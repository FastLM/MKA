from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from mka.cuda.ops import (
    fastmka_attn,
    fused_route_mka_attn,
    has_fastmka_cuda,
    has_fused_route_mka_cuda,
)
from .session_memory import causal_prefix_ema, repeat_retrieval_to_length


@dataclass
class FastMKAConfig:
    hidden_size: int
    num_heads: int
    use_l3: bool = True
    ema_beta: float = 0.9
    dropout_p: float = 0.0
    use_cuda_kernel: bool = True
    # Fused route+KV+attention CUDA kernel (MHA only); requires no KV cache.
    # Keep default off: current reference kernel is functionally correct but not yet optimized.
    use_fused_route_cuda: bool = False


class FastMKAAttention(nn.Module):
    """
    Route-Fused MKA (FastMKA):
    1) build L1/L2/(L3)
    2) lambda = softmax(MLP(Q))
    3) X_fused = sum(lambda_l * L_l)
    4) single K,V projection + single causal attention
    """

    def __init__(self, cfg: FastMKAConfig):
        super().__init__()
        assert cfg.hidden_size % cfg.num_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.hidden_size // cfg.num_heads

        self.wq = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.wk = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.wv = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

        levels = 3 if cfg.use_l3 else 2
        self.router = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size // 2, levels),
        )
        self.dropout = nn.Dropout(cfg.dropout_p)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.cfg.num_heads
        return x.view(b, t, h, self.head_dim).transpose(1, 2)  # [B,H,T,Dh]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def forward(
        self,
        x: torch.Tensor,
        l3_memory: Optional[torch.Tensor] = None,
        kv_cache_k: Optional[torch.Tensor] = None,
        kv_cache_v: Optional[torch.Tensor] = None,
    ):
        b, t, _ = x.shape
        q = self.wq(x)  # [B,T,D]

        # Build hierarchical memory
        l1 = x
        l2 = causal_prefix_ema(x, beta=self.cfg.ema_beta)
        levels = [l1, l2]
        if self.cfg.use_l3:
            if l3_memory is None:
                l3_memory = torch.zeros((b, 1, x.size(-1)), dtype=x.dtype, device=x.device)
            l3 = repeat_retrieval_to_length(l3_memory, t)
            levels.append(l3)

        logits = self.router(q)
        lam = torch.softmax(logits, dim=-1)  # [B,T,L]

        fused = 0.0
        for i, mem in enumerate(levels):
            fused = fused + lam[..., i : i + 1] * mem

        k_new = self._reshape_heads(self.wk(fused))  # [B,H,T,Dh]
        v_new = self._reshape_heads(self.wv(fused))
        qh = self._reshape_heads(q)

        if kv_cache_k is not None and kv_cache_v is not None:
            k_tot = torch.cat([kv_cache_k, k_new], dim=2)
            v_tot = torch.cat([kv_cache_v, v_new], dim=2)
        else:
            k_tot, v_tot = k_new, v_new

        d_hidden = self.cfg.hidden_size
        h, dh = self.cfg.num_heads, self.head_dim
        l3_for_fused = levels[2] if len(levels) == 3 else torch.zeros_like(l1)

        scale = 1.0 / math.sqrt(self.head_dim)
        dtype_ok = qh.dtype in (torch.float16, torch.bfloat16, torch.float32)
        no_kv_cache = kv_cache_k is None and kv_cache_v is None
        can_use_fused = (
            self.cfg.use_cuda_kernel
            and self.cfg.use_fused_route_cuda
            and x.is_cuda
            and no_kv_cache
            and has_fused_route_mka_cuda()
            and h * dh == d_hidden
            and self.head_dim <= 256
            and dtype_ok
        )
        can_use_kernel = (
            self.cfg.use_cuda_kernel
            and x.is_cuda
            and has_fastmka_cuda()
            and self.head_dim <= 256
            and dtype_ok
        )
        if can_use_fused:
            out = fused_route_mka_attn(
                qh,
                l1,
                l2,
                l3_for_fused,
                lam,
                self.wk.weight,
                self.wv.weight,
            )
        elif can_use_kernel:
            out = fastmka_attn(qh, k_tot, v_tot, causal=True)
        else:
            # When no cache is involved, SDPA can dispatch to optimized kernels.
            if kv_cache_k is None and kv_cache_v is None:
                out = torch.nn.functional.scaled_dot_product_attention(
                    qh,
                    k_tot,
                    v_tot,
                    attn_mask=None,
                    dropout_p=self.cfg.dropout_p if self.training else 0.0,
                    is_causal=True,
                )
            else:
                scores = torch.matmul(qh, k_tot.transpose(-1, -2)) * scale  # [B,H,T,Ttot]
                t_tot = k_tot.size(2)
                q_pos = torch.arange(t_tot - t, t_tot, device=x.device).view(1, 1, t, 1)
                k_pos = torch.arange(t_tot, device=x.device).view(1, 1, 1, t_tot)
                causal = k_pos <= q_pos
                scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
                attn = torch.softmax(scores, dim=-1)
                attn = self.dropout(attn)
                out = torch.matmul(attn, v_tot)
        out = self._merge_heads(out)
        out = self.wo(out)
        return out, k_tot.detach(), v_tot.detach(), lam
