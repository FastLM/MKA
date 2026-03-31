from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mka.cuda.ops import fastmka_attn, has_fastmka_cuda
from mka.layers.session_memory import causal_prefix_ema, repeat_retrieval_to_length


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, kv_h, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, kv_h, n_rep, s, d)
    return hidden_states.reshape(b, kv_h * n_rep, s, d)


class HFFastMKAAttention(nn.Module):
    """
    HF-compatible FastMKA drop-in attention wrapper.
    Supports Llama/Qwen-style attention modules with q_proj/k_proj/v_proj/o_proj.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        use_l3: bool = False,
        ema_beta: float = 0.9,
        use_cuda_kernel: bool = True,
        prefer_sdpa: bool = True,
    ):
        super().__init__()
        self.original_attn = original_attn
        self.use_l3 = use_l3
        self.ema_beta = ema_beta
        self.use_cuda_kernel = use_cuda_kernel
        self.prefer_sdpa = prefer_sdpa

        self.hidden_size = original_attn.hidden_size
        self.num_heads = original_attn.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(original_attn, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Reuse pretrained projections to keep compatibility.
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.rotary_emb = getattr(original_attn, "rotary_emb", None)
        self.attention_dropout = float(getattr(original_attn, "attention_dropout", 0.0))

        levels = 3 if use_l3 else 2
        self.router = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size // 2, levels),
        )

    def _shape(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        b, t, _ = hidden_states.shape
        q = self.q_proj(hidden_states)

        l1 = hidden_states
        l2 = causal_prefix_ema(hidden_states, beta=self.ema_beta)
        levels = [l1, l2]
        if self.use_l3:
            l3 = torch.zeros((b, 1, self.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
            levels.append(repeat_retrieval_to_length(l3, t))

        lam = torch.softmax(self.router(q), dim=-1)
        fused = 0.0
        for i, mem in enumerate(levels):
            fused = fused + lam[..., i : i + 1] * mem

        q_states = self._shape(q, self.num_heads)
        k_states = self._shape(self.k_proj(fused), self.num_key_value_heads)
        v_states = self._shape(self.v_proj(fused), self.num_key_value_heads)

        if self.rotary_emb is not None:
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                # Llama/Qwen usually expose rotary_emb(value_states, position_ids)
                cos, sin = self.rotary_emb(v_states, position_ids)
            q_states, k_states = _apply_rotary_pos_emb(q_states, k_states, cos, sin)

        if past_key_value is not None:
            pk, pv = past_key_value
            k_states = torch.cat([pk, k_states], dim=2)
            v_states = torch.cat([pv, v_states], dim=2)

        present = (k_states, v_states) if use_cache else None

        k_states = _repeat_kv(k_states, self.num_key_value_groups)
        v_states = _repeat_kv(v_states, self.num_key_value_groups)

        can_use_kernel = (
            self.use_cuda_kernel
            and q_states.is_cuda
            and has_fastmka_cuda()
            and self.head_dim <= 256
            and attention_mask is None
            and q_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )
        can_use_sdpa = not output_attentions and q_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
        use_sdpa_first = self.prefer_sdpa and can_use_sdpa
        if use_sdpa_first and past_key_value is None and attention_mask is None:
            attn_output = F.scaled_dot_product_attention(
                q_states,
                k_states,
                v_states,
                attn_mask=None,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True,
            )
            attn_weights = None
        elif use_sdpa_first and attention_mask is not None:
            attn_output = F.scaled_dot_product_attention(
                q_states,
                k_states,
                v_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
            attn_weights = None
        elif can_use_kernel:
            attn_output = fastmka_attn(q_states, k_states, v_states, causal=True)
            attn_weights = None
        else:
            # Prefer SDPA to keep flash/memory-efficient kernels enabled in fallback path.
            # This avoids a large regression from explicit matmul+softmax when kernel is unavailable.
            if can_use_sdpa and past_key_value is None and attention_mask is None:
                attn_output = F.scaled_dot_product_attention(
                    q_states,
                    k_states,
                    v_states,
                    attn_mask=None,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=True,
                )
                attn_weights = None
            elif can_use_sdpa and attention_mask is not None:
                attn_output = F.scaled_dot_product_attention(
                    q_states,
                    k_states,
                    v_states,
                    attn_mask=attention_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=False,
                )
                attn_weights = None
            else:
                attn_weights = torch.matmul(q_states, k_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_states.dtype)
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                attn_output = torch.matmul(attn_weights, v_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, present
