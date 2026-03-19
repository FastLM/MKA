from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .session_memory import causal_prefix_ema, repeat_retrieval_to_length


@dataclass
class MKAFullConfig:
    hidden_size: int
    num_heads: int
    use_l3: bool = True
    ema_beta: float = 0.9


class MKAFullAttention(nn.Module):
    """Full 3-path MKA with routing over per-level attention outputs."""

    def __init__(self, cfg: MKAFullConfig):
        super().__init__()
        assert cfg.hidden_size % cfg.num_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.hidden_size // cfg.num_heads

        self.wq = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.wk = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.wv = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.router = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_size // 2, 3 if cfg.use_l3 else 2),
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.cfg.num_heads
        return x.view(b, t, h, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, dh = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * dh)

    def _causal_attend(self, qh: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        kh = self._reshape_heads(self.wk(mem))
        vh = self._reshape_heads(self.wv(mem))
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
        t = qh.size(2)
        mask = torch.tril(torch.ones((t, t), device=qh.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask.view(1, 1, t, t), torch.finfo(scores.dtype).min)
        return torch.matmul(torch.softmax(scores, dim=-1), vh)

    def forward(self, x: torch.Tensor, l3_memory: Optional[torch.Tensor] = None):
        b, t, _ = x.shape
        q = self.wq(x)
        qh = self._reshape_heads(q)

        l1 = x
        l2 = causal_prefix_ema(x, beta=self.cfg.ema_beta)
        levels = [l1, l2]
        if self.cfg.use_l3:
            if l3_memory is None:
                l3_memory = torch.zeros((b, 1, x.size(-1)), dtype=x.dtype, device=x.device)
            levels.append(repeat_retrieval_to_length(l3_memory, t))

        lam = torch.softmax(self.router(q), dim=-1)  # [B,T,L]
        mixed = 0.0
        for i, mem in enumerate(levels):
            path_out = self._causal_attend(qh, mem)  # [B,H,T,Dh]
            mixed = mixed + lam[..., i : i + 1].unsqueeze(1) * path_out

        out = self.wo(self._merge_heads(mixed))
        return out, lam
