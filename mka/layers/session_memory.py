from __future__ import annotations

import torch


def causal_prefix_ema(x: torch.Tensor, beta: float = 0.9) -> torch.Tensor:
    """
    Causal session summary used as L2 memory.

    Args:
        x: [B, T, D]
        beta: EMA coefficient.
    Returns:
        [B, T, D] causal summary where each timestep only sees <= t.
    """
    b, t, d = x.shape
    out = torch.zeros_like(x)
    state = torch.zeros((b, d), dtype=x.dtype, device=x.device)
    for i in range(t):
        state = beta * state + (1.0 - beta) * x[:, i, :]
        out[:, i, :] = state
    return out


def repeat_retrieval_to_length(mem: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Utility for L3 integration:
    convert [B, 1, D] or [B, T3, D] memory to [B, T, D] via repeat/truncate.
    """
    if mem.size(1) == target_len:
        return mem
    if mem.size(1) == 1:
        return mem.repeat(1, target_len, 1)
    if mem.size(1) > target_len:
        return mem[:, :target_len, :]
    # If shorter than target_len and not singleton, pad by last token.
    pad_n = target_len - mem.size(1)
    last = mem[:, -1:, :].repeat(1, pad_n, 1)
    return torch.cat([mem, last], dim=1)
