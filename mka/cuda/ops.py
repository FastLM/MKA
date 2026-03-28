from __future__ import annotations

from typing import Optional

import torch

_EXT: Optional[object] = None
_FUSED_EXT: Optional[object] = None


def _load_ext() -> Optional[object]:
    global _EXT
    if _EXT is not None:
        return _EXT
    try:
        import fastmka_cuda  # type: ignore

        _EXT = fastmka_cuda
        return _EXT
    except Exception:
        _EXT = None
        return None


def has_fastmka_cuda() -> bool:
    return _load_ext() is not None


def _load_fused_ext() -> Optional[object]:
    global _FUSED_EXT
    if _FUSED_EXT is not None:
        return _FUSED_EXT
    try:
        import fused_route_mka_cuda  # type: ignore

        _FUSED_EXT = fused_route_mka_cuda
        return _FUSED_EXT
    except Exception:
        _FUSED_EXT = None
        return None


def has_fused_route_mka_cuda() -> bool:
    return _load_fused_ext() is not None


def fused_route_mka_attn(
    q: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    l3: torch.Tensor,
    lam: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
) -> torch.Tensor:
    """Fused route + K/V projection + causal attention; matches FastMKA math when H*Dh == D."""
    ext = _load_fused_ext()
    if ext is None:
        raise RuntimeError(
            "fused_route_mka_cuda is not available. Build extensions from mka/cuda/build.py."
        )
    return ext.forward(
        q.contiguous(),
        l1.contiguous(),
        l2.contiguous(),
        l3.contiguous(),
        lam.contiguous(),
        wk.contiguous(),
        wv.contiguous(),
    )[0]


def fastmka_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    ext = _load_ext()
    if ext is None:
        raise RuntimeError("fastmka_cuda extension is not available. Build it from mka/cuda/build.py.")
    return ext.fastmka_attn(q.contiguous(), k.contiguous(), v.contiguous(), causal)
