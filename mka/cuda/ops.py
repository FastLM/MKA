from __future__ import annotations

from typing import Optional

import torch

_EXT: Optional[object] = None


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


def fastmka_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    ext = _load_ext()
    if ext is None:
        raise RuntimeError("fastmka_cuda extension is not available. Build it from mka/cuda/build.py.")
    return ext.fastmka_attn(q.contiguous(), k.contiguous(), v.contiguous(), causal)
