from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from .attention import HFFastMKAAttention


@dataclass
class HFPatchConfig:
    enabled: bool = True
    mode: str = "fastmka"  # fastmka | off
    use_l3: bool = False
    ema_beta: float = 0.9
    use_cuda_kernel: bool = True
    verbose: bool = True


def _iter_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model layout: cannot find decoder layers.")


def apply_hf_attention_patch(model: nn.Module, cfg: HFPatchConfig) -> int:
    if not cfg.enabled or cfg.mode == "off":
        return 0

    patched = 0
    for layer in _iter_layers(model):
        if not hasattr(layer, "self_attn"):
            continue
        attn = layer.self_attn
        required = all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
        if not required:
            continue
        layer.self_attn = HFFastMKAAttention(
            attn, use_l3=cfg.use_l3, ema_beta=cfg.ema_beta, use_cuda_kernel=cfg.use_cuda_kernel
        )
        patched += 1

    if cfg.verbose:
        print(f"[MKA patch] patched self_attn layers: {patched}")
    return patched


def parse_patch_config(raw: dict[str, Any]) -> HFPatchConfig:
    return HFPatchConfig(
        enabled=bool(raw.get("enabled", True)),
        mode=str(raw.get("mode", "fastmka")),
        use_l3=bool(raw.get("use_l3", False)),
        ema_beta=float(raw.get("ema_beta", 0.9)),
        use_cuda_kernel=bool(raw.get("use_cuda_kernel", True)),
        verbose=bool(raw.get("verbose", True)),
    )
