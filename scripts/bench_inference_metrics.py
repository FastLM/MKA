"""
Inference metrics: prefill vs decode throughput, per-phase GPU peak memory,
and KV-cache size estimate from `past_key_values` tensors (for artifact comparisons).

Usage:
  python scripts/bench_inference_metrics.py --config configs/hf_qwen_fastmka.yaml
"""
from __future__ import annotations

import argparse
import time
from typing import Any, cast

import torch
import torch.nn as nn
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from mka.hf import apply_hf_attention_patch, parse_patch_config
from mka.utils.repro import set_global_seed


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _tensor_storage_bytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, (tuple, list)):
        return sum(_tensor_storage_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_tensor_storage_bytes(v) for v in obj.values())
    return 0


def estimate_kv_cache_bytes(past_key_values: Any) -> int:
    """Sum storage of KV tensors in HF `past_key_values` (tuple or `DynamicCache`-style)."""
    if past_key_values is None:
        return 0
    try:
        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            return _tensor_storage_bytes(past_key_values.key_cache) + _tensor_storage_bytes(
                past_key_values.value_cache
            )
        return _tensor_storage_bytes(past_key_values)
    except Exception:
        return 0


def bench_prefill_decode(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
    decode_steps: int,
    dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    vocab = getattr(model.config, "vocab_size", 32000)
    input_ids = torch.randint(1, min(vocab, 50000), (1, seq_len), device=device)

    # Prefill
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
        out = model(input_ids=input_ids, use_cache=True)
    _sync()
    prefill_s = time.perf_counter() - t0
    past = out.past_key_values
    logits = out.logits
    next_id = logits[:, -1:, :].argmax(dim=-1)
    kv_prefill = estimate_kv_cache_bytes(past)
    prefill_peak_gb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**3) if device.type == "cuda" else 0.0
    )

    # Decode (autoregressive steps)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync()
    t1 = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
        for _ in range(decode_steps):
            out = model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1:, :].argmax(dim=-1)
    _sync()
    decode_s = time.perf_counter() - t1
    kv_decode = estimate_kv_cache_bytes(past)
    decode_peak_gb = (
        torch.cuda.max_memory_allocated(device) / (1024.0**3) if device.type == "cuda" else 0.0
    )

    return {
        "prefill_tokens": float(seq_len),
        "prefill_s": float(prefill_s),
        "prefill_tok_s": float(seq_len) / max(prefill_s, 1e-9),
        "decode_tokens": float(decode_steps),
        "decode_s": float(decode_s),
        "decode_tok_s": float(decode_steps) / max(decode_s, 1e-9),
        "prefill_peak_gpu_memory_gb": float(prefill_peak_gb),
        "decode_peak_gpu_memory_gb": float(decode_peak_gb),
        "kv_cache_bytes_after_prefill": float(kv_prefill),
        "kv_cache_bytes_after_decode": float(kv_decode),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Prefill/decode throughput, KV size, and peak GPU memory.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--decode-steps", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(0)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if cfg.get("tp_size", 1) > 1:
        model_kwargs["tp_plan"] = "auto"

    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=dtype,
        **model_kwargs,
    )
    patch_cfg = parse_patch_config(cfg.get("mka_patch", {}))
    apply_hf_attention_patch(model, patch_cfg)
    model = cast(nn.Module, model).to(device)
    model.eval()

    seq_len = int(cfg["seq_len"])

    for _ in range(args.warmup):
        bench_prefill_decode(model, device, seq_len, args.decode_steps, dtype)

    stats = bench_prefill_decode(model, device, seq_len, args.decode_steps, dtype)

    print("--- inference metrics (forward-only; no backward) ---")
    print(f"seed={seed}")
    for k in (
        "prefill_tok_s",
        "prefill_s",
        "decode_tok_s",
        "decode_s",
        "prefill_peak_gpu_memory_gb",
        "decode_peak_gpu_memory_gb",
        "kv_cache_bytes_after_prefill",
        "kv_cache_bytes_after_decode",
    ):
        v = stats[k]
        if "tok_s" in k:
            print(f"{k}={v:.2f}")
        elif "bytes" in k:
            print(f"{k}={v:.0f}")
        elif "memory" in k:
            print(f"{k}={v:.4f}")
        else:
            print(f"{k}={v:.6f}")
    print(
        "Note: HBM read/write bandwidth needs Nsight/CUPTI or `nvidia-smi dmon` on the host; not exposed by PyTorch alone."
    )


if __name__ == "__main__":
    main()
