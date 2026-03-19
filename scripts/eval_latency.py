from __future__ import annotations

import argparse
import time

import torch

from mka.layers.fastmka import FastMKAConfig, FastMKAAttention


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn = FastMKAAttention(
        FastMKAConfig(hidden_size=args.hidden_size, num_heads=args.num_heads, use_l3=False)
    ).to(device)
    x = torch.randn(1, args.seq_len, args.hidden_size, device=device, dtype=torch.bfloat16)

    for _ in range(args.warmup):
        _ = attn(x)
    torch.cuda.synchronize() if device == "cuda" else None

    t0 = time.time()
    for _ in range(args.iters):
        _ = attn(x)
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = (time.time() - t0) / args.iters
    print(f"latency_ms_per_forward={elapsed * 1000:.3f}")


if __name__ == "__main__":
    main()
