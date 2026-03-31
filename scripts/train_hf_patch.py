from __future__ import annotations

import argparse
import math
import os
import time
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from datasets import load_dataset  # type: ignore[import-not-found]
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from mka.cuda.ops import has_fastmka_cuda, has_fused_route_mka_cuda
from mka.hf import apply_hf_attention_patch, parse_patch_config
from mka.utils.repro import set_global_seed


def collate_fn(batch, tokenizer, max_len):
    texts = [x["text"] for x in batch if x.get("text", "").strip()]
    tok = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    return tok["input_ids"]


@torch.no_grad()
def eval_ppl_wikitext_validation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    seq_len: int,
    max_batches: int,
    use_autocast: bool,
) -> float:
    """Mean token cross-entropy on wikitext-2 validation (approximate PPL = exp(loss))."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, seq_len),
    )
    model.eval()
    total_nll = 0.0
    n_tokens = 0
    for step, input_ids in enumerate(dl):
        if step >= max_batches:
            break
        input_ids = input_ids.to(device)
        labels = input_ids.clone()
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=use_autocast):
            out = model(input_ids=input_ids, labels=labels)
        loss = float(out.loss.item())
        total_nll += loss * input_ids.numel()
        n_tokens += input_ids.numel()
    model.train()
    mean_nll = total_nll / max(n_tokens, 1)
    return math.exp(mean_nll)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument(
        "--eval-ppl",
        action="store_true",
        help="After training, run validation perplexity on wikitext-2 (small; see --eval-ppl-max-batches).",
    )
    parser.add_argument("--eval-ppl-max-batches", type=int, default=50)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    if bool(cfg.get("deterministic", False)):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cpu")
    if use_ddp:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if cfg.get("tp_size", 1) > 1:
        model_kwargs["tp_plan"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float16,
        **model_kwargs,
    )
    patch_cfg = parse_patch_config(cfg.get("mka_patch", {}))
    apply_hf_attention_patch(model, patch_cfg)
    model.to(device)  # type: ignore[attr-defined]
    if use_ddp:
        model = DDP(
            model,
            device_ids=[args.local_rank] if use_cuda else None,
            output_device=args.local_rank if use_cuda else None,
        )
    model.train()

    max_steps = int(cfg["max_steps"])
    warmup_steps = int(cfg.get("warmup_steps", 0))

    if args.local_rank == 0:
        print(
            "[MKA CUDA] fastmka_cuda (attention kernel) available:",
            has_fastmka_cuda(),
            "| fused_route_mka_cuda available:",
            has_fused_route_mka_cuda(),
        )
        print(f"[repro] seed={seed} warmup_steps={warmup_steps} deterministic={bool(cfg.get('deterministic', False))}")
        print(
            "[metrics] train_throughput_tok_s uses wall time after warmup only (if warmup_steps>0); "
            "full step = forward + backward + optimizer. For inference prefill/decode only, "
            "use scripts/bench_inference_metrics.py.",
        )

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    if use_ddp:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=args.local_rank,
            shuffle=True,
            seed=seed,
        )
    else:
        sampler = None

    gen = torch.Generator()
    gen.manual_seed(seed)
    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer, cfg["seq_len"]),
        generator=gen if sampler is None else None,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    loss_accum = 0.0
    loss_steps = 0
    measuring = False
    t_meas0: float | None = None
    seen_tokens = 0

    if sampler is not None:
        sampler.set_epoch(0)

    t_loop_start = time.perf_counter()
    for step, input_ids in enumerate(dl):
        if step >= max_steps:
            break
        input_ids = input_ids.to(device)
        labels = input_ids.clone()
        with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=torch.bfloat16, enabled=use_cuda):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_accum += float(loss.item())
        loss_steps += 1
        if (step + 1) % 5 == 0 and args.local_rank == 0:
            print(f"step={step+1} loss={loss.item():.4f}")

        if step < warmup_steps:
            continue
        if not measuring:
            measuring = True
            if use_cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            t_meas0 = time.perf_counter()
            seen_tokens = 0
        seen_tokens += input_ids.numel()

    if use_cuda:
        torch.cuda.synchronize(device)

    peak_mem_gb = 0.0
    if use_cuda:
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024.0**3)

    if args.local_rank == 0:
        if warmup_steps >= max_steps:
            print(
                "warning: warmup_steps >= max_steps; no timed steps. Increase max_steps or lower warmup_steps."
            )
        mean_loss = loss_accum / max(loss_steps, 1)
        print(f"train_mean_loss={mean_loss:.6f}")
        print(f"train_total_elapsed_s={time.perf_counter() - t_loop_start:.4f}")
        print(f"peak_gpu_memory_gb={peak_mem_gb:.4f}")

        if measuring and t_meas0 is not None:
            elapsed = time.perf_counter() - t_meas0
            train_tok_s = seen_tokens / max(elapsed, 1e-9)
            print(f"train_throughput_tok_s={train_tok_s:.2f}")
            print(f"train_timed_elapsed_s={elapsed:.4f}")
            print(f"train_timed_tokens={seen_tokens}")
            print(f"warmup_steps_excluded={warmup_steps}")
        else:
            print("train_throughput_tok_s=nan (no step after warmup; see warning above)")

        if args.eval_ppl:
            ppl = eval_ppl_wikitext_validation(
                cast(nn.Module, model.module if use_ddp else model),
                tokenizer,
                device,
                seq_len=int(cfg["seq_len"]),
                max_batches=int(args.eval_ppl_max_batches),
                use_autocast=use_cuda,
            )
            print(f"wikitext2_validation_ppl_approx={ppl:.4f} (max_batches={args.eval_ppl_max_batches})")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
