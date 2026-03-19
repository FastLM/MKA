from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import yaml
from datasets import load_dataset  # type: ignore[import-not-found]
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from mka.hf import apply_hf_attention_patch, parse_patch_config


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.backends.cuda.matmul.allow_tf32 = True
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
        # HF native tensor parallel support (for supported models/versions).
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

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=args.local_rank, shuffle=True) if use_ddp else None
    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer, cfg["seq_len"]),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # bf16 path; keep interface explicit.
    t0 = time.time()
    seen_tokens = 0

    if sampler is not None:
        sampler.set_epoch(0)
    for step, input_ids in enumerate(dl):
        if step >= int(cfg["max_steps"]):
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

        seen_tokens += input_ids.numel()
        if (step + 1) % 5 == 0 and args.local_rank == 0:
            print(f"step={step+1} loss={loss.item():.4f}")

    if use_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    if args.local_rank == 0:
        print(f"throughput_tok_s={seen_tokens / max(elapsed, 1e-6):.2f}")
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
