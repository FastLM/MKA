from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from mka.layers.fastmka import FastMKAConfig, FastMKAAttention


@dataclass
class TinyLMConfig:
    vocab_size: int
    hidden_size: int
    num_heads: int
    max_len: int


class TinyLM(nn.Module):
    def __init__(self, cfg: TinyLMConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.attn = FastMKAAttention(
            FastMKAConfig(hidden_size=cfg.hidden_size, num_heads=cfg.num_heads, use_l3=False)
        )
        self.ln = nn.LayerNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        x = self.emb(input_ids)
        x, _, _, _ = self.attn(x)
        x = self.ln(x)
        return self.head(x)


def collate_fn(batch, tokenizer, max_len):
    texts = [x["text"] for x in batch if x["text"].strip()]
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
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, cfg["seq_len"]),
    )

    model = TinyLM(
        TinyLMConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=cfg["hidden_size"],
            num_heads=cfg["num_heads"],
            max_len=cfg["seq_len"],
        )
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    t0 = time.time()
    tokens = 0

    for step, input_ids in enumerate(dl):
        if step >= cfg["max_steps"]:
            break
        input_ids = input_ids.to(device)
        logits = model(input_ids[:, :-1])
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        tokens += input_ids.numel()
        if (step + 1) % 10 == 0:
            print(f"step={step+1} loss={loss.item():.4f}")

    elapsed = time.time() - t0
    print(f"throughput_tok_s={tokens / max(elapsed, 1e-6):.2f}")


if __name__ == "__main__":
    main()
