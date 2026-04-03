# MKA Implementation

This ropo contains code implemnentation of our paper `MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning `

The main idea include:
- `MKA` (3-path hierarchical memory attention)
- `FastMKA` (route-fused variant for speed)
- CUDA extensions for fused routing + online softmax
- Reproducible training/evaluation scripts

Our code repo include follwing:
- `mka/layers/`: PyTorch modules (`MKAFullAttention`, `FastMKAAttention`)
- `mka/hf/`: HuggingFace monkey patch support (Qwen/Llama style `self_attn`)
- `mka/cuda/`: CUDA extensions (`fastmka_attn`, optional `fused_route_mka`)
- `mka/config/`: optional YAML fields for `memory_hierarchy` etc.
- `mka/utils/repro.py`: global RNG seeding for reproducible runs
- `scripts/`: train/eval/benchmark entry points
- `configs/`: experiment configs

To get started, run following scripts

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_wikitext2.py --config configs/qwen7b_fastmka.yaml
```

## Features

1. **FastMKA forward path** in PyTorch:
   - L1 local memory (`X`)
   - L2 causal session summary (prefix EMA)
   - Optional L3 retrieved memory
   - Learned routing gate (`softmax(MLP(Q))`)
   - Route-fusion before single KV projection

2. **MKA full path** in PyTorch:
   - Per-level attention over L1/L2/L3
   - Soft mixture over outputs

3. **CUDA design**:
   - Tiled QK score calculation
   - Online max/denominator (`m`, `z`) update
   - Fused route application before attention
   - Causal masking support

4. **HuggingFace direct patch path**:
   - `mka/hf/attention.py`: `HFFastMKAAttention` wrapper
   - `mka/hf/patch.py`: monkey patch over decoder `self_attn`
   - `scripts/train_hf_patch.py`: train loop with patched attention
   - `configs/hf_qwen_fastmka.yaml`, `configs/hf_llama_fastmka.yaml`
   - `scripts/launch_dp_torchrun.sh`: DP launch (torchrun)
   - `scripts/launch_tp_dp_accelerate.sh`: TP+DP launch path (accelerate + HF TP)

## Usage

### 1. Build CUDA kernel

```bash
cd mka/cuda
python build.py build_ext --inplace
cd ../..
```

If build fails with `CUDA_HOME environment variable is not set`, export your CUDA path first, e.g.
`export CUDA_HOME=/usr/local/cuda`.

### 2. Execute on GPU(s)

2.1 For Single GPU

```bash
python scripts/train_hf_patch.py --config configs/hf_qwen_fastmka.yaml
```

2.2 For Multi-GPU DP

```bash
bash scripts/launch_dp_torchrun.sh configs/hf_qwen_fastmka.yaml 4
```

2.3 For Multi-GPU TP+DP

1. Set `tp_size` in config (`>1`).
2. Launch:

```bash
bash scripts/launch_tp_dp_accelerate.sh configs/hf_qwen_fastmka.yaml 4
```

Notes:
- TP relies on HuggingFace native `tp_plan="auto"` support for the model/version.
- Dependencies use lower bounds in `requirements.txt` (adjust `torch` for your CUDA wheel).
- **Training throughput** (`train_throughput_tok_s`) is measured **after** `warmup_steps` (see YAML) and includes forward + backward + optimizer. **Inference** prefill/decode (forward-only) is reported by `scripts/bench_inference_metrics.py`.
- FastMKA CUDA kernel is used automatically when:
  - extension `fastmka_cuda` is available,
  - tensor is CUDA,
  - `head_dim <= 256`,
  - no extra additive attention mask is required.

### Metrics

- YAML: `seed`, `warmup_steps` (exclude cold-start from timed throughput), optional `deterministic: true` (slower, stricter cudnn).
- `train_hf_patch.py`: logs `train_mean_loss`, `train_total_elapsed_s`, `train_throughput_tok_s` (post-warmup), `peak_gpu_memory_gb`, optional `--eval-ppl` for validation PPL.
- `bench_inference_metrics.py`: `prefill_tok_s`, `decode_tok_s`, per-phase peak GPU memory, `kv_cache_bytes_*` from `past_key_values`. HBM bandwidth is not available from PyTorch alone; use Nsight / `nvidia-smi dmon` on the host.

### Memory hierarchy (Block-MKA + FastMKA)

**Block-MKA** (§4.2) maps memory to compute tiers: **L1** on-chip SRAM (tiled attention, online softmax with running max and partition sum); **L2** HBM (activations, Q/K/V, fused KV cache); **L3** DRAM (vectorized hash, chunk recall). **FastMKA** (Algorithm 2) **route-fuses** L1/L2/(L3) into one hidden representation, then **one** KV projection and **one** causal attention — the dominant data path is **fused activations → KV on HBM → attention** (see detials in our paper Tables 4–6).

YAML `memory_hierarchy` records these tiers for reproducibility (`mka/config/memory_hierarchy.py`). Older keys (`hbm_enabled`, `dram_staging`, `ssd_tier_path`) still parse as aliases.

**Scripts vs metrics**

- Training: `train_hf_patch.py` — tokens/s includes backward + optimizer
- Inference-style: `bench_inference_metrics.py` — prefill vs decode, KV bytes, per-phase GPU peak memory.
- HBM bandwidth: use **Nsight Compute** / vendor tools, not PyTorch alone.

**References for implementation quality**

- FlashAttention-2 / SDPA for L1+L2 fused attention paths.
- Paged KV / disk offload patterns (vLLM, FlashInfer-class) when extending L3 or spill.
- ZeRO-Infinity–style paths for `ssd_spill_path` experiments.

## Evaluation

For evaluation of LongBench and RULER.
Use:

- `scripts/run_longbench.sh` for LongBench workflow
- `scripts/run_ruler.sh` for RULER workflow

## Cite
```
@inproceedings{mka2026,
  title     = {MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning},
  author    = {Dong Liu and Yanxuan Yu and Ben Lengerich and Ying Nian Wu},
  booktitle = {Proceedings of the ACM International Conference on Computing Frontiers (CF '26)},
  year      = {2026}
}
```
