# MKA Code Folder

This ropo contains code implemnentation of our paper MKA:

- `MKA` (3-path hierarchical memory attention)
- `FastMKA` (route-fused variant for speed)
- CUDA extension scaffolding for fused routing + online softmax
- Reproducible training/evaluation scripts following the paper setup

## Structure

- `mka/layers/`: PyTorch modules (`MKAFullAttention`, `FastMKAAttention`)
- `mka/hf/`: HuggingFace monkey patch support (Qwen/Llama style `self_attn`)
- `mka/cuda/`: CUDA/C++ extension skeleton for fused kernels
- `scripts/`: train/eval/benchmark entry points
- `configs/`: experiment configs

## Quick Start

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_wikitext2.py --config configs/qwen7b_fastmka.yaml
```

## What Is Implemented

1. **FastMKA forward path** in PyTorch:
   - L1 local memory (`X`)
   - L2 causal session summary (prefix EMA)
   - Optional L3 retrieved memory
   - Learned routing gate (`softmax(MLP(Q))`)
   - Route-fusion before single KV projection

2. **MKA full path** in PyTorch:
   - Per-level attention over L1/L2/L3
   - Soft mixture over outputs

3. **CUDA design (detailed skeleton)**:
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

## HuggingFace Patch Usage

### Build CUDA kernel (recommended)

```bash
cd mka/cuda
python build.py build_ext --inplace
cd ../..
```

If build fails with `CUDA_HOME environment variable is not set`, export your CUDA path first, e.g.
`export CUDA_HOME=/usr/local/cuda`.

### Single GPU

```bash
python scripts/train_hf_patch.py --config configs/hf_qwen_fastmka.yaml
```

### Multi-GPU DP

```bash
bash scripts/launch_dp_torchrun.sh configs/hf_qwen_fastmka.yaml 4
```

### Multi-GPU TP+DP

1. Set `tp_size` in config (`>1`).
2. Launch:

```bash
bash scripts/launch_tp_dp_accelerate.sh configs/hf_qwen_fastmka.yaml 4
```

Notes:
- TP relies on HuggingFace native `tp_plan="auto"` support for the model/version.
- For strict reproducibility, pin `transformers` and CUDA versions used in your paper runs.
- FastMKA CUDA kernel is used automatically when:
  - extension `fastmka_cuda` is available,
  - tensor is CUDA,
  - `head_dim <= 256`,
  - no extra additive attention mask is required.

## Benchmark Evaluation

The paper reports LongBench and RULER.
Use:

- `scripts/run_longbench.sh` for LongBench workflow
- `scripts/run_ruler.sh` for RULER workflow

