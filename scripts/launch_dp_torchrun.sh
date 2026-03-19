#!/usr/bin/env bash
set -euo pipefail

# Data Parallel launch with torchrun.
# Usage:
#   bash scripts/launch_dp_torchrun.sh configs/hf_qwen_fastmka.yaml 4

CFG_PATH="${1:-configs/hf_qwen_fastmka.yaml}"
NPROC="${2:-1}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  scripts/train_hf_patch.py \
  --config "${CFG_PATH}"
