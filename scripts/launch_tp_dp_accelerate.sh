#!/usr/bin/env bash
set -euo pipefail

# TP+DP launch path:
# - DP is managed by accelerate processes.
# - TP is enabled in config via tp_size > 1 and model load with tp_plan=auto.
#
# Usage:
#   bash scripts/launch_tp_dp_accelerate.sh configs/hf_qwen_fastmka.yaml 4

CFG_PATH="${1:-configs/hf_qwen_fastmka.yaml}"
NUM_PROC="${2:-1}"

accelerate launch \
  --num_processes "${NUM_PROC}" \
  --mixed_precision bf16 \
  scripts/train_hf_patch.py \
  --config "${CFG_PATH}"
