#!/usr/bin/env bash
set -euo pipefail

# External benchmark integration steps for LongBench.
# This repository does not include LongBench data/scripts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${ROOT_DIR}/external/LongBench"

echo "[1/5] Clone LongBench"
mkdir -p "${ROOT_DIR}/external"
if [ ! -d "${BENCH_DIR}" ]; then
  git clone https://github.com/THUDM/LongBench.git "${BENCH_DIR}"
fi

echo "[2/5] Install LongBench dependencies"
python -m pip install -r "${BENCH_DIR}/requirements.txt"

echo "[3/5] Export model path"
echo "Set your checkpoint path:"
echo "  export MODEL_PATH=/path/to/fastmka-checkpoint"

echo "[4/5] Run evaluation (official scripts)"
echo "Example:"
echo "  cd ${BENCH_DIR}"
echo "  python pred.py --model_path \$MODEL_PATH --max_length 128000 --max_gen 128"
echo "  python eval.py --pred_dir ./pred"

echo "[5/5] Record outputs"
echo "Collect QA / Summarization / Code / Avg for paper table."
