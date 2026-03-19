#!/usr/bin/env bash
set -euo pipefail

# External benchmark integration steps for RULER.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${ROOT_DIR}/external/RULER"

echo "[1/5] Clone RULER"
mkdir -p "${ROOT_DIR}/external"
if [ ! -d "${BENCH_DIR}" ]; then
  git clone https://github.com/hsiehjackson/RULER.git "${BENCH_DIR}"
fi

echo "[2/5] Install dependencies"
python -m pip install -r "${BENCH_DIR}/requirements.txt"

echo "[3/5] Prepare model"
echo "  export MODEL_PATH=/path/to/fastmka-checkpoint"

echo "[4/5] Run passkey retrieval"
echo "Example:"
echo "  cd ${BENCH_DIR}"
echo "  python eval.py --task passkey --model_path \$MODEL_PATH --max_new_tokens 1"

echo "[5/5] Aggregate 4K/8K/16K/32K/64K/128K scores for table."
