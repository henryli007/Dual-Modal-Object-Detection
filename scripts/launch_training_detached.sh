#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

RUN_NAME="${1:-advanced_smallobj_v2}"
MODEL_CONFIG="${2:-configs/model/two_stream_smallobj_v2.toml}"
TRAIN_CONFIG="${3:-configs/train/advanced_smallobj_v2.toml}"
EXTRA_ARGS=("${@:4}")

LOG_PATH="$LOG_DIR/${RUN_NAME}.log"
PID_PATH="$LOG_DIR/${RUN_NAME}.pid"

: > "$LOG_PATH"

setsid bash -lc "
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /root/miniconda3/etc/profile.d/conda.sh
conda activate thu-bdc2026
cd '$ROOT_DIR'
exec python -u scripts/train_detector.py --model-config '$MODEL_CONFIG' --train-config '$TRAIN_CONFIG' ${EXTRA_ARGS[*]}
" >> "$LOG_PATH" 2>&1 < /dev/null &

echo $! > "$PID_PATH"
echo "pid=$!"
echo "log=$LOG_PATH"
