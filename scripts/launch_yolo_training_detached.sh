#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${1:-yolo_rgbt_fused_m_v1}"
TRAIN_CONFIG="${2:-configs/train/yolo_rgbt_fused_v1.toml}"
LOG_PATH="logs/train_${RUN_NAME}.log"
PID_PATH="logs/train_${RUN_NAME}.pid"

mkdir -p logs
nohup bash -lc "
source /root/miniconda3/etc/profile.d/conda.sh &&
conda activate thu-bdc2026 &&
cd '$ROOT_DIR' &&
python scripts/train_yolo.py --train-config '$TRAIN_CONFIG'
" > "$LOG_PATH" 2>&1 &
echo "$!" > "$PID_PATH"
echo "started ${RUN_NAME} pid=$(cat "$PID_PATH") log=${LOG_PATH}"
