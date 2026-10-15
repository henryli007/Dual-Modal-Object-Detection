#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG="logs/train_yolo_rgbt_4ch_m_v1.log"
PID="logs/train_yolo_rgbt_4ch_m_v1.pid"
RUN_DIR="outputs/yolo/yolo_rgbt_4ch_m_v1"
LAST="$RUN_DIR/weights/last.pt"
mkdir -p logs

if [[ -s "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "already running yolo_rgbt_4ch_m_v1 pid=$(cat "$PID") log=$LOG"
  exit 0
fi

ARGS=(--train-config configs/train/yolo_rgbt_4ch_m_v1.toml)
if [[ -s "$LAST" ]]; then
  ARGS+=(--resume --resume-from "$LAST")
fi

nohup /root/miniconda3/envs/thu-bdc2026/bin/python -u scripts/train_yolo.py "${ARGS[@]}" \
  > "$LOG" 2>&1 &

echo $! > "$PID"
echo "started yolo_rgbt_4ch_m_v1 pid=$(cat "$PID") log=$LOG"
