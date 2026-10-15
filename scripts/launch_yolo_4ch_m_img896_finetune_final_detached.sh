#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NAME="yolo_rgbt_4ch_m_img896_finetune_final_v1"
LOG="logs/train_${NAME}.log"
PID="logs/train_${NAME}.pid"
RUN_DIR="outputs/yolo/${NAME}"
LAST="${RUN_DIR}/weights/last.pt"
BASE="outputs/yolo/yolo_rgbt_4ch_m_img704_parallel_v1/weights/best.pt"
mkdir -p logs

if [[ -s "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
  echo "already running ${NAME} pid=$(cat "$PID") log=$LOG"
  exit 0
fi

if [[ ! -s "$BASE" && ! -s "$LAST" ]]; then
  echo "missing base checkpoint: $BASE"
  exit 2
fi

ARGS=(--train-config configs/train/yolo_rgbt_4ch_m_img896_finetune_final_v1.toml)
if [[ -s "$LAST" ]]; then
  ARGS+=(--resume --resume-from "$LAST")
fi

setsid /root/miniconda3/envs/thu-bdc2026/bin/python -u scripts/train_yolo.py "${ARGS[@]}" \
  > "$LOG" 2>&1 < /dev/null &

echo $! > "$PID"
echo "started ${NAME} pid=$(cat "$PID") log=$LOG"
