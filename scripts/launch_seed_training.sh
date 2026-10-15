#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_PREFIX="${1:-competitive_v3}"
MODEL_CONFIG="${2:-configs/model/two_stream_competitive_v3.toml}"
TRAIN_CONFIG="${3:-configs/train/competitive_v3.toml}"
SEED="${4:-3407}"

OUTPUT_DIR="outputs/models/${RUN_PREFIX}_seed${SEED}"
RUN_NAME="${RUN_PREFIX}_seed${SEED}"

scripts/launch_training_detached.sh \
  "$RUN_NAME" \
  "$MODEL_CONFIG" \
  "$TRAIN_CONFIG" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"
