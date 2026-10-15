#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="$ROOT_DIR/data/raw/official/GAIIC2024-赛道1-目标检测任务"

cd "$ROOT_DIR"
bash scripts/extract_dataset.sh
python scripts/index_dataset.py --dataset-root "$DATASET_ROOT"
python scripts/build_manifest.py --dataset-root "$DATASET_ROOT"
python scripts/generate_dataset_report_figures.py --dataset-root "$DATASET_ROOT"
