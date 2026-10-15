#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python scripts/download_baidu_share.py \
  --share-url "https://pan.baidu.com/s/1sJi4BJCjK0iaqg1TfnQimA?pwd=gvci" \
  --password "gvci" \
  --output-dir "data/raw"
