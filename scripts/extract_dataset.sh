#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT_DIR/data/raw"
OUT_DIR="$RAW_DIR/official"

ARCHIVE="$(find "$RAW_DIR" -maxdepth 1 -type f -name '*.tar.gz' | head -n 1)"

if [[ -z "${ARCHIVE:-}" ]]; then
  echo "No .tar.gz archive found under $RAW_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
tar -xzf "$ARCHIVE" -C "$OUT_DIR"
echo "Extracted $ARCHIVE to $OUT_DIR"
