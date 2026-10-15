#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_rgbt_detection.data.coco_checks import summarize_coco
from uav_rgbt_detection.project_paths import ensure_project_dirs
from uav_rgbt_detection.utils.io import write_json


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def build_summary(dataset_root: Path) -> dict:
    train_rgb = dataset_root / "train" / "rgb"
    train_tir = dataset_root / "train" / "tir"
    val_rgb = dataset_root / "val" / "rgb"
    val_tir = dataset_root / "val" / "tir"
    test_rgb = dataset_root / "test" / "rgb"
    test_tir = dataset_root / "test" / "tir"

    summary = {
        "dataset_root": str(dataset_root),
        "counts": {
            "train_rgb": count_images(train_rgb),
            "train_tir": count_images(train_tir),
            "val_rgb": count_images(val_rgb),
            "val_tir": count_images(val_tir),
            "test_rgb": count_images(test_rgb),
            "test_tir": count_images(test_tir),
        },
        "annotations": {},
    }

    for split in ["train", "val"]:
        ann_path = dataset_root / split / f"{split}.json"
        if ann_path.exists():
            summary["annotations"][split] = summarize_coco(ann_path)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", default="data/interim/dataset_summary.json")
    return parser.parse_args()


def main() -> int:
    ensure_project_dirs()
    args = parse_args()
    summary = build_summary(Path(args.dataset_root).resolve())
    write_json(args.output, summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
