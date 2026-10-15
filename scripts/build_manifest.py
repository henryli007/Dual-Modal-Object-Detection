#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_rgbt_detection.project_paths import ensure_project_dirs
from uav_rgbt_detection.utils.io import write_json


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_split_manifest(dataset_root: Path, split: str) -> list[dict]:
    rgb_dir = dataset_root / split / "rgb"
    tir_dir = dataset_root / split / "tir"
    ann_path = dataset_root / split / f"{split}.json"

    ann_by_image_id: dict[int, list[dict]] = defaultdict(list)
    image_meta_by_id: dict[int, dict] = {}
    categories: dict[int, str] = {}

    if ann_path.exists():
        coco = load_coco(ann_path)
        categories = {item["id"]: item["name"] for item in coco["categories"]}
        image_meta_by_id = {item["id"]: item for item in coco["images"]}
        for ann in coco["annotations"]:
            ann_by_image_id[ann["image_id"]].append(ann)

    rows = []
    for rgb_path in sorted(rgb_dir.glob("*.jpg")):
        stem = rgb_path.stem
        tir_path = tir_dir / rgb_path.name
        image_id = int(stem)
        image_meta = image_meta_by_id.get(image_id, {})
        anns = ann_by_image_id.get(image_id, [])
        rows.append(
            {
                "split": split,
                "image_id": image_id,
                "file_name": rgb_path.name,
                "rgb_path": str(rgb_path),
                "tir_path": str(tir_path),
                "width": image_meta.get("width"),
                "height": image_meta.get("height"),
                "num_boxes": len(anns),
                "categories": sorted({categories.get(ann["category_id"], str(ann["category_id"])) for ann in anns}),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", default="data/interim/dataset_manifest.json")
    return parser.parse_args()


def main() -> int:
    ensure_project_dirs()
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    payload = {
        "dataset_root": str(dataset_root),
        "splits": {
            split: build_split_manifest(dataset_root, split)
            for split in ["train", "val", "test"]
        },
    }
    write_json(args.output, payload)
    print(
        {
            "output": args.output,
            "train": len(payload["splits"]["train"]),
            "val": len(payload["splits"]["val"]),
            "test": len(payload["splits"]["test"]),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
