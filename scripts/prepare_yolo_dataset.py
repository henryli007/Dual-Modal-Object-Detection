#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_rgbt_detection.utils.io import write_json

CATEGORY_TO_YOLO = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
NAMES = ["car", "truck", "bus", "van", "freight_car"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/raw/official/GAIIC2024-赛道1-目标检测任务")
    parser.add_argument("--output-root", default="data/processed/yolo")
    parser.add_argument("--view", choices=["rgb", "tir", "rgbt_fused", "rgbt_4ch", "rgbt_6ch"], default="rgbt_fused")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--max-images", type=int, default=0, help="Optional per-split limit for smoke checks.")
    return parser.parse_args()


def load_annotations(dataset_root: Path, split: str) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    payload = json.loads((dataset_root / split / f"{split}.json").read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in payload["images"]}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for ann in payload["annotations"]:
        if int(ann["category_id"]) in CATEGORY_TO_YOLO:
            annotations[int(ann["image_id"])].append(ann)
    return images, annotations


def ensure_clean(path: Path, rebuild: bool) -> None:
    if path.exists() and rebuild:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_image(rgb_path: Path, tir_path: Path, output_path: Path, view: str) -> None:
    if output_path.exists():
        return
    if view == "rgb":
        if not output_path.exists():
            output_path.symlink_to(rgb_path.resolve())
        return
    if view == "tir":
        tir = Image.open(tir_path).convert("RGB")
        tir.save(output_path, quality=95)
        return
    if view in {"rgbt_4ch", "rgbt_6ch"}:
        import numpy as np

        if not output_path.exists():
            output_path.symlink_to(rgb_path.resolve())
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        tir = np.asarray(Image.open(tir_path).convert("L"), dtype=np.uint8)
        if view == "rgbt_4ch":
            stacked = np.concatenate([rgb, tir[..., None]], axis=2)
        else:
            tir3 = np.repeat(tir[..., None], 3, axis=2)
            stacked = np.concatenate([rgb, tir3], axis=2)
        np.save(output_path.with_suffix(".npy"), stacked)
        return

    rgb = Image.open(rgb_path).convert("RGB")
    tir = Image.open(tir_path).convert("L")
    r, g, _ = rgb.split()
    fused = Image.merge("RGB", (r, g, tir))
    fused.save(output_path, quality=95)


def write_label(label_path: Path, anns: list[dict], width: int, height: int) -> None:
    lines = []
    for ann in anns:
        x, y, w, h = [float(value) for value in ann["bbox"]]
        if w <= 1 or h <= 1:
            continue
        cls = CATEGORY_TO_YOLO[int(ann["category_id"])]
        xc = (x + 0.5 * w) / width
        yc = (y + 0.5 * h) / height
        lines.append(f"{cls} {xc:.8f} {yc:.8f} {w / width:.8f} {h / height:.8f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def convert_labeled_split(dataset_root: Path, output_root: Path, split: str, view: str, max_images: int = 0) -> dict:
    images, annotations = load_annotations(dataset_root, split)
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    for idx, (image_id, info) in enumerate(sorted(images.items())):
        if max_images and idx >= max_images:
            break
        file_name = info["file_name"]
        width = int(info.get("width", 640))
        height = int(info.get("height", 512))
        rgb_path = dataset_root / split / "rgb" / file_name
        tir_path = dataset_root / split / "tir" / file_name
        stem = f"{image_id:05d}"
        save_image(rgb_path, tir_path, image_dir / f"{stem}.jpg", view)
        anns = annotations.get(image_id, [])
        write_label(label_dir / f"{stem}.txt", anns, width, height)
        total_boxes += len(anns)
    return {"split": split, "images": len(images), "boxes": total_boxes}


def convert_test_split(dataset_root: Path, output_root: Path, view: str, max_images: int = 0) -> dict:
    image_dir = output_root / "images" / "test"
    image_dir.mkdir(parents=True, exist_ok=True)
    rgb_paths = sorted((dataset_root / "test" / "rgb").glob("*.jpg"))
    if max_images:
        rgb_paths = rgb_paths[:max_images]
    for rgb_path in rgb_paths:
        tir_path = dataset_root / "test" / "tir" / rgb_path.name
        save_image(rgb_path, tir_path, image_dir / rgb_path.name, view)
    return {"split": "test", "images": len(rgb_paths)}


def write_dataset_yaml(output_root: Path, channels: int = 3) -> Path:
    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"channels: {channels}",
                "names:",
                *[f"  {idx}: {name}" for idx, name in enumerate(NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def main() -> int:
    args = parse_args()
    dataset_root = PROJECT_ROOT / args.dataset_root
    output_root = PROJECT_ROOT / args.output_root / args.view
    ensure_clean(output_root, args.rebuild)
    summary = {
        "view": args.view,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "splits": [
            convert_labeled_split(dataset_root, output_root, "train", args.view, args.max_images),
            convert_labeled_split(dataset_root, output_root, "val", args.view, args.max_images),
            convert_test_split(dataset_root, output_root, args.view, args.max_images),
        ],
        "yaml": str(write_dataset_yaml(output_root, channels={"rgbt_4ch": 4, "rgbt_6ch": 6}.get(args.view, 3))),
    }
    write_json(output_root / "summary.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
