#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_json", nargs="+")
    parser.add_argument("--dataset-root", default="data/raw/official/GAIIC2024-赛道1-目标检测任务")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-detections-per-image", type=int, default=100)
    return parser.parse_args()


def load_expected(dataset_root: Path, split: str) -> tuple[set[int], set[int]]:
    if split in {"train", "val"}:
        payload = json.loads((dataset_root / split / f"{split}.json").read_text(encoding="utf-8"))
        image_ids = {int(item["id"]) for item in payload["images"]}
        category_ids = {int(item["id"]) for item in payload["categories"]}
        return image_ids, category_ids

    image_ids = {int(path.stem) for path in (dataset_root / split / "rgb").glob("*.jpg")}
    category_ids = {1, 2, 3, 4, 5}
    return image_ids, category_ids


def summarize(path: Path, expected_images: set[int], expected_categories: set[int], max_det: int) -> dict:
    predictions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError(f"{path} must contain a JSON list")

    image_counts: Counter[int] = Counter()
    category_counts: Counter[int] = Counter()
    scores: list[float] = []
    invalid = 0
    out_of_bounds = 0
    bad_images: set[int] = set()
    bad_categories: set[int] = set()

    for item in predictions:
        try:
            image_id = int(item["image_id"])
            category_id = int(item["category_id"])
            x, y, w, h = [float(value) for value in item["bbox"]]
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue

        image_counts[image_id] += 1
        category_counts[category_id] += 1
        scores.append(score)
        if image_id not in expected_images:
            bad_images.add(image_id)
        if category_id not in expected_categories:
            bad_categories.add(category_id)
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > 640.5 or y + h > 512.5:
            out_of_bounds += 1

    over_limit = {image_id: count for image_id, count in image_counts.items() if count > max_det}
    missing_images = sorted(expected_images - set(image_counts))
    score_summary = {
        "min": min(scores) if scores else None,
        "median": median(scores) if scores else None,
        "max": max(scores) if scores else None,
    }
    detections_per_image = list(image_counts.values())

    return {
        "file": str(path),
        "size_mb": round(path.stat().st_size / 1_000_000, 3),
        "predictions": len(predictions),
        "covered_images": len(image_counts),
        "missing_images": len(missing_images),
        "missing_image_examples": missing_images[:10],
        "invalid_records": invalid,
        "bad_image_ids": sorted(bad_images)[:10],
        "bad_category_ids": sorted(bad_categories),
        "out_of_bounds_or_nonpositive_boxes": out_of_bounds,
        "images_over_max_detections": len(over_limit),
        "max_detections_on_one_image": max(detections_per_image) if detections_per_image else 0,
        "detections_per_image_median": median(detections_per_image) if detections_per_image else 0,
        "category_counts": dict(sorted(category_counts.items())),
        "score": score_summary,
    }


def main() -> int:
    args = parse_args()
    dataset_root = PROJECT_ROOT / args.dataset_root
    expected_images, expected_categories = load_expected(dataset_root, args.split)
    reports = [
        summarize(Path(prediction_json), expected_images, expected_categories, args.max_detections_per_image)
        for prediction_json in args.prediction_json
    ]
    print(json.dumps(reports if len(reports) > 1 else reports[0], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
