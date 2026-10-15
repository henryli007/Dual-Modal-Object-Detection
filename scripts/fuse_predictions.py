#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import torch
from torchvision.ops import batched_nms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_rgbt_detection.utils.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-json", nargs="+", required=True)
    parser.add_argument("--weights", nargs="*", type=float, default=[])
    parser.add_argument("--method", choices=["nms", "wbf"], default="wbf")
    parser.add_argument("--iou-threshold", type=float, default=0.55)
    parser.add_argument("--skip-box-threshold", type=float, default=0.001)
    parser.add_argument("--max-detections-per-image", type=int, default=100)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def to_xywh(box):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), 640.0))
    y1 = max(0.0, min(float(y1), 512.0))
    x2 = max(0.0, min(float(x2), 640.0))
    y2 = max(0.0, min(float(y2), 512.0))
    return [round(x1, 1), round(y1, 1), round(max(0.0, x2 - x1), 1), round(max(0.0, y2 - y1), 1)]


def box_iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def fuse_wbf(items, weights, iou_threshold: float):
    clusters: list[dict] = []
    for item in sorted(items, key=lambda x: x["weighted_score"], reverse=True):
        matched = False
        for cluster in clusters:
            if item["category_id"] != cluster["category_id"]:
                continue
            if box_iou(item["box_xyxy"], cluster["box_xyxy"]) >= iou_threshold:
                cluster["members"].append(item)
                total_weight = sum(member["weight"] * member["score"] for member in cluster["members"])
                coords = [0.0, 0.0, 0.0, 0.0]
                for member in cluster["members"]:
                    contribution = member["weight"] * member["score"]
                    for idx, value in enumerate(member["box_xyxy"]):
                        coords[idx] += contribution * value
                cluster["box_xyxy"] = [value / max(total_weight, 1e-12) for value in coords]
                cluster["score"] = sum(member["score"] * member["weight"] for member in cluster["members"]) / sum(
                    member["weight"] for member in cluster["members"]
                )
                matched = True
                break
        if not matched:
            clusters.append(
                {
                    "category_id": item["category_id"],
                    "box_xyxy": list(item["box_xyxy"]),
                    "score": item["score"],
                    "members": [item],
                }
            )
    fused = []
    for cluster in clusters:
        fused.append(
            {
                "category_id": cluster["category_id"],
                "box_xyxy": cluster["box_xyxy"],
                "score": round(float(cluster["score"]), 6),
            }
        )
    return fused


def fuse_nms(items, iou_threshold: float):
    boxes = torch.tensor([item["box_xyxy"] for item in items], dtype=torch.float32)
    scores = torch.tensor([item["weighted_score"] for item in items], dtype=torch.float32)
    labels = torch.tensor([item["category_id"] for item in items], dtype=torch.int64)
    keep = batched_nms(boxes, scores, labels, iou_threshold).tolist()
    return [
        {
            "category_id": items[idx]["category_id"],
            "box_xyxy": items[idx]["box_xyxy"],
            "score": round(float(items[idx]["score"]), 6),
        }
        for idx in keep
    ]


def main() -> int:
    args = parse_args()
    prediction_paths = [Path(path) for path in args.prediction_json]
    weights = args.weights or [1.0] * len(prediction_paths)
    if len(weights) != len(prediction_paths):
        raise ValueError("weights length must match prediction-json length")

    grouped: dict[int, list[dict]] = defaultdict(list)
    for path, weight in zip(prediction_paths, weights):
        for item in read_json(path):
            if float(item["score"]) < args.skip_box_threshold:
                continue
            grouped[int(item["image_id"])].append(
                {
                    "image_id": int(item["image_id"]),
                    "category_id": int(item["category_id"]),
                    "box_xyxy": to_xyxy(item["bbox"]),
                    "score": float(item["score"]),
                    "weight": float(weight),
                    "weighted_score": float(item["score"]) * float(weight),
                }
            )

    fused_predictions = []
    for image_id, items in grouped.items():
        if args.method == "wbf":
            fused = fuse_wbf(items, weights, args.iou_threshold)
        else:
            fused = fuse_nms(items, args.iou_threshold)
        fused = sorted(fused, key=lambda x: x["score"], reverse=True)[: args.max_detections_per_image]
        for item in fused:
            if item["box_xyxy"][2] <= item["box_xyxy"][0] or item["box_xyxy"][3] <= item["box_xyxy"][1]:
                continue
            fused_predictions.append(
                {
                    "image_id": image_id,
                    "category_id": item["category_id"],
                    "bbox": to_xywh(item["box_xyxy"]),
                    "score": item["score"],
                }
            )

    write_json(args.output, fused_predictions)
    print({"output": args.output, "num_predictions": len(fused_predictions)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
