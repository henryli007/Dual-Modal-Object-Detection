#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_rgbt_detection.engine.coco_eval import evaluate_coco
from uav_rgbt_detection.utils.config import load_toml
from uav_rgbt_detection.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default="configs/train/yolo_rgbt_fused_v1.toml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", default="")
    parser.add_argument("--confidence", type=float, default=-1.0)
    parser.add_argument("--iou", type=float, default=-1.0)
    parser.add_argument("--max-detections", type=int, default=0)
    parser.add_argument("--array-batch-size", type=int, default=8)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    return parser.parse_args()


def clip_xyxy(box, width: float = 640.0, height: float = 512.0):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(x1, width))
    y1 = max(0.0, min(y1, height))
    x2 = max(0.0, min(x2, width))
    y2 = max(0.0, min(y2, height))
    return x1, y1, x2, y2


def image_id_from_path(path: str) -> int:
    return int(Path(path).stem)


def append_result_predictions(predictions: list[dict], result, image_id: int) -> None:
    boxes = result.boxes
    if boxes is None:
        return
    for xyxy, score, cls in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
        x1, y1, x2, y2 = clip_xyxy(xyxy)
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue
        predictions.append(
            {
                "image_id": image_id,
                "category_id": int(cls) + 1,
                "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                "score": round(float(score), 6),
            }
        )


def main() -> int:
    from ultralytics import YOLO

    args = parse_args()
    cfg = load_toml(args.train_config)
    pred_cfg = cfg.get("prediction", {})
    view = str(cfg["view"])
    source = PROJECT_ROOT / "data" / "processed" / "yolo" / view / "images" / args.split
    conf = args.confidence if args.confidence >= 0 else float(pred_cfg.get("confidence", 0.001))
    iou = args.iou if args.iou >= 0 else float(pred_cfg.get("iou", 0.7))
    max_det = args.max_detections or int(pred_cfg.get("max_detections", 150))
    augment = False if args.no_tta else bool(args.tta or pred_cfg.get("tta", False))

    model = YOLO(args.weights)
    predictions = []
    if view in {"rgbt_4ch", "rgbt_6ch"}:
        npy_paths = sorted(source.glob("*.npy"))
        batch_size = max(1, int(args.array_batch_size))
        for start in range(0, len(npy_paths), batch_size):
            batch_paths = npy_paths[start : start + batch_size]
            arrays = [np.load(path) for path in batch_paths]
            results = model.predict(
                source=arrays,
                imgsz=int(cfg["image_size"]),
                conf=conf,
                iou=iou,
                max_det=max_det,
                augment=augment,
                device=cfg["device"],
                verbose=False,
            )
            for path, result in zip(batch_paths, results):
                append_result_predictions(predictions, result, image_id_from_path(str(path)))
    else:
        results = model.predict(
            source=str(source),
            imgsz=int(cfg["image_size"]),
            conf=conf,
            iou=iou,
            max_det=max_det,
            augment=augment,
            device=cfg["device"],
            verbose=False,
            stream=True,
        )
        for result in results:
            append_result_predictions(predictions, result, image_id_from_path(result.path))

    write_json(args.output, predictions)
    report = {"output": args.output, "split": args.split, "num_predictions": len(predictions)}
    if args.split == "val":
        metrics = evaluate_coco(
            PROJECT_ROOT / "data/raw/official/GAIIC2024-赛道1-目标检测任务/val/val.json",
            predictions,
        )
        metrics["num_predictions"] = len(predictions)
        report.update(metrics)
        if args.metrics_output:
            Path(args.metrics_output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
