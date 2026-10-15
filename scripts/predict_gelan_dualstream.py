#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gelan-root", default="third_party/gelan_rgbt")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.0001)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--detect-head", type=int, default=4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=4_950_000)
    parser.add_argument("--min-det", type=int, default=20)
    return parser.parse_args()


def clamp_box(box: list[float]) -> list[float] | None:
    x, y, w, h = [float(v) for v in box]
    x1 = max(0.0, min(x, 640.0))
    y1 = max(0.0, min(y, 512.0))
    x2 = max(0.0, min(x + w, 640.0))
    y2 = max(0.0, min(y + h, 512.0))
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    return [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)]


def compact_predictions(raw_path: Path, output_path: Path, max_det: int, max_bytes: int, min_det: int) -> tuple[int, int]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    grouped: dict[int, list[dict]] = defaultdict(list)
    for item in raw:
        if item.get("category_id") is None or item.get("bbox") is None or item.get("score") is None:
            continue
        box = clamp_box(item["bbox"])
        if box is None:
            continue
        grouped[int(item["image_id"])].append(
            {
                "image_id": int(item["image_id"]),
                "category_id": int(item["category_id"]),
                "bbox": box,
                "score": round(float(item["score"]), 5),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_max = max_det
    while current_max >= min_det:
        compact = []
        for image_id in sorted(grouped):
            compact.extend(sorted(grouped[image_id], key=lambda x: x["score"], reverse=True)[:current_max])
        output_path.write_text(json.dumps(compact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        if output_path.stat().st_size <= max_bytes:
            return len(compact), current_max
        current_max -= 5
    raise RuntimeError(f"{output_path} is still larger than {max_bytes} bytes at max_det={min_det}")


def main() -> int:
    args = parse_args()
    gelan_root = Path(args.gelan_root).resolve()
    code_root = gelan_root / "code"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    from val import run_mlab

    data_path = gelan_root / "data" / "contest_data" / args.split
    output_path = Path(args.output).resolve()
    raw_path = output_path.with_name(output_path.stem + "_raw.json")

    run_mlab(
        data=str(data_path),
        weights=str(Path(args.weights).resolve()),
        save_dir=str(raw_path),
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        task=args.split,
        device=args.device,
        workers=args.workers,
        detect_head=args.detect_head,
        augment=args.augment,
        save_json=True,
        half=True,
    )
    num_predictions, kept_per_image = compact_predictions(
        raw_path,
        output_path,
        max_det=args.max_det,
        max_bytes=args.max_bytes,
        min_det=args.min_det,
    )
    print(
        {
            "raw_output": str(raw_path),
            "output": str(output_path),
            "num_predictions": num_predictions,
            "max_detections_per_image": kept_per_image,
            "size_mb": round(output_path.stat().st_size / 1_000_000, 3),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
