#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_rgbt_detection.data.rgbt_coco import RGBTDetectionDataset, collate_rgbt
from uav_rgbt_detection.data.transforms import RGBTTestTransform
from uav_rgbt_detection.engine.trainer import evaluate_model
from uav_rgbt_detection.models.rgbt_faster_rcnn import build_model
from uav_rgbt_detection.utils.config import load_toml
from uav_rgbt_detection.utils.checkpoint import load_checkpoint
from uav_rgbt_detection.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/raw/official/GAIIC2024-赛道1-目标检测任务")
    parser.add_argument("--model-config", default="configs/model/two_stream_baseline.toml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--max-detections-per-image", type=int, default=100)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-scales", nargs="*", type=float, default=[1.0])
    parser.add_argument("--tta-nms-iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def parse_anchor_sizes(model_cfg: dict) -> tuple[tuple[int, ...], ...] | None:
    anchor_sizes = model_cfg.get("anchor_sizes")
    if not anchor_sizes:
        return None
    return tuple((int(size),) for size in anchor_sizes)


def build_model_kwargs(model_cfg: dict) -> dict:
    return {
        "anchor_sizes": parse_anchor_sizes(model_cfg),
        "fusion_variant": str(model_cfg.get("fusion_variant", "basic")),
        "neck_variant": str(model_cfg.get("neck_variant", "fpn")),
    }


def main() -> int:
    args = parse_args()
    dataset_root = PROJECT_ROOT / args.dataset_root
    model_cfg = load_toml(args.model_config)
    dataset = RGBTDetectionDataset(dataset_root, args.split, transforms=RGBTTestTransform())
    if args.subset and args.subset < len(dataset):
        dataset = Subset(dataset, list(range(args.subset)))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_rgbt,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=6,
        pretrained_backbone=False,
        **build_model_kwargs(model_cfg),
    )
    state = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    metrics = evaluate_model(
        model,
        loader,
        device,
        dataset_root / args.split / f"{args.split}.json",
        score_threshold=args.score_threshold,
        max_detections_per_image=args.max_detections_per_image,
        tta_hflip=args.tta_hflip,
        tta_scales=tuple(args.tta_scales),
        tta_nms_iou_threshold=args.tta_nms_iou_threshold,
    )
    print(metrics)
    if args.output:
        write_json(args.output, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
