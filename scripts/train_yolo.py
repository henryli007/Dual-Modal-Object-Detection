#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_rgbt_detection.utils.config import load_toml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default="configs/train/yolo_rgbt_fused_v1.toml")
    parser.add_argument("--data", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", default="")
    return parser.parse_args()


def main() -> int:
    from ultralytics import YOLO

    args = parse_args()
    cfg = load_toml(args.train_config)
    view = str(cfg["view"])
    data = args.data or str((PROJECT_ROOT / "data" / "processed" / "yolo" / view / "dataset.yaml").resolve())
    run_dir = PROJECT_ROOT / str(cfg["project"]) / str(cfg["name"])
    resume_from = Path(args.resume_from).expanduser() if args.resume_from else run_dir / "weights" / "last.pt"
    model_name = args.model or (str(resume_from) if args.resume and resume_from.exists() else str(cfg["model"]))
    epochs = args.epochs or int(cfg["epochs"])
    batch = args.batch_size or int(cfg["batch_size"])
    imgsz = args.image_size or int(cfg["image_size"])
    aug = cfg.get("augmentation", {})

    model = YOLO(model_name)
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=int(cfg["workers"]),
        device=args.device or cfg["device"],
        seed=int(cfg["seed"]),
        project=str(PROJECT_ROOT / str(cfg["project"])),
        name=str(cfg["name"]),
        exist_ok=True,
        patience=int(cfg["patience"]),
        cache=bool(cfg["cache"]),
        pretrained=bool(cfg["pretrained"]),
        optimizer=str(cfg["optimizer"]),
        lr0=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        close_mosaic=int(cfg["close_mosaic"]),
        resume=bool(args.resume),
        cos_lr=bool(cfg.get("cos_lr", False)),
        warmup_epochs=float(cfg.get("warmup_epochs", 3.0)),
        save_period=int(cfg.get("save_period", -1)),
        rect=bool(cfg.get("rect", False)),
        multi_scale=bool(cfg.get("multi_scale", False)),
        amp=bool(cfg.get("amp", True)),
        plots=bool(cfg.get("plots", True)),
        box=float(cfg.get("box_loss_gain", 7.5)),
        cls=float(cfg.get("cls_loss_gain", 0.5)),
        dfl=float(cfg.get("dfl_loss_gain", 1.5)),
        hsv_h=float(aug.get("hsv_h", 0.015)),
        hsv_s=float(aug.get("hsv_s", 0.7)),
        hsv_v=float(aug.get("hsv_v", 0.4)),
        degrees=float(aug.get("degrees", 0.0)),
        translate=float(aug.get("translate", 0.1)),
        scale=float(aug.get("scale", 0.5)),
        shear=float(aug.get("shear", 0.0)),
        fliplr=float(aug.get("fliplr", 0.5)),
        mosaic=float(aug.get("mosaic", 1.0)),
        mixup=float(aug.get("mixup", 0.0)),
        copy_paste=float(aug.get("copy_paste", 0.0)),
        erasing=float(aug.get("erasing", 0.4)),
        auto_augment=str(aug.get("auto_augment", "randaugment")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
