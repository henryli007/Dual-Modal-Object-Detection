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
from uav_rgbt_detection.data.transforms import RGBTTestTransform, RGBTTrainTransform
from uav_rgbt_detection.engine.trainer import evaluate_model, train_one_epoch
from uav_rgbt_detection.models.rgbt_faster_rcnn import build_model
from uav_rgbt_detection.project_paths import ensure_project_dirs
from uav_rgbt_detection.utils.checkpoint import load_checkpoint, make_inference_state_dict, save_checkpoint
from uav_rgbt_detection.utils.config import load_toml
from uav_rgbt_detection.utils.ema import ModelEMA
from uav_rgbt_detection.utils.io import write_json
from uav_rgbt_detection.utils.logging import CSVLogger
from uav_rgbt_detection.utils.randomness import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", default="configs/dataset/gaicc2024_rgbt.toml")
    parser.add_argument("--model-config", default="configs/model/two_stream_baseline.toml")
    parser.add_argument("--train-config", default="configs/train/baseline.toml")
    parser.add_argument("--resume", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--val-subset", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=-1)
    return parser.parse_args()


def maybe_subset(dataset, subset_size: int):
    if subset_size and subset_size < len(dataset):
        return Subset(dataset, list(range(subset_size)))
    return dataset


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


def build_scheduler(optimizer, train_cfg: dict):
    scheduler_cfg = train_cfg.get("scheduler", {})
    name = str(scheduler_cfg.get("name", "step")).lower()
    epochs = int(train_cfg["epochs"])
    if name == "cosine":
        min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.05))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=float(train_cfg["learning_rate"]) * min_lr_ratio,
        )
    step_size = int(scheduler_cfg.get("step_size", max(1, epochs // 3)))
    gamma = float(scheduler_cfg.get("gamma", 0.1))
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def main() -> int:
    ensure_project_dirs()
    args = parse_args()
    dataset_cfg = load_toml(args.dataset_config)
    model_cfg = load_toml(args.model_config)
    train_cfg = load_toml(args.train_config)
    if args.epochs > 0:
        train_cfg["epochs"] = args.epochs
    if args.batch_size > 0:
        train_cfg["batch_size"] = args.batch_size
    if args.learning_rate > 0:
        train_cfg["learning_rate"] = args.learning_rate
    if args.seed >= 0:
        train_cfg["seed"] = args.seed

    dataset_root = PROJECT_ROOT / dataset_cfg["dataset_root"]
    output_dir = Path(args.output_dir or train_cfg["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(train_cfg["seed"]))
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")

    augmentation_cfg = model_cfg.get("augmentation", {})
    train_dataset = RGBTDetectionDataset(
        dataset_root,
        "train",
        transforms=RGBTTrainTransform(
            horizontal_flip_prob=float(augmentation_cfg.get("horizontal_flip", 0.5)),
            multi_scale_min=float(augmentation_cfg.get("multi_scale_min", 1.0)),
            multi_scale_max=float(augmentation_cfg.get("multi_scale_max", 1.0)),
            color_jitter_rgb=float(augmentation_cfg.get("color_jitter_rgb", 0.0)),
            tir_gain=float(augmentation_cfg.get("tir_gain", 0.0)),
        ),
    )
    val_dataset = RGBTDetectionDataset(dataset_root, "val", transforms=RGBTTestTransform())
    train_dataset = maybe_subset(train_dataset, args.train_subset)
    val_dataset = maybe_subset(val_dataset, args.val_subset)

    workers = args.num_workers if args.num_workers >= 0 else int(train_cfg["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_rgbt,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, int(train_cfg["batch_size"]) // 2),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_rgbt,
    )

    num_classes = 5 + 1
    model_kwargs = build_model_kwargs(model_cfg)
    model = build_model(
        num_classes=num_classes,
        pretrained_backbone=bool(model_cfg.get("pretrained", True)),
        **model_kwargs,
    )
    model.to(device)
    ema_decay = float(train_cfg.get("ema_decay", 0.0))
    ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0 else None
    ema_eval_model = None
    if ema is not None:
        ema_eval_model = build_model(
            num_classes=num_classes,
            pretrained_backbone=False,
            **model_kwargs,
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=bool(train_cfg["mixed_precision"]) and device.type == "cuda")
    scheduler = build_scheduler(optimizer, train_cfg)

    start_epoch = 1
    best_map = -1.0
    if args.resume:
        state = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if args.learning_rate > 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = float(args.learning_rate)
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        if ema is not None and state.get("ema") is not None:
            ema.load_state_dict(state["ema"])
        start_epoch = int(state["epoch"]) + 1
        best_map = float(state.get("best_map", -1.0))

    csv_logger = CSVLogger(
        output_dir / "training_log.csv",
        ["epoch", "train_loss", "train_steps", "skipped_steps", "val_mAP", "val_mAP50", "val_mAP75", "val_mAR100", "num_predictions", "lr"],
    )

    gt_json = dataset_root / "val" / "val.json"
    best_path = output_dir / "best_model.pth"
    last_checkpoint_path = output_dir / "last_checkpoint.pth"
    summary_path = output_dir / "summary.json"

    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler if scaler.is_enabled() else None,
            device,
            epoch,
            ema=ema,
            log_interval=int(train_cfg["logging"]["log_interval"]),
            grad_clip_norm=float(train_cfg["gradient_clip_norm"]),
        )
        model_for_eval = model
        model_state_for_save = model.state_dict()
        if ema is not None and ema_eval_model is not None:
            ema.copy_to(ema_eval_model)
            model_for_eval = ema_eval_model
            model_state_for_save = ema.state_dict()
        metrics = evaluate_model(
            model_for_eval,
            val_loader,
            device,
            gt_json,
            score_threshold=float(train_cfg["evaluation"].get("score_threshold", 0.05)),
            max_detections_per_image=100,
            tta_hflip=bool(train_cfg["evaluation"].get("tta_hflip", False)),
        )
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_steps": train_stats["steps"],
            "skipped_steps": train_stats["skipped_steps"],
            "val_mAP": metrics["mAP"],
            "val_mAP50": metrics["mAP50"],
            "val_mAP75": metrics["mAP75"],
            "val_mAR100": metrics["mAR100"],
            "num_predictions": metrics["num_predictions"],
            "lr": lr,
        }
        csv_logger.log(row)
        print(f"[val] epoch={epoch} mAP={metrics['mAP']:.4f} mAP50={metrics['mAP50']:.4f} preds={metrics['num_predictions']}")

        current_best_map = max(best_map, metrics["mAP"])
        inference_model_state = make_inference_state_dict(model_state_for_save)
        model_state = {
            "epoch": epoch,
            "model": inference_model_state,
            "best_map": current_best_map,
            "dataset_root": str(dataset_root),
        }
        checkpoint_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "ema": ema.state_dict() if ema is not None else None,
            "best_map": current_best_map,
            "dataset_root": str(dataset_root),
        }
        save_checkpoint(last_checkpoint_path, checkpoint_state)
        if metrics["mAP"] > best_map:
            best_map = current_best_map
            save_checkpoint(best_path, model_state)
        scheduler.step()

        write_json(
            summary_path,
            {
                "best_map": best_map,
                "last_epoch": epoch,
                "dataset_root": str(dataset_root),
                "output_dir": str(output_dir),
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
