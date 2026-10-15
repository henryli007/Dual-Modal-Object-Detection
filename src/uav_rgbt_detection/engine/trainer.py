from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms

from uav_rgbt_detection.engine.coco_eval import evaluate_coco


def move_targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    moved = []
    for target in targets:
        current = {}
        for key, value in target.items():
            if torch.is_tensor(value):
                current[key] = value.to(device)
            else:
                current[key] = value
        moved.append(current)
    return moved


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scaler,
    device: torch.device,
    epoch: int,
    ema=None,
    log_interval: int = 50,
    grad_clip_norm: float | None = None,
) -> dict:
    model.train()
    running_loss = 0.0
    steps = 0
    skipped_steps = 0
    start = time.time()

    for step, (rgbs, tirs, targets, _) in enumerate(loader, start=1):
        rgbs = [img.to(device, non_blocking=True) for img in rgbs]
        tirs = [img.to(device, non_blocking=True) for img in tirs]
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            loss_dict = model(rgbs, tirs, targets)
            loss = sum(loss_dict.values())
        finite_loss = torch.isfinite(loss.detach())
        finite_parts = all(torch.isfinite(value.detach()).all() for value in loss_dict.values())
        if not bool(finite_loss) or not finite_parts:
            skipped_steps += 1
            debug_parts = {key: float(value.detach().float().cpu()) for key, value in loss_dict.items()}
            print(
                f"[warn] epoch={epoch} step={step}/{len(loader)} non-finite loss detected; "
                f"skipping update loss_dict={debug_parts}",
                flush=True,
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if not torch.isfinite(grad_norm):
                    skipped_steps += 1
                    print(
                        f"[warn] epoch={epoch} step={step}/{len(loader)} non-finite grad norm; skipping update",
                        flush=True,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if not torch.isfinite(grad_norm):
                    skipped_steps += 1
                    print(
                        f"[warn] epoch={epoch} step={step}/{len(loader)} non-finite grad norm; skipping update",
                        flush=True,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
            optimizer.step()
        if ema is not None:
            ema.update(model)

        running_loss += float(loss.item())
        steps += 1
        if step % log_interval == 0:
            elapsed = time.time() - start
            print(
                f"[train] epoch={epoch} step={step}/{len(loader)} "
                f"loss={running_loss / steps:.4f} elapsed={elapsed/60:.1f}m",
                flush=True,
            )

    return {"loss": running_loss / max(1, steps), "steps": steps, "skipped_steps": skipped_steps}


@torch.inference_mode()
def _run_model_once(model, rgbs: list[torch.Tensor], tirs: list[torch.Tensor]):
    return model(rgbs, tirs)


@torch.inference_mode()
def _resize_batch(images: list[torch.Tensor], scale: float) -> list[torch.Tensor]:
    if math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-6):
        return images
    resized = []
    for image in images:
        height, width = image.shape[-2:]
        new_height = max(32, int(round(height * scale)))
        new_width = max(32, int(round(width * scale)))
        resized_image = F.interpolate(
            image.unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        resized.append(resized_image)
    return resized


@torch.inference_mode()
def _rescale_outputs(outputs, scale: float):
    if math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-6):
        return outputs
    inv_scale = 1.0 / scale
    scaled_outputs = []
    for output in outputs:
        current = {key: value.clone() for key, value in output.items()}
        current["boxes"] = current["boxes"] * inv_scale
        scaled_outputs.append(current)
    return scaled_outputs


@torch.inference_mode()
def _flip_outputs_horizontally(outputs, image_widths: list[int]):
    flipped_outputs = []
    for output, width in zip(outputs, image_widths):
        current = {key: value.clone() for key, value in output.items()}
        boxes = current["boxes"]
        x1 = width - boxes[:, 2]
        x2 = width - boxes[:, 0]
        current["boxes"][:, 0] = x1
        current["boxes"][:, 2] = x2
        flipped_outputs.append(current)
    return flipped_outputs


@torch.inference_mode()
def _merge_tta_outputs(outputs_a, outputs_b, max_detections_per_image: int, nms_iou_threshold: float = 0.5):
    merged = []
    for first, second in zip(outputs_a, outputs_b):
        boxes = torch.cat([first["boxes"], second["boxes"]], dim=0)
        scores = torch.cat([first["scores"], second["scores"]], dim=0)
        labels = torch.cat([first["labels"], second["labels"]], dim=0)
        keep = batched_nms(boxes, scores, labels, nms_iou_threshold)
        keep = keep[:max_detections_per_image]
        merged.append(
            {
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep],
            }
        )
    return merged


@torch.inference_mode()
def run_inference(
    model,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.05,
    max_detections_per_image: int = 100,
    tta_hflip: bool = False,
    tta_scales: tuple[float, ...] = (1.0,),
    tta_nms_iou_threshold: float = 0.5,
) -> list[dict]:
    model.eval()
    predictions: list[dict] = []
    for rgbs, tirs, targets, _ in loader:
        rgbs = [img.to(device, non_blocking=True) for img in rgbs]
        tirs = [img.to(device, non_blocking=True) for img in tirs]
        image_widths = [int(img.shape[-1]) for img in rgbs]
        merged_outputs = None
        for scale in tta_scales:
            scaled_rgbs = _resize_batch(rgbs, scale)
            scaled_tirs = _resize_batch(tirs, scale)
            outputs = _run_model_once(model, scaled_rgbs, scaled_tirs)
            outputs = _rescale_outputs(outputs, scale)
            if merged_outputs is None:
                merged_outputs = outputs
            else:
                merged_outputs = _merge_tta_outputs(
                    merged_outputs,
                    outputs,
                    max_detections_per_image=max_detections_per_image,
                    nms_iou_threshold=tta_nms_iou_threshold,
                )
            if tta_hflip:
                flipped_rgbs = [torch.flip(img, dims=[2]) for img in scaled_rgbs]
                flipped_tirs = [torch.flip(img, dims=[2]) for img in scaled_tirs]
                flipped_outputs = _run_model_once(model, flipped_rgbs, flipped_tirs)
                flipped_outputs = _flip_outputs_horizontally(
                    _rescale_outputs(flipped_outputs, scale),
                    image_widths,
                )
                merged_outputs = _merge_tta_outputs(
                    merged_outputs,
                    flipped_outputs,
                    max_detections_per_image=max_detections_per_image,
                    nms_iou_threshold=tta_nms_iou_threshold,
                )
        outputs = merged_outputs if merged_outputs is not None else []
        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].detach().cpu()
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()
            keep = scores >= score_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
            if len(scores) > max_detections_per_image:
                boxes = boxes[:max_detections_per_image]
                scores = scores[:max_detections_per_image]
                labels = labels[:max_detections_per_image]
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                width = float(target["orig_size"][1].item()) if "orig_size" in target else float(image_widths[0])
                height = float(target["orig_size"][0].item()) if "orig_size" in target else float(rgbs[0].shape[-2])
                x1 = max(0.0, min(float(x1), width))
                y1 = max(0.0, min(float(y1), height))
                x2 = max(0.0, min(float(x2), width))
                y2 = max(0.0, min(float(y2), height))
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width <= 0 or box_height <= 0:
                    continue
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label.item()),
                        "bbox": [round(x1, 1), round(y1, 1), round(box_width, 1), round(box_height, 1)],
                        "score": round(float(score.item()), 6),
                    }
                )
    return predictions


def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    gt_json_path: str | Path,
    score_threshold: float = 0.05,
    max_detections_per_image: int = 100,
    tta_hflip: bool = False,
    tta_scales: tuple[float, ...] = (1.0,),
    tta_nms_iou_threshold: float = 0.5,
) -> dict:
    predictions = run_inference(
        model,
        loader,
        device,
        score_threshold=score_threshold,
        max_detections_per_image=max_detections_per_image,
        tta_hflip=tta_hflip,
        tta_scales=tta_scales,
        tta_nms_iou_threshold=tta_nms_iou_threshold,
    )
    metrics = evaluate_coco(gt_json_path, predictions)
    metrics["num_predictions"] = len(predictions)
    return metrics
