#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_rgbt_detection.project_paths import FIGURE_ROOT, ensure_project_dirs
from uav_rgbt_detection.viz.cvpr_style import apply_cvpr_style


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig: plt.Figure, stem: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def sample_ids_with_boxes(coco: dict, k: int, seed: int = 3407) -> list[int]:
    rng = random.Random(seed)
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)
    candidates = [img_id for img_id, anns in ann_by_img.items() if len(anns) >= 4]
    return sorted(rng.sample(candidates, k=min(k, len(candidates))))


def draw_boxes(ax: plt.Axes, image: np.ndarray, anns: list[dict], categories: dict[int, str], title: str) -> None:
    colors = {
        "car": "#22C55E",
        "bus": "#3B82F6",
        "truck": "#F59E0B",
        "van": "#A855F7",
        "freight_car": "#EF4444",
    }
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    for ann in anns:
        x, y, w, h = ann["bbox"]
        name = categories[ann["category_id"]]
        color = colors.get(name, "#FFFFFF")
        rect = patches.Rectangle((x, y), w, h, linewidth=1.4, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x,
            max(0, y - 4),
            name,
            color="white",
            fontsize=7,
            bbox={"facecolor": color, "edgecolor": color, "pad": 0.15},
        )


def figure_rgb_tir_samples(dataset_root: Path) -> None:
    coco = load_coco(dataset_root / "train" / "train.json")
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    image_meta = {item["id"]: item for item in coco["images"]}
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    sample_ids = sample_ids_with_boxes(coco, k=3)
    fig, axes = plt.subplots(len(sample_ids), 2, figsize=(10.4, 4.8 * len(sample_ids)))
    if len(sample_ids) == 1:
        axes = np.array([axes])

    for row, image_id in enumerate(sample_ids):
        file_name = image_meta[image_id]["file_name"]
        rgb = np.array(Image.open(dataset_root / "train" / "rgb" / file_name).convert("RGB"))
        tir = np.array(Image.open(dataset_root / "train" / "tir" / file_name).convert("RGB"))
        anns = ann_by_img[image_id]
        draw_boxes(axes[row, 0], rgb, anns, categories, f"RGB: {file_name}")
        draw_boxes(axes[row, 1], tir, anns, categories, "TIR")

    fig.suptitle("Paired RGB-T training samples with COCO boxes", fontsize=13, y=0.995)
    save(fig, "sample_pairs_with_boxes")


def figure_category_distribution(dataset_root: Path) -> None:
    coco = load_coco(dataset_root / "train" / "train.json")
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    counter = Counter(categories[ann["category_id"]] for ann in coco["annotations"])

    order = ["car", "truck", "bus", "freight_car", "van"]
    values = [counter[item] for item in order]
    colors = ["#22C55E", "#F59E0B", "#3B82F6", "#EF4444", "#A855F7"]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(order, values, color=colors, width=0.64)
    ax.set_ylabel("Number of boxes")
    ax.set_title("Training annotation distribution")
    ax.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.01, f"{value:,}", ha="center", va="bottom", fontsize=9)
    save(fig, "category_distribution")


def figure_box_size_distribution(dataset_root: Path) -> None:
    coco = load_coco(dataset_root / "train" / "train.json")
    widths = np.array([ann["bbox"][2] for ann in coco["annotations"]], dtype=float)
    heights = np.array([ann["bbox"][3] for ann in coco["annotations"]], dtype=float)
    areas = widths * heights

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    axes[0].hist(np.clip(np.sqrt(areas), 0, 160), bins=40, color="#2B6CB0", alpha=0.9)
    axes[0].set_title("Object scale distribution")
    axes[0].set_xlabel("sqrt(box area)")
    axes[0].set_ylabel("Count")

    axes[1].scatter(widths[::20], heights[::20], s=6, alpha=0.18, color="#D97706")
    axes[1].set_title("Bounding box width-height distribution")
    axes[1].set_xlabel("Width")
    axes[1].set_ylabel("Height")
    axes[1].set_xlim(0, 220)
    axes[1].set_ylim(0, 220)
    save(fig, "box_geometry_distribution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    return parser.parse_args()


def main() -> int:
    ensure_project_dirs()
    apply_cvpr_style()
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    figure_rgb_tir_samples(dataset_root)
    figure_category_distribution(dataset_root)
    figure_box_size_distribution(dataset_root)
    print(f"Report figures written to {FIGURE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
