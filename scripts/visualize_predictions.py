#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/raw/official/GAIIC2024-赛道1-目标检测任务")
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--output-dir", default="outputs/figures/predictions")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = PROJECT_ROOT / args.dataset_root
    predictions = json.loads(Path(args.prediction_json).read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for pred in predictions:
        grouped[int(pred["image_id"])].append(pred)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = {1: "car", 2: "truck", 3: "bus", 4: "van", 5: "freight_car"}
    colors = {1: "#22C55E", 2: "#F59E0B", 3: "#3B82F6", 4: "#A855F7", 5: "#EF4444"}

    ranked = sorted(grouped.items(), key=lambda kv: max(item["score"] for item in kv[1]), reverse=True)[: args.top_k]
    for image_id, preds in ranked:
        file_name = f"{image_id:05d}.jpg"
        image = Image.open(dataset_root / args.split / "rgb" / file_name).convert("RGB")
        fig, ax = plt.subplots(figsize=(6.4, 5.12))
        ax.imshow(image)
        ax.axis("off")
        for pred in preds:
            x, y, w, h = pred["bbox"]
            label = int(pred["category_id"])
            score = float(pred["score"])
            color = colors.get(label, "#FFFFFF")
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color, facecolor="none"))
            ax.text(x, max(0, y - 4), f"{categories.get(label, label)} {score:.2f}", color="white", fontsize=7, bbox={"facecolor": color, "edgecolor": color, "pad": 0.15})
        fig.savefig(output_dir / f"{file_name}", bbox_inches="tight")
        plt.close(fig)
    print({"output_dir": str(output_dir), "num_images": len(ranked)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
