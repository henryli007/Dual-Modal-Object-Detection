#!/usr/bin/env python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uav_rgbt_detection.project_paths import FIGURE_ROOT, ensure_project_dirs
from uav_rgbt_detection.viz.cvpr_style import apply_cvpr_style


def save(fig: plt.Figure, stem: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_dataset_overview() -> None:
    splits = ["Train", "Val", "Test"]
    counts = np.array([17990, 1469, 1000])
    colors = ["#2B6CB0", "#2F855A", "#D69E2E"]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.bar(splits, counts, color=colors, width=0.62)
    ax.set_ylabel("Paired RGB-T samples")
    ax.set_title("Dataset split overview")
    ax.set_ylim(0, 20000)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 350, f"{value:,}", ha="center", va="bottom")
    ax.text(
        0.98,
        0.96,
        "Resolution: 640 x 512\nAnnotations: COCO",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#D9E2EC", "boxstyle": "round,pad=0.3"},
    )
    save(fig, "dataset_overview")


def figure_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    boxes = [
        (3, 16, 16, 8, "#E6F0FF", "RGB Input"),
        (3, 6, 16, 8, "#FFF4E5", "TIR Input"),
        (24, 16, 16, 8, "#DBEAFE", "RGB Encoder"),
        (24, 6, 16, 8, "#FDE68A", "TIR Encoder"),
        (46, 11, 16, 8, "#E9D8FD", "Cross-modal\nFusion"),
        (68, 11, 12, 8, "#D9F99D", "FPN"),
        (84, 11, 12, 8, "#FECACA", "Detection\nHead"),
    ]

    for x, y, w, h, color, label in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#334155", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)

    arrows = [
        ((19, 20), (24, 20)),
        ((19, 10), (24, 10)),
        ((40, 20), (46, 15)),
        ((40, 10), (46, 15)),
        ((62, 15), (68, 15)),
        ((80, 15), (84, 15)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": "#334155"})

    ax.text(50, 28, "Proposed two-stream RGB-T detection baseline", ha="center", va="center", fontsize=12)
    ax.text(50, 2.5, "The scaffold is ready for RGB-only, TIR-only, and fusion ablations.", ha="center", va="center", fontsize=9, color="#475569")
    save(fig, "proposed_pipeline")


def figure_modality_roles() -> None:
    labels = ["Daylight texture", "Low-light robustness", "Thermal contrast", "Small-object context", "Background suppression"]
    rgb = np.array([0.92, 0.28, 0.18, 0.71, 0.45])
    tir = np.array([0.38, 0.91, 0.95, 0.58, 0.80])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    rgb_vals = np.concatenate([rgb, rgb[:1]])
    tir_vals = np.concatenate([tir, tir[:1]])

    fig = plt.figure(figsize=(5.2, 4.8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, rgb_vals, color="#2563EB", label="RGB", linewidth=2.2)
    ax.fill(angles, rgb_vals, color="#2563EB", alpha=0.16)
    ax.plot(angles, tir_vals, color="#F59E0B", label="TIR", linewidth=2.2)
    ax.fill(angles, tir_vals, color="#F59E0B", alpha=0.16)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Complementary modality strengths", pad=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10))
    save(fig, "modality_complementarity")


def main() -> int:
    ensure_project_dirs()
    apply_cvpr_style()
    figure_dataset_overview()
    figure_pipeline()
    figure_modality_roles()
    print(f"Figures written to {FIGURE_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
