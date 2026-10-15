#!/usr/bin/env python
from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"
LOG_DIR = PROJECT_ROOT / "docs" / "experiment_logs"
FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"


def setup_style() -> None:
    if FONT_PATH.exists():
        fm.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = "Noto Sans CJK SC"
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "axes.linewidth": 0.9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def read_text(path: Path) -> str:
    if path.suffix == ".gz":
        return gzip.decompress(path.read_bytes()).decode("utf-8", "ignore")
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_val_rows(path: Path) -> list[dict[str, float]]:
    text = read_text(path).replace("\r", "\n")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("all "):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 7:
            continue
        rows.append(
            {
                "index": len(rows),
                "images": int(parts[1]),
                "instances": int(parts[2]),
                "precision": float(parts[3]),
                "recall": float(parts[4]),
                "map50": float(parts[5]),
                "map5095": float(parts[6]),
            }
        )
    return rows


def save_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_gelan_training_curve(rows: list[dict[str, float]]) -> None:
    x = np.array([r["index"] for r in rows])
    map5095 = np.array([r["map5095"] for r in rows])
    map50 = np.array([r["map50"] for r in rows])
    recall = np.array([r["recall"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(x, map5095, marker="o", markersize=3.2, linewidth=2.0, color="#2563EB", label="mAP@[.50:.95]")
    ax.plot(x, map50, marker="s", markersize=3.0, linewidth=1.8, color="#16A34A", label="mAP@.50")
    ax.plot(x, recall, linewidth=1.5, color="#F97316", alpha=0.85, label="Recall")
    best = int(map5095.argmax())
    ax.scatter([best], [map5095[best]], s=58, color="#DC2626", zorder=5)
    ax.set_title("双流 GELAN 训练验证曲线")
    ax.set_xlabel("验证序号")
    ax.set_ylabel("COCO 指标")
    ax.set_ylim(0.38, 0.90)
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, "gelan_training_curve_cn")


def plot_method_iteration() -> None:
    methods = ["双流\nFaster R-CNN", "YOLO\nRGB-T", "双流\nGELAN", "GELAN\n泛化微调"]
    local = np.array([0.5865, 0.6280, 0.6410, 0.6500])
    online = np.array([0.382, 0.310, 0.450, 0.448])

    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    b1 = ax.bar(x - width / 2, local, width, color="#2563EB", label="本地验证 mAP")
    b2 = ax.bar(x + width / 2, online, width, color="#F97316", label="线上反馈 mAP")
    ax.set_title("主要方法迭代与泛化差距")
    ax.set_ylabel("COCO mAP")
    ax.set_xticks(x, methods)
    ax.set_ylim(0.25, 0.70)
    ax.legend(frameon=False, loc="upper left")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, "method_iteration_gap_cn")


def plot_gelan_online_probe() -> None:
    labels = ["早期\nprobe", "e18", "e24", "e33", "e40"]
    local = np.array([np.nan, 0.602, 0.618, 0.631, 0.641])
    online = np.array([0.424366, 0.42417, 0.434, 0.450, 0.450])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(x, online, marker="o", linewidth=2.0, color="#DC2626", label="线上反馈")
    ax.plot(x[1:], local[1:], marker="s", linewidth=2.0, color="#2563EB", label="本地验证")
    ax.set_title("GELAN 中间 checkpoint 的线上探测")
    ax.set_xlabel("导出版本")
    ax.set_ylabel("mAP")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.40, 0.67)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, "gelan_online_probe_cn")


def plot_final_submission_distribution() -> None:
    submission = PROJECT_ROOT / "deliverables" / "final_submission_gelan_dualstream_e40_under5mb.json"
    data = json.loads(submission.read_text(encoding="utf-8"))
    counter = Counter(int(item["category_id"]) for item in data)
    names = {1: "car", 2: "truck", 3: "bus", 4: "freight_car", 5: "van"}
    ids = [1, 2, 3, 4, 5]
    values = [counter[i] for i in ids]
    colors = ["#2563EB", "#16A34A", "#F97316", "#7C3AED", "#DC2626"]

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    bars = ax.bar([names[i] for i in ids], values, color=colors, width=0.62)
    ax.set_title("最终提交预测类别分布")
    ax.set_ylabel("预测框数量")
    ax.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.012, f"{value:,}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, "final_submission_category_distribution_cn")


def main() -> int:
    setup_style()
    main_log = LOG_DIR / "gelan_dualstream_main_train.log.gz"
    ft_log = LOG_DIR / "gelan_generalization_finetune.log.gz"
    main_rows = parse_val_rows(main_log)
    ft_rows = parse_val_rows(ft_log)
    save_csv(main_rows, PROJECT_ROOT / "docs" / "gelan_main_val_metrics.csv")
    save_csv(ft_rows, PROJECT_ROOT / "docs" / "gelan_finetune_val_metrics.csv")
    plot_gelan_training_curve(main_rows)
    plot_method_iteration()
    plot_gelan_online_probe()
    plot_final_submission_distribution()
    print(f"figures written to {FIGURE_DIR}")
    print(f"main validation rows: {len(main_rows)}, finetune rows: {len(ft_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
