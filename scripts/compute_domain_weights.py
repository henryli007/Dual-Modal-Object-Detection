#!/usr/bin/env python3
"""Compute test-domain similarity weights for UAV-RGBT training images.

The script uses unlabeled image statistics from RGB/TIR pairs, projects them
with PCA, then assigns larger sampling weights to training images closer to the
test split in PCA space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


def list_pairs(data_root: Path, split: str) -> list[tuple[str, Path, Path]]:
    rgb_dir = data_root / split / "rgb"
    tir_dir = data_root / split / "tir"
    pairs = []
    for rgb_path in sorted(rgb_dir.glob("*")):
        if not rgb_path.is_file():
            continue
        tir_path = tir_dir / rgb_path.name
        if tir_path.is_file():
            pairs.append((rgb_path.stem, rgb_path, tir_path))
    if not pairs:
        raise FileNotFoundError(f"no RGB/TIR pairs found for split={split} under {data_root}")
    return pairs


def percentiles(x: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(x, [1, 5, 25, 50, 75, 95, 99])]


def gray_features(gray: np.ndarray, prefix: str) -> list[tuple[str, float]]:
    gray = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    names = [
        (f"{prefix}_mean", float(gray.mean())),
        (f"{prefix}_std", float(gray.std())),
        (f"{prefix}_dark", float((gray < 0.08).mean())),
        (f"{prefix}_bright", float((gray > 0.92).mean())),
        (f"{prefix}_edge_mean", float(edge.mean())),
        (f"{prefix}_edge_std", float(edge.std())),
        (f"{prefix}_lap_var", float(lap.var())),
    ]
    names.extend((f"{prefix}_p{p}", v) for p, v in zip([1, 5, 25, 50, 75, 95, 99], percentiles(gray)))
    return names


def extract_features(rgb_path: Path, tir_path: Path, size: int) -> tuple[list[str], np.ndarray]:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    tir = cv2.imread(str(tir_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    if tir is None:
        raise FileNotFoundError(tir_path)

    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    tir = cv2.resize(tir, (size, size), interpolation=cv2.INTER_AREA)
    rgb_float = rgb.astype(np.float32) / 255.0
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)

    items: list[tuple[str, float]] = []
    for channel, name in enumerate(["b", "g", "r"]):
        c = rgb_float[:, :, channel]
        items.append((f"rgb_{name}_mean", float(c.mean())))
        items.append((f"rgb_{name}_std", float(c.std())))
        items.extend((f"rgb_{name}_p{p}", v) for p, v in zip([5, 50, 95], percentiles(c)[1:6:2]))

    items.extend(gray_features(rgb_gray, "rgb_gray"))
    items.extend(gray_features(tir, "tir"))
    items.append(("rgb_sat_mean", float((hsv[:, :, 1] / 255.0).mean())))
    items.append(("rgb_sat_std", float((hsv[:, :, 1] / 255.0).std())))
    items.append(("rgb_val_mean", float((hsv[:, :, 2] / 255.0).mean())))
    items.append(("rgb_val_std", float((hsv[:, :, 2] / 255.0).std())))

    rgb_g = rgb_gray.astype(np.float32) / 255.0
    tir_g = tir.astype(np.float32) / 255.0
    diff = np.abs(rgb_g - tir_g)
    corr = float(np.corrcoef(rgb_g.reshape(-1), tir_g.reshape(-1))[0, 1])
    if not math.isfinite(corr):
        corr = 0.0
    items.append(("rgb_tir_absdiff_mean", float(diff.mean())))
    items.append(("rgb_tir_absdiff_std", float(diff.std())))
    items.append(("rgb_tir_corr", corr))

    names = [k for k, _ in items]
    values = np.asarray([v for _, v in items], dtype=np.float64)
    return names, values


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return (x - mean) / std, mean, std


def pca_project(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    components = vt[:n_components]
    return x @ components.T, components


def knn_distance(query: np.ndarray, ref: np.ndarray, k: int, chunk: int = 512) -> np.ndarray:
    out = np.empty(query.shape[0], dtype=np.float64)
    k = min(k, ref.shape[0])
    ref_norm = np.sum(ref * ref, axis=1)[None, :]
    for start in range(0, query.shape[0], chunk):
        q = query[start:start + chunk]
        dist2 = np.sum(q * q, axis=1)[:, None] + ref_norm - 2.0 * q @ ref.T
        dist2 = np.maximum(dist2, 0.0)
        kth = np.partition(dist2, kth=k - 1, axis=1)[:, :k]
        out[start:start + chunk] = np.sqrt(kth).mean(axis=1)
    return out


def weights_from_distance(dist: np.ndarray, alpha: float, min_weight: float, max_weight: float) -> np.ndarray:
    median = float(np.median(dist))
    scale = float(np.percentile(dist, 75) - np.percentile(dist, 25))
    if scale < 1e-8:
        scale = float(np.std(dist) + 1e-6)
    similarity = np.exp(-(dist - median) / scale)
    similarity /= similarity.mean()
    weights = (1.0 - alpha) + alpha * similarity
    weights /= weights.mean()
    weights = np.clip(weights, min_weight, max_weight)
    weights /= weights.mean()
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/root/external_solutions/2024_GAIIC_rank4_object_detection/data/contest_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/UAV-RGBT-Detection/data/domain_weights"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--weight-split", default="train", help="split to export into the training JSON weights mapping")
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--knn", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--min-weight", type=float, default=0.35)
    parser.add_argument("--max-weight", type=float, default=3.0)
    args = parser.parse_args()

    rows = []
    feature_names = None
    for split in args.splits:
        pairs = list_pairs(args.data_root, split)
        for i, (image_id, rgb_path, tir_path) in enumerate(pairs, 1):
            names, feat = extract_features(rgb_path, tir_path, args.size)
            if feature_names is None:
                feature_names = names
            rows.append({
                "split": split,
                "image_id": image_id,
                "rgb_path": str(rgb_path),
                "tir_path": str(tir_path),
                "feature": feat,
            })
            if i % 1000 == 0:
                print(f"{split}: extracted {i}/{len(pairs)}")

    x = np.stack([r["feature"] for r in rows], axis=0)
    z, mean, std = standardize(x)
    n_components = min(args.pca_components, z.shape[1], z.shape[0])
    p, components = pca_project(z, n_components)

    test_mask = np.asarray([r["split"] == "test" for r in rows])
    test_p = p[test_mask]
    centroid = test_p.mean(axis=0)
    centroid_dist = np.linalg.norm(p - centroid[None, :], axis=1)
    knn_dist = knn_distance(p, test_p, args.knn)

    for i, row in enumerate(rows):
        row["centroid_dist"] = float(centroid_dist[i])
        row["knn_dist"] = float(knn_dist[i])
        row["pca"] = [float(v) for v in p[i]]

    weight_mask = np.asarray([r["split"] == args.weight_split for r in rows])
    weight_dist = np.asarray([r["knn_dist"] for r in rows if r["split"] == args.weight_split], dtype=np.float64)
    weights = weights_from_distance(weight_dist, args.alpha, args.min_weight, args.max_weight)

    weight_iter = iter(weights.tolist())
    weight_map = {}
    for row in rows:
        if row["split"] == args.weight_split:
            w = float(next(weight_iter))
            row["weight"] = w
            weight_map[row["image_id"]] = w
            weight_map[Path(row["rgb_path"]).name] = w
            weight_map[row["rgb_path"]] = w
        else:
            row["weight"] = 1.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"rank4_{args.weight_split}_to_test_pca_weights.json"
    csv_path = args.output_dir / f"rank4_{args.weight_split}_to_test_pca_weights.csv"

    payload = {
        "metadata": {
            "data_root": str(args.data_root),
            "splits": args.splits,
            "weight_split": args.weight_split,
            "feature_count": len(feature_names or []),
            "features": feature_names,
            "pca_components": int(n_components),
            "knn": int(args.knn),
            "alpha": float(args.alpha),
            "min_weight": float(args.min_weight),
            "max_weight": float(args.max_weight),
            "standardizer_mean": mean.tolist(),
            "standardizer_std": std.tolist(),
            "pca_basis": components.tolist(),
            "test_centroid": centroid.tolist(),
        },
        "weights": weight_map,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image_id", "weight", "knn_dist", "centroid_dist", "rgb_path", "tir_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    print(f"saved JSON: {json_path} ({json_path.stat().st_size / 1024 / 1024:.3f} MB)")
    print(f"saved CSV:  {csv_path} ({csv_path.stat().st_size / 1024 / 1024:.3f} MB)")
    print(
        "weights:",
        f"min={weights.min():.3f}",
        f"p10={np.percentile(weights, 10):.3f}",
        f"p50={np.percentile(weights, 50):.3f}",
        f"p90={np.percentile(weights, 90):.3f}",
        f"max={weights.max():.3f}",
    )


if __name__ == "__main__":
    main()
