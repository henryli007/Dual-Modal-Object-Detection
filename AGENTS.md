# AGENTS

This repository is for the UAV RGB-T object detection course project described in `/root/计算机视觉综合实践-无人机视角下的双光目标检测.pdf`.

## Working rules

1. Keep all work isolated to this repository. Do not edit unrelated projects under `/root`.
2. Treat `data/raw` as immutable once the dataset is downloaded and extracted.
3. Put all generated artifacts under `outputs/` and all runtime logs under `logs/`.
4. Prefer configuration-driven changes through `configs/*.toml` instead of hard-coded paths.
5. Preserve dual-modality assumptions everywhere:
   - RGB and TIR files must stay paired.
   - Any split, augmentation, or sampling step must preserve pair identity.
   - COCO annotations must stay traceable back to paired images.
6. Before adding a new training component, update:
   - `README.md`
   - `WORKLOG.md`
   - the relevant config file under `configs/`
7. For long-running jobs, use `nohup` or a detached shell and write a PID file to `logs/`.

## Expected project phases

1. Download and verify the official dataset.
2. Extract and index the RGB-T pairs.
3. Build a reproducible baseline.
4. Add stronger fusion, augmentation, and validation analysis.
5. Generate submission JSON and report assets.

## Directory intent

- `configs/`: dataset, model, and experiment configuration.
- `data/`: raw downloads, extracted dataset, processed artifacts, submissions.
- `docs/`: task brief, architecture notes, experiment protocol.
- `scripts/`: entry-point scripts for download, extraction, indexing, and visualization.
- `src/uav_rgbt_detection/`: reusable Python package.
- `outputs/figures/`: report-quality figures.
- `logs/`: nohup and runtime logs.
