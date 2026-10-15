# Architecture

## Baseline detector

The current baseline is a custom RGB-T Faster R-CNN implementation built on top of `torchvision`.

### Data flow

1. Load paired RGB and TIR images with shared COCO boxes.
2. Run two separate `ResNet-50` backbones:
   - one for RGB
   - one for TIR
3. Fuse feature maps stage by stage with lightweight gated fusion blocks.
4. Pass fused multi-scale features into an FPN.
5. Use Faster R-CNN RPN + RoI heads for final detection.

### Why this baseline

- uses both modalities directly
- keeps pretrained ImageNet initialization
- stays inside one self-contained codebase
- is easy to ablate into RGB-only, TIR-only, or stronger fusion variants

## Current code boundaries

- `src/uav_rgbt_detection/data/`
  - dataset loading
  - train/test transforms
- `src/uav_rgbt_detection/models/`
  - RGB-T backbone
  - Faster R-CNN wrapper
- `src/uav_rgbt_detection/engine/`
  - train loop
  - COCO evaluation
  - inference/export
- `scripts/`
  - CLI entry points

## Planned extension points

- stronger cross-modal fusion blocks
- better augmentation
- TTA and score calibration
- class-balanced sampling
- experiment sweeps and ensembling
