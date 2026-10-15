# Task Brief

Source: `/root/计算机视觉综合实践-无人机视角下的双光目标检测.pdf`

## Problem

Detect multiple object categories from paired UAV-view RGB and thermal infrared imagery.

## Dataset facts from the PDF

- image size: `640 x 512`
- train: `17,990` RGB-T pairs with `train.json`
- val: `1,469` RGB-T pairs with `val.json`
- test: `1,000` RGB-T pairs
- annotations/results: COCO format

## Expected file organization

```text
test/
  rgb/
  tir/
train/
  rgb/
  tir/
  train.json
val/
  rgb/
  tir/
  val.json
```

## Constraints

- external datasets are disallowed unless approved
- open pretrained models are allowed, but final documentation must report source link and md5
- final deliverables include:
  - website submission material
  - experiment report within 3 pages
  - reproducible code
  - clear README
  - team contribution statement

## Engineering implications

- Pair consistency is a first-class invariant.
- All preprocessing must preserve RGB/TIR alignment.
- COCO validation must be automated early.
- Report assets should be generated from scripts, not manually assembled.
