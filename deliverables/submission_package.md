# Submission Package

## Final JSON

Recommended final submission:

```text
deliverables/final_submission_gelan_dualstream_e40_under5mb.json
```

Equivalent project path:

```text
deliverables/final_submission_gelan_dualstream_e40_under5mb.json
```

Audit result:

- size: `4.697 MB`
- predictions: `58,410`
- covered test images: `1000/1000`
- missing images: `0`
- invalid records: `0`
- invalid category ids: `0`
- out-of-bounds or non-positive boxes: `0`
- max detections on one image: `60`

## Representative Ablation Files

The following JSON files are kept as experiment references:

```text
data/submissions/submission_final_rgbt_v5_advanced_selfwbf.json
data/submissions/submission_yolo11m_rgbt4ch_img704_best_map0p628_iou070_th0p002_notta_under5mb.json
data/submissions/submission_fused_competitive_v3_advanced_yolox_w025_iou060_th0p002_under5mb.json
deliverables/final_submission_gelan_dualstream_e40_under5mb.json
```

## Method Summary

The project explored three stages:

1. dual-stream Faster R-CNN with RGB/TIR gated fusion, TTA, and WBF;
2. YOLO RGB-T lines, including 3-channel fused input and 4-channel `RGB + TIR`;
3. final dual-stream GELAN detector.

The final GELAN submission was chosen because it had the best online feedback among stable candidates. Later generalization fine-tuning improved local validation mAP but reduced online score, so it was not retained.

## Model and Data Packaging

Large training checkpoints, YOLO derived caches, raw prediction files, and logs were removed before packaging. The final submission JSON is self-contained for website scoring.

To reproduce training, prepare the official dataset under:

```text
data/raw/official/GAIIC2024-赛道1-目标检测任务
```

Then follow `README.md` and `WORKLOG.md`.

## Pretrained Weights Declaration

No extra external training dataset was used. Open-source pretrained weights used during experiments:

| Weight | MD5 |
| --- | --- |
| `resnet50-11ad3fa6.pth` | `012571d812f34f8442473d8b827077b5` |
| `gelan-e.pt` | `5b4cf9356e206bf5621e2f4e41120d4b` |
| `yolov9-e.pt` | `0be4b088452771e1349fc368e0956f19` |
| `yolo11m.pt` | used in YOLO ablations; not retained after cleanup |
