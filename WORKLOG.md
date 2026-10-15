# WORKLOG

## 2026-06-14

- Created a standalone project under `/root/UAV-RGBT-Detection`.
- Read the task PDF and extracted the core constraints:
  - paired RGB-T detection
  - COCO annotations
  - `640 x 512` images
  - train/val/test counts from the PDF
- Verified the Baidu share metadata and confirmed the published archive:
  - `GAIIC2024-赛道1-目标检测任务.tar.gz`
  - about `5.15 GB`
- Implemented repository scaffolding, task brief, configs, and scripts.
- Added a direct dataset downloader with a browser-token fallback for Baidu share.
- Added dataset indexing and CVPR-style figure generation scripts.
- Verified that the current anonymous Baidu share flow returns an encrypted payload instead of a plain `dlink`, so the archive metadata is saved but the final file download is still blocked without an account-backed path or an official client flow.

## 2026-06-16

- Confirmed the local archive exists at `data/raw/GAIIC2024-赛道1-目标检测任务.tar.gz`.
- Extracted the official dataset under `data/raw/official/GAIIC2024-赛道1-目标检测任务`.
- Validated split sizes:
  - train RGB/TIR: `17,990 / 17,990`
  - val RGB/TIR: `1,469 / 1,469`
  - test RGB/TIR: `1,000 / 1,000`
- Validated annotation counts:
  - train boxes: `316,409`
  - val boxes: `24,490`
- Confirmed the dataset contains `5` categories:
  - `car`, `truck`, `bus`, `freight_car`, `van`
- Added `build_manifest.py` to generate a dataset-level manifest.
- Added `generate_dataset_report_figures.py` to generate report figures from real RGB-T samples and annotations.
- Updated the dataset config root to the real extracted path.

## 2026-06-16 Training stack

- Installed `torchvision` and `pycocotools` into the existing `thu-bdc2026` conda environment.
- Implemented a complete RGB-T detection pipeline:
  - paired COCO dataset loader
  - shared RGB/TIR transforms
  - dual-stream ResNet-50 + gated fusion + FPN backbone
  - Faster R-CNN detector wrapper
  - COCO evaluation
  - validation and test inference export
- Added runnable scripts:
  - `train_detector.py`
  - `evaluate_detector.py`
  - `predict_test.py`
  - `visualize_predictions.py`
- Added shell wrappers:
  - `run_train.sh`
  - `run_eval.sh`
  - `run_predict.sh`
- Added `docs/architecture.md`.
- Ran smoke checks:
  - forward/loss computation on real samples
  - one-epoch miniature training run
  - validation evaluation
  - test submission export
- Refined checkpoint handling:
  - lightweight `best_model.pth`
  - resumable `last_checkpoint.pth`

## Current status

- The project is runnable end to end on the local official dataset:
  - train
  - resume
  - validate
  - export test submission JSON
  - visualize predictions
- Current validated submission-ready result:
  - advanced small-object config + self-ensemble WBF
  - val `mAP=0.586521`
  - final JSON prepared at `data/submissions/submission_final_rgbt_v5_advanced_selfwbf.json`
- Started a stronger training line:
  - model config `configs/model/two_stream_smallobj_v2.toml`
  - train config `configs/train/advanced_smallobj_v2.toml`
- Remaining work is experimental rather than structural:
  - tune the baseline
  - run ablations
  - package the final report and submission artifacts

## 2026-06-29 YOLO training line

- Added a second detector family based on Ultralytics YOLO.
- Added YOLO-format data preparation:
  - `scripts/prepare_yolo_dataset.py`
  - views: `rgb`, `tir`, `rgbt_fused`
  - derived data root: `data/processed/yolo/<view>/`
- Added YOLO train/export entry points:
  - `scripts/train_yolo.py`
  - `scripts/predict_yolo.py`
- Added experiment configs:
  - `configs/train/yolo_rgbt_fused_v1.toml`
  - `configs/train/yolo_rgb_v1.toml`
  - `configs/train/yolo_tir_v1.toml`
- The `rgbt_fused` view uses RGB `R/G` plus TIR as the third channel. This keeps
  compatibility with 3-channel pretrained YOLO weights while injecting thermal
  information.
- Installed `ultralytics` in the `thu-bdc2026` environment.
- Ran the first YOLO11m `rgbt_fused` training pass:
  - 20 epochs
  - image size `640`
  - batch size `16`
  - best Ultralytics internal `mAP50-95=0.63165`
  - unified COCO no-TTA `mAP=0.5822916741933056`
  - unified COCO TTA `mAP=0.5919447032568088`
- Exported YOLO test candidates:
  - `data/submissions/submission_yolo_rgbt_fused_m_v1_tta_max100.json`
  - `data/submissions/submission_yolo_rgbt_fused_m_v1_tta_max150.json`
- Tested WBF between the YOLO line and the current Faster R-CNN ensemble.
  - Best tested cross-framework WBF val `mAP=0.5915675592382932`.
  - This did not beat the Faster R-CNN two-model WBF val `mAP=0.6023439540474061`.
- Fine-tuned the YOLO11m `rgbt_fused` checkpoint:
  - config `configs/train/yolo_rgbt_fused_m_v1_finetune.toml`
  - initialized from `outputs/yolo/yolo_rgbt_fused_m_v1/weights/best.pt`
  - 15 fine-tune epochs, image size `640`, batch size `16`
  - best Ultralytics internal `mAP50-95=0.63941`
  - unified COCO TTA `mAP=0.6080596504668253`
  - unified COCO TTA `mAP50=0.8012227820925757`
  - unified COCO TTA `mAP75=0.7292524777534056`
- Exported the current recommended test submission:
  - `data/submissions/submission_yolo_rgbt_fused_m_v1_finetune_tta_max100.json`
  - `68,004` predictions
  - covers all `1,000` test images
  - no invalid category ids, non-positive boxes, or out-of-bounds boxes
- Swept post-export confidence thresholds on the fine-tuned YOLO validation
  predictions.
  - `0.001` remained best with `mAP=0.6080596504668253`
  - higher thresholds from `0.002` to `0.05` gradually reduced mAP

## 2026-06-30 to 2026-07-02 Dual-stream GELAN line

- Added a stronger RGB-T detector family based on a dual-stream GELAN-style
  detection network. This line was introduced after the Faster R-CNN and YOLO
  experiments had largely saturated.
- Fixed two environment compatibility issues found during training:
  - NumPy AP computation now uses `np.trapezoid` when available and falls back to
    `np.trapz` on older versions.
  - Pillow text rendering now uses `font.getbbox()` when `font.getsize()` is not
    available.
- Hardened long-running training:
  - checkpoints were written to `/root/autodl-tmp` instead of the smaller root
    filesystem;
  - per-epoch checkpoint saving was disabled;
  - only `best.pt` and `last.pt` were kept during training;
  - interrupted runs could resume from the latest checkpoint.
- In parallel, upgraded the YOLO branch from a 3-channel fused baseline to a
  4-channel `RGB + TIR` training view. The 4-channel YOLO reached about `0.628`
  local validation mAP, but its standalone public-test score was only about
  `0.31`, so it was kept as an ablation instead of the final method.
- Exported GELAN checkpoints at multiple training stages for online probing:
  - early probe: online `0.424366`;
  - local `mAP50-95=0.602`: online `0.42417`;
  - local `mAP50-95=0.618`: online improved by about `0.01`;
  - local `mAP50-95=0.631`: online reached about `0.45`.
- Continued training to a local validation checkpoint with
  `mAP50-95=0.641` and `mAP50=0.843`. This checkpoint produced the final
  selected JSON:
  - `deliverables/final_submission_gelan_dualstream_e40_under5mb.json`
  - `58,410` predictions
  - `4.697 MB`
  - covers all `1,000` test images
  - no invalid categories or out-of-bounds/non-positive boxes
- Tested two follow-up directions after the e40 checkpoint:
  - generalization fine-tuning with stronger augmentation and low learning rate;
  - post-export threshold, max-detection, and category-score calibration.
- Generalization fine-tuning increased local validation mAP to `0.649~0.650`,
  but public-test feedback decreased, so this direction was stopped and the
  generated checkpoints were removed.
- Post-export calibration produced only about `0.0001~0.0002` public-test
  improvement. The gain was too small to justify keeping the search artifacts,
  so the temporary candidates were deleted.
- Before packaging, removed heavy or nonessential artifacts:
  - training checkpoints;
  - YOLO derived `.npy` caches;
  - raw prediction JSON files;
  - smoke submissions and failed candidates;
  - long training logs;
  - Python bytecode caches.
