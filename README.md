# UAV-RGBT-Detection

无人机视角 RGB-T 双光目标检测课程项目。目标是在官方训练集/验证集上训练检测器，并对 `1000` 对测试集 RGB/TIR 图像导出 COCO detection result JSON。

## 任务与数据

- 图像尺寸：`640 x 512`
- 训练集：`17,990` 对 RGB-T 图像，COCO 标注
- 验证集：`1,469` 对 RGB-T 图像，COCO 标注
- 测试集：`1,000` 对 RGB-T 图像，无公开标注
- 类别：`car`, `truck`, `bus`, `freight_car`, `van`
- 最终提交：单个 COCO 结果 JSON，字段包含 `image_id/category_id/bbox/score`

本项目未使用额外外部数据集。训练中使用了开源预训练模型初始化，见“预训练模型声明”。

## 项目结构

```text
UAV-RGBT-Detection/
├── configs/                 # 数据、模型、训练配置
├── data/                    # 指向 /root/autodl-tmp/UAV-RGBT-Detection/data 的符号链接
│   └── submissions/         # 保留的代表性提交 JSON
├── docs/                    # 任务要求、报告草稿、方法说明
├── scripts/                 # 数据准备、训练、评估、推理、审计脚本
├── src/uav_rgbt_detection/  # 双流 Faster R-CNN 代码
├── WORKLOG.md               # 实验过程记录
└── README.md
```

大权重、派生 YOLO 缓存、raw 预测文件和训练日志已清理，避免服务器磁盘被占满。官方原始数据保留在 `data/raw/official/GAIIC2024-赛道1-目标检测任务`。

## 环境

当前实验环境：

```bash
conda activate thu-bdc2026
```

主要依赖：

- Python 3.12
- PyTorch / torchvision
- pycocotools
- ultralytics
- numpy, Pillow, matplotlib

安装本项目本地包：

```bash
cd /root/UAV-RGBT-Detection
pip install -e .
```

## 实验路线

### 1. 非 YOLO 双流检测器

首先实现了 RGB/TIR 成对读取、同步变换和 COCO 评估，构建双流 Faster R-CNN：

- RGB 分支和 TIR 分支分别提取特征
- 使用门控融合与 FPN 进行多尺度检测
- 加入小目标增强、TTA 和 WBF 后处理

代表结果：

- 本地 val COCO mAP 约 `0.5865`
- 代表提交：`data/submissions/submission_final_rgbt_v5_advanced_selfwbf.json`

### 2. YOLO RGB-T 路线

随后尝试 YOLO 系列：

- 3 通道 RGB-T 融合：RGB 中两个通道加 TIR，兼容常规预训练权重
- 4 通道 RGB-T：使用 `RGB + TIR` `.npy` 输入，保留更多模态信息
- 尝试不同输入尺寸、TTA 和后处理阈值

4 通道 YOLO 在本地 val mAP 可达到约 `0.628`，但线上测试泛化明显较弱，单独提交约 `0.31`，因此最终没有作为主方案。

代表提交：

- `data/submissions/submission_yolo11m_rgbt4ch_img704_best_map0p628_iou070_th0p002_notta_under5mb.json`

### 3. 双流 GELAN 检测器

最后采用更强的 GELAN 检测结构进行 RGB-T 双流建模：

- RGB/TIR 双输入检测头
- 使用官方训练集训练，不使用额外外部数据
- 修复了当前环境中的 NumPy/Pillow 兼容问题
- 训练时严格控制 checkpoint，只保留 `best.pt/last.pt`
- 多次导出中间 checkpoint 进行线上验证，发现本地 val 与线上 test 存在明显分布差异

最终选择本地 val `mAP50-95=0.641` 的 checkpoint 导出的提交文件作为当前最优候选：

- `deliverables/final_submission_gelan_dualstream_e40_under5mb.json`

### 4. 负结果与结论

实验中还尝试了两类方法，但最终未采用：

- 基于当前 GELAN checkpoint 的强增强泛化微调：本地 val 从 `0.641` 提升到 `0.649/0.650`，但线上分数下降。
- 输出后处理搜索：减少框数、提高阈值会削弱召回；轻微类别分数校准只有约 `0.0001~0.0002` 的提升，收益太小。

因此最终结论是：本项目当前最可靠提交是 GELAN 双流检测器 e40 导出的 JSON，而不是继续追逐本地 val mAP。

## 关键命令

### 数据索引与统计

```bash
python scripts/index_dataset.py --dataset-root data/raw/official/GAIIC2024-赛道1-目标检测任务
python scripts/build_manifest.py --dataset-root data/raw/official/GAIIC2024-赛道1-目标检测任务
python scripts/generate_dataset_report_figures.py --dataset-root data/raw/official/GAIIC2024-赛道1-目标检测任务
```

### 训练双流 Faster R-CNN

```bash
python scripts/train_detector.py --train-config configs/train/advanced_smallobj_v2.toml
```

### 准备 4 通道 YOLO 数据

```bash
python scripts/prepare_yolo_dataset.py --view rgbt_4ch
```

### 训练 4 通道 YOLO

```bash
python scripts/train_yolo.py --train-config configs/train/yolo_rgbt_4ch_m_img704_parallel_v1.toml
```

### 导出 GELAN 双流检测结果

将 GELAN 双流检测代码放在 `third_party/gelan_rgbt/`，或通过 `--gelan-root` 指定路径。该目录需包含推理入口 `code/val.py` 和整理后的 `data/contest_data/{val,test}`。

```bash
python scripts/predict_gelan_dualstream.py \
  --gelan-root third_party/gelan_rgbt \
  --weights outputs/gelan_dualstream_e40/best.pt \
  --split test \
  --output deliverables/final_submission_gelan_dualstream_e40_under5mb.json \
  --batch-size 1 \
  --conf-thres 0.0001 \
  --iou-thres 0.7 \
  --detect-head 4
```

### 审计提交文件

```bash
python scripts/audit_submission.py \
  deliverables/final_submission_gelan_dualstream_e40_under5mb.json \
  --split test --max-detections-per-image 100
```

## 当前保留的提交文件

```text
deliverables/final_submission_gelan_dualstream_e40_under5mb.json
data/submissions/submission_fused_competitive_v3_advanced_yolox_w025_iou060_th0p002_under5mb.json
data/submissions/submission_yolo11m_rgbt4ch_img704_best_map0p628_iou070_th0p002_notta_under5mb.json
data/submissions/submission_final_rgbt_v5_advanced_selfwbf.json
```

最终推荐提交：

```text
deliverables/final_submission_gelan_dualstream_e40_under5mb.json
```

该文件审计结果：

- 文件大小：`4.697 MB`
- 预测数：`58,410`
- 覆盖测试图片：`1000/1000`
- 单图最多预测：`60`
- 无非法类别、无越界框、无非正宽高框

## 训练记录与报告图

训练过程中的关键日志和曲线数据保存在：

```text
docs/experiment_logs/gelan_dualstream_main_train.log.gz
docs/experiment_logs/gelan_generalization_finetune.log.gz
docs/gelan_main_val_metrics.csv
docs/gelan_finetune_val_metrics.csv
```

报告图位于 `docs/figures/`，均提供 PNG 和 PDF：

```text
gelan_training_curve_cn
method_iteration_gap_cn
gelan_online_probe_cn
final_submission_category_distribution_cn
```

重新生成图表：

```bash
python scripts/generate_experiment_figures.py
```

## 预训练模型声明

本项目未使用额外外部检测数据集。使用过的开源预训练权重如下，权重文件不随最终代码包提交：

| 权重 | 来源 | MD5 |
| --- | --- | --- |
| `resnet50-11ad3fa6.pth` | torchvision ResNet-50 | `012571d812f34f8442473d8b827077b5` |
| `gelan-e.pt` | GELAN/YOLOv9 开源预训练权重 | `5b4cf9356e206bf5621e2f4e41120d4b` |
| `yolov9-e.pt` | YOLOv9 开源预训练权重 | `0be4b088452771e1349fc368e0956f19` |
| `yolo11m.pt` | Ultralytics YOLO11m 预训练权重 | 中间实验使用，清理后未随项目保留 |

## 清理说明

为了便于代码提交和服务器再次打开，已删除：

- 训练权重和 checkpoint
- YOLO 派生 `.npy` 数据缓存
- raw 预测 JSON
- smoke/失败候选提交
- 大训练日志
- Python `__pycache__`

如需复现实验，需要重新下载或生成对应权重与派生数据。
