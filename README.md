# Worker Detection in Construction Site CCTV
**SmartSites Technical Task | AI & Computer Vision Intern**

## Overview
This notebook compares vanilla YOLOv11 inference against SAHI (Slicing Aided Hyper Inference) for worker detection on construction site images, and includes DINOv2 patch-level feature visualization as an exploration of unsupervised scene understanding.

## Structure
- **Part A** — Paper review of Akyon et al., 2022 (SAHI)
- **Part B** — Proof of concept on the MOCS dataset

## Dataset
[MOCS Dataset](https://www.kaggle.com/datasets/xiaopan9802/mocs-dataset) — construction site images with workers and machinery from fixed CCTV cameras. 20 validation images used.

## Models
| Model | Weights | Purpose |
|---|---|---|
| YOLOv11n | COCO pretrained | Vanilla inference |
| YOLOv11n + SAHI | COCO pretrained | Sliced inference |
| DINOv3 ViT-B/14 | Self-supervised pretrained | Patch feature extraction |

No training involved. Pretrained weights only.

## Pipeline

### Vanilla YOLO
Full image resized to 640×640 → single forward pass → person class filtered (class 0).

### YOLO + SAHI
Image sliced into overlapping 640×640 patches → YOLO on each patch → merged back with NMS → person class filtered.

Slice size: `640×640` | Overlap: `25%` | Confidence threshold: `0.3`

### DINOv2 Feature Visualization
Patch features extracted at 980×980 resolution → L2 normalized → agglomerative clustering (cosine distance, tau controls granularity) → PCA RGB visualization.

## Results
SAHI consistently detects more workers than vanilla YOLO, particularly distant or partially occluded workers that shrink to 10–15 pixels at full image resolution. False positives appear mainly on scaffolding and machinery parts that resemble human silhouettes at crop resolution.

## Beyond SAHI
For a production pipeline:
- **DINOv2-guided SAHI** — use patch clusters to identify regions of interest before slicing, instead of uniform slicing
- **Kalman filter tracker** — maintain worker ID continuity across frames
- **Frame skipping** — run inference every N frames at 30fps to meet latency requirements

## Setup

```bash
pip install ultralytics sahi torch torchvision scikit-learn matplotlib opencv-python
```

Run on Kaggle with T4 GPU. Open `sample-task-maryam-amjad.ipynb`.

## References
- [SAHI Paper](https://arxiv.org/abs/2202.06934) — Akyon et al., 2022
- [SAHI Library](https://github.com/obss/sahi)
- [YOLOv11](https://github.com/ultralytics/ultralytics)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [MOCS Dataset](https://www.kaggle.com/datasets/xiaopan9802/mocs-dataset)
