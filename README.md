# Evidence Timeline Reconstruction System
### UCF-Crime Dataset · 5-Group Classification · 5-Model Comparative Analysis

A deep learning pipeline that processes CCTV/surveillance video, detects criminal activity frame-by-frame, and reconstructs a timestamped evidence timeline. Trains and compares five architectures spanning 3D video CNNs, 2D CNNs, Vision Transformers, and temporal RNNs on the UCF-Crime dataset.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Class Grouping Strategy](#3-class-grouping-strategy)
4. [Model Architectures](#4-model-architectures)
5. [Results](#5-results)
6. [Project Structure](#6-project-structure)
7. [Setup & Installation](#7-setup--installation)
8. [Usage](#8-usage)
9. [Configuration](#9-configuration)
10. [XAI — Explainability](#10-xai--explainability)
11. [Demo & Video Annotation](#11-demo--video-annotation)
12. [Key Findings](#12-key-findings)

---

## 1. Project Overview

| | |
|---|---|
| **Task** | Multi-class anomaly detection + evidence timeline reconstruction |
| **Dataset** | UCF-Crime (1,100 videos, 14 crime classes) |
| **Grouping** | 14 classes → 5 semantic groups |
| **Models** | 5 architectures (3D CNN, 2D CNN, ViT, Temporal RNN) |
| **Hardware** | NVIDIA RTX 4070 (12 GB VRAM) |
| **Framework** | PyTorch 2.x + CUDA |
| **XAI** | GradCAM + LIME on all 5 models |

**Pipeline steps:**
```
Raw Videos → Frame Extraction → Dataset Splits → Training →
Evaluation → XAI → Timeline Reconstruction → Annotated Video Demo
```

---

## 2. Dataset

**UCF-Crime** — a large-scale anomaly detection benchmark containing real-world CCTV footage.

| Split | Videos | Frames (@ 3 fps) |
|---|---|---|
| Train | 770 | ~107,800 |
| Validation | 159 | ~22,300 |
| Test | 171 | ~21,400 |
| **Total** | **1,100** | **~143,650** |

**14 Original Classes:**
`Abuse · Arrest · Arson · Assault · Burglary · Explosion · Fighting · RoadAccidents · Robbery · Shooting · Shoplifting · Stealing · Vandalism · Normal`

**Preprocessing:**
- Frames extracted at **3 fps**, resized to **224 × 224**
- Optical flow computed (Farneback method)
- Stratified 70 / 15 / 15 train / val / test split
- Class weights computed (inverse frequency) to handle imbalance

---

## 3. Class Grouping Strategy

14 fine-grained classes are merged into **5 semantic groups** to address class imbalance (some classes have only 50 videos) and improve training stability.

| Group | Original Classes | Videos |
|---|---|---|
| **Normal** | Normal | 150 |
| **Theft** | Robbery, Stealing, Burglary, Shoplifting | 400 |
| **Violence** | Abuse, Assault, Fighting | 150 |
| **Hazard** | Arson, Explosion, Vandalism, RoadAccidents | 300 |
| **Enforcement** | Arrest, Shooting | 100 |

**Rationale:** Groups are formed by visual similarity and temporal patterns — a model learning "Theft" generalises across stealing behaviours, whereas learning "Shoplifting" separately from 50 examples does not.

---

## 4. Model Architectures

### Model 1 — R(2+1)D-18 (3D Video CNN)
- **Backbone:** R(2+1)D-18, Kinetics-400 pretrained
- **Input:** (B, 16, 3, 112, 112) — 16 consecutive frames (dense sampling)
- **Head:** Global average pool → Dropout(0.5) → Linear(512, 5)
- **Training:** Video-level, differential LR (backbone × 0.1)
- **Rationale:** Factorised 3D convolutions decompose spatial + temporal learning; motion is the primary crime signal in CCTV footage.

### Model 2 — EfficientNet-B3 (2D CNN, Frozen)
- **Backbone:** EfficientNet-B3, ImageNet pretrained, **fully frozen**
- **Input:** (B, 20, 3, 224, 224) — 20 frames sparse-sampled per video
- **Head:** Simple linear probe (1536 → 5), Dropout(0.5)
- **Training:** Frame-level (each frame = independent sample), head only
- **Rationale:** Tests whether frozen ImageNet features transfer to surveillance footage.

### Model 3 — ConvNeXt-Tiny (Modern 2D CNN, Frozen)
- **Backbone:** ConvNeXt-Tiny, ImageNet pretrained, **fully frozen**
- **Input:** Same as EfficientNet-B3
- **Head:** Linear probe (768 → 5)
- **Rationale:** Modern CNN design with depthwise convolutions and layer norm; comparison with EfficientNet.

### Model 4 — Swin Transformer-Tiny (Vision Transformer, Frozen)
- **Backbone:** Swin-T, ImageNet pretrained, **fully frozen**
- **Input:** (B, 20, 3, 224, 224), batch_size=128
- **Head:** Linear probe (768 → 5)
- **Rationale:** Hierarchical ViT with shifted-window attention; tests self-attention feature transfer.

### Model 5 — CNN-LSTM (Temporal RNN)
- **Backbone:** ResNet-50, pretrained, layers 3+4 unfrozen
- **Temporal head:** 2-layer BiLSTM (hidden=512) + class head (Linear → 5)
- **Input:** (B, 16, 3, 224, 224) — 16 frames per video
- **Training:** Video-level (same speed as 2D models)
- **Rationale:** Explicit temporal modelling via LSTM; captures sequential crime patterns.

---

## 5. Results

### Test Set Performance (5 Classes, 171 Videos)

| Model | Architecture | Test Acc | F1 Macro | Precision | Recall | ROC-AUC | Epochs | Train Time |
|---|---|---|---|---|---|---|---|---|
| **CNNLSTM** | ResNet50 + BiLSTM | **43.9%** | 0.3901 | 0.4383 | 0.4441 | 0.7221 | 33 | 27 min |
| R2Plus1D | 3D Video CNN | 39.8% | **0.3884** | **0.4425** | **0.4733** | **0.7517** | 37 | 2h 10m |
| SwinT | Vision Transformer | 29.3% | 0.3153 | 0.4067 | 0.3776 | 0.7028 | 27 | 32 min |
| ConvNeXtTiny | Modern 2D CNN | 28.4% | 0.2881 | 0.4187 | 0.3721 | 0.6961 | 58 | 44 min |
| EfficientNetB3 | Compound-scaled CNN | 21.7% | 0.2169 | 0.3521 | 0.3464 | 0.6618 | 40 | 30 min |

> **Baseline (random):** 20% (5 classes, uniform prior)

### Validation Performance (Best Val F1 During Training)

| Model | Best Val F1 | Best Val Acc | Best Epoch |
|---|---|---|---|
| **CNNLSTM** | **0.4848** | **54.1%** | 19 |
| R2Plus1D | 0.3931 | 44.0% | 23 |
| ConvNeXtTiny | 0.3150 | 29.6% | 44 |
| SwinT | 0.3097 | 28.9% | 13 |
| EfficientNetB3 | 0.2468 | 25.1% | 26 |

---

## 6. Project Structure

```
evidence_timeline_reconstruction/
│
├── configs/
│   └── config.yaml                  ← All hyperparameters, model configs, paths
│
├── data/
│   ├── raw/                         ← UCF-Crime videos (14 class folders)
│   │   ├── Abuse/ · Arrest/ · ... · Normal/
│   ├── processed/
│   │   ├── frames/                  ← Extracted JPEG frames @ 3 fps
│   │   ├── optical_flow/            ← Farneback flow maps
│   │   └── segments/                ← Segmentation overlays + stats
│   └── splits/
│       ├── splits.json              ← Train/val/test video lists
│       ├── class_weights.json       ← Inverse-frequency class weights
│       ├── summary.json             ← Dataset statistics
│       └── Temporal_Anomaly_Annotation_for_Testing_Videos.txt
│
├── models/
│   ├── weights/                     ← Best model weights (*_best.pt)
│   │   ├── CNNLSTM_best.pt
│   │   ├── R2Plus1D_best.pt
│   │   ├── EfficientNetB3_best.pt
│   │   ├── ConvNeXtTiny_best.pt
│   │   └── SwinT_best.pt
│   └── checkpoints/                 ← Resume checkpoints (*_checkpoint.pt)
│
├── notebooks/
│   └── EDA_Visualization.ipynb      ← Dataset EDA (distribution, samples, augmentation)
│
├── src/
│   ├── data/
│   │   ├── dataset.py               ← VideoLevelDataset, SequenceDataset, DataLoaders
│   │   ├── preprocess.py            ← Frame extraction, optical flow, splits
│   │   └── segmentation.py          ← Watershed / GrabCut / Contour segmentation
│   ├── models/
│   │   └── model_builder.py         ← All 5 model classes + build_model() factory
│   ├── training/
│   │   └── train.py                 ← Training loop (AMP, mixup, early stopping, tqdm)
│   └── evaluation/
│       ├── evaluate.py              ← Test metrics, confusion matrix, ROC, HTML report
│       ├── xai.py                   ← GradCAM + LIME on all 5 models
│       ├── model_summary.py         ← Per-model summary + comparative HTML dashboard
│       └── timeline_reconstruct.py  ← Frame-level anomaly timeline reconstruction
│
├── results/
│   ├── logs/                        ← Training history JSON (loss, acc, F1 per epoch)
│   ├── metrics/                     ← Test set metrics JSON per model + combined
│   ├── plots/                       ← Learning curves, confusion matrices, comparatives
│   ├── xai/                         ← GradCAM overlays, LIME maps, histograms (per model)
│   ├── timeline/                    ← Timeline reconstruction outputs
│   ├── demo/                        ← Annotated video demo outputs
│   ├── model_summary/               ← Per-model summary JSON + HTML dashboard
│   ├── index.html                   ← Master navigation dashboard
│   └── evaluation_report.html       ← Full comparative evaluation report
│
├── demo.py                          ← Single-video anomaly detection + annotated MP4
├── run_all.py                       ← Full pipeline runner (skip-train supported)
└── requirements.txt
```

---

## 7. Setup & Installation

### Requirements
- Python 3.9+
- NVIDIA GPU with CUDA (RTX 3060+ recommended; tested on RTX 4070 12 GB)
- 35+ GB disk space (UCF-Crime raw + processed frames)
- 16+ GB RAM

### Install

```bash
# Clone / navigate to project
cd evidence_timeline_reconstruction

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
pip install -r requirements.txt
```

### Place UCF-Crime Videos

Download UCF-Crime from the official source and organize as:

```
data/raw/
  Abuse/         ← ~50 .mp4 files
  Arrest/        ← ~50 .mp4 files
  Arson/
  Assault/
  Burglary/
  Explosion/
  Fighting/
  RoadAccidents/
  Robbery/
  Shooting/
  Shoplifting/
  Stealing/
  Vandalism/
  Normal/        ← ~150 .mp4 files
```

---

## 8. Usage

### Full Pipeline (from scratch)

```bash
# Step 1 — Preprocess: extract frames, build splits, compute class weights
python src/data/preprocess.py --skip-flow

# Step 2 — Train all 5 models
python src/training/train.py

# Step 3 — Evaluate all models on test set
python src/evaluation/evaluate.py

# Step 4 — Generate GradCAM + LIME for all models
python src/evaluation/xai.py --config configs/config.yaml --all-models --num-samples 20

# Step 5 — Model summary dashboard
python src/evaluation/model_summary.py

# Step 6 — Run demo on a random video
python demo.py --config configs/config.yaml
```

### Run Everything (post-training)

```bash
python run_all.py --skip-train
```

Runs evaluation → XAI → timeline → demo → model summary → EDA notebook → master dashboard in one command.

### Train a Single Model

```bash
python src/training/train.py --model CNNLSTM --config configs/config.yaml
python src/training/train.py --model R2Plus1D --config configs/config.yaml
```

### Resume Training from Checkpoint

```bash
python src/training/train.py --model R2Plus1D --resume
```

### Demo on a Specific Video

```bash
# Random video (different every run)
python demo.py --config configs/config.yaml

# Specific video
python demo.py --video data/raw/Robbery/Robbery001_x264.mp4 --config configs/config.yaml

# Use a specific model
python demo.py --config configs/config.yaml --model CNNLSTM

# Adjust detection sensitivity
python demo.py --config configs/config.yaml --threshold 0.4 --smooth 15
```

Demo outputs (saved to `results/demo/<video_name>/`):
- `*_annotated.mp4` — Full video at original fps with ANOMALY/NORMAL banner, class label, confidence, anomaly score bar, per-class confidence panel
- `*_timeline.png` — 4-panel timeline (class confidence, anomaly probability, predicted class, detection bar)
- `*_report.html` — HTML report with embedded video player + segment table
- `*_result.json` — Machine-readable segment data

---

## 9. Configuration

All hyperparameters live in `configs/config.yaml`. Key sections:

```yaml
dataset:
  train_split: 0.70
  val_split:   0.15
  test_split:  0.15

frames:
  fps: 3          # extraction fps
  img_size: 224

training:
  epochs: 60
  early_stopping_patience: 15
  learning_rate: 1e-3
  weight_decay: 4e-3
  batch_size: 8          # global default (overridden per model)
  num_workers: 6
  prefetch_factor: 4
  mixed_precision: true  # AMP — FP16 forward, FP32 gradients
  scheduler: "cosine"

models:
  - name: "CNNLSTM"
    video_level: true    # routes through VideoLevelDataset (~100 batches/epoch)
    batch_size: 16
    ...
```

### GPU Optimizations (applied at runtime)

```python
torch.backends.cudnn.benchmark    = True   # auto-tune CUDA kernels
torch.backends.cuda.matmul.allow_tf32 = True  # tensor cores (RTX 30/40)
torch.backends.cudnn.allow_tf32   = True
```

---

## 10. XAI — Explainability

Two explanation methods are applied to all 5 trained models:

### GradCAM (Gradient-weighted Class Activation Maps)
- Hooks the last Conv2d (2D models) or Conv3d (3D models) layer
- Computes gradient-weighted activation maps
- Overlays a colour heatmap on the original frame showing **which spatial regions** drove the prediction
- For frame-level models (EfficientNet/ConvNeXt/Swin): averages across the frame batch dimension
- For CNNLSTM: LSTM layers set to train mode before backward (cuDNN requirement)

### LIME (Local Interpretable Model-agnostic Explanations)
- SLIC superpixel segmentation of the input frame
- 80 perturbations (superpixel on/off masks) per sample
- Chunked inference (4 at a time) to avoid CUDA OOM
- Highlights which image **regions** (superpixels) contribute positively to the prediction

### XAI Outputs (`results/xai/<ModelName>/`)
```
gradcam/     ← 20 GradCAM overlay images per model
lime/        ← 20 LIME explanation images per model
histograms/  ← Per-class pixel intensity histograms
```

---

## 11. Demo & Video Annotation

`demo.py` runs the full evidence reconstruction pipeline on a single video:

```
Input Video
    ↓ Extract frames @ 3 fps (tmp dir)
    ↓ Batch inference → (N, 5) softmax probabilities
    ↓ Temporal smoothing (moving average, window=7)
    ↓ Hysteresis class stabilisation (10-frame lock)
    ↓ Anomaly segment extraction (min_len=3 frames)
    ↓ Write annotated MP4 @ original fps
    ↓ Generate timeline PNG + HTML report
    ↓ Display live in OpenCV window (press Q to stop)
```

**Annotation overlay (burned onto every frame):**
- Top banner: `ANOMALY` (red) or `NORMAL` (green)
- Centre: stabilised class label + confidence %
- Right: anomaly score (0.00–1.00)
- Bottom-left: timestamp (MM:SS)
- Bottom edge: thin anomaly score progress bar
- Right panel: live per-class confidence bar chart (display only, not saved)

**Hysteresis:** prevents rapid flickering between visually similar classes (e.g. Hazard ↔ Violence). A new class must hold dominance for 10 consecutive inference frames before the displayed label switches.

---

## 12. Key Findings

### 1. Temporal RNN outperforms frozen 2D CNNs
CNNLSTM (ResNet-50 + BiLSTM) achieved the highest test accuracy (43.9%) despite being the simplest temporal model. Frozen ImageNet backbones (EfficientNet, ConvNeXt, Swin) barely exceeded the 20% random baseline, confirming that **ImageNet features do not transfer well to low-resolution surveillance footage**.

### 2. R2Plus1D has the highest ROC-AUC (0.7517)
Despite lower accuracy, R2Plus1D's motion-based features produce the best ranking/discrimination. It uses dense consecutive-frame sampling specifically to capture motion between adjacent frames — the primary anomaly signal in CCTV.

### 3. Overfitting is the dominant failure mode
All models show a significant train/val gap:

| Model | Train Loss (final) | Val Loss (final) | Gap |
|---|---|---|---|
| CNNLSTM | 0.41 | 2.05 | 1.64 |
| R2Plus1D | 0.56 | 2.30 | 1.74 |
| ConvNeXtTiny | 0.89 | 1.78 | 0.89 |

Root cause: ~55–150 videos per class is insufficient for fine-tuning large backbones (34M+ parameters).

### 4. Group-level classification is more stable than 14-class
With ~55 videos per fine-grained class, 14-class training is dominated by noise. Merging into 5 semantic groups yields 100–400 videos/class, providing sufficient signal for generalisation.

### 5. Training time vs. performance trade-off
R2Plus1D required **2h 10m** to train but achieved comparable test F1 to CNNLSTM which trained in **27 minutes**. For resource-constrained settings, CNNLSTM is the preferred model.

---

## Outputs Reference

| File / Folder | Description |
|---|---|
| `results/logs/*.json` | Per-epoch: loss, accuracy, F1, epoch time |
| `results/metrics/*.json` | Test set: accuracy, F1, precision, recall, ROC-AUC |
| `results/plots/*_learning_curves.png` | Train/val loss + metric curves |
| `results/plots/*_confusion_matrix.png` | Normalised confusion matrix |
| `results/plots/comparative_*.png` | Cross-model bar charts, radar, ROC, heatmap |
| `results/xai/*/gradcam/` | GradCAM heatmap overlays |
| `results/xai/*/lime/` | LIME superpixel explanations |
| `results/model_summary/model_summary.html` | Comparative dashboard (all models) |
| `results/evaluation_report.html` | Full evaluation HTML report |
| `results/demo/*/annotated.mp4` | Annotated video with live predictions |

---

## Citation

```
UCF-Crime Dataset:
Sultani, W., Chen, C., & Shah, M. (2018).
Real-world anomaly detection in surveillance videos.
CVPR 2018. https://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf
```
