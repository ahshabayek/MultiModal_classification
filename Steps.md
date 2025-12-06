# ViLBERT Hateful Memes Classification: Setup & Training Guide

This document provides step-by-step instructions for setting up, training, and evaluating ViLBERT models for the Hateful Memes Challenge.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Preparation](#2-data-preparation)
3. [Weight Downloads](#3-weight-downloads)
4. [Training Pipelines](#4-training-pipelines)
5. [Evaluation & Inference](#5-evaluation--inference)
6. [Troubleshooting](#6-troubleshooting)
7. [Resume Training](#7-resume-training)

---

## 1. Environment Setup

### 1.1 Prerequisites

- Python 3.9+
- CUDA 11.x+ (for GPU training)
- 16GB+ GPU memory recommended (32GB for VG features)

### 1.2 Install Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd MultiModal_classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Kedro
pip install kedro kedro-viz
```

### 1.3 Verify Installation

```bash
# Check Kedro installation
kedro info

# List available pipelines
kedro registry list
```

---

## 2. Data Preparation

### 2.1 Download Hateful Memes Dataset

The dataset is automatically downloaded from HuggingFace when running the data processing pipeline:

```bash
kedro run --pipeline=data_processing
```

This will:
1. Download the Hateful Memes dataset from HuggingFace
2. Extract images to `data/01_raw/hateful_memes/`
3. Create train/val/test splits

### 2.2 Manual Download (Alternative)

If automatic download fails:

```bash
# Install HuggingFace datasets
pip install datasets

# Download manually
python -c "
from datasets import load_dataset
ds = load_dataset('facebook/hateful_memes')
ds.save_to_disk('data/01_raw/hateful_memes_hf')
"
```

### 2.3 Verify Data

```bash
# Check data directory
ls -la data/01_raw/hateful_memes/

# Expected structure:
# data/01_raw/hateful_memes/
# ├── img/
# │   ├── 00001.png
# │   ├── 00002.png
# │   └── ...
# ├── train.jsonl
# ├── dev_seen.jsonl
# └── test_seen.jsonl
```

---

## 3. Weight Downloads

### 3.1 Facebook ViLBERT Pretrained Weights

Required for: `vilbert_frcnn_train`, `vilbert_vg_train`, `vilbert_lmdb_train`

```bash
# Create weights directory
mkdir -p weights

# Download Facebook's Conceptual Captions pretrained weights
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin \
     -O weights/vilbert_pretrained_cc.bin
```

### 3.2 Visual Genome Faster R-CNN Weights

Required for: `vilbert_vg_train`

```bash
# Download from Google Drive (or use gdown)
pip install gdown
gdown 18n_3V1rywgeADZ3oONO0DsuuS9eMW6sN -O weights/faster_rcnn_res101_vg.pth

# Alternative: Download manually from
# https://drive.google.com/file/d/18n_3V1rywgeADZ3oONO0DsuuS9eMW6sN/view
```

### 3.3 Facebook's Official LMDB Features (Optional)

Required for: `vilbert_lmdb_train` (closest to Facebook baseline)

```bash
# Download from MMF (large file ~10GB)
mkdir -p data/03_features/mmf

# Follow MMF instructions:
# https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes

# The file should be placed at:
# data/03_features/mmf/detectron.lmdb
```

### 3.4 Verify Weights

```bash
# Check weights directory
ls -la weights/

# Expected files:
# weights/
# ├── vilbert_pretrained_cc.bin      (~450MB)
# └── faster_rcnn_res101_vg.pth      (~500MB)
```

---

## 4. Training Pipelines

### 4.1 Quick Start: Default Pipeline

Best for initial experiments with ResNet-152 grid features:

```bash
# Full training pipeline (data processing + training)
kedro run --pipeline=vilbert_train

# Expected time: ~2-4 hours on V100
# Expected AUROC: 0.62-0.66
```

### 4.2 Faster R-CNN (COCO) Pipeline

Uses object detection features from COCO-pretrained detector:

```bash
# Requires: Facebook weights
kedro run --pipeline=vilbert_frcnn_train

# Expected time: ~4-6 hours on V100
# Expected AUROC: 0.62-0.65
```

### 4.3 Visual Genome Pipeline (Recommended)

Closest to Facebook's setup with VG-pretrained detector:

```bash
# Requires: Facebook weights + VG Faster R-CNN weights
kedro run --pipeline=vilbert_vg_train

# Expected time: ~12-24 hours on V100 (83 epochs)
# Expected AUROC: 0.68-0.72
```

### 4.4 LMDB Pipeline (Facebook Match)

Uses Facebook's exact precomputed features:

```bash
# Requires: Facebook weights + detectron.lmdb
kedro run --pipeline=vilbert_lmdb_train

# Expected time: ~4-6 hours on V100
# Expected AUROC: ~0.70
```

### 4.5 Precomputed Features Pipeline

For custom precomputed features in HDF5 format:

```bash
# First, extract features
python scripts/extract_features.py \
    --extractor fasterrcnn_vg \
    --output data/03_features/vg_features_100.h5 \
    --max_regions 100

# Then train
kedro run --pipeline=vilbert_precomputed_train
```

### 4.6 Pipeline Configuration

Edit `conf/base/parameters.yml` to customize:

```yaml
# Example: Change learning rate and epochs
training:
  learning_rate: 1.0e-5  # Lower for less overfitting
  num_epochs: 50         # More epochs
  batch_size: 16         # Reduce if OOM

# Example: Change feature extractor
vilbert:
  feature_extractor: "clip"  # Options: resnet, clip, fasterrcnn, fasterrcnn_vg
  max_regions: 36
```

---

## 5. Evaluation & Inference

### 5.1 Validate Trained Model

```bash
# Validate on test set
kedro run --pipeline=vilbert_validate

# Results saved to:
# data/05_model_output/test_metrics.json
```

### 5.2 View Results

```bash
# View test metrics
cat data/05_model_output/test_metrics.json

# View training history
cat data/05_model_output/training_history.json
```

### 5.3 Run Inference

```bash
# With HuggingFace pretrained weights
kedro run --pipeline=vilbert_inference

# With locally trained weights
kedro run --pipeline=vilbert_inference_local
```

### 5.4 Visualize Training

```bash
# Launch Kedro-Viz
kedro viz

# Open http://localhost:4141 in browser
```

---

## 6. Troubleshooting

### 6.1 CUDA Out of Memory

```bash
# Reduce batch size in conf/base/parameters.yml
training:
  batch_size: 16  # or 8

# Or reduce max_regions
vilbert:
  max_regions: 36  # instead of 100
```

### 6.2 Missing Weights Error

```bash
# Error: "Facebook weights not found"
# Solution: Download weights (see Section 3)

wget https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin \
     -O weights/vilbert_pretrained_cc.bin
```

### 6.3 HuggingFace Download Issues

```bash
# Error: "Connection refused" or timeout
# Solution: Set HuggingFace mirror or use offline mode

export HF_DATASETS_OFFLINE=1
export HF_HOME=/path/to/cache
```

### 6.4 Slow Training

```bash
# Enable mixed precision training (add to parameters.yml)
training:
  use_amp: true  # Automatic mixed precision

# Use multiple workers
training:
  num_workers: 4
```

### 6.5 Early Stopping Too Aggressive

```bash
# Increase patience in parameters.yml
training:
  early_stopping_patience: 10  # instead of 5
```

---

## 7. Resume Training

### 7.1 From Checkpoint

If training was interrupted, resume from the last checkpoint:

```bash
# Check for saved checkpoint
ls data/05_model_output/*.pt

# Modify parameters.yml to load checkpoint
vilbert:
  checkpoint_path: "data/05_model_output/vilbert_best.pt"
  resume_training: true
```

### 7.2 From Specific Epoch

```python
# In a Python script or notebook
import torch

# Load checkpoint
checkpoint = torch.load("data/05_model_output/vilbert_best.pt")
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Best AUROC: {checkpoint.get('best_auroc', 'N/A')}")
```

---

## 8. Quick Reference

### 8.1 All Available Pipelines

| Pipeline | Command | Description |
|----------|---------|-------------|
| `data_processing` | `kedro run --pipeline=data_processing` | Load & preprocess data |
| `vilbert_train` | `kedro run --pipeline=vilbert_train` | Default training (ResNet) |
| `vilbert_frcnn_train` | `kedro run --pipeline=vilbert_frcnn_train` | FRCNN COCO features |
| `vilbert_vg_train` | `kedro run --pipeline=vilbert_vg_train` | FRCNN Visual Genome |
| `vilbert_precomputed_train` | `kedro run --pipeline=vilbert_precomputed_train` | Precomputed HDF5 |
| `vilbert_lmdb_train` | `kedro run --pipeline=vilbert_lmdb_train` | Facebook LMDB features |
| `vilbert_validate` | `kedro run --pipeline=vilbert_validate` | Validate on test set |
| `vilbert_inference` | `kedro run --pipeline=vilbert_inference` | Inference (pretrained) |
| `vilbert_inference_local` | `kedro run --pipeline=vilbert_inference_local` | Inference (local) |

### 8.2 Key Files

| File | Purpose |
|------|---------|
| `conf/base/parameters.yml` | All hyperparameters |
| `conf/base/catalog.yml` | Data catalog |
| `data/05_model_output/test_metrics.json` | Test results |
| `data/05_model_output/training_history.json` | Training curves |
| `data/05_model_output/vilbert_best.pt` | Best checkpoint |
| `RESULTS_ANALYSIS.md` | Comprehensive results analysis |

### 8.3 Verified Results

| Pipeline | Test AUROC | Test Accuracy | Training Time | Status |
|----------|------------|---------------|---------------|--------|
| **`vilbert_lmdb_train`** | **0.7433** | **70.6%** | ~13 min | **Best** |
| `vilbert_train` | 0.6645 | 62.1% | ~3 hours | Verified |
| `vilbert_frcnn_train` | 0.6235 | 61.1% | ~5 hours | Verified |
| `vilbert_precomputed_train` | 0.5878* | ~62%* | - | Overfitting |
| Facebook Baseline | 0.7045 | ~65% | - | Reference |

*Best validation AUROC (stopped due to overfitting)

**We exceeded Facebook's baseline by +3.88% AUROC!**

---

## 9. Next Steps

1. **Best results achieved**: `vilbert_lmdb_train` with official LMDB features (0.7433 AUROC)
2. **For quick experiments**: Run `vilbert_train` with ResNet features (0.6645 AUROC)
3. **To improve further**: Try ensemble methods or larger models
4. **Not recommended**: `vilbert_precomputed_train` had feature mapping issues

---

*Last updated: December 2024*
