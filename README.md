# MultiModal Classification

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

A multimodal classification project using ViLBERT (Vision-and-Language BERT) for the Hateful Memes dataset. This project implements binary classification to detect hateful content in memes by combining image and text understanding.

**Best Result:** 0.7580 AUROC (Facebook LMDB features), exceeding Facebook's baseline (0.7045) by +5.35%.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Available Pipelines](#available-pipelines)
- [Configuration](#configuration)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Results Summary](#results-summary)
- [Adding New Models](#adding-new-models)

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- ~10GB disk space for data and weights

### Step 1: Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd MultiModal_classification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install the project in editable mode
pip install -e .
```

### Step 2: Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Additional Dependencies

```bash
# Core ML libraries
pip install transformers datasets huggingface_hub
pip install mlflow kedro-mlflow

# For Faster R-CNN features
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# For DINOv2 features
pip install timm

# For LMDB features
pip install lmdb
```

### Step 4: Download Pretrained Weights

```bash
# Create weights directory
mkdir -p weights

# Download Facebook's ViLBERT pretrained weights (Conceptual Captions)
# Option 1: Using gdown
pip install gdown
gdown --id 1kuPr3OAN5oVJQYSXHhkv5tXgCn-mvJBj -O weights/vilbert_pretrained_cc.bin

# Option 2: Manual download from:
# https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin
```

### Step 5: Download Facebook's LMDB Features (Recommended)

For best results, use Facebook's precomputed features:

```bash
# Create features directory
mkdir -p data/03_features/mmf

# Download LMDB features (~3GB)
# These are ResNeXt-152 features pretrained on Visual Genome
gdown --id <facebook-lmdb-id> -O data/03_features/mmf/detectron.lmdb

# Or follow instructions at:
# https://github.com/facebookresearch/mmf/tree/main/tools/scripts/features
```

### Verify Installation

```bash
# Check Kedro installation
kedro info

# List available pipelines
kedro registry list

# Run a quick test (data processing only)
kedro run --pipeline=data_processing
```

---

## Quick Start

### 1. Process Data (First Time Only)

```bash
kedro run --pipeline=data_processing
```

This downloads the Hateful Memes dataset from HuggingFace and images from Google Drive.

### 2. Train Best Model (LMDB Features)

```bash
kedro run --pipeline=vilbert_lmdb_train
```

### 3. Train with On-the-fly Features (No LMDB Required)

```bash
# Best on-the-fly model (ROI pooling)
kedro run --pipeline=vilbert_resnet152_roi_train

# Or with DINOv2
kedro run --pipeline=vilbert_dinov2_train
```

### 4. View Results

```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

---

## Available Pipelines

### Data Processing

| Pipeline | Command | Description |
|----------|---------|-------------|
| Data Processing | `kedro run --pipeline=data_processing` | Load data from HuggingFace, download images |

### Training Pipelines (Full: Data + Training)

| Pipeline | Command | Feature Extractor | Expected AUROC |
|----------|---------|-------------------|----------------|
| **LMDB (Best)** | `kedro run --pipeline=vilbert_lmdb_train` | Facebook LMDB (ResNeXt-152 VG) | **0.7580** |
| **ROI Pool (Best On-the-fly)** | `kedro run --pipeline=vilbert_resnet152_roi_train` | ResNet-152 + ROI Pooling | **0.7197** |
| DINOv2 Multi-Layer | `kedro run --pipeline=vilbert_dinov2_multilayer_train` | DINOv2 ViT-L (4 layers) | 0.7171 |
| DINOv2 | `kedro run --pipeline=vilbert_dinov2_train` | DINOv2 ViT-L (single layer) | 0.7069 |
| ResNet-152 Grid | `kedro run --pipeline=vilbert_resnet152_grid_train` | ResNet-152 Grid Pooling | 0.6658 |
| Faster R-CNN R50 | `kedro run --pipeline=vilbert_frcnn_train` | Faster R-CNN ResNet-50 (COCO) | 0.6472 |
| Faster R-CNN R152 | `kedro run --pipeline=vilbert_frcnn_resnet152_train` | Faster R-CNN ResNet-152 (COCO) | 0.6334 |
| X-152++ | `kedro run --pipeline=vilbert_x152_train` | Facebook X-152++ (VQA winner) | TBD |

### Model-Only Pipelines (Skip Data Processing)

Use these if data is already processed:

```bash
kedro run --pipeline=model_training_lmdb
kedro run --pipeline=model_training_resnet152_roi
kedro run --pipeline=model_training_dinov2
kedro run --pipeline=model_training_dinov2_multilayer
kedro run --pipeline=model_training_resnet152_grid
kedro run --pipeline=model_training_frcnn
kedro run --pipeline=model_training_frcnn_resnet152
```

### Inference & Validation

| Pipeline | Command | Description |
|----------|---------|-------------|
| Validate | `kedro run --pipeline=vilbert_validate` | Evaluate trained model on test set |
| Inference (Pretrained) | `kedro run --pipeline=vilbert_inference` | Run inference with HuggingFace weights |
| Inference (Local) | `kedro run --pipeline=vilbert_inference_local` | Run inference with locally trained weights |

---

## Configuration

All configuration is in `conf/base/parameters.yml`.

### Key Training Parameters

```yaml
training:
  batch_size: 16          # 16 for LMDB, 32 for others
  num_epochs: 20          # With early stopping
  learning_rate: 1.0e-5   # Conservative for fine-tuning
  weight_decay: 0.01      # AdamW regularization
  warmup_steps: 2000      # Linear warmup
  early_stopping_patience: 5
  gradient_clip: 1.0
  loss_type: "ce"         # Options: ce, focal, label_smoothing
```

### Feature Extractor Selection

```yaml
vilbert:
  feature_extractor: "resnet"  # Options: resnet, fasterrcnn, dinov2, etc.
  max_regions: 36              # 36 for most, 100 for LMDB
  freeze_bert_layers: 0        # 0 = train all layers
```

### Model-Specific Configs

Each feature extractor has its own configuration section:
- `vilbert_lmdb` - Facebook LMDB features
- `vilbert_resnet152_roi` - ROI pooling
- `vilbert_dinov2` - DINOv2 features
- `vilbert_dinov2_multilayer` - Multi-layer DINOv2
- `vilbert_frcnn` - Faster R-CNN ResNet-50
- `vilbert_frcnn_resnet152` - Faster R-CNN ResNet-152

---

## MLflow Experiment Tracking

### Starting MLflow UI

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

### Logged Metrics

**Training:**
- `train_loss`, `val_loss` - Loss per epoch
- `val_auroc`, `val_accuracy`, `val_f1` - Validation metrics
- Best model checkpoint

**Evaluation:**
- `test_auroc`, `test_accuracy`, `test_precision`, `test_recall`, `test_f1`
- Confusion matrix (TP, TN, FP, FN)

---

## Results Summary

| Feature Extractor | AUROC | Accuracy | Notes |
|-------------------|-------|----------|-------|
| Facebook LMDB (ResNeXt-152 VG) | **0.7580** | 71.9% | Best overall, requires LMDB download |
| ResNet-152 ROI Pool (ImageNet) | **0.7197** | 68.9% | Best on-the-fly extraction |
| DINOv2 Multi-Layer (ViT-L) | 0.7171 | 68.1% | Uses label smoothing (0.1) |
| DINOv2 Single-Layer (ViT-L) | 0.7069 | 67.1% | Self-supervised features |
| Facebook Baseline | 0.7045 | ~65% | Reference from paper |
| ResNet-152 Grid (no ROI) | 0.6658 | 65.6% | Simple grid pooling |
| Faster R-CNN R50 (COCO) | 0.6472 | 63.5% | COCO pretrained |
| Faster R-CNN R152 (COCO) | 0.6334 | 63.5% | COCO pretrained |

### Key Findings

1. **Precomputed features dominate:** LMDB achieves best results due to Visual Genome pretraining alignment with ViLBERT.

2. **ROI pooling helps significantly:** ROI approach (0.7197) outperforms grid pooling (0.6658) by +5.4%.

3. **DINOv2 beats COCO detectors:** Self-supervised DINOv2 (0.7069) outperforms COCO Faster R-CNN (0.6334) by +7.4%.

4. **BERT freezing hurts on-the-fly extractors:** Freezing first 6 BERT layers has negligible effect on LMDB (-0.03%) but hurts ROI (-1.77%) and DINOv2 (-2.66%).

5. **Focal loss doesn't help:** Despite class imbalance (53.6%/46.4%), focal loss hurt all models by 0.35-1.07%.

---

## Data Loading

The project automatically loads data from HuggingFace (`neuralcatcher/hateful_memes`) and downloads images from Google Drive.

**Data flow:**
1. Load dataset from HuggingFace (train/validation/test splits)
2. Remove duplicates
3. Download image archive from Google Drive (~1GB)
4. Fetch any missing images from HuggingFace backup
5. Preprocess text and validate image paths

---

## Adding New Models

### Project Structure

```
src/multimodalclassification/
├── models/
│   ├── __init__.py              # Registry exports
│   ├── base.py                  # Base classes and registries
│   ├── vilbert_hf.py            # HuggingFace ViLBERT
│   ├── vilbert_facebook.py      # Facebook weights ViLBERT
│   └── feature_extractors/
│       ├── __init__.py          # Feature extractor exports
│       ├── resnet.py            # ResNet-152 grid features
│       ├── dinov2.py            # DINOv2 ViT features
│       ├── fasterrcnn.py        # Faster R-CNN features
│       └── resnet152_roi.py     # ROI pooling features
└── pipelines/
    └── model_training/
        ├── nodes.py             # Training logic
        └── pipeline.py          # Pipeline definitions
```

### Adding a New Feature Extractor

1. Create a new file in `src/multimodalclassification/models/feature_extractors/`

2. Inherit from `BaseFeatureExtractor`:

```python
from ..base import BaseFeatureExtractor, register_feature_extractor

@register_feature_extractor("my_extractor")
class MyFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, output_dim=2048, num_regions=36, device="cuda", **kwargs):
        super().__init__(output_dim, num_regions, device)
        self.backbone = ...  # Your model
    
    def extract_features(self, image):
        # Returns (features, spatial_locations)
        # features: [num_regions, output_dim]
        # spatial: [num_regions, 5] (x1, y1, x2, y2, area)
        ...
```

3. Export in `feature_extractors/__init__.py`

4. Add configuration in `conf/base/parameters.yml`

5. Create pipeline in `pipeline_registry.py`

### Available Feature Extractors

| Extractor | Description | Output Dim | Regions |
|-----------|-------------|------------|---------|
| `resnet` | ResNet-152 grid features | 2048 | 36 |
| `resnet152_roi` | ResNet-152 with ROI pooling | 2048 | 36 |
| `dinov2` | DINOv2 ViT-L features | 1024→2048 | 36 |
| `dinov2_multilayer` | DINOv2 multi-layer fusion | 4096→2048 | 36 |
| `fasterrcnn` | Faster R-CNN R50 (COCO) | 2048 | 36 |
| `fasterrcnn_resnet152` | Faster R-CNN R152 (COCO) | 2048 | 36 |
| `lmdb` | Facebook precomputed features | 2048 | 100 |

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in `conf/base/parameters.yml`:
```yaml
training:
  batch_size: 8  # or 4
```

### Missing LMDB Features

If you don't have Facebook's LMDB features, use on-the-fly extraction:
```bash
kedro run --pipeline=vilbert_resnet152_roi_train
```

### Slow Training

DINOv2 extraction adds ~0.5s/image. Use precomputed features for faster training:
```bash
kedro run --pipeline=vilbert_lmdb_train
```

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -e .
pip install transformers datasets timm detectron2
```

---

## References

- [ViLBERT Paper](https://arxiv.org/abs/1908.02265) (Lu et al., NeurIPS 2019)
- [Hateful Memes Challenge](https://arxiv.org/abs/2005.04790) (Kiela et al., NeurIPS 2020)
- [Facebook MMF](https://github.com/facebookresearch/mmf)
- [DINOv2](https://arxiv.org/abs/2304.07193) (Oquab et al., 2023)

---

## License

This project is for educational and research purposes.
