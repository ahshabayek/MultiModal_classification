# ViLBERT Hateful Memes Classification: Comprehensive Results Analysis

## Executive Summary

This document provides a comprehensive analysis of all ViLBERT model architectures and training configurations implemented for the Hateful Memes Challenge. We compare our implementations against Facebook's official baseline (AUROC 0.7045).

### Results Summary

| Model Configuration | Feature Extractor | Test AUROC | Test Accuracy | F1 | Status |
|---------------------|-------------------|------------|---------------|-----|--------|
| **vilbert_lmdb_train** | Facebook LMDB (ResNeXt-152 VG) | **0.7433** | **70.60%** | - | **Best - Trained** |
| Facebook Baseline | ResNeXt-152 (VG LMDB) | 0.7045 | ~65% | ~0.50 | Reference |
| vilbert_train (HF) | ResNet-152 Grid | 0.6645 | 62.1% | 0.225 | Trained |
| vilbert_frcnn_train | Faster R-CNN (COCO) | 0.6235 | 61.1% | 0.382 | Trained |
| vilbert_precomputed_train | Precomputed HDF5 (VG) | 0.5878* | ~62%* | - | Trained (overfitting) |
| vilbert_vg_train | Faster R-CNN (VG) | - | - | - | Not run |

*Best validation AUROC before overfitting

**Current Best Result: 0.7433 AUROC** (vilbert_lmdb_train with Facebook's official LMDB features)

**Improvement over Facebook Baseline: +3.88% AUROC (+5.6% Accuracy)**

---

## 1. Model Architectures

### 1.1 Architecture Overview

We implement three ViLBERT variants, each designed for different use cases:

| Implementation | File | Purpose | Parameters |
|----------------|------|---------|------------|
| **ViLBERT Core** | `vilbert_core.py` | Custom from-scratch implementation | ~191M |
| **ViLBERT Facebook Arch** | `vilbert_facebook_arch.py` | Exact Facebook checkpoint structure | ~227M |
| **ViLBERT HuggingFace** | `vilbert_hf.py` | Community weights wrapper | varies |

### 1.2 ViLBERT Core Architecture

**Location**: `src/multimodalclassification/models/vilbert_core.py`

```
Architecture:
├── Text Stream (BERT-base)
│   ├── 12 transformer layers
│   ├── 768-dim hidden size
│   ├── 12 attention heads
│   └── 3072 intermediate size
│
├── Visual Stream (Custom)
│   ├── 6 transformer layers
│   ├── 768-dim hidden size
│   ├── 12 attention heads
│   └── 3072 intermediate size
│
├── Co-Attention Layers (6)
│   ├── Bidirectional cross-modal attention
│   ├── Visual→Text: visual queries attend to text
│   └── Text→Visual: text queries attend to visual
│
└── Classification Head
    ├── Concatenate [CLS] + mean visual pooling
    ├── Dense: 1536 → 768 → 2
    └── Dropout: 0.5
```

**Key Features**:
- Uses pretrained BERT-base for text encoding
- Custom visual transformer stream
- Co-attention after text layers [1, 3, 5, 7, 9, 11]
- Supports any visual feature extractor

### 1.3 ViLBERT Facebook Architecture

**Location**: `src/multimodalclassification/models/vilbert_facebook_arch.py`

```
Architecture (matches Facebook checkpoint):
├── Text Stream (BERT-base)
│   ├── 12 transformer layers
│   ├── 768-dim hidden size
│   ├── 12 attention heads
│   └── 3072 intermediate size
│
├── Visual Stream (Facebook spec)
│   ├── 6 transformer layers
│   ├── 1024-dim hidden size (larger than core)
│   ├── 8 attention heads
│   └── 1024 intermediate size
│
├── Co-Attention Layers (6)
│   ├── BiAttention: query1/key1/value1 (visual 1024→1024)
│   ├── BiAttention: query2/key2/value2 (text 768→1024)
│   ├── BiOutput: dense1/dense2 + LayerNorm
│   └── Separate FFNs for visual and text
│
├── Poolers
│   ├── t_pooler: 768 → 1024
│   └── v_pooler: 1024 → 1024
│
└── Classification Head
    ├── Concatenate: 2048-dim
    ├── Dense: 2048 → 1024 → 2
    └── Dropout: 0.1
```

**Key Differences from Core**:
- Visual hidden size: 1024 (vs 768)
- Visual attention heads: 8 (vs 12)
- Separate dimension projections for cross-modal attention
- Matches Facebook's pretrained weight structure exactly

### 1.4 ViLBERT HuggingFace Wrapper

**Location**: `src/multimodalclassification/models/vilbert_hf.py`

This is a convenience wrapper that:
- Loads weights from HuggingFace Hub (`visualjoyce/transformers4vl-vilbert`)
- Wraps the ViLBERT Core implementation
- Provides `from_pretrained()` API similar to transformers library

---

## 2. Visual Feature Extractors

### 2.1 Available Extractors

| Extractor | File | Pretrained On | Output | Regions | Achieved AUROC |
|-----------|------|---------------|--------|---------|----------------|
| **Facebook LMDB** | `lmdb_dataset.py` | Visual Genome (ResNeXt-152) | 2048-dim | 100 | **0.7433** |
| **Precomputed HDF5** | `precomputed_dataset.py` | Visual Genome (ResNet-101) | 2048-dim | 100 | 0.5878 |
| **ResNet-152** | `resnet.py` | ImageNet | 2048-dim | 36 (6×6 grid) | 0.6645 |
| **Faster R-CNN (COCO)** | `fasterrcnn.py` | COCO | 2048-dim | 36 detected | 0.6235 |
| **CLIP** | `clip.py` | WebImageText | 2048-dim | 36 (6×6 grid) | Not tested |
| **Faster R-CNN (VG)** | `fasterrcnn_vg.py` | Visual Genome | 2048-dim | 100 detected | Not tested |

### 2.2 Feature Extractor Details

#### Facebook LMDB (Best Performance)
```python
# Configuration
source: Facebook's official detectron.lmdb
model: ResNeXt-152-32x8d with attribute loss
pretraining: Visual Genome (1600 classes + 400 attributes)
regions: 100 per image
feature_dim: 2048
```
- **Pros**: Best performance, exact Facebook features
- **Cons**: Requires downloading ~10GB LMDB file

#### ResNet-152 (Grid-based)
```python
# Configuration
model: ResNet-152 pretrained on ImageNet
grid_size: 6x6 = 36 regions
feature_dim: 2048 (from layer4 output)
spatial_encoding: Normalized (x1, y1, x2, y2, area)
```
- **Pros**: Fast, reliable, no detection overhead
- **Cons**: Uniform grid ignores semantic regions

#### Faster R-CNN COCO (Object Detection)
```python
# Configuration
model: ResNet-50-FPN pretrained on COCO (80 classes)
max_regions: 36 (top confidence)
confidence_threshold: 0.2
feature_dim: 2048 (from box features)
```
- **Pros**: Object-centric regions
- **Cons**: COCO classes don't cover meme content well

---

## 3. Training Pipelines

### 3.1 Available Pipelines

```bash
# Best performing - Facebook's official LMDB features
kedro run --pipeline=vilbert_lmdb_train

# Default: HuggingFace weights + ResNet-152 grid features
kedro run --pipeline=vilbert_train

# Facebook weights + Faster R-CNN (COCO)
kedro run --pipeline=vilbert_frcnn_train

# Facebook weights + Precomputed HDF5 features
kedro run --pipeline=vilbert_precomputed_train

# Facebook weights + Faster R-CNN (Visual Genome)
kedro run --pipeline=vilbert_vg_train

# Validation and Inference
kedro run --pipeline=vilbert_validate
kedro run --pipeline=vilbert_inference
kedro run --pipeline=vilbert_inference_local
```

### 3.2 Pipeline Configuration Summary

| Pipeline | Model Weights | Feature Extractor | LR | Epochs | Loss | Test AUROC |
|----------|---------------|-------------------|-----|--------|------|------------|
| `vilbert_lmdb_train` | Facebook CC | Facebook LMDB | 1e-5 | 20 | CE | **0.7433** |
| `vilbert_train` | HuggingFace | ResNet-152 | 5e-5 | 20 | Focal | 0.6645 |
| `vilbert_frcnn_train` | Facebook CC | Faster R-CNN COCO | 5e-5 | 20 | Focal | 0.6235 |
| `vilbert_precomputed_train` | Facebook CC | Precomputed HDF5 | 1e-5 | 83 | CE | 0.5878* |

*Best validation AUROC (overfitting issues)

---

## 4. Detailed Training Results

### 4.1 vilbert_lmdb_train (Facebook LMDB Features) - BEST

**Configuration**:
```yaml
feature_extractor: lmdb (Facebook's detectron.lmdb)
max_regions: 100
learning_rate: 1.0e-5
num_epochs: 20
loss_type: ce
early_stopping_patience: 5
```

**Test Results**:
```
Test AUROC: 0.7433
Test Accuracy: 70.60%
```

**Training History**:

| Epoch | Val AUROC | Val Acc | Notes |
|-------|-----------|---------|-------|
| 1 | 0.5237 | - | Initial |
| 2 | 0.6523 | 65.0% | +0.13 |
| 3 | 0.6968 | - | +0.04 |
| 4 | 0.7330 | - | +0.04 |
| 5 | 0.7376 | 69.2% | |
| 6 | 0.7456 | - | |
| **9** | **0.7488** | **69.4%** | **Best validation** |
| 14 | - | - | Early stopping |

**Key Observations**:
- **Best validation AUROC: 0.7488** at epoch 9
- **Test AUROC: 0.7433** (+3.88% above Facebook baseline!)
- **Test Accuracy: 70.60%** (+5.6% above Facebook baseline!)
- Training completed in 766 seconds (~12.8 minutes)
- Early stopping triggered at epoch 14

### 4.2 vilbert_train (HuggingFace + ResNet-152)

**Configuration**:
```yaml
feature_extractor: resnet
max_regions: 36
learning_rate: 5.0e-5
num_epochs: 20
loss_type: focal
focal_alpha: 0.35
focal_gamma: 2.0
```

**Test Results**:
```json
{
  "loss": 0.6893,
  "accuracy": 0.6213,
  "auroc": 0.6645,
  "f1": 0.2251
}
```

**Training History** (11 epochs before early stopping):

| Epoch | Train Loss | Val Loss | Val AUROC | Notes |
|-------|------------|----------|-----------|-------|
| 1 | 0.0789 | 0.6644 | 0.5704 | Initial |
| 2 | 0.0729 | 0.6619 | 0.5799 | |
| 3 | 0.0690 | 0.6486 | 0.6156 | Improving |
| 4 | 0.0625 | 0.6503 | 0.6211 | |
| 5 | 0.0568 | 0.6518 | 0.6117 | |
| **6** | **0.0500** | **0.6872** | **0.6224** | **Best AUROC** |
| 7 | 0.0468 | 0.6736 | 0.6067 | Declining |
| 8 | 0.0428 | 0.6779 | 0.5960 | |
| 9 | 0.0359 | 0.6873 | 0.5985 | |
| 10 | 0.0304 | 0.6962 | 0.6144 | |
| 11 | 0.0263 | 0.8387 | 0.6153 | Early stop |

**Observations**:
- Best validation AUROC: **0.6224** at epoch 6
- Clear overfitting: train loss ↓ 0.079→0.026, val loss ↑ 0.664→0.839
- Early stopping triggered correctly at epoch 11

### 4.3 vilbert_frcnn_train (Facebook Weights + Faster R-CNN COCO)

**Configuration**:
```yaml
feature_extractor: fasterrcnn
max_regions: 36
learning_rate: 5.0e-5
num_epochs: 20
loss_type: focal
facebook_weights_path: weights/vilbert_pretrained_cc.bin
```

**Test Results**:
```json
{
  "loss": 0.6564,
  "accuracy": 0.6107,
  "auroc": 0.6235,
  "f1": 0.3820
}
```

**Training History** (9 epochs before early stopping):

| Epoch | Train Loss | Val Loss | Val AUROC | Notes |
|-------|------------|----------|-----------|-------|
| 1 | 0.0776 | 0.6612 | 0.5784 | Initial |
| 2 | 0.0750 | 0.6623 | 0.5796 | |
| 3 | 0.0717 | 0.6525 | 0.6047 | Improving |
| **4** | **0.0658** | **0.6501** | **0.6110** | **Best AUROC** |
| 5 | 0.0582 | 0.6680 | 0.6107 | Plateau |
| 6 | 0.0529 | 0.6620 | 0.6016 | Declining |
| 7 | 0.0489 | 0.6552 | 0.6020 | |
| 8 | 0.0458 | 0.7254 | 0.6010 | |
| 9 | 0.0416 | 0.7109 | 0.5972 | Early stop |

**Observations**:
- Best validation AUROC: **0.6110** at epoch 4
- Lower AUROC than ResNet grid features (0.6235 vs 0.6645)
- Better F1 score (0.382 vs 0.225) due to better class balance
- COCO-pretrained detector may not capture meme-relevant objects

### 4.4 vilbert_precomputed_train (Precomputed VG Features)

**Configuration**:
```yaml
feature_extractor: precomputed HDF5
max_regions: 100
learning_rate: 1.0e-5
num_epochs: 83
loss_type: ce
early_stopping_patience: 100
```

**Results** (manually stopped at epoch 41 due to overfitting):
- **Best validation AUROC: 0.5878** at epoch 5
- Severe overfitting: train loss → 0, val AUROC declined to ~0.53
- Issue: Likely mismatch between precomputed features and image IDs

**Observations**:
- The precomputed HDF5 features performed worse than expected
- Possible issue with feature-image ID mapping
- Facebook's official LMDB features work much better

---

## 5. Final Results Comparison

### 5.1 All Models vs Facebook Baseline

| Model | Test AUROC | Test Acc | vs Facebook |
|-------|------------|----------|-------------|
| **vilbert_lmdb_train** | **0.7433** | **70.60%** | **+3.88%** |
| Facebook Baseline | 0.7045 | ~65% | - |
| vilbert_train | 0.6645 | 62.1% | -4.00% |
| vilbert_frcnn_train | 0.6235 | 61.1% | -8.10% |
| vilbert_precomputed_train | 0.5878* | ~62%* | -11.67% |

*Best validation AUROC (test not run due to early termination)

### 5.2 Key Findings

1. **Facebook's LMDB features are crucial**: Using the exact same features as Facebook (detectron.lmdb) gives the best results.

2. **We exceeded Facebook's baseline**: Our best model achieves **0.7433 AUROC** vs Facebook's **0.7045** (+3.88% improvement).

3. **Lower learning rate helps**: Using 1e-5 instead of 5e-5 reduces overfitting.

4. **Cross-entropy beats focal loss**: For this dataset, standard CE performs better than focal loss.

5. **Feature quality matters most**: The gap between LMDB (0.7433) and ResNet grid (0.6645) is ~8%, showing visual features are the most important factor.

---

## 6. ViLBERT Architecture Deep Dive

### 6.1 Two-Stream Transformer Design

```
                    ┌─────────────────────────────────────────────┐
                    │              ViLBERT Architecture            │
                    └─────────────────────────────────────────────┘

    ┌──────────────┐                              ┌──────────────┐
    │  Text Input  │                              │ Visual Input │
    │  (tokens)    │                              │ (regions)    │
    └──────┬───────┘                              └──────┬───────┘
           │                                             │
           ▼                                             ▼
    ┌──────────────┐                              ┌──────────────┐
    │    BERT      │                              │   Visual     │
    │  Embeddings  │                              │  Embeddings  │
    │  (768-dim)   │                              │  (1024-dim)  │
    └──────┬───────┘                              └──────┬───────┘
           │                                             │
           ▼                                             ▼
    ┌──────────────┐                              ┌──────────────┐
    │ Text Layer 0 │                              │              │
    └──────┬───────┘                              │              │
           │                                      │              │
    ┌──────┴───────┐         Co-Attention         │              │
    │ Text Layer 1 │◄────────────────────────────►│ Visual L0    │
    └──────┬───────┘             (1)              └──────┬───────┘
           │                                             │
    ┌──────┴───────┐                                     │
    │ Text Layer 2 │                                     │
    └──────┬───────┘                                     │
           │                                             │
    ┌──────┴───────┐         Co-Attention         ┌──────┴───────┐
    │ Text Layer 3 │◄────────────────────────────►│ Visual L1    │
    └──────┬───────┘             (2)              └──────┬───────┘
           │                                             │
           ▼                                             ▼
          ...                   ...                     ...
           │                                             │
    ┌──────┴───────┐         Co-Attention         ┌──────┴───────┐
    │ Text Layer 11│◄────────────────────────────►│ Visual L5    │
    └──────┬───────┘             (6)              └──────┬───────┘
           │                                             │
           ▼                                             ▼
    ┌──────────────┐                              ┌──────────────┐
    │  [CLS] Pool  │                              │  Mean Pool   │
    │  (1024-dim)  │                              │  (1024-dim)  │
    └──────┬───────┘                              └──────┬───────┘
           │                                             │
           └─────────────────┬───────────────────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ Concatenate  │
                      │  (2048-dim)  │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │  Classifier  │
                      │   (2 class)  │
                      └──────────────┘
```

### 6.2 Co-Attention Mechanism

The key innovation in ViLBERT is **bidirectional co-attention**:

```python
# Visual attending to Text
v_query = W_q1 @ visual_hidden     # [B, V, 1024]
t_key = W_k2 @ text_hidden         # [B, T, 1024]
t_value = W_v2 @ text_hidden       # [B, T, 1024]

v_scores = v_query @ t_key.T / sqrt(128)  # [B, 8, V, T]
v_context = softmax(v_scores) @ t_value   # [B, V, 1024]

# Text attending to Visual
t_query = W_q2 @ text_hidden       # [B, T, 1024]
v_key = W_k1 @ visual_hidden       # [B, V, 1024]
v_value = W_v1 @ visual_hidden     # [B, V, 1024]

t_scores = t_query @ v_key.T / sqrt(128)  # [B, 8, T, V]
t_context = softmax(t_scores) @ v_value   # [B, T, 1024]
```

### 6.3 Co-Attention Schedule

| Text Layer | Visual Layer | Co-Attention Index |
|------------|--------------|-------------------|
| 1 | 0 | 0 |
| 3 | 1 | 1 |
| 5 | 2 | 2 |
| 7 | 3 | 3 |
| 9 | 4 | 4 |
| 11 | 5 | 5 |

---

## 7. Loss Functions

### 7.1 Available Loss Functions

| Loss Type | Formula | Use Case | Performance |
|-----------|---------|----------|-------------|
| **Cross-Entropy** | `-log(p_y)` | Standard classification | **Best (0.7433)** |
| **Focal Loss** | `-α(1-p)^γ log(p)` | Imbalanced data | Good (0.6645) |
| **Label Smoothing** | `(1-ε)p + ε/K` | Prevent overconfidence | Not tested |

### 7.2 Focal Loss Configuration

```python
# For Hateful Memes (~35% hateful, 65% not hateful)
focal_alpha: 0.35  # Weight for minority class
focal_gamma: 2.0   # Focusing parameter
```

### 7.3 Key Finding: Cross-Entropy Wins

Facebook uses standard cross-entropy, and our experiments confirm it performs better than focal loss for this dataset when using proper features.

---

## 8. Recommendations

### 8.1 Best Configuration (Achieved 0.7433 AUROC)

```yaml
# Use Facebook's official LMDB features
vilbert_lmdb:
  facebook_weights_path: "weights/vilbert_pretrained_cc.bin"
  lmdb_path: "data/03_features/mmf/detectron.lmdb"
  max_regions: 100

training_lmdb:
  batch_size: 32
  num_epochs: 20
  learning_rate: 1.0e-5
  warmup_steps: 2000
  early_stopping_patience: 5
  loss_type: "ce"
  use_linear_decay: true
```

### 8.2 Quick Start

```bash
# Best performing pipeline
kedro run --pipeline=vilbert_lmdb_train
```

### 8.3 For Custom Features

If you want to use your own features:
1. Use lower learning rate (1e-5)
2. Use cross-entropy loss
3. Use 100 regions if possible
4. Ensure feature-image ID mapping is correct

---

## 9. Project Structure

```
MultiModal_classification/
├── src/multimodalclassification/
│   ├── models/
│   │   ├── base.py                          # Abstract base class
│   │   ├── vilbert_core.py                  # Custom ViLBERT (768-dim visual)
│   │   ├── vilbert_facebook_arch.py         # Facebook-matching (1024-dim visual)
│   │   ├── vilbert_hf.py                    # HuggingFace wrapper
│   │   └── feature_extractors/
│   │       ├── __init__.py                  # get_feature_extractor()
│   │       ├── resnet.py                    # ResNet-152 grid features
│   │       ├── clip.py                      # CLIP semantic features
│   │       ├── fasterrcnn.py                # Faster R-CNN (COCO)
│   │       └── fasterrcnn_vg.py             # Faster R-CNN (Visual Genome)
│   │
│   ├── pipelines/
│   │   ├── data_processing/
│   │   │   ├── nodes.py                     # Data loading & preprocessing
│   │   │   ├── pipeline.py
│   │   │   ├── augmentation.py              # Data augmentation
│   │   │   ├── lmdb_dataset.py              # LMDB feature reader
│   │   │   └── precomputed_dataset.py       # HDF5 feature reader
│   │   │
│   │   └── model_training/
│   │       ├── nodes.py                     # Training logic
│   │       ├── pipeline.py                  # Pipeline definitions
│   │       ├── losses.py                    # Loss functions
│   │       └── visual_features.py           # Feature extraction utils
│   │
│   └── pipeline_registry.py                 # Register all pipelines
│
├── conf/base/
│   ├── parameters.yml                       # All hyperparameters
│   └── catalog.yml                          # Data catalog
│
├── data/
│   ├── 01_raw/hateful_memes/               # Raw images
│   ├── 02_intermediate/                     # Processed data
│   ├── 03_features/                         # Extracted features
│   │   ├── vg_features_100.h5              # Precomputed VG features
│   │   └── mmf/detectron.lmdb              # Facebook's features
│   ├── 04_models/                           # (legacy)
│   └── 05_model_output/
│       ├── vilbert_best.pt                  # Best checkpoint (LMDB)
│       ├── test_metrics.json                # Test results
│       ├── training_history.json            # Training curves
│       └── frcnn/                           # FRCNN pipeline outputs
│
├── weights/
│   ├── vilbert_pretrained_cc.bin           # Facebook CC weights
│   └── faster_rcnn_res101_vg.pth           # VG Faster R-CNN
│
├── scripts/
│   └── extract_features.py                  # Precompute features
│
├── RESULTS_ANALYSIS.md                      # This document
└── Steps.md                                 # Setup & training guide
```

---

## 10. References

1. Lu et al., "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations" (NeurIPS 2019)
2. Kiela et al., "The Hateful Memes Challenge" (NeurIPS 2020)
3. Facebook MMF Framework: https://github.com/facebookresearch/mmf
4. ViLBERT Multi-Task: https://github.com/jiasenlu/vilbert_beta
5. HuggingFace ViLBERT: https://huggingface.co/visualjoyce/transformers4vl-vilbert

---

## Appendix A: Full Training Logs

### A.1 vilbert_lmdb_train Training History

```
Epoch 1: val_auroc=0.5237
Epoch 2: val_auroc=0.6523, val_acc=0.6500
Epoch 3: val_auroc=0.6968
Epoch 4: val_auroc=0.7330
Epoch 5: val_auroc=0.7376, val_acc=0.6922
Epoch 6: val_auroc=0.7456
Epoch 7: val_auroc=0.7441, val_acc=0.6937
Epoch 8: val_auroc=0.7483
Epoch 9: val_auroc=0.7488 (Best)
Epoch 10: val_auroc=0.7388
Epoch 11: val_auroc=0.7427, val_acc=0.7000
Epoch 12: val_auroc=0.7424
Epoch 13: val_auroc=0.7267
Epoch 14: Early stopping

Test: AUROC=0.7433, Acc=0.7060
```

### A.2 vilbert_train Training History

```json
{
  "train_loss": [0.0789, 0.0729, 0.0690, 0.0625, 0.0568, 0.0500, 0.0468, 0.0428, 0.0359, 0.0304, 0.0263],
  "val_loss": [0.6644, 0.6619, 0.6486, 0.6503, 0.6518, 0.6872, 0.6736, 0.6779, 0.6873, 0.6962, 0.8387],
  "val_auroc": [0.5704, 0.5799, 0.6156, 0.6211, 0.6117, 0.6224, 0.6067, 0.5960, 0.5985, 0.6144, 0.6153]
}
```

### A.3 vilbert_frcnn_train Training History

```json
{
  "train_loss": [0.0776, 0.0750, 0.0717, 0.0658, 0.0582, 0.0529, 0.0489, 0.0458, 0.0416],
  "val_loss": [0.6612, 0.6623, 0.6525, 0.6501, 0.6680, 0.6620, 0.6552, 0.7254, 0.7109],
  "val_auroc": [0.5784, 0.5796, 0.6047, 0.6110, 0.6107, 0.6016, 0.6020, 0.6010, 0.5972]
}
```

---

*Last updated: December 4, 2024*
