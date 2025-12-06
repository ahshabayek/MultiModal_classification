# ViLBERT Model Implementations - Complete Guide

You have **4 ViLBERT implementations** in your codebase, each serving different purposes:

---

## 1. `vilbert_core.py` - Custom Implementation (768-dim visual)

**Location**: `src/multimodalclassification/models/vilbert_core.py`

This is a **from-scratch implementation** of the ViLBERT architecture based on the original paper.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ViLBERT CORE ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TEXT INPUT                           VISUAL INPUT                         │
│   [CLS] tokens [SEP]                   Faster R-CNN regions (2048-dim)      │
│        │                                      │                             │
│        ▼                                      ▼                             │
│   ┌──────────────┐                    ┌─────────────────┐                   │
│   │ BERT         │                    │ Visual          │                   │
│   │ Embeddings   │                    │ Embeddings      │                   │
│   │ (768-dim)    │                    │ 2048 → 768      │                   │
│   └──────────────┘                    └─────────────────┘                   │
│        │                                      │                             │
│        ▼                                      ▼                             │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                    ViLBERT ENCODER                                │     │
│   │  ┌─────────────┐                    ┌─────────────┐               │     │
│   │  │ Text Layer 0│                    │             │               │     │
│   │  │ Text Layer 1│◄───Co-Attention───►│ Vis Layer 0 │               │     │
│   │  │ Text Layer 2│                    │             │               │     │
│   │  │ Text Layer 3│◄───Co-Attention───►│ Vis Layer 1 │               │     │
│   │  │ Text Layer 4│                    │             │               │     │
│   │  │ Text Layer 5│◄───Co-Attention───►│ Vis Layer 2 │               │     │
│   │  │ ...         │       ...          │ ...         │               │     │
│   │  │ Text Layer11│◄───Co-Attention───►│ Vis Layer 5 │               │     │
│   │  └─────────────┘                    └─────────────┘               │     │
│   │     12 layers                          6 layers                   │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│        │                                      │                             │
│        ▼                                      ▼                             │
│   ┌──────────────┐                    ┌─────────────────┐                   │
│   │ Text Pooler  │                    │ Visual Pooler   │                   │
│   │ [CLS] → 768  │                    │ mean → 768      │                   │
│   └──────────────┘                    └─────────────────┘                   │
│        │                                      │                             │
│        └──────────────┬───────────────────────┘                             │
│                       ▼                                                     │
│              ┌────────────────┐                                             │
│              │ Classifier     │                                             │
│              │ 1536 → 768 → 2 │                                             │
│              └────────────────┘                                             │
│                       │                                                     │
│                       ▼                                                     │
│                   [logits]                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| `BertSelfAttention` | Standard self-attention within each modality |
| `BertCoAttention` | **Cross-modal attention** - Query from one modality, Key/Value from the other |
| `BertConnectionLayer` | Bidirectional co-attention (visual↔text) + FFN for both streams |
| `ViLBERTEmbeddings` | Projects 2048-dim visual features → 768-dim + spatial embeddings |
| `ViLBERTEncoder` | 12 text layers, 6 visual layers, 6 co-attention connections |

### Configuration

```python
{
    "hidden_size": 768,           # Both text AND visual use 768
    "v_feature_size": 2048,       # Input visual features
    "v_num_hidden_layers": 6,     # Visual transformer layers
    "t_num_hidden_layers": 12,    # Text transformer layers  
    "num_co_layers": 6,           # Co-attention connections
    "max_regions": 100,           # Max visual regions
}
```

### Use Case
- Custom training from scratch
- Research and experimentation
- When you don't have Facebook's pretrained weights

---

## 2. `vilbert_facebook_arch.py` - Facebook's Exact Architecture (1024-dim visual)

**Location**: `src/multimodalclassification/models/vilbert_facebook_arch.py`

This implementation **exactly matches Facebook's pretrained weight structure** from `vilbert-multi-task`.

### Key Difference: 1024-dim Visual Stream

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  FACEBOOK ARCHITECTURE (Different Dimensions!)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TEXT STREAM (768-dim)              VISUAL STREAM (1024-dim)               │
│        │                                      │                             │
│        ▼                                      ▼                             │
│   ┌──────────────┐                    ┌─────────────────┐                   │
│   │ BERT         │                    │ v_embeddings    │                   │
│   │ embeddings   │                    │ 2048 → 1024     │  ◄── Different!   │
│   │ (768-dim)    │                    │ + location (5)  │                   │
│   └──────────────┘                    └─────────────────┘                   │
│        │                                      │                             │
│        ▼                                      ▼                             │
│   ┌─────────────┐                     ┌─────────────┐                       │
│   │ layer[0-11] │                     │ v_layer[0-5]│                       │
│   │ 768-dim     │                     │ 1024-dim    │  ◄── Different!       │
│   │ 12 heads    │                     │ 8 heads     │                       │
│   └─────────────┘                     └─────────────┘                       │
│        │                                      │                             │
│        └──────────────┬───────────────────────┘                             │
│                       ▼                                                     │
│              ┌────────────────────────────────────────┐                     │
│              │         c_layer (Co-Attention)         │                     │
│              │  ┌─────────────────────────────────┐   │                     │
│              │  │ biattention:                    │   │                     │
│              │  │   query1/key1/value1 (1024)     │   │                     │
│              │  │   query2/key2/value2 (768→1024) │   │                     │
│              │  ├─────────────────────────────────┤   │                     │
│              │  │ biOutput:                       │   │                     │
│              │  │   dense1: 1024 → 1024           │   │                     │
│              │  │   dense2: 1024 → 768            │   │                     │
│              │  └─────────────────────────────────┘   │                     │
│              └────────────────────────────────────────┘                     │
│        │                                      │                             │
│        ▼                                      ▼                             │
│   ┌──────────────┐                    ┌─────────────────┐                   │
│   │ t_pooler     │                    │ v_pooler        │                   │
│   │ 768 → 1024   │                    │ 1024 → 1024     │                   │
│   └──────────────┘                    └─────────────────┘                   │
│        │                                      │                             │
│        └──────────────┬───────────────────────┘                             │
│                       ▼                                                     │
│              ┌────────────────┐                                             │
│              │ Classifier     │                                             │
│              │ 2048 → 1024 → 2│                                             │
│              └────────────────┘                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuration

```python
{
    "hidden_size": 768,           # Text stream
    "v_hidden_size": 1024,        # Visual stream (DIFFERENT!)
    "bi_hidden_size": 1024,       # Co-attention dimension
    "v_num_attention_heads": 8,   # Visual attention heads
    "num_co_attention_layers": 6, # Co-attention layers
}
```

### Weight Structure (matches Facebook's checkpoint)

```
bert.embeddings.*              # Text embeddings
bert.v_embeddings.*            # Visual embeddings (image_embeddings, location_embeddings)
bert.encoder.layer[0-11].*     # Text transformer layers
bert.encoder.v_layer[0-5].*    # Visual transformer layers  
bert.encoder.c_layer[0-5].*    # Co-attention layers
bert.t_pooler.*                # Text pooler
bert.v_pooler.*                # Visual pooler
```

### Use Case
- Loading Facebook's official pretrained weights (`pretrained_model.bin`)
- Achieving best results with LMDB features
- **This is what achieved 0.7433 AUROC!**

---

## 3. `vilbert_hf.py` - HuggingFace Wrapper

**Location**: `src/multimodalclassification/models/vilbert_hf.py`

A **convenience wrapper** that loads ViLBERT from HuggingFace Hub.

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     ViLBERTHuggingFace                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    Wrapper Layer                         │   │
│   │  - from_pretrained() - downloads from HuggingFace       │   │
│   │  - freeze_layers() - efficient fine-tuning              │   │
│   │  - forward() - delegates to inner model                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │           vilbert_core.ViLBERTForClassification          │   │
│   │                    (768-dim visual)                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Available Models

| Model ID | Description |
|----------|-------------|
| `visualjoyce/transformers4vl-vilbert` | Pretrained on Conceptual Captions |
| `visualjoyce/transformers4vl-vilbert-mt` | Multi-Task pretrained |

### Usage

```python
from multimodalclassification.models import ViLBERTHuggingFace

model = ViLBERTHuggingFace.from_pretrained(
    "visualjoyce/transformers4vl-vilbert",
    num_labels=2
)
```

### Use Case
- Quick experimentation with community weights
- When you don't have Facebook's official weights
- Grid-based features (ResNet, CLIP)

---

## 4. `vilbert_facebook.py` - Facebook Weights Loader

**Location**: `src/multimodalclassification/models/vilbert_facebook.py`

A **wrapper for loading Facebook's official weights** with proper key mapping.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ViLBERTFacebook                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    Wrapper Layer                         │   │
│   │  - from_pretrained() - loads Facebook checkpoint        │   │
│   │  - _load_facebook_weights() - key mapping               │   │
│   │  - freeze_layers() - efficient fine-tuning              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │     data/04_models/vilbert.ViLBERTForClassification      │   │
│   │              (imports from old location)                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Use Case
- Loading `weights/vilbert_pretrained_cc.bin`
- Used by some legacy pipelines

---

## Comparison Table

| Feature | vilbert_core | vilbert_facebook_arch | vilbert_hf | vilbert_facebook |
|---------|--------------|----------------------|------------|------------------|
| **Visual Dim** | 768 | **1024** | 768 | varies |
| **Text Dim** | 768 | 768 | 768 | 768 |
| **Visual Heads** | 12 | **8** | 12 | varies |
| **Pooler Output** | 768+768=1536 | **1024+1024=2048** | 768+768=1536 | varies |
| **Weight Source** | Random/BERT | Facebook official | HuggingFace | Facebook |
| **Best For** | Research | **Production** | Quick tests | Legacy |

---

## Co-Attention Mechanism (The Key Innovation)

All implementations share the **co-attention** concept:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CO-ATTENTION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   VISUAL STREAM                              TEXT STREAM                    │
│   v_hidden [B, 100, 1024]                    t_hidden [B, 128, 768]         │
│        │                                           │                        │
│        │    ┌─────────────────────────────────┐    │                        │
│        └───►│      BIDIRECTIONAL ATTENTION     │◄───┘                        │
│             │                                  │                            │
│             │  Visual → Text:                  │                            │
│             │    Q = v_hidden                  │                            │
│             │    K,V = t_hidden                │                            │
│             │    "What text is relevant        │                            │
│             │     to this image region?"       │                            │
│             │                                  │                            │
│             │  Text → Visual:                  │                            │
│             │    Q = t_hidden                  │                            │
│             │    K,V = v_hidden                │                            │
│             │    "What image region is         │                            │
│             │     relevant to this word?"      │                            │
│             └─────────────────────────────────┘                             │
│                      │                │                                     │
│                      ▼                ▼                                     │
│             v_updated           t_updated                                   │
│             (enriched with      (enriched with                              │
│              text info)          visual info)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Works for Hateful Memes

1. **Text alone**: "Love the way you look" → Not hateful
2. **Image alone**: Photo of a person → Not hateful  
3. **Combined via co-attention**: The model learns that the text referring to the specific visual content can be hateful (e.g., mocking appearance)

---

## Which Model Achieved 0.7433 AUROC?

The **`vilbert_lmdb_train` pipeline** uses:
- **Model**: `vilbert_facebook_arch.py` (1024-dim visual)
- **Features**: Facebook's LMDB (ResNeXt-152-32x8d, Visual Genome)
- **Result**: Test AUROC 0.7433, Test Accuracy 70.60%

This is defined in `nodes.py:1359`:
```python
def train_vilbert_with_lmdb_features(...):
    """
    This uses Facebook's EXACT precomputed features extracted with ResNeXt-152
    """
```

---

# Input Preprocessing Pipeline

## Overview: What Gets Fed to ViLBERT?

ViLBERT does **NOT** receive raw images and text. Instead, inputs are preprocessed:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INPUT PREPROCESSING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   RAW INPUTS                                                                │
│   ┌─────────────┐              ┌─────────────────────────────────┐          │
│   │ Raw Image   │              │ Raw Text                        │          │
│   │ (JPEG/PNG)  │              │ "this is some meme text"        │          │
│   └──────┬──────┘              └────────────────┬────────────────┘          │
│          │                                      │                           │
│          ▼                                      ▼                           │
│   ┌──────────────────────┐              ┌──────────────────────┐            │
│   │  VISUAL FEATURE      │              │  TEXT TOKENIZATION   │            │
│   │  EXTRACTION          │              │  (BERT Tokenizer)    │            │
│   │                      │              │                      │            │
│   │  Options:            │              │  "bert-base-uncased" │            │
│   │  - ResNet-152        │              │                      │            │
│   │  - CLIP              │              └──────────┬───────────┘            │
│   │  - Faster R-CNN      │                         │                        │
│   │  - LMDB (precomputed)│                         │                        │
│   └──────────┬───────────┘                         │                        │
│              │                                     │                        │
│              ▼                                     ▼                        │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                     MODEL INPUTS                                  │     │
│   │                                                                   │     │
│   │  VISUAL:                          TEXT:                          │     │
│   │  - visual_features [B, R, 2048]   - input_ids [B, 128]           │     │
│   │  - spatial_locations [B, R, 5]    - attention_mask [B, 128]      │     │
│   │  - visual_attention_mask [B, R]   - token_type_ids [B, 128]      │     │
│   │                                                                   │     │
│   │  B = batch size, R = num_regions (36-100)                        │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Text Preprocessing

All models use the same text preprocessing via **BERT Tokenizer**:

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Example text
text = "when you can't pay your bills but still buy coffee"

# Tokenization
encoding = tokenizer(
    text,
    max_length=128,              # Max sequence length
    padding="max_length",        # Pad to max_length
    truncation=True,             # Truncate if too long
    return_tensors="pt",         # Return PyTorch tensors
)

# Output:
# encoding["input_ids"]      -> [1, 128] token IDs
# encoding["attention_mask"] -> [1, 128] 1s for real tokens, 0s for padding
# encoding["token_type_ids"] -> [1, 128] segment IDs (all 0s for single sentence)
```

### Tokenization Example

```
Input:  "when you can't pay your bills"
         ↓
Tokens: ['[CLS]', 'when', 'you', 'can', "'", 't', 'pay', 'your', 'bills', '[SEP]', '[PAD]'...]
         ↓
IDs:    [101, 2043, 2017, 2064, 1005, 1056, 3477, 2115, 5765, 102, 0, 0, 0...]
         ↓
Mask:   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0...]
```

---

## Visual Preprocessing

Visual preprocessing varies by **feature extractor type**. There are 4 approaches:

### Approach 1: ResNet-152 (Grid Features)

**File**: `src/multimodalclassification/models/feature_extractors/resnet.py`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ResNet-152 FEATURE EXTRACTION                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Raw Image (any size)                                                      │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ 1. Resize to 224×224                 │                                  │
│   │ 2. ToTensor (normalize to 0-1)       │                                  │
│   │ 3. Normalize(mean, std)              │  ImageNet normalization          │
│   │    mean=[0.485, 0.456, 0.406]        │                                  │
│   │    std=[0.229, 0.224, 0.225]         │                                  │
│   └──────────────────────────────────────┘                                  │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ ResNet-152 Backbone                  │                                  │
│   │ (remove final pooling + FC)          │                                  │
│   │                                      │                                  │
│   │ Output: [1, 2048, 7, 7]              │  Feature maps                    │
│   └──────────────────────────────────────┘                                  │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ Adaptive Pool to 6×6 grid            │                                  │
│   │                                      │                                  │
│   │ Output: [36, 2048]                   │  36 regions × 2048 features      │
│   └──────────────────────────────────────┘                                  │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ Generate Grid Spatial Locations      │                                  │
│   │                                      │                                  │
│   │ For 6×6 grid:                        │                                  │
│   │ [x1, y1, x2, y2, area]               │                                  │
│   │ [0.0, 0.0, 0.167, 0.167, 0.028]      │  First cell                      │
│   │ [0.167, 0.0, 0.333, 0.167, 0.028]    │  Second cell                     │
│   │ ...                                  │                                  │
│   └──────────────────────────────────────┘                                  │
│                                                                             │
│   OUTPUT:                                                                   │
│   - visual_features: [36, 2048]                                             │
│   - spatial_locations: [36, 5]                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Pros**: Fast, no object detection overhead
**Cons**: Grid-based, may miss object semantics
**Expected AUROC**: ~0.65-0.67

---

### Approach 2: CLIP (Semantic Grid Features)

**File**: `src/multimodalclassification/models/feature_extractors/clip.py`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLIP FEATURE EXTRACTION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Raw Image (any size)                                                      │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ CLIP Processor                       │                                  │
│   │ (model-specific preprocessing)       │                                  │
│   └──────────────────────────────────────┘                                  │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ CLIP Vision Transformer              │                                  │
│   │ "openai/clip-vit-base-patch32"       │                                  │
│   │                                      │                                  │
│   │ Input: 224×224 → 7×7 patches         │                                  │
│   │ Output: [1, 49+1, 768]               │  49 patches + CLS token          │
│   └──────────────────────────────────────┘                                  │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ Remove CLS token                     │                                  │
│   │ Project: 768 → 2048                  │                                  │
│   │ Interpolate to num_regions           │                                  │
│   │                                      │                                  │
│   │ Output: [36, 2048]                   │                                  │
│   └──────────────────────────────────────┘                                  │
│                                                                             │
│   OUTPUT:                                                                   │
│   - visual_features: [36, 2048]                                             │
│   - spatial_locations: [36, 5] (grid-based)                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Pros**: Semantically rich features from vision-language pretraining
**Cons**: Still grid-based, not object-centric
**Expected AUROC**: ~0.68-0.70

---

### Approach 3: Faster R-CNN (Object Detection)

**File**: `src/multimodalclassification/models/feature_extractors/fasterrcnn.py` (COCO)
**File**: `src/multimodalclassification/models/feature_extractors/fasterrcnn_vg.py` (Visual Genome)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FASTER R-CNN FEATURE EXTRACTION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Raw Image (any size)                                                      │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ 1. ToTensor                          │                                  │
│   │ 2. (No resize - detection works      │                                  │
│   │     at original resolution)          │                                  │
│   └──────────────────────────────────────┘                                  │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ Faster R-CNN Detector                │                                  │
│   │                                      │                                  │
│   │ COCO version:                        │                                  │
│   │   ResNet-50-FPN, 91 classes          │                                  │
│   │                                      │                                  │
│   │ Visual Genome version:               │                                  │
│   │   ResNet-101, 1600 classes           │                                  │
│   │                                      │                                  │
│   │ Output: boxes, scores, labels        │                                  │
│   └──────────────────────────────────────┘                                  │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ Filter by confidence (>0.2)          │                                  │
│   │ Take top 36 scoring boxes            │                                  │
│   │ Pad with grid if < 36 detections     │                                  │
│   └──────────────────────────────────────┘                                  │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ ROI Pooling                          │                                  │
│   │ Extract features for each box        │                                  │
│   │                                      │                                  │
│   │ Output: [36, 2048]                   │                                  │
│   └──────────────────────────────────────┘                                  │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ Normalize bounding boxes             │                                  │
│   │ [x1/W, y1/H, x2/W, y2/H, area]       │                                  │
│   │                                      │                                  │
│   │ Output: [36, 5]                      │                                  │
│   └──────────────────────────────────────┘                                  │
│                                                                             │
│   OUTPUT:                                                                   │
│   - visual_features: [36, 2048]  (actual object regions!)                   │
│   - spatial_locations: [36, 5]   (actual bounding boxes!)                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Pros**: Object-centric features, actual detected regions
**Cons**: Slower, COCO has domain mismatch with ViLBERT pretraining
**Expected AUROC**: ~0.62-0.68 (COCO), ~0.68-0.72 (Visual Genome)

---

### Approach 4: LMDB Precomputed Features (Facebook's Approach)

**File**: `src/multimodalclassification/pipelines/data_processing/lmdb_dataset.py`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LMDB PRECOMPUTED FEATURES (BEST!)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   NO IMAGE PROCESSING AT RUNTIME!                                           │
│                                                                             │
│   Image ID (e.g., "12345")                                                  │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ LMDB Database Lookup                 │                                  │
│   │ detectron.lmdb                       │                                  │
│   │                                      │                                  │
│   │ Key: image_id                        │                                  │
│   │ Value: pickled dict                  │                                  │
│   └──────────────────────────────────────┘                                  │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ Unpickle stored features             │                                  │
│   │                                      │                                  │
│   │ {                                    │                                  │
│   │   "features": [100, 2048],           │  Visual features                 │
│   │   "boxes": [100, 4],                 │  Bounding boxes                  │
│   │   "cls_prob": [100, 1600],           │  Class probabilities (VG)        │
│   │   "objects": [100,]                  │  Object class IDs                │
│   │ }                                    │                                  │
│   └──────────────────────────────────────┘                                  │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────────────────┐                                  │
│   │ Convert boxes to spatial format      │                                  │
│   │ [x1/1000, y1/1000, x2/1000,          │                                  │
│   │  y2/1000, area]                      │                                  │
│   └──────────────────────────────────────┘                                  │
│                                                                             │
│   OUTPUT:                                                                   │
│   - visual_features: [100, 2048]                                            │
│   - spatial_locations: [100, 5]                                             │
│                                                                             │
│   These features were extracted by Facebook using:                          │
│   - ResNeXt-152-32x8d backbone                                              │
│   - Pretrained on Visual Genome (1600 classes)                              │
│   - Bottom-Up Attention approach                                            │
│   - 100 regions per image (adaptive)                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Pros**: Facebook's exact features, no runtime extraction, fastest training
**Cons**: Must have precomputed LMDB file
**AUROC Achieved**: **0.7433** (best result!)

---

## Complete Data Flow Example

Here's what happens when you load a single sample:

```python
# Sample from Hateful Memes dataset
sample = {
    "id": "42953",
    "img": "data/01_raw/hateful_memes/img/42953.png",
    "text": "when you can't pay your bills but still buy coffee",
    "label": 0  # not hateful
}

# === TEXT PROCESSING ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encoding = tokenizer(
    sample["text"],
    max_length=128,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

# Result:
# input_ids:      [101, 2043, 2017, 2064, ..., 0, 0, 0]  shape: [128]
# attention_mask: [1, 1, 1, 1, ..., 0, 0, 0]             shape: [128]
# token_type_ids: [0, 0, 0, 0, ..., 0, 0, 0]             shape: [128]

# === VISUAL PROCESSING (Option A: On-the-fly) ===
from PIL import Image
image = Image.open(sample["img"]).convert("RGB")
visual_features, spatial_locations = feature_extractor.extract_features(image)

# Result:
# visual_features:    [36, 2048]  (36 regions, 2048-dim features)
# spatial_locations:  [36, 5]     (x1, y1, x2, y2, area for each region)

# === VISUAL PROCESSING (Option B: LMDB precomputed) ===
# Just lookup by ID - features already extracted!
lmdb_data = lmdb_txn.get(sample["id"].encode())
data_dict = pickle.loads(lmdb_data)
visual_features = torch.tensor(data_dict["features"])      # [100, 2048]
spatial_locations = normalize_boxes(data_dict["boxes"])    # [100, 5]

# === FINAL BATCH (what model receives) ===
batch = {
    "input_ids":             torch.tensor([...]),  # [B, 128]
    "attention_mask":        torch.tensor([...]),  # [B, 128]
    "token_type_ids":        torch.tensor([...]),  # [B, 128]
    "visual_features":       torch.tensor([...]),  # [B, 100, 2048]
    "spatial_locations":     torch.tensor([...]),  # [B, 100, 5]
    "visual_attention_mask": torch.tensor([...]),  # [B, 100]
    "labels":                torch.tensor([...]),  # [B]
}
```

---

## Dataset Classes Summary

| Dataset Class | Visual Source | Pipeline |
|---------------|---------------|----------|
| `HatefulMemesDataset` | On-the-fly extraction (ResNet/CLIP/FRCNN) | `vilbert_train`, `vilbert_frcnn_train`, `vilbert_vg_train` |
| `PrecomputedFeaturesDataset` | HDF5 file (custom extracted) | `vilbert_precomputed_train` |
| `LMDBFeaturesDataset` | Facebook's detectron.lmdb | `vilbert_lmdb_train` |

---

## Configuration Parameters

In `conf/base/parameters.yml`:

```yaml
vilbert:
  max_seq_length: 128        # Max text tokens
  max_regions: 36            # Regions for on-the-fly extraction
  visual_feature_dim: 2048   # Feature dimension
  feature_extractor: "resnet"  # Options: resnet, clip, fasterrcnn, fasterrcnn_vg

vilbert_lmdb:
  max_seq_length: 128
  num_regions: 100           # LMDB has 100 regions per image
  visual_feature_dim: 2048
  lmdb_path: "data/01_raw/hateful_memes/detectron.lmdb"
```
