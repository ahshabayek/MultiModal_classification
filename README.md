# MultiModalClassification

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

A multimodal classification project using ViLBERT (Vision-and-Language BERT) for the Hateful Memes dataset. This project implements binary classification to detect hateful content in memes by combining image and text understanding.

## Data Loading

The project automatically loads data from HuggingFace (`neuralcatcher/hateful_memes`) and downloads images from Google Drive. This mirrors the approach in `notebooks/hateful-memes.ipynb`.

**Data flow:**
1. Load dataset from HuggingFace (train/validation/test splits)
2. Remove duplicates
3. Download image archive from Google Drive (~1GB)
4. Fetch any missing images from HuggingFace backup
5. Preprocess text and validate image paths

## Available Pipelines

The project provides multiple pipeline modes for different use cases:

| Pipeline | Command | Description |
|----------|---------|-------------|
| Data Processing | `kedro run --pipeline=data_processing` | Load data from HuggingFace and preprocess |
| Training (HF) | `kedro run --pipeline=vilbert_train` | Full pipeline with HuggingFace weights + ResNet features |
| Training (FRCNN) | `kedro run --pipeline=vilbert_frcnn_train` | Facebook weights + Faster R-CNN (COCO) features |
| Training (VG) | `kedro run --pipeline=vilbert_vg_train` | Facebook weights + Faster R-CNN (Visual Genome) features |
| Validation | `kedro run --pipeline=vilbert_validate` | Validate a locally trained model on the test set |
| Inference (Pretrained) | `kedro run --pipeline=vilbert_inference` | Run inference using pretrained HuggingFace weights |
| Inference (Local) | `kedro run --pipeline=vilbert_inference_local` | Run inference using locally trained weights |

**Model-only pipelines** (use if data is already processed):
- `kedro run --pipeline=model_training`
- `kedro run --pipeline=model_validation`
- `kedro run --pipeline=model_inference`
- `kedro run --pipeline=model_inference_local`

## MLflow Experiment Tracking

This project uses MLflow for experiment tracking via the `kedro-mlflow` plugin.

### What Gets Logged

**During Training:**
- Model parameters (total params, trainable params, device)
- Training configuration (learning rate, batch size, epochs, etc.)
- Per-epoch metrics: train_loss, val_loss, val_auroc, val_accuracy, val_f1
- Best model checkpoint

**During Inference/Evaluation:**
- Inference metrics: AUROC, accuracy, precision, recall, F1
- Confusion matrix values (TP, TN, FP, FN)
- Prediction validation checks

### Viewing MLflow UI

To view experiment results in the MLflow dashboard:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Start MLflow UI
mlflow ui --backend-store-uri mlruns
```

Then open http://localhost:5000 in your browser.

## Prediction Validation Checks

The inference pipelines include automatic validation checks on predictions:

| Check | Description |
|-------|-------------|
| `no_null_predictions` | Ensures no null values in predictions |
| `valid_binary_predictions` | Verifies predictions are 0 or 1 |
| `valid_probability_range` | Confirms probabilities are in [0, 1] |
| `no_null_probabilities` | Ensures no null probability values |
| `prediction_count` | Verifies predictions were generated |
| `class_distribution` | Checks for reasonable class balance |
| `better_than_random` | Validates AUROC > 0.5 (better than random) |

## Performance Improvement Techniques

The project implements several techniques proven to improve ViLBERT performance on Hateful Memes:

### 1. Caption Enrichment (CES) - +2-6% AUROC

Adds image captions to the text input using BLIP model. This helps the model understand visual content better.

```yaml
# In conf/base/parameters.yml
data_processing:
  use_captions: true  # Enable caption enrichment
```

Reference: [Caption Enriched Samples for Improving Hateful Memes Detection](https://aclanthology.org/2021.emnlp-main.738/) (EMNLP 2021)

### 2. Focal Loss - +1-2% AUROC

Handles class imbalance by down-weighting easy examples and focusing on hard negatives.

```yaml
# In conf/base/parameters.yml
training:
  loss_type: "focal"  # Options: ce, focal, label_smoothing, focal_smoothing
  focal_alpha: 0.35   # Weight for minority class
  focal_gamma: 2.0    # Focusing parameter
```

### 3. Proper Visual Feature Extraction

Uses pretrained ResNet-152 or CLIP for visual features instead of raw pixels.

```yaml
# In conf/base/parameters.yml
vilbert:
  feature_extractor: "resnet"  # or "clip" for better results
```

### 4. Training Hyperparameters (Facebook MMF Baseline)

Matches the official Facebook MMF configuration:

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-5 |
| Batch size | 32 |
| Warmup steps | 2000 |
| Optimizer | AdamW |
| Gradient clipping | 1.0 |

### Expected Performance

| Configuration | Expected AUROC |
|--------------|----------------|
| Baseline (pixel features) | ~0.62 |
| + ResNet features | ~0.65-0.68 |
| + Focal loss | ~0.66-0.69 |
| + Caption enrichment | ~0.68-0.72 |
| + CLIP features | ~0.70-0.72 |
| Facebook ViLBERT baseline | 0.7045 |

---

## Adding New Models

This project uses a modular architecture that makes it easy to add new models and feature extractors. All models are registered using decorators and follow a consistent interface.

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
│       ├── clip.py              # CLIP semantic features
│       └── fasterrcnn.py        # Faster R-CNN object features
└── pipelines/
    └── model_training/
        ├── nodes.py             # Training logic
        └── pipeline.py          # Pipeline definitions
```

### Adding a New Multimodal Model

1. **Create a new file** in `src/multimodalclassification/models/` (e.g., `my_model.py`)

2. **Inherit from `BaseMultimodalModel`** and implement required methods:

```python
# src/multimodalclassification/models/my_model.py
import torch
import torch.nn as nn
from .base import BaseMultimodalModel, register_model


@register_model("my_model")  # Register with a unique name
class MyMultimodalModel(BaseMultimodalModel):
    """
    My custom multimodal model.
    
    Combines visual and text features for classification.
    """
    
    def __init__(self, num_labels: int = 2, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        # Initialize your model components here
        self.text_encoder = ...
        self.vision_encoder = ...
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
        visual_attention_mask: torch.Tensor = None,
        spatial_locations: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Text attention mask [batch, seq_len]
            visual_features: Visual features [batch, num_regions, feature_dim]
            visual_attention_mask: Visual attention mask [batch, num_regions]
            spatial_locations: Bounding box coordinates [batch, num_regions, 5]
            labels: Ground truth labels [batch]
        
        Returns:
            dict with 'logits' and optionally 'loss'
        """
        # Your forward logic here
        logits = self.classifier(combined_features)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {"logits": logits, "loss": loss}
    
    @classmethod
    def from_pretrained(cls, model_path: str, num_labels: int = 2, **kwargs):
        """Load pretrained weights."""
        model = cls(num_labels=num_labels, **kwargs)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        return model
    
    def get_num_parameters(self) -> tuple:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
```

3. **Export in `__init__.py`**:

```python
# In src/multimodalclassification/models/__init__.py
from .my_model import MyMultimodalModel
```

4. **Create a pipeline** (optional - for separate training):

```python
# In src/multimodalclassification/pipelines/model_training/pipeline.py
def create_my_model_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func=create_dataloaders, ...),
        node(func=load_my_model, ...),
        node(func=train_model, ...),
        node(func=evaluate_model, ...),
    ])
```

5. **Add configuration** in `conf/base/parameters.yml`:

```yaml
my_model:
  num_labels: 2
  pretrained_path: "weights/my_model.bin"
  feature_extractor: "resnet"  # or clip, fasterrcnn
```

### Adding a New Feature Extractor

1. **Create a new file** in `src/multimodalclassification/models/feature_extractors/`

2. **Inherit from `BaseFeatureExtractor`**:

```python
# src/multimodalclassification/models/feature_extractors/my_extractor.py
import torch
from PIL import Image
from ..base import BaseFeatureExtractor, register_feature_extractor


@register_feature_extractor("my_extractor")
class MyFeatureExtractor(BaseFeatureExtractor):
    """
    My custom visual feature extractor.
    """
    
    def __init__(
        self,
        output_dim: int = 2048,
        num_regions: int = 36,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(output_dim, num_regions, device)
        # Initialize your backbone
        self.backbone = ...
        self.backbone.eval()
    
    def extract_features(self, image: Image.Image) -> tuple:
        """
        Extract visual features from an image.
        
        Args:
            image: PIL Image
        
        Returns:
            tuple of (features, spatial_locations)
            - features: [num_regions, output_dim]
            - spatial_locations: [num_regions, 5] (x1, y1, x2, y2, area)
        """
        with torch.no_grad():
            # Your extraction logic
            features = self.backbone(preprocessed_image)
            spatial = self._compute_spatial_locations(...)
        
        return features, spatial
```

3. **Export in feature_extractors/__init__.py**:

```python
from .my_extractor import MyFeatureExtractor
```

### Using Registry Functions

The registry system allows dynamic model/extractor loading:

```python
from multimodalclassification.models import get_model, get_feature_extractor

# Load a model by name
model = get_model("vilbert_hf", num_labels=2)

# Load a feature extractor by name  
extractor = get_feature_extractor("resnet", output_dim=2048, num_regions=36)

# List available models/extractors
from multimodalclassification.models import list_available_models, list_available_extractors
list_available_models()
list_available_extractors()
```

### Available Models

| Model Name | Description | Feature Extractor |
|------------|-------------|-------------------|
| `vilbert_hf` | ViLBERT with HuggingFace community weights | Any |
| `vilbert_facebook` | ViLBERT with Facebook's official CC weights | fasterrcnn (recommended) |

### Available Feature Extractors

| Extractor | Description | Output Dim | Regions |
|-----------|-------------|------------|---------|
| `resnet` | ResNet-152 grid features | 2048 | 36 (6x6) |
| `clip` | CLIP ViT-B/32 semantic features | 512 | 49 (7x7) |
| `fasterrcnn` | Faster R-CNN object detection (COCO, 91 classes) | 2048 | 36 |
| `fasterrcnn_vg` | Faster R-CNN object detection (Visual Genome, 1600 classes) | 2048 | 36 |

**Note:** The `fasterrcnn_vg` extractor requires downloading pretrained weights from:
https://drive.google.com/file/d/18n_3V1rywgeADZ3oONO0DsuuS9eMW6sN/view

Save to `weights/faster_rcnn_res101_vg.pth`

---

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the files `tests/test_run.py` and `tests/pipelines/data_science/test_pipeline.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.

## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
