"""
Multimodal Classification Models

This module provides various models and feature extractors for
multimodal (vision + language) classification tasks.

Available Models:
    - ViLBERTHuggingFace: ViLBERT with HuggingFace community weights
    - ViLBERTFacebook: ViLBERT with Facebook's official weights

Available Feature Extractors:
    - ResNetFeatureExtractor: Grid-based features using ResNet-152
    - CLIPFeatureExtractor: Semantic features using CLIP
    - FasterRCNNFeatureExtractor: Object-based features using Faster R-CNN

Usage:
    # Load a model
    from multimodalclassification.models import ViLBERTHuggingFace
    model = ViLBERTHuggingFace.from_pretrained("visualjoyce/transformers4vl-vilbert")

    # Load a feature extractor
    from multimodalclassification.models import get_feature_extractor
    extractor = get_feature_extractor("resnet", output_dim=2048)

Adding a New Model:
    See README.md for detailed instructions on adding new models.
    In brief:
    1. Create a new file in models/ (e.g., my_model.py)
    2. Inherit from BaseMultimodalModel
    3. Implement forward() and from_pretrained() methods
    4. Register with @register_model("my_model")
"""

from .base import (
    FEATURE_EXTRACTOR_REGISTRY,
    MODEL_REGISTRY,
    BaseFeatureExtractor,
    BaseMultimodalModel,
    get_feature_extractor,
    get_model,
    register_feature_extractor,
    register_model,
)
from .feature_extractors import (
    CLIPFeatureExtractor,
    FasterRCNNFeatureExtractor,
    FasterRCNNVGExtractor,
    ResNetFeatureExtractor,
    download_vg_weights,
)
from .vilbert_facebook import ViLBERTFacebook, load_vilbert_facebook
from .vilbert_facebook_arch import (
    ViLBERTForClassification as ViLBERTFacebookArch,
)
from .vilbert_facebook_arch import (
    get_facebook_vilbert_config,
    load_facebook_weights,
)
from .vilbert_hf import ViLBERTHuggingFace, load_vilbert_from_huggingface

__all__ = [
    # Base classes
    "BaseMultimodalModel",
    "BaseFeatureExtractor",
    # Registries
    "MODEL_REGISTRY",
    "FEATURE_EXTRACTOR_REGISTRY",
    # Decorators
    "register_model",
    "register_feature_extractor",
    # Factory functions
    "get_model",
    "get_feature_extractor",
    # Models
    "ViLBERTHuggingFace",
    "ViLBERTFacebook",
    "ViLBERTFacebookArch",
    "load_vilbert_from_huggingface",
    "load_vilbert_facebook",
    "get_facebook_vilbert_config",
    "load_facebook_weights",
    # Feature Extractors
    "ResNetFeatureExtractor",
    "CLIPFeatureExtractor",
    "FasterRCNNFeatureExtractor",
    "FasterRCNNVGExtractor",
    "download_vg_weights",
]


def list_available_models():
    """List all registered models."""
    print("\n" + "=" * 60)
    print("Available Models")
    print("=" * 60)
    for name, cls in MODEL_REGISTRY.items():
        doc = cls.__doc__.split("\n")[0] if cls.__doc__ else "No description"
        print(f"\n  {name}:")
        print(f"    {doc}")
    print("=" * 60 + "\n")


def list_available_extractors():
    """List all registered feature extractors."""
    print("\n" + "=" * 60)
    print("Available Feature Extractors")
    print("=" * 60)
    for name, cls in FEATURE_EXTRACTOR_REGISTRY.items():
        doc = cls.__doc__.split("\n")[0] if cls.__doc__ else "No description"
        print(f"\n  {name}:")
        print(f"    {doc}")
    print("=" * 60 + "\n")
