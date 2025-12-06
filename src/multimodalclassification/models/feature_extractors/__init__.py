"""
Visual Feature Extractors

This module provides various visual feature extractors for multimodal models.

Available Extractors:
    - ResNetFeatureExtractor: Grid-based features using ResNet-152
    - CLIPFeatureExtractor: Semantic features using CLIP
    - FasterRCNNFeatureExtractor: Object-based features using Faster R-CNN (COCO)
    - FasterRCNNVGExtractor: Object-based features using Faster R-CNN (Visual Genome)
    - GridFeatsX152Extractor: Facebook's X-152++ from grid-feats-vqa (best quality)
    - GridFeatsX152StandardExtractor: Facebook's standard X-152

Usage:
    from multimodalclassification.models.feature_extractors import get_feature_extractor

    extractor = get_feature_extractor("resnet", output_dim=2048, num_regions=36)
    features, spatial = extractor.extract_features(image)

    # For best quality features (requires detectron2):
    extractor = get_feature_extractor("grid_x152", weights_path="weights/X-152pp.pth")
"""

from .clip import CLIPFeatureExtractor
from .fasterrcnn import FasterRCNNFeatureExtractor
from .fasterrcnn_vg import FasterRCNNVGExtractor, download_vg_weights
from .resnet import ResNetFeatureExtractor

# Conditionally import grid features extractor (requires detectron2)
try:
    from .grid_feats_x152 import (
        GridFeatsX152Extractor,
        GridFeatsX152StandardExtractor,
        download_x152pp_weights,
    )

    GRID_FEATS_AVAILABLE = True
except ImportError:
    GRID_FEATS_AVAILABLE = False
    GridFeatsX152Extractor = None
    GridFeatsX152StandardExtractor = None
    download_x152pp_weights = None

__all__ = [
    "ResNetFeatureExtractor",
    "CLIPFeatureExtractor",
    "FasterRCNNFeatureExtractor",
    "FasterRCNNVGExtractor",
    "GridFeatsX152Extractor",
    "GridFeatsX152StandardExtractor",
    "get_feature_extractor",
    "download_vg_weights",
    "download_x152pp_weights",
    "GRID_FEATS_AVAILABLE",
]


def get_feature_extractor(name: str, **kwargs):
    """
    Get a feature extractor by name.

    Args:
        name: Extractor name. Options:
            - "resnet": ResNet-152 grid features (fast, ~0.66 AUROC)
            - "clip": CLIP semantic features
            - "fasterrcnn": Faster R-CNN COCO features (~0.62 AUROC)
            - "fasterrcnn_vg": Faster R-CNN Visual Genome features
            - "grid_x152": Facebook's X-152++ (best quality, requires detectron2)
            - "grid_x152_standard": Facebook's standard X-152
        **kwargs: Arguments passed to the extractor constructor

    Returns:
        Initialized feature extractor

    Example:
        >>> extractor = get_feature_extractor("resnet", output_dim=2048)
        >>> features, spatial = extractor.extract_features(image)

        # For best quality (requires detectron2):
        >>> extractor = get_feature_extractor(
        ...     "grid_x152", weights_path="weights/X-152pp.pth"
        ... )
    """
    extractors = {
        "resnet": ResNetFeatureExtractor,
        "clip": CLIPFeatureExtractor,
        "fasterrcnn": FasterRCNNFeatureExtractor,
        "fasterrcnn_vg": FasterRCNNVGExtractor,
    }

    # Add grid features extractors if detectron2 is available
    if GRID_FEATS_AVAILABLE:
        extractors["grid_x152"] = GridFeatsX152Extractor
        extractors["grid_x152_standard"] = GridFeatsX152StandardExtractor

    if name not in extractors:
        available = list(extractors.keys())
        if name in ("grid_x152", "grid_x152_standard") and not GRID_FEATS_AVAILABLE:
            raise ImportError(
                f"'{name}' requires detectron2. Install with: "
                "pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            )
        raise ValueError(f"Unknown extractor: {name}. Available: {available}")

    return extractors[name](**kwargs)
