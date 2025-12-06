"""
Base Model Classes for Multimodal Classification

This module provides abstract base classes that define the interface
for all models in the pipeline. New models should inherit from these
base classes to ensure compatibility with the training pipeline.

Example:
    To create a new model, inherit from BaseMultimodalModel:

    >>> class MyNewModel(BaseMultimodalModel):
    ...     def __init__(self, config):
    ...         super().__init__(config)
    ...         # Initialize your model components
    ...
    ...     def forward(self, input_ids, attention_mask, visual_features, ...):
    ...         # Implement forward pass
    ...         return {"logits": logits, "loss": loss}
    ...
    ...     @classmethod
    ...     def from_pretrained(cls, model_path, **kwargs):
    ...         # Load pretrained weights
    ...         return model
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BaseMultimodalModel(nn.Module, ABC):
    """
    Abstract base class for all multimodal classification models.

    All models in this pipeline should inherit from this class to ensure
    a consistent interface for training, evaluation, and inference.

    Required methods to implement:
        - forward(): Process inputs and return logits/loss
        - from_pretrained(): Load pretrained weights

    Optional methods to override:
        - freeze_layers(): Freeze specific layers for fine-tuning
        - get_num_parameters(): Return parameter counts
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.num_labels = self.config.get("num_labels", 2)

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor,
        visual_attention_mask: Optional[torch.Tensor] = None,
        spatial_locations: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Text token IDs [batch, seq_len]
            attention_mask: Text attention mask [batch, seq_len]
            visual_features: Visual region features [batch, num_regions, feature_dim]
            visual_attention_mask: Visual region mask [batch, num_regions]
            spatial_locations: Normalized bbox coords [batch, num_regions, 5]
            token_type_ids: Text segment IDs [batch, seq_len]
            labels: Ground truth labels [batch] for computing loss

        Returns:
            Dictionary containing:
                - logits: Classification logits [batch, num_labels]
                - loss: Cross-entropy loss (if labels provided)
                - (optional) pooled_output, hidden_states, etc.
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        model_path: str,
        num_labels: int = 2,
        **kwargs,
    ) -> "BaseMultimodalModel":
        """
        Load a pretrained model from a checkpoint or model hub.

        Args:
            model_path: Path to checkpoint or HuggingFace model ID
            num_labels: Number of classification labels
            **kwargs: Additional model-specific arguments

        Returns:
            Initialized model with loaded weights
        """
        pass

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions from logits.

        Args:
            logits: Raw model output [batch, num_labels]

        Returns:
            Predicted class indices [batch]
        """
        return torch.argmax(logits, dim=-1)

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions from logits.

        Args:
            logits: Raw model output [batch, num_labels]

        Returns:
            Class probabilities [batch, num_labels]
        """
        return F.softmax(logits, dim=-1)

    def freeze_layers(self, num_layers: int = 0) -> None:
        """
        Freeze the first N layers for efficient fine-tuning.

        Override this method to implement model-specific freezing logic.

        Args:
            num_layers: Number of layers to freeze from the bottom
        """
        logger.warning(
            f"{self.__class__.__name__} does not implement freeze_layers(). "
            "All parameters will remain trainable."
        )

    def get_num_parameters(self) -> Tuple[int, int]:
        """
        Return the total and trainable parameter counts.

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def save_pretrained(self, save_path: str) -> None:
        """
        Save model weights to a file.

        Args:
            save_path: Path to save the model checkpoint
        """
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config,
                "num_labels": self.num_labels,
            },
            save_path,
        )
        logger.info(f"Model saved to {save_path}")


class BaseFeatureExtractor(nn.Module, ABC):
    """
    Abstract base class for visual feature extractors.

    All feature extractors should inherit from this class to ensure
    a consistent interface for extracting visual features from images.

    Required methods to implement:
        - extract_features(): Extract features from a single image
        - forward(): Batch feature extraction
    """

    def __init__(
        self,
        output_dim: int = 2048,
        num_regions: int = 36,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the feature extractor.

        Args:
            output_dim: Output feature dimension per region
            num_regions: Number of visual regions to extract
            device: Device to run the extractor on
        """
        super().__init__()
        self.output_dim = output_dim
        self.num_regions = num_regions
        self.device = device

    @abstractmethod
    def extract_features(self, image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract visual features from a single image.

        Args:
            image: PIL Image or torch.Tensor

        Returns:
            Tuple of:
                - visual_features: [num_regions, output_dim]
                - spatial_locations: [num_regions, 5] (x1, y1, x2, y2, area)
        """
        pass

    @abstractmethod
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch feature extraction.

        Args:
            images: Batch of images [batch, 3, H, W]

        Returns:
            Tuple of:
                - visual_features: [batch, num_regions, output_dim]
                - spatial_locations: [batch, num_regions, 5]
        """
        pass

    def _generate_grid_spatial(self, num_regions: int = None) -> torch.Tensor:
        """
        Generate spatial locations for grid-based regions.

        Args:
            num_regions: Number of regions (defaults to self.num_regions)

        Returns:
            Spatial coordinates [num_regions, 5]
        """
        if num_regions is None:
            num_regions = self.num_regions

        grid_size = int(num_regions**0.5)
        spatial = []

        for i in range(grid_size):
            for j in range(grid_size):
                x1 = j / grid_size
                y1 = i / grid_size
                x2 = (j + 1) / grid_size
                y2 = (i + 1) / grid_size
                area = (x2 - x1) * (y2 - y1)
                spatial.append([x1, y1, x2, y2, area])

        return torch.tensor(spatial, device=self.device)


# Registry for models and feature extractors
MODEL_REGISTRY: Dict[str, type] = {}
FEATURE_EXTRACTOR_REGISTRY: Dict[str, type] = {}


def register_model(name: str):
    """
    Decorator to register a model class.

    Usage:
        @register_model("my_model")
        class MyModel(BaseMultimodalModel):
            ...
    """

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def register_feature_extractor(name: str):
    """
    Decorator to register a feature extractor class.

    Usage:
        @register_feature_extractor("my_extractor")
        class MyExtractor(BaseFeatureExtractor):
            ...
    """

    def decorator(cls):
        FEATURE_EXTRACTOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs) -> BaseMultimodalModel:
    """
    Get a model instance by name.

    Args:
        name: Registered model name
        **kwargs: Arguments to pass to the model constructor

    Returns:
        Initialized model instance
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)


def get_feature_extractor(name: str, **kwargs) -> BaseFeatureExtractor:
    """
    Get a feature extractor instance by name.

    Args:
        name: Registered extractor name
        **kwargs: Arguments to pass to the extractor constructor

    Returns:
        Initialized feature extractor instance
    """
    if name not in FEATURE_EXTRACTOR_REGISTRY:
        available = list(FEATURE_EXTRACTOR_REGISTRY.keys())
        raise ValueError(f"Unknown feature extractor: {name}. Available: {available}")
    return FEATURE_EXTRACTOR_REGISTRY[name](**kwargs)
