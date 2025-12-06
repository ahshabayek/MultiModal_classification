"""
ViLBERT with Facebook Official Weights

This model uses Facebook's official pretrained weights from the
vilbert-multi-task repository. It's designed for use with Faster R-CNN
visual features (object-based regions).

Weights: https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin
Pretrained on: Conceptual Captions
Expected AUROC: ~0.70 with proper Faster R-CNN (Visual Genome) features

Usage:
    from multimodalclassification.models import ViLBERTFacebook

    model = ViLBERTFacebook.from_pretrained(
        "weights/vilbert_pretrained_cc.bin",
        num_labels=2
    )

Note:
    For best results, use with Visual Genome Faster R-CNN features.
    Using COCO Faster R-CNN features may result in lower performance
    due to domain mismatch with pretraining.
"""

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseMultimodalModel, register_model

logger = logging.getLogger(__name__)


def get_vilbert_config() -> Dict[str, Any]:
    """Get default ViLBERT configuration matching Facebook's setup."""
    return {
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "v_hidden_size": 1024,
        "v_num_attention_heads": 8,
        "v_num_hidden_layers": 6,
        "v_intermediate_size": 1024,
        "v_attention_probs_dropout_prob": 0.1,
        "v_hidden_act": "gelu",
        "v_hidden_dropout_prob": 0.1,
        "v_feature_size": 2048,
        "v_loc_size": 5,
        "num_labels": 2,
    }


@register_model("vilbert_facebook")
class ViLBERTFacebook(BaseMultimodalModel):
    """
    ViLBERT model with Facebook's official pretrained weights.

    This model is designed to work with Faster R-CNN visual features,
    matching Facebook's original setup for the Hateful Memes challenge.

    Key differences from ViLBERTHuggingFace:
        - Uses Facebook's official weights (not community uploads)
        - Optimized for Faster R-CNN object-based features
        - Includes weight mapping for Facebook's checkpoint format

    Attributes:
        model: The underlying ViLBERTForClassification model
        config: Model configuration dictionary
        num_labels: Number of output classes
    """

    # Default paths for Facebook weights
    DEFAULT_WEIGHTS_URL = (
        "https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin"
    )
    DEFAULT_WEIGHTS_PATH = "weights/vilbert_pretrained_cc.bin"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        num_labels: int = 2,
        bert_model_name: str = "bert-base-uncased",
    ):
        """
        Initialize ViLBERT model.

        Args:
            config: Model configuration dictionary
            num_labels: Number of classification labels
            bert_model_name: BERT model to use for text encoding
        """
        config = config or get_vilbert_config()
        config["num_labels"] = num_labels
        super().__init__(config)

        # Import the underlying ViLBERT implementation
        import sys

        sys.path.insert(0, os.path.join(os.getcwd(), "data", "04_models"))
        from vilbert import ViLBERTForClassification

        self.model = ViLBERTForClassification(
            config=self.config,
            num_labels=num_labels,
            bert_model_name=bert_model_name,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = None,
        num_labels: int = 2,
        **kwargs,
    ) -> "ViLBERTFacebook":
        """
        Load ViLBERT with Facebook's official weights.

        Args:
            model_path: Path to the Facebook checkpoint file
                       If None, uses default path: weights/vilbert_pretrained_cc.bin
            num_labels: Number of classification labels
            **kwargs: Additional arguments

        Returns:
            ViLBERTFacebook model with loaded weights

        Raises:
            FileNotFoundError: If weights file not found
        """
        if model_path is None:
            model_path = cls.DEFAULT_WEIGHTS_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Facebook weights not found at: {model_path}\n"
                f"Download with: python scripts/download_weights.py --source vilbert_cc --output ./weights/"
            )

        logger.info(f"Loading ViLBERT with Facebook weights from: {model_path}")

        # Create model instance
        model = cls(num_labels=num_labels, **kwargs)

        # Load Facebook's weights
        model._load_facebook_weights(model_path)

        return model

    def _load_facebook_weights(self, weight_path: str):
        """
        Load Facebook's official pretrained weights.

        Facebook's checkpoint has a specific format that requires
        careful key mapping to our model architecture.
        """
        logger.info(f"Loading Facebook weights from: {weight_path}")

        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model_state_dict = self.model.state_dict()
        new_state_dict = {}
        matched = 0
        unmatched = []

        for key, value in state_dict.items():
            # Facebook's keys often start with "bert."
            # Try various key mappings
            possible_keys = [
                key,
                f"model.{key}",
                f"model.vilbert.{key}",
                key.replace("bert.", "model.vilbert.bert."),
                key.replace("bert.", "vilbert.bert."),
                key.replace("v_embeddings.", "model.vilbert.v_embeddings."),
                key.replace("encoder.", "model.vilbert.encoder."),
            ]

            found = False
            for pkey in possible_keys:
                if pkey in model_state_dict:
                    if model_state_dict[pkey].shape == value.shape:
                        new_state_dict[pkey] = value
                        matched += 1
                        found = True
                        break

            if not found:
                unmatched.append(key)

        self.model.load_state_dict(new_state_dict, strict=False)

        logger.info(f"Loaded {matched} weight tensors from Facebook checkpoint")
        if unmatched:
            logger.debug(f"Unmatched keys ({len(unmatched)}): {unmatched[:10]}...")

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
        """Forward pass through ViLBERT."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_features=visual_features,
            visual_attention_mask=visual_attention_mask,
            spatial_locations=spatial_locations,
            labels=labels,
        )

    def freeze_layers(self, num_layers: int = 6) -> None:
        """Freeze the first N BERT layers for efficient fine-tuning."""
        # Freeze embeddings
        for param in self.model.vilbert.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified layers
        for i, layer in enumerate(self.model.vilbert.bert.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        total, trainable = self.get_num_parameters()
        logger.info(f"Froze {num_layers} layers. Trainable: {trainable:,} / {total:,}")

    # Alias for backwards compatibility
    def freeze_bert_layers(self, num_layers: int = 6) -> None:
        """Alias for freeze_layers()."""
        self.freeze_layers(num_layers)


def load_vilbert_facebook(
    weights_path: str = None,
    num_labels: int = 2,
    freeze_layers: int = 0,
    **kwargs,
) -> ViLBERTFacebook:
    """
    Convenience function to load ViLBERT with Facebook weights.

    Args:
        weights_path: Path to Facebook checkpoint (default: weights/vilbert_pretrained_cc.bin)
        num_labels: Number of classification labels
        freeze_layers: Number of layers to freeze (0 = none)
        **kwargs: Additional arguments

    Returns:
        ViLBERTFacebook model ready for fine-tuning
    """
    model = ViLBERTFacebook.from_pretrained(
        model_path=weights_path,
        num_labels=num_labels,
        **kwargs,
    )

    if freeze_layers > 0:
        model.freeze_layers(freeze_layers)

    return model
