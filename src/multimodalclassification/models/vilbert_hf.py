"""
ViLBERT with HuggingFace Weights

This model uses community-uploaded ViLBERT weights from HuggingFace Hub.
It's designed for use with grid-based visual features (ResNet, CLIP).

Model: visualjoyce/transformers4vl-vilbert
Pretrained on: Conceptual Captions
Expected AUROC: ~0.65-0.67 with ResNet features

Usage:
    from multimodalclassification.models import ViLBERTHuggingFace

    model = ViLBERTHuggingFace.from_pretrained(
        "visualjoyce/transformers4vl-vilbert",
        num_labels=2
    )
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMultimodalModel, register_model

logger = logging.getLogger(__name__)

# Try to import huggingface_hub
try:
    from huggingface_hub import hf_hub_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


def get_vilbert_config() -> Dict[str, Any]:
    """Get default ViLBERT configuration."""
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


@register_model("vilbert_hf")
class ViLBERTHuggingFace(BaseMultimodalModel):
    """
    ViLBERT model loaded from HuggingFace Hub.

    This model wraps the custom ViLBERT implementation and provides
    easy loading from HuggingFace community uploads.

    Attributes:
        model: The underlying ViLBERTForClassification model
        config: Model configuration dictionary
        num_labels: Number of output classes
    """

    # Available models on HuggingFace
    AVAILABLE_MODELS = {
        "visualjoyce/transformers4vl-vilbert": "ViLBERT pretrained on Conceptual Captions",
        "visualjoyce/transformers4vl-vilbert-mt": "ViLBERT Multi-Task pretrained",
    }

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
        from .vilbert_core import ViLBERTForClassification

        self.model = ViLBERTForClassification(
            config=self.config,
            num_labels=num_labels,
            bert_model_name=bert_model_name,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "visualjoyce/transformers4vl-vilbert",
        num_labels: int = 2,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        **kwargs,
    ) -> "ViLBERTHuggingFace":
        """
        Load ViLBERT from HuggingFace Hub or local path.

        Args:
            model_name_or_path: HuggingFace repo ID or local path
            num_labels: Number of classification labels
            cache_dir: Directory to cache downloaded files
            force_download: Whether to force re-download
            **kwargs: Additional arguments

        Returns:
            ViLBERTHuggingFace model with loaded weights
        """
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            )

        logger.info(f"Loading ViLBERT from: {model_name_or_path}")

        # Check if it's a local path
        if os.path.isdir(model_name_or_path):
            weight_path = os.path.join(model_name_or_path, "pytorch_model.bin")
            if not os.path.exists(weight_path):
                weight_path = os.path.join(model_name_or_path, "model.bin")
        else:
            # Download from HuggingFace Hub
            weight_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename="pytorch_model.bin",
                cache_dir=cache_dir,
                force_download=force_download,
            )
            logger.info(f"Downloaded weights to: {weight_path}")

        # Create model instance
        model = cls(num_labels=num_labels, **kwargs)

        # Load weights
        model._load_pretrained_weights(weight_path)

        return model

    def _load_pretrained_weights(self, weight_path: str):
        """Load pretrained weights with flexible key matching."""
        logger.info(f"Loading weights from: {weight_path}")

        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)

        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        model_state_dict = self.model.state_dict()
        new_state_dict = {}
        matched_keys = 0

        for key, value in state_dict.items():
            possible_keys = [
                key,
                key.replace("module.", ""),
                key.replace("vilbert.", ""),
                key.replace("model.", ""),
                f"model.{key}",
                f"vilbert.{key}",
            ]

            for possible_key in possible_keys:
                if possible_key in model_state_dict:
                    if model_state_dict[possible_key].shape == value.shape:
                        new_state_dict[possible_key] = value
                        matched_keys += 1
                        break

        self.model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Loaded {matched_keys} weight tensors")

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


def load_vilbert_from_huggingface(
    model_name: str = "visualjoyce/transformers4vl-vilbert",
    num_labels: int = 2,
    freeze_bert_layers: int = 0,
    **kwargs,
) -> ViLBERTHuggingFace:
    """
    Convenience function to load ViLBERT from HuggingFace.

    Args:
        model_name: HuggingFace model repository name
        num_labels: Number of classification labels
        freeze_bert_layers: Number of BERT layers to freeze (0 = none)
        **kwargs: Additional arguments

    Returns:
        ViLBERTHuggingFace model ready for fine-tuning
    """
    model = ViLBERTHuggingFace.from_pretrained(
        model_name, num_labels=num_labels, **kwargs
    )

    if freeze_bert_layers > 0:
        model.freeze_layers(freeze_bert_layers)

    return model
