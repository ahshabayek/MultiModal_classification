"""
ViLBERT from HuggingFace Hub (Community Upload)

This module provides an easy way to load ViLBERT pretrained weights
from HuggingFace Hub without manually downloading files.

Available Models on HuggingFace Hub:
- visualjoyce/transformers4vl-vilbert (Conceptual Captions pretrained)
- visualjoyce/transformers4vl-vilbert-mt (Multi-task pretrained)

Usage:
    from models.vilbert_huggingface import ViLBERTHuggingFace

    # Load pretrained ViLBERT from HuggingFace
    model = ViLBERTHuggingFace.from_pretrained(
        "visualjoyce/transformers4vl-vilbert",
        num_labels=2
    )

    # Or use the convenience function
    model = load_vilbert_from_huggingface(num_labels=2)
"""

import logging
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import huggingface_hub
try:
    from huggingface_hub import HfApi, hf_hub_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Import our custom ViLBERT implementation
try:
    from .vilbert import (
        BertLayerNorm,
        ViLBERTEmbeddings,
        ViLBERTEncoder,
        ViLBERTForClassification,
        ViLBERTModel,
        get_vilbert_config,
    )
except ImportError:
    from vilbert import (
        BertLayerNorm,
        ViLBERTEmbeddings,
        ViLBERTEncoder,
        ViLBERTForClassification,
        ViLBERTModel,
        get_vilbert_config,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Available HuggingFace models
HUGGINGFACE_VILBERT_MODELS = {
    "visualjoyce/transformers4vl-vilbert": {
        "description": "ViLBERT pretrained on Conceptual Captions",
        "filename": "pytorch_model.bin",
        "config_file": "config.json",
    },
    "visualjoyce/transformers4vl-vilbert-mt": {
        "description": "ViLBERT Multi-Task pretrained",
        "filename": "pytorch_model.bin",
        "config_file": "config.json",
    },
}


class ViLBERTHuggingFace(nn.Module):
    """
    ViLBERT model that can be loaded directly from HuggingFace Hub.

    This is a wrapper around our custom ViLBERT implementation that
    handles downloading and loading weights from HuggingFace.

    Example:
        >>> model = ViLBERTHuggingFace.from_pretrained(
        ...     "visualjoyce/transformers4vl-vilbert", num_labels=2
        ... )
        >>>
        >>> # Forward pass
        >>> outputs = model(
        ...     input_ids=input_ids,
        ...     attention_mask=attention_mask,
        ...     visual_features=visual_features,
        ...     visual_attention_mask=visual_attention_mask,
        ...     labels=labels,
        ... )
        >>> loss = outputs["loss"]
        >>> logits = outputs["logits"]
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
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
        super().__init__()

        self.config = config or get_vilbert_config()
        self.num_labels = num_labels

        # Create the underlying ViLBERT model
        self.model = ViLBERTForClassification(
            config=self.config, num_labels=num_labels, bert_model_name=bert_model_name
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
            **kwargs: Additional arguments passed to model init

        Returns:
            ViLBERTHuggingFace model with loaded weights

        Example:
            >>> model = ViLBERTHuggingFace.from_pretrained(
            ...     "visualjoyce/transformers4vl-vilbert", num_labels=2
            ... )
        """
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required to load from HuggingFace. "
                "Install with: pip install huggingface_hub"
            )

        logger.info(f"Loading ViLBERT from: {model_name_or_path}")

        # Check if it's a local path
        if os.path.isdir(model_name_or_path):
            weight_path = os.path.join(model_name_or_path, "pytorch_model.bin")
            if not os.path.exists(weight_path):
                weight_path = os.path.join(model_name_or_path, "model.bin")
        else:
            # Download from HuggingFace Hub
            try:
                weight_path = hf_hub_download(
                    repo_id=model_name_or_path,
                    filename="pytorch_model.bin",
                    cache_dir=cache_dir,
                    force_download=force_download,
                )
                logger.info(f"Downloaded weights to: {weight_path}")
            except Exception as e:
                logger.error(f"Failed to download from HuggingFace: {e}")
                raise

        # Create model instance
        model = cls(num_labels=num_labels, **kwargs)

        # Load weights
        model._load_pretrained_weights(weight_path)

        return model

    def _load_pretrained_weights(self, weight_path: str, strict: bool = False):
        """
        Load pretrained weights into the model.

        Args:
            weight_path: Path to the weight file
            strict: Whether to strictly enforce key matching
        """
        logger.info(f"Loading weights from: {weight_path}")

        state_dict = torch.load(weight_path, map_location="cpu")

        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        # Get model state dict
        model_state_dict = self.model.state_dict()

        # Match keys and load
        new_state_dict = {}
        matched_keys = 0
        unmatched_keys = []

        for key, value in state_dict.items():
            # Try different key formats
            possible_keys = [
                key,
                key.replace("module.", ""),
                key.replace("vilbert.", ""),
                key.replace("model.", ""),
                f"model.{key}",
                f"vilbert.{key}",
            ]

            matched = False
            for possible_key in possible_keys:
                if possible_key in model_state_dict:
                    if model_state_dict[possible_key].shape == value.shape:
                        new_state_dict[possible_key] = value
                        matched_keys += 1
                        matched = True
                        break

            if not matched:
                unmatched_keys.append(key)

        # Load matched weights
        self.model.load_state_dict(new_state_dict, strict=False)

        logger.info(f"Loaded {matched_keys} weight tensors")
        logger.info(f"Unmatched: {len(unmatched_keys)} keys")

        if unmatched_keys and len(unmatched_keys) < 20:
            logger.debug(f"Unmatched keys: {unmatched_keys}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        visual_features: torch.Tensor = None,
        visual_attention_mask: Optional[torch.Tensor] = None,
        spatial_locations: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ViLBERT.

        Args:
            input_ids: Text token IDs [batch, seq_len]
            attention_mask: Text attention mask [batch, seq_len]
            token_type_ids: Text segment IDs [batch, seq_len]
            visual_features: Faster R-CNN region features [batch, num_regions, 2048]
            visual_attention_mask: Visual region mask [batch, num_regions]
            spatial_locations: Normalized bbox coords [batch, num_regions, 5]
            labels: Ground truth labels [batch] for computing loss

        Returns:
            Dictionary with:
                - logits: Classification logits [batch, num_labels]
                - loss: Cross-entropy loss (if labels provided)
                - pooled_output: Concatenated pooled features
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_features=visual_features,
            visual_attention_mask=visual_attention_mask,
            spatial_locations=spatial_locations,
            labels=labels,
        )

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Get class predictions from logits."""
        return torch.argmax(logits, dim=-1)

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Get probability predictions from logits."""
        return F.softmax(logits, dim=-1)

    def freeze_bert_layers(self, num_layers: int = 6):
        """
        Freeze the first N BERT layers for efficient fine-tuning.

        Args:
            num_layers: Number of layers to freeze (from bottom)
        """
        # Freeze embeddings
        for param in self.model.vilbert.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified layers
        for i, layer in enumerate(self.model.vilbert.bert.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Froze {num_layers} BERT layers")
        logger.info(f"Trainable: {trainable:,} / {total:,} parameters")

    def get_num_parameters(self) -> Tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


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

    Example:
        >>> # Load for Hateful Memes (binary classification)
        >>> model = load_vilbert_from_huggingface(num_labels=2)
        >>>
        >>> # Load with frozen layers for efficient fine-tuning
        >>> model = load_vilbert_from_huggingface(num_labels=2, freeze_bert_layers=6)
    """
    model = ViLBERTHuggingFace.from_pretrained(
        model_name, num_labels=num_labels, **kwargs
    )

    if freeze_bert_layers > 0:
        model.freeze_bert_layers(freeze_bert_layers)

    return model


def list_available_models():
    """List available ViLBERT models on HuggingFace Hub."""
    print("\n" + "=" * 60)
    print("Available ViLBERT Models on HuggingFace Hub")
    print("=" * 60)

    for repo_id, info in HUGGINGFACE_VILBERT_MODELS.items():
        print(f"\n🤗 {repo_id}")
        print(f"   {info['description']}")
        print(f"   Files: {info['filename']}")

    print("\n" + "-" * 60)
    print("Usage:")
    print(
        '  model = ViLBERTHuggingFace.from_pretrained("visualjoyce/transformers4vl-vilbert")'
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # List available models
    list_available_models()

    # Test loading (if huggingface_hub is available)
    if HF_HUB_AVAILABLE:
        print("\n🔧 Testing model loading from HuggingFace...")

        try:
            # Load model
            model = ViLBERTHuggingFace.from_pretrained(
                "visualjoyce/transformers4vl-vilbert", num_labels=2
            )

            total, trainable = model.get_num_parameters()
            print(f"✓ Model loaded successfully!")
            print(f"  Total parameters: {total:,}")
            print(f"  Trainable parameters: {trainable:,}")

            # Test forward pass with dummy data
            print("\n🧪 Testing forward pass...")
            batch_size = 2
            seq_len = 128
            num_regions = 36

            dummy_inputs = {
                "input_ids": torch.randint(0, 30000, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len),
                "visual_features": torch.randn(batch_size, num_regions, 2048),
                "visual_attention_mask": torch.ones(batch_size, num_regions),
                "labels": torch.tensor([0, 1]),
            }

            outputs = model(**dummy_inputs)

            print(f"✓ Forward pass successful!")
            print(f"  Logits shape: {outputs['logits'].shape}")
            print(f"  Loss: {outputs['loss'].item():.4f}")

            # Test with frozen layers
            print("\n🔒 Testing layer freezing...")
            model.freeze_bert_layers(6)
            total, trainable = model.get_num_parameters()
            print(f"  After freezing 6 layers:")
            print(f"  Trainable parameters: {trainable:,}")

            print("\n✅ All tests passed!")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("\n⚠️  Install huggingface_hub to test: pip install huggingface_hub")
