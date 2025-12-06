"""
CLIP Feature Extractor

Extracts semantic visual features using OpenAI's CLIP vision encoder.
CLIP features are semantically richer than ResNet features and often
provide better performance for multimodal tasks.

Expected AUROC: ~0.68-0.70
Speed: Medium (transformer-based)

Usage:
    from multimodalclassification.models.feature_extractors import CLIPFeatureExtractor

    extractor = CLIPFeatureExtractor(output_dim=2048, num_regions=36)
    features, spatial = extractor.extract_features(image)
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from ..base import BaseFeatureExtractor, register_feature_extractor

logger = logging.getLogger(__name__)


@register_feature_extractor("clip")
class CLIPFeatureExtractor(BaseFeatureExtractor):
    """
    Extract visual features using CLIP's vision encoder.

    CLIP features are a good alternative to Faster R-CNN features
    and often provide competitive or better results due to their
    semantic richness from vision-language pretraining.

    Attributes:
        model: CLIP vision model
        processor: CLIP image processor
        projection: Linear projection to output dimension
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_dim: int = 2048,
        num_regions: int = 36,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize CLIP feature extractor.

        Args:
            model_name: HuggingFace CLIP model name
            output_dim: Output feature dimension
            num_regions: Number of visual regions
            device: Device to run the extractor on
        """
        super().__init__(output_dim, num_regions, device)

        self.use_clip = False

        try:
            from transformers import CLIPModel, CLIPProcessor

            logger.info(f"Loading CLIP model: {model_name}")

            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            self.model.to(device)

            # Get CLIP's hidden size
            clip_hidden_size = self.model.config.vision_config.hidden_size

            # Project to output_dim
            self.projection = nn.Sequential(
                nn.Linear(clip_hidden_size, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            ).to(device)

            self.use_clip = True
            logger.info(
                f"CLIP feature extractor initialized "
                f"(num_regions={num_regions}, output_dim={output_dim})"
            )

        except ImportError:
            logger.warning(
                "transformers not available. Install with: pip install transformers"
            )
            self._init_resnet_fallback()

    def _init_resnet_fallback(self):
        """Initialize ResNet as fallback if CLIP not available."""
        from torchvision.models import ResNet152_Weights, resnet152

        logger.info("Falling back to ResNet-152 features")

        resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.backbone.eval()
        self.backbone.to(self.device)

        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(start_dim=2),
        ).to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract visual features from an image.

        Args:
            image: PIL Image

        Returns:
            Tuple of:
                - visual_features: [num_regions, output_dim]
                - spatial_locations: [num_regions, 5]
        """
        if self.use_clip:
            return self._extract_clip_features(image)
        else:
            return self._extract_resnet_features(image)

    def _extract_clip_features(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features using CLIP."""
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get CLIP features
        outputs = self.model.vision_model(**inputs, output_hidden_states=True)

        # Use patch embeddings as regions (excluding CLS token)
        hidden_states = outputs.last_hidden_state
        patch_features = hidden_states[:, 1:, :]  # Remove CLS token

        # Project to output dimension
        features = self.projection(patch_features)

        # Interpolate to num_regions if needed
        num_patches = features.shape[1]
        if num_patches != self.num_regions:
            grid_size = int(num_patches**0.5)
            features = features.view(1, grid_size, grid_size, self.output_dim)
            features = features.permute(0, 3, 1, 2)

            new_grid = int(self.num_regions**0.5)
            features = torch.nn.functional.interpolate(
                features,
                size=(new_grid, new_grid),
                mode="bilinear",
                align_corners=False,
            )
            features = features.permute(0, 2, 3, 1)
            features = features.view(1, -1, self.output_dim)

        features = features.squeeze(0)
        spatial = self._generate_grid_spatial()

        return features, spatial

    def _extract_resnet_features(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features using ResNet fallback."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        features = self.backbone(img_tensor)
        features = features.view(1, 2048, -1).permute(0, 2, 1)

        if features.shape[1] != self.num_regions:
            features = features.permute(0, 2, 1)
            features = torch.nn.functional.interpolate(
                features.unsqueeze(-1),
                size=(self.num_regions, 1),
                mode="bilinear",
                align_corners=False,
            ).squeeze(-1)
            features = features.permute(0, 2, 1)

        features = features.squeeze(0)

        if features.shape[-1] != self.output_dim:
            if features.shape[-1] < self.output_dim:
                padding = torch.zeros(
                    features.shape[0],
                    self.output_dim - features.shape[-1],
                    device=self.device,
                )
                features = torch.cat([features, padding], dim=-1)
            else:
                features = features[:, : self.output_dim]

        spatial = self._generate_grid_spatial()
        return features, spatial

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch feature extraction."""
        batch_features = []
        batch_spatial = []

        for img in images:
            img_pil = transforms.ToPILImage()(img.cpu())
            features, spatial = self.extract_features(img_pil)
            batch_features.append(features)
            batch_spatial.append(spatial)

        return torch.stack(batch_features), torch.stack(batch_spatial)
