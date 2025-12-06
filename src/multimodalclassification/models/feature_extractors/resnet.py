"""ResNet-152 grid-based feature extractor."""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet152_Weights, resnet152

from ..base import BaseFeatureExtractor, register_feature_extractor

logger = logging.getLogger(__name__)


@register_feature_extractor("resnet")
class ResNetFeatureExtractor(BaseFeatureExtractor):
    """Extracts features from a grid of image regions using ResNet-152."""

    def __init__(
        self,
        output_dim: int = 2048,
        num_regions: int = 36,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(output_dim, num_regions, device)

        logger.info("Loading ResNet-152 feature extractor...")

        resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.backbone.eval()
        self.backbone.to(device)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        logger.info(f"ResNet-152 initialized (num_regions={num_regions})")

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        features = self.backbone(img_tensor)

        batch, channels, h, w = features.shape
        grid_size = int(self.num_regions**0.5)

        features = features.permute(0, 2, 3, 1).reshape(batch, h * w, channels)
        features = features.view(batch, h, w, channels).permute(0, 3, 1, 2)
        features = torch.nn.functional.adaptive_avg_pool2d(
            features, (grid_size, grid_size)
        )
        features = features.view(batch, channels, -1).permute(0, 2, 1).squeeze(0)

        if features.shape[-1] < self.output_dim:
            padding = torch.zeros(
                features.shape[0],
                self.output_dim - features.shape[-1],
                device=self.device,
            )
            features = torch.cat([features, padding], dim=-1)
        elif features.shape[-1] > self.output_dim:
            features = features[:, : self.output_dim]

        return features, self._generate_grid_spatial()

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_features, batch_spatial = [], []
        for img in images:
            img_pil = transforms.ToPILImage()(img.cpu())
            features, spatial = self.extract_features(img_pil)
            batch_features.append(features)
            batch_spatial.append(spatial)
        return torch.stack(batch_features), torch.stack(batch_spatial)
