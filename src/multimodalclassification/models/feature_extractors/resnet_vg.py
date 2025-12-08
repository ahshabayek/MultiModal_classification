"""
Simple ResNet-101 Feature Extractor with Visual Genome Pretrained Weights

This extracts grid-based features using ONLY the ResNet-101 backbone from the
Visual Genome checkpoint - NO detection head, NO RPN, NO ROI pooling.

This is analogous to the simple ResNet-152 extractor (which achieved 0.6645 AUROC)
but uses VG-pretrained weights instead of ImageNet weights.

The hypothesis is that VG pretraining should provide better visual features
for meme understanding than ImageNet classification pretraining.
"""

import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet101_Weights, resnet101

from ..base import BaseFeatureExtractor, register_feature_extractor

logger = logging.getLogger(__name__)


class VGResNet101Backbone(nn.Module):
    """
    ResNet-101 backbone matching the Visual Genome checkpoint structure.

    Only loads RCNN_base (conv1 through layer3) and RCNN_top (layer4).
    No detection heads, no RPN - pure feature extraction.
    """

    def __init__(self):
        super().__init__()

        # Load ImageNet pretrained ResNet-101 as initialization
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)

        # RCNN_base: conv1 through layer3 (matches VG checkpoint structure)
        self.RCNN_base = nn.Sequential(
            resnet.conv1,  # 0
            resnet.bn1,  # 1
            resnet.relu,  # 2
            resnet.maxpool,  # 3
            resnet.layer1,  # 4
            resnet.layer2,  # 5
            resnet.layer3,  # 6
        )

        # RCNN_top: layer4 (outputs 2048-dim features)
        self.RCNN_top = resnet.layer4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 2048-dim features.

        Args:
            x: [B, 3, H, W] input image tensor

        Returns:
            features: [B, 2048, H/32, W/32] feature map
        """
        x = self.RCNN_base(x)  # [B, 1024, H/16, W/16]
        x = self.RCNN_top(x)  # [B, 2048, H/32, W/32]
        return x


def load_vg_backbone_weights(model: VGResNet101Backbone, checkpoint_path: str) -> dict:
    """
    Load only the backbone weights from VG checkpoint.

    Ignores RPN, classifier, and bbox predictor weights.

    Returns:
        dict with loading statistics
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    model_state = model.state_dict()
    loaded = {}
    skipped = {}

    for ckpt_key, ckpt_value in state_dict.items():
        # Only load backbone weights (RCNN_base and RCNN_top)
        if not (ckpt_key.startswith("RCNN_base") or ckpt_key.startswith("RCNN_top")):
            skipped[ckpt_key] = "not backbone"
            continue

        model_key = ckpt_key

        # Handle RCNN_top key mapping: checkpoint has RCNN_top.0.X, model has RCNN_top.X
        if ckpt_key.startswith("RCNN_top.0."):
            model_key = "RCNN_top." + ckpt_key[len("RCNN_top.0.") :]

        if model_key in model_state:
            if model_state[model_key].shape == ckpt_value.shape:
                loaded[model_key] = ckpt_value
            else:
                skipped[ckpt_key] = (
                    f"shape mismatch: {model_state[model_key].shape} vs {ckpt_value.shape}"
                )
        else:
            skipped[ckpt_key] = "key not in model"

    model.load_state_dict(loaded, strict=False)

    stats = {
        "loaded": len(loaded),
        "total_model": len(model_state),
        "skipped": len(skipped),
        "skipped_keys": list(skipped.keys())[:10],  # First 10 skipped
    }

    logger.info(
        f"Loaded {stats['loaded']}/{stats['total_model']} backbone weights from VG checkpoint"
    )

    return stats


@register_feature_extractor("resnet_vg")
class ResNetVGExtractor(BaseFeatureExtractor):
    """
    Simple grid-based feature extractor using VG-pretrained ResNet-101.

    This is the VG equivalent of the simple ResNet-152 extractor:
    - Uses ResNet-101 backbone pretrained on Visual Genome
    - Extracts grid-based features (no object detection)
    - No RPN, no ROI pooling - just spatial grid pooling

    Expected to outperform ImageNet ResNet-152 (0.6645) due to VG pretraining.

    Args:
        output_dim: Output feature dimension (default: 2048)
        num_regions: Number of visual regions to extract (default: 36, i.e., 6x6 grid)
        weights_path: Path to VG checkpoint (default: weights/faster_rcnn_res101_vg.pth)
        device: Device to run on
    """

    def __init__(
        self,
        output_dim: int = 2048,
        num_regions: int = 36,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(output_dim, num_regions, device)

        # Default weights path
        if weights_path is None:
            weights_path = "weights/faster_rcnn_res101_vg.pth"

        logger.info("Initializing VG ResNet-101 backbone...")

        # Initialize backbone
        self.backbone = VGResNet101Backbone()

        # Load VG weights
        if os.path.exists(weights_path):
            logger.info(f"Loading VG backbone weights from {weights_path}")
            stats = load_vg_backbone_weights(self.backbone, weights_path)
            self.has_vg_weights = stats["loaded"] > 0
            logger.info(f"VG backbone weights loaded: {stats['loaded']} tensors")
        else:
            logger.warning(
                f"VG weights not found at {weights_path}, using ImageNet weights"
            )
            self.has_vg_weights = False

        # Move to device and freeze
        self.backbone.to(self.device).eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Image preprocessing (same as ResNet-152 extractor)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Grid size for spatial pooling
        self.grid_size = int(num_regions**0.5)  # 6 for 36 regions

        logger.info(
            f"VG ResNet-101 initialized: num_regions={num_regions} ({self.grid_size}x{self.grid_size} grid), "
            f"output_dim={output_dim}, vg_weights={'loaded' if self.has_vg_weights else 'ImageNet fallback'}"
        )

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract grid-based visual features.

        Args:
            image: PIL Image

        Returns:
            features: [num_regions, output_dim] region features
            spatial: [num_regions, 5] normalized spatial info
        """
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract features through full backbone
        features = self.backbone(img_tensor)  # [1, 2048, H/32, W/32]

        batch, channels, h, w = features.shape

        # Adaptive pool to grid_size x grid_size
        features = nn.functional.adaptive_avg_pool2d(
            features, (self.grid_size, self.grid_size)
        )

        # Reshape to [num_regions, channels]
        features = (
            features.view(batch, channels, -1).permute(0, 2, 1).squeeze(0)
        )  # [36, 2048]

        # Handle dimension mismatch
        if features.shape[-1] < self.output_dim:
            padding = torch.zeros(
                features.shape[0],
                self.output_dim - features.shape[-1],
                device=self.device,
            )
            features = torch.cat([features, padding], dim=-1)
        elif features.shape[-1] > self.output_dim:
            features = features[:, : self.output_dim]

        # Generate grid spatial info
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
