"""
ResNet-152 Feature Extractor with ROI Pooling

This combines:
- ResNet-152 backbone (ImageNet pretrained) - same as vilbert_train
- ROI pooling pipeline (like the VG/COCO Faster R-CNN models)

This tests whether the detection-style ROI pooling helps or hurts
when using the same strong ImageNet backbone.

Comparison:
- vilbert_train (ResNet-152 + grid pooling): 0.6645 AUROC
- vilbert_frcnn_resnet152_train (ResNet-152 + COCO detection): 0.6334 AUROC
- This model (ResNet-152 + ROI pooling, no detection): ???

The hypothesis is that ROI pooling on proposed regions might capture
better object-centric features than simple grid pooling.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet152_Weights, resnet152
from torchvision.ops import RoIPool, nms

from ..base import BaseFeatureExtractor, register_feature_extractor

logger = logging.getLogger(__name__)


class ResNet152Backbone(nn.Module):
    """
    ResNet-152 split into base (conv1-layer3) and top (layer4).

    This structure allows ROI pooling between base and top,
    matching the Faster R-CNN architecture.
    """

    def __init__(self):
        super().__init__()

        resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)

        # Base: conv1 through layer3 (outputs 1024-dim, stride 16)
        self.base = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        # Top: layer4 (outputs 2048-dim)
        self.top = resnet.layer4

        # Average pooling for final features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_base(self, x: torch.Tensor) -> torch.Tensor:
        """Extract base features (before ROI pooling)."""
        return self.base(x)

    def forward_top(self, x: torch.Tensor) -> torch.Tensor:
        """Extract top features from ROI-pooled regions."""
        x = self.top(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


@register_feature_extractor("resnet152_roi")
class ResNet152ROIExtractor(BaseFeatureExtractor):
    """
    ResNet-152 with ROI pooling for region-based feature extraction.

    Architecture:
        Image → ResNet base (conv1-layer3) → ROI Pool → ResNet top (layer4) → Features

    This is similar to Faster R-CNN feature extraction but:
    - Uses ImageNet-pretrained ResNet-152 (not COCO or VG)
    - Uses grid-based proposals (not learned RPN)
    - No detection head, just feature extraction

    Args:
        output_dim: Output feature dimension (default: 2048)
        num_regions: Number of regions to extract (default: 36)
        roi_size: ROI pooling output size (default: 14)
        use_multi_scale: Use multi-scale region proposals (default: True)
        device: Device to run on
    """

    def __init__(
        self,
        output_dim: int = 2048,
        num_regions: int = 36,
        roi_size: int = 14,
        use_multi_scale: bool = True,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(output_dim, num_regions, device)

        self.roi_size = roi_size
        self.use_multi_scale = use_multi_scale

        logger.info("Initializing ResNet-152 with ROI pooling...")

        # Initialize backbone
        self.backbone = ResNet152Backbone()
        self.backbone.to(self.device).eval()

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ROI pooling layer
        # After base (layer3), spatial stride is 16, so spatial_scale = 1/16
        self.roi_pool = RoIPool(output_size=(roi_size, roi_size), spatial_scale=1 / 16)

        # Image preprocessing - larger size for better region proposals
        self.transform = transforms.Compose(
            [
                transforms.Resize((600, 600)),  # Larger than grid-based (224)
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        logger.info(
            f"ResNet-152 ROI initialized: num_regions={num_regions}, "
            f"roi_size={roi_size}, multi_scale={use_multi_scale}"
        )

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract ROI-pooled features from image regions.

        Args:
            image: PIL Image

        Returns:
            features: [num_regions, output_dim] region features
            spatial: [num_regions, 5] normalized spatial locations
        """
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]

        # Extract base features
        base_features = self.backbone.forward_base(img_tensor)  # [1, 1024, H/16, W/16]

        # Generate region proposals
        boxes = self._generate_proposals(img_h, img_w)

        # ROI pooling
        batch_idx = torch.zeros(len(boxes), 1, device=self.device)
        rois = torch.cat([batch_idx, boxes], dim=1)  # [N, 5]

        pooled = self.roi_pool(base_features, rois)  # [N, 1024, 14, 14]

        # Extract top features for each region
        features = self.backbone.forward_top(pooled)  # [N, 2048]

        # Normalize spatial locations
        spatial = self._normalize_boxes(boxes, img_w, img_h)

        return features, spatial

    def _generate_proposals(self, img_h: int, img_w: int) -> torch.Tensor:
        """
        Generate region proposals.

        Uses multi-scale grid if enabled, otherwise simple grid.
        """
        if self.use_multi_scale:
            return self._generate_multi_scale_proposals(img_h, img_w)
        else:
            return self._generate_grid_proposals(img_h, img_w)

    def _generate_grid_proposals(self, img_h: int, img_w: int) -> torch.Tensor:
        """Generate simple grid-based proposals."""
        grid_size = int(self.num_regions**0.5)  # 6 for 36 regions
        cell_h = img_h / grid_size
        cell_w = img_w / grid_size

        boxes = []
        for i in range(grid_size):
            for j in range(grid_size):
                x1 = j * cell_w
                y1 = i * cell_h
                x2 = (j + 1) * cell_w
                y2 = (i + 1) * cell_h
                boxes.append([x1, y1, x2, y2])

        return torch.tensor(boxes, device=self.device, dtype=torch.float32)

    def _generate_multi_scale_proposals(self, img_h: int, img_w: int) -> torch.Tensor:
        """
        Generate multi-scale region proposals.

        Creates proposals at multiple scales and aspect ratios,
        then selects the top num_regions based on coverage/diversity.
        """
        boxes = []

        # Multiple scales (fraction of image size)
        scales = [0.15, 0.25, 0.35, 0.5, 0.7]
        aspect_ratios = [0.5, 0.75, 1.0, 1.33, 2.0]

        for scale in scales:
            for ar in aspect_ratios:
                # Calculate box dimensions
                box_w = img_w * scale
                box_h = box_w / ar

                # Clamp to image bounds
                box_h = min(box_h, img_h * 0.95)
                box_w = min(box_w, img_w * 0.95)

                # Stride for this scale
                stride_x = max(box_w * 0.4, 20)
                stride_y = max(box_h * 0.4, 20)

                x = 0
                while x + box_w <= img_w:
                    y = 0
                    while y + box_h <= img_h:
                        boxes.append([x, y, x + box_w, y + box_h])
                        y += stride_y
                    x += stride_x

        boxes = torch.tensor(boxes, device=self.device, dtype=torch.float32)

        # If we have more than needed, select diverse subset
        if len(boxes) > self.num_regions:
            boxes = self._select_diverse_boxes(boxes, img_h, img_w)
        elif len(boxes) < self.num_regions:
            # Pad with grid boxes
            grid_boxes = self._generate_grid_proposals(img_h, img_w)
            boxes = torch.cat([boxes, grid_boxes], dim=0)[: self.num_regions]

        return boxes[: self.num_regions]

    def _select_diverse_boxes(
        self, boxes: torch.Tensor, img_h: int, img_w: int
    ) -> torch.Tensor:
        """
        Select a diverse subset of boxes using NMS-like approach.

        Prioritizes boxes that cover different parts of the image.
        """
        # Score boxes by their "centrality" - prefer boxes covering different areas
        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2 / img_w
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2 / img_h

        # Compute area (prefer medium-sized boxes)
        widths = (boxes[:, 2] - boxes[:, 0]) / img_w
        heights = (boxes[:, 3] - boxes[:, 1]) / img_h
        areas = widths * heights

        # Score: prefer medium areas and diverse positions
        area_score = 1.0 - torch.abs(areas - 0.15)  # Prefer ~15% of image

        # Use NMS to get diverse boxes
        scores = area_score
        keep = nms(boxes, scores, iou_threshold=0.5)

        # Take top num_regions
        if len(keep) >= self.num_regions:
            return boxes[keep[: self.num_regions]]
        else:
            # If not enough after NMS, add remaining boxes
            remaining = [i for i in range(len(boxes)) if i not in keep]
            keep = torch.cat(
                [
                    keep,
                    torch.tensor(
                        remaining[: self.num_regions - len(keep)], device=self.device
                    ),
                ]
            )
            return boxes[keep[: self.num_regions]]

    def _normalize_boxes(
        self, boxes: torch.Tensor, img_w: int, img_h: int
    ) -> torch.Tensor:
        """Normalize boxes to [0, 1] and add area."""
        normalized = boxes.clone()
        normalized[:, [0, 2]] /= img_w
        normalized[:, [1, 3]] /= img_h
        normalized = normalized.clamp(0, 1)

        # Compute area
        widths = normalized[:, 2] - normalized[:, 0]
        heights = normalized[:, 3] - normalized[:, 1]
        areas = widths * heights

        # [x1, y1, x2, y2, area]
        spatial = torch.cat([normalized, areas.unsqueeze(1)], dim=1)
        return spatial

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
