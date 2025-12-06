"""Faster R-CNN (COCO) feature extractor."""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)

from ..base import BaseFeatureExtractor, register_feature_extractor

logger = logging.getLogger(__name__)


@register_feature_extractor("fasterrcnn")
class FasterRCNNFeatureExtractor(BaseFeatureExtractor):
    """Extracts object-based features using Faster R-CNN pretrained on COCO."""

    def __init__(
        self,
        output_dim: int = 2048,
        num_regions: int = 36,
        confidence_threshold: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(output_dim, num_regions, device)
        self.confidence_threshold = confidence_threshold

        logger.info("Loading Faster R-CNN ResNet50-FPN-v2...")

        self.detector = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        )
        self.detector.eval()
        self.detector.to(device)

        for param in self.detector.parameters():
            param.requires_grad = False

        self.feature_projection = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        ).to(device)

        self.backbone = self.detector.backbone
        self.roi_pooler = self.detector.roi_heads.box_roi_pool
        self.transform = transforms.ToTensor()

        logger.info(
            f"Faster R-CNN initialized (num_regions={num_regions}, threshold={confidence_threshold})"
        )

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        img_tensor = self.transform(image).to(self.device)
        img_width, img_height = image.size

        detections = self.detector([img_tensor])[0]
        boxes, scores = detections["boxes"], detections["scores"]

        keep = scores >= self.confidence_threshold
        boxes, scores = boxes[keep], scores[keep]

        if len(boxes) > self.num_regions:
            _, indices = scores.topk(self.num_regions)
            boxes = boxes[indices]
        elif len(boxes) < self.num_regions:
            boxes = self._pad_boxes_with_grid(boxes, img_width, img_height)

        features = self._extract_roi_features(img_tensor, boxes)
        spatial = self._normalize_boxes(boxes, img_width, img_height)

        return features, spatial

    def _extract_roi_features(
        self, img_tensor: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        features = self.backbone(img_tensor.unsqueeze(0))
        pooled = self.roi_pooler(
            features, [boxes], [(img_tensor.shape[1], img_tensor.shape[2])]
        )
        pooled_flat = pooled.view(pooled.size(0), -1)
        return self.feature_projection(pooled_flat)

    def _pad_boxes_with_grid(
        self, boxes: torch.Tensor, img_width: int, img_height: int
    ) -> torch.Tensor:
        num_needed = self.num_regions - len(boxes)
        if num_needed <= 0:
            return boxes[: self.num_regions]

        grid_size = int(num_needed**0.5) + 1
        cell_w, cell_h = img_width / grid_size, img_height / grid_size

        grid_boxes = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(grid_boxes) >= num_needed:
                    break
                grid_boxes.append(
                    [j * cell_w, i * cell_h, (j + 1) * cell_w, (i + 1) * cell_h]
                )
            if len(grid_boxes) >= num_needed:
                break

        grid_boxes = torch.tensor(
            grid_boxes,
            device=self.device,
            dtype=boxes.dtype if len(boxes) > 0 else torch.float32,
        )

        return torch.cat([boxes, grid_boxes], dim=0) if len(boxes) > 0 else grid_boxes

    def _normalize_boxes(
        self, boxes: torch.Tensor, img_width: int, img_height: int
    ) -> torch.Tensor:
        normalized = boxes.clone()
        normalized[:, [0, 2]] /= img_width
        normalized[:, [1, 3]] /= img_height

        widths = normalized[:, 2] - normalized[:, 0]
        heights = normalized[:, 3] - normalized[:, 1]
        areas = widths * heights

        return torch.cat([normalized, areas.unsqueeze(1)], dim=1)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_features, batch_spatial = [], []
        for img in images:
            img_pil = transforms.ToPILImage()(img.cpu())
            features, spatial = self.extract_features(img_pil)
            batch_features.append(features)
            batch_spatial.append(spatial)
        return torch.stack(batch_features), torch.stack(batch_spatial)
