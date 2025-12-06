"""
Visual Feature Extraction for ViLBERT

The original Facebook ViLBERT uses Faster R-CNN features extracted from a
ResNeXT-152 model trained on Visual Genome. This module provides options:

1. CLIP features (recommended) - Good performance, easy to use
2. ResNet features - Simpler but less effective
3. Pre-extracted features - If you have .lmdb files from MMF

For best results matching the Facebook baseline (~0.70 AUROC), you should use
pre-extracted Faster R-CNN features or fine-tune with CLIP features.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class CLIPVisualFeatureExtractor(nn.Module):
    """
    Extract visual features using CLIP's vision encoder.

    CLIP features are a good alternative to Faster R-CNN features
    and often provide competitive or better results.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_dim: int = 2048,
        num_regions: int = 36,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_regions = num_regions
        self.device = device

        try:
            from transformers import CLIPModel, CLIPProcessor

            logger.info(f"Loading CLIP model: {model_name}")
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            self.model.to(device)

            # Get CLIP's hidden size
            clip_hidden_size = self.model.config.vision_config.hidden_size

            # Project to output_dim and create region features
            self.projection = nn.Sequential(
                nn.Linear(clip_hidden_size, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            ).to(device)

            self.use_clip = True
            logger.info("CLIP feature extractor initialized")

        except ImportError:
            logger.warning("transformers not available, falling back to ResNet")
            self.use_clip = False
            self._init_resnet_fallback()

    def _init_resnet_fallback(self):
        """Initialize ResNet as fallback feature extractor."""
        from torchvision.models import ResNet152_Weights, resnet152

        resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.backbone.eval()
        self.backbone.to(self.device)

        # Project ResNet features (2048) to output_dim
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),  # Create 36 regions
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
            visual_features: [num_regions, output_dim]
            spatial_locations: [num_regions, 5] normalized bbox coords
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
        # Shape: [1, num_patches + 1, hidden_size]
        hidden_states = outputs.last_hidden_state
        patch_features = hidden_states[:, 1:, :]  # Remove CLS token

        # Project to output dimension
        features = self.projection(patch_features)  # [1, num_patches, output_dim]

        # Interpolate to num_regions if needed
        num_patches = features.shape[1]
        if num_patches != self.num_regions:
            # Reshape to 2D grid, interpolate, reshape back
            grid_size = int(num_patches**0.5)
            features = features.view(1, grid_size, grid_size, self.output_dim)
            features = features.permute(0, 3, 1, 2)  # [1, dim, h, w]

            new_grid = int(self.num_regions**0.5)
            features = torch.nn.functional.interpolate(
                features,
                size=(new_grid, new_grid),
                mode="bilinear",
                align_corners=False,
            )
            features = features.permute(0, 2, 3, 1)  # [1, h, w, dim]
            features = features.view(1, -1, self.output_dim)  # [1, num_regions, dim]

        features = features.squeeze(0)  # [num_regions, output_dim]

        # Generate spatial locations (grid-based)
        spatial = self._generate_grid_spatial(self.num_regions)

        return features, spatial

    def _extract_resnet_features(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features using ResNet."""
        # Transform image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract features
        features = self.backbone(img_tensor)  # [1, 2048, 7, 7]

        # Reshape to regions
        features = features.view(1, 2048, -1).permute(0, 2, 1)  # [1, 49, 2048]

        # Interpolate to num_regions
        if features.shape[1] != self.num_regions:
            features = features.permute(0, 2, 1)  # [1, 2048, 49]
            features = torch.nn.functional.interpolate(
                features.unsqueeze(-1),
                size=(self.num_regions, 1),
                mode="bilinear",
                align_corners=False,
            ).squeeze(-1)
            features = features.permute(0, 2, 1)  # [1, num_regions, 2048]

        features = features.squeeze(0)  # [num_regions, 2048]

        # Pad/truncate to output_dim if needed
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

        spatial = self._generate_grid_spatial(self.num_regions)

        return features, spatial

    def _generate_grid_spatial(self, num_regions: int) -> torch.Tensor:
        """Generate spatial locations for grid-based regions."""
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

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch feature extraction.

        Args:
            images: Batch of images [batch, 3, H, W]

        Returns:
            visual_features: [batch, num_regions, output_dim]
            spatial_locations: [batch, num_regions, 5]
        """
        batch_features = []
        batch_spatial = []

        for img in images:
            # Convert tensor to PIL
            img_pil = transforms.ToPILImage()(img.cpu())
            features, spatial = self.extract_features(img_pil)
            batch_features.append(features)
            batch_spatial.append(spatial)

        return torch.stack(batch_features), torch.stack(batch_spatial)


class ResNetFeatureExtractor(nn.Module):
    """
    Simple ResNet-based feature extractor.

    This is a fallback when CLIP is not available. It won't achieve
    the same performance as Faster R-CNN features but is easy to use.
    """

    def __init__(
        self,
        output_dim: int = 2048,
        num_regions: int = 36,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        from torchvision.models import ResNet152_Weights, resnet152

        self.output_dim = output_dim
        self.num_regions = num_regions
        self.device = device

        # Load pretrained ResNet-152
        resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)

        # Remove final pooling and FC layers to get feature maps
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.backbone.eval()
        self.backbone.to(device)

        # Freeze backbone
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

        logger.info("ResNet-152 feature extractor initialized")

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from a single image."""
        # Transform
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract features [1, 2048, 7, 7]
        features = self.backbone(img_tensor)

        # Reshape to [1, 49, 2048]
        batch, channels, h, w = features.shape
        features = features.view(batch, channels, -1).permute(0, 2, 1)

        # Adaptive pooling to num_regions
        grid_size = int(self.num_regions**0.5)
        features = features.view(batch, h, w, channels)
        features = features.permute(0, 3, 1, 2)  # [1, 2048, 7, 7]
        features = torch.nn.functional.adaptive_avg_pool2d(
            features, (grid_size, grid_size)
        )
        features = features.view(batch, channels, -1).permute(0, 2, 1)  # [1, 36, 2048]

        features = features.squeeze(0)  # [36, 2048]

        # Generate spatial locations
        spatial = self._generate_grid_spatial()

        return features, spatial

    def _generate_grid_spatial(self) -> torch.Tensor:
        """Generate spatial locations for grid regions."""
        grid_size = int(self.num_regions**0.5)
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


class FasterRCNNFeatureExtractor(nn.Module):
    """
    Extract visual features using Faster R-CNN object detection.

    This extractor uses torchvision's Faster R-CNN to detect objects and
    extract region-based features, similar to Facebook's original ViLBERT setup.

    Key differences from Facebook's setup:
    - Facebook used ResNeXt-152 backbone trained on Visual Genome (1600 classes)
    - This uses ResNet-50-FPN trained on COCO (91 classes)

    The advantage is that features are object-centric (actual detected objects)
    rather than uniform grid cells, which better aligns with ViLBERT's pretraining.
    """

    def __init__(
        self,
        output_dim: int = 2048,
        num_regions: int = 36,
        confidence_threshold: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

        self.output_dim = output_dim
        self.num_regions = num_regions
        self.confidence_threshold = confidence_threshold
        self.device = device

        logger.info("Loading Faster R-CNN ResNet50-FPN-v2 (COCO pretrained)...")

        # Load pretrained Faster R-CNN
        self.detector = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        )
        self.detector.eval()
        self.detector.to(device)

        # Freeze detector
        for param in self.detector.parameters():
            param.requires_grad = False

        # The backbone outputs 256-dim features from FPN
        # We need to project to output_dim (2048)
        self.feature_projection = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),  # ROI pooled features are 256x7x7
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        ).to(device)

        # For extracting features from backbone
        self.backbone = self.detector.backbone

        # ROI pooler from the detector
        self.roi_pooler = self.detector.roi_heads.box_roi_pool

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # Image normalization (Faster R-CNN does its own normalization)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        logger.info(
            f"Faster R-CNN feature extractor initialized "
            f"(num_regions={num_regions}, output_dim={output_dim})"
        )

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract visual features from detected object regions.

        Args:
            image: PIL Image

        Returns:
            visual_features: [num_regions, output_dim]
            spatial_locations: [num_regions, 5] normalized bbox coords (x1, y1, x2, y2, area)
        """
        # Convert to tensor
        img_tensor = self.transform(image).to(self.device)
        img_width, img_height = image.size

        # Get detections
        detections = self.detector([img_tensor])[0]

        boxes = detections["boxes"]  # [N, 4]
        scores = detections["scores"]  # [N]

        # Filter by confidence
        keep = scores >= self.confidence_threshold
        boxes = boxes[keep]
        scores = scores[keep]

        # Sort by score and take top num_regions
        if len(boxes) > self.num_regions:
            _, indices = scores.topk(self.num_regions)
            boxes = boxes[indices]
        elif len(boxes) < self.num_regions:
            # Pad with grid-based regions if not enough detections
            boxes = self._pad_boxes_with_grid(boxes, img_width, img_height)

        # Extract features using backbone and ROI pooling
        features = self._extract_roi_features(img_tensor, boxes)

        # Normalize spatial locations
        spatial = self._normalize_boxes(boxes, img_width, img_height)

        return features, spatial

    def _extract_roi_features(
        self, img_tensor: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        """Extract features for each ROI using backbone + ROI pooling."""
        # Get backbone features
        img_list = [img_tensor]
        features = self.backbone(img_tensor.unsqueeze(0))

        # ROI pooling expects boxes as list of tensors
        # Each tensor is [N, 4] for N boxes in that image
        pooled_features = self.roi_pooler(
            features, [boxes], [(img_tensor.shape[1], img_tensor.shape[2])]
        )

        # pooled_features: [num_regions, 256, 7, 7]
        pooled_flat = pooled_features.view(pooled_features.size(0), -1)

        # Project to output_dim
        projected = self.feature_projection(pooled_flat)

        return projected

    def _pad_boxes_with_grid(
        self, boxes: torch.Tensor, img_width: int, img_height: int
    ) -> torch.Tensor:
        """Pad boxes with grid-based regions if not enough detections."""
        num_existing = len(boxes)
        num_needed = self.num_regions - num_existing

        if num_needed <= 0:
            return boxes[: self.num_regions]

        # Generate grid boxes
        grid_size = int(num_needed**0.5) + 1
        cell_w = img_width / grid_size
        cell_h = img_height / grid_size

        grid_boxes = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(grid_boxes) >= num_needed:
                    break
                x1 = j * cell_w
                y1 = i * cell_h
                x2 = (j + 1) * cell_w
                y2 = (i + 1) * cell_h
                grid_boxes.append([x1, y1, x2, y2])
            if len(grid_boxes) >= num_needed:
                break

        grid_boxes = torch.tensor(grid_boxes, device=self.device, dtype=boxes.dtype)

        if num_existing > 0:
            return torch.cat([boxes, grid_boxes], dim=0)
        else:
            return grid_boxes

    def _normalize_boxes(
        self, boxes: torch.Tensor, img_width: int, img_height: int
    ) -> torch.Tensor:
        """Normalize boxes to [0, 1] and compute area."""
        # boxes: [N, 4] in format [x1, y1, x2, y2]
        normalized = boxes.clone()
        normalized[:, [0, 2]] /= img_width
        normalized[:, [1, 3]] /= img_height

        # Compute area
        widths = normalized[:, 2] - normalized[:, 0]
        heights = normalized[:, 3] - normalized[:, 1]
        areas = widths * heights

        # Concatenate: [x1, y1, x2, y2, area]
        spatial = torch.cat([normalized, areas.unsqueeze(1)], dim=1)

        return spatial

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch feature extraction.

        Args:
            images: Batch of images [batch, 3, H, W]

        Returns:
            visual_features: [batch, num_regions, output_dim]
            spatial_locations: [batch, num_regions, 5]
        """
        batch_features = []
        batch_spatial = []

        for img in images:
            img_pil = transforms.ToPILImage()(img.cpu())
            features, spatial = self.extract_features(img_pil)
            batch_features.append(features)
            batch_spatial.append(spatial)

        return torch.stack(batch_features), torch.stack(batch_spatial)


def get_feature_extractor(
    extractor_type: str = "clip",
    output_dim: int = 2048,
    num_regions: int = 36,
    device: str = None,
    **kwargs,
) -> nn.Module:
    """
    Get a visual feature extractor.

    Args:
        extractor_type: "clip", "resnet", or "fasterrcnn"
        output_dim: Output feature dimension
        num_regions: Number of visual regions
        device: Device to use
        **kwargs: Additional arguments for specific extractors

    Returns:
        Feature extractor module
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if extractor_type == "clip":
        return CLIPVisualFeatureExtractor(
            output_dim=output_dim,
            num_regions=num_regions,
            device=device,
        )
    elif extractor_type == "resnet":
        return ResNetFeatureExtractor(
            output_dim=output_dim,
            num_regions=num_regions,
            device=device,
        )
    elif extractor_type == "fasterrcnn":
        return FasterRCNNFeatureExtractor(
            output_dim=output_dim,
            num_regions=num_regions,
            confidence_threshold=kwargs.get("confidence_threshold", 0.2),
            device=device,
        )
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
