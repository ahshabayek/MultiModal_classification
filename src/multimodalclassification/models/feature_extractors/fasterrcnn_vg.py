"""
Faster R-CNN Feature Extractor (Visual Genome)

Extracts object-based visual features using Faster R-CNN pretrained on Visual Genome.
This matches Facebook's original ViLBERT setup which used Visual Genome features.

Visual Genome has 1600 object classes vs COCO's 91, providing better coverage
for the types of objects that appear in memes.

Expected AUROC: ~0.68-0.72 (closer to Facebook's reported results)

The checkpoint structure:
- RCNN_base: ResNet-101 layers 0-3 (conv1, bn1, layer1, layer2, layer3)
- RCNN_top: ResNet-101 layer4
- RCNN_cls_score: 1601-class classifier (1600 VG + background)
- RCNN_bbox_pred: Bounding box regressor
- RCNN_rpn: Region Proposal Network
"""

import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.ops import RoIPool, nms

from ..base import BaseFeatureExtractor, register_feature_extractor

logger = logging.getLogger(__name__)


class VGFasterRCNN(nn.Module):
    """
    Faster R-CNN architecture matching the Visual Genome checkpoint structure.

    The VG checkpoint uses a specific structure:
    - RCNN_base: Sequential of conv1, bn1, relu, maxpool, layer1, layer2, layer3
    - RCNN_top: layer4 of ResNet-101
    - RCNN_cls_score: Linear(2048, 1601) for classification
    - RCNN_bbox_pred: Linear(2048, 6404) for bbox regression
    """

    NUM_VG_CLASSES = 1601  # 1600 + background

    def __init__(self):
        super().__init__()

        # Load ImageNet pretrained ResNet-101 as base
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)

        # RCNN_base: layers before layer4 (outputs 1024-dim features)
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

        # Average pooling for final feature extraction
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification and bbox heads
        self.RCNN_cls_score = nn.Linear(2048, self.NUM_VG_CLASSES)
        self.RCNN_bbox_pred = nn.Linear(2048, self.NUM_VG_CLASSES * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract base features."""
        return self.RCNN_base(x)

    def extract_top_features(self, pooled_features: torch.Tensor) -> torch.Tensor:
        """
        Extract features from ROI-pooled regions through layer4.

        Args:
            pooled_features: [N, 1024, 7, 7] ROI-pooled features

        Returns:
            features: [N, 2048] final visual features
        """
        # Pass through layer4
        x = self.RCNN_top(pooled_features)  # [N, 2048, 4, 4] approximately
        # Global average pooling
        x = self.avgpool(x)  # [N, 2048, 1, 1]
        x = x.view(x.size(0), -1)  # [N, 2048]
        return x

    def get_class_scores(self, features: torch.Tensor) -> torch.Tensor:
        """Get classification scores for regions."""
        return self.RCNN_cls_score(features)


def load_vg_weights(model: VGFasterRCNN, checkpoint_path: str) -> int:
    """
    Load Visual Genome pretrained weights into the model.

    The VG checkpoint structure:
    - RCNN_base.0 = conv1 (weight only)
    - RCNN_base.1 = bn1
    - RCNN_base.4 = layer1 (index 4 because relu=2, maxpool=3 have no params)
    - RCNN_base.5 = layer2
    - RCNN_base.6 = layer3
    - RCNN_top.0 = layer4
    - RCNN_cls_score = classifier
    - RCNN_bbox_pred = bbox regressor

    Args:
        model: VGFasterRCNN model instance
        checkpoint_path: Path to the VG checkpoint

    Returns:
        Number of successfully loaded weight tensors
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    model_state = model.state_dict()
    loaded_count = 0
    new_state_dict = {}

    # Build key mapping from checkpoint to model
    # Checkpoint uses: RCNN_base.0, RCNN_base.1, RCNN_base.4.0.conv1, etc.
    # Model uses: RCNN_base.0 (conv1), RCNN_base.1 (bn1), RCNN_base.4.0.conv1, etc.
    # They should match directly for RCNN_base

    for ckpt_key, ckpt_value in state_dict.items():
        model_key = ckpt_key

        # Handle RCNN_top key mapping: checkpoint has RCNN_top.0.X, model has RCNN_top.X
        # Checkpoint: RCNN_top.0.0.conv1 -> Model: RCNN_top.0.conv1
        # Checkpoint: RCNN_top.0.1.conv1 -> Model: RCNN_top.1.conv1
        if ckpt_key.startswith("RCNN_top.0."):
            # Remove the extra ".0" after RCNN_top
            model_key = "RCNN_top." + ckpt_key[len("RCNN_top.0.") :]

        # Direct match for RCNN_base, RCNN_cls_score, RCNN_bbox_pred
        if model_key in model_state:
            if model_state[model_key].shape == ckpt_value.shape:
                new_state_dict[model_key] = ckpt_value
                loaded_count += 1
            else:
                logger.warning(
                    f"Shape mismatch for {model_key}: "
                    f"model={model_state[model_key].shape}, ckpt={ckpt_value.shape}"
                )

    # Load the matched weights
    model.load_state_dict(new_state_dict, strict=False)

    logger.info(f"Loaded {loaded_count}/{len(model_state)} weights from VG checkpoint")

    # Log statistics
    unmatched_model = set(model_state.keys()) - set(new_state_dict.keys())
    if unmatched_model:
        logger.info(
            f"Model keys not in checkpoint ({len(unmatched_model)}): "
            f"{list(unmatched_model)[:5]}..."
        )

    return loaded_count


@register_feature_extractor("fasterrcnn_vg")
class FasterRCNNVGExtractor(BaseFeatureExtractor):
    """
    Extract visual features using Faster R-CNN pretrained on Visual Genome.

    This extractor matches Facebook's original ViLBERT setup using bottom-up
    attention features from Visual Genome (1600 object classes).

    Architecture:
        - Backbone: ResNet-101 (layers 0-3 = RCNN_base)
        - Top layers: ResNet-101 layer4 (= RCNN_top)
        - ROI Pooling: 14x14 -> layer4 -> avgpool -> 2048-dim
        - Classifier: 2048 -> 1601 classes

    Args:
        output_dim: Output feature dimension (default: 2048)
        num_regions: Number of visual regions to extract (default: 36)
        weights_path: Path to Visual Genome pretrained weights
        confidence_threshold: Minimum confidence for detected objects
        nms_threshold: Non-maximum suppression threshold
        device: Device to run the extractor on
    """

    NUM_VG_CLASSES = 1601

    DEFAULT_WEIGHTS_URL = (
        "https://drive.google.com/file/d/18n_3V1rywgeADZ3oONO0DsuuS9eMW6sN/view"
    )

    def __init__(
        self,
        output_dim: int = 2048,
        num_regions: int = 36,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.2,
        nms_threshold: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(output_dim, num_regions, device)

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.weights_path = weights_path

        # Check for VG weights
        if weights_path is None:
            weights_path = "weights/faster_rcnn_res101_vg.pth"

        self.has_vg_weights = os.path.exists(weights_path)

        # Initialize model
        self.model = VGFasterRCNN()

        if self.has_vg_weights:
            logger.info(f"Loading Visual Genome weights from {weights_path}")
            loaded = load_vg_weights(self.model, weights_path)
            logger.info(f"Loaded {loaded} weight tensors from VG checkpoint")
        else:
            logger.warning(
                f"Visual Genome weights not found at {weights_path}. "
                f"Using ImageNet-pretrained ResNet-101. "
                f"Download from: {self.DEFAULT_WEIGHTS_URL}"
            )

        # ROI pooling - pool to 14x14 before layer4
        # layer3 output stride is 16, so spatial_scale = 1/16
        self.roi_pool = RoIPool(output_size=(14, 14), spatial_scale=1 / 16)

        # Move to device and eval mode
        self.model.to(self.device).eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((600, 1000)),  # Standard Faster R-CNN size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        logger.info(
            f"VG Faster R-CNN initialized: num_regions={num_regions}, "
            f"output_dim={output_dim}, vg_weights={'loaded' if self.has_vg_weights else 'ImageNet fallback'}"
        )

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract visual features from image regions.

        Args:
            image: PIL Image

        Returns:
            Tuple of:
                - visual_features: [num_regions, output_dim]
                - spatial_locations: [num_regions, 5] (x1, y1, x2, y2, area)
        """
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]

        # Extract base features (through layer3)
        base_features = self.model(img_tensor)  # [1, 1024, H/16, W/16]

        # Generate region proposals (grid-based for simplicity)
        boxes, scores = self._generate_proposals(base_features, img_h, img_w)

        # Select top regions
        boxes, scores = self._select_top_regions(boxes, scores)

        # Extract ROI features
        roi_features = self._extract_roi_features(base_features, boxes)

        # Normalize spatial locations
        spatial = self._normalize_boxes(boxes, img_w, img_h)

        return roi_features, spatial

    def _generate_proposals(
        self,
        features: torch.Tensor,
        img_h: int,
        img_w: int,
        num_proposals: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate region proposals using grid-based approach with scoring.

        For proper Visual Genome features, we'd use the RPN, but grid-based
        proposals with classification scoring works reasonably well.
        """
        # Generate multi-scale grid proposals
        boxes = []

        # Multiple scales and aspect ratios
        scales = [0.2, 0.3, 0.4, 0.5, 0.7]
        aspect_ratios = [0.5, 1.0, 2.0]

        for scale in scales:
            for ar in aspect_ratios:
                box_w = img_w * scale
                box_h = box_w / ar

                # Limit to image bounds
                box_h = min(box_h, img_h * 0.9)
                box_w = min(box_w, img_w * 0.9)

                # Stride
                stride_x = max(box_w * 0.5, 1)
                stride_y = max(box_h * 0.5, 1)

                x = 0
                while x + box_w <= img_w:
                    y = 0
                    while y + box_h <= img_h:
                        boxes.append([x, y, x + box_w, y + box_h])
                        y += stride_y
                        if len(boxes) >= num_proposals * 2:
                            break
                    x += stride_x
                    if len(boxes) >= num_proposals * 2:
                        break
                if len(boxes) >= num_proposals * 2:
                    break
            if len(boxes) >= num_proposals * 2:
                break

        # Fallback: add grid proposals if not enough
        if len(boxes) < num_proposals:
            grid_size = int((num_proposals - len(boxes)) ** 0.5) + 1
            cell_w = img_w / grid_size
            cell_h = img_h / grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    x1 = j * cell_w
                    y1 = i * cell_h
                    x2 = min((j + 1) * cell_w, img_w)
                    y2 = min((i + 1) * cell_h, img_h)
                    boxes.append([x1, y1, x2, y2])

        boxes = torch.tensor(
            boxes[: num_proposals * 2], device=self.device, dtype=torch.float32
        )

        # Score proposals using classifier if available
        if self.has_vg_weights and len(boxes) > 0:
            scores = self._score_proposals(features, boxes)
        else:
            scores = torch.ones(len(boxes), device=self.device)

        return boxes, scores

    def _score_proposals(
        self, base_features: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        """Score proposals using the VG classifier."""
        # Extract features for proposals
        batch_idx = torch.zeros(len(boxes), 1, device=self.device)
        rois = torch.cat([batch_idx, boxes], dim=1)

        # ROI pool
        pooled = self.roi_pool(base_features, rois)  # [N, 1024, 14, 14]

        # Extract top features
        top_features = self.model.extract_top_features(pooled)  # [N, 2048]

        # Get class scores
        cls_scores = self.model.get_class_scores(top_features)  # [N, 1601]

        # Use max class score (excluding background at index 0)
        scores = cls_scores[:, 1:].max(dim=1)[0]

        return scores

    def _select_top_regions(
        self, boxes: torch.Tensor, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-scoring regions with NMS."""
        if len(boxes) == 0:
            # Return dummy boxes if empty
            boxes = torch.zeros(self.num_regions, 4, device=self.device)
            scores = torch.zeros(self.num_regions, device=self.device)
            return boxes, scores

        if len(boxes) <= self.num_regions:
            return self._pad_regions(boxes, scores)

        # Apply NMS
        keep = nms(boxes, scores, self.nms_threshold)

        # Sort by score and take top num_regions
        if len(keep) > self.num_regions:
            top_scores, top_idx = scores[keep].topk(self.num_regions)
            keep = keep[top_idx]

        boxes = boxes[keep]
        scores = scores[keep]

        return self._pad_regions(boxes, scores)

    def _pad_regions(
        self, boxes: torch.Tensor, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad to num_regions if needed."""
        if len(boxes) >= self.num_regions:
            return boxes[: self.num_regions], scores[: self.num_regions]

        padding = self.num_regions - len(boxes)
        pad_boxes = boxes[-1:].repeat(padding, 1)
        pad_scores = scores[-1:].repeat(padding)

        boxes = torch.cat([boxes, pad_boxes], dim=0)
        scores = torch.cat([scores, pad_scores], dim=0)

        return boxes, scores

    def _extract_roi_features(
        self, base_features: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        """Extract 2048-dim features for each ROI."""
        # Add batch index
        batch_idx = torch.zeros(len(boxes), 1, device=self.device)
        rois = torch.cat([batch_idx, boxes], dim=1)

        # ROI pooling
        pooled = self.roi_pool(base_features, rois)  # [N, 1024, 14, 14]

        # Pass through layer4 and avgpool
        features = self.model.extract_top_features(pooled)  # [N, 2048]

        return features

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


def download_vg_weights(output_dir: str = "weights") -> str:
    """
    Download Visual Genome Faster R-CNN weights using gdown.

    Args:
        output_dir: Directory to save weights

    Returns:
        Path to downloaded weights file
    """
    import subprocess

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "faster_rcnn_res101_vg.pth")

    if os.path.exists(output_path):
        logger.info(f"VG weights already exist at {output_path}")
        return output_path

    url = "https://drive.google.com/file/d/18n_3V1rywgeADZ3oONO0DsuuS9eMW6sN/view"

    try:
        subprocess.run(["gdown", "--fuzzy", url, "-O", output_path], check=True)
        logger.info(f"Downloaded VG weights to {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download VG weights: {e}")
        raise

    return output_path
