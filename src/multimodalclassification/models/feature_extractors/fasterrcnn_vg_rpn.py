"""
Faster R-CNN Feature Extractor with Visual Genome RPN

This implementation properly loads and uses the trained RPN from the Visual Genome
checkpoint, providing learned region proposals instead of grid-based proposals.

The checkpoint structure (faster_rcnn_res101_vg.pth):
- RCNN_base: ResNet-101 layers 0-3 (conv1, bn1, layer1, layer2, layer3) -> 1024-dim
- RCNN_top: ResNet-101 layer4 -> 2048-dim
- RCNN_rpn: Region Proposal Network (trained on Visual Genome)
- RCNN_cls_score: 1601-class classifier (1600 VG + background)
- RCNN_bbox_pred: Bounding box regressor

Expected improvement over grid-based: +5-10% AUROC
"""

import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.ops import RoIPool, batched_nms, clip_boxes_to_image, nms

from ..base import BaseFeatureExtractor, register_feature_extractor

logger = logging.getLogger(__name__)


class RPN(nn.Module):
    """
    Region Proposal Network matching the VG checkpoint structure.

    Architecture:
    - RPN_Conv: 3x3 conv (1024 -> 512)
    - RPN_cls_score: 1x1 conv (512 -> num_anchors * 2)
    - RPN_bbox_pred: 1x1 conv (512 -> num_anchors * 4)

    Uses 12 anchors (3 scales × 4 aspect ratios) per location.
    """

    def __init__(self, in_channels: int = 1024, num_anchors: int = 12):
        super().__init__()
        self.num_anchors = num_anchors

        # Matching checkpoint structure
        self.RPN_Conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.RPN_cls_score = nn.Conv2d(512, num_anchors * 2, kernel_size=1)
        self.RPN_bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1)

        # Anchor scales and ratios (from bottom-up-attention config)
        self.anchor_scales = [4, 8, 16, 32]  # Base sizes
        self.anchor_ratios = [0.5, 1.0, 2.0]  # Aspect ratios
        self.feat_stride = 16  # Feature stride from layer3

    def forward(
        self, features: torch.Tensor, img_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate region proposals from feature map.

        Args:
            features: [1, 1024, H, W] feature map from RCNN_base
            img_size: (height, width) of input image

        Returns:
            proposals: [N, 4] top proposals (x1, y1, x2, y2)
            scores: [N] objectness scores
        """
        batch_size = features.size(0)

        # RPN forward
        rpn_conv = F.relu(self.RPN_Conv(features))

        # Classification scores (objectness)
        rpn_cls_score = self.RPN_cls_score(rpn_conv)  # [B, 24, H, W]
        rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 24]
        rpn_cls_score = rpn_cls_score.view(batch_size, -1, 2)  # [B, H*W*12, 2]
        rpn_cls_prob = F.softmax(rpn_cls_score, dim=-1)
        rpn_cls_prob_fg = rpn_cls_prob[:, :, 1]  # Foreground probability

        # Bounding box deltas
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv)  # [B, 48, H, W]
        rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 48]
        rpn_bbox_pred = rpn_bbox_pred.view(batch_size, -1, 4)  # [B, H*W*12, 4]

        # Generate anchors
        feat_h, feat_w = features.size(2), features.size(3)
        anchors = self._generate_anchors(feat_h, feat_w, features.device)

        # Apply bbox deltas to anchors
        proposals = self._apply_deltas(anchors, rpn_bbox_pred[0])

        # Clip to image bounds
        proposals = clip_boxes_to_image(proposals, img_size)

        # Get scores
        scores = rpn_cls_prob_fg[0]

        return proposals, scores

    def _generate_anchors(
        self, feat_h: int, feat_w: int, device: torch.device
    ) -> torch.Tensor:
        """Generate anchor boxes for each feature map location."""
        # Base anchors at (0, 0)
        base_anchors = []
        for scale in self.anchor_scales:
            for ratio in self.anchor_ratios:
                h = scale * self.feat_stride * (ratio**0.5)
                w = scale * self.feat_stride / (ratio**0.5)
                base_anchors.append([-w / 2, -h / 2, w / 2, h / 2])

        base_anchors = torch.tensor(base_anchors, device=device, dtype=torch.float32)

        # Generate shifts for each feature map location
        shift_x = (
            torch.arange(0, feat_w, device=device) * self.feat_stride
            + self.feat_stride // 2
        )
        shift_y = (
            torch.arange(0, feat_h, device=device) * self.feat_stride
            + self.feat_stride // 2
        )

        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1).reshape(
            -1, 4
        )

        # Combine base anchors with shifts
        anchors = base_anchors.unsqueeze(0) + shifts.unsqueeze(1)
        anchors = anchors.reshape(-1, 4)

        return anchors

    def _apply_deltas(
        self, anchors: torch.Tensor, deltas: torch.Tensor
    ) -> torch.Tensor:
        """Apply predicted deltas to anchors."""
        # Anchor dimensions
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights

        # Deltas
        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2].clamp(max=4.0)  # Prevent exp overflow
        dh = deltas[:, 3].clamp(max=4.0)

        # Apply
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        # Convert back to (x1, y1, x2, y2)
        proposals = torch.stack(
            [
                pred_ctr_x - 0.5 * pred_w,
                pred_ctr_y - 0.5 * pred_h,
                pred_ctr_x + 0.5 * pred_w,
                pred_ctr_y + 0.5 * pred_h,
            ],
            dim=-1,
        )

        return proposals


class VGFasterRCNNWithRPN(nn.Module):
    """
    Faster R-CNN with Visual Genome pretrained RPN.

    This model properly loads:
    - RCNN_base: ResNet-101 conv1 through layer3
    - RCNN_top: ResNet-101 layer4
    - RCNN_rpn: Trained Region Proposal Network
    - RCNN_cls_score: 1601-class VG classifier
    """

    NUM_VG_CLASSES = 1601

    def __init__(self):
        super().__init__()

        # Load ImageNet pretrained ResNet-101
        resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)

        # RCNN_base: through layer3 (outputs 1024-dim)
        self.RCNN_base = nn.Sequential(
            resnet.conv1,  # 0
            resnet.bn1,  # 1
            resnet.relu,  # 2
            resnet.maxpool,  # 3
            resnet.layer1,  # 4
            resnet.layer2,  # 5
            resnet.layer3,  # 6
        )

        # RCNN_top: layer4 (outputs 2048-dim)
        self.RCNN_top = resnet.layer4

        # RPN
        self.RCNN_rpn = RPN(in_channels=1024, num_anchors=12)

        # Average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification and bbox heads
        self.RCNN_cls_score = nn.Linear(2048, self.NUM_VG_CLASSES)
        self.RCNN_bbox_pred = nn.Linear(2048, self.NUM_VG_CLASSES * 4)

    def get_base_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through layer3."""
        return self.RCNN_base(x)

    def get_proposals(
        self, features: torch.Tensor, img_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get region proposals from RPN."""
        return self.RCNN_rpn(features, img_size)

    def extract_roi_features(self, pooled: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dim features from ROI-pooled regions."""
        x = self.RCNN_top(pooled)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def get_class_scores(self, features: torch.Tensor) -> torch.Tensor:
        """Get VG class scores."""
        return self.RCNN_cls_score(features)


def load_vg_checkpoint(model: VGFasterRCNNWithRPN, checkpoint_path: str) -> dict:
    """
    Load Visual Genome checkpoint into model.

    Returns dict with loading statistics.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    model_state = model.state_dict()
    loaded = {}
    skipped = {}

    for ckpt_key, ckpt_value in state_dict.items():
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
        "total": len(model_state),
        "skipped": len(skipped),
    }

    logger.info(f"Loaded {stats['loaded']}/{stats['total']} weights from VG checkpoint")

    # Check RPN weights specifically
    rpn_loaded = [k for k in loaded if "rpn" in k.lower()]
    logger.info(f"RPN weights loaded: {len(rpn_loaded)} ({rpn_loaded})")

    return stats


@register_feature_extractor("fasterrcnn_vg_rpn")
class FasterRCNNVGRPNExtractor(BaseFeatureExtractor):
    """
    Faster R-CNN with Visual Genome pretrained RPN for region proposals.

    This extractor uses the TRAINED RPN from the Visual Genome checkpoint
    instead of grid-based proposals, which should significantly improve
    feature quality for ViLBERT.

    Args:
        output_dim: Output feature dimension (default: 2048)
        num_regions: Number of visual regions to extract (default: 36)
        weights_path: Path to VG checkpoint
        nms_threshold: NMS threshold for proposal filtering
        pre_nms_top_n: Number of proposals before NMS
        post_nms_top_n: Number of proposals after NMS
        device: Device to run on
    """

    def __init__(
        self,
        output_dim: int = 2048,
        num_regions: int = 36,
        weights_path: Optional[str] = None,
        nms_threshold: float = 0.7,
        pre_nms_top_n: int = 6000,
        post_nms_top_n: int = 300,
        min_box_size: float = 16,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(output_dim, num_regions, device)

        self.nms_threshold = nms_threshold
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.min_box_size = min_box_size

        # Default weights path
        if weights_path is None:
            weights_path = "weights/faster_rcnn_res101_vg.pth"

        # Initialize model
        self.model = VGFasterRCNNWithRPN()

        # Load VG weights
        if os.path.exists(weights_path):
            logger.info(f"Loading Visual Genome checkpoint from {weights_path}")
            stats = load_vg_checkpoint(self.model, weights_path)
            self.has_vg_weights = stats["loaded"] > 0
        else:
            logger.warning(f"VG weights not found at {weights_path}")
            self.has_vg_weights = False

        # ROI pooling - 14x14 before layer4
        self.roi_pool = RoIPool(output_size=(14, 14), spatial_scale=1 / 16)

        # Move to device and eval mode
        self.model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Image preprocessing (same as bottom-up-attention)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Target size for images
        self.target_size = 600
        self.max_size = 1000

        logger.info(
            f"VG Faster R-CNN with RPN initialized: "
            f"num_regions={num_regions}, output_dim={output_dim}, "
            f"vg_weights={'loaded' if self.has_vg_weights else 'not found'}"
        )

    def _resize_image(self, image: Image.Image) -> Tuple[Image.Image, float]:
        """Resize image maintaining aspect ratio."""
        w, h = image.size
        scale = self.target_size / min(w, h)
        if max(w, h) * scale > self.max_size:
            scale = self.max_size / max(w, h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        image = image.resize((new_w, new_h), Image.BILINEAR)
        return image, scale

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract visual features using VG-trained RPN proposals.

        Args:
            image: PIL Image

        Returns:
            features: [num_regions, 2048] region features
            spatial: [num_regions, 5] normalized boxes
        """
        orig_w, orig_h = image.size

        # Resize image
        image_resized, scale = self._resize_image(image)
        img_tensor = self.transform(image_resized).unsqueeze(0).to(self.device)

        img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]

        # Get base features
        base_features = self.model.get_base_features(img_tensor)

        # Get RPN proposals
        proposals, scores = self.model.get_proposals(base_features, (img_h, img_w))

        # Filter proposals
        boxes, scores = self._filter_proposals(proposals, scores, (img_h, img_w))

        # Extract ROI features
        roi_features = self._extract_roi_features(base_features, boxes)

        # Score regions using VG classifier and select top ones
        cls_scores = self.model.get_class_scores(roi_features)
        region_scores = cls_scores[:, 1:].max(dim=1)[
            0
        ]  # Max class score (excluding bg)

        # Select top num_regions
        if len(boxes) > self.num_regions:
            _, top_idx = region_scores.topk(self.num_regions)
            boxes = boxes[top_idx]
            roi_features = roi_features[top_idx]
        elif len(boxes) < self.num_regions:
            boxes, roi_features = self._pad_regions(
                boxes, roi_features, base_features, img_w, img_h
            )

        # Scale boxes back to original image size
        boxes_orig = boxes / scale

        # Normalize spatial info
        spatial = self._normalize_boxes(boxes_orig, orig_w, orig_h)

        return roi_features, spatial

    def _filter_proposals(
        self, proposals: torch.Tensor, scores: torch.Tensor, img_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter proposals by score, size, and NMS."""
        # Remove small boxes
        widths = proposals[:, 2] - proposals[:, 0]
        heights = proposals[:, 3] - proposals[:, 1]
        keep = (widths >= self.min_box_size) & (heights >= self.min_box_size)
        proposals = proposals[keep]
        scores = scores[keep]

        if len(proposals) == 0:
            return proposals, scores

        # Take top pre_nms_top_n by score
        if len(scores) > self.pre_nms_top_n:
            _, top_idx = scores.topk(self.pre_nms_top_n)
            proposals = proposals[top_idx]
            scores = scores[top_idx]

        # Apply NMS
        keep = nms(proposals, scores, self.nms_threshold)

        # Take top post_nms_top_n
        if len(keep) > self.post_nms_top_n:
            keep = keep[: self.post_nms_top_n]

        return proposals[keep], scores[keep]

    def _extract_roi_features(
        self, base_features: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        """Extract 2048-dim features for each ROI."""
        if len(boxes) == 0:
            return torch.zeros(0, 2048, device=self.device)

        # Add batch index
        batch_idx = torch.zeros(len(boxes), 1, device=self.device)
        rois = torch.cat([batch_idx, boxes], dim=1)

        # ROI pool
        pooled = self.roi_pool(base_features, rois)

        # Extract features through layer4
        features = self.model.extract_roi_features(pooled)

        return features

    def _pad_regions(
        self,
        boxes: torch.Tensor,
        features: torch.Tensor,
        base_features: torch.Tensor,
        img_w: int,
        img_h: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad with grid regions if not enough proposals."""
        num_needed = self.num_regions - len(boxes)

        if num_needed <= 0:
            return boxes[: self.num_regions], features[: self.num_regions]

        # Generate grid boxes
        grid_size = int(num_needed**0.5) + 1
        cell_w = img_w / grid_size
        cell_h = img_h / grid_size

        grid_boxes = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(grid_boxes) >= num_needed:
                    break
                x1 = j * cell_w
                y1 = i * cell_h
                x2 = min((j + 1) * cell_w, img_w)
                y2 = min((i + 1) * cell_h, img_h)
                grid_boxes.append([x1, y1, x2, y2])
            if len(grid_boxes) >= num_needed:
                break

        grid_boxes = torch.tensor(grid_boxes, device=self.device, dtype=torch.float32)

        # Extract features for grid boxes
        grid_features = self._extract_roi_features(base_features, grid_boxes)

        # Combine
        if len(boxes) > 0:
            boxes = torch.cat([boxes, grid_boxes], dim=0)
            features = torch.cat([features, grid_features], dim=0)
        else:
            boxes = grid_boxes
            features = grid_features

        return boxes[: self.num_regions], features[: self.num_regions]

    def _normalize_boxes(
        self, boxes: torch.Tensor, img_w: int, img_h: int
    ) -> torch.Tensor:
        """Normalize boxes to [0, 1] and add area."""
        normalized = boxes.clone()
        normalized[:, [0, 2]] /= img_w
        normalized[:, [1, 3]] /= img_h
        normalized = normalized.clamp(0, 1)

        widths = normalized[:, 2] - normalized[:, 0]
        heights = normalized[:, 3] - normalized[:, 1]
        areas = widths * heights

        return torch.cat([normalized, areas.unsqueeze(1)], dim=1)

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
