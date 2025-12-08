"""Faster R-CNN with ResNet-152 FPN backbone (COCO) feature extractor.

This extractor uses a ResNet-152 backbone with Feature Pyramid Network (FPN)
for Faster R-CNN object detection. The detection head weights are initialized
from the ResNet-50 FPN v2 model (COCO pretrained), while the backbone uses
ImageNet-pretrained ResNet-152 weights.

This allows us to test whether a stronger backbone improves feature quality
for ViLBERT, compared to the standard ResNet-50 FPN version.
"""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet152_Weights, resnet152
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)

from ..base import BaseFeatureExtractor, register_feature_extractor

logger = logging.getLogger(__name__)


def build_resnet152_fpn_backbone(trainable_layers: int = 3) -> BackboneWithFPN:
    """
    Build a ResNet-152 backbone with FPN for Faster R-CNN.

    Args:
        trainable_layers: Number of trainable (not frozen) layers starting from final block.
                         Valid values are between 0 and 5, with 5 meaning all backbone
                         layers are trainable.

    Returns:
        BackboneWithFPN: ResNet-152 backbone with Feature Pyramid Network
    """
    # Load pretrained ResNet-152
    backbone = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)

    # Freeze layers based on trainable_layers parameter
    # ResNet layers: conv1, bn1, layer1, layer2, layer3, layer4
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][
        :trainable_layers
    ]
    if trainable_layers == 5:
        layers_to_train.append("bn1")

    for name, parameter in backbone.named_parameters():
        if all(not name.startswith(layer) for layer in layers_to_train):
            parameter.requires_grad_(False)

    # Extract feature layers for FPN
    # ResNet-152 channel dimensions: layer1=256, layer2=512, layer3=1024, layer4=2048
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256  # FPN output channels

    # Build the backbone with FPN
    # First, create a feature extractor from ResNet
    from torchvision.models._utils import IntermediateLayerGetter

    body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool(),
    )

    backbone_fpn = BackboneWithFPN(
        backbone=body,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool(),
    )

    # Manually construct since BackboneWithFPN expects specific structure
    class ResNet152FPNBackbone(nn.Module):
        def __init__(self, body, fpn, out_channels):
            super().__init__()
            self.body = body
            self.fpn = fpn
            self.out_channels = out_channels

        def forward(self, x):
            x = self.body(x)
            x = self.fpn(x)
            return x

    return ResNet152FPNBackbone(body, fpn, out_channels)


def build_fasterrcnn_resnet152_fpn(
    num_classes: int = 91,  # COCO has 91 classes (including background)
    trainable_backbone_layers: int = 3,
) -> FasterRCNN:
    """
    Build Faster R-CNN with ResNet-152 FPN backbone.

    The model uses:
    - ResNet-152 backbone pretrained on ImageNet
    - Feature Pyramid Network for multi-scale features
    - Standard Faster R-CNN detection head

    Args:
        num_classes: Number of output classes (91 for COCO)
        trainable_backbone_layers: Number of trainable backbone layers

    Returns:
        FasterRCNN model with ResNet-152 FPN backbone
    """
    backbone = build_resnet152_fpn_backbone(trainable_backbone_layers)

    # RPN anchor generator (same as ResNet-50 FPN v2)
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # ROI pooler
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    # Build Faster R-CNN
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # Use same hyperparameters as ResNet-50 FPN v2
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
    )

    return model


@register_feature_extractor("fasterrcnn_resnet152")
class FasterRCNNResNet152Extractor(BaseFeatureExtractor):
    """
    Extracts object-based features using Faster R-CNN with ResNet-152 FPN backbone.

    This extractor uses a stronger ResNet-152 backbone compared to the standard
    ResNet-50 version, potentially providing better visual features for ViLBERT.

    Note: The detection head is NOT pretrained on COCO (only backbone on ImageNet),
    so detection quality may be lower than ResNet-50 FPN v2. However, we primarily
    use this for feature extraction rather than detection accuracy.

    Args:
        output_dim: Output feature dimension (default: 2048)
        num_regions: Number of visual regions to extract (default: 36)
        confidence_threshold: Minimum confidence for detected objects (default: 0.2)
        use_pretrained_detector: If True, initialize detection head from ResNet-50 FPN v2
        device: Device to run on
    """

    def __init__(
        self,
        output_dim: int = 2048,
        num_regions: int = 36,
        confidence_threshold: float = 0.2,
        use_pretrained_detector: bool = True,
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(output_dim, num_regions, device)
        self.confidence_threshold = confidence_threshold

        logger.info("Building Faster R-CNN with ResNet-152 FPN backbone...")

        # Build model
        self.detector = build_fasterrcnn_resnet152_fpn(
            num_classes=91,
            trainable_backbone_layers=0,  # Freeze backbone for feature extraction
        )

        # Optionally load detection head weights from pretrained ResNet-50 FPN v2
        if use_pretrained_detector:
            self._init_detection_head_from_resnet50()

        self.detector.eval()
        self.detector.to(device)

        for param in self.detector.parameters():
            param.requires_grad = False

        # Feature projection: 256 channels * 7x7 pooled = 12544 -> output_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        ).to(device)

        self.backbone = self.detector.backbone
        self.roi_pooler = self.detector.roi_heads.box_roi_pool
        self.transform = transforms.ToTensor()

        # Count parameters
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        total_params = sum(p.numel() for p in self.detector.parameters())

        logger.info(
            f"Faster R-CNN ResNet-152 FPN initialized: "
            f"num_regions={num_regions}, threshold={confidence_threshold}, "
            f"backbone_params={backbone_params:,}, total_params={total_params:,}"
        )

    def _init_detection_head_from_resnet50(self):
        """
        Initialize RPN and ROI heads from pretrained ResNet-50 FPN v2.

        This gives us COCO-pretrained detection capability while using
        the stronger ResNet-152 backbone for features.
        """
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        logger.info("Loading detection head weights from ResNet-50 FPN v2...")

        # Load pretrained ResNet-50 model
        pretrained = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        )

        # Copy RPN weights (only matching keys due to architecture differences)
        rpn_state = pretrained.rpn.state_dict()
        own_rpn_state = self.detector.rpn.state_dict()
        matched_rpn = {
            k: v
            for k, v in rpn_state.items()
            if k in own_rpn_state and own_rpn_state[k].shape == v.shape
        }
        self.detector.rpn.load_state_dict(matched_rpn, strict=False)
        logger.info(f"Loaded {len(matched_rpn)}/{len(own_rpn_state)} RPN weights")

        # Copy ROI heads weights (box_head, box_predictor)
        roi_state = pretrained.roi_heads.state_dict()
        own_roi_state = self.detector.roi_heads.state_dict()
        matched_roi = {
            k: v
            for k, v in roi_state.items()
            if k in own_roi_state and own_roi_state[k].shape == v.shape
        }
        self.detector.roi_heads.load_state_dict(matched_roi, strict=False)
        logger.info(f"Loaded {len(matched_roi)}/{len(own_roi_state)} ROI heads weights")

        logger.info("Detection head weights loaded from ResNet-50 FPN v2")

        del pretrained

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract visual features from detected regions.

        Args:
            image: PIL Image

        Returns:
            Tuple of:
                - features: [num_regions, output_dim] region features
                - spatial: [num_regions, 5] normalized boxes (x1, y1, x2, y2, area)
        """
        img_tensor = self.transform(image).to(self.device)
        img_width, img_height = image.size

        # Run detection
        detections = self.detector([img_tensor])[0]
        boxes, scores = detections["boxes"], detections["scores"]

        # Filter by confidence
        keep = scores >= self.confidence_threshold
        boxes, scores = boxes[keep], scores[keep]

        # Select top regions or pad with grid
        if len(boxes) > self.num_regions:
            _, indices = scores.topk(self.num_regions)
            boxes = boxes[indices]
        elif len(boxes) < self.num_regions:
            boxes = self._pad_boxes_with_grid(boxes, img_width, img_height)

        # Extract ROI features
        features = self._extract_roi_features(img_tensor, boxes)
        spatial = self._normalize_boxes(boxes, img_width, img_height)

        return features, spatial

    def _extract_roi_features(
        self, img_tensor: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        """Extract features from ROI-pooled regions."""
        # Get FPN features
        features = self.backbone(img_tensor.unsqueeze(0))

        # ROI pool from multi-scale features
        pooled = self.roi_pooler(
            features, [boxes], [(img_tensor.shape[1], img_tensor.shape[2])]
        )

        # Flatten and project to output_dim
        pooled_flat = pooled.view(pooled.size(0), -1)
        return self.feature_projection(pooled_flat)

    def _pad_boxes_with_grid(
        self, boxes: torch.Tensor, img_width: int, img_height: int
    ) -> torch.Tensor:
        """Pad with grid boxes if fewer than num_regions detected."""
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
        """Normalize box coordinates to [0, 1] and compute area."""
        normalized = boxes.clone()
        normalized[:, [0, 2]] /= img_width
        normalized[:, [1, 3]] /= img_height

        widths = normalized[:, 2] - normalized[:, 0]
        heights = normalized[:, 3] - normalized[:, 1]
        areas = widths * heights

        return torch.cat([normalized, areas.unsqueeze(1)], dim=1)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch feature extraction."""
        batch_features, batch_spatial = [], []
        for img in images:
            img_pil = transforms.ToPILImage()(img.cpu())
            features, spatial = self.extract_features(img_pil)
            batch_features.append(features)
            batch_spatial.append(spatial)
        return torch.stack(batch_features), torch.stack(batch_spatial)
