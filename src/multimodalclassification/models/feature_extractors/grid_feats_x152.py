"""X-152++ feature extractor from Facebook's grid-feats-vqa (2020 VQA Challenge winner)."""

import logging
import os
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ..base import BaseFeatureExtractor, register_feature_extractor

logger = logging.getLogger(__name__)

DETECTRON2_AVAILABLE = False
try:
    import detectron2.data.transforms as T
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model

    DETECTRON2_AVAILABLE = True
except ImportError:
    logger.warning(
        "detectron2 not installed. Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'"
    )


def download_x152pp_weights(output_path: str = "weights/X-152pp.pth") -> str:
    """Download X-152++ weights (~1.5GB)."""
    import urllib.request

    url = "https://dl.fbaipublicfiles.com/grid-feats-vqa/X-152pp/X-152pp.pth"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        return output_path

    logger.info(f"Downloading X-152++ weights from {url}...")
    urllib.request.urlretrieve(url, output_path)
    logger.info(f"Downloaded to {output_path}")
    return output_path


def get_x152pp_cfg(weights_path: str):
    """Create detectron2 config for X-152++ model."""
    if not DETECTRON2_AVAILABLE:
        raise ImportError("detectron2 required")

    cfg = get_cfg()

    # ResNeXt-152 32x8d backbone
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.RESNETS.DEPTH = 152
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res5"]
    cfg.MODEL.RESNETS.NORM = "SyncBN"
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.RESNETS.NUM_GROUPS = 32
    cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 8
    cfg.MODEL.RESNETS.RES5_DILATION = 1

    # BGR with ImageNet mean
    cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
    cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

    # ROI Heads (Visual Genome 1600 classes)
    cfg.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1600
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"

    # RPN
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
    cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
    cfg.MODEL.RPN.IN_FEATURES = ["res5"]
    cfg.MODEL.RPN.ANCHOR_SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.RPN.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

    # Input
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    cfg.freeze()
    return cfg


@register_feature_extractor("grid_x152")
class GridFeatsX152Extractor(BaseFeatureExtractor):
    """X-152++ feature extractor using ResNeXt-152 trained on Visual Genome."""

    def __init__(
        self,
        weights_path: str = "weights/X-152pp.pth",
        output_dim: int = 2048,
        num_regions: int = 100,
        confidence_threshold: float = 0.2,
        nms_threshold: float = 0.5,
        device: str = None,
        auto_download: bool = True,
    ):
        if not DETECTRON2_AVAILABLE:
            raise ImportError(
                "detectron2 required. Install: pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(output_dim, num_regions, device)
        self.confidence_threshold = confidence_threshold

        if auto_download and not os.path.exists(weights_path):
            download_x152pp_weights(weights_path)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        logger.info(f"Loading X-152++ from {weights_path}...")

        self.cfg = get_x152pp_cfg(weights_path)
        self.model = build_model(self.cfg)
        DetectionCheckpointer(self.model).load(weights_path)

        self.model.eval()
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False

        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"X-152++ initialized (regions={num_regions}, params={num_params:,})"
        )

    def _preprocess_image(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        img_bgr = np.array(image)[:, :, ::-1]
        img_transformed = self.aug.get_transform(img_bgr).apply_image(img_bgr)
        img_tensor = torch.as_tensor(
            img_transformed.astype("float32").transpose(2, 0, 1)
        ).to(self.device)
        return img_tensor, (image.height, image.width)

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        img_tensor, (height, width) = self._preprocess_image(image)

        inputs = {"image": img_tensor, "height": height, "width": width}
        images = self.model.preprocess_image([inputs])
        features = self.model.backbone(images.tensor)

        proposals, _ = self.model.proposal_generator(images, features, None)
        box_features = self._extract_roi_features(features, proposals)

        boxes = proposals[0].proposal_boxes.tensor
        scores = proposals[0].objectness_logits

        return self._select_regions(box_features, boxes, scores, height, width)

    def _extract_roi_features(self, features: dict, proposals: list) -> torch.Tensor:
        feature_map = [features["res5"]]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        box_features = self.model.roi_heads.box_pooler(feature_map, proposal_boxes)
        return self.model.roi_heads.box_head(box_features)

    def _select_regions(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = torch.sigmoid(scores)
        keep = probs >= self.confidence_threshold
        features, boxes, probs = features[keep], boxes[keep], probs[keep]

        if len(features) > self.num_regions:
            _, indices = probs.topk(self.num_regions)
            features, boxes = features[indices], boxes[indices]
        elif len(features) < self.num_regions:
            pad = self.num_regions - len(features)
            features = torch.cat(
                [features, torch.zeros(pad, features.shape[-1], device=self.device)]
            )
            boxes = torch.cat([boxes, torch.zeros(pad, 4, device=self.device)])

        # Normalize spatial features
        x1, y1, x2, y2 = (
            boxes[:, 0] / width,
            boxes[:, 1] / height,
            boxes[:, 2] / width,
            boxes[:, 3] / height,
        )
        area = (x2 - x1) * (y2 - y1)
        spatial = torch.stack([x1, y1, x2, y2, area], dim=1)

        return features, spatial

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        from torchvision import transforms

        batch_features, batch_spatial = [], []
        for img in images:
            img_pil = transforms.ToPILImage()(img.cpu())
            features, spatial = self.extract_features(img_pil)
            batch_features.append(features)
            batch_spatial.append(spatial)
        return torch.stack(batch_features), torch.stack(batch_spatial)


@register_feature_extractor("grid_x152_standard")
class GridFeatsX152StandardExtractor(GridFeatsX152Extractor):
    """Standard X-152 (without ++ improvements). Slightly smaller/faster."""

    def __init__(self, weights_path: str = "weights/X-152.pth", **kwargs):
        if kwargs.get("auto_download", True) and not os.path.exists(weights_path):
            self._download_weights(weights_path)
        super().__init__(weights_path=weights_path, auto_download=False, **kwargs)

    @staticmethod
    def _download_weights(output_path: str):
        import urllib.request

        url = "https://dl.fbaipublicfiles.com/grid-feats-vqa/X-152/X-152.pth"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Downloading X-152 weights...")
        urllib.request.urlretrieve(url, output_path)
