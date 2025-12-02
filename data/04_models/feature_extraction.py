"""
Visual Feature Extraction using Faster R-CNN (Detectron2)

Extracts region features from images for ViLBERT/VisualBERT models.
Based on Bottom-Up Attention approach (Anderson et al., 2018).

Features:
- 2048-dimensional features per region (fc6 layer)
- 10-100 regions per image (default 36)
- Bounding box coordinates (normalized)
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# Detectron2 imports (handle import errors gracefully)
try:
    import detectron2
    from detectron2 import model_zoo
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.structures import Boxes, ImageList

    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    print("Warning: Detectron2 not available. Install with: pip install detectron2")


class FasterRCNNFeatureExtractor:
    """
    Extract visual features using Faster R-CNN backbone.

    This extractor uses a pretrained Faster R-CNN model to extract
    region-of-interest (ROI) features for vision-language models.
    """

    def __init__(
        self,
        model_config: str = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
        num_features: int = 36,
        feature_dim: int = 2048,
        min_boxes: int = 10,
        max_boxes: int = 100,
        conf_threshold: float = 0.2,
        device: str = "cuda",
    ):
        """
        Initialize the feature extractor.

        Args:
            model_config: Detectron2 model config file
            num_features: Fixed number of features to extract (if set)
            feature_dim: Dimension of extracted features
            min_boxes: Minimum number of boxes to extract
            max_boxes: Maximum number of boxes to extract
            conf_threshold: Confidence threshold for proposals
            device: Device to run inference on
        """
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.min_boxes = min_boxes
        self.max_boxes = max_boxes
        self.conf_threshold = conf_threshold
        self.device = device

        if DETECTRON2_AVAILABLE:
            self._setup_model(model_config)
        else:
            print(
                "Warning: Using dummy feature extractor. Install detectron2 for real features."
            )
            self.model = None

    def _setup_model(self, model_config: str):
        """Set up the Detectron2 model."""
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_config))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf_threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
        cfg.MODEL.DEVICE = self.device

        self.cfg = cfg
        self.model = build_model(cfg)
        self.model.eval()

        # Load checkpoint
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

    def preprocess_image(
        self, image: Union[str, Image.Image, np.ndarray]
    ) -> torch.Tensor:
        """
        Preprocess image for Detectron2.

        Args:
            image: Image path, PIL Image, or numpy array

        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert BGR to RGB if needed (Detectron2 expects BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1]  # RGB to BGR

        # Convert to tensor
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).float()

        return image

    def extract_features(
        self, image: Union[str, Image.Image, np.ndarray, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract ROI features from a single image.

        Args:
            image: Input image (path, PIL, numpy, or tensor)

        Returns:
            Dictionary containing:
                - features: [num_regions, feature_dim] ROI features
                - boxes: [num_regions, 4] Bounding boxes (x1, y1, x2, y2)
                - normalized_boxes: [num_regions, 5] Normalized bbox + area
                - scores: [num_regions] Confidence scores
                - num_boxes: Number of valid boxes
        """
        if self.model is None:
            return self._dummy_features()

        with torch.no_grad():
            # Preprocess
            if not isinstance(image, torch.Tensor):
                image = self.preprocess_image(image)

            height, width = image.shape[1], image.shape[2]
            image = image.to(self.device)

            # Create image list
            images = ImageList.from_tensors(
                [image], self.cfg.MODEL.RPN.SIZE_DIVISIBILITY
            )

            # Get backbone features
            features = self.model.backbone(images.tensor)

            # Get proposals
            proposals, _ = self.model.proposal_generator(images, features)

            # Get ROI features
            box_features = self._extract_roi_features(features, proposals)

            # Get boxes and scores
            boxes = proposals[0].proposal_boxes.tensor
            scores = proposals[0].objectness_logits.sigmoid()

            # Select top boxes
            num_boxes = min(self.max_boxes, len(boxes))
            num_boxes = max(self.min_boxes, num_boxes)

            if self.num_features is not None:
                num_boxes = self.num_features

            # Sort by score and select top
            sorted_indices = torch.argsort(scores, descending=True)[:num_boxes]

            selected_features = box_features[sorted_indices]
            selected_boxes = boxes[sorted_indices]
            selected_scores = scores[sorted_indices]

            # Normalize boxes
            normalized_boxes = self._normalize_boxes(selected_boxes, height, width)

            return {
                "features": selected_features.cpu(),
                "boxes": selected_boxes.cpu(),
                "normalized_boxes": normalized_boxes.cpu(),
                "scores": selected_scores.cpu(),
                "num_boxes": num_boxes,
                "image_size": (height, width),
            }

    def _extract_roi_features(self, features: Dict, proposals: List) -> torch.Tensor:
        """Extract ROI features from backbone features."""
        # Get the feature maps to pool from
        feature_names = ["p2", "p3", "p4", "p5"]
        feature_maps = [features[f] for f in feature_names if f in features]

        if not feature_maps:
            feature_maps = [features["res4"]]  # Fallback for non-FPN

        # Get proposal boxes
        proposal_boxes = [p.proposal_boxes for p in proposals]

        # ROI pooling
        box_features = self.model.roi_heads.box_pooler(feature_maps, proposal_boxes)

        # Pass through box head to get final features
        box_features = self.model.roi_heads.box_head(box_features)

        return box_features

    def _normalize_boxes(
        self, boxes: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        Normalize bounding boxes and add area feature.

        Returns: [num_boxes, 5] with (x1, y1, x2, y2, area) all normalized
        """
        normalized = boxes.clone()
        normalized[:, [0, 2]] /= width  # x coordinates
        normalized[:, [1, 3]] /= height  # y coordinates

        # Compute normalized area
        area = (normalized[:, 2] - normalized[:, 0]) * (
            normalized[:, 3] - normalized[:, 1]
        )

        return torch.cat([normalized, area.unsqueeze(1)], dim=1)

    def _dummy_features(self) -> Dict[str, torch.Tensor]:
        """Return dummy features when Detectron2 is not available."""
        num_regions = self.num_features if self.num_features else 36
        return {
            "features": torch.randn(num_regions, self.feature_dim),
            "boxes": torch.zeros(num_regions, 4),
            "normalized_boxes": torch.zeros(num_regions, 5),
            "scores": torch.ones(num_regions),
            "num_boxes": num_regions,
            "image_size": (224, 224),
        }

    def extract_batch(
        self, images: List[Union[str, Image.Image]], show_progress: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """Extract features from a batch of images."""
        results = []
        iterator = tqdm(images, desc="Extracting features") if show_progress else images

        for image in iterator:
            result = self.extract_features(image)
            results.append(result)

        return results


class PreExtractedFeatureLoader:
    """
    Load pre-extracted visual features from H5 or numpy files.

    Useful when features have been pre-computed for efficiency.
    """

    def __init__(
        self,
        feature_dir: str,
        feature_format: str = "h5",  # 'h5' or 'npy'
    ):
        """
        Initialize the feature loader.

        Args:
            feature_dir: Directory containing feature files
            feature_format: Format of feature files ('h5' or 'npy')
        """
        self.feature_dir = feature_dir
        self.feature_format = feature_format

        # Build index of available features
        self._build_index()

    def _build_index(self):
        """Build an index of available feature files."""
        self.feature_index = {}

        if not os.path.exists(self.feature_dir):
            print(f"Warning: Feature directory {self.feature_dir} does not exist")
            return

        for filename in os.listdir(self.feature_dir):
            if filename.endswith(f".{self.feature_format}"):
                # Extract image ID from filename
                image_id = os.path.splitext(filename)[0]
                self.feature_index[image_id] = os.path.join(self.feature_dir, filename)

    def load_features(self, image_id: str) -> Dict[str, torch.Tensor]:
        """Load features for a specific image ID."""
        if image_id not in self.feature_index:
            raise KeyError(f"Features not found for image ID: {image_id}")

        filepath = self.feature_index[image_id]

        if self.feature_format == "h5":
            return self._load_h5(filepath)
        else:
            return self._load_npy(filepath)

    def _load_h5(self, filepath: str) -> Dict[str, torch.Tensor]:
        """Load features from H5 file."""
        with h5py.File(filepath, "r") as f:
            return {
                "features": torch.from_numpy(f["features"][:]),
                "boxes": torch.from_numpy(f["boxes"][:]) if "boxes" in f else None,
                "normalized_boxes": torch.from_numpy(f["normalized_boxes"][:])
                if "normalized_boxes" in f
                else None,
                "num_boxes": f["features"].shape[0],
            }

    def _load_npy(self, filepath: str) -> Dict[str, torch.Tensor]:
        """Load features from numpy file."""
        data = np.load(filepath, allow_pickle=True).item()
        return {
            "features": torch.from_numpy(data["features"]),
            "boxes": torch.from_numpy(data.get("boxes", np.zeros((36, 4)))),
            "normalized_boxes": torch.from_numpy(
                data.get("normalized_boxes", np.zeros((36, 5)))
            ),
            "num_boxes": data["features"].shape[0],
        }

    def has_features(self, image_id: str) -> bool:
        """Check if features exist for an image ID."""
        return image_id in self.feature_index


def extract_and_save_features(
    image_dir: str,
    output_dir: str,
    image_list: Optional[List[str]] = None,
    num_features: int = 36,
    batch_size: int = 1,
):
    """
    Extract and save features for a dataset.

    Args:
        image_dir: Directory containing images
        output_dir: Directory to save features
        image_list: Optional list of image filenames to process
        num_features: Number of features per image
        batch_size: Batch size for extraction
    """
    os.makedirs(output_dir, exist_ok=True)

    extractor = FasterRCNNFeatureExtractor(num_features=num_features)

    # Get list of images
    if image_list is None:
        image_list = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    print(f"Extracting features for {len(image_list)} images...")

    for image_name in tqdm(image_list):
        image_path = os.path.join(image_dir, image_name)

        # Skip if already processed
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.h5")
        if os.path.exists(output_path):
            continue

        # Extract features
        features = extractor.extract_features(image_path)

        # Save to H5
        with h5py.File(output_path, "w") as f:
            f.create_dataset("features", data=features["features"].numpy())
            f.create_dataset("boxes", data=features["boxes"].numpy())
            f.create_dataset(
                "normalized_boxes", data=features["normalized_boxes"].numpy()
            )
            f.create_dataset("scores", data=features["scores"].numpy())

    print(f"Features saved to {output_dir}")


class VisualFeatureEmbedding(nn.Module):
    """
    Neural network module to embed visual features for ViLBERT.

    Takes raw Faster R-CNN features and prepares them for the transformer.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 768,
        spatial_dim: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project visual features
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # Project spatial features
        self.spatial_proj = nn.Linear(spatial_dim, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, features: torch.Tensor, spatial_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Embed visual features.

        Args:
            features: [batch, num_regions, feature_dim]
            spatial_features: [batch, num_regions, 5]

        Returns:
            Embedded features [batch, num_regions, hidden_dim]
        """
        # Project features
        x = self.feature_proj(features)

        # Add spatial features if provided
        if spatial_features is not None:
            x = x + self.spatial_proj(spatial_features)

        x = self.layer_norm(x)
        x = self.dropout(x)

        return x


if __name__ == "__main__":
    print("Testing Feature Extractor...")

    # Test with dummy features (when Detectron2 is not available)
    extractor = FasterRCNNFeatureExtractor(num_features=36)

    # Extract dummy features
    features = extractor._dummy_features()
    print(f"Feature shape: {features['features'].shape}")
    print(f"Boxes shape: {features['boxes'].shape}")
    print(f"Normalized boxes shape: {features['normalized_boxes'].shape}")

    # Test visual embedding
    embedding = VisualFeatureEmbedding()
    embedded = embedding(
        features["features"].unsqueeze(0), features["normalized_boxes"].unsqueeze(0)
    )
    print(f"Embedded features shape: {embedded.shape}")
