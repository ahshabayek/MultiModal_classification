"""
DINOv2 Feature Extractor

Extracts visual features using Meta's DINOv2 Vision Transformer.
DINOv2 provides semantically rich self-supervised features that capture
dense visual information without the domain bias of object detection models.

Key advantages over Faster R-CNN:
- No domain mismatch (not limited to COCO's 80 classes)
- Dense patch features (every region is meaningful)
- Self-supervised (no label bias)
- State-of-the-art visual representations

Region Selection Strategies:
- "interpolate": Bilinear interpolation from patch grid (default, fast)
- "attention": Select top-K patches by self-attention score (recommended)
  Uses DINOv2's attention maps to identify most salient regions,
  similar to how object detection selects meaningful regions.

Expected AUROC: ~0.70-0.73 with attention selection
Speed: Medium (transformer-based, similar to CLIP)

Usage:
    from multimodalclassification.models.feature_extractors import DINOv2FeatureExtractor

    # Standard interpolation (baseline)
    extractor = DINOv2FeatureExtractor(model_size="large", num_regions=36)

    # Attention-weighted selection (recommended)
    extractor = DINOv2FeatureExtractor(
        model_size="large",
        num_regions=36,
        region_selection="attention"
    )
    features, spatial = extractor.extract_features(image)
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from ..base import BaseFeatureExtractor, register_feature_extractor

logger = logging.getLogger(__name__)


@register_feature_extractor("dinov2")
class DINOv2FeatureExtractor(BaseFeatureExtractor):
    """
    Extract visual features using Meta's DINOv2 Vision Transformer.

    DINOv2 is a self-supervised vision foundation model that produces
    semantically rich patch-level features. Each patch embedding captures
    dense visual information about that region of the image.

    Attributes:
        model: DINOv2 vision transformer
        projection: Linear projection to output dimension
        transform: Image preprocessing transform
    """

    # DINOv2 model configurations
    MODEL_CONFIGS = {
        "small": {
            "name": "dinov2_vits14",
            "hidden_size": 384,
            "patch_size": 14,
            "hf_name": "facebook/dinov2-small",
        },
        "base": {
            "name": "dinov2_vitb14",
            "hidden_size": 768,
            "patch_size": 14,
            "hf_name": "facebook/dinov2-base",
        },
        "large": {
            "name": "dinov2_vitl14",
            "hidden_size": 1024,
            "patch_size": 14,
            "hf_name": "facebook/dinov2-large",
        },
        "giant": {
            "name": "dinov2_vitg14",
            "hidden_size": 1536,
            "patch_size": 14,
            "hf_name": "facebook/dinov2-giant",
        },
    }

    def __init__(
        self,
        model_size: str = "large",
        output_dim: int = 2048,
        num_regions: int = 36,
        use_registers: bool = False,
        region_selection: str = "interpolate",
        device: str = None,
    ):
        """
        Initialize DINOv2 feature extractor.

        Args:
            model_size: Model size - "small", "base", "large", or "giant"
            output_dim: Output feature dimension (default 2048 for ViLBERT)
            num_regions: Number of visual regions to output
            use_registers: Whether to use DINOv2 with registers (v2 variant)
            region_selection: Region selection strategy:
                - "interpolate": Bilinear interpolation from patch grid (fast, baseline)
                - "attention": Select top-K patches by self-attention score (recommended)
            device: Device to run on (default: cuda if available)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(output_dim, num_regions, device)

        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model_size: {model_size}. "
                f"Options: {list(self.MODEL_CONFIGS.keys())}"
            )

        if region_selection not in ["interpolate", "attention"]:
            raise ValueError(
                f"Unknown region_selection: {region_selection}. "
                f"Options: ['interpolate', 'attention']"
            )

        self.model_size = model_size
        self.config = self.MODEL_CONFIGS[model_size]
        self.hidden_size = self.config["hidden_size"]
        self.patch_size = self.config["patch_size"]
        self.use_registers = use_registers
        self.region_selection = region_selection

        # Load model
        self._load_model()

        # Projection layer to map DINOv2 features to output_dim
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        ).to(device)

        # Initialize projection weights
        self._init_projection_weights()

        # Image preprocessing (DINOv2 uses ImageNet normalization)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    518, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(518),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Calculate grid size for spatial locations
        # 518 / 14 = 37 patches per side
        self.grid_size = 518 // self.patch_size  # 37

        total_params = sum(p.numel() for p in self.model.parameters())
        proj_params = sum(p.numel() for p in self.projection.parameters())
        logger.info(
            f"DINOv2 {model_size} initialized: "
            f"hidden_size={self.hidden_size}, "
            f"num_regions={num_regions}, "
            f"output_dim={output_dim}, "
            f"region_selection={region_selection}, "
            f"model_params={total_params:,}, "
            f"projection_params={proj_params:,}"
        )

    def _load_model(self):
        """Load DINOv2 model from torch.hub or HuggingFace."""
        model_name = self.config["name"]
        if self.use_registers:
            model_name += "_reg"

        try:
            # Try torch.hub first (Facebook's official)
            logger.info(f"Loading DINOv2 from torch.hub: {model_name}")
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=True,
            )
            self.model.eval()
            self.model.to(self.device)
            self.use_transformers = False
            logger.info(f"Loaded DINOv2 {self.model_size} from torch.hub")

        except Exception as e:
            logger.warning(f"torch.hub failed: {e}. Trying HuggingFace...")
            try:
                from transformers import AutoModel

                hf_name = self.config["hf_name"]
                logger.info(f"Loading DINOv2 from HuggingFace: {hf_name}")
                self.model = AutoModel.from_pretrained(hf_name)
                self.model.eval()
                self.model.to(self.device)
                self.use_transformers = True
                logger.info(f"Loaded DINOv2 {self.model_size} from HuggingFace")

            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load DINOv2 from both torch.hub and HuggingFace. "
                    f"torch.hub error: {e}, HuggingFace error: {e2}"
                )

    def _init_projection_weights(self):
        """Initialize projection layer weights."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract visual features from an image.

        Args:
            image: PIL Image

        Returns:
            Tuple of:
                - visual_features: [num_regions, output_dim]
                - spatial_locations: [num_regions, 5]
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Extract features based on selection strategy
        if self.region_selection == "attention":
            return self._extract_attention_weighted(img_tensor)
        else:
            return self._extract_interpolated(img_tensor)

    def _extract_interpolated(
        self, img_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features using bilinear interpolation (original method).

        Args:
            img_tensor: Preprocessed image tensor [1, 3, H, W]

        Returns:
            Tuple of features and spatial locations
        """
        # Extract features
        if self.use_transformers:
            outputs = self.model(img_tensor, output_hidden_states=True)
            patch_features = outputs.last_hidden_state[:, 1:, :]
        else:
            features_dict = self.model.forward_features(img_tensor)
            if isinstance(features_dict, dict):
                patch_features = features_dict["x_norm_patchtokens"]
            else:
                patch_features = features_dict[:, 1:, :]

        # Project to output dimension
        projected = self.projection(patch_features)  # [1, num_patches, output_dim]

        # Reshape to grid and interpolate to num_regions
        num_patches = projected.shape[1]
        grid_size = int(num_patches**0.5)  # 37 for 518x518

        # Reshape to [1, output_dim, grid_size, grid_size]
        features_grid = projected.permute(0, 2, 1).view(
            1, self.output_dim, grid_size, grid_size
        )

        # Interpolate to target grid size
        target_grid = int(self.num_regions**0.5)  # 6 for 36 regions
        features_resized = torch.nn.functional.interpolate(
            features_grid,
            size=(target_grid, target_grid),
            mode="bilinear",
            align_corners=False,
        )

        # Reshape back to [num_regions, output_dim]
        features = features_resized.permute(0, 2, 3, 1).view(-1, self.output_dim)

        # Generate spatial locations for grid
        spatial = self._generate_grid_spatial()

        return features, spatial

    def _extract_attention_weighted(
        self, img_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features using attention-weighted region selection.

        Uses DINOv2's self-attention to identify the most salient patches,
        similar to how object detection selects semantically meaningful regions.

        The attention from CLS token to patch tokens indicates patch importance.
        We select the top-K most attended patches as our regions.

        Args:
            img_tensor: Preprocessed image tensor [1, 3, H, W]

        Returns:
            Tuple of features and spatial locations
        """
        if self.use_transformers:
            # HuggingFace API with attention output
            outputs = self.model(
                img_tensor,
                output_hidden_states=True,
                output_attentions=True,
            )
            patch_features = outputs.last_hidden_state[
                :, 1:, :
            ]  # [1, num_patches, hidden]
            # Get attention from last layer: [1, num_heads, seq_len, seq_len]
            last_attn = outputs.attentions[-1]
            # CLS attention to patches: average over heads, take CLS row (index 0), exclude CLS column
            cls_attn = last_attn[:, :, 0, 1:].mean(dim=1)  # [1, num_patches]
        else:
            # torch.hub API - need to use forward hook to get attention
            patch_features, cls_attn = self._forward_with_attention(img_tensor)

        # patch_features: [1, num_patches, hidden_size]
        # cls_attn: [1, num_patches] - attention scores from CLS to each patch

        num_patches = patch_features.shape[1]
        grid_size = int(num_patches**0.5)  # 37 for 518x518

        # Select top-K patches by attention score
        _, top_indices = torch.topk(
            cls_attn, self.num_regions, dim=1
        )  # [1, num_regions]
        top_indices = top_indices.squeeze(0)  # [num_regions]

        # Sort indices to maintain some spatial coherence (optional but helps)
        top_indices, _ = torch.sort(top_indices)

        # Select features for top patches
        selected_features = patch_features[
            0, top_indices, :
        ]  # [num_regions, hidden_size]

        # Project to output dimension
        projected = self.projection(selected_features)  # [num_regions, output_dim]

        # Generate spatial locations for selected patches
        spatial = self._generate_patch_spatial(top_indices, grid_size)

        return projected, spatial

    def _forward_with_attention(
        self, img_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that captures attention weights using hooks.

        Args:
            img_tensor: Preprocessed image tensor [1, 3, H, W]

        Returns:
            Tuple of (patch_features, cls_attention)
        """
        attention_weights = []

        def attention_hook(module, input, output):
            # DINOv2 attention returns (attn_output, attn_weights)
            # We need to capture the attention weights before softmax or after
            pass

        # For DINOv2 torch.hub, we need to access attention differently
        # The model uses Block modules with Attention sub-modules

        # Get the last transformer block's attention
        last_block = self.model.blocks[-1]

        # Register hook on attention module
        attn_output = []

        def get_attention(module, input, output):
            # input[0] is x, shape [B, N, C]
            B, N, C = input[0].shape
            qkv = module.qkv(input[0])
            qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Compute attention scores
            scale = (C // module.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)

            # Store CLS attention (row 0, columns 1: for patches)
            cls_attn = attn[:, :, 0, 1:].mean(dim=1)  # [B, num_patches]
            attn_output.append(cls_attn)

        hook = last_block.attn.register_forward_hook(get_attention)

        try:
            # Forward pass
            features_dict = self.model.forward_features(img_tensor)
            if isinstance(features_dict, dict):
                patch_features = features_dict["x_norm_patchtokens"]
            else:
                patch_features = features_dict[:, 1:, :]

            cls_attn = attn_output[0] if attn_output else None
        finally:
            hook.remove()

        if cls_attn is None:
            # Fallback: use feature norm as saliency proxy
            logger.warning("Could not extract attention, using feature norm as proxy")
            cls_attn = patch_features.norm(dim=-1)  # [1, num_patches]

        return patch_features, cls_attn

    def _generate_patch_spatial(
        self, patch_indices: torch.Tensor, grid_size: int
    ) -> torch.Tensor:
        """
        Generate spatial location features for selected patches.

        Args:
            patch_indices: Indices of selected patches [num_regions]
            grid_size: Size of the patch grid (e.g., 37)

        Returns:
            Tensor of shape [num_regions, 5] with (x1, y1, x2, y2, area)
        """
        spatial = torch.zeros(self.num_regions, 5, device=self.device)

        for i, idx in enumerate(patch_indices):
            idx = idx.item()
            row = idx // grid_size
            col = idx % grid_size

            x1 = col / grid_size
            y1 = row / grid_size
            x2 = (col + 1) / grid_size
            y2 = (row + 1) / grid_size
            area = (x2 - x1) * (y2 - y1)

            spatial[i] = torch.tensor([x1, y1, x2, y2, area], device=self.device)

        return spatial

    def _generate_grid_spatial(self) -> torch.Tensor:
        """
        Generate spatial location features for a uniform grid.

        Returns:
            Tensor of shape [num_regions, 5] with (x1, y1, x2, y2, area)
        """
        grid_size = int(self.num_regions**0.5)
        spatial = torch.zeros(self.num_regions, 5, device=self.device)

        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                x1 = j / grid_size
                y1 = i / grid_size
                x2 = (j + 1) / grid_size
                y2 = (i + 1) / grid_size
                area = (x2 - x1) * (y2 - y1)
                spatial[idx] = torch.tensor([x1, y1, x2, y2, area], device=self.device)

        return spatial

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch feature extraction.

        Args:
            images: Batch of images [B, C, H, W]

        Returns:
            Tuple of:
                - visual_features: [B, num_regions, output_dim]
                - spatial_locations: [B, num_regions, 5]
        """
        batch_features = []
        batch_spatial = []

        for img in images:
            # Convert tensor to PIL Image
            img_pil = transforms.ToPILImage()(img.cpu())
            features, spatial = self.extract_features(img_pil)
            batch_features.append(features)
            batch_spatial.append(spatial)

        return torch.stack(batch_features), torch.stack(batch_spatial)
