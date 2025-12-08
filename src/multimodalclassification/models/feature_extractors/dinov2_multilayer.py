"""
DINOv2 Multi-Layer Feature Extractor

Extracts and fuses visual features from multiple layers of DINOv2 Vision Transformer.
Unlike the standard DINOv2 extractor that uses only the last layer, this implementation
combines features from multiple transformer layers to capture both low-level patterns
(text edges, textures) and high-level semantics.

Hypothesis: For hateful meme detection, earlier layers may capture important
low-level cues like embedded text patterns, while later layers capture semantic
content. Fusing multiple layers should provide richer representations.

Layer Selection (for ViT-L/14 with 24 layers):
- Layer 6: Low-level features (edges, textures, text patterns)
- Layer 12: Mid-level features (parts, local structures)
- Layer 18: High-level features (objects, regions)
- Layer 24: Semantic features (scene understanding)

Fusion Strategies:
- "concat": Concatenate features from all layers, then project
- "weighted_sum": Learnable weighted sum of layer features
- "attention": Cross-layer attention pooling

Expected AUROC: ~0.71-0.73 (hypothesis: better than single-layer 0.7056)

Usage:
    from multimodalclassification.models.feature_extractors import DINOv2MultiLayerExtractor

    extractor = DINOv2MultiLayerExtractor(
        model_size="large",
        num_regions=36,
        layer_indices=[6, 12, 18, 24],
        fusion_strategy="concat"
    )
    features, spatial = extractor.extract_features(image)
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from ..base import BaseFeatureExtractor, register_feature_extractor

logger = logging.getLogger(__name__)


@register_feature_extractor("dinov2_multilayer")
class DINOv2MultiLayerExtractor(BaseFeatureExtractor):
    """
    Extract and fuse visual features from multiple DINOv2 transformer layers.

    This extractor captures features at different levels of abstraction:
    - Early layers: edges, textures, text patterns
    - Middle layers: parts, local structures
    - Late layers: objects, semantic content

    Attributes:
        model: DINOv2 vision transformer
        layer_indices: Which transformer layers to extract from
        fusion: Fusion module to combine multi-layer features
        projection: Final projection to output dimension
    """

    # DINOv2 model configurations
    MODEL_CONFIGS = {
        "small": {
            "name": "dinov2_vits14",
            "hidden_size": 384,
            "patch_size": 14,
            "num_layers": 12,
            "default_layers": [3, 6, 9, 12],
        },
        "base": {
            "name": "dinov2_vitb14",
            "hidden_size": 768,
            "patch_size": 14,
            "num_layers": 12,
            "default_layers": [3, 6, 9, 12],
        },
        "large": {
            "name": "dinov2_vitl14",
            "hidden_size": 1024,
            "patch_size": 14,
            "num_layers": 24,
            "default_layers": [6, 12, 18, 24],
        },
        "giant": {
            "name": "dinov2_vitg14",
            "hidden_size": 1536,
            "patch_size": 14,
            "num_layers": 40,
            "default_layers": [10, 20, 30, 40],
        },
    }

    FUSION_STRATEGIES = ["concat", "weighted_sum", "attention"]

    def __init__(
        self,
        model_size: str = "large",
        output_dim: int = 2048,
        num_regions: int = 36,
        layer_indices: Optional[List[int]] = None,
        fusion_strategy: str = "concat",
        device: str = None,
    ):
        """
        Initialize DINOv2 multi-layer feature extractor.

        Args:
            model_size: Model size - "small", "base", "large", or "giant"
            output_dim: Output feature dimension (default 2048 for ViLBERT)
            num_regions: Number of visual regions to output
            layer_indices: Which layers to extract features from (1-indexed).
                          If None, uses default layers for the model size.
            fusion_strategy: How to combine multi-layer features:
                - "concat": Concatenate and project (default)
                - "weighted_sum": Learnable weighted combination
                - "attention": Cross-layer attention pooling
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

        if fusion_strategy not in self.FUSION_STRATEGIES:
            raise ValueError(
                f"Unknown fusion_strategy: {fusion_strategy}. "
                f"Options: {self.FUSION_STRATEGIES}"
            )

        self.model_size = model_size
        self.config = self.MODEL_CONFIGS[model_size]
        self.hidden_size = self.config["hidden_size"]
        self.patch_size = self.config["patch_size"]
        self.num_layers = self.config["num_layers"]
        self.fusion_strategy = fusion_strategy

        # Set layer indices (convert to 0-indexed internally)
        if layer_indices is None:
            self.layer_indices = self.config["default_layers"]
        else:
            self.layer_indices = layer_indices

        # Validate layer indices
        for idx in self.layer_indices:
            if idx < 1 or idx > self.num_layers:
                raise ValueError(
                    f"Layer index {idx} out of range [1, {self.num_layers}]"
                )

        self.num_extraction_layers = len(self.layer_indices)

        # Load model
        self._load_model()

        # Build fusion module
        self._build_fusion_module()

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
        self.grid_size = 518 // self.patch_size  # 37

        total_params = sum(p.numel() for p in self.model.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        proj_params = sum(p.numel() for p in self.projection.parameters())

        logger.info(
            f"DINOv2 MultiLayer {model_size} initialized: "
            f"layers={self.layer_indices}, "
            f"fusion={fusion_strategy}, "
            f"hidden_size={self.hidden_size}, "
            f"num_regions={num_regions}, "
            f"output_dim={output_dim}, "
            f"model_params={total_params:,}, "
            f"fusion_params={fusion_params:,}, "
            f"projection_params={proj_params:,}"
        )

    def _load_model(self):
        """Load DINOv2 model from torch.hub."""
        model_name = self.config["name"]

        try:
            logger.info(f"Loading DINOv2 from torch.hub: {model_name}")
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=True,
            )
            self.model.eval()
            self.model.to(self.device)
            logger.info(f"Loaded DINOv2 {self.model_size} from torch.hub")

        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv2: {e}")

    def _build_fusion_module(self):
        """Build the fusion module based on the selected strategy."""
        total_hidden = self.hidden_size * self.num_extraction_layers

        if self.fusion_strategy == "concat":
            # Concatenate all layer features, then project
            self.fusion = nn.Identity()
            fusion_input_dim = total_hidden

        elif self.fusion_strategy == "weighted_sum":
            # Learnable weights for each layer
            self.fusion = nn.Sequential(
                LayerWeightedSum(self.num_extraction_layers, self.hidden_size),
            )
            fusion_input_dim = self.hidden_size

        elif self.fusion_strategy == "attention":
            # Cross-layer attention pooling
            self.fusion = CrossLayerAttention(
                num_layers=self.num_extraction_layers,
                hidden_size=self.hidden_size,
                num_heads=8,
            )
            fusion_input_dim = self.hidden_size

        self.fusion.to(self.device)

        # Final projection to output dimension
        self.projection = nn.Sequential(
            nn.Linear(fusion_input_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        ).to(self.device)

        # Initialize projection weights
        self._init_projection_weights()

    def _init_projection_weights(self):
        """Initialize projection layer weights."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _extract_multilayer_features(
        self, img_tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Extract features from multiple transformer layers.

        Args:
            img_tensor: Preprocessed image tensor [1, 3, H, W]

        Returns:
            List of feature tensors, one per layer [1, num_patches, hidden_size]
        """
        layer_features = []

        # Register hooks to capture intermediate layer outputs
        captured_outputs = {}

        def make_hook(layer_idx):
            def hook(module, input, output):
                captured_outputs[layer_idx] = output

            return hook

        # Register hooks for each layer we want to capture
        hooks = []
        for layer_idx in self.layer_indices:
            # DINOv2 layers are 0-indexed internally
            block = self.model.blocks[layer_idx - 1]
            hook = block.register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)

        try:
            # Forward pass
            with torch.no_grad():
                _ = self.model.forward_features(img_tensor)

            # Collect features from each layer (excluding CLS token)
            for layer_idx in self.layer_indices:
                output = captured_outputs[layer_idx]
                # Remove CLS token (first token)
                patch_features = output[:, 1:, :]  # [1, num_patches, hidden_size]
                layer_features.append(patch_features)

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return layer_features

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract multi-layer visual features from an image.

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

        # Extract features from multiple layers
        layer_features = self._extract_multilayer_features(img_tensor)

        # Fuse multi-layer features
        if self.fusion_strategy == "concat":
            # Concatenate along feature dimension
            # Each layer: [1, num_patches, hidden_size]
            # Result: [1, num_patches, hidden_size * num_layers]
            fused = torch.cat(layer_features, dim=-1)
        else:
            # Stack layers: [1, num_layers, num_patches, hidden_size]
            stacked = torch.stack(layer_features, dim=1)
            # Apply fusion: [1, num_patches, hidden_size]
            fused = self.fusion(stacked)

        # Reshape to grid and interpolate to num_regions
        num_patches = fused.shape[1]
        grid_size = int(num_patches**0.5)  # 37 for 518x518
        feature_dim = fused.shape[-1]

        # Reshape to [1, feature_dim, grid_size, grid_size]
        features_grid = fused.permute(0, 2, 1).view(
            1, feature_dim, grid_size, grid_size
        )

        # Interpolate to target grid size
        target_grid = int(self.num_regions**0.5)  # 6 for 36 regions
        features_resized = torch.nn.functional.interpolate(
            features_grid,
            size=(target_grid, target_grid),
            mode="bilinear",
            align_corners=False,
        )

        # Reshape to [num_regions, feature_dim]
        features_flat = features_resized.permute(0, 2, 3, 1).view(-1, feature_dim)

        # Project to output dimension
        projected = self.projection(features_flat)  # [num_regions, output_dim]

        # Generate spatial locations
        spatial = self._generate_grid_spatial()

        return projected, spatial

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


class LayerWeightedSum(nn.Module):
    """
    Learnable weighted sum of multi-layer features.

    Learns a scalar weight for each layer and computes weighted average.
    """

    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.num_layers = num_layers
        # Learnable weights (initialized to uniform)
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Stacked layer features [B, num_layers, num_patches, hidden_size]

        Returns:
            Weighted sum [B, num_patches, hidden_size]
        """
        # Normalize weights with softmax
        weights = torch.softmax(self.layer_weights, dim=0)
        # Weighted sum: [B, num_patches, hidden_size]
        return torch.einsum("blph,l->bph", x, weights)


class CrossLayerAttention(nn.Module):
    """
    Cross-layer attention pooling.

    Uses multi-head attention to dynamically weight and combine
    features from different layers based on their content.
    """

    def __init__(self, num_layers: int, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Learnable query for pooling
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )

        # Layer norm
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Stacked layer features [B, num_layers, num_patches, hidden_size]

        Returns:
            Attention-pooled features [B, num_patches, hidden_size]
        """
        B, L, P, H = x.shape

        # Process each patch position independently
        # Reshape: [B * P, L, H]
        x_flat = x.permute(0, 2, 1, 3).reshape(B * P, L, H)

        # Expand query for batch: [B * P, 1, H]
        query = self.query.expand(B * P, -1, -1)

        # Cross-attention: query attends to all layers
        # Output: [B * P, 1, H]
        attn_out, _ = self.attention(query, x_flat, x_flat)

        # Reshape back: [B, P, H]
        output = attn_out.squeeze(1).view(B, P, H)

        return self.norm(output)
