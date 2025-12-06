"""
Dataset and DataLoader for precomputed visual features.

This module provides efficient data loading using precomputed features
stored in HDF5 format, similar to Facebook's detectron.lmdb approach.
"""

import logging
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class PrecomputedFeaturesDataset(Dataset):
    """
    Dataset that loads precomputed visual features from HDF5.

    This is much faster than extracting features on-the-fly and provides
    consistent features across training epochs (like Facebook's approach).

    Uses lazy loading to avoid HDF5 pickling issues with multiprocessing.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        features_path: str,
        id_map_path: str,
        tokenizer: BertTokenizer,
        max_seq_length: int = 128,
        num_regions: int = 100,
        visual_feature_dim: int = 2048,
    ):
        """
        Args:
            data: DataFrame with columns ['id', 'text', 'label', 'img_path']
            features_path: Path to HDF5 file with precomputed features
            id_map_path: Path to .npy file with image_id -> index mapping
            tokenizer: BERT tokenizer
            max_seq_length: Maximum sequence length for text
            num_regions: Number of visual regions per image
            visual_feature_dim: Dimension of visual features
        """
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_regions = num_regions
        self.visual_feature_dim = visual_feature_dim
        self.features_path = features_path

        # Load ID mapping
        self.id_map = np.load(id_map_path, allow_pickle=True).item()

        # Lazy loading - don't open HDF5 file in __init__ (not picklable)
        self._h5_file = None

        logger.info(
            f"Loaded precomputed features mapping: {len(self.id_map)} images, "
            f"{self.num_regions} regions, {self.visual_feature_dim}D"
        )

    def _get_h5_file(self):
        """Lazily open HDF5 file (needed for multiprocessing compatibility)."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.features_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]

        # Get image ID and look up feature index
        img_id = str(row["id"])

        if img_id in self.id_map:
            feat_idx = self.id_map[img_id]
            h5_file = self._get_h5_file()
            visual_feats = torch.tensor(
                h5_file["visual_features"][feat_idx], dtype=torch.float32
            )
            spatial_feats = torch.tensor(
                h5_file["spatial_features"][feat_idx], dtype=torch.float32
            )
        else:
            # Fallback: zero features if not found
            logger.warning(f"Image ID {img_id} not found in precomputed features")
            visual_feats = torch.zeros(
                self.num_regions, self.visual_feature_dim, dtype=torch.float32
            )
            spatial_feats = torch.zeros(self.num_regions, 5, dtype=torch.float32)

        # Tokenize text
        text = str(row.get("text", ""))
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Get label
        label = int(row.get("label", 0))

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids", torch.zeros_like(encoding["input_ids"])
            ).squeeze(0),
            "visual_features": visual_feats,
            "spatial_locations": spatial_feats,  # Model expects spatial_locations
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def __del__(self):
        """Close HDF5 file on cleanup."""
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except:
                pass


def create_precomputed_dataloaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features_path: str,
    id_map_path: str,
    batch_size: int = 32,
    max_seq_length: int = 128,
    num_regions: int = 100,
    visual_feature_dim: int = 2048,
    num_workers: int = 0,  # Use 0 for HDF5 compatibility
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for precomputed features.

    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        test_data: Test DataFrame
        features_path: Path to HDF5 features file
        id_map_path: Path to ID mapping file
        batch_size: Batch size
        max_seq_length: Max text sequence length
        num_regions: Number of visual regions
        visual_feature_dim: Visual feature dimension
        num_workers: DataLoader workers (0 recommended for HDF5)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create datasets
    train_dataset = PrecomputedFeaturesDataset(
        data=train_data,
        features_path=features_path,
        id_map_path=id_map_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        num_regions=num_regions,
        visual_feature_dim=visual_feature_dim,
    )

    val_dataset = PrecomputedFeaturesDataset(
        data=val_data,
        features_path=features_path,
        id_map_path=id_map_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        num_regions=num_regions,
        visual_feature_dim=visual_feature_dim,
    )

    test_dataset = PrecomputedFeaturesDataset(
        data=test_data,
        features_path=features_path,
        id_map_path=id_map_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        num_regions=num_regions,
        visual_feature_dim=visual_feature_dim,
    )

    # Create dataloaders (num_workers=0 for HDF5 compatibility)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Precomputed DataLoaders: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
