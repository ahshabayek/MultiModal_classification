"""Dataset for loading visual features from Facebook's detectron.lmdb."""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import lmdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

LMDB_URL = "https://dl.fbaipublicfiles.com/mmf/data/datasets/hateful_memes/defaults/features/features_2020_10_01.tar.gz"


def download_lmdb_features(output_dir: str = "data/03_features/mmf") -> str:
    """Download Facebook's LMDB features if not present."""
    output_dir = Path(output_dir)
    lmdb_path = output_dir / "detectron.lmdb"

    if lmdb_path.exists():
        return str(lmdb_path)

    logger.info(f"LMDB not found at {lmdb_path}, downloading...")
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = output_dir / "features.tar.gz"

    # Download
    if not tar_path.exists():
        import subprocess

        logger.info(f"Downloading ~10GB from {LMDB_URL}...")
        try:
            subprocess.run(
                ["wget", "-O", str(tar_path), "--progress=bar:force", LMDB_URL],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            import urllib.request

            urllib.request.urlretrieve(LMDB_URL, str(tar_path))

    # Extract
    if not lmdb_path.exists():
        import tarfile

        logger.info("Extracting tarball...")
        with tarfile.open(str(tar_path), "r:gz") as tar:
            tar.extractall(path=str(output_dir))

    logger.info(f"LMDB ready at {lmdb_path}")
    return str(lmdb_path)


class LMDBFeaturesDataset(Dataset):
    """Loads precomputed visual features from Facebook's detectron.lmdb."""

    def __init__(
        self,
        data: pd.DataFrame,
        lmdb_path: str,
        tokenizer: BertTokenizer,
        max_seq_length: int = 128,
        num_regions: int = 100,
        visual_feature_dim: int = 2048,
    ):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_regions = num_regions
        self.visual_feature_dim = visual_feature_dim
        self.lmdb_path = lmdb_path
        self._env = None

        self._verify_lmdb()

    def _verify_lmdb(self):
        try:
            env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                max_readers=1,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with env.begin(write=False) as txn:
                stat = txn.stat()
                logger.info(f"LMDB opened: {stat['entries']} entries")
            env.close()
        except Exception as e:
            logger.error(f"Failed to open LMDB at {self.lmdb_path}: {e}")
            raise

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                max_readers=1,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self._env

    def __len__(self) -> int:
        return len(self.data)

    def _query_lmdb(self, img_id: str):
        """Query LMDB with various key formats."""
        env = self._get_env()
        with env.begin(write=False) as txn:
            for key in [
                img_id,
                img_id.encode(),
                f"{img_id}.png".encode(),
                img_id.zfill(5).encode(),
            ]:
                if isinstance(key, str):
                    key = key.encode()
                item = txn.get(key)
                if item is not None:
                    return pickle.loads(item)
        return None

    def _extract_features(self, data_dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract visual features and spatial locations from LMDB data."""
        if data_dict is None:
            return (
                torch.zeros(self.num_regions, self.visual_feature_dim),
                torch.zeros(self.num_regions, 5),
            )

        # Extract features
        if isinstance(data_dict, dict):
            features = (
                data_dict.get("features")
                or data_dict.get("feature")
                or data_dict.get("fc6")
            )
            boxes = data_dict.get("boxes") or data_dict.get("bbox")
        else:
            features = data_dict
            boxes = None

        # Process features
        if features is not None:
            features = np.array(features, dtype=np.float32)
            features = self._pad_or_truncate(features, self.visual_feature_dim)
            visual_feats = torch.tensor(features, dtype=torch.float32)
        else:
            visual_feats = torch.zeros(self.num_regions, self.visual_feature_dim)

        # Process spatial features
        spatial_feats = self._process_boxes(boxes)

        return visual_feats, spatial_feats

    def _pad_or_truncate(self, arr: np.ndarray, feat_dim: int) -> np.ndarray:
        """Pad or truncate array to num_regions."""
        if len(arr) < self.num_regions:
            pad = np.zeros((self.num_regions - len(arr), feat_dim), dtype=np.float32)
            return np.vstack([arr, pad])
        return arr[: self.num_regions]

    def _process_boxes(self, boxes) -> torch.Tensor:
        """Convert bounding boxes to normalized spatial features [x1, y1, x2, y2, area]."""
        if boxes is None:
            return torch.zeros(self.num_regions, 5)

        boxes = np.array(boxes, dtype=np.float32)
        if len(boxes.shape) != 2 or boxes.shape[1] < 4:
            return torch.zeros(self.num_regions, 5)

        # Normalize by assumed 1000x1000 image size
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        area = (w * h) / 1000000.0

        spatial = np.column_stack(
            [
                boxes[:, 0] / 1000.0,
                boxes[:, 1] / 1000.0,
                boxes[:, 2] / 1000.0,
                boxes[:, 3] / 1000.0,
                area,
            ]
        )

        spatial = self._pad_or_truncate(spatial, 5)
        return torch.tensor(spatial, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        img_id = str(row["id"])

        # Get visual features from LMDB
        data_dict = self._query_lmdb(img_id)
        if data_dict is None:
            logger.warning(f"Image ID {img_id} not found in LMDB")
        visual_feats, spatial_feats = self._extract_features(data_dict)

        # Tokenize text
        text = str(row.get("text", ""))
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids", torch.zeros_like(encoding["input_ids"])
            ).squeeze(0),
            "visual_features": visual_feats,
            "spatial_locations": spatial_feats,
            "labels": torch.tensor(int(row.get("label", 0)), dtype=torch.long),
        }

    def __del__(self):
        if self._env is not None:
            try:
                self._env.close()
            except:
                pass


def create_lmdb_dataloaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    lmdb_path: str = "data/03_features/mmf/detectron.lmdb",
    batch_size: int = 32,
    max_seq_length: int = 128,
    num_regions: int = 100,
    visual_feature_dim: int = 2048,
    num_workers: int = 0,
    auto_download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for LMDB features."""
    # Auto-download if LMDB doesn't exist
    if auto_download and not Path(lmdb_path).exists():
        lmdb_dir = str(Path(lmdb_path).parent)
        lmdb_path = download_lmdb_features(lmdb_dir)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def make_dataset(data):
        return LMDBFeaturesDataset(
            data=data,
            lmdb_path=lmdb_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            num_regions=num_regions,
            visual_feature_dim=visual_feature_dim,
        )

    train_dataset = make_dataset(train_data)
    val_dataset = make_dataset(val_data)
    test_dataset = make_dataset(test_data)

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
        f"LMDB DataLoaders: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
