#!/usr/bin/env python3
"""
Precompute Visual Genome Faster R-CNN features for all images.

This script extracts and caches visual features for the Hateful Memes dataset,
similar to Facebook's detectron.lmdb approach.

Usage:
    python scripts/extract_features.py --output data/03_features/vg_features.h5

Features stored per image:
    - visual_features: [num_regions, 2048] - ROI pooled features
    - spatial_features: [num_regions, 5] - normalized bbox (x1, y1, x2, y2, area)
    - num_boxes: actual detections before padding
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multimodalclassification.models.feature_extractors.fasterrcnn_vg import (
    FasterRCNNVGExtractor,
)


def load_dataset_info(data_dir: str = "data/01_raw/hateful_memes"):
    """Load image paths and IDs from the dataset."""
    from datasets import load_dataset

    print("Loading Hateful Memes dataset from HuggingFace...")
    dataset = load_dataset("limjiayi/hateful_memes_expanded")

    # Collect all unique image IDs across splits
    all_samples = []
    for split_name in ["train", "validation", "test"]:
        split = dataset[split_name]
        for item in split:
            img_id = item["id"]
            # Image path format: data/01_raw/hateful_memes/img/{id}.png
            img_path = os.path.join(data_dir, "img", f"{img_id}.png")
            if os.path.exists(img_path):
                all_samples.append(
                    {
                        "id": img_id,
                        "img_path": img_path,
                        "split": split_name,
                    }
                )

    # Remove duplicates by ID
    seen_ids = set()
    unique_samples = []
    for sample in all_samples:
        if sample["id"] not in seen_ids:
            seen_ids.add(sample["id"])
            unique_samples.append(sample)

    print(f"Found {len(unique_samples)} unique images")
    return unique_samples


def extract_features(
    samples: list,
    output_path: str,
    weights_path: str = "weights/faster_rcnn_res101_vg.pth",
    num_regions: int = 100,
    batch_size: int = 1,  # Process one at a time for simplicity
    device: str = "cuda",
):
    """Extract and save features for all images."""

    # Initialize feature extractor
    print(f"Initializing VG Faster R-CNN with {num_regions} regions...")
    extractor = FasterRCNNVGExtractor(
        output_dim=2048,
        num_regions=num_regions,
        weights_path=weights_path,
        confidence_threshold=0.2,
        nms_threshold=0.3,
        device=device,
    )
    extractor.eval()

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create HDF5 file
    print(f"Creating HDF5 file: {output_path}")
    with h5py.File(output_path, "w") as f:
        # Create datasets with compression
        n_samples = len(samples)

        # Visual features: [N, num_regions, 2048]
        visual_ds = f.create_dataset(
            "visual_features",
            shape=(n_samples, num_regions, 2048),
            dtype=np.float32,
            chunks=(1, num_regions, 2048),
            compression="gzip",
            compression_opts=4,
        )

        # Spatial features: [N, num_regions, 5] - x1, y1, x2, y2, area (normalized)
        spatial_ds = f.create_dataset(
            "spatial_features",
            shape=(n_samples, num_regions, 5),
            dtype=np.float32,
            chunks=(1, num_regions, 5),
            compression="gzip",
            compression_opts=4,
        )

        # Number of actual boxes (before padding)
        num_boxes_ds = f.create_dataset(
            "num_boxes",
            shape=(n_samples,),
            dtype=np.int32,
        )

        # Image IDs (stored as variable-length strings)
        dt = h5py.special_dtype(vlen=str)
        ids_ds = f.create_dataset(
            "image_ids",
            shape=(n_samples,),
            dtype=dt,
        )

        # Also create an ID to index mapping
        id_to_idx = {}

        # Extract features
        print("Extracting features...")
        for idx, sample in enumerate(tqdm(samples, desc="Extracting")):
            img_path = sample["img_path"]
            img_id = sample["id"]

            try:
                # Load image
                image = Image.open(img_path).convert("RGB")

                # Extract features
                with torch.no_grad():
                    visual_feats, spatial_feats = extractor.extract_features(image)

                # Convert to numpy
                visual_feats = visual_feats.cpu().numpy()  # [num_regions, 2048]
                spatial_feats = spatial_feats.cpu().numpy()  # [num_regions, 5]

                # Count actual boxes (non-zero features)
                # Features are padded by repeating last detection, so count unique
                num_boxes = num_regions  # Default
                for i in range(num_regions - 1, 0, -1):
                    if not np.allclose(visual_feats[i], visual_feats[i - 1], atol=1e-6):
                        num_boxes = i + 1
                        break

                # Store
                visual_ds[idx] = visual_feats
                spatial_ds[idx] = spatial_feats
                num_boxes_ds[idx] = num_boxes
                ids_ds[idx] = str(img_id)
                id_to_idx[str(img_id)] = idx

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Store zeros for failed images
                visual_ds[idx] = np.zeros((num_regions, 2048), dtype=np.float32)
                spatial_ds[idx] = np.zeros((num_regions, 5), dtype=np.float32)
                num_boxes_ds[idx] = 0
                ids_ds[idx] = str(img_id)
                id_to_idx[str(img_id)] = idx

        # Store ID to index mapping as attributes
        f.attrs["num_samples"] = n_samples
        f.attrs["num_regions"] = num_regions
        f.attrs["feature_dim"] = 2048

    # Also save ID mapping separately for fast lookup
    mapping_path = output_path.replace(".h5", "_id_map.npy")
    np.save(mapping_path, id_to_idx)
    print(f"Saved ID mapping to {mapping_path}")

    print(f"Done! Features saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Extract VG Faster R-CNN features")
    parser.add_argument(
        "--output",
        type=str,
        default="data/03_features/vg_features.h5",
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/faster_rcnn_res101_vg.pth",
        help="Path to VG Faster R-CNN weights",
    )
    parser.add_argument(
        "--num-regions",
        type=int,
        default=100,
        help="Number of regions to extract per image",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/01_raw/hateful_memes",
        help="Path to hateful memes data directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    # Load dataset info
    samples = load_dataset_info(args.data_dir)

    # Extract features
    extract_features(
        samples=samples,
        output_path=args.output,
        weights_path=args.weights,
        num_regions=args.num_regions,
        device=args.device,
    )


if __name__ == "__main__":
    main()
