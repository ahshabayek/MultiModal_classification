#!/usr/bin/env python3
"""
Hateful Memes Dataset Download & Validation Script

This script downloads and validates the Hateful Memes dataset from Hugging Face,
which is the verified alternative source since the original competition ended.

Usage:
    python download_and_validate_dataset.py --output-dir data/01_raw/hateful_memes

Requirements:
    pip install datasets pillow pandas tqdm
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import datasets
    except ImportError:
        missing.append("datasets")

    try:
        from PIL import Image
    except ImportError:
        missing.append("pillow")

    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    if missing:
        print("❌ Missing required packages:")
        print(f"   pip install {' '.join(missing)}")
        return False

    return True


def download_from_huggingface(
    output_dir: str,
    dataset_name: str = "neuralcatcher/hateful_memes",
    save_images: bool = True
) -> Dict[str, Any]:
    """
    Download the Hateful Memes dataset from Hugging Face.

    Args:
        output_dir: Directory to save the dataset
        dataset_name: Hugging Face dataset identifier
        save_images: Whether to save images to disk

    Returns:
        Dictionary with dataset statistics
    """
    from datasets import load_dataset
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading dataset from: {dataset_name}")
    print(f"📂 Output directory: {output_path}")
    print()

    # Load dataset from Hugging Face
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("\nTrying alternative source...")
        try:
            dataset = load_dataset("limjiayi/hateful_memes_expanded")
            print("✓ Loaded from limjiayi/hateful_memes_expanded")
        except Exception as e2:
            print(f"❌ Alternative also failed: {e2}")
            return None

    stats = {
        "source": dataset_name,
        "splits": {},
        "total_samples": 0,
        "total_hateful": 0,
    }

    # Create image directory
    img_dir = output_path / "img"
    img_dir.mkdir(exist_ok=True)

    # Process each split
    for split_name in dataset.keys():
        split_data = dataset[split_name]
        split_size = len(split_data)

        print(f"\n📊 Processing {split_name}: {split_size} samples")

        # Convert to pandas and save as JSONL
        records = []
        hateful_count = 0

        for i, example in enumerate(tqdm(split_data, desc=f"  {split_name}")):
            record = {
                "id": example.get("id", str(i)),
                "img": example.get("img", f"img/{i:05d}.png"),
                "text": example.get("text", ""),
                "label": example.get("label", 0)
            }
            records.append(record)

            if record["label"] == 1:
                hateful_count += 1

            # Save image if present and save_images is True
            if save_images and "image" in example and example["image"] is not None:
                img_filename = Path(record["img"]).name
                img_path = img_dir / img_filename
                if not img_path.exists():
                    try:
                        example["image"].save(img_path)
                    except Exception as e:
                        pass  # Skip problematic images

        # Save JSONL file
        jsonl_path = output_path / f"{split_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  ✓ Saved {jsonl_path.name}: {split_size} samples")
        print(f"    Hateful: {hateful_count} ({hateful_count/split_size:.1%})")

        stats["splits"][split_name] = {
            "count": split_size,
            "hateful": hateful_count,
            "not_hateful": split_size - hateful_count,
            "hateful_ratio": hateful_count / split_size if split_size > 0 else 0
        }
        stats["total_samples"] += split_size
        stats["total_hateful"] += hateful_count

    # Count images
    image_count = len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg")))
    stats["image_count"] = image_count

    # Save statistics
    stats_path = output_path / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅ Dataset downloaded successfully!")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Total hateful: {stats['total_hateful']} ({stats['total_hateful']/stats['total_samples']:.1%})")
    print(f"   Images saved: {image_count}")
    print(f"   Statistics: {stats_path}")

    return stats


def validate_dataset(data_dir: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a downloaded Hateful Memes dataset.

    Args:
        data_dir: Path to the dataset directory

    Returns:
        Tuple of (is_valid, validation_stats)
    """
    from PIL import Image

    data_path = Path(data_dir)

    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    print(f"Path: {data_path}")

    validation = {
        "path": str(data_path),
        "errors": [],
        "warnings": [],
        "splits": {},
    }

    # Check directory exists
    if not data_path.exists():
        validation["errors"].append(f"Directory not found: {data_path}")
        return False, validation

    # Check for JSONL files
    jsonl_files = list(data_path.glob("*.jsonl"))
    if not jsonl_files:
        validation["errors"].append("No JSONL files found")
        return False, validation

    print(f"\n📄 Found {len(jsonl_files)} JSONL files:")

    # Validate each JSONL file
    total_samples = 0
    missing_images = []

    img_dir = data_path / "img"
    available_images = set()
    if img_dir.exists():
        available_images = {f.name for f in img_dir.iterdir() if f.is_file()}
        print(f"📷 Found {len(available_images)} images in img/ directory")

    for jsonl_file in sorted(jsonl_files):
        split_name = jsonl_file.stem
        print(f"\n  Validating {split_name}...")

        split_stats = {
            "count": 0,
            "hateful": 0,
            "valid_entries": 0,
            "missing_images": 0,
            "invalid_labels": 0,
        }

        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                        split_stats["count"] += 1

                        # Check required fields
                        has_id = "id" in entry
                        has_img = "img" in entry
                        has_text = "text" in entry
                        has_label = "label" in entry

                        if not all([has_id, has_img, has_text]):
                            continue

                        # Check label
                        label = entry.get("label")
                        if label not in [0, 1, None]:
                            split_stats["invalid_labels"] += 1
                        elif label == 1:
                            split_stats["hateful"] += 1

                        # Check image exists
                        if available_images:
                            img_name = Path(entry["img"]).name
                            if img_name not in available_images:
                                split_stats["missing_images"] += 1
                                if len(missing_images) < 5:
                                    missing_images.append(img_name)

                        split_stats["valid_entries"] += 1

                    except json.JSONDecodeError:
                        validation["errors"].append(f"{split_name} line {line_num}: Invalid JSON")

            total_samples += split_stats["count"]
            validation["splits"][split_name] = split_stats

            print(f"    ✓ {split_stats['count']} entries")
            print(f"      Valid: {split_stats['valid_entries']}")
            print(f"      Hateful: {split_stats['hateful']}")
            if split_stats["missing_images"] > 0:
                print(f"      ⚠ Missing images: {split_stats['missing_images']}")

        except Exception as e:
            validation["errors"].append(f"Error reading {split_name}: {e}")

    # Check image quality (sample)
    if available_images:
        print("\n🖼️ Checking image quality (sample)...")
        sample_images = list(available_images)[:10]
        valid_images = 0
        for img_name in sample_images:
            try:
                img_path = img_dir / img_name
                with Image.open(img_path) as img:
                    if img.size[0] > 0 and img.size[1] > 0:
                        valid_images += 1
            except Exception:
                pass
        print(f"    {valid_images}/{len(sample_images)} sample images valid")

    # Summary
    validation["total_samples"] = total_samples
    validation["total_images"] = len(available_images)

    is_valid = len(validation["errors"]) == 0 and total_samples > 0

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {total_samples}")
    print(f"Total images: {len(available_images)}")

    if validation["errors"]:
        print(f"\n❌ Errors ({len(validation['errors'])}):")
        for err in validation["errors"][:5]:
            print(f"   - {err}")

    if validation["warnings"]:
        print(f"\n⚠️ Warnings ({len(validation['warnings'])}):")
        for warn in validation["warnings"][:5]:
            print(f"   - {warn}")

    if is_valid:
        print("\n✅ Dataset is VALID and ready for use!")
    else:
        print("\n❌ Dataset validation FAILED. See errors above.")

    return is_valid, validation


def compare_to_original_stats(stats: Dict) -> None:
    """Compare downloaded dataset to original paper statistics."""

    print("\n" + "=" * 60)
    print("COMPARISON TO ORIGINAL DATASET")
    print("=" * 60)

    # Original stats from the paper
    original = {
        "train": {"count": 8500, "hateful_ratio": 0.35},
        "dev_seen": {"count": 500, "hateful_ratio": 0.50},
        "dev_unseen": {"count": 540, "hateful_ratio": 0.50},
        "test_seen": {"count": 1000, "hateful_ratio": 0.50},
        "test_unseen": {"count": 1000, "hateful_ratio": 0.50},
    }

    for split_name, orig_stats in original.items():
        if split_name in stats.get("splits", {}):
            downloaded = stats["splits"][split_name]
            count_match = abs(downloaded["count"] - orig_stats["count"]) < 100
            ratio_match = abs(downloaded.get("hateful_ratio", 0) - orig_stats["hateful_ratio"]) < 0.1

            status = "✓" if count_match else "⚠"
            print(f"\n{status} {split_name}:")
            print(f"   Downloaded: {downloaded['count']} samples")
            print(f"   Original:   ~{orig_stats['count']} samples")
            if "hateful_ratio" in downloaded:
                print(f"   Hateful ratio: {downloaded['hateful_ratio']:.1%} (expected ~{orig_stats['hateful_ratio']:.0%})")
        else:
            print(f"\n⚠ {split_name}: NOT FOUND in downloaded data")


def main():
    parser = argparse.ArgumentParser(
        description="Download and validate Hateful Memes dataset from Hugging Face"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/01_raw/hateful_memes",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="neuralcatcher/hateful_memes",
        help="Hugging Face dataset identifier"
    )
    parser.add_argument(
        "--validate-only", "-v",
        action="store_true",
        help="Only validate existing dataset, don't download"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't save images to disk (faster, uses less space)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HATEFUL MEMES DATASET DOWNLOADER")
    print("=" * 60)
    print(f"\nSource: Hugging Face ({args.dataset})")
    print(f"Output: {args.output_dir}")
    print()

    # Check dependencies
    if not check_dependencies():
        return 1

    if args.validate_only:
        # Just validate existing dataset
        is_valid, stats = validate_dataset(args.output_dir)
        return 0 if is_valid else 1

    # Download dataset
    stats = download_from_huggingface(
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        save_images=not args.no_images
    )

    if stats is None:
        return 1

    # Validate downloaded dataset
    is_valid, validation = validate_dataset(args.output_dir)

    # Compare to original
    compare_to_original_stats(stats)

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Your dataset is ready in: {output_dir}

2. Update your Kedro catalog.yml:
   raw_train_data:
     type: pandas.JSONDataset
     filepath: {output_dir}/train.jsonl
     load_args:
       lines: true

3. Run your data processing pipeline:
   kedro run --pipeline=data_processing

4. Start training:
   kedro run --pipeline=model_training
""".format(output_dir=args.output_dir))

    return 0 if is_valid else 1


if __name__ == "__main__":
    exit(main())
