#!/usr/bin/env python
"""
Download Pretrained ViLBERT Weights

Sources:
1. Original vilbert_beta (Conceptual Captions pretrained)
2. Facebook MMF model zoo (Hateful Memes fine-tuned)

Usage:
    python download_weights.py --source vilbert_cc --output ./weights/
    python download_weights.py --source mmf --output ./weights/
"""

import argparse
import os
import subprocess
import sys
from typing import Optional

# URLs for pretrained weights
WEIGHT_SOURCES = {
    # Original ViLBERT pretrained on Conceptual Captions
    "vilbert_cc": {
        "url": "https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin",
        "filename": "vilbert_pretrained_cc.bin",
        "description": "ViLBERT pretrained on Conceptual Captions (recommended)",
    },
    # Multi-task ViLBERT
    "vilbert_multi_task": {
        "url": "https://dl.fbaipublicfiles.com/vilbert-multi-task/multi_task_model.bin",
        "filename": "vilbert_multi_task.bin",
        "description": "ViLBERT multi-task model",
    },
}

# HuggingFace Hub models (community-uploaded)
HUGGINGFACE_MODELS = {
    "vilbert_hf": {
        "repo_id": "visualjoyce/transformers4vl-vilbert",
        "filename": "pytorch_model.bin",
        "description": "ViLBERT from HuggingFace Hub (community upload)",
    },
    "vilbert_mt_hf": {
        "repo_id": "visualjoyce/transformers4vl-vilbert-mt",
        "filename": "pytorch_model.bin",
        "description": "ViLBERT Multi-Task from HuggingFace Hub",
    },
}


def download_file(url: str, output_path: str) -> bool:
    """Download a file using wget or curl."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    # Try wget first
    try:
        result = subprocess.run(
            ["wget", "-O", output_path, url], check=True, capture_output=True, text=True
        )
        print("Download complete!")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try curl
    try:
        result = subprocess.run(
            ["curl", "-L", "-o", output_path, url],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Download complete!")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try Python requests
    try:
        import requests

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = downloaded / total_size * 100
                    print(f"\rDownloading: {percent:.1f}%", end="")
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def download_mmf_weights(output_dir: str) -> bool:
    """
    Download weights from MMF model zoo.

    Note: MMF uses its own download mechanism. This provides instructions.
    """
    print("\n" + "=" * 60)
    print("MMF Model Zoo Instructions")
    print("=" * 60)
    print("""
To download pretrained weights from MMF model zoo:

1. Install MMF:
   pip install mmf

2. Download ViLBERT pretrained on Conceptual Captions:
   mmf_convert model=vilbert dataset=hateful_memes

3. Or use the checkpoint zoo key directly in your config:
   checkpoint.resume_zoo=vilbert.pretrained.cc

Available model zoo keys for Hateful Memes:
- vilbert.pretrained.cc             (Conceptual Captions pretrained)
- vilbert.finetuned.hateful_memes.direct
- visual_bert.pretrained.coco       (COCO pretrained)
- visual_bert.finetuned.hateful_memes.from_coco

MMF will download these automatically when you run:
   mmf_run config=projects/hateful_memes/configs/vilbert/from_cc.yaml \\
           model=vilbert dataset=hateful_memes
""")
    return True


def download_from_huggingface(source: str, output_dir: str) -> bool:
    """
    Download pretrained weights from HuggingFace Hub.

    Args:
        source: Key from HUGGINGFACE_MODELS
        output_dir: Directory to save weights

    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "Error: huggingface_hub not installed. Install with: pip install huggingface_hub"
        )
        return False

    if source not in HUGGINGFACE_MODELS:
        print(f"Unknown HuggingFace source: {source}")
        return False

    model_info = HUGGINGFACE_MODELS[source]
    print(f"\n{model_info['description']}")
    print(f"Downloading from: {model_info['repo_id']}")

    try:
        # Download from HuggingFace Hub
        downloaded_path = hf_hub_download(
            repo_id=model_info["repo_id"], filename=model_info["filename"]
        )

        # Copy to output directory
        output_path = os.path.join(output_dir, f"{source}.bin")
        import shutil

        shutil.copy(downloaded_path, output_path)

        print(f"✓ Downloaded to: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading from HuggingFace: {e}")
        return False


def download_weights(source: str, output_dir: str) -> bool:
    """Download pretrained weights from specified source."""
    os.makedirs(output_dir, exist_ok=True)

    if source == "mmf":
        return download_mmf_weights(output_dir)

    # Check if it's a HuggingFace source
    if source in HUGGINGFACE_MODELS:
        return download_from_huggingface(source, output_dir)

    if source not in WEIGHT_SOURCES:
        print(f"Unknown source: {source}")
        all_sources = (
            list(WEIGHT_SOURCES.keys()) + list(HUGGINGFACE_MODELS.keys()) + ["mmf"]
        )
        print(f"Available sources: {all_sources}")
        return False

    weight_info = WEIGHT_SOURCES[source]
    output_path = os.path.join(output_dir, weight_info["filename"])

    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        response = input("Download again? [y/N]: ")
        if response.lower() != "y":
            return True

    print(f"\n{weight_info['description']}")
    return download_file(weight_info["url"], output_path)


def verify_weights(weight_path: str) -> bool:
    """Verify that downloaded weights are valid."""
    import torch

    print(f"\nVerifying weights: {weight_path}")

    try:
        state_dict = torch.load(weight_path, map_location="cpu")

        # Check if it's a checkpoint or raw state dict
        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

        num_params = len(state_dict)
        print(f"✓ Successfully loaded weights with {num_params} parameter tensors")

        # Print some key info
        if num_params > 0:
            keys = list(state_dict.keys())
            print(f"  First key: {keys[0]}")
            print(f"  Last key: {keys[-1]}")

            # Estimate total parameters
            total = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
            print(f"  Total parameters: {total:,}")

        return True
    except Exception as e:
        print(f"✗ Failed to load weights: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download ViLBERT pretrained weights")

    all_sources = (
        list(WEIGHT_SOURCES.keys()) + list(HUGGINGFACE_MODELS.keys()) + ["mmf", "all"]
    )

    parser.add_argument(
        "--source",
        type=str,
        default="vilbert_cc",
        choices=all_sources,
        help="Weight source to download",
    )
    parser.add_argument(
        "--output", type=str, default="./weights", help="Output directory for weights"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify downloaded weights"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available weight sources"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable pretrained weight sources:")
        print("=" * 60)

        print("\n📥 Facebook Sources (Direct Download):")
        for key, info in WEIGHT_SOURCES.items():
            print(f"\n  {key}:")
            print(f"    {info['description']}")
            print(f"    URL: {info['url']}")

        print("\n🤗 HuggingFace Hub Sources:")
        for key, info in HUGGINGFACE_MODELS.items():
            print(f"\n  {key}:")
            print(f"    {info['description']}")
            print(f"    Repo: {info['repo_id']}")

        print(f"\n📦 mmf:")
        print(f"    Weights from Facebook MMF model zoo (instructions)")
        return

    if args.source == "all":
        for source in WEIGHT_SOURCES.keys():
            print(f"\n{'=' * 60}")
            print(f"Downloading: {source}")
            print("=" * 60)
            download_weights(source, args.output)
    else:
        download_weights(args.source, args.output)

    if args.verify:
        print("\n" + "=" * 60)
        print("Verifying downloaded weights")
        print("=" * 60)

        for filename in os.listdir(args.output):
            if filename.endswith(".bin") or filename.endswith(".pth"):
                weight_path = os.path.join(args.output, filename)
                verify_weights(weight_path)


if __name__ == "__main__":
    main()
