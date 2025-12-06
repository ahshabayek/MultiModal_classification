#!/usr/bin/env python
"""
Download Facebook's LMDB features for Hateful Memes.

Usage:
    python scripts/download_lmdb.py
    python scripts/download_lmdb.py --output data/03_features/mmf
"""

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path

LMDB_URL = "https://dl.fbaipublicfiles.com/mmf/data/datasets/hateful_memes/defaults/features/features_2020_10_01.tar.gz"
EXPECTED_SIZE_GB = 9.6


def download_file(url: str, output_path: str) -> bool:
    """Download file using wget, curl, or requests."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    print(f"Expected size: ~{EXPECTED_SIZE_GB} GB (this may take 20-30 minutes)")

    # Try wget
    try:
        subprocess.run(
            ["wget", "-O", output_path, "--progress=bar:force", url],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try curl
    try:
        subprocess.run(
            ["curl", "-L", "-o", output_path, "--progress-bar", url],
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Try Python requests
    try:
        import requests

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        block_size = 8192

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    percent = downloaded / total_size * 100
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(
                        f"\rProgress: {percent:.1f}% ({downloaded_mb:.0f}/{total_mb:.0f} MB)",
                        end="",
                    )
        print()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def extract_tarball(tar_path: str, output_dir: str) -> bool:
    """Extract tar.gz file."""
    print(f"Extracting {tar_path}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
        print("Extraction complete")
        return True
    except Exception as e:
        print(f"Extraction error: {e}")
        return False


def verify_lmdb(lmdb_path: str) -> bool:
    """Verify LMDB database is readable."""
    try:
        import lmdb

        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            num_entries = txn.stat()["entries"]
        env.close()
        print(f"LMDB verified: {num_entries} entries")
        return num_entries > 0
    except Exception as e:
        print(f"LMDB verification failed: {e}")
        return False


def download_lmdb(
    output_dir: str = "data/03_features/mmf", keep_tarball: bool = False
) -> str:
    """
    Download and extract Facebook's LMDB features.

    Args:
        output_dir: Directory to save files
        keep_tarball: Whether to keep the tar.gz after extraction

    Returns:
        Path to detectron.lmdb directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lmdb_path = output_dir / "detectron.lmdb"
    tar_path = output_dir / "features.tar.gz"

    # Check if already exists
    if lmdb_path.exists():
        print(f"LMDB already exists at {lmdb_path}")
        if verify_lmdb(str(lmdb_path)):
            return str(lmdb_path)
        print("Existing LMDB is corrupted, re-downloading...")

    # Download
    if not tar_path.exists():
        if not download_file(LMDB_URL, str(tar_path)):
            raise RuntimeError("Download failed")
    else:
        print(f"Tarball already exists at {tar_path}")

    # Extract
    if not extract_tarball(str(tar_path), str(output_dir)):
        raise RuntimeError("Extraction failed")

    # Verify
    if not verify_lmdb(str(lmdb_path)):
        raise RuntimeError("LMDB verification failed")

    # Cleanup
    if not keep_tarball and tar_path.exists():
        print(f"Removing tarball to save space...")
        tar_path.unlink()

    print(f"\nLMDB ready at: {lmdb_path}")
    return str(lmdb_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download Facebook's LMDB features for Hateful Memes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/03_features/mmf",
        help="Output directory",
    )
    parser.add_argument(
        "--keep-tarball",
        action="store_true",
        help="Keep the tar.gz file after extraction",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing LMDB",
    )

    args = parser.parse_args()

    if args.verify_only:
        lmdb_path = Path(args.output) / "detectron.lmdb"
        if lmdb_path.exists():
            if verify_lmdb(str(lmdb_path)):
                print("LMDB is valid")
                sys.exit(0)
            else:
                print("LMDB is invalid")
                sys.exit(1)
        else:
            print(f"LMDB not found at {lmdb_path}")
            sys.exit(1)

    try:
        download_lmdb(args.output, args.keep_tarball)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
