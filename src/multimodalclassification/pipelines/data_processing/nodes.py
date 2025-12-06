"""
Data Processing Nodes for Hateful Memes Classification

Uses HuggingFace datasets to load the hateful memes dataset with images.
Supports caption enrichment for improved performance (+2-6% AUROC).
"""

import logging
import os
import warnings
from typing import Any, Dict, Tuple

import pandas as pd
import requests
from datasets import Dataset as HFDataset
from datasets import Image as HFImage
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def enrich_with_captions(
    df: pd.DataFrame,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """
    Enrich dataset with image captions using BLIP.

    This implements Caption Enriched Samples (CES) which improved
    ViLBERT AUROC by +2-6% on Hateful Memes.

    Reference: "Caption Enriched Samples for Improving Hateful Memes Detection" (EMNLP 2021)
    """
    if not parameters.get("use_captions", False):
        return df

    try:
        from .augmentation import enrich_with_captions as _enrich
    except ImportError:
        from augmentation import enrich_with_captions as _enrich

    cache_path = parameters.get(
        "caption_cache_path", "data/02_intermediate/captions.csv"
    )

    logger.info("Enriching dataset with image captions...")
    df = _enrich(
        df,
        image_column="img_path",
        text_column="text",
        output_column="text_enriched",
        cache_path=cache_path,
    )

    # Use enriched text as the main text
    df["text_original"] = df["text"]
    df["text"] = df["text_enriched"]

    return df


def load_hateful_memes_from_huggingface(
    parameters: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Load Hateful Memes dataset from HuggingFace.

    This replicates the loading logic from hateful-memes.ipynb.
    """
    logger.info("Loading Hateful Memes dataset from HuggingFace...")

    # Suppress HuggingFace token warning
    warnings.filterwarnings(
        "ignore", message="Error while fetching `HF_TOKEN` secret value from your vault"
    )

    # Load the public dataset
    dataset = load_dataset("neuralcatcher/hateful_memes")
    logger.info(f"Loaded dataset: {dataset}")

    # Remove duplicates from each split
    for split_name, split_data in dataset.items():
        dataset[split_name] = HFDataset.from_pandas(
            pd.DataFrame(split_data).drop_duplicates(), preserve_index=False
        )
    logger.info("Removed duplicates from all splits")

    # Get data directory from parameters or use default
    data_dir = parameters.get("data_dir", "data/01_raw/hateful_memes")
    img_dir = os.path.join(data_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    # Download images if needed
    _download_images(dataset, data_dir, parameters)

    # Convert to pandas DataFrames with absolute image paths
    datasets_dict = {}
    for split_name, split_data in dataset.items():
        df = pd.DataFrame(split_data)
        # Convert relative img paths to absolute paths
        df["img_path"] = df["img"].apply(
            lambda x: os.path.abspath(os.path.join(data_dir, x))
        )
        datasets_dict[split_name] = df
        logger.info(f"  {split_name}: {len(df)} samples")

    return datasets_dict


def _download_images(dataset, data_dir: str, parameters: Dict[str, Any]) -> None:
    """Download images from Google Drive archive or fetch missing ones."""
    import tarfile

    import gdown

    img_dir = os.path.join(data_dir, "img")
    archive_path = os.path.join(data_dir, "img.tar.gz")

    # Check if images already exist
    if os.path.exists(img_dir) and len(os.listdir(img_dir)) > 100:
        logger.info(f"Images already exist in {img_dir}, skipping download")
    else:
        # Download from Google Drive
        gdrive_url = parameters.get(
            "gdrive_img_url",
            "https://drive.google.com/uc?id=1VZ2WQrh4MRStFfWRSx0ezYJ_DlcaCGwI",
        )

        if not os.path.exists(archive_path):
            logger.info(f"Downloading image archive from Google Drive...")
            try:
                gdown.download(gdrive_url, archive_path, quiet=False)
                logger.info("Download complete.")
            except Exception as e:
                logger.warning(f"Failed to download from Google Drive: {e}")
                logger.info("Will try to fetch individual missing images instead.")

        # Extract archive if it exists
        if os.path.exists(archive_path):
            logger.info("Extracting image archive...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(data_dir)
            logger.info("Extraction complete.")

    # Fetch any missing images from HuggingFace
    base_url = (
        "https://huggingface.co/datasets/limjiayi/hateful_memes_expanded/resolve/main"
    )
    missing_count = 0

    for split_name, split_data in dataset.items():
        for sample in split_data:
            img_path = os.path.join(data_dir, sample["img"])
            if not os.path.exists(img_path):
                try:
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    response = requests.get(f"{base_url}/{sample['img']}")
                    response.raise_for_status()
                    with open(img_path, "wb") as f:
                        f.write(response.content)
                    missing_count += 1
                except Exception as e:
                    logger.warning(f"Failed to download {sample['img']}: {e}")

    if missing_count > 0:
        logger.info(f"Downloaded {missing_count} missing images from HuggingFace")


def load_and_validate_data(
    parameters: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Load and validate all data splits from HuggingFace."""
    logger.info("Loading and validating data...")

    datasets = load_hateful_memes_from_huggingface(parameters)

    # Validate each split
    for name, df in datasets.items():
        logger.info(f"  {name}: {len(df)} samples")
        if "label" in df.columns:
            hateful = (df["label"] == 1).sum()
            ratio = hateful / len(df) if len(df) > 0 else 0
            logger.info(f"    Hateful: {hateful} ({ratio:.1%})")

        # Check image paths exist
        if "img_path" in df.columns:
            existing = df["img_path"].apply(os.path.exists).sum()
            logger.info(f"    Images found: {existing}/{len(df)}")

    return datasets


def create_train_val_split(
    datasets: Dict[str, pd.DataFrame], parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Create train/val/test splits from HuggingFace dataset splits."""
    logger.info("Creating train/val/test splits...")

    use_dev_as_val = parameters.get("use_dev_as_val", True)
    random_seed = parameters.get("random_seed", 42)

    # Map HuggingFace split names to our naming
    # HuggingFace has: train, validation, test
    train_df = datasets.get("train", pd.DataFrame()).copy()

    # Use 'validation' split from HuggingFace as val, or split from train
    if use_dev_as_val and "validation" in datasets:
        val_df = datasets["validation"].copy()
        logger.info("  Using HuggingFace 'validation' split as validation set")
    else:
        val_ratio = parameters.get("val_split_ratio", 0.1)
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=train_df["label"],
        )
        logger.info(f"  Split {val_ratio:.0%} from train as validation")

    # Test set
    test_df = datasets.get("test", pd.DataFrame()).copy()

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    split_info = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_hateful_ratio": float((train_df["label"] == 1).mean())
        if len(train_df) > 0
        else 0,
        "val_hateful_ratio": float((val_df["label"] == 1).mean())
        if len(val_df) > 0
        else 0,
        "val_source": "huggingface_validation" if use_dev_as_val else "train_split",
    }

    logger.info(f"  Train: {split_info['train_size']} samples")
    logger.info(f"  Val: {split_info['val_size']} samples")
    logger.info(f"  Test: {split_info['test_size']} samples")

    return train_df, val_df, test_df, split_info


def preprocess_data(df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess text and validate image paths."""
    df = df.copy()
    max_length = parameters.get("max_text_length", 512)
    lowercase = parameters.get("lowercase", False)

    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip()
        if lowercase:
            text = text.lower()
        if len(text) > max_length * 4:
            text = text[: max_length * 4]
        return text

    df["text_clean"] = df["text"].apply(clean_text)

    # Validate image paths
    if "img_path" in df.columns:
        df["img_exists"] = df["img_path"].apply(os.path.exists)
        missing = (~df["img_exists"]).sum()
        if missing > 0:
            logger.warning(f"  {missing} images not found!")

    logger.info(
        f"  Text preprocessing: avg length = {df['text_clean'].str.len().mean():.1f} chars"
    )
    return df


def process_train_data(
    train_df: pd.DataFrame, parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Process training data with optional caption enrichment."""
    logger.info("Processing training data...")
    df = preprocess_data(train_df, parameters)

    # Add caption enrichment if enabled
    df = enrich_with_captions(df, parameters)

    return df


def process_val_data(val_df: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Process validation data with optional caption enrichment."""
    logger.info("Processing validation data...")
    df = preprocess_data(val_df, parameters)
    df = enrich_with_captions(df, parameters)
    return df


def process_test_data(
    test_df: pd.DataFrame, parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Process test data with optional caption enrichment."""
    logger.info("Processing test data...")
    df = preprocess_data(test_df, parameters)
    df = enrich_with_captions(df, parameters)
    return df


def compute_dataset_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute comprehensive dataset statistics."""
    logger.info("Computing dataset statistics...")

    def get_split_stats(df: pd.DataFrame) -> Dict[str, Any]:
        stats = {
            "count": len(df),
            "hateful_count": int((df["label"] == 1).sum())
            if "label" in df.columns
            else 0,
            "not_hateful_count": int((df["label"] == 0).sum())
            if "label" in df.columns
            else 0,
            "hateful_ratio": float((df["label"] == 1).mean())
            if "label" in df.columns
            else 0,
        }
        if "text_clean" in df.columns:
            stats["avg_text_length"] = float(df["text_clean"].str.len().mean())
        if "img_exists" in df.columns:
            stats["images_found"] = int(df["img_exists"].sum())
        return stats

    statistics = {
        "splits": {
            "train": get_split_stats(train_df),
            "val": get_split_stats(val_df),
            "test": get_split_stats(test_df),
        },
        "total_samples": len(train_df) + len(val_df) + len(test_df),
        "split_info": split_info,
    }

    logger.info(f"  Total samples: {statistics['total_samples']}")
    return statistics
