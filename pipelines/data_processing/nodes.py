"""
Data Processing Nodes for Hateful Memes Classification

Location: src/multimodalclassification/pipelines/data_processing/nodes.py
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Any

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_and_validate_data(
    train_data: pd.DataFrame,
    dev_seen_data: pd.DataFrame,
    dev_unseen_data: pd.DataFrame,
    test_seen_data: pd.DataFrame,
    test_unseen_data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """
    Load and validate all data splits.

    Args:
        train_data: Training DataFrame from catalog
        dev_seen_data: Dev seen DataFrame
        dev_unseen_data: Dev unseen DataFrame
        test_seen_data: Test seen DataFrame
        test_unseen_data: Test unseen DataFrame
        parameters: Data processing parameters

    Returns:
        Dictionary with validated DataFrames
    """
    logger.info("Validating loaded data...")

    datasets = {
        "train": train_data,
        "dev_seen": dev_seen_data,
        "dev_unseen": dev_unseen_data,
        "test_seen": test_seen_data,
        "test_unseen": test_unseen_data
    }

    for name, df in datasets.items():
        logger.info(f"  {name}: {len(df)} samples")

        # Check label distribution
        if "label" in df.columns:
            hateful = (df["label"] == 1).sum()
            ratio = hateful / len(df) if len(df) > 0 else 0
            logger.info(f"    Hateful: {hateful} ({ratio:.1%})")

    return datasets


def create_train_val_split(
    datasets: Dict[str, pd.DataFrame],
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Create train/val/test splits.

    Args:
        datasets: Dictionary of DataFrames
        parameters: Data processing parameters

    Returns:
        Tuple of (train_df, val_df, test_seen_df, test_unseen_df, split_info)
    """
    logger.info("Creating train/val/test splits...")

    use_dev_as_val = parameters.get("use_dev_as_val", True)
    random_seed = parameters.get("random_seed", 42)

    train_df = datasets["train"].copy()
    test_seen_df = datasets["test_seen"].copy()
    test_unseen_df = datasets["test_unseen"].copy()

    if use_dev_as_val:
        val_df = datasets["dev_seen"].copy()
        logger.info("  Using dev_seen as validation set")
    else:
        val_ratio = parameters.get("val_split_ratio", 0.1)
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=train_df["label"]
        )
        logger.info(f"  Split training data with {val_ratio:.1%} validation")

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_seen_df = test_seen_df.reset_index(drop=True)
    test_unseen_df = test_unseen_df.reset_index(drop=True)

    split_info = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_seen_size": len(test_seen_df),
        "test_unseen_size": len(test_unseen_df),
        "train_hateful_ratio": float((train_df["label"] == 1).mean()),
        "val_hateful_ratio": float((val_df["label"] == 1).mean()),
        "val_source": "dev_seen" if use_dev_as_val else "train_split"
    }

    logger.info(f"  Train: {split_info['train_size']} samples")
    logger.info(f"  Val: {split_info['val_size']} samples")
    logger.info(f"  Test (seen): {split_info['test_seen_size']} samples")
    logger.info(f"  Test (unseen): {split_info['test_unseen_size']} samples")

    return train_df, val_df, test_seen_df, test_unseen_df, split_info


def preprocess_text(
    df: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Preprocess text data.
    """
    df = df.copy()

    max_length = parameters.get("max_text_length", 128)
    lowercase = parameters.get("lowercase", False)

    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip()
        if lowercase:
            text = text.lower()
        if len(text) > max_length * 4:
            text = text[:max_length * 4]
        return text

    df["text_clean"] = df["text"].apply(clean_text)

    avg_len = df["text_clean"].str.len().mean()
    logger.info(f"  Text preprocessing: avg length = {avg_len:.1f} chars")

    return df


def process_train_data(
    train_df: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Process training data."""
    logger.info("Processing training data...")
    return preprocess_text(train_df, parameters)


def process_val_data(
    val_df: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Process validation data."""
    logger.info("Processing validation data...")
    return preprocess_text(val_df, parameters)


def process_test_seen_data(
    test_seen_df: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Process test (seen) data."""
    logger.info("Processing test (seen) data...")
    return preprocess_text(test_seen_df, parameters)


def process_test_unseen_data(
    test_unseen_df: pd.DataFrame,
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Process test (unseen) data."""
    logger.info("Processing test (unseen) data...")
    return preprocess_text(test_unseen_df, parameters)


def compute_dataset_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_seen_df: pd.DataFrame,
    test_unseen_df: pd.DataFrame,
    split_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute comprehensive dataset statistics.
    """
    logger.info("Computing dataset statistics...")

    def get_split_stats(df: pd.DataFrame) -> Dict[str, Any]:
        stats = {
            "count": len(df),
            "hateful_count": int((df["label"] == 1).sum()),
            "not_hateful_count": int((df["label"] == 0).sum()),
            "hateful_ratio": float((df["label"] == 1).mean()),
        }
        if "text_clean" in df.columns:
            stats["avg_text_length"] = float(df["text_clean"].str.len().mean())
            stats["max_text_length"] = int(df["text_clean"].str.len().max())
            stats["min_text_length"] = int(df["text_clean"].str.len().min())
        return stats

    statistics = {
        "splits": {
            "train": get_split_stats(train_df),
            "val": get_split_stats(val_df),
            "test_seen": get_split_stats(test_seen_df),
            "test_unseen": get_split_stats(test_unseen_df),
        },
        "total_samples": len(train_df) + len(val_df) + len(test_seen_df) + len(test_unseen_df),
        "split_info": split_info
    }

    logger.info(f"  Total samples: {statistics['total_samples']}")

    return statistics
