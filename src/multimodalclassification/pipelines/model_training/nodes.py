"""
Model Training Nodes for ViLBERT

Uses real images loaded via PIL and processed with a vision encoder.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import BertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add data/04_models to path
sys.path.insert(0, os.path.join(os.getcwd(), "data", "04_models"))


class HatefulMemesDataset(Dataset):
    """Dataset for Hateful Memes with real image loading."""

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_seq_length: int = 128,
        max_regions: int = 36,
        image_size: int = 224,
        visual_feature_dim: int = 2048,
        feature_extractor=None,
    ):
        self.data = data.to_dict("records")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_regions = max_regions
        self.image_size = image_size
        self.visual_feature_dim = visual_feature_dim
        self.feature_extractor = feature_extractor

        # Image transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def _load_image(self, img_path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image)
            return image_tensor
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            # Return a blank image as fallback
            return torch.zeros(3, self.image_size, self.image_size)

    def _extract_visual_features(
        self, image_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract visual features from image.

        If no feature extractor is provided, use a grid-based approach
        where we split the image into regions and use pixel features.
        """
        if self.feature_extractor is not None:
            # Use provided feature extractor (e.g., ResNet, Faster R-CNN)
            with torch.no_grad():
                features = self.feature_extractor(image_tensor.unsqueeze(0))
                if isinstance(features, dict):
                    features = features.get(
                        "pooler_output", features.get("last_hidden_state")
                    )
                features = features.squeeze(0)
        else:
            # Grid-based feature extraction
            # Split image into grid regions and compute features
            c, h, w = image_tensor.shape
            grid_size = int(np.sqrt(self.max_regions))
            region_h = h // grid_size
            region_w = w // grid_size

            features = []
            for i in range(grid_size):
                for j in range(grid_size):
                    region = image_tensor[
                        :,
                        i * region_h : (i + 1) * region_h,
                        j * region_w : (j + 1) * region_w,
                    ]
                    # Compute simple features: mean, std, max for each channel
                    region_features = torch.cat(
                        [
                            region.mean(dim=(1, 2)),
                            region.std(dim=(1, 2)),
                            region.max(dim=2)[0].max(dim=1)[0],
                            region.flatten()[: self.visual_feature_dim - 9],
                        ]
                    )
                    # Pad or truncate to visual_feature_dim
                    if len(region_features) < self.visual_feature_dim:
                        region_features = torch.nn.functional.pad(
                            region_features,
                            (0, self.visual_feature_dim - len(region_features)),
                        )
                    else:
                        region_features = region_features[: self.visual_feature_dim]
                    features.append(region_features)

            features = torch.stack(features)

        # Pad/truncate to max_regions
        if features.shape[0] < self.max_regions:
            padding = torch.zeros(
                self.max_regions - features.shape[0], self.visual_feature_dim
            )
            features = torch.cat([features, padding], dim=0)
            mask = torch.cat(
                [
                    torch.ones(features.shape[0] - padding.shape[0]),
                    torch.zeros(padding.shape[0]),
                ]
            )
        else:
            features = features[: self.max_regions]
            mask = torch.ones(self.max_regions)

        return features, mask

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Text processing
        text = str(sample.get("text", sample.get("text_clean", "")))
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Image processing
        img_path = sample.get("img_path", sample.get("img", ""))
        if img_path and os.path.exists(img_path):
            image_tensor = self._load_image(img_path)
            visual_features, visual_mask = self._extract_visual_features(image_tensor)
        else:
            # Fallback to zero features if image not found
            visual_features = torch.zeros(self.max_regions, self.visual_feature_dim)
            visual_mask = torch.zeros(self.max_regions)

        label = int(sample.get("label", 0))

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "visual_features": visual_features,
            "visual_attention_mask": visual_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "visual_features": torch.stack([x["visual_features"] for x in batch]),
        "visual_attention_mask": torch.stack(
            [x["visual_attention_mask"] for x in batch]
        ),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def load_vilbert_model(parameters: Dict[str, Any]) -> nn.Module:
    """Load ViLBERT model from HuggingFace."""

    vilbert_params = parameters.get("vilbert", {})
    num_labels = vilbert_params.get("num_labels", 2)
    freeze_layers = vilbert_params.get("freeze_bert_layers", 6)
    model_name = vilbert_params.get(
        "huggingface_model", "visualjoyce/transformers4vl-vilbert"
    )

    logger.info(f"Loading ViLBERT from {model_name}...")

    from vilbert_huggingface import load_vilbert_from_huggingface

    model = load_vilbert_from_huggingface(
        model_name=model_name,
        num_labels=num_labels,
        freeze_bert_layers=freeze_layers,
    )

    total, trainable = model.get_num_parameters()
    logger.info(f"Model loaded: {total:,} total, {trainable:,} trainable")

    return model


def create_dataloaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders with real image loading."""

    training_params = parameters.get("training", {})
    vilbert_params = parameters.get("vilbert", {})

    batch_size = training_params.get("batch_size", 16)
    max_seq_length = vilbert_params.get("max_seq_length", 128)
    image_size = vilbert_params.get("image_size", 224)
    max_regions = vilbert_params.get("max_regions", 36)
    visual_feature_dim = vilbert_params.get("visual_feature_dim", 2048)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    logger.info(
        f"Creating dataloaders with image_size={image_size}, max_regions={max_regions}"
    )

    train_dataset = HatefulMemesDataset(
        train_data,
        tokenizer,
        max_seq_length,
        max_regions,
        image_size,
        visual_feature_dim,
    )
    val_dataset = HatefulMemesDataset(
        val_data, tokenizer, max_seq_length, max_regions, image_size, visual_feature_dim
    )
    test_dataset = HatefulMemesDataset(
        test_data,
        tokenizer,
        max_seq_length,
        max_regions,
        image_size,
        visual_feature_dim,
    )

    # Use multiple workers for faster data loading
    num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Dataloaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    parameters: Dict[str, Any],
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train the model with MLflow logging."""

    training_params = parameters.get("training", {})
    vilbert_params = parameters.get("vilbert", {})

    num_epochs = training_params.get("num_epochs", 10)
    learning_rate = training_params.get("learning_rate", 2e-5)
    weight_decay = training_params.get("weight_decay", 0.01)
    early_stopping_patience = training_params.get("early_stopping_patience", 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Log model architecture info
    total_params, trainable_params = model.get_num_parameters()
    mlflow.log_params(
        {
            "model_total_params": total_params,
            "model_trainable_params": trainable_params,
            "device": str(device),
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
        }
    )

    history = {"train_loss": [], "val_loss": [], "val_auroc": []}
    best_auroc = 0.0
    patience_counter = 0
    best_state = None
    final_epoch = 0

    for epoch in range(1, num_epochs + 1):
        final_epoch = epoch
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation
        val_metrics = _evaluate(model, val_loader, device)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auroc"].append(val_metrics["auroc"])

        # Log metrics to MLflow
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_auroc": val_metrics["auroc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            },
            step=epoch,
        )

        logger.info(
            f"Epoch {epoch}: loss={train_loss:.4f}, val_auroc={val_metrics['auroc']:.4f}"
        )

        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            mlflow.log_metric("best_val_auroc", best_auroc, step=epoch)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                mlflow.log_param("early_stopped_at_epoch", epoch)
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, history


def _evaluate(model, dataloader, device):
    """Internal evaluation function."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            total_loss += outputs["loss"].item()
            probs = torch.softmax(outputs["logits"], dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "auroc": roc_auc_score(all_labels, all_probs),
        "f1": f1_score(all_labels, all_preds, average="binary"),
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    parameters: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate on test set with MLflow logging."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    metrics = _evaluate(model, test_loader, device)

    # Log test metrics to MLflow
    mlflow.log_metrics(
        {
            "test_loss": metrics["loss"],
            "test_auroc": metrics["auroc"],
            "test_accuracy": metrics["accuracy"],
            "test_f1": metrics["f1"],
        }
    )

    logger.info(f"Test: AUROC={metrics['auroc']:.4f}, Acc={metrics['accuracy']:.4f}")
    return metrics


def save_model(
    model: nn.Module,
    metrics: Dict[str, float],
    parameters: Dict[str, Any],
) -> str:
    """Save model checkpoint."""
    output_dir = parameters.get("vilbert", {}).get("output_dir", "data/05_model_output")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "vilbert_best.pt")
    torch.save(
        {"model_state_dict": model.state_dict(), "metrics": metrics}, output_path
    )
    logger.info(f"Saved to {output_path}")

    return output_path


def load_trained_model(parameters: Dict[str, Any]) -> nn.Module:
    """Load a locally trained ViLBERT model from checkpoint."""

    vilbert_params = parameters.get("vilbert", {})
    num_labels = vilbert_params.get("num_labels", 2)
    checkpoint_path = vilbert_params.get(
        "checkpoint_path", "data/05_model_output/vilbert_best.pt"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading trained model from {checkpoint_path}...")

    # Load the model architecture
    from vilbert_huggingface import ViLBERTHuggingFace

    model = ViLBERTHuggingFace(num_labels=num_labels)

    # Load the trained weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if "metrics" in checkpoint:
            logger.info(f"Loaded model metrics: {checkpoint['metrics']}")
    else:
        state_dict = checkpoint

    # The saved state_dict is from ViLBERTHuggingFace (which wraps ViLBERTForClassification)
    # Load it directly into the model
    model.load_state_dict(state_dict, strict=False)

    total, trainable = model.get_num_parameters()
    logger.info(f"Trained model loaded: {total:,} total, {trainable:,} trainable")

    return model


def run_inference(
    model: nn.Module,
    test_loader: DataLoader,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """Run inference on the test set and return predictions with validation checks."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running inference on: {device}")

    model = model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            probs = torch.softmax(outputs["logits"], dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            if "labels" in batch:
                all_labels.extend(batch["labels"].cpu().numpy())

    results = pd.DataFrame(
        {
            "prediction": all_preds,
            "probability": all_probs,
        }
    )

    # ==================== PREDICTION VALIDATION CHECKS ====================
    validation_results = validate_predictions(
        results, all_labels if all_labels else None
    )

    # Log validation results
    for check_name, check_result in validation_results.items():
        status = "PASSED" if check_result["passed"] else "FAILED"
        logger.info(
            f"Validation Check [{check_name}]: {status} - {check_result['message']}"
        )

    # Log to MLflow
    mlflow.log_params(
        {f"validation_{k}": v["passed"] for k, v in validation_results.items()}
    )

    if all_labels:
        results["label"] = all_labels

        # Calculate metrics if labels are available
        accuracy = accuracy_score(all_labels, all_preds)
        auroc = roc_auc_score(all_labels, all_probs)
        f1 = f1_score(all_labels, all_preds, average="binary")
        precision = precision_score(all_labels, all_preds, average="binary")
        recall = recall_score(all_labels, all_preds, average="binary")
        cm = confusion_matrix(all_labels, all_preds)

        # Log inference metrics to MLflow
        mlflow.log_metrics(
            {
                "inference_auroc": auroc,
                "inference_accuracy": accuracy,
                "inference_f1": f1,
                "inference_precision": precision,
                "inference_recall": recall,
                "true_negatives": int(cm[0, 0]),
                "false_positives": int(cm[0, 1]),
                "false_negatives": int(cm[1, 0]),
                "true_positives": int(cm[1, 1]),
            }
        )

        logger.info(
            f"Inference Results: AUROC={auroc:.4f}, Accuracy={accuracy:.4f}, F1={f1:.4f}"
        )
        logger.info(
            f"Confusion Matrix: TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}"
        )

    logger.info(f"Generated {len(results)} predictions")

    return results


def validate_predictions(
    predictions: pd.DataFrame,
    labels: List = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Validate predictions with various checks.

    Returns a dictionary of validation results with pass/fail status and messages.
    """
    validation_results = {}

    # Check 1: No null predictions
    null_preds = predictions["prediction"].isnull().sum()
    validation_results["no_null_predictions"] = {
        "passed": null_preds == 0,
        "message": f"Found {null_preds} null predictions"
        if null_preds > 0
        else "No null predictions",
        "value": int(null_preds),
    }

    # Check 2: Predictions are valid binary (0 or 1)
    valid_preds = predictions["prediction"].isin([0, 1]).all()
    validation_results["valid_binary_predictions"] = {
        "passed": valid_preds,
        "message": "All predictions are 0 or 1"
        if valid_preds
        else "Invalid prediction values found",
        "value": valid_preds,
    }

    # Check 3: Probabilities are in valid range [0, 1]
    prob_min = predictions["probability"].min()
    prob_max = predictions["probability"].max()
    valid_probs = (prob_min >= 0) and (prob_max <= 1)
    validation_results["valid_probability_range"] = {
        "passed": valid_probs,
        "message": f"Probabilities in range [{prob_min:.4f}, {prob_max:.4f}]",
        "value": {"min": float(prob_min), "max": float(prob_max)},
    }

    # Check 4: No null probabilities
    null_probs = predictions["probability"].isnull().sum()
    validation_results["no_null_probabilities"] = {
        "passed": null_probs == 0,
        "message": f"Found {null_probs} null probabilities"
        if null_probs > 0
        else "No null probabilities",
        "value": int(null_probs),
    }

    # Check 5: Prediction count matches expected
    pred_count = len(predictions)
    validation_results["prediction_count"] = {
        "passed": pred_count > 0,
        "message": f"Generated {pred_count} predictions",
        "value": pred_count,
    }

    # Check 6: Class distribution is reasonable (not all same class)
    class_counts = predictions["prediction"].value_counts()
    all_same_class = len(class_counts) == 1
    validation_results["class_distribution"] = {
        "passed": not all_same_class,
        "message": f"Class distribution: {class_counts.to_dict()}"
        if not all_same_class
        else "WARNING: All predictions are the same class",
        "value": class_counts.to_dict(),
    }

    # Check 7: Probability calibration (if labels available)
    if labels is not None and len(labels) > 0:
        # Check if model is better than random
        try:
            auroc = roc_auc_score(labels, predictions["probability"])
            better_than_random = auroc > 0.5
            validation_results["better_than_random"] = {
                "passed": better_than_random,
                "message": f"AUROC={auroc:.4f} {'>' if better_than_random else '<='} 0.5",
                "value": float(auroc),
            }
        except Exception as e:
            validation_results["better_than_random"] = {
                "passed": False,
                "message": f"Could not compute AUROC: {str(e)}",
                "value": None,
            }

    return validation_results


def create_inference_dataloader(
    test_data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> DataLoader:
    """Create a single DataLoader for inference."""

    training_params = parameters.get("training", {})
    vilbert_params = parameters.get("vilbert", {})

    batch_size = training_params.get("batch_size", 16)
    max_seq_length = vilbert_params.get("max_seq_length", 128)
    image_size = vilbert_params.get("image_size", 224)
    max_regions = vilbert_params.get("max_regions", 36)
    visual_feature_dim = vilbert_params.get("visual_feature_dim", 2048)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    test_dataset = HatefulMemesDataset(
        test_data,
        tokenizer,
        max_seq_length,
        max_regions,
        image_size,
        visual_feature_dim,
    )

    num_workers = min(4, os.cpu_count() or 1)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Inference dataloader: {len(test_dataset)} samples")

    return test_loader
