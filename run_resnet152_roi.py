#!/usr/bin/env python3
"""Run ResNet-152 ROI training directly without kedro CLI."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import directly from the nodes module to avoid kedro imports
import importlib.util

# Load nodes module directly
nodes_path = os.path.join(
    os.path.dirname(__file__),
    "src/multimodalclassification/pipelines/model_training/nodes.py",
)


# We need to mock kedro before importing
class MockPipeline:
    pass


class MockNode:
    pass


# Create a mock kedro module
import types

kedro_mock = types.ModuleType("kedro")
kedro_mock.pipeline = types.ModuleType("kedro.pipeline")
kedro_mock.pipeline.Pipeline = MockPipeline
kedro_mock.pipeline.node = MockNode
kedro_mock.pipeline.pipeline = lambda x: x
sys.modules["kedro"] = kedro_mock
sys.modules["kedro.pipeline"] = kedro_mock.pipeline

import logging

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import BertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Now import project modules
from multimodalclassification.models.feature_extractors import get_feature_extractor
from multimodalclassification.models.vilbert_facebook_arch import (
    ViLBERTForClassification,
)


class SimpleDataset(Dataset):
    """Simple dataset for training."""

    def __init__(
        self, data, tokenizer, feature_extractor, max_seq_length=128, max_regions=36
    ):
        self.data = data.to_dict("records") if isinstance(data, pd.DataFrame) else data
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_seq_length = max_seq_length
        self.max_regions = max_regions
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Text processing
        text = item.get("text", "")
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Image processing
        img_path = item.get("img", item.get("image", ""))
        try:
            image = Image.open(img_path).convert("RGB")
            features, spatial = self.feature_extractor.extract_features(image)
        except Exception as e:
            features = torch.zeros(self.max_regions, 2048)
            spatial = torch.zeros(self.max_regions, 5)

        label = item.get("label", 0)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "visual_features": features,
            "spatial_features": spatial,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_data(parameters):
    """Load data from CSV files."""
    data_dir = parameters.get("data_processing", {}).get(
        "data_dir", "data/01_raw/hateful_memes"
    )

    # Try to load from intermediate directory first (already processed)
    intermediate_dir = "data/02_intermediate"
    train_path = os.path.join(intermediate_dir, "processed_train.csv")
    val_path = os.path.join(intermediate_dir, "processed_val.csv")
    test_path = os.path.join(intermediate_dir, "processed_test.csv")

    if os.path.exists(train_path):
        logger.info("Loading from intermediate directory...")
        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)
        test_data = pd.read_csv(test_path)
    else:
        # Load from raw
        logger.info("Loading from raw directory...")
        train_jsonl = os.path.join(data_dir, "train.jsonl")
        val_jsonl = os.path.join(data_dir, "dev_seen.jsonl")
        test_jsonl = os.path.join(data_dir, "test_seen.jsonl")

        train_data = pd.read_json(train_jsonl, lines=True)
        val_data = pd.read_json(val_jsonl, lines=True)
        test_data = pd.read_json(test_jsonl, lines=True)

        # Fix image paths
        for df in [train_data, val_data, test_data]:
            df["img"] = df["img"].apply(
                lambda x: os.path.join(data_dir, x) if not x.startswith("/") else x
            )

    logger.info(
        f"Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples"
    )
    return train_data, val_data, test_data


def create_model(parameters):
    """Create ViLBERT model with Facebook weights."""
    vilbert_params = parameters.get("vilbert_resnet152_roi", {})
    weights_path = vilbert_params.get(
        "facebook_weights_path", "weights/vilbert_pretrained_cc.bin"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ViLBERTForClassification(num_labels=2)

    if os.path.exists(weights_path):
        logger.info(f"Loading Facebook weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    return model, device


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", disable=True):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        visual_features = batch["visual_features"].to(device)
        spatial_features = batch["spatial_features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            spatial_features=spatial_features,
            labels=labels,
        )

        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            visual_features = batch["visual_features"].to(device)
            spatial_features = batch["spatial_features"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=visual_features,
                spatial_features=spatial_features,
            )

            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)

    return {"accuracy": accuracy, "auroc": auroc, "f1": f1}


def main():
    # Load parameters
    with open("conf/base/parameters.yml", "r") as f:
        parameters = yaml.safe_load(f)

    print("=" * 60)
    print("ResNet-152 ROI Training Pipeline")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Get training params
    training_params = parameters.get("training_resnet152_roi", {})
    vilbert_params = parameters.get("vilbert_resnet152_roi", {})

    batch_size = training_params.get("batch_size", 32)
    num_epochs = training_params.get("num_epochs", 20)
    learning_rate = training_params.get("learning_rate", 1e-5)
    patience = training_params.get("early_stopping_patience", 5)

    # Step 1: Load data
    print("\n[1/5] Loading data...")
    train_data, val_data, test_data = load_data(parameters)

    # Step 2: Create feature extractor
    print("\n[2/5] Creating ResNet-152 ROI feature extractor...")
    feature_extractor = get_feature_extractor(
        "resnet152_roi",
        output_dim=2048,
        num_regions=vilbert_params.get("max_regions", 36),
        roi_size=vilbert_params.get("roi_size", 14),
        use_multi_scale=vilbert_params.get("use_multi_scale", True),
        device=device,
    )

    # Step 3: Create datasets and dataloaders
    print("\n[3/5] Creating dataloaders...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = SimpleDataset(train_data, tokenizer, feature_extractor)
    val_dataset = SimpleDataset(val_data, tokenizer, feature_extractor)
    test_dataset = SimpleDataset(test_data, tokenizer, feature_extractor)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Step 4: Create model
    print("\n[4/5] Loading ViLBERT model...")
    model, device = create_model(parameters)

    # Step 5: Train
    print(f"\n[5/5] Training for {num_epochs} epochs...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_auroc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_auroc={val_metrics['auroc']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)

    print("\nTest Results:")
    print("=" * 60)
    print(f"  AUROC: {test_metrics['auroc']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Best Val AUROC: {best_val_auroc:.4f}")

    # Save model
    output_dir = vilbert_params.get("output_dir", "data/05_model_output/resnet152_roi")
    os.makedirs(output_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "test_metrics": test_metrics,
            "best_val_auroc": best_val_auroc,
        },
        os.path.join(output_dir, "vilbert_resnet152_roi_best.pt"),
    )

    # Save metrics
    import json

    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nModel saved to {output_dir}")
    print("\nTraining complete!")

    return test_metrics


if __name__ == "__main__":
    main()
