"""Model training nodes for ViLBERT."""

import logging
import os
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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import BertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_FEATURE_EXTRACTOR = None


def get_feature_extractor(
    extractor_type: str = "resnet",
    output_dim: int = 2048,
    num_regions: int = 36,
    device: str = None,
    **kwargs,
):
    """Get or create a shared feature extractor (singleton pattern)."""
    global _FEATURE_EXTRACTOR

    if _FEATURE_EXTRACTOR is None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            from multimodalclassification.models.feature_extractors import (
                get_feature_extractor as get_extractor,
            )

            _FEATURE_EXTRACTOR = get_extractor(
                extractor_type,
                output_dim=output_dim,
                num_regions=num_regions,
                device=device,
                **kwargs,
            )
        except ImportError:
            from .visual_features import (
                CLIPVisualFeatureExtractor,
                FasterRCNNFeatureExtractor,
                ResNetFeatureExtractor,
            )

            extractors = {
                "clip": CLIPVisualFeatureExtractor,
                "fasterrcnn": FasterRCNNFeatureExtractor,
                "resnet": ResNetFeatureExtractor,
            }
            cls = extractors.get(extractor_type, ResNetFeatureExtractor)
            _FEATURE_EXTRACTOR = cls(
                output_dim=output_dim, num_regions=num_regions, device=device, **kwargs
            )

        logger.info(f"Initialized {extractor_type} feature extractor")

    return _FEATURE_EXTRACTOR


class HatefulMemesDataset(Dataset):
    """Dataset for Hateful Memes with visual feature extraction."""

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_seq_length: int = 128,
        max_regions: int = 36,
        image_size: int = 224,
        visual_feature_dim: int = 2048,
        feature_extractor=None,
        use_cached_features: bool = True,
    ):
        self.data = data.to_dict("records")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_regions = max_regions
        self.image_size = image_size
        self.visual_feature_dim = visual_feature_dim
        self.feature_extractor = feature_extractor
        self.use_cached_features = use_cached_features
        self.feature_cache = {}
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

    def _load_image(self, img_path: str) -> Image.Image:
        try:
            return Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            return Image.new("RGB", (self.image_size, self.image_size))

    def _extract_features(self, img_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_cached_features and img_path in self.feature_cache:
            return self.feature_cache[img_path]

        image = self._load_image(img_path)

        if self.feature_extractor is not None:
            try:
                features, spatial = self.feature_extractor.extract_features(image)
                features, spatial = features.cpu(), spatial.cpu()
            except Exception as e:
                logger.warning(f"Feature extraction failed for {img_path}: {e}")
                features, spatial = self._fallback_features()
        else:
            features, spatial = self._fallback_features()

        if self.use_cached_features:
            self.feature_cache[img_path] = (features, spatial)

        return features, spatial

    def _fallback_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.zeros(self.max_regions, self.visual_feature_dim)
        spatial = self._generate_grid_spatial()
        return features, spatial

    def _generate_grid_spatial(self) -> torch.Tensor:
        grid_size = int(self.max_regions**0.5)
        spatial = []
        for i in range(grid_size):
            for j in range(grid_size):
                x1, y1 = j / grid_size, i / grid_size
                x2, y2 = (j + 1) / grid_size, (i + 1) / grid_size
                spatial.append([x1, y1, x2, y2, (x2 - x1) * (y2 - y1)])
        return torch.tensor(spatial)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = str(sample.get("text", sample.get("text_clean", "")))
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        img_path = sample.get("img_path", sample.get("img", ""))
        if img_path and os.path.exists(img_path):
            visual_features, spatial_locations = self._extract_features(img_path)
        else:
            visual_features, spatial_locations = self._fallback_features()

        # Pad or truncate to max_regions
        if visual_features.shape[0] < self.max_regions:
            pad_size = self.max_regions - visual_features.shape[0]
            visual_features = torch.cat(
                [visual_features, torch.zeros(pad_size, self.visual_feature_dim)], dim=0
            )
            spatial_locations = torch.cat(
                [spatial_locations, torch.zeros(pad_size, 5)], dim=0
            )

        visual_features = visual_features[: self.max_regions]
        spatial_locations = spatial_locations[: self.max_regions]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "visual_features": visual_features,
            "visual_attention_mask": torch.ones(self.max_regions),
            "spatial_locations": spatial_locations,
            "labels": torch.tensor(int(sample.get("label", 0)), dtype=torch.long),
        }


def collate_fn(batch):
    return {k: torch.stack([x[k] for x in batch]) for k in batch[0].keys()}


# Model loading functions


def _load_facebook_model(parameters: Dict[str, Any], config_key: str) -> nn.Module:
    """Helper to load ViLBERT with Facebook weights."""
    vilbert_params = parameters.get(config_key, parameters.get("vilbert", {}))
    num_labels = vilbert_params.get("num_labels", 2)
    weights_path = vilbert_params.get(
        "facebook_weights_path", "weights/vilbert_pretrained_cc.bin"
    )

    logger.info(f"Loading ViLBERT from {weights_path}...")

    from multimodalclassification.models import (
        ViLBERTFacebookArch,
        get_facebook_vilbert_config,
        load_facebook_weights,
    )

    config = get_facebook_vilbert_config()
    model = ViLBERTFacebookArch(config, num_labels=num_labels)

    if os.path.exists(weights_path):
        loaded = load_facebook_weights(model, weights_path)
        logger.info(f"Loaded {loaded} weight tensors from Facebook checkpoint")
    else:
        logger.warning(f"Facebook weights not found at {weights_path}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total:,} total, {trainable:,} trainable params")

    return model


def load_vilbert_model(parameters: Dict[str, Any]) -> nn.Module:
    """Load ViLBERT model from HuggingFace."""
    vilbert_params = parameters.get("vilbert", {})
    num_labels = vilbert_params.get("num_labels", 2)
    freeze_layers = vilbert_params.get("freeze_bert_layers", 0)
    model_name = vilbert_params.get(
        "huggingface_model", "visualjoyce/transformers4vl-vilbert"
    )

    logger.info(f"Loading ViLBERT from {model_name}...")

    from multimodalclassification.models import load_vilbert_from_huggingface

    model = load_vilbert_from_huggingface(
        model_name=model_name, num_labels=num_labels, freeze_bert_layers=freeze_layers
    )

    total, trainable = model.get_num_parameters()
    logger.info(f"Model: {total:,} total, {trainable:,} trainable params")
    return model


def load_vilbert_facebook(parameters: Dict[str, Any]) -> nn.Module:
    return _load_facebook_model(parameters, "vilbert_frcnn")


def load_vilbert_vg(parameters: Dict[str, Any]) -> nn.Module:
    return _load_facebook_model(parameters, "vilbert_vg")


def load_vilbert_lmdb(parameters: Dict[str, Any]) -> nn.Module:
    return _load_facebook_model(parameters, "vilbert_lmdb")


def load_vilbert_x152(parameters: Dict[str, Any]) -> nn.Module:
    return _load_facebook_model(parameters, "vilbert_x152")


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

    from multimodalclassification.models import ViLBERTHuggingFace

    model = ViLBERTHuggingFace(num_labels=num_labels)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    total, trainable = model.get_num_parameters()
    logger.info(f"Loaded: {total:,} total, {trainable:,} trainable params")
    return model


# DataLoader creation functions


def _create_dataloaders_with_extractor(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    parameters: Dict[str, Any],
    training_key: str,
    vilbert_key: str,
    extractor_type: str,
    **extractor_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Helper to create dataloaders with a specific feature extractor."""
    global _FEATURE_EXTRACTOR
    _FEATURE_EXTRACTOR = None

    training_params = parameters.get(training_key, parameters.get("training", {}))
    vilbert_params = parameters.get(vilbert_key, parameters.get("vilbert", {}))

    batch_size = training_params.get("batch_size", 32)
    max_seq_length = vilbert_params.get("max_seq_length", 128)
    max_regions = vilbert_params.get("max_regions", 36)
    visual_feature_dim = vilbert_params.get("visual_feature_dim", 2048)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor = get_feature_extractor(
        extractor_type=extractor_type,
        output_dim=visual_feature_dim,
        num_regions=max_regions,
        device=device,
        **extractor_kwargs,
    )

    logger.info(f"Creating {extractor_type} dataloaders, batch_size={batch_size}")

    def make_dataset(data):
        return HatefulMemesDataset(
            data,
            tokenizer,
            max_seq_length,
            max_regions,
            visual_feature_dim=visual_feature_dim,
            feature_extractor=feature_extractor,
        )

    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

    train_dataset, val_dataset, test_dataset = (
        make_dataset(train_data),
        make_dataset(val_data),
        make_dataset(test_data),
    )
    logger.info(
        f"Datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return (
        make_loader(train_dataset, True),
        make_loader(val_dataset, False),
        make_loader(test_dataset, False),
    )


def create_dataloaders(train_data, val_data, test_data, parameters):
    vilbert_params = parameters.get("vilbert", {})
    return _create_dataloaders_with_extractor(
        train_data,
        val_data,
        test_data,
        parameters,
        "training",
        "vilbert",
        vilbert_params.get("feature_extractor", "resnet"),
    )


def create_dataloaders_frcnn(train_data, val_data, test_data, parameters):
    vilbert_params = parameters.get("vilbert_frcnn", {})
    return _create_dataloaders_with_extractor(
        train_data,
        val_data,
        test_data,
        parameters,
        "training_frcnn",
        "vilbert_frcnn",
        "fasterrcnn",
        confidence_threshold=vilbert_params.get("frcnn_confidence_threshold", 0.2),
    )


def create_dataloaders_vg(train_data, val_data, test_data, parameters):
    vilbert_params = parameters.get("vilbert_vg", {})
    return _create_dataloaders_with_extractor(
        train_data,
        val_data,
        test_data,
        parameters,
        "training_vg",
        "vilbert_vg",
        "fasterrcnn_vg",
        weights_path=vilbert_params.get(
            "vg_weights_path", "weights/faster_rcnn_res101_vg.pth"
        ),
        confidence_threshold=vilbert_params.get("frcnn_confidence_threshold", 0.2),
        nms_threshold=vilbert_params.get("nms_threshold", 0.3),
    )


def create_dataloaders_x152(train_data, val_data, test_data, parameters):
    vilbert_params = parameters.get("vilbert_x152", {})
    return _create_dataloaders_with_extractor(
        train_data,
        val_data,
        test_data,
        parameters,
        "training_x152",
        "vilbert_x152",
        "grid_x152",
        weights_path=vilbert_params.get("x152_weights_path", "weights/X-152pp.pth"),
        confidence_threshold=vilbert_params.get("confidence_threshold", 0.2),
        nms_threshold=vilbert_params.get("nms_threshold", 0.5),
        auto_download=vilbert_params.get("auto_download_weights", True),
    )


def create_dataloaders_precomputed(train_data, val_data, test_data, parameters):
    from multimodalclassification.pipelines.data_processing.precomputed_dataset import (
        create_precomputed_dataloaders,
    )

    training_params = parameters.get(
        "training_precomputed", parameters.get("training_vg", {})
    )
    vilbert_params = parameters.get(
        "vilbert_precomputed", parameters.get("vilbert_vg", {})
    )

    return create_precomputed_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        features_path=vilbert_params.get(
            "precomputed_features_path", "data/03_features/vg_features_100.h5"
        ),
        id_map_path=vilbert_params.get(
            "precomputed_id_map_path", "data/03_features/vg_features_100_id_map.npy"
        ),
        batch_size=training_params.get("batch_size", 32),
        max_seq_length=vilbert_params.get("max_seq_length", 128),
        num_regions=vilbert_params.get("max_regions", 100),
        visual_feature_dim=vilbert_params.get("visual_feature_dim", 2048),
        num_workers=0,
    )


def create_dataloaders_lmdb(train_data, val_data, test_data, parameters):
    from multimodalclassification.pipelines.data_processing.lmdb_dataset import (
        create_lmdb_dataloaders,
    )

    training_params = parameters.get("training_lmdb", parameters.get("training_vg", {}))
    vilbert_params = parameters.get("vilbert_lmdb", parameters.get("vilbert_vg", {}))

    logger.info(f"Creating LMDB dataloaders from {vilbert_params.get('lmdb_path')}")

    return create_lmdb_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        lmdb_path=vilbert_params.get(
            "lmdb_path", "data/03_features/mmf/detectron.lmdb"
        ),
        batch_size=training_params.get("batch_size", 32),
        max_seq_length=vilbert_params.get("max_seq_length", 128),
        num_regions=vilbert_params.get("max_regions", 100),
        visual_feature_dim=vilbert_params.get("visual_feature_dim", 2048),
        num_workers=0,
    )


def create_inference_dataloader(
    test_data: pd.DataFrame, parameters: Dict[str, Any]
) -> DataLoader:
    vilbert_params = parameters.get("vilbert", {})
    training_params = parameters.get("training", {})

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_extractor = get_feature_extractor(
        extractor_type=vilbert_params.get("feature_extractor", "resnet"),
        output_dim=vilbert_params.get("visual_feature_dim", 2048),
        num_regions=vilbert_params.get("max_regions", 36),
        device=device,
    )

    dataset = HatefulMemesDataset(
        test_data,
        tokenizer,
        vilbert_params.get("max_seq_length", 128),
        vilbert_params.get("max_regions", 36),
        visual_feature_dim=vilbert_params.get("visual_feature_dim", 2048),
        feature_extractor=feature_extractor,
    )

    logger.info(f"Inference dataloader: {len(dataset)} samples")
    return DataLoader(
        dataset,
        batch_size=training_params.get("batch_size", 32),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )


# Training functions


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    parameters: Dict[str, Any],
    training_config_key: str = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train the model."""
    if training_config_key and training_config_key in parameters:
        training_params = parameters[training_config_key]
    else:
        training_params = parameters.get("training", {})

    num_epochs = training_params.get("num_epochs", 20)
    learning_rate = training_params.get("learning_rate", 5e-5)
    weight_decay = training_params.get("weight_decay", 0.01)
    warmup_steps = training_params.get("warmup_steps", 2000)
    early_stopping_patience = training_params.get("early_stopping_patience", 5)
    gradient_clip = training_params.get("gradient_clip", 1.0)
    loss_type = training_params.get("loss_type", "focal")
    focal_alpha = training_params.get("focal_alpha", 0.35)
    focal_gamma = training_params.get("focal_gamma", 2.0)
    label_smoothing = training_params.get("label_smoothing", 0.1)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Training on {device}, lr={learning_rate}, epochs={num_epochs}")

    model = model.to(device)

    try:
        from .losses import get_loss_function
    except ImportError:
        from losses import get_loss_function

    loss_fn = get_loss_function(
        loss_type=loss_type,
        alpha=focal_alpha,
        gamma=focal_gamma,
        smoothing=label_smoothing,
    )
    use_custom_loss = loss_type != "ce"

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    total_params, trainable_params = model.get_num_parameters()
    mlflow.log_params(
        {
            "model_total_params": total_params,
            "model_trainable_params": trainable_params,
            "device": str(device),
            "train_samples": len(train_loader.dataset),
            "learning_rate": learning_rate,
            "loss_type": loss_type,
        }
    )

    history = {"train_loss": [], "val_loss": [], "val_auroc": []}
    best_auroc, patience_counter, best_state = 0.0, 0, None

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, num_batches = 0.0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(**batch)
            loss = (
                loss_fn(outputs["logits"], batch["labels"])
                if use_custom_loss
                else outputs["loss"]
            )

            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches
        history["train_loss"].append(train_loss)

        val_metrics = _evaluate(model, val_loader, device)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auroc"].append(val_metrics["auroc"])

        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_auroc": val_metrics["auroc"],
                "val_accuracy": val_metrics["accuracy"],
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
            logger.info(f"New best AUROC: {best_auroc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    mlflow.log_metric("final_best_auroc", best_auroc)
    return model, history


def train_model_vg(model, train_loader, val_loader, parameters):
    return train_model(model, train_loader, val_loader, parameters, "training_vg")


def train_model_frcnn(model, train_loader, val_loader, parameters):
    return train_model(model, train_loader, val_loader, parameters, "training_frcnn")


def train_model_lmdb(model, train_loader, val_loader, parameters):
    return train_model(model, train_loader, val_loader, parameters, "training_lmdb")


def train_model_x152(model, train_loader, val_loader, parameters):
    return train_model(model, train_loader, val_loader, parameters, "training_x152")


# Evaluation functions


def _evaluate(model, dataloader, device):
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
    model: nn.Module, test_loader: DataLoader, parameters: Dict[str, Any]
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    metrics = _evaluate(model, test_loader, device)

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
    model: nn.Module, metrics: Dict[str, float], parameters: Dict[str, Any]
) -> str:
    output_dir = parameters.get("vilbert", {}).get("output_dir", "data/05_model_output")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "vilbert_best.pt")
    torch.save(
        {"model_state_dict": model.state_dict(), "metrics": metrics}, output_path
    )
    logger.info(f"Saved to {output_path}")
    return output_path


def run_inference(
    model: nn.Module, test_loader: DataLoader, parameters: Dict[str, Any]
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds, all_probs, all_labels = [], [], []

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

    results = pd.DataFrame({"prediction": all_preds, "probability": all_probs})

    if all_labels:
        results["label"] = all_labels
        auroc = roc_auc_score(all_labels, all_probs)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")

        mlflow.log_metrics(
            {
                "inference_auroc": auroc,
                "inference_accuracy": accuracy,
                "inference_f1": f1,
            }
        )
        logger.info(
            f"Inference: AUROC={auroc:.4f}, Accuracy={accuracy:.4f}, F1={f1:.4f}"
        )

    logger.info(f"Generated {len(results)} predictions")
    return results
