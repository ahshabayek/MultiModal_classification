import mlflow
import mlflow.pytorch
import torch
from transformers import (
    AutoModelForSequenceClassification,
    CLIPModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from typing import Dict, Any
import json

class MultiModalExperiment:
    """Base class for multi-modal experiments with MLflow tracking"""

    def __init__(self, model_name: str, experiment_name: str = "hateful-memes"):
        self.model_name = model_name
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str = None):
        """Start MLflow run with proper tags"""
        mlflow.start_run(run_name=run_name or f"{self.model_name}_training")
        mlflow.set_tags({
            "model_architecture": self.model_name,
            "dataset": "facebook-hateful-memes",
            "task": "binary-classification",
            "team": "Hateful3"
        })

    def log_model_params(self, params: Dict[str, Any]):
        """Log model hyperparameters"""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log evaluation metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(self, model, signature=None):
        """Log the trained model"""
        mlflow.pytorch.log_model(model, f"{self.model_name}_model", signature=signature)

    def end_run(self):
        """End MLflow run"""
        mlflow.end_run()

def train_vilbert(train_data, val_data, parameters):
    """Train ViLBERT model with MLflow tracking"""

    experiment = MultiModalExperiment("vilbert", parameters["mlflow"]["experiment_name"])
    experiment.start_run("vilbert_finetuning")

    # Log hyperparameters
    experiment.log_model_params({
        "model_architecture": "vilbert",
        "pretrained_model": parameters["vilbert"]["model_name"],
        "batch_size": parameters["training"]["batch_size"],
        "learning_rate": parameters["training"]["learning_rate"],
        "num_epochs": parameters["training"]["num_epochs"],
        "max_seq_length": parameters["vilbert"]["max_seq_length"],
        "use_faster_rcnn": parameters["vilbert"]["use_faster_rcnn"]
    })

    # Initialize model
    from transformers import ViltModel, ViltProcessor

    model = ViltModel.from_pretrained(parameters["vilbert"]["model_name"])
    processor = ViltProcessor.from_pretrained(parameters["vilbert"]["model_name"])

    # Training loop with metric logging
    best_val_auc = 0
    for epoch in range(parameters["training"]["num_epochs"]):
        # Training code here...
        train_loss = 0.5  # Placeholder

        # Validation
        val_metrics = evaluate_model(model, val_data, processor)

        # Log metrics
        experiment.log_metrics({
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"]
        }, step=epoch)

        # Save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            experiment.log_model(model)

    # Log final results
    mlflow.log_metric("best_val_auc", best_val_auc)

    experiment.end_run()
    return model

def train_visualbert(train_data, val_data, parameters):
    """Train VisualBERT model with MLflow tracking"""

    experiment = MultiModalExperiment("visualbert", parameters["mlflow"]["experiment_name"])
    experiment.start_run("visualbert_finetuning")

    # Log hyperparameters
    experiment.log_model_params({
        "model_architecture": "visualbert",
        "pretrained_model": parameters["visualbert"]["model_name"],
        "batch_size": parameters["training"]["batch_size"],
        "learning_rate": parameters["training"]["learning_rate"],
        "num_epochs": parameters["training"]["num_epochs"],
        "max_seq_length": parameters["visualbert"]["max_seq_length"],
        "visual_embedding_dim": parameters["visualbert"]["visual_embedding_dim"]
    })

    # Model initialization and training similar to ViLBERT
    # ... training code ...

    experiment.end_run()
    return model

def train_clip(train_data, val_data, parameters):
    """Train CLIP model with MLflow tracking"""

    experiment = MultiModalExperiment("clip", parameters["mlflow"]["experiment_name"])
    experiment.start_run("clip_finetuning")

    # Log hyperparameters
    experiment.log_model_params({
        "model_architecture": "clip",
        "pretrained_model": parameters["clip"]["model_name"],
        "batch_size": parameters["training"]["batch_size"],
        "learning_rate": parameters["training"]["learning_rate"],
        "num_epochs": parameters["training"]["num_epochs"],
        "freeze_vision_model": parameters["clip"]["freeze_vision_model"],
        "freeze_text_model": parameters["clip"]["freeze_text_model"]
    })

    # Initialize CLIP
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(parameters["clip"]["model_name"])
    processor = CLIPProcessor.from_pretrained(parameters["clip"]["model_name"])

    # Freeze layers if specified
    if parameters["clip"]["freeze_vision_model"]:
        for param in model.vision_model.parameters():
            param.requires_grad = False

    if parameters["clip"]["freeze_text_model"]:
        for param in model.text_model.parameters():
            param.requires_grad = False

    # Training loop with contrastive loss
    # ... training code ...

    experiment.end_run()
    return model

def evaluate_model(model, test_data, processor):
    """Common evaluation function for all models"""
    # Placeholder for evaluation logic
    predictions = []
    true_labels = []

    # Get predictions
    # ... evaluation code ...

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )
    auc = roc_auc_score(true_labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

def compare_models():
    """Compare all three models using MLflow"""

    import mlflow
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get experiment
    experiment = mlflow.get_experiment_by_name("hateful-memes-comparison")

    # Get all runs
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.val_auc DESC"]
    )

    # Create comparison DataFrame
    comparison_df = runs[['tags.model_architecture', 'metrics.val_accuracy',
                          'metrics.val_precision', 'metrics.val_recall',
                          'metrics.val_f1', 'metrics.val_auc']]

    # Log comparison as artifact
    with mlflow.start_run(run_name="model_comparison"):
        # Save comparison table
        comparison_df.to_csv("model_comparison.csv")
        mlflow.log_artifact("model_comparison.csv")

        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_auc']

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            data = runs.groupby('tags.model_architecture')[f'metrics.{metric}'].mean()
            data.plot(kind='bar', ax=ax)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)

        plt.tight_layout()
        plt.savefig('model_comparison.png')
        mlflow.log_artifact('model_comparison.png')

        # Log best model info
        best_run = runs.iloc[0]
        mlflow.log_metric("best_auc", best_run['metrics.val_auc'])
        mlflow.log_param("best_model", best_run['tags.model_architecture'])

    return comparison_df
