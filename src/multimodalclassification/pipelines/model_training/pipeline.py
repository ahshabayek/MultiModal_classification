"""
ViLBERT Pipelines

Provides separate pipelines for different modes:
- vilbert_train: Full training pipeline (load pretrained -> train -> evaluate -> save)
- vilbert_validate: Validate a trained model on validation/test set
- vilbert_inference: Run inference with pretrained HuggingFace weights
- vilbert_inference_local: Run inference with locally trained weights

Usage:
    kedro run --pipeline=vilbert_train
    kedro run --pipeline=vilbert_validate
    kedro run --pipeline=vilbert_inference
    kedro run --pipeline=vilbert_inference_local

These pipelines use processed data from the data_processing pipeline, which loads
data from HuggingFace with real images.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_dataloaders,
    create_inference_dataloader,
    evaluate_model,
    load_trained_model,
    load_vilbert_model,
    run_inference,
    save_model,
    train_model,
)


def create_training_pipeline(**kwargs) -> Pipeline:
    """Create the full ViLBERT training pipeline.

    Flow: Load pretrained model -> Create dataloaders -> Train -> Evaluate -> Save

    Inputs from data_processing pipeline:
        - processed_train: Training data with images
        - processed_val: Validation data with images
        - processed_test: Test data with images
    """
    return pipeline(
        [
            node(
                func=create_dataloaders,
                inputs=[
                    "processed_train",
                    "processed_val",
                    "processed_test",
                    "parameters",
                ],
                outputs=["train_loader", "val_loader", "test_loader"],
                name="create_dataloaders",
            ),
            node(
                func=load_vilbert_model,
                inputs="parameters",
                outputs="vilbert_model",
                name="load_pretrained_model",
            ),
            node(
                func=train_model,
                inputs=["vilbert_model", "train_loader", "val_loader", "parameters"],
                outputs=["trained_model", "training_history"],
                name="train_model",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "test_loader", "parameters"],
                outputs="test_metrics",
                name="evaluate_model",
            ),
            node(
                func=save_model,
                inputs=["trained_model", "test_metrics", "parameters"],
                outputs="model_path",
                name="save_model",
            ),
        ]
    )


def create_validation_pipeline(**kwargs) -> Pipeline:
    """Create pipeline to validate a locally trained model.

    Flow: Load trained model -> Create test dataloader -> Evaluate
    """
    return pipeline(
        [
            node(
                func=create_inference_dataloader,
                inputs=["processed_test", "parameters"],
                outputs="test_loader",
                name="create_test_dataloader",
            ),
            node(
                func=load_trained_model,
                inputs="parameters",
                outputs="trained_model",
                name="load_trained_model",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "test_loader", "parameters"],
                outputs="validation_metrics",
                name="evaluate_trained_model",
            ),
        ]
    )


def create_inference_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for inference using pretrained HuggingFace weights.

    Flow: Load pretrained model from HuggingFace -> Create dataloader -> Run inference
    """
    return pipeline(
        [
            node(
                func=create_inference_dataloader,
                inputs=["processed_test", "parameters"],
                outputs="inference_loader",
                name="create_inference_dataloader",
            ),
            node(
                func=load_vilbert_model,
                inputs="parameters",
                outputs="pretrained_model",
                name="load_pretrained_model",
            ),
            node(
                func=run_inference,
                inputs=["pretrained_model", "inference_loader", "parameters"],
                outputs="predictions",
                name="run_inference",
            ),
        ]
    )


def create_inference_local_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for inference using locally trained weights.

    Flow: Load trained model from checkpoint -> Create dataloader -> Run inference
    """
    return pipeline(
        [
            node(
                func=create_inference_dataloader,
                inputs=["processed_test", "parameters"],
                outputs="inference_loader",
                name="create_inference_dataloader",
            ),
            node(
                func=load_trained_model,
                inputs="parameters",
                outputs="local_model",
                name="load_local_model",
            ),
            node(
                func=run_inference,
                inputs=["local_model", "inference_loader", "parameters"],
                outputs="predictions",
                name="run_inference",
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Create the default pipeline (training).

    This is kept for backward compatibility.
    """
    return create_training_pipeline(**kwargs)
