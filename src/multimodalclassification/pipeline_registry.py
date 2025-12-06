from typing import Dict

from kedro.pipeline import Pipeline

from multimodalclassification.pipelines.data_processing import (
    create_pipeline as create_data_processing_pipeline,
)
from multimodalclassification.pipelines.model_training import (
    create_frcnn_training_pipeline,
    create_inference_local_pipeline,
    create_inference_pipeline,
    create_lmdb_training_pipeline,
    create_precomputed_training_pipeline,
    create_training_pipeline,
    create_validation_pipeline,
    create_vg_training_pipeline,
    create_x152_training_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register all pipelines.

    Available pipelines:
    - data_processing: Load data from HuggingFace and preprocess
    - vilbert_train: Full training (data processing + training + evaluation)
    - vilbert_validate: Validate locally trained model on test set
    - vilbert_inference: Run inference with pretrained HuggingFace weights
    - vilbert_inference_local: Run inference with locally trained weights
    - vilbert_frcnn_train: Training with Faster R-CNN (COCO) + Facebook weights
    - vilbert_vg_train: Training with Faster R-CNN (Visual Genome) + Facebook weights

    Usage:
        kedro run --pipeline=data_processing
        kedro run --pipeline=vilbert_train
        kedro run --pipeline=vilbert_frcnn_train
        kedro run --pipeline=vilbert_vg_train
        kedro run --pipeline=vilbert_validate
        kedro run --pipeline=vilbert_inference
        kedro run --pipeline=vilbert_inference_local
    """

    # Data processing pipeline (loads from HuggingFace)
    data_processing = create_data_processing_pipeline()

    # Model pipelines
    model_training = create_training_pipeline()
    model_training_frcnn = create_frcnn_training_pipeline()
    model_training_vg = create_vg_training_pipeline()
    model_training_precomputed = create_precomputed_training_pipeline()
    model_training_lmdb = create_lmdb_training_pipeline()
    model_training_x152 = create_x152_training_pipeline()
    model_validation = create_validation_pipeline()
    model_inference = create_inference_pipeline()
    model_inference_local = create_inference_local_pipeline()

    # Combined pipelines (data processing + model)
    vilbert_train = data_processing + model_training
    vilbert_frcnn_train = data_processing + model_training_frcnn
    vilbert_vg_train = data_processing + model_training_vg
    vilbert_precomputed_train = data_processing + model_training_precomputed
    vilbert_lmdb_train = data_processing + model_training_lmdb
    vilbert_x152_train = data_processing + model_training_x152
    vilbert_validate = data_processing + model_validation
    vilbert_inference = data_processing + model_inference
    vilbert_inference_local = data_processing + model_inference_local

    return {
        "__default__": vilbert_train,
        # Data processing only
        "data_processing": data_processing,
        # Full pipelines (data processing + model)
        "vilbert": vilbert_train,
        "vilbert_train": vilbert_train,
        "vilbert_frcnn_train": vilbert_frcnn_train,  # Faster R-CNN (COCO) + Facebook weights
        "vilbert_vg_train": vilbert_vg_train,  # Faster R-CNN (Visual Genome) + Facebook weights
        "vilbert_precomputed_train": vilbert_precomputed_train,  # Precomputed features (Facebook-style)
        "vilbert_lmdb_train": vilbert_lmdb_train,  # Facebook's official LMDB features (ResNeXt-152)
        "vilbert_x152_train": vilbert_x152_train,  # Facebook's X-152++ from grid-feats-vqa (best quality)
        "vilbert_validate": vilbert_validate,
        "vilbert_inference": vilbert_inference,
        "vilbert_inference_local": vilbert_inference_local,
        # Model-only pipelines (assumes data is already processed)
        "model_training": model_training,
        "model_training_frcnn": model_training_frcnn,
        "model_training_vg": model_training_vg,
        "model_training_precomputed": model_training_precomputed,
        "model_training_lmdb": model_training_lmdb,
        "model_training_x152": model_training_x152,
        "model_validation": model_validation,
        "model_inference": model_inference,
        "model_inference_local": model_inference_local,
    }
