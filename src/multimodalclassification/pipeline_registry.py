from typing import Dict

from kedro.pipeline import Pipeline

from multimodalclassification.pipelines.data_processing import (
    create_pipeline as create_data_processing_pipeline,
)
from multimodalclassification.pipelines.model_training import (
    create_dinov2_multilayer_training_pipeline,
    create_dinov2_training_pipeline,
    create_frcnn_resnet152_training_pipeline,
    create_frcnn_training_pipeline,
    create_inference_local_pipeline,
    create_inference_pipeline,
    create_lmdb_training_pipeline,
    create_precomputed_training_pipeline,
    create_resnet152_grid_training_pipeline,
    create_resnet152_roi_training_pipeline,
    create_resnet_vg_training_pipeline,
    create_training_pipeline,
    create_validation_pipeline,
    create_vg_rpn_training_pipeline,
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
    model_training_frcnn_resnet152 = create_frcnn_resnet152_training_pipeline()
    model_training_dinov2 = create_dinov2_training_pipeline()
    model_training_dinov2_multilayer = create_dinov2_multilayer_training_pipeline()
    model_training_vg = create_vg_training_pipeline()
    model_training_vg_rpn = create_vg_rpn_training_pipeline()
    model_training_resnet_vg = create_resnet_vg_training_pipeline()
    model_training_resnet152_grid = create_resnet152_grid_training_pipeline()
    model_training_resnet152_roi = create_resnet152_roi_training_pipeline()
    model_training_precomputed = create_precomputed_training_pipeline()
    model_training_lmdb = create_lmdb_training_pipeline()
    model_training_x152 = create_x152_training_pipeline()
    model_validation = create_validation_pipeline()
    model_inference = create_inference_pipeline()
    model_inference_local = create_inference_local_pipeline()

    # Combined pipelines (data processing + model)
    vilbert_train = data_processing + model_training
    vilbert_frcnn_train = data_processing + model_training_frcnn
    vilbert_frcnn_resnet152_train = data_processing + model_training_frcnn_resnet152
    vilbert_dinov2_train = data_processing + model_training_dinov2
    vilbert_dinov2_multilayer_train = data_processing + model_training_dinov2_multilayer
    vilbert_vg_train = data_processing + model_training_vg
    vilbert_vg_rpn_train = data_processing + model_training_vg_rpn
    vilbert_resnet_vg_train = data_processing + model_training_resnet_vg
    vilbert_resnet152_grid_train = data_processing + model_training_resnet152_grid
    vilbert_resnet152_roi_train = data_processing + model_training_resnet152_roi
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
        "vilbert_frcnn_train": vilbert_frcnn_train,  # Faster R-CNN ResNet-50 (COCO) + Facebook weights
        "vilbert_frcnn_resnet152_train": vilbert_frcnn_resnet152_train,  # Faster R-CNN ResNet-152 (COCO) + Facebook weights
        "vilbert_dinov2_train": vilbert_dinov2_train,  # DINOv2 Vision Transformer + Facebook weights
        "vilbert_dinov2_multilayer_train": vilbert_dinov2_multilayer_train,  # DINOv2 Multi-Layer fusion + Facebook weights
        "vilbert_vg_train": vilbert_vg_train,  # Faster R-CNN (Visual Genome) + Facebook weights
        "vilbert_vg_rpn_train": vilbert_vg_rpn_train,  # VG Faster R-CNN with trained RPN + Facebook weights
        "vilbert_resnet_vg_train": vilbert_resnet_vg_train,  # Simple VG ResNet-101 backbone (no detection)
        "vilbert_resnet152_grid_train": vilbert_resnet152_grid_train,  # ResNet-152 with grid pooling (Facebook CC weights, no ROI)
        "vilbert_resnet152_roi_train": vilbert_resnet152_roi_train,  # ResNet-152 with ROI pooling (ImageNet, no detection)
        "vilbert_precomputed_train": vilbert_precomputed_train,  # Precomputed features (Facebook-style)
        "vilbert_lmdb_train": vilbert_lmdb_train,  # Facebook's official LMDB features (ResNeXt-152)
        "vilbert_x152_train": vilbert_x152_train,  # Facebook's X-152++ from grid-feats-vqa (best quality)
        "vilbert_validate": vilbert_validate,
        "vilbert_inference": vilbert_inference,
        "vilbert_inference_local": vilbert_inference_local,
        # Model-only pipelines (assumes data is already processed)
        "model_training": model_training,
        "model_training_frcnn": model_training_frcnn,
        "model_training_frcnn_resnet152": model_training_frcnn_resnet152,
        "model_training_dinov2": model_training_dinov2,
        "model_training_dinov2_multilayer": model_training_dinov2_multilayer,
        "model_training_vg": model_training_vg,
        "model_training_vg_rpn": model_training_vg_rpn,
        "model_training_resnet_vg": model_training_resnet_vg,
        "model_training_resnet152_grid": model_training_resnet152_grid,
        "model_training_resnet152_roi": model_training_resnet152_roi,
        "model_training_precomputed": model_training_precomputed,
        "model_training_lmdb": model_training_lmdb,
        "model_training_x152": model_training_x152,
        "model_validation": model_validation,
        "model_inference": model_inference,
        "model_inference_local": model_inference_local,
    }
