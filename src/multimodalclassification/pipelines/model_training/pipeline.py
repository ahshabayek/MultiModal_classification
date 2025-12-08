"""ViLBERT training pipelines."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_dataloaders,
    create_dataloaders_dinov2,
    create_dataloaders_dinov2_multilayer,
    create_dataloaders_frcnn,
    create_dataloaders_frcnn_resnet152,
    create_dataloaders_lmdb,
    create_dataloaders_precomputed,
    create_dataloaders_resnet152_grid,
    create_dataloaders_resnet152_roi,
    create_dataloaders_resnet_vg,
    create_dataloaders_vg,
    create_dataloaders_vg_rpn,
    create_dataloaders_x152,
    create_inference_dataloader,
    evaluate_model,
    load_trained_model,
    load_vilbert_dinov2,
    load_vilbert_dinov2_multilayer,
    load_vilbert_facebook,
    load_vilbert_frcnn_resnet152,
    load_vilbert_lmdb,
    load_vilbert_model,
    load_vilbert_resnet152_grid,
    load_vilbert_resnet152_roi,
    load_vilbert_resnet_vg,
    load_vilbert_vg,
    load_vilbert_vg_rpn,
    load_vilbert_x152,
    run_inference,
    save_model,
    train_model,
    train_model_dinov2,
    train_model_dinov2_multilayer,
    train_model_frcnn,
    train_model_frcnn_resnet152,
    train_model_lmdb,
    train_model_resnet152_grid,
    train_model_resnet152_roi,
    train_model_resnet_vg,
    train_model_vg,
    train_model_vg_rpn,
    train_model_x152,
)


def _create_training_pipeline_nodes(
    create_loaders_fn, load_model_fn, train_fn, suffix=""
) -> list:
    """Helper to create standard training pipeline nodes."""
    s = f"_{suffix}" if suffix else ""
    return [
        node(
            create_loaders_fn,
            ["processed_train", "processed_val", "processed_test", "parameters"],
            [f"train_loader{s}", f"val_loader{s}", f"test_loader{s}"],
            name=f"create_dataloaders{s}",
        ),
        node(
            load_model_fn,
            "parameters",
            f"vilbert_model{s}",
            name=f"load_pretrained_model{s}",
        ),
        node(
            train_fn,
            [f"vilbert_model{s}", f"train_loader{s}", f"val_loader{s}", "parameters"],
            [f"trained_model{s}", f"training_history{s}"],
            name=f"train_model{s}",
        ),
        node(
            evaluate_model,
            [f"trained_model{s}", f"test_loader{s}", "parameters"],
            f"test_metrics{s}",
            name=f"evaluate_model{s}",
        ),
        node(
            save_model,
            [f"trained_model{s}", f"test_metrics{s}", "parameters"],
            f"model_path{s}",
            name=f"save_model{s}",
        ),
    ]


def create_training_pipeline(**kwargs) -> Pipeline:
    """Default training with HuggingFace weights + ResNet features."""
    return pipeline(
        _create_training_pipeline_nodes(
            create_dataloaders, load_vilbert_model, train_model
        )
    )


def create_frcnn_training_pipeline(**kwargs) -> Pipeline:
    """Training with Facebook weights + Faster R-CNN (COCO) features."""
    return pipeline(
        _create_training_pipeline_nodes(
            create_dataloaders_frcnn, load_vilbert_facebook, train_model_frcnn, "frcnn"
        )
    )


def create_frcnn_resnet152_training_pipeline(**kwargs) -> Pipeline:
    """Training with Facebook weights + Faster R-CNN ResNet-152 FPN (COCO) features."""
    return pipeline(
        _create_training_pipeline_nodes(
            create_dataloaders_frcnn_resnet152,
            load_vilbert_frcnn_resnet152,
            train_model_frcnn_resnet152,
            "frcnn_resnet152",
        )
    )


def create_dinov2_training_pipeline(**kwargs) -> Pipeline:
    """Training with Facebook weights + DINOv2 Vision Transformer features."""
    return pipeline(
        _create_training_pipeline_nodes(
            create_dataloaders_dinov2,
            load_vilbert_dinov2,
            train_model_dinov2,
            "dinov2",
        )
    )


def create_dinov2_multilayer_training_pipeline(**kwargs) -> Pipeline:
    """Training with Facebook weights + DINOv2 Multi-Layer feature fusion.

    Extracts features from multiple DINOv2 transformer layers (e.g., 6, 12, 18, 24)
    and fuses them to capture both low-level patterns and high-level semantics.
    """
    return pipeline(
        _create_training_pipeline_nodes(
            create_dataloaders_dinov2_multilayer,
            load_vilbert_dinov2_multilayer,
            train_model_dinov2_multilayer,
            "dinov2_multilayer",
        )
    )


def create_vg_training_pipeline(**kwargs) -> Pipeline:
    """Training with Facebook weights + Faster R-CNN (Visual Genome) features."""
    return pipeline(
        _create_training_pipeline_nodes(
            create_dataloaders_vg, load_vilbert_vg, train_model_vg, "vg"
        )
    )


def create_vg_rpn_training_pipeline(**kwargs) -> Pipeline:
    """Training with Facebook weights + VG Faster R-CNN with trained RPN.

    This uses the TRAINED RPN from the Visual Genome checkpoint for learned
    region proposals instead of grid-based proposals.
    """
    return pipeline(
        _create_training_pipeline_nodes(
            create_dataloaders_vg_rpn, load_vilbert_vg_rpn, train_model_vg_rpn, "vg_rpn"
        )
    )


def create_resnet_vg_training_pipeline(**kwargs) -> Pipeline:
    """Training with Facebook weights + simple VG ResNet-101 backbone (no detection).

    This uses ONLY the ResNet-101 backbone from the VG checkpoint with grid-based
    pooling - NO RPN, NO ROI pooling, NO detection head.
    Similar to simple ResNet-152 but with VG-pretrained weights.
    """
    return pipeline(
        _create_training_pipeline_nodes(
            create_dataloaders_resnet_vg,
            load_vilbert_resnet_vg,
            train_model_resnet_vg,
            "resnet_vg",
        )
    )


def create_resnet152_roi_training_pipeline(**kwargs) -> Pipeline:
    """Training with Facebook weights + ResNet-152 with ROI pooling.

    This tests whether ROI pooling helps with an ImageNet backbone:
    - Uses ResNet-152 pretrained on ImageNet (same as vilbert_train)
    - Applies ROI pooling on multi-scale proposals (like detection models)
    - NO detection training (no COCO/VG classification head)

    Comparison:
    - vilbert_train (ResNet-152 + simple grid): 0.6645 AUROC
    - vilbert_frcnn_resnet152_train (ResNet-152 + COCO detection): 0.6334 AUROC
    - This model (ResNet-152 + ROI pooling, no detection): ???
    """
    return pipeline(
        _create_training_pipeline_nodes(
            create_dataloaders_resnet152_roi,
            load_vilbert_resnet152_roi,
            train_model_resnet152_roi,
            "resnet152_roi",
        )
    )


def create_resnet152_grid_training_pipeline(**kwargs) -> Pipeline:
    """Training with Facebook weights + ResNet-152 simple grid (NO ROI pooling).

    This is a control experiment to isolate the effect of ROI pooling:
    - Same ViLBERT weights as vilbert_resnet152_roi_train (Facebook CC)
    - Same ResNet-152 backbone (ImageNet pretrained)
    - Simple 6x6 grid pooling (NO ROI pooling)

    Comparison:
    - vilbert_resnet152_roi_train (Facebook CC + ROI pooling): 0.7197 AUROC
    - This model (Facebook CC + grid pooling): ???
    """
    return pipeline(
        _create_training_pipeline_nodes(
            create_dataloaders_resnet152_grid,
            load_vilbert_resnet152_grid,
            train_model_resnet152_grid,
            "resnet152_grid",
        )
    )


def create_precomputed_training_pipeline(**kwargs) -> Pipeline:
    """Training with Facebook weights + precomputed HDF5 features."""
    return pipeline(
        _create_training_pipeline_nodes(
            create_dataloaders_precomputed,
            load_vilbert_vg,
            train_model_vg,
            "precomputed",
        )
    )


def create_lmdb_training_pipeline(**kwargs) -> Pipeline:
    """Training with Facebook weights + official LMDB features (best performance)."""
    return pipeline(
        _create_training_pipeline_nodes(
            create_dataloaders_lmdb, load_vilbert_lmdb, train_model_lmdb, "lmdb"
        )
    )


def create_x152_training_pipeline(**kwargs) -> Pipeline:
    """Training with Facebook weights + X-152++ features (requires detectron2)."""
    return pipeline(
        _create_training_pipeline_nodes(
            create_dataloaders_x152, load_vilbert_x152, train_model_x152, "x152"
        )
    )


def create_validation_pipeline(**kwargs) -> Pipeline:
    """Validate a locally trained model on test set."""
    return pipeline(
        [
            node(
                create_inference_dataloader,
                ["processed_test", "parameters"],
                "test_loader",
                name="create_test_dataloader",
            ),
            node(
                load_trained_model,
                "parameters",
                "trained_model",
                name="load_trained_model",
            ),
            node(
                evaluate_model,
                ["trained_model", "test_loader", "parameters"],
                "validation_metrics",
                name="evaluate_trained_model",
            ),
        ]
    )


def create_inference_pipeline(**kwargs) -> Pipeline:
    """Run inference with pretrained HuggingFace weights."""
    return pipeline(
        [
            node(
                create_inference_dataloader,
                ["processed_test", "parameters"],
                "inference_loader",
                name="create_inference_dataloader",
            ),
            node(
                load_vilbert_model,
                "parameters",
                "pretrained_model",
                name="load_pretrained_model",
            ),
            node(
                run_inference,
                ["pretrained_model", "inference_loader", "parameters"],
                "predictions",
                name="run_inference",
            ),
        ]
    )


def create_inference_local_pipeline(**kwargs) -> Pipeline:
    """Run inference with locally trained weights."""
    return pipeline(
        [
            node(
                create_inference_dataloader,
                ["processed_test", "parameters"],
                "inference_loader",
                name="create_inference_dataloader",
            ),
            node(
                load_trained_model, "parameters", "local_model", name="load_local_model"
            ),
            node(
                run_inference,
                ["local_model", "inference_loader", "parameters"],
                "predictions",
                name="run_inference",
            ),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Default pipeline (training). Kept for backward compatibility."""
    return create_training_pipeline(**kwargs)
