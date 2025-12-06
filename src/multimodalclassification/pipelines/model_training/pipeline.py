"""ViLBERT training pipelines."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_dataloaders,
    create_dataloaders_frcnn,
    create_dataloaders_lmdb,
    create_dataloaders_precomputed,
    create_dataloaders_vg,
    create_dataloaders_x152,
    create_inference_dataloader,
    evaluate_model,
    load_trained_model,
    load_vilbert_facebook,
    load_vilbert_lmdb,
    load_vilbert_model,
    load_vilbert_vg,
    load_vilbert_x152,
    run_inference,
    save_model,
    train_model,
    train_model_frcnn,
    train_model_lmdb,
    train_model_vg,
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


def create_vg_training_pipeline(**kwargs) -> Pipeline:
    """Training with Facebook weights + Faster R-CNN (Visual Genome) features."""
    return pipeline(
        _create_training_pipeline_nodes(
            create_dataloaders_vg, load_vilbert_vg, train_model_vg, "vg"
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
