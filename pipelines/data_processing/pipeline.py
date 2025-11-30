"""
Data Processing Pipeline for Hateful Memes Classification

Location: src/multimodalclassification/pipelines/data_processing/pipeline.py
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    load_and_validate_data,
    create_train_val_split,
    process_train_data,
    process_val_data,
    process_test_seen_data,
    process_test_unseen_data,
    compute_dataset_statistics,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data processing pipeline."""
    return pipeline([
        node(
            func=load_and_validate_data,
            inputs=[
                "hateful_memes_train",
                "hateful_memes_dev_seen",
                "hateful_memes_dev_unseen",
                "hateful_memes_test_seen",
                "hateful_memes_test_unseen",
                "params:data_processing"
            ],
            outputs="validated_datasets",
            name="load_and_validate_data",
        ),
        node(
            func=create_train_val_split,
            inputs=[
                "validated_datasets",
                "params:data_processing"
            ],
            outputs=[
                "train_split",
                "val_split",
                "test_seen_split",
                "test_unseen_split",
                "split_info"
            ],
            name="create_splits",
        ),
        node(
            func=process_train_data,
            inputs=["train_split", "params:data_processing"],
            outputs="processed_train",
            name="process_train",
        ),
        node(
            func=process_val_data,
            inputs=["val_split", "params:data_processing"],
            outputs="processed_val",
            name="process_val",
        ),
        node(
            func=process_test_seen_data,
            inputs=["test_seen_split", "params:data_processing"],
            outputs="processed_test_seen",
            name="process_test_seen",
        ),
        node(
            func=process_test_unseen_data,
            inputs=["test_unseen_split", "params:data_processing"],
            outputs="processed_test_unseen",
            name="process_test_unseen",
        ),
        node(
            func=compute_dataset_statistics,
            inputs=[
                "processed_train",
                "processed_val",
                "processed_test_seen",
                "processed_test_unseen",
                "split_info"
            ],
            outputs="data_splits_info",
            name="compute_statistics",
        ),
    ])
