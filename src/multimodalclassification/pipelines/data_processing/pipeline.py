"""
Data Processing Pipeline for Hateful Memes Classification

Loads data from HuggingFace datasets and prepares it for model training.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    compute_dataset_statistics,
    create_train_val_split,
    load_and_validate_data,
    process_test_data,
    process_train_data,
    process_val_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data processing pipeline."""
    return pipeline(
        [
            node(
                func=load_and_validate_data,
                inputs="params:data_processing",
                outputs="validated_datasets",
                name="load_and_validate_data",
            ),
            node(
                func=create_train_val_split,
                inputs=["validated_datasets", "params:data_processing"],
                outputs=[
                    "train_split",
                    "val_split",
                    "test_split",
                    "split_info",
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
                func=process_test_data,
                inputs=["test_split", "params:data_processing"],
                outputs="processed_test",
                name="process_test",
            ),
            node(
                func=compute_dataset_statistics,
                inputs=[
                    "processed_train",
                    "processed_val",
                    "processed_test",
                    "split_info",
                ],
                outputs="data_splits_info",
                name="compute_statistics",
            ),
        ]
    )
