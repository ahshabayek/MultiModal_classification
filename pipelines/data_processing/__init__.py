"""
Data Processing Pipeline

This pipeline handles loading, validating, and preprocessing the
Hateful Memes dataset for multimodal classification.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
