"""Dataset management utilities for evaluation framework."""

from .config import get_dataset_configs, get_datasets_by_language, get_datasets_by_task
from .downloader import DatasetConfig, DatasetDownloader

__all__ = [
    "DatasetDownloader",
    "DatasetConfig",
    "get_dataset_configs",
    "get_datasets_by_task",
    "get_datasets_by_language",
]
