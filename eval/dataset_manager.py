"""Convenience functions for dataset management in the evaluation framework.

This module provides high-level functions for managing datasets in the evaluation
framework, including downloading, listing, and setting up datasets for various
evaluation tasks.
"""

import logging
from typing import Any

from .config import get_settings
from .datasets.config import DatasetSettings, get_dataset_configs
from .datasets.downloader import DatasetDownloader

logger = logging.getLogger(__name__)


def create_dataset_downloader(settings: Any = None) -> DatasetDownloader:
    """
    Create a dataset downloader using evaluation framework settings.

    Parameters
    ----------
    settings : Any, optional
        Evaluation framework settings object. If None, loads default settings.

    Returns
    -------
    DatasetDownloader
        Configured DatasetDownloader instance.
    """
    settings = get_settings()

    # Create DatasetSettings from evaluation settings
    dataset_settings = DatasetSettings(
        base_download_dir=settings.datasets.base_download_dir,
        cache_dir=settings.datasets.cache_dir,
        sample_size=settings.datasets.sample_size,
        save_samples=settings.datasets.save_samples,
        force_redownload=settings.datasets.force_redownload,
    )

    return DatasetDownloader(dataset_settings)


def download_datasets(
    dataset_keys: list[str] | None = None,
    task: str | None = None,
    language: str | None = None,
    recommended_only: bool = False,
    settings: Any = None,
    **kwargs: Any,
) -> dict[str, dict | None]:
    """
    Download datasets with flexible filtering options.

    Parameters
    ----------
    dataset_keys : list[str] | None, optional
        Specific dataset keys to download. Takes precedence over other options.
    task : str | None, optional
        Download datasets for specific task (stt, diarization, etc.).
    language : str | None, optional
        Download datasets for specific language (arabic, english, etc.).
    recommended_only : bool, optional
        Download only recommended datasets, by default False.
    settings : Any, optional
        Evaluation framework settings object.
    **kwargs : Any
        Additional arguments passed to download methods.

    Returns
    -------
    dict[str, dict | None]
        Dictionary mapping dataset keys to loaded datasets (None if failed).
    """
    downloader = create_dataset_downloader(settings)

    if dataset_keys:
        logger.info(f"Downloading datasets: {dataset_keys}")
        return downloader.download_multiple(dataset_keys, **kwargs)
    elif task:
        logger.info(f"Downloading datasets for task: {task}")
        return downloader.download_by_task(task, **kwargs)
    elif language:
        logger.info(f"Downloading datasets for language: {language}")
        return downloader.download_by_language(language, **kwargs)
    elif recommended_only:
        logger.info("Downloading recommended datasets")
        return downloader.download_recommended(**kwargs)
    else:
        logger.info("No specific selection criteria provided, downloading recommended datasets")
        return downloader.download_recommended(**kwargs)


def list_available_datasets() -> dict[str, dict]:
    """
    List all available datasets configured for evaluation.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping dataset keys to summary information including
        name, repo_id, task, language, description, and local_dir.
    """
    configs = get_dataset_configs()
    return {key: {"name": config.name, "repo_id": config.repo_id, "task": config.task, "language": config.language, "description": config.description, "local_dir": config.local_dir} for key, config in configs.items()}


def get_dataset_status(settings: Any = None) -> dict[str, dict[str, bool | str | int]]:
    """
    Get download status for all configured datasets.

    Parameters
    ----------
    settings : Any, optional
        Evaluation framework settings object.

    Returns
    -------
    dict[str, dict[str, bool | str | int]]
        Dictionary mapping dataset keys to status information including
        downloaded status, name, task, language, sample counts, and directory.
    """
    downloader = create_dataset_downloader(settings)
    return downloader.get_download_status()


def setup_eval_datasets(
    quick_setup: bool = True,
    settings: Any = None,
) -> dict[str, dict | None]:
    """
    Set up datasets for evaluation framework.

    Parameters
    ----------
    quick_setup : bool, optional
        If True, downloads recommended datasets only. If False, downloads all.
    settings : Any, optional
        Evaluation framework settings object.

    Returns
    -------
    dict[str, dict | None]
        Dictionary mapping dataset keys to loaded datasets.
    """
    downloader = create_dataset_downloader(settings)

    if quick_setup:
        logger.info("Setting up recommended datasets for evaluation")
        return downloader.download_recommended()
    else:
        logger.info("Setting up all available datasets for evaluation")
        return downloader.download_all()


# Specific convenience functions for different use cases
def setup_stt_datasets(settings: Any = None) -> dict[str, dict | None]:
    """
    Download all Speech-to-Text (STT) datasets for evaluation.

    Parameters
    ----------
    settings : Any, optional
        Evaluation framework settings object.

    Returns
    -------
    dict[str, dict | None]
        Dictionary mapping STT dataset keys to loaded datasets.
    """
    return download_datasets(test="stt", settings=settings)


def setup_diarization_datasets(settings: Any = None) -> dict[str, dict | None]:
    """
    Download all speaker diarization datasets for evaluation.

    Parameters
    ----------
    settings : Any, optional
        Evaluation framework settings object.

    Returns
    -------
    dict[str, dict | None]
        Dictionary mapping diarization dataset keys to loaded datasets.
    """
    return download_datasets(task="diarization", settings=settings)


def setup_arabic_datasets(settings: Any = None) -> dict[str, dict | None]:
    """
    Download all Arabic language datasets for evaluation.

    Parameters
    ----------
    settings : Any, optional
        Evaluation framework settings object.

    Returns
    -------
    dict[str, dict | None]
        Dictionary mapping Arabic dataset keys to loaded datasets.
    """
    return download_datasets(language="arabic", settings=settings)


def setup_english_datasets(settings: Any = None) -> dict[str, dict | None]:
    """
    Download all English language datasets for evaluation.

    Parameters
    ----------
    settings : Any, optional
        Evaluation framework settings object.

    Returns
    -------
    dict[str, dict | None]
        Dictionary mapping English dataset keys to loaded datasets.
    """
    return download_datasets(language="english", settings=settings)
