"""Hugging Face dataset downloader for evaluation framework."""

import json
import logging
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from datasets import DatasetDict, load_dataset
except ImportError:
    raise ImportError("Please install datasets: pip install datasets")

from .config import DatasetConfig, DatasetSettings, get_dataset_configs

# Configure logging
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """
    Manages downloading and caching of Hugging Face datasets for evaluation.

    Attributes
    ----------
    settings : DatasetSettings
        Configuration settings for dataset management.
    """

    def __init__(self, settings: DatasetSettings | None = None) -> None:
        """
        Initialize the dataset downloader with configuration settings.

        Parameters
        ----------
        settings : DatasetSettings | None, optional
            Dataset settings configuration. If None, uses default settings.
        """
        self.settings = settings or DatasetSettings()

        # Ensure directories exist
        self.settings.base_download_dir.mkdir(parents=True, exist_ok=True)
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Dataset downloader initialized with base directory: {self.settings.base_download_dir}")

    def download_dataset(
        self,
        config: DatasetConfig,
        force_redownload: bool | None = None,
        save_samples: bool | None = None,
        sample_size: int | None = None,
    ) -> DatasetDict | None:
        """
        Download and cache a single dataset from Hugging Face Hub.

        Parameters
        ----------
        config : DatasetConfig
            Dataset configuration containing repository ID and settings.
        force_redownload : bool | None, optional
            Whether to force re-download even if cached.
        save_samples : bool | None, optional
            Whether to save sample data as JSON files.
        sample_size : int | None, optional
            Number of samples to save per split.

        Returns
        -------
        DatasetDict | None
            Loaded dataset or None if download failed.
        """
        # Use settings defaults if not specified
        force_redownload = force_redownload if force_redownload is not None else self.settings.force_redownload
        save_samples = save_samples if save_samples is not None else self.settings.save_samples
        sample_size = sample_size if sample_size is not None else self.settings.sample_size

        # Create dataset-specific directory
        dataset_dir = self.settings.base_download_dir / config.local_dir
        cache_dir = self.settings.cache_dir / config.local_dir

        # Check if already exists and not forcing redownload
        if not force_redownload and self._is_dataset_cached(dataset_dir):
            logger.info(f"Dataset {config.name} already cached at {dataset_dir}")
            try:
                return self._load_cached_dataset(dataset_dir)
            except Exception as e:
                logger.warning(f"Failed to load cached dataset: {e}. Re-downloading...")

        # Create directories
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {config.name} from {config.repo_id}")

        try:
            # Prepare download arguments
            download_kwargs = {"cache_dir": str(cache_dir), "trust_remote_code": config.trust_remote_code}

            if config.subset:
                download_kwargs["name"] = config.subset

            # Download the dataset
            ds = load_dataset(config.repo_id, **download_kwargs)

            logger.info(f"Successfully downloaded {config.name}")
            logger.info(f"Available splits: {list(ds.keys())}")

            # Filter to requested splits if specified
            if config.splits:
                available_splits = set(ds.keys())
                requested_splits = set(config.splits)
                missing_splits = requested_splits - available_splits

                if missing_splits:
                    logger.warning(f"Requested splits not found in {config.name}: {missing_splits}")

                # Keep only available requested splits
                valid_splits = requested_splits & available_splits
                if valid_splits:
                    ds = DatasetDict({split: ds[split] for split in valid_splits})
                else:
                    logger.warning(f"No valid splits found for {config.name}")

            # Log dataset information
            total_samples = sum(len(split_data) for split_data in ds.values())
            logger.info(f"Dataset {config.name} contains {total_samples:,} total samples")

            # Save dataset info and samples
            self._save_dataset_info(ds, config, dataset_dir)

            if save_samples:
                self._save_samples(ds, dataset_dir, sample_size)

            # Save the dataset in parquet format for efficient access
            self._save_as_parquet(ds, dataset_dir)

            logger.info(f"Dataset {config.name} successfully processed and saved to {dataset_dir}")
            return ds

        except Exception as e:
            logger.error(f"Failed to download {config.name}: {e}")
            logger.error(f"Repository: {config.repo_id}, Subset: {config.subset}")
            return None

    def download_multiple(self, dataset_keys: list[str], **kwargs) -> dict[str, DatasetDict | None]:
        """
        Download multiple datasets sequentially from the configured dataset collection.

        Parameters
        ----------
        dataset_keys : list[str]
            List of dataset configuration keys to download.
        **kwargs : Any
            Additional keyword arguments passed to download_dataset method.

        Returns
        -------
        dict[str, DatasetDict | None]
            Dictionary mapping dataset keys to their downloaded DatasetDict objects.
        """
        configs = get_dataset_configs()
        results = {}

        logger.info(f"Starting download of {len(dataset_keys)} datasets")

        for i, key in enumerate(dataset_keys, 1):
            logger.info(f"Processing dataset {i}/{len(dataset_keys)}: {key}")

            if key not in configs:
                logger.error(f"Dataset key '{key}' not found in configuration")
                results[key] = None
                continue

            config = configs[key]
            logger.info(f"Downloading {config.name}")

            results[key] = self.download_dataset(config, **kwargs)

        successful = sum(1 for ds in results.values() if ds is not None)
        logger.info(f"Download complete: {successful}/{len(dataset_keys)} datasets successful")

        return results

    def download_all(self, **kwargs) -> dict[str, DatasetDict | None]:
        """
        Download all configured datasets from the evaluation framework.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments passed to download_dataset method.

        Returns
        -------
        dict[str, DatasetDict | None]
            Dictionary mapping all configured dataset keys to their downloaded datasets.
        """
        configs = get_dataset_configs()
        logger.info(f"Downloading all {len(configs)} configured datasets")
        return self.download_multiple(list(configs.keys()), **kwargs)

    def download_by_task(self, task: str, **kwargs) -> dict[str, DatasetDict | None]:
        """
        Download all datasets configured for a specific machine learning task.

        Parameters
        ----------
        task : str
            Task type identifier (e.g., "stt", "diarization").
        **kwargs : Any
            Additional keyword arguments passed to download_dataset method.

        Returns
        -------
        dict[str, DatasetDict | None]
            Dictionary mapping dataset keys to their downloaded datasets for the specified task.
        """
        from .config import get_datasets_by_task

        task_configs = get_datasets_by_task(task)
        if not task_configs:
            logger.warning(f"No datasets found for task: {task}")
            return {}

        logger.info(f"Downloading {len(task_configs)} datasets for task: {task}")
        return self.download_multiple(list(task_configs.keys()), **kwargs)

    def download_by_language(self, language: str, **kwargs) -> dict[str, DatasetDict | None]:
        """
        Download all datasets configured for a specific language.

        Parameters
        ----------
        language : str
            Language identifier (e.g., "arabic", "english").
        **kwargs : Any
            Additional arguments passed to download_dataset.

        Returns
        -------
        dict[str, DatasetDict | None]
            Dictionary mapping dataset keys to downloaded datasets.
        """
        from .config import get_datasets_by_language

        lang_configs = get_datasets_by_language(language)
        if not lang_configs:
            logger.warning(f"No datasets found for language: {language}")
            return {}

        logger.info(f"Downloading {len(lang_configs)} datasets for language: {language}")
        return self.download_multiple(list(lang_configs.keys()), **kwargs)

    def download_recommended(self, **kwargs) -> dict[str, DatasetDict | None]:
        """
        Download the curated set of recommended datasets.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments passed to download_dataset.

        Returns
        -------
        dict[str, DatasetDict | None]
            Dictionary mapping recommended dataset keys to downloaded datasets.
        """
        from .config import get_eval_recommended_datasets

        recommended = get_eval_recommended_datasets()
        logger.info(f"Downloading {len(recommended)} recommended evaluation datasets")
        return self.download_multiple(list(recommended.keys()), **kwargs)

    def list_available_datasets(self) -> dict[str, DatasetConfig]:
        """
        Get all available dataset configurations from the registry.

        Returns
        -------
        dict[str, DatasetConfig]
            Dictionary mapping dataset keys to their complete configuration objects.
        """
        return get_dataset_configs()

    def get_dataset_info(self, dataset_key: str) -> DatasetConfig | None:
        """
        Get configuration information for a specific dataset.

        Parameters
        ----------
        dataset_key : str
            The unique identifier key for the dataset.

        Returns
        -------
        DatasetConfig | None
            Dataset configuration object or None if not found.
        """
        configs = get_dataset_configs()
        return configs.get(dataset_key)

    def get_download_status(self) -> dict[str, dict[str, Any]]:
        """
        Get download status for all configured datasets.

        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping dataset keys to status information including
            download status, metadata, and error information.
        """
        configs = get_dataset_configs()
        status = {}

        for key, config in configs.items():
            dataset_dir = self.settings.base_download_dir / config.local_dir
            info_file = dataset_dir / "dataset_info.json"

            if info_file.exists():
                try:
                    with open(info_file, encoding="utf-8") as f:
                        info = json.load(f)

                    status[key] = {"downloaded": True, "name": config.name, "task": config.task, "language": config.language, "total_samples": info.get("total_samples", 0), "splits": info.get("splits", {}), "directory": str(dataset_dir)}
                except Exception as e:
                    status[key] = {"downloaded": False, "error": str(e), "name": config.name, "directory": str(dataset_dir)}
            else:
                status[key] = {"downloaded": False, "name": config.name, "task": config.task, "language": config.language, "directory": str(dataset_dir)}

        return status

    def _is_dataset_cached(self, dataset_dir: Path) -> bool:
        """
        Check if a dataset is already cached locally.

        Parameters
        ----------
        dataset_dir : Path
            Local directory path where the dataset should be stored.

        Returns
        -------
        bool
            True if the dataset is cached, False otherwise.
        """
        return (dataset_dir / "dataset_info.json").exists()

    def _load_cached_dataset(self, dataset_dir: Path) -> DatasetDict | None:
        """
        Load a dataset from cached files (placeholder implementation).

        Parameters
        ----------
        dataset_dir : Path
            Local directory path containing the cached dataset files.

        Returns
        -------
        DatasetDict | None
            Currently returns None as loading from cache is not implemented.
        """
        # This is a placeholder - could implement loading from parquet files
        # for faster access in the future
        logger.debug(f"Cached dataset loading not implemented for {dataset_dir}")
        return None

    def _save_dataset_info(self, ds: DatasetDict, config: DatasetConfig, dataset_dir: Path):
        """
        Save comprehensive dataset information and metadata to JSON file.

        Parameters
        ----------
        ds : DatasetDict
            The downloaded dataset containing all splits and data.
        config : DatasetConfig
            Original dataset configuration used for downloading.
        dataset_dir : Path
            Local directory where the dataset is stored.
        """
        info = {
            "name": config.name,
            "repo_id": config.repo_id,
            "subset": config.subset,
            "description": config.description,
            "language": config.language,
            "task": config.task,
            "splits": {split_name: len(split_data) for split_name, split_data in ds.items()},
            "total_samples": sum(len(split_data) for split_data in ds.values()),
            "columns": list(next(iter(ds.values())).column_names) if ds else [],
            "download_timestamp": pd.Timestamp.now().isoformat() if pd else None,
        }

        info_file = dataset_dir / "dataset_info.json"
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.debug(f"Dataset info saved to {info_file}")

    def _save_samples(self, ds: DatasetDict, dataset_dir: Path, sample_size: int):
        """
        Save sample data from each dataset split for inspection.

        Parameters
        ----------
        ds : DatasetDict
            The downloaded dataset containing splits.
        dataset_dir : Path
            Local directory where sample files should be saved.
        sample_size : int
            Number of samples to extract from each split.
        """
        for split_name, split_data in ds.items():
            split_dir = dataset_dir / split_name
            split_dir.mkdir(exist_ok=True)

            # Convert to pandas if available, otherwise use dataset methods
            if pd:
                df = split_data.to_pandas()
                sample_df = df.head(sample_size)

                # Remove audio column for JSON serialization
                sample_for_json = sample_df.drop(columns=["audio"], errors="ignore")
                sample_data = sample_for_json.to_dict("records")
            else:
                # Fallback without pandas
                sample_data = []
                for i, example in enumerate(split_data):
                    if i >= sample_size:
                        break
                    # Remove audio data for JSON serialization
                    clean_example = {k: v for k, v in example.items() if k != "audio"}
                    sample_data.append(clean_example)

            # Save sample as JSON
            sample_file = split_dir / f"{split_name}_sample.json"

            with open(sample_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(sample_data)} samples to {sample_file}")

    def _save_as_parquet(self, ds: DatasetDict, dataset_dir: Path):
        """
        Save dataset splits as parquet files for efficient access.

        Parameters
        ----------
        ds : DatasetDict
            The downloaded dataset containing splits.
        dataset_dir : Path
            Local directory where parquet files should be saved.
        """
        if not pd:
            logger.warning("Pandas not available, skipping parquet export")
            return

        for split_name, split_data in ds.items():
            split_dir = dataset_dir / split_name
            split_dir.mkdir(exist_ok=True)

            # Convert to pandas and save
            df = split_data.to_pandas()
            parquet_file = split_dir / f"{split_name}.parquet"
            df.to_parquet(parquet_file, index=False)

            logger.debug(f"Saved split '{split_name}' as {parquet_file}")


# Convenience functions for the evaluation framework
def create_downloader(base_dir: str | Path = "datasets") -> DatasetDownloader:
    """
    Create a dataset downloader with standard settings.

    Parameters
    ----------
    base_dir : str | Path, optional
        Base directory for storing downloaded datasets.

    Returns
    -------
    DatasetDownloader
        Configured DatasetDownloader instance.
    """
    settings = DatasetSettings(base_download_dir=Path(base_dir))
    return DatasetDownloader(settings)


def download_eval_datasets(dataset_keys: list[str] | None = None, base_dir: str | Path = "datasets", **kwargs) -> dict[str, DatasetDict | None]:
    """
    Download datasets for evaluation workflows.

    Parameters
    ----------
    dataset_keys : list[str] | None, optional
        List of dataset keys to download. If None, downloads recommended datasets.
    base_dir : str | Path, optional
        Base directory for storing downloaded datasets.
    **kwargs : Any
        Additional arguments passed to download methods.

    Returns
    -------
    dict[str, DatasetDict | None]
        Dictionary mapping dataset keys to downloaded datasets.
    """
    downloader = create_downloader(base_dir)

    if dataset_keys:
        return downloader.download_multiple(dataset_keys, **kwargs)
    else:
        return downloader.download_recommended(**kwargs)
