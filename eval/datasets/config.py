"""Dataset configuration for Hugging Face datasets in the evaluation framework."""

from pathlib import Path

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """
    Configuration for a single Hugging Face dataset used in evaluation.

    Attributes
    ----------
    name : str
        Human-readable friendly name for the dataset.
    repo_id : str
        Hugging Face Hub repository identifier.
    subset : str | None, optional
        Dataset subset or configuration name if applicable.
    splits : list[str] | None, optional
        Specific dataset splits to download.
    local_dir : str
        Local directory path for dataset storage.
    description : str | None, optional
        Brief description of the dataset.
    language : str | None, optional
        Primary language of the dataset.
    task : str
        Machine learning task type (e.g., "stt", "diarization").
    trust_remote_code : bool, optional
        Whether to trust remote code when loading the dataset.
    """

    name: str = Field(..., description="Friendly name for the dataset")
    repo_id: str = Field(..., description="Hugging Face repository ID")
    subset: str | None = Field(None, description="Dataset subset/configuration")
    splits: list[str] | None = Field(None, description="Specific splits to download (all if None)")
    local_dir: str = Field(..., description="Local directory name for storage")
    description: str | None = Field(None, description="Dataset description")
    language: str | None = Field(None, description="Primary language of the dataset")
    task: str = Field(..., description="ML task (stt, diarization, etc.)")
    trust_remote_code: bool = Field(False, description="Whether to trust remote code")

    class Config:
        """Pydantic model configuration."""

        extra = "allow"  # Allow additional fields for dataset-specific options


class DatasetSettings(BaseModel):
    """
    Dataset management settings for the evaluation framework.

    Attributes
    ----------
    base_download_dir : Path, optional
        Base directory for downloading and storing datasets.
    cache_dir : Path, optional
        Cache directory for Hugging Face datasets library.
    sample_size : int, optional
        Number of samples to save in JSON files for inspection.
    save_samples : bool, optional
        Whether to save sample files for dataset inspection.
    force_redownload : bool, optional
        Whether to force re-download datasets even if already cached, by default False.
        Useful for getting latest dataset versions or recovering from corrupted downloads.

    Notes
    -----
    - Directories are created automatically if they don't exist
    - Cache directory should have sufficient space for dataset downloads
    - Sample size affects both storage requirements and inspection utility
    - Force redownload should be used sparingly to avoid unnecessary network usage
    - Settings can be overridden per-download operation if needed

    See Also
    --------
    DatasetConfig : Configuration for individual datasets
    DatasetDownloader : Main class that uses these settings
    """

    base_download_dir: Path = Field(default=Path("datasets"), description="Base directory for downloading datasets")
    cache_dir: Path = Field(default=Path("datasets/.cache"), description="Cache directory for Hugging Face datasets")
    sample_size: int = Field(default=100, ge=1, le=10000, description="Number of samples to save in JSON files for inspection")
    save_samples: bool = Field(default=True, description="Whether to save sample files for inspection")
    force_redownload: bool = Field(default=False, description="Whether to force re-download even if cached")


def get_dataset_configs() -> dict[str, DatasetConfig]:
    """
    Get all configured datasets for the evaluation framework.

    Returns
    -------
    dict[str, DatasetConfig]
        Dictionary mapping dataset keys to their configuration objects.
    """
    return {
        # Speech-to-Text Datasets
        "arabic_common_voice": DatasetConfig(
            name="Arabic Common Voice 17.0",
            repo_id="Geethuzzz/common_voice_17_0_arabic_New_cleaned",
            subset=None,
            splits=["train", "validation", "test"],
            local_dir="stt/arabic_common_voice",
            description="Cleaned Arabic Common Voice dataset for speech recognition evaluation",
            language="arabic",
            task="stt",
            trust_remote_code=True,
        ),
        "librispeech_clean": DatasetConfig(
            name="LibriSpeech Clean",
            repo_id="openslr/librispeech_asr",
            subset="clean",
            splits=["train.clean.100", "train.clean.360", "validation.clean", "test.clean"],
            local_dir="stt/librispeech_clean",
            description="LibriSpeech clean subset for English speech recognition evaluation",
            language="english",
            task="stt",
            trust_remote_code=True,
        ),
        "librispeech_other": DatasetConfig(
            name="LibriSpeech Other",
            repo_id="openslr/librispeech_asr",
            subset="other",
            splits=["train.other.500", "validation.other", "test.other"],
            local_dir="stt/librispeech_other",
            description="LibriSpeech other subset with more challenging audio for evaluation",
            language="english",
            task="stt",
            trust_remote_code=True,
        ),
        "common_voice_en": DatasetConfig(
            name="Common Voice English",
            repo_id="mozilla-foundation/common_voice_17_0",
            subset="en",
            splits=["train", "validation", "test"],
            local_dir="stt/common_voice_english",
            description="Mozilla Common Voice English dataset for speech recognition evaluation",
            language="english",
            task="stt",
            trust_remote_code=False,
        ),
        # Speaker Diarization Datasets
        "voxceleb1": DatasetConfig(
            name="VoxCeleb1",
            repo_id="facebook/voxceleb1",
            subset=None,
            splits=["train", "test"],
            local_dir="diarization/voxceleb1",
            description="Speaker verification dataset for diarization evaluation",
            language="english",
            task="diarization",
            trust_remote_code=False,
        ),
        "callhome_diarization": DatasetConfig(
            name="CallHome Diarization",
            repo_id="pyannote/callhome-diarization",
            subset=None,
            splits=["train", "development", "test"],
            local_dir="diarization/callhome",
            description="Telephone conversation diarization evaluation dataset",
            language="english",
            task="diarization",
            trust_remote_code=False,
        ),
        "ami_diarization": DatasetConfig(
            name="AMI Meeting Corpus",
            repo_id="edinburghcstr/ami",
            subset="ihm",  # Individual headset microphone
            splits=["train", "validation", "test"],
            local_dir="diarization/ami_meeting",
            description="Meeting diarization dataset for evaluation",
            language="english",
            task="diarization",
            trust_remote_code=False,
        ),
        # Multi-language and specialized datasets
        "multilingual_librispeech": DatasetConfig(
            name="Multilingual LibriSpeech",
            repo_id="facebook/multilingual_librispeech",
            subset="german",  # Can be changed to other languages
            splits=["train", "dev", "test"],
            local_dir="stt/multilingual_librispeech_de",
            description="German subset of Multilingual LibriSpeech for cross-lingual evaluation",
            language="german",
            task="stt",
            trust_remote_code=False,
        ),
    }


def get_datasets_by_task(task: str) -> dict[str, DatasetConfig]:
    """
    Get datasets filtered by machine learning task type.

    Parameters
    ----------
    task : str
        Machine learning task identifier (e.g., "stt", "diarization").

    Returns
    -------
    dict[str, DatasetConfig]
        Dictionary mapping dataset keys to configuration objects for matching task.
    """
    all_configs = get_dataset_configs()
    return {key: config for key, config in all_configs.items() if config.task == task}


def get_datasets_by_language(language: str) -> dict[str, DatasetConfig]:
    """
    Get datasets filtered by primary language.

    Parameters
    ----------
    language : str
        Language identifier (e.g., "arabic", "english").

    Returns
    -------
    dict[str, DatasetConfig]
        Dictionary mapping dataset keys to configuration objects for matching language.
    """
    all_configs = get_dataset_configs()
    return {key: config for key, config in all_configs.items() if config.language and config.language.lower() == language.lower()}


def get_eval_recommended_datasets() -> dict[str, DatasetConfig]:
    """
    Get curated recommended datasets for evaluation testing.

    Returns
    -------
    dict[str, DatasetConfig]
        Dictionary mapping dataset keys to recommended dataset configurations.
    """
    all_configs = get_dataset_configs()

    # Recommended datasets for comprehensive evaluation
    recommended_keys = [
        "arabic_common_voice",  # Arabic STT
        "librispeech_clean",  # English STT (clean)
        "callhome_diarization",  # Diarization evaluation
    ]

    return {key: all_configs[key] for key in recommended_keys if key in all_configs}
