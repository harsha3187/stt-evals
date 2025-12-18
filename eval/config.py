"""Configuration management using Pydantic Settings.

This module provides a robust configuration system following modern Python practices:
- Type-safe configuration with Pydantic
- Environment variable support
- .env file loading
- Validation and default values
- Separation of concerns between different evaluation tasks
"""

from functools import lru_cache
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class STTConfig(BaseModel):
    """Configuration for Speech-to-Text evaluation."""

    ground_truth_file: Path = Field(default=Path("datasets/stt/ground_truth.txt"), description="Path to ground truth transcriptions file")
    generated_file: Path = Field(default=Path("datasets/stt/generated.txt"), description="Path to generated transcriptions file")
    language: Literal["en", "ar"] = Field(default="en", description="Language code for evaluation")
    output_file: Path = Field(default=Path("results/stt_results.json"), description="Output file for STT evaluation results")


class DiarizationConfig(BaseModel):
    """Configuration for Speaker Diarization evaluation."""

    ground_truth_rttm: Path = Field(default=Path("datasets/diarization/ground_truth.rttm"), description="Path to ground truth RTTM file")
    generated_rttm: Path = Field(default=Path("datasets/diarization/generated.rttm"), description="Path to generated RTTM file")
    collar: float = Field(default=0.25, ge=0.0, le=1.0, description="Tolerance collar in seconds")
    output_file: Path = Field(default=Path("results/diarization_results.json"), description="Output file for diarization evaluation results")


class DatasetDownloadConfig(BaseModel):
    """Configuration for dataset downloading and management."""

    base_download_dir: Path = Field(default=Path("datasets"), description="Base directory for downloading datasets")
    cache_dir: Path = Field(default=Path("eval_datasets/.cache"), description="Cache directory for Hugging Face datasets")
    sample_size: int = Field(default=100, ge=1, le=10000, description="Number of samples to save in JSON files for inspection")
    save_samples: bool = Field(default=True, description="Whether to save sample files for inspection")
    force_redownload: bool = Field(default=False, description="Whether to force re-download even if cached")
    auto_download: bool = Field(default=False, description="Whether to automatically download missing datasets")


class RAGConfig(BaseModel):
    """Configuration for RAG (Retrieval-Augmented Generation) evaluation.

    Simplified configuration focusing on summarization quality metrics only.
    Follows the same pattern as STT evaluation for consistency.
    """

    # Required files for summarization evaluation
    ground_truth_file: Path = Field(default=Path("datasets/rag/ground_truth.jsonl"), description="Path to ground truth answers file (JSONL format with 'text' field)")
    generated_file: Path = Field(default=Path("datasets/rag/generated.jsonl"), description="Path to generated answers file (JSONL format with 'text' field)")

    # Evaluation parameters
    language: Literal["en", "ar"] = Field(default="en", description="Language code for evaluation")

    # Output configuration
    output_file: Path = Field(default=Path("results/rag_results.json"), description="Output file for RAG evaluation results")


class CloudLoggingConfig(BaseSettings):
    """Configuration for cloud logging integration with Azure ML and MLflow.

    This configuration manages connection details for Azure ML workspace
    and MLflow experiment tracking, providing comprehensive logging capabilities
    for evaluation results, metrics, and artifacts.

    Environment Variables (via EVAL_CLOUD_ prefix)
    ----------
    EVAL_CLOUD_WORKSPACE_NAME : str
        Azure ML workspace name
    EVAL_CLOUD_RESOURCE_GROUP : str
        Azure resource group name
    EVAL_CLOUD_SUBSCRIPTION_ID : str
        Azure subscription ID
    EVAL_CLOUD_EXPERIMENT_NAME : str, optional
        MLflow experiment name, default: "evaluation-experiments"
    EVAL_CLOUD_ENABLE_LOGGING : bool, optional
        Enable cloud logging, default: False

    Examples
    --------
    >>> settings = EvaluationSettings()
    >>> if settings.cloud_logging.is_valid():
    ...     from eval.eval_logger import EvaluationLogger
    ...     logger = EvaluationLogger(settings.cloud_logging)
    """

    model_config = SettingsConfigDict(env_file=Path(__file__).parent / ".env", env_file_encoding="utf-8", env_prefix="EVAL_CLOUD_", case_sensitive=False, extra="ignore")

    workspace_name: str = Field(default="", description="Azure ML workspace name")
    resource_group: str = Field(default="", description="Azure resource group name")
    subscription_id: str = Field(default="", description="Azure subscription ID")
    experiment_name: str = Field(default="evaluation-experiments", description="MLflow experiment name")
    enable_logging: bool = Field(default=False, description="Enable cloud logging for evaluation metrics and artifacts")

    def is_valid(self) -> bool:
        """Check if configuration has all required fields for cloud logging.

        Returns
        -------
        bool
            True if all required fields are present and logging is enabled
        """
        return bool(self.workspace_name and self.resource_group and self.subscription_id and self.enable_logging)


class EvaluationSettings(BaseSettings):
    """Main evaluation settings with environment variable support.

    This class automatically loads configuration from:
    1. Environment variables
    2. .env file in eval directory
    3. Default values defined in the model

    Examples
    --------
    Environment variables:
        EVAL_LOG_LEVEL=DEBUG
        EVAL_STT__LANGUAGE=ar
        EVAL_DIARIZATION__COLLAR=0.5
        EVAL_DATASETS__BASE_DOWNLOAD_DIR=datasets
        EVAL_CLOUD_ENABLE_LOGGING=true
        EVAL_CLOUD_WORKSPACE_NAME=my-workspace

    .env file (eval/.env):
        EVAL_LOG_LEVEL=INFO
        EVAL_STT__REFERENCES_FILE=datasets/stt/references.txt
        EVAL_COMBINED_OUTPUT_FILE=results/combined_evaluation_results.json
        EVAL_DATASETS__CACHE_DIR=datasets/.cache
        EVAL_CLOUD_ENABLE_LOGGING=true
        EVAL_CLOUD_WORKSPACE_NAME=my-workspace
        EVAL_CLOUD_RESOURCE_GROUP=my-rg
        EVAL_CLOUD_SUBSCRIPTION_ID=12345678-1234-1234-1234-123456789012
    """

    model_config = SettingsConfigDict(env_file=Path(__file__).parent / ".env", env_file_encoding="utf-8", env_prefix="EVAL_", env_nested_delimiter="__", case_sensitive=False, extra="ignore")

    # Logging configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", description="Logging level for the application")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Format string for log messages")

    # Task-specific configurations
    stt: STTConfig = Field(default_factory=STTConfig, description="Speech-to-Text evaluation configuration")
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig, description="Speaker Diarization evaluation configuration")
    rag: RAGConfig = Field(default_factory=RAGConfig, description="RAG evaluation configuration")

    # Dataset management configuration
    datasets: DatasetDownloadConfig = Field(default_factory=DatasetDownloadConfig, description="Dataset downloading and management configuration")

    # Cloud logging integration configuration
    cloud_logging: CloudLoggingConfig = Field(default_factory=CloudLoggingConfig, description="Cloud logging integration configuration")

    # General settings
    combined_output_file: Path = Field(default=Path("results/combined_evaluation_results.json"), description="Output file for combined evaluation results")
    results_directory: Path = Field(default=Path("results"), description="Base directory for storing evaluation results")
    datasets_directory: Path = Field(default=Path("datasets"), description="Base directory containing evaluation datasets")

    def model_post_init(self, __context) -> None:
        """Post-initialization setup."""
        # Create directories if they don't exist
        self.results_directory.mkdir(parents=True, exist_ok=True)
        self.datasets_directory.mkdir(parents=True, exist_ok=True)

        # Create dataset download directories
        self.datasets.base_download_dir.mkdir(parents=True, exist_ok=True)
        self.datasets.cache_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(level=getattr(logging, self.log_level), format=self.log_format)

        logger.info(f"Configuration loaded with log level: {self.log_level}")
        logger.debug(f"Results directory: {self.results_directory}")
        logger.debug(f"Datasets directory: {self.datasets_directory}")
        logger.debug(f"Dataset download directory: {self.datasets.base_download_dir}")
        logger.debug(f"Dataset cache directory: {self.datasets.cache_dir}")
        if self.cloud_logging.is_valid():
            logger.debug(f"Cloud logging enabled for workspace: {self.cloud_logging.workspace_name}")

    def validate_paths(self) -> bool:
        """Validate that all required paths exist.

        Returns
        -------
        bool
            True if all paths are valid, False otherwise
        """
        required_paths = [
            self.stt.ground_truth_file,
            self.stt.generated_file,
            self.diarization.ground_truth_rttm,
            self.diarization.generated_rttm,
            self.rag.ground_truth_file,
            self.rag.generated_file,
        ]

        all_valid = True
        for path in required_paths:
            if not path.exists():
                logger.warning(f"Required file not found: {path}")
                all_valid = False
            else:
                logger.debug(f"Found required file: {path}")

        return all_valid

    def get_cloud_logging_config(self) -> CloudLoggingConfig:
        """Get cloud logging configuration.

        Returns
        -------
        CloudLoggingConfig
            Cloud logging configuration
        """
        return self.cloud_logging

    def get_summary(self) -> str:
        """Get a human-readable summary of current configuration.

        Returns
        -------
        str
            Formatted configuration summary
        """
        return f"""Evaluation Configuration Summary:
├── Logging: {self.log_level}
├── Results Directory: {self.results_directory}
├── Datasets Directory: {self.datasets_directory}
├── Combined Output: {self.combined_output_file}
├── STT Configuration:
│   ├── Ground Truth: {self.stt.ground_truth_file}
│   ├── Generated: {self.stt.generated_file}
│   ├── Language: {self.stt.language}
│   └── Output: {self.stt.output_file}
├── Diarization Configuration:
│   ├── Ground Truth RTTM: {self.diarization.ground_truth_rttm}
│   ├── Generated RTTM: {self.diarization.generated_rttm}
│   ├── Collar: {self.diarization.collar}s
│   └── Output: {self.diarization.output_file}
└── RAG Configuration:
    ├── Ground Truth: {self.rag.ground_truth_file}
    ├── Generated: {self.rag.generated_file}
    ├── Language: {self.rag.language}
    └── Output: {self.rag.output_file}"""


# Global settings instance


@lru_cache
def get_settings() -> EvaluationSettings:
    """Get the global settings instance.

    Returns
    -------
    EvaluationSettings
        The configured settings instance
    """
    return EvaluationSettings()


def reload_settings() -> EvaluationSettings:
    """Reload settings from environment and .env file.

    Returns
    -------
    EvaluationSettings
        The newly loaded settings instance
    """
    global settings
    settings = EvaluationSettings()
    return settings


def _format_field_value(field_info) -> str:
    """Format a field's default value for .env file."""
    default = field_info.default
    if default is None:
        return ""
    if isinstance(default, Path):
        return str(default)
    if isinstance(default, bool):
        return str(default).lower()
    if isinstance(default, list):
        return ",".join(map(str, default))
    return str(default)


def _get_field_line(field_name: str, env_var: str, value: str, critical_fields: set) -> str:
    """Get the formatted line for a field."""
    is_critical = field_name in critical_fields or field_name.endswith(("_key", "_endpoint"))

    if not is_critical:
        return f"# {env_var}={value}"

    # Critical fields handling
    field_lower = field_name.lower()
    if "key" in field_lower or "token" in field_lower:
        return f"{env_var}=your-{field_name.replace('_', '-')}-here"
    elif "endpoint" in field_lower:
        return f"{env_var}=https://your-endpoint-here.com/"
    else:
        return f"{env_var}={value}"


def _add_config_section(lines: list, title: str, model_class, prefix: str, critical_fields: set, include_all: bool) -> None:
    """Add a configuration section from a Pydantic model."""
    lines.append(f"# {title}")
    lines.append("")

    for field_name, field_info in model_class.model_fields.items():
        # Skip if not including all and not a critical field
        if not include_all and field_name not in critical_fields:
            continue

        # Add description as comment
        if field_info.description:
            lines.append(f"# {field_info.description}")

        # Format environment variable name and value
        env_var = f"{prefix}{field_name}".upper()
        value = _format_field_value(field_info)

        # Add the field line
        lines.append(_get_field_line(field_name, env_var, value, critical_fields))
        lines.append("")


def generate_env_example(output_path: Path = Path("eval/.env.example"), include_all: bool = False) -> None:
    """Generate .env.example file from Pydantic model definitions.

    This function auto-generates an .env.example file from the Pydantic configuration
    models, ensuring the example file stays in sync with the code.

    Parameters
    ----------
    output_path : Path
        Path where .env.example should be written
    include_all : bool
        If True, include all fields. If False, only include critical/required fields
        (API keys, endpoints, etc.)

    Examples
    --------
    Generate minimal .env.example with only critical fields:
    >>> generate_env_example()

    Generate complete .env.example with all fields:
    >>> generate_env_example(include_all=True)
    """
    lines = [
        "# Evaluation Framework Configuration",
        "# Auto-generated from eval/config.py Pydantic models",
        "# Copy this file to .env and update with your values",
        "",
        "# This file is automatically generated to stay in sync with code.",
        "# To regenerate: python -m eval.config --generate-env-example",
        "",
    ]

    # Critical fields that should always be in the minimal .env.example
    critical_fields = {
        "log_level",
        "results_directory",
        # Azure/OpenAI API configuration
        "use_azure",
        "azure_endpoint",
        "azure_api_key",
        "azure_deployment",
        "azure_embeddings_deployment",
        "ragas_model",
        "judge_model",
        # Cloud logging configuration
        "workspace_name",
        "resource_group",
        "subscription_id",
        "enable_logging",
        "experiment_name",
    }

    # Main settings
    _add_config_section(lines, "General Settings", EvaluationSettings, "EVAL_", critical_fields, include_all)

    # STT Configuration
    _add_config_section(lines, "Speech-to-Text Configuration", STTConfig, "EVAL_STT__", critical_fields, include_all)

    # Diarization Configuration
    _add_config_section(lines, "Diarization Configuration", DiarizationConfig, "EVAL_DIARIZATION__", critical_fields, include_all)

    # RAG Configuration
    _add_config_section(lines, "RAG Configuration", RAGConfig, "EVAL_RAG__", critical_fields, include_all)

    # Cloud Logging Configuration
    _add_config_section(lines, "Cloud Logging Configuration", CloudLoggingConfig, "EVAL_CLOUD_", critical_fields, include_all)

    # Add example usage section
    lines.extend(
        [
            "",
            "# ===== Example Configurations =====",
            "",
            "# Example 1: Azure OpenAI with GPT-4",
            "# EVAL_RAG__USE_AZURE=true",
            "# EVAL_RAG__AZURE_ENDPOINT=https://my-resource.openai.azure.com/",
            "# EVAL_RAG__AZURE_API_KEY=your-key",
            "# EVAL_RAG__AZURE_DEPLOYMENT=gpt-4-deployment",
            "# EVAL_RAG__AZURE_EMBEDDINGS_DEPLOYMENT=text-embedding-3-large",
            "",
            "# Example 2: Standard OpenAI",
            "# EVAL_RAG__USE_AZURE=false",
            "# OPENAI_API_KEY=sk-proj-...",
            "# EVAL_RAG__JUDGE_MODEL=gpt-4",
            "# EVAL_RAG__RAGAS_MODEL=gpt-3.5-turbo",
            "",
            "# Example 3: Azure ML Cloud Logging",
            "# EVAL_CLOUD_ENABLE_LOGGING=true",
            "# EVAL_CLOUD_WORKSPACE_NAME=my-ml-workspace",
            "# EVAL_CLOUD_RESOURCE_GROUP=my-resource-group",
            "# EVAL_CLOUD_SUBSCRIPTION_ID=12345678-1234-1234-1234-123456789012",
            "# EVAL_CLOUD_EXPERIMENT_NAME=evaluation-experiments",
            "",
        ]
    )

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info(f"✅ Generated {output_path}")
    logger.info(f"   Fields included: {'all' if include_all else 'critical only'}")


def validate_env_config() -> bool:
    """Validate current environment configuration.

    Checks if .env file exists and all required fields are set.

    Returns
    -------
    bool
        True if configuration is valid, False otherwise
    """
    import os

    logger.info("🔍 Validating environment configuration...")

    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        logger.warning("⚠️  No .env file found")
        logger.info("   Create one with: cp eval/.env.example .env")
        return False

    logger.info(f"✅ Found .env file: {env_file}")

    # Try to load settings
    try:
        settings = EvaluationSettings()
        logger.info("✅ Configuration loaded successfully")

        # Check critical paths
        logger.info("\n📂 Checking paths...")
        if settings.validate_paths():
            logger.info("✅ All paths are valid")
        else:
            logger.warning("⚠️  Some paths are missing (see warnings above)")

        # Check API keys if RAG is enabled
        if settings.rag.enable_ragas or settings.rag.enable_llm_judge:
            logger.info("\n🔑 Checking API keys...")
            if settings.rag.use_azure:
                if not settings.rag.azure_api_key and not os.getenv("AZURE_OPENAI_API_KEY"):
                    logger.error("❌ Azure OpenAI API key not set")
                    return False
                logger.info("✅ Azure OpenAI configured")
            else:
                if not os.getenv("OPENAI_API_KEY"):
                    logger.error("❌ OpenAI API key not set")
                    return False
                logger.info("✅ OpenAI API key configured")

        logger.info("\n✅ Configuration is valid")
        return True

    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return False


def show_config_diff() -> None:
    """Show difference between current config and defaults."""
    logger.info("📊 Configuration Comparison")
    logger.info("=" * 80)

    settings = get_settings()

    # Validate paths
    if settings.validate_paths():
        logger.info("All required paths are valid")
    else:
        logger.warning("Some required paths are missing")


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "--generate-env-example":
            include_all = "--all" in sys.argv
            generate_env_example(include_all=include_all)
            print("\n✅ Generated eval/.env.example")
            if not include_all:
                print("   (Minimal version with critical fields only)")
                print("   For complete example: python -m eval.config --generate-env-example --all")

        elif command == "--validate":
            is_valid = validate_env_config()
            sys.exit(0 if is_valid else 1)

        elif command == "--diff":
            show_config_diff()

        elif command == "--help":
            print("""
Evaluation Configuration Management

Usage:
  python -m eval.config [COMMAND]

Commands:
  --help                    Show this help message
  --generate-env-example    Generate minimal .env.example (critical fields only)
  --generate-env-example --all  Generate complete .env.example (all fields)
  --validate                Validate current .env configuration
  --diff                    Show differences from default configuration
  (no command)              Show current configuration summary

Examples:
  # Generate minimal .env.example
  python -m eval.config --generate-env-example

  # Generate complete .env.example with all fields
  python -m eval.config --generate-env-example --all

  # Validate current configuration
  python -m eval.config --validate

  # Show config differences
  python -m eval.config --diff

  # Show current config summary
  python -m eval.config
            """)
        else:
            print(f"Unknown command: {command}")
            print("Run: python -m eval.config --help")
            sys.exit(1)
    else:
        # Default: show configuration summary
        settings = get_settings()

        logger.info("=== Evaluation Settings Summary ===")
        logger.info("\n" + settings.get_summary())

        # Validate paths
        logger.info("\n" + "=" * 80)
        if settings.validate_paths():
            logger.info("✅ All required paths are valid")
        else:
            logger.warning("⚠️  Some required paths are missing")

        # Show JSON serialization
        import json

        config_dict = settings.model_dump()
        logger.info("\n" + "=" * 80)
        logger.info("JSON Configuration:")
        logger.info(json.dumps(config_dict, indent=2, default=str))
