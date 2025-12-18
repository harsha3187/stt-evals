"""Evaluation Logging Module.

This module provides cloud logging integration for evaluation results,
supporting Azure ML and MLflow for experiment tracking and artifact management.

The module follows the same structure as the STT evaluation module,
with main functionality in main.py and configuration in config.py.

Features:
- Cloud logging to Azure ML workspaces
- MLflow experiment tracking
- Automatic dataset artifact upload
- Support for all evaluation task types (STT, Diarization, RAG)
- Environment-based configuration
- Robust error handling and logging

Usage:
    Basic logging:
    >>> from eval.eval_logger import log_evaluation_result
    >>> run_id = log_evaluation_result(stt_result)

    Advanced usage:
    >>> from eval.eval_logger import EvaluationLogger
    >>> from eval.eval_logger.config import CloudLoggingConfig
    >>>
    >>> config = CloudLoggingConfig.from_env()
    >>> logger = EvaluationLogger(config)
    >>> run_id = logger.log_evaluation_result(result, dataset_paths)

    Integration script:
    >>> python -m eval.eval_logger.integration --check-config
    >>> python -m eval.eval_logger.integration --result-file results/eval.json

Environment Variables:
    EVAL_CLOUD_WORKSPACE_NAME: Azure ML workspace name
    EVAL_CLOUD_RESOURCE_GROUP: Azure resource group name
    EVAL_CLOUD_SUBSCRIPTION_ID: Azure subscription ID
    EVAL_CLOUD_EXPERIMENT_NAME: MLflow experiment name (optional)
    EVAL_CLOUD_ENABLE_LOGGING: Enable cloud logging (true/false)
"""

__version__ = "1.0.0"
__author__ = "Oryx CAP Upskilling Team"

# Import main evaluation logging classes and functions
from eval.eval_logger.main import (
    EvaluationLogger,
    get_evaluation_logger,
    log_evaluation_result,
)

__all__ = [
    "EvaluationLogger",
    "get_evaluation_logger",
    "log_evaluation_result",
]
