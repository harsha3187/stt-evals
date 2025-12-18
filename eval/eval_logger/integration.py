#!/usr/bin/env python3
"""Standalone evaluation logging integration script.

This script can be used to push existing evaluation results to cloud platforms
without re-running the evaluations. It supports both individual result files
and batch processing of multiple results.

Usage:
    # Push a single result file
    python -m eval.eval_logger.integration --result-file results/stt_evaluation.json

    # Push all results from a directory
    python -m eval.eval_logger.integration --results-dir results/

    # Push with specific experiment name
    python -m eval.eval_logger.integration --result-file results/stt_evaluation.json --experiment-name my-experiment

    # Check cloud configuration
    python -m eval.eval_logger.integration --check-config

    # List existing experiments
    python -m eval.eval_logger.integration --list-experiments
"""

import argparse
import json
import logging
from pathlib import Path
import sys

from eval.config import EvaluationSettings
from eval.eval_logger.main import EvaluationLogger

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_result_file(result_file: Path) -> dict | None:
    """Load evaluation result from file.

    Parameters
    ----------
    result_file : Path
        Path to the result JSON file

    Returns
    -------
    Dict, optional
        Loaded result data or None if failed

    Examples
    --------
    >>> result_data = load_result_file(Path("results/stt_eval.json"))
    >>> if result_data:
    ...     print(f"Loaded {result_data.get('task_type')} result")
    """
    try:
        with open(result_file) as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load result file {result_file}: {e}")
        return None


def find_dataset_files(result_data: dict, base_dir: Path) -> dict[str, Path]:  # noqa: C901
    """Find dataset files associated with a result.

    This function searches for common dataset file patterns based on the
    task type to automatically associate datasets with evaluation results.

    Parameters
    ----------
    result_data : Dict
        Result data dictionary containing task type information
    base_dir : Path
        Base directory to search for dataset files

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping dataset names to file paths

    Examples
    --------
    >>> result_data = {"task_type": "stt", "language": "en"}
    >>> datasets = find_dataset_files(result_data, Path("./datasets"))
    >>> print(list(datasets.keys()))  # ['references', 'hypotheses']
    """
    dataset_paths = {}
    task_type = result_data.get("task_type", "")

    # Look for common dataset file patterns
    if task_type == "stt":
        # Look for reference and hypothesis files
        for pattern in ["*ref*", "*reference*", "*references*"]:
            files = list(base_dir.glob(f"**/{pattern}.txt")) + list(base_dir.glob(f"**/{pattern}.jsonl"))
            if files:
                dataset_paths["references"] = files[0]
                break

        for pattern in ["*hyp*", "*hypothesis*", "*hypotheses*"]:
            files = list(base_dir.glob(f"**/{pattern}.txt")) + list(base_dir.glob(f"**/{pattern}.jsonl"))
            if files:
                dataset_paths["hypotheses"] = files[0]
                break

    elif task_type == "diarization":
        # Look for RTTM files
        for pattern in ["*ref*", "*reference*", "*ground_truth*"]:
            files = list(base_dir.glob(f"**/{pattern}.rttm"))
            if files:
                dataset_paths["ground_truth_rttm"] = files[0]
                break

        for pattern in ["*hyp*", "*hypothesis*", "*generated*"]:
            files = list(base_dir.glob(f"**/{pattern}.rttm"))
            if files:
                dataset_paths["generated_rttm"] = files[0]
                break

    elif task_type == "rag":
        # Look for RAG dataset files
        for file_type in ["queries", "corpus", "qrels", "answers"]:
            files = list(base_dir.glob(f"**/*{file_type}*.jsonl"))
            if files:
                dataset_paths[file_type] = files[0]

    return dataset_paths


def push_single_result(result_file: Path, evaluation_logger: EvaluationLogger, find_datasets: bool = True) -> bool:
    """Push a single result file to cloud platform.

    Parameters
    ----------
    result_file : Path
        Path to the result file
    evaluation_logger : EvaluationLogger
        Evaluation logger instance
    find_datasets : bool, optional
        Whether to automatically find dataset files, by default True

    Returns
    -------
    bool
        True if successful, False otherwise

    Examples
    --------
    >>> logger = EvaluationLogger()
    >>> success = push_single_result(Path("results/eval.json"), logger)
    >>> if success:
    ...     print("Successfully uploaded to cloud")
    """
    logger.info(f"Processing result file: {result_file}")

    # Load result data
    result_data = load_result_file(result_file)
    if not result_data:
        return False

    # Find dataset files if requested
    dataset_paths = {}
    if find_datasets:
        # Look in the same directory and parent directories
        search_dirs = [result_file.parent, result_file.parent.parent]
        for search_dir in search_dirs:
            if search_dir.exists():
                dataset_paths.update(find_dataset_files(result_data, search_dir))

    # Convert result data to appropriate model type
    # For now, we'll create a generic result object
    class GenericResult:
        """Generic result wrapper for cloud logging."""

        def __init__(self, data):
            self.__dict__.update(data)

        def dict(self):
            return self.__dict__

    result_obj = GenericResult(result_data.get("results", result_data))

    # Log to cloud platform
    try:
        run_id = evaluation_logger.log_evaluation_result(result=result_obj, dataset_paths=dataset_paths, session_id=result_data.get("session_id"))

        if run_id:
            logger.info(f"✓ Successfully logged to cloud - Run ID: {run_id}")
            if dataset_paths:
                logger.info(f"  Uploaded {len(dataset_paths)} dataset file(s)")
            return True
        else:
            logger.error("Failed to log to cloud")
            return False

    except Exception as e:
        logger.error(f"Error logging to cloud: {e}")
        return False


def push_results_directory(results_dir: Path, evaluation_logger: EvaluationLogger, pattern: str = "**/*.json") -> dict[str, int]:
    """Push all result files from a directory to cloud platform.

    Parameters
    ----------
    results_dir : Path
        Directory containing result files
    evaluation_logger : EvaluationLogger
        Evaluation logger instance
    pattern : str, optional
        File pattern to match, by default "**/*.json"

    Returns
    -------
    Dict[str, int]
        Dictionary with counts of successful and failed uploads

    Examples
    --------
    >>> logger = EvaluationLogger()
    >>> results = push_results_directory(Path("./results"), logger)
    >>> print(f"Success: {results['successful']}, Failed: {results['failed']}")
    """
    logger.info(f"Processing results directory: {results_dir}")

    result_files = list(results_dir.glob(pattern))
    if not result_files:
        logger.warning(f"No result files found in {results_dir} with pattern {pattern}")
        return {"successful": 0, "failed": 0}

    logger.info(f"Found {len(result_files)} result file(s)")

    successful = 0
    failed = 0

    for result_file in result_files:
        if push_single_result(result_file, evaluation_logger):
            successful += 1
        else:
            failed += 1

    logger.info(f"Batch processing complete: {successful} successful, {failed} failed")
    return {"successful": successful, "failed": failed}


def check_cloud_config() -> bool:
    """Check cloud logging configuration and connectivity.

    Returns
    -------
    bool
        True if configuration is valid and connection works

    Examples
    --------
    >>> if check_cloud_config():
    ...     print("Cloud logging is properly configured")
    ... else:
    ...     print("Configuration issues detected")
    """
    logger.info("Checking cloud logging configuration...")

    try:
        # Load settings
        settings = EvaluationSettings()

        # Check if cloud logging is enabled using environment-aware config
        config = settings.get_cloud_logging_config()
        if not config.enable_logging:
            logger.warning("Cloud logging is disabled in configuration")
            logger.info("To enable, set EVAL_CLOUD_ENABLE_LOGGING=true in your .env file")
            return False

        # Check configuration (already loaded above)
        if not config.is_valid():
            logger.error("Cloud logging configuration is incomplete:")
            if not config.workspace_name:
                logger.error("  - Missing workspace name (EVAL_CLOUD_WORKSPACE_NAME)")
            if not config.resource_group:
                logger.error("  - Missing resource group (EVAL_CLOUD_RESOURCE_GROUP)")
            if not config.subscription_id:
                logger.error("  - Missing subscription ID (EVAL_CLOUD_SUBSCRIPTION_ID)")
            return False

        # Test connection
        evaluation_logger = EvaluationLogger(config)
        if not evaluation_logger.is_enabled():
            logger.error("Failed to connect to cloud workspace")
            return False

        logger.info("✓ Cloud logging configuration is valid and connection successful")
        logger.info(f"  Workspace: {config.workspace_name}")
        logger.info(f"  Resource Group: {config.resource_group}")
        logger.info(f"  Subscription: {config.subscription_id}")
        logger.info(f"  Experiment: {config.experiment_name}")

        return True

    except Exception as e:
        logger.error(f"Error checking cloud logging configuration: {e}")
        return False


def list_experiments() -> bool:
    """List existing cloud experiments.

    Returns
    -------
    bool
        True if successful

    Examples
    --------
    >>> if list_experiments():
    ...     print("Successfully retrieved experiment list")
    """
    logger.info("Listing cloud experiments...")

    try:
        settings = EvaluationSettings()
        evaluation_logger = EvaluationLogger(settings.get_cloud_logging_config())

        if not evaluation_logger.is_enabled():
            logger.error("Cloud logging is not properly configured")
            return False

        experiments = evaluation_logger.list_experiments()
        if experiments is None:
            logger.error("Failed to retrieve experiments")
            return False

        if not experiments:
            logger.info("No experiment runs found")
            return True

        logger.info(f"Found {len(experiments)} experiment run(s):")

        for i, exp in enumerate(experiments[:10], 1):  # Show first 10
            run_id = exp.get("run_id", "N/A")
            task_type = exp.get("tags.task_type", "N/A")
            timestamp = exp.get("start_time", "N/A")
            status = exp.get("status", "N/A")

            logger.info(f"  {i}. Run ID: {run_id}")
            logger.info(f"     Task: {task_type}, Status: {status}")
            logger.info(f"     Started: {timestamp}")

            # Show key metrics if available
            metrics = []
            for col in exp.keys():
                if col.startswith("metrics.") and exp[col] is not None:
                    metric_name = col.replace("metrics.", "")
                    metrics.append(f"{metric_name}: {exp[col]:.4f}")

            if metrics:
                logger.info(f"     Metrics: {', '.join(metrics[:3])}")  # Show first 3 metrics
            logger.info("")

        if len(experiments) > 10:
            logger.info(f"... and {len(experiments) - 10} more runs")

        return True

    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        return False


def main():  # noqa: C901
    """Main entry point for the integration script.

    Handles command-line arguments and dispatches to appropriate functions.
    """
    parser = argparse.ArgumentParser(description="Cloud logging integration for evaluation results", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)

    # Mutually exclusive group for different operations
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--result-file", type=Path, help="Path to a single result JSON file to upload")
    group.add_argument("--results-dir", type=Path, help="Directory containing result files to upload")
    group.add_argument("--check-config", action="store_true", help="Check cloud logging configuration and connectivity")
    group.add_argument("--list-experiments", action="store_true", help="List existing cloud experiments")

    # Optional arguments
    parser.add_argument("--experiment-name", help="Custom experiment name (overrides config)")
    parser.add_argument("--no-datasets", action="store_true", help="Don't automatically search for dataset files")
    parser.add_argument("--pattern", default="**/*.json", help="File pattern for batch processing (default: **/*.json)")

    args = parser.parse_args()

    try:
        # Handle config check
        if args.check_config:
            success = check_cloud_config()
            sys.exit(0 if success else 1)

        # Handle list experiments
        if args.list_experiments:
            success = list_experiments()
            sys.exit(0 if success else 1)

        # Load cloud logging configuration
        settings = EvaluationSettings()
        cloud_config = settings.get_cloud_logging_config()

        # Override experiment name if provided
        if args.experiment_name:
            cloud_config.experiment_name = args.experiment_name

        # Initialize evaluation logger
        evaluation_logger = EvaluationLogger(cloud_config)

        if not evaluation_logger.is_enabled():
            logger.error("Cloud logging is not properly configured")
            logger.info("Run with --check-config to diagnose configuration issues")
            sys.exit(1)

        # Handle single result file
        if args.result_file:
            if not args.result_file.exists():
                logger.error(f"Result file not found: {args.result_file}")
                sys.exit(1)

            success = push_single_result(args.result_file, evaluation_logger, find_datasets=not args.no_datasets)
            sys.exit(0 if success else 1)

        # Handle results directory
        if args.results_dir:
            if not args.results_dir.exists():
                logger.error(f"Results directory not found: {args.results_dir}")
                sys.exit(1)

            if not args.results_dir.is_dir():
                logger.error(f"Path is not a directory: {args.results_dir}")
                sys.exit(1)

            results = push_results_directory(args.results_dir, evaluation_logger, args.pattern)

            success = results["failed"] == 0
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
