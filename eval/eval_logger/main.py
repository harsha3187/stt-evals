"""Evaluation logging service for metrics and artifacts.

This module provides a comprehensive logging service that integrates with cloud platforms
to track evaluation metrics, datasets, and results as experiments.

Features:
- Log evaluation metrics to cloud experiments
- Upload datasets and results as artifacts
- Environment-based configuration
- Support for all evaluation task types (STT, Diarization, RAG)
- Robust error handling and logging
- Async support for better performance
"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

from eval.config import CloudLoggingConfig
from eval.models import (
    CombinedEvaluationResult,
    DiarizationEvaluationResult,
    RAGEvaluationResult,
    STTEvaluationResult,
)

logger = logging.getLogger(__name__)


class EvaluationLogger:
    """Cloud logging service for evaluation metrics and artifacts.

    This service provides integration with Azure ML and MLflow for tracking
    evaluation experiments, metrics, and artifacts across different task types.

    Parameters
    ----------
    config : CloudLoggingConfig, optional
        Cloud logging configuration. If None, loads from environment variables.

    Examples
    --------
    >>> from eval.eval_logger import EvaluationLogger
    >>> from eval.eval_logger.config import CloudLoggingConfig
    >>>
    >>> config = CloudLoggingConfig()
    >>> logger = EvaluationLogger(config)
    >>>
    >>> if logger.is_enabled():
    ...     run_id = logger.log_evaluation_result(stt_result)

    Notes
    -----
    Requires Azure ML SDK and MLflow for cloud functionality. If dependencies
    are not available, logging will be disabled gracefully.
    """

    def __init__(self, config: CloudLoggingConfig | None = None):
        """Initialize evaluation logger.

        Parameters
        ----------
        config : CloudLoggingConfig, optional
            Cloud logging configuration. If None, loads from environment variables.
        """
        self.config = config or CloudLoggingConfig()
        self.ml_client: Any | None = None
        self.mlflow_configured = False

        if not self.config.is_valid():
            logger.warning("Cloud logging configuration is incomplete. Logging will be disabled.")
            return

        if not self.config.enable_logging:
            logger.info("Cloud logging is disabled via configuration.")
            return

        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize cloud clients and MLflow.

        Sets up Azure ML client and configures MLflow tracking URI.
        Handles import errors gracefully if dependencies are not available.
        """
        try:
            # Import Azure ML dependencies here to avoid import errors if not available
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            import mlflow

            # Initialize cloud client
            credential = DefaultAzureCredential()
            self.ml_client = MLClient(credential=credential, subscription_id=self.config.subscription_id, resource_group_name=self.config.resource_group, workspace_name=self.config.workspace_name)

            # Configure MLflow
            workspace_details = self.ml_client.workspaces.get(self.config.workspace_name)
            mlflow_uri = workspace_details.mlflow_tracking_uri
            mlflow.set_tracking_uri(mlflow_uri)

            self.mlflow_configured = True
            logger.info(f"✓ Connected to Cloud Workspace: {self.config.workspace_name}")
            logger.info(f"✓ MLflow tracking URI: {mlflow_uri}")

        except ImportError as e:
            logger.warning(f"Cloud logging dependencies not available: {e}")
            self.ml_client = None
            self.mlflow_configured = False
        except Exception as e:
            logger.error(f"Failed to initialize cloud clients: {e}")
            self.ml_client = None
            self.mlflow_configured = False

    def is_enabled(self) -> bool:
        """Check if cloud logging is enabled and properly configured.

        Returns
        -------
        bool
            True if logging is enabled and all clients are initialized
        """
        return self.config.enable_logging and self.config.is_valid() and self.ml_client is not None and self.mlflow_configured

    def log_evaluation_result(self, result: STTEvaluationResult | DiarizationEvaluationResult | RAGEvaluationResult | CombinedEvaluationResult, dataset_paths: dict[str, Path] | None = None, session_id: str | None = None) -> str | None:
        """Log evaluation result to cloud platform.

        Parameters
        ----------
        result : Union[STTEvaluationResult, DiarizationEvaluationResult, RAGEvaluationResult, CombinedEvaluationResult]
            Evaluation result to log
        dataset_paths : Dict[str, Path], optional
            Dictionary mapping dataset names to file paths
        session_id : str, optional
            Session ID for grouping related runs

        Returns
        -------
        str, optional
            Run ID if successful, None otherwise

        Examples
        --------
        >>> dataset_paths = {"references": Path("refs.txt"), "hypotheses": Path("hyps.txt")}
        >>> run_id = logger.log_evaluation_result(stt_result, dataset_paths, "session_123")
        """
        if not self.is_enabled():
            logger.debug("Cloud logging is disabled or not configured.")
            return None

        try:
            import mlflow

            # Set experiment
            mlflow.set_experiment(self.config.experiment_name)

            # Create run name
            task_type = self._get_task_type(result)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{task_type}_{timestamp}"
            if session_id:
                run_name += f"_{session_id[:8]}"

            # Start MLflow run
            with mlflow.start_run(run_name=run_name) as run:
                # Log metadata
                self._log_metadata(result, session_id)

                # Log metrics
                self._log_metrics(result)

                # Log parameters
                self._log_parameters(result)

                # Upload datasets as artifacts
                if dataset_paths:
                    self._upload_datasets(dataset_paths)

                # Upload result as artifact
                self._upload_result_artifact(result, run_name)

                logger.info(f"✓ Logged evaluation result to cloud: {run_name}")
                logger.info(f"  - Run ID: {run.info.run_id}")

                return run.info.run_id

        except Exception as e:
            logger.error(f"Failed to log evaluation result to cloud: {e}")
            return None

    def _get_task_type(self, result: Any) -> str:
        """Get task type from result object.

        Parameters
        ----------
        result : Any
            Evaluation result object

        Returns
        -------
        str
            Task type identifier
        """
        if isinstance(result, STTEvaluationResult):
            return "stt"
        elif isinstance(result, DiarizationEvaluationResult):
            return "diarization"
        elif isinstance(result, RAGEvaluationResult):
            return "rag"
        elif isinstance(result, CombinedEvaluationResult):
            return "combined"
        else:
            return "unknown"

    def _log_metadata(self, result: Any, session_id: str | None) -> None:
        """Log metadata as tags.

        Parameters
        ----------
        result : Any
            Evaluation result object
        session_id : str, optional
            Session ID for grouping
        """
        import mlflow

        tags = {
            "task_type": self._get_task_type(result),
            "timestamp": datetime.now().isoformat(),
            "evaluation_framework_version": "1.0.0",
        }

        if session_id:
            tags["session_id"] = session_id

        # Add task-specific metadata
        if hasattr(result, "language"):
            tags["language"] = result.language

        if hasattr(result, "dataset_info"):
            if result.dataset_info:
                tags.update({f"dataset_{k}": str(v) for k, v in result.dataset_info.items()})

        mlflow.set_tags(tags)

    def _log_metrics(self, result: Any) -> None:
        """Log evaluation metrics based on result type.

        Parameters
        ----------
        result : Any
            Evaluation result object
        """
        if isinstance(result, STTEvaluationResult):
            self._log_stt_metrics(result)
        elif isinstance(result, DiarizationEvaluationResult):
            self._log_diarization_metrics(result)
        elif isinstance(result, RAGEvaluationResult):
            self._log_rag_metrics(result)
        elif isinstance(result, CombinedEvaluationResult):
            self._log_combined_metrics(result)

    def _log_stt_metrics(self, result: STTEvaluationResult) -> None:
        """Log STT-specific metrics dynamically.

        Parameters
        ----------
        result : STTEvaluationResult
            STT evaluation result object
        """
        self._log_metrics_dynamically(result, "stt")

    def _log_diarization_metrics(self, result: DiarizationEvaluationResult) -> None:
        """Log diarization-specific metrics dynamically.

        Parameters
        ----------
        result : DiarizationEvaluationResult
            Diarization evaluation result object
        """
        self._log_metrics_dynamically(result, "diarization")

    def _log_rag_metrics(self, result: RAGEvaluationResult) -> None:
        """Log RAG-specific metrics dynamically.

        Parameters
        ----------
        result : RAGEvaluationResult
            RAG evaluation result object
        """
        import mlflow

        # Log summarization metrics if available
        if hasattr(result, "summarization_metrics") and result.summarization_metrics:
            self._log_metrics_dynamically(result.summarization_metrics, "rag")

            # Log aggregate metrics if available
            if hasattr(result.summarization_metrics, "get_summary"):
                try:
                    summary = result.summarization_metrics.get_summary()
                    if isinstance(summary, dict):
                        for key, value in summary.items():
                            if isinstance(value, int | float):
                                mlflow.log_metric(f"rag_{key}", float(value))
                except Exception as e:
                    logger.debug(f"Could not log summary metrics: {e}")

        # Log legacy fields if present (for backward compatibility)
        self._log_legacy_rag_metrics(result)

    def _log_metrics_dynamically(self, result: Any, task_prefix: str = "") -> None:
        """Dynamically log metrics from any result object.

        This method introspects the result object and logs all numeric fields as metrics,
        including both normal metrics and rubric scores with _rubric suffix.

        Parameters
        ----------
        result : Any
            Result object with metrics to log
        task_prefix : str, optional
            Prefix to add to metric names, by default ""
        """
        import mlflow

        # Get all attributes from the result object
        if hasattr(result, "__dict__"):
            attributes = result.__dict__
        elif hasattr(result, "model_dump"):
            # Pydantic model
            attributes = result.model_dump()
        elif hasattr(result, "dict"):
            # Legacy Pydantic model
            attributes = result.dict()
        else:
            # Try to convert to dict
            try:
                attributes = vars(result)
            except TypeError:
                logger.warning(f"Could not extract attributes from result object of type {type(result)}")
                return

        # Fields to exclude from metric logging (non-numeric metadata)
        excluded_fields = {"evaluation_timestamp", "language", "sample_count", "session_id", "task_type", "timestamp", "results", "tasks_performed", "execution_time"}

        prefix = f"{task_prefix}_" if task_prefix else ""

        for field_name, value in attributes.items():
            if field_name in excluded_fields:
                continue

            # Handle rubric objects
            if field_name.endswith("_rubric") and hasattr(value, "rubric_score"):
                try:
                    mlflow.log_metric(f"{prefix}{field_name}_score", float(value.rubric_score))
                    if hasattr(value, "rubric_label"):
                        mlflow.set_tag(f"{prefix}{field_name}_label", str(value.rubric_label))
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not log rubric metric {field_name}: {e}")

            # Handle numeric values
            elif isinstance(value, int | float) and value is not None:
                try:
                    mlflow.log_metric(f"{prefix}{field_name}", float(value))
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not log metric {field_name}: {e}")

            # Handle nested objects (like summarization_metrics)
            elif hasattr(value, "__dict__") or hasattr(value, "model_dump") or hasattr(value, "dict"):
                # Recursively log nested metrics with extended prefix
                nested_prefix = f"{prefix}{field_name}_" if not field_name.endswith("_metrics") else prefix
                self._log_metrics_dynamically(value, nested_prefix.rstrip("_"))

    def _log_legacy_rag_metrics(self, result: RAGEvaluationResult) -> None:
        """Log legacy RAG metrics for backward compatibility.

        Parameters
        ----------
        result : RAGEvaluationResult
            RAG evaluation result object
        """
        import mlflow

        # Legacy retrieval metrics
        if hasattr(result, "retrieval_metrics") and result.retrieval_metrics:
            for metrics in result.retrieval_metrics:
                if hasattr(metrics, "k"):
                    k = metrics.k
                    for metric_name in ["ndcg", "mrr", "precision", "recall", "map_score", "hit_rate"]:
                        if hasattr(metrics, metric_name):
                            value = getattr(metrics, metric_name)
                            if isinstance(value, int | float):
                                mlflow.log_metric(f"retrieval_{metric_name}_k{k}", float(value))

        # Legacy RAGAS metrics
        if hasattr(result, "ragas_metrics") and result.ragas_metrics:
            for metric_name in ["faithfulness", "answer_relevance", "context_precision", "context_recall", "context_utilization"]:
                if hasattr(result.ragas_metrics, metric_name):
                    value = getattr(result.ragas_metrics, metric_name)
                    if isinstance(value, int | float):
                        mlflow.log_metric(f"ragas_{metric_name}", float(value))

        # Legacy LLM Judge metrics
        if hasattr(result, "llm_judge_metrics") and result.llm_judge_metrics:
            for metric_name in ["relevance", "groundedness", "completeness", "coherence", "fluency"]:
                if hasattr(result.llm_judge_metrics, metric_name):
                    value = getattr(result.llm_judge_metrics, metric_name)
                    if isinstance(value, int | float):
                        mlflow.log_metric(f"llm_judge_{metric_name}", float(value))

    def _log_combined_metrics(self, result: CombinedEvaluationResult) -> None:
        """Log combined evaluation metrics.

        Parameters
        ----------
        result : CombinedEvaluationResult
            Combined evaluation result object
        """
        # Log individual task results
        if result.stt_result:
            self._log_stt_metrics(result.stt_result)

        if result.diarization_result:
            self._log_diarization_metrics(result.diarization_result)

        if result.rag_result:
            self._log_rag_metrics(result.rag_result)

    def _log_parameters(self, result: Any) -> None:
        """Log evaluation parameters.

        Parameters
        ----------
        result : Any
            Evaluation result object
        """
        import mlflow

        params = {}

        # Common parameters
        if hasattr(result, "evaluation_config"):
            config = result.evaluation_config
            if config:
                params.update(
                    {
                        "evaluation_timestamp": getattr(config, "timestamp", ""),
                        "evaluation_version": getattr(config, "version", ""),
                    }
                )

        # Task-specific parameters
        if hasattr(result, "language"):
            params["language"] = result.language

        if hasattr(result, "collar") and result.collar is not None:
            params["collar"] = result.collar

        mlflow.log_params(params)

    def _upload_datasets(self, dataset_paths: dict[str, Path]) -> None:
        """Upload dataset files as artifacts.

        Parameters
        ----------
        dataset_paths : Dict[str, Path]
            Dictionary mapping dataset names to file paths
        """
        import mlflow

        for dataset_name, dataset_path in dataset_paths.items():
            if dataset_path.exists():
                try:
                    mlflow.log_artifact(str(dataset_path), f"datasets/{dataset_name}")
                    logger.debug(f"✓ Uploaded dataset: {dataset_name} -> {dataset_path}")
                except Exception as e:
                    logger.warning(f"Failed to upload dataset {dataset_name}: {e}")

    def _upload_result_artifact(self, result: Any, run_name: str) -> None:
        """Upload evaluation result as JSON artifact.

        Parameters
        ----------
        result : Any
            Evaluation result object
        run_name : str
            Name of the run for artifact naming
        """
        import mlflow

        try:
            # Convert result to dict
            if hasattr(result, "dict"):
                result_dict = result.dict()
            elif hasattr(result, "__dict__"):
                result_dict = result.__dict__
            else:
                result_dict = {"result": str(result)}

            # Create temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
                json.dump(result_dict, temp_file, indent=2, default=str)
                temp_file_path = temp_file.name

            try:
                # Upload artifact
                mlflow.log_artifact(temp_file_path, "results")
                logger.debug(f"✓ Uploaded result artifact: {run_name}")
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)

        except Exception as e:
            logger.warning(f"Failed to upload result artifact: {e}")

    def list_experiments(self) -> list[dict[str, Any]] | None:
        """[Optional Function]:List all experiments in the workspace.

        Returns
        -------
        List[Dict[str, Any]], optional
            List of experiment run data, None if failed
        """
        if not self.is_enabled():
            return None

        try:
            import mlflow

            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if not experiment:
                return []

            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=50, order_by=["start_time DESC"])

            return runs.to_dict("records") if not runs.empty else []

        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return None

    def get_metrics_summary(self) -> dict[str, Any] | None:
        """[Optional Function]: Get summary of metrics across all runs.

        Returns
        -------
        Dict[str, Any], optional
            Summary statistics for all metrics, None if failed
        """
        if not self.is_enabled():
            return None

        try:
            import mlflow

            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if not experiment:
                return {"total_runs": 0, "metrics": {}}

            runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=1000)

            if runs_df.empty:
                return {"total_runs": 0, "metrics": {}}

            # Calculate summary statistics
            summary = {"total_runs": len(runs_df), "metrics": {}}

            metric_columns = [col for col in runs_df.columns if col.startswith("metrics.")]
            for metric_col in metric_columns:
                metric_name = metric_col.replace("metrics.", "")
                values = runs_df[metric_col].dropna()

                if not values.empty:
                    summary["metrics"][metric_name] = {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max()), "std": float(values.std()), "count": int(len(values))}

            return summary

        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return None


# Global logger instance
_global_logger: EvaluationLogger | None = None


def get_evaluation_logger() -> EvaluationLogger:
    """Get global evaluation logger instance.

    Returns
    -------
    EvaluationLogger
        Global evaluation logger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = EvaluationLogger()
    return _global_logger


def log_evaluation_result(result: STTEvaluationResult | DiarizationEvaluationResult | RAGEvaluationResult | CombinedEvaluationResult, dataset_paths: dict[str, Path] | None = None, session_id: str | None = None) -> str | None:
    """Convenience function to log evaluation result.

    Parameters
    ----------
    result : Union[STTEvaluationResult, DiarizationEvaluationResult, RAGEvaluationResult, CombinedEvaluationResult]
        Evaluation result to log
    dataset_paths : Dict[str, Path], optional
        Dictionary mapping dataset names to file paths
    session_id : str, optional
        Session ID for grouping related runs

    Returns
    -------
    str, optional
        Run ID if successful, None otherwise

    Examples
    --------
    >>> from eval.eval_logger import log_evaluation_result
    >>> run_id = log_evaluation_result(stt_result, {"refs": Path("refs.txt")})
    """
    logger_instance = get_evaluation_logger()
    return logger_instance.log_evaluation_result(result, dataset_paths, session_id)
