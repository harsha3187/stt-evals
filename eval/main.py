"""Main evaluation orchestrator.

This module provides a unified interface to run evaluation across all supported tasks:
- Speech-to-Text (ASR) evaluation
- Speaker Diarization evaluation
- Retrieval evaluation (future)
- RAG answer quality evaluation (future)

Note: E402 (imports not at top) is intentional - cache paths must be set before importing ML libraries.

Supported file formats for STT evaluation:
- TXT format: Each line contains a transcription
- JSONL format: Each line contains {"id": "audio_id", "text": "transcription"}

Usage:
    # TXT format with default test name and output
    python main.py --task stt --references ref.txt --hypotheses hyp.txt

    # JSONL format with custom test name
    python main.py --task stt --references ref.jsonl --hypotheses hyp.jsonl --test-name "my_test"

    # Custom output directory
    python main.py --task stt --references ref.txt --hypotheses hyp.txt --output "/path/to/results"

    # Other tasks
    python main.py --task diarization --reference ref.rttm --hypothesis hyp.rttm
    python main.py --task all --config eval_config.yaml
"""
# ruff: noqa: E402

# IMPORTANT: Configure cache directories BEFORE any imports
# This prevents HuggingFace libraries from creating 'datasets' folder
import os
from pathlib import Path as _Path

_EVAL_CACHE_DIR = _Path(__file__).parent / ".model_cache"
_EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set ALL HuggingFace-related cache environment variables
os.environ.setdefault("TRANSFORMERS_CACHE", str(_EVAL_CACHE_DIR / "transformers"))
os.environ.setdefault("HF_HOME", str(_EVAL_CACHE_DIR / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_EVAL_CACHE_DIR / "datasets"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_EVAL_CACHE_DIR / "sentence_transformers"))
os.environ.setdefault("TORCH_HOME", str(_EVAL_CACHE_DIR / "torch"))

# Now safe to import other modules
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import uuid

from eval.config import CloudLoggingConfig, EvaluationSettings
from eval.models import (
    CombinedEvaluationResult,
    DiarizationEvaluationResult,
    RAGEvaluationResult,
    STTEvaluationResult,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import evaluation logging (conditionally to handle missing dependencies)
try:
    from eval.eval_logger import EvaluationLogger

    CLOUD_LOGGING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Cloud logging not available: {e}")
    CLOUD_LOGGING_AVAILABLE = False

# Import evaluation modules (conditionally to handle missing dependencies)
try:
    from eval.diarization import (
        evaluate_diarization_model,
        load_rttm_file,
    )

    DIARIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Diarization module not available: {e}")
    DIARIZATION_AVAILABLE = False

try:
    from eval.stt import (
        evaluate_asr_model,
    )

    STT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"stt not available: {e}")
    STT_AVAILABLE = False

try:
    from eval.rag import evaluate_rag_model

    RAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG module not available: {e}")
    RAG_AVAILABLE = False


class EvaluationOrchestrator:
    """Main orchestrator for running evaluation tasks.

    Coordinates multiple evaluation tasks and manages results using structured
    Pydantic models for type safety and validation. Supports both individual
    task execution and combined evaluation sessions.
    """

    def __init__(self, config: dict[str, Any] | None = None, settings: EvaluationSettings | None = None):
        """Initialize the evaluation orchestrator.

        Parameters
        ----------
        config : dict[str, Any] | None, optional
            Configuration dictionary for evaluation tasks, by default None
        settings : EvaluationSettings | None, optional
            Evaluation settings including Azure ML configuration, by default None
            If None, will create default EvaluationSettings instance
        """
        self.config = config or {}
        # Always ensure settings is initialized - create default if None
        self.settings = settings if settings is not None else EvaluationSettings()
        self.results = {}
        self.session_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.tasks_performed = []

        # Initialize evaluation logger if available and enabled
        self.evaluation_logger = None

        if CLOUD_LOGGING_AVAILABLE and self.settings:
            try:
                # Use get_cloud_logging_config to ensure environment variables are loaded
                cloud_config: CloudLoggingConfig = self.settings.get_cloud_logging_config()
                logger.info(cloud_config)
                if cloud_config.enable_logging:
                    self.evaluation_logger = EvaluationLogger(cloud_config)
                    if self.evaluation_logger.is_enabled():
                        logger.info("✓ Cloud logging enabled")
                    else:
                        logger.warning("Cloud logging configuration incomplete")
                        self.evaluation_logger = None
            except Exception as e:
                logger.warning(f"Failed to initialize evaluation logger: {e}")
                self.evaluation_logger = None

    def run_stt_evaluation(
        self,
        ground_truth: list[str],
        generated: list[str],
        language: str = "en",
        output_file: str | None = None,
        ground_truth_file: Path | None = None,
        generated_file: Path | None = None,
        test_name: str | None = None,
    ):
        """Run Speech-to-Text evaluation.

        Parameters
        ----------
        ground_truth : list[str]
            List of ground truth transcriptions
        generated : list[str]
            List of generated/predicted transcriptions
        language : str, optional
            Language code ("en" or "ar"), by default "en"
        output_file : str | None, optional
            Optional output file to save results, by default None
        ground_truth_file : Path | None, optional
            Path to ground truth file for Azure ML logging, by default None
        generated_file : Path | None, optional
            Path to generated file for Azure ML logging, by default None
        test_name : str | None, optional
            Name for the evaluation test, by default None (auto-generated)

        Returns
        -------
        STTEvaluationResult
            Structured evaluation results with type safety and validation

        Raises
        ------
        ImportError
            If Speech-to-text evaluation module is not available
        """
        if not STT_AVAILABLE:
            raise ImportError("Speech-to-text evaluation module is not available")

        logger.info(f"Starting Speech-to-Text evaluation for {len(ground_truth)} samples (language: {language})")

        try:
            # Run evaluation
            results = evaluate_asr_model(ground_truth, generated, language=language)

            # Log key results
            logger.info(f"STT Results - WER: {results.wer:.4f}, CER: {results.cer:.4f}")

            # Prepare dataset paths for Azure ML logging
            dataset_paths = {}
            if ground_truth_file and ground_truth_file.exists():
                dataset_paths["ground_truth"] = ground_truth_file
            if generated_file and generated_file.exists():
                dataset_paths["generated"] = generated_file

            # Resolve output path and save results if needed
            if not output_file:
                if not test_name:
                    # Generate test name if not provided
                    resolved_test_name = test_name if test_name else _generate_default_test_name("stt")
                else:
                    resolved_test_name = test_name
                # Resolve the output path using the same logic as CLI
                resolved_output_path = _resolve_output_path(output_file, resolved_test_name, "stt")
                self._save_results(results, str(resolved_output_path), "stt", dataset_paths)
            else:
                self._save_results(results, str(output_file), "stt", dataset_paths)

            self.results["stt"] = results
            self.tasks_performed.append("stt")
            return results

        except Exception as e:
            logger.error(f"Error in Speech-to-Text evaluation: {e}")
            raise

    def run_diarization_evaluation(
        self,
        ground_truth_rttm: str | Path,
        generated_rttm: str | Path,
        collar: float = 0.25,
        output_file: str | None = None,
        test_name: str | None = None,
        ground_truth_file: Path | None = None,
        generated_file: Path | None = None,
    ):
        """Run Speaker Diarization evaluation.

        Parameters
        ----------
        ground_truth_rttm : str | Path
            Path to ground truth RTTM file
        generated_rttm : str | Path
            Path to generated RTTM file
        collar : float, optional
            Tolerance collar in seconds, by default 0.25
        output_file : str | None, optional
            Optional output file to save results, by default None
        test_name : str | None, optional
            Name for the evaluation test, by default None (auto-generated)

        Returns
        -------
        DiarizationEvaluationResult
            Structured evaluation results with type safety and validation

        Raises
        ------
        ImportError
            If Diarization evaluation module is not available
        """
        if not DIARIZATION_AVAILABLE:
            raise ImportError("Diarization evaluation module is not available")

        logger.info(f"Starting Speaker Diarization evaluation (collar: {collar}s)")

        try:
            # Convert to Path objects
            ground_truth_path = Path(ground_truth_rttm)
            generated_path = Path(generated_rttm)

            # Load RTTM files
            ground_truth_segments = load_rttm_file(ground_truth_path)
            generated_segments = load_rttm_file(generated_path)

            logger.info(f"Loaded {len(ground_truth_segments)} ground truth and {len(generated_segments)} generated segments")

            # Run evaluation
            results = evaluate_diarization_model(ground_truth_segments, generated_segments, collar=collar)

            # Log key results
            logger.info(f"Diarization Results - DER: {results.der:.4f}, JER: {results.jer:.4f}")

            # Prepare dataset paths for Azure ML logging
            dataset_paths = {}
            if ground_truth_file:
                dataset_paths["ground_truth_rttm"] = ground_truth_file
            elif ground_truth_path:
                dataset_paths["ground_truth_rttm"] = ground_truth_path
            if generated_file:
                dataset_paths["generated_rttm"] = generated_file
            elif generated_path:
                dataset_paths["generated_rttm"] = generated_path

            # Resolve output path and save results if needed
            if not output_file:
                if not test_name:
                    # Generate test name if not provided
                    resolved_test_name = test_name if test_name else _generate_default_test_name("diarization")
                else:
                    resolved_test_name = test_name
                # Resolve the output path using the same logic as CLI
                resolved_output_path = _resolve_output_path(output_file, resolved_test_name, "diarization")
                self._save_results(results, str(resolved_output_path), "diarization", dataset_paths if dataset_paths else None)
            else:
                self._save_results(results, str(output_file), "diarization", dataset_paths)

            self.results["diarization"] = results
            self.tasks_performed.append("diarization")
            return results

        except Exception as e:
            logger.error(f"Error in Diarization evaluation: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

    def run_rag_evaluation(
        self,
        ground_truth_file: str | Path,
        generated_file: str | Path,
        language: str = "en",
        output_file: str | None = None,
        test_name: str | None = None,
    ):
        """Run RAG (Retrieval-Augmented Generation) summarization evaluation.

        Simplified RAG evaluation focusing on summarization quality metrics only.
        Follows the same pattern as STT evaluation for consistency.

        Parameters
        ----------
        references_file : str | Path
            Path to reference texts file (JSONL format with 'text' field)
        candidates_file : str | Path
            Path to generated candidate texts file (JSONL format with 'text' field)
        language : str, optional
            Language code ("en" or "ar"), by default "en"
        output_file : str | None, optional
            Optional output file to save results, by default None
        test_name : str | None, optional
            Name for the evaluation test, by default None (auto-generated)

        Returns
        -------
        RAGEvaluationResult
            Structured evaluation results with type safety and validation

        Raises
        ------
        ImportError
            If RAG evaluation module is not available
        """
        if not RAG_AVAILABLE:
            raise ImportError("RAG evaluation module is not available")

        logger.info("Starting RAG summarization evaluation")
        logger.info(f"Language: {language}")

        try:
            # Run evaluation using the simplified function
            results = evaluate_rag_model(ground_truth_file=ground_truth_file, generated_file=generated_file, language=language)

            # Log results
            logger.info("RAG Evaluation Results:")
            logger.info(f"  Sample Count: {results.summarization_metrics.sample_count}")
            logger.info(f"  Language: {results.summarization_metrics.language}")

            # Log summarization metrics
            sm = results.summarization_metrics
            logger.info("  Summarization Metrics:")
            logger.info(f"    ROUGE-1 F1: {sm.rouge_1_f1:.4f} (Rubric: {sm.rouge_1_f1_rubric.rubric_score}/4 - {sm.rouge_1_f1_rubric.rubric_label})")
            logger.info(f"    ROUGE-2 F1: {sm.rouge_2_f1:.4f} (Rubric: {sm.rouge_2_f1_rubric.rubric_score}/4 - {sm.rouge_2_f1_rubric.rubric_label})")
            logger.info(f"    ROUGE-L F1: {sm.rouge_l_f1:.4f} (Rubric: {sm.rouge_l_f1_rubric.rubric_score}/4 - {sm.rouge_l_f1_rubric.rubric_label})")
            logger.info(f"    METEOR: {sm.meteor_score:.4f} (Rubric: {sm.meteor_score_rubric.rubric_score}/4 - {sm.meteor_score_rubric.rubric_label})")
            logger.info(f"    Cosine Similarity: {sm.semantic_similarity_cosine:.4f} (Rubric: {sm.semantic_similarity_cosine_rubric.rubric_score}/4 - {sm.semantic_similarity_cosine_rubric.rubric_label})")
            logger.info(f"    Euclidean Similarity: {sm.semantic_similarity_euclidean:.4f} (Rubric: {sm.semantic_similarity_euclidean_rubric.rubric_score}/4 - {sm.semantic_similarity_euclidean_rubric.rubric_label})")
            logger.info(f"    Clinical Similarity: {sm.clinical_embedding_similarity:.4f} (Rubric: {sm.clinical_embedding_similarity_rubric.rubric_score}/4 - {sm.clinical_embedding_similarity_rubric.rubric_label})")

            summary = sm.get_summary()
            logger.info(f"    Overall Assessment: {summary['overall_assessment']} (Avg Rubric: {summary['average_rubric_score']:.2f}/4)")

            # Prepare dataset paths for Azure ML logging
            dataset_paths = {"ground_truth": Path(ground_truth_file), "generated": Path(generated_file)}

            # Save results if output_file is provided
            # if output_file:
            #     self._save_results(results, str(output_file), "rag", dataset_paths)

            # Resolve output path and save results if needed
            if not output_file:
                if not test_name:
                    # Generate test name if not provided
                    resolved_test_name = test_name if test_name else _generate_default_test_name("rag")
                else:
                    resolved_test_name = test_name
                # Resolve the output path using the same logic as CLI
                resolved_output_path = _resolve_output_path(output_file, resolved_test_name, "rag")
                self._save_results(results, str(resolved_output_path), "rag", dataset_paths if dataset_paths else None)
            else:
                self._save_results(results, str(output_file), "rag", dataset_paths)

            self.results["rag"] = results
            self.tasks_performed.append("rag")
            return results

        except Exception as e:
            logger.error(f"Error in RAG evaluation: {e}")
            raise

    def run_all_evaluations(self, config_file: str | Path, output_directory: str | Path | None = None):  # noqa: C901
        """Run all configured evaluations.

        Parameters
        ----------
        config_file : str | Path
            Path to configuration file
        output_directory : str | Path | None, optional
            Directory to save combined results, by default None

        Returns
        -------
        CombinedEvaluationResult | dict[str, dict[str, float]]
            Combined evaluation results (Pydantic model if available, dict otherwise)
        """
        logger.info(f"Running evaluations from config: {config_file}")

        try:
            config = self._load_config(config_file)

            all_results = {}
            stt_result = None
            diarization_result = None
            rag_result = None

            # Run STT evaluation if configured
            if "stt" in config:
                stt_config = config["stt"]
                references = self._load_transcriptions_file(stt_config["references"])
                hypotheses = self._load_transcriptions_file(stt_config["hypotheses"])
                language = stt_config.get("language", "en")
                output_file = stt_config.get("output_file")

                stt_result = self.run_stt_evaluation(references, hypotheses, language, output_file)
                all_results["stt"] = stt_result

            # Run Diarization evaluation if configured
            if "diarization" in config:
                diar_config = config["diarization"]
                collar = diar_config.get("collar", 0.25)
                output_file = diar_config.get("output_file")

                diarization_result = self.run_diarization_evaluation(
                    diar_config["ground_truth_rttm"],
                    diar_config["generated_rttm"],
                    collar,
                    output_file,
                )
                all_results["diarization"] = diarization_result

            # Run RAG evaluation if configured
            if "rag" in config:
                rag_config = config["rag"]
                language = rag_config.get("language", "en")
                k_values = rag_config.get("k_values", [5, 10, 20])
                enable_ragas = rag_config.get("enable_ragas", True)
                enable_llm_judge = rag_config.get("enable_llm_judge", True)
                output_file = rag_config.get("output_file")

                rag_result = self.run_rag_evaluation(
                    rag_config["queries_file"],
                    rag_config["corpus_file"],
                    rag_config["qrels_file"],
                    rag_config["answers_file"],
                    language,
                    k_values,
                    enable_ragas,
                    enable_llm_judge,
                    output_file,
                )
                all_results["rag"] = rag_result

            # Prepare all dataset paths for Azure ML logging
            all_dataset_paths = {}

            # Add STT dataset paths if available
            if "stt" in config:
                stt_config = config["stt"]
                if "ground_truth_file" in stt_config:
                    all_dataset_paths["stt_ground_truth"] = Path(stt_config["ground_truth_file"])
                if "generated_file" in stt_config:
                    all_dataset_paths["stt_generated"] = Path(stt_config["generated_file"])

            # Add diarization dataset paths if available
            if "diarization" in config:
                diar_config = config["diarization"]
                if "ground_truth_rttm" in diar_config:
                    all_dataset_paths["diarization_ground_truth"] = Path(diar_config["ground_truth_rttm"])
                if "generated_rttm" in diar_config:
                    all_dataset_paths["diarization_generated"] = Path(diar_config["generated_rttm"])

            # Add RAG dataset paths if available
            if "rag" in config:
                rag_config = config["rag"]
                for file_type in ["queries_file", "corpus_file", "qrels_file", "answers_file"]:
                    if file_type in rag_config:
                        all_dataset_paths[f"rag_{file_type.replace('_file', '')}"] = Path(rag_config[file_type])

            # Save combined results
            if output_directory:
                output_directory = Path(output_directory)
                output_directory.parent.mkdir(parents=True, exist_ok=True)
                output_file = str(output_directory)
            else:
                output_file = config.get("combined_output", "evaluation_results.json")
            self._save_results(all_results, output_file, "combined", all_dataset_paths)

            self.results = all_results

            # Return structured result
            evaluation_time = time.time() - self.start_time
            return CombinedEvaluationResult(
                stt_results=stt_result if isinstance(stt_result, STTEvaluationResult) else None,
                diarization_results=diarization_result if isinstance(diarization_result, DiarizationEvaluationResult) else None,
                rag_results=rag_result if isinstance(rag_result, RAGEvaluationResult) else None,
                evaluation_session_id=self.session_id,
                tasks_performed=self.tasks_performed,
                total_evaluation_time=evaluation_time,
                configuration_file=Path(config_file),
                output_directory=Path(output_file).parent if output_file else None,
            )

        except Exception as e:
            logger.error(f"Error running all evaluations: {e}")
            raise

    def _load_config(self, config_file: str | Path) -> dict[str, Any]:
        """Load configuration from file."""
        config_path = Path(config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_path) as f:
            if config_path.suffix.lower() == ".json":
                return json.load(f)
            elif config_path.suffix.lower() in [".yaml", ".yml"]:
                try:
                    import yaml

                    return yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config files")
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    def _load_transcriptions_file(self, file_path: str | Path) -> list[str]:
        """Load transcriptions from either TXT or JSONL format files.

        Supports two file formats:
        1. TXT format: Each line contains a transcription
        2. JSONL format: Each line contains a JSON object with "text" field
           Example: {"id": "audio1.mp3", "text": "hello world"}

        Parameters
        ----------
        file_path : str | Path
            Path to the transcriptions file (.txt or .jsonl)

        Returns
        -------
        list[str]
            List of transcription texts

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        ValueError
            If JSONL format is malformed or missing required "text" field
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Transcriptions file not found: {file_path}")

        transcriptions = []

        # Determine file format based on extension
        if file_path.suffix.lower() == ".jsonl":
            # Load JSONL format
            with open(file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parse JSON object
                        data = json.loads(line)

                        # Extract text field
                        if "text" not in data:
                            raise ValueError(f"Missing 'text' field in JSON object at line {line_num}")

                        transcription = data["text"]
                        if not isinstance(transcription, str):
                            raise ValueError(f"'text' field must be a string at line {line_num}")

                        transcriptions.append(transcription.strip())

                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON format at line {line_num}: {e}")

        else:
            # Load TXT format (default for any other extension)
            with open(file_path, encoding="utf-8") as f:
                transcriptions = [line.strip() for line in f if line.strip()]

        if not transcriptions:
            logger.warning(f"No transcriptions found in file: {file_path}")
        else:
            logger.info(f"Loaded {len(transcriptions)} transcriptions from {file_path} ({file_path.suffix} format)")

        return transcriptions

    def _save_results(self, results: Any, output_file: str, task_type: str, dataset_paths: dict[str, Path] | None = None) -> None:
        """Save results to file and optionally log to Azure ML.

        Parameters
        ----------
        results : Any
            Results to save (Pydantic model, dict, or combined results)
        output_file : str
            Path to output file
        task_type : str
            Type of task ("stt", "diarization", "rag", "combined")
        dataset_paths : dict[str, Path] | None, optional
            Dictionary mapping dataset names to file paths for Azure ML logging
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Pydantic models to dictionaries for JSON serialization
        if hasattr(results, "model_dump"):
            # Single Pydantic model
            results_dict = results.model_dump()
        elif isinstance(results, dict):
            # Dictionary containing nested Pydantic models
            results_dict = {}
            for key, value in results.items():
                if hasattr(value, "model_dump"):
                    results_dict[key] = value.model_dump()
                else:
                    results_dict[key] = value
        else:
            # Should not happen with Pydantic-only approach
            results_dict = results

        # Add metadata
        output_data = {
            "task_type": task_type,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "results": results_dict,
        }

        # Save to local file
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"Results saved to: {output_path}")

        # Log to cloud if enabled
        if self.evaluation_logger and self.evaluation_logger.is_enabled():
            try:
                run_id = self.evaluation_logger.log_evaluation_result(result=results, dataset_paths=dataset_paths, session_id=self.session_id)
                if run_id:
                    logger.info(f"✓ Logged to cloud - Run ID: {run_id}")
                else:
                    logger.warning("Failed to log to cloud")
            except Exception as e:
                logger.warning(f"Error logging to cloud: {e}")


def _generate_default_test_name(task: str) -> str:
    """Generate a default test name using task and timestamp.

    Parameters
    ----------
    task : str
        The evaluation task name

    Returns
    -------
    str
        Default test name in format: task_YYYYMMDD_HHMMSS
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{task}_{timestamp}"


def _resolve_output_path(output_arg: str | None, test_name: str, task: str) -> Path:
    """Resolve the output path for evaluation results.

    Parameters
    ----------
    output_arg : str | None
        Output path provided via command line argument
    test_name : str
        Name of the evaluation test
    task : str
        The evaluation task name

    Returns
    -------
    Path
        Resolved output path for saving results
    """
    if output_arg:
        # If output path is provided, use it as the directory and add the test name
        output_path = Path(output_arg)
        if output_path.is_file() or output_path.suffix:
            # If it's a specific file, use it as-is
            return output_path
        else:
            # If it's a directory, create a file within it
            return output_path / f"{test_name}_results.json"
    else:
        # Default to eval/results/task_datetime/ directory with task_evaluation_results.json file
        # Always use absolute path to eval/results regardless of current working directory
        eval_dir = Path(__file__).parent  # This is the eval/ directory
        results_dir = eval_dir / "results" / test_name
        return results_dir / f"{task}_evaluation_results.json"


def create_sample_config() -> None:
    """Create a sample environment configuration file."""
    from eval.config import generate_env_example

    generate_env_example(Path(".env.example"))

    logger.info("Sample environment configuration created: .env.example")


def main() -> None:  # noqa: C901
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluation orchestrator for ASR, Diarization, and RAG")

    parser.add_argument(
        "--task",
        choices=["stt", "diarization", "rag", "all"],
        required=True,
        help="Evaluation task to run",
    )

    # Speech-to-Text arguments
    parser.add_argument("--ground-truth", help="Path to ground truth transcriptions file (.txt or .jsonl)")
    parser.add_argument("--generated", help="Path to generated/predicted transcriptions file (.txt or .jsonl)")
    parser.add_argument("--language", default="en", choices=["en", "ar"], help="Language code")

    # Diarization arguments
    parser.add_argument("--ground-truth-rttm", help="Path to ground truth RTTM file")
    parser.add_argument("--generated-rttm", help="Path to generated/predicted RTTM file")
    parser.add_argument("--collar", type=float, default=0.25, help="Tolerance collar in seconds")

    # RAG arguments (simplified - focuses on summarization quality only)
    parser.add_argument("--ground-truth-rag", help="Path to ground truth texts file (JSONL format with 'text' field)")
    parser.add_argument("--generated-rag", help="Path to generated/predicted texts file (JSONL format with 'text' field)")

    # General arguments
    parser.add_argument("--test-name", help="Name for the evaluation test (default: task_YYYYMMDD_HHMMSS)")
    parser.add_argument("--config", help="Path to configuration file (for --task all)")
    parser.add_argument("--output", help="Output path/directory for results (default: eval/results/test_name/)")
    parser.add_argument(
        "--create-sample-config",
        action="store_true",
        help="Create a sample configuration file",
    )

    # Handle create sample config before parsing other args
    if "--create-sample-config" in sys.argv:
        create_sample_config()
        return

    args = parser.parse_args()

    try:
        # Generate test name if not provided
        test_name = args.test_name if args.test_name else _generate_default_test_name(args.task)

        # Resolve output path
        output_path = _resolve_output_path(args.output, test_name, args.task)

        logger.info(f"Starting evaluation '{test_name}' for task: {args.task}")
        logger.info(f"Results will be saved to: {output_path}")

        # Load evaluation settings (includes Azure ML configuration)
        settings = EvaluationSettings()

        orchestrator = EvaluationOrchestrator(settings=settings)

        if args.task == "stt":
            ground_truth_arg = getattr(args, "ground_truth", None)
            generated_arg = getattr(args, "generated", None)

            if not ground_truth_arg or not generated_arg:
                parser.error("--ground-truth and --generated are required for stt task")

            ground_truth = orchestrator._load_transcriptions_file(ground_truth_arg)
            generated = orchestrator._load_transcriptions_file(generated_arg)

            orchestrator.run_stt_evaluation(ground_truth, generated, args.language, str(output_path), ground_truth_file=Path(ground_truth_arg), generated_file=Path(generated_arg), test_name=test_name)

        elif args.task == "diarization":
            ground_truth_rttm_arg = getattr(args, "ground_truth_rttm", None)
            generated_rttm_arg = getattr(args, "generated_rttm", None)

            if not ground_truth_rttm_arg or not generated_rttm_arg:
                parser.error("--ground-truth-rttm and --generated-rttm are required for diarization task")

            logger.info(f"Results will be saved to temp: {output_path}")

            orchestrator.run_diarization_evaluation(ground_truth_rttm_arg, generated_rttm_arg, args.collar, str(output_path), test_name=test_name)

        elif args.task == "rag":
            ground_truth_rag_arg = getattr(args, "ground_truth_rag", None)
            generated_rag_arg = getattr(args, "generated_rag", None)

            if not ground_truth_rag_arg or not generated_rag_arg:
                parser.error("--ground-truth-rag and --generated-rag are required for rag task")

            orchestrator.run_rag_evaluation(ground_truth_rag_arg, generated_rag_arg, args.language, str(output_path), test_name=test_name)

        elif args.task == "all":
            if not args.config:
                parser.error("--config is required for 'all' task")

            orchestrator.run_all_evaluations(args.config, str(output_path))

        logger.info("Evaluation completed")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
