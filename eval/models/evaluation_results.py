"""Pydantic models for evaluation results.

This module defines structured data models for all evaluation outputs following
the backend instructions guidelines. These models provide type safety, validation,
and clear documentation for evaluation results.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RubricScore(BaseModel):
    """Rubric score information for a metric following ADR-003 specifications."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    raw_value: float = Field(..., description="Original metric value")
    rubric_score: int = Field(..., ge=1, le=4, description="Rubric score: 1=Poor, 2=Fair, 3=Good, 4=Excellent")
    rubric_label: str = Field(..., description="Human-readable rubric label")
    metric_name: str = Field(..., description="Name of the metric")


class STTEvaluationResult(BaseModel):
    """Results from Speech-to-Text evaluation.

    Contains comprehensive ASR evaluation metrics including word-level,
    character-level, and Levenshtein distance measurements with ADR-003 rubric scores.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    # Word-level metrics from jiwer
    wer: float = Field(..., ge=0.0, description="Word Error Rate: percentage of words incorrectly predicted")
    mer: float = Field(..., ge=0.0, description="Match Error Rate: percentage of word matches that are errors")
    wip: float = Field(..., ge=0.0, le=1.0, description="Word Information Preserved: fraction of word information retained")
    wil: float = Field(..., ge=0.0, le=1.0, description="Word Information Lost: fraction of word information lost")

    # Character-level metrics
    cer: float = Field(..., ge=0.0, description="Character Error Rate: percentage of characters incorrectly predicted")

    # Levenshtein distance metrics
    word_levenshtein_distance: float = Field(..., ge=0.0, description="Average word-level Levenshtein distance")
    char_levenshtein_distance: float = Field(..., ge=0.0, description="Average character-level Levenshtein distance")
    normalized_word_ld: float = Field(..., ge=0.0, le=1.0, description="Normalized word-level Levenshtein distance (0-1 scale)")
    normalized_char_ld: float = Field(..., ge=0.0, le=1.0, description="Normalized character-level Levenshtein distance (0-1 scale)")

    # Rubric scores (ADR-003 4-point scale)
    wer_rubric: RubricScore = Field(..., description="Rubric score for WER")
    mer_rubric: RubricScore = Field(..., description="Rubric score for MER")
    wip_rubric: RubricScore = Field(..., description="Rubric score for WIP")
    wil_rubric: RubricScore = Field(..., description="Rubric score for WIL")
    cer_rubric: RubricScore = Field(..., description="Rubric score for CER")
    word_ld_rubric: RubricScore = Field(..., description="Rubric score for word Levenshtein distance")
    char_ld_rubric: RubricScore = Field(..., description="Rubric score for character Levenshtein distance")
    normalized_word_ld_rubric: RubricScore = Field(..., description="Rubric score for normalized word LD")
    normalized_char_ld_rubric: RubricScore = Field(..., description="Rubric score for normalized character LD")

    # Metadata
    language: str = Field(..., pattern=r"^(en|ar)$", description="Language code used for evaluation")
    sample_count: int = Field(..., ge=1, description="Number of samples evaluated")
    evaluation_timestamp: datetime = Field(default_factory=datetime.now, description="When the evaluation was performed")

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of key metrics with rubric scores.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the most important metrics and their rubric scores for quick review
        """
        return {
            "wer": self.wer,
            "wer_rubric": f"{self.wer_rubric.rubric_score}/4 ({self.wer_rubric.rubric_label})",
            "cer": self.cer,
            "cer_rubric": f"{self.cer_rubric.rubric_score}/4 ({self.cer_rubric.rubric_label})",
            "normalized_word_ld": self.normalized_word_ld,
            "normalized_word_ld_rubric": f"{self.normalized_word_ld_rubric.rubric_score}/4 ({self.normalized_word_ld_rubric.rubric_label})",
            "language": self.language,
            "sample_count": self.sample_count,
        }


class DiarizationEvaluationResult(BaseModel):
    """Results from Speaker Diarization evaluation.

    Contains DER, JER, SER, and BER metrics following NIST/DIHARD and X-LANCE/BER
    evaluation standards with ADR-003 rubric scores.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    # DER components
    der: float = Field(..., ge=0.0, description="Diarization Error Rate: overall error rate including all components")
    missed_speech: float = Field(..., ge=0.0, le=1.0, description="Missed Speech Rate: fraction of reference speech not detected")
    false_alarm: float = Field(..., ge=0.0, description="False Alarm Rate: fraction of non-speech detected as speech")
    speaker_confusion: float = Field(..., ge=0.0, le=1.0, description="Speaker Confusion Rate: fraction of speech attributed to wrong speaker")

    # JER metric
    jer: float = Field(..., ge=0.0, le=1.0, description="Jaccard Error Rate: 1 - average Jaccard index over mapped speakers")

    # SER metric (X-LANCE/BER)
    ser: float = Field(..., ge=0.0, le=1.0, description="Segment Error Rate: segment-level error via connected sub-graphs and adaptive IoU")

    # BER metrics (X-LANCE/BER)
    ber: float = Field(..., ge=0.0, description="Balanced Error Rate: speaker-weighted combination of duration and segment errors")
    ber_ref_part: float = Field(..., ge=0.0, description="BER reference part: weighted average of DER and SER for matched speakers")
    ber_fa_dur: float = Field(..., ge=0.0, description="BER false alarm duration rate")
    ber_fa_seg: float = Field(..., ge=0.0, description="BER false alarm segment rate")
    ber_fa_mean: float = Field(..., ge=0.0, description="BER false alarm mean: harmonic mean of FA duration and segment rates")

    # Rubric scores (ADR-003 4-point scale)
    der_rubric: RubricScore = Field(..., description="Rubric score for DER")
    jer_rubric: RubricScore = Field(..., description="Rubric score for JER")
    ser_rubric: RubricScore = Field(..., description="Rubric score for SER")
    ber_rubric: RubricScore = Field(..., description="Rubric score for BER")
    missed_speech_rubric: RubricScore = Field(..., description="Rubric score for missed speech")
    false_alarm_rubric: RubricScore = Field(..., description="Rubric score for false alarm")
    speaker_confusion_rubric: RubricScore = Field(..., description="Rubric score for speaker confusion")

    # Evaluation parameters
    collar: float = Field(..., ge=0.0, description="Tolerance collar used in evaluation (seconds)")
    total_speech_time: float = Field(..., ge=0.0, description="Total duration of speech in reference (seconds)")

    # Metadata
    reference_speakers: int = Field(..., ge=0, description="Number of speakers in reference")
    hypothesis_speakers: int = Field(..., ge=0, description="Number of speakers in hypothesis")
    evaluation_timestamp: datetime = Field(default_factory=datetime.now, description="When the evaluation was performed")

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of key metrics.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the most important metrics for quick review
        """
        return {
            "der": self.der,
            "jer": self.jer,
            "ser": self.ser,
            "ber": self.ber,
            "missed_speech": self.missed_speech,
            "false_alarm": self.false_alarm,
            "speaker_confusion": self.speaker_confusion,
            "collar": self.collar,
        }


"""
Design Decision: Separate Pydantic Models for RAG Metrics

The RAG evaluation results use separate Pydantic models (RetrievalMetrics, RAGASMetrics,
LLMJudgeMetrics) rather than combining everything into one large model. This design provides:

1. **Separation of Concerns**: Each metric type has distinct validation rules and scales
   - RetrievalMetrics: 0-1 scale for IR metrics (nDCG, MRR, etc.)
   - RAGASMetrics: 0-1 scale for answer quality
   - LLMJudgeMetrics: 1-4 scale for G-Eval rubric scoring

2. **Optional Metrics**: Different evaluation scenarios require different metrics.
   Separate models allow clean Optional[RAGASMetrics] and Optional[LLMJudgeMetrics]
   without complex conditional validation in a single large model.

3. **Type Safety**: Each model has its own field constraints and validation logic.
   For example, LLMJudgeMetrics uses 1-4 scale while others use 0-1 scale.

4. **Extensibility**: New metric types can be added without modifying existing models.
   Future additions (e.g., ContextualMetrics, HallucinationMetrics) follow same pattern.

5. **Reusability**: Individual metric models can be used independently in other contexts
   or composed differently for different evaluation frameworks.

6. **Clear Documentation**: Each model has focused documentation specific to its metrics,
   making it easier for users to understand what each metric type represents.

This follows the Composition over Inheritance principle and Single Responsibility Principle
from SOLID design patterns. RAGEvaluationResult serves as the aggregate model that
composes these specialized models together.
"""


class RAGSummarizationMetrics(BaseModel):
    """Summarization quality metrics for RAG evaluation following ADR-003 specifications.

    Contains comprehensive summarization metrics including traditional text overlap
    metrics and semantic similarity using clinical embeddings optimized for healthcare domain.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    # Traditional text overlap metrics
    rouge_1_f1: float = Field(..., ge=0.0, le=1.0, description="ROUGE-1 F1 score: unigram overlap")
    rouge_2_f1: float = Field(..., ge=0.0, le=1.0, description="ROUGE-2 F1 score: bigram overlap")
    rouge_l_f1: float = Field(..., ge=0.0, le=1.0, description="ROUGE-L F1 score: longest common subsequence")
    meteor_score: float = Field(..., ge=0.0, le=1.0, description="METEOR score: semantic alignment with synonymy")

    # Semantic similarity metrics using clinical embeddings
    semantic_similarity_cosine: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity using clinical embeddings")
    semantic_similarity_euclidean: float = Field(..., ge=0.0, le=1.0, description="Normalized Euclidean similarity")
    clinical_embedding_similarity: float = Field(..., ge=0.0, le=1.0, description="Healthcare domain-specific weighted similarity")

    # Rubric scores (ADR-003 4-point scale)
    rouge_1_f1_rubric: RubricScore = Field(..., description="Rubric score for ROUGE-1 F1")
    rouge_2_f1_rubric: RubricScore = Field(..., description="Rubric score for ROUGE-2 F1")
    rouge_l_f1_rubric: RubricScore = Field(..., description="Rubric score for ROUGE-L F1")
    meteor_score_rubric: RubricScore = Field(..., description="Rubric score for METEOR")
    semantic_similarity_cosine_rubric: RubricScore = Field(..., description="Rubric score for cosine similarity")
    semantic_similarity_euclidean_rubric: RubricScore = Field(..., description="Rubric score for euclidean similarity")
    clinical_embedding_similarity_rubric: RubricScore = Field(..., description="Rubric score for clinical embedding similarity")

    # Metadata
    language: str = Field(..., pattern=r"^(en|ar)$", description="Language code used for evaluation")
    sample_count: int = Field(..., ge=1, description="Number of reference-candidate pairs evaluated")
    evaluation_timestamp: datetime = Field(default_factory=datetime.now, description="When the evaluation was performed")

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of key metrics.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the most important metrics for quick review
        """
        return {
            "average_rouge_score": (self.rouge_1_f1 + self.rouge_2_f1 + self.rouge_l_f1) / 3.0,
            "average_semantic_score": (self.semantic_similarity_cosine + self.semantic_similarity_euclidean + self.clinical_embedding_similarity) / 3.0,
            "meteor_score": self.meteor_score,
            "average_rubric_score": (
                self.rouge_1_f1_rubric.rubric_score
                + self.rouge_2_f1_rubric.rubric_score
                + self.rouge_l_f1_rubric.rubric_score
                + self.meteor_score_rubric.rubric_score
                + self.semantic_similarity_cosine_rubric.rubric_score
                + self.semantic_similarity_euclidean_rubric.rubric_score
                + self.clinical_embedding_similarity_rubric.rubric_score
            )
            / 7.0,
            "overall_assessment": self._get_overall_assessment(),
            "sample_count": self.sample_count,
            "language": self.language,
        }

    def _get_overall_assessment(self) -> str:
        """Get overall qualitative assessment based on average rubric score.

        Returns
        -------
        str
            Overall assessment: Poor, Fair, Good, or Excellent
        """
        avg_rubric = (
            self.rouge_1_f1_rubric.rubric_score
            + self.rouge_2_f1_rubric.rubric_score
            + self.rouge_l_f1_rubric.rubric_score
            + self.meteor_score_rubric.rubric_score
            + self.semantic_similarity_cosine_rubric.rubric_score
            + self.semantic_similarity_euclidean_rubric.rubric_score
            + self.clinical_embedding_similarity_rubric.rubric_score
        ) / 7.0

        if avg_rubric >= 3.5:
            return "Excellent"
        elif avg_rubric >= 2.5:
            return "Good"
        elif avg_rubric >= 1.5:
            return "Fair"
        else:
            return "Poor"


class RAGEvaluationResult(BaseModel):
    """Results from RAG (Retrieval-Augmented Generation) summarization evaluation.

    Simplified RAG evaluation focusing on summarization quality metrics only.
    Follows the same pattern as STT evaluation for consistency.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    # Summarization metrics (core evaluation)
    summarization_metrics: RAGSummarizationMetrics = Field(..., description="Comprehensive summarization quality metrics")

    # Evaluation metadata
    evaluation_timestamp: datetime = Field(default_factory=datetime.now, description="When the evaluation was performed")

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of key metrics with rubric scores.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the most important metrics and their rubric scores for quick review
        """
        return self.summarization_metrics.get_summary()

    def get_best_k(self) -> int:
        """Determine the best K value based on average performance.

        Returns
        -------
        int
            K value with highest average metric scores
        """
        best_k = self.retrieval_metrics[0].k
        best_avg = 0.0

        for metrics in self.retrieval_metrics:
            avg = (metrics.ndcg + metrics.mrr + metrics.precision + metrics.recall + metrics.map_score) / 5.0
            if avg > best_avg:
                best_avg = avg
                best_k = metrics.k

        return best_k


class CombinedEvaluationResult(BaseModel):
    """Combined results from multiple evaluation tasks.

    Aggregates results from STT and Diarization evaluations with
    metadata about the overall evaluation session.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    # Session metadata
    evaluation_session_id: str = Field(..., description="Unique identifier for this evaluation session")
    tasks_performed: list[str] = Field(..., description="List of evaluation tasks that were performed")
    total_evaluation_time: float | None = Field(default=None, ge=0.0, description="Total time taken for all evaluations (seconds)")

    stt_results: STTEvaluationResult | None = Field(default=None, description="Speech-to-Text evaluation results if performed")
    diarization_results: DiarizationEvaluationResult | None = Field(default=None, description="Diarization evaluation results if performed")
    rag_results: RAGEvaluationResult | None = Field(default=None, description="RAG evaluation results if performed")

    # Configuration used
    configuration_file: Path | None = Field(default=None, description="Path to configuration file used (if any)")
    output_directory: Path | None = Field(default=None, description="Directory where results were saved")

    evaluation_timestamp: datetime = Field(default_factory=datetime.now, description="When the evaluation session started")

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics from all performed evaluations.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all metrics with task prefixes
        """
        all_metrics = {"session_id": self.evaluation_session_id, "tasks": self.tasks_performed, "timestamp": self.evaluation_timestamp}

        if self.stt_results:
            for key, value in self.stt_results.model_dump().items():
                all_metrics[f"stt_{key}"] = value

        if self.diarization_results:
            for key, value in self.diarization_results.model_dump().items():
                all_metrics[f"diarization_{key}"] = value

        if self.rag_results:
            for key, value in self.rag_results.model_dump().items():
                all_metrics[f"rag_{key}"] = value

        return all_metrics

    def has_results(self) -> bool:
        """Check if any evaluation results are available.

        Returns
        -------
        bool
            True if at least one evaluation result is present
        """
        return self.stt_results is not None or self.diarization_results is not None or self.rag_results is not None

    def get_summary_report(self) -> str:
        """Generate a human-readable summary report.

        Returns
        -------
        str
            Formatted summary report of all evaluation results
        """
        lines = [f"Evaluation Session: {self.evaluation_session_id}", f"Timestamp: {self.evaluation_timestamp.isoformat()}", f"Tasks Performed: {', '.join(self.tasks_performed)}", ""]

        if self.stt_results:
            lines.extend(
                [
                    "Speech-to-Text Results:",
                    f"  WER: {self.stt_results.wer:.4f}",
                    f"  CER: {self.stt_results.cer:.4f}",
                    f"  Normalized Word LD: {self.stt_results.normalized_word_ld:.4f}",
                    f"  Language: {self.stt_results.language}",
                    f"  Samples: {self.stt_results.sample_count}",
                    "",
                ]
            )

        if self.diarization_results:
            lines.extend(
                [
                    "Diarization Results:",
                    f"  DER: {self.diarization_results.der:.4f}",
                    f"  JER: {self.diarization_results.jer:.4f}",
                    f"  Missed Speech: {self.diarization_results.missed_speech:.4f}",
                    f"  False Alarm: {self.diarization_results.false_alarm:.4f}",
                    f"  Speaker Confusion: {self.diarization_results.speaker_confusion:.4f}",
                    f"  Collar: {self.diarization_results.collar}s",
                    "",
                ]
            )

        if self.rag_results:
            lines.extend(["RAG Evaluation Results:", f"  Language: {self.rag_results.language}", f"  Query Count: {self.rag_results.query_count}", f"  Avg Chunk Relevance: {self.rag_results.avg_chunk_relevance:.4f}", ""])

            # Add retrieval metrics for each K
            for metrics in self.rag_results.retrieval_metrics:
                lines.extend(
                    [
                        f"  Retrieval Metrics @K={metrics.k}:",
                        f"    nDCG: {metrics.ndcg:.4f}",
                        f"    MRR: {metrics.mrr:.4f}",
                        f"    Precision: {metrics.precision:.4f}",
                        f"    Recall: {metrics.recall:.4f}",
                        f"    MAP: {metrics.map_score:.4f}",
                        f"    Hit Rate: {metrics.hit_rate:.4f}",
                        "",
                    ]
                )

            # Add RAGAS metrics if available
            if self.rag_results.ragas_metrics:
                lines.extend(["  RAGAS Metrics:", f"    Faithfulness: {self.rag_results.ragas_metrics.faithfulness:.4f}", f"    Answer Relevance: {self.rag_results.ragas_metrics.answer_relevance:.4f}", ""])

            # Add LLM Judge metrics if available
            if self.rag_results.llm_judge_metrics:
                lines.extend(
                    [
                        "  LLM-as-Judge Metrics:",
                        f"    Relevance: {self.rag_results.llm_judge_metrics.relevance:.2f}/4.0",
                        f"    Groundedness: {self.rag_results.llm_judge_metrics.groundedness:.2f}/4.0",
                        f"    Completeness: {self.rag_results.llm_judge_metrics.completeness:.2f}/4.0",
                        f"    Coherence: {self.rag_results.llm_judge_metrics.coherence:.2f}/4.0",
                        f"    Fluency: {self.rag_results.llm_judge_metrics.fluency:.2f}/4.0",
                        f"    Average: {self.rag_results.llm_judge_metrics.get_average_score():.2f}/4.0",
                        "",
                    ]
                )

        return "\n".join(lines)


# Utility functions for model creation
def create_stt_result(metrics: dict[str, float], language: str, sample_count: int) -> STTEvaluationResult:
    """Create STTEvaluationResult from metrics dictionary.

    Parameters
    ----------
    metrics : dict[str, float]
        Dictionary containing evaluation metrics
    language : str
        Language code used for evaluation
    sample_count : int
        Number of samples evaluated

    Returns
    -------
    STTEvaluationResult
        Validated STT evaluation result model

    Raises
    ------
    ValueError
        If required metrics are missing or invalid
    """
    from ..rubric_scorer import RubricScorer

    required_metrics = ["wer", "mer", "wip", "wil", "cer", "word_levenshtein_distance", "char_levenshtein_distance", "normalized_word_ld", "normalized_char_ld"]

    missing_metrics = [m for m in required_metrics if m not in metrics]
    if missing_metrics:
        raise ValueError(f"Missing required metrics: {missing_metrics}")

    # Create rubric scores for each metric with proper field name mapping
    metric_to_field_mapping = {
        "word_levenshtein_distance": "word_ld",
        "char_levenshtein_distance": "char_ld",
    }

    rubric_data = {}
    for metric_name in required_metrics:
        if metric_name in metrics:
            rubric_info = RubricScorer.get_metric_rubric_info(metric_name, metrics[metric_name])
            field_name = metric_to_field_mapping.get(metric_name, metric_name)
            rubric_data[f"{field_name}_rubric"] = RubricScore(**rubric_info)

    return STTEvaluationResult(language=language, sample_count=sample_count, **metrics, **rubric_data)


def create_diarization_result(metrics: dict[str, float], collar: float, total_speech_time: float, reference_speakers: int, hypothesis_speakers: int) -> DiarizationEvaluationResult:
    """Create DiarizationEvaluationResult from metrics dictionary.

    Parameters
    ----------
    metrics : dict[str, float]
        Dictionary containing evaluation metrics (DER, JER, SER, BER)
    collar : float
        Tolerance collar used in evaluation
    total_speech_time : float
        Total speech duration in reference
    reference_speakers : int
        Number of reference speakers
    hypothesis_speakers : int
        Number of hypothesis speakers

    Returns
    -------
    DiarizationEvaluationResult
        Validated diarization evaluation result model

    Raises
    ------
    ValueError
        If required metrics are missing or invalid
    """
    from ..rubric_scorer import RubricScorer

    required_metrics = ["der", "missed_speech", "false_alarm", "speaker_confusion", "jer", "ser", "ber"]

    missing_metrics = [m for m in required_metrics if m not in metrics]
    if missing_metrics:
        raise ValueError(f"Missing required metrics: {missing_metrics}")

    # Create rubric scores for each metric
    rubric_data = {}
    scoreable_metrics = ["der", "missed_speech", "false_alarm", "speaker_confusion", "jer", "ser", "ber"]
    for metric_name in scoreable_metrics:
        if metric_name in metrics:
            rubric_info = RubricScorer.get_metric_rubric_info(metric_name, metrics[metric_name])
            rubric_data[f"{metric_name}_rubric"] = RubricScore(**rubric_info)

    return DiarizationEvaluationResult(collar=collar, total_speech_time=total_speech_time, reference_speakers=reference_speakers, hypothesis_speakers=hypothesis_speakers, **metrics, **rubric_data)


def create_rag_result(metrics: dict[str, float], language: str, sample_count: int) -> RAGEvaluationResult:
    """Create RAGEvaluationResult from metrics dictionary.

    Parameters
    ----------
    metrics : dict[str, float]
        Dictionary containing summarization evaluation metrics
    language : str
        Language code used for evaluation
    sample_count : int
        Number of samples evaluated

    Returns
    -------
    RAGEvaluationResult
        Validated RAG evaluation result model

    Raises
    ------
    ValueError
        If required metrics are missing or invalid
    """
    from ..rubric_scorer import RubricScorer

    required_metrics = ["rouge_1_f1", "rouge_2_f1", "rouge_l_f1", "meteor_score", "semantic_similarity_cosine", "semantic_similarity_euclidean", "clinical_embedding_similarity"]

    missing_metrics = [m for m in required_metrics if m not in metrics]
    if missing_metrics:
        raise ValueError(f"Missing required summarization metrics: {missing_metrics}")

    # Create rubric scores for each metric
    rubric_data = {}
    for metric_name in required_metrics:
        if metric_name in metrics:
            rubric_info = RubricScorer.get_metric_rubric_info(metric_name, metrics[metric_name])
            rubric_data[f"{metric_name}_rubric"] = RubricScore(**rubric_info)

    # Create the summarization metrics object
    summarization_metrics_data = {"language": language, "sample_count": sample_count, **metrics, **rubric_data}

    summarization_metrics = RAGSummarizationMetrics(**summarization_metrics_data)

    return RAGEvaluationResult(summarization_metrics=summarization_metrics)
