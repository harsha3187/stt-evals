"""Rubric scoring system for evaluation metrics based on ADR-003 specifications.

This module implements the rubric score mapping defined in ADR-003 Appendix,
providing 4-point scale scoring (1=Poor, 2=Fair, 3=Good, 4=Excellent) for
all evaluation metrics including ASR, Diarization, Retrieval, and RAG.
"""


class RubricScorer:
    """Converts raw evaluation metrics to 4-point rubric scores based on ADR-003 specifications."""

    # Rubric thresholds from ADR-003 Appendix
    RUBRIC_THRESHOLDS = {
        # ASR/STT Metrics
        "wer": {
            4: (0.0, 0.05),  # Excellent: < 5%
            3: (0.05, 0.15),  # Good: 5-15%
            2: (0.15, 0.30),  # Fair: 15-30%
            1: (0.30, float("inf")),  # Poor: > 30%
        },
        "mer": {
            4: (0.0, 0.05),  # Excellent: < 5%
            3: (0.05, 0.15),  # Good: 5-15%
            2: (0.15, 0.30),  # Fair: 15-30%
            1: (0.30, float("inf")),  # Poor: > 30%
        },
        "wip": {
            4: (0.95, 1.0),  # Excellent: > 95%
            3: (0.80, 0.95),  # Good: 80-95%
            2: (0.60, 0.80),  # Fair: 60-80%
            1: (0.0, 0.60),  # Poor: < 60%
        },
        "wil": {
            4: (0.0, 0.05),  # Excellent: < 5%
            3: (0.05, 0.20),  # Good: 5-20%
            2: (0.20, 0.40),  # Fair: 20-40%
            1: (0.40, float("inf")),  # Poor: > 40%
        },
        "cer": {
            4: (0.0, 0.05),  # Excellent: < 5%
            3: (0.05, 0.15),  # Good: 5-15%
            2: (0.15, 0.30),  # Fair: 15-30%
            1: (0.30, float("inf")),  # Poor: > 30%
        },
        "normalized_word_ld": {
            4: (0.0, 0.05),  # Excellent: < 5%
            3: (0.05, 0.15),  # Good: 5-15%
            2: (0.15, 0.30),  # Fair: 15-30%
            1: (0.30, float("inf")),  # Poor: > 30%
        },
        "normalized_char_ld": {
            4: (0.0, 0.05),  # Excellent: < 5%
            3: (0.05, 0.15),  # Good: 5-15%
            2: (0.15, 0.30),  # Fair: 15-30%
            1: (0.30, float("inf")),  # Poor: > 30%
        },
        # Levenshtein Distance (raw) - using edit count thresholds
        "word_levenshtein_distance": {
            4: (0.0, 5.0),  # Excellent: < 5 edits
            3: (5.0, 15.0),  # Good: 5-15 edits
            2: (15.0, 30.0),  # Fair: 15-30 edits
            1: (30.0, float("inf")),  # Poor: > 30 edits
        },
        "char_levenshtein_distance": {
            4: (0.0, 5.0),  # Excellent: < 5 edits
            3: (5.0, 15.0),  # Good: 5-15 edits
            2: (15.0, 30.0),  # Fair: 15-30 edits
            1: (30.0, float("inf")),  # Poor: > 30 edits
        },
        # Diarization Metrics
        "der": {
            4: (0.0, 0.10),  # Excellent: < 10%
            3: (0.10, 0.20),  # Good: 10-20%
            2: (0.20, 0.30),  # Fair: 20-30%
            1: (0.30, 1.0),  # Poor: > 30%
        },
        "jer": {
            4: (0.0, 0.10),  # Excellent: < 10%
            3: (0.10, 0.20),  # Good: 10-20%
            2: (0.20, 0.30),  # Fair: 20-30%
            1: (0.30, 1.0),  # Poor: > 30%
        },
        "missed_speech": {
            4: (0.0, 0.05),  # Excellent: < 5%
            3: (0.05, 0.15),  # Good: 5-15%
            2: (0.15, 0.30),  # Fair: 15-30%
            1: (0.30, 1.0),  # Poor: > 30%
        },
        "false_alarm": {
            4: (0.0, 0.05),  # Excellent: < 5%
            3: (0.05, 0.15),  # Good: 5-15%
            2: (0.15, 0.30),  # Fair: 15-30%
            1: (0.30, float("inf")),  # Poor: > 30%
        },
        "speaker_confusion": {
            4: (0.0, 0.05),  # Excellent: < 5%
            3: (0.05, 0.15),  # Good: 5-15%
            2: (0.15, 0.30),  # Fair: 15-30%
            1: (0.30, 1.0),  # Poor: > 30%
        },
        "ser": {
            4: (0.0, 0.15),  # Excellent: < 15%
            3: (0.15, 0.30),  # Good: 15-30%
            2: (0.30, 0.50),  # Fair: 30-50%
            1: (0.50, 1.0),  # Poor: > 50%
        },
        "ber": {
            4: (0.0, 0.15),  # Excellent: < 15%
            3: (0.15, 0.30),  # Good: 15-30%
            2: (0.30, 0.50),  # Fair: 30-50%
            1: (0.50, float("inf")),  # Poor: > 50%
        },
        # Retrieval Metrics
        "ndcg_at_k": {
            4: (0.90, 1.0),  # Excellent: > 90%
            3: (0.75, 0.90),  # Good: 75-90%
            2: (0.50, 0.75),  # Fair: 50-75%
            1: (0.0, 0.50),  # Poor: < 50%
        },
        "mrr_at_k": {
            4: (0.90, 1.0),  # Excellent: > 90%
            3: (0.75, 0.90),  # Good: 75-90%
            2: (0.50, 0.75),  # Fair: 50-75%
            1: (0.0, 0.50),  # Poor: < 50%
        },
        "precision_at_k": {
            4: (0.80, 1.0),  # Excellent: > 80%
            3: (0.60, 0.80),  # Good: 60-80%
            2: (0.30, 0.60),  # Fair: 30-60%
            1: (0.0, 0.30),  # Poor: < 30%
        },
        "recall_at_k": {
            4: (0.80, 1.0),  # Excellent: > 80%
            3: (0.60, 0.80),  # Good: 60-80%
            2: (0.30, 0.60),  # Fair: 30-60%
            1: (0.0, 0.30),  # Poor: < 30%
        },
        "map_at_k": {
            4: (0.80, 1.0),  # Excellent: > 80%
            3: (0.60, 0.80),  # Good: 60-80%
            2: (0.30, 0.60),  # Fair: 30-60%
            1: (0.0, 0.30),  # Poor: < 30%
        },
        "hit_rate_at_k": {
            4: (1.0, 1.0),  # Excellent: = 100%
            3: (0.75, 1.0),  # Good: ≥ 75%
            2: (0.50, 0.75),  # Fair: ≥ 50%
            1: (0.0, 0.50),  # Poor: < 50%
        },
        # Summarization Metrics (following ADR-003 for text quality)
        "rouge_1_f1": {
            4: (0.80, 1.0),  # Excellent: > 80%
            3: (0.60, 0.80),  # Good: 60-80%
            2: (0.40, 0.60),  # Fair: 40-60%
            1: (0.0, 0.40),  # Poor: < 40%
        },
        "rouge_2_f1": {
            4: (0.70, 1.0),  # Excellent: > 70%
            3: (0.50, 0.70),  # Good: 50-70%
            2: (0.30, 0.50),  # Fair: 30-50%
            1: (0.0, 0.30),  # Poor: < 30%
        },
        "rouge_l_f1": {
            4: (0.75, 1.0),  # Excellent: > 75%
            3: (0.55, 0.75),  # Good: 55-75%
            2: (0.35, 0.55),  # Fair: 35-55%
            1: (0.0, 0.35),  # Poor: < 35%
        },
        "meteor_score": {
            4: (0.70, 1.0),  # Excellent: > 70%
            3: (0.50, 0.70),  # Good: 50-70%
            2: (0.30, 0.50),  # Fair: 30-50%
            1: (0.0, 0.30),  # Poor: < 30%
        },
        "semantic_similarity_cosine": {
            4: (0.90, 1.0),  # Excellent: > 90%
            3: (0.80, 0.90),  # Good: 80-90%
            2: (0.70, 0.80),  # Fair: 70-80%
            1: (0.0, 0.70),  # Poor: < 70%
        },
        "semantic_similarity_euclidean": {
            4: (0.90, 1.0),  # Excellent: > 90% (normalized inverse distance)
            3: (0.80, 0.90),  # Good: 80-90%
            2: (0.70, 0.80),  # Fair: 70-80%
            1: (0.0, 0.70),  # Poor: < 70%
        },
        "clinical_embedding_similarity": {
            4: (0.85, 1.0),  # Excellent: > 85% (clinical domain specific)
            3: (0.75, 0.85),  # Good: 75-85%
            2: (0.65, 0.75),  # Fair: 65-75%
            1: (0.0, 0.65),  # Poor: < 65%
        },
    }

    # Score labels for human-readable output
    SCORE_LABELS = {4: "Excellent", 3: "Good", 2: "Fair", 1: "Poor"}

    @classmethod
    def get_rubric_score(cls, metric_name: str, metric_value: float) -> int:
        """Get rubric score (1-4) for a metric value.

        Parameters
        ----------
        metric_name : str
            Name of the metric (e.g., 'wer', 'der', 'precision_at_k')
        metric_value : float
            Raw metric value

        Returns
        -------
        int
            Rubric score from 1 (Poor) to 4 (Excellent)

        Raises
        ------
        ValueError
            If metric_name is not recognized
        """
        if metric_name not in cls.RUBRIC_THRESHOLDS:
            raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {list(cls.RUBRIC_THRESHOLDS.keys())}")

        thresholds = cls.RUBRIC_THRESHOLDS[metric_name]

        # Check each score level (4 to 1)
        for score in [4, 3, 2, 1]:
            min_val, max_val = thresholds[score]
            if min_val <= metric_value < max_val:
                return score
            # Handle edge case for exact max value in highest range
            elif score == 4 and metric_value == max_val and max_val != float("inf"):
                return score

        # Fallback - should not reach here with proper thresholds
        return 1

    @classmethod
    def get_rubric_label(cls, score: int) -> str:
        """Get human-readable label for rubric score.

        Parameters
        ----------
        score : int
            Rubric score (1-4)

        Returns
        -------
        str
            Human-readable label (Poor, Fair, Good, Excellent)
        """
        return cls.SCORE_LABELS.get(score, "Unknown")

    @classmethod
    def get_metric_rubric_info(cls, metric_name: str, metric_value: float) -> dict[str, int | str | float]:
        """Get complete rubric information for a metric.

        Parameters
        ----------
        metric_name : str
            Name of the metric
        metric_value : float
            Raw metric value

        Returns
        -------
        Dict[str, Union[int, str, float]]
            Dictionary containing:
            - 'raw_value': Original metric value
            - 'rubric_score': Score from 1-4
            - 'rubric_label': Human-readable label
            - 'metric_name': Name of the metric
        """
        rubric_score = cls.get_rubric_score(metric_name, metric_value)
        return {"raw_value": metric_value, "rubric_score": rubric_score, "rubric_label": cls.get_rubric_label(rubric_score), "metric_name": metric_name}

    @classmethod
    def get_available_metrics(cls) -> list[str]:
        """Get list of all supported metrics.

        Returns
        -------
        list[str]
            List of metric names that can be scored
        """
        return list(cls.RUBRIC_THRESHOLDS.keys())
