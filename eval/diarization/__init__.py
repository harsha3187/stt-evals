"""Diarization Evaluation Module.

This module provides evaluation metrics for Speaker Diarization systems.
Implements DER (Diarization Error Rate) and JER (Jaccard Error Rate) calculations
as specified in the evaluation ADR.

Metrics computed:
- DER: Diarization Error Rate (missed speech + false alarm + speaker confusion)
- JER: Jaccard Error Rate (1 - avg. Jaccard over mapped speakers)

Datasets supported:
- VoxConverse (EN with RTTM labels)
- QASR (AR with speaker segments when available)

Frameworks used:
- pyannote.metrics for DER/JER from RTTM
- NIST dscore for DER/JER breakdown
"""

__version__ = "1.0.0"
__author__ = "Oryx CAP Upskilling Team"

# Import main evaluation classes and functions
from .main import (
    DiarizationEvaluator,
    calculate_der,
    calculate_jer,
    evaluate_diarization_model,
    load_rttm_file,
    parse_rttm_line,
)

__all__: list[str] = ["DiarizationEvaluator", "calculate_der", "calculate_jer", "evaluate_diarization_model", "load_rttm_file", "parse_rttm_line"]
