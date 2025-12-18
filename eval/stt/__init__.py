"""Speech-to-Text Evaluation Module.

This module provides evaluation metrics for Automatic Speech Recognition (ASR) systems.
Implements WER, MER, WIP, WIL, CER, and Levenshtein Distance calculations as specified
in the evaluation ADR.

Metrics computed:
- WER: Word Error Rate (substitutions/insertions/deletions)
- MER: Match-based word error rate
- WIP: Word Information Preserved
- WIL: Word Information Lost
- CER: Character Error Rate
- LD: Levenshtein Distance (word/character level)
- NLD: Normalized Levenshtein Distance

Datasets supported:
- Common Voice Mozilla (EN, AR)
- LibriSpeech (EN)
- QASR (AR - MSA + dialects)
- MGB2/MGB3 (AR broadcast)
"""

__version__ = "1.0.0"
__author__ = "Oryx CAP Upskilling Team"

# Import main evaluation classes and functions
from .main import (
    SpeechToTextEvaluator,
    calculate_cer,
    calculate_levenshtein_distance,
    calculate_mer,
    calculate_wer,
    calculate_wip_wil,
    evaluate_asr_model,
    normalize_text,
)

__all__ = ["SpeechToTextEvaluator", "calculate_wer", "calculate_mer", "calculate_wip_wil", "calculate_cer", "calculate_levenshtein_distance", "normalize_text", "evaluate_asr_model"]
