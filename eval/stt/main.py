"""Speech-to-Text evaluation implementation following ADR-003 specifications.

This module implements comprehensive ASR evaluation metrics as defined in ADR-003:
- WER (Word Error Rate): Word substitutions/insertions/deletions
- MER (Match Error Rate): Match-based word error diagnostic
- WIP/WIL (Word Information Preserved/Lost): Info-centric diagnostics
- CER (Character Error Rate): Character-level edits for Arabic orthography
- LD/NLD (Levenshtein Distance): Raw and normalized edit distances

Supports bilingual evaluation (English and Arabic) with proper text normalization
including Unicode NFKC, Arabic diacritic stripping, and standardized punctuation.

References
----------
- ADR-003: Evaluation framework specification
- jiwer: Word-level metrics computation
- RapidFuzz: Levenshtein distance calculations
"""

import logging
import unicodedata

import jiwer
import rapidfuzz

# Import evaluation models
from eval.models import create_stt_result

logger = logging.getLogger(__name__)


class SpeechToTextEvaluator:
    """Comprehensive ASR evaluation following ADR-003 specifications.

    This evaluator implements all metrics specified in ADR-003 for bilingual
    (English/Arabic) ASR evaluation with proper text normalization and
    minimum-edit alignment.

    Metrics Implemented
    -------------------
    - WER: Word Error Rate (substitutions + insertions + deletions)
    - MER: Match Error Rate (diagnostic complement to WER)
    - WIP: Word Information Preserved (info-centric diagnostic)
    - WIL: Word Information Lost (info-centric diagnostic)
    - CER: Character Error Rate (sensitive to Arabic orthography)
    - LD: Levenshtein Distance (word and character level)
    - NLD: Normalized Levenshtein Distance (0-1 scale)

    Text Normalization
    ------------------
    - Lowercase conversion
    - Unicode NFKC normalization
    - Arabic diacritic stripping (configurable)
    - Standardized punctuation handling
    - Whitespace tokenization for word-level metrics

    Examples
    --------
    >>> evaluator = SpeechToTextEvaluator(language="en")
    >>> refs = ["hello world", "test sentence"]
    >>> hyps = ["hello word", "test sentance"]
    >>> results = evaluator.evaluate_full(refs, hyps)
    >>> print(f"WER: {results.wer:.4f}")
    """

    def __init__(self, language: str = "en", strip_arabic_diacritics: bool = True, normalize_punctuation: bool = True) -> None:
        """Initialize the ASR evaluator.

        Parameters
        ----------
        language : str, optional
            Language code ("en" for English, "ar" for Arabic), by default "en"
        strip_arabic_diacritics : bool, optional
            Whether to remove Arabic diacritics during normalization, by default True
        normalize_punctuation : bool, optional
            Whether to standardize punctuation marks, by default True

        Raises
        ------
        ValueError
            If language is not "en" or "ar"
        """
        if language.lower() not in ("en", "ar"):
            raise ValueError(f"Unsupported language: {language}. Must be 'en' or 'ar'")

        self.language = language.lower()
        self.strip_arabic_diacritics = strip_arabic_diacritics
        self.normalize_punctuation = normalize_punctuation

    def normalize_text(self, text: str) -> str:
        """Normalize text according to ADR-003 evaluation standards.

        Applies comprehensive text normalization for consistent ASR evaluation:
        1. Lowercase conversion
        2. Unicode NFKC normalization (canonical decomposition + compatibility)
        3. Arabic diacritic stripping (if enabled and language is Arabic)
        4. Standardized punctuation handling
        5. Whitespace normalization

        Parameters
        ----------
        text : str
            Raw input text to normalize

        Returns
        -------
        str
            Normalized text ready for metric computation

        Notes
        -----
        Unicode NFKC normalization ensures consistent character representation
        across different encodings and input methods, crucial for Arabic text.
        Arabic diacritic removal follows the ADR specification for optional
        primary evaluation runs without diacritics.

        Examples
        --------
        >>> evaluator = SpeechToTextEvaluator(language="ar")
        >>> text = "مَرْحَباً بِكَ"  # Arabic with diacritics
        >>> normalized = evaluator.normalize_text(text)
        >>> print(normalized)  # "مرحبا بك"
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected string, got {type(text)}")

        # Convert to lowercase
        text = text.lower()

        # Unicode normalization (NFKC) - canonical decomposition + compatibility
        text = unicodedata.normalize("NFKC", text)

        # Strip Arabic diacritics if enabled and language is Arabic
        if self.language == "ar" and self.strip_arabic_diacritics:
            text = self._strip_arabic_diacritics(text)

        # Standardized punctuation handling
        if self.normalize_punctuation:
            text = self._normalize_punctuation(text)

        # Normalize whitespace (collapse multiple spaces, strip edges)
        text = " ".join(text.split())

        return text

    def _strip_arabic_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics from text following ADR-003 specification.

        Removes all Arabic diacritical marks (tashkeel) to enable consistent
        evaluation across different transcription standards and input methods.

        Parameters
        ----------
        text : str
            Arabic text with potential diacritics

        Returns
        -------
        str
            Text with all Arabic diacritics removed

        Notes
        -----
        Covers the complete Unicode range of Arabic diacritics including:
        - Short vowels (fatha, damma, kasra)
        - Sukun (absence of vowel)
        - Shadda (gemination)
        - Tanween (nunation)
        - Various other diacritical marks
        """
        # Complete Arabic diacritics Unicode ranges (U+064B to U+0652, U+0653 to U+065F, U+0670)
        arabic_diacritics = [
            "\u064b",  # Fathatan
            "\u064c",  # Dammatan
            "\u064d",  # Kasratan
            "\u064e",  # Fatha
            "\u064f",  # Damma
            "\u0650",  # Kasra
            "\u0651",  # Shadda
            "\u0652",  # Sukun
            "\u0653",  # Maddah Above
            "\u0654",  # Hamza Above
            "\u0655",  # Hamza Below
            "\u0656",  # Subscript Alef
            "\u0657",  # Inverted Damma
            "\u0658",  # Mark Noon Ghunna
            "\u0659",  # Zwarakay
            "\u065a",  # Vowel Sign Small V Above
            "\u065b",  # Vowel Sign Inverted Small V Above
            "\u065c",  # Vowel Sign Dot Below
            "\u065d",  # Reversed Damma
            "\u065e",  # Fatha With Two Dots
            "\u065f",  # Wavy Hamza Below
            "\u0670",  # Superscript Alef
        ]

        for diacritic in arabic_diacritics:
            text = text.replace(diacritic, "")

        return text

    def _normalize_punctuation(self, text: str) -> str:
        """Standardize punctuation marks for consistent evaluation.

        Parameters
        ----------
        text : str
            Input text with potentially varied punctuation

        Returns
        -------
        str
            Text with standardized punctuation marks

        Notes
        -----
        Normalizes common punctuation variations including:
        - Smart quotes to standard quotes
        - Various apostrophe forms to standard apostrophe
        - Unicode dash variants to standard hyphen
        """
        # Smart quotes to standard quotes
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")

        # Various dash forms to standard hyphen
        text = text.replace("—", "-").replace("–", "-")

        # Normalize ellipsis
        text = text.replace("…", "...")

        return text

    def evaluate_full(self, references: list[str], hypotheses: list[str]):
        """Perform complete evaluation with all metrics.

        Parameters
        ----------
        references : list[str]
            List of reference transcriptions
        hypotheses : list[str]
            List of hypothesis transcriptions

        Returns
        -------
        STTEvaluationResult | dict[str, float]
            Structured evaluation results (Pydantic model if available, dict otherwise)

        Raises
        ------
        ValueError
            If references and hypotheses have different lengths
        """
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses must have same length")

        # Normalize all texts
        norm_refs = [self.normalize_text(ref) for ref in references]
        norm_hyps = [self.normalize_text(hyp) for hyp in hypotheses]

        results = {}

        # Word-level metrics using jiwer
        results.update(self._calculate_word_metrics(norm_refs, norm_hyps))

        # Character-level metrics
        results["cer"] = self._calculate_cer(norm_refs, norm_hyps)

        # Levenshtein distances
        word_ld, char_ld = self._calculate_levenshtein_distances(norm_refs, norm_hyps)
        results["word_levenshtein_distance"] = word_ld
        results["char_levenshtein_distance"] = char_ld

        # Normalized Levenshtein distances
        results["normalized_word_ld"] = self._calculate_normalized_ld(norm_refs, norm_hyps, level="word")
        results["normalized_char_ld"] = self._calculate_normalized_ld(norm_refs, norm_hyps, level="char")

        # Return structured Pydantic model
        return create_stt_result(metrics=results, language=self.language, sample_count=len(references))

    def _calculate_word_metrics(self, references: list[str], hypotheses: list[str]) -> dict[str, float]:
        """Calculate word-level metrics using jiwer."""
        # Use lists directly - jiwer can handle list inputs
        wer = jiwer.wer(references, hypotheses)
        mer = jiwer.mer(references, hypotheses)
        wip = jiwer.wip(references, hypotheses)
        wil = jiwer.wil(references, hypotheses)

        return {"wer": wer, "mer": mer, "wip": wip, "wil": wil}

    def _calculate_cer(self, references: list[str], hypotheses: list[str]) -> float:
        """Calculate Character Error Rate."""
        # Use lists directly - jiwer.cer can handle list inputs
        return jiwer.cer(references, hypotheses)

    def _calculate_levenshtein_distances(self, references: list[str], hypotheses: list[str]) -> tuple[float, float]:
        """Calculate average Levenshtein distances at word and character level."""
        word_distances = []
        char_distances = []

        for ref, hyp in zip(references, hypotheses):
            # Word-level distance
            ref_words = ref.split()
            hyp_words = hyp.split()
            word_dist = rapidfuzz.distance.Levenshtein.distance(ref_words, hyp_words)
            word_distances.append(word_dist)

            # Character-level distance
            char_dist = rapidfuzz.distance.Levenshtein.distance(ref, hyp)
            char_distances.append(char_dist)

        return sum(word_distances) / len(word_distances), sum(char_distances) / len(char_distances)

    def _calculate_normalized_ld(self, references: list[str], hypotheses: list[str], level: str = "word") -> float:
        """Calculate normalized Levenshtein distance."""
        distances = []

        for ref, hyp in zip(references, hypotheses):
            if level == "word":
                ref_items = ref.split()
                hyp_items = hyp.split()
            else:  # character level
                ref_items = list(ref)
                hyp_items = list(hyp)

            distance = rapidfuzz.distance.Levenshtein.distance(ref_items, hyp_items)
            max_len = max(len(ref_items), len(hyp_items), 1)  # Avoid division by zero
            normalized_distance = distance / max_len
            distances.append(normalized_distance)

        return sum(distances) / len(distances)


# Convenience functions for direct metric calculation
def calculate_wer(references: list[str], hypotheses: list[str], language: str = "en") -> float:
    """Calculate Word Error Rate."""
    evaluator = SpeechToTextEvaluator(language=language)
    # Normalize texts
    norm_refs = [evaluator.normalize_text(ref) for ref in references]
    norm_hyps = [evaluator.normalize_text(hyp) for hyp in hypotheses]
    # Calculate directly using jiwer with lists
    return jiwer.wer(norm_refs, norm_hyps)


def calculate_mer(references: list[str], hypotheses: list[str], language: str = "en") -> float:
    """Calculate Match Error Rate."""
    evaluator = SpeechToTextEvaluator(language=language)
    # Normalize texts
    norm_refs = [evaluator.normalize_text(ref) for ref in references]
    norm_hyps = [evaluator.normalize_text(hyp) for hyp in hypotheses]
    # Calculate directly using jiwer with lists
    return jiwer.mer(norm_refs, norm_hyps)


def calculate_wip_wil(references: list[str], hypotheses: list[str], language: str = "en") -> tuple[float, float]:
    """Calculate Word Information Preserved and Lost."""
    evaluator = SpeechToTextEvaluator(language=language)
    # Normalize texts
    norm_refs = [evaluator.normalize_text(ref) for ref in references]
    norm_hyps = [evaluator.normalize_text(hyp) for hyp in hypotheses]
    # Calculate directly using jiwer with lists
    wip = jiwer.wip(norm_refs, norm_hyps)
    wil = jiwer.wil(norm_refs, norm_hyps)
    return wip, wil


def calculate_cer(references: list[str], hypotheses: list[str], language: str = "en") -> float:
    """Calculate Character Error Rate."""
    evaluator = SpeechToTextEvaluator(language=language)
    # Normalize texts
    norm_refs = [evaluator.normalize_text(ref) for ref in references]
    norm_hyps = [evaluator.normalize_text(hyp) for hyp in hypotheses]
    # Calculate directly using jiwer with lists
    return jiwer.cer(norm_refs, norm_hyps)


def calculate_levenshtein_distance(references: list[str], hypotheses: list[str], level: str = "word", normalize: bool = False, language: str = "en") -> float:
    """Calculate Levenshtein Distance."""
    evaluator = SpeechToTextEvaluator(language=language)
    # Normalize texts
    norm_refs = [evaluator.normalize_text(ref) for ref in references]
    norm_hyps = [evaluator.normalize_text(hyp) for hyp in hypotheses]

    if normalize:
        return evaluator._calculate_normalized_ld(norm_refs, norm_hyps, level=level)
    else:
        word_ld, char_ld = evaluator._calculate_levenshtein_distances(norm_refs, norm_hyps)
        return word_ld if level == "word" else char_ld


def normalize_text(text: str, language: str = "en", strip_arabic_diacritics: bool = True) -> str:
    """Normalize text according to evaluation standards."""
    evaluator = SpeechToTextEvaluator(language=language, strip_arabic_diacritics=strip_arabic_diacritics)
    return evaluator.normalize_text(text)


def evaluate_asr_model(references: list[str], hypotheses: list[str], language: str = "en"):
    """Comprehensive ASR model evaluation.

    Parameters
    ----------
    references : list[str]
        List of reference transcriptions
    hypotheses : list[str]
        List of hypothesis transcriptions
    language : str, optional
        Language code for evaluation ("en" or "ar"), by default "en"

    Returns
    -------
    STTEvaluationResult | dict[str, float]
        Structured evaluation results (Pydantic model if available, dict otherwise)

    Examples
    --------
    >>> refs = ["hello world", "test sentence"]
    >>> hyps = ["hello word", "test sentance"]
    >>> results = evaluate_asr_model(refs, hyps, language="en")
    >>> print(f"WER: {results.wer:.4f}" if hasattr(results, 'wer') else f"WER: {results['wer']:.4f}")
    """
    evaluator = SpeechToTextEvaluator(language=language)
    return evaluator.evaluate_full(references, hypotheses)


if __name__ == "__main__":
    # Example usage
    references = ["hello world this is a test", "another example sentence"]
    hypotheses = ["hello word this is test", "another exampl sentence"]

    evaluator = SpeechToTextEvaluator(language="en")
    results = evaluator.evaluate_full(references, hypotheses)

    logger.info("Speech-to-Text Evaluation Results (with ADR-003 Rubric Scores):")
    logger.info(f"WER: {results.wer:.4f} -> {results.wer_rubric.rubric_score}/4 ({results.wer_rubric.rubric_label})")
    logger.info(f"MER: {results.mer:.4f} -> {results.mer_rubric.rubric_score}/4 ({results.mer_rubric.rubric_label})")
    logger.info(f"WIP: {results.wip:.4f} -> {results.wip_rubric.rubric_score}/4 ({results.wip_rubric.rubric_label})")
    logger.info(f"WIL: {results.wil:.4f} -> {results.wil_rubric.rubric_score}/4 ({results.wil_rubric.rubric_label})")
    logger.info(f"CER: {results.cer:.4f} -> {results.cer_rubric.rubric_score}/4 ({results.cer_rubric.rubric_label})")
    logger.info(f"Word LD: {results.word_levenshtein_distance:.4f} -> {results.word_ld_rubric.rubric_score}/4 ({results.word_ld_rubric.rubric_label})")
    logger.info(f"Char LD: {results.char_levenshtein_distance:.4f} -> {results.char_ld_rubric.rubric_score}/4 ({results.char_ld_rubric.rubric_label})")
    logger.info(f"Normalized Word LD: {results.normalized_word_ld:.4f} -> {results.normalized_word_ld_rubric.rubric_score}/4 ({results.normalized_word_ld_rubric.rubric_label})")
    logger.info(f"Normalized Char LD: {results.normalized_char_ld:.4f} -> {results.normalized_char_ld_rubric.rubric_score}/4 ({results.normalized_char_ld_rubric.rubric_label})")
