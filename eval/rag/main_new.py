"""RAG summarization evaluation implementation following ADR-003 specifications.

This module implements summarization quality metrics for RAG evaluation as defined in ADR-003:
- ROUGE-1, ROUGE-2, ROUGE-L F1 scores for n-gram overlap
- METEOR score for semantic alignment with synonymy
- Semantic similarity using clinical embeddings (MedEmbed for healthcare)
- Cosine similarity and normalized Euclidean distance metrics

Supports bilingual evaluation (English and Arabic) with proper text normalization
following the same patterns as STT evaluation for consistency.

References
----------
- ADR-003: Evaluation framework specification
- rouge-score: ROUGE metrics computation
- nltk: METEOR score computation
- sentence-transformers: Clinical embeddings (MedEmbed-large-v0.1)
"""

import json
import logging
from pathlib import Path
from typing import Any
import unicodedata

# Import evaluation models
from eval.models import create_rag_result

logger = logging.getLogger(__name__)

# Global model cache to avoid re-downloading models
_MODEL_CACHE = {}


def _get_cached_sentence_transformer(model_name: str, fallback_model: str = None):
    """Get a cached SentenceTransformer model or load it if not cached.

    Parameters
    ----------
    model_name : str
        Primary model name to load
    fallback_model : str, optional
        Fallback model name if primary fails

    Returns
    -------
    SentenceTransformer
        The loaded model instance

    Raises
    ------
    ImportError
        If sentence_transformers is not available
    """
    # Try primary model first
    if model_name in _MODEL_CACHE:
        logger.debug(f"Using cached model: {model_name}")
        return _MODEL_CACHE[model_name]

    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading model: {model_name} (first time, will be cached)")
        model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model
        logger.info(f"Successfully loaded and cached model: {model_name}")
        return model
    except Exception as e:
        logger.warning(f"Failed to load primary model {model_name}: {e}")

        # Try fallback if provided
        if fallback_model and fallback_model != model_name:
            try:
                logger.info(f"Loading fallback model: {fallback_model}")
                model = SentenceTransformer(fallback_model)
                _MODEL_CACHE[fallback_model] = model
                logger.info(f"Successfully loaded and cached fallback model: {fallback_model}")
                return model
            except Exception as fallback_e:
                logger.error(f"Failed to load fallback model {fallback_model}: {fallback_e}")
                raise fallback_e
        else:
            raise e


def clear_model_cache():
    """Clear the global model cache to free memory."""
    global _MODEL_CACHE
    cache_size = len(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    logger.info(f"Cleared model cache ({cache_size} models removed)")


class RAGSummarizationEvaluator:
    """Comprehensive summarization evaluation for RAG systems following ADR-003 specifications.

    This evaluator implements summarization quality metrics for RAG evaluation with proper
    text normalization and clinical domain optimization for healthcare applications.

    Metrics Implemented
    -------------------
    - ROUGE-1, ROUGE-2, ROUGE-L F1: N-gram overlap metrics
    - METEOR: Semantic alignment metric with synonymy support
    - Semantic Similarity (Cosine): Vector similarity using clinical embeddings
    - Semantic Similarity (Euclidean): Normalized distance-based similarity
    - Clinical Embedding Similarity: Healthcare domain-specific weighted similarity

    Text Normalization
    ------------------
    - Lowercase conversion
    - Unicode NFKC normalization
    - Arabic diacritic stripping (configurable)
    - Standardized punctuation handling
    - Whitespace tokenization

    Examples
    --------
    >>> evaluator = RAGSummarizationEvaluator(language="en")
    >>> refs = ["The patient shows signs of improvement.", "Treatment is effective."]
    >>> cands = ["Patient is improving significantly.", "The treatment works well."]
    >>> results = evaluator.evaluate_full(refs, cands)
    >>> print(f"ROUGE-1 F1: {results.rouge_1_f1:.4f}")
    """

    def __init__(self, language: str = "en", strip_arabic_diacritics: bool = True, normalize_punctuation: bool = True) -> None:
        """Initialize the RAG summarization evaluator.

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

        Applies comprehensive text normalization for consistent RAG evaluation:
        1. Lowercase conversion
        2. Unicode NFKC normalization (canonical decomposition + compatibility)
        3. Arabic diacritic stripping (if enabled and language is Arabic)
        4. Standardized punctuation handling
        5. Whitespace normalization

        Parameters
        ----------
        text : str
            Input text to normalize

        Returns
        -------
        str
            Normalized text ready for evaluation
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Lowercase conversion
        normalized = text.lower()

        # Step 2: Unicode NFKC normalization
        normalized = unicodedata.normalize("NFKC", normalized)

        # Step 3: Arabic diacritic stripping (if enabled)
        if self.language == "ar" and self.strip_arabic_diacritics:
            # Remove Arabic diacritics (short vowels, shadda, etc.)
            arabic_diacritics = [
                "\u064b",
                "\u064c",
                "\u064d",
                "\u064e",
                "\u064f",  # Tanween, Fatha, Damma
                "\u0650",
                "\u0651",
                "\u0652",
                "\u0653",
                "\u0654",  # Kasra, Shadda, Sukun, etc.
                "\u0655",
                "\u0656",
                "\u0657",
                "\u0658",
                "\u0659",
                "\u065a",
                "\u065b",
                "\u065c",
                "\u065d",
                "\u065e",
                "\u0670",
                "\u06d6",
                "\u06d7",
                "\u06d8",
                "\u06d9",
                "\u06da",
                "\u06db",
                "\u06dc",
                "\u06df",
                "\u06e0",
                "\u06e1",
                "\u06e2",
                "\u06e3",
                "\u06e4",
                "\u06e7",
                "\u06e8",
                "\u06ea",
                "\u06eb",
                "\u06ec",
                "\u06ed",
            ]
            for diacritic in arabic_diacritics:
                normalized = normalized.replace(diacritic, "")

        # Step 4: Standardized punctuation handling (if enabled)
        if self.normalize_punctuation:
            # Replace various punctuation marks with standard forms
            punctuation_map = {
                """: "'", """: "'",
                '"': '"',
                '"': '"',  # Smart quotes
                "–": "-",
                "—": "-",  # Dashes
                "…": "...",  # Ellipsis
                "«": '"',
                "»": '"',  # Guillemets
            }
            for old_punct, new_punct in punctuation_map.items():
                normalized = normalized.replace(old_punct, new_punct)

        # Step 5: Whitespace normalization
        normalized = " ".join(normalized.split())

        return normalized

    def compute_summarization_metrics(self, references: list[str], candidates: list[str]) -> dict[str, float]:
        """Compute comprehensive summarization quality metrics.

        Parameters
        ----------
        references : list[str]
            Reference/ground truth texts
        candidates : list[str]
            Generated candidate texts

        Returns
        -------
        dict[str, float]
            Dictionary containing all computed metrics

        Raises
        ------
        ValueError
            If references and candidates have different lengths
        ImportError
            If required libraries are not available
        """
        if len(references) != len(candidates):
            raise ValueError(f"Length mismatch: {len(references)} references vs {len(candidates)} candidates")

        if not references:
            raise ValueError("Empty references and candidates provided")

        logger.info(f"Computing summarization metrics for {len(references)} pairs")

        # Normalize all texts
        norm_refs = [self.normalize_text(ref) for ref in references]
        norm_cands = [self.normalize_text(cand) for cand in candidates]

        metrics = {}

        # Compute ROUGE metrics
        try:
            from rouge_score import rouge_scorer

            rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

            rouge_1_scores = []
            rouge_2_scores = []
            rouge_l_scores = []

            for ref, cand in zip(norm_refs, norm_cands):
                if not ref.strip() or not cand.strip():
                    rouge_1_scores.append(0.0)
                    rouge_2_scores.append(0.0)
                    rouge_l_scores.append(0.0)
                    continue

                scores = rouge_scorer_obj.score(ref, cand)
                rouge_1_scores.append(scores["rouge1"].fmeasure)
                rouge_2_scores.append(scores["rouge2"].fmeasure)
                rouge_l_scores.append(scores["rougeL"].fmeasure)

            metrics["rouge_1_f1"] = float(sum(rouge_1_scores) / len(rouge_1_scores))
            metrics["rouge_2_f1"] = float(sum(rouge_2_scores) / len(rouge_2_scores))
            metrics["rouge_l_f1"] = float(sum(rouge_l_scores) / len(rouge_l_scores))

        except ImportError as e:
            logger.warning(f"ROUGE computation failed - missing dependency: {e}")
            metrics["rouge_1_f1"] = 0.0
            metrics["rouge_2_f1"] = 0.0
            metrics["rouge_l_f1"] = 0.0

        # Compute METEOR score
        try:
            import nltk
            from nltk.translate.meteor_score import meteor_score

            # Ensure NLTK data is available
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                logger.info("Downloading NLTK punkt tokenizer...")
                nltk.download("punkt", quiet=True)

            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                logger.info("Downloading NLTK wordnet...")
                nltk.download("wordnet", quiet=True)

            meteor_scores = []
            for ref, cand in zip(norm_refs, norm_cands):
                if not ref.strip() or not cand.strip():
                    meteor_scores.append(0.0)
                    continue

                ref_tokens = ref.split()
                cand_tokens = cand.split()

                if not ref_tokens or not cand_tokens:
                    meteor_scores.append(0.0)
                    continue

                score = meteor_score([ref_tokens], cand_tokens)
                meteor_scores.append(score)

            metrics["meteor_score"] = float(sum(meteor_scores) / len(meteor_scores))

        except ImportError as e:
            logger.warning(f"METEOR computation failed - missing dependency: {e}")
            metrics["meteor_score"] = 0.0
        except Exception as e:
            logger.warning(f"METEOR computation failed: {e}")
            metrics["meteor_score"] = 0.0

        # Compute semantic similarity using clinical embeddings
        try:
            import numpy as np
            from scipy.spatial.distance import euclidean
            from sklearn.metrics.pairwise import cosine_similarity

            # Filter out empty pairs
            valid_pairs = [(ref, cand) for ref, cand in zip(norm_refs, norm_cands) if ref.strip() and cand.strip()]

            if valid_pairs:
                valid_refs, valid_cands = zip(*valid_pairs)

                # Choose appropriate model based on language and domain
                if self.language == "en":
                    # Use clinical/medical specific model for English
                    model_name = "abhinand/MedEmbed-large-v0.1"  # Clinical embeddings
                    fallback_model = "all-MiniLM-L6-v2"  # Lightweight fallback
                else:
                    # Use multilingual model for Arabic
                    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

                # Use cached model loading
                model = _get_cached_sentence_transformer(model_name, fallback_model)

                # Generate embeddings
                ref_embeddings = model.encode(list(valid_refs))
                cand_embeddings = model.encode(list(valid_cands))

                # Compute similarities
                cosine_scores = []
                euclidean_scores = []
                clinical_scores = []

                for ref_emb, cand_emb in zip(ref_embeddings, cand_embeddings):
                    # Cosine similarity
                    cos_sim = cosine_similarity([ref_emb], [cand_emb])[0][0]
                    cosine_scores.append(max(0.0, float(cos_sim)))

                    # Normalized Euclidean similarity (convert distance to similarity)
                    eucl_dist = euclidean(ref_emb, cand_emb)
                    normalized_eucl_sim = 1.0 / (1.0 + eucl_dist / np.sqrt(len(ref_emb)))
                    euclidean_scores.append(float(normalized_eucl_sim))

                    # Clinical embedding similarity (weighted for healthcare domain)
                    clinical_sim = 0.7 * cos_sim + 0.3 * normalized_eucl_sim
                    clinical_scores.append(max(0.0, float(clinical_sim)))

                metrics["semantic_similarity_cosine"] = float(sum(cosine_scores) / len(cosine_scores))
                metrics["semantic_similarity_euclidean"] = float(sum(euclidean_scores) / len(euclidean_scores))
                metrics["clinical_embedding_similarity"] = float(sum(clinical_scores) / len(clinical_scores))
            else:
                metrics["semantic_similarity_cosine"] = 0.0
                metrics["semantic_similarity_euclidean"] = 0.0
                metrics["clinical_embedding_similarity"] = 0.0

        except ImportError as e:
            logger.warning(f"Semantic similarity computation failed - missing dependency: {e}")
            metrics["semantic_similarity_cosine"] = 0.0
            metrics["semantic_similarity_euclidean"] = 0.0
            metrics["clinical_embedding_similarity"] = 0.0
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            metrics["semantic_similarity_cosine"] = 0.0
            metrics["semantic_similarity_euclidean"] = 0.0
            metrics["clinical_embedding_similarity"] = 0.0

        logger.info(f"Computed metrics: ROUGE-1={metrics['rouge_1_f1']:.4f}, ROUGE-2={metrics['rouge_2_f1']:.4f}, METEOR={metrics['meteor_score']:.4f}")

        return metrics

    def evaluate_full(self, references: list[str], candidates: list[str]) -> Any:
        """Perform complete RAG summarization evaluation.

        Parameters
        ----------
        references : list[str]
            List of reference texts
        candidates : list[str]
            List of candidate texts

        Returns
        -------
        RAGEvaluationResult
            Comprehensive evaluation results with Pydantic validation

        Raises
        ------
        ValueError
            If input validation fails
        """
        if not references or not candidates:
            raise ValueError("References and candidates cannot be empty")

        if len(references) != len(candidates):
            raise ValueError(f"Length mismatch: {len(references)} references != {len(candidates)} candidates")

        logger.info(f"Starting RAG summarization evaluation for {len(references)} pairs")
        logger.info(f"Language: {self.language}")

        # Compute all metrics
        metrics = self.compute_summarization_metrics(references, candidates)

        # Create structured result using Pydantic model
        result = create_rag_result(metrics, self.language, len(references))

        logger.info("RAG summarization evaluation completed successfully")
        return result


def load_jsonl_texts(file_path: Path, text_field: str = "text") -> list[str]:
    """Load texts from JSONL file.

    Parameters
    ----------
    file_path : Path
        Path to JSONL file
    text_field : str, optional
        Field name containing the text, by default "text"

    Returns
    -------
    list[str]
        List of extracted texts

    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If file format is invalid or text field is missing
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    texts = []
    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                if text_field not in data:
                    raise ValueError(f"Missing '{text_field}' field in line {line_num}")
                texts.append(str(data[text_field]))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in line {line_num}: {e}")

    if not texts:
        raise ValueError(f"No valid texts found in {file_path}")

    logger.info(f"Loaded {len(texts)} texts from {file_path}")
    return texts


def evaluate_rag_model(references_file: str | Path, candidates_file: str | Path, language: str = "en", text_field: str = "text") -> Any:
    """Evaluate RAG model summarization quality using file inputs.

    This function follows the same pattern as evaluate_asr_model from STT module
    for consistency and ease of use.

    Parameters
    ----------
    references_file : str | Path
        Path to reference texts file (JSONL format)
    candidates_file : str | Path
        Path to candidate texts file (JSONL format)
    language : str, optional
        Language code ("en" or "ar"), by default "en"
    text_field : str, optional
        JSON field containing text data, by default "text"

    Returns
    -------
    RAGEvaluationResult
        Comprehensive evaluation results with metrics and rubric scores

    Raises
    ------
    FileNotFoundError
        If input files do not exist
    ValueError
        If file formats are invalid or texts don't match
    """
    references_path = Path(references_file)
    candidates_path = Path(candidates_file)

    logger.info(f"Loading reference texts from {references_path}")
    references = load_jsonl_texts(references_path, text_field)

    logger.info(f"Loading candidate texts from {candidates_path}")
    candidates = load_jsonl_texts(candidates_path, text_field)

    if len(references) != len(candidates):
        raise ValueError(f"Mismatch: {len(references)} references != {len(candidates)} candidates")

    # Initialize evaluator and run evaluation
    evaluator = RAGSummarizationEvaluator(language=language)
    return evaluator.evaluate_full(references, candidates)


def preload_models(language: str = "en") -> None:
    """Preload embedding models to avoid download during evaluation.

    This is useful for production environments where you want to download
    models once during initialization.

    Parameters
    ----------
    language : str, optional
        Language code to determine which models to preload, by default "en"
    """
    logger.info("Preloading embedding models for RAG evaluation...")

    try:
        if language == "en":
            model_name = "abhinand/MedEmbed-large-v0.1"  # Clinical embeddings
            fallback_model = "all-MiniLM-L6-v2"  # Lightweight fallback
        else:
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        # Preload the model
        model = _get_cached_sentence_transformer(model_name, fallback_model)
        model_key = model_name if model_name in _MODEL_CACHE else fallback_model
        logger.info(f"Successfully preloaded embedding model: {model_key}")

    except Exception as e:
        logger.warning(f"Failed to preload embedding models: {e}")
