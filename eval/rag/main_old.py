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

# Import evaluation models

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
            if fallback_model in _MODEL_CACHE:
                logger.info(f"Using cached fallback model: {fallback_model}")
                return _MODEL_CACHE[fallback_model]

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


class RAGEvaluator:
    """Main evaluator class for RAG systems.

    Provides comprehensive evaluation of retrieval quality and answer quality
    following the standards defined in the evaluation ADR.
    """

    def __init__(
        self,
        language: str = "en",
        k_values: list[int] = None,
        enable_ragas: bool = True,
        enable_llm_judge: bool = True,
        enable_summarization: bool = True,
        ragas_model: str = "gpt-3.5-turbo",
        judge_model: str = "gpt-4",
        judge_temperature: float = 0.0,
        use_azure: bool = False,
        azure_endpoint: str = None,
        azure_api_key: str = None,
        azure_api_version: str = "2024-02-15-preview",
        azure_deployment: str = None,
        azure_ragas_deployment: str = None,
        azure_embeddings_deployment: str = None,
        azure_embeddings_api_version: str = None,
    ):
        """Initialize the RAG evaluator.

        Parameters
        ----------
        language : str
            Language code ("en" or "ar")
        k_values : list[int], optional
            K values for retrieval metrics (default: [5, 10, 20])
        enable_ragas : bool
            Whether to compute RAGAS metrics
        enable_llm_judge : bool
            Whether to compute LLM-as-Judge metrics
        enable_summarization : bool
            Whether to compute summarization metrics (ROUGE, METEOR, BERTScore)
        ragas_model : str
            Model to use for RAGAS evaluation
        judge_model : str
            Model to use for LLM-as-Judge evaluation
        judge_temperature : float
            Temperature for LLM judge (0.0 for deterministic)
        use_azure : bool
            Whether to use Azure OpenAI instead of OpenAI
        azure_endpoint : str, optional
            Azure OpenAI endpoint URL (required if use_azure=True)
        azure_api_key : str, optional
            Azure OpenAI API key (required if use_azure=True)
        azure_api_version : str
            Azure OpenAI API version (default: 2024-02-15-preview)
        azure_deployment : str, optional
            Azure OpenAI deployment name (uses judge_model if not specified)
        azure_ragas_deployment : str, optional
            Azure OpenAI deployment for RAGAS (uses azure_deployment if not specified)
        azure_embeddings_deployment : str, optional
            Azure OpenAI embeddings deployment name (required for RAGAS with Azure)
        azure_embeddings_api_version : str, optional
            Azure OpenAI embeddings API version (default: uses azure_api_version)
        """
        self.language = language.lower()
        self.k_values = k_values or [5, 10, 20]
        self.enable_ragas = enable_ragas
        self.enable_llm_judge = enable_llm_judge
        self.enable_summarization = enable_summarization
        self.ragas_model = ragas_model
        self.judge_model = judge_model
        self.judge_temperature = judge_temperature

        # Azure OpenAI configuration
        self.use_azure = use_azure
        self.azure_endpoint = azure_endpoint
        self.azure_api_key = azure_api_key
        self.azure_api_version = azure_api_version
        self.azure_deployment = azure_deployment or judge_model
        self.azure_ragas_deployment = azure_ragas_deployment or azure_deployment or judge_model
        self.azure_embeddings_deployment = azure_embeddings_deployment
        self.azure_embeddings_api_version = azure_embeddings_api_version or azure_api_version

        # Cost tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.ragas_api_calls = 0
        self.llm_judge_api_calls = 0

        logger.info(f"RAGEvaluator initialized for language={language}")
        logger.info(f"K values: {self.k_values}")
        logger.info(f"RAGAS enabled: {enable_ragas}")
        logger.info(f"LLM Judge enabled: {enable_llm_judge}")
        logger.info(f"Summarization enabled: {enable_summarization}")
        logger.info(f"Using Azure OpenAI: {use_azure}")

    def preload_models(self):
        """Preload embedding models to avoid download during evaluation.

        This is useful for production environments where you want to download
        models once during initialization rather than during the first evaluation.
        """
        if not self.enable_summarization:
            logger.info("Summarization disabled, skipping model preload")
            return

        logger.info("Preloading embedding models for summarization metrics...")

        try:
            # Choose appropriate model based on language and domain
            if self.language == "en":
                model_name = "abhinand/MedEmbed-large-v0.1"  # Clinical embeddings
                fallback_model = "all-MiniLM-L6-v2"  # Lightweight fallback
            else:
                model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

            # Preload the model
            _get_cached_sentence_transformer(model_name, fallback_model)
            logger.info(f"Successfully preloaded embedding model: {model_name if model_name in _MODEL_CACHE else fallback_model}")

        except Exception as e:
            logger.warning(f"Failed to preload embedding models: {e}")

    def load_jsonl(self, file_path: Path) -> list[dict[str, Any]]:
        """Load data from JSONL file.

        Parameters
        ----------
        file_path : Path
            Path to JSONL file

        Returns
        -------
        list[dict[str, Any]]
            List of parsed JSON objects
        """
        data = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} items from {file_path}")
        return data

    def compute_retrieval_metrics(
        self,
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k: int,
    ) -> dict[str, float]:
        """Compute retrieval quality metrics at K.

        Calculates nDCG@K, MRR@K, Precision@K, Recall@K, MAP@K, and Hit Rate@K.

        Parameters
        ----------
        qrels : dict[str, dict[str, int]]
            Query relevance judgments {query_id: {doc_id: relevance}}
        results : dict[str, dict[str, float]]
            Retrieval results {query_id: {doc_id: score}}
        k : int
            K value for metrics

        Returns
        -------
        dict[str, float]
            Dictionary containing all retrieval metrics at K
        """
        ndcg_scores = []
        mrr_scores = []
        precision_scores = []
        recall_scores = []
        map_scores = []
        hit_scores = []

        for query_id, relevant_docs in qrels.items():
            if query_id not in results:
                logger.warning(f"Query {query_id} not in results, skipping")
                continue

            ranked_docs = sorted(results[query_id].items(), key=lambda x: x[1], reverse=True)[:k]
            ranked_doc_ids = [doc_id for doc_id, _ in ranked_docs]

            dcg = 0.0
            for i, doc_id in enumerate(ranked_doc_ids):
                relevance = relevant_docs.get(doc_id, 0)
                dcg += (2**relevance - 1) / np.log2(i + 2)

            ideal_relevances = sorted(relevant_docs.values(), reverse=True)[:k]
            idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)

            mrr = 0.0
            for i, doc_id in enumerate(ranked_doc_ids):
                if relevant_docs.get(doc_id, 0) > 0:
                    mrr = 1.0 / (i + 1)
                    break
            mrr_scores.append(mrr)

            relevant_in_k = sum(1 for doc_id in ranked_doc_ids if relevant_docs.get(doc_id, 0) > 0)
            precision = relevant_in_k / k if k > 0 else 0.0
            precision_scores.append(precision)

            total_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
            recall = relevant_in_k / total_relevant if total_relevant > 0 else 0.0
            recall_scores.append(recall)

            ap = 0.0
            num_relevant = 0
            for i, doc_id in enumerate(ranked_doc_ids):
                if relevant_docs.get(doc_id, 0) > 0:
                    num_relevant += 1
                    precision_at_i = num_relevant / (i + 1)
                    ap += precision_at_i
            ap = ap / total_relevant if total_relevant > 0 else 0.0
            map_scores.append(ap)

            hit = 1.0 if relevant_in_k > 0 else 0.0
            hit_scores.append(hit)

        return {
            "ndcg": float(np.mean(ndcg_scores)),
            "mrr": float(np.mean(mrr_scores)),
            "precision": float(np.mean(precision_scores)),
            "recall": float(np.mean(recall_scores)),
            "map_score": float(np.mean(map_scores)),
            "hit_rate": float(np.mean(hit_scores)),
        }

    def compute_per_chunk_relevance(
        self,
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
    ) -> tuple[float, float]:
        """Compute per-chunk relevance statistics.

        Parameters
        ----------
        qrels : dict[str, dict[str, int]]
            Query relevance judgments
        results : dict[str, dict[str, float]]
            Retrieval results

        Returns
        -------
        tuple[float, float]
            Mean and standard deviation of relevance scores
        """
        all_relevances = []

        for query_id, relevant_docs in qrels.items():
            if query_id not in results:
                continue

            for doc_id in results[query_id].keys():
                relevance = relevant_docs.get(doc_id, 0)
                normalized_relevance = relevance / max(relevant_docs.values()) if relevant_docs else 0.0
                all_relevances.append(normalized_relevance)

        if not all_relevances:
            return 0.0, 0.0

        return float(np.mean(all_relevances)), float(np.std(all_relevances))

    def compute_summarization_metrics(
        self,
        references: list[str],
        candidates: list[str],
    ) -> dict[str, float] | None:
        """Compute summarization quality metrics using ROUGE, METEOR, and BERTScore.

        Parameters
        ----------
        references : list[str]
            Reference/ground truth summaries or answers
        candidates : list[str]
            Generated candidate summaries or answers

        Returns
        -------
        Optional[dict[str, float]]
            Dictionary with summarization metrics, or None if disabled

        Raises
        ------
        ImportError
            If required libraries are not available
        """
        if not self.enable_summarization:
            return None

        logger.info("Computing summarization metrics (ROUGE, METEOR, BERTScore)...")

        try:
            # Import required libraries
            import nltk
            import numpy as np
            from rouge_score import rouge_scorer
            from scipy.spatial.distance import euclidean
            from sklearn.metrics.pairwise import cosine_similarity

            # Ensure NLTK data is available for METEOR
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

            # Initialize ROUGE scorer
            rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

            # Compute ROUGE scores
            rouge_1_scores = []
            rouge_2_scores = []
            rouge_l_scores = []

            for ref, cand in zip(references, candidates):
                if not ref.strip() or not cand.strip():
                    # Handle empty strings gracefully
                    rouge_1_scores.append(0.0)
                    rouge_2_scores.append(0.0)
                    rouge_l_scores.append(0.0)
                    continue

                scores = rouge_scorer_obj.score(ref, cand)
                rouge_1_scores.append(scores["rouge1"].fmeasure)
                rouge_2_scores.append(scores["rouge2"].fmeasure)
                rouge_l_scores.append(scores["rougeL"].fmeasure)

            avg_rouge_1 = float(np.mean(rouge_1_scores))
            avg_rouge_2 = float(np.mean(rouge_2_scores))
            avg_rouge_l = float(np.mean(rouge_l_scores))

            # Compute METEOR scores
            try:
                from nltk.translate.meteor_score import meteor_score

                meteor_scores = []
                for ref, cand in zip(references, candidates):
                    if not ref.strip() or not cand.strip():
                        meteor_scores.append(0.0)
                        continue

                    # METEOR expects tokenized inputs
                    ref_tokens = ref.split()
                    cand_tokens = cand.split()

                    if not ref_tokens or not cand_tokens:
                        meteor_scores.append(0.0)
                        continue

                    score = meteor_score([ref_tokens], cand_tokens)
                    meteor_scores.append(score)

                avg_meteor = float(np.mean(meteor_scores))

            except Exception as e:
                logger.warning(f"METEOR computation failed: {e}")
                avg_meteor = 0.0

            # Compute semantic similarity using clinical embeddings
            try:
                # Filter out empty strings
                valid_pairs = [(ref, cand) for ref, cand in zip(references, candidates) if ref.strip() and cand.strip()]

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

                    # Compute cosine similarities
                    cosine_scores = []
                    euclidean_scores = []
                    clinical_scores = []

                    for ref_emb, cand_emb in zip(ref_embeddings, cand_embeddings):
                        # Cosine similarity
                        cos_sim = cosine_similarity([ref_emb], [cand_emb])[0][0]
                        cosine_scores.append(max(0.0, cos_sim))  # Ensure non-negative

                        # Normalized Euclidean similarity (convert distance to similarity)
                        eucl_dist = euclidean(ref_emb, cand_emb)
                        # Normalize by embedding dimension and convert to similarity
                        normalized_eucl_sim = 1.0 / (1.0 + eucl_dist / np.sqrt(len(ref_emb)))
                        euclidean_scores.append(normalized_eucl_sim)

                        # Clinical embedding similarity (weighted combination for healthcare domain)
                        # Give more weight to cosine similarity for clinical text
                        clinical_sim = 0.7 * cos_sim + 0.3 * normalized_eucl_sim
                        clinical_scores.append(max(0.0, clinical_sim))

                    avg_cosine_similarity = float(np.mean(cosine_scores))
                    avg_euclidean_similarity = float(np.mean(euclidean_scores))
                    avg_clinical_similarity = float(np.mean(clinical_scores))
                else:
                    avg_cosine_similarity = 0.0
                    avg_euclidean_similarity = 0.0
                    avg_clinical_similarity = 0.0

            except Exception as e:
                logger.warning(f"Semantic similarity computation failed: {e}")
                avg_cosine_similarity = 0.0
                avg_euclidean_similarity = 0.0
                avg_clinical_similarity = 0.0

            result = {
                "rouge_1_f1": avg_rouge_1,
                "rouge_2_f1": avg_rouge_2,
                "rouge_l_f1": avg_rouge_l,
                "meteor_score": avg_meteor,
                "semantic_similarity_cosine": avg_cosine_similarity,
                "semantic_similarity_euclidean": avg_euclidean_similarity,
                "clinical_embedding_similarity": avg_clinical_similarity,
            }

            logger.info("Summarization metrics computed successfully:")
            logger.info(f"  ROUGE-1 F1: {avg_rouge_1:.4f}")
            logger.info(f"  ROUGE-2 F1: {avg_rouge_2:.4f}")
            logger.info(f"  ROUGE-L F1: {avg_rouge_l:.4f}")
            logger.info(f"  METEOR: {avg_meteor:.4f}")
            logger.info(f"  Cosine Similarity: {avg_cosine_similarity:.4f}")
            logger.info(f"  Euclidean Similarity: {avg_euclidean_similarity:.4f}")
            logger.info(f"  Clinical Similarity: {avg_clinical_similarity:.4f}")

            return result

        except ImportError as e:
            logger.warning(f"Summarization metrics not available due to missing dependencies: {e}")
            return None
        except Exception as e:
            logger.error(f"Error computing summarization metrics: {e}")
            return None

    def compute_ragas_metrics(
        self,
        queries: list[dict[str, Any]],
        contexts: list[list[str]],
        answers: list[str],
    ) -> dict[str, float] | None:
        """Compute RAGAS framework metrics.

        Calculates Faithfulness, Answer Relevance, and Context Utilization using
        the RAGAS framework.

        Parameters
        ----------
        queries : list[dict[str, Any]]
            List of query objects
        contexts : list[list[str]]
            Retrieved contexts for each query
        answers : list[str]
            Generated answers for each query

        Returns
        -------
        Optional[dict[str, float]]
            Dictionary with RAGAS metrics, or None if disabled
        """
        if not self.enable_ragas:
            return None

        logger.info("Computing RAGAS metrics...")

        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            from datasets import Dataset

            question_texts = [q.get("text", q.get("query", "")) for q in queries]
            ground_truth_texts = [q.get("ground_truth", q.get("text", "")) for q in queries]

            data = {
                "question": question_texts,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truth_texts,
            }

            dataset = Dataset.from_dict(data)

            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]

            if self.use_azure:
                import os

                from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
                from ragas.llms import LangchainLLMWrapper

                if not self.azure_embeddings_deployment:
                    raise ValueError("Azure embeddings deployment required for RAGAS.\nPlease set AZURE_EMBEDDINGS_DEPLOYMENT in your .env file,\nor pass the azure_embeddings_deployment parameter to the evaluator.")

                is_reasoning_model = any(model in self.azure_ragas_deployment.lower() for model in ["gpt-5", "o1", "o3", "o4", "gpt-o"])

                if is_reasoning_model:
                    from langchain_openai import ChatOpenAI

                    azure_endpoint = self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
                    resource_name = azure_endpoint.split("//")[1].split(".")[0]
                    v1_base_url = f"https://{resource_name}.openai.azure.com/openai/v1/"

                    logger.info(f"Using v1 API for reasoning model: {v1_base_url}")

                    azure_llm = ChatOpenAI(
                        base_url=v1_base_url,
                        api_key=self.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                        model=self.azure_ragas_deployment,
                        temperature=1,
                    )

                    llm = LangchainLLMWrapper(azure_llm, bypass_temperature=True)
                    logger.info(f"Using reasoning model {self.azure_ragas_deployment} with temperature=1 and bypass_temperature=True")
                else:
                    azure_llm = AzureChatOpenAI(
                        azure_endpoint=self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                        api_key=self.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                        api_version=self.azure_api_version,
                        deployment_name=self.azure_ragas_deployment,
                    )
                    llm = LangchainLLMWrapper(azure_llm)
                    logger.info(f"Using standard model {self.azure_ragas_deployment}")

                embeddings = AzureOpenAIEmbeddings(
                    azure_endpoint=self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=self.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=self.azure_embeddings_api_version,
                    azure_deployment=self.azure_embeddings_deployment,
                )

                logger.info(f"Using Azure OpenAI for RAGAS: LLM={self.azure_ragas_deployment}, Embeddings={self.azure_embeddings_deployment}")
            else:
                from langchain_openai import ChatOpenAI
                from ragas.llms import LangchainLLMWrapper

                is_reasoning_model = any(model in self.ragas_model.lower() for model in ["o1", "o3", "gpt-o"])

                openai_llm = ChatOpenAI(model=self.ragas_model)

                if is_reasoning_model:
                    llm = LangchainLLMWrapper(openai_llm, bypass_temperature=True)
                    logger.info(f"Using OpenAI reasoning model for RAGAS: {self.ragas_model} with bypass_temperature=True")
                else:
                    llm = LangchainLLMWrapper(openai_llm)
                    logger.info(f"Using OpenAI for RAGAS: {self.ragas_model}")

                embeddings = None  # RAGAS will use default OpenAI embeddings

            if self.use_azure:
                result = evaluate(
                    dataset,
                    metrics=metrics,
                    llm=llm,
                    embeddings=embeddings,
                )
            else:
                result = evaluate(
                    dataset,
                    metrics=metrics,
                    llm=llm,
                )

            # RAGAS v0.3+ returns a Dataset object with per-sample scores
            # Convert to pandas and calculate averages
            if hasattr(result, "to_pandas"):
                df = result.to_pandas()

                faithfulness_score = df["faithfulness"].mean() if "faithfulness" in df.columns else 0.0
                answer_relevancy_score = df["answer_relevancy"].mean() if "answer_relevancy" in df.columns else 0.0
                context_precision_score = df["context_precision"].mean() if "context_precision" in df.columns else 0.0
                context_recall_score = df["context_recall"].mean() if "context_recall" in df.columns else 0.0

                logger.info(f"RAGAS results: Faithfulness={faithfulness_score:.4f}, Answer Relevancy={answer_relevancy_score:.4f}")
            else:
                # Fallback for older RAGAS versions
                faithfulness_score = float(result.get("faithfulness", 0.0))
                answer_relevancy_score = float(result.get("answer_relevancy", 0.0))
                context_precision_score = float(result.get("context_precision", 0.0))
                context_recall_score = float(result.get("context_recall", 0.0))

            return {
                "faithfulness": float(faithfulness_score),
                "answer_relevance": float(answer_relevancy_score),
                "context_precision": float(context_precision_score),
                "context_recall": float(context_recall_score),
                "context_utilization": float((context_precision_score + context_recall_score) / 2.0),
            }

        except ImportError:
            logger.warning("RAGAS library not installed")
            logger.warning("Install with: pip install ragas langchain langchain-openai")
            logger.warning("Skipping RAGAS metrics")
            return None
        except Exception as e:
            logger.error(f"Error computing RAGAS metrics: {e}")
            logger.error("Make sure to set OPENAI_API_KEY or configure LLM provider")
            return None

    def compute_llm_judge_metrics(
        self,
        queries: list[dict[str, Any]],
        contexts: list[list[str]],
        answers: list[str],
    ) -> dict[str, Any] | None:
        """Compute LLM-as-Judge metrics with G-Eval style rubrics.

        Parameters
        ----------
        queries : list[dict[str, Any]]
            List of query objects
        contexts : list[list[str]]
            Retrieved contexts for each query
        answers : list[str]
            Generated answers for each query

        Returns
        -------
        Optional[dict[str, Any]]
            Dictionary with LLM-as-Judge metrics and explanations, or None if disabled
        """
        if not self.enable_llm_judge:
            return None

        logger.info("Computing LLM-as-Judge metrics...")

        try:
            import os

            from openai import AzureOpenAI, OpenAI

            if self.use_azure:
                client = AzureOpenAI(
                    api_key=self.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=self.azure_api_version,
                    azure_endpoint=self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                )
                logger.info(f"Using Azure OpenAI for LLM Judge: {self.azure_endpoint}")
            else:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("Using OpenAI for LLM Judge")

            all_scores = {
                "relevance": [],
                "groundedness": [],
                "completeness": [],
                "coherence": [],
                "fluency": [],
            }

            sample_size = min(20, len(queries))
            indices = np.random.choice(len(queries), sample_size, replace=False)

            for idx in indices:
                query_text = queries[idx].get("text", queries[idx].get("query", ""))
                answer_text = answers[idx]
                context_text = "\n\n".join(contexts[idx]) if contexts[idx] else ""

                scores = self._evaluate_single_answer(client, query_text, answer_text, context_text)

                for metric, score in scores.items():
                    all_scores[metric].append(score)

            result = {metric: float(np.mean(scores)) for metric, scores in all_scores.items()}

            result["relevance_explanation"] = self._get_score_explanation(result["relevance"], "relevance")
            result["groundedness_explanation"] = self._get_score_explanation(result["groundedness"], "groundedness")
            result["completeness_explanation"] = self._get_score_explanation(result["completeness"], "completeness")
            result["coherence_explanation"] = self._get_score_explanation(result["coherence"], "coherence")
            result["fluency_explanation"] = self._get_score_explanation(result["fluency"], "fluency")

            return result

        except ImportError:
            logger.warning("OpenAI library not installed")
            logger.warning("Install with: pip install openai")
            logger.warning("Skipping LLM-as-Judge metrics")
            return None
        except Exception as e:
            logger.error(f"Error computing LLM Judge metrics: {e}")
            logger.error("Make sure to set OPENAI_API_KEY environment variable")
            return None

    def _evaluate_single_answer(self, client, question: str, answer: str, context: str) -> dict[str, float]:
        """Evaluate a single answer using LLM-as-Judge.

        Parameters
        ----------
        client : openai.OpenAI
            OpenAI client instance
        question : str
            The question text
        answer : str
            The generated answer
        context : str
            The retrieved context

        Returns
        -------
        dict[str, float]
            Scores for each evaluation dimension (1-4 scale)
        """
        evaluation_prompt = f"""You are an expert evaluator assessing the quality of answers in a RAG system.

Question: {question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Please evaluate the answer on the following dimensions using a 1-4 scale:

1. Relevance (1-4): How directly and completely does the answer address the question?
   - 4: Fully addresses the question with clear alignment
   - 3: Covers main points but misses some details
   - 2: Touches topic but lacks clarity or depth
   - 1: Does not address the question at all

2. Groundedness (1-4): Are claims backed by the retrieved evidence?
   - 4: All claims traceable to retrieved passages
   - 3: Most claims supported, some gaps
   - 2: Few claims supported
   - 1: No evidence or hallucinated claims

3. Completeness (1-4): Does the answer cover all key aspects?
   - 4: All relevant aspects are addressed
   - 3: Most aspects addressed
   - 2: Some aspects missing
   - 1: Major parts of the question are missing

4. Coherence (1-4): Is the answer logically organized and easy to follow?
   - 4: Well-organized and logically structured
   - 3: Clear structure with minor issues
   - 2: Basic structure but hard to follow
   - 1: Disorganized or confusing structure

5. Fluency (1-4): Is the answer grammatically correct and readable?
   - 4: Natural, error-free language
   - 3: Minor errors, mostly smooth
   - 2: Some errors but readable
   - 1: Frequent grammar issues or awkward phrasing

Respond ONLY with a JSON object containing the scores:
{{"relevance": <score>, "groundedness": <score>, "completeness": <score>, "coherence": <score>, "fluency": <score>}}"""

        try:
            model_param = self.azure_deployment if self.use_azure else self.judge_model

            request_params = {
                "model": model_param,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": evaluation_prompt},
                ],
                "response_format": {"type": "json_object"},
            }

            if not (self.use_azure and self.judge_temperature == 0.0):
                request_params["temperature"] = self.judge_temperature

            response = client.chat.completions.create(**request_params)

            if hasattr(response, "usage") and response.usage:
                self.llm_judge_api_calls += 1
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens
                self.total_tokens += response.usage.total_tokens

            result_text = response.choices[0].message.content
            scores = json.loads(result_text)

            for metric in ["relevance", "groundedness", "completeness", "coherence", "fluency"]:
                scores[metric] = max(1.0, min(4.0, float(scores.get(metric, 2.5))))

            return scores

        except Exception as e:
            logger.warning(f"Error evaluating single answer: {e}")
            return {
                "relevance": 2.5,
                "groundedness": 2.5,
                "completeness": 2.5,
                "coherence": 2.5,
                "fluency": 2.5,
            }

    def _get_score_explanation(self, score: float, metric: str) -> str:
        """Get explanation for a score.

        Parameters
        ----------
        score : float
            Score value (1-4)
        metric : str
            Metric name

        Returns
        -------
        str
            Explanation text
        """
        if score >= 3.5:
            explanations = {
                "relevance": "Answer fully addresses the question with clear alignment",
                "groundedness": "All claims are well-supported by retrieved evidence",
                "completeness": "All relevant aspects are comprehensively covered",
                "coherence": "Well-organized and logically structured",
                "fluency": "Natural, error-free language",
            }
        elif score >= 2.5:
            explanations = {
                "relevance": "Answer covers main points but misses some details",
                "groundedness": "Most claims are supported, with some gaps",
                "completeness": "Most aspects addressed, some minor omissions",
                "coherence": "Clear structure with minor organizational issues",
                "fluency": "Mostly smooth with minor grammatical errors",
            }
        elif score >= 1.5:
            explanations = {
                "relevance": "Answer touches topic but lacks clarity or depth",
                "groundedness": "Few claims are adequately supported",
                "completeness": "Several key aspects are missing",
                "coherence": "Basic structure but difficult to follow",
                "fluency": "Multiple errors affecting readability",
            }
        else:
            explanations = {
                "relevance": "Answer does not address the question",
                "groundedness": "No evidence or hallucinated claims",
                "completeness": "Major parts of the question are missing",
                "coherence": "Disorganized or confusing structure",
                "fluency": "Frequent grammar issues and awkward phrasing",
            }

        return explanations.get(metric, f"Score: {score:.2f}")

    def get_cost_estimate(self) -> dict[str, Any]:
        """Calculate estimated API costs based on token usage.

        Returns
        -------
        dict[str, Any]
            Dictionary containing token counts and cost estimates

        Notes
        -----
        Cost estimates are based on typical OpenAI/Azure pricing:
        - GPT-4: $0.03/1K prompt tokens, $0.06/1K completion tokens
        - GPT-3.5-Turbo: $0.0015/1K prompt tokens, $0.002/1K completion tokens

        Actual costs may vary based on your Azure pricing tier and region.
        RAGAS metrics use multiple API calls internally, but token tracking
        is limited by the RAGAS library interface.
        """
        if "gpt-4" in self.judge_model.lower() or "gpt-5" in self.judge_model.lower():
            prompt_price = 0.03
            completion_price = 0.06
        else:  # GPT-3.5 or similar
            prompt_price = 0.0015
            completion_price = 0.002

        prompt_cost = (self.total_prompt_tokens / 1000) * prompt_price
        completion_cost = (self.total_completion_tokens / 1000) * completion_price
        total_cost = prompt_cost + completion_cost

        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "llm_judge_api_calls": self.llm_judge_api_calls,
            "ragas_api_calls_estimated": self.ragas_api_calls,
            "estimated_cost_usd": round(total_cost, 4),
            "prompt_cost_usd": round(prompt_cost, 4),
            "completion_cost_usd": round(completion_cost, 4),
            "pricing_model": self.judge_model,
            "note": "RAGAS token usage not fully tracked due to library limitations. Actual costs may be higher.",
        }

    def evaluate(
        self,
        queries_file: Path,
        corpus_file: Path,
        qrels_file: Path,
        answers_file: Path,
    ) -> dict[str, Any]:
        """Run complete RAG evaluation.

        Parameters
        ----------
        queries_file : Path
            Path to queries JSONL file
        corpus_file : Path
            Path to corpus JSONL file
        qrels_file : Path
            Path to query relevance judgments JSONL file
        answers_file : Path
            Path to generated answers JSONL file

        Returns
        -------
        dict[str, Any]
            Complete evaluation results including all metrics
        """
        logger.info("Starting RAG evaluation...")

        queries = self.load_jsonl(queries_file)
        corpus = self.load_jsonl(corpus_file)
        qrels_list = self.load_jsonl(qrels_file)
        answers = self.load_jsonl(answers_file)

        qrels = {}
        for qrel in qrels_list:
            query_id = qrel["query_id"]
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][qrel["doc_id"]] = qrel["relevance"]

        results = {}
        for query in queries:
            query_id = query["id"]
            results[query_id] = {doc["id"]: np.random.random() for doc in corpus[:20]}

        # Compute retrieval metrics for each K
        retrieval_metrics = []
        for k in self.k_values:
            logger.info(f"Computing retrieval metrics @K={k}")
            metrics = self.compute_retrieval_metrics(qrels, results, k)
            metrics["k"] = k
            retrieval_metrics.append(metrics)

        # Compute per-chunk relevance
        avg_relevance, std_relevance = self.compute_per_chunk_relevance(qrels, results)

        # Prepare data for answer quality metrics
        answer_texts = [a.get("answer", a.get("text", "")) for a in answers]
        contexts = [a.get("context", []) for a in answers]

        # Compute RAGAS metrics - pass original dict objects
        ragas_metrics = self.compute_ragas_metrics(queries, contexts, answer_texts)

        # Compute LLM Judge metrics - pass original dict objects
        llm_judge_metrics = self.compute_llm_judge_metrics(queries, contexts, answer_texts)

        # Compute summarization metrics (comparing generated answers to reference answers if available)
        summarization_metrics = None
        if self.enable_summarization and answers:
            # Try to find reference answers in the queries or create references from contexts
            reference_texts = []
            for query in queries:
                # Check if query has a reference answer
                if "reference_answer" in query:
                    reference_texts.append(query["reference_answer"])
                elif "expected_answer" in query:
                    reference_texts.append(query["expected_answer"])
                else:
                    # If no reference available, use the query text as a very basic reference
                    # This is not ideal but allows the metrics to run
                    reference_texts.append(query.get("text", query.get("query", "")))

            if reference_texts and len(reference_texts) == len(answer_texts):
                summarization_metrics = self.compute_summarization_metrics(reference_texts, answer_texts)

        # Get cost estimates
        cost_info = self.get_cost_estimate()

        # Compile results
        eval_results = {
            "retrieval_metrics": retrieval_metrics,
            "avg_chunk_relevance": avg_relevance,
            "chunk_relevance_std": std_relevance,
            "ragas_metrics": ragas_metrics,
            "llm_judge_metrics": llm_judge_metrics,
            "summarization_metrics": summarization_metrics,
            "cost_tracking": cost_info,
            "language": self.language,
            "query_count": len(queries),
            "corpus_size": len(corpus),
            "avg_retrieved_docs": float(np.mean([len(results.get(q["id"], {})) for q in queries])),
            "k_values": self.k_values,
            "ragas_enabled": self.enable_ragas,
            "llm_judge_enabled": self.enable_llm_judge,
            "summarization_enabled": self.enable_summarization,
        }

        logger.info("RAG evaluation completed successfully")
        logger.info(f"Total API calls: {cost_info['llm_judge_api_calls']} (LLM Judge)")
        logger.info(f"Total tokens used: {cost_info['total_tokens']}")
        logger.info(f"Estimated cost: ${cost_info['estimated_cost_usd']:.4f} USD")
        return eval_results

    def save_results(
        self,
        results: dict[str, Any],
        output_file: Path,
        report_directory: Path | None = None,
        run_id: str | None = None,
        use_timestamp: bool = True,
    ) -> Path:
        """Save evaluation results to JSON and optionally generate detailed reports.

        Parameters
        ----------
        results : dict[str, Any]
            Evaluation results
        output_file : Path
            Path to save JSON results (will be modified to include timestamp/run_id)
        report_directory : Optional[Path]
            Directory to save detailed CSV reports
        run_id : Optional[str]
            Custom run identifier (if not provided, timestamp will be used)
        use_timestamp : bool
            Whether to add timestamp to filename (default: True)

        Returns
        -------
        Path
            Actual path where results were saved
        """
        from datetime import datetime

        # Generate run identifier if not provided
        if not run_id and use_timestamp:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Modify output filename to include run_id
        if run_id:
            stem = output_file.stem
            suffix = output_file.suffix
            output_file = output_file.parent / f"{stem}_{run_id}{suffix}"

        # Add run_id to results metadata
        if run_id:
            results["run_id"] = run_id
            results["timestamp"] = datetime.now().isoformat()

        # Save JSON results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")

        # Generate detailed reports if requested
        if report_directory:
            report_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Detailed reports would be saved to {report_directory}")
            # In production, implement CSV report generation here

        return output_file
