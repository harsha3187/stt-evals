"""RAG (Retrieval-Augmented Generation) evaluation module.

This module implements simplified RAG evaluation focusing on summarization quality metrics:
- ROUGE-1, ROUGE-2, ROUGE-L F1 scores
- METEOR score
- Semantic similarity using clinical embeddings (MedEmbed for healthcare)
- Cosine similarity and normalized Euclidean distance

Follows the same evaluation patterns as STT module for consistency.
Supports bilingual evaluation (English and Arabic) with proper text normalization.
"""

from .main import evaluate_rag_model

__all__ = ["evaluate_rag_model"]
