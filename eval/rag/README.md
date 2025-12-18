# RAG Evaluation Framework

Comprehensive evaluation framework for RAG (Retrieval-Augmented Generation) systems, implementing the metrics and protocols defined in [ADR-003](../../docs/adrs/003-evals.md).

## Overview

This module provides evaluation for:

1. **Retrieval Quality Metrics** - Using BEIR + pytrec_eval
   - nDCG@K (Normalized Discounted Cumulative Gain)
   - MRR@K (Mean Reciprocal Rank)
   - Precision@K
   - Recall@K
   - MAP@K (Mean Average Precision)
   - Hit Rate@K (Binary success metric)
   - Per-chunk relevance scoring

2. **RAGAS Framework Metrics** - Answer quality evaluation
   - Faithfulness (answer claims backed by context)
   - Answer Relevance (semantic alignment with question)
   - Context Precision (relevance of retrieved chunks)
   - Context Recall (coverage of relevant information)
   - Context Utilization (how well chunks are used)

3. **LLM-as-Judge Metrics** - G-Eval style rubric scoring (1-4 scale)
   - Relevance (addresses the question)
   - Groundedness (claims backed by evidence)
   - Completeness (covers all key aspects)
   - Coherence (logical organization)
   - Fluency (grammar and readability)

## Features

- **Bilingual Support**: English and Arabic evaluation
- **Configurable K Values**: Evaluate at K=5, 10, 20, or custom values
- **Type-Safe Models**: Pydantic models for all results
- **Flexible Configuration**: Environment variables, .env files, or code
- **JSON/CSV Reporting**: Dashboard-ready output formats
- **Per-Query & Aggregate Stats**: Detailed analysis at all levels

## Installation

### Dependencies

All dependencies are managed via Poetry and declared in the root `pyproject.toml`:

```bash
# Install all dependencies (including RAG evaluation dependencies)
poetry install
```

The following dependencies are included:
- **Core**: `pydantic`, `pydantic-settings`, `numpy`, `pandas`
- **RAGAS metrics**: `ragas`, `langchain`, `langchain-openai`, `datasets`
- **LLM-as-Judge**: `openai`
- **Data sources**: `kaggle`, `python-dotenv`

### Environment Variables

Set your API key for RAGAS and LLM-as-Judge:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Data Format

### Input Files (JSONL format)

#### Queries File (`queries.jsonl`)
```jsonl
{"id": "q1", "text": "What are the symptoms of diabetes?"}
{"id": "q2", "text": "How is hypertension treated?"}
```

#### Corpus File (`corpus.jsonl`)
```jsonl
{"id": "doc1", "text": "Diabetes symptoms include increased thirst..."}
{"id": "doc2", "text": "Hypertension treatment involves medication..."}
```

#### Query Relevance Judgments (`qrels.jsonl`)
```jsonl
{"query_id": "q1", "doc_id": "doc1", "relevance": 2}
{"query_id": "q1", "doc_id": "doc5", "relevance": 1}
```

Relevance scale: 0 (not relevant), 1 (somewhat relevant), 2 (highly relevant)

#### Answers File (`answers.jsonl`)
```jsonl
{"query_id": "q1", "answer": "Diabetes symptoms include...", "context": ["doc1", "doc5"]}
{"query_id": "q2", "answer": "Hypertension is treated by...", "context": ["doc2"]}
```

## Usage

### Command Line

#### Basic RAG Evaluation
```bash
python main.py --task rag \
    --queries datasets/rag/queries.jsonl \
    --corpus datasets/rag/corpus.jsonl \
    --qrels datasets/rag/qrels.jsonl \
    --answers datasets/rag/answers.jsonl \
    --language en \
    --output results/rag_results.json
```

#### Custom K Values
```bash
python main.py --task rag \
    --queries datasets/rag/queries.jsonl \
    --corpus datasets/rag/corpus.jsonl \
    --qrels datasets/rag/qrels.jsonl \
    --answers datasets/rag/answers.jsonl \
    --k-values 3 5 10 20 \
    --language ar
```

#### Disable RAGAS or LLM Judge
```bash
python main.py --task rag \
    --queries datasets/rag/queries.jsonl \
    --corpus datasets/rag/corpus.jsonl \
    --qrels datasets/rag/qrels.jsonl \
    --answers datasets/rag/answers.jsonl \
    --no-ragas \
    --no-llm-judge
```

### Python API

```python
from pathlib import Path
from eval.rag import RAGEvaluator
from eval.models import RAGEvaluationResult

# Initialize evaluator
evaluator = RAGEvaluator(
    language="en",
    k_values=[5, 10, 20],
    enable_ragas=True,
    enable_llm_judge=True,
    judge_model="gpt-4"
)

# Run evaluation
results_dict = evaluator.evaluate(
    queries_file=Path("datasets/rag/queries.jsonl"),
    corpus_file=Path("datasets/rag/corpus.jsonl"),
    qrels_file=Path("datasets/rag/qrels.jsonl"),
    answers_file=Path("datasets/rag/answers.jsonl")
)

# Convert to typed model
from eval.models import RAGEvaluationResult, RetrievalMetrics

retrieval_metrics = [RetrievalMetrics(**m) for m in results_dict["retrieval_metrics"]]
results = RAGEvaluationResult(
    retrieval_metrics=retrieval_metrics,
    avg_chunk_relevance=results_dict["avg_chunk_relevance"],
    chunk_relevance_std=results_dict["chunk_relevance_std"],
    # ... other fields
)

# Get summary
summary = results.get_summary()
print(f"Best K value: {results.get_best_k()}")

# Save results
evaluator.save_results(
    results_dict,
    Path("results/rag_results.json"),
    report_directory=Path("results/rag_reports")
)
```

### Configuration File

Create `eval_config.yaml`:

```yaml
rag:
  queries_file: datasets/rag/queries.jsonl
  corpus_file: datasets/rag/corpus.jsonl
  qrels_file: datasets/rag/qrels.jsonl
  answers_file: datasets/rag/answers.jsonl
  language: en
  k_values: [5, 10, 20]
  enable_ragas: true
  enable_llm_judge: true
  output_file: results/rag_results.json
```

Run with:
```bash
python main.py --task all --config eval_config.yaml
```

### Environment Variables

Configure via `.env` file or environment variables with `EVAL_` prefix:

```bash
# .env
RAG__LANGUAGE=en
RAG__K_VALUES=[5, 10, 20]
RAG__ENABLE_RAGAS=true
RAG__ENABLE_LLM_JUDGE=true
RAG__JUDGE_MODEL=gpt-4
RAG__QUERIES_FILE=datasets/rag/queries.jsonl
RAG__CORPUS_FILE=datasets/rag/corpus.jsonl
RAG__QRELS_FILE=datasets/rag/qrels.jsonl
RAG__ANSWERS_FILE=datasets/rag/answers.jsonl
```

## Output Format

### JSON Results Structure

```json
{
  "task_type": "rag",
  "timestamp": "2025-10-23T12:00:00",
  "session_id": "uuid-here",
  "results": {
    "retrieval_metrics": [
      {
        "k": 5,
        "ndcg": 0.85,
        "mrr": 0.90,
        "precision": 0.80,
        "recall": 0.75,
        "map_score": 0.82,
        "hit_rate": 0.95
      }
    ],
    "avg_chunk_relevance": 0.78,
    "chunk_relevance_std": 0.15,
    "ragas_metrics": {
      "faithfulness": 0.88,
      "answer_relevance": 0.85,
      "context_precision": 0.80,
      "context_recall": 0.82,
      "context_utilization": 0.79
    },
    "llm_judge_metrics": {
      "relevance": 3.8,
      "groundedness": 3.9,
      "completeness": 3.5,
      "coherence": 4.0,
      "fluency": 4.0,
      "relevance_explanation": "Answer directly addresses the question..."
    },
    "language": "en",
    "query_count": 100,
    "corpus_size": 1000,
    "avg_retrieved_docs": 10.5
  }
}
```

## Interpretation Guide

### Retrieval Metrics

| Metric | Range | Good | Fair | Poor | Interpretation |
|--------|-------|------|------|------|----------------|
| nDCG@K | 0-1 | >0.90 | 0.75-0.90 | <0.50 | Rank quality with graded relevance |
| MRR@K | 0-1 | >0.90 | 0.75-0.90 | <0.50 | Speed to first relevant result |
| Precision@K | 0-1 | >0.80 | 0.60-0.80 | <0.30 | Accuracy of top-K results |
| Recall@K | 0-1 | >0.80 | 0.60-0.80 | <0.30 | Coverage of relevant items |
| MAP@K | 0-1 | >0.80 | 0.60-0.80 | <0.30 | Overall precision across queries |
| Hit Rate@K | 0-1 | 1.0 | ≥0.75 | <0.50 | At least one relevant in top-K |

### RAGAS Metrics

| Metric | Range | Good | Fair | Poor | Interpretation |
|--------|-------|------|------|------|----------------|
| Faithfulness | 0-1 | >0.85 | 0.70-0.85 | <0.60 | Claims supported by context |
| Answer Relevance | 0-1 | >0.85 | 0.70-0.85 | <0.60 | Addresses the question |
| Context Precision | 0-1 | >0.80 | 0.65-0.80 | <0.50 | Retrieved chunks are relevant |
| Context Recall | 0-1 | >0.80 | 0.65-0.80 | <0.50 | All relevant info retrieved |

### LLM-as-Judge Metrics

| Score | Label | Description |
|-------|-------|-------------|
| 4.0 | Excellent | Meets all criteria with no issues |
| 3.0-3.9 | Good | Meets most criteria with minor issues |
| 2.0-2.9 | Fair | Meets some criteria but has notable gaps |
| 1.0-1.9 | Poor | Fails to meet most criteria |

## Datasets

### English Datasets
- **MTSDialog**: Dialog → SOAP/summary (1.7k dialogs)
- **MedDialogEN**: Doctor-patient dialogs (260k dialogs)

### Arabic Datasets
- **AHQAD**: Arabic healthcare Q&A (808k pairs, 90 categories)
- **Arabic Medical Dialogue**: Mental health consultations (3.6k dialogs)
- **MedArabiQ**: 7 Arabic medical reasoning tasks

See [ADR-003](../../docs/adrs/003-evals.md) for complete dataset details.

## Implementation Notes

### Fully Implemented Features

1. **Retrieval Quality Metrics**: Complete implementation of nDCG, MRR, Precision, Recall, MAP, Hit Rate using numpy calculations

2. **RAGAS Integration**: Full integration with the RAGAS library for Faithfulness, Answer Relevance, Context Precision/Recall
   - Dependencies: `ragas`, `langchain`, `langchain-openai`, `datasets` (installed via Poetry)
   - Requires: `OPENAI_API_KEY` environment variable

3. **LLM-as-Judge**: Complete implementation using OpenAI API with G-Eval style rubrics
   - Evaluates: Relevance, Groundedness, Completeness, Coherence, Fluency (1-4 scale)
   - Dependencies: `openai` (installed via Poetry)
   - Requires: `OPENAI_API_KEY` environment variable
   - Samples up to 20 queries for efficiency
   - Includes detailed rubric prompts and score explanations

4. **Error Handling**: Graceful degradation when optional libraries are not installed
   - Will skip RAGAS metrics if library not available
   - Will skip LLM-as-Judge if API key not configured
   - Continues with retrieval metrics in all cases

### Production Checklist

- [x] Complete retrieval metrics implementation
- [x] RAGAS library integration
- [x] LLM-as-Judge with OpenAI API
- [ ] Integrate with actual retrieval system (currently uses mock results)
- [ ] Configure cost tracking for LLM API calls
- [ ] Implement CSV report generation for detailed analysis
- [ ] Add statistical significance testing
- [ ] Support for Azure OpenAI and other LLM providers

## References

- ADR-003: [Evaluation Standards](../../docs/adrs/003-evals.md)
- RAGAS Framework: [docs.ragas.io](https://docs.ragas.io)
- BEIR Benchmark: [beir-cellar.github.io](https://beir-cellar.github.io)
- G-Eval Paper: [arxiv.org/abs/2303.16634](https://arxiv.org/abs/2303.16634)

## Contributing

When adding new metrics or features:

1. Update the Pydantic models in `models/evaluation_results.py`
2. Implement the metric calculation in `eval/rag/main.py`
3. Add tests for the new metric
4. Update this README and the ADR

## License

See project LICENSE file.
