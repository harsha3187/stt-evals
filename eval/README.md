# Evaluation Framework

An end-to-end evaluation framework for **Speech-to-Text (STT)**, **Speaker Diarization**, and **Retrieval-Augmented Generation (RAG)** systems with type-safe Pydantic models, modern configuration management, and production-ready Azure ML logging.

## Overview

This framework provides end-to-end evaluation capabilities for AI systems with cloud experiment tracking, standardized metrics reporting and dataset manager. Built for the CAP to assess different services that uses the language models across multiple tasks.

## Key Features

- **Multi-Task Evaluation**: STT, Diarization, and RAG evaluation in one unified framework
- **Type Safety**: Pydantic models ensure data integrity and automatic validation
- **Cloud Integration**: Azure ML and MLflow experiment tracking for evaluation results
- **Automated Dataset Management**: Download and organize evaluation datasets from Hugging Face Hub
- **Flexible Output**: Save results locally and/or log to cloud with consistent paths

## Project Structure

```text
eval/
├── config.py                 # Configuration management with Pydantic Settings
├── main.py                   # Main orchestrator for evaluation tasks
├── dataset_manager.py        # Automated dataset management
├── .env.example              # Environment variables
├── models/                   # Pydantic data models for type-safe results
│   ├── __init__.py
│   └── evaluation_results.py
├── stt/                      # Speech-to-Text evaluation
│   ├── __init__.py
│   └── main.py
├── diarization/              # Speaker Diarization evaluation
│   ├── __init__.py
│   └── main.py
├── rag/                      # RAG (Retrieval-Augmented Generation) evaluation
│   ├── __init__.py
│   └── main.py
├── eval_logger/              # Cloud logging and experiment tracking
│   ├── __init__.py
│   ├── main.py               # Core evaluation logging functionality
│   └── integration.py        # Integration script for batch operations
├── datasets/                 # Evaluation datasets storage
└── results/                  # Local evaluation results
```

## Evaluation Tasks

### Speech-to-Text (STT) Evaluation

Evaluation of Arabic and English speech recognition systems using industry-standard metrics.

**Supported Metrics:**
- **WER** (Word Error Rate) - Primary metric for transcription accuracy
- **CER** (Character Error Rate) - Fine-grained character-level accuracy
- **MER** (Match Error Rate) - Alternative word-level metric
- **WIP/WIL** (Word Information Preserved/Lost) - Information theory metrics

**File Format Support:**
- **TXT Format**: One transcription per line (traditional format)
- **JSONL Format**: JSON objects with "text" field for structured data
- **Mixed Formats**: References and hypotheses can use different formats

**Quick Usage:**
```python
from eval.main import EvaluationOrchestrator

orchestrator = EvaluationOrchestrator()
result = orchestrator.run_stt_evaluation(
    ground_truth=["hello world", "test sentence"],
    generated=["hello word", "test sentance"],
    language="en",
    test_name="my_stt_test"
)
print(f"WER: {result.wer:.4f}, CER: {result.cer:.4f}")
```

### Speaker Diarization Evaluation

Evaluation of speaker diarization systems using RTTM (Rich Transcription Time Marked) format.

**Supported Metrics:**
- **DER** (Diarization Error Rate) - Primary diarization accuracy metric
- **JER** (Jaccard Error Rate) - Speaker overlap accuracy
- **Missed Speech/False Alarm/Speaker Confusion** - Detailed error breakdown

**Quick Usage:**
```python
result = orchestrator.run_diarization_evaluation(
    ground_truth_rttm="reference.rttm",
    generated_rttm="hypothesis.rttm",
    collar=0.25,  # 250ms tolerance
    test_name="my_diarization_test"
)
print(f"DER: {result.der:.4f}, JER: {result.jer:.4f}")
```

### RAG (Retrieval-Augmented Generation) Evaluation

Comprehensive evaluation of RAG systems with retrieval and generation quality metrics.

**Supported Metrics:**

- **Traditional NLP Metrics**: ROUGE (1/2/L), METEOR, Semantic Similarity
- **Embedding-based**: Clinical embedding similarity with cosine/euclidean distance  
- **Rubric-based Scoring**: Human-interpretable quality assessment for all metrics
- **Statistical Analysis**: Sample count tracking and evaluation timestamps

**Quick Usage:**

```python
result = orchestrator.run_rag_evaluation(
    ground_truth_file="references.jsonl",  # JSONL with 'text' field
    generated_file="candidates.jsonl",  # JSONL with 'text' field
    language="en",  # Arabic or English supported
    test_name="my_rag_test"
)
print(f"ROUGE-L: {result.summarization_metrics.rouge_l_f1:.4f}")
print(f"Clinical Similarity: {result.summarization_metrics.clinical_embedding_similarity:.4f}")
```

## Cloud Logging & Experiment Tracking

The evaluation framework integrates with **Azure ML** and **MLflow** for professional experiment tracking and result management. All evaluation runs can be automatically logged to the cloud for team collaboration and historical analysis.

### Why Use Cloud Logging?

- **Team Collaboration**: Share evaluation results across the CAP project team
- **Historical Tracking**: Compare model performance over time and across datasets
- **Experiment Management**: Organize evaluations by tasks, datasets, and sessions
- **Artifact Storage**: Automatically upload datasets and results for reproducibility
- **Visual Analytics**: Use Azure ML Studio for metric visualization and comparison

### Quick Setup

1. **Configure Environment Variables:**
   ```bash
   # Required Azure ML configuration
   export EVAL_CLOUD_WORKSPACE_NAME="Azure Ml Workspace name"
   export EVAL_CLOUD_RESOURCE_GROUP="Azure resource group"  
   export EVAL_CLOUD_SUBSCRIPTION_ID="your-subscription-id"
   export EVAL_CLOUD_ENABLE_LOGGING="true"
   
   # Optional (defaults provided)
   export EVAL_CLOUD_EXPERIMENT_NAME="cap-evaluation-experiments"
   ```

2. **Automatic Logging in Evaluation:**
   ```python
   from eval.main import EvaluationOrchestrator
   
   # Cloud logging is automatically enabled when configured
   orchestrator = EvaluationOrchestrator()
   result = orchestrator.run_stt_evaluation(
       references=refs,
       hypotheses=hyps,
       test_name="arabic_stt_baseline"  # Results saved locally AND logged to cloud
   )
   # ✓ Results automatically uploaded to Azure ML
   ```

3. **Manual Cloud Logging:**
   ```python
   from eval.eval_logger import log_evaluation_result
   
   # Log any existing result to cloud
   run_id = log_evaluation_result(
       result=stt_result,
       dataset_paths={"references": "refs.txt", "hypotheses": "hyps.txt"}
   )
   print(f"Logged to Azure ML with run ID: {run_id}")
   ```

### Batch Operations

Upload existing results without re-running evaluations:

```bash
# Check your cloud configuration
python -m eval.eval_logger.integration --check-config

# Upload a single result file  
python -m eval.eval_logger.integration --result-file results/stt_eval.json

# Batch upload all results from directory
python -m eval.eval_logger.integration --results-dir results/

# List existing experiments in Azure ML
python -m eval.eval_logger.integration --list-experiments
```

### Cloud Logging Features

- **Multi-Task Support**: STT, Diarization, RAG, and Combined evaluation logging
- **Session Grouping**: Group related runs with session IDs for batch experiments  
- **Dataset Tracking**: Automatically upload and version datasets used in evaluations
- **Metric Visualization**: View results in Azure ML Studio with charts and comparisons
- **Artifact Management**: Store evaluation configurations, outputs, and analysis notebooks
- **Type-Safe Logging**: Automatic filtering of non-numeric fields prevents logging errors
- **RAG Metrics Support**: Full support for traditional NLP, embedding, and rubric-based metrics

## Running Evaluations

### Command Line Interface (Recommended)

The easiest way to run evaluations with automatic result saving and cloud logging:

```bash
# STT Evaluation - Basic usage
python -m eval.main --task stt --ground-truth refs.txt --generated hyps.txt

# STT with custom test name and Arabic language
python -m eval.main --task stt --ground-truth refs.jsonl --generated hyps.jsonl \
    --language ar --test-name "arabic_stt_experiment"

# Diarization Evaluation
python -m eval.main --task diarization --ground-truth-rttm ref.rttm --generated-rttm hyp.rttm \
    --collar 0.25 --test-name "diarization_baseline"

# RAG Evaluation
python -m eval.main --task rag --ground-truth-rag references.jsonl --generated-rag candidates.jsonl \
    --language ar --test-name "rag_summarization_test"

# Combined evaluation from config file
python -m eval.main --task all --config evaluation_config.yaml
```

### Python API (For Integration)

Use the Python API when integrating with other systems or custom workflows:

```python
from eval.main import EvaluationOrchestrator

# Initialize orchestrator (auto-loads configuration)
orchestrator = EvaluationOrchestrator()

# Run individual evaluations
stt_result = orchestrator.run_stt_evaluation(
    references=["مرحبا بالعالم", "جملة تجريبية"],
    hypotheses=["مرحبا بالعالم", "جملة تجريبيه"],
    language="ar",
    test_name="arabic_stt_test"
)

# Run RAG evaluation with file inputs
rag_result = orchestrator.run_rag_evaluation(
    ground_truth_file="reference_summaries.jsonl",
    generated_file="generated_summaries.jsonl",
    language="ar",
    test_name="my_rag_test"  # Automatically saves and logs
)
```

### Output Path Management

The framework automatically manages result paths for consistency:

- **Default behavior**: Results saved to `eval/results/{task}_{timestamp}/{task}_evaluation_results.json`
- **Custom test name**: Results saved to `eval/results/{test_name}/{task}_evaluation_results.json`
- **Custom output path**: Full control over where results are saved
- **Cloud logging**: Always enabled when configured, regardless of local path settings

Examples:
```bash
# Auto-generated paths
python -m eval.main --task stt --ground-truth refs.txt --generated hyps.txt
# → eval/results/stt_20251112_143052/stt_evaluation_results.json

# Custom test name
python -m eval.main --task stt --ground-truth refs.txt --generated hyps.txt --test-name "baseline"
# → eval/results/baseline/stt_evaluation_results.json

# RAG evaluation with custom test name
python -m eval.main --task rag --ground-truth-rag refs.jsonl --generated-rag cands.jsonl --test-name "clinical_rag"
# → eval/results/clinical_rag/rag_evaluation_results.json

# Custom output directory
python -m eval.main --task stt --ground-truth refs.txt --generated hyps.txt --output "/custom/results/"
# → /custom/results/stt_20251112_143052_results.json
```

## Dataset Management

The evaluation framework includes automated dataset management that downloads and organizes evaluation datasets from Hugging Face Hub.

### Quick Dataset Setup

```python
from eval.dataset_manager import setup_eval_datasets

# Download recommended datasets for evaluation
datasets = setup_eval_datasets(quick_setup=True)

# Download datasets for specific tasks
from eval.dataset_manager import setup_stt_datasets
stt_datasets = setup_stt_datasets()
```

### Dataset Features

- **Automated Downloads**: Download datasets from Hugging Face Hub with intelligent caching
- **Multi-task Support**: Organize datasets by task (STT, diarization, RAG) and language
- **Flexible Selection**: Download specific datasets, task-based collections, or recommended sets
- **Status Monitoring**: Track download progress and dataset availability
- **Sample Generation**: Create inspection samples for quick dataset exploration

For detailed dataset documentation, see [Dataset Management Documentation](datasets/README.md).

## Installation and Setup

### Prerequisites

This project uses a DevContainer for consistent development. See the [main project README](../README.md) for complete setup instructions.

### Quick Start

1. **Open in DevContainer** (automatically installs all dependencies)
2. **Set up environment variables:**

   ```bash
   cp eval/.env.example eval/.env
   # Edit eval/.env with your Azure ML configuration
   ```

3. **Verify installation:**

   ```bash
   python -m eval.main --create-sample-config
   python -m eval.config --validate
   ```

### Dependencies

The evaluation framework requires:

- **Core**: `pydantic`, `pydantic-settings` for configuration and data models
- **STT**: `jiwer`, `rapidfuzz` for speech recognition metrics
- **Diarization**: `numpy`, `scipy` for audio processing and metrics
- **RAG**: `datasets`, `sentence-transformers` for retrieval evaluation
- **Cloud Logging**: `azure-ai-ml`, `mlflow` for Azure ML integration

All dependencies are pre-installed in the DevContainer via Poetry.

## Data Models

The framework uses type-safe Pydantic models for structured results:

### STTEvaluationResult
- **Metrics**: WER, CER, MER, WIP, WIL with Levenshtein distances
- **Metadata**: Language, sample count, evaluation timestamp

### DiarizationEvaluationResult
- **Metrics**: DER, JER with detailed error breakdown (missed speech, false alarms, speaker confusion)
- **Configuration**: Collar tolerance and timing statistics

### RAGEvaluationResult
- **Traditional Metrics**: ROUGE-1/2/L F1 scores, METEOR score for text quality
- **Semantic Similarity**: Cosine and Euclidean distance-based similarity measures  
- **Clinical Embeddings**: Specialized medical domain embedding similarity
- **Rubric Scoring**: Human-interpretable scores for all traditional metrics
- **Metadata**: Language, sample count, evaluation timestamp for tracking

### CombinedEvaluationResult
- **Session Tracking**: Unique session ID, tasks performed, execution time
- **Multi-task Results**: Optional STT, diarization, and RAG results
- **Configuration**: Source files and output directory tracking

## 🔧 Troubleshooting

### Common Issues

- **Import errors**: Ensure you're in the DevContainer or run `poetry install`
- **Azure ML connection**: Verify your Azure credentials with `az login`
- **Path issues**: All paths are automatically resolved to absolute paths
- **Configuration errors**: Run `python -m eval.config --validate` to check settings
- **RAG cloud logging errors**: Fixed - non-numeric fields are automatically filtered during cloud logging

### Debug Mode

Enable detailed logging:

```bash
export EVAL_LOG_LEVEL=DEBUG
python -m eval.main --task stt --references refs.txt --hypotheses hyps.txt
```

### Getting Help

- Check the [main project README](../README.md) for DevContainer setup
- Review configuration with `python -m eval.config --diff`
- Test individual components with provided examples in each module
