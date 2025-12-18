# RAG Evaluation Test Pipeline

This directory contains an end-to-end test pipeline for the RAG evaluation library (`eval/rag`), using the **AHQAD Arabic Healthcare Q&A dataset**.

## 📁 Directory Structure

```
tests/eval/rag/
├── README.md                   # This file
├── requirements.txt            # Test-specific dependencies
├── test_rag_pipeline.py        # Main test script (uses the library)
├── scripts/
│   └── prepare_ahqad.py        # Dataset download and conversion
├── data/
│   └── ahqad/                  # Prepared dataset (created by prepare_ahqad.py)
│       ├── queries.jsonl
│       ├── corpus.jsonl
│       ├── qrels.jsonl
│       ├── answers.jsonl
│       └── metadata.json
└── results/
    └── ahqad_test_results.json # Evaluation results (created by test)
```

## 🎯 Purpose

This test demonstrates:
1. How to use the RAG evaluation library as a reusable component
2. How to prepare a dataset for RAG evaluation
3. How to run retrieval metrics, RAGAS metrics, and LLM-as-Judge evaluation
4. How to interpret the results according to ADR-003 standards

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd tests/eval/rag
pip install -r requirements.txt
```

For full evaluation (RAGAS + LLM-as-Judge), also install:

```bash
pip install ragas langchain langchain-openai datasets openai
```

### Step 2: Set Up Kaggle API (for dataset download)

1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Step 3: Prepare the AHQAD Dataset

```bash
# Download and convert full dataset
python scripts/prepare_ahqad.py

# Or download and convert a sample (faster for testing)
python scripts/prepare_ahqad.py --sample 100

# Or use a specific output directory
python scripts/prepare_ahqad.py --output-dir data/ahqad_sample --sample 50
```

**Output:** Creates JSONL files in `data/ahqad/` directory.

### Step 4: Configure Azure OpenAI

Copy the example configuration and fill in your Azure OpenAI credentials:

```bash
cp .env.example .env
# Edit .env with your Azure OpenAI credentials:
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_API_KEY
# - AZURE_DEPLOYMENT_NAME
# - AZURE_EMBEDDINGS_DEPLOYMENT
```

### Step 5: Run the Evaluation Test

```bash
# Quick test - retrieval metrics only (no API costs)
python test_rag_pipeline.py

# Full test - all metrics (requires Azure OpenAI credentials in .env)
python test_rag_pipeline.py --enable-ragas --enable-llm-judge
```

**Output:** Creates `results/ahqad_test_results.json` with all evaluation metrics.

## 🔑 Azure OpenAI Configuration

This evaluation framework requires Azure OpenAI for enterprise security, compliance, and data residency.

### Getting Azure OpenAI Credentials

1. **Access Azure Portal**: Go to [https://portal.azure.com](https://portal.azure.com)
2. **Create or Select Resource**: Navigate to your Azure OpenAI resource
3. **Get Credentials**:
   - Go to "Keys and Endpoint" section
   - Copy the **Endpoint URL** (e.g., `https://your-resource.openai.azure.com/`)
   - Copy **KEY 1** or **KEY 2**
4. **Create Model Deployments**:
   - Go to "Model deployments" or Azure OpenAI Studio
   - Create a GPT-4 deployment (for LLM Judge)
   - Create a text-embedding-3-large deployment (for RAGAS)
   - Note: Deployment names can be different from model names

### Configuration File

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
USE_AZURE=true
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-api-key-here
AZURE_DEPLOYMENT_NAME=gpt-4
AZURE_EMBEDDINGS_DEPLOYMENT=text-embedding-3-large
```

### Available Models

Common Azure OpenAI deployments:
- **LLM Models**: gpt-4, gpt-4-turbo, gpt-4o, gpt-35-turbo, gpt-5-preview
- **Embeddings**: text-embedding-3-large, text-embedding-3-small, text-embedding-ada-002

### Cost Optimization

Use different deployments for cost optimization:
```bash
AZURE_DEPLOYMENT_NAME=gpt-4           # High-quality judge
AZURE_RAGAS_DEPLOYMENT=gpt-35-turbo   # Cost-effective RAGAS
AZURE_EMBEDDINGS_DEPLOYMENT=text-embedding-3-small  # Smaller embeddings
```

## 📋 Detailed Usage

### Dataset Preparation Options

```bash
# Full dataset (808k Q&A pairs)
python scripts/prepare_ahqad.py

# Sample for quick testing
python scripts/prepare_ahqad.py --sample 100

# Custom output directory
python scripts/prepare_ahqad.py --output-dir /path/to/output

# Skip download if data already exists
python scripts/prepare_ahqad.py --skip-download
```

### Evaluation Test Options

```bash
# Basic test (retrieval metrics only)
python test_rag_pipeline.py

# Enable RAGAS metrics
python test_rag_pipeline.py --enable-ragas

# Enable LLM-as-Judge
python test_rag_pipeline.py --enable-llm-judge

# Full evaluation
python test_rag_pipeline.py --enable-ragas --enable-llm-judge

# Custom K values for retrieval metrics
python test_rag_pipeline.py --k-values 3 5 10 20

# Use custom data directory
python test_rag_pipeline.py --data-dir data/ahqad_sample

# Use custom LLM judge model
python test_rag_pipeline.py --enable-llm-judge --judge-model gpt-3.5-turbo

# Combine options
python test_rag_pipeline.py \
    --enable-ragas \
    --enable-llm-judge \
    --k-values 5 10 \
    --judge-model gpt-4 \
    --output-dir results/full_evaluation
```

## 📊 Understanding the Results

### Output Structure

The test creates a JSON file with the following structure:

```json
{
  "retrieval_metrics": [
    {"k": 5, "ndcg": 0.85, "mrr": 0.90, ...},
    {"k": 10, "ndcg": 0.82, "mrr": 0.88, ...},
    {"k": 20, "ndcg": 0.78, "mrr": 0.85, ...}
  ],
  "avg_chunk_relevance": 0.78,
  "chunk_relevance_std": 0.15,
  "ragas_metrics": {
    "faithfulness": 0.88,
    "answer_relevance": 0.85,
    "context_precision": 0.80,
    "context_recall": 0.82,
    "context_utilization": 0.81
  },
  "llm_judge_metrics": {
    "relevance": 3.8,
    "groundedness": 3.9,
    "completeness": 3.5,
    "coherence": 4.0,
    "fluency": 3.2
  },
  "cost_tracking": {
    "total_tokens": 15420,
    "prompt_tokens": 12340,
    "completion_tokens": 3080,
    "llm_judge_api_calls": 20,
    "estimated_cost_usd": 0.5544,
    "pricing_model": "gpt-4"
  },
  "language": "ar",
  "query_count": 100,
  "corpus_size": 100
}
```

### Interpreting Metrics (Based on ADR-003)

#### Retrieval Metrics (0-1 scale)

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| nDCG@K | >0.90 | 0.75-0.90 | 0.50-0.75 | <0.50 |
| MRR@K | >0.90 | 0.75-0.90 | 0.50-0.75 | <0.50 |
| Precision@K | >0.80 | 0.60-0.80 | 0.30-0.60 | <0.30 |
| Recall@K | >0.80 | 0.60-0.80 | 0.30-0.60 | <0.30 |

#### RAGAS Metrics (0-1 scale)

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| Faithfulness | >0.85 | 0.70-0.85 | 0.60-0.70 | <0.60 |
| Answer Relevance | >0.85 | 0.70-0.85 | 0.60-0.70 | <0.60 |

#### LLM-as-Judge Metrics (1-4 scale)

| Score | Label | Meaning |
|-------|-------|---------|
| ≥3.5 | Excellent | Meets all criteria with no issues |
| 2.5-3.5 | Good | Meets most criteria with minor issues |
| 1.5-2.5 | Fair | Meets some criteria but has notable gaps |
| <1.5 | Poor | Fails to meet most criteria |

### Console Output

The test script provides detailed console output:

```
================================================================================
RAG EVALUATION RESULTS SUMMARY
================================================================================

📊 Dataset Information:
   Language: ar
   Total Queries: 100
   Corpus Size: 100

🔍 Retrieval Metrics:

   @K=10:
      nDCG:      0.8523
      MRR:       0.9012
      Precision: 0.7800
      Recall:    0.8200
      MAP:       0.8150
      Hit Rate:  0.9500

⚖️  LLM-as-Judge Metrics (1-4 scale):
   Relevance:     3.80
   Groundedness:  3.90
   Completeness:  3.50
   Coherence:     4.00
   Fluency:       3.20

💰 API Usage & Cost Tracking:
   LLM Judge API Calls:  20
   Total Tokens:         15,420
     - Prompt Tokens:    12,340
     - Completion Tokens: 3,080
   Estimated Cost:       $0.5544 USD
     - Prompt Cost:      $0.3702
     - Completion Cost:  $0.1848
   Pricing Model:        gpt-4

   ℹ️  Note: RAGAS token usage not fully tracked due to library limitations. Actual costs may be higher.

================================================================================
RESULTS INTERPRETATION (Based on ADR-003 Thresholds)
================================================================================

🔍 Retrieval Quality @K=10:
   nDCG: 0.8523 - ✅ GOOD (0.75-0.90)
   Precision: 0.7800 - ✅ GOOD (0.60-0.80)
   Recall: 0.8200 - ✅ EXCELLENT (>0.80)
```

## 🧪 What This Test Validates

### Library Functionality

- ✅ RAG evaluation library can be imported and used independently
- ✅ All retrieval metrics calculate correctly (nDCG, MRR, Precision, Recall, MAP, Hit Rate)
- ✅ RAGAS integration works when enabled
- ✅ LLM-as-Judge integration works when enabled
- ✅ JSONL data format is correctly parsed
- ✅ Results are properly formatted and saved

### Dataset Suitability

- ✅ AHQAD dataset converts correctly to RAG evaluation format
- ✅ Arabic text is handled properly (UTF-8 encoding)
- ✅ Ground truth Q&A pairs produce expected high scores
- ✅ Medical domain content is suitable for evaluation

### ADR-003 Compliance

- ✅ All required metrics from ADR-003 Section 2.3 (Retrieval)
- ✅ All required metrics from ADR-003 Section 2.4 (Answer Quality)
- ✅ Rubric-based scoring (1-4 scale) matches ADR-003 Appendix
- ✅ Arabic language support as specified in ADR-003

## 💡 Expected Results

For the AHQAD test dataset (ground truth Q&A pairs), you should see:

- **High retrieval metrics** (>0.8): Since each question maps to its correct answer
- **High RAGAS faithfulness** (>0.85): Answers are ground truth, fully faithful
- **Good LLM judge scores** (3.5-4.0): Medical Q&A should be relevant and complete
- **Lower fluency scores for Arabic**: Medical terminology may affect fluency ratings

If scores are unexpectedly low (<0.5), check:
1. Data conversion was successful
2. JSONL files are properly formatted
3. API keys are set correctly for RAGAS/LLM-Judge

## 💰 Azure OpenAI Cost Tracking

The evaluation library automatically tracks Azure OpenAI API usage and estimates costs when using RAGAS or LLM-as-Judge metrics.

### What Gets Tracked

- **Token Usage**: Prompt tokens and completion tokens from Azure OpenAI API calls
- **API Calls**: Number of calls to Azure OpenAI for LLM-as-Judge evaluation
- **Cost Estimation**: Calculated based on Azure OpenAI pricing for GPT-4/GPT-3.5

### Cost Information in Results

The `cost_tracking` field in results includes:

```json
{
  "total_tokens": 15420,
  "prompt_tokens": 12340,
  "completion_tokens": 3080,
  "llm_judge_api_calls": 20,
  "ragas_api_calls_estimated": 0,
  "estimated_cost_usd": 0.5544,
  "prompt_cost_usd": 0.3702,
  "completion_cost_usd": 0.1848,
  "pricing_model": "gpt-4",
  "note": "RAGAS token usage not fully tracked due to library limitations. Actual costs may be higher."
}
```

### Pricing Models

The library uses these default pricing estimates (per 1,000 tokens):

| Model | Prompt Tokens | Completion Tokens |
|-------|--------------|-------------------|
| GPT-4 / GPT-5 | $0.03 | $0.06 |
| GPT-3.5-Turbo | $0.0015 | $0.002 |

**Note**: Actual Azure pricing may vary based on your subscription tier and region. Check the Azure Portal for accurate costs.

### Typical Costs

For AHQAD dataset evaluation (100 samples):

| Component | API Calls | Tokens (est.) | Cost (est.) |
|-----------|----------|---------------|-------------|
| Retrieval Metrics | 0 | 0 | $0 |
| LLM-as-Judge | ~20 | ~15K | $0.50-1.00 |
| RAGAS | ~400* | ~800K* | $2.00-4.00 |
| **Full Evaluation** | **~420** | **~815K** | **$2.50-5.00** |

*RAGAS token counts are estimates as the library doesn't expose detailed usage.

### Monitoring Azure OpenAI Costs

Track actual costs in Azure Portal:

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Azure OpenAI resource (Cognitive Services)
3. Select "Cost Management" → "Cost analysis"
4. Filter by resource and time range
5. Compare with evaluation timestamps from results JSON
6. Set up budget alerts to prevent overspending

**Note**: Azure OpenAI pricing varies by region and subscription tier. Check your specific pricing in the Azure Portal.

### Cost Control Tips

1. **Use `--sample N`** to limit evaluation to N queries
2. **Start with retrieval-only** tests (free, no API calls)
3. **Test LLM-Judge first** (cheaper than full RAGAS)
4. **Monitor token usage** in results before scaling up
5. **Set Azure spending limits** in the portal

## 🔧 Troubleshooting

### "Kaggle credentials not found"

```bash
# Set up Kaggle API credentials
mkdir -p ~/.kaggle
# Download kaggle.json from https://www.kaggle.com/settings/account
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### "RAGAS metrics not available"

```bash
# Install RAGAS dependencies
pip install ragas langchain langchain-openai datasets

# Configure Azure OpenAI credentials in .env
cp .env.example .env
# Edit .env and set:
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_API_KEY
# - AZURE_DEPLOYMENT_NAME (e.g., gpt-4)
# - AZURE_EMBEDDINGS_DEPLOYMENT (e.g., text-embedding-3-large)
```

### "No CSV file found"

```bash
# Re-download the dataset
python scripts/prepare_ahqad.py --output-dir data/ahqad
```

### "Module 'eval.rag' not found"

```bash
# Make sure you're running from the tests/eval/rag directory
cd tests/eval/rag
python test_rag_pipeline.py
```

## 📚 Related Documentation

- **ADR-003**: `docs/adrs/003-evals.md` - Evaluation standards and metrics
- **Library README**: `eval/rag/README.md` - RAG evaluation library documentation
- **AHQAD Dataset**: https://www.kaggle.com/datasets/abdoashraf90/ahqad-arabic-healthcare-q-and-a-dataset

## 🤝 Contributing

When modifying the test pipeline:

1. Keep test code separate from library code (`eval/rag/`)
2. Update this README if you add new test options
3. Ensure the test still validates all ADR-003 requirements
4. Add example output for new metrics

## 📝 License

This test code follows the project's main license. The AHQAD dataset is licensed under CC BY 4.0.
