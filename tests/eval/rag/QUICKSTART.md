# Quick Start Guide - RAG Evaluation Test

**Goal:** Test the RAG evaluation library with the AHQAD Arabic medical Q&A dataset in under 10 minutes.

## Prerequisites

- Python 3.13+
- pip
- Kaggle account (free)

## 3-Step Quick Start

### 1️⃣ Set up Kaggle credentials (one-time)

```bash
# Get API credentials from Kaggle
# Visit: https://www.kaggle.com/settings/account → "Create New API Token"

# Install credentials
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2️⃣ Run automated setup

```bash
cd tests/eval/rag
./quick_start.sh
```

**What it does:**
- Installs dependencies
- Downloads AHQAD dataset (100 sample Q&A pairs)
- Runs basic evaluation (no API costs)
- Shows results in terminal

### 3️⃣ View results

```bash
cat results/ahqad_test_results.json
```

## ✅ Done!

You should see:
- ✅ Retrieval metrics (nDCG, Precision, Recall, etc.)
- ✅ Evaluation scores @K=5, 10, 20
- ✅ Results saved to JSON file

## 🎯 What Was Tested

This validates that:
1. The `eval/rag` library works correctly as a reusable component
2. AHQAD dataset converts to evaluation format successfully
3. All retrieval metrics calculate properly
4. Arabic text is handled correctly

## 📚 Next Steps

### Run Full Evaluation (with RAGAS + LLM-as-Judge)

```bash
# Install optional dependencies
pip install ragas langchain langchain-openai datasets openai

# Set API key
export OPENAI_API_KEY="your-key-here"

# Run full evaluation
python test_rag_pipeline.py --enable-ragas --enable-llm-judge
```

**Cost:** ~$2-5 for 100 samples

### Use More Data

```bash
# Prepare 1,000 samples
python scripts/prepare_ahqad.py --sample 1000

# Run test
python test_rag_pipeline.py
```

### Customize Evaluation

```bash
# Custom K values
python test_rag_pipeline.py --k-values 3 5 10

# Different LLM judge model
python test_rag_pipeline.py --enable-llm-judge --judge-model gpt-3.5-turbo
```

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Kaggle credentials not found" | Follow step 1️⃣ above |
| "Module eval.rag not found" | Run from `tests/eval/rag/` directory |
| "No CSV file found" | Re-run `python scripts/prepare_ahqad.py` |

## 📖 Full Documentation

- **Detailed README:** [README.md](README.md)
- **Library Docs:** `../../../eval/rag/README.md`
- **ADR-003:** `../../../docs/adrs/003-evals.md`

---

**Time to complete:** ~5-10 minutes (first time)
**API costs:** $0 (basic test), ~$2-5 (full test with 100 samples)
**Dataset:** AHQAD Arabic Healthcare Q&A (CC BY 4.0)
