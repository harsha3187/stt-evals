#!/bin/bash
# RAGAS Integration Test Script
# Tests RAGAS metrics with Azure OpenAI gpt-5 deployment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "RAGAS Integration Test"
echo "================================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "test_rag_pipeline.py" ]; then
    echo -e "${RED}❌ Error: Please run this script from tests/eval/rag/ directory${NC}"
    exit 1
fi

# Check if dataset is prepared
if [ ! -f "data/ahqad/queries.jsonl" ]; then
    echo -e "${RED}❌ Error: AHQAD dataset not found!${NC}"
    echo "Please prepare the dataset first:"
    echo "  python scripts/prepare_ahqad.py --sample 100"
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ Error: .env file not found!${NC}"
    echo "Please create .env from .env.example and add your Azure credentials"
    exit 1
fi

echo -e "${BLUE}Step 1: Checking Dependencies${NC}"
echo "----------------------------------------"

# Check if packages are installed
echo "Checking for RAGAS..."
if python -c "import ragas" 2>/dev/null; then
    RAGAS_VERSION=$(python -c "import ragas; print(ragas.__version__)")
    echo -e "${GREEN}✅ RAGAS installed (version: $RAGAS_VERSION)${NC}"
else
    echo -e "${YELLOW}⚠️  RAGAS not installed${NC}"
    echo ""
    read -p "Install RAGAS and dependencies? [Y/n] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "Installing RAGAS dependencies..."
        pip install ragas langchain langchain-openai datasets
        echo -e "${GREEN}✅ Dependencies installed${NC}"
    else
        echo -e "${RED}❌ Cannot continue without RAGAS${NC}"
        exit 1
    fi
fi

# Verify other dependencies
echo ""
echo "Verifying dependencies..."
python -c "from langchain_openai import AzureChatOpenAI; print('✅ langchain-openai: OK')" || {
    echo -e "${RED}❌ langchain-openai not found${NC}"
    exit 1
}
python -c "import datasets; print('✅ datasets: OK')" || {
    echo -e "${RED}❌ datasets not found${NC}"
    exit 1
}

echo ""
echo -e "${BLUE}Step 2: Environment Check${NC}"
echo "----------------------------------------"

# Source .env file to check variables
source .env

if [ "$USE_AZURE" = "true" ]; then
    echo -e "${GREEN}✅ Azure OpenAI: Enabled${NC}"
    echo "   Endpoint: $AZURE_OPENAI_ENDPOINT"
    echo "   Deployment: $AZURE_DEPLOYMENT_NAME"
else
    echo -e "${YELLOW}⚠️  Azure OpenAI: Disabled${NC}"
    echo "   Will use standard OpenAI instead"
fi

echo ""
echo -e "${BLUE}Step 3: Dataset Information${NC}"
echo "----------------------------------------"

# Count samples in dataset
QUERY_COUNT=$(wc -l < data/ahqad/queries.jsonl)
echo "   Queries: $QUERY_COUNT"
echo "   Corpus: $(wc -l < data/ahqad/corpus.jsonl)"
echo "   QRels: $(wc -l < data/ahqad/qrels.jsonl)"
echo "   Answers: $(wc -l < data/ahqad/answers.jsonl)"

echo ""
echo -e "${BLUE}Step 4: Test Configuration${NC}"
echo "----------------------------------------"
echo "   This test will:"
echo "   • Evaluate $QUERY_COUNT Arabic medical Q&A pairs"
echo "   • Use Azure OpenAI gpt-5 deployment"
echo "   • Compute RAGAS metrics:"
echo "     - Faithfulness (claims supported by context)"
echo "     - Answer Relevance (semantic alignment)"
echo "     - Context Precision (retrieval quality)"
echo "     - Context Recall (retrieval coverage)"
echo ""
echo "   Expected:"
echo "   • Duration: ~3-5 minutes"
echo "   • API Calls: ~400 to Azure gpt-5"
echo "   • Cost: ~\$1-2 (depends on Azure pricing)"
echo ""

read -p "Continue with RAGAS evaluation? [Y/n] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Test cancelled."
    exit 0
fi

echo ""
echo -e "${BLUE}Step 5: Running RAGAS Evaluation${NC}"
echo "----------------------------------------"
echo "Starting evaluation (this will take a few minutes)..."
echo ""

# Record start time
START_TIME=$(date +%s)

# Run the evaluation
python test_rag_pipeline.py --data-dir data/ahqad --enable-ragas

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ RAGAS Evaluation Complete!${NC}"
echo "================================================================================"
echo "Duration: ${MINUTES}m ${SECONDS}s"
echo ""

echo -e "${BLUE}Step 6: Results Summary${NC}"
echo "----------------------------------------"

# Check if results file exists
if [ -f "results/ahqad_test_results.json" ]; then
    echo "Results saved to: results/ahqad_test_results.json"
    echo ""

    # Try to display RAGAS metrics if jq is available
    if command -v jq &> /dev/null; then
        echo "RAGAS Metrics:"
        jq '.ragas_metrics' results/ahqad_test_results.json 2>/dev/null || {
            echo "Use this command to view RAGAS metrics:"
            echo "  cat results/ahqad_test_results.json | jq .ragas_metrics"
        }
    else
        echo "Install jq to view formatted results:"
        echo "  sudo apt-get install jq"
        echo ""
        echo "Or view raw JSON:"
        echo "  cat results/ahqad_test_results.json"
    fi
else
    echo -e "${YELLOW}⚠️  Results file not found${NC}"
fi

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Review RAGAS metrics in the results file"
echo "2. Test full evaluation (RAGAS + LLM-Judge):"
echo "   python test_rag_pipeline.py --data-dir data/ahqad --enable-ragas --enable-llm-judge"
echo "3. Monitor Azure costs in Azure Portal"
echo ""
echo "================================================================================"
