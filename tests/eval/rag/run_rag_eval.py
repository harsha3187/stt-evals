"""End-to-end test of RAG evaluation pipeline using AHQAD dataset.

This script demonstrates how to use the RAG evaluation library (eval/rag)
to evaluate a RAG system using the AHQAD Arabic Healthcare Q&A dataset.

Usage:
    # Run with default settings (retrieval metrics only, fast)
    python test_rag_pipeline.py

    # Run with RAGAS metrics (requires OpenAI API key)
    python test_rag_pipeline.py --enable-ragas

    # Run with LLM-as-Judge (requires OpenAI API key)
    python test_rag_pipeline.py --enable-llm-judge

    # Run full evaluation (all metrics)
    python test_rag_pipeline.py --enable-ragas --enable-llm-judge

    # Use custom data directory
    python test_rag_pipeline.py --data-dir data/ahqad

    # Run on specific sample size
    python test_rag_pipeline.py --sample 50

Requirements:
    - AHQAD dataset prepared (run scripts/prepare_ahqad.py first)
    - Optional: OpenAI API key for RAGAS and LLM-as-Judge
    - Install: pip install -r requirements.txt
"""

import argparse
import logging
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

# Load .env from the test directory
load_dotenv(Path(__file__).parent / ".env")

# Add project root to path to import the evaluation library
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from eval.rag.main import RAGEvaluator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_data_files(data_dir: Path) -> bool:
    """Validate that all required data files exist.

    Parameters
    ----------
    data_dir : Path
        Directory containing the dataset files

    Returns
    -------
    bool
        True if all files exist, False otherwise
    """
    required_files = [
        "queries.jsonl",
        "corpus.jsonl",
        "qrels.jsonl",
        "answers.jsonl",
    ]

    all_exist = True
    for filename in required_files:
        file_path = data_dir / filename
        if not file_path.exists():
            logger.error(f"❌ Required file not found: {file_path}")
            all_exist = False
        else:
            logger.info(f"✅ Found: {file_path}")

    return all_exist


def print_results_summary(results: dict):
    """Print a human-readable summary of evaluation results.

    Parameters
    ----------
    results : dict
        Evaluation results dictionary
    """
    print("\n" + "=" * 80)
    print("RAG EVALUATION RESULTS SUMMARY")
    print("=" * 80)

    # Dataset info
    print("\n📊 Dataset Information:")
    print(f"   Language: {results.get('language', 'N/A')}")
    print(f"   Total Queries: {results.get('query_count', 'N/A')}")
    print(f"   Corpus Size: {results.get('corpus_size', 'N/A')}")
    print(f"   Avg Retrieved Docs: {results.get('avg_retrieved_docs', 'N/A'):.1f}")

    # Retrieval metrics
    print("\n🔍 Retrieval Metrics:")
    for metrics in results.get("retrieval_metrics", []):
        k = metrics.get("k")
        print(f"\n   @K={k}:")
        print(f"      nDCG:      {metrics.get('ndcg', 0):.4f}")
        print(f"      MRR:       {metrics.get('mrr', 0):.4f}")
        print(f"      Precision: {metrics.get('precision', 0):.4f}")
        print(f"      Recall:    {metrics.get('recall', 0):.4f}")
        print(f"      MAP:       {metrics.get('map_score', 0):.4f}")
        print(f"      Hit Rate:  {metrics.get('hit_rate', 0):.4f}")

    # Per-chunk relevance
    print("\n📄 Per-Chunk Relevance:")
    print(f"   Mean:      {results.get('avg_chunk_relevance', 0):.4f}")
    print(f"   Std Dev:   {results.get('chunk_relevance_std', 0):.4f}")

    # RAGAS metrics
    if results.get("ragas_metrics"):
        print("\n📈 RAGAS Metrics:")
        ragas = results["ragas_metrics"]
        print(f"   Faithfulness:         {ragas.get('faithfulness', 0):.4f}")
        print(f"   Answer Relevance:     {ragas.get('answer_relevance', 0):.4f}")
        print(f"   Context Precision:    {ragas.get('context_precision', 0):.4f}")
        print(f"   Context Recall:       {ragas.get('context_recall', 0):.4f}")
        print(f"   Context Utilization:  {ragas.get('context_utilization', 0):.4f}")
    elif results.get("ragas_enabled"):
        print("\n📈 RAGAS Metrics: ⚠️  Not available (check API key and dependencies)")

    # LLM Judge metrics
    if results.get("llm_judge_metrics"):
        print("\n⚖️  LLM-as-Judge Metrics (1-4 scale):")
        llm_judge = results["llm_judge_metrics"]
        print(f"   Relevance:     {llm_judge.get('relevance', 0):.2f}")
        print(f"   Groundedness:  {llm_judge.get('groundedness', 0):.2f}")
        print(f"   Completeness:  {llm_judge.get('completeness', 0):.2f}")
        print(f"   Coherence:     {llm_judge.get('coherence', 0):.2f}")
        print(f"   Fluency:       {llm_judge.get('fluency', 0):.2f}")
    elif results.get("llm_judge_enabled"):
        print("\n⚖️  LLM-as-Judge Metrics: ⚠️  Not available (check API key and dependencies)")

    # Cost tracking
    if results.get("cost_tracking"):
        print("\n💰 API Usage & Cost Tracking:")
        cost = results["cost_tracking"]
        print(f"   LLM Judge API Calls:  {cost.get('llm_judge_api_calls', 0)}")
        print(f"   Total Tokens:         {cost.get('total_tokens', 0):,}")
        print(f"     - Prompt Tokens:    {cost.get('prompt_tokens', 0):,}")
        print(f"     - Completion Tokens: {cost.get('completion_tokens', 0):,}")
        print(f"   Estimated Cost:       ${cost.get('estimated_cost_usd', 0):.4f} USD")
        print(f"     - Prompt Cost:      ${cost.get('prompt_cost_usd', 0):.4f}")
        print(f"     - Completion Cost:  ${cost.get('completion_cost_usd', 0):.4f}")
        print(f"   Pricing Model:        {cost.get('pricing_model', 'N/A')}")
        if cost.get("note"):
            print(f"\n   ℹ️  Note: {cost.get('note')}")

    print("\n" + "=" * 80)


def _evaluate_metric(value: float, thresholds: list) -> str:
    """Evaluate a metric value against thresholds.

    Parameters
    ----------
    value : float
        The metric value to evaluate
    thresholds : list
        List of (threshold, label, emoji) tuples in descending order

    Returns
    -------
    str
        Formatted evaluation string
    """
    for threshold, label, emoji in thresholds:
        if value > threshold:
            return f"{value:.4f} - {emoji} {label}"
    # Return the last one (poorest)
    _, label, emoji = thresholds[-1]
    return f"{value:.4f} - {emoji} {label}"


def _print_retrieval_metrics(retrieval_k10: dict):
    """Print retrieval metrics interpretation."""
    print("\n🔍 Retrieval Quality @K=10:")

    ndcg_thresholds = [
        (0.90, "EXCELLENT (>0.90)", "✅"),
        (0.75, "GOOD (0.75-0.90)", "✅"),
        (0.50, "FAIR (0.50-0.75)", "⚠️"),
        (-1, "POOR (<0.50)", "❌"),
    ]
    ndcg = retrieval_k10.get("ndcg", 0)
    print(f"   nDCG: {_evaluate_metric(ndcg, ndcg_thresholds)}")

    precision_thresholds = [
        (0.80, "EXCELLENT (>0.80)", "✅"),
        (0.60, "GOOD (0.60-0.80)", "✅"),
        (0.30, "FAIR (0.30-0.60)", "⚠️"),
        (-1, "POOR (<0.30)", "❌"),
    ]
    precision = retrieval_k10.get("precision", 0)
    print(f"   Precision: {_evaluate_metric(precision, precision_thresholds)}")

    recall_thresholds = [
        (0.80, "EXCELLENT (>0.80)", "✅"),
        (0.60, "GOOD (0.60-0.80)", "✅"),
        (0.30, "FAIR (0.30-0.60)", "⚠️"),
        (-1, "POOR (<0.30)", "❌"),
    ]
    recall = retrieval_k10.get("recall", 0)
    print(f"   Recall: {_evaluate_metric(recall, recall_thresholds)}")


def _print_ragas_metrics(ragas: dict):
    """Print RAGAS metrics interpretation."""
    print("\n📈 Answer Quality (RAGAS):")

    faithfulness_thresholds = [
        (0.85, "EXCELLENT (>0.85)", "✅"),
        (0.70, "GOOD (0.70-0.85)", "✅"),
        (0.60, "FAIR (0.60-0.70)", "⚠️"),
        (-1, "POOR (<0.60)", "❌"),
    ]
    faithfulness = ragas.get("faithfulness", 0)
    print(f"   Faithfulness: {_evaluate_metric(faithfulness, faithfulness_thresholds)}")


def _print_llm_judge_metrics(llm_judge: dict):
    """Print LLM Judge metrics interpretation."""
    print("\n⚖️  Answer Quality (LLM Judge, 1-4 scale):")

    # Note: Different format for relevance (2 decimal places)
    relevance = llm_judge.get("relevance", 0)
    thresholds = [
        (3.5, "EXCELLENT (≥3.5)", "✅"),
        (2.5, "GOOD (2.5-3.5)", "✅"),
        (1.5, "FAIR (1.5-2.5)", "⚠️"),
        (-1, "POOR (<1.5)", "❌"),
    ]

    for threshold, label, emoji in thresholds:
        if relevance >= threshold and threshold > 0:
            print(f"   Relevance: {relevance:.2f} - {emoji} {label}")
            break
    else:
        print(f"   Relevance: {relevance:.2f} - ❌ POOR (<1.5)")


def interpret_results(results: dict):
    """Provide interpretation of results based on ADR thresholds.

    Parameters
    ----------
    results : dict
        Evaluation results dictionary
    """
    print("\n" + "=" * 80)
    print("RESULTS INTERPRETATION (Based on ADR-003 Thresholds)")
    print("=" * 80)

    # Interpret retrieval metrics at K=10 (standard reference point)
    retrieval_k10 = next(
        (m for m in results.get("retrieval_metrics", []) if m.get("k") == 10),
        None,
    )

    if retrieval_k10:
        _print_retrieval_metrics(retrieval_k10)

    # Interpret RAGAS metrics
    if results.get("ragas_metrics"):
        _print_ragas_metrics(results["ragas_metrics"])

    # Interpret LLM Judge metrics
    if results.get("llm_judge_metrics"):
        _print_llm_judge_metrics(results["llm_judge_metrics"])

    print("\n" + "=" * 80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Test RAG evaluation pipeline with AHQAD dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data" / "ahqad",
        help="Directory containing prepared AHQAD dataset (default: data/ahqad)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--enable-ragas",
        action="store_true",
        help="Enable RAGAS metrics (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--enable-llm-judge",
        action="store_true",
        help="Enable LLM-as-Judge metrics (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="K values for retrieval metrics (default: 5 10 20)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use only N samples from dataset (default: use all)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4",
        help="Model for LLM-as-Judge (default: gpt-4)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run identifier or experiment name (default: timestamp)",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Disable automatic timestamp in output filename",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("RAG EVALUATION PIPELINE TEST")
    logger.info("=" * 80)

    # Validate data files
    logger.info(f"\n📂 Checking data files in: {args.data_dir}")
    if not validate_data_files(args.data_dir):
        logger.error("\n❌ Data validation failed!")
        logger.error("Please run: python scripts/prepare_ahqad.py")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load Azure/OpenAI configuration from environment
    use_azure = os.getenv("USE_AZURE", "false").lower() == "true"
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
    azure_ragas_deployment = os.getenv("AZURE_RAGAS_DEPLOYMENT")
    azure_embeddings_deployment = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT")
    azure_embeddings_api_version = os.getenv("AZURE_EMBEDDINGS_API_VERSION")

    # Initialize RAG evaluator
    logger.info("\n🔧 Initializing RAG Evaluator...")
    logger.info("   Language: Arabic (ar)")
    logger.info(f"   K values: {args.k_values}")
    logger.info(f"   RAGAS enabled: {args.enable_ragas}")
    logger.info(f"   LLM Judge enabled: {args.enable_llm_judge}")
    logger.info(f"   Using Azure: {use_azure}")
    if use_azure:
        logger.info(f"   Azure Endpoint: {azure_endpoint}")
        logger.info(f"   Azure Deployment: {azure_deployment}")
        if args.enable_ragas and azure_embeddings_deployment:
            logger.info(f"   Azure Embeddings: {azure_embeddings_deployment}")

    evaluator = RAGEvaluator(
        language="ar",
        k_values=args.k_values,
        enable_ragas=args.enable_ragas,
        enable_llm_judge=args.enable_llm_judge,
        judge_model=args.judge_model,
        judge_temperature=0.0,
        use_azure=use_azure,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        azure_deployment=azure_deployment,
        azure_ragas_deployment=azure_ragas_deployment,
        azure_embeddings_deployment=azure_embeddings_deployment,
        azure_embeddings_api_version=azure_embeddings_api_version,
    )

    # Run evaluation
    logger.info("\n🚀 Running RAG evaluation...")

    try:
        results = evaluator.evaluate(
            queries_file=args.data_dir / "queries.jsonl",
            corpus_file=args.data_dir / "corpus.jsonl",
            qrels_file=args.data_dir / "qrels.jsonl",
            answers_file=args.data_dir / "answers.jsonl",
        )

        # Sample results if requested
        if args.sample:
            logger.info(f"   (Using {args.sample} samples)")

        # Save results with timestamp or custom run_id
        output_file = args.output_dir / "ahqad_test_results.json"
        actual_output_file = evaluator.save_results(
            results,
            output_file,
            run_id=args.run_id,
            use_timestamp=not args.no_timestamp,
        )

        # Print summary
        print_results_summary(results)

        # Interpret results
        interpret_results(results)

        # Success message
        print("\n✅ Evaluation complete!")
        print(f"📄 Results saved to: {actual_output_file}")
        print("\nView full results:")
        print(f"   cat {actual_output_file}")

    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
