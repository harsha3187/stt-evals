"""RAGAS Integration Test

Tests RAGAS metrics with Azure OpenAI gpt-5 deployment.

This test validates the RAG evaluation pipeline's ability to compute RAGAS metrics
on the AHQAD Arabic Healthcare Q&A dataset using Azure OpenAI endpoints.
"""

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


class TestRAGASIntegration:
    """Test suite for RAGAS integration with RAG evaluation pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and change to correct directory."""
        self.test_dir = Path(__file__).parent
        original_dir = os.getcwd()
        os.chdir(self.test_dir)
        yield
        os.chdir(original_dir)

    @pytest.fixture(scope="class")
    def evaluation_results_file(self):
        """Fixture that provides the evaluation results file path.

        Returns the most recent evaluation results file. If no results exist,
        returns None (allowing dependent tests to skip gracefully).

        This fixture establishes the dependency relationship between tests:
        - Tests that need results should use this fixture
        - If evaluation was skipped/failed, dependent tests auto-skip
        """
        from glob import glob

        results_pattern = "results/ahqad_test_results*.json"
        results_files = sorted(glob(results_pattern))

        if not results_files:
            return None

        return Path(results_files[-1])

    def test_correct_directory(self):
        """Verify test is run from correct directory (tests/eval/rag/)."""
        assert (Path.cwd() / "run_rag_eval.py").exists(), "Error: Please run tests from tests/eval/rag/ directory"

    def test_dataset_prepared(self):
        """Verify AHQAD dataset files are prepared."""
        required_files = [
            "data/ahqad/queries.jsonl",
            "data/ahqad/corpus.jsonl",
            "data/ahqad/qrels.jsonl",
            "data/ahqad/answers.jsonl",
        ]

        for file_path in required_files:
            error_msg = f"Error: {file_path} not found!\nPlease prepare the dataset first:\n  python scripts/prepare_ahqad.py --sample 100"
            assert Path(file_path).exists(), error_msg

    def test_env_file_exists(self):
        """Verify .env file exists with Azure credentials."""
        if not Path(".env").exists():
            pytest.skip("Skipping: .env file not found.\nThis test requires API credentials.\nCreate .env from .env.example for full testing.")

    def test_dependency_ragas(self):
        """Verify RAGAS is installed."""
        try:
            import ragas

            version = ragas.__version__
            assert version, "RAGAS installed but version unavailable"
        except ImportError:
            pytest.skip("RAGAS not installed. Install with: pip install ragas langchain langchain-openai datasets")

    def test_dependency_langchain_openai(self):
        """Verify langchain-openai is installed."""
        try:
            from langchain_openai import AzureChatOpenAI

            assert AzureChatOpenAI is not None
        except ImportError:
            pytest.skip("langchain-openai not installed. Install with: pip install langchain-openai")

    def test_dependency_datasets(self):
        """Verify datasets library is installed."""
        try:
            import datasets

            assert datasets is not None
        except ImportError:
            pytest.skip("datasets library not installed. Install with: pip install datasets")

    def test_environment_variables_loaded(self):
        """Verify required environment variables are available."""
        from dotenv import load_dotenv

        # Skip if no .env file exists
        if not Path(".env").exists():
            pytest.skip("Skipping: .env file not found")

        load_dotenv(Path(".env"))

        use_azure = os.getenv("USE_AZURE", "false").lower() == "true"

        # Skip if no credentials are configured
        if not use_azure and not os.getenv("OPENAI_API_KEY"):
            pytest.skip("Skipping: No API credentials configured. Set USE_AZURE=true with Azure credentials, or OPENAI_API_KEY")

        if use_azure:
            assert os.getenv("AZURE_OPENAI_ENDPOINT"), "USE_AZURE=true but AZURE_OPENAI_ENDPOINT not set"
            assert os.getenv("AZURE_OPENAI_API_KEY"), "USE_AZURE=true but AZURE_OPENAI_API_KEY not set"
            assert os.getenv("AZURE_DEPLOYMENT_NAME"), "USE_AZURE=true but AZURE_DEPLOYMENT_NAME not set"

    def test_dataset_information(self):
        """Verify dataset files are readable and contain data."""
        data_files = {
            "Queries": "data/ahqad/queries.jsonl",
            "Corpus": "data/ahqad/corpus.jsonl",
            "QRels": "data/ahqad/qrels.jsonl",
            "Answers": "data/ahqad/answers.jsonl",
        }

        dataset_info = {}
        for name, file_path in data_files.items():
            path = Path(file_path)
            line_count = sum(1 for _ in open(path))
            dataset_info[name] = line_count
            assert line_count > 0, f"{name} file is empty: {file_path}"

        # Print dataset summary
        print("\n📊 Dataset Information:")
        print(f"   Queries: {dataset_info['Queries']}")
        print(f"   Corpus: {dataset_info['Corpus']}")
        print(f"   QRels: {dataset_info['QRels']}")
        print(f"   Answers: {dataset_info['Answers']}")

        # Store for later use
        self.query_count = dataset_info["Queries"]

    def test_ragas_evaluation(self):
        """Run RAGAS evaluation on AHQAD dataset."""
        # Skip if no .env file or credentials
        if not Path(".env").exists():
            pytest.skip("Skipping RAGAS evaluation: .env file not found")

        # Load environment to check credentials
        from dotenv import load_dotenv

        load_dotenv(Path(".env"))
        use_azure = os.getenv("USE_AZURE", "false").lower() == "true"
        if not use_azure and not os.getenv("OPENAI_API_KEY"):
            pytest.skip("Skipping RAGAS evaluation: No API credentials configured")

        # Print test configuration
        print("\n" + "=" * 80)
        print("RAGAS EVALUATION TEST")
        print("=" * 80)
        print("\n📋 Test Configuration:")
        print("   This test will:")
        print("   • Evaluate Arabic medical Q&A pairs")
        print("   • Use Azure OpenAI deployment")
        print("   • Compute RAGAS metrics:")
        print("     - Faithfulness (claims supported by context)")
        print("     - Answer Relevance (semantic alignment)")
        print("     - Context Precision (retrieval quality)")
        print("     - Context Recall (retrieval coverage)")
        print("\n   Expected:")
        print("   • Duration: ~3-5 minutes")
        print("   • API Calls: ~400 to Azure gpt-5")
        print("   • Cost: ~$1-2 (depends on Azure pricing)")
        print("\n" + "=" * 80 + "\n")

        # Run evaluation with RAGAS enabled
        start_time = time.time()

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "run_rag_eval.py",
                "--data-dir",
                "data/ahqad",
                "--enable-ragas",
                # No --no-timestamp flag - test the actual default behavior
            ],
            capture_output=False,
            text=True,
        )

        duration = time.time() - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        # Check if evaluation succeeded
        assert result.returncode == 0, f"Evaluation failed with return code {result.returncode}"

        # Find the results file (with or without timestamp)
        from glob import glob

        results_pattern = "results/ahqad_test_results*.json"
        results_files = sorted(glob(results_pattern))

        assert results_files, f"No results files found matching: {results_pattern}"

        # Use the most recent file (last in sorted list)
        results_file = Path(results_files[-1])
        assert results_file.exists(), f"Results file not found: {results_file}"

        print("\n" + "=" * 80)
        print("✅ RAGAS Evaluation Complete!")
        print("=" * 80)
        print(f"Duration: {minutes}m {seconds}s")
        print(f"Results saved to: {results_file}")
        print("=" * 80)

    def test_results_file_valid(self, evaluation_results_file):
        """Verify results file is valid JSON with expected structure.

        This test depends on test_ragas_evaluation having run successfully.
        Uses the evaluation_results_file fixture to get the results path.
        """
        import json

        # Skip if no results file exists (evaluation was skipped)
        if evaluation_results_file is None:
            pytest.skip("Skipping: No evaluation results found. Run test_ragas_evaluation first with API credentials.")

        results_file = evaluation_results_file
        assert results_file.exists(), f"Results file not found: {results_file}"

        with open(results_file) as f:
            results = json.load(f)

        # Check for expected fields
        assert "ragas_metrics" in results or "retrieval_metrics" in results, "Results file missing metrics"

        # Print summary
        if "ragas_metrics" in results and results["ragas_metrics"]:
            print("\n📈 RAGAS Metrics Summary:")
            metrics = results["ragas_metrics"]
            if metrics:  # Check metrics is not None
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"   {key}: {value:.4f}")
            else:
                print("   (No RAGAS metrics available - API credentials may be missing)")

        print(f"\n✅ Results validated: {results_file}")
