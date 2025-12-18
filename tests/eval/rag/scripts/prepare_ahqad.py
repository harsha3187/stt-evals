"""Download and convert AHQAD dataset to RAG evaluation format.

This script downloads the AHQAD Arabic Healthcare Q&A dataset from Kaggle
and converts it to the JSONL format required by the RAG evaluation library.

Dataset: https://www.kaggle.com/datasets/abdoashraf90/ahqad-arabic-healthcare-q-and-a-dataset
License: CC BY 4.0

Usage:
    # Download and convert full dataset
    python prepare_ahqad.py

    # Download and convert sample (100 pairs)
    python prepare_ahqad.py --sample 100

    # Specify output directory
    python prepare_ahqad.py --output-dir ../data/ahqad

Requirements:
    - Kaggle API credentials configured (~/.kaggle/kaggle.json)
    - Install: pip install kaggle pandas
    - See: https://github.com/Kaggle/kaggle-api#api-credentials
"""

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import zipfile

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured.

    Returns
    -------
    bool
        True if credentials exist, False otherwise
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.error("Kaggle credentials not found!")
        logger.error("Please set up Kaggle API credentials:")
        logger.error("1. Go to https://www.kaggle.com/settings/account")
        logger.error("2. Click 'Create New API Token'")
        logger.error("3. Save kaggle.json to ~/.kaggle/")
        logger.error("4. chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True


def download_dataset(output_dir: Path):
    """Download AHQAD dataset from Kaggle.

    Parameters
    ----------
    output_dir : Path
        Directory to save the downloaded dataset
    """
    logger.info("Downloading AHQAD dataset from Kaggle...")

    dataset_name = "abdoashraf90/ahqad-arabic-healthcare-q-and-a-dataset"

    try:
        # Download using Kaggle API
        subprocess.run(  # noqa: S603
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(output_dir)],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )

        logger.info(f"✅ Dataset downloaded to {output_dir}")

        # Unzip the dataset
        zip_file = output_dir / "ahqad-arabic-healthcare-q-and-a-dataset.zip"
        if zip_file.exists():
            logger.info("Extracting dataset...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(output_dir)
            logger.info("✅ Dataset extracted")
            zip_file.unlink()  # Remove zip file

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.error(f"Error output: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        sys.exit(1)


def load_ahqad_csv(data_dir: Path) -> pd.DataFrame:
    """Load AHQAD dataset from CSV file.

    Parameters
    ----------
    data_dir : Path
        Directory containing the dataset

    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    # Try common CSV filenames
    possible_names = [
        "ahqad.csv",
        "AHQAD.csv",
        "arabic_healthcare_qa.csv",
        "dataset.csv",
    ]

    csv_file = None
    for name in possible_names:
        candidate = data_dir / name
        if candidate.exists():
            csv_file = candidate
            break

    if csv_file is None:
        # List all CSV files in directory
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            csv_file = csv_files[0]
            logger.warning(f"Using first CSV file found: {csv_file.name}")
        else:
            logger.error(f"No CSV file found in {data_dir}")
            logger.error(f"Directory contents: {list(data_dir.glob('*'))}")
            sys.exit(1)

    logger.info(f"Loading data from {csv_file}")
    df = pd.read_csv(csv_file)
    logger.info(f"✅ Loaded {len(df)} Q&A pairs")
    logger.info(f"Columns: {df.columns.tolist()}")

    return df


def convert_to_jsonl(df: pd.DataFrame, output_dir: Path, sample_size: int = None):  # noqa: C901
    """Convert AHQAD dataset to JSONL format for RAG evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        AHQAD dataset
    output_dir : Path
        Output directory for JSONL files
    sample_size : int, optional
        Number of samples to use (None = use all)
    """
    # Sample if requested
    if sample_size and sample_size < len(df):
        logger.info(f"Sampling {sample_size} random Q&A pairs (seed=42)")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Detect column names (different versions of dataset may have different names)
    question_col = None
    answer_col = None
    category_col = None

    for col in df.columns:
        col_lower = col.lower()
        if "question" in col_lower or "query" in col_lower:
            question_col = col
        elif "answer" in col_lower or "response" in col_lower:
            answer_col = col
        elif "category" in col_lower or "topic" in col_lower:
            category_col = col

    if not question_col or not answer_col:
        logger.error(f"Could not detect question/answer columns in: {df.columns.tolist()}")
        logger.error("Please check the CSV structure")
        sys.exit(1)

    logger.info(f"Using columns: question='{question_col}', answer='{answer_col}', category='{category_col}'")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to queries.jsonl
    queries_file = output_dir / "queries.jsonl"
    queries = []
    for idx, row in df.iterrows():
        query = {
            "id": f"q{idx}",
            "text": str(row[question_col]),
        }
        if category_col and pd.notna(row[category_col]):
            query["category"] = str(row[category_col])
        queries.append(query)

    with open(queries_file, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    logger.info(f"✅ Created {queries_file} ({len(queries)} queries)")

    # Convert to corpus.jsonl (answers as documents)
    corpus_file = output_dir / "corpus.jsonl"
    corpus = []
    for idx, row in df.iterrows():
        doc = {
            "id": f"doc{idx}",
            "text": str(row[answer_col]),
        }
        if category_col and pd.notna(row[category_col]):
            doc["category"] = str(row[category_col])
        corpus.append(doc)

    with open(corpus_file, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info(f"✅ Created {corpus_file} ({len(corpus)} documents)")

    # Create qrels.jsonl (query relevance judgments)
    # Each question maps to its corresponding answer with high relevance
    qrels_file = output_dir / "qrels.jsonl"
    qrels = []
    for idx, _ in df.iterrows():
        qrel = {
            "query_id": f"q{idx}",
            "doc_id": f"doc{idx}",
            "relevance": 2,  # Highly relevant (ground truth)
        }
        qrels.append(qrel)

    with open(qrels_file, "w", encoding="utf-8") as f:
        for qrel in qrels:
            f.write(json.dumps(qrel, ensure_ascii=False) + "\n")
    logger.info(f"✅ Created {qrels_file} ({len(qrels)} relevance judgments)")

    # Create answers.jsonl (generated answers)
    # For testing, we use the ground truth answers
    answers_file = output_dir / "answers.jsonl"
    answers = []
    for idx, row in df.iterrows():
        answer = {
            "query_id": f"q{idx}",
            "answer": str(row[answer_col]),
            "context": [f"doc{idx}"],  # Retrieved context
        }
        answers.append(answer)

    with open(answers_file, "w", encoding="utf-8") as f:
        for ans in answers:
            f.write(json.dumps(ans, ensure_ascii=False) + "\n")
    logger.info(f"✅ Created {answers_file} ({len(answers)} answers)")

    # Create metadata file
    metadata_file = output_dir / "metadata.json"
    metadata = {
        "dataset": "AHQAD",
        "description": "Arabic Healthcare Q&A Dataset",
        "source": "https://www.kaggle.com/datasets/abdoashraf90/ahqad-arabic-healthcare-q-and-a-dataset",
        "license": "CC BY 4.0",
        "language": "ar",
        "total_pairs": len(df),
        "sample_size": sample_size or len(df),
        "categories": df[category_col].nunique() if category_col else None,
        "files": {
            "queries": "queries.jsonl",
            "corpus": "corpus.jsonl",
            "qrels": "qrels.jsonl",
            "answers": "answers.jsonl",
        },
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Created {metadata_file}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Dataset preparation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total Q&A pairs: {len(df)}")
    logger.info("=" * 60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Download and convert AHQAD dataset for RAG evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "ahqad",
        help="Output directory for converted dataset (default: ../data/ahqad)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size (default: use full dataset)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (use existing data)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("AHQAD Dataset Preparation")
    logger.info("=" * 60)

    # Check Kaggle credentials
    if not args.skip_download:
        if not check_kaggle_credentials():
            sys.exit(1)

        # Download dataset
        download_dir = args.output_dir / "raw"
        download_dir.mkdir(parents=True, exist_ok=True)
        download_dataset(download_dir)
    else:
        download_dir = args.output_dir / "raw"
        logger.info(f"Skipping download, using existing data in {download_dir}")

    # Load dataset
    df = load_ahqad_csv(download_dir)

    # Convert to JSONL
    convert_to_jsonl(df, args.output_dir, sample_size=args.sample)

    logger.info("\n📝 Next steps:")
    logger.info("1. Run the RAG evaluation test:")
    logger.info(f"   python test_rag_pipeline.py --data-dir {args.output_dir}")
    logger.info("2. View results in: results/ahqad_test_results.json")


if __name__ == "__main__":
    main()
