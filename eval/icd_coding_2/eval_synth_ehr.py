"""
Evaluate ICD-10 coding service on HuggingFace FiscaAI/synth-ehr-icd10cm-prompt dataset.

This script loads synthetic EHR data with ground truth ICD-10 codes,
runs the automatic coding service, and compares predictions against truth.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import sys

from datasets import load_dataset

from src.backend.service.icd_coding_2 import ICDCodingRequest, ICDCodingService


def eval_on_synth_ehr_dataset(num_samples: int = 10, min_confidence: float = 0.0) -> None:
    """
    Evaluate ICD-10 coding service on HuggingFace dataset.

    Parameters
    ----------
    num_samples : int
        Number of samples to evaluate
    min_confidence : float
        Minimum confidence threshold for predictions
    """
    # Create output directory
    output_dir = Path(__file__).parent.parent / "results" / "icd_coding_2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"eval_results_{timestamp}.json"
    summary_file = output_dir / f"eval_summary_{timestamp}.txt"

    print("📦 Loading FiscaAI/synth-ehr-icd10cm-prompt dataset...")

    # Load dataset from HuggingFace
    dataset = load_dataset("FiscaAI/synth-ehr-icd10cm-prompt", split="train")
    print(f"✅ Loaded {len(dataset)} samples\n")

    # Initialize service
    print("🏥 Initializing ICD-10 coding service...")
    service = ICDCodingService()
    print()

    # Evaluate on samples
    results = []
    correct_exact = 0
    correct_any = 0
    total_ground_truth = 0
    total_predicted = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    print(f"🔬 Evaluating on {num_samples} samples...\n")
    print("=" * 80)

    for i, sample in enumerate(dataset.select(range(num_samples))):
        try:
            print(f"\n📋 Sample {i + 1}/{num_samples}")
            print("-" * 80)

            # Extract text and ground truth codes
            clinical_text = sample.get("user", sample.get("text", sample.get("prompt", "")))
            ground_truth_codes = sample.get("codes", [])

            # Parse ground truth codes
            ground_truth = []
            if isinstance(ground_truth_codes, list):
                ground_truth = [str(c).replace(".", "") for c in ground_truth_codes]
            elif isinstance(ground_truth_codes, str):
                # Try to extract ICD codes from string
                codes = re.findall(r"\b[A-TV-Z]\d{2,3}(?:\.\d{1,4})?\b", ground_truth_codes)
                ground_truth = [c.replace(".", "") for c in codes]

            if not ground_truth:
                print("⚠️  No ground truth codes found, skipping...")
                continue

            print(f"Clinical Text: {clinical_text[:150]}...")
            print(f"\n🎯 Ground Truth: {', '.join(ground_truth)}")

            # Get predictions with timing
            start_time = datetime.now()
            request = ICDCodingRequest(text=clinical_text, max_codes=20, top_k_candidates=100, min_confidence=min_confidence)
            result = service.assign_codes(request)
            end_time = datetime.now()
            call_duration_ms = (end_time - start_time).total_seconds() * 1000

            predicted_codes = [code.code for code in result.codes]
            print(f"🤖 Predicted ({len(predicted_codes)}): {', '.join(predicted_codes)}")

            # Calculate metrics
            gt_set = set(ground_truth)
            pred_set = set(predicted_codes)

            exact_match = gt_set == pred_set
            any_match = len(gt_set & pred_set) > 0

            # Calculate per-sample metrics for F1
            true_positives = len(gt_set & pred_set)
            false_positives = len(pred_set - gt_set)
            false_negatives = len(gt_set - pred_set)

            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives

            if exact_match:
                correct_exact += 1
                print("✅ Exact match!")
            elif any_match:
                correct_any += 1
                overlap = gt_set & pred_set
                print(f"✓ Partial match: {', '.join(overlap)}")
            else:
                print("❌ No match")

            # Show confidence scores
            print("\n📊 Predictions with confidence:")
            for code in result.codes[:5]:  # Show top 5
                match_indicator = "✓" if code.code in gt_set else " "
                print(f"  {match_indicator} {code.code}: {code.short_desc} (conf: {code.confidence:.2f})")

            total_ground_truth += len(ground_truth)
            total_predicted += len(predicted_codes)

            results.append(
                {
                    "sample_id": i,
                    "clinical_text": clinical_text[:200] + "..." if len(clinical_text) > 200 else clinical_text,
                    "ground_truth": ground_truth,
                    "predicted": predicted_codes,
                    "predicted_with_scores": [{"code": c.code, "desc": c.short_desc, "confidence": c.confidence} for c in result.codes],
                    "exact_match": exact_match,
                    "any_match": any_match,
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                    "processing_time_ms": result.processing_time_ms,
                    "total_call_time_ms": call_duration_ms,
                    "timestamp": start_time.isoformat(),
                }
            )

            # Save progress every 10 samples
            if (i + 1) % 10 == 0:
                progress_file = output_dir / f"eval_progress_{timestamp}.json"
                with open(progress_file, "w") as f:
                    json.dump({"completed": len(results), "results": results}, f, indent=2)
                print(f"\n💾 Progress saved: {len(results)} samples completed")

        except Exception as e:
            print(f"❌ Error processing sample {i + 1}: {e}")
            continue

    # Print summary
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nSamples evaluated: {len(results)}")
    print(f"Exact matches: {correct_exact} ({correct_exact / len(results) * 100:.1f}%)")
    print(f"Partial matches: {correct_any} ({correct_any / len(results) * 100:.1f}%)")
    print(f"No matches: {len(results) - correct_any} ({(len(results) - correct_any) / len(results) * 100:.1f}%)")
    print(f"\nAvg ground truth codes per sample: {total_ground_truth / len(results):.1f}")
    print(f"Avg predicted codes per sample: {total_predicted / len(results):.1f}")

    # Calculate micro-averaged metrics (across all codes)
    accuracy = correct_exact / len(results) if len(results) > 0 else 0.0
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0.0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n📈 PERFORMANCE METRICS")
    print("-" * 80)
    print(f"Accuracy (exact match):    {accuracy * 100:.2f}%")
    print(f"Precision (micro):         {precision * 100:.2f}%")
    print(f"Recall (micro):            {recall * 100:.2f}%")
    print(f"F1 Score (micro):          {f1_score * 100:.2f}%")

    avg_time = sum(r["processing_time_ms"] for r in results) / len(results)
    avg_call_time = sum(r["total_call_time_ms"] for r in results) / len(results)
    print(f"\nAvg processing time: {avg_time:.0f}ms")
    print(f"Avg total call time: {avg_call_time:.0f}ms")

    # Save detailed results to JSON
    evaluation_data = {
        "metadata": {
            "dataset": "FiscaAI/synth-ehr-icd10cm-prompt",
            "num_samples": num_samples,
            "min_confidence": min_confidence,
            "evaluation_timestamp": timestamp,
            "samples_evaluated": len(results),
        },
        "metrics": {
            "exact_matches": correct_exact,
            "partial_matches": correct_any,
            "no_matches": len(results) - correct_any,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "avg_ground_truth_codes": total_ground_truth / len(results) if len(results) > 0 else 0,
            "avg_predicted_codes": total_predicted / len(results) if len(results) > 0 else 0,
            "avg_processing_time_ms": avg_time,
            "avg_total_call_time_ms": avg_call_time,
        },
        "results": results,
    }

    with open(results_file, "w") as f:
        json.dump(evaluation_data, f, indent=2)
    print(f"\n💾 Detailed results saved to: {results_file}")

    # Save summary report
    separator = "=" * 80
    subseparator = "-" * 80
    total_eval_time = sum(r["total_call_time_ms"] for r in results) / 1000

    summary_content = f"""\
{separator}
ICD-10 CODING EVALUATION SUMMARY
{separator}

Dataset: FiscaAI/synth-ehr-icd10cm-prompt
Evaluation Time: {timestamp}
Samples Requested: {num_samples}
Samples Evaluated: {len(results)}
Min Confidence: {min_confidence}

{subseparator}
PERFORMANCE METRICS
{subseparator}
Accuracy (exact match):    {accuracy * 100:.2f}%
Precision (micro):         {precision * 100:.2f}%
Recall (micro):            {recall * 100:.2f}%
F1 Score (micro):          {f1_score * 100:.2f}%

{subseparator}
MATCH STATISTICS
{subseparator}
Exact matches:             {correct_exact} ({correct_exact / len(results) * 100:.1f}%)
Partial matches:           {correct_any} ({correct_any / len(results) * 100:.1f}%)
No matches:                {len(results) - correct_any} ({(len(results) - correct_any) / len(results) * 100:.1f}%)

{subseparator}
CODE STATISTICS
{subseparator}
Avg ground truth codes:    {total_ground_truth / len(results):.1f}
Avg predicted codes:       {total_predicted / len(results):.1f}

{subseparator}
TIMING
{subseparator}
Avg processing time:       {avg_time:.0f}ms
Avg total call time:       {avg_call_time:.0f}ms
Total evaluation time:     {total_eval_time:.1f}s
"""
    with open(summary_file, "w") as f:
        f.write(summary_content)
    print(f"📄 Summary report saved to: {summary_file}")


if __name__ == "__main__":
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    min_confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    print("🚀 ICD-10 Coding Evaluation")
    print("Dataset: FiscaAI/synth-ehr-icd10cm-prompt")
    print(f"Samples: {num_samples}")
    print(f"Min Confidence: {min_confidence}\n")

    eval_on_synth_ehr_dataset(num_samples, min_confidence)
