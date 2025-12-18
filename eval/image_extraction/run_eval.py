"""Evaluate image extraction quality across test cases."""

import asyncio
import json
import logging
from pathlib import Path
import re
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from src.backend.core.telemetry import initialize_telemetry
from src.backend.service.search.image_extraction.extract import extract_images
from src.backend.service.search.image_extraction.image_models import ImageSearchMetadata

logger = logging.getLogger(__name__)


def normalize_image_url(url: str) -> str:
    """Normalize image URL by removing dimension-specific parameters and path components.

    This allows matching images that are the same but served at different sizes.

    Examples
    --------
    - /500px-Image.png and /960px-Image.png -> both normalize to /Image.png
    - /thmb/xxx=/750x0/filters:... and /thmb/xxx=/1500x0/filters:... -> both normalize to /thmb/xxx=/filters:...
    """
    # Parse URL
    parsed = urlparse(url)
    path = parsed.path

    # Remove dimension prefixes like /500px-, /960px-, /1800x1200_
    path = re.sub(r"/\d+px-", "/", path)
    path = re.sub(r"/\d+x\d+_", "/", path)

    # For thumbnail URLs (e.g., verywellhealth), normalize the dimension part
    # /thmb/xxx=/750x0/ -> /thmb/xxx=/
    path = re.sub(r"=/\d+x\d+/", "=/", path)

    # Remove query parameters that specify dimensions
    query_params = parse_qs(parsed.query)
    # Keep only non-dimension parameters
    filtered_params = {k: v for k, v in query_params.items() if not any(dim in k.lower() for dim in ["width", "height", "w", "h", "size", "resize"])}
    new_query = urlencode(filtered_params, doseq=True)

    # Reconstruct URL
    normalized = urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, new_query, ""))
    return normalized


def load_test_cases(jsonl_path: Path) -> list[dict]:
    """Load test cases from JSONL file."""
    test_cases = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                test_case = json.loads(line)
                test_cases.append(test_case)
    return test_cases


async def evaluate_test_case(test_case: dict) -> dict:
    """Evaluate a single test case."""
    url = test_case["url"]
    query = test_case["query"]
    should_contain = test_case["should_contain"]
    should_not_contain = test_case["should_not_contain"]
    notes = test_case["notes"]

    logger.info(f"Evaluating: {url}")

    metadata = ImageSearchMetadata()

    try:
        results = await extract_images(url, query, search_metadata=metadata)
        extracted_urls = [img.image_url for img in results]

        # Normalize URLs for comparison (ignore dimension differences)
        normalized_extracted = {normalize_image_url(u): u for u in extracted_urls}
        normalized_expected = {normalize_image_url(u): u for u in should_contain}
        normalized_unwanted = {normalize_image_url(u): u for u in should_not_contain}

        # Check expected images
        found_expected = []
        missing_expected = []
        for norm_url, orig_url in normalized_expected.items():
            if norm_url in normalized_extracted:
                found_expected.append(orig_url)
            else:
                missing_expected.append(orig_url)

        # Check unwanted images
        found_unwanted = []
        for norm_url, orig_url in normalized_unwanted.items():
            if norm_url in normalized_extracted:
                found_unwanted.append(orig_url)

        # Calculate metrics
        total_extracted = len(extracted_urls)
        expected_count = len(should_contain)
        found_count = len(found_expected)
        precision = found_count / total_extracted if total_extracted > 0 else 0
        recall = found_count / expected_count if expected_count > 0 else 1.0  # 1.0 if no expectations

        passed = len(missing_expected) == 0 and len(found_unwanted) == 0

        return {
            "url": url,
            "query": query,
            "notes": notes,
            "total_extracted": total_extracted,
            "found_expected": found_count,
            "missing_expected": len(missing_expected),
            "found_unwanted": len(found_unwanted),
            "precision": precision,
            "recall": recall,
            "passed": passed,
            "missing_urls": missing_expected,
            "unwanted_urls": found_unwanted,
            "extracted_images": results,  # Store full ImageResult objects
        }
    except Exception as e:
        logger.error(f"Error evaluating {url}: {e}")
        return {
            "url": url,
            "query": query,
            "notes": notes,
            "total_extracted": 0,
            "found_expected": 0,
            "missing_expected": len(should_contain),
            "found_unwanted": 0,
            "precision": 0.0,
            "recall": 0.0,
            "passed": False,
            "missing_urls": should_contain,
            "unwanted_urls": [],
            "extracted_images": [],
            "error": str(e),
        }


async def run_evaluation():
    """Run evaluation on all test cases."""
    jsonl_path = Path(__file__).parent / "test_cases.jsonl"
    test_cases = load_test_cases(jsonl_path)

    logger.info(f"Loaded {len(test_cases)} test cases")

    results = []
    for test_case in test_cases:
        result = await evaluate_test_case(test_case)
        results.append(result)

    # Print summary
    print("\n" + "=" * 80)
    print("IMAGE EXTRACTION EVALUATION RESULTS")
    print("=" * 80)

    total_passed = sum(1 for r in results if r["passed"])
    total_cases = len(results)

    for i, result in enumerate(results, 1):
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"\n[{i}/{total_cases}] {status}")
        print(f"URL: {result['url']}")
        print(f"Query: {result['query']}")
        print(f"Extracted: {result['total_extracted']} images")
        print(f"Found expected: {result['found_expected']}")
        print(f"Missing expected: {result['missing_expected']}")
        print(f"Found unwanted: {result['found_unwanted']}")
        print(f"Precision: {result['precision']:.2%}, Recall: {result['recall']:.2%}")
        if result["notes"]:
            print(f"Notes: {result['notes']}")
        if "error" in result:
            print(f"Error: {result['error']}")

        # Print extracted images with details
        if result.get("extracted_images"):
            print(f"\n  Extracted Images ({len(result['extracted_images'])}):")
            for idx, img in enumerate(result["extracted_images"], 1):
                print(f"    {idx}. Relevance: {img.relevance_score:.2f}")
                print(f"       URL: {img.image_url}")
                if img.alt_text:
                    print(f"       Alt: {img.alt_text[:100]}{'...' if len(img.alt_text) > 100 else ''}")

        if result.get("missing_urls"):
            print(f"\n  Missing URLs: {result['missing_urls'][:2]}...")  # Show first 2
        if result.get("unwanted_urls"):
            print(f"  Unwanted URLs: {result['unwanted_urls']}")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {total_passed}/{total_cases} test cases passed ({total_passed / total_cases:.1%})")
    print("=" * 80)

    return results


if __name__ == "__main__":
    initialize_telemetry(None)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    asyncio.run(run_evaluation())
