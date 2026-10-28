"""Rewrite custom retrieval short queries into long query lists per household.

For each household in HOUSEHOLDS, calls the Gemini-backed LLMRewriter in
"auto" mode to expand each short query in SHORT_QUERIES into 1-6 retrieval
sentences grounded in that home's sensor inventory.

Updates metadata/custom_retrieval_queries.json:
  - Replaces any existing entries for the processed households.
  - Keeps the `long_queries` field as a list[str] (not a single string).

Usage
-----
  conda activate discover-v2-env
  python src/evals/imwut_paper/rewrite_custom_queries.py [--households milan aruba cairo]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path so `src.query` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.query.llm_rewriter import LLMRewriter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHORT_QUERIES: list[str] = [
    "Kitchen Activity",
    "Eating a Meal",
    "Working at a Desk",
    "Watching TV",
    "Using Bathroom at Night",
    "Using Kitchen at Night",
    "Wandering around the house during the day",
    "Wandering around the house during the night",
    "Sedentary Activity",
    "Sedentary Activity in the Morning",
]

HOUSEHOLDS: list[str] = ["milan", "aruba", "cairo"]

_OUTPUT_PATH = _PROJECT_ROOT / "metadata" / "custom_retrieval_queries.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rewrite_queries_for_home(home: str) -> list[dict]:
    """Return a list of {short_query, long_queries} dicts for one household."""
    print(f"\n{'='*60}")
    print(f"  Processing household: {home.upper()}")
    print(f"{'='*60}")

    rewriter = LLMRewriter(home=home, backend="gemini")

    results: list[dict] = []
    for short_query in SHORT_QUERIES:
        print(f"\n  Query: {short_query!r}")
        reasoning, sentences = rewriter.rewrite(short_query, mode="auto")
        print(f"  -> {len(sentences)} sentence(s) generated")
        for i, s in enumerate(sentences, 1):
            print(f"     [{i}] {s}")
        results.append({
            "short_query": short_query,
            "long_queries": sentences,
        })

    return results


def load_existing(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"\nSaved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--households",
        nargs="+",
        default=HOUSEHOLDS,
        help="Households to process (default: milan aruba cairo)",
    )
    args = parser.parse_args()

    data = load_existing(_OUTPUT_PATH)

    for home in args.households:
        entries = rewrite_queries_for_home(home)
        data[home] = {"queries": entries}

    save(data, _OUTPUT_PATH)
    print("\nDone.")


if __name__ == "__main__":
    main()
