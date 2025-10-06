#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
"""
Summary script for embedding evaluation results.
Compares different evaluation runs and provides insights.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def load_evaluation_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all evaluation results from directory."""
    results_dir = Path(results_dir)
    results = []

    for json_file in results_dir.glob("evaluation_results_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                data['filename'] = json_file.name
                results.append(data)
        except Exception as e:
            print(f"⚠️  Could not load {json_file}: {e}")

    return results


def create_summary_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create summary table of evaluation results."""

    summary_data = []

    for result in results:
        params = result.get('evaluation_params', {})
        metrics = result.get('metrics_summary', {})

        row = {
            'filename': result['filename'],
            'samples': params.get('max_samples', 'Unknown'),
            'k_neighbors': params.get('k_neighbors', 'Unknown'),
            'l1_accuracy': metrics.get('l1_accuracy', 0),
            'l1_f1_macro': metrics.get('l1_f1_macro', 0),
            'l1_f1_weighted': metrics.get('l1_f1_weighted', 0),
            'l1_classes': metrics.get('l1_classes', 0),
            'l2_accuracy': metrics.get('l2_accuracy', 0),
            'l2_f1_macro': metrics.get('l2_f1_macro', 0),
            'l2_f1_weighted': metrics.get('l2_f1_weighted', 0),
            'l2_classes': metrics.get('l2_classes', 0),
        }
        summary_data.append(row)

    return pd.DataFrame(summary_data)


def print_detailed_analysis(results: List[Dict[str, Any]]):
    """Print detailed analysis of results."""

    print("🔍 DETAILED EVALUATION ANALYSIS")
    print("=" * 60)

    for result in results:
        params = result.get('evaluation_params', {})
        metrics = result.get('metrics_summary', {})

        print(f"\n📊 {result['filename']}")
        print("-" * 40)
        print(f"Configuration:")
        print(f"  Samples: {params.get('max_samples', 'Unknown')}")
        print(f"  K-Neighbors: {params.get('k_neighbors', 'Unknown')}")

        print(f"\nL1 (Primary Activities):")
        print(f"  Classes: {metrics.get('l1_classes', 0)}")
        print(f"  Accuracy: {metrics.get('l1_accuracy', 0):.4f}")
        print(f"  F1-Macro: {metrics.get('l1_f1_macro', 0):.4f}")
        print(f"  F1-Weighted: {metrics.get('l1_f1_weighted', 0):.4f}")

        print(f"\nL2 (Secondary Activities):")
        print(f"  Classes: {metrics.get('l2_classes', 0)}")
        print(f"  Accuracy: {metrics.get('l2_accuracy', 0):.4f}")
        print(f"  F1-Macro: {metrics.get('l2_f1_macro', 0):.4f}")
        print(f"  F1-Weighted: {metrics.get('l2_f1_weighted', 0):.4f}")


def analyze_performance_insights(results: List[Dict[str, Any]]):
    """Analyze performance insights from results."""

    print("\n💡 PERFORMANCE INSIGHTS")
    print("=" * 60)

    if len(results) < 2:
        print("Need at least 2 evaluation runs for comparison.")
        return

    # Compare k=1 vs k=3 performance
    k1_results = [r for r in results if r.get('evaluation_params', {}).get('k_neighbors') == 1]
    k3_results = [r for r in results if r.get('evaluation_params', {}).get('k_neighbors') == 3]

    if k1_results and k3_results:
        print("\n🔄 K-Neighbors Comparison:")
        k1_f1 = k1_results[0].get('metrics_summary', {}).get('l1_f1_macro', 0)
        k3_f1 = k3_results[0].get('metrics_summary', {}).get('l1_f1_macro', 0)

        print(f"  K=1 F1-Macro: {k1_f1:.4f}")
        print(f"  K=3 F1-Macro: {k3_f1:.4f}")
        print(f"  Performance: {'K=1 better' if k1_f1 > k3_f1 else 'K=3 better'}")
        print(f"  Difference: {abs(k1_f1 - k3_f1):.4f}")

    # Compare sample size effects
    sample_sizes = {}
    for result in results:
        samples = result.get('evaluation_params', {}).get('max_samples')
        f1_score = result.get('metrics_summary', {}).get('l1_f1_macro', 0)
        if samples:
            sample_sizes[samples] = f1_score

    if len(sample_sizes) > 1:
        print(f"\n📊 Sample Size Effects:")
        for samples, f1 in sorted(sample_sizes.items()):
            print(f"  {samples:,} samples: F1-Macro = {f1:.4f}")

    # Activity distribution insights
    print(f"\n🏷️  Activity Recognition Insights:")

    # Find best performing run
    best_result = max(results, key=lambda x: x.get('metrics_summary', {}).get('l1_f1_macro', 0))
    best_f1 = best_result.get('metrics_summary', {}).get('l1_f1_macro', 0)

    print(f"  Best F1-Macro (L1): {best_f1:.4f}")
    print(f"  Best Configuration: {best_result.get('evaluation_params', {})}")

    # Performance analysis
    if best_f1 > 0.4:
        print("  ✅ Good performance - embeddings capture activity patterns well")
    elif best_f1 > 0.25:
        print("  ⚠️  Moderate performance - some activity confusion expected")
    else:
        print("  ❌ Low performance - embeddings may need improvement")

    # L1 vs L2 comparison
    l1_f1 = best_result.get('metrics_summary', {}).get('l1_f1_macro', 0)
    l2_f1 = best_result.get('metrics_summary', {}).get('l2_f1_macro', 0)

    print(f"\n🎯 L1 vs L2 Performance:")
    print(f"  L1 (Primary) F1: {l1_f1:.4f}")
    print(f"  L2 (Secondary) F1: {l2_f1:.4f}")
    print(f"  Better: {'L1' if l1_f1 > l2_f1 else 'L2'}")

    if abs(l1_f1 - l2_f1) < 0.05:
        print("  Similar performance between L1 and L2 labels")
    elif l1_f1 > l2_f1:
        print("  L1 labels are easier to distinguish in embedding space")
    else:
        print("  L2 labels provide better semantic clustering")


def main():
    parser = argparse.ArgumentParser(description='Summarize embedding evaluation results')
    parser.add_argument('--results_dir', type=str, default='results/evals/milan/embedding_evaluation',
                       help='Directory containing evaluation results')

    args = parser.parse_args()

    print("📋 EMBEDDING EVALUATION SUMMARY")
    print("=" * 60)

    # Load results
    results = load_evaluation_results(args.results_dir)

    if not results:
        print(f"No evaluation results found in {args.results_dir}")
        return

    print(f"Found {len(results)} evaluation result(s)")

    # Create summary table
    summary_df = create_summary_table(results)

    print(f"\n📊 SUMMARY TABLE")
    print("-" * 60)
    print(summary_df.to_string(index=False, float_format='%.4f'))

    # Detailed analysis
    print_detailed_analysis(results)

    # Performance insights
    analyze_performance_insights(results)

    print(f"\n✅ Analysis complete! Results loaded from: {args.results_dir}")


if __name__ == "__main__":
    main()
