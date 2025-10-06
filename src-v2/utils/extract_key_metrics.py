#!/usr/bin/env python3
"""
Extract and display key metrics from WandB run.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  WandB not available. Install with: pip install wandb")


def load_key_metrics_config() -> Dict:
    """Load key metrics configuration."""
    config_path = Path(__file__).parent.parent / "config" / "key_metrics.json"

    if not config_path.exists():
        print(f"❌ Key metrics config not found at: {config_path}")
        return {}

    with open(config_path, 'r') as f:
        return json.load(f)


def extract_latest_metrics(run_path: str, metric_keys: List[str]) -> Dict[str, float]:
    """Extract latest values for specified metrics from WandB run."""
    if not WANDB_AVAILABLE:
        return {}

    try:
        # Initialize WandB API
        api = wandb.Api()
        run = api.run(run_path)

        # Get run history (metrics over time)
        history = run.history()

        if history.empty:
            print(f"❌ No history found for run: {run_path}")
            return {}

        # Extract latest values for each metric
        latest_metrics = {}
        for key in metric_keys:
            if key in history.columns:
                # Get the last non-null value
                values = history[key].dropna()
                if len(values) > 0:
                    latest_metrics[key] = float(values.iloc[-1])
                else:
                    latest_metrics[key] = None
            else:
                latest_metrics[key] = None

        return latest_metrics

    except Exception as e:
        print(f"❌ Error extracting metrics from {run_path}: {e}")
        return {}


def format_metric_value(value: Optional[float]) -> str:
    """Format metric value for display."""
    if value is None:
        return "N/A"
    elif isinstance(value, float):
        if abs(value) < 0.01:
            return f"{value:.6f}"
        elif abs(value) < 0.1:
            return f"{value:.4f}"
        else:
            return f"{value:.3f}"
    else:
        return str(value)


def print_key_metrics(run_path: str):
    """Print key metrics for the specified run."""
    print(f"\n🔍 Extracting Key Metrics from WandB Run")
    print(f"📊 Run: {run_path}")
    print("=" * 80)

    # Load configuration
    config = load_key_metrics_config()
    if not config:
        return

    # Collect all metric keys
    all_metric_keys = []
    for category in config["key_metrics"].values():
        for metric in category:
            all_metric_keys.extend(metric["wandb_keys"])

    # Extract metrics
    latest_metrics = extract_latest_metrics(run_path, all_metric_keys)

    if not latest_metrics:
        print("❌ No metrics extracted")
        return

    # Display metrics by category
    for category_name, category_metrics in config["key_metrics"].items():
        print(f"\n📈 {category_name.replace('_', ' ').title()}")
        print("-" * 60)

        for metric in category_metrics:
            name = metric["name"]
            keys = metric["wandb_keys"]
            description = metric["description"]

            print(f"\n🎯 {name}")
            print(f"   {description}")

            # Display train/val values
            for key in keys:
                value = latest_metrics.get(key)
                formatted_value = format_metric_value(value)

                # Extract split (train/val) from key
                if key.startswith("train/"):
                    split = "Train"
                elif key.startswith("val/"):
                    split = "Val  "
                else:
                    split = "     "

                print(f"   {split}: {formatted_value:>10}")

    print("\n" + "=" * 80)
    print("✅ Key metrics extraction complete!")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python extract_key_metrics.py <wandb_run_path>")
        print("Example: python extract_key_metrics.py username/project/run_id")
        print("Or: python extract_key_metrics.py username/project/run_name")
        return

    run_path = sys.argv[1]
    print_key_metrics(run_path)


if __name__ == "__main__":
    main()
