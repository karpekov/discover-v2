#!/usr/bin/env python3
"""
Merge multiple classification probing evaluation results into a single report.

Finds results_*.json files in clf_probing folders and compares different models.

Usage:
    python src/evals/merge_clf_probing_evals.py --dataset milan --model-regex ".*_v1"
    python src/evals/merge_clf_probing_evals.py --dataset milan --model-regex ".*seq.*" --output-name seq_models_only
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ClfProbingEvalsMerger:
    """Merge and analyze multiple classification probing evaluation results."""

    def __init__(self, dataset: str, results_base_dir: str = "results/evals"):
        self.dataset = dataset
        self.results_base_dir = Path(results_base_dir)
        self.dataset_dir = self.results_base_dir / dataset

        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.dataset_dir}")

    def find_results_files(self, model_regex: str) -> List[Dict[str, Any]]:
        """Find all results_*.json files in clf_probing folders matching the regex.

        Args:
            model_regex: Regex pattern to match model names
        """
        pattern = re.compile(model_regex)
        results_files = []

        # Search for clf_probing results
        for results_file in self.dataset_dir.rglob("clf_probing/results_*.json"):
            # Extract model name from path
            # Path structure: results/evals/{dataset}/{sampling_strategy}/{model_name}/clf_probing/results_{classifier}_{epochs}_{label_level}.json
            model_name = results_file.parent.parent.name
            sampling_strategy = results_file.parent.parent.parent.name

            # Check if model matches regex
            if pattern.match(model_name):
                # Parse classifier type and other info from filename
                filename = results_file.name
                # Format: results_{classifier}_e{epochs}_l{label_level}.json
                # e.g., results_mlp_e50_ll2.json or results_lin_e50_ll2.json

                classifier_type = None
                epochs = None
                label_level = None

                if '_lin_' in filename or filename.startswith('results_lin_'):
                    classifier_type = 'linear'
                elif '_mlp_' in filename or filename.startswith('results_mlp_'):
                    classifier_type = 'mlp'

                # Extract epochs (e.g., e50)
                epoch_match = re.search(r'_e(\d+)_', filename)
                if epoch_match:
                    epochs = int(epoch_match.group(1))

                # Extract label level (e.g., ll2)
                label_match = re.search(r'_ll(\d+)', filename)
                if label_match:
                    label_level = f"L{label_match.group(1)}"

                results_files.append({
                    'model_name': model_name,
                    'sampling_strategy': sampling_strategy,
                    'file_path': results_file,
                    'classifier_type': classifier_type,
                    'epochs': epochs,
                    'label_level': label_level
                })

        return sorted(results_files, key=lambda x: (x['sampling_strategy'], x['model_name'], x.get('classifier_type', '')))

    def load_results(self, results_files: List[Dict[str, Any]]) -> pd.DataFrame:
        """Load all results files and merge into a DataFrame."""
        all_data = []

        for item in results_files:
            model_name = item['model_name']
            sampling_strategy = item['sampling_strategy']
            file_path = item['file_path']
            classifier_type = item.get('classifier_type', 'unknown')
            epochs = item.get('epochs', 0)
            label_level = item.get('label_level', 'unknown')

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Extract metrics
                metrics = data.get('metrics', {})
                hyperparams = data.get('hyperparameters', {})

                # Get validation metrics (primary)
                val_metrics = metrics.get('validation', {})
                train_metrics = metrics.get('train', {})
                best_metrics = metrics.get('best', {})

                # Use best validation metrics if available, otherwise use validation
                accuracy = best_metrics.get('val_accuracy', val_metrics.get('accuracy', 0))
                f1_weighted = best_metrics.get('val_f1_weighted', val_metrics.get('f1_weighted', 0))
                f1_macro = val_metrics.get('f1_macro', 0)

                # Training metrics
                train_accuracy = train_metrics.get('accuracy', 0)
                train_f1_weighted = train_metrics.get('f1_weighted', 0)
                train_f1_macro = train_metrics.get('f1_macro', 0)

                row = {
                    'model_name': model_name,
                    'sampling_strategy': sampling_strategy,
                    'classifier_type': classifier_type,
                    'epochs': epochs,
                    'label_level': label_level,
                    'val_accuracy': accuracy,
                    'val_f1_weighted': f1_weighted,
                    'val_f1_macro': f1_macro,
                    'train_accuracy': train_accuracy,
                    'train_f1_weighted': train_f1_weighted,
                    'train_f1_macro': train_f1_macro,
                    'num_classes': hyperparams.get('num_classes', 0),
                    'learning_rate': hyperparams.get('learning_rate', 0),
                    'batch_size': hyperparams.get('batch_size', 0),
                    'use_class_weights': hyperparams.get('use_class_weights', False),
                }

                all_data.append(row)

            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not all_data:
            raise ValueError("No valid results found!")

        df = pd.DataFrame(all_data)
        return df

    def create_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a summary table with best metrics for each model."""
        summary_data = []

        # Group by model and sampling strategy
        for (model_name, sampling_strategy), group in df.groupby(['model_name', 'sampling_strategy']):
            # Get both linear and MLP results if available
            linear_results = group[group['classifier_type'] == 'linear']
            mlp_results = group[group['classifier_type'] == 'mlp']

            row = {
                'Model': model_name,
                'Sampling': sampling_strategy,
            }

            # Linear classifier metrics
            if len(linear_results) > 0:
                lin_row = linear_results.iloc[0]
                row['Linear_Val_Accuracy'] = lin_row['val_accuracy']
                row['Linear_Val_F1_Weighted'] = lin_row['val_f1_weighted']
                row['Linear_Val_F1_Macro'] = lin_row['val_f1_macro']
            else:
                row['Linear_Val_Accuracy'] = np.nan
                row['Linear_Val_F1_Weighted'] = np.nan
                row['Linear_Val_F1_Macro'] = np.nan

            # MLP classifier metrics
            if len(mlp_results) > 0:
                mlp_row = mlp_results.iloc[0]
                row['MLP_Val_Accuracy'] = mlp_row['val_accuracy']
                row['MLP_Val_F1_Weighted'] = mlp_row['val_f1_weighted']
                row['MLP_Val_F1_Macro'] = mlp_row['val_f1_macro']
            else:
                row['MLP_Val_Accuracy'] = np.nan
                row['MLP_Val_F1_Weighted'] = np.nan
                row['MLP_Val_F1_Macro'] = np.nan

            # Best overall (max of linear and MLP)
            best_acc = max(
                row.get('Linear_Val_Accuracy', 0) if not pd.isna(row.get('Linear_Val_Accuracy')) else 0,
                row.get('MLP_Val_Accuracy', 0) if not pd.isna(row.get('MLP_Val_Accuracy')) else 0
            )
            best_f1w = max(
                row.get('Linear_Val_F1_Weighted', 0) if not pd.isna(row.get('Linear_Val_F1_Weighted')) else 0,
                row.get('MLP_Val_F1_Weighted', 0) if not pd.isna(row.get('MLP_Val_F1_Weighted')) else 0
            )
            row['Best_Val_Accuracy'] = best_acc if best_acc > 0 else np.nan
            row['Best_Val_F1_Weighted'] = best_f1w if best_f1w > 0 else np.nan

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)

        # Sort by Best Val F1 Weighted (descending)
        if len(summary_df) > 0 and 'Best_Val_F1_Weighted' in summary_df.columns:
            summary_df = summary_df.sort_values('Best_Val_F1_Weighted', ascending=False, na_position='last')

        return summary_df

    def create_markdown_report(self, df: pd.DataFrame, summary_df: pd.DataFrame,
                               output_path: Path):
        """Create a markdown report with tables."""
        with open(output_path, 'w') as f:
            f.write(f"# Classification Probing Evaluation Results - {self.dataset.upper()}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Models:** {len(summary_df)}\n\n")

            # Summary table
            f.write("## Summary - Classification Probing Performance\n\n")
            f.write("Models sorted by Best Validation F1-Weighted score (primary metric).\n\n")

            # Format summary table
            summary_formatted = summary_df.copy()
            metric_cols = [col for col in summary_formatted.columns if col not in ['Model', 'Sampling']]
            for col in metric_cols:
                summary_formatted[col] = summary_formatted[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )

            # Write header
            f.write("| " + " | ".join(summary_formatted.columns) + " |\n")
            f.write("| " + " | ".join(["---"] * len(summary_formatted.columns)) + " |\n")
            # Write rows
            for _, row in summary_formatted.iterrows():
                f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
            f.write("\n\n")

            # Top 5 models
            f.write("## 🏆 Top 5 Models (Best Validation F1-Weighted)\n\n")
            if 'Best_Val_F1_Weighted' in summary_df.columns:
                top5 = summary_df.head(5)[['Model', 'Sampling', 'Best_Val_F1_Weighted', 'Best_Val_Accuracy']]
                top5_formatted = top5.copy()
                for col in ['Best_Val_F1_Weighted', 'Best_Val_Accuracy']:
                    top5_formatted[col] = top5_formatted[col].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                    )

                f.write("| " + " | ".join(top5_formatted.columns) + " |\n")
                f.write("| " + " | ".join(["---"] * len(top5_formatted.columns)) + " |\n")
                for _, row in top5_formatted.iterrows():
                    f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
                f.write("\n\n")

            # Linear vs MLP comparison
            f.write("## Linear vs MLP Classifier Comparison\n\n")

            linear_df = df[df['classifier_type'] == 'linear']
            mlp_df = df[df['classifier_type'] == 'mlp']

            if len(linear_df) > 0 and len(mlp_df) > 0:
                comparison_data = []
                for (model_name, sampling_strategy), group in df.groupby(['model_name', 'sampling_strategy']):
                    lin_group = group[group['classifier_type'] == 'linear']
                    mlp_group = group[group['classifier_type'] == 'mlp']

                    if len(lin_group) > 0 and len(mlp_group) > 0:
                        lin_row = lin_group.iloc[0]
                        mlp_row = mlp_group.iloc[0]

                        comparison_data.append({
                            'Model': model_name,
                            'Sampling': sampling_strategy,
                            'Linear_F1_Weighted': lin_row['val_f1_weighted'],
                            'MLP_F1_Weighted': mlp_row['val_f1_weighted'],
                            'Difference': mlp_row['val_f1_weighted'] - lin_row['val_f1_weighted'],
                            'Best': 'MLP' if mlp_row['val_f1_weighted'] > lin_row['val_f1_weighted'] else 'Linear'
                        })

                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    comp_df = comp_df.sort_values('Difference', ascending=False)

                    comp_formatted = comp_df.copy()
                    for col in ['Linear_F1_Weighted', 'MLP_F1_Weighted', 'Difference']:
                        comp_formatted[col] = comp_formatted[col].apply(lambda x: f"{x:.4f}")

                    f.write("| " + " | ".join(comp_formatted.columns) + " |\n")
                    f.write("| " + " | ".join(["---"] * len(comp_formatted.columns)) + " |\n")
                    for _, row in comp_formatted.iterrows():
                        f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
                    f.write("\n\n")

            # Metrics legend
            f.write("## Metrics Legend\n\n")
            f.write("- **Val_Accuracy**: Validation accuracy (best checkpoint)\n")
            f.write("- **Val_F1_Weighted**: Validation F1 score weighted by support (best checkpoint)\n")
            f.write("- **Val_F1_Macro**: Validation F1 score macro average\n")
            f.write("- **Linear**: Simple linear classifier (single layer)\n")
            f.write("- **MLP**: Multi-layer perceptron classifier\n")
            f.write("- **Best**: Best performance across Linear and MLP classifiers\n\n")

    def create_comparison_charts(self, df: pd.DataFrame, summary_df: pd.DataFrame,
                                 output_dir: Path):
        """Create comparison charts for classification probing metrics."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Linear vs MLP comparison (bar chart)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Group by classifier type
        linear_df = df[df['classifier_type'] == 'linear']
        mlp_df = df[df['classifier_type'] == 'mlp']

        if len(linear_df) > 0 and len(mlp_df) > 0:
            # Accuracy comparison
            ax1 = axes[0]
            x = np.arange(1)
            width = 0.35

            linear_acc_mean = linear_df['val_accuracy'].mean()
            linear_acc_std = linear_df['val_accuracy'].std()
            mlp_acc_mean = mlp_df['val_accuracy'].mean()
            mlp_acc_std = mlp_df['val_accuracy'].std()

            ax1.bar(x - width/2, [linear_acc_mean], width, yerr=[linear_acc_std],
                   label='Linear', alpha=0.8, capsize=5)
            ax1.bar(x + width/2, [mlp_acc_mean], width, yerr=[mlp_acc_std],
                   label='MLP', alpha=0.8, capsize=5)

            ax1.set_ylabel('Validation Accuracy', fontsize=11)
            ax1.set_title('Average Validation Accuracy: Linear vs MLP', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels([''])
            ax1.legend(fontsize=9)
            ax1.grid(axis='y', alpha=0.3)
            ax1.set_ylim([0, 100])

            # F1 Weighted comparison
            ax2 = axes[1]
            linear_f1_mean = linear_df['val_f1_weighted'].mean()
            linear_f1_std = linear_df['val_f1_weighted'].std()
            mlp_f1_mean = mlp_df['val_f1_weighted'].mean()
            mlp_f1_std = mlp_df['val_f1_weighted'].std()

            ax2.bar(x - width/2, [linear_f1_mean], width, yerr=[linear_f1_std],
                   label='Linear', alpha=0.8, capsize=5)
            ax2.bar(x + width/2, [mlp_f1_mean], width, yerr=[mlp_f1_std],
                   label='MLP', alpha=0.8, capsize=5)

            ax2.set_ylabel('Validation F1-Weighted', fontsize=11)
            ax2.set_title('Average Validation F1-Weighted: Linear vs MLP', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([''])
            ax2.legend(fontsize=9)
            ax2.grid(axis='y', alpha=0.3)
            ax2.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_dir / 'linear_vs_mlp_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Chart saved: {output_dir / 'linear_vs_mlp_comparison.png'}")

        # 2. Top models comparison
        if len(summary_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            top_n = min(15, len(summary_df))
            top_models = summary_df.head(top_n)

            # Accuracy
            ax1 = axes[0]
            x = np.arange(len(top_models))
            width = 0.35

            linear_acc = top_models['Linear_Val_Accuracy'].fillna(0)
            mlp_acc = top_models['MLP_Val_Accuracy'].fillna(0)

            ax1.bar(x - width/2, linear_acc, width, label='Linear', alpha=0.8)
            ax1.bar(x + width/2, mlp_acc, width, label='MLP', alpha=0.8)

            ax1.set_xlabel('Model', fontsize=11)
            ax1.set_ylabel('Validation Accuracy', fontsize=11)
            ax1.set_title('Top Models - Validation Accuracy', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"{row['Model'][:20]}..." if len(row['Model']) > 20 else row['Model']
                                for _, row in top_models.iterrows()],
                               rotation=45, ha='right', fontsize=8)
            ax1.legend(fontsize=9)
            ax1.grid(axis='y', alpha=0.3)
            ax1.set_ylim([0, 100])

            # F1 Weighted
            ax2 = axes[1]
            linear_f1 = top_models['Linear_Val_F1_Weighted'].fillna(0)
            mlp_f1 = top_models['MLP_Val_F1_Weighted'].fillna(0)

            ax2.bar(x - width/2, linear_f1, width, label='Linear', alpha=0.8)
            ax2.bar(x + width/2, mlp_f1, width, label='MLP', alpha=0.8)

            ax2.set_xlabel('Model', fontsize=11)
            ax2.set_ylabel('Validation F1-Weighted', fontsize=11)
            ax2.set_title('Top Models - Validation F1-Weighted', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"{row['Model'][:20]}..." if len(row['Model']) > 20 else row['Model']
                                for _, row in top_models.iterrows()],
                               rotation=45, ha='right', fontsize=8)
            ax2.legend(fontsize=9)
            ax2.grid(axis='y', alpha=0.3)
            ax2.set_ylim([0, 1])

            plt.tight_layout()
            plt.savefig(output_dir / 'top_models_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Chart saved: {output_dir / 'top_models_comparison.png'}")

        # 3. Heatmap: All models, all metrics
        if len(summary_df) > 0:
            fig, ax = plt.subplots(figsize=(14, max(8, len(summary_df) * 0.3)))

            metric_cols = ['Linear_Val_Accuracy', 'Linear_Val_F1_Weighted', 'MLP_Val_Accuracy', 'MLP_Val_F1_Weighted']
            available_cols = [col for col in metric_cols if col in summary_df.columns]

            if available_cols:
                heatmap_data = summary_df[['Model'] + available_cols].set_index('Model')
                heatmap_data = heatmap_data.fillna(0)

                # Truncate model names
                heatmap_data.index = [name[:40] + '...' if len(name) > 40 else name
                                     for name in heatmap_data.index]

                # Create heatmap
                data_min = heatmap_data.min().min()
                data_max = heatmap_data.max().max()
                data_center = (data_min + data_max) / 2

                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                           ax=ax, cbar_kws={'label': 'Score'},
                           vmin=data_min, vmax=data_max, linewidths=0.5, center=data_center)
                ax.set_title('Classification Probing Metrics Heatmap', fontsize=12, fontweight='bold')
                ax.set_xlabel('Metric', fontsize=11)
                ax.set_ylabel('Model', fontsize=11)

                plt.tight_layout()
                plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Chart saved: {output_dir / 'metrics_heatmap.png'}")

        # 4. Sampling strategy comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, metric in enumerate(['val_accuracy', 'val_f1_weighted']):
            ax = axes[idx]

            # Group by sampling strategy and classifier type
            sampling_stats = df.groupby(['sampling_strategy', 'classifier_type'])[metric].agg(['mean', 'std', 'count'])

            sampling_strategies = sorted(df['sampling_strategy'].unique())
            x = np.arange(len(sampling_strategies))
            width = 0.35

            for classifier_type in ['linear', 'mlp']:
                means = []
                stds = []
                for sampling in sampling_strategies:
                    if (sampling, classifier_type) in sampling_stats.index:
                        stats = sampling_stats.loc[(sampling, classifier_type)]
                        means.append(stats['mean'])
                        stds.append(stats['std'])
                    else:
                        means.append(0)
                        stds.append(0)

                offset = -width/2 if classifier_type == 'linear' else width/2
                label = 'Linear' if classifier_type == 'linear' else 'MLP'
                ax.bar(x + offset, means, width, yerr=stds,
                      label=label, capsize=5, alpha=0.7)

            ax.set_xlabel('Sampling Strategy', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'Average {metric.replace("_", " ").title()} by Sampling Strategy', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(sampling_strategies, rotation=45, ha='right')
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            if metric == 'val_accuracy':
                ax.set_ylim([0, 100])
            else:
                ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_dir / 'sampling_strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Chart saved: {output_dir / 'sampling_strategy_comparison.png'}")

    def run(self, model_regex: str, output_name: str = None):
        """Run the complete merging pipeline.

        Args:
            model_regex: Regex pattern to match model names
            output_name: Output directory name (default: auto-generated)
        """
        print("="*80)
        print("CLASSIFICATION PROBING EVALUATIONS MERGER")
        print("="*80)
        print(f"Dataset: {self.dataset}")
        print(f"Model regex: {model_regex}")
        print("")

        # Find results files
        print("🔍 Finding results files...")
        results_files = self.find_results_files(model_regex)

        if not results_files:
            print("❌ No results files found matching the regex!")
            return

        print(f"✅ Found {len(results_files)} result files:")
        for item in results_files:
            classifier_str = item.get('classifier_type', 'unknown')
            print(f"   - {item['sampling_strategy']}/{item['model_name']} ({classifier_str})")
        print("")

        # Load results
        print("📊 Loading and merging results...")
        df = self.load_results(results_files)
        print(f"✅ Loaded {len(df)} result entries")
        print("")

        # Create summary
        print("📋 Creating summary table...")
        summary_df = self.create_summary_table(df)
        print(f"✅ Summary created with {len(summary_df)} models")
        print("")

        # Create output directory
        if output_name is None:
            output_name = f"clf_probing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        output_dir = self.results_base_dir / self.dataset / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save merged data
        print("💾 Saving merged data...")
        df.to_csv(output_dir / 'detailed_results.csv', index=False)
        summary_df.to_csv(output_dir / 'summary_results.csv', index=False)
        df.to_json(output_dir / 'detailed_results.json', orient='records', indent=2)
        summary_df.to_json(output_dir / 'summary_results.json', orient='records', indent=2)
        print("✅ Saved results")
        print("")

        # Create markdown report
        print("📝 Creating markdown report...")
        self.create_markdown_report(df, summary_df, output_dir / 'RESULTS_REPORT.md')
        print("")

        # Create charts
        print("📊 Creating comparison charts...")
        charts_dir = output_dir / 'charts'
        self.create_comparison_charts(df, summary_df, charts_dir)
        print("")

        # Print final summary
        print("="*80)
        print("✅ MERGE COMPLETE!")
        print("="*80)
        print(f"Output directory: {output_dir}")
        print(f"Total models: {len(summary_df)}")
        print("")
        print("📁 Files created:")
        print(f"  - detailed_results.csv / .json")
        print(f"  - summary_results.csv / .json")
        print(f"  - RESULTS_REPORT.md")
        print(f"  - charts/ (4 comparison visualizations)")
        print("")
        print("Top 3 Models (Best Validation F1-Weighted):")
        if 'Best_Val_F1_Weighted' in summary_df.columns:
            for idx, row in summary_df.head(3).iterrows():
                print(f"  {idx+1}. {row['Model']}")
                print(f"     Sampling: {row['Sampling']}")
                print(f"     Best F1-Weighted: {row['Best_Val_F1_Weighted']:.4f}")
                print(f"     Best Accuracy: {row['Best_Val_Accuracy']:.4f}")
                print("")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple classification probing evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all v1 models for Milan
  python src/evals/merge_clf_probing_evals.py --dataset milan --model-regex ".*_v1"

  # Merge only sequence models
  python src/evals/merge_clf_probing_evals.py --dataset milan --model-regex ".*seq.*"

  # Merge with custom output name
  python src/evals/merge_clf_probing_evals.py --dataset milan --model-regex ".*_v1" --output-name all_v1_models
        """
    )

    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., milan, aruba)')
    parser.add_argument('--model-regex', type=str, required=True,
                       help='Regular expression to match model names')
    parser.add_argument('--output-name', type=str, default=None,
                       help='Output directory name (default: auto-generated with timestamp)')
    parser.add_argument('--results-dir', type=str, default='results/evals',
                       help='Base results directory (default: results/evals)')

    args = parser.parse_args()

    # Run merger
    merger = ClfProbingEvalsMerger(args.dataset, args.results_dir)
    merger.run(args.model_regex, args.output_name)


if __name__ == '__main__':
    main()

