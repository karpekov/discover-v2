#!/usr/bin/env python3
"""
Merge multiple comprehensive_results.json files into a single report.

Usage:
    python src/evals/merge_embedding_evals.py --dataset milan --model-regex ".*_v1"
    python src/evals/merge_embedding_evals.py --dataset milan --model-regex "fd60.*seq.*"
    python src/evals/merge_embedding_evals.py --dataset milan --model-regex ".*img.*" --output-name image_models
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


class EmbeddingEvalsMerger:
    """Merge and analyze multiple embedding evaluation results."""

    def __init__(self, dataset: str, results_base_dir: str = "results/evals"):
        self.dataset = dataset
        self.results_base_dir = Path(results_base_dir)
        self.dataset_dir = self.results_base_dir / dataset

        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.dataset_dir}")

    def find_results_files(self, model_regex: str) -> List[Dict[str, Any]]:
        """Find all comprehensive_results.json files matching the regex."""
        pattern = re.compile(model_regex)
        results_files = []

        # Search through all subdirectories
        for results_file in self.dataset_dir.rglob("comprehensive_results.json"):
            # Extract model name from path
            # Path structure: results/evals/{dataset}/{sampling_strategy}/{model_name}/comprehensive_results.json
            model_name = results_file.parent.name
            sampling_strategy = results_file.parent.parent.name

            # Check if model matches regex
            if pattern.match(model_name):
                results_files.append({
                    'model_name': model_name,
                    'sampling_strategy': sampling_strategy,
                    'file_path': results_file
                })

        return sorted(results_files, key=lambda x: (x['sampling_strategy'], x['model_name']))

    def load_results(self, results_files: List[Dict[str, Any]]) -> pd.DataFrame:
        """Load all results files and merge into a DataFrame."""
        all_data = []

        for item in results_files:
            model_name = item['model_name']
            sampling_strategy = item['sampling_strategy']
            file_path = item['file_path']

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Extract key metrics for each configuration
                # JSON structure: {"text_noproj": {"l1": {...}, "l2": {...}}, ...}
                model_types = [
                    ('text_noproj', 'Text (No Proj)'),
                    ('text_proj', 'Text (Proj)'),
                    ('sensor', 'Sensor'),
                ]

                levels = ['l1', 'l2']

                for model_type_key, display_name in model_types:
                    if model_type_key in data:
                        for level in levels:
                            if level in data[model_type_key]:
                                metrics = data[model_type_key][level]

                                row = {
                                    'model_name': model_name,
                                    'sampling_strategy': sampling_strategy,
                                    'label_level': level.upper(),
                                    'model_type': display_name,
                                    'accuracy': metrics.get('accuracy', 0),
                                    'f1_macro': metrics.get('f1_macro', 0),
                                    'f1_weighted': metrics.get('f1_weighted', 0),
                                    'precision_macro': metrics.get('precision_macro', 0),
                                    'recall_macro': metrics.get('recall_macro', 0),
                                    'num_classes': metrics.get('num_classes', 0),
                                    'num_samples': metrics.get('num_samples', 0),
                                }

                                all_data.append(row)

            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
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
            # Get L1 sensor metrics (primary metric of interest)
            l1_sensor = group[(group['label_level'] == 'L1') & (group['model_type'] == 'Sensor')]
            l2_sensor = group[(group['label_level'] == 'L2') & (group['model_type'] == 'Sensor')]

            if len(l1_sensor) > 0 and len(l2_sensor) > 0:
                l1_row = l1_sensor.iloc[0]
                l2_row = l2_sensor.iloc[0]

                summary_data.append({
                    'Model': model_name,
                    'Sampling': sampling_strategy,
                    'L1_F1_Weighted': l1_row['f1_weighted'],
                    'L1_F1_Macro': l1_row['f1_macro'],
                    'L1_Accuracy': l1_row['accuracy'],
                    'L2_F1_Weighted': l2_row['f1_weighted'],
                    'L2_F1_Macro': l2_row['f1_macro'],
                    'L2_Accuracy': l2_row['accuracy'],
                    'L1_Classes': l1_row['num_classes'],
                    'L2_Classes': l2_row['num_classes'],
                    'Samples': l1_row['num_samples'],
                })

        summary_df = pd.DataFrame(summary_data)

        # Sort by L1 F1 Weighted (descending)
        if len(summary_df) > 0:
            summary_df = summary_df.sort_values('L1_F1_Weighted', ascending=False)

        return summary_df

    def create_markdown_report(self, df: pd.DataFrame, summary_df: pd.DataFrame,
                               output_path: Path):
        """Create a markdown report with tables."""
        with open(output_path, 'w') as f:
            f.write(f"# Embedding Evaluation Results - {self.dataset.upper()}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Models:** {len(summary_df)}\n\n")

            # Summary table
            f.write("## Summary - Sensor Embeddings Performance\n\n")
            f.write("Models sorted by L1 F1-Weighted score (primary metric).\n\n")

            # Format summary table (manual markdown generation)
            summary_formatted = summary_df.copy()
            for col in ['L1_F1_Weighted', 'L1_F1_Macro', 'L1_Accuracy',
                       'L2_F1_Weighted', 'L2_F1_Macro', 'L2_Accuracy']:
                summary_formatted[col] = summary_formatted[col].apply(lambda x: f"{x:.4f}")

            # Write header
            f.write("| " + " | ".join(summary_formatted.columns) + " |\n")
            f.write("| " + " | ".join(["---"] * len(summary_formatted.columns)) + " |\n")
            # Write rows
            for _, row in summary_formatted.iterrows():
                f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
            f.write("\n\n")

            # Top 5 models
            f.write("## 🏆 Top 5 Models (L1 F1-Weighted)\n\n")
            top5 = summary_df.head(5)[['Model', 'Sampling', 'L1_F1_Weighted', 'L1_Accuracy']]
            top5_formatted = top5.copy()
            top5_formatted['L1_F1_Weighted'] = top5_formatted['L1_F1_Weighted'].apply(lambda x: f"{x:.4f}")
            top5_formatted['L1_Accuracy'] = top5_formatted['L1_Accuracy'].apply(lambda x: f"{x:.4f}")

            # Write header
            f.write("| " + " | ".join(top5_formatted.columns) + " |\n")
            f.write("| " + " | ".join(["---"] * len(top5_formatted.columns)) + " |\n")
            # Write rows
            for _, row in top5_formatted.iterrows():
                f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
            f.write("\n\n")

            # Detailed breakdown by label level
            f.write("## Detailed Results by Label Level\n\n")

            for level in ['L1', 'L2']:
                f.write(f"### {level} - Primary Activities\n\n" if level == 'L1' else f"### {level} - Secondary Activities\n\n")

                level_df = df[df['label_level'] == level].copy()

                # Pivot for better comparison
                pivot_df = level_df.pivot_table(
                    index=['model_name', 'sampling_strategy'],
                    columns='model_type',
                    values='f1_weighted',
                    aggfunc='first'
                ).reset_index()

                pivot_df.columns.name = None
                pivot_df = pivot_df.sort_values('Sensor', ascending=False)

                # Format
                for col in pivot_df.columns:
                    if col not in ['model_name', 'sampling_strategy']:
                        pivot_df[col] = pivot_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

                pivot_df = pivot_df.rename(columns={'model_name': 'Model', 'sampling_strategy': 'Sampling'})

                # Write header
                f.write("| " + " | ".join(pivot_df.columns) + " |\n")
                f.write("| " + " | ".join(["---"] * len(pivot_df.columns)) + " |\n")
                # Write rows
                for _, row in pivot_df.iterrows():
                    f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
                f.write("\n\n")

            # Metrics legend
            f.write("## Metrics Legend\n\n")
            f.write("- **F1-Weighted**: F1 score weighted by support (accounts for class imbalance)\n")
            f.write("- **F1-Macro**: Unweighted average F1 score across all classes\n")
            f.write("- **Accuracy**: Overall classification accuracy\n")
            f.write("- **Text (No Proj)**: Text embeddings without projection (baseline)\n")
            f.write("- **Text (Proj)**: Text embeddings with learned projection\n")
            f.write("- **Sensor**: Sensor embeddings (primary model output)\n\n")

    def add_comprehensive_tables_to_report(self, output_dir: Path):
        """Add comprehensive tables to the markdown report."""
        report_path = output_dir / 'RESULTS_REPORT.md'

        # Read existing report
        with open(report_path, 'r') as f:
            content = f.read()

        # Add comprehensive tables section
        with open(report_path, 'a') as f:
            f.write("\n---\n\n")
            f.write("# Comprehensive Comparison Tables\n\n")
            f.write("These tables compare all model configurations across sampling strategies.\n\n")
            f.write("**Structure:**\n")
            f.write("- **Columns**: FL_20, FL_50, FD_60, FD_120 (each with 3 sub-columns)\n")
            f.write("  - `Sensor`: Sensor embedding performance\n")
            f.write("  - `Text`: Text-only embedding (no projection)\n")
            f.write("  - `Text+Proj`: Text embedding with projection\n")
            f.write("- **Rows**: Model configurations\n")
            f.write("  - Sequence vs Image encoders\n")
            f.write("  - Linear vs MLP projection\n")
            f.write("  - CLIP vs CLIP+MLM loss\n\n")

            # Add each comprehensive table
            for level in ['L1', 'L2']:
                for metric in ['f1_weighted', 'f1_macro']:
                    metric_name = metric.replace('_', ' ').title()
                    csv_file = output_dir / f'comprehensive_table_{level}_{metric}.csv'
                    heatmap_file = output_dir / f'comprehensive_heatmap_{level}_{metric}.png'

                    if csv_file.exists():
                        f.write(f"## {level} {metric_name}\n\n")

                        # Add heatmap reference
                        if heatmap_file.exists():
                            f.write(f"![{level} {metric_name} Heatmap](comprehensive_heatmap_{level}_{metric}.png)\n\n")

                        # Read and format the CSV
                        comp_df = pd.read_csv(csv_file)

                        # Format numbers
                        for col in comp_df.columns:
                            if col != 'Configuration':
                                comp_df[col] = comp_df[col].apply(
                                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                                )

                        # Write table header
                        f.write("| " + " | ".join(comp_df.columns) + " |\n")
                        f.write("| " + " | ".join(["---"] * len(comp_df.columns)) + " |\n")

                        # Write rows
                        for _, row in comp_df.iterrows():
                            f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")

                        f.write("\n")

                        # Add key insights
                        f.write("**Key Insights:**\n")

                        # Find best configuration overall
                        numeric_cols = [col for col in comp_df.columns if col != 'Configuration']
                        for col in numeric_cols:
                            comp_df[col] = pd.to_numeric(comp_df[col].replace('N/A', np.nan), errors='coerce')

                        # Calculate mean performance per configuration
                        comp_df['mean_score'] = comp_df[numeric_cols].mean(axis=1, skipna=True)
                        best_config_idx = comp_df['mean_score'].idxmax()
                        best_config = comp_df.loc[best_config_idx, 'Configuration']
                        best_score = comp_df.loc[best_config_idx, 'mean_score']

                        f.write(f"- Best overall configuration: **{best_config}** (avg: {best_score:.4f})\n")

                        # Find best per sampling strategy
                        sensor_cols = [col for col in numeric_cols if 'Sensor' in col]
                        for col in sensor_cols:
                            sampling = col.replace('_Sensor', '')
                            if comp_df[col].notna().any():  # Check if there are any non-NaN values
                                best_idx = comp_df[col].idxmax(skipna=True)
                                if pd.notna(best_idx):
                                    best_conf = comp_df.loc[best_idx, 'Configuration']
                                    best_val = comp_df.loc[best_idx, col]
                                    f.write(f"- Best for {sampling}: **{best_conf}** ({best_val:.4f})\n")

                        f.write("\n")

            # Add charts section
            f.write("\n---\n\n")
            f.write("# Comparison Charts\n\n")
            f.write("## Model Architecture Comparisons\n\n")

            charts = [
                ("projection_comparison.png", "Linear vs MLP Projection"),
                ("encoder_type_comparison.png", "Sequence vs Image Encoder"),
                ("embedding_type_comparison.png", "Embedding Type Comparison"),
                ("sampling_strategy_comparison.png", "Sampling Strategy Performance"),
            ]

            for chart_file, title in charts:
                chart_path = output_dir / 'charts' / chart_file
                if chart_path.exists():
                    f.write(f"### {title}\n\n")
                    f.write(f"![{title}](charts/{chart_file})\n\n")

        print(f"✅ Comprehensive tables added to markdown report")

    def create_comprehensive_tables(self, df: pd.DataFrame, output_dir: Path):
        """Create comprehensive comparison tables as requested."""

        # Define the structure
        sampling_strategies = ['FL_20', 'FL_50', 'FD_60', 'FD_120']

        # Create tables for L1 (both F1-weighted and F1-macro)
        for level in ['L1', 'L2']:
            for metric in ['f1_weighted', 'f1_macro']:
                # Initialize the comprehensive table
                rows_data = []

                # Define row structure
                row_configs = [
                    ('Seq - Linear - CLIP', '_seq_', 'projlin', '_clip_v1'),
                    ('Seq - Linear - CLIP+MLM', '_seq_', 'projlin', 'clipmlm_v1'),
                    ('Seq - MLP - CLIP', '_seq_', 'projmlp', '_clip_v1'),
                    ('Seq - MLP - CLIP+MLM', '_seq_', 'projmlp', 'clipmlm_v1'),
                    ('Img - Linear - CLIP', '_img_', 'projlin', '_v1'),
                    ('Img - MLP - CLIP', '_img_', 'projmlp', '_v1'),
                ]

                for row_name, encoder_type, proj_type, loss_suffix in row_configs:
                    row = {'Configuration': row_name}

                    for sampling in sampling_strategies:
                        # Find matching models
                        matching_df = df[
                            (df['label_level'] == level) &
                            (df['sampling_strategy'].str.contains(sampling)) &
                            (df['model_name'].str.contains(encoder_type)) &
                            (df['model_name'].str.contains(proj_type)) &
                            (df['model_name'].str.endswith(loss_suffix))
                        ]

                        if len(matching_df) > 0:
                            # Get sensor, text_noproj, text_proj values
                            sensor_val = matching_df[matching_df['model_type'] == 'Sensor'][metric].values
                            text_noproj_val = matching_df[matching_df['model_type'] == 'Text (No Proj)'][metric].values
                            text_proj_val = matching_df[matching_df['model_type'] == 'Text (Proj)'][metric].values

                            row[f'{sampling}_Sensor'] = sensor_val[0] if len(sensor_val) > 0 else np.nan
                            row[f'{sampling}_Text'] = text_noproj_val[0] if len(text_noproj_val) > 0 else np.nan
                            row[f'{sampling}_Text+Proj'] = text_proj_val[0] if len(text_proj_val) > 0 else np.nan
                        else:
                            row[f'{sampling}_Sensor'] = np.nan
                            row[f'{sampling}_Text'] = np.nan
                            row[f'{sampling}_Text+Proj'] = np.nan

                    rows_data.append(row)

                # Create DataFrame
                comp_df = pd.DataFrame(rows_data)

                # Save as CSV
                csv_path = output_dir / f'comprehensive_table_{level}_{metric}.csv'
                comp_df.to_csv(csv_path, index=False)
                print(f"✅ Comprehensive table saved: {csv_path}")

                # Create heatmap
                fig, ax = plt.subplots(figsize=(16, 8))

                # Prepare data for heatmap (exclude Configuration column)
                heatmap_data = comp_df.set_index('Configuration')

                # Create heatmap with dynamic scale based on actual data range
                # Get the actual min and max values in the data (excluding NaN)
                data_min = heatmap_data.min().min()
                data_max = heatmap_data.max().max()
                data_center = (data_min + data_max) / 2

                sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn',
                           ax=ax, cbar_kws={'label': metric.replace('_', ' ').title()},
                           vmin=data_min, vmax=data_max, linewidths=0.5, center=data_center)

                ax.set_title(f'{level} {metric.replace("_", " ").title()} - Comprehensive Comparison',
                            fontsize=14, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('Configuration', fontweight='bold')

                # Rotate column labels for readability
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)

                plt.tight_layout()
                heatmap_path = output_dir / f'comprehensive_heatmap_{level}_{metric}.png'
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Comprehensive heatmap saved: {heatmap_path}")

    def create_comparison_charts(self, df: pd.DataFrame, summary_df: pd.DataFrame,
                                 output_dir: Path):
        """Create comparison charts for all metrics."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Overall F1-Weighted comparison (L1 and L2, Sensor only)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, level in enumerate(['L1', 'L2']):
            level_df = df[(df['label_level'] == level) & (df['model_type'] == 'Sensor')]
            level_df = level_df.sort_values('f1_weighted', ascending=False).head(15)

            ax = axes[idx]
            y_pos = np.arange(len(level_df))
            ax.barh(y_pos, level_df['f1_weighted'], color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{row['model_name'][:30]}..." if len(row['model_name']) > 30
                                else row['model_name']
                                for _, row in level_df.iterrows()], fontsize=8)
            ax.set_xlabel('F1-Weighted Score')
            ax.set_title(f'{level} Sensor F1-Weighted (Top 15)')
            ax.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, v in enumerate(level_df['f1_weighted']):
                ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / 'f1_weighted_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Chart saved: {output_dir / 'f1_weighted_comparison.png'}")

        # 2. Model type comparison (Text vs Sensor)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        metrics = ['f1_weighted', 'f1_macro', 'accuracy']
        levels = ['L1', 'L2']

        for level_idx, level in enumerate(levels):
            for metric_idx, metric in enumerate(metrics):
                ax = axes[level_idx, metric_idx]

                level_df = df[df['label_level'] == level]

                # Pivot for grouped bar chart
                pivot_data = level_df.pivot_table(
                    index=['model_name', 'sampling_strategy'],
                    columns='model_type',
                    values=metric,
                    aggfunc='first'
                ).reset_index()

                # Get top 10 by Sensor performance
                pivot_data = pivot_data.sort_values('Sensor', ascending=False).head(10)

                x = np.arange(len(pivot_data))
                width = 0.25

                for i, col in enumerate(['Text (No Proj)', 'Text (Proj)', 'Sensor']):
                    if col in pivot_data.columns:
                        ax.bar(x + i * width, pivot_data[col], width,
                              label=col, alpha=0.8)

                ax.set_xlabel('Model')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{level} - {metric.replace("_", " ").title()}')
                ax.set_xticks(x + width)
                ax.set_xticklabels([f"{row['model_name'][:15]}..."
                                    for _, row in pivot_data.iterrows()],
                                   rotation=45, ha='right', fontsize=7)
                ax.legend(fontsize=8)
                ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'model_type_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Chart saved: {output_dir / 'model_type_comparison.png'}")

        # 3. Sampling strategy comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, level in enumerate(['L1', 'L2']):
            ax = axes[idx]

            level_df = df[(df['label_level'] == level) & (df['model_type'] == 'Sensor')]

            # Group by sampling strategy
            sampling_stats = level_df.groupby('sampling_strategy')['f1_weighted'].agg(['mean', 'std', 'count'])
            sampling_stats = sampling_stats.sort_values('mean', ascending=False)

            x = np.arange(len(sampling_stats))
            ax.bar(x, sampling_stats['mean'], yerr=sampling_stats['std'],
                  capsize=5, color='coral', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(sampling_stats.index, rotation=45, ha='right')
            ax.set_ylabel('F1-Weighted Score')
            ax.set_title(f'{level} - Average F1-Weighted by Sampling Strategy')
            ax.grid(axis='y', alpha=0.3)

            # Add count annotations
            for i, (idx_val, row) in enumerate(sampling_stats.iterrows()):
                ax.text(i, row['mean'] + row['std'] + 0.01,
                       f"n={int(row['count'])}",
                       ha='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / 'sampling_strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Chart saved: {output_dir / 'sampling_strategy_comparison.png'}")

        # 4. L1 vs L2 scatter plot (Sensor performance)
        fig, ax = plt.subplots(figsize=(10, 10))

        # Merge L1 and L2 metrics
        l1_sensor = df[(df['label_level'] == 'L1') & (df['model_type'] == 'Sensor')][
            ['model_name', 'sampling_strategy', 'f1_weighted']
        ].rename(columns={'f1_weighted': 'l1_f1'})

        l2_sensor = df[(df['label_level'] == 'L2') & (df['model_type'] == 'Sensor')][
            ['model_name', 'sampling_strategy', 'f1_weighted']
        ].rename(columns={'f1_weighted': 'l2_f1'})

        scatter_df = pd.merge(l1_sensor, l2_sensor,
                             on=['model_name', 'sampling_strategy'])

        # Color by sampling strategy
        unique_strategies = scatter_df['sampling_strategy'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_strategies)))
        color_map = dict(zip(unique_strategies, colors))

        for strategy in unique_strategies:
            strategy_df = scatter_df[scatter_df['sampling_strategy'] == strategy]
            ax.scatter(strategy_df['l1_f1'], strategy_df['l2_f1'],
                      label=strategy, alpha=0.6, s=100,
                      color=color_map[strategy])

        # Add diagonal line
        max_val = max(scatter_df['l1_f1'].max(), scatter_df['l2_f1'].max())
        min_val = min(scatter_df['l1_f1'].min(), scatter_df['l2_f1'].min())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='y=x')

        ax.set_xlabel('L1 F1-Weighted Score')
        ax.set_ylabel('L2 F1-Weighted Score')
        ax.set_title('L1 vs L2 Performance (Sensor Embeddings)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'l1_vs_l2_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Chart saved: {output_dir / 'l1_vs_l2_scatter.png'}")

        # 5. Heatmap of all models
        if len(summary_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(summary_df) * 0.3)))

            for idx, level_prefix in enumerate(['L1', 'L2']):
                ax = axes[idx]

                # Select relevant columns
                cols = [f'{level_prefix}_F1_Weighted', f'{level_prefix}_F1_Macro',
                       f'{level_prefix}_Accuracy']

                heatmap_data = summary_df[['Model'] + cols].set_index('Model')

                # Truncate model names for readability
                heatmap_data.index = [name[:40] + '...' if len(name) > 40 else name
                                     for name in heatmap_data.index]

                # Convert to numeric
                for col in cols:
                    heatmap_data[col] = pd.to_numeric(heatmap_data[col])

                # Get dynamic scale based on actual data range
                data_min = heatmap_data[cols].min().min()
                data_max = heatmap_data[cols].max().max()
                data_center = (data_min + data_max) / 2

                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                           ax=ax, cbar_kws={'label': 'Score'},
                           vmin=data_min, vmax=data_max, linewidths=0.5, center=data_center)
                ax.set_title(f'{level_prefix} Metrics Heatmap')
                ax.set_xlabel('Metric')
                ax.set_ylabel('Model')

            plt.tight_layout()
        plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Chart saved: {output_dir / 'metrics_heatmap.png'}")

        # 6. Linear vs MLP Projection Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, level in enumerate(['L1', 'L2']):
            ax = axes[idx]

            level_df = df[(df['label_level'] == level) & (df['model_type'] == 'Sensor')]

            # Extract projection type
            level_df = level_df.copy()
            level_df['proj_type'] = level_df['model_name'].apply(
                lambda x: 'Linear' if 'projlin' in x else 'MLP' if 'projmlp' in x else 'Unknown'
            )

            # Group by projection type
            proj_stats = level_df.groupby('proj_type')['f1_weighted'].agg(['mean', 'std', 'count'])
            proj_stats = proj_stats.sort_values('mean', ascending=False)

            x = np.arange(len(proj_stats))
            ax.bar(x, proj_stats['mean'], yerr=proj_stats['std'],
                  capsize=5, color=['steelblue', 'coral'][:len(proj_stats)], alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(proj_stats.index)
            ax.set_ylabel('F1-Weighted Score')
            ax.set_title(f'{level} - Linear vs MLP Projection')
            ax.grid(axis='y', alpha=0.3)

            # Add count annotations
            for i, (idx_val, row) in enumerate(proj_stats.iterrows()):
                ax.text(i, row['mean'] + row['std'] + 0.01,
                       f"n={int(row['count'])}",
                       ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'projection_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Chart saved: {output_dir / 'projection_comparison.png'}")

        # 7. Sequence vs Image Encoder Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, level in enumerate(['L1', 'L2']):
            ax = axes[idx]

            level_df = df[(df['label_level'] == level) & (df['model_type'] == 'Sensor')]

            # Extract encoder type
            level_df = level_df.copy()
            level_df['encoder_type'] = level_df['model_name'].apply(
                lambda x: 'Sequence' if '_seq_' in x else 'Image' if '_img_' in x else 'Unknown'
            )

            # Group by encoder type
            encoder_stats = level_df.groupby('encoder_type')['f1_weighted'].agg(['mean', 'std', 'count'])
            encoder_stats = encoder_stats.sort_values('mean', ascending=False)

            x = np.arange(len(encoder_stats))
            ax.bar(x, encoder_stats['mean'], yerr=encoder_stats['std'],
                  capsize=5, color=['forestgreen', 'darkorange'][:len(encoder_stats)], alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(encoder_stats.index)
            ax.set_ylabel('F1-Weighted Score')
            ax.set_title(f'{level} - Sequence vs Image Encoder')
            ax.grid(axis='y', alpha=0.3)

            # Add count annotations
            for i, (idx_val, row) in enumerate(encoder_stats.iterrows()):
                ax.text(i, row['mean'] + row['std'] + 0.01,
                       f"n={int(row['count'])}",
                       ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'encoder_type_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Chart saved: {output_dir / 'encoder_type_comparison.png'}")

        # 8. Embedding Type Comparison (Sensor vs Text vs Text+Proj)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, level in enumerate(['L1', 'L2']):
            ax = axes[idx]

            level_df = df[df['label_level'] == level]

            # Group by model type
            embed_stats = level_df.groupby('model_type')['f1_weighted'].agg(['mean', 'std', 'count'])
            embed_stats = embed_stats.sort_values('mean', ascending=False)

            x = np.arange(len(embed_stats))
            colors = {'Sensor': 'steelblue', 'Text (Proj)': 'coral', 'Text (No Proj)': 'lightgray'}
            bar_colors = [colors.get(idx_val, 'gray') for idx_val in embed_stats.index]

            ax.bar(x, embed_stats['mean'], yerr=embed_stats['std'],
                  capsize=5, color=bar_colors, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(embed_stats.index, rotation=15, ha='right')
            ax.set_ylabel('F1-Weighted Score')
            ax.set_title(f'{level} - Embedding Type Comparison')
            ax.grid(axis='y', alpha=0.3)

            # Add count annotations
            for i, (idx_val, row) in enumerate(embed_stats.iterrows()):
                ax.text(i, row['mean'] + row['std'] + 0.01,
                       f"n={int(row['count'])}",
                       ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'embedding_type_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Chart saved: {output_dir / 'embedding_type_comparison.png'}")

    def run(self, model_regex: str, output_name: str = None):
        """Run the complete merging pipeline."""
        print("="*80)
        print("EMBEDDING EVALUATIONS MERGER")
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

        print(f"✅ Found {len(results_files)} models:")
        for item in results_files:
            print(f"   - {item['sampling_strategy']}/{item['model_name']}")
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
            output_name = f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        output_dir = self.results_base_dir / self.dataset / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save merged data
        print("💾 Saving merged data...")

        # CSV files
        df.to_csv(output_dir / 'detailed_results.csv', index=False)
        summary_df.to_csv(output_dir / 'summary_results.csv', index=False)
        print(f"✅ Saved: {output_dir / 'detailed_results.csv'}")
        print(f"✅ Saved: {output_dir / 'summary_results.csv'}")

        # JSON files
        df.to_json(output_dir / 'detailed_results.json', orient='records', indent=2)
        summary_df.to_json(output_dir / 'summary_results.json', orient='records', indent=2)
        print(f"✅ Saved: {output_dir / 'detailed_results.json'}")
        print(f"✅ Saved: {output_dir / 'summary_results.json'}")
        print("")

        # Create markdown report
        print("📝 Creating markdown report...")
        self.create_markdown_report(df, summary_df, output_dir / 'RESULTS_REPORT.md')
        print("")

        # Create comprehensive tables
        print("📊 Creating comprehensive comparison tables...")
        self.create_comprehensive_tables(df, output_dir)
        print("")

        # Add comprehensive tables to markdown report
        print("📝 Adding comprehensive tables to report...")
        self.add_comprehensive_tables_to_report(output_dir)
        print("")

        # Create charts
        print("📊 Creating comparison charts...")
        charts_dir = output_dir / 'charts'
        self.create_comparison_charts(df, summary_df, charts_dir)
        print("")

        # Print summary
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
        print(f"  - comprehensive_table_L1_f1_weighted.csv (+ heatmap)")
        print(f"  - comprehensive_table_L1_f1_macro.csv (+ heatmap)")
        print(f"  - comprehensive_table_L2_f1_weighted.csv (+ heatmap)")
        print(f"  - comprehensive_table_L2_f1_macro.csv (+ heatmap)")
        print(f"  - charts/ (11 comparison visualizations)")
        print("")
        print("Top 3 Models (L1 F1-Weighted):")
        for idx, row in summary_df.head(3).iterrows():
            print(f"  {idx+1}. {row['Model']}")
            print(f"     Sampling: {row['Sampling']}")
            print(f"     L1 F1-Weighted: {row['L1_F1_Weighted']:.4f}")
            print(f"     L1 Accuracy: {row['L1_Accuracy']:.4f}")
            print("")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple comprehensive_results.json files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all v1 models for Milan
  python src/evals/merge_embedding_evals.py --dataset milan --model-regex ".*_v1"

  # Merge only FD60 sequence models
  python src/evals/merge_embedding_evals.py --dataset milan --model-regex "fd60.*seq.*"

  # Merge only image models with custom output name
  python src/evals/merge_embedding_evals.py --dataset milan --model-regex ".*img.*" --output-name image_models_analysis
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
    merger = EmbeddingEvalsMerger(args.dataset, args.results_dir)
    merger.run(args.model_regex, args.output_name)


if __name__ == '__main__':
    main()

