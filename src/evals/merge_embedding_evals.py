#!/usr/bin/env python3
"""
Merge multiple comprehensive_results.json files into a single report.

Supports both classification and retrieval evaluation merging.

Usage:
    # Classification evals (default)
    python src/evals/merge_embedding_evals.py --dataset milan --model-regex ".*_v1"

    # Retrieval evals
    python src/evals/merge_embedding_evals.py --dataset milan --model-regex ".*_v1" --eval-type retrieval

    # Both types
    python src/evals/merge_embedding_evals.py --dataset milan --model-regex ".*_v1" --eval-type both
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

    def find_results_files(self, model_regex: str, eval_type: str = 'classification') -> List[Dict[str, Any]]:
        """Find all comprehensive_results.json files matching the regex.

        Args:
            model_regex: Regex pattern to match model names
            eval_type: 'classification', 'retrieval', or 'both'
        """
        pattern = re.compile(model_regex)
        results_files = []

        if eval_type in ['classification', 'both']:
            # Search for classification results
            for results_file in self.dataset_dir.rglob("comprehensive_results.json"):
                # Skip retrieval subdirectory
                if 'retrieval' in str(results_file):
                    continue

                # Extract model name from path
                # Path structure: results/evals/{dataset}/{sampling_strategy}/{model_name}/comprehensive_results.json
                model_name = results_file.parent.name
                sampling_strategy = results_file.parent.parent.name

                # Check if model matches regex
                if pattern.match(model_name):
                    results_files.append({
                        'model_name': model_name,
                        'sampling_strategy': sampling_strategy,
                        'file_path': results_file,
                        'eval_type': 'classification'
                    })

        if eval_type in ['retrieval', 'both']:
            # Search for retrieval results
            # Try both comprehensive_results.json and retrieval_metrics.json
            for results_file in self.dataset_dir.rglob("retrieval/*.json"):
                # Skip if it's not one of the expected files
                if results_file.name not in ['comprehensive_results.json', 'retrieval_metrics.json']:
                    continue

                # Extract model name from path
                # Path structure: results/evals/{dataset}/{sampling_strategy}/{model_name}/retrieval/{filename}.json
                model_name = results_file.parent.parent.name
                sampling_strategy = results_file.parent.parent.parent.name

                # Check if model matches regex
                if pattern.match(model_name):
                    results_files.append({
                        'model_name': model_name,
                        'sampling_strategy': sampling_strategy,
                        'file_path': results_file,
                        'eval_type': 'retrieval'
                    })

        return sorted(results_files, key=lambda x: (x['sampling_strategy'], x['model_name'], x.get('eval_type', '')))

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

    def load_retrieval_results(self, results_files: List[Dict[str, Any]]) -> pd.DataFrame:
        """Load all retrieval results files and merge into a DataFrame."""
        all_data = []

        for item in results_files:
            if item.get('eval_type') != 'retrieval':
                continue

            model_name = item['model_name']
            sampling_strategy = item['sampling_strategy']
            file_path = item['file_path']

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Handle different possible structures
                retrieval_data = None
                if 'retrieval_metrics' in data:
                    retrieval_data = data['retrieval_metrics']
                elif 'instance_to_instance' in data or 'prototype_based' in data:
                    retrieval_data = data
                else:
                    print(f"⚠️  Unexpected structure in {file_path}, skipping...")
                    continue

                # Extract metrics for each label level
                label_levels = ['l1', 'l2']
                if isinstance(retrieval_data, dict):
                    # Check which label levels are present
                    available_levels = [level for level in label_levels if level in retrieval_data]
                    if not available_levels:
                        # Try uppercase
                        available_levels = [level for level in ['L1', 'L2'] if level in retrieval_data]
                        label_levels = ['L1', 'L2'] if available_levels else []

                for level in label_levels:
                    if level not in retrieval_data:
                        continue

                    level_data = retrieval_data[level]

                    # Check if this is the new format (with recall_at_k) or old format
                    if 'recall_at_k' in level_data:
                        # New format: retrieval_metrics -> L1 -> recall_at_k -> text2sensor -> 10 -> {macro, weighted}
                        recall_data = level_data['recall_at_k']

                        # Instance-to-instance directions
                        for direction in ['text2sensor', 'sensor2text']:
                            if direction in recall_data:
                                direction_data = recall_data[direction]
                                for k_str, metrics in direction_data.items():
                                    k = int(k_str)
                                    if isinstance(metrics, dict):
                                        macro = metrics.get('macro', 0)
                                        weighted = metrics.get('weighted', 0)
                                    else:
                                        macro = metrics
                                        weighted = metrics

                                    row = {
                                        'model_name': model_name,
                                        'sampling_strategy': sampling_strategy,
                                        'label_level': level.upper(),
                                        'retrieval_type': 'instance_to_instance',
                                        'direction': direction,
                                        'k': k,
                                        'macro': macro,
                                        'weighted': weighted
                                    }
                                    all_data.append(row)

                        # Prototype-based directions
                        for direction in ['prototype2sensor', 'prototype2text']:
                            if direction in recall_data:
                                direction_data = recall_data[direction]
                                for k_str, metrics in direction_data.items():
                                    k = int(k_str)
                                    if isinstance(metrics, dict):
                                        macro = metrics.get('macro', 0)
                                        weighted = metrics.get('weighted', 0)
                                    else:
                                        macro = metrics
                                        weighted = metrics

                                    row = {
                                        'model_name': model_name,
                                        'sampling_strategy': sampling_strategy,
                                        'label_level': level.upper(),
                                        'retrieval_type': 'prototype_based',
                                        'direction': direction,
                                        'k': k,
                                        'macro': macro,
                                        'weighted': weighted
                                    }
                                    all_data.append(row)

                    else:
                        # Old format: instance_to_instance/prototype_based structure
                        # Instance-to-instance retrieval
                        if 'instance_to_instance' in level_data:
                            instance_data = level_data['instance_to_instance']
                            if 'overall' in instance_data:
                                overall = instance_data['overall']
                                for direction in ['text2sensor', 'sensor2text']:
                                    if direction in overall:
                                        direction_data = overall[direction]
                                        for k_str, metrics in direction_data.items():
                                            k = int(k_str)
                                            if isinstance(metrics, dict):
                                                macro = metrics.get('macro', 0)
                                                weighted = metrics.get('weighted', 0)
                                            else:
                                                macro = metrics
                                                weighted = metrics

                                            row = {
                                                'model_name': model_name,
                                                'sampling_strategy': sampling_strategy,
                                                'label_level': level.upper(),
                                                'retrieval_type': 'instance_to_instance',
                                                'direction': direction,
                                                'k': k,
                                                'macro': macro,
                                                'weighted': weighted
                                            }
                                            all_data.append(row)

                        # Prototype-based retrieval
                        if 'prototype_based' in level_data:
                            prototype_data = level_data['prototype_based']
                            if 'overall' in prototype_data:
                                overall = prototype_data['overall']
                                for direction in ['prototype2sensor', 'prototype2text']:
                                    if direction in overall:
                                        direction_data = overall[direction]
                                        for k_str, metrics in direction_data.items():
                                            k = int(k_str)
                                            if isinstance(metrics, dict):
                                                macro = metrics.get('macro', 0)
                                                weighted = metrics.get('weighted', 0)
                                            else:
                                                macro = metrics
                                                weighted = metrics

                                            row = {
                                                'model_name': model_name,
                                                'sampling_strategy': sampling_strategy,
                                                'label_level': level.upper(),
                                                'retrieval_type': 'prototype_based',
                                                'direction': direction,
                                                'k': k,
                                                'macro': macro,
                                                'weighted': weighted
                                            }
                                            all_data.append(row)

            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not all_data:
            raise ValueError("No valid retrieval results found!")

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

    def add_comprehensive_tables_to_report(self, output_dir: Path, report_filename: str = 'RESULTS_REPORT.md'):
        """Add comprehensive tables to the markdown report."""
        report_path = output_dir / report_filename

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

    def create_retrieval_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a summary table with best retrieval metrics for each model."""
        summary_data = []

        # Group by model and sampling strategy
        for (model_name, sampling_strategy), group in df.groupby(['model_name', 'sampling_strategy']):
            # Get metrics for K=50 (middle value, commonly used)
            k50_data = group[group['k'] == 50]

            if len(k50_data) == 0:
                # Try K=10 or K=100 if 50 not available
                k50_data = group[group['k'] == 10]
                if len(k50_data) == 0:
                    k50_data = group[group['k'] == 100]

            if len(k50_data) == 0:
                continue

            # Extract best metrics for each direction/type
            row = {
                'Model': model_name,
                'Sampling': sampling_strategy,
            }

            # Instance-to-instance metrics
            instance_l1 = k50_data[
                (k50_data['label_level'] == 'L1') &
                (k50_data['retrieval_type'] == 'instance_to_instance')
            ]
            instance_l2 = k50_data[
                (k50_data['label_level'] == 'L2') &
                (k50_data['retrieval_type'] == 'instance_to_instance')
            ]

            # Text2Sensor
            t2s_l1 = instance_l1[instance_l1['direction'] == 'text2sensor']
            t2s_l2 = instance_l2[instance_l2['direction'] == 'text2sensor']
            if len(t2s_l1) > 0:
                row['L1_T2S_Weighted'] = t2s_l1.iloc[0]['weighted']
                row['L1_T2S_Macro'] = t2s_l1.iloc[0]['macro']
            if len(t2s_l2) > 0:
                row['L2_T2S_Weighted'] = t2s_l2.iloc[0]['weighted']
                row['L2_T2S_Macro'] = t2s_l2.iloc[0]['macro']

            # Sensor2Text
            s2t_l1 = instance_l1[instance_l1['direction'] == 'sensor2text']
            s2t_l2 = instance_l2[instance_l2['direction'] == 'sensor2text']
            if len(s2t_l1) > 0:
                row['L1_S2T_Weighted'] = s2t_l1.iloc[0]['weighted']
                row['L1_S2T_Macro'] = s2t_l1.iloc[0]['macro']
            if len(s2t_l2) > 0:
                row['L2_S2T_Weighted'] = s2t_l2.iloc[0]['weighted']
                row['L2_S2T_Macro'] = s2t_l2.iloc[0]['macro']

            # Prototype-based metrics
            proto_l1 = k50_data[
                (k50_data['label_level'] == 'L1') &
                (k50_data['retrieval_type'] == 'prototype_based')
            ]
            proto_l2 = k50_data[
                (k50_data['label_level'] == 'L2') &
                (k50_data['retrieval_type'] == 'prototype_based')
            ]

            # Prototype2Sensor
            p2s_l1 = proto_l1[proto_l1['direction'] == 'prototype2sensor']
            p2s_l2 = proto_l2[proto_l2['direction'] == 'prototype2sensor']
            if len(p2s_l1) > 0:
                row['L1_P2S_Weighted'] = p2s_l1.iloc[0]['weighted']
                row['L1_P2S_Macro'] = p2s_l1.iloc[0]['macro']
            if len(p2s_l2) > 0:
                row['L2_P2S_Weighted'] = p2s_l2.iloc[0]['weighted']
                row['L2_P2S_Macro'] = p2s_l2.iloc[0]['macro']

            # Prototype2Text
            p2t_l1 = proto_l1[proto_l1['direction'] == 'prototype2text']
            p2t_l2 = proto_l2[proto_l2['direction'] == 'prototype2text']
            if len(p2t_l1) > 0:
                row['L1_P2T_Weighted'] = p2t_l1.iloc[0]['weighted']
                row['L1_P2T_Macro'] = p2t_l1.iloc[0]['macro']
            if len(p2t_l2) > 0:
                row['L2_P2T_Weighted'] = p2t_l2.iloc[0]['weighted']
                row['L2_P2T_Macro'] = p2t_l2.iloc[0]['macro']

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)

        # Sort by L1 Text2Sensor Weighted (primary metric)
        if len(summary_df) > 0 and 'L1_T2S_Weighted' in summary_df.columns:
            summary_df = summary_df.sort_values('L1_T2S_Weighted', ascending=False, na_position='last')

        return summary_df

    def create_retrieval_charts(self, df: pd.DataFrame, summary_df: pd.DataFrame,
                                output_dir: Path):
        """Create comparison charts for retrieval metrics."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Overall comparison: All directions, all K values (L1)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        k_values = sorted(df['k'].unique())
        directions = ['text2sensor', 'sensor2text', 'prototype2sensor', 'prototype2text']
        direction_labels = {
            'text2sensor': 'Text → Sensor',
            'sensor2text': 'Sensor → Text',
            'prototype2sensor': 'Prototype → Sensor',
            'prototype2text': 'Prototype → Text'
        }

        for level_idx, level in enumerate(['L1', 'L2']):
            level_df = df[df['label_level'] == level]

            # Top subplot: Weighted metrics
            ax_weighted = axes[level_idx, 0]
            for direction in directions:
                dir_df = level_df[level_df['direction'] == direction]
                if len(dir_df) == 0:
                    continue

                # Average across models for each K
                k_means = dir_df.groupby('k')['weighted'].mean()
                k_stds = dir_df.groupby('k')['weighted'].std()

                ax_weighted.plot(k_means.index, k_means.values,
                                marker='o', label=direction_labels.get(direction, direction),
                                linewidth=2, markersize=8)
                ax_weighted.fill_between(k_means.index,
                                        k_means.values - k_stds.values,
                                        k_means.values + k_stds.values,
                                        alpha=0.2)

            ax_weighted.set_xlabel('K (Number of Neighbors)', fontsize=11)
            ax_weighted.set_ylabel('Label-Recall@K (Weighted)', fontsize=11)
            ax_weighted.set_title(f'{level} - Weighted Average Across Models', fontsize=12, fontweight='bold')
            ax_weighted.legend(fontsize=9)
            ax_weighted.grid(alpha=0.3)
            ax_weighted.set_ylim([0, 1])

            # Bottom subplot: Macro metrics
            ax_macro = axes[level_idx, 1]
            for direction in directions:
                dir_df = level_df[level_df['direction'] == direction]
                if len(dir_df) == 0:
                    continue

                k_means = dir_df.groupby('k')['macro'].mean()
                k_stds = dir_df.groupby('k')['macro'].std()

                ax_macro.plot(k_means.index, k_means.values,
                             marker='o', label=direction_labels.get(direction, direction),
                             linewidth=2, markersize=8)
                ax_macro.fill_between(k_means.index,
                                     k_means.values - k_stds.values,
                                     k_means.values + k_stds.values,
                                     alpha=0.2)

            ax_macro.set_xlabel('K (Number of Neighbors)', fontsize=11)
            ax_macro.set_ylabel('Label-Recall@K (Macro)', fontsize=11)
            ax_macro.set_title(f'{level} - Macro Average Across Models', fontsize=12, fontweight='bold')
            ax_macro.legend(fontsize=9)
            ax_macro.grid(alpha=0.3)
            ax_macro.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_dir / 'retrieval_overall_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Chart saved: {output_dir / 'retrieval_overall_comparison.png'}")

        # 2. Top models comparison (K=50, L1, Weighted)
        if len(summary_df) > 0 and 'L1_T2S_Weighted' in summary_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            top_n = min(15, len(summary_df))
            top_models = summary_df.head(top_n)

            # Instance-to-instance
            ax1 = axes[0]
            x = np.arange(len(top_models))
            width = 0.35

            t2s_values = top_models['L1_T2S_Weighted'].fillna(0)
            s2t_values = top_models['L1_S2T_Weighted'].fillna(0)

            ax1.bar(x - width/2, t2s_values, width, label='Text → Sensor', alpha=0.8)
            ax1.bar(x + width/2, s2t_values, width, label='Sensor → Text', alpha=0.8)

            ax1.set_xlabel('Model', fontsize=11)
            ax1.set_ylabel('Label-Recall@50 (Weighted)', fontsize=11)
            ax1.set_title('L1 - Instance-to-Instance Retrieval (Top Models)', fontsize=12, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"{row['Model'][:20]}..." if len(row['Model']) > 20 else row['Model']
                                for _, row in top_models.iterrows()],
                               rotation=45, ha='right', fontsize=8)
            ax1.legend(fontsize=9)
            ax1.grid(axis='y', alpha=0.3)
            ax1.set_ylim([0, 1])

            # Prototype-based
            ax2 = axes[1]
            p2s_values = top_models['L1_P2S_Weighted'].fillna(0)
            p2t_values = top_models['L1_P2T_Weighted'].fillna(0)

            ax2.bar(x - width/2, p2s_values, width, label='Prototype → Sensor', alpha=0.8)
            ax2.bar(x + width/2, p2t_values, width, label='Prototype → Text', alpha=0.8)

            ax2.set_xlabel('Model', fontsize=11)
            ax2.set_ylabel('Label-Recall@50 (Weighted)', fontsize=11)
            ax2.set_title('L1 - Prototype-Based Retrieval (Top Models)', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"{row['Model'][:20]}..." if len(row['Model']) > 20 else row['Model']
                                for _, row in top_models.iterrows()],
                               rotation=45, ha='right', fontsize=8)
            ax2.legend(fontsize=9)
            ax2.grid(axis='y', alpha=0.3)
            ax2.set_ylim([0, 1])

            plt.tight_layout()
            plt.savefig(output_dir / 'retrieval_top_models.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Chart saved: {output_dir / 'retrieval_top_models.png'}")

        # 3. Heatmap: All models, all metrics (K=50, L1)
        if len(summary_df) > 0:
            fig, ax = plt.subplots(figsize=(14, max(8, len(summary_df) * 0.3)))

            # Select relevant columns
            metric_cols = [col for col in summary_df.columns
                          if col not in ['Model', 'Sampling'] and 'L1' in col and 'Weighted' in col]

            if metric_cols:
                heatmap_data = summary_df[['Model'] + metric_cols].set_index('Model')
                heatmap_data = heatmap_data.fillna(0)

                # Truncate model names
                heatmap_data.index = [name[:40] + '...' if len(name) > 40 else name
                                     for name in heatmap_data.index]

                # Create heatmap
                data_min = heatmap_data.min().min()
                data_max = heatmap_data.max().max()
                data_center = (data_min + data_max) / 2

                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                           ax=ax, cbar_kws={'label': 'Label-Recall@50 (Weighted)'},
                           vmin=data_min, vmax=data_max, linewidths=0.5, center=data_center)
                ax.set_title('L1 Retrieval Metrics Heatmap (K=50, Weighted)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Metric', fontsize=11)
                ax.set_ylabel('Model', fontsize=11)

                plt.tight_layout()
                plt.savefig(output_dir / 'retrieval_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Chart saved: {output_dir / 'retrieval_heatmap.png'}")

        # 4. Sampling strategy comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for level_idx, level in enumerate(['L1', 'L2']):
            ax = axes[level_idx]
            level_df = df[df['label_level'] == level]

            # Group by sampling strategy and direction
            sampling_stats = level_df.groupby(['sampling_strategy', 'direction'])['weighted'].agg(['mean', 'std', 'count'])

            sampling_strategies = sorted(level_df['sampling_strategy'].unique())
            x = np.arange(len(sampling_strategies))
            width = 0.2

            directions_to_plot = ['text2sensor', 'sensor2text']
            colors = ['steelblue', 'coral']

            for dir_idx, direction in enumerate(directions_to_plot):
                means = []
                stds = []
                for sampling in sampling_strategies:
                    if (sampling, direction) in sampling_stats.index:
                        stats = sampling_stats.loc[(sampling, direction)]
                        means.append(stats['mean'])
                        stds.append(stats['std'])
                    else:
                        means.append(0)
                        stds.append(0)

                ax.bar(x + dir_idx * width, means, width, yerr=stds,
                      label=direction_labels.get(direction, direction),
                      capsize=5, color=colors[dir_idx], alpha=0.7)

            ax.set_xlabel('Sampling Strategy', fontsize=11)
            ax.set_ylabel('Label-Recall@K (Weighted)', fontsize=11)
            ax.set_title(f'{level} - Average by Sampling Strategy', fontsize=12, fontweight='bold')
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(sampling_strategies, rotation=45, ha='right')
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(output_dir / 'retrieval_sampling_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Chart saved: {output_dir / 'retrieval_sampling_comparison.png'}")

    def create_retrieval_markdown_report(self, df: pd.DataFrame, summary_df: pd.DataFrame,
                                        output_path: Path):
        """Create a markdown report for retrieval results."""
        with open(output_path, 'w') as f:
            f.write(f"# Retrieval Evaluation Results - {self.dataset.upper()}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Models:** {len(summary_df)}\n\n")

            # Summary table
            f.write("## Summary - Retrieval Performance (K=50)\n\n")
            f.write("Models sorted by L1 Text→Sensor Weighted score (primary metric).\n\n")

            # Format summary table
            summary_formatted = summary_df.copy()
            metric_cols = [col for col in summary_formatted.columns if col not in ['Model', 'Sampling']]
            for col in metric_cols:
                summary_formatted[col] = summary_formatted[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

            # Write header
            f.write("| " + " | ".join(summary_formatted.columns) + " |\n")
            f.write("| " + " | ".join(["---"] * len(summary_formatted.columns)) + " |\n")
            # Write rows
            for _, row in summary_formatted.iterrows():
                f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
            f.write("\n\n")

            # Top 5 models
            f.write("## 🏆 Top 5 Models (L1 Text→Sensor Weighted)\n\n")
            if 'L1_T2S_Weighted' in summary_df.columns:
                top5 = summary_df.head(5)[['Model', 'Sampling', 'L1_T2S_Weighted', 'L1_S2T_Weighted']]
                top5_formatted = top5.copy()
                for col in ['L1_T2S_Weighted', 'L1_S2T_Weighted']:
                    top5_formatted[col] = top5_formatted[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

                f.write("| " + " | ".join(top5_formatted.columns) + " |\n")
                f.write("| " + " | ".join(["---"] * len(top5_formatted.columns)) + " |\n")
                for _, row in top5_formatted.iterrows():
                    f.write("| " + " | ".join(str(v) for v in row.values) + " |\n")
                f.write("\n\n")

            # Detailed breakdown by K value
            f.write("## Detailed Results by K Value\n\n")
            for level in ['L1', 'L2']:
                f.write(f"### {level}\n\n")
                level_df = df[df['label_level'] == level]

                for k in sorted(level_df['k'].unique()):
                    f.write(f"#### K={k}\n\n")
                    k_df = level_df[level_df['k'] == k]

                    # Pivot table: models vs directions
                    pivot_df = k_df.pivot_table(
                        index=['model_name', 'sampling_strategy'],
                        columns='direction',
                        values='weighted',
                        aggfunc='first'
                    ).reset_index()

                    pivot_df.columns.name = None
                    pivot_df = pivot_df.sort_values('text2sensor', ascending=False, na_position='last')

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
                    f.write("\n")

            # Metrics legend
            f.write("## Metrics Legend\n\n")
            f.write("- **Label-Recall@K**: Proportion of top-K retrieved neighbors that share the same label as the query\n")
            f.write("- **Weighted**: Average weighted by label prevalence (accounts for class imbalance)\n")
            f.write("- **Macro**: Unweighted average across all labels\n")
            f.write("- **Text → Sensor**: Text queries retrieving sensor embeddings\n")
            f.write("- **Sensor → Text**: Sensor queries retrieving text embeddings\n")
            f.write("- **Prototype → Sensor**: Text prototype queries retrieving sensor embeddings\n")
            f.write("- **Prototype → Text**: Text prototype queries retrieving text embeddings\n\n")

    def run(self, model_regex: str, output_name: str = None, eval_type: str = 'classification'):
        """Run the complete merging pipeline.

        Args:
            model_regex: Regex pattern to match model names
            output_name: Output directory name (default: auto-generated)
            eval_type: 'classification', 'retrieval', or 'both'
        """
        print("="*80)
        print("EMBEDDING EVALUATIONS MERGER")
        print("="*80)
        print(f"Dataset: {self.dataset}")
        print(f"Model regex: {model_regex}")
        print(f"Evaluation type: {eval_type}")
        print("")

        # Find results files
        print("🔍 Finding results files...")
        results_files = self.find_results_files(model_regex, eval_type)

        if not results_files:
            print("❌ No results files found matching the regex!")
            return

        print(f"✅ Found {len(results_files)} result files:")
        for item in results_files:
            eval_type_str = item.get('eval_type', 'classification')
            print(f"   - {item['sampling_strategy']}/{item['model_name']} ({eval_type_str})")
        print("")

        # Create output directory
        if output_name is None:
            suffix = eval_type if eval_type != 'both' else 'all'
            output_name = f"{suffix}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        output_dir = self.results_base_dir / self.dataset / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process classification results
        if eval_type in ['classification', 'both']:
            classification_files = [f for f in results_files if f.get('eval_type') == 'classification']
            if classification_files:
                print("="*80)
                print("PROCESSING CLASSIFICATION RESULTS")
                print("="*80)

                print("📊 Loading and merging classification results...")
                df_class = self.load_results(classification_files)
                print(f"✅ Loaded {len(df_class)} classification result entries")
                print("")

                print("📋 Creating classification summary table...")
                summary_df_class = self.create_summary_table(df_class)
                print(f"✅ Summary created with {len(summary_df_class)} models")
                print("")

                print("💾 Saving classification data...")
                df_class.to_csv(output_dir / 'classification_detailed_results.csv', index=False)
                summary_df_class.to_csv(output_dir / 'classification_summary_results.csv', index=False)
                df_class.to_json(output_dir / 'classification_detailed_results.json', orient='records', indent=2)
                summary_df_class.to_json(output_dir / 'classification_summary_results.json', orient='records', indent=2)
                print("✅ Saved classification results")
                print("")

                print("📝 Creating classification markdown report...")
                self.create_markdown_report(df_class, summary_df_class, output_dir / 'CLASSIFICATION_RESULTS_REPORT.md')
                print("")

                print("📊 Creating comprehensive comparison tables...")
                self.create_comprehensive_tables(df_class, output_dir)
                print("")

                print("📝 Adding comprehensive tables to report...")
                self.add_comprehensive_tables_to_report(output_dir, 'CLASSIFICATION_RESULTS_REPORT.md')
                print("")

                print("📊 Creating classification comparison charts...")
                charts_dir = output_dir / 'classification_charts'
                self.create_comparison_charts(df_class, summary_df_class, charts_dir)
                print("")

        # Process retrieval results
        if eval_type in ['retrieval', 'both']:
            retrieval_files = [f for f in results_files if f.get('eval_type') == 'retrieval']
            if retrieval_files:
                print("="*80)
                print("PROCESSING RETRIEVAL RESULTS")
                print("="*80)

                print("📊 Loading and merging retrieval results...")
                df_retrieval = self.load_retrieval_results(retrieval_files)
                print(f"✅ Loaded {len(df_retrieval)} retrieval result entries")
                print("")

                print("📋 Creating retrieval summary table...")
                summary_df_retrieval = self.create_retrieval_summary_table(df_retrieval)
                print(f"✅ Summary created with {len(summary_df_retrieval)} models")
                print("")

                print("💾 Saving retrieval data...")
                df_retrieval.to_csv(output_dir / 'retrieval_detailed_results.csv', index=False)
                summary_df_retrieval.to_csv(output_dir / 'retrieval_summary_results.csv', index=False)
                df_retrieval.to_json(output_dir / 'retrieval_detailed_results.json', orient='records', indent=2)
                summary_df_retrieval.to_json(output_dir / 'retrieval_summary_results.json', orient='records', indent=2)
                print("✅ Saved retrieval results")
                print("")

                print("📝 Creating retrieval markdown report...")
                self.create_retrieval_markdown_report(df_retrieval, summary_df_retrieval,
                                                     output_dir / 'RETRIEVAL_RESULTS_REPORT.md')
                print("")

                print("📊 Creating retrieval comparison charts...")
                charts_dir = output_dir / 'retrieval_charts'
                self.create_retrieval_charts(df_retrieval, summary_df_retrieval, charts_dir)
                print("")

        # Print final summary
        print("="*80)
        print("✅ MERGE COMPLETE!")
        print("="*80)
        print(f"Output directory: {output_dir}")
        print("")
        print("📁 Files created:")

        if eval_type in ['classification', 'both']:
            classification_files = [f for f in results_files if f.get('eval_type') == 'classification']
            if classification_files:
                print("\n📊 Classification Results:")
                print(f"  - classification_detailed_results.csv / .json")
                print(f"  - classification_summary_results.csv / .json")
                print(f"  - CLASSIFICATION_RESULTS_REPORT.md")
                print(f"  - comprehensive_table_L1_f1_weighted.csv (+ heatmap)")
                print(f"  - comprehensive_table_L1_f1_macro.csv (+ heatmap)")
                print(f"  - comprehensive_table_L2_f1_weighted.csv (+ heatmap)")
                print(f"  - comprehensive_table_L2_f1_macro.csv (+ heatmap)")
                print(f"  - classification_charts/ (11 comparison visualizations)")

        if eval_type in ['retrieval', 'both']:
            retrieval_files = [f for f in results_files if f.get('eval_type') == 'retrieval']
            if retrieval_files:
                print("\n🔍 Retrieval Results:")
                print(f"  - retrieval_detailed_results.csv / .json")
                print(f"  - retrieval_summary_results.csv / .json")
                print(f"  - RETRIEVAL_RESULTS_REPORT.md")
                print(f"  - retrieval_charts/ (4 comparison visualizations)")

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
    parser.add_argument('--eval-type', type=str, default='classification',
                       choices=['classification', 'retrieval', 'both'],
                       help='Type of evaluation to merge (default: classification)')
    parser.add_argument('--output-name', type=str, default=None,
                       help='Output directory name (default: auto-generated with timestamp)')
    parser.add_argument('--results-dir', type=str, default='results/evals',
                       help='Base results directory (default: results/evals)')

    args = parser.parse_args()

    # Run merger
    merger = EmbeddingEvalsMerger(args.dataset, args.results_dir)
    merger.run(args.model_regex, args.output_name, args.eval_type)


if __name__ == '__main__':
    main()

