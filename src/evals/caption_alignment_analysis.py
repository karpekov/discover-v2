#!/usr/bin/env python3
"""
Caption-based alignment analysis script.
Compares sensor embeddings with their actual corresponding captions (not prototypes),
stratified by activity labels to understand real-world alignment performance.
"""

import sys
import os
# Add both src directory and project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import argparse

from models.text_encoder import TextEncoder, build_text_encoder
from models.sensor_encoder import SensorEncoder
from alignment.dataset import AlignmentDataset
from dataio.collate import create_data_loader
from utils.device_utils import get_optimal_device, log_device_info

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


class CaptionAlignmentAnalyzer:
    """Analyze alignment between sensor embeddings and their actual captions."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_optimal_device()
        log_device_info(self.device)

        self._load_models()
        self._load_data()

    def _load_models(self):
        """Load trained models."""
        print(f"🔄 Loading models from {self.config['checkpoint_path']}")

        checkpoint = torch.load(self.config['checkpoint_path'], map_location=self.device, weights_only=False)

        # Get config from checkpoint and handle dataclass/dict
        raw_config = checkpoint.get('config', {})
        # Handle both dict and dataclass config objects
        if hasattr(raw_config, '__dataclass_fields__'):
            # It's an AlignmentConfig dataclass - get encoder config from it
            model_config = getattr(raw_config, 'encoder', {}) or {}
        else:
            # It's a plain dict
            model_config = raw_config

        # Text encoder - use robust 3-tier fallback to detect correct encoder
        from evals.eval_utils import create_text_encoder_from_checkpoint
        self.text_encoder = create_text_encoder_from_checkpoint(
            checkpoint=checkpoint,
            device=self.device,
            data_path=self.config.get('data_path')  # For .npz fallback
        )
        if self.text_encoder is None:
            # Fallback to standard text encoder
            text_model_name = model_config.get('text_model_name', 'sentence-transformers/all-MiniLM-L6-v2')
            eval_config = {"text_model_name": text_model_name, "use_cached_embeddings": False}
            self.text_encoder = build_text_encoder(eval_config)
        self.text_encoder.to(self.device)

        # Sensor encoder - check if it's ChronosEncoder or SensorEncoder
        self.vocab_sizes = checkpoint.get('vocab_sizes', {})

        # Check checkpoint format - AlignmentTrainer uses 'model_state_dict', AlignmentModel uses individual dicts
        if 'model_state_dict' in checkpoint:
            # Load from AlignmentModel - need to extract sensor_encoder weights
            from alignment.model import AlignmentModel
            full_model = AlignmentModel.load(self.config['checkpoint_path'], device=self.device)
            self.sensor_encoder = full_model.sensor_encoder
        elif 'chronos_encoder_state_dict' in checkpoint or 'sensor_encoder_state_dict' in checkpoint:
            # Old format - individual state dicts
            if 'chronos_encoder_state_dict' in checkpoint:
                from models.chronos_encoder import ChronosEncoder
                self.sensor_encoder = ChronosEncoder(
                    vocab_sizes=self.vocab_sizes,
                    chronos_model_name=model_config.get('chronos_model_name', 'amazon/chronos-2'),
                    projection_hidden_dim=model_config.get('projection_hidden_dim', 256),
                    projection_dropout=model_config.get('projection_dropout', 0.1),
                    output_dim=model_config.get('output_dim', 512),
                    sequence_length=model_config.get('sequence_length', 50)
                )
                self.sensor_encoder.load_state_dict(checkpoint['chronos_encoder_state_dict'])
            else:
                # Standard SensorEncoder
                self.sensor_encoder = SensorEncoder(
                    vocab_sizes=self.vocab_sizes,
                    d_model=model_config.get('d_model', 768),
                    n_layers=model_config.get('n_layers', 6),
                    n_heads=model_config.get('n_heads', 8),
                    d_ff=model_config.get('d_ff', 3072),
                    max_seq_len=model_config.get('max_seq_len', 512),
                    dropout=model_config.get('dropout', 0.1),
                    fourier_bands=model_config.get('fourier_bands', 12),
                    use_rope_time=model_config.get('use_rope_time', False),
                    use_rope_2d=model_config.get('use_rope_2d', False)
                )
                self.sensor_encoder.load_state_dict(checkpoint['sensor_encoder_state_dict'])
        else:
            raise ValueError("Checkpoint format not recognized - missing both 'model_state_dict' and 'sensor_encoder_state_dict'")

        self.sensor_encoder.to(self.device)
        self.sensor_encoder.eval()

        print("✅ Models loaded successfully")

    def _load_data(self):
        """Load test dataset."""
        # Try to find pre-computed text embeddings in the same directory
        data_path = Path(self.config['data_path'])
        data_dir = data_path.parent
        data_split = data_path.stem  # 'test', 'train', or 'val'

        # Look for matching text embeddings file
        text_embeddings_path = None
        npz_pattern = f"{data_split}_embeddings_*.npz"
        npz_files = list(data_dir.glob(npz_pattern))

        if npz_files:
            # Prefer CLIP embeddings if available (matches checkpoint dimension)
            for npz_file in npz_files:
                if 'clip' in npz_file.name.lower():
                    text_embeddings_path = str(npz_file)
                    print(f"📝 Found pre-computed text embeddings: {npz_file.name}")
                    break

            # Fallback to first available
            if not text_embeddings_path and npz_files:
                text_embeddings_path = str(npz_files[0])
                print(f"📝 Found pre-computed text embeddings: {npz_files[0].name}")
        else:
            print(f"⚠️  No pre-computed text embeddings found matching pattern: {npz_pattern}")
            print(f"   Text embeddings will be computed on-the-fly (slower)")

        # Load vocabulary
        with open(self.config['vocab_path'], 'r') as f:
            vocab = json.load(f)

        self.dataset = AlignmentDataset(
            data_path=self.config['data_path'],
            text_embeddings_path=text_embeddings_path,
            captions_path=None,  # We don't need captions file, just embeddings
            text_encoder_config_path=None,
            vocab=vocab,
            device=self.device
        )

        print(f"📊 Dataset loaded: {len(self.dataset)} samples")

    def extract_caption_alignments(self, max_samples: int = 5000) -> Dict[str, Any]:
        """Extract sensor embeddings and their corresponding caption embeddings."""
        print(f"🔄 Extracting caption alignments from {min(max_samples, len(self.dataset))} samples...")

        # Create data loader using AlignmentDataset's collate function
        from torch.utils.data import DataLoader
        data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            collate_fn=self.dataset.collate_fn
        )

        sensor_embeddings = []
        text_embeddings = []
        labels_l1 = []
        labels_l2 = []
        captions = []
        similarities = []
        samples_processed = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if samples_processed >= max_samples:
                    break

                # Extract CLIP-projected embeddings
                # AlignmentDataset returns sensor_data as a nested dict
                sensor_data = batch['sensor_data']
                sensor_emb = self.sensor_encoder.forward_clip(
                    input_data=sensor_data,
                    attention_mask=batch['attention_mask']
                )

                # Text embeddings are already CLIP-projected in the batch
                text_emb = batch['text_embeddings']

                # Compute pairwise similarities (sensor[i] with text[i])
                batch_similarities = torch.sum(sensor_emb * text_emb, dim=1).cpu().numpy()

                sensor_embeddings.append(sensor_emb.cpu().numpy())
                text_embeddings.append(text_emb.cpu().numpy())
                similarities.extend(batch_similarities)

                # Extract labels and captions from batch
                batch_size_actual = sensor_emb.shape[0]

                # Get sample IDs and look up labels from original data
                sample_ids = batch.get('sample_ids', [])

                # For AlignmentDataset, we need to get labels from the original data
                batch_labels_l1 = []
                batch_labels_l2 = []
                batch_captions = []

                for sample_id in sample_ids:
                    # Find sample in dataset by ID
                    sample_data = None
                    for sample in self.dataset.sensor_data:
                        if sample.get('sample_id') == sample_id:
                            sample_data = sample
                            break

                    if sample_data and 'metadata' in sample_data and 'ground_truth_labels' in sample_data['metadata']:
                        gt_labels = sample_data['metadata']['ground_truth_labels']
                        batch_labels_l1.append(gt_labels.get('primary_l1', 'Unknown'))
                        batch_labels_l2.append(gt_labels.get('primary_l2', 'Unknown'))
                    else:
                        batch_labels_l1.append('Unknown')
                        batch_labels_l2.append('Unknown')

                    batch_captions.append('')  # Captions not needed for visualization

                # Add all labels from this batch
                for i in range(min(batch_size_actual, len(batch_labels_l1))):
                    if samples_processed >= max_samples:
                        break

                    labels_l1.append(batch_labels_l1[i])
                    labels_l2.append(batch_labels_l2[i])
                    captions.append(batch_captions[i])
                    samples_processed += 1

                if batch_idx % 20 == 0:
                    print(f"  Processed {samples_processed}/{min(max_samples, len(self.dataset))} samples...")

        # Concatenate embeddings
        sensor_embeddings = np.vstack(sensor_embeddings)[:samples_processed]
        text_embeddings = np.vstack(text_embeddings)[:samples_processed]
        similarities = np.array(similarities[:samples_processed])

        print(f"📈 Extracted {sensor_embeddings.shape[0]} sensor-caption pairs")
        print(f"📊 Mean caption alignment: {np.mean(similarities):.4f} ± {np.std(similarities):.4f}")

        return {
            'sensor_embeddings': sensor_embeddings,
            'text_embeddings': text_embeddings,
            'similarities': similarities,
            'labels_l1': labels_l1,
            'labels_l2': labels_l2,
            'captions': captions
        }

    def filter_noisy_labels(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out noisy labels."""
        exclude_labels = {
            'other', 'no_activity', 'unknown', 'none', 'null', 'nan',
            'no activity', 'other activity', 'miscellaneous', 'misc'
        }

        valid_indices = []
        for i, (l1, l2) in enumerate(zip(data['labels_l1'], data['labels_l2'])):
            l1_lower = l1.lower().strip()
            l2_lower = l2.lower().strip()

            if l1_lower not in exclude_labels and l2_lower not in exclude_labels:
                valid_indices.append(i)

        if not valid_indices:
            print("⚠️  Warning: All samples filtered out!")
            return data

        # Filter all arrays
        filtered_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                filtered_data[key] = value[valid_indices]
            elif isinstance(value, list):
                filtered_data[key] = [value[i] for i in valid_indices]
            else:
                filtered_data[key] = value

        print(f"🧹 Filtered out noisy labels:")
        print(f"   Original samples: {len(data['labels_l1'])}")
        print(f"   Filtered samples: {len(filtered_data['labels_l1'])}")
        print(f"   Removed: {len(data['labels_l1']) - len(filtered_data['labels_l1'])} samples")

        return filtered_data

    def compute_stratified_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute alignment metrics stratified by activity labels."""
        print("🔄 Computing stratified alignment metrics...")

        similarities = data['similarities']
        labels_l1 = data['labels_l1']
        labels_l2 = data['labels_l2']

        metrics = {
            'l1_metrics': {},
            'l2_metrics': {},
            'overall_stats': {}
        }

        # Overall statistics
        metrics['overall_stats'] = {
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'median_similarity': float(np.median(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'total_samples': len(similarities)
        }

        # L1 (Primary) activity metrics
        print("   Computing L1 activity metrics...")
        l1_activities = sorted(list(set(labels_l1)))

        for activity in l1_activities:
            activity_mask = np.array(labels_l1) == activity
            activity_similarities = similarities[activity_mask]

            if len(activity_similarities) > 0:
                metrics['l1_metrics'][activity] = {
                    'mean_similarity': float(np.mean(activity_similarities)),
                    'std_similarity': float(np.std(activity_similarities)),
                    'median_similarity': float(np.median(activity_similarities)),
                    'min_similarity': float(np.min(activity_similarities)),
                    'max_similarity': float(np.max(activity_similarities)),
                    'sample_count': int(len(activity_similarities)),
                    'percentile_25': float(np.percentile(activity_similarities, 25)),
                    'percentile_75': float(np.percentile(activity_similarities, 75))
                }

        # L2 (Secondary) activity metrics
        print("   Computing L2 activity metrics...")
        l2_activities = sorted(list(set(labels_l2)))

        for activity in l2_activities:
            activity_mask = np.array(labels_l2) == activity
            activity_similarities = similarities[activity_mask]

            if len(activity_similarities) > 0:
                metrics['l2_metrics'][activity] = {
                    'mean_similarity': float(np.mean(activity_similarities)),
                    'std_similarity': float(np.std(activity_similarities)),
                    'median_similarity': float(np.median(activity_similarities)),
                    'min_similarity': float(np.min(activity_similarities)),
                    'max_similarity': float(np.max(activity_similarities)),
                    'sample_count': int(len(activity_similarities)),
                    'percentile_25': float(np.percentile(activity_similarities, 25)),
                    'percentile_75': float(np.percentile(activity_similarities, 75))
                }

        print("✅ Stratified metrics computed")
        return metrics

    def create_stratified_visualizations(self, data: Dict[str, Any], metrics: Dict[str, Any],
                                       output_dir: Path) -> None:
        """Create comprehensive stratified alignment visualizations."""
        print("🔄 Creating stratified visualizations...")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract model name and test data name for chart subtitles
        model_name = Path(self.config['checkpoint_path']).parent.name if 'checkpoint_path' in self.config else 'unknown_model'
        test_data_name = Path(self.config['data_path']).stem if 'data_path' in self.config else 'unknown_data'

        # 1. Overall distribution comparison
        self._plot_overall_distribution(data, metrics, output_dir, model_name, test_data_name)

        # 2. L1 activity comparison
        self._plot_activity_comparison(metrics, 'l1_metrics', 'L1 (Primary)', output_dir, model_name, test_data_name)

        # 3. L2 activity comparison
        self._plot_activity_comparison(metrics, 'l2_metrics', 'L2 (Secondary)', output_dir, model_name, test_data_name)

        # 4. Box plots by activity
        self._plot_activity_boxplots(data, output_dir)

        # 5. Correlation analysis
        self._plot_correlation_analysis(data, metrics, output_dir)

        print("✅ Stratified visualizations created")

    def _plot_overall_distribution(self, data: Dict[str, Any], metrics: Dict[str, Any],
                                 output_dir: Path, model_name: str = "", test_data_name: str = ""):
        """Plot overall distribution of caption alignments."""
        similarities = data['similarities']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Histogram
        ax1.hist(similarities, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        ax1.axvline(np.mean(similarities), color='red', linestyle='--',
                   label=f'Mean: {np.mean(similarities):.4f}')
        ax1.axvline(np.median(similarities), color='orange', linestyle='--',
                   label=f'Median: {np.median(similarities):.4f}')
        ax1.set_xlabel('Cosine Similarity (Sensor ↔ Caption)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Sensor-Caption Similarities')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cumulative distribution
        sorted_sims = np.sort(similarities)
        cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
        ax2.plot(sorted_sims, cumulative, linewidth=2)
        ax2.axvline(np.mean(similarities), color='red', linestyle='--', alpha=0.7)
        ax2.axvline(np.median(similarities), color='orange', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Cosine Similarity')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution of Similarities')
        ax2.grid(True, alpha=0.3)

        # Q-Q plot against normal distribution
        from scipy import stats
        stats.probplot(similarities, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot vs Normal Distribution')
        ax3.grid(True, alpha=0.3)

        # Statistics summary
        stats_text = f"""
        Statistics Summary:
        Mean: {metrics['overall_stats']['mean_similarity']:.4f}
        Std: {metrics['overall_stats']['std_similarity']:.4f}
        Median: {metrics['overall_stats']['median_similarity']:.4f}
        Min: {metrics['overall_stats']['min_similarity']:.4f}
        Max: {metrics['overall_stats']['max_similarity']:.4f}
        Samples: {metrics['overall_stats']['total_samples']}

        Percentiles:
        25th: {np.percentile(similarities, 25):.4f}
        75th: {np.percentile(similarities, 75):.4f}
        95th: {np.percentile(similarities, 95):.4f}
        """
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='center', fontfamily='monospace')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Summary Statistics')

        plt.tight_layout()
        plt.savefig(output_dir / 'caption_alignment_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_activity_comparison(self, metrics: Dict[str, Any], metric_key: str,
                                title_prefix: str, output_dir: Path, model_name: str = "", test_data_name: str = ""):
        """Plot comparison of activities for L1 or L2."""
        activity_metrics = metrics[metric_key]

        if not activity_metrics:
            return

        activities = []
        mean_sims = []
        std_sims = []
        sample_counts = []

        for activity, stats in activity_metrics.items():
            activities.append(activity)
            mean_sims.append(stats['mean_similarity'])
            std_sims.append(stats['std_similarity'])
            sample_counts.append(stats['sample_count'])

        # Sort by mean similarity
        sorted_indices = np.argsort(mean_sims)[::-1]
        activities = [activities[i] for i in sorted_indices]
        mean_sims = [mean_sims[i] for i in sorted_indices]
        std_sims = [std_sims[i] for i in sorted_indices]
        sample_counts = [sample_counts[i] for i in sorted_indices]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Mean similarity with error bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(activities)))
        bars = ax1.bar(range(len(activities)), mean_sims, yerr=std_sims,
                      capsize=5, alpha=0.8, color=colors, edgecolor='black')
        ax1.set_xlabel('Activity')
        ax1.set_ylabel('Mean Cosine Similarity')
        # Set title with subtitle if available
        title1 = f'{title_prefix} Activities: Caption Alignment (Mean ± Std)'
        if model_name and test_data_name:
            title1 += f'\nModel: {model_name} | Data: {test_data_name}'
        ax1.set_title(title1)
        ax1.set_xticks(range(len(activities)))
        ax1.set_xticklabels(activities, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, mean_sim, sample_count) in enumerate(zip(bars, mean_sims, sample_counts)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_sims[i] + 0.005,
                    f'{mean_sim:.3f}\n(n={sample_count})', ha='center', va='bottom', fontsize=9)

        # Sample counts (log scale for better visualization)
        bars2 = ax2.bar(range(len(activities)), sample_counts, alpha=0.8,
                       color=colors, edgecolor='black')
        ax2.set_xlabel('Activity')
        ax2.set_ylabel('Number of Samples (log scale)')
        ax2.set_yscale('log')
        # Set title with subtitle if available
        title2 = f'{title_prefix} Activities: Sample Distribution'
        if model_name and test_data_name:
            title2 += f'\nModel: {model_name} | Data: {test_data_name}'
        ax2.set_title(title2)
        ax2.set_xticks(range(len(activities)))
        ax2.set_xticklabels(activities, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, count in zip(bars2, sample_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    str(count), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        filename = f'caption_alignment_{metric_key.replace("_", "")}.png'
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_activity_boxplots(self, data: Dict[str, Any], output_dir: Path):
        """Create box plots showing distribution of similarities by activity."""
        similarities = data['similarities']
        labels_l1 = data['labels_l1']
        labels_l2 = data['labels_l2']

        # Create DataFrames for plotting
        df_l1 = pd.DataFrame({
            'similarity': similarities,
            'activity': labels_l1,
            'level': 'L1'
        })

        df_l2 = pd.DataFrame({
            'similarity': similarities,
            'activity': labels_l2,
            'level': 'L2'
        })

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # L1 box plot
        l1_order = df_l1.groupby('activity')['similarity'].mean().sort_values(ascending=False).index
        sns.boxplot(data=df_l1, x='activity', y='similarity', order=l1_order, ax=ax1)
        ax1.set_title('L1 (Primary) Activities: Caption Alignment Distribution')
        ax1.set_xlabel('Activity')
        ax1.set_ylabel('Cosine Similarity')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # L2 box plot
        l2_order = df_l2.groupby('activity')['similarity'].mean().sort_values(ascending=False).index
        sns.boxplot(data=df_l2, x='activity', y='similarity', order=l2_order, ax=ax2)
        ax2.set_title('L2 (Secondary) Activities: Caption Alignment Distribution')
        ax2.set_xlabel('Activity')
        ax2.set_ylabel('Cosine Similarity')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'caption_alignment_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_correlation_analysis(self, data: Dict[str, Any], metrics: Dict[str, Any],
                                 output_dir: Path):
        """Plot correlation between sample size and alignment quality."""
        l1_metrics = metrics['l1_metrics']

        sample_sizes = []
        mean_similarities = []
        std_similarities = []
        activities = []

        for activity, stats in l1_metrics.items():
            sample_sizes.append(stats['sample_count'])
            mean_similarities.append(stats['mean_similarity'])
            std_similarities.append(stats['std_similarity'])
            activities.append(activity)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Sample size vs mean similarity
        scatter = ax1.scatter(sample_sizes, mean_similarities, s=100, alpha=0.7,
                            c=mean_similarities, cmap='viridis', edgecolors='black')

        for i, activity in enumerate(activities):
            ax1.annotate(activity, (sample_sizes[i], mean_similarities[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        if len(sample_sizes) > 1:
            correlation, p_value = pearsonr(sample_sizes, mean_similarities)
            ax1.set_title(f'Sample Size vs Mean Alignment\nCorrelation: {correlation:.3f} (p={p_value:.3f})')
        else:
            ax1.set_title('Sample Size vs Mean Alignment')

        ax1.set_xlabel('Number of Samples')
        ax1.set_ylabel('Mean Caption Alignment')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Mean Similarity')

        # Sample size vs std similarity
        scatter2 = ax2.scatter(sample_sizes, std_similarities, s=100, alpha=0.7,
                             c=std_similarities, cmap='plasma', edgecolors='black')

        for i, activity in enumerate(activities):
            ax2.annotate(activity, (sample_sizes[i], std_similarities[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        if len(sample_sizes) > 1:
            correlation2, p_value2 = pearsonr(sample_sizes, std_similarities)
            ax2.set_title(f'Sample Size vs Alignment Variability\nCorrelation: {correlation2:.3f} (p={p_value2:.3f})')
        else:
            ax2.set_title('Sample Size vs Alignment Variability')

        ax2.set_xlabel('Number of Samples')
        ax2.set_ylabel('Std Caption Alignment')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Std Similarity')

        plt.tight_layout()
        plt.savefig(output_dir / 'caption_alignment_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_stratified_report(self, metrics: Dict[str, Any], output_dir: Path):
        """Save detailed stratified metrics report."""
        report_path = output_dir / 'caption_alignment_report.txt'

        with open(report_path, 'w') as f:
            f.write("CAPTION-BASED ALIGNMENT ANALYSIS REPORT\n")
            f.write("=" * 55 + "\n\n")

            # Overall statistics
            overall = metrics['overall_stats']
            f.write("OVERALL CAPTION ALIGNMENT METRICS:\n")
            f.write(f"  Mean Similarity: {overall['mean_similarity']:.4f}\n")
            f.write(f"  Std Similarity:  {overall['std_similarity']:.4f}\n")
            f.write(f"  Median Similarity: {overall['median_similarity']:.4f}\n")
            f.write(f"  Min/Max Similarity: {overall['min_similarity']:.4f} / {overall['max_similarity']:.4f}\n")
            f.write(f"  Total Samples: {overall['total_samples']}\n\n")

            # L1 activity metrics
            f.write("L1 (PRIMARY) ACTIVITY ALIGNMENT METRICS:\n")
            f.write("-" * 50 + "\n")

            l1_activities = sorted(metrics['l1_metrics'].keys(),
                                 key=lambda x: metrics['l1_metrics'][x]['mean_similarity'],
                                 reverse=True)

            for activity in l1_activities:
                stats = metrics['l1_metrics'][activity]
                f.write(f"\n{activity}:\n")
                f.write(f"  Mean Similarity: {stats['mean_similarity']:.4f} ± {stats['std_similarity']:.4f}\n")
                f.write(f"  Median Similarity: {stats['median_similarity']:.4f}\n")
                f.write(f"  Range: {stats['min_similarity']:.4f} to {stats['max_similarity']:.4f}\n")
                f.write(f"  25th-75th Percentile: {stats['percentile_25']:.4f} to {stats['percentile_75']:.4f}\n")
                f.write(f"  Sample Count: {stats['sample_count']}\n")

            # L2 activity metrics
            f.write("\n\nL2 (SECONDARY) ACTIVITY ALIGNMENT METRICS:\n")
            f.write("-" * 50 + "\n")

            l2_activities = sorted(metrics['l2_metrics'].keys(),
                                 key=lambda x: metrics['l2_metrics'][x]['mean_similarity'],
                                 reverse=True)

            for activity in l2_activities:
                stats = metrics['l2_metrics'][activity]
                f.write(f"\n{activity}:\n")
                f.write(f"  Mean Similarity: {stats['mean_similarity']:.4f} ± {stats['std_similarity']:.4f}\n")
                f.write(f"  Median Similarity: {stats['median_similarity']:.4f}\n")
                f.write(f"  Range: {stats['min_similarity']:.4f} to {stats['max_similarity']:.4f}\n")
                f.write(f"  25th-75th Percentile: {stats['percentile_25']:.4f} to {stats['percentile_75']:.4f}\n")
                f.write(f"  Sample Count: {stats['sample_count']}\n")

        print(f"📄 Stratified metrics report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Caption-based embedding alignment analysis')

    parser.add_argument('--checkpoint', type=str,
                       default='models/milan_20epochs_final/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str,
                       default='data/processed/casas/milan/training_20/test.json',
                       help='Path to test data')
    parser.add_argument('--vocab', type=str,
                       default='data/processed/casas/milan/training_20/vocab.json',
                       help='Path to vocabulary file')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum number of samples to analyze')
    parser.add_argument('--output_dir', type=str, default='results/evals/milan/caption_alignment',
                       help='Output directory for results')
    parser.add_argument('--filter_noisy', action='store_true',
                       help='Filter out noisy labels like no_activity')

    args = parser.parse_args()

    config = {
        'checkpoint_path': args.checkpoint,
        'data_path': args.data,
        'vocab_path': args.vocab,
    }

    # Run analysis
    analyzer = CaptionAlignmentAnalyzer(config)

    # Extract caption alignments
    data = analyzer.extract_caption_alignments(max_samples=args.max_samples)

    # Filter noisy labels if requested
    if args.filter_noisy:
        data = analyzer.filter_noisy_labels(data)

    # Compute stratified metrics
    metrics = analyzer.compute_stratified_metrics(data)

    # Create visualizations
    output_dir = Path(args.output_dir)
    analyzer.create_stratified_visualizations(data, metrics, output_dir)

    # Skip saving report as requested

    print(f"\n✅ Caption-based alignment analysis complete!")
    print(f"📁 Results saved in: {output_dir}")

    # Print summary
    overall = metrics['overall_stats']
    print(f"\n🎯 CAPTION ALIGNMENT SUMMARY:")
    print(f"   Overall Mean Similarity: {overall['mean_similarity']:.4f} ± {overall['std_similarity']:.4f}")
    print(f"   Median Similarity: {overall['median_similarity']:.4f}")
    print(f"   Total Samples: {overall['total_samples']}")

    # Top 3 L1 activities
    l1_sorted = sorted(metrics['l1_metrics'].items(),
                      key=lambda x: x[1]['mean_similarity'], reverse=True)[:3]
    print(f"\n🏆 TOP 3 L1 ACTIVITIES:")
    for activity, stats in l1_sorted:
        print(f"   {activity}: {stats['mean_similarity']:.4f} (n={stats['sample_count']})")


if __name__ == "__main__":
    main()

