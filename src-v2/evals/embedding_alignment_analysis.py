#!/usr/bin/env python3
"""
Comprehensive embedding alignment analysis script.
Computes detailed metrics on how well sensor and text embeddings are aligned.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import argparse

from models.text_encoder import TextEncoder, build_text_encoder
from models.sensor_encoder import SensorEncoder
from dataio.dataset import SmartHomeDataset
from dataio.collate import create_data_loader
from utils.device_utils import get_optimal_device, log_device_info
from utils.label_utils import convert_labels_to_text

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr


class EmbeddingAlignmentAnalyzer:
    """Analyze alignment between sensor and text embeddings."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_optimal_device()
        log_device_info(self.device)

        self._load_models()
        self._load_data()

    def _load_models(self):
        """Load trained models."""
        print(f"🔄 Loading models from {self.config['checkpoint_path']}")

        checkpoint = torch.load(self.config['checkpoint_path'], map_location=self.device)

        # Text encoder - use config from checkpoint
        model_config = checkpoint.get('config', {})
        text_model_name = model_config.get('text_model_name', 'thenlper/gte-base')
        # Use text encoder factory to handle different encoder types

        eval_config = model_config.copy() if "model_config" in locals() else {"text_model_name": text_model_name}

        eval_config["use_cached_embeddings"] = False  # Compute embeddings on-the-fly for eval

        self.text_encoder = build_text_encoder(eval_config)
        self.text_encoder.to(self.device)

        # Sensor encoder - use config from checkpoint
        self.vocab_sizes = checkpoint['vocab_sizes']
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
        self.sensor_encoder.to(self.device)
        self.sensor_encoder.eval()

        print("✅ Models loaded successfully")

    def _load_data(self):
        """Load test dataset."""
        self.dataset = SmartHomeDataset(
            data_path=self.config['data_path'],
            vocab_path=self.config['vocab_path'],
            sequence_length=20,
            max_captions=1
        )

        self.data_loader = create_data_loader(
            dataset=self.dataset,
            text_encoder=self.text_encoder,
            span_masker=None,
            vocab_sizes=self.vocab_sizes,
            device=self.device,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            apply_mlm=False
        )

        print(f"📊 Dataset loaded: {len(self.dataset)} samples")

    def extract_embeddings(self, max_samples: int = 5000) -> Dict[str, Any]:
        """Extract sensor and text embeddings."""
        print(f"🔄 Extracting embeddings from {min(max_samples, len(self.dataset))} samples...")

        sensor_embeddings = []
        labels_l1 = []
        labels_l2 = []
        samples_processed = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                if samples_processed >= max_samples:
                    break

                # Extract CLIP-projected sensor embeddings
                sensor_emb = self.sensor_encoder.forward_clip(
                    categorical_features=batch['categorical_features'],
                    coordinates=batch['coordinates'],
                    time_deltas=batch['time_deltas'],
                    mask=batch['mask']
                )

                sensor_embeddings.append(sensor_emb.cpu().numpy())

                # Extract labels
                batch_size_actual = sensor_emb.shape[0]
                for i in range(batch_size_actual):
                    if samples_processed >= max_samples:
                        break

                    sample_idx = batch_idx * 64 + i
                    if sample_idx < len(self.dataset):
                        original_sample = self.dataset.data[sample_idx]

                        label_l1 = original_sample.get('first_activity', 'Unknown')
                        label_l2 = original_sample.get('first_activity_l2', 'Unknown')

                        labels_l1.append(label_l1)
                        labels_l2.append(label_l2)
                        samples_processed += 1

        # Concatenate sensor embeddings
        sensor_embeddings = np.vstack(sensor_embeddings)[:samples_processed]

        # Create text embeddings for unique labels
        unique_labels_l1 = sorted(list(set(labels_l1)))
        text_descriptions = convert_labels_to_text(unique_labels_l1, single_description=True)

        with torch.no_grad():
            text_embeddings = self.text_encoder.encode_texts_clip(text_descriptions, self.device).cpu().numpy()

        print(f"📈 Extracted {sensor_embeddings.shape[0]} sensor embeddings")
        print(f"📈 Extracted {text_embeddings.shape[0]} text embeddings")

        return {
            'sensor_embeddings': sensor_embeddings,
            'text_embeddings': text_embeddings,
            'labels_l1': labels_l1,
            'labels_l2': labels_l2,
            'unique_labels': unique_labels_l1,
            'text_descriptions': text_descriptions
        }


    def compute_alignment_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comprehensive alignment metrics."""
        print("🔄 Computing alignment metrics...")

        sensor_emb = data['sensor_embeddings']
        text_emb = data['text_embeddings']
        labels = data['labels_l1']
        unique_labels = data['unique_labels']

        metrics = {}

        # 1. Cross-modal similarity for each activity
        print("   Computing cross-modal similarities...")
        cross_modal_sims = {}

        for i, label in enumerate(unique_labels):
            # Get sensor embeddings for this label
            label_mask = np.array(labels) == label
            if np.sum(label_mask) == 0:
                continue

            label_sensor_emb = sensor_emb[label_mask]
            label_text_emb = text_emb[i:i+1]  # Single text embedding

            # Compute similarities
            similarities = cosine_similarity(label_sensor_emb, label_text_emb).flatten()

            cross_modal_sims[label] = {
                'mean_similarity': float(np.mean(similarities)),
                'std_similarity': float(np.std(similarities)),
                'median_similarity': float(np.median(similarities)),
                'min_similarity': float(np.min(similarities)),
                'max_similarity': float(np.max(similarities)),
                'num_samples': int(np.sum(label_mask))
            }

        metrics['cross_modal_similarities'] = cross_modal_sims

        # 2. Overall alignment score
        all_similarities = []
        for label_stats in cross_modal_sims.values():
            # Weight by number of samples
            weight = label_stats['num_samples']
            all_similarities.extend([label_stats['mean_similarity']] * weight)

        metrics['overall_alignment'] = {
            'mean_similarity': float(np.mean(all_similarities)),
            'std_similarity': float(np.std(all_similarities)),
            'median_similarity': float(np.median(all_similarities))
        }

        # 3. Intra-modal clustering quality
        print("   Computing intra-modal clustering quality...")

        # Group sensor embeddings by label
        sensor_clusters = defaultdict(list)
        for i, label in enumerate(labels):
            sensor_clusters[label].append(sensor_emb[i])

        # Compute silhouette scores for sensor embeddings
        if len(unique_labels) > 1:
            # Create cluster labels for silhouette score
            cluster_labels = [unique_labels.index(label) for label in labels if label in unique_labels]
            valid_sensor_emb = sensor_emb[[i for i, label in enumerate(labels) if label in unique_labels]]

            if len(set(cluster_labels)) > 1 and len(valid_sensor_emb) > 1:
                sensor_silhouette = silhouette_score(valid_sensor_emb, cluster_labels)
                metrics['sensor_clustering_quality'] = float(sensor_silhouette)
            else:
                metrics['sensor_clustering_quality'] = 0.0
        else:
            metrics['sensor_clustering_quality'] = 0.0

        # 4. Text-to-text similarity matrix
        print("   Computing text-to-text similarities...")
        text_text_sim = cosine_similarity(text_emb, text_emb)
        metrics['text_similarity_matrix'] = text_text_sim

        # 5. Sensor centroid to text similarity
        print("   Computing sensor centroid to text similarities...")
        centroid_similarities = {}

        for i, label in enumerate(unique_labels):
            label_mask = np.array(labels) == label
            if np.sum(label_mask) == 0:
                continue

            # Compute centroid of sensor embeddings for this label
            label_sensor_emb = sensor_emb[label_mask]
            centroid = np.mean(label_sensor_emb, axis=0, keepdims=True)

            # Similarity to corresponding text embedding
            sim_to_text = cosine_similarity(centroid, text_emb[i:i+1])[0, 0]
            centroid_similarities[label] = float(sim_to_text)

        metrics['centroid_to_text_similarities'] = centroid_similarities

        print("✅ Alignment metrics computed")
        return metrics

    def create_alignment_visualizations(self, data: Dict[str, Any], metrics: Dict[str, Any],
                                      output_dir: Path) -> None:
        """Create comprehensive alignment visualization charts."""
        print("🔄 Creating alignment visualizations...")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract model name and test data name for chart subtitles
        model_name = Path(self.config['checkpoint_path']).parent.name if 'checkpoint_path' in self.config else 'unknown_model'
        test_data_name = Path(self.config['data_path']).stem if 'data_path' in self.config else 'unknown_data'

        # 1. Cross-modal similarity distribution
        self._plot_similarity_distribution(metrics, output_dir, model_name, test_data_name)

        # 2. Per-activity alignment scores
        self._plot_per_activity_alignment(metrics, output_dir, model_name, test_data_name)

        # 3. Text similarity heatmap
        self._plot_text_similarity_heatmap(data, metrics, output_dir)

        # 4. Alignment quality vs sample size
        self._plot_alignment_vs_sample_size(metrics, output_dir)

        print("✅ Visualizations created")

    def _plot_similarity_distribution(self, metrics: Dict[str, Any], output_dir: Path, model_name: str = "", test_data_name: str = ""):
        """Plot distribution of cross-modal similarities."""
        similarities = []
        labels = []

        for label, stats in metrics['cross_modal_similarities'].items():
            similarities.extend([stats['mean_similarity']] * stats['num_samples'])
            labels.extend([label] * stats['num_samples'])

        plt.figure(figsize=(12, 6))

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(similarities, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(similarities), color='red', linestyle='--',
                   label=f'Mean: {np.mean(similarities):.3f}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Distribution of Cross-Modal Similarities')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Box plot by activity
        plt.subplot(1, 2, 2)
        df = pd.DataFrame({'similarity': similarities, 'activity': labels})
        activity_order = df.groupby('activity')['similarity'].mean().sort_values(ascending=False).index

        sns.boxplot(data=df, y='activity', x='similarity', order=activity_order)
        plt.title('Cross-Modal Similarity by Activity')
        plt.xlabel('Cosine Similarity')

        plt.tight_layout()
        plt.savefig(output_dir / 'similarity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_per_activity_alignment(self, metrics: Dict[str, Any], output_dir: Path, model_name: str = "", test_data_name: str = ""):
        """Plot alignment scores per activity."""
        activities = []
        mean_sims = []
        std_sims = []
        sample_counts = []

        for activity, stats in metrics['cross_modal_similarities'].items():
            activities.append(activity)
            mean_sims.append(stats['mean_similarity'])
            std_sims.append(stats['std_similarity'])
            sample_counts.append(stats['num_samples'])

        # Sort by mean similarity
        sorted_indices = np.argsort(mean_sims)[::-1]
        activities = [activities[i] for i in sorted_indices]
        mean_sims = [mean_sims[i] for i in sorted_indices]
        std_sims = [std_sims[i] for i in sorted_indices]
        sample_counts = [sample_counts[i] for i in sorted_indices]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Mean similarity with error bars
        bars = ax1.bar(range(len(activities)), mean_sims, yerr=std_sims,
                      capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Activity')
        ax1.set_ylabel('Mean Cosine Similarity')
        # Set title with subtitle if available
        title1 = 'Cross-Modal Alignment by Activity (Mean ± Std)'
        if model_name and test_data_name:
            title1 += f'\nModel: {model_name} | Data: {test_data_name}'
        ax1.set_title(title1)
        ax1.set_xticks(range(len(activities)))
        ax1.set_xticklabels(activities, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, mean_sim, std_sim) in enumerate(zip(bars, mean_sims, std_sims)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_sim + 0.01,
                    f'{mean_sim:.3f}', ha='center', va='bottom', fontsize=9)

        # Sample counts
        bars2 = ax2.bar(range(len(activities)), sample_counts, alpha=0.7,
                       color='lightcoral', edgecolor='darkred')
        ax2.set_xlabel('Activity')
        ax2.set_ylabel('Number of Samples')
        # Set title with subtitle if available
        title2 = 'Sample Count by Activity'
        if model_name and test_data_name:
            title2 += f'\nModel: {model_name} | Data: {test_data_name}'
        ax2.set_title(title2)
        ax2.set_xticks(range(len(activities)))
        ax2.set_xticklabels(activities, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, count in zip(bars2, sample_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sample_counts)*0.01,
                    str(count), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'per_activity_alignment.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_text_similarity_heatmap(self, data: Dict[str, Any], metrics: Dict[str, Any],
                                    output_dir: Path):
        """Plot text-to-text similarity heatmap."""
        text_sim_matrix = metrics['text_similarity_matrix']
        labels = data['unique_labels']

        plt.figure(figsize=(12, 10))

        # Create heatmap
        mask = np.triu(np.ones_like(text_sim_matrix, dtype=bool), k=1)
        sns.heatmap(text_sim_matrix,
                   xticklabels=labels,
                   yticklabels=labels,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   mask=mask,
                   cbar_kws={'label': 'Cosine Similarity'})

        plt.title('Text Embedding Similarity Matrix\n(Lower triangle shows similarities)')
        plt.xlabel('Activity')
        plt.ylabel('Activity')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'text_similarity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_alignment_vs_sample_size(self, metrics: Dict[str, Any], output_dir: Path):
        """Plot alignment quality vs sample size."""
        sample_sizes = []
        mean_similarities = []
        activities = []

        for activity, stats in metrics['cross_modal_similarities'].items():
            sample_sizes.append(stats['num_samples'])
            mean_similarities.append(stats['mean_similarity'])
            activities.append(activity)

        plt.figure(figsize=(10, 6))

        # Scatter plot
        scatter = plt.scatter(sample_sizes, mean_similarities, s=100, alpha=0.7,
                            c=mean_similarities, cmap='viridis', edgecolors='black')

        # Add labels for each point
        for i, activity in enumerate(activities):
            plt.annotate(activity, (sample_sizes[i], mean_similarities[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Correlation
        if len(sample_sizes) > 1:
            correlation, p_value = pearsonr(sample_sizes, mean_similarities)
            plt.title(f'Alignment Quality vs Sample Size\n'
                     f'Correlation: {correlation:.3f} (p={p_value:.3f})')
        else:
            plt.title('Alignment Quality vs Sample Size')

        plt.xlabel('Number of Samples')
        plt.ylabel('Mean Cross-Modal Similarity')
        plt.colorbar(scatter, label='Mean Similarity')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'alignment_vs_sample_size.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_metrics_report(self, metrics: Dict[str, Any], output_dir: Path):
        """Save detailed metrics report."""
        report_path = output_dir / 'alignment_metrics_report.txt'

        with open(report_path, 'w') as f:
            f.write("EMBEDDING ALIGNMENT ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Overall alignment
            overall = metrics['overall_alignment']
            f.write("OVERALL ALIGNMENT METRICS:\n")
            f.write(f"  Mean Similarity: {overall['mean_similarity']:.4f}\n")
            f.write(f"  Std Similarity:  {overall['std_similarity']:.4f}\n")
            f.write(f"  Median Similarity: {overall['median_similarity']:.4f}\n")
            f.write(f"  Sensor Clustering Quality (Silhouette): {metrics['sensor_clustering_quality']:.4f}\n\n")

            # Per-activity metrics
            f.write("PER-ACTIVITY ALIGNMENT METRICS:\n")
            f.write("-" * 40 + "\n")

            # Sort by mean similarity
            activities = list(metrics['cross_modal_similarities'].keys())
            activities.sort(key=lambda x: metrics['cross_modal_similarities'][x]['mean_similarity'], reverse=True)

            for activity in activities:
                stats = metrics['cross_modal_similarities'][activity]
                f.write(f"\n{activity}:\n")
                f.write(f"  Mean Similarity: {stats['mean_similarity']:.4f}\n")
                f.write(f"  Std Similarity:  {stats['std_similarity']:.4f}\n")
                f.write(f"  Min/Max Similarity: {stats['min_similarity']:.4f} / {stats['max_similarity']:.4f}\n")
                f.write(f"  Sample Count: {stats['num_samples']}\n")

            # Centroid similarities
            f.write("\n\nSENSOR CENTROID TO TEXT SIMILARITIES:\n")
            f.write("-" * 40 + "\n")
            centroid_sims = metrics['centroid_to_text_similarities']
            for activity in sorted(centroid_sims.keys(), key=lambda x: centroid_sims[x], reverse=True):
                f.write(f"  {activity}: {centroid_sims[activity]:.4f}\n")

        print(f"📄 Metrics report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive embedding alignment analysis')

    parser.add_argument('--checkpoint', type=str,
                       default='models/milan_20epochs_final/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str,
                       default='data/data_for_alignment/milan_training_20/milan_test.json',
                       help='Path to test data')
    parser.add_argument('--vocab', type=str,
                       default='data/data_for_alignment/milan_training_20/milan_vocab.json',
                       help='Path to vocabulary file')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum number of samples to analyze')
    parser.add_argument('--output_dir', type=str, default='src-v2/evals/alignment_analysis',
                       help='Output directory for results')

    args = parser.parse_args()

    config = {
        'checkpoint_path': args.checkpoint,
        'data_path': args.data,
        'vocab_path': args.vocab,
    }

    # Run analysis
    analyzer = EmbeddingAlignmentAnalyzer(config)

    # Extract embeddings
    data = analyzer.extract_embeddings(max_samples=args.max_samples)

    # Compute metrics
    metrics = analyzer.compute_alignment_metrics(data)

    # Create visualizations
    output_dir = Path(args.output_dir)
    analyzer.create_alignment_visualizations(data, metrics, output_dir)

    # Skip saving report as requested

    print(f"\n✅ Embedding alignment analysis complete!")
    print(f"📁 Results saved in: {output_dir}")

    # Print summary
    overall = metrics['overall_alignment']
    print(f"\n🎯 ALIGNMENT SUMMARY:")
    print(f"   Overall Mean Similarity: {overall['mean_similarity']:.4f}")
    print(f"   Sensor Clustering Quality: {metrics['sensor_clustering_quality']:.4f}")
    print(f"   Number of Activities: {len(metrics['cross_modal_similarities'])}")


if __name__ == "__main__":
    main()

