#!/usr/bin/env python3

"""
t-SNE visualization and evaluation of SCAN clustering results.
Visualizes embeddings colored by ground truth labels and cluster predictions.

Usage:
    # Basic usage (auto-detects test data from model config)
    python src/evals/scan_tsne_visualization.py \
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1

    # With explicit paths
    python src/evals/scan_tsne_visualization.py \
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \
        --data_path data/processed/casas/milan/FL_20/test.json \
        --vocab_path data/processed/casas/milan/FL_20/vocab.json

    # Custom output directory
    python src/evals/scan_tsne_visualization.py \
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \
        --output_dir results/scan/milan/scan_fl20_20cl_discover_v1
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
import torch
from torch.utils.data import DataLoader
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.scan_model import SCANClusteringModel
from dataio.dataset import SmartHomeDataset
from utils.device_utils import get_optimal_device


class SCANEvaluator:
    """
    Evaluator for SCAN clustering with t-SNE visualizations and metrics.
    """

    def __init__(
        self,
        model_dir: str,
        data_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        checkpoint_name: str = 'best_model.pt',
        keep_all_labels: bool = False
    ):
        """
        Initialize SCAN evaluator.

        Args:
            model_dir: Path to trained SCAN model directory
            data_path: Path to test data (auto-detected if None)
            vocab_path: Path to vocabulary (auto-detected if None)
            output_dir: Output directory (auto-generated if None)
            checkpoint_name: Checkpoint file name (default: best_model.pt)
            keep_all_labels: If True, include all labels including No_Activity/Unknown
        """
        self.model_dir = Path(model_dir)
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = self.model_dir / checkpoint_name
        self.keep_all_labels = keep_all_labels
        self.label_suffix = '_all_labels' if keep_all_labels else ''

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = get_optimal_device()
        print(f"Using device: {self.device}")

        # Load checkpoint to get config
        print(f"Loading checkpoint: {self.checkpoint_path}")
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

        # Extract config
        self.config = self.checkpoint.get('config', {})
        self.num_clusters = self.config.get('num_clusters', 20)

        # Auto-detect paths from config if not provided
        if data_path is None:
            train_data_path = self.config.get('train_data_path', '')
            # Replace train.json with test.json
            self.data_path = train_data_path.replace('train.json', 'test.json')
            print(f"Auto-detected test data: {self.data_path}")
        else:
            self.data_path = data_path

        if vocab_path is None:
            self.vocab_path = self.config.get('vocab_path', '')
            print(f"Auto-detected vocab: {self.vocab_path}")
        else:
            self.vocab_path = vocab_path

        # Setup output directory
        if output_dir is None:
            # Extract dataset info from data path
            # e.g., data/processed/casas/milan/FL_20/test.json -> milan, FL_20
            data_path_parts = Path(self.data_path).parts
            dataset_name = 'unknown'
            data_config = 'unknown'

            for i, part in enumerate(data_path_parts):
                if part == 'casas' and i + 1 < len(data_path_parts):
                    dataset_name = data_path_parts[i + 1]
                    if i + 2 < len(data_path_parts):
                        data_config = data_path_parts[i + 2]
                    break

            model_name = self.model_dir.name
            self.output_dir = Path('results/scan') / dataset_name / model_name
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        # Initialize components
        self.model = None
        self.dataset = None
        self.embeddings = None
        self.cluster_predictions = None
        self.ground_truth_labels = None
        self.tsne_embeddings = None

        # Load label colors from metadata
        self._load_label_colors()

    def _load_label_colors(self):
        """Load label colors from city metadata."""
        try:
            metadata_path = Path(__file__).parent.parent.parent / "metadata" / "casas_metadata.json"
            with open(metadata_path, 'r') as f:
                city_metadata = json.load(f)

            # Try to detect dataset from data path
            dataset_name = 'milan'  # default
            if 'aruba' in self.data_path.lower():
                dataset_name = 'aruba'
            elif 'cairo' in self.data_path.lower():
                dataset_name = 'cairo'

            dataset_metadata = city_metadata.get(dataset_name, {})

            # Load L1 colors - Milan uses 'label', others use 'label_color'
            self.label_colors = dataset_metadata.get('label',
                                dataset_metadata.get('label_color',
                                dataset_metadata.get('lable', {})))

            # Load L2 colors from label_deepcasas_color
            self.label_colors_l2 = dataset_metadata.get('label_deepcasas_color', {})

            if self.label_colors:
                print(f"Loaded {len(self.label_colors)} L1 label colors from {dataset_name} metadata")
            else:
                print(f"No L1 label colors found in {dataset_name} metadata, using default colors")
                self.label_colors = {}

            if self.label_colors_l2:
                print(f"Loaded {len(self.label_colors_l2)} L2 label colors from {dataset_name} metadata")

        except Exception as e:
            print(f"Could not load label colors: {e}")
            self.label_colors = {}
            self.label_colors_l2 = {}

    def load_model_and_data(self):
        """Load the trained SCAN model and test dataset."""
        print("\n" + "="*60)
        print("LOADING MODEL AND DATA")
        print("="*60)

        # Get pretrained model path from config
        pretrained_model_path = self.config.get('pretrained_model_path', '')

        print(f"Loading SCAN model (num_clusters={self.num_clusters})...")
        print(f"Pretrained model: {pretrained_model_path}")

        # Load SCAN model
        self.model = SCANClusteringModel(
            pretrained_model_path=pretrained_model_path,
            num_clusters=self.num_clusters,
            dropout=self.config.get('dropout', 0.1),
            freeze_encoder=True,
            vocab_path=self.vocab_path,
            device=self.device
        )

        # Load the SCAN checkpoint weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print("Loading test dataset...")
        self.dataset = SmartHomeDataset(
            data_path=self.data_path,
            vocab_path=self.vocab_path,
            sequence_length=self.config.get('sequence_length', 50),
            max_captions=1,
            caption_types='long'
        )

        print(f"Loaded {len(self.dataset)} test samples")

    def _move_to_device(self, obj):
        """Recursively move tensors to device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._move_to_device(item) for item in obj]
        else:
            return obj

    def extract_embeddings_and_predictions(self, max_samples: int = 5000):
        """Extract embeddings and cluster predictions from the model."""
        print("\n" + "="*60)
        print("EXTRACTING EMBEDDINGS AND PREDICTIONS")
        print("="*60)

        # Create collator for sensor data
        def collate_fn(batch):
            all_categorical = [sample['categorical_features'] for sample in batch]
            all_coordinates = [sample['coordinates'] for sample in batch]
            all_time_deltas = [sample['time_deltas'] for sample in batch]
            all_masks = [sample['mask'] for sample in batch]

            coordinates = torch.stack(all_coordinates)
            time_deltas = torch.stack(all_time_deltas)
            masks = torch.stack(all_masks)

            categorical_features = {}
            for field in all_categorical[0].keys():
                field_tensors = [sample[field] for sample in all_categorical]
                categorical_features[field] = torch.stack(field_tensors)

            input_data = {
                'categorical_features': categorical_features,
                'coordinates': coordinates,
                'time_deltas': time_deltas
            }

            # Also extract ground truth labels
            gt_labels = []
            for sample in batch:
                label = sample.get('first_activity', sample.get('activity', 'Unknown'))
                gt_labels.append(label)

            return {
                'input_data': input_data,
                'mask': masks,
                'gt_labels': gt_labels
            }

        # Filter out "No_Activity" samples (unless keep_all_labels is True)
        if self.keep_all_labels:
            print("Keeping all labels (including No_Activity/Unknown)...")
            valid_indices = list(range(len(self.dataset)))
            print(f"Found {len(valid_indices)} total samples")
        else:
            print("Filtering samples...")
            no_activity_variations = ['No_Activity', 'no_activity', 'No Activity', 'no activity', 'Unknown', 'unknown', '']

            valid_indices = []
            for i in range(len(self.dataset)):
                sample = self.dataset[i]
                activity = sample.get('first_activity', sample.get('activity', 'Unknown'))
                if activity.strip().lower() not in [var.lower() for var in no_activity_variations]:
                    valid_indices.append(i)

            print(f"Found {len(valid_indices)} valid samples (excluding No_Activity)")

        # Limit samples if needed
        if len(valid_indices) > max_samples:
            np.random.seed(42)
            valid_indices = np.random.choice(valid_indices, max_samples, replace=False).tolist()
            print(f"Sampled {max_samples} samples for visualization")

        subset_dataset = torch.utils.data.Subset(self.dataset, valid_indices)

        dataloader = DataLoader(
            dataset=subset_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        embeddings = []
        cluster_predictions = []
        ground_truth_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                input_data = self._move_to_device(batch['input_data'])
                mask = batch['mask'].to(self.device)

                # Get embeddings and cluster predictions
                logits, batch_embeddings = self.model(
                    input_data=input_data,
                    attention_mask=mask,
                    return_embeddings=True
                )

                batch_predictions = torch.argmax(logits, dim=1)

                embeddings.append(batch_embeddings.cpu().numpy())
                cluster_predictions.append(batch_predictions.cpu().numpy())
                ground_truth_labels.extend(batch['gt_labels'])

                if (batch_idx + 1) % 20 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")

        self.embeddings = np.vstack(embeddings)
        self.cluster_predictions = np.concatenate(cluster_predictions)
        self.ground_truth_labels = np.array(ground_truth_labels)

        print(f"\nExtracted embeddings shape: {self.embeddings.shape}")
        print(f"Unique ground truth labels: {len(np.unique(self.ground_truth_labels))}")
        print(f"Unique cluster predictions: {len(np.unique(self.cluster_predictions))}")

        # Print distribution of ground truth labels
        print("\nGround truth label distribution:")
        label_counts = Counter(self.ground_truth_labels)
        for label, count in label_counts.most_common():
            print(f"  {label}: {count} ({100*count/len(self.ground_truth_labels):.1f}%)")

    def compute_tsne(self, perplexity: int = 30, random_state: int = 42):
        """Compute t-SNE embeddings."""
        print("\n" + "="*60)
        print("COMPUTING t-SNE")
        print("="*60)
        print(f"Perplexity: {perplexity}, Samples: {len(self.embeddings)}")

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=1000,
            verbose=1
        )

        self.tsne_embeddings = tsne.fit_transform(self.embeddings)
        print("t-SNE computation completed")

    def create_visualizations(self):
        """Create t-SNE visualizations colored by ground truth and cluster predictions."""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)

        plt.style.use('default')

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        # ========================================
        # Plot 1: Colored by ground truth labels
        # ========================================
        ax1 = axes[0]
        unique_gt_labels = sorted(np.unique(self.ground_truth_labels))

        for i, label in enumerate(unique_gt_labels):
            mask = self.ground_truth_labels == label

            # Get color from metadata or use colormap
            if label in self.label_colors:
                color = self.label_colors[label]
            else:
                color = plt.cm.tab20(i / max(len(unique_gt_labels), 1))

            ax1.scatter(
                self.tsne_embeddings[mask, 0],
                self.tsne_embeddings[mask, 1],
                c=[color],
                label=label.replace('_', ' '),
                alpha=0.7,
                s=25,
                edgecolors='white',
                linewidth=0.3
            )

        ax1.set_title('t-SNE by Ground Truth Activities', fontsize=16, fontweight='bold', pad=15)
        ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # ========================================
        # Plot 2: Colored by SCAN cluster labels (categorical)
        # ========================================
        ax2 = axes[1]
        unique_clusters = sorted(np.unique(self.cluster_predictions))

        # Create categorical cluster labels
        cluster_labels = [f'Cluster {c}' for c in unique_clusters]
        colors_cluster = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster_id in enumerate(unique_clusters):
            mask = self.cluster_predictions == cluster_id
            ax2.scatter(
                self.tsne_embeddings[mask, 0],
                self.tsne_embeddings[mask, 1],
                c=[colors_cluster[i]],
                label=f'Cluster {cluster_id}',  # Categorical label
                alpha=0.7,
                s=25,
                edgecolors='white',
                linewidth=0.3
            )

        ax2.set_title('t-SNE by SCAN Cluster Predictions', fontsize=16, fontweight='bold', pad=15)
        ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        # Save the plot
        output_path = self.output_dir / f'tsne_visualization{self.label_suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {output_path}")

        # Also save individual plots
        self._save_individual_plots()

    def _save_individual_plots(self):
        """Save individual t-SNE plots."""
        # Ground truth plot
        fig, ax = plt.subplots(figsize=(14, 12))
        unique_gt_labels = sorted(np.unique(self.ground_truth_labels))

        for i, label in enumerate(unique_gt_labels):
            mask = self.ground_truth_labels == label
            color = self.label_colors.get(label, plt.cm.tab20(i / max(len(unique_gt_labels), 1)))
            ax.scatter(
                self.tsne_embeddings[mask, 0],
                self.tsne_embeddings[mask, 1],
                c=[color],
                label=label.replace('_', ' '),
                alpha=0.7,
                s=30,
                edgecolors='white',
                linewidth=0.3
            )

        ax.set_title('t-SNE Embeddings by Ground Truth Activity', fontsize=16, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'tsne_ground_truth{self.label_suffix}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        # Cluster plot
        fig, ax = plt.subplots(figsize=(14, 12))
        unique_clusters = sorted(np.unique(self.cluster_predictions))
        colors_cluster = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster_id in enumerate(unique_clusters):
            mask = self.cluster_predictions == cluster_id
            ax.scatter(
                self.tsne_embeddings[mask, 0],
                self.tsne_embeddings[mask, 1],
                c=[colors_cluster[i]],
                label=f'Cluster {cluster_id}',
                alpha=0.7,
                s=30,
                edgecolors='white',
                linewidth=0.3
            )

        ax.set_title('t-SNE Embeddings by SCAN Cluster', fontsize=16, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'tsne_clusters{self.label_suffix}.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Saved: {self.output_dir / 'tsne_ground_truth.png'}")
        print(f"Saved: {self.output_dir / 'tsne_clusters.png'}")

    def compute_clustering_metrics(self) -> Dict:
        """Compute clustering evaluation metrics."""
        print("\n" + "="*60)
        print("COMPUTING CLUSTERING METRICS")
        print("="*60)

        # Create numerical labels for ground truth
        unique_gt_labels = np.unique(self.ground_truth_labels)
        gt_label_to_idx = {label: idx for idx, label in enumerate(unique_gt_labels)}
        gt_numerical = np.array([gt_label_to_idx[label] for label in self.ground_truth_labels])

        # Compute metrics
        ari = adjusted_rand_score(gt_numerical, self.cluster_predictions)
        nmi = normalized_mutual_info_score(gt_numerical, self.cluster_predictions)
        homogeneity = homogeneity_score(gt_numerical, self.cluster_predictions)
        completeness = completeness_score(gt_numerical, self.cluster_predictions)
        v_measure = v_measure_score(gt_numerical, self.cluster_predictions)

        # Compute cluster purity
        purity = self._compute_purity(gt_numerical, self.cluster_predictions)

        metrics = {
            'adjusted_rand_index': float(ari),
            'normalized_mutual_info': float(nmi),
            'homogeneity': float(homogeneity),
            'completeness': float(completeness),
            'v_measure': float(v_measure),
            'purity': float(purity),
            'num_ground_truth_classes': int(len(unique_gt_labels)),
            'num_predicted_clusters': int(len(np.unique(self.cluster_predictions))),
            'num_samples': int(len(self.ground_truth_labels)),
            'ground_truth_labels': list(unique_gt_labels)
        }

        # Print metrics
        print(f"\nAdjusted Rand Index (ARI):     {ari:.4f}")
        print(f"Normalized Mutual Info (NMI):  {nmi:.4f}")
        print(f"Homogeneity:                   {homogeneity:.4f}")
        print(f"Completeness:                  {completeness:.4f}")
        print(f"V-Measure:                     {v_measure:.4f}")
        print(f"Purity:                        {purity:.4f}")
        print(f"\nGround truth classes:          {len(unique_gt_labels)}")
        print(f"Predicted clusters:            {len(np.unique(self.cluster_predictions))}")
        print(f"Number of samples:             {len(self.ground_truth_labels)}")

        return metrics

    def _compute_purity(self, gt_labels, cluster_predictions):
        """Compute cluster purity score."""
        contingency = {}
        for gt, pred in zip(gt_labels, cluster_predictions):
            if pred not in contingency:
                contingency[pred] = {}
            contingency[pred][gt] = contingency[pred].get(gt, 0) + 1

        total_correct = sum(max(cluster.values()) for cluster in contingency.values())
        return total_correct / len(gt_labels)

    def create_confusion_matrix(self):
        """Create a confusion matrix between ground truth and clusters."""
        print("\n" + "="*60)
        print("CREATING CONFUSION MATRIX")
        print("="*60)

        unique_gt_labels = sorted(np.unique(self.ground_truth_labels))
        unique_clusters = sorted(np.unique(self.cluster_predictions))

        # Create confusion matrix data
        confusion_data = []
        for gt_label in unique_gt_labels:
            gt_mask = self.ground_truth_labels == gt_label
            for cluster_id in unique_clusters:
                cluster_mask = self.cluster_predictions == cluster_id
                overlap = np.sum(gt_mask & cluster_mask)
                if overlap > 0:
                    confusion_data.append({
                        'ground_truth': gt_label,
                        'cluster': f'Cluster {cluster_id}',  # Categorical
                        'count': overlap
                    })

        df = pd.DataFrame(confusion_data)
        if not df.empty:
            pivot_df = df.pivot(index='ground_truth', columns='cluster', values='count').fillna(0)

            # Reorder columns
            cluster_cols = [f'Cluster {i}' for i in unique_clusters]
            pivot_df = pivot_df[cluster_cols]

            # Create heatmap
            plt.figure(figsize=(14, 10))
            sns.heatmap(
                pivot_df,
                annot=True,
                fmt='g',
                cmap='Blues',
                cbar_kws={'label': 'Sample Count'},
                linewidths=0.5
            )
            plt.title('Confusion Matrix: Ground Truth vs SCAN Clusters', fontsize=14, fontweight='bold')
            plt.xlabel('SCAN Cluster', fontsize=12)
            plt.ylabel('Ground Truth Activity', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            confusion_path = self.output_dir / f'confusion_matrix{self.label_suffix}.png'
            plt.savefig(confusion_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Saved: {confusion_path}")

            # Save CSV
            csv_path = self.output_dir / f'confusion_matrix{self.label_suffix}.csv'
            pivot_df.to_csv(csv_path)
            print(f"Saved: {csv_path}")

    def create_cluster_composition_chart(self):
        """Create a stacked bar chart showing cluster composition."""
        print("Creating cluster composition chart...")

        unique_clusters = sorted(np.unique(self.cluster_predictions))
        unique_gt_labels = sorted(np.unique(self.ground_truth_labels))

        # Build composition data
        composition = {cluster: Counter() for cluster in unique_clusters}
        for gt, pred in zip(self.ground_truth_labels, self.cluster_predictions):
            composition[pred][gt] += 1

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(16, 8))

        bottom = np.zeros(len(unique_clusters))
        x = np.arange(len(unique_clusters))

        for i, gt_label in enumerate(unique_gt_labels):
            counts = [composition[cluster].get(gt_label, 0) for cluster in unique_clusters]
            color = self.label_colors.get(gt_label, plt.cm.tab20(i / max(len(unique_gt_labels), 1)))
            ax.bar(x, counts, bottom=bottom, label=gt_label.replace('_', ' '), color=color)
            bottom += counts

        ax.set_xlabel('SCAN Cluster', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Cluster Composition by Ground Truth Activity', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Cluster {c}' for c in unique_clusters], rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.output_dir / f'cluster_composition{self.label_suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {output_path}")

    def save_evaluation_report(self, metrics: Dict):
        """Save comprehensive evaluation report."""
        print("\n" + "="*60)
        print("SAVING EVALUATION REPORT")
        print("="*60)

        report = {
            'evaluation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_dir': str(self.model_dir),
            'checkpoint': self.checkpoint_name,
            'data_path': self.data_path,
            'config': {
                'num_clusters': self.num_clusters,
                'pretrained_model': self.config.get('pretrained_model_path', 'unknown')
            },
            'metrics': metrics,
            'cluster_distribution': {
                f'Cluster {int(c)}': int(count)
                for c, count in sorted(Counter(self.cluster_predictions).items())
            },
            'output_files': [
                f'tsne_visualization{self.label_suffix}.png',
                f'tsne_ground_truth{self.label_suffix}.png',
                f'tsne_clusters{self.label_suffix}.png',
                f'confusion_matrix{self.label_suffix}.png',
                f'confusion_matrix{self.label_suffix}.csv',
                f'cluster_composition{self.label_suffix}.png',
                f'evaluation_report{self.label_suffix}.json',
                f'evaluation_report{self.label_suffix}.txt'
            ]
        }

        # Save JSON report
        json_path = self.output_dir / f'evaluation_report{self.label_suffix}.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved: {json_path}")

        # Save text report
        txt_path = self.output_dir / f'evaluation_report{self.label_suffix}.txt'
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SCAN CLUSTERING EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Timestamp: {report['evaluation_timestamp']}\n")
            f.write(f"Model: {self.model_dir.name}\n")
            f.write(f"Checkpoint: {self.checkpoint_name}\n")
            f.write(f"Test Data: {self.data_path}\n\n")

            f.write("-"*70 + "\n")
            f.write("CLUSTERING METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Adjusted Rand Index (ARI):     {metrics['adjusted_rand_index']:.4f}\n")
            f.write(f"Normalized Mutual Info (NMI):  {metrics['normalized_mutual_info']:.4f}\n")
            f.write(f"Homogeneity:                   {metrics['homogeneity']:.4f}\n")
            f.write(f"Completeness:                  {metrics['completeness']:.4f}\n")
            f.write(f"V-Measure:                     {metrics['v_measure']:.4f}\n")
            f.write(f"Purity:                        {metrics['purity']:.4f}\n\n")

            f.write("-"*70 + "\n")
            f.write("DATASET INFO\n")
            f.write("-"*70 + "\n")
            f.write(f"Number of samples:             {metrics['num_samples']}\n")
            f.write(f"Ground truth classes:          {metrics['num_ground_truth_classes']}\n")
            f.write(f"Predicted clusters:            {metrics['num_predicted_clusters']}\n\n")

            f.write("-"*70 + "\n")
            f.write("CLUSTER DISTRIBUTION\n")
            f.write("-"*70 + "\n")
            for cluster, count in report['cluster_distribution'].items():
                pct = 100 * count / metrics['num_samples']
                f.write(f"{cluster}: {count} samples ({pct:.1f}%)\n")

            f.write("\n" + "="*70 + "\n")

        print(f"Saved: {txt_path}")

    def run_full_evaluation(self, max_samples: int = 5000, perplexity: int = 30):
        """Run the complete evaluation pipeline."""
        print("\n" + "="*70)
        print("SCAN CLUSTERING EVALUATION")
        print("="*70)

        self.load_model_and_data()
        self.extract_embeddings_and_predictions(max_samples=max_samples)
        self.compute_tsne(perplexity=perplexity)
        self.create_visualizations()
        metrics = self.compute_clustering_metrics()
        self.create_confusion_matrix()
        self.create_cluster_composition_chart()
        self.save_evaluation_report(metrics)

        print("\n" + "="*70)
        print("EVALUATION COMPLETED!")
        print(f"Results saved to: {self.output_dir}")
        print("="*70)

        return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SCAN clustering with t-SNE visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (auto-detects test data from model config)
    python src/evals/scan_tsne_visualization.py \\
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1

    # With explicit paths
    python src/evals/scan_tsne_visualization.py \\
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \\
        --data_path data/processed/casas/milan/FL_20/test.json \\
        --vocab_path data/processed/casas/milan/FL_20/vocab.json
        """
    )

    parser.add_argument('--model_dir', type=str, required=True,
                       help='Path to trained SCAN model directory')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                       help='Checkpoint file name (default: best_model.pt)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to test dataset (auto-detected if not provided)')
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to vocabulary file (auto-detected if not provided)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (auto-generated if not provided)')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum samples for visualization (default: 5000)')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity (default: 30)')
    parser.add_argument('--keep_all_labels', action='store_true',
                       help='Include all labels (including No_Activity/Unknown). Default: exclude them.')

    args = parser.parse_args()

    evaluator = SCANEvaluator(
        model_dir=args.model_dir,
        data_path=args.data_path,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint,
        keep_all_labels=args.keep_all_labels
    )

    evaluator.run_full_evaluation(
        max_samples=args.max_samples,
        perplexity=args.perplexity
    )


if __name__ == '__main__':
    main()
