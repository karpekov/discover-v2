#!/usr/bin/env python3

"""
t-SNE visualization of SCAN clustering results.
Visualizes embeddings colored by ground truth labels and cluster predictions.
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.scan_model import SCANClusteringModel
from dataio.dataset import SmartHomeDataset
from utils.device_utils import get_optimal_device


class SCANVisualizationAnalyzer:
    """
    Analyzer for SCAN clustering results with t-SNE visualizations.
    """

    def __init__(
        self,
        model_path: str,
        data_path: str,
        vocab_path: str,
        output_dir: str = "src/analysis/scan_visualization"
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = get_optimal_device()
        print(f"Using device: {self.device}")

        # Initialize components
        self.model = None
        self.dataset = None
        self.embeddings = None
        self.cluster_predictions = None
        self.ground_truth_labels = None

        # Load label colors from metadata
        self._load_label_colors()

    def _load_label_colors(self):
        """Load label colors from city metadata."""
        try:
            metadata_path = Path(__file__).parent.parent.parent / "metadata" / "casas_metadata.json"
            with open(metadata_path, 'r') as f:
                city_metadata = json.load(f)

            # Get Milan metadata (assuming we're working with Milan dataset)
            milan_metadata = city_metadata.get('milan', {})

            # Load L1 colors - try both 'label_color' and 'lable' (typo in the original file)
            self.label_colors = milan_metadata.get('label_color', milan_metadata.get('lable', {}))

            # Load L2 colors from label_deepcasas_color
            self.label_colors_l2 = milan_metadata.get('label_deepcasas_color', {})

            if self.label_colors:
                print(f"🎨 Loaded {len(self.label_colors)} L1 label colors from metadata")
            else:
                print("⚠️  No L1 label colors found in metadata, using default colors")
                self.label_colors = {}

            if self.label_colors_l2:
                print(f"🎨 Loaded {len(self.label_colors_l2)} L2 label colors from metadata")
            else:
                print("⚠️  No L2 label colors found in metadata, using default colors")
                self.label_colors_l2 = {}

        except Exception as e:
            print(f"⚠️  Could not load label colors: {e}")
            self.label_colors = {}
            self.label_colors_l2 = {}

    def load_model_and_data(self):
        """Load the trained SCAN model and test dataset."""
        print("Loading SCAN model...")

        # First load the checkpoint to get the number of clusters
        checkpoint = torch.load(self.model_path, map_location='cpu')

        # Extract num_clusters from checkpoint config
        if 'config' in checkpoint:
            num_clusters = checkpoint['config'].get('num_clusters', 20)
        elif 'hyperparameters' in checkpoint:
            num_clusters = checkpoint['hyperparameters'].get('num_clusters', 20)
        else:
            # Try to infer from model state dict
            clustering_head_keys = [k for k in checkpoint['model_state_dict'].keys() if 'clustering_head' in k and 'weight' in k]
            if clustering_head_keys:
                # Get the output dimension of the last layer
                last_layer_key = max(clustering_head_keys, key=lambda x: int(x.split('.')[1]))
                num_clusters = checkpoint['model_state_dict'][last_layer_key].shape[0]
                print(f"Inferred num_clusters={num_clusters} from model weights")
            else:
                num_clusters = 20  # fallback
                print("Warning: Could not determine num_clusters from checkpoint, using default 20")

        print(f"Using num_clusters={num_clusters}")

        # Load SCAN model
        self.model = SCANClusteringModel(
            pretrained_model_path=self.model_path.replace('best_model.pt', '../milan_baseline_50/best_model.pt'),
            num_clusters=num_clusters,
            dropout=0.1,
            freeze_encoder=True
        )

        # Load the SCAN checkpoint
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print("Loading test dataset...")
        # Load test dataset
        self.dataset = SmartHomeDataset(
            data_path=self.data_path,
            vocab_path=self.vocab_path,
            sequence_length=50,
            max_captions=1,
            caption_types='long'
        )

        print(f"Loaded {len(self.dataset)} samples")

    def extract_embeddings_and_predictions(self, max_samples: int = 5000):
        """Extract embeddings and cluster predictions from the model."""
        print(f"Extracting embeddings and predictions (max {max_samples} samples)...")

        # Create simple collator for sensor data only
        def simple_collate_fn(batch):
            batch_size = len(batch)

            # Extract components
            all_categorical = [sample['categorical_features'] for sample in batch]
            all_coordinates = [sample['coordinates'] for sample in batch]
            all_time_deltas = [sample['time_deltas'] for sample in batch]
            all_masks = [sample['mask'] for sample in batch]

            # Stack tensors
            coordinates = torch.stack(all_coordinates).to(self.device)
            time_deltas = torch.stack(all_time_deltas).to(self.device)
            masks = torch.stack(all_masks).to(self.device)

            # Stack categorical features
            categorical_features = {}
            for field in all_categorical[0].keys():
                field_tensors = [sample[field] for sample in all_categorical]
                categorical_features[field] = torch.stack(field_tensors).to(self.device)

            return {
                'categorical_features': categorical_features,
                'coordinates': coordinates,
                'time_deltas': time_deltas,
                'mask': masks
            }

        # Filter out "No_Activity" samples and limit dataset size for visualization
        print("Filtering out 'No_Activity' samples...")
        valid_indices = []
        no_activity_variations = ['No_Activity', 'no_activity', 'No Activity', 'no activity', 'Unknown', 'unknown', '']

        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            activity = sample.get('first_activity', 'Unknown')
            # Case-insensitive filtering for no activity variants
            if activity.strip().lower() not in [var.lower() for var in no_activity_variations]:
                valid_indices.append(i)

        # Debug: Print sample of activities found
        sample_activities = []
        for i in range(min(100, len(self.dataset))):
            sample = self.dataset[i]
            activity = sample.get('first_activity', 'Unknown')
            sample_activities.append(activity)

        from collections import Counter
        activity_counts = Counter(sample_activities)
        print(f"Sample activity distribution (first 100 samples):")
        for activity, count in activity_counts.most_common(10):
            print(f"  {activity}: {count}")

        print(f"Found {len(valid_indices)} valid samples (excluding No_Activity variations)")
        dataset_size = min(len(valid_indices), max_samples)
        selected_indices = np.random.choice(valid_indices, dataset_size, replace=False)
        subset_dataset = torch.utils.data.Subset(self.dataset, selected_indices)
        indices = selected_indices

        dataloader = DataLoader(
            dataset=subset_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            collate_fn=simple_collate_fn
        )

        embeddings = []
        cluster_predictions = []
        ground_truth_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Get embeddings and cluster predictions
                logits, batch_embeddings = self.model(
                    categorical_features=batch['categorical_features'],
                    coordinates=batch['coordinates'],
                    time_deltas=batch['time_deltas'],
                    mask=batch['mask'],
                    return_embeddings=True
                )

                # Get cluster predictions
                batch_predictions = torch.argmax(logits, dim=1)

                embeddings.append(batch_embeddings.cpu().numpy())
                cluster_predictions.append(batch_predictions.cpu().numpy())

                # Extract ground truth labels from original samples
                start_idx = batch_idx * dataloader.batch_size
                end_idx = min(start_idx + len(batch_embeddings), len(indices))
                batch_gt_labels = []

                for i in range(start_idx, end_idx):
                    original_idx = indices[i]
                    sample = self.dataset[original_idx]
                    # Use first_activity as ground truth label
                    gt_label = sample.get('first_activity', 'Unknown')
                    batch_gt_labels.append(gt_label)

                ground_truth_labels.extend(batch_gt_labels)

                if batch_idx % 20 == 0:
                    print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

        self.embeddings = np.vstack(embeddings)
        self.cluster_predictions = np.concatenate(cluster_predictions)
        self.ground_truth_labels = np.array(ground_truth_labels)

        print(f"Extracted embeddings shape: {self.embeddings.shape}")
        print(f"Unique ground truth labels: {len(np.unique(self.ground_truth_labels))}")
        print(f"Unique cluster predictions: {len(np.unique(self.cluster_predictions))}")

    def compute_tsne(self, perplexity: int = 30, random_state: int = 42):
        """Compute t-SNE embeddings."""
        print(f"Computing t-SNE with perplexity={perplexity}...")

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
            verbose=1
        )

        self.tsne_embeddings = tsne.fit_transform(self.embeddings)
        print("t-SNE computation completed")

    def create_visualizations(self):
        """Create t-SNE visualizations colored by ground truth and cluster predictions."""
        print("Creating visualizations...")

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        # Plot 1: Colored by ground truth labels
        ax1 = axes[0]
        unique_gt_labels = np.unique(self.ground_truth_labels)

        # Use colors from city metadata where available, otherwise fall back to default colors
        for i, label in enumerate(unique_gt_labels):
            mask = self.ground_truth_labels == label

            # Try to get color from metadata first
            if hasattr(self, 'label_colors') and label in self.label_colors:
                color = self.label_colors[label]
            else:
                # Fallback to Set3 colormap for labels not in metadata
                color = plt.cm.Set3(i / len(unique_gt_labels))

            ax1.scatter(
                self.tsne_embeddings[mask, 0],
                self.tsne_embeddings[mask, 1],
                c=[color],
                label=label.replace('_', ' '),  # Clean up label names
                alpha=0.8,
                s=30,
                edgecolors='white',
                linewidth=0.5
            )

        ax1.set_title('t-SNE colored by Ground Truth Activities\n(No_Activity samples excluded)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Colored by cluster predictions
        ax2 = axes[1]
        unique_clusters = np.unique(self.cluster_predictions)
        colors_cluster = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster_id in enumerate(unique_clusters):
            mask = self.cluster_predictions == cluster_id
            ax2.scatter(
                self.tsne_embeddings[mask, 0],
                self.tsne_embeddings[mask, 1],
                c=[colors_cluster[i]],
                label=f'Cluster {cluster_id}',
                alpha=0.8,
                s=30,
                edgecolors='white',
                linewidth=0.5
            )

        ax2.set_title('t-SNE colored by SCAN Cluster Predictions\n(No_Activity samples excluded)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot
        output_path = self.output_dir / 'scan_tsne_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")

    def compute_clustering_metrics(self):
        """Compute clustering evaluation metrics."""
        print("Computing clustering metrics...")

        # Create numerical labels for ground truth
        unique_gt_labels = np.unique(self.ground_truth_labels)
        gt_label_to_idx = {label: idx for idx, label in enumerate(unique_gt_labels)}
        gt_numerical = np.array([gt_label_to_idx[label] for label in self.ground_truth_labels])

        # Compute metrics
        ari = adjusted_rand_score(gt_numerical, self.cluster_predictions)
        nmi = normalized_mutual_info_score(gt_numerical, self.cluster_predictions)

        metrics = {
            'adjusted_rand_index': ari,
            'normalized_mutual_info': nmi,
            'num_ground_truth_classes': len(unique_gt_labels),
            'num_predicted_clusters': len(np.unique(self.cluster_predictions)),
            'num_samples': len(self.ground_truth_labels)
        }

        print("\n" + "="*50)
        print("CLUSTERING EVALUATION METRICS")
        print("="*50)
        print(f"Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
        print(f"Number of ground truth classes: {len(unique_gt_labels)}")
        print(f"Number of predicted clusters: {len(np.unique(self.cluster_predictions))}")
        print(f"Number of samples: {len(self.ground_truth_labels)}")
        print("="*50)

        # Save metrics
        metrics_path = self.output_dir / 'clustering_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

        return metrics

    def create_confusion_matrix(self):
        """Create a confusion matrix between ground truth and clusters."""
        print("Creating confusion matrix...")

        # Create a mapping table
        unique_gt_labels = np.unique(self.ground_truth_labels)
        unique_clusters = np.unique(self.cluster_predictions)

        # Create confusion matrix
        confusion_data = []
        for gt_label in unique_gt_labels:
            gt_mask = self.ground_truth_labels == gt_label
            for cluster_id in unique_clusters:
                cluster_mask = self.cluster_predictions == cluster_id
                overlap = np.sum(gt_mask & cluster_mask)
                if overlap > 0:
                    confusion_data.append({
                        'ground_truth': gt_label,
                        'cluster_id': cluster_id,
                        'count': overlap
                    })

        # Convert to DataFrame and create pivot table
        df = pd.DataFrame(confusion_data)
        if not df.empty:
            pivot_df = df.pivot(index='ground_truth', columns='cluster_id', values='count').fillna(0)

            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_df, annot=True, fmt='g', cmap='Blues', cbar_kws={'label': 'Sample Count'})
            plt.title('Confusion Matrix: Ground Truth vs SCAN Clusters\n(No_Activity samples excluded)', fontsize=14, fontweight='bold')
            plt.xlabel('SCAN Cluster ID')
            plt.ylabel('Ground Truth Activity')
            plt.tight_layout()

            # Save confusion matrix
            confusion_path = self.output_dir / 'confusion_matrix.png'
            plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to: {confusion_path}")

            plt.show()

            # Save the data
            pivot_df.to_csv(self.output_dir / 'confusion_matrix.csv')
            print(f"Saved confusion matrix data to: {self.output_dir / 'confusion_matrix.csv'}")

    def run_full_analysis(self, max_samples: int = 5000, perplexity: int = 30):
        """Run the complete analysis pipeline."""
        print("Starting SCAN clustering visualization analysis...")
        print("="*60)

        self.load_model_and_data()
        self.extract_embeddings_and_predictions(max_samples=max_samples)
        self.compute_tsne(perplexity=perplexity)
        self.create_visualizations()
        self.compute_clustering_metrics()
        self.create_confusion_matrix()

        print("\n" + "="*60)
        print("Analysis completed!")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Create t-SNE visualizations for SCAN clustering results')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained SCAN model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test dataset (JSON file)')
    parser.add_argument('--vocab_path', type=str, required=True,
                       help='Path to vocabulary file')
    parser.add_argument('--output_dir', type=str,
                       default='src/analysis/scan_visualization',
                       help='Output directory for visualizations')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum number of samples to use for visualization')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='Perplexity parameter for t-SNE')

    args = parser.parse_args()

    # Create analyzer and run analysis
    analyzer = SCANVisualizationAnalyzer(
        model_path=args.model_path,
        data_path=args.data_path,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir
    )

    analyzer.run_full_analysis(
        max_samples=args.max_samples,
        perplexity=args.perplexity
    )


if __name__ == '__main__':
    main()
