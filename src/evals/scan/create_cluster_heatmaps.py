#!/usr/bin/env python3

"""
Create spatial density heatmaps for each SCAN cluster on house floor plans.

For each cluster, extracts sensor activations (ON signals) and creates
KDE (Kernel Density Estimation) plots on the house floor plan.

Usage:
    # Basic usage
    python src/evals/scan/create_cluster_heatmaps.py \
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1

    # With custom data directory
    python src/evals/scan/create_cluster_heatmaps.py \
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \
        --data_dir data/processed/casas/milan/FL_20

    # Limit samples per cluster for faster processing
    python src/evals/scan/create_cluster_heatmaps.py \
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \
        --max_samples_per_cluster 500
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
import torch
from torch.utils.data import DataLoader, Subset

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from models.scan_model import SCANClusteringModel
from dataio.dataset import SmartHomeDataset
from utils.device_utils import get_optimal_device


class ClusterHeatmapGenerator:
    """
    Generate spatial density heatmaps for SCAN clusters on house floor plans.
    """

    def __init__(
        self,
        model_dir: str,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        checkpoint_name: str = 'best_model.pt',
        max_samples_per_cluster: int = 1000
    ):
        """
        Initialize cluster heatmap generator.

        Args:
            model_dir: Path to trained SCAN model directory
            data_dir: Path to data directory (auto-detected if None)
            output_dir: Output directory (auto-generated if None)
            checkpoint_name: Checkpoint file name
            max_samples_per_cluster: Maximum samples to use per cluster
        """
        self.model_dir = Path(model_dir)
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = self.model_dir / checkpoint_name
        self.max_samples_per_cluster = max_samples_per_cluster

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = get_optimal_device()
        print(f"Using device: {self.device}")

        # Load checkpoint to get config
        print(f"Loading checkpoint: {self.checkpoint_path}")
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

        self.config = self.checkpoint.get('config', {})
        self.num_clusters = self.config.get('num_clusters', 20)
        self.sequence_length = self.config.get('sequence_length', 50)

        # Determine data directory
        if data_dir is None:
            train_data_path = self.config.get('train_data_path', '')
            self.data_dir = Path(train_data_path).parent
            print(f"Auto-detected data directory: {self.data_dir}")
        else:
            self.data_dir = Path(data_dir)

        self.vocab_path = self.data_dir / 'vocab.json'

        # Detect dataset name (milan, aruba, cairo)
        self.dataset_name = self._detect_dataset_name()
        print(f"Detected dataset: {self.dataset_name}")

        # Setup output directory
        if output_dir is None:
            model_name = self.model_dir.name
            self.output_dir = Path('results/scan') / self.dataset_name / model_name
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        # Initialize components
        self.model = None
        self.floor_plan_img = None  # Will be set by _load_metadata

        # Load metadata (this sets floor_plan_img)
        self._load_metadata()

    def _detect_dataset_name(self) -> str:
        """Detect dataset name from data path."""
        data_str = str(self.data_dir).lower()
        for name in ['milan', 'aruba', 'cairo', 'kyoto']:
            if name in data_str:
                return name
        return 'unknown'

    def _load_metadata(self):
        """Load sensor coordinates and floor plan."""
        metadata_path = Path(__file__).parent.parent.parent.parent / "metadata" / "casas_metadata.json"

        try:
            with open(metadata_path, 'r') as f:
                all_metadata = json.load(f)

            self.metadata = all_metadata.get(self.dataset_name, {})
            self.sensor_coords = self.metadata.get('sensor_coordinates', {})
            self.sensor_locations = self.metadata.get('sensor_location', {})

            if self.sensor_coords:
                print(f"Loaded {len(self.sensor_coords)} sensor coordinates")
            else:
                print("WARNING: No sensor coordinates found in metadata")

        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.metadata = {}
            self.sensor_coords = {}
            self.sensor_locations = {}

        # Load floor plan image
        floor_plan_path = Path(__file__).parent.parent.parent.parent / "metadata" / "floor_plans_augmented" / f"{self.dataset_name}.png"

        if floor_plan_path.exists():
            try:
                self.floor_plan_img = plt.imread(str(floor_plan_path))
                self.img_height, self.img_width = self.floor_plan_img.shape[:2]
                print(f"Loaded floor plan: {floor_plan_path} ({self.img_width}x{self.img_height})")
            except Exception as e:
                print(f"Error loading floor plan: {e}")
                self.floor_plan_img = None
                self.img_width, self.img_height = 800, 600
        else:
            print(f"Floor plan not found: {floor_plan_path}")
            self.floor_plan_img = None
            self.img_width, self.img_height = 800, 600

    def load_model(self):
        """Load the trained SCAN model."""
        print("\n" + "="*60)
        print("LOADING SCAN MODEL")
        print("="*60)

        pretrained_model_path = self.config.get('pretrained_model_path', '')

        self.model = SCANClusteringModel(
            pretrained_model_path=pretrained_model_path,
            num_clusters=self.num_clusters,
            dropout=self.config.get('dropout', 0.1),
            freeze_encoder=True,
            vocab_path=str(self.vocab_path),
            device=self.device
        )

        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Loaded model with {self.num_clusters} clusters")

    def _collate_fn(self, batch):
        """Collate function for batching samples."""
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

        return {
            'input_data': input_data,
            'mask': masks
        }

    def _move_to_device(self, obj):
        """Recursively move tensors to device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._move_to_device(item) for item in obj]
        return obj

    def load_data_and_get_clusters(self) -> Tuple[Dict[int, List[Dict]], Dict[int, str]]:
        """
        Load test data, get cluster assignments, and organize samples by cluster.

        Returns:
            Tuple of (cluster_samples dict, cluster_majority_labels dict)
        """
        print("\n" + "="*60)
        print("LOADING TEST DATA AND ASSIGNING CLUSTERS")
        print("="*60)

        # Load test split only
        data_path = self.data_dir / 'test.json'
        if not data_path.exists():
            raise FileNotFoundError(f"Test data not found: {data_path}")

        print(f"Loading test data from: {data_path}")

        # Load raw JSON for sensor extraction
        with open(data_path, 'r') as f:
            data = json.load(f)
        raw_samples = data.get('samples', data if isinstance(data, list) else [])

        # Create dataset for model inference
        dataset = SmartHomeDataset(
            data_path=str(data_path),
            vocab_path=str(self.vocab_path),
            sequence_length=self.sequence_length,
            max_captions=1,
            caption_types='long'
        )

        print(f"Loaded {len(raw_samples)} test samples")

        # Get cluster assignments
        print("\nGetting cluster assignments...")
        cluster_assignments = []

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_fn
        )

        with torch.no_grad():
            for batch in dataloader:
                input_data = self._move_to_device(batch['input_data'])
                mask = batch['mask'].to(self.device)

                logits = self.model(input_data=input_data, attention_mask=mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                cluster_assignments.extend(preds)

        print(f"Got {len(cluster_assignments)} cluster assignments")

        # Organize samples by cluster
        cluster_samples = defaultdict(list)
        cluster_gt_counts = defaultdict(Counter)

        for sample, cluster in zip(raw_samples, cluster_assignments):
            cluster_samples[cluster].append(sample)

            # Track ground truth for majority label
            gt_label = self._get_ground_truth_label(sample)
            cluster_gt_counts[cluster][gt_label] += 1

        # Compute majority labels
        cluster_majority_labels = {}
        for cluster in range(self.num_clusters):
            if cluster_gt_counts[cluster]:
                cluster_majority_labels[cluster] = cluster_gt_counts[cluster].most_common(1)[0][0]
            else:
                cluster_majority_labels[cluster] = 'Unknown'

        # Print cluster statistics
        print("\nCluster sample counts:")
        for cluster in sorted(cluster_samples.keys()):
            majority = cluster_majority_labels[cluster]
            print(f"  Cluster {cluster}: {len(cluster_samples[cluster])} samples (majority: {majority})")

        return dict(cluster_samples), cluster_majority_labels

    def _get_ground_truth_label(self, sample: Dict) -> str:
        """Extract ground truth label from a sample."""
        if 'metadata' in sample and 'ground_truth_labels' in sample['metadata']:
            return sample['metadata']['ground_truth_labels'].get('primary_l1', 'Unknown')
        elif 'first_activity' in sample:
            return sample.get('first_activity', 'Unknown')
        elif 'activity' in sample:
            return sample.get('activity', 'Unknown')
        return 'Unknown'

    def extract_sensor_activations(
        self,
        samples: List[Dict],
        only_on_signals: bool = True
    ) -> List[Tuple[float, float]]:
        """
        Extract sensor coordinates from samples.

        Args:
            samples: List of sample dictionaries
            only_on_signals: If True, only include ON/OPEN signals

        Returns:
            List of (x, y) coordinates
        """
        coordinates = []

        # Limit samples if needed
        if len(samples) > self.max_samples_per_cluster:
            np.random.seed(42)
            indices = np.random.choice(len(samples), self.max_samples_per_cluster, replace=False)
            samples = [samples[i] for i in indices]

        for sample in samples:
            sensor_sequence = sample.get('sensor_sequence', sample.get('events', []))

            for event in sensor_sequence:
                # Get sensor ID
                sensor_id = event.get('sensor_id', event.get('sensor', ''))

                # Filter by signal type if requested
                if only_on_signals:
                    state = str(event.get('state', event.get('event_type', ''))).upper()
                    if state not in ['ON', 'OPEN', '1', 'TRUE']:
                        continue

                # Get coordinates (flip Y from Cartesian to image coords)
                if sensor_id in self.sensor_coords:
                    x, y = self.sensor_coords[sensor_id]
                    y_flipped = self.img_height - y  # Convert from Cartesian (y=0 bottom) to image (y=0 top)
                    coordinates.append((x, y_flipped))

        return coordinates

    def create_cluster_heatmaps(
        self,
        cluster_samples: Dict[int, List[Dict]],
        cluster_majority_labels: Dict[int, str]
    ):
        """Create KDE heatmaps for all clusters."""
        print("\n" + "="*60)
        print("CREATING CLUSTER HEATMAPS")
        print("="*60)

        # Get clusters with samples
        active_clusters = sorted([c for c in cluster_samples.keys() if len(cluster_samples[c]) > 0])

        if not active_clusters:
            print("No clusters with samples found!")
            return

        # Calculate grid dimensions
        n_clusters = len(active_clusters)
        n_cols = min(5, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols

        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_clusters == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, cluster in enumerate(active_clusters):
            print(f"  Processing Cluster {cluster}...")
            ax = axes[idx]

            samples = cluster_samples[cluster]
            majority_label = cluster_majority_labels.get(cluster, 'Unknown')

            # Extract sensor coordinates (ON signals only)
            coords = self.extract_sensor_activations(samples, only_on_signals=True)

            if len(coords) < 3:
                # Not enough points for KDE
                if self.floor_plan_img is not None:
                    # CRITICAL: Use extent to map image to coordinate space
                    ax.imshow(self.floor_plan_img,
                              extent=[0, self.img_width, self.img_height, 0],
                              alpha=0.6, aspect='auto', zorder=0)
                ax.set_xlim(0, self.img_width)
                ax.set_ylim(self.img_height, 0)
                ax.set_title(f'Cluster {cluster}\n({majority_label})\n{len(samples)} samples, <3 ON events',
                            fontsize=9, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]  # Keep original Y coordinates

            # Display floor plan background using image coordinates (y=0 at top)
            # CRITICAL: Use extent to map image to coordinate space (like marble code)
            if self.floor_plan_img is not None:
                ax.imshow(self.floor_plan_img,
                          extent=[0, self.img_width, self.img_height, 0],
                          alpha=0.6, aspect='auto', zorder=0)

            # Create KDE heatmap overlay
            try:
                # Create grid for KDE
                x_grid = np.linspace(0, self.img_width, 100)
                y_grid = np.linspace(0, self.img_height, 100)
                X, Y = np.meshgrid(x_grid, y_grid)
                positions = np.vstack([X.ravel(), Y.ravel()])

                # Compute KDE
                values = np.vstack([x_coords, y_coords])
                kernel = stats.gaussian_kde(values, bw_method=0.15)
                Z = np.reshape(kernel(positions).T, X.shape)

                # Plot KDE as contour lines only (not filled) to preserve floor plan visibility
                levels = np.linspace(Z.max() * 0.1, Z.max(), 8)
                ax.contour(X, Y, Z, levels=levels, cmap='Reds', linewidths=1.5, alpha=0.8)

                # Add scatter points
                ax.scatter(x_coords, y_coords, c='red', s=8, alpha=0.7, edgecolors='darkred', linewidths=0.5)

            except Exception as e:
                print(f"    KDE failed: {e}, using scatter plot")
                ax.scatter(x_coords, y_coords, c='red', s=12, alpha=0.7, edgecolors='darkred', linewidths=0.5)

            # Add sensor labels (flip Y from Cartesian to image coords)
            for sensor_id, (sx, sy) in self.sensor_coords.items():
                sy_flipped = self.img_height - sy
                ax.annotate(sensor_id, (sx, sy_flipped), fontsize=4, alpha=0.6,
                           ha='center', va='bottom', color='blue')

            # Image coordinates: y=0 at top, increases downward
            ax.set_xlim(0, self.img_width)
            ax.set_ylim(self.img_height, 0)  # Inverted so y=0 at top like image

            ax.set_title(f'Cluster {cluster}\n({majority_label})\n{len(samples)} samples, {len(coords)} events',
                        fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        for idx in range(len(active_clusters), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'SCAN Cluster Spatial Heatmaps - {self.dataset_name.capitalize()}\n'
                    f'Sensor Activation Density (ON signals only)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save
        output_path = self.output_dir / 'cluster_heatmaps.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"\nSaved: {output_path}")

        # Also create individual cluster heatmaps
        self._create_individual_heatmaps(cluster_samples, cluster_majority_labels)

    def _create_individual_heatmaps(
        self,
        cluster_samples: Dict[int, List[Dict]],
        cluster_majority_labels: Dict[int, str]
    ):
        """Create individual high-resolution heatmaps for each cluster."""
        print("\nCreating individual cluster heatmaps...")

        individual_dir = self.output_dir / 'cluster_heatmaps_individual'
        individual_dir.mkdir(exist_ok=True)

        for cluster, samples in cluster_samples.items():
            if len(samples) == 0:
                continue

            majority_label = cluster_majority_labels.get(cluster, 'Unknown')
            coords = self.extract_sensor_activations(samples, only_on_signals=True)

            fig, ax = plt.subplots(figsize=(12, 10))

            # Display floor plan background (with extent like marble code)
            if self.floor_plan_img is not None:
                ax.imshow(self.floor_plan_img,
                          extent=[0, self.img_width, self.img_height, 0],
                          alpha=0.6, aspect='auto', zorder=0)

            if len(coords) >= 3:
                x_coords = [c[0] for c in coords]
                y_coords = [c[1] for c in coords]  # Keep original Y coordinates

                try:
                    x_grid = np.linspace(0, self.img_width, 150)
                    y_grid = np.linspace(0, self.img_height, 150)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    positions = np.vstack([X.ravel(), Y.ravel()])

                    values = np.vstack([x_coords, y_coords])
                    kernel = stats.gaussian_kde(values, bw_method=0.12)
                    Z = np.reshape(kernel(positions).T, X.shape)

                    # Use contour lines instead of filled contours
                    levels = np.linspace(Z.max() * 0.1, Z.max(), 10)
                    contour = ax.contour(X, Y, Z, levels=levels, cmap='Reds', linewidths=2, alpha=0.8)
                    plt.colorbar(contour, ax=ax, label='Density')

                    ax.scatter(x_coords, y_coords, c='red', s=15, alpha=0.7, edgecolors='darkred', linewidths=0.5)

                except Exception as e:
                    ax.scatter([c[0] for c in coords], [c[1] for c in coords],
                              c='red', s=20, alpha=0.7, edgecolors='darkred', linewidths=0.5)

            # Add sensor labels (flip Y from Cartesian to image coords)
            for sensor_id, (sx, sy) in self.sensor_coords.items():
                sy_flipped = self.img_height - sy
                ax.annotate(sensor_id, (sx, sy_flipped), fontsize=7, alpha=0.8,
                           ha='center', va='bottom', fontweight='bold', color='blue')

            # Image coordinates: y=0 at top
            ax.set_xlim(0, self.img_width)
            ax.set_ylim(self.img_height, 0)

            ax.set_title(f'Cluster {cluster} - {majority_label}\n'
                        f'{len(samples)} samples, {len(coords)} ON events',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')

            output_path = individual_dir / f'cluster_{cluster:02d}_{majority_label.replace(" ", "_")}.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

        print(f"Saved individual heatmaps to: {individual_dir}")

    def run(self):
        """Run the full heatmap generation pipeline."""
        print("\n" + "="*70)
        print("SCAN CLUSTER HEATMAP GENERATION")
        print("="*70)

        self.load_model()
        cluster_samples, cluster_majority_labels = self.load_data_and_get_clusters()
        self.create_cluster_heatmaps(cluster_samples, cluster_majority_labels)

        print("\n" + "="*70)
        print("HEATMAP GENERATION COMPLETED!")
        print(f"Results saved to: {self.output_dir}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Create spatial density heatmaps for SCAN clusters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python src/evals/scan/create_cluster_heatmaps.py \\
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1

    # With custom data directory
    python src/evals/scan/create_cluster_heatmaps.py \\
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \\
        --data_dir data/processed/casas/milan/FL_20

    # Limit samples for faster processing
    python src/evals/scan/create_cluster_heatmaps.py \\
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \\
        --max_samples_per_cluster 500
        """
    )

    parser.add_argument('--model_dir', type=str, required=True,
                       help='Path to trained SCAN model directory')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to data directory (auto-detected if not provided)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (auto-generated if not provided)')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                       help='Checkpoint file name (default: best_model.pt)')
    parser.add_argument('--max_samples_per_cluster', type=int, default=1000,
                       help='Maximum samples per cluster (default: 1000)')

    args = parser.parse_args()

    generator = ClusterHeatmapGenerator(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint,
        max_samples_per_cluster=args.max_samples_per_cluster
    )

    generator.run()


if __name__ == '__main__':
    main()

