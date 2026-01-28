#!/usr/bin/env python3

"""
Longitudinal analysis of activity distributions over time.
Compares activity patterns between the first N weeks and last N weeks of data.

Usage:
    # Basic usage (auto-detects first/last 2 weeks)
    python src/evals/long/temporal_distribution.py \
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1

    # Specify number of weeks to compare
    python src/evals/long/temporal_distribution.py \
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \
        --weeks 2

    # Custom week start dates (compare week starting Nov 25 vs week starting Dec 23)
    python src/evals/long/temporal_distribution.py \
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \
        --first_start 2009-11-25 \
        --last_start 2009-12-23 \
        --weeks 1

    # Custom data directory
    python src/evals/long/temporal_distribution.py \
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \
        --data_dir data/processed/casas/milan/FL_20
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from models.scan_model import SCANClusteringModel
from dataio.dataset import SmartHomeDataset
from utils.device_utils import get_optimal_device


class TemporalDistributionAnalyzer:
    """
    Analyzer for comparing activity distributions across different time periods.
    """

    def __init__(
        self,
        model_dir: str,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        checkpoint_name: str = 'best_model.pt',
        weeks: int = 2,
        first_start: Optional[str] = None,
        last_start: Optional[str] = None,
        keep_all_labels: bool = False
    ):
        """
        Initialize temporal distribution analyzer.

        Args:
            model_dir: Path to trained SCAN model directory
            data_dir: Path to data directory (auto-detected if None)
            output_dir: Output directory (auto-generated if None)
            checkpoint_name: Checkpoint file name
            weeks: Number of weeks to compare (first N vs last N)
            first_start: Optional start date for first period (YYYY-MM-DD format)
            last_start: Optional start date for last period (YYYY-MM-DD format)
            keep_all_labels: If True, include all labels including No_Activity/Unknown
        """
        self.model_dir = Path(model_dir)
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = self.model_dir / checkpoint_name
        self.weeks = weeks
        self.keep_all_labels = keep_all_labels
        self.label_suffix = '_all_labels' if keep_all_labels else ''

        # Parse custom start dates if provided
        self.first_start = datetime.strptime(first_start, '%Y-%m-%d') if first_start else None
        self.last_start = datetime.strptime(last_start, '%Y-%m-%d') if last_start else None

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

        # Setup output directory
        if output_dir is None:
            # Extract dataset info
            data_parts = self.data_dir.parts
            dataset_name = 'unknown'

            for i, part in enumerate(data_parts):
                if part == 'casas' and i + 1 < len(data_parts):
                    dataset_name = data_parts[i + 1]
                    break

            model_name = self.model_dir.name
            self.output_dir = Path('results/scan') / dataset_name / model_name
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

        # Load label colors
        self._load_label_colors()

        # Initialize components
        self.model = None
        self.raw_samples = []  # Raw JSON samples with timestamps
        self.datasets = {}  # SmartHomeDataset for each split

    def _load_label_colors(self):
        """Load label colors from metadata."""
        try:
            metadata_path = Path(__file__).parent.parent.parent.parent / "metadata" / "casas_metadata.json"
            with open(metadata_path, 'r') as f:
                city_metadata = json.load(f)

            # Detect dataset
            dataset_name = 'milan'
            data_str = str(self.data_dir).lower()
            if 'aruba' in data_str:
                dataset_name = 'aruba'
            elif 'cairo' in data_str:
                dataset_name = 'cairo'

            dataset_metadata = city_metadata.get(dataset_name, {})
            self.label_colors = dataset_metadata.get('label',
                                dataset_metadata.get('label_color', {}))

            if self.label_colors:
                print(f"Loaded {len(self.label_colors)} label colors from {dataset_name} metadata")
            else:
                self.label_colors = {}

        except Exception as e:
            print(f"Could not load label colors: {e}")
            self.label_colors = {}

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

    def load_all_data(self):
        """Load all data and extract timestamps."""
        print("\n" + "="*60)
        print("LOADING ALL DATA")
        print("="*60)

        splits = ['train', 'val', 'test']
        self.raw_samples = []

        # Load raw JSON for timestamp extraction
        for split in splits:
            data_path = self.data_dir / f'{split}.json'
            if data_path.exists():
                print(f"Loading {split} data from: {data_path}")
                with open(data_path, 'r') as f:
                    data = json.load(f)

                samples = data.get('samples', data if isinstance(data, list) else [])
                print(f"  Found {len(samples)} samples in {split}")

                # Extract timestamp info and store raw sample reference
                for idx, sample in enumerate(samples):
                    ts_info = self._extract_timestamp(sample)
                    if ts_info:
                        self.raw_samples.append({
                            'datetime': ts_info,
                            'split': split,
                            'idx': idx,
                            'sample': sample
                        })

                # Also create SmartHomeDataset for this split
                self.datasets[split] = SmartHomeDataset(
                    data_path=str(data_path),
                    vocab_path=str(self.vocab_path),
                    sequence_length=self.sequence_length,
                    max_captions=1,
                    caption_types='long'
                )
            else:
                print(f"  {split}.json not found, skipping")

        print(f"\nTotal samples with valid timestamps: {len(self.raw_samples)}")

        if self.raw_samples:
            # Get date range
            dates = [s['datetime'] for s in self.raw_samples]
            min_date = min(dates)
            max_date = max(dates)
            date_range = (max_date - min_date).days
            print(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({date_range} days)")

    def _extract_timestamp(self, sample: Dict) -> Optional[datetime]:
        """Extract timestamp from a sample's first event."""
        sensor_sequence = sample.get('sensor_sequence', sample.get('events', []))

        if not sensor_sequence:
            return None

        first_event = sensor_sequence[0]
        timestamp = first_event.get('timestamp') or first_event.get('datetime')

        if timestamp is None:
            return None

        # Parse timestamp
        try:
            if isinstance(timestamp, str):
                # Try various formats
                for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                    try:
                        dt = datetime.strptime(timestamp.split('.')[0], fmt.split('.')[0])
                        return dt
                    except ValueError:
                        continue
                # Try dateutil parser
                from dateutil import parser
                return parser.parse(timestamp)
            elif isinstance(timestamp, (int, float)):
                # Unix timestamp (possibly in milliseconds)
                if timestamp > 1e12:
                    timestamp = timestamp / 1000
                return datetime.fromtimestamp(timestamp)
        except Exception:
            pass

        return None

    def _get_ground_truth_label(self, sample: Dict) -> str:
        """Extract ground truth label from a sample."""
        if 'metadata' in sample and 'ground_truth_labels' in sample['metadata']:
            return sample['metadata']['ground_truth_labels'].get('primary_l1', 'Unknown')
        elif 'first_activity' in sample:
            return sample.get('first_activity', 'Unknown')
        elif 'activity' in sample:
            return sample.get('activity', 'Unknown')
        return 'Unknown'

    def _move_to_device(self, obj):
        """Recursively move tensors to device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._move_to_device(item) for item in obj]
        return obj

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

    def filter_by_time_period(self, period: str) -> Tuple[List[Dict], str]:
        """
        Filter raw samples by time period.

        Args:
            period: 'first' or 'last'

        Returns:
            Tuple of (filtered samples, date range string)
        """
        if not self.raw_samples:
            return [], ""

        dates = [s['datetime'] for s in self.raw_samples]
        min_date = min(dates)
        max_date = max(dates)

        weeks_delta = timedelta(weeks=self.weeks)

        if period == 'first':
            # Use custom start date if provided, otherwise use data start
            start = self.first_start if self.first_start else min_date
            end = start + weeks_delta
            filtered = [s for s in self.raw_samples if start <= s['datetime'] <= end]
        else:  # 'last'
            # Use custom start date if provided, otherwise use data end minus weeks
            if self.last_start:
                start = self.last_start
                end = start + weeks_delta
            else:
                end = max_date
                start = end - weeks_delta
            filtered = [s for s in self.raw_samples if start <= s['datetime'] <= end]

        # Compute actual date range from filtered samples
        if filtered:
            filtered_dates = [s['datetime'] for s in filtered]
            date_range = f"{min(filtered_dates).strftime('%Y-%m-%d')} to {max(filtered_dates).strftime('%Y-%m-%d')}"
        else:
            date_range = "No data"

        return filtered, date_range

    def get_predictions_for_samples(self, sample_infos: List[Dict]) -> Tuple[List[str], List[int]]:
        """
        Get ground truth labels and cluster predictions for a list of samples.

        Args:
            sample_infos: List of sample info dicts with 'split', 'idx', 'sample' keys

        Returns:
            Tuple of (ground_truth_labels, cluster_predictions)
        """
        # Filter out No_Activity samples (unless keep_all_labels is True)
        if self.keep_all_labels:
            valid_infos = [(info, self._get_ground_truth_label(info['sample'])) for info in sample_infos]
        else:
            no_activity_variations = ['No_Activity', 'no_activity', 'No Activity', 'no activity',
                                       'Unknown', 'unknown', '', 'Other_Activity', 'other_activity']

            valid_infos = []
            for info in sample_infos:
                gt = self._get_ground_truth_label(info['sample'])
                if gt.strip().lower() not in [var.lower() for var in no_activity_variations]:
                    valid_infos.append((info, gt))

        if not valid_infos:
            return [], []

        ground_truth_labels = []
        cluster_predictions = []

        # Group by split for efficient dataset access
        by_split = {}
        for info, gt in valid_infos:
            split = info['split']
            if split not in by_split:
                by_split[split] = []
            by_split[split].append((info['idx'], gt))

        # Process each split
        for split, idx_gt_pairs in by_split.items():
            if split not in self.datasets:
                continue

            dataset = self.datasets[split]
            indices = [idx for idx, _ in idx_gt_pairs]
            gts = [gt for _, gt in idx_gt_pairs]

            # Create subset dataset
            subset = Subset(dataset, indices)

            dataloader = DataLoader(
                dataset=subset,
                batch_size=64,
                shuffle=False,
                num_workers=0,
                collate_fn=self._collate_fn
            )

            # Get predictions
            batch_gts = []
            with torch.no_grad():
                for batch in dataloader:
                    input_data = self._move_to_device(batch['input_data'])
                    mask = batch['mask'].to(self.device)

                    # Get cluster predictions
                    logits = self.model(input_data=input_data, attention_mask=mask)
                    preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

                    cluster_predictions.extend(preds)
                    batch_gts.extend(batch['gt_labels'])

            ground_truth_labels.extend(gts)

        return ground_truth_labels, cluster_predictions

    def create_temporal_comparison(self):
        """Create the temporal comparison visualization."""
        # Build description of comparison
        if self.first_start or self.last_start:
            first_desc = self.first_start.strftime('%Y-%m-%d') if self.first_start else "data start"
            last_desc = self.last_start.strftime('%Y-%m-%d') if self.last_start else "data end"
            print("\n" + "="*60)
            print(f"CREATING TEMPORAL COMPARISON")
            print(f"First period: {self.weeks} week(s) starting {first_desc}")
            print(f"Last period: {self.weeks} week(s) starting {last_desc}")
            print("="*60)
        else:
            print("\n" + "="*60)
            print(f"CREATING TEMPORAL COMPARISON (First {self.weeks} weeks vs Last {self.weeks} weeks)")
            print("="*60)

        # Filter samples
        first_period, first_range = self.filter_by_time_period('first')
        last_period, last_range = self.filter_by_time_period('last')

        print(f"First period: {len(first_period)} samples ({first_range})")
        print(f"Last period: {len(last_period)} samples ({last_range})")

        if not first_period or not last_period:
            print("ERROR: Not enough data in one or both periods")
            return

        # Get ground truth labels and cluster predictions
        print("\nProcessing first period samples...")
        first_gt, first_clusters = self.get_predictions_for_samples(first_period)
        print(f"  Got {len(first_gt)} valid samples with predictions")

        print("Processing last period samples...")
        last_gt, last_clusters = self.get_predictions_for_samples(last_period)
        print(f"  Got {len(last_gt)} valid samples with predictions")

        if not first_gt or not last_gt:
            print("ERROR: No valid samples found for analysis")
            return

        # Create visualizations
        self._create_distribution_plots(
            first_gt, last_gt,
            first_clusters, last_clusters,
            first_range, last_range
        )

    def _create_distribution_plots(
        self,
        first_gt: List[str],
        last_gt: List[str],
        first_clusters: List[int],
        last_clusters: List[int],
        first_range: str,
        last_range: str
    ):
        """Create the 3x4 distribution comparison plots with delta row."""

        # Create figure with 3x4 subplots
        fig, axes = plt.subplots(3, 4, figsize=(36, 18))

        # Get all unique labels and clusters
        all_gt_labels = sorted(set(first_gt + last_gt))
        all_clusters = sorted(set(first_clusters + last_clusters))

        # Count distributions
        first_gt_counts = Counter(first_gt)
        last_gt_counts = Counter(last_gt)
        first_cluster_counts = Counter(first_clusters)
        last_cluster_counts = Counter(last_clusters)

        # Build cluster->GT mapping for stacked charts
        first_cluster_gt = self._build_cluster_gt_mapping(first_clusters, first_gt, all_clusters, all_gt_labels)
        last_cluster_gt = self._build_cluster_gt_mapping(last_clusters, last_gt, all_clusters, all_gt_labels)

        # Compute majority labels for each cluster (using combined data for consistency)
        combined_cluster_gt = {cl: Counter() for cl in all_clusters}
        for cl, gt in zip(first_clusters + last_clusters, first_gt + last_gt):
            combined_cluster_gt[cl][gt] += 1
        cluster_majority_label = self._get_cluster_majority_labels(combined_cluster_gt, all_clusters)

        # Map samples to majority labels
        first_majority = [cluster_majority_label[cl] for cl in first_clusters]
        last_majority = [cluster_majority_label[cl] for cl in last_clusters]
        first_majority_counts = Counter(first_majority)
        last_majority_counts = Counter(last_majority)

        # ========================================
        # Row 1: First N weeks
        # ========================================

        # Col 1: Ground Truth (colored)
        self._plot_histogram(
            axes[0, 0], first_gt_counts, all_gt_labels,
            f"First {self.weeks} Weeks - Ground Truth\n({first_range})",
            is_cluster=False
        )

        # Col 2: Clusters (grey)
        self._plot_histogram(
            axes[0, 1], first_cluster_counts, all_clusters,
            f"First {self.weeks} Weeks - SCAN Clusters\n({first_range})",
            is_cluster=True, use_grey=True
        )

        # Col 3: Clusters stacked by GT
        self._plot_stacked_cluster_gt(
            axes[0, 2], first_cluster_gt, all_clusters, all_gt_labels,
            f"First {self.weeks} Weeks - Clusters by Activity\n({first_range})"
        )

        # Col 4: Majority label assignment
        self._plot_histogram(
            axes[0, 3], first_majority_counts, all_gt_labels,
            f"First {self.weeks} Weeks - Majority Label\n({first_range})",
            is_cluster=False
        )

        # ========================================
        # Row 2: Last N weeks
        # ========================================

        # Col 1: Ground Truth (colored)
        self._plot_histogram(
            axes[1, 0], last_gt_counts, all_gt_labels,
            f"Last {self.weeks} Weeks - Ground Truth\n({last_range})",
            is_cluster=False
        )

        # Col 2: Clusters (grey)
        self._plot_histogram(
            axes[1, 1], last_cluster_counts, all_clusters,
            f"Last {self.weeks} Weeks - SCAN Clusters\n({last_range})",
            is_cluster=True, use_grey=True
        )

        # Col 3: Clusters stacked by GT
        self._plot_stacked_cluster_gt(
            axes[1, 2], last_cluster_gt, all_clusters, all_gt_labels,
            f"Last {self.weeks} Weeks - Clusters by Activity\n({last_range})"
        )

        # Col 4: Majority label assignment
        self._plot_histogram(
            axes[1, 3], last_majority_counts, all_gt_labels,
            f"Last {self.weeks} Weeks - Majority Label\n({last_range})",
            is_cluster=False
        )

        # ========================================
        # Row 3: Delta (change over time)
        # ========================================

        # Compute deltas (normalized to compare fairly)
        first_total = len(first_gt)
        last_total = len(last_gt)

        # Col 1: GT delta
        gt_delta = {label: (last_gt_counts.get(label, 0) / last_total * 100) -
                          (first_gt_counts.get(label, 0) / first_total * 100)
                   for label in all_gt_labels}
        self._plot_delta_chart(
            axes[2, 0], gt_delta, all_gt_labels,
            f"Change in Ground Truth\n(Last - First, pp)",
            is_cluster=False
        )

        # Col 2: Cluster delta
        cluster_delta = {cl: (last_cluster_counts.get(cl, 0) / last_total * 100) -
                            (first_cluster_counts.get(cl, 0) / first_total * 100)
                        for cl in all_clusters}
        self._plot_delta_chart(
            axes[2, 1], cluster_delta, all_clusters,
            f"Change in Clusters\n(Last - First, pp)",
            is_cluster=True
        )

        # Col 3: Stacked cluster delta (composition change)
        self._plot_stacked_delta(
            axes[2, 2], first_cluster_gt, last_cluster_gt, all_clusters, all_gt_labels,
            f"Change in Cluster Composition\n(Last - First, count)"
        )

        # Col 4: Majority label delta
        majority_delta = {label: (last_majority_counts.get(label, 0) / last_total * 100) -
                                (first_majority_counts.get(label, 0) / first_total * 100)
                        for label in all_gt_labels}
        self._plot_delta_chart(
            axes[2, 3], majority_delta, all_gt_labels,
            f"Change in Majority Label\n(Last - First, pp)",
            is_cluster=False
        )

        # Add main title
        fig.suptitle(
            f'Temporal Activity Distribution Comparison\n'
            f'First {self.weeks} Weeks ({len(first_gt)} samples) vs Last {self.weeks} Weeks ({len(last_gt)} samples)',
            fontsize=18, fontweight='bold', y=1.02
        )

        plt.tight_layout()

        # Save plot
        output_path = self.output_dir / f'temporal_distribution_{self.weeks}weeks{self.label_suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"\nSaved: {output_path}")

        # Also create normalized version
        self._create_normalized_plots(
            first_gt_counts, last_gt_counts,
            first_cluster_counts, last_cluster_counts,
            all_gt_labels, all_clusters,
            first_range, last_range,
            len(first_gt), len(last_gt),
            first_cluster_gt, last_cluster_gt,
            first_majority_counts, last_majority_counts,
            cluster_majority_label
        )

        # Save statistics
        self._save_statistics(
            first_gt_counts, last_gt_counts,
            first_cluster_counts, last_cluster_counts,
            first_range, last_range,
            cluster_majority_label
        )

    def _get_cluster_majority_labels(
        self,
        cluster_gt: Dict[int, Counter],
        all_clusters: List[int]
    ) -> Dict[int, str]:
        """Get the majority ground truth label for each cluster."""
        majority_labels = {}
        for cl in all_clusters:
            if cluster_gt[cl]:
                majority_labels[cl] = cluster_gt[cl].most_common(1)[0][0]
            else:
                majority_labels[cl] = 'Unknown'
        return majority_labels

    def _build_cluster_gt_mapping(
        self,
        clusters: List[int],
        gt_labels: List[str],
        all_clusters: List[int],
        all_gt_labels: List[str]
    ) -> Dict[int, Counter]:
        """Build mapping from cluster to ground truth label counts."""
        cluster_gt = {cl: Counter() for cl in all_clusters}
        for cl, gt in zip(clusters, gt_labels):
            cluster_gt[cl][gt] += 1
        return cluster_gt

    def _plot_stacked_cluster_gt(
        self,
        ax,
        cluster_gt: Dict[int, Counter],
        all_clusters: List[int],
        all_gt_labels: List[str],
        title: str
    ):
        """Plot stacked bar chart of clusters colored by ground truth labels (raw counts)."""
        x_pos = np.arange(len(all_clusters))
        x_labels = [f'C{c}' for c in all_clusters]

        # Build stacked data with raw counts
        bottom = np.zeros(len(all_clusters))

        for gt_label in all_gt_labels:
            values = np.array([cluster_gt[cl].get(gt_label, 0) for cl in all_clusters])
            color = self.label_colors.get(gt_label, plt.cm.tab20(all_gt_labels.index(gt_label) % 20))
            ax.bar(x_pos, values, bottom=bottom, label=gt_label.replace('_', ' '),
                   color=color, edgecolor='white', linewidth=0.3)
            bottom += values

        ax.set_xlabel('Cluster', fontsize=11)
        ax.set_ylabel('Frequency (Count)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=7, ncol=2)

    def _plot_delta_chart(
        self,
        ax,
        delta: Dict,
        labels: List,
        title: str,
        is_cluster: bool = False
    ):
        """Plot delta (change) chart with positive/negative bars."""
        if is_cluster:
            x_labels = [f'C{c}' for c in labels]
        else:
            x_labels = [str(l).replace('_', ' ') for l in labels]

        values = [delta.get(l, 0) for l in labels]
        x_pos = np.arange(len(labels))

        # Color based on positive/negative
        colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]

        bars = ax.bar(x_pos, values, color=colors, edgecolor='white', linewidth=0.5)

        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xlabel('Activity' if not is_cluster else 'Cluster', fontsize=11)
        ax.set_ylabel('Change (pp)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, values):
            if abs(val) > 0.3:
                va = 'bottom' if val >= 0 else 'top'
                ax.annotate(f'{val:+.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va=va, fontsize=7)

    def _plot_stacked_delta(
        self,
        ax,
        first_cluster_gt: Dict[int, Counter],
        last_cluster_gt: Dict[int, Counter],
        all_clusters: List[int],
        all_gt_labels: List[str],
        title: str
    ):
        """Plot stacked delta chart showing count changes per cluster by GT label."""
        x_pos = np.arange(len(all_clusters))
        x_labels = [f'C{c}' for c in all_clusters]

        # Compute raw count deltas (last - first)
        positive_bottom = np.zeros(len(all_clusters))
        negative_bottom = np.zeros(len(all_clusters))

        for gt_label in all_gt_labels:
            deltas = []
            for cl in all_clusters:
                first_count = first_cluster_gt[cl].get(gt_label, 0)
                last_count = last_cluster_gt[cl].get(gt_label, 0)
                deltas.append(last_count - first_count)

            color = self.label_colors.get(gt_label, plt.cm.tab20(all_gt_labels.index(gt_label) % 20))

            # Split into positive and negative
            pos_vals = np.array([max(0, d) for d in deltas])
            neg_vals = np.array([min(0, d) for d in deltas])

            if any(v > 0 for v in pos_vals):
                ax.bar(x_pos, pos_vals, bottom=positive_bottom, label=gt_label.replace('_', ' '),
                       color=color, edgecolor='white', linewidth=0.3)
                positive_bottom += pos_vals

            if any(v < 0 for v in neg_vals):
                ax.bar(x_pos, neg_vals, bottom=negative_bottom,
                       color=color, edgecolor='white', linewidth=0.3)
                negative_bottom += neg_vals

        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xlabel('Cluster', fontsize=11)
        ax.set_ylabel('Change (Count)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=6, ncol=2)

    def _plot_histogram(
        self,
        ax,
        counts: Counter,
        labels: List,
        title: str,
        is_cluster: bool = False,
        use_grey: bool = False
    ):
        """Plot a single histogram."""

        if is_cluster:
            x_labels = [f'C{c}' for c in labels]
            if use_grey:
                colors = ['#7f8c8d'] * len(labels)  # Grey for clusters
            else:
                colors = [plt.cm.tab20(i % 20) for i in range(len(labels))]
        else:
            x_labels = [str(l).replace('_', ' ') for l in labels]
            colors = [self.label_colors.get(l, plt.cm.tab20(i % 20)) for i, l in enumerate(labels)]

        values = [counts.get(l, 0) for l in labels]

        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, values, color=colors, edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Activity' if not is_cluster else 'Cluster', fontsize=11)
        ax.set_ylabel('Frequency (Count)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add count labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8)

    def _create_normalized_plots(
        self,
        first_gt: Counter,
        last_gt: Counter,
        first_clusters: Counter,
        last_clusters: Counter,
        all_gt_labels: List,
        all_clusters: List,
        first_range: str,
        last_range: str,
        first_total: int,
        last_total: int,
        first_cluster_gt: Optional[Dict[int, Counter]] = None,
        last_cluster_gt: Optional[Dict[int, Counter]] = None,
        first_majority: Optional[Counter] = None,
        last_majority: Optional[Counter] = None,
        cluster_majority_label: Optional[Dict[int, str]] = None
    ):
        """Create normalized (percentage) comparison plots with 3x4 layout."""

        fig, axes = plt.subplots(3, 4, figsize=(36, 18))

        # Normalize to percentages
        first_gt_pct = {k: 100*v/first_total for k, v in first_gt.items()}
        last_gt_pct = {k: 100*v/last_total for k, v in last_gt.items()}
        first_cluster_pct = {k: 100*v/first_total for k, v in first_clusters.items()}
        last_cluster_pct = {k: 100*v/last_total for k, v in last_clusters.items()}
        first_majority_pct = {k: 100*v/first_total for k, v in first_majority.items()} if first_majority else {}
        last_majority_pct = {k: 100*v/last_total for k, v in last_majority.items()} if last_majority else {}

        # ========================================
        # Row 1: First N weeks (normalized)
        # ========================================
        self._plot_normalized_histogram(axes[0, 0], first_gt_pct, all_gt_labels,
            f"First {self.weeks} Weeks - Ground Truth\n({first_range})", is_cluster=False)
        self._plot_normalized_histogram(axes[0, 1], first_cluster_pct, all_clusters,
            f"First {self.weeks} Weeks - SCAN Clusters\n({first_range})", is_cluster=True, use_grey=True)
        if first_cluster_gt:
            self._plot_stacked_cluster_gt_normalized(axes[0, 2], first_cluster_gt, all_clusters, all_gt_labels,
                f"First {self.weeks} Weeks - Clusters by Activity\n({first_range})")
        self._plot_normalized_histogram(axes[0, 3], first_majority_pct, all_gt_labels,
            f"First {self.weeks} Weeks - Majority Label\n({first_range})", is_cluster=False)

        # ========================================
        # Row 2: Last N weeks (normalized)
        # ========================================
        self._plot_normalized_histogram(axes[1, 0], last_gt_pct, all_gt_labels,
            f"Last {self.weeks} Weeks - Ground Truth\n({last_range})", is_cluster=False)
        self._plot_normalized_histogram(axes[1, 1], last_cluster_pct, all_clusters,
            f"Last {self.weeks} Weeks - SCAN Clusters\n({last_range})", is_cluster=True, use_grey=True)
        if last_cluster_gt:
            self._plot_stacked_cluster_gt_normalized(axes[1, 2], last_cluster_gt, all_clusters, all_gt_labels,
                f"Last {self.weeks} Weeks - Clusters by Activity\n({last_range})")
        self._plot_normalized_histogram(axes[1, 3], last_majority_pct, all_gt_labels,
            f"Last {self.weeks} Weeks - Majority Label\n({last_range})", is_cluster=False)

        # ========================================
        # Row 3: Delta (change over time)
        # ========================================
        gt_delta = {label: last_gt_pct.get(label, 0) - first_gt_pct.get(label, 0)
                   for label in all_gt_labels}
        cluster_delta = {cl: last_cluster_pct.get(cl, 0) - first_cluster_pct.get(cl, 0)
                        for cl in all_clusters}
        majority_delta = {label: last_majority_pct.get(label, 0) - first_majority_pct.get(label, 0)
                        for label in all_gt_labels}

        self._plot_delta_chart(axes[2, 0], gt_delta, all_gt_labels,
            f"Change in Ground Truth\n(Last - First, pp)", is_cluster=False)
        self._plot_delta_chart(axes[2, 1], cluster_delta, all_clusters,
            f"Change in Clusters\n(Last - First, pp)", is_cluster=True)
        if first_cluster_gt and last_cluster_gt:
            self._plot_stacked_delta(axes[2, 2], first_cluster_gt, last_cluster_gt,
                all_clusters, all_gt_labels, f"Change in Cluster Composition\n(Last - First, count)")
        self._plot_delta_chart(axes[2, 3], majority_delta, all_gt_labels,
            f"Change in Majority Label\n(Last - First, pp)", is_cluster=False)

        fig.suptitle(
            f'Temporal Activity Distribution (Normalized)\n'
            f'First {self.weeks} Weeks vs Last {self.weeks} Weeks',
            fontsize=18, fontweight='bold', y=1.02
        )

        plt.tight_layout()

        output_path = self.output_dir / f'temporal_distribution_{self.weeks}weeks_normalized{self.label_suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {output_path}")

    def _plot_normalized_histogram(
        self, ax, counts: Dict, labels: List, title: str,
        is_cluster: bool = False, use_grey: bool = False
    ):
        """Plot a normalized histogram with percentages."""
        if is_cluster:
            x_labels = [f'C{c}' for c in labels]
            colors = ['#7f8c8d'] * len(labels) if use_grey else [plt.cm.tab20(j % 20) for j in range(len(labels))]
        else:
            x_labels = [str(l).replace('_', ' ') for l in labels]
            colors = [self.label_colors.get(l, plt.cm.tab20(j % 20)) for j, l in enumerate(labels)]

        values = [counts.get(l, 0) for l in labels]
        x_pos = np.arange(len(labels))

        bars = ax.bar(x_pos, values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Activity' if not is_cluster else 'Cluster', fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            if val > 0.5:
                ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=7)

    def _plot_stacked_cluster_gt_normalized(
        self, ax, cluster_gt: Dict[int, Counter], all_clusters: List[int],
        all_gt_labels: List[str], title: str
    ):
        """Plot stacked bar chart with raw counts (same as main plot, broken down by GT)."""
        x_pos = np.arange(len(all_clusters))
        x_labels = [f'C{c}' for c in all_clusters]

        # Use raw counts (not normalized)
        bottom = np.zeros(len(all_clusters))
        for gt_label in all_gt_labels:
            values = np.array([cluster_gt[cl].get(gt_label, 0) for cl in all_clusters])
            color = self.label_colors.get(gt_label, plt.cm.tab20(all_gt_labels.index(gt_label) % 20))
            ax.bar(x_pos, values, bottom=bottom, label=gt_label.replace('_', ' '),
                   color=color, edgecolor='white', linewidth=0.3)
            bottom += values

        ax.set_xlabel('Cluster', fontsize=11)
        ax.set_ylabel('Frequency (Count)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=6, ncol=2)

    def _save_statistics(
        self,
        first_gt: Counter,
        last_gt: Counter,
        first_clusters: Counter,
        last_clusters: Counter,
        first_range: str,
        last_range: str,
        cluster_majority_label: Optional[Dict[int, str]] = None
    ):
        """Save distribution statistics to JSON and text files."""

        stats = {
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': str(self.model_dir.name),
            'weeks_compared': self.weeks,
            'first_period': {
                'date_range': first_range,
                'total_samples': sum(first_gt.values()),
                'ground_truth_distribution': dict(first_gt),
                'cluster_distribution': {f'Cluster_{k}': v for k, v in first_clusters.items()}
            },
            'last_period': {
                'date_range': last_range,
                'total_samples': sum(last_gt.values()),
                'ground_truth_distribution': dict(last_gt),
                'cluster_distribution': {f'Cluster_{k}': v for k, v in last_clusters.items()}
            }
        }

        if cluster_majority_label:
            stats['cluster_majority_labels'] = {f'Cluster_{k}': v for k, v in cluster_majority_label.items()}

        # Save JSON
        json_path = self.output_dir / f'temporal_stats_{self.weeks}weeks{self.label_suffix}.json'
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved: {json_path}")

        # Save text report
        txt_path = self.output_dir / f'temporal_stats_{self.weeks}weeks{self.label_suffix}.txt'
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"TEMPORAL DISTRIBUTION ANALYSIS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Model: {self.model_dir.name}\n")
            f.write(f"Weeks compared: {self.weeks}\n\n")

            # Cluster majority label mapping
            if cluster_majority_label:
                f.write("-"*70 + "\n")
                f.write("CLUSTER -> MAJORITY LABEL MAPPING\n")
                f.write("-"*70 + "\n")
                for cl in sorted(cluster_majority_label.keys()):
                    f.write(f"  Cluster {cl}: {cluster_majority_label[cl]}\n")
                f.write("\n")

            f.write("-"*70 + "\n")
            f.write(f"FIRST {self.weeks} WEEKS ({first_range})\n")
            f.write("-"*70 + "\n")
            f.write(f"Total samples: {sum(first_gt.values())}\n\n")
            f.write("Ground Truth Distribution:\n")
            for label, count in sorted(first_gt.items(), key=lambda x: -x[1]):
                pct = 100 * count / sum(first_gt.values())
                f.write(f"  {label}: {count} ({pct:.1f}%)\n")

            f.write("\n")
            f.write("-"*70 + "\n")
            f.write(f"LAST {self.weeks} WEEKS ({last_range})\n")
            f.write("-"*70 + "\n")
            f.write(f"Total samples: {sum(last_gt.values())}\n\n")
            f.write("Ground Truth Distribution:\n")
            for label, count in sorted(last_gt.items(), key=lambda x: -x[1]):
                pct = 100 * count / sum(last_gt.values())
                f.write(f"  {label}: {count} ({pct:.1f}%)\n")

            f.write("\n" + "="*70 + "\n")

        print(f"Saved: {txt_path}")

    def run_analysis(self):
        """Run the full temporal analysis."""
        print("\n" + "="*70)
        print("TEMPORAL DISTRIBUTION ANALYSIS")
        print("="*70)

        self.load_model()
        self.load_all_data()
        self.create_temporal_comparison()

        print("\n" + "="*70)
        print("ANALYSIS COMPLETED!")
        print(f"Results saved to: {self.output_dir}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Temporal distribution analysis for SCAN clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (auto-detect first/last 2 weeks)
    python src/evals/long/temporal_distribution.py \\
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1

    # Compare first/last 4 weeks
    python src/evals/long/temporal_distribution.py \\
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \\
        --weeks 4

    # Custom week start dates (compare week starting Nov 25 vs Dec 23)
    python src/evals/long/temporal_distribution.py \\
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \\
        --first_start 2009-11-25 \\
        --last_start 2009-12-23 \\
        --weeks 1
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
    parser.add_argument('--weeks', type=int, default=2,
                       help='Number of weeks to compare (default: 2)')
    parser.add_argument('--first_start', type=str, default=None,
                       help='Start date for first period (YYYY-MM-DD). If not provided, uses data start.')
    parser.add_argument('--last_start', type=str, default=None,
                       help='Start date for last period (YYYY-MM-DD). If not provided, uses (data end - weeks).')
    parser.add_argument('--keep_all_labels', action='store_true',
                       help='Include all labels (including No_Activity/Unknown). Default: exclude them.')

    args = parser.parse_args()

    analyzer = TemporalDistributionAnalyzer(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint,
        weeks=args.weeks,
        first_start=args.first_start,
        last_start=args.last_start,
        keep_all_labels=args.keep_all_labels
    )

    analyzer.run_analysis()


if __name__ == '__main__':
    main()
