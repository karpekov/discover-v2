#!/usr/bin/env python3
"""
Publication-ready temporal distribution comparison charts for SCAN clusters.

Creates professional 2x2 charts comparing activity distributions between
two time periods (first week vs last week) with horizontal grouped bars.

Sample usage:
MILAN:
python src/evals/scan/temporal_diff_publication.py \
    --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \
    --weeks 1 \
    --first_start 2009-11-25 \
    --last_start 2009-12-23 \
    --label_level l2

ARUBA:
python src/evals/scan/temporal_diff_publication.py \
    --model_dir trained_models/aruba/scan_fl20_20cl_discover_v1 \
    --weeks 1 \
    --first_start 2011-02-10 \
    --last_start 2011-05-20 \
    --label_level l2

CAIRO:
python src/evals/scan/temporal_diff_publication.py \
    --model_dir trained_models/cairo/scan_fl20_20cl_discover_v1 \
    --weeks 1 \
    --first_start 2009-06-10 \
    --last_start 2009-07-25 \
    --label_level l2

Output: 4 chart variants (normalized/raw × with/without no_activity)
"""

import argparse
import json
import sys
import yaml
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# Use a clean, professional font stack
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'text.color': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
})

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.dataio.dataset import SmartHomeDataset
from src.models.scan_model import SCANClusteringModel

# Hard-coded cluster descriptions for each dataset
CLUSTER_TO_NAMES = {
    "milan": {
        0: "Sitting in living room armchair",
        1: "Movement near TV room armchair",
        2: "Desk area movement",
        3: "Movement in master bathroom",
        4: "Walking towards living room armchair",
        5: "Movement between master bedroom and bathroom",
        6: "Movement all over kitchen",
        7: "Movement between dining and living rooms",
        8: "Movement around living room armchair",
        9: "Movement all over kitchen",
        10: "Walking towards TV room armchair",
        11: "Movement between kitchen and living room",
        12: "Sitting in TV room armchair",
        13: "Walking into guest bathroom",
        14: "Movement all over kitchen",
        15: "Movement near medicine cabinet in the kitchen",
        16: "Movement near stove in the kitchen",
        17: "Movement near fridge in the kitchen",
        18: "Bed motion in master bedroom",
        19: "Movement all over the house"
    },
    "aruba": {},
    "cairo": {}
}

# Cluster ordering with group labels (for grouped display)
# Format: list of tuples (group_name, [cluster_ids])
CLUSTER_GROUPS = {
    "milan": [
        ("Kitchen", [6, 9, 14, 16, 17, 15]),
        ("Eat", [7, 11]),
        ("Work", [2]),
        ("Relax", [0, 4, 8, 1, 10, 12]),
        ("Sleep", [18]),
        ("Bathroom", [3, 5, 13]),
        ("General", [19]),
    ],
    "aruba": None,  # Use default (no groups)
    "cairo": None   # Use default (no groups)
}

# Flat cluster order (derived from groups or use default)
def get_cluster_order(dataset_name):
    """Get flat cluster order from groups."""
    groups = CLUSTER_GROUPS.get(dataset_name)
    if groups:
        return [c for _, clusters in groups for c in clusters]
    return None

CLUSTER_ORDER = {
    "milan": get_cluster_order("milan"),
    "aruba": None,
    "cairo": None
}

# Activity label ordering (for ground truth labels)
# L1 = coarse labels, L2 = fine-grained labels
ACTIVITY_ORDER = {
    "milan": {
        "l1": [
            "Kitchen_Activity",
            "Eve_Meds",
            "Dining_Rm_Activity",
            "Read",
            "Watch_TV",
            "Desk_Activity",
            "Master_Bedroom_Activity",
            "Sleep",
            "Master_Bathroom",
            "Bed_to_Toilet",
            "Guest_Bathroom",
            "Leave_Home",
            "Meditate",
            "no_activity",
            "No_Activity",
            "Unknown",
        ],
        "l2": [
            "Cook",
            "Eat",
            "Take_medicine",
            "Work",
            "Relax",
            "Sleep",
            "Bathing",
            "Bed_to_toilet",
            "Leave_Home",
            "Other",
            "No_Activity",
            "no_activity",
            "Unknown",
        ],
    },
    "aruba": None,
    "cairo": None
}


class TemporalDiffPublicationChart:
    """Creates publication-ready temporal distribution comparison charts."""

    # Professional color palette
    COLORS = {
        'week1': '#4C72B0',      # Steel blue
        'week2': '#DD8452',      # Warm orange
        'positive': '#55A868',   # Green for increase
        'negative': '#C44E52',   # Red for decrease
        'neutral': '#8C8C8C',    # Gray
    }

    # Activity colors for consistency
    ACTIVITY_COLORS = {
        'Kitchen_Activity': '#E24A33',
        'Watch_TV': '#348ABD',
        'Sleep': '#988ED5',
        'Master_Bathroom': '#8EBA42',
        'Read': '#FBC15E',
        'Bed_to_Toilet': '#FFB5B8',
        'Leave_Home': '#777777',
        'Enter_Home': '#999999',
        'Work': '#56B4E9',
        'Morning_Meds': '#F0E442',
        'Eve_Meds': '#D55E00',
        'Chores': '#CC79A7',
        'Dining_Rm_Activity': '#E69F00',
        'Guest_Bathroom': '#009E73',
        'no_activity': '#CCCCCC',
        'No_Activity': '#CCCCCC',
        'Unknown': '#AAAAAA',
        'Other_Activity': '#BBBBBB',
    }

    def __init__(
        self,
        model_dir: str,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        weeks: int = 1,
        first_start: Optional[str] = None,
        last_start: Optional[str] = None,
        checkpoint: str = 'best_model.pt',
        device: Optional[str] = None,
        label_level: str = 'l1'
    ):
        self.model_dir = Path(model_dir)
        self.weeks = weeks
        self.first_start = first_start
        self.last_start = last_start
        self.checkpoint = checkpoint
        self.label_level = label_level  # 'l1' or 'l2'

        # Setup device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

        # Load model config (try YAML first, then JSON)
        config_path = self.model_dir / 'config.yaml'
        if not config_path.exists():
            config_path = self.model_dir / 'config.json'

        with open(config_path, 'r') as f:
            if config_path.suffix == '.yaml':
                self.config = yaml.safe_load(f)
            else:
                self.config = json.load(f)

        self.num_clusters = self.config.get('num_clusters', 20)

        # Setup data directory
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            # Extract from train_data_path or data_dir
            train_path = self.config.get('train_data_path', '')
            if train_path:
                self.data_dir = Path(train_path).parent
            else:
                self.data_dir = Path(self.config.get('data_dir', ''))

        self.vocab_path = self.data_dir / 'vocab.json'

        # Detect dataset name
        self.dataset_name = self._detect_dataset_name()

        # Setup output directory
        model_name = self.model_dir.name
        if output_dir is None:
            self.output_dir = Path('results/scan') / self.dataset_name / model_name / 'temporal_diff_pub'
        else:
            self.output_dir = Path(output_dir) / 'temporal_diff_pub'

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

    def _detect_dataset_name(self) -> str:
        """Detect dataset name from data path."""
        data_str = str(self.data_dir).lower()
        for name in ['milan', 'aruba', 'cairo', 'kyoto', 'tulum']:
            if name in data_str:
                return name
        return 'unknown'

    def load_model(self):
        """Load the trained SCAN model."""
        print("\nLoading SCAN model...")
        pretrained_model_path = self.config.get('pretrained_model_path', '')

        self.model = SCANClusteringModel(
            pretrained_model_path=pretrained_model_path,
            num_clusters=self.num_clusters,
            dropout=self.config.get('dropout', 0.1),
            freeze_encoder=True,
            vocab_path=str(self.vocab_path),
            device=self.device
        )

        checkpoint_path = self.model_dir / self.checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded model with {self.num_clusters} clusters")

    def load_all_data(self) -> Tuple[List[Dict], Dict]:
        """Load all raw data splits (train, val, test) with timestamps."""
        print("\nLoading all data splits...")
        all_samples = []
        datasets = {}  # Store SmartHomeDatasets for cluster prediction

        for split in ['train', 'val', 'test']:
            split_path = self.data_dir / f'{split}.json'
            if split_path.exists():
                # Load raw JSON for timestamps
                with open(split_path, 'r') as f:
                    data = json.load(f)

                samples = data.get('samples', data if isinstance(data, list) else [])
                print(f"  Found {len(samples)} raw samples in {split}")

                for idx, sample in enumerate(samples):
                    sample['_split'] = split
                    sample['_split_idx'] = idx
                    all_samples.append(sample)

                # Also load SmartHomeDataset for this split
                datasets[split] = SmartHomeDataset(str(split_path), str(self.vocab_path))

        print(f"Total samples: {len(all_samples)}")
        self.datasets = datasets
        return all_samples

    def _parse_timestamp(self, sample: Dict) -> Optional[datetime]:
        """Extract timestamp from sample's first event."""
        sensor_sequence = sample.get('sensor_sequence', sample.get('events', []))

        if not sensor_sequence:
            return None

        first_event = sensor_sequence[0]
        ts = first_event.get('timestamp') or first_event.get('datetime')

        if ts is None:
            return None

        try:
            if isinstance(ts, str):
                # Try various formats
                for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                    try:
                        return datetime.strptime(ts.split('.')[0], fmt.split('.')[0])
                    except ValueError:
                        continue
                # Try dateutil parser
                from dateutil import parser
                return parser.parse(ts)
            elif isinstance(ts, (int, float)):
                # Unix timestamp
                if ts > 1e12:
                    ts = ts / 1000
                return datetime.fromtimestamp(ts)
        except Exception:
            pass
        return None

    def _get_ground_truth_label(self, sample: Dict) -> str:
        """Extract ground truth label from sample based on label_level (l1 or l2)."""
        if self.label_level == 'l2':
            # Try L2 labels first
            if 'metadata' in sample and 'ground_truth_labels' in sample['metadata']:
                return sample['metadata']['ground_truth_labels'].get('primary_l2', 'Unknown')
            return sample.get('first_activity_l2', sample.get('activity_l2', 'Unknown'))
        else:
            # Default to L1 labels
            if 'metadata' in sample and 'ground_truth_labels' in sample['metadata']:
                return sample['metadata']['ground_truth_labels'].get('primary_l1', 'Unknown')
            return sample.get('first_activity', sample.get('activity', 'Unknown'))

    def filter_by_time_periods(
        self,
        samples: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], Tuple[datetime, datetime], Tuple[datetime, datetime]]:
        """Filter samples into first and last N weeks."""
        # Get timestamps
        samples_with_ts = []
        for sample in samples:
            ts = self._parse_timestamp(sample)
            if ts:
                samples_with_ts.append((sample, ts))

        if not samples_with_ts:
            raise ValueError("No samples with valid timestamps found")

        samples_with_ts.sort(key=lambda x: x[1])
        min_ts = samples_with_ts[0][1]
        max_ts = samples_with_ts[-1][1]

        print(f"\nData range: {min_ts.date()} to {max_ts.date()}")

        # Determine period boundaries
        period_days = self.weeks * 7

        if self.first_start:
            first_start = datetime.strptime(self.first_start, '%Y-%m-%d')
        else:
            first_start = min_ts

        if self.last_start:
            last_start = datetime.strptime(self.last_start, '%Y-%m-%d')
        else:
            last_start = max_ts - timedelta(days=period_days)

        first_end = first_start + timedelta(days=period_days)
        last_end = last_start + timedelta(days=period_days)

        print(f"First period: {first_start.date()} to {first_end.date()}")
        print(f"Last period: {last_start.date()} to {last_end.date()}")

        # Filter samples
        first_period = [s for s, ts in samples_with_ts if first_start <= ts < first_end]
        last_period = [s for s, ts in samples_with_ts if last_start <= ts < last_end]

        print(f"First period samples: {len(first_period)}")
        print(f"Last period samples: {len(last_period)}")

        return (first_period, last_period,
                (first_start, first_end), (last_start, last_end))

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function for batching samples."""
        all_categorical = [sample['categorical_features'] for sample in batch]
        all_coordinates = [sample['coordinates'] for sample in batch]
        all_time_deltas = [sample['time_deltas'] for sample in batch]
        all_masks = [sample['mask'] for sample in batch]

        # Stack categorical features
        categorical_batch = {}
        cat_keys = all_categorical[0].keys()
        for key in cat_keys:
            tensors = [cat[key] for cat in all_categorical]
            categorical_batch[key] = torch.stack(tensors)

        # Structure expected by model
        input_data = {
            'categorical_features': categorical_batch,
            'coordinates': torch.stack(all_coordinates),
            'time_deltas': torch.stack(all_time_deltas)
        }

        return {
            'input_data': input_data,
            'mask': torch.stack(all_masks)
        }

    def get_cluster_assignments(self, samples: List[Dict]) -> List[int]:
        """Get cluster assignments for samples using the pre-loaded datasets."""
        print(f"Getting cluster assignments for {len(samples)} samples...")

        batch_size = 64

        # Group samples by split to use the correct dataset
        samples_by_split = defaultdict(list)
        for i, sample in enumerate(samples):
            split = sample.get('_split', 'test')
            idx = sample.get('_split_idx', i)
            samples_by_split[split].append((i, idx))

        # Create result array
        cluster_results = [None] * len(samples)

        with torch.no_grad():
            for split, indices in samples_by_split.items():
                if split not in self.datasets:
                    continue

                dataset = self.datasets[split]

                # Process in batches
                for batch_start in range(0, len(indices), batch_size):
                    batch_indices = indices[batch_start:batch_start + batch_size]

                    # Get samples from dataset
                    batch_samples = [dataset[idx] for _, idx in batch_indices]
                    batch = self._collate_fn(batch_samples)

                    # Move to device
                    def move_to_device(obj):
                        if isinstance(obj, torch.Tensor):
                            return obj.to(self.device)
                        elif isinstance(obj, dict):
                            return {k: move_to_device(v) for k, v in obj.items()}
                        return obj

                    input_data = move_to_device(batch['input_data'])
                    mask = batch['mask'].to(self.device)

                    # Get cluster predictions
                    logits = self.model(
                        input_data=input_data,
                        attention_mask=mask,
                        return_embeddings=False
                    )
                    preds = torch.argmax(logits, dim=1).cpu().numpy()

                    # Store results
                    for (orig_idx, _), pred in zip(batch_indices, preds):
                        cluster_results[orig_idx] = int(pred)

        return cluster_results

    def compute_distributions(
        self,
        first_samples: List[Dict],
        last_samples: List[Dict],
        first_clusters: List[int],
        last_clusters: List[int],
        keep_all_labels: bool = False
    ) -> Dict[str, Any]:
        """Compute label and cluster distributions."""

        exclude_labels = set() if keep_all_labels else {'no_activity', 'No_Activity', 'Unknown', 'Other_Activity', ''}

        # Ground truth distributions
        first_gt = [self._get_ground_truth_label(s) for s in first_samples]
        last_gt = [self._get_ground_truth_label(s) for s in last_samples]

        # Filter if needed
        if not keep_all_labels:
            filtered_first = [(gt, cl) for gt, cl in zip(first_gt, first_clusters)
                            if gt.lower() not in [l.lower() for l in exclude_labels]]
            filtered_last = [(gt, cl) for gt, cl in zip(last_gt, last_clusters)
                           if gt.lower() not in [l.lower() for l in exclude_labels]]

            first_gt = [x[0] for x in filtered_first]
            first_clusters = [x[1] for x in filtered_first]
            last_gt = [x[0] for x in filtered_last]
            last_clusters = [x[1] for x in filtered_last]

        # Count distributions
        first_gt_counts = Counter(first_gt)
        last_gt_counts = Counter(last_gt)
        first_cl_counts = Counter(first_clusters)
        last_cl_counts = Counter(last_clusters)

        # Get all labels/clusters
        all_gt_labels = sorted(set(first_gt_counts.keys()) | set(last_gt_counts.keys()))
        all_clusters = sorted(set(first_cl_counts.keys()) | set(last_cl_counts.keys()))

        return {
            'gt_labels': all_gt_labels,
            'clusters': all_clusters,
            'first_gt_counts': first_gt_counts,
            'last_gt_counts': last_gt_counts,
            'first_cl_counts': first_cl_counts,
            'last_cl_counts': last_cl_counts,
            'first_total': len(first_gt),
            'last_total': len(last_gt),
        }

    def create_publication_chart(
        self,
        data: Dict[str, Any],
        first_period: Tuple[datetime, datetime],
        last_period: Tuple[datetime, datetime],
        normalized: bool = False,
        keep_all_labels: bool = False
    ):
        """Create a publication-ready 2x2 chart with shared axes and connected panels."""

        suffix = '_normalized' if normalized else ''
        suffix += '_all_labels' if keep_all_labels else ''
        suffix += f'_{self.label_level}'  # Add label level (l1 or l2)

        # Period labels for legend
        p1_label = f"Week 1 ({first_period[0].strftime('%b %d')})"
        p2_label = f"Week {self.weeks + int((last_period[0] - first_period[0]).days / 7)} ({last_period[0].strftime('%b %d')})"

        # Prepare cluster labels with custom names and ordering
        cluster_names = CLUSTER_TO_NAMES.get(self.dataset_name, {})
        cluster_order = CLUSTER_ORDER.get(self.dataset_name)

        # Get ordered list of clusters
        if cluster_order:
            # Use custom order, only include clusters that exist in data
            ordered_clusters = [c for c in cluster_order if c in data['clusters']]
            # Add any missing clusters at the end
            for c in data['clusters']:
                if c not in ordered_clusters:
                    ordered_clusters.append(c)
        else:
            ordered_clusters = data['clusters']

        # Create labels with format "CL{num}: {description}"
        def make_cluster_label(c):
            if c in cluster_names:
                return f"CL{c}: {cluster_names[c]}"
            return f"CL{c}"

        cluster_labels = [make_cluster_label(c) for c in ordered_clusters]

        # Create count dictionaries with the new label keys
        cluster_first = {make_cluster_label(k): data['first_cl_counts'].get(k, 0) for k in ordered_clusters}
        cluster_last = {make_cluster_label(k): data['last_cl_counts'].get(k, 0) for k in ordered_clusters}

        # Apply custom ordering to ground truth labels based on label level
        activity_order_dict = ACTIVITY_ORDER.get(self.dataset_name)
        activity_order = activity_order_dict.get(self.label_level) if activity_order_dict else None
        if activity_order:
            # Reorder labels based on predefined order
            ordered_gt_labels = [l for l in activity_order if l in data['gt_labels']]
            # Add any labels not in the predefined order at the end
            for l in data['gt_labels']:
                if l not in ordered_gt_labels:
                    ordered_gt_labels.append(l)
        else:
            ordered_gt_labels = data['gt_labels']

        # Pre-compute all values to determine shared x-axis ranges
        if normalized:
            gt_first_vals = [data['first_gt_counts'].get(l, 0) / data['first_total'] * 100 for l in ordered_gt_labels]
            gt_last_vals = [data['last_gt_counts'].get(l, 0) / data['last_total'] * 100 for l in ordered_gt_labels]
            cl_first_vals = [cluster_first.get(l, 0) / data['first_total'] * 100 for l in cluster_labels]
            cl_last_vals = [cluster_last.get(l, 0) / data['last_total'] * 100 for l in cluster_labels]
        else:
            gt_first_vals = [data['first_gt_counts'].get(l, 0) for l in ordered_gt_labels]
            gt_last_vals = [data['last_gt_counts'].get(l, 0) for l in ordered_gt_labels]
            cl_first_vals = [cluster_first.get(l, 0) for l in cluster_labels]
            cl_last_vals = [cluster_last.get(l, 0) for l in cluster_labels]

        # Compute deltas
        gt_deltas = [gt_last_vals[i] - gt_first_vals[i] for i in range(len(ordered_gt_labels))]
        cl_deltas = [cl_last_vals[i] - cl_first_vals[i] for i in range(len(cluster_labels))]

        # Separate x-axis ranges for each distribution chart
        gt_dist_max = max(gt_first_vals + gt_last_vals) * 1.1 if gt_first_vals else 10
        cl_dist_max = max(cl_first_vals + cl_last_vals) * 1.1 if cl_first_vals else 10

        # Shared x-axis range for delta column (both activity and cluster)
        all_deltas = gt_deltas + cl_deltas
        delta_max = max(abs(d) for d in all_deltas) * 1.3 if all_deltas else 10

        # Compute row heights proportional to number of labels (so bar widths are identical)
        n_gt_labels = len(ordered_gt_labels)
        n_cl_labels = len(cluster_labels)
        height_ratios = [n_gt_labels, n_cl_labels]

        # Adjust figure height based on total labels
        total_labels = n_gt_labels + n_cl_labels
        fig_height = max(8, total_labels * 0.35)  # Scale height with number of labels

        # Create figure with GridSpec for tighter column spacing
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(10, fig_height))  # Narrower overall width

        # GridSpec: narrower left column, wider right column, height proportional to labels
        gs = GridSpec(2, 2, figure=fig, wspace=0.05, hspace=0.25,
                      width_ratios=[0.5, 0.5],  # Left column narrower
                      height_ratios=height_ratios)

        axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]

        # Row 1: Ground truth labels
        self._plot_grouped_bars(
            axes[0][0],
            ordered_gt_labels,
            data['first_gt_counts'],
            data['last_gt_counts'],
            data['first_total'] if normalized else 1,
            data['last_total'] if normalized else 1,
            normalized,
            p1_label, p2_label,
            title='CASAS Label Distribution',
            x_max=gt_dist_max
        )

        self._plot_delta_bars(
            axes[0][1],
            ordered_gt_labels,
            gt_deltas,
            normalized,
            title='Change in Label Distribution',
            x_max=delta_max,
            show_y_labels=False
        )

        # Add connecting tick lines for row 1
        self._add_connecting_ticks(axes[0][0], axes[0][1], len(ordered_gt_labels))

        # Add top rule (spine) for the top row charts
        axes[0][0].spines['top'].set_visible(True)
        axes[0][0].spines['top'].set_color('#333333')
        axes[0][0].spines['top'].set_linewidth(0.8)
        axes[0][1].spines['top'].set_visible(True)
        axes[0][1].spines['top'].set_color('#333333')
        axes[0][1].spines['top'].set_linewidth(0.8)

        # Row 2: Clusters
        self._plot_grouped_bars(
            axes[1][0],
            cluster_labels,
            cluster_first,
            cluster_last,
            data['first_total'] if normalized else 1,
            data['last_total'] if normalized else 1,
            normalized,
            p1_label, p2_label,
            title='Cluster Distribution',
            x_max=cl_dist_max
        )

        self._plot_delta_bars(
            axes[1][1],
            cluster_labels,
            cl_deltas,
            normalized,
            title='Change in Cluster Distribution',
            x_max=delta_max,
            show_y_labels=False
        )

        # Add top rule (spine) for the bottom row charts
        axes[1][0].spines['top'].set_visible(True)
        axes[1][0].spines['top'].set_color('#333333')
        axes[1][0].spines['top'].set_linewidth(0.8)
        axes[1][1].spines['top'].set_visible(True)
        axes[1][1].spines['top'].set_color('#333333')
        axes[1][1].spines['top'].set_linewidth(0.8)

        # Compute group boundaries for clusters
        cluster_groups = CLUSTER_GROUPS.get(self.dataset_name)
        if cluster_groups:
            group_boundaries = []  # List of y positions where groups end
            group_labels_info = []  # List of (y_center, group_name)
            current_y = 0
            for group_name, group_clusters in cluster_groups:
                # Only count clusters that are in our ordered list
                count = sum(1 for c in group_clusters if c in ordered_clusters)
                if count > 0:
                    group_start = current_y
                    group_end = current_y + count
                    group_center = (group_start + group_end - 1) / 2
                    group_labels_info.append((group_center, group_name))
                    if group_end < len(cluster_labels):
                        group_boundaries.append(group_end - 0.5)
                    current_y = group_end

            # Add connecting lines only at group boundaries
            self._add_connecting_ticks(axes[1][0], axes[1][1], len(cluster_labels),
                                       group_boundaries=group_boundaries)
        else:
            # No groups, use standard connecting ticks
            self._add_connecting_ticks(axes[1][0], axes[1][1], len(cluster_labels))

        # Main title
        norm_text = 'Normalized ' if normalized else ''
        fig.suptitle(
            f'{norm_text}Temporal Distribution Comparison — {self.dataset_name.capitalize()} Dataset\n'
            f'{first_period[0].strftime("%b %d, %Y")} vs {last_period[0].strftime("%b %d, %Y")} ({self.weeks}-week periods)',
            fontsize=14, fontweight='bold', y=0.98
        )

        # Adjust layout (GridSpec handles spacing, just leave room for title)
        fig.subplots_adjust(top=0.88)  # More space below title

        # Save
        output_path = self.output_dir / f'temporal_diff{suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        print(f"Saved: {output_path}")
        print(f"Saved: {output_path.with_suffix('.pdf')}")

    def _add_connecting_ticks(self, ax_left, ax_right, n_labels: int,
                                group_boundaries: List[float] = None):
        """Add light horizontal lines spanning from labels through both charts.

        Args:
            ax_left: Left axes (distribution)
            ax_right: Right axes (delta)
            n_labels: Number of labels
            group_boundaries: Optional list of y positions where to draw lines.
                             If None, draws lines between every row.
        """
        if group_boundaries is not None:
            # Only draw lines at specified group boundaries
            for y_line in group_boundaries:
                # Draw full-width line on left chart (extends into label area)
                ax_left.axhline(y=y_line, color='#AAAAAA', linewidth=1.0,
                               linestyle='-', zorder=0, clip_on=False,
                               xmin=-0.5, xmax=1.0)

                # Draw full-width line on right chart
                ax_right.axhline(y=y_line, color='#AAAAAA', linewidth=1.0,
                                linestyle='-', zorder=0)
        else:
            # Draw lines between every row (default behavior)
            y_pos = np.arange(n_labels)
            for y in y_pos:
                for offset in [-0.5, 0.5]:
                    y_line = y + offset

                    if y_line < -0.5 or y_line > n_labels - 0.5:
                        continue

                    ax_left.axhline(y=y_line, color='#E0E0E0', linewidth=0.6,
                                   linestyle='-', zorder=0, clip_on=False,
                                   xmin=-0.3, xmax=1.0)

                    ax_right.axhline(y=y_line, color='#E0E0E0', linewidth=0.6,
                                    linestyle='-', zorder=0)

    def _add_group_labels(self, ax, group_labels_info: List[Tuple[float, str]]):
        """Add group labels on the left side of the chart (like merged cells).

        Args:
            ax: The axes to add labels to
            group_labels_info: List of (y_center, group_name) tuples
        """
        for y_center, group_name in group_labels_info:
            # Add text label to the left of the axes
            ax.annotate(
                group_name,
                xy=(-0.35, y_center),  # Position in axes coordinates
                xycoords=('axes fraction', 'data'),
                fontsize=9,
                fontweight='bold',
                ha='center',
                va='center',
                rotation=90,  # Vertical text
                color='#555555',
                clip_on=False
            )

    def _plot_grouped_bars(
        self,
        ax,
        labels: List[str],
        first_counts: Dict,
        last_counts: Dict,
        first_total: int,
        last_total: int,
        normalized: bool,
        p1_label: str,
        p2_label: str,
        title: str,
        x_max: float = None
    ):
        """Plot horizontal grouped bars for two periods."""

        n = len(labels)
        if n == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return

        # Bar positions
        bar_height = 0.35
        y_pos = np.arange(n)

        # Get values
        if normalized:
            first_vals = [first_counts.get(l, 0) / first_total * 100 for l in labels]
            last_vals = [last_counts.get(l, 0) / last_total * 100 for l in labels]
            xlabel = 'Percent of Data Samples'
        else:
            first_vals = [first_counts.get(l, 0) for l in labels]
            last_vals = [last_counts.get(l, 0) for l in labels]
            xlabel = 'Count'

        # Plot bars - both periods with solid colors (no hatch pattern)
        bars1 = ax.barh(y_pos - bar_height/2, first_vals, bar_height,
                       color=self.COLORS['week1'], label=p1_label,
                       edgecolor='white', linewidth=0.5)

        bars2 = ax.barh(y_pos + bar_height/2, last_vals, bar_height,
                       color=self.COLORS['week2'], label=p2_label,
                       edgecolor='white', linewidth=0.5)

        # Style
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight='bold', pad=10)
        ax.legend(loc='lower right', framealpha=0.9, edgecolor='none', facecolor='#F0F0F0', fontsize=8)

        # Set shared x-axis range
        if x_max is not None:
            ax.set_xlim(0, x_max)

        # Add % sign to x-axis ticks for normalized charts
        if normalized:
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))

        # Invert y-axis so first item is at top
        ax.invert_yaxis()

        # Add subtle grid
        ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_axisbelow(True)

        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _plot_delta_bars(
        self,
        ax,
        labels: List[str],
        deltas: List[float],
        normalized: bool,
        title: str,
        x_max: float = None,
        show_y_labels: bool = True
    ):
        """Plot horizontal delta bars showing change between periods."""

        n = len(labels)
        if n == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return

        y_pos = np.arange(n)
        bar_height = 0.6

        xlabel = 'Change (pp)' if normalized else 'Change'

        # Color based on direction
        colors = [self.COLORS['positive'] if d >= 0 else self.COLORS['negative'] for d in deltas]

        # Plot
        bars = ax.barh(y_pos, deltas, bar_height, color=colors, edgecolor='white', linewidth=0.5)

        # Add value labels
        max_delta = max(abs(d) for d in deltas) if deltas else 1
        for i, (bar, delta) in enumerate(zip(bars, deltas)):
            width = bar.get_width()
            sign = '+' if delta >= 0 else ''
            if normalized:
                label = f'{sign}{delta:.1f}%'
            else:
                label = f'{sign}{int(delta)}'

            # Position label - always outside for cleaner look
            offset = max_delta * 0.08 if deltas else 1
            x_pos = width + offset if width >= 0 else width - offset
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                   label, ha='left' if width >= 0 else 'right',
                   va='center', fontsize=8, color='#333333')

        # Style
        ax.set_yticks(y_pos)
        if show_y_labels:
            ax.set_yticklabels(labels)
        else:
            ax.set_yticklabels([])  # Hide y-labels
            ax.tick_params(axis='y', length=0)  # Hide tick marks too

        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight='bold', pad=10)

        # Add vertical line at 0
        ax.axvline(x=0, color='#333333', linewidth=1, linestyle='-')

        # Invert y-axis to match left plot
        ax.invert_yaxis()

        # Add subtle grid
        ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_axisbelow(True)

        # Set shared symmetric x-axis
        if x_max is not None:
            ax.set_xlim(-x_max, x_max)
        else:
            max_abs = max(abs(d) for d in deltas) if deltas else 1
            ax.set_xlim(-max_abs * 1.3, max_abs * 1.3)

        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Also remove left spine since we don't have y-labels
        if not show_y_labels:
            ax.spines['left'].set_visible(False)

        # Legend for colors (only on first delta chart)
        increase_patch = mpatches.Patch(color=self.COLORS['positive'], label='Increase')
        decrease_patch = mpatches.Patch(color=self.COLORS['negative'], label='Decrease')
        ax.legend(handles=[increase_patch, decrease_patch], loc='lower left',
                 framealpha=0.9, edgecolor='none', facecolor='#F0F0F0', fontsize=8)

    def run(self):
        """Run the full analysis and generate all chart variants."""
        print("\n" + "=" * 70)
        print("PUBLICATION-READY TEMPORAL DISTRIBUTION CHARTS")
        print("=" * 70)

        # Load model and data
        self.load_model()
        all_samples = self.load_all_data()

        # Filter by time periods
        first_samples, last_samples, first_period, last_period = \
            self.filter_by_time_periods(all_samples)

        # Get cluster assignments
        first_clusters = self.get_cluster_assignments(first_samples)
        last_clusters = self.get_cluster_assignments(last_samples)

        # Generate 4 chart variants
        for keep_all in [False, True]:
            label_text = "with all labels" if keep_all else "excluding no_activity"
            print(f"\n--- Computing distributions ({label_text}) ---")

            data = self.compute_distributions(
                first_samples, last_samples,
                first_clusters, last_clusters,
                keep_all_labels=keep_all
            )

            for normalized in [False, True]:
                norm_text = "normalized" if normalized else "raw counts"
                print(f"Creating chart: {norm_text}, {label_text}")

                self.create_publication_chart(
                    data, first_period, last_period,
                    normalized=normalized,
                    keep_all_labels=keep_all
                )

        print("\n" + "=" * 70)
        print(f"All charts saved to: {self.output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Create publication-ready temporal distribution comparison charts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (1 week comparison)
    python src/evals/scan/temporal_diff_publication.py \\
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1

    # Custom week start dates
    python src/evals/scan/temporal_diff_publication.py \\
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \\
        --first_start 2009-11-25 --last_start 2009-12-23

    # 2-week periods
    python src/evals/scan/temporal_diff_publication.py \\
        --model_dir trained_models/milan/scan_fl20_20cl_discover_v1 \\
        --weeks 2
        """
    )

    parser.add_argument('--model_dir', type=str, required=True,
                       help='Path to trained SCAN model directory')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to data directory (auto-detected if not provided)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (auto-generated if not provided)')
    parser.add_argument('--weeks', type=int, default=1,
                       help='Number of weeks for each period (default: 1)')
    parser.add_argument('--first_start', type=str, default=None,
                       help='Start date for first period (YYYY-MM-DD)')
    parser.add_argument('--last_start', type=str, default=None,
                       help='Start date for last period (YYYY-MM-DD)')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                       help='Checkpoint file name (default: best_model.pt)')
    parser.add_argument('--label_level', type=str, default='l1', choices=['l1', 'l2'],
                       help='Label granularity: l1 (coarse) or l2 (fine-grained) (default: l1)')

    args = parser.parse_args()

    analyzer = TemporalDiffPublicationChart(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        weeks=args.weeks,
        first_start=args.first_start,
        last_start=args.last_start,
        checkpoint=args.checkpoint,
        label_level=args.label_level
    )

    analyzer.run()


if __name__ == '__main__':
    main()

