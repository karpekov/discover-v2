#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
"""
Embedding visualization script for evaluating learned representations.
Creates 2D projections (t-SNE/UMAP) of sensor sequence embeddings colored by CASAS ground truth labels.

Usage:
    python src/evals/visualize_embeddings.py \
        --checkpoint trained_models/milan/tiny_50/best_model.pt \
        --train_data src/data/processed/casas/milan/training_50/train.json \
        --test_data src/data/processed/casas/milan/training_50/presegmented_test.json \
        --vocab src/data/processed/casas/milan/training_50/vocab.json \
        --output_dir results/evals/milan/tiny_50 \
        --max_samples 10000
"""

import torch
import numpy as np
import pandas as pd
import json
import argparse
import os
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not available. Install with: pip install umap-learn")

# Visualization
try:
    import altair as alt
    HAS_ALTAIR = True
    alt.data_transformers.enable('json')
except ImportError:
    HAS_ALTAIR = False
    print("Altair not available. Install with: pip install altair")

import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from models.text_encoder import TextEncoder, build_text_encoder
from models.sensor_encoder import SensorEncoder
from dataio.dataset import SmartHomeDataset
from dataio.collate import create_data_loader
from utils.device_utils import get_optimal_device, log_device_info
from utils.label_utils import convert_labels_to_text


class EmbeddingVisualizer:
    """Visualize sensor sequence embeddings with ground truth labels."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_optimal_device()
        log_device_info(self.device)

        # Load models
        self._load_models()

        # Load datasets
        self._load_datasets()

        # Load label colors from metadata
        self._load_label_colors()

    def _perform_clustering(self, embeddings: np.ndarray, n_clusters: int = 50, use_cosine: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Perform clustering on embeddings and return cluster labels."""
        print(f"🔄 Performing clustering with {n_clusters} clusters (cosine={use_cosine})...")

        if use_cosine:
            # Normalize embeddings for cosine distance
            embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # Custom K-means with cosine distance
            np.random.seed(42)
            n_samples, n_features = embeddings_normalized.shape

            # Initialize centroids randomly
            centroid_indices = np.random.choice(n_samples, n_clusters, replace=False)
            centroids = embeddings_normalized[centroid_indices].copy()

            max_iter = 300
            tol = 1e-4

            for iteration in range(max_iter):
                # Compute similarities and assign clusters
                similarities = cosine_similarity(embeddings_normalized, centroids)
                cluster_labels = np.argmax(similarities, axis=1)

                # Update centroids
                new_centroids = np.zeros((n_clusters, n_features))
                for k in range(n_clusters):
                    cluster_mask = cluster_labels == k
                    if np.sum(cluster_mask) > 0:
                        cluster_mean = np.mean(embeddings_normalized[cluster_mask], axis=0)
                        new_centroids[k] = cluster_mean / np.linalg.norm(cluster_mean)
                    else:
                        new_centroids[k] = centroids[k]

                # Check convergence
                centroid_shift = np.mean(np.linalg.norm(new_centroids - centroids, axis=1))
                if centroid_shift < tol:
                    print(f"    Converged after {iteration + 1} iterations")
                    break
                centroids = new_centroids

            cluster_labels = np.argmax(cosine_similarity(embeddings_normalized, centroids), axis=1)
        else:
            # Standard K-means with Euclidean distance
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

        # Create cluster label strings
        cluster_label_strings = [f"Cluster_{i:02d}" for i in cluster_labels]

        print(f"✅ Clustering complete. Found {len(np.unique(cluster_labels))} unique clusters")
        return cluster_labels, cluster_label_strings


    def _load_models(self):
        """Load trained models from checkpoint."""
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
        model_config = checkpoint.get('config', {})
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

        # Set to eval mode
        self.sensor_encoder.eval()
        self.text_encoder.eval()

        print("✅ Models loaded successfully")

    def _load_datasets(self):
        """Load training and test datasets."""
        datasets = {}

        if self.config['train_data_path'] and os.path.exists(self.config['train_data_path']):
            datasets['train'] = SmartHomeDataset(
                data_path=self.config['train_data_path'],
                vocab_path=self.config['vocab_path'],
                sequence_length=self.config['sequence_length']
            )
            print(f"📊 Train dataset: {len(datasets['train'])} samples")

        if self.config['test_data_path'] and os.path.exists(self.config['test_data_path']):
            datasets['test'] = SmartHomeDataset(
                data_path=self.config['test_data_path'],
                vocab_path=self.config['vocab_path'],
                sequence_length=20,  # Use the actual sequence length from data
                max_captions=1
            )
            print(f"📊 Test dataset: {len(datasets['test'])} samples")

        self.datasets = datasets

    def _load_label_colors(self):
        """Load label colors from city metadata."""
        try:
            metadata_path = Path(__file__).parent.parent.parent / "metadata" / "city_metadata.json"
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


    def extract_text_embeddings(self, labels: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Extract text embeddings for activity labels."""

        # Get unique labels
        unique_labels = sorted(list(set(labels)))
        print(f"🔤 Extracting text embeddings for {len(unique_labels)} unique labels...")

        # Convert to text descriptions
        descriptions = convert_labels_to_text(unique_labels, single_description=True)

        # Extract CLIP projected text embeddings (512-dim)
        self.text_encoder.eval()
        with torch.no_grad():
            text_embeddings = self.text_encoder.encode_texts_clip(descriptions, self.device).cpu().numpy()

        print(f"✅ Extracted {text_embeddings.shape[0]} text embeddings")
        for label, desc in zip(unique_labels, descriptions):
            print(f"    {label} → '{desc}'")

        return text_embeddings, unique_labels

    def extract_embeddings(self, max_samples: int = 10000,
                          split: str = 'test',
                          batch_size: int = 64,
                          include_text_embeddings: bool = True) -> Tuple[np.ndarray, List[str], List[str], List[Dict], np.ndarray, List[str]]:
        """Extract embeddings from the specified dataset split with random sampling."""
        import random
        from torch.utils.data import Subset

        if split not in self.datasets:
            raise ValueError(f"Split '{split}' not available. Available: {list(self.datasets.keys())}")

        dataset = self.datasets[split]
        actual_samples = min(max_samples, len(dataset))

        print(f"🔄 Extracting embeddings from {actual_samples} {split} samples...")

        # Create a random subset if needed
        if actual_samples < len(dataset):
            # Set consistent seed for reproducible sampling
            import random
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

            # Create random indices for sampling
            all_indices = list(range(len(dataset)))
            random.shuffle(all_indices)
            selected_indices = all_indices[:actual_samples]
            print(f"🎲 Randomly sampling {actual_samples} from {len(dataset)} total samples (seed=42)")

            # Create subset dataset
            dataset = Subset(dataset, selected_indices)
        else:
            print(f"📊 Using all {len(dataset)} samples")

        # Create data loader with shuffling for better visualization diversity
        data_loader = create_data_loader(
            dataset=dataset,
            text_encoder=self.text_encoder,
            span_masker=None,
            vocab_sizes=self.vocab_sizes,
            device=self.device,
            batch_size=batch_size,
            shuffle=True,  # Enable shuffling for better diversity
            num_workers=0,  # For MPS compatibility
            apply_mlm=False  # No masking for evaluation
        )

        embeddings = []
        labels = []
        labels_l2 = []
        metadata = []
        samples_processed = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if samples_processed >= actual_samples:
                    break

                # Extract CLIP projected embeddings (512-dim)
                sensor_emb = self.sensor_encoder.forward_clip(
                    categorical_features=batch['categorical_features'],
                    coordinates=batch['coordinates'],
                    time_deltas=batch['time_deltas'],
                    mask=batch['mask']
                )

                embeddings.append(sensor_emb.cpu().numpy())

                # Extract labels and metadata for this batch
                batch_size_actual = sensor_emb.shape[0]

                # Get ground truth labels directly from the batch (much more reliable!)
                batch_labels_l1 = batch.get('activity_labels', ['Unknown'] * batch_size_actual)
                batch_labels_l2 = batch.get('activity_labels_l2', ['Unknown'] * batch_size_actual)
                batch_captions = batch.get('captions', [''] * batch_size_actual)

                for i in range(batch_size_actual):
                    if samples_processed >= actual_samples:
                        break

                    # Use labels directly from the batch - this is the correct approach!
                    label = batch_labels_l1[i] if i < len(batch_labels_l1) else 'Unknown'
                    label_l2 = batch_labels_l2[i] if i < len(batch_labels_l2) else 'Unknown'
                    caption = batch_captions[i] if i < len(batch_captions) else ''

                    labels.append(label)
                    labels_l2.append(label_l2)

                    # Additional metadata
                    sample_metadata = {
                        'sample_idx': samples_processed,  # Use processed count as index
                        'split': split,
                        'caption': caption
                    }
                    metadata.append(sample_metadata)

                    samples_processed += 1

                if batch_idx % 20 == 0:
                    print(f"  Processed {samples_processed}/{actual_samples} samples...")

        # Concatenate embeddings
        embeddings = np.vstack(embeddings)[:actual_samples]

        print(f"📈 Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
        print(f"📊 Label distribution:")

        # Print label statistics
        from collections import Counter
        label_counts = Counter(labels)
        for label, count in label_counts.most_common(10):
            print(f"    {label}: {count}")

        # Extract text embeddings for labels if requested
        text_embeddings = np.array([])
        text_labels = []
        if include_text_embeddings:
            text_embeddings, text_labels = self.extract_text_embeddings(labels)

        return embeddings, labels, labels_l2, metadata, text_embeddings, text_labels

    def filter_noisy_labels(self, embeddings: np.ndarray, labels: List[str], labels_l2: List[str], metadata: List[Dict]) -> Tuple[np.ndarray, List[str], List[str], List[Dict]]:
        """Filter out noisy/uninformative labels like 'Other', 'No_Activity', etc."""

        # Define labels to exclude (case-insensitive)
        exclude_labels = {
            'other', 'no_activity', 'unknown', 'none', 'null', 'nan',
            'no activity', 'other activity', 'miscellaneous', 'misc'
        }

        # Find valid indices (keep samples that don't have excluded labels in either L1 or L2)
        valid_indices = []
        for i, (l1, l2) in enumerate(zip(labels, labels_l2)):
            l1_lower = l1.lower().strip()
            l2_lower = l2.lower().strip()

            # Keep sample if neither L1 nor L2 labels are in exclude list
            if l1_lower not in exclude_labels and l2_lower not in exclude_labels:
                valid_indices.append(i)

        if not valid_indices:
            print("⚠️  Warning: All samples filtered out!")
            return embeddings, labels, labels_l2, metadata

        # Filter arrays and lists
        filtered_embeddings = embeddings[valid_indices]
        filtered_labels = [labels[i] for i in valid_indices]
        filtered_labels_l2 = [labels_l2[i] for i in valid_indices]
        filtered_metadata = [metadata[i] for i in valid_indices]

        print(f"🧹 Filtered out noisy labels:")
        print(f"   Original samples: {len(labels)}")
        print(f"   Filtered samples: {len(filtered_labels)}")
        print(f"   Removed: {len(labels) - len(filtered_labels)} samples")

        return filtered_embeddings, filtered_labels, filtered_labels_l2, filtered_metadata

    def create_2d_projection(self, embeddings: np.ndarray,
                           method: str = 'tsne',
                           random_state: int = 42,
                           text_embeddings: np.ndarray = None) -> np.ndarray:
        """Create 2D projection using t-SNE or UMAP."""

        # Combine sensor and text embeddings if text embeddings provided
        combined_embeddings = embeddings
        n_sensor_embeddings = embeddings.shape[0]

        if text_embeddings is not None and text_embeddings.size > 0:
            combined_embeddings = np.vstack([embeddings, text_embeddings])
            print(f"📊 Combined embeddings: {n_sensor_embeddings} sensor + {text_embeddings.shape[0]} text = {combined_embeddings.shape[0]} total")
        else:
            print(f"📊 Sensor embeddings only: {embeddings.shape[0]} samples")

        print(f"🔄 Creating 2D projection using {method.upper()}...")

        if method.lower() == 'tsne':
            reducer = TSNE(
                n_components=2,
                random_state=random_state,
                perplexity=min(30, len(combined_embeddings) // 4),
                max_iter=1000,
                verbose=1
            )
        elif method.lower() == 'umap':
            if not HAS_UMAP:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")

            reducer = umap.UMAP(
                n_components=2,
                random_state=random_state,
                n_neighbors=min(15, len(combined_embeddings) // 10),
                min_dist=0.1,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'")

        projection = reducer.fit_transform(combined_embeddings)

        print(f"✅ 2D projection complete: {projection.shape}")
        return projection

    def create_altair_plot(self, projection: np.ndarray,
                          labels: List[str],
                          labels_l2: List[str],
                          metadata: List[Dict],
                          method: str,
                          split: str) -> alt.Chart:
        """Create interactive Altair scatter plot."""

        if not HAS_ALTAIR:
            raise ImportError("Altair not available. Install with: pip install altair")

        # Create DataFrame
        df = pd.DataFrame({
            'x': projection[:, 0],
            'y': projection[:, 1],
            'activity': labels,
            'activity_l2': labels_l2,
            'sample_idx': [m['sample_idx'] for m in metadata],
            'caption': [m['caption'][:100] + '...' if len(m['caption']) > 100 else m['caption'] for m in metadata]
        })

        # Create color palette using metadata colors
        unique_labels = df['activity'].unique()

        # Build color scale using metadata colors where available
        color_domain = []
        color_range = []

        for label in sorted(unique_labels):
            color_domain.append(label)
            if label in self.label_colors:
                color_range.append(self.label_colors[label])
            else:
                # Fallback to default colors for labels not in metadata
                color_range.append(None)  # Will use Altair's default scheme

        # Create color scale
        if any(color_range):  # If we have any custom colors
            # Filter out None values and create custom scale
            custom_colors = [(domain, color) for domain, color in zip(color_domain, color_range) if color is not None]
            if len(custom_colors) == len(color_domain):
                # All labels have custom colors
                color_scale = alt.Scale(domain=color_domain, range=color_range)
            else:
                # Mix of custom and default colors - use category20 as fallback
                color_scale = alt.Scale(scheme='category20')
        else:
            # No custom colors available
            color_scale = alt.Scale(scheme='category20')

        # Base chart
        base = alt.Chart(df).add_selection(
            alt.selection_single(on='mouseover', empty='none')
        )

        # Main scatter plot
        scatter = base.mark_circle(size=60, opacity=0.7).encode(
            x=alt.X('x:Q', title=f'{method.upper()} 1'),
            y=alt.Y('y:Q', title=f'{method.upper()} 2'),
            color=alt.Color(
                'activity:N',
                title='Ground Truth Activity',
                scale=color_scale,
                legend=alt.Legend(orient='right', titleLimit=200)
            ),
            tooltip=[
                alt.Tooltip('activity:N', title='Activity'),
                alt.Tooltip('activity_l2:N', title='Activity L2'),
                alt.Tooltip('sample_idx:O', title='Sample Index'),
                alt.Tooltip('caption:N', title='Caption')
            ]
        ).properties(
            width=800,
            height=600,
            title=f'{method.upper()} Visualization of Sensor Embeddings ({split} split, {len(df)} samples)'
        )

        return scatter

    def create_matplotlib_plot(self, projection: np.ndarray,
                              labels: List[str],
                              labels_l2: List[str],
                              method: str,
                              split: str,
                              save_path: str = None,
                              text_labels: List[str] = None,
                              n_sensor_samples: int = None) -> plt.Figure:
        """Create matplotlib scatter plot."""

        # Split projection into sensor and text parts
        if n_sensor_samples is not None and text_labels is not None:
            sensor_projection = projection[:n_sensor_samples]
            text_projection = projection[n_sensor_samples:]

            # Create DataFrame for sensor embeddings
            df_sensor = pd.DataFrame({
                'x': sensor_projection[:, 0],
                'y': sensor_projection[:, 1],
                'activity': labels,
                'activity_l2': labels_l2,
                'type': 'sensor'
            })

            # Create DataFrame for text embeddings
            df_text = pd.DataFrame({
                'x': text_projection[:, 0],
                'y': text_projection[:, 1],
                'activity': text_labels,
                'activity_l2': text_labels,  # Use same for both
                'type': 'text'
            })

            df = pd.concat([df_sensor, df_text], ignore_index=True)
        else:
            # Original behavior - sensor embeddings only
            df = pd.DataFrame({
                'x': projection[:, 0],
                'y': projection[:, 1],
                'activity': labels,
                'activity_l2': labels_l2,
                'type': 'sensor'
            })

        # Set up the plot
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: Primary activities
        unique_labels = df['activity'].unique()

        # Use custom colors from metadata where available
        colors = {}
        for label in unique_labels:
            if label in self.label_colors:
                colors[label] = self.label_colors[label]
            else:
                # Fallback to tab20 colormap
                colors[label] = plt.cm.tab20(len(colors) % 20)

        # Plot sensor embeddings first (small circles)
        sensor_df = df[df['type'] == 'sensor']
        for label in sensor_df['activity'].unique():
            mask = sensor_df['activity'] == label
            ax1.scatter(
                sensor_df.loc[mask, 'x'],
                sensor_df.loc[mask, 'y'],
                c=[colors[label]],
                label=f'{label} (sensor)',
                alpha=0.6,
                s=30,
                marker='o'
            )

        # Plot text embeddings (large diamonds with labels)
        text_df = df[df['type'] == 'text']
        if not text_df.empty:
            for _, row in text_df.iterrows():
                label = row['activity']
                ax1.scatter(
                    row['x'], row['y'],
                    c=[colors[label]],
                    s=200,  # Large size
                    marker='D',  # Diamond shape
                    alpha=1.0,
                    edgecolors='black',
                    linewidths=2,
                    label=f'{label} (text)'
                )

                # Add text label next to diamond
                ax1.annotate(
                    label,
                    (row['x'], row['y']),
                    xytext=(10, 10),  # Offset from point
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'),
                    ha='left'
                )

        ax1.set_xlabel(f'{method.upper()} 1')
        ax1.set_ylabel(f'{method.upper()} 2')
        ax1.set_title(f'{method.upper()} - Primary Activities ({split} split, {len(df)} samples)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot 2: L2 activities
        unique_labels_l2 = df['activity_l2'].unique()

        # Use custom L2 colors from metadata where available
        colors_l2 = {}
        for label in unique_labels_l2:
            if label in self.label_colors_l2:
                colors_l2[label] = self.label_colors_l2[label]
            else:
                # Fallback to tab10 colormap
                colors_l2[label] = plt.cm.tab10(len(colors_l2) % 10)

        for label in unique_labels_l2:
            mask = df['activity_l2'] == label
            ax2.scatter(
                df.loc[mask, 'x'],
                df.loc[mask, 'y'],
                c=[colors_l2[label]],
                label=label,
                alpha=0.7,
                s=30
            )

        ax2.set_xlabel(f'{method.upper()} 1')
        ax2.set_ylabel(f'{method.upper()} 2')
        ax2.set_title(f'{method.upper()} - L2 Activities ({split} split, {len(df)} samples)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {save_path}")

        return fig

    def create_dual_matplotlib_plot(self, projection: np.ndarray,
                                   ground_truth_labels: List[str],
                                   cluster_labels: List[str],
                                   method: str, split: str, save_path: str = None) -> plt.Figure:
        """Create dual matplotlib plots: ground truth vs cluster labels."""

        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: Ground truth labels (L1) - use colors from metadata
        unique_gt_labels = list(set(ground_truth_labels))
        gt_color_map = {}

        # Use colors from city_metadata if available, otherwise fall back to default colors
        for label in unique_gt_labels:
            if hasattr(self, 'label_colors') and label in self.label_colors:
                gt_color_map[label] = self.label_colors[label]
            else:
                # Fallback to tab20 colors if not in metadata
                fallback_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_gt_labels)))
                gt_color_map[label] = fallback_colors[list(unique_gt_labels).index(label)]

        for label in unique_gt_labels:
            mask = np.array(ground_truth_labels) == label
            if np.any(mask):
                ax1.scatter(projection[mask, 0], projection[mask, 1],
                           c=[gt_color_map[label]], label=label, alpha=0.6, s=20)

        ax1.set_title(f'Ground Truth Labels (L1)\n{method.upper()} - {split} - {len(projection)} samples',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'{method.upper()}-1', fontsize=12)
        ax1.set_ylabel(f'{method.upper()}-2', fontsize=12)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cluster labels
        unique_cluster_labels = list(set(cluster_labels))
        cluster_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_cluster_labels)))
        cluster_color_map = {label: color for label, color in zip(unique_cluster_labels, cluster_colors)}

        for label in unique_cluster_labels:
            mask = np.array(cluster_labels) == label
            if np.any(mask):
                ax2.scatter(projection[mask, 0], projection[mask, 1],
                           c=[cluster_color_map[label]], label=label, alpha=0.6, s=20)

        ax2.set_title(f'Cluster Labels (K-means, n={len(unique_cluster_labels)})\n{method.upper()} - {split} - {len(projection)} samples',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'{method.upper()}-1', fontsize=12)
        ax2.set_ylabel(f'{method.upper()}-2', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Dual plot saved to: {save_path}")

        return fig

    def run_visualization(self, max_samples: int = 10000,
                         split: str = 'test',
                         method: str = 'tsne',
                         use_altair: bool = True,
                         save_plots: bool = True,
                         filter_noisy_labels: bool = False,
                         include_text_embeddings: bool = False,
                         include_clustering: bool = False,
                         n_clusters: int = 25) -> Dict[str, Any]:
        """Run the complete embedding visualization pipeline."""

        print(f"🚀 Starting embedding visualization...")
        print(f"   Max samples: {max_samples}")
        print(f"   Split: {split}")
        print(f"   Method: {method}")
        print(f"   Visualization: {'Altair' if use_altair else 'Matplotlib'}")
        print(f"   Include text embeddings: {include_text_embeddings}")
        print(f"   Include clustering: {include_clustering}")
        if include_clustering:
            print(f"   Number of clusters: {n_clusters}")

        # Extract embeddings (optionally including text embeddings)
        embeddings, labels, labels_l2, metadata, text_embeddings, text_labels = self.extract_embeddings(
            max_samples=max_samples,
            split=split,
            include_text_embeddings=include_text_embeddings
        )

        # Filter out noisy labels if requested
        if filter_noisy_labels:
            embeddings, labels, labels_l2, metadata = self.filter_noisy_labels(
                embeddings, labels, labels_l2, metadata
            )
            # Re-extract text embeddings for the filtered labels (only if including text embeddings)
            if include_text_embeddings:
                text_embeddings, text_labels = self.extract_text_embeddings(labels)

        # Perform clustering if requested
        cluster_labels_numeric = None
        cluster_labels = None
        if include_clustering:
            cluster_labels_numeric, cluster_labels = self._perform_clustering(embeddings, n_clusters=n_clusters, use_cosine=True)

        # Create 2D projection (optionally combining sensor and text embeddings)
        text_emb_for_projection = text_embeddings if include_text_embeddings else None
        projection = self.create_2d_projection(embeddings, method=method, text_embeddings=text_emb_for_projection)

        # Create visualizations
        results = {
            'embeddings': embeddings,
            'projection': projection,
            'labels': labels,
            'labels_l2': labels_l2,
            'metadata': metadata,
            'cluster_labels': cluster_labels,
            'cluster_labels_numeric': cluster_labels_numeric
        }

        # Output directory
        output_dir = Path(self.config.get('output_dir', './embedding_visualizations'))
        output_dir.mkdir(parents=True, exist_ok=True)

        if use_altair and HAS_ALTAIR:
            # Create Altair plot
            chart = self.create_altair_plot(projection, labels, labels_l2, metadata, method, split)

            if save_plots:
                # Save as HTML
                html_path = output_dir / f"embeddings_{method}_{split}_{len(embeddings)}samples.html"
                chart.save(str(html_path))
                print(f"📊 Interactive plot saved: {html_path}")

            results['altair_chart'] = chart

        # Create matplotlib plot(s)
        if include_clustering and cluster_labels is not None:
            # Create dual plot with ground truth vs cluster labels
            dual_matplotlib_path = output_dir / f"embeddings_{method}_{split}_{len(embeddings)}samples_dual.png" if save_plots else None
            fig = self.create_dual_matplotlib_plot(
                projection, labels, cluster_labels, method, split,
                str(dual_matplotlib_path) if dual_matplotlib_path else None
            )
            results['matplotlib_fig'] = fig

            # Also create individual plots for comparison
            gt_matplotlib_path = output_dir / f"embeddings_{method}_{split}_{len(embeddings)}samples_groundtruth.png" if save_plots else None
            fig_gt = self.create_matplotlib_plot(
                projection, labels, labels_l2, method, split,
                str(gt_matplotlib_path) if gt_matplotlib_path else None,
                text_labels=text_labels if include_text_embeddings else None,
                n_sensor_samples=len(embeddings)
            )
            results['matplotlib_fig_groundtruth'] = fig_gt
        else:
            # Standard single plot
            matplotlib_path = output_dir / f"embeddings_{method}_{split}_{len(embeddings)}samples.png" if save_plots else None
            fig = self.create_matplotlib_plot(
                projection, labels, labels_l2, method, split,
                str(matplotlib_path) if matplotlib_path else None,
                text_labels=text_labels if include_text_embeddings else None,
                n_sensor_samples=len(embeddings)
            )
            results['matplotlib_fig'] = fig

        # Skip saving raw data (numpy arrays) as requested

        print("✅ Embedding visualization complete!")
        return results


def main():
    parser = argparse.ArgumentParser(description='Visualize sensor sequence embeddings with ground truth labels')

    # Model and data paths
    parser.add_argument('--checkpoint', type=str, default='./outputs/milan_training/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--train_data', type=str,
                       default='../data/processed_experiments/experiment_milan_training/milan/milan_train.json',
                       help='Path to training data')
    parser.add_argument('--test_data', type=str,
                       default='../data/processed_experiments/experiment_milan_training/milan/milan_test.json',
                       help='Path to test data')
    parser.add_argument('--vocab', type=str,
                       default='../data/processed_experiments/experiment_milan_training/milan/milan_vocab.json',
                       help='Path to vocabulary file')

    # Visualization parameters
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum number of samples to visualize (default: 5000)')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'both'], default='test',
                       help='Dataset split to visualize')
    parser.add_argument('--method', type=str, choices=['tsne', 'umap'], default='tsne',
                       help='Dimensionality reduction method')
    parser.add_argument('--use_altair', action='store_true', default=True,
                       help='Use Altair for interactive plots')
    parser.add_argument('--use_matplotlib', action='store_true',
                       help='Use matplotlib instead of Altair')
    parser.add_argument('--output_dir', type=str, default='results/evals/milan/embedding_visualizations',
                       help='Output directory for plots and data')
    parser.add_argument('--filter_noisy_labels', action='store_true',
                       help='Filter out noisy labels like "Other" and "No_Activity"')
    parser.add_argument('--include_text_embeddings', action='store_true',
                       help='Include text embeddings in the visualization (default: False, sensor embeddings only)')
    parser.add_argument('--include_clustering', action='store_true',
                       help='Include clustering analysis and create dual plots (ground truth vs clusters)')
    parser.add_argument('--n_clusters', type=int, default=25,
                       help='Number of clusters for K-means clustering (default: 25)')

    args = parser.parse_args()

    # Configuration
    config = {
        'checkpoint_path': args.checkpoint,
        'train_data_path': args.train_data,
        'test_data_path': args.test_data,
        'vocab_path': args.vocab,
        'output_dir': args.output_dir,

        # Model architecture (should match training config)
        'text_model_name': 'thenlper/gte-base',
        'd_model': 768,
        'n_layers': 6,
        'n_heads': 8,
        'd_ff': 3072,
        'dropout': 0.1,
        'sequence_length': 512,  # Match training config
        'use_rope_time': False,
        'use_rope_2d': False,
        'fourier_num_bands': 12,

        # MLM config (needed for loading)
        'mlm_field_priors': {
            'room_id': 0.30, 'event_type': 0.20, 'sensor_id': 0.20,
            'tod_bucket': 0.15, 'delta_t_bucket': 0.10, 'sensor_type': 0.05
        },
        'mask_prob': 0.25,
        'mean_span_length': 3.0,
    }

    # Initialize visualizer
    visualizer = EmbeddingVisualizer(config)

    # Run visualization
    if args.split == 'both':
        # Visualize both splits
        for split in ['train', 'test']:
            if split in visualizer.datasets:
                visualizer.run_visualization(
                    max_samples=args.max_samples,
                    split=split,
                    method=args.method,
                    use_altair=args.use_altair and not args.use_matplotlib,
                    save_plots=True,
                    filter_noisy_labels=args.filter_noisy_labels,
                    include_text_embeddings=args.include_text_embeddings,
                    include_clustering=args.include_clustering,
                    n_clusters=args.n_clusters
                )
    else:
        visualizer.run_visualization(
            max_samples=args.max_samples,
            split=args.split,
            method=args.method,
            use_altair=args.use_altair and not args.use_matplotlib,
            save_plots=True,
            filter_noisy_labels=args.filter_noisy_labels,
            include_text_embeddings=args.include_text_embeddings,
            include_clustering=args.include_clustering,
            n_clusters=args.n_clusters
        )

    print("\n🎯 Embedding visualization complete!")
    print(f"📁 Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
