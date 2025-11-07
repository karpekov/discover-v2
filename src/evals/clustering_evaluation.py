#!/usr/bin/env python3
"""
Clustering-based Activity Recognition Evaluation

This script evaluates how well unsupervised clustering of sensor embeddings
can recover ground truth activity labels. It uses K-means and DBSCAN clustering
algorithms and assigns majority labels to each cluster.

Usage:
    python clustering_evaluation.py \
        --checkpoint path/to/model.pt \
        --test_data path/to/test.json \
        --vocab path/to/vocab.json \
        --output_dir path/to/output \
        --max_samples 10000 \
        --n_clusters 50 \
        --filter_noisy_labels
"""

import argparse
import json
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, homogeneity_score, completeness_score,
    v_measure_score, classification_report
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Import project modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataio.dataset import SmartHomeDataset
from dataio.collate import create_data_loader
from models.text_encoder import TextEncoder, build_text_encoder
from models.sensor_encoder import SensorEncoder
from utils.device_utils import get_optimal_device, log_device_info


class CosineKMeans:
    """K-means clustering using cosine distance."""

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        """Fit the cosine K-means clustering."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Normalize input vectors (important for cosine distance)
        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
        n_samples, n_features = X_normalized.shape

        # Initialize centroids randomly
        centroid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = X_normalized[centroid_indices].copy()

        for iteration in range(self.max_iter):
            # Compute cosine similarities between points and centroids
            similarities = cosine_similarity(X_normalized, centroids)

            # Assign each point to the closest centroid (highest cosine similarity)
            labels = np.argmax(similarities, axis=1)

            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                cluster_mask = labels == k
                if np.sum(cluster_mask) > 0:
                    # Average of points in cluster, then normalize
                    cluster_mean = np.mean(X_normalized[cluster_mask], axis=0)
                    new_centroids[k] = cluster_mean / np.linalg.norm(cluster_mean)
                else:
                    # Keep old centroid if no points assigned
                    new_centroids[k] = centroids[k]

            # Check for convergence
            centroid_shift = np.mean(np.linalg.norm(new_centroids - centroids, axis=1))
            if centroid_shift < self.tol:
                break

            centroids = new_centroids

        self.cluster_centers_ = centroids
        self.labels_ = labels
        return self

    def fit_predict(self, X):
        """Fit the model and return cluster labels."""
        self.fit(X)
        return self.labels_


class ClusteringEvaluator:
    """Evaluates activity recognition through unsupervised clustering."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the clustering evaluator."""
        self.config = config
        self.device = get_optimal_device()
        log_device_info(self.device)

        # Load models and data
        self._load_models()
        self._load_datasets()

    def _load_models(self):
        """Load trained models from checkpoint."""
        print(f"🔄 Loading models from {self.config['checkpoint_path']}")

        checkpoint = torch.load(self.config['checkpoint_path'], map_location=self.device)

        # Sensor encoder - check if it's ChronosEncoder or SensorEncoder
        self.vocab_sizes = checkpoint.get('vocab_sizes', {})
        model_config = checkpoint.get('config', {})

        # Check if this is a Chronos checkpoint
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

        self.sensor_encoder.to(self.device)
        self.sensor_encoder.eval()

        print("✅ Models loaded successfully")

    def _simple_collate_fn(self, batch):
        """Simple collate function for clustering that doesn't need text processing."""
        from collections import defaultdict

        # Initialize batch structure
        collated_batch = {
            'categorical_features': defaultdict(list),
            'coordinates': [],
            'time_deltas': [],
            'mask': [],
            'activity_labels': [],
            'activity_labels_l2': []
        }

        for sample in batch:
            # Add categorical features
            for key, value in sample.get('categorical_features', {}).items():
                collated_batch['categorical_features'][key].append(value)

            # Add other features
            collated_batch['coordinates'].append(sample.get('coordinates', torch.zeros(20, 2)))
            collated_batch['time_deltas'].append(sample.get('time_deltas', torch.zeros(20)))
            collated_batch['mask'].append(sample.get('mask', torch.ones(20, dtype=torch.bool)))

            # Get activity labels using the correct keys
            label_l1 = sample.get('first_activity', 'Unknown')
            label_l2 = sample.get('first_activity_l2', 'Unknown')

            collated_batch['activity_labels'].append(label_l1)
            collated_batch['activity_labels_l2'].append(label_l2)

        # Stack tensors and move to device
        for key in collated_batch['categorical_features']:
            collated_batch['categorical_features'][key] = torch.stack(collated_batch['categorical_features'][key]).to(self.device)

        collated_batch['coordinates'] = torch.stack(collated_batch['coordinates']).to(self.device)
        collated_batch['time_deltas'] = torch.stack(collated_batch['time_deltas']).to(self.device)
        collated_batch['mask'] = torch.stack(collated_batch['mask']).to(self.device)

        return collated_batch

    def _load_datasets(self):
        """Load datasets."""
        self.datasets = {}

        if self.config['test_data_path'] and Path(self.config['test_data_path']).exists():
            self.datasets['test'] = SmartHomeDataset(
                data_path=self.config['test_data_path'],
                vocab_path=self.config['vocab_path'],
                sequence_length=20,
                max_captions=1
            )
            print(f"📊 Test dataset: {len(self.datasets['test'])} samples")
        else:
            raise ValueError(f"Test dataset not found: {self.config['test_data_path']}")

    def extract_embeddings_and_labels(self, split: str, max_samples: int = 10000, filter_noisy: bool = True) -> Dict[str, Any]:
        """Extract embeddings and ground truth labels from dataset with random sampling."""
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

        # Create simple data loader without text processing
        from torch.utils.data import DataLoader
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: self._simple_collate_fn(batch)
        )

        embeddings = []
        labels_l1 = []
        labels_l2 = []
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

                # Extract labels for this batch
                batch_size_actual = sensor_emb.shape[0]

                # Get ground truth labels directly from the batch (much more reliable!)
                batch_labels_l1 = batch.get('activity_labels', ['Unknown'] * batch_size_actual)
                batch_labels_l2 = batch.get('activity_labels_l2', ['Unknown'] * batch_size_actual)

                for i in range(batch_size_actual):
                    if samples_processed >= actual_samples:
                        break

                    # Use labels directly from the batch - this is the correct approach!
                    label_l1 = batch_labels_l1[i] if i < len(batch_labels_l1) else 'Unknown'
                    label_l2 = batch_labels_l2[i] if i < len(batch_labels_l2) else 'Unknown'
                    labels_l1.append(label_l1)
                    labels_l2.append(label_l2)
                    samples_processed += 1

                # Progress update
                if batch_idx % 20 == 0:
                    print(f"  Processed {samples_processed}/{actual_samples} samples...")

        # Concatenate embeddings
        embeddings = np.concatenate(embeddings, axis=0)
        print(f"📈 Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

        # Print label distribution
        l1_counter = Counter(labels_l1)
        l2_counter = Counter(labels_l2)
        print(f"📊 L1 Labels: {len(l1_counter)} unique ({l1_counter.most_common(5)})")
        print(f"📊 L2 Labels: {len(l2_counter)} unique ({l2_counter.most_common(5)})")

        # Filter noisy labels if requested
        if filter_noisy:
            original_size = len(embeddings)

            # Define noisy labels
            noisy_l1_labels = {'no_activity', 'Master_Bedroom_Activity', 'Meditate', 'Read', 'Watch_TV', 'Desk_Activity', 'Sleep'}
            noisy_l2_labels = {'No_Activity', 'Other'}

            # Create filter mask
            keep_mask = []
            for i, (l1, l2) in enumerate(zip(labels_l1, labels_l2)):
                keep = l1 not in noisy_l1_labels and l2 not in noisy_l2_labels
                keep_mask.append(keep)

            # Count removed labels (before filtering)
            original_labels_l1 = labels_l1.copy()
            original_labels_l2 = labels_l2.copy()

            removed_l1 = Counter([l1 for i, (l1, keep) in enumerate(zip(original_labels_l1, keep_mask)) if not keep])
            removed_l2 = Counter([l2 for i, (l2, keep) in enumerate(zip(original_labels_l2, keep_mask)) if not keep])

            # Apply filter
            keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
            embeddings = embeddings[keep_indices]
            labels_l1 = [labels_l1[i] for i in keep_indices]
            labels_l2 = [labels_l2[i] for i in keep_indices]

            print("🧹 Filtered out noisy labels:")
            print(f"   Original samples: {original_size}")
            print(f"   Filtered samples: {len(embeddings)}")
            print(f"   Removed: {original_size - len(embeddings)} samples")
            print(f"   Removed L1 labels: {dict(removed_l1)}")
            print(f"   Removed L2 labels: {dict(removed_l2)}")

        return {
            'embeddings': embeddings,
            'labels_l1': labels_l1,
            'labels_l2': labels_l2,
            'n_samples': len(embeddings)
        }

    def perform_clustering(self, embeddings: np.ndarray, n_clusters: int = 50) -> Dict[str, Any]:
        """Perform clustering using K-means (Euclidean & Cosine) and DBSCAN."""
        print(f"🔄 Performing clustering with {len(embeddings)} samples...")

        results = {}

        # K-means clustering (Euclidean distance)
        print("   Running K-means clustering (Euclidean)...")
        kmeans_euclidean = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_euclidean_labels = kmeans_euclidean.fit_predict(embeddings)

        # Calculate silhouette score for Euclidean K-means
        kmeans_euclidean_silhouette = silhouette_score(embeddings, kmeans_euclidean_labels)

        results['kmeans_euclidean'] = {
            'labels': kmeans_euclidean_labels,
            'n_clusters': len(np.unique(kmeans_euclidean_labels)),
            'silhouette_score': kmeans_euclidean_silhouette,
            'algorithm': 'K-means (Euclidean)'
        }

        print(f"   K-means (Euclidean): {results['kmeans_euclidean']['n_clusters']} clusters, silhouette: {kmeans_euclidean_silhouette:.4f}")

        # K-means clustering (Cosine distance)
        print("   Running K-means clustering (Cosine)...")
        kmeans_cosine = CosineKMeans(n_clusters=n_clusters, random_state=42)
        kmeans_cosine_labels = kmeans_cosine.fit_predict(embeddings)

        # Calculate silhouette score for Cosine K-means using cosine distance
        try:
            # For cosine silhouette, we need to use cosine distance metric
            cosine_dist_matrix = cosine_distances(embeddings)
            from sklearn.metrics import silhouette_score as silhouette_score_func
            kmeans_cosine_silhouette = silhouette_score_func(cosine_dist_matrix, kmeans_cosine_labels, metric='precomputed')
        except:
            # Fallback to regular silhouette score
            kmeans_cosine_silhouette = silhouette_score(embeddings, kmeans_cosine_labels)

        results['kmeans_cosine'] = {
            'labels': kmeans_cosine_labels,
            'n_clusters': len(np.unique(kmeans_cosine_labels)),
            'silhouette_score': kmeans_cosine_silhouette,
            'algorithm': 'K-means (Cosine)'
        }

        print(f"   K-means (Cosine): {results['kmeans_cosine']['n_clusters']} clusters, silhouette: {kmeans_cosine_silhouette:.4f}")

        # DBSCAN clustering
        print("   Running DBSCAN clustering...")
        # Use adaptive eps based on data scale
        eps = np.percentile(np.linalg.norm(embeddings, axis=1), 75) * 0.1
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = dbscan.fit_predict(embeddings)

        # Calculate silhouette score for DBSCAN (if we have more than 1 cluster)
        n_dbscan_clusters = len(np.unique(dbscan_labels[dbscan_labels != -1]))
        if n_dbscan_clusters > 1:
            # Only calculate silhouette for non-noise points
            non_noise_mask = dbscan_labels != -1
            if np.sum(non_noise_mask) > 1:
                dbscan_silhouette = silhouette_score(embeddings[non_noise_mask], dbscan_labels[non_noise_mask])
            else:
                dbscan_silhouette = -1.0
        else:
            dbscan_silhouette = -1.0

        n_noise = np.sum(dbscan_labels == -1)

        results['dbscan'] = {
            'labels': dbscan_labels,
            'n_clusters': n_dbscan_clusters,
            'n_noise': n_noise,
            'silhouette_score': dbscan_silhouette,
            'eps': eps,
            'algorithm': 'DBSCAN'
        }

        print(f"   DBSCAN: {n_dbscan_clusters} clusters, {n_noise} noise points, silhouette: {dbscan_silhouette:.4f}")

        return results

    def assign_cluster_labels(self, cluster_labels: np.ndarray, ground_truth_labels: List[str]) -> Tuple[Dict[int, str], np.ndarray]:
        """Assign majority ground truth label to each cluster."""
        cluster_to_label = {}
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue

            # Get ground truth labels for samples in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_gt_labels = [ground_truth_labels[i] for i in range(len(ground_truth_labels)) if cluster_mask[i]]

            # Assign majority label
            if cluster_gt_labels:
                majority_label = Counter(cluster_gt_labels).most_common(1)[0][0]
                cluster_to_label[cluster_id] = majority_label
            else:
                cluster_to_label[cluster_id] = 'Unknown'

        # Create predicted labels array
        predicted_labels = np.array(['Unknown'] * len(cluster_labels))
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id in cluster_to_label:
                predicted_labels[i] = cluster_to_label[cluster_id]
            else:
                predicted_labels[i] = 'Noise'  # For DBSCAN noise points

        return cluster_to_label, predicted_labels

    def compute_clustering_metrics(self, ground_truth: List[str], predicted: np.ndarray,
                                 cluster_labels: np.ndarray) -> Dict[str, float]:
        """Compute clustering and classification metrics."""
        # Filter out noise points for classification metrics
        valid_mask = predicted != 'Noise'
        gt_valid = [ground_truth[i] for i in range(len(ground_truth)) if valid_mask[i]]
        pred_valid = predicted[valid_mask]

        if len(gt_valid) == 0:
            return {
                'accuracy': 0.0,
                'f1_macro': 0.0,
                'f1_micro': 0.0,
                'f1_weighted': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'n_samples': 0,
                'n_classes': 0,
                'adjusted_rand_score': 0.0,
                'normalized_mutual_info': 0.0
            }

        # Encode labels for sklearn metrics
        le = LabelEncoder()
        all_labels = list(set(gt_valid + list(pred_valid)))
        le.fit(all_labels)

        gt_encoded = le.transform(gt_valid)
        pred_encoded = le.transform(pred_valid)

        # Classification metrics
        accuracy = accuracy_score(gt_encoded, pred_encoded)
        f1_macro = f1_score(gt_encoded, pred_encoded, average='macro', zero_division=0)
        f1_micro = f1_score(gt_encoded, pred_encoded, average='micro', zero_division=0)
        f1_weighted = f1_score(gt_encoded, pred_encoded, average='weighted', zero_division=0)
        precision_macro = precision_score(gt_encoded, pred_encoded, average='macro', zero_division=0)
        recall_macro = recall_score(gt_encoded, pred_encoded, average='macro', zero_division=0)

        # Clustering metrics (using all samples including noise)
        gt_all_encoded = le.fit_transform(ground_truth)

        # For clustering metrics, we need to handle the case where cluster_labels might have -1 (noise)
        cluster_labels_clean = cluster_labels.copy()
        if np.any(cluster_labels_clean == -1):
            # Assign unique cluster IDs to noise points
            max_cluster = np.max(cluster_labels_clean)
            noise_mask = cluster_labels_clean == -1
            cluster_labels_clean[noise_mask] = np.arange(max_cluster + 1, max_cluster + 1 + np.sum(noise_mask))

        try:
            ari = adjusted_rand_score(gt_all_encoded, cluster_labels_clean)
            nmi = normalized_mutual_info_score(gt_all_encoded, cluster_labels_clean)
            homogeneity = homogeneity_score(gt_all_encoded, cluster_labels_clean)
            completeness = completeness_score(gt_all_encoded, cluster_labels_clean)
            v_measure = v_measure_score(gt_all_encoded, cluster_labels_clean)
        except:
            ari = nmi = homogeneity = completeness = v_measure = 0.0

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'n_samples': len(gt_valid),
            'n_classes': len(np.unique(gt_encoded)),
            'adjusted_rand_score': ari,
            'normalized_mutual_info': nmi,
            'homogeneity_score': homogeneity,
            'completeness_score': completeness,
            'v_measure_score': v_measure,
            'per_class_metrics': self._compute_per_class_metrics(gt_valid, pred_valid) if len(gt_valid) > 0 else {}
        }

    def _compute_per_class_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute detailed per-class metrics."""
        try:
            # Get classification report
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

            per_class_stats = {}
            for class_name, metrics in report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    per_class_stats[class_name] = {
                        'precision': metrics.get('precision', 0.0),
                        'recall': metrics.get('recall', 0.0),
                        'f1_score': metrics.get('f1-score', 0.0),
                        'support': metrics.get('support', 0)
                    }

            return per_class_stats
        except:
            return {}

    def _compute_cluster_purity_stats(self, cluster_labels: np.ndarray, ground_truth_labels: List[str]) -> Dict[str, Any]:
        """Compute cluster purity and completeness statistics."""
        cluster_stats = {}
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue

            # Get samples in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_gt_labels = [ground_truth_labels[i] for i in range(len(ground_truth_labels)) if cluster_mask[i]]

            if not cluster_gt_labels:
                continue

            # Compute purity (most frequent class / total samples in cluster)
            label_counts = Counter(cluster_gt_labels)
            most_common_label, most_common_count = label_counts.most_common(1)[0]
            purity = most_common_count / len(cluster_gt_labels)

            # Compute size and distribution
            cluster_stats[f'cluster_{cluster_id}'] = {
                'size': len(cluster_gt_labels),
                'purity': purity,
                'dominant_class': most_common_label,
                'dominant_class_count': most_common_count,
                'class_distribution': dict(label_counts),
                'entropy': self._compute_entropy(list(label_counts.values()))
            }

        return cluster_stats

    def _compute_entropy(self, counts: List[int]) -> float:
        """Compute entropy of a distribution."""
        if not counts or sum(counts) == 0:
            return 0.0

        total = sum(counts)
        probs = [c / total for c in counts]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy

    def run_clustering_evaluation(self, split: str = 'test', max_samples: int = 10000,
                                n_clusters: int = 50, filter_noisy: bool = True) -> Dict[str, Any]:
        """Run complete clustering evaluation."""
        print("🚀 Starting clustering evaluation...")

        # Extract embeddings and labels
        data = self.extract_embeddings_and_labels(split, max_samples, filter_noisy)

        # Perform clustering
        clustering_results = self.perform_clustering(data['embeddings'], n_clusters)

        # Evaluate each clustering algorithm
        evaluation_results = {}

        for algorithm_name, cluster_result in clustering_results.items():
            print(f"\n🔄 Evaluating {algorithm_name.upper()} clustering...")

            cluster_labels = cluster_result['labels']

            # Evaluate L1 labels
            l1_cluster_to_label, l1_predicted = self.assign_cluster_labels(cluster_labels, data['labels_l1'])
            l1_metrics = self.compute_clustering_metrics(data['labels_l1'], l1_predicted, cluster_labels)

            # Evaluate L2 labels
            l2_cluster_to_label, l2_predicted = self.assign_cluster_labels(cluster_labels, data['labels_l2'])
            l2_metrics = self.compute_clustering_metrics(data['labels_l2'], l2_predicted, cluster_labels)

            # Compute cluster purity statistics
            l1_cluster_stats = self._compute_cluster_purity_stats(cluster_labels, data['labels_l1'])
            l2_cluster_stats = self._compute_cluster_purity_stats(cluster_labels, data['labels_l2'])

            evaluation_results[algorithm_name] = {
                'clustering_info': cluster_result,
                'l1_evaluation': {
                    'cluster_to_label': l1_cluster_to_label,
                    'predicted_labels': l1_predicted,
                    'metrics': l1_metrics,
                    'cluster_stats': l1_cluster_stats
                },
                'l2_evaluation': {
                    'cluster_to_label': l2_cluster_to_label,
                    'predicted_labels': l2_predicted,
                    'metrics': l2_metrics,
                    'cluster_stats': l2_cluster_stats
                }
            }

            # Print results
            print(f"✅ {algorithm_name.upper()} L1 Metrics:")
            print(f"    Accuracy: {l1_metrics['accuracy']:.4f}")
            print(f"    F1 (Macro): {l1_metrics['f1_macro']:.4f}")
            print(f"    F1 (Weighted): {l1_metrics['f1_weighted']:.4f}")
            print(f"    Classes: {l1_metrics['n_classes']}")
            print(f"    Samples: {l1_metrics['n_samples']}")

            print(f"✅ {algorithm_name.upper()} L2 Metrics:")
            print(f"    Accuracy: {l2_metrics['accuracy']:.4f}")
            print(f"    F1 (Macro): {l2_metrics['f1_macro']:.4f}")
            print(f"    F1 (Weighted): {l2_metrics['f1_weighted']:.4f}")
            print(f"    Classes: {l2_metrics['n_classes']}")
            print(f"    Samples: {l2_metrics['n_samples']}")

        return {
            'data': data,
            'evaluation_results': evaluation_results,
            'parameters': {
                'max_samples': max_samples,
                'n_clusters': n_clusters,
                'filter_noisy': filter_noisy
            }
        }

    def create_clustering_visualizations(self, results: Dict[str, Any], output_dir: Path):
        """Create comprehensive clustering visualization charts."""
        print("🔄 Creating clustering visualizations...")

        output_dir.mkdir(parents=True, exist_ok=True)
        evaluation_results = results['evaluation_results']

        # Extract model name and test data name for chart subtitles
        model_name = Path(self.config['checkpoint_path']).parent.name if 'checkpoint_path' in self.config else 'unknown_model'
        test_data_name = Path(self.config['test_data_path']).stem if 'test_data_path' in self.config else 'unknown_data'

        # 1. Create F1 scores comparison chart
        self._create_f1_comparison_chart(evaluation_results, output_dir, len(results['data']['embeddings']), model_name, test_data_name)

        # 2. Create confusion matrices
        self._create_confusion_matrices(evaluation_results, results['data'], output_dir)

        # 3. Create clustering metrics chart
        self._create_clustering_metrics_chart(evaluation_results, output_dir)

        # 4. Create cluster purity analysis chart
        self._create_cluster_purity_chart(evaluation_results, output_dir)

        # 5. Create per-class performance chart
        self._create_per_class_performance_chart(evaluation_results, output_dir)

        # 6. Create cluster size distribution chart
        self._create_cluster_size_distribution_chart(evaluation_results, output_dir)

        print("✅ Visualizations created")

    def _create_f1_comparison_chart(self, evaluation_results: Dict[str, Any], output_dir: Path, n_samples: int, model_name: str = "", test_data_name: str = ""):
        """Create F1 scores comparison chart."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.3, wspace=0.3)

        # Colors for algorithms
        colors = {
            'kmeans_euclidean': {'L1': '#1976D2', 'L2': '#FF6F00'},  # Blue and Orange
            'kmeans_cosine': {'L1': '#9C27B0', 'L2': '#FF9800'},     # Purple and Deep Orange
            'dbscan': {'L1': '#388E3C', 'L2': '#D32F2F'}             # Green and Red
        }

        # Overall metrics comparison
        ax_overall = fig.add_subplot(gs[0, :])

        metrics = ['f1_macro', 'f1_micro', 'f1_weighted', 'accuracy', 'precision_macro', 'recall_macro']
        metric_labels = ['F1 Macro', 'F1 Micro', 'F1 Weighted', 'Accuracy', 'Precision', 'Recall']

        x = np.arange(len(metrics))
        width = 0.15  # Reduced width to prevent overlap
        n_algorithms = len(evaluation_results)

        for i, (algorithm, results) in enumerate(evaluation_results.items()):
            l1_values = [results['l1_evaluation']['metrics'][m] for m in metrics]
            l2_values = [results['l2_evaluation']['metrics'][m] for m in metrics]

            # Calculate proper offsets for each algorithm and level
            base_offset = (i - (n_algorithms - 1) / 2) * width * 2.2  # Space between algorithms

            ax_overall.bar(x + base_offset - width*0.6, l1_values, width,
                          label=f'{algorithm.upper()} L1', color=colors[algorithm]['L1'], alpha=0.8)
            ax_overall.bar(x + base_offset + width*0.6, l2_values, width,
                          label=f'{algorithm.upper()} L2', color=colors[algorithm]['L2'], alpha=0.8)

        ax_overall.set_xlabel('Metrics')
        ax_overall.set_ylabel('Score')
        # Set title with subtitle if available
        title = 'Clustering-based Classification Performance Comparison'
        if model_name and test_data_name:
            title += f'\nModel: {model_name} | Data: {test_data_name}'
        ax_overall.set_title(title, fontsize=14, fontweight='bold')
        ax_overall.set_xticks(x)
        ax_overall.set_xticklabels(metric_labels)
        ax_overall.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_overall.grid(True, alpha=0.3)
        ax_overall.set_ylim(0, 1.0)

        # Per-class F1 scores for L1 (best performing K-means)
        best_kmeans = 'kmeans_euclidean'  # Default
        if 'kmeans_cosine' in evaluation_results and 'kmeans_euclidean' in evaluation_results:
            # Choose the one with better L1 F1-macro performance
            cosine_f1 = evaluation_results['kmeans_cosine']['l1_evaluation']['metrics']['f1_macro']
            euclidean_f1 = evaluation_results['kmeans_euclidean']['l1_evaluation']['metrics']['f1_macro']
            best_kmeans = 'kmeans_cosine' if cosine_f1 > euclidean_f1 else 'kmeans_euclidean'
        elif 'kmeans_cosine' in evaluation_results:
            best_kmeans = 'kmeans_cosine'

        if best_kmeans in evaluation_results:
            ax_l1 = fig.add_subplot(gs[1, 0])
            title = f'{evaluation_results[best_kmeans]["clustering_info"]["algorithm"]} L1 Per-Class F1 Scores'
            self._plot_per_class_f1(ax_l1, evaluation_results[best_kmeans], 'l1_evaluation',
                                   title, colors[best_kmeans]['L1'])

        # Per-class F1 scores for L2 (best performing K-means)
        if best_kmeans in evaluation_results:
            ax_l2 = fig.add_subplot(gs[1, 1])
            title = f'{evaluation_results[best_kmeans]["clustering_info"]["algorithm"]} L2 Per-Class F1 Scores'
            self._plot_per_class_f1(ax_l2, evaluation_results[best_kmeans], 'l2_evaluation',
                                   title, colors[best_kmeans]['L2'])

        plt.tight_layout()
        output_path = output_dir / f"clustering_f1_comparison_{n_samples}samples.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 F1 comparison chart saved: {output_path}")

    def _plot_per_class_f1(self, ax, algorithm_results: Dict[str, Any], level: str, title: str, color: str):
        """Plot per-class F1 scores."""
        # This is a simplified version - in practice, you'd need to compute per-class F1 scores
        # For now, we'll show overall metrics as a placeholder
        metrics = algorithm_results[level]['metrics']

        metric_names = ['F1 Macro', 'F1 Weighted', 'Accuracy', 'Precision', 'Recall']
        values = [metrics['f1_macro'], metrics['f1_weighted'], metrics['accuracy'],
                 metrics['precision_macro'], metrics['recall_macro']]

        bars = ax.barh(metric_names, values, color=color, alpha=0.7)
        ax.set_xlabel('Score')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', va='center', fontsize=10)

    def _create_confusion_matrices(self, evaluation_results: Dict[str, Any], data: Dict[str, Any], output_dir: Path):
        """Create confusion matrices for clustering results."""
        n_algorithms = len(evaluation_results)
        fig, axes = plt.subplots(2, n_algorithms, figsize=(6*n_algorithms, 10))

        if n_algorithms == 1:
            axes = axes.reshape(2, 1)

        for i, (algorithm, results) in enumerate(evaluation_results.items()):
            # L1 confusion matrix
            ax_l1 = axes[0, i]
            self._plot_confusion_matrix(ax_l1, data['labels_l1'],
                                      results['l1_evaluation']['predicted_labels'],
                                      f'{algorithm.upper()} L1 Activities', 'Blues')

            # L2 confusion matrix
            ax_l2 = axes[1, i]
            self._plot_confusion_matrix(ax_l2, data['labels_l2'],
                                      results['l2_evaluation']['predicted_labels'],
                                      f'{algorithm.upper()} L2 Activities', 'Oranges')

        plt.tight_layout()
        output_path = output_dir / f"clustering_confusion_matrices_{len(data['embeddings'])}samples.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 Confusion matrices saved: {output_path}")

    def _plot_confusion_matrix(self, ax, y_true: List[str], y_pred: np.ndarray, title: str, colormap: str):
        """Plot a single confusion matrix."""
        # Filter out noise points
        valid_mask = y_pred != 'Noise'
        y_true_valid = [y_true[i] for i in range(len(y_true)) if valid_mask[i]]
        y_pred_valid = y_pred[valid_mask]

        if len(y_true_valid) == 0:
            ax.text(0.5, 0.5, 'No valid predictions', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return

        # Get unique labels
        labels = sorted(list(set(y_true_valid) | set(y_pred_valid)))

        # Limit to top 12 classes for readability
        if len(labels) > 12:
            label_counts = Counter(y_true_valid)
            top_labels = [label for label, _ in label_counts.most_common(12)]

            # Filter data to top labels
            mask = [(true in top_labels and pred in top_labels)
                   for true, pred in zip(y_true_valid, y_pred_valid)]
            y_true_filtered = [y_true_valid[i] for i, keep in enumerate(mask) if keep]
            y_pred_filtered = [y_pred_valid[i] for i, keep in enumerate(mask) if keep]
            labels = top_labels

            y_true_valid = y_true_filtered
            y_pred_valid = y_pred_filtered

        # Create confusion matrix
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=labels)

        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

        # Plot
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=colormap, vmin=0, vmax=1)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add labels
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)

        # Truncate long labels
        short_labels = [label[:12] + '...' if len(label) > 12 else label for label in labels]
        ax.set_xticklabels(short_labels, rotation=45, ha='right')
        ax.set_yticklabels(short_labels)

        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(title)

    def _create_clustering_metrics_chart(self, evaluation_results: Dict[str, Any], output_dir: Path):
        """Create clustering quality metrics chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        algorithms = list(evaluation_results.keys())

        # Clustering quality metrics
        silhouette_scores = [evaluation_results[alg]['clustering_info']['silhouette_score'] for alg in algorithms]
        n_clusters = [evaluation_results[alg]['clustering_info']['n_clusters'] for alg in algorithms]

        x = np.arange(len(algorithms))
        width = 0.35

        ax1.bar(x - width/2, silhouette_scores, width, label='Silhouette Score', color='skyblue', alpha=0.7)
        ax1_twin = ax1.twinx()
        ax1_twin.bar(x + width/2, n_clusters, width, label='Number of Clusters', color='lightcoral', alpha=0.7)

        ax1.set_xlabel('Clustering Algorithm')
        ax1.set_ylabel('Silhouette Score', color='blue')
        ax1_twin.set_ylabel('Number of Clusters', color='red')
        ax1.set_title('Clustering Quality Metrics')
        ax1.set_xticks(x)
        ax1.set_xticklabels([alg.upper() for alg in algorithms])

        # Add value labels
        for i, (sil, n_clust) in enumerate(zip(silhouette_scores, n_clusters)):
            ax1.text(i - width/2, sil + 0.01, f'{sil:.3f}', ha='center', va='bottom')
            ax1_twin.text(i + width/2, n_clust + 0.5, f'{n_clust}', ha='center', va='bottom')

        # External clustering validation metrics
        ari_scores = [evaluation_results[alg]['l1_evaluation']['metrics']['adjusted_rand_score'] for alg in algorithms]
        nmi_scores = [evaluation_results[alg]['l1_evaluation']['metrics']['normalized_mutual_info'] for alg in algorithms]

        ax2.bar(x - width/2, ari_scores, width, label='Adjusted Rand Index', color='lightgreen', alpha=0.7)
        ax2.bar(x + width/2, nmi_scores, width, label='Normalized Mutual Info', color='gold', alpha=0.7)

        ax2.set_xlabel('Clustering Algorithm')
        ax2.set_ylabel('Score')
        ax2.set_title('External Validation Metrics (vs L1 Labels)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([alg.upper() for alg in algorithms])
        ax2.legend()
        ax2.set_ylim(0, 1.0)

        # Add value labels
        for i, (ari, nmi) in enumerate(zip(ari_scores, nmi_scores)):
            ax2.text(i - width/2, ari + 0.01, f'{ari:.3f}', ha='center', va='bottom')
            ax2.text(i + width/2, nmi + 0.01, f'{nmi:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        output_path = output_dir / "clustering_quality_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 Clustering quality metrics saved: {output_path}")

    def _create_cluster_purity_chart(self, evaluation_results: Dict[str, Any], output_dir: Path):
        """Create cluster purity analysis chart."""
        n_algorithms = len(evaluation_results)
        fig, axes = plt.subplots(2, n_algorithms, figsize=(6*n_algorithms, 12))
        if n_algorithms == 1:
            axes = axes.reshape(2, 1)
        fig.suptitle('Cluster Purity Analysis', fontsize=16, fontweight='bold')

        algorithms = list(evaluation_results.keys())
        colors = {
            'kmeans_euclidean': '#1976D2',
            'kmeans_cosine': '#9C27B0',
            'dbscan': '#388E3C'
        }

        for i, algorithm in enumerate(algorithms):
            l1_stats = evaluation_results[algorithm]['l1_evaluation']['cluster_stats']
            l2_stats = evaluation_results[algorithm]['l2_evaluation']['cluster_stats']

            # L1 Purity Distribution
            ax_l1_purity = axes[0, i]
            if l1_stats:
                purities = [stats['purity'] for stats in l1_stats.values()]
                sizes = [stats['size'] for stats in l1_stats.values()]

                scatter = ax_l1_purity.scatter(sizes, purities, alpha=0.7, c=colors.get(algorithm, '#666666'))
                ax_l1_purity.set_xlabel('Cluster Size')
                ax_l1_purity.set_ylabel('Purity')
                ax_l1_purity.set_title(f'{algorithm.upper()} L1 Cluster Purity vs Size')
                ax_l1_purity.grid(True, alpha=0.3)

                # Add trend line
                if len(sizes) > 1:
                    z = np.polyfit(sizes, purities, 1)
                    p = np.poly1d(z)
                    ax_l1_purity.plot(sorted(sizes), p(sorted(sizes)), "--", alpha=0.8, color=colors.get(algorithm, '#666666'))

            # L2 Purity Distribution
            ax_l2_purity = axes[1, i]
            if l2_stats:
                purities = [stats['purity'] for stats in l2_stats.values()]
                sizes = [stats['size'] for stats in l2_stats.values()]

                scatter = ax_l2_purity.scatter(sizes, purities, alpha=0.7, c=colors.get(algorithm, '#666666'))
                ax_l2_purity.set_xlabel('Cluster Size')
                ax_l2_purity.set_ylabel('Purity')
                ax_l2_purity.set_title(f'{algorithm.upper()} L2 Cluster Purity vs Size')
                ax_l2_purity.grid(True, alpha=0.3)

                # Add trend line
                if len(sizes) > 1:
                    z = np.polyfit(sizes, purities, 1)
                    p = np.poly1d(z)
                    ax_l2_purity.plot(sorted(sizes), p(sorted(sizes)), "--", alpha=0.8, color=colors.get(algorithm, '#666666'))

        plt.tight_layout()
        output_path = output_dir / "cluster_purity_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 Cluster purity analysis saved: {output_path}")

    def _create_per_class_performance_chart(self, evaluation_results: Dict[str, Any], output_dir: Path):
        """Create per-class performance comparison chart."""
        n_algorithms = len(evaluation_results)
        fig, axes = plt.subplots(2, n_algorithms, figsize=(6*n_algorithms, 12))
        if n_algorithms == 1:
            axes = axes.reshape(2, 1)
        fig.suptitle('Per-Class Performance Analysis', fontsize=16, fontweight='bold')

        algorithms = list(evaluation_results.keys())
        colors = {
            'kmeans_euclidean': '#1976D2',
            'kmeans_cosine': '#9C27B0',
            'dbscan': '#388E3C'
        }

        for i, algorithm in enumerate(algorithms):
            l1_per_class = evaluation_results[algorithm]['l1_evaluation']['metrics'].get('per_class_metrics', {})
            l2_per_class = evaluation_results[algorithm]['l2_evaluation']['metrics'].get('per_class_metrics', {})

            # L1 Per-class F1 scores
            ax_l1 = axes[0, i]
            if l1_per_class:
                classes = list(l1_per_class.keys())[:10]  # Top 10 classes
                f1_scores = [l1_per_class[cls]['f1_score'] for cls in classes]

                bars = ax_l1.barh(range(len(classes)), f1_scores, color=colors.get(algorithm, '#666666'), alpha=0.7)
                ax_l1.set_yticks(range(len(classes)))
                ax_l1.set_yticklabels([cls[:12] + '...' if len(cls) > 12 else cls for cls in classes])
                ax_l1.set_xlabel('F1 Score')
                ax_l1.set_title(f'{algorithm.upper()} L1 Per-Class F1 Scores')
                ax_l1.grid(True, alpha=0.3)

                # Add value labels
                for bar, score in zip(bars, f1_scores):
                    ax_l1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                              f'{score:.3f}', va='center', fontsize=9)

            # L2 Per-class F1 scores
            ax_l2 = axes[1, i]
            if l2_per_class:
                classes = list(l2_per_class.keys())[:10]  # Top 10 classes
                f1_scores = [l2_per_class[cls]['f1_score'] for cls in classes]

                bars = ax_l2.barh(range(len(classes)), f1_scores, color=colors.get(algorithm, '#666666'), alpha=0.7)
                ax_l2.set_yticks(range(len(classes)))
                ax_l2.set_yticklabels([cls[:12] + '...' if len(cls) > 12 else cls for cls in classes])
                ax_l2.set_xlabel('F1 Score')
                ax_l2.set_title(f'{algorithm.upper()} L2 Per-Class F1 Scores')
                ax_l2.grid(True, alpha=0.3)

                # Add value labels
                for bar, score in zip(bars, f1_scores):
                    ax_l2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                              f'{score:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        output_path = output_dir / "per_class_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 Per-class performance chart saved: {output_path}")

    def _create_cluster_size_distribution_chart(self, evaluation_results: Dict[str, Any], output_dir: Path):
        """Create cluster size distribution chart."""
        fig, axes = plt.subplots(2, len(evaluation_results), figsize=(6*len(evaluation_results), 10))
        if len(evaluation_results) == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle('Cluster Size Distribution Analysis', fontsize=16, fontweight='bold')

        algorithms = list(evaluation_results.keys())
        colors = {'kmeans': '#1976D2', 'dbscan': '#388E3C'}

        for i, algorithm in enumerate(algorithms):
            l1_stats = evaluation_results[algorithm]['l1_evaluation']['cluster_stats']
            l2_stats = evaluation_results[algorithm]['l2_evaluation']['cluster_stats']

            # L1 Cluster sizes histogram
            ax_l1 = axes[0, i]
            if l1_stats:
                sizes = [stats['size'] for stats in l1_stats.values()]
                ax_l1.hist(sizes, bins=min(20, len(sizes)), alpha=0.7, color=colors.get(algorithm, '#666666'))
                ax_l1.set_xlabel('Cluster Size')
                ax_l1.set_ylabel('Number of Clusters')
                ax_l1.set_title(f'{algorithm.upper()} L1 Cluster Size Distribution')
                ax_l1.grid(True, alpha=0.3)

                # Add statistics
                mean_size = np.mean(sizes)
                std_size = np.std(sizes)
                ax_l1.axvline(mean_size, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_size:.1f}')
                ax_l1.legend()

            # L2 Cluster sizes histogram
            ax_l2 = axes[1, i]
            if l2_stats:
                sizes = [stats['size'] for stats in l2_stats.values()]
                ax_l2.hist(sizes, bins=min(20, len(sizes)), alpha=0.7, color=colors.get(algorithm, '#666666'))
                ax_l2.set_xlabel('Cluster Size')
                ax_l2.set_ylabel('Number of Clusters')
                ax_l2.set_title(f'{algorithm.upper()} L2 Cluster Size Distribution')
                ax_l2.grid(True, alpha=0.3)

                # Add statistics
                mean_size = np.mean(sizes)
                std_size = np.std(sizes)
                ax_l2.axvline(mean_size, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_size:.1f}')
                ax_l2.legend()

        plt.tight_layout()
        output_path = output_dir / "cluster_size_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 Cluster size distribution chart saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Clustering-based Activity Recognition Evaluation')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--test_data', required=True, help='Path to test dataset')
    parser.add_argument('--vocab', required=True, help='Path to vocabulary file')
    parser.add_argument('--output_dir', required=True, help='Path to output directory for results')
    parser.add_argument('--max_samples', type=int, default=10000, help='Maximum number of samples to use')
    parser.add_argument('--n_clusters', type=int, default=50, help='Number of clusters for K-means')
    parser.add_argument('--filter_noisy_labels', action='store_true', help='Filter out noisy labels')

    args = parser.parse_args()

    # Create config
    config = {
        'checkpoint_path': args.checkpoint,
        'vocab_path': args.vocab,
        'test_data_path': args.test_data
    }

    # Create evaluator
    evaluator = ClusteringEvaluator(config)

    # Run evaluation
    results = evaluator.run_clustering_evaluation(
        split='test',
        max_samples=args.max_samples,
        n_clusters=args.n_clusters,
        filter_noisy=args.filter_noisy_labels
    )

    # Create visualizations
    output_dir = Path(args.output_dir)
    evaluator.create_clustering_visualizations(results, output_dir)

    # Save results
    results_file = output_dir / f"clustering_evaluation_results_{args.max_samples}samples.json"

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_numpy(v) for k, v in obj.items()}  # Convert keys to strings
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    # Save results (excluding embeddings to keep file size reasonable)
    results_to_save = {
        'evaluation_results': convert_numpy(results['evaluation_results']),
        'parameters': results['parameters'],
        'n_samples': results['data']['n_samples']
    }

    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"💾 Results saved: {results_file}")
    print("✅ Clustering evaluation complete!")

    # Print detailed summary
    print("\n🎯 COMPREHENSIVE CLUSTERING EVALUATION SUMMARY")
    print("=" * 70)
    for algorithm, eval_results in results['evaluation_results'].items():
        print(f"\n{algorithm.upper()} CLUSTERING RESULTS:")
        print("-" * 40)

        # Clustering info
        clustering_info = eval_results['clustering_info']
        print(f"  📊 Clusters: {clustering_info['n_clusters']}")
        print(f"  📈 Silhouette Score: {clustering_info['silhouette_score']:.4f}")
        if 'n_noise' in clustering_info:
            print(f"  🔇 Noise Points: {clustering_info['n_noise']}")

        # L1 metrics
        l1_metrics = eval_results['l1_evaluation']['metrics']
        l1_cluster_stats = eval_results['l1_evaluation']['cluster_stats']
        print(f"\n  🎯 L1 (Primary Activities) Performance:")
        print(f"     Accuracy: {l1_metrics['accuracy']:.4f}")
        print(f"     F1-Macro: {l1_metrics['f1_macro']:.4f}")
        print(f"     F1-Weighted: {l1_metrics['f1_weighted']:.4f}")
        print(f"     Homogeneity: {l1_metrics['homogeneity_score']:.4f}")
        print(f"     Completeness: {l1_metrics['completeness_score']:.4f}")
        print(f"     V-Measure: {l1_metrics['v_measure_score']:.4f}")
        print(f"     Classes: {l1_metrics['n_classes']}")
        print(f"     Samples: {l1_metrics['n_samples']}")

        # L1 cluster purity stats
        if l1_cluster_stats:
            purities = [stats['purity'] for stats in l1_cluster_stats.values()]
            sizes = [stats['size'] for stats in l1_cluster_stats.values()]
            print(f"     Avg Cluster Purity: {np.mean(purities):.4f}")
            print(f"     Avg Cluster Size: {np.mean(sizes):.1f}")

        # L2 metrics
        l2_metrics = eval_results['l2_evaluation']['metrics']
        l2_cluster_stats = eval_results['l2_evaluation']['cluster_stats']
        print(f"\n  🎯 L2 (Secondary Activities) Performance:")
        print(f"     Accuracy: {l2_metrics['accuracy']:.4f}")
        print(f"     F1-Macro: {l2_metrics['f1_macro']:.4f}")
        print(f"     F1-Weighted: {l2_metrics['f1_weighted']:.4f}")
        print(f"     Homogeneity: {l2_metrics['homogeneity_score']:.4f}")
        print(f"     Completeness: {l2_metrics['completeness_score']:.4f}")
        print(f"     V-Measure: {l2_metrics['v_measure_score']:.4f}")
        print(f"     Classes: {l2_metrics['n_classes']}")
        print(f"     Samples: {l2_metrics['n_samples']}")

        # L2 cluster purity stats
        if l2_cluster_stats:
            purities = [stats['purity'] for stats in l2_cluster_stats.values()]
            sizes = [stats['size'] for stats in l2_cluster_stats.values()]
            print(f"     Avg Cluster Purity: {np.mean(purities):.4f}")
            print(f"     Avg Cluster Size: {np.mean(sizes):.1f}")

        # Top performing classes
        l1_per_class = l1_metrics.get('per_class_metrics', {})
        if l1_per_class:
            top_l1_classes = sorted(l1_per_class.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:3]
            print(f"\n  🏆 Top 3 L1 Classes:")
            for cls, metrics in top_l1_classes:
                print(f"     {cls}: F1={metrics['f1_score']:.3f}, Support={metrics['support']}")

        l2_per_class = l2_metrics.get('per_class_metrics', {})
        if l2_per_class:
            top_l2_classes = sorted(l2_per_class.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:3]
            print(f"\n  🏆 Top 3 L2 Classes:")
            for cls, metrics in top_l2_classes:
                print(f"     {cls}: F1={metrics['f1_score']:.3f}, Support={metrics['support']}")

    print(f"\n📁 All results and visualizations saved in: {output_dir}")
    print("✨ Analysis complete!")


if __name__ == "__main__":
    main()
