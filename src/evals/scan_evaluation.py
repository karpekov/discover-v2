#!/usr/bin/env python3

"""
SCAN Clustering Evaluation Script

This script dynamically detects the number of clusters from the SCAN model checkpoint,
so it works with models trained with any number of clusters (20, 50, etc.).

Sample Usage:
python src/evals/scan_evaluation.py \
    --scan_model_path trained_models/milan/scan_50clusters_finetune/best_model.pt \
    --baseline_model_path trained_models/milan/baseline_50/best_model.pt \
    --data_path data/processed/casas/milan/training_50/presegmented_test.json \
    --vocab_path data/processed/casas/milan/training_50/vocab.json \
    --output_dir results/evals/milan/baseline_50/scan_evaluations_50clusters \
    --max_samples 10000

"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.scan_model import SCANClusteringModel
from models.text_encoder import TextEncoder, build_text_encoder
from dataio.dataset import SmartHomeDataset
from utils.device_utils import get_optimal_device
from evals.evaluate_embeddings import EmbeddingEvaluator
from torch.utils.data import Dataset


class IndexedSubset(Dataset):
    """A subset dataset that preserves original indices for proper data alignment."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        sample = self.dataset[original_idx]
        # Add the original index to the sample for tracking
        sample['_original_idx'] = original_idx
        return sample


class SCANEvaluator:
    """
    Evaluator for SCAN clustering results with two evaluation methods:
    1. Confusion matrix argmax assignment
    2. Cluster centroid to text prototype matching
    """

    def __init__(
        self,
        scan_model_path: str,
        baseline_model_path: str,
        data_path: str,
        vocab_path: str,
        output_dir: str = "results/evals/milan/baseline_50/scan_evaluations"
    ):
        self.scan_model_path = scan_model_path
        self.baseline_model_path = baseline_model_path
        self.data_path = data_path
        self.vocab_path = vocab_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = get_optimal_device()
        print(f"Using device: {self.device}")

        # Initialize components
        self.scan_model = None
        self.text_encoder = None
        self.dataset = None
        self.embeddings = None
        self.cluster_predictions = None
        self.ground_truth_labels = None

        # For method 2 - will be populated later
        self.embedding_evaluator = None

    def load_models_and_data(self):
        """Load SCAN model, text encoder, and test dataset."""
        print("Loading models and data...")

        # 1. Load SCAN model
        print("Loading SCAN model...")

        # First load the checkpoint to get the number of clusters
        checkpoint = torch.load(self.scan_model_path, map_location='cpu')

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

        self.scan_model = SCANClusteringModel(
            pretrained_model_path=self.baseline_model_path,
            num_clusters=num_clusters,
            dropout=0.1,
            freeze_encoder=True  # This will be loaded from checkpoint anyway
        )

        # Load SCAN checkpoint
        self.scan_model.load_state_dict(checkpoint['model_state_dict'])
        self.scan_model = self.scan_model.to(self.device)
        self.scan_model.eval()

        # 2. Load text encoder (for method 2)
        print("Loading text encoder...")
        # Load config from baseline model to get text model name
        baseline_checkpoint = torch.load(self.baseline_model_path, map_location='cpu')
        text_model_name = baseline_checkpoint.get('config', {}).get('text_model_name', 'thenlper/gte-base')

        # Use text encoder factory to handle different encoder types


        eval_config = model_config.copy() if "model_config" in locals() else {"text_model_name": text_model_name}


        eval_config["use_cached_embeddings"] = False  # Compute embeddings on-the-fly for eval


        self.text_encoder = build_text_encoder(eval_config)
        self.text_encoder.to(self.device)
        self.text_encoder.eval()

        # 3. Load test dataset
        print("Loading test dataset...")
        self.dataset = SmartHomeDataset(
            data_path=self.data_path,
            vocab_path=self.vocab_path,
            sequence_length=50,
            max_captions=1,
            caption_types='long'
        )

        print(f"Loaded {len(self.dataset)} samples")

    def extract_embeddings_and_predictions(self, max_samples: int = 10000):
        """Extract embeddings, cluster predictions, and ground truth labels."""
        print(f"Extracting embeddings and predictions (max {max_samples} samples)...")

        # Create collator function
        def simple_collate_fn(batch):
            batch_size = len(batch)

            # Extract components
            all_categorical = [sample['categorical_features'] for sample in batch]
            all_coordinates = [sample['coordinates'] for sample in batch]
            all_time_deltas = [sample['time_deltas'] for sample in batch]
            all_masks = [sample['mask'] for sample in batch]

            # Extract original indices and activities for validation
            original_indices = [sample['_original_idx'] for sample in batch]
            activities = [sample.get('first_activity', 'Unknown') for sample in batch]

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
                'mask': masks,
                'original_indices': original_indices,
                'activities': activities
            }

        # Filter out "No_Activity" samples
        print("Filtering out 'No_Activity' samples...")
        valid_indices = []
        no_activity_variations = ['No_Activity', 'no_activity', 'No Activity', 'no activity', 'Unknown', 'unknown', '']

        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            activity = sample.get('first_activity', 'Unknown')
            if activity.strip().lower() not in [var.lower() for var in no_activity_variations]:
                valid_indices.append(i)

        print(f"Found {len(valid_indices)} valid samples (excluding No_Activity)")
        dataset_size = min(len(valid_indices), max_samples)
        selected_indices = np.random.choice(valid_indices, dataset_size, replace=False)

        # Use our custom IndexedSubset to preserve original indices
        subset_dataset = IndexedSubset(self.dataset, selected_indices)

        dataloader = DataLoader(
            dataset=subset_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            collate_fn=simple_collate_fn
        )

        embeddings = []
        clip_embeddings = []  # For method 2
        cluster_predictions = []
        ground_truth_labels = []

        # Initialize original indices buffer for Method 4
        self._orig_idx_buf = getattr(self, "_orig_idx_buf", [])

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Get embeddings and cluster predictions
                logits, batch_embeddings = self.scan_model(
                    categorical_features=batch['categorical_features'],
                    coordinates=batch['coordinates'],
                    time_deltas=batch['time_deltas'],
                    mask=batch['mask'],
                    return_embeddings=True
                )

                # Get cluster predictions
                batch_predictions = torch.argmax(logits, dim=1)

                # Get CLIP projected embeddings for method 2
                batch_clip_embeddings = self.scan_model.sensor_encoder.forward_clip(
                    categorical_features=batch['categorical_features'],
                    coordinates=batch['coordinates'],
                    time_deltas=batch['time_deltas'],
                    mask=batch['mask']
                )

                embeddings.append(batch_embeddings.cpu().numpy())
                clip_embeddings.append(batch_clip_embeddings.cpu().numpy())
                cluster_predictions.append(batch_predictions.cpu().numpy())

                # Extract ground truth labels directly from batch (much cleaner!)
                batch_gt_labels = batch['activities']
                ground_truth_labels.extend(batch_gt_labels)

                # Store original indices for Method 4
                self._orig_idx_buf.extend(batch['original_indices'])

                if batch_idx % 20 == 0:
                    print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
                    print(f"  Batch size: {len(batch_embeddings)}")
                    print(f"  Sample original indices: {batch['original_indices'][:3]}")
                    print(f"  Sample activities: {batch_gt_labels[:3]}")

                # Validation: ensure batch consistency
                assert len(batch_embeddings) == len(batch_gt_labels), \
                    f"Batch size mismatch: embeddings {len(batch_embeddings)} != labels {len(batch_gt_labels)}"
                assert len(batch_predictions) == len(batch_gt_labels), \
                    f"Batch size mismatch: predictions {len(batch_predictions)} != labels {len(batch_gt_labels)}"

        self.embeddings = np.vstack(embeddings)
        self.clip_embeddings = np.vstack(clip_embeddings)
        self.cluster_predictions = np.concatenate(cluster_predictions)
        self.ground_truth_labels = np.array(ground_truth_labels)

        # Store original indices for Method 4
        self.original_indices = np.array(self._orig_idx_buf)

        # Validate data alignment
        self._validate_data_alignment()

        print(f"Extracted embeddings shape: {self.embeddings.shape}")
        print(f"Extracted CLIP embeddings shape: {self.clip_embeddings.shape}")
        print(f"Unique ground truth labels: {len(np.unique(self.ground_truth_labels))}")
        print(f"Unique cluster predictions: {len(np.unique(self.cluster_predictions))}")

    def _validate_data_alignment(self):
        """Validate that all extracted data arrays are properly aligned."""
        n_samples = len(self.embeddings)

        # Check that all arrays have the same length
        assert len(self.clip_embeddings) == n_samples, f"CLIP embeddings length mismatch: {len(self.clip_embeddings)} != {n_samples}"
        assert len(self.cluster_predictions) == n_samples, f"Cluster predictions length mismatch: {len(self.cluster_predictions)} != {n_samples}"
        assert len(self.ground_truth_labels) == n_samples, f"Ground truth labels length mismatch: {len(self.ground_truth_labels)} != {n_samples}"

        print(f"✅ Data alignment validation passed: {n_samples} samples across all arrays")

        # Additional sanity checks
        print(f"📊 Sample validation:")
        print(f"  First 3 cluster predictions: {self.cluster_predictions[:3]}")
        print(f"  First 3 ground truth labels: {self.ground_truth_labels[:3]}")
        print(f"  Embeddings dtype: {self.embeddings.dtype}, CLIP embeddings dtype: {self.clip_embeddings.dtype}")

        # Check for any obvious issues
        if np.any(np.isnan(self.embeddings)):
            print("⚠️  Warning: NaN values found in embeddings")
        if np.any(np.isnan(self.clip_embeddings)):
            print("⚠️  Warning: NaN values found in CLIP embeddings")
        if len(np.unique(self.ground_truth_labels)) < 2:
            print("⚠️  Warning: Very few unique ground truth labels found")
        if len(np.unique(self.cluster_predictions)) < 2:
            print("⚠️  Warning: Very few unique cluster predictions found")

    def evaluate_method1_argmax_assignment(self) -> Dict[str, Any]:
        """
        Method 1: Confusion matrix argmax assignment.
        For each cluster, assign the most frequent ground truth label.
        """
        print("\n" + "="*60)
        print("METHOD 1: CONFUSION MATRIX ARGMAX ASSIGNMENT")
        print("="*60)

        # Create confusion matrix data
        unique_gt_labels = np.unique(self.ground_truth_labels)
        unique_clusters = np.unique(self.cluster_predictions)

        print(f"Ground truth labels: {len(unique_gt_labels)}")
        print(f"Predicted clusters: {len(unique_clusters)}")

        # Build confusion matrix: rows = ground truth, cols = clusters
        confusion_data = np.zeros((len(unique_gt_labels), len(unique_clusters)))

        gt_to_idx = {label: idx for idx, label in enumerate(unique_gt_labels)}
        cluster_to_idx = {cluster: idx for idx, cluster in enumerate(unique_clusters)}

        for gt_label, cluster_pred in zip(self.ground_truth_labels, self.cluster_predictions):
            gt_idx = gt_to_idx[gt_label]
            cluster_idx = cluster_to_idx[cluster_pred]
            confusion_data[gt_idx, cluster_idx] += 1

        # For each cluster, find the most frequent ground truth label (argmax)
        cluster_to_label_assignment = {}
        for cluster_idx, cluster_id in enumerate(unique_clusters):
            # Get the column for this cluster
            cluster_column = confusion_data[:, cluster_idx]
            # Find the ground truth label with maximum count
            max_gt_idx = np.argmax(cluster_column)
            assigned_label = unique_gt_labels[max_gt_idx]
            cluster_to_label_assignment[cluster_id] = assigned_label

            print(f"Cluster {cluster_id} -> {assigned_label} ({cluster_column[max_gt_idx]:.0f} samples)")

        # Create predictions based on cluster assignments
        predicted_labels = [cluster_to_label_assignment[cluster_id] for cluster_id in self.cluster_predictions]

        # Compute evaluation metrics
        metrics = self._compute_evaluation_metrics(
            self.ground_truth_labels,
            predicted_labels,
            "Method 1 (Argmax Assignment)"
        )

        # Save confusion matrix visualization
        self._create_cluster_confusion_matrix(
            confusion_data,
            unique_gt_labels,
            unique_clusters,
            title="Cluster-Ground Truth Confusion Matrix (Method 1)",
            save_path=self.output_dir / "method1_cluster_confusion_matrix.png"
        )

        return {
            'method': 'argmax_assignment',
            'cluster_assignments': cluster_to_label_assignment,
            'predicted_labels': predicted_labels,
            'metrics': metrics,
            'confusion_matrix': confusion_data
        }

    def evaluate_method2_centroid_matching(self) -> Dict[str, Any]:
        """
        Method 2: Cluster centroid to text prototype matching.
        Compute cluster centroids and match them to text prototypes.
        """
        print("\n" + "="*60)
        print("METHOD 2: CLUSTER CENTROID TO TEXT PROTOTYPE MATCHING")
        print("="*60)

        unique_clusters = np.unique(self.cluster_predictions)
        unique_gt_labels = np.unique(self.ground_truth_labels)

        print(f"Computing centroids for {len(unique_clusters)} clusters...")

        # Compute cluster centroids using CLIP projected embeddings
        cluster_centroids = {}
        for cluster_id in unique_clusters:
            cluster_mask = self.cluster_predictions == cluster_id
            cluster_embeddings = self.clip_embeddings[cluster_mask]

            # Use mean as centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            cluster_centroids[cluster_id] = centroid

            print(f"Cluster {cluster_id}: {np.sum(cluster_mask)} samples")

        # Create text prototypes using the same method as in evaluate_embeddings.py
        print("Creating text prototypes...")

        # Use embedding evaluator's method to create text prototypes
        # First, we need to create a temporary config and evaluator
        temp_config = {
            'checkpoint_path': self.baseline_model_path,
            'train_data_path': None,  # We'll use ground truth labels directly
            'test_data_path': self.data_path,
            'vocab_path': self.vocab_path,
            'output_dir': str(self.output_dir)
        }

        self.embedding_evaluator = EmbeddingEvaluator(temp_config)

        # Create text prototypes for unique ground truth labels
        prototypes, label_counts = self.embedding_evaluator.create_text_prototypes(unique_gt_labels.tolist())

        print(f"Created {len(prototypes)} text prototypes")

        # Match cluster centroids to text prototypes using cosine similarity
        print("Matching cluster centroids to text prototypes...")

        # Convert prototypes to array
        prototype_labels = list(prototypes.keys())
        prototype_embeddings = np.array([prototypes[label] for label in prototype_labels])

        # Normalize embeddings for cosine similarity
        centroid_embeddings = np.array([cluster_centroids[cluster_id] for cluster_id in unique_clusters])
        centroid_embeddings_norm = centroid_embeddings / (np.linalg.norm(centroid_embeddings, axis=1, keepdims=True) + 1e-8)
        prototype_embeddings_norm = prototype_embeddings / (np.linalg.norm(prototype_embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarities
        similarities = np.dot(centroid_embeddings_norm, prototype_embeddings_norm.T)

        # For each cluster, find the most similar text prototype
        cluster_to_label_assignment = {}
        for i, cluster_id in enumerate(unique_clusters):
            best_prototype_idx = np.argmax(similarities[i])
            assigned_label = prototype_labels[best_prototype_idx]
            similarity_score = similarities[i, best_prototype_idx]

            cluster_to_label_assignment[cluster_id] = assigned_label
            print(f"Cluster {cluster_id} -> {assigned_label} (similarity: {similarity_score:.4f})")

        # Create predictions based on cluster assignments
        predicted_labels = [cluster_to_label_assignment[cluster_id] for cluster_id in self.cluster_predictions]

        # Compute evaluation metrics
        metrics = self._compute_evaluation_metrics(
            self.ground_truth_labels,
            predicted_labels,
            "Method 2 (Centroid Matching)"
        )

        # Save similarity matrix visualization
        self._create_similarity_matrix(
            similarities,
            unique_clusters,
            prototype_labels,
            title="Cluster Centroid to Text Prototype Similarity Matrix (Method 2)",
            save_path=self.output_dir / "method2_similarity_matrix.png"
        )

        return {
            'method': 'centroid_matching',
            'cluster_centroids': cluster_centroids,
            'text_prototypes': prototypes,
            'cluster_assignments': cluster_to_label_assignment,
            'predicted_labels': predicted_labels,
            'metrics': metrics,
            'similarity_matrix': similarities
        }

    def evaluate_method3_prototype_argmax(self) -> Dict[str, Any]:
        """
        Method 3: Prototype argmax method with majority voting.
        For each sample in each cluster, find the closest text prototype,
        then use majority voting to assign cluster labels.
        """
        print("\n" + "="*60)
        print("METHOD 3: PROTOTYPE ARGMAX WITH MAJORITY VOTING")
        print("="*60)

        unique_clusters = np.unique(self.cluster_predictions)
        unique_gt_labels = np.unique(self.ground_truth_labels)

        print(f"Analyzing {len(unique_clusters)} clusters with individual sample prototype matching...")

        # Create text prototypes (reuse from method 2 if available)
        if not hasattr(self, 'embedding_evaluator') or self.embedding_evaluator is None:
            print("Creating text prototypes...")
            temp_config = {
                'checkpoint_path': self.baseline_model_path,
                'train_data_path': None,
                'test_data_path': self.data_path,
                'vocab_path': self.vocab_path,
                'output_dir': str(self.output_dir)
            }
            self.embedding_evaluator = EmbeddingEvaluator(temp_config)

        # Create text prototypes for unique ground truth labels
        prototypes, label_counts = self.embedding_evaluator.create_text_prototypes(unique_gt_labels.tolist())

        print(f"Created {len(prototypes)} text prototypes")

        # Convert prototypes to array for efficient computation
        prototype_labels = list(prototypes.keys())
        prototype_embeddings = np.array([prototypes[label] for label in prototype_labels])

        # Normalize embeddings for cosine similarity
        clip_embeddings_norm = self.clip_embeddings / (np.linalg.norm(self.clip_embeddings, axis=1, keepdims=True) + 1e-8)
        prototype_embeddings_norm = prototype_embeddings / (np.linalg.norm(prototype_embeddings, axis=1, keepdims=True) + 1e-8)

        print("Computing prototype assignments for each sample...")

        # For each sample, find the closest prototype
        similarities_all = np.dot(clip_embeddings_norm, prototype_embeddings_norm.T)
        sample_prototype_assignments = np.argmax(similarities_all, axis=1)
        sample_assigned_labels = [prototype_labels[idx] for idx in sample_prototype_assignments]

        # For each cluster, perform majority voting
        cluster_to_label_assignment = {}
        cluster_voting_details = {}

        # Validate that arrays are aligned before processing
        assert len(sample_assigned_labels) == len(self.cluster_predictions), \
            f"Sample assignments length mismatch: {len(sample_assigned_labels)} != {len(self.cluster_predictions)}"

        for cluster_id in unique_clusters:
            cluster_mask = self.cluster_predictions == cluster_id
            cluster_sample_labels = [sample_assigned_labels[i] for i in range(len(sample_assigned_labels)) if cluster_mask[i]]

            # Additional validation
            cluster_size = np.sum(cluster_mask)
            assert len(cluster_sample_labels) == cluster_size, \
                f"Cluster {cluster_id} sample labels mismatch: {len(cluster_sample_labels)} != {cluster_size}"

            # Count votes for each label in this cluster
            label_votes = Counter(cluster_sample_labels)

            # Assign the most frequent label
            if label_votes:
                assigned_label, vote_count = label_votes.most_common(1)[0]
                cluster_to_label_assignment[cluster_id] = assigned_label
                cluster_voting_details[cluster_id] = dict(label_votes)

                total_samples = np.sum(cluster_mask)
                print(f"Cluster {cluster_id} -> {assigned_label} ({vote_count}/{total_samples} votes, {vote_count/total_samples:.2%})")

                # Show top 3 vote getters if there are multiple candidates
                if len(label_votes) > 1:
                    top_votes = label_votes.most_common(3)
                    vote_summary = ", ".join([f"{label}: {count}" for label, count in top_votes])
                    print(f"  Vote breakdown: {vote_summary}")
            else:
                # Fallback (shouldn't happen)
                cluster_to_label_assignment[cluster_id] = unique_gt_labels[0]
                cluster_voting_details[cluster_id] = {}

        # Create predictions based on cluster assignments
        predicted_labels = [cluster_to_label_assignment[cluster_id] for cluster_id in self.cluster_predictions]

        # Compute evaluation metrics
        metrics = self._compute_evaluation_metrics(
            self.ground_truth_labels,
            predicted_labels,
            "Method 3 (Prototype Argmax)"
        )

        # Create voting analysis visualization
        self._create_voting_analysis_visualization(
            cluster_voting_details,
            unique_clusters,
            title="Cluster Prototype Voting Analysis (Method 3)",
            save_path=self.output_dir / "method3_voting_analysis.png"
        )

        # Create sample-level similarity matrix (for a subset of samples for visualization)
        self._create_sample_prototype_similarity_matrix(
            similarities_all[:min(100, len(similarities_all))],  # Limit to first 100 samples for readability
            prototype_labels,
            title="Sample-Prototype Similarity Matrix (Method 3, First 100 Samples)",
            save_path=self.output_dir / "method3_sample_similarity_matrix.png"
        )

        return {
            'method': 'prototype_argmax',
            'text_prototypes': prototypes,
            'cluster_assignments': cluster_to_label_assignment,
            'predicted_labels': predicted_labels,
            'metrics': metrics,
            'voting_details': cluster_voting_details,
            'sample_assignments': sample_assigned_labels,
            'similarity_matrix': similarities_all
        }

    def evaluate_method4_caption_clusters(self) -> Dict[str, Any]:
        """
        Method 4: Caption Clusters (Text-Space).
        For each cluster, collect all captions of member samples, encode them,
        and compute cluster centroids in text embedding space.
        """
        print("\n" + "="*60)
        print("METHOD 4: CAPTION CLUSTERS (TEXT-SPACE)")
        print("="*60)

        unique_clusters = np.unique(self.cluster_predictions)
        unique_gt_labels = np.unique(self.ground_truth_labels)

        print(f"Analyzing {len(unique_clusters)} clusters using caption embeddings...")

        # Ensure we have text encoder available
        if not hasattr(self, 'embedding_evaluator') or self.embedding_evaluator is None:
            print("Creating text encoder...")
            temp_config = {
                'checkpoint_path': self.baseline_model_path,
                'train_data_path': None,
                'test_data_path': self.data_path,
                'vocab_path': self.vocab_path,
                'output_dir': str(self.output_dir)
            }
            self.embedding_evaluator = EmbeddingEvaluator(temp_config)

        # Step 1: Build cluster captions from actual cluster members (not global activity pools)
        print("Collecting captions from cluster members...")
        cluster_captions = {}
        caption_collection_stats = {}

        for cluster_id in unique_clusters:
            cluster_mask = (self.cluster_predictions == cluster_id)
            member_indices = self.original_indices[cluster_mask]

            captions = []
            for original_idx in member_indices:
                sample = self.dataset[original_idx]
                sample_captions = sample.get('captions', [])

                if isinstance(sample_captions, list):
                    for item in sample_captions:
                        if isinstance(item, str):
                            captions.append(item.strip())
                        elif isinstance(item, list) and item and isinstance(item[0], str):
                            captions.append(item[0].strip())

            # Deduplicate and cap to 200 captions per cluster
            captions = list({c for c in captions if c})
            if len(captions) > 200:
                rng = np.random.default_rng(42)
                captions = list(rng.choice(captions, 200, replace=False))

            cluster_captions[cluster_id] = captions
            caption_collection_stats[cluster_id] = {
                'total_samples': int(cluster_mask.sum()),
                'samples_with_captions': int(cluster_mask.sum()),  # All samples have captions in our case
                'unique_captions': len(captions),
                'caption_coverage': int(len(captions) > 0)
            }

            print(f"Cluster {cluster_id}: {len(captions)} unique captions from {len(member_indices)} samples")

        # Step 2: Compute robust text centroids (spherical mean)
        print("Computing cluster centroids in text embedding space...")
        cluster_text_centroids = {}

        for cluster_id, captions in cluster_captions.items():
            if not captions:
                print(f"Warning: No captions found for cluster {cluster_id}, skipping...")
                continue

            # Encode all captions for this cluster
            with torch.no_grad():
                E = self.text_encoder.encode_texts_clip(captions, self.device).cpu().numpy()

            # Normalize embeddings and compute spherical mean
            E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
            m = E.mean(axis=0)
            m = m / (np.linalg.norm(m) + 1e-8)

            cluster_text_centroids[cluster_id] = m
            print(f"Cluster {cluster_id}: Computed spherical mean from {len(captions)} captions")

        # Step 3: Create label prototypes using existing descriptions
        print("Creating label prototypes...")
        prototypes, label_counts = self.embedding_evaluator.create_text_prototypes(unique_gt_labels.tolist())

        prototype_labels = list(prototypes.keys())
        prototype_embeddings = np.array([prototypes[label] for label in prototype_labels])

        # Step 4: Label assignment with Hungarian algorithm (avoid collapse)
        print("Assigning labels using Hungarian algorithm...")

        # Normalize prototype embeddings
        P = prototype_embeddings / (np.linalg.norm(prototype_embeddings, axis=1, keepdims=True) + 1e-8)

        # Get valid cluster centroids
        valid_cluster_ids = sorted(cluster_text_centroids.keys())
        C = np.array([cluster_text_centroids[cid] for cid in valid_cluster_ids])
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarities
        S = C @ P.T

        # Apply Hungarian algorithm on square submatrix
        M = min(len(valid_cluster_ids), len(prototype_labels))
        row_ind, col_ind = linear_sum_assignment(-S[:M, :M])  # Negative for maximization

        # Build assignment dictionary
        cluster_to_label_assignment = {}
        similarity_scores = {}

        # Assign matched pairs
        for r, c in zip(row_ind, col_ind):
            cluster_id = valid_cluster_ids[r]
            assigned_label = prototype_labels[c]
            similarity = S[r, c]

            cluster_to_label_assignment[cluster_id] = assigned_label
            similarity_scores[cluster_id] = similarity
            print(f"Cluster {cluster_id} -> {assigned_label} (similarity: {similarity:.4f}) [Hungarian]")

        # Assign leftover clusters using argmax
        for i, cluster_id in enumerate(valid_cluster_ids):
            if cluster_id not in cluster_to_label_assignment:
                best_prototype_idx = int(np.argmax(S[i]))
                assigned_label = prototype_labels[best_prototype_idx]
                similarity = S[i, best_prototype_idx]

                cluster_to_label_assignment[cluster_id] = assigned_label
                similarity_scores[cluster_id] = similarity
                print(f"Cluster {cluster_id} -> {assigned_label} (similarity: {similarity:.4f}) [Argmax]")

        # Handle clusters without captions (fallback)
        for cluster_id in unique_clusters:
            if cluster_id not in cluster_to_label_assignment:
                cluster_to_label_assignment[cluster_id] = prototype_labels[0]
                similarity_scores[cluster_id] = 0.0
                print(f"Cluster {cluster_id} -> {prototype_labels[0]} (similarity: 0.0000) [Fallback]")

        # Step 5: Sanity checks and logging
        print("\nSanity checks:")

        # Check for label collapse
        assigned_labels = list(cluster_to_label_assignment.values())
        label_counts_assigned = Counter(assigned_labels)
        most_common_label, most_common_count = label_counts_assigned.most_common(1)[0]
        collapse_ratio = most_common_count / len(assigned_labels)

        if collapse_ratio > 0.5:
            print(f"⚠️  WARNING: {collapse_ratio:.1%} of clusters assigned to '{most_common_label}' - possible label collapse!")
        else:
            print(f"✅ Label distribution looks good (max {collapse_ratio:.1%} for '{most_common_label}')")

        # Check prototype similarity
        pairwise_sims = P @ P.T
        median_prototype_sim = np.median(pairwise_sims[np.triu_indices_from(pairwise_sims, k=1)])

        if median_prototype_sim > 0.8:
            print(f"⚠️  WARNING: High median pairwise cosine similarity among prototypes: {median_prototype_sim:.3f}")
        else:
            print(f"✅ Prototype diversity looks good (median pairwise sim: {median_prototype_sim:.3f})")

        # Log per-cluster statistics
        print(f"\nCluster caption statistics:")
        for cluster_id in sorted(caption_collection_stats.keys()):
            stats = caption_collection_stats[cluster_id]
            print(f"  Cluster {cluster_id}: {stats['unique_captions']} captions, coverage: {stats['caption_coverage']}")

        # Step 6: Create predictions and evaluate
        predicted_labels = [cluster_to_label_assignment.get(cluster_id, prototype_labels[0]) for cluster_id in self.cluster_predictions]

        # Compute evaluation metrics
        metrics = self._compute_evaluation_metrics(
            self.ground_truth_labels,
            predicted_labels,
            "Method 4 (Caption Clusters)"
        )

        # Create visualization showing caption statistics
        self._create_caption_analysis_visualization(
            caption_collection_stats,
            unique_clusters,
            title="Caption Collection Statistics (Method 4)",
            save_path=self.output_dir / "method4_caption_statistics.png"
        )

        # Create similarity matrix for cluster centroids vs prototypes
        if cluster_text_centroids:
            centroid_embeddings = np.array([cluster_text_centroids[cid] for cid in sorted(cluster_text_centroids.keys())])
            similarities_matrix = np.dot(centroid_embeddings, P.T)  # P is the normalized prototype embeddings

            self._create_caption_similarity_matrix(
                similarities_matrix,
                sorted(cluster_text_centroids.keys()),
                prototype_labels,
                title="Caption Cluster Centroids vs Label Prototypes (Method 4)",
                save_path=self.output_dir / "method4_caption_similarity_matrix.png"
            )

        return {
            'method': 'caption_clusters',
            'cluster_captions': cluster_captions,
            'cluster_text_centroids': cluster_text_centroids,
            'text_prototypes': prototypes,
            'cluster_assignments': cluster_to_label_assignment,
            'predicted_labels': predicted_labels,
            'metrics': metrics,
            'similarity_scores': similarity_scores,
            'caption_stats': caption_collection_stats
        }

    def _compute_evaluation_metrics(self, true_labels, pred_labels, method_name: str) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
        print(f"\nComputing metrics for {method_name}...")

        # Convert to lists if they are numpy arrays
        if hasattr(true_labels, 'tolist'):
            true_labels = true_labels.tolist()
        if hasattr(pred_labels, 'tolist'):
            pred_labels = pred_labels.tolist()

        # Get unique labels
        unique_labels = sorted(list(set(true_labels + pred_labels)))

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'f1_macro': f1_score(true_labels, pred_labels, average='macro', zero_division=0),
            'f1_weighted': f1_score(true_labels, pred_labels, average='weighted', zero_division=0),
            'precision_macro': precision_score(true_labels, pred_labels, average='macro', zero_division=0),
            'recall_macro': recall_score(true_labels, pred_labels, average='macro', zero_division=0),
            'num_samples': len(true_labels),
            'num_classes': len(unique_labels),
            'unique_labels': unique_labels
        }

        # Per-class F1 scores
        per_class_f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0, labels=unique_labels)
        metrics['per_class_f1'] = dict(zip(unique_labels, per_class_f1))

        # Classification report
        metrics['classification_report'] = classification_report(
            true_labels, pred_labels,
            target_names=unique_labels,
            zero_division=0,
            output_dict=True
        )

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(true_labels, pred_labels, labels=unique_labels)

        # Print summary
        print(f"✅ {method_name} Metrics:")
        print(f"    Accuracy:        {metrics['accuracy']:.4f}")
        print(f"    F1 (Macro):      {metrics['f1_macro']:.4f}")
        print(f"    F1 (Weighted):   {metrics['f1_weighted']:.4f}")
        print(f"    Precision:       {metrics['precision_macro']:.4f}")
        print(f"    Recall:          {metrics['recall_macro']:.4f}")
        print(f"    Classes:         {metrics['num_classes']}")
        print(f"    Samples:         {metrics['num_samples']}")

        return metrics

    def _create_cluster_confusion_matrix(self, confusion_data: np.ndarray, gt_labels: List[str],
                                       clusters: List[int], title: str, save_path: Path):
        """Create cluster-ground truth confusion matrix visualization."""

        plt.figure(figsize=(14, 10))

        # Normalize by cluster (columns) to show proportions
        confusion_normalized = confusion_data / (confusion_data.sum(axis=0, keepdims=True) + 1e-8)

        # Create heatmap
        sns.heatmap(
            confusion_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=[f'Cluster {c}' for c in clusters],
            yticklabels=[label.replace('_', ' ') for label in gt_labels],
            cbar_kws={'label': 'Proportion'}
        )

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('SCAN Clusters')
        plt.ylabel('Ground Truth Activities')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Confusion matrix saved: {save_path}")
        plt.close()

    def _create_similarity_matrix(self, similarity_matrix: np.ndarray, clusters: List[int],
                                prototype_labels: List[str], title: str, save_path: Path):
        """Create cluster centroid to text prototype similarity matrix visualization."""

        plt.figure(figsize=(16, 10))

        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            xticklabels=[label.replace('_', ' ') for label in prototype_labels],
            yticklabels=[f'Cluster {c}' for c in clusters],
            cbar_kws={'label': 'Cosine Similarity'},
            center=0
        )

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Text Prototypes (Ground Truth Activities)')
        plt.ylabel('SCAN Clusters')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Similarity matrix saved: {save_path}")
        plt.close()

    def _create_voting_analysis_visualization(self, voting_details: Dict[int, Dict[str, int]],
                                            clusters: List[int], title: str, save_path: Path):
        """Create voting analysis visualization showing vote distribution for each cluster."""

        # Find all unique labels across all clusters
        all_labels = set()
        for cluster_votes in voting_details.values():
            all_labels.update(cluster_votes.keys())
        all_labels = sorted(list(all_labels))

        # Create a matrix: rows = clusters, cols = labels
        vote_matrix = np.zeros((len(clusters), len(all_labels)))

        for i, cluster_id in enumerate(clusters):
            cluster_votes = voting_details.get(cluster_id, {})
            for j, label in enumerate(all_labels):
                vote_matrix[i, j] = cluster_votes.get(label, 0)

        plt.figure(figsize=(16, 10))

        # Create heatmap
        sns.heatmap(
            vote_matrix,
            annot=True,
            fmt='g',
            cmap='YlOrRd',
            xticklabels=[label.replace('_', ' ') for label in all_labels],
            yticklabels=[f'Cluster {c}' for c in clusters],
            cbar_kws={'label': 'Vote Count'}
        )

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Prototype Labels')
        plt.ylabel('SCAN Clusters')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Voting analysis saved: {save_path}")
        plt.close()

    def _create_sample_prototype_similarity_matrix(self, similarity_matrix: np.ndarray,
                                                 prototype_labels: List[str], title: str, save_path: Path):
        """Create sample-prototype similarity matrix visualization."""

        plt.figure(figsize=(14, 8))

        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            cmap='RdYlBu_r',
            xticklabels=[label.replace('_', ' ') for label in prototype_labels],
            yticklabels=False,  # Too many samples to show individual labels
            cbar_kws={'label': 'Cosine Similarity'},
            center=0
        )

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Text Prototypes')
        plt.ylabel('Samples')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Sample similarity matrix saved: {save_path}")
        plt.close()

    def _create_caption_analysis_visualization(self, caption_stats: Dict[int, Dict[str, Any]],
                                             clusters: List[int], title: str, save_path: Path):
        """Create visualization showing caption collection statistics for each cluster."""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        cluster_ids = sorted(clusters)

        # Extract statistics
        total_samples = [caption_stats[cid]['total_samples'] for cid in cluster_ids]
        samples_with_captions = [caption_stats[cid]['samples_with_captions'] for cid in cluster_ids]
        unique_captions = [caption_stats[cid]['unique_captions'] for cid in cluster_ids]
        caption_coverage = [caption_stats[cid]['caption_coverage'] * 100 for cid in cluster_ids]

        # Plot 1: Total samples per cluster
        ax1.bar(range(len(cluster_ids)), total_samples, color='skyblue', alpha=0.7)
        ax1.set_title('Total Samples per Cluster', fontweight='bold')
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Samples')
        ax1.set_xticks(range(len(cluster_ids)))
        ax1.set_xticklabels([f'C{cid}' for cid in cluster_ids], rotation=45)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Samples with captions
        ax2.bar(range(len(cluster_ids)), samples_with_captions, color='lightgreen', alpha=0.7)
        ax2.set_title('Samples with Captions per Cluster', fontweight='bold')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Samples with Captions')
        ax2.set_xticks(range(len(cluster_ids)))
        ax2.set_xticklabels([f'C{cid}' for cid in cluster_ids], rotation=45)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Unique captions per cluster
        ax3.bar(range(len(cluster_ids)), unique_captions, color='orange', alpha=0.7)
        ax3.set_title('Unique Captions per Cluster', fontweight='bold')
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Unique Captions')
        ax3.set_xticks(range(len(cluster_ids)))
        ax3.set_xticklabels([f'C{cid}' for cid in cluster_ids], rotation=45)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Caption coverage percentage
        bars = ax4.bar(range(len(cluster_ids)), caption_coverage, color='coral', alpha=0.7)
        ax4.set_title('Caption Coverage per Cluster (%)', fontweight='bold')
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Coverage Percentage')
        ax4.set_xticks(range(len(cluster_ids)))
        ax4.set_xticklabels([f'C{cid}' for cid in cluster_ids], rotation=45)
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)

        # Add percentage labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Caption analysis saved: {save_path}")
        plt.close()

    def _create_caption_similarity_matrix(self, similarity_matrix: np.ndarray,
                                        cluster_ids: List[int], prototype_labels: List[str],
                                        title: str, save_path: Path):
        """Create caption cluster centroid to prototype similarity matrix visualization."""

        plt.figure(figsize=(16, 10))

        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            xticklabels=[label.replace('_', ' ') for label in prototype_labels],
            yticklabels=[f'Cluster {cid}' for cid in cluster_ids],
            cbar_kws={'label': 'Cosine Similarity'},
            center=0
        )

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Text Prototypes (Ground Truth Activities)')
        plt.ylabel('Caption-based Cluster Centroids')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Caption similarity matrix saved: {save_path}")
        plt.close()

    def create_comparison_visualizations(self, method1_results: Dict[str, Any],
                                       method2_results: Dict[str, Any],
                                       method3_results: Dict[str, Any] = None,
                                       method4_results: Dict[str, Any] = None):
        """Create comparison visualizations between the methods."""
        print("\nCreating comparison visualizations...")

        # 1. F1 scores comparison
        if method4_results is not None:
            self._create_f1_comparison_chart_four_methods(
                method1_results['metrics'],
                method2_results['metrics'],
                method3_results['metrics'],
                method4_results['metrics']
            )
        elif method3_results is not None:
            self._create_f1_comparison_chart_three_methods(
                method1_results['metrics'],
                method2_results['metrics'],
                method3_results['metrics']
            )
        else:
            self._create_f1_comparison_chart(method1_results['metrics'], method2_results['metrics'])

        # 2. Per-class F1 comparison
        if method4_results is not None:
            self._create_per_class_f1_comparison_four_methods(
                method1_results['metrics'],
                method2_results['metrics'],
                method3_results['metrics'],
                method4_results['metrics']
            )
        elif method3_results is not None:
            self._create_per_class_f1_comparison_three_methods(
                method1_results['metrics'],
                method2_results['metrics'],
                method3_results['metrics']
            )
        else:
            self._create_per_class_f1_comparison(method1_results['metrics'], method2_results['metrics'])

        # 3. Confusion matrices comparison
        if method4_results is not None:
            self._create_confusion_matrices_comparison_four_methods(
                method1_results['metrics'],
                method2_results['metrics'],
                method3_results['metrics'],
                method4_results['metrics']
            )
        elif method3_results is not None:
            self._create_confusion_matrices_comparison_three_methods(
                method1_results['metrics'],
                method2_results['metrics'],
                method3_results['metrics']
            )
        else:
            self._create_confusion_matrices_comparison(method1_results['metrics'], method2_results['metrics'])

    def _create_f1_comparison_chart(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any]):
        """Create F1 scores comparison chart."""

        fig, ax = plt.subplots(figsize=(12, 8))

        # Metrics to compare
        metric_names = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
        metric_labels = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'Precision', 'Recall']

        method1_scores = [metrics1.get(metric, 0) for metric in metric_names]
        method2_scores = [metrics2.get(metric, 0) for metric in metric_names]

        x = np.arange(len(metric_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, method1_scores, width, label='Method 1 (Argmax)',
                      color='#1976D2', alpha=0.8)
        bars2 = ax.bar(x + width/2, method2_scores, width, label='Method 2 (Centroid)',
                      color='#FF6F00', alpha=0.8)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('SCAN Evaluation Methods Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        save_path = self.output_dir / 'methods_comparison_f1_scores.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 F1 comparison chart saved: {save_path}")
        plt.close()

    def _create_f1_comparison_chart_three_methods(self, metrics1: Dict[str, Any],
                                                metrics2: Dict[str, Any],
                                                metrics3: Dict[str, Any]):
        """Create F1 scores comparison chart for three methods."""

        fig, ax = plt.subplots(figsize=(14, 8))

        # Metrics to compare
        metric_names = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
        metric_labels = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'Precision', 'Recall']

        method1_scores = [metrics1.get(metric, 0) for metric in metric_names]
        method2_scores = [metrics2.get(metric, 0) for metric in metric_names]
        method3_scores = [metrics3.get(metric, 0) for metric in metric_names]

        x = np.arange(len(metric_names))
        width = 0.25

        bars1 = ax.bar(x - width, method1_scores, width, label='Method 1 (Argmax)',
                      color='#1976D2', alpha=0.8)
        bars2 = ax.bar(x, method2_scores, width, label='Method 2 (Centroid)',
                      color='#FF6F00', alpha=0.8)
        bars3 = ax.bar(x + width, method3_scores, width, label='Method 3 (Prototype Argmax)',
                      color='#4CAF50', alpha=0.8)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('SCAN Evaluation Methods Comparison (3 Methods)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        save_path = self.output_dir / 'methods_comparison_f1_scores_three_methods.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 F1 comparison chart (3 methods) saved: {save_path}")
        plt.close()

    def _create_f1_comparison_chart_four_methods(self, metrics1: Dict[str, Any],
                                               metrics2: Dict[str, Any],
                                               metrics3: Dict[str, Any],
                                               metrics4: Dict[str, Any]):
        """Create F1 scores comparison chart for four methods."""

        fig, ax = plt.subplots(figsize=(16, 8))

        # Metrics to compare
        metric_names = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
        metric_labels = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'Precision', 'Recall']

        method1_scores = [metrics1.get(metric, 0) for metric in metric_names]
        method2_scores = [metrics2.get(metric, 0) for metric in metric_names]
        method3_scores = [metrics3.get(metric, 0) for metric in metric_names]
        method4_scores = [metrics4.get(metric, 0) for metric in metric_names]

        x = np.arange(len(metric_names))
        width = 0.2

        bars1 = ax.bar(x - 1.5*width, method1_scores, width, label='Method 1 (Argmax)',
                      color='#1976D2', alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, method2_scores, width, label='Method 2 (Centroid)',
                      color='#FF6F00', alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, method3_scores, width, label='Method 3 (Prototype Argmax)',
                      color='#4CAF50', alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, method4_scores, width, label='Method 4 (Caption Clusters)',
                      color='#9C27B0', alpha=0.8)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('SCAN Evaluation Methods Comparison (4 Methods)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)

        plt.tight_layout()

        save_path = self.output_dir / 'methods_comparison_f1_scores_four_methods.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 F1 comparison chart (4 methods) saved: {save_path}")
        plt.close()

    def _create_per_class_f1_comparison(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any]):
        """Create per-class F1 scores comparison."""

        # Get all unique labels from both methods
        all_labels = set()
        if 'per_class_f1' in metrics1:
            all_labels.update(metrics1['per_class_f1'].keys())
        if 'per_class_f1' in metrics2:
            all_labels.update(metrics2['per_class_f1'].keys())

        all_labels = sorted(list(all_labels))

        if not all_labels:
            print("No per-class F1 data available for comparison")
            return

        # Limit to top 15 for readability
        if len(all_labels) > 15:
            all_labels = all_labels[:15]

        method1_f1 = [metrics1.get('per_class_f1', {}).get(label, 0) for label in all_labels]
        method2_f1 = [metrics2.get('per_class_f1', {}).get(label, 0) for label in all_labels]

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(all_labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, method1_f1, width, label='Method 1 (Argmax)',
                      color='#1976D2', alpha=0.8)
        bars2 = ax.bar(x + width/2, method2_f1, width, label='Method 2 (Centroid)',
                      color='#FF6F00', alpha=0.8)

        ax.set_xlabel('Activity Labels')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Scores Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([label.replace('_', ' ') for label in all_labels], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        save_path = self.output_dir / 'per_class_f1_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Per-class F1 comparison saved: {save_path}")
        plt.close()

    def _create_per_class_f1_comparison_three_methods(self, metrics1: Dict[str, Any],
                                                    metrics2: Dict[str, Any],
                                                    metrics3: Dict[str, Any]):
        """Create per-class F1 scores comparison for three methods."""

        # Get all unique labels from all methods
        all_labels = set()
        for metrics in [metrics1, metrics2, metrics3]:
            if 'per_class_f1' in metrics:
                all_labels.update(metrics['per_class_f1'].keys())

        all_labels = sorted(list(all_labels))

        if not all_labels:
            print("No per-class F1 data available for comparison")
            return

        # Limit to top 12 for readability
        if len(all_labels) > 12:
            all_labels = all_labels[:12]

        method1_f1 = [metrics1.get('per_class_f1', {}).get(label, 0) for label in all_labels]
        method2_f1 = [metrics2.get('per_class_f1', {}).get(label, 0) for label in all_labels]
        method3_f1 = [metrics3.get('per_class_f1', {}).get(label, 0) for label in all_labels]

        fig, ax = plt.subplots(figsize=(16, 8))

        x = np.arange(len(all_labels))
        width = 0.25

        bars1 = ax.bar(x - width, method1_f1, width, label='Method 1 (Argmax)',
                      color='#1976D2', alpha=0.8)
        bars2 = ax.bar(x, method2_f1, width, label='Method 2 (Centroid)',
                      color='#FF6F00', alpha=0.8)
        bars3 = ax.bar(x + width, method3_f1, width, label='Method 3 (Prototype Argmax)',
                      color='#4CAF50', alpha=0.8)

        ax.set_xlabel('Activity Labels')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Scores Comparison (3 Methods)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([label.replace('_', ' ') for label in all_labels], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        save_path = self.output_dir / 'per_class_f1_comparison_three_methods.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Per-class F1 comparison (3 methods) saved: {save_path}")
        plt.close()

    def _create_per_class_f1_comparison_four_methods(self, metrics1: Dict[str, Any],
                                                   metrics2: Dict[str, Any],
                                                   metrics3: Dict[str, Any],
                                                   metrics4: Dict[str, Any]):
        """Create per-class F1 scores comparison for four methods."""

        # Get all unique labels from all methods
        all_labels = set()
        for metrics in [metrics1, metrics2, metrics3, metrics4]:
            if 'per_class_f1' in metrics:
                all_labels.update(metrics['per_class_f1'].keys())

        all_labels = sorted(list(all_labels))

        if not all_labels:
            print("No per-class F1 data available for comparison")
            return

        # Limit to top 10 for readability
        if len(all_labels) > 10:
            all_labels = all_labels[:10]

        method1_f1 = [metrics1.get('per_class_f1', {}).get(label, 0) for label in all_labels]
        method2_f1 = [metrics2.get('per_class_f1', {}).get(label, 0) for label in all_labels]
        method3_f1 = [metrics3.get('per_class_f1', {}).get(label, 0) for label in all_labels]
        method4_f1 = [metrics4.get('per_class_f1', {}).get(label, 0) for label in all_labels]

        fig, ax = plt.subplots(figsize=(18, 8))

        x = np.arange(len(all_labels))
        width = 0.2

        bars1 = ax.bar(x - 1.5*width, method1_f1, width, label='Method 1 (Argmax)',
                      color='#1976D2', alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, method2_f1, width, label='Method 2 (Centroid)',
                      color='#FF6F00', alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, method3_f1, width, label='Method 3 (Prototype Argmax)',
                      color='#4CAF50', alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, method4_f1, width, label='Method 4 (Caption Clusters)',
                      color='#9C27B0', alpha=0.8)

        ax.set_xlabel('Activity Labels')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Scores Comparison (4 Methods)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([label.replace('_', ' ') for label in all_labels], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        save_path = self.output_dir / 'per_class_f1_comparison_four_methods.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Per-class F1 comparison (4 methods) saved: {save_path}")
        plt.close()

    def _create_confusion_matrices_comparison(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any]):
        """Create side-by-side confusion matrices comparison."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')

        # Method 1 confusion matrix
        if 'confusion_matrix' in metrics1 and 'unique_labels' in metrics1:
            cm1 = metrics1['confusion_matrix']
            labels1 = metrics1['unique_labels']

            # Normalize confusion matrix
            cm1_normalized = cm1.astype('float') / (cm1.sum(axis=1)[:, np.newaxis] + 1e-8)

            sns.heatmap(
                cm1_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=[label.replace('_', ' ')[:10] for label in labels1],
                yticklabels=[label.replace('_', ' ')[:10] for label in labels1],
                ax=ax1,
                cbar_kws={'label': 'Normalized Count'}
            )

            ax1.set_title('Method 1 (Argmax Assignment)', fontweight='bold')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            ax1.tick_params(axis='x', rotation=45)

        # Method 2 confusion matrix
        if 'confusion_matrix' in metrics2 and 'unique_labels' in metrics2:
            cm2 = metrics2['confusion_matrix']
            labels2 = metrics2['unique_labels']

            # Normalize confusion matrix
            cm2_normalized = cm2.astype('float') / (cm2.sum(axis=1)[:, np.newaxis] + 1e-8)

            sns.heatmap(
                cm2_normalized,
                annot=True,
                fmt='.2f',
                cmap='Oranges',
                xticklabels=[label.replace('_', ' ')[:10] for label in labels2],
                yticklabels=[label.replace('_', ' ')[:10] for label in labels2],
                ax=ax2,
                cbar_kws={'label': 'Normalized Count'}
            )

            ax2.set_title('Method 2 (Centroid Matching)', fontweight='bold')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        save_path = self.output_dir / 'confusion_matrices_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Confusion matrices comparison saved: {save_path}")
        plt.close()

    def _create_confusion_matrices_comparison_three_methods(self, metrics1: Dict[str, Any],
                                                          metrics2: Dict[str, Any],
                                                          metrics3: Dict[str, Any]):
        """Create side-by-side confusion matrices comparison for three methods."""

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle('Confusion Matrices Comparison (3 Methods)', fontsize=16, fontweight='bold')

        # Method 1 confusion matrix
        if 'confusion_matrix' in metrics1 and 'unique_labels' in metrics1:
            cm1 = metrics1['confusion_matrix']
            labels1 = metrics1['unique_labels']

            # Normalize confusion matrix
            cm1_normalized = cm1.astype('float') / (cm1.sum(axis=1)[:, np.newaxis] + 1e-8)

            sns.heatmap(
                cm1_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=[label.replace('_', ' ')[:8] for label in labels1],
                yticklabels=[label.replace('_', ' ')[:8] for label in labels1],
                ax=ax1,
                cbar_kws={'label': 'Normalized Count'}
            )

            ax1.set_title('Method 1 (Argmax)', fontweight='bold')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            ax1.tick_params(axis='x', rotation=45, labelsize=8)
            ax1.tick_params(axis='y', labelsize=8)

        # Method 2 confusion matrix
        if 'confusion_matrix' in metrics2 and 'unique_labels' in metrics2:
            cm2 = metrics2['confusion_matrix']
            labels2 = metrics2['unique_labels']

            # Normalize confusion matrix
            cm2_normalized = cm2.astype('float') / (cm2.sum(axis=1)[:, np.newaxis] + 1e-8)

            sns.heatmap(
                cm2_normalized,
                annot=True,
                fmt='.2f',
                cmap='Oranges',
                xticklabels=[label.replace('_', ' ')[:8] for label in labels2],
                yticklabels=[label.replace('_', ' ')[:8] for label in labels2],
                ax=ax2,
                cbar_kws={'label': 'Normalized Count'}
            )

            ax2.set_title('Method 2 (Centroid)', fontweight='bold')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            ax2.tick_params(axis='x', rotation=45, labelsize=8)
            ax2.tick_params(axis='y', labelsize=8)

        # Method 3 confusion matrix
        if 'confusion_matrix' in metrics3 and 'unique_labels' in metrics3:
            cm3 = metrics3['confusion_matrix']
            labels3 = metrics3['unique_labels']

            # Normalize confusion matrix
            cm3_normalized = cm3.astype('float') / (cm3.sum(axis=1)[:, np.newaxis] + 1e-8)

            sns.heatmap(
                cm3_normalized,
                annot=True,
                fmt='.2f',
                cmap='Greens',
                xticklabels=[label.replace('_', ' ')[:8] for label in labels3],
                yticklabels=[label.replace('_', ' ')[:8] for label in labels3],
                ax=ax3,
                cbar_kws={'label': 'Normalized Count'}
            )

            ax3.set_title('Method 3 (Prototype Argmax)', fontweight='bold')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
            ax3.tick_params(axis='x', rotation=45, labelsize=8)
            ax3.tick_params(axis='y', labelsize=8)

        plt.tight_layout()

        save_path = self.output_dir / 'confusion_matrices_comparison_three_methods.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Confusion matrices comparison (3 methods) saved: {save_path}")
        plt.close()

    def _create_confusion_matrices_comparison_four_methods(self, metrics1: Dict[str, Any],
                                                         metrics2: Dict[str, Any],
                                                         metrics3: Dict[str, Any],
                                                         metrics4: Dict[str, Any]):
        """Create side-by-side confusion matrices comparison for four methods."""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Confusion Matrices Comparison (4 Methods)', fontsize=16, fontweight='bold')

        # Method 1 confusion matrix (Top Left)
        if 'confusion_matrix' in metrics1 and 'unique_labels' in metrics1:
            cm1 = metrics1['confusion_matrix']
            labels1 = metrics1['unique_labels']
            cm1_normalized = cm1.astype('float') / (cm1.sum(axis=1)[:, np.newaxis] + 1e-8)

            sns.heatmap(
                cm1_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=[label.replace('_', ' ')[:8] for label in labels1],
                yticklabels=[label.replace('_', ' ')[:8] for label in labels1],
                ax=ax1,
                cbar_kws={'label': 'Normalized Count'}
            )
            ax1.set_title('Method 1 (Argmax)', fontweight='bold')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            ax1.tick_params(axis='x', rotation=45, labelsize=8)
            ax1.tick_params(axis='y', labelsize=8)

        # Method 2 confusion matrix (Top Right)
        if 'confusion_matrix' in metrics2 and 'unique_labels' in metrics2:
            cm2 = metrics2['confusion_matrix']
            labels2 = metrics2['unique_labels']
            cm2_normalized = cm2.astype('float') / (cm2.sum(axis=1)[:, np.newaxis] + 1e-8)

            sns.heatmap(
                cm2_normalized,
                annot=True,
                fmt='.2f',
                cmap='Oranges',
                xticklabels=[label.replace('_', ' ')[:8] for label in labels2],
                yticklabels=[label.replace('_', ' ')[:8] for label in labels2],
                ax=ax2,
                cbar_kws={'label': 'Normalized Count'}
            )
            ax2.set_title('Method 2 (Centroid)', fontweight='bold')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            ax2.tick_params(axis='x', rotation=45, labelsize=8)
            ax2.tick_params(axis='y', labelsize=8)

        # Method 3 confusion matrix (Bottom Left)
        if 'confusion_matrix' in metrics3 and 'unique_labels' in metrics3:
            cm3 = metrics3['confusion_matrix']
            labels3 = metrics3['unique_labels']
            cm3_normalized = cm3.astype('float') / (cm3.sum(axis=1)[:, np.newaxis] + 1e-8)

            sns.heatmap(
                cm3_normalized,
                annot=True,
                fmt='.2f',
                cmap='Greens',
                xticklabels=[label.replace('_', ' ')[:8] for label in labels3],
                yticklabels=[label.replace('_', ' ')[:8] for label in labels3],
                ax=ax3,
                cbar_kws={'label': 'Normalized Count'}
            )
            ax3.set_title('Method 3 (Prototype Argmax)', fontweight='bold')
            ax3.set_xlabel('Predicted')
            ax3.set_ylabel('Actual')
            ax3.tick_params(axis='x', rotation=45, labelsize=8)
            ax3.tick_params(axis='y', labelsize=8)

        # Method 4 confusion matrix (Bottom Right)
        if 'confusion_matrix' in metrics4 and 'unique_labels' in metrics4:
            cm4 = metrics4['confusion_matrix']
            labels4 = metrics4['unique_labels']
            cm4_normalized = cm4.astype('float') / (cm4.sum(axis=1)[:, np.newaxis] + 1e-8)

            sns.heatmap(
                cm4_normalized,
                annot=True,
                fmt='.2f',
                cmap='Purples',
                xticklabels=[label.replace('_', ' ')[:8] for label in labels4],
                yticklabels=[label.replace('_', ' ')[:8] for label in labels4],
                ax=ax4,
                cbar_kws={'label': 'Normalized Count'}
            )
            ax4.set_title('Method 4 (Caption Clusters)', fontweight='bold')
            ax4.set_xlabel('Predicted')
            ax4.set_ylabel('Actual')
            ax4.tick_params(axis='x', rotation=45, labelsize=8)
            ax4.tick_params(axis='y', labelsize=8)

        plt.tight_layout()

        save_path = self.output_dir / 'confusion_matrices_comparison_four_methods.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Confusion matrices comparison (4 methods) saved: {save_path}")
        plt.close()

    def create_text_summary_report(self, method1_results: Dict[str, Any],
                                 method2_results: Dict[str, Any],
                                 method3_results: Dict[str, Any] = None,
                                 method4_results: Dict[str, Any] = None,
                                 max_samples: int = 10000):
        """Create comprehensive text summary report."""
        print("Creating text summary report...")

        report_path = self.output_dir / f'scan_evaluation_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SCAN CLUSTERING EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"SCAN Model: {self.scan_model_path}\n")
            f.write(f"Baseline Model: {self.baseline_model_path}\n")
            f.write(f"Test Data: {self.data_path}\n")
            f.write(f"Max Samples: {max_samples}\n")
            f.write(f"Unique Ground Truth Labels: {len(np.unique(self.ground_truth_labels))}\n")
            f.write(f"Unique Predicted Clusters: {len(np.unique(self.cluster_predictions))}\n\n")

            # Summary comparison table
            if method4_results is not None:
                f.write("SUMMARY COMPARISON (4 METHODS)\n")
                f.write("-" * 130 + "\n")
                f.write(f"{'Metric':<20} {'Method 1 (Argmax)':<20} {'Method 2 (Centroid)':<20} {'Method 3 (Proto-Argmax)':<20} {'Method 4 (Caption)':<20} {'Best Method':<15}\n")
                f.write("-" * 130 + "\n")

                metrics1 = method1_results['metrics']
                metrics2 = method2_results['metrics']
                metrics3 = method3_results['metrics']
                metrics4 = method4_results['metrics']

                comparison_metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
                for metric in comparison_metrics:
                    val1 = metrics1.get(metric, 0)
                    val2 = metrics2.get(metric, 0)
                    val3 = metrics3.get(metric, 0)
                    val4 = metrics4.get(metric, 0)

                    # Find best method
                    values = [val1, val2, val3, val4]
                    best_idx = np.argmax(values)
                    best_method = ['Method 1', 'Method 2', 'Method 3', 'Method 4'][best_idx]

                    f.write(f"{metric:<20} {val1:<20.4f} {val2:<20.4f} {val3:<20.4f} {val4:<20.4f} {best_method:<15}\n")
            elif method3_results is not None:
                f.write("SUMMARY COMPARISON (3 METHODS)\n")
                f.write("-" * 110 + "\n")
                f.write(f"{'Metric':<20} {'Method 1 (Argmax)':<20} {'Method 2 (Centroid)':<20} {'Method 3 (Proto-Argmax)':<20} {'Best Method':<15}\n")
                f.write("-" * 110 + "\n")

                metrics1 = method1_results['metrics']
                metrics2 = method2_results['metrics']
                metrics3 = method3_results['metrics']

                comparison_metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
                for metric in comparison_metrics:
                    val1 = metrics1.get(metric, 0)
                    val2 = metrics2.get(metric, 0)
                    val3 = metrics3.get(metric, 0)

                    # Find best method
                    values = [val1, val2, val3]
                    best_idx = np.argmax(values)
                    best_method = ['Method 1', 'Method 2', 'Method 3'][best_idx]

                    f.write(f"{metric:<20} {val1:<20.4f} {val2:<20.4f} {val3:<20.4f} {best_method:<15}\n")
            else:
                f.write("SUMMARY COMPARISON\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Metric':<20} {'Method 1 (Argmax)':<20} {'Method 2 (Centroid)':<20} {'Difference':<15}\n")
                f.write("-" * 80 + "\n")

                metrics1 = method1_results['metrics']
                metrics2 = method2_results['metrics']

                comparison_metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
                for metric in comparison_metrics:
                    val1 = metrics1.get(metric, 0)
                    val2 = metrics2.get(metric, 0)
                    diff = val2 - val1
                    f.write(f"{metric:<20} {val1:<20.4f} {val2:<20.4f} {diff:<+15.4f}\n")

            f.write("\n\n")

            # Method 1 detailed results
            f.write("METHOD 1: CONFUSION MATRIX ARGMAX ASSIGNMENT\n")
            f.write("="*60 + "\n")
            f.write("Description: For each cluster, assign the most frequent ground truth label\n\n")

            f.write("Cluster Assignments:\n")
            for cluster_id, assigned_label in method1_results['cluster_assignments'].items():
                f.write(f"  Cluster {cluster_id} -> {assigned_label}\n")
            f.write("\n")

            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy:        {metrics1.get('accuracy', 0):.4f}\n")
            f.write(f"  F1 (Macro):      {metrics1.get('f1_macro', 0):.4f}\n")
            f.write(f"  F1 (Weighted):   {metrics1.get('f1_weighted', 0):.4f}\n")
            f.write(f"  Precision:       {metrics1.get('precision_macro', 0):.4f}\n")
            f.write(f"  Recall:          {metrics1.get('recall_macro', 0):.4f}\n")
            f.write(f"  Classes:         {metrics1.get('num_classes', 0)}\n")
            f.write(f"  Samples:         {metrics1.get('num_samples', 0)}\n\n")

            # Top performing classes for Method 1
            if 'per_class_f1' in metrics1:
                f.write("Top Performing Classes (F1-Score):\n")
                sorted_f1_1 = sorted(metrics1['per_class_f1'].items(), key=lambda x: x[1], reverse=True)[:10]
                for class_name, f1_score in sorted_f1_1:
                    f.write(f"  {class_name:<25} {f1_score:.4f}\n")
                f.write("\n")

            # Method 2 detailed results
            f.write("METHOD 2: CLUSTER CENTROID TO TEXT PROTOTYPE MATCHING\n")
            f.write("="*60 + "\n")
            f.write("Description: Match cluster centroids to text prototypes using cosine similarity\n\n")

            f.write("Cluster Assignments:\n")
            for cluster_id, assigned_label in method2_results['cluster_assignments'].items():
                f.write(f"  Cluster {cluster_id} -> {assigned_label}\n")
            f.write("\n")

            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy:        {metrics2.get('accuracy', 0):.4f}\n")
            f.write(f"  F1 (Macro):      {metrics2.get('f1_macro', 0):.4f}\n")
            f.write(f"  F1 (Weighted):   {metrics2.get('f1_weighted', 0):.4f}\n")
            f.write(f"  Precision:       {metrics2.get('precision_macro', 0):.4f}\n")
            f.write(f"  Recall:          {metrics2.get('recall_macro', 0):.4f}\n")
            f.write(f"  Classes:         {metrics2.get('num_classes', 0)}\n")
            f.write(f"  Samples:         {metrics2.get('num_samples', 0)}\n\n")

            # Top performing classes for Method 2
            if 'per_class_f1' in metrics2:
                f.write("Top Performing Classes (F1-Score):\n")
                sorted_f1_2 = sorted(metrics2['per_class_f1'].items(), key=lambda x: x[1], reverse=True)[:10]
                for class_name, f1_score in sorted_f1_2:
                    f.write(f"  {class_name:<25} {f1_score:.4f}\n")
                f.write("\n")

            # Method 3 detailed results (if available)
            if method3_results is not None:
                f.write("METHOD 3: PROTOTYPE ARGMAX WITH MAJORITY VOTING\n")
                f.write("="*60 + "\n")
                f.write("Description: Find closest prototype for each sample, use majority voting per cluster\n\n")

                f.write("Cluster Assignments:\n")
                for cluster_id, assigned_label in method3_results['cluster_assignments'].items():
                    f.write(f"  Cluster {cluster_id} -> {assigned_label}\n")
                f.write("\n")

                metrics3 = method3_results['metrics']
                f.write("Performance Metrics:\n")
                f.write(f"  Accuracy:        {metrics3.get('accuracy', 0):.4f}\n")
                f.write(f"  F1 (Macro):      {metrics3.get('f1_macro', 0):.4f}\n")
                f.write(f"  F1 (Weighted):   {metrics3.get('f1_weighted', 0):.4f}\n")
                f.write(f"  Precision:       {metrics3.get('precision_macro', 0):.4f}\n")
                f.write(f"  Recall:          {metrics3.get('recall_macro', 0):.4f}\n")
                f.write(f"  Classes:         {metrics3.get('num_classes', 0)}\n")
                f.write(f"  Samples:         {metrics3.get('num_samples', 0)}\n\n")

                # Top performing classes for Method 3
                if 'per_class_f1' in metrics3:
                    f.write("Top Performing Classes (F1-Score):\n")
                    sorted_f1_3 = sorted(metrics3['per_class_f1'].items(), key=lambda x: x[1], reverse=True)[:10]
                    for class_name, f1_score in sorted_f1_3:
                        f.write(f"  {class_name:<25} {f1_score:.4f}\n")
                    f.write("\n")

                # Voting analysis
                if 'voting_details' in method3_results:
                    f.write("Voting Analysis (Sample clusters with vote breakdown):\n")
                    for cluster_id, votes in list(method3_results['voting_details'].items())[:5]:  # Show first 5
                        total_votes = sum(votes.values())
                        f.write(f"  Cluster {cluster_id} ({total_votes} votes):\n")
                        for label, count in sorted(votes.items(), key=lambda x: x[1], reverse=True)[:3]:
                            percentage = count / total_votes * 100
                            f.write(f"    {label}: {count} ({percentage:.1f}%)\n")
                    f.write("\n")

            # Method 4 detailed results (if available)
            if method4_results is not None:
                f.write("METHOD 4: CAPTION CLUSTERS (TEXT-SPACE)\n")
                f.write("="*60 + "\n")
                f.write("Description: Collect captions from cluster members, encode them, compute spherical mean centroids\n\n")

                f.write("Cluster Assignments:\n")
                for cluster_id, assigned_label in method4_results['cluster_assignments'].items():
                    f.write(f"  Cluster {cluster_id} -> {assigned_label}\n")
                f.write("\n")

                metrics4 = method4_results['metrics']
                f.write("Performance Metrics:\n")
                f.write(f"  Accuracy:        {metrics4.get('accuracy', 0):.4f}\n")
                f.write(f"  F1 (Macro):      {metrics4.get('f1_macro', 0):.4f}\n")
                f.write(f"  F1 (Weighted):   {metrics4.get('f1_weighted', 0):.4f}\n")
                f.write(f"  Precision:       {metrics4.get('precision_macro', 0):.4f}\n")
                f.write(f"  Recall:          {metrics4.get('recall_macro', 0):.4f}\n")
                f.write(f"  Classes:         {metrics4.get('num_classes', 0)}\n")
                f.write(f"  Samples:         {metrics4.get('num_samples', 0)}\n\n")

                # Top performing classes for Method 4
                if 'per_class_f1' in metrics4:
                    f.write("Top Performing Classes (F1-Score):\n")
                    sorted_f1_4 = sorted(metrics4['per_class_f1'].items(), key=lambda x: x[1], reverse=True)[:10]
                    for class_name, f1_score in sorted_f1_4:
                        f.write(f"  {class_name:<25} {f1_score:.4f}\n")
                    f.write("\n")

                # Caption statistics
                if 'caption_stats' in method4_results:
                    f.write("Caption Collection Statistics (Sample clusters):\n")
                    caption_stats = method4_results['caption_stats']
                    for cluster_id, stats in list(caption_stats.items())[:5]:  # Show first 5
                        f.write(f"  Cluster {cluster_id}:\n")
                        f.write(f"    Total samples: {stats.get('total_samples', 0)}\n")
                        f.write(f"    Samples with captions: {stats.get('samples_with_captions', 0)}\n")
                        f.write(f"    Unique captions: {stats.get('unique_captions', 0)}\n")
                        f.write(f"    Coverage: {stats.get('caption_coverage', 0)*100:.1f}%\n")
                    f.write("\n")

            # Key insights
            f.write("KEY INSIGHTS\n")
            f.write("="*60 + "\n")

            # Find better performing method
            if method4_results is not None:
                f1_scores = {
                    'Method 1 (Argmax Assignment)': metrics1.get('f1_macro', 0),
                    'Method 2 (Centroid Matching)': metrics2.get('f1_macro', 0),
                    'Method 3 (Prototype Argmax)': metrics3.get('f1_macro', 0),
                    'Method 4 (Caption Clusters)': metrics4.get('f1_macro', 0)
                }
                best_method = max(f1_scores.items(), key=lambda x: x[1])
                f.write(f"Best performing method: {best_method[0]} (F1-Macro: {best_method[1]:.4f})\n")

                # Compare all methods
                f.write("\nMethod Comparison:\n")
                for method, score in sorted(f1_scores.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {method}: {score:.4f}\n")
                f.write("\n")
            elif method3_results is not None:
                f1_scores = {
                    'Method 1 (Argmax Assignment)': metrics1.get('f1_macro', 0),
                    'Method 2 (Centroid Matching)': metrics2.get('f1_macro', 0),
                    'Method 3 (Prototype Argmax)': metrics3.get('f1_macro', 0)
                }
                best_method = max(f1_scores.items(), key=lambda x: x[1])
                f.write(f"Best performing method: {best_method[0]} (F1-Macro: {best_method[1]:.4f})\n")

                # Compare all methods
                f.write("\nMethod Comparison:\n")
                for method, score in sorted(f1_scores.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {method}: {score:.4f}\n")
                f.write("\n")
            else:
                if metrics2.get('f1_macro', 0) > metrics1.get('f1_macro', 0):
                    better_method = "Method 2 (Centroid Matching)"
                    improvement = metrics2.get('f1_macro', 0) - metrics1.get('f1_macro', 0)
                else:
                    better_method = "Method 1 (Argmax Assignment)"
                    improvement = metrics1.get('f1_macro', 0) - metrics2.get('f1_macro', 0)

                f.write(f"Best performing method: {better_method}\n")
                f.write(f"F1-Macro improvement: {improvement:+.4f}\n\n")

            # Analysis of cluster quality
            f.write("Cluster Quality Analysis:\n")
            f.write(f"  Total clusters: {len(np.unique(self.cluster_predictions))}\n")
            f.write(f"  Ground truth classes: {len(np.unique(self.ground_truth_labels))}\n")

            # Cluster size distribution
            cluster_sizes = Counter(self.cluster_predictions)
            f.write(f"  Cluster size statistics:\n")
            f.write(f"    Mean size: {np.mean(list(cluster_sizes.values())):.1f}\n")
            f.write(f"    Std size: {np.std(list(cluster_sizes.values())):.1f}\n")
            f.write(f"    Min size: {min(cluster_sizes.values())}\n")
            f.write(f"    Max size: {max(cluster_sizes.values())}\n\n")

        print(f"💾 Text summary report saved: {report_path}")

    def run_full_evaluation(self, max_samples: int = 10000):
        """Run complete SCAN evaluation pipeline."""
        print("Starting SCAN clustering evaluation...")
        print("="*60)

        # Load models and data
        self.load_models_and_data()

        # Extract embeddings and predictions
        self.extract_embeddings_and_predictions(max_samples=max_samples)

        # Run Method 1: Argmax assignment
        method1_results = self.evaluate_method1_argmax_assignment()

        # Run Method 2: Centroid matching
        method2_results = self.evaluate_method2_centroid_matching()

        # Run Method 3: Prototype argmax with majority voting
        method3_results = self.evaluate_method3_prototype_argmax()

        # Run Method 4: Caption clusters
        method4_results = self.evaluate_method4_caption_clusters()

        # Create comparison visualizations
        self.create_comparison_visualizations(method1_results, method2_results, method3_results, method4_results)

        # Create text summary report
        self.create_text_summary_report(method1_results, method2_results, method3_results, method4_results, max_samples)

        # Save detailed results (only essential metrics, avoiding complex numpy objects)
        detailed_results = {
            'config': {
                'scan_model_path': self.scan_model_path,
                'baseline_model_path': self.baseline_model_path,
                'data_path': self.data_path,
                'vocab_path': self.vocab_path,
                'max_samples': max_samples
            },
            'method1_results': {
                'method': method1_results['method'],
                'cluster_assignments': {str(k): v for k, v in method1_results['cluster_assignments'].items()},
                'metrics': {
                    'accuracy': float(method1_results['metrics'].get('accuracy', 0)),
                    'f1_macro': float(method1_results['metrics'].get('f1_macro', 0)),
                    'f1_weighted': float(method1_results['metrics'].get('f1_weighted', 0)),
                    'precision_macro': float(method1_results['metrics'].get('precision_macro', 0)),
                    'recall_macro': float(method1_results['metrics'].get('recall_macro', 0)),
                    'num_classes': int(method1_results['metrics'].get('num_classes', 0)),
                    'num_samples': int(method1_results['metrics'].get('num_samples', 0)),
                    'per_class_f1': {k: float(v) for k, v in method1_results['metrics'].get('per_class_f1', {}).items()}
                }
            },
            'method2_results': {
                'method': method2_results['method'],
                'cluster_assignments': {str(k): v for k, v in method2_results['cluster_assignments'].items()},
                'metrics': {
                    'accuracy': float(method2_results['metrics'].get('accuracy', 0)),
                    'f1_macro': float(method2_results['metrics'].get('f1_macro', 0)),
                    'f1_weighted': float(method2_results['metrics'].get('f1_weighted', 0)),
                    'precision_macro': float(method2_results['metrics'].get('precision_macro', 0)),
                    'recall_macro': float(method2_results['metrics'].get('recall_macro', 0)),
                    'num_classes': int(method2_results['metrics'].get('num_classes', 0)),
                    'num_samples': int(method2_results['metrics'].get('num_samples', 0)),
                    'per_class_f1': {k: float(v) for k, v in method2_results['metrics'].get('per_class_f1', {}).items()}
                }
            },
            'method3_results': {
                'method': method3_results['method'],
                'cluster_assignments': {str(k): v for k, v in method3_results['cluster_assignments'].items()},
                'metrics': {
                    'accuracy': float(method3_results['metrics'].get('accuracy', 0)),
                    'f1_macro': float(method3_results['metrics'].get('f1_macro', 0)),
                    'f1_weighted': float(method3_results['metrics'].get('f1_weighted', 0)),
                    'precision_macro': float(method3_results['metrics'].get('precision_macro', 0)),
                    'recall_macro': float(method3_results['metrics'].get('recall_macro', 0)),
                    'num_classes': int(method3_results['metrics'].get('num_classes', 0)),
                    'num_samples': int(method3_results['metrics'].get('num_samples', 0)),
                    'per_class_f1': {k: float(v) for k, v in method3_results['metrics'].get('per_class_f1', {}).items()}
                },
                'voting_details': {str(k): v for k, v in method3_results.get('voting_details', {}).items()}
            },
            'method4_results': {
                'method': method4_results['method'],
                'cluster_assignments': {str(k): v for k, v in method4_results['cluster_assignments'].items()},
                'metrics': {
                    'accuracy': float(method4_results['metrics'].get('accuracy', 0)),
                    'f1_macro': float(method4_results['metrics'].get('f1_macro', 0)),
                    'f1_weighted': float(method4_results['metrics'].get('f1_weighted', 0)),
                    'precision_macro': float(method4_results['metrics'].get('precision_macro', 0)),
                    'recall_macro': float(method4_results['metrics'].get('recall_macro', 0)),
                    'num_classes': int(method4_results['metrics'].get('num_classes', 0)),
                    'num_samples': int(method4_results['metrics'].get('num_samples', 0)),
                    'per_class_f1': {k: float(v) for k, v in method4_results['metrics'].get('per_class_f1', {}).items()}
                },
                'similarity_scores': {str(k): float(v) for k, v in method4_results.get('similarity_scores', {}).items()},
                'caption_stats': {str(k): v for k, v in method4_results.get('caption_stats', {}).items()}
            },
            'summary': {
                'method1_f1_macro': float(method1_results['metrics'].get('f1_macro', 0)),
                'method2_f1_macro': float(method2_results['metrics'].get('f1_macro', 0)),
                'method3_f1_macro': float(method3_results['metrics'].get('f1_macro', 0)),
                'method4_f1_macro': float(method4_results['metrics'].get('f1_macro', 0)),
                'best_method': max([
                    ('method1', method1_results['metrics'].get('f1_macro', 0)),
                    ('method2', method2_results['metrics'].get('f1_macro', 0)),
                    ('method3', method3_results['metrics'].get('f1_macro', 0)),
                    ('method4', method4_results['metrics'].get('f1_macro', 0))
                ], key=lambda x: x[1])[0]
            }
        }

        results_file = self.output_dir / f'scan_evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)  # Use default=str to handle numpy types
        print(f"💾 Detailed results saved: {results_file}")

        print("\n" + "="*60)
        print("🎯 SCAN EVALUATION COMPLETED!")
        print("="*60)
        print(f"Results saved in: {self.output_dir}")

        return detailed_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate SCAN clustering results')
    parser.add_argument('--scan_model_path', type=str, required=True,
                       help='Path to trained SCAN model checkpoint')
    parser.add_argument('--baseline_model_path', type=str, required=True,
                       help='Path to baseline pre-trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test dataset (JSON file)')
    parser.add_argument('--vocab_path', type=str, required=True,
                       help='Path to vocabulary file')
    parser.add_argument('--output_dir', type=str,
                       default='results/evals/milan/baseline_50/scan_evaluations',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=10000,
                       help='Maximum number of samples to evaluate')

    args = parser.parse_args()

    # Create evaluator and run evaluation
    evaluator = SCANEvaluator(
        scan_model_path=args.scan_model_path,
        baseline_model_path=args.baseline_model_path,
        data_path=args.data_path,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir
    )

    results = evaluator.run_full_evaluation(max_samples=args.max_samples)

    print(f"\n✅ SCAN evaluation complete! Results saved in: {args.output_dir}")


if __name__ == '__main__':
    main()
