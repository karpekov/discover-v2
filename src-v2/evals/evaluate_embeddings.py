#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
"""
Embedding-based evaluation script for CASAS activity recognition.
Computes nearest neighbor predictions in embedding space and evaluates against ground truth labels.

Sample Usage:
python src-v2/evals/evaluate_embeddings.py \
    --checkpoint src-v2/trained_models/milan_tiny_50/best_model.pt \
    --train_data data/processed/casas/milan/training_50/train.json \
    --test_data data/processed/casas/milan/training_50/presegmented_test.json \
    --vocab data/processed/casas/milan/training_50/vocab.json \
    --output_dir src-v2/analysis/milan_tiny_50_TEST \
    --max_samples 10000 \
    --compare_filtering
"""

import torch
import numpy as np
import pandas as pd
import json
import argparse
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Evaluation metrics
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

# Local imports
from models.text_encoder import TextEncoder
from models.sensor_encoder import SensorEncoder
from dataio.dataset import SmartHomeDataset
from dataio.collate import create_data_loader
from utils.device_utils import get_optimal_device, log_device_info
from utils.label_utils import convert_labels_to_text


class EmbeddingEvaluator:
    """Evaluate embedding-based activity recognition using nearest neighbors."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_optimal_device()
        log_device_info(self.device)

        # Load models and data
        self._load_models()
        self._load_datasets()

        # Load label colors from metadata
        self._load_label_colors()

    def _load_models(self):
        """Load trained models from checkpoint."""
        print(f"🔄 Loading models from {self.config['checkpoint_path']}")

        checkpoint = torch.load(self.config['checkpoint_path'], map_location=self.device)

        # Text encoder - use config from checkpoint
        model_config = checkpoint.get('config', {})
        text_model_name = model_config.get('text_model_name', 'thenlper/gte-base')
        self.text_encoder = TextEncoder(text_model_name)
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
        self.sensor_encoder.eval()

        print("✅ Models loaded successfully")

    def _load_datasets(self):
        """Load datasets."""
        self.datasets = {}

        if self.config['train_data_path'] and Path(self.config['train_data_path']).exists():
            self.datasets['train'] = SmartHomeDataset(
                data_path=self.config['train_data_path'],
                vocab_path=self.config['vocab_path'],
                sequence_length=20,
                max_captions=1
            )
            print(f"📊 Train dataset: {len(self.datasets['train'])} samples")

        if self.config['test_data_path'] and Path(self.config['test_data_path']).exists():
            self.datasets['test'] = SmartHomeDataset(
                data_path=self.config['test_data_path'],
                vocab_path=self.config['vocab_path'],
                sequence_length=20,
                max_captions=1
            )
            print(f"📊 Test dataset: {len(self.datasets['test'])} samples")

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

    def extract_embeddings_and_labels(self, split: str, max_samples: int = 10000) -> Tuple[np.ndarray, List[str], List[str]]:
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

        # Create data loader with shuffling
        data_loader = create_data_loader(
            dataset=dataset,
            text_encoder=self.text_encoder,
            span_masker=None,
            vocab_sizes=self.vocab_sizes,
            device=self.device,
            batch_size=64,
            shuffle=True,  # Always shuffle for good measure
            num_workers=0,
            apply_mlm=False
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

                if batch_idx % 20 == 0:
                    print(f"  Processed {samples_processed}/{actual_samples} samples...")

        # Concatenate embeddings
        embeddings = np.vstack(embeddings)[:actual_samples]

        print(f"📈 Extracted {embeddings.shape[0]} embeddings")
        print(f"📊 L1 Labels: {len(set(labels_l1))} unique ({Counter(labels_l1).most_common(5)})")
        print(f"📊 L2 Labels: {len(set(labels_l2))} unique ({Counter(labels_l2).most_common(5)})")

        return embeddings, labels_l1, labels_l2

    def filter_noisy_labels(self, embeddings: np.ndarray, labels_l1: List[str], labels_l2: List[str],
                           original_indices: List[int] = None) -> Tuple[np.ndarray, List[str], List[str], List[int]]:
        """Filter out noisy/uninformative labels like 'Other', 'No_Activity', etc."""

        # Define labels to exclude (case-insensitive)
        exclude_labels = {
            # 'other',
            'no_activity',
            # 'unknown', 'none', 'null', 'nan',
            # 'no activity', 'other activity', 'miscellaneous', 'misc'
        }

        # If no original indices provided, create them
        if original_indices is None:
            original_indices = list(range(len(labels_l1)))

        # Find valid indices (keep samples that don't have excluded labels in either L1 or L2)
        valid_indices = []
        valid_original_indices = []

        for i, (l1, l2) in enumerate(zip(labels_l1, labels_l2)):
            l1_lower = l1.lower().strip()
            l2_lower = l2.lower().strip()

            # Keep sample if neither L1 nor L2 labels are in exclude list
            if l1_lower not in exclude_labels and l2_lower not in exclude_labels:
                valid_indices.append(i)
                valid_original_indices.append(original_indices[i])

        if not valid_indices:
            print("⚠️  Warning: All samples filtered out!")
            return embeddings, labels_l1, labels_l2, original_indices

        # Filter arrays and lists
        filtered_embeddings = embeddings[valid_indices]
        filtered_labels_l1 = [labels_l1[i] for i in valid_indices]
        filtered_labels_l2 = [labels_l2[i] for i in valid_indices]

        print(f"🧹 Filtered out noisy labels:")
        print(f"   Original samples: {len(labels_l1)}")
        print(f"   Filtered samples: {len(filtered_labels_l1)}")
        print(f"   Removed: {len(labels_l1) - len(filtered_labels_l1)} samples")

        # Show what was filtered out
        removed_l1 = Counter([labels_l1[i] for i in range(len(labels_l1)) if i not in valid_indices])
        removed_l2 = Counter([labels_l2[i] for i in range(len(labels_l2)) if i not in valid_indices])

        if removed_l1:
            print(f"   Removed L1 labels: {dict(removed_l1.most_common())}")
        if removed_l2:
            print(f"   Removed L2 labels: {dict(removed_l2.most_common())}")

        return filtered_embeddings, filtered_labels_l1, filtered_labels_l2, valid_original_indices

    def create_text_prototypes(self, labels: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """Create text-based prototypes using the text encoder with multiple captions per label."""

        print(f"🔄 Creating text-based label prototypes...")

        # Get unique labels and their counts
        label_counts = Counter(labels)
        unique_labels = list(label_counts.keys())

        print(f"📝 Encoding {len(unique_labels)} unique labels with text encoder...")

        # Convert labels to multiple natural language descriptions
        label_descriptions_lists = convert_labels_to_text(unique_labels)

        # Create prototypes dictionary by averaging embeddings for multiple captions
        prototypes = {}
        self.text_encoder.eval()

        with torch.no_grad():
            for i, label in enumerate(unique_labels):
                descriptions = label_descriptions_lists[i]

                # Encode all descriptions for this label
                caption_embeddings = self.text_encoder.encode_texts_clip(descriptions, self.device).cpu().numpy()

                # Average the embeddings to create a single prototype
                prototype_embedding = np.mean(caption_embeddings, axis=0)
                prototypes[label] = prototype_embedding

        print(f"✅ Created {len(prototypes)} text-based prototypes")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            descriptions = label_descriptions_lists[unique_labels.index(label)]
            print(f"    {label}: {count} samples → {len(descriptions)} captions (e.g., '{descriptions[0]}')")

        return prototypes, dict(label_counts)

    def create_nshot_text_prototypes(self, embeddings: np.ndarray, labels_l1: List[str],
                                   labels_l2: List[str], original_indices: List[int], n_shots: int = 1) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Create n-shot text prototypes by sampling n examples per class and encoding their captions."""

        print(f"🔄 Creating {n_shots}-shot text prototypes from actual captions...")

        # Group samples by label, storing both filtered indices and original indices
        l1_samples = defaultdict(list)
        l2_samples = defaultdict(list)

        for i, (l1, l2) in enumerate(zip(labels_l1, labels_l2)):
            l1_samples[l1].append((i, original_indices[i]))  # (filtered_idx, original_idx)
            l2_samples[l2].append((i, original_indices[i]))

        # Sample n examples per class and get their captions
        def create_prototypes_for_level(samples_dict, dataset_split='test'):
            prototypes = {}

            for label, sample_pairs in samples_dict.items():
                if len(sample_pairs) < n_shots:
                    print(f"⚠️  Label '{label}' has only {len(sample_pairs)} samples, using all available")
                    selected_pairs = sample_pairs
                else:
                    # Randomly sample n_shots examples
                    np.random.seed(42)  # For reproducibility
                    selected_indices = np.random.choice(len(sample_pairs), n_shots, replace=False)
                    selected_pairs = [sample_pairs[i] for i in selected_indices]

                # Get captions for selected samples using original indices
                captions = []
                dataset = self.datasets[dataset_split]
                for filtered_idx, original_idx in selected_pairs:
                    if hasattr(dataset, 'data') and original_idx < len(dataset.data):
                        sample = dataset.data[original_idx]  # Use original index
                        # Get captions from the sample
                        sample_captions = sample.get('captions', [])
                        if sample_captions:
                            # Use the first caption or a random one
                            caption = sample_captions[0] if isinstance(sample_captions[0], str) else sample_captions[0][0]
                            captions.append(caption)

                if not captions:
                    # Fallback to generated descriptions if no captions available
                    descriptions = convert_labels_to_text([label])[0]
                    captions = descriptions[:n_shots] if len(descriptions) >= n_shots else descriptions

                # Encode captions and average
                if captions:
                    self.text_encoder.eval()
                    with torch.no_grad():
                        caption_embeddings = self.text_encoder.encode_texts_clip(captions, self.device).cpu().numpy()
                        prototype_embedding = np.mean(caption_embeddings, axis=0)
                        prototypes[label] = prototype_embedding

            return prototypes

        prototypes_l1 = create_prototypes_for_level(l1_samples)
        prototypes_l2 = create_prototypes_for_level(l2_samples)

        print(f"✅ Created {n_shots}-shot text prototypes: L1={len(prototypes_l1)}, L2={len(prototypes_l2)}")

        return prototypes_l1, prototypes_l2

    def create_nshot_sensor_prototypes(self, embeddings: np.ndarray, labels_l1: List[str],
                                     labels_l2: List[str], n_shots: int = 1) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Create n-shot sensor prototypes by sampling n examples per class from sensor embeddings."""

        print(f"🔄 Creating {n_shots}-shot sensor prototypes from sensor embeddings...")

        # Group samples by label
        l1_samples = defaultdict(list)
        l2_samples = defaultdict(list)

        for i, (l1, l2) in enumerate(zip(labels_l1, labels_l2)):
            l1_samples[l1].append(i)
            l2_samples[l2].append(i)

        def create_prototypes_for_level(samples_dict):
            prototypes = {}

            for label, sample_indices in samples_dict.items():
                if len(sample_indices) < n_shots:
                    print(f"⚠️  Label '{label}' has only {len(sample_indices)} samples, using all available")
                    selected_indices = sample_indices
                else:
                    # Randomly sample n_shots examples
                    np.random.seed(42)  # For reproducibility
                    selected_indices = np.random.choice(sample_indices, n_shots, replace=False)

                # Get embeddings for selected samples and average
                selected_embeddings = embeddings[selected_indices]
                prototype_embedding = np.mean(selected_embeddings, axis=0)
                prototypes[label] = prototype_embedding

            return prototypes

        prototypes_l1 = create_prototypes_for_level(l1_samples)
        prototypes_l2 = create_prototypes_for_level(l2_samples)

        print(f"✅ Created {n_shots}-shot sensor prototypes: L1={len(prototypes_l1)}, L2={len(prototypes_l2)}")

        return prototypes_l1, prototypes_l2


    def predict_labels_knn(self, query_embeddings: np.ndarray,
                          prototypes: Dict[str, np.ndarray],
                          k: int = 1) -> List[str]:
        """Predict labels using k-nearest neighbors between sensor and text embeddings."""

        print(f"🔄 Predicting labels using {k}-NN cross-modal comparison...")
        print(f"    Sensor embeddings: {query_embeddings.shape[0]} samples")
        print(f"    Text prototypes: {len(prototypes)} activities")

        # Convert prototypes to arrays
        prototype_labels = list(prototypes.keys())
        prototype_embeddings = np.array([prototypes[label] for label in prototype_labels])

        # Normalize embeddings for cosine similarity
        query_embeddings_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        prototype_embeddings_norm = prototype_embeddings / (np.linalg.norm(prototype_embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarities
        similarities = np.dot(query_embeddings_norm, prototype_embeddings_norm.T)

        if k == 1:
            # Simple nearest neighbor
            nearest_indices = np.argmax(similarities, axis=1)
            predictions = [prototype_labels[idx] for idx in nearest_indices]
        else:
            # k-NN with majority voting
            top_k_indices = np.argsort(similarities, axis=1)[:, -k:]
            predictions = []

            for indices in top_k_indices:
                # Get top-k labels and vote
                top_labels = [prototype_labels[idx] for idx in indices]
                # Simple majority vote (could be weighted by similarity)
                prediction = Counter(top_labels).most_common(1)[0][0]
                predictions.append(prediction)

        return predictions

    def evaluate_predictions(self, true_labels: List[str], pred_labels: List[str],
                           label_type: str) -> Dict[str, Any]:
        """Compute evaluation metrics."""

        print(f"🔄 Computing metrics for {label_type} labels...")

        # Filter out unknown labels for fair evaluation
        valid_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels))
                        if true != 'Unknown' and pred != 'Unknown']

        if not valid_indices:
            print(f"⚠️  No valid predictions for {label_type}")
            return {}

        true_filtered = [true_labels[i] for i in valid_indices]
        pred_filtered = [pred_labels[i] for i in valid_indices]

        # Get unique labels
        unique_labels = sorted(list(set(true_filtered + pred_filtered)))

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(true_filtered, pred_filtered),
            'f1_macro': f1_score(true_filtered, pred_filtered, average='macro', zero_division=0),
            'f1_micro': f1_score(true_filtered, pred_filtered, average='micro', zero_division=0),
            'f1_weighted': f1_score(true_filtered, pred_filtered, average='weighted', zero_division=0),
            'precision_macro': precision_score(true_filtered, pred_filtered, average='macro', zero_division=0),
            'recall_macro': recall_score(true_filtered, pred_filtered, average='macro', zero_division=0),
            'num_samples': len(true_filtered),
            'num_classes': len(unique_labels),
            'unique_labels': unique_labels
        }

        # Per-class F1 scores
        per_class_f1 = f1_score(true_filtered, pred_filtered, average=None, zero_division=0, labels=unique_labels)
        metrics['per_class_f1'] = dict(zip(unique_labels, per_class_f1))

        # Classification report
        metrics['classification_report'] = classification_report(
            true_filtered, pred_filtered,
            target_names=unique_labels,
            zero_division=0,
            output_dict=True
        )

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(true_filtered, pred_filtered, labels=unique_labels)

        print(f"✅ {label_type} Metrics:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 (Macro): {metrics['f1_macro']:.4f}")
        print(f"    F1 (Micro): {metrics['f1_micro']:.4f}")
        print(f"    F1 (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"    Classes: {metrics['num_classes']}")
        print(f"    Samples: {metrics['num_samples']}")

        return metrics

    def create_confusion_matrix_plot(self, confusion_matrix: np.ndarray,
                                   labels: List[str],
                                   title: str,
                                   save_path: str = None) -> plt.Figure:
        """Create confusion matrix visualization with improved readability for many classes."""

        # Limit to top classes for readability
        if len(labels) > 20:
            print(f"⚠️  Too many classes ({len(labels)}), showing top 15 by frequency")
            # This is simplified - in practice you'd want to select top classes by frequency
            labels = labels[:15]
            confusion_matrix = confusion_matrix[:15, :15]

        # Dynamic figure sizing based on number of classes
        num_classes = len(labels)
        if num_classes <= 8:
            figsize = (10, 8)
            annot_fontsize = 10
        elif num_classes <= 12:
            figsize = (12, 10)
            annot_fontsize = 8
        elif num_classes <= 16:
            figsize = (14, 12)
            annot_fontsize = 7
        else:
            figsize = (16, 14)
            annot_fontsize = 6

        fig, ax = plt.subplots(figsize=figsize)

        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-8)

        # Truncate long labels for better display
        display_labels = []
        for label in labels:
            # Replace underscores with spaces and truncate if too long
            clean_label = label.replace('_', ' ')
            if len(clean_label) > 12:
                clean_label = clean_label[:10] + '..'
            display_labels.append(clean_label)

        # Create heatmap with improved settings
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=display_labels,
            yticklabels=display_labels,
            ax=ax,
            cbar_kws={'label': 'Normalized Count', 'shrink': 0.8},
            annot_kws={'size': annot_fontsize},
            square=True,  # Make cells square for better readability
            linewidths=0.5,  # Add thin lines between cells
            linecolor='white'
        )

        ax.set_title(f'Confusion Matrix - {title}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)

        # Rotate x-axis labels to 90 degrees as requested
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Confusion matrix saved: {save_path}")

        return fig

    def create_combined_confusion_matrices(self,
                                         metrics_l1_filtered: Dict[str, Any],
                                         metrics_l2_filtered: Dict[str, Any],
                                         metrics_l1_unfiltered: Dict[str, Any],
                                         metrics_l2_unfiltered: Dict[str, Any],
                                         title: str, subtitle: str = "", save_path: str = None) -> plt.Figure:
        """Create combined confusion matrices plot with 4 matrices."""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        # Set main title and subtitle
        if subtitle:
            fig.suptitle(f'Confusion Matrices - {title}\n{subtitle}', fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f'Confusion Matrices - {title}', fontsize=16, fontweight='bold')

        # Colors for different label levels
        l1_cmap = 'Blues'  # Blue for L1
        l2_cmap = 'Oranges'  # Orange for L2

        # Helper function to create individual confusion matrix
        def plot_confusion_matrix(ax, cm, labels, title, cmap, max_labels=18):
            if len(labels) > max_labels:
                print(f"⚠️ Too many classes ({len(labels)}), showing top {max_labels}")
                cm = cm[:max_labels, :max_labels]
                labels = labels[:max_labels]

            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

            # Determine font sizes based on number of classes
            num_classes = len(labels)
            if num_classes <= 8:
                annot_fontsize = 8
                tick_fontsize = 9
            elif num_classes <= 12:
                annot_fontsize = 6
                tick_fontsize = 8
            else:
                annot_fontsize = 5
                tick_fontsize = 7

            # Truncate and clean labels for better display
            display_labels = []
            for lbl in labels:
                clean_lbl = lbl.replace('_', ' ')
                if len(clean_lbl) > 10:
                    clean_lbl = clean_lbl[:8] + '..'
                display_labels.append(clean_lbl)

            # Create heatmap
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2f',
                cmap=cmap,
                xticklabels=display_labels,
                yticklabels=display_labels,
                ax=ax,
                cbar_kws={'label': 'Normalized Count', 'shrink': 0.6},
                annot_kws={'size': annot_fontsize},
                square=True,
                linewidths=0.3,
                linecolor='white'
            )

            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=9)
            ax.set_ylabel('True Label', fontsize=9)

            # Rotate x-axis labels to 90 degrees as requested
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=tick_fontsize)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=tick_fontsize)

        # Plot L1 Filtered (Top Left)
        if 'confusion_matrix' in metrics_l1_filtered and 'unique_labels' in metrics_l1_filtered:
            plot_confusion_matrix(
                ax1,
                metrics_l1_filtered['confusion_matrix'],
                metrics_l1_filtered['unique_labels'],
                'L1 Primary (Filtered)',
                l1_cmap
            )
        else:
            ax1.text(0.5, 0.5, 'L1 Filtered\nData Not Available',
                    transform=ax1.transAxes, ha='center', va='center', fontsize=12)
            ax1.set_title('L1 Primary (Filtered)')

        # Plot L2 Filtered (Top Right)
        if 'confusion_matrix' in metrics_l2_filtered and 'unique_labels' in metrics_l2_filtered:
            plot_confusion_matrix(
                ax2,
                metrics_l2_filtered['confusion_matrix'],
                metrics_l2_filtered['unique_labels'],
                'L2 Secondary (Filtered)',
                l2_cmap
            )
        else:
            ax2.text(0.5, 0.5, 'L2 Filtered\nData Not Available',
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('L2 Secondary (Filtered)')

        # Plot L1 Unfiltered (Bottom Left)
        if 'confusion_matrix' in metrics_l1_unfiltered and 'unique_labels' in metrics_l1_unfiltered:
            plot_confusion_matrix(
                ax3,
                metrics_l1_unfiltered['confusion_matrix'],
                metrics_l1_unfiltered['unique_labels'],
                'L1 Primary (With Noisy)',
                l1_cmap
            )
        else:
            ax3.text(0.5, 0.5, 'L1 Unfiltered\nData Not Available',
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('L1 Primary (With Noisy)')

        # Plot L2 Unfiltered (Bottom Right)
        if 'confusion_matrix' in metrics_l2_unfiltered and 'unique_labels' in metrics_l2_unfiltered:
            plot_confusion_matrix(
                ax4,
                metrics_l2_unfiltered['confusion_matrix'],
                metrics_l2_unfiltered['unique_labels'],
                'L2 Secondary (With Noisy)',
                l2_cmap
            )
        else:
            ax4.text(0.5, 0.5, 'L2 Unfiltered\nData Not Available',
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('L2 Secondary (With Noisy)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Combined confusion matrices saved: {save_path}")

        return fig

    def create_f1_scores_plot(self, metrics_l1_filtered: Dict[str, Any], metrics_l2_filtered: Dict[str, Any],
                             metrics_l1_unfiltered: Dict[str, Any] = None, metrics_l2_unfiltered: Dict[str, Any] = None,
                             title: str = "", subtitle: str = "", save_path: str = None) -> plt.Figure:
        """Create comprehensive F1 scores visualization comparing filtered vs unfiltered results."""

        # Use 2x2 layout: overall metrics centered top, per-class charts in bottom row
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Overall metrics centered in top row (spans 3 columns)
        ax1 = fig.add_subplot(gs[0, :])

        # Per-class charts in bottom row
        ax2 = fig.add_subplot(gs[1, 0])  # L1 per-class (left)
        ax3 = fig.add_subplot(gs[1, 2])  # L2 per-class (right)

        # Set main title and subtitle
        if subtitle:
            fig.suptitle(f'F1 Score Analysis - {title}\n{subtitle}', fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f'F1 Score Analysis - {title}', fontsize=16, fontweight='bold')

        # Updated colors: Blue for L1, Orange for L2
        l1_color = '#1976D2'  # Blue
        l2_color = '#FF6F00'  # Orange

        # 1. Overall Performance Metrics Comparison (Top Left)
        overall_metrics = ['f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'accuracy']

        if metrics_l1_unfiltered and metrics_l2_unfiltered:
            # Compare filtered vs unfiltered
            l1_filtered = [metrics_l1_filtered.get(metric, 0) for metric in overall_metrics]
            l1_unfiltered = [metrics_l1_unfiltered.get(metric, 0) for metric in overall_metrics]
            l2_filtered = [metrics_l2_filtered.get(metric, 0) for metric in overall_metrics]
            l2_unfiltered = [metrics_l2_unfiltered.get(metric, 0) for metric in overall_metrics]

            x = np.arange(len(overall_metrics))
            width = 0.2

            bars1 = ax1.bar(x - 1.5*width, l1_unfiltered, width, label='L1 (with noisy)',
                           color=l1_color, alpha=0.6)
            bars2 = ax1.bar(x - 0.5*width, l1_filtered, width, label='L1 (filtered)',
                           color=l1_color, alpha=0.9)
            bars3 = ax1.bar(x + 0.5*width, l2_unfiltered, width, label='L2 (with noisy)',
                           color=l2_color, alpha=0.6)
            bars4 = ax1.bar(x + 1.5*width, l2_filtered, width, label='L2 (filtered)',
                           color=l2_color, alpha=0.9)

            all_bars = [bars1, bars2, bars3, bars4]
        else:
            # Just show filtered results
            l1_scores = [metrics_l1_filtered.get(metric, 0) for metric in overall_metrics]
            l2_scores = [metrics_l2_filtered.get(metric, 0) for metric in overall_metrics]

            x = np.arange(len(overall_metrics))
            width = 0.35

            bars1 = ax1.bar(x - width/2, l1_scores, width, label='L1 (Primary)',
                           color=l1_color, alpha=0.8)
            bars2 = ax1.bar(x + width/2, l2_scores, width, label='L2 (Secondary)',
                           color=l2_color, alpha=0.8)

            all_bars = [bars1, bars2]

        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Overall Performance Metrics')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['F1 Macro', 'F1 Weighted', 'Precision', 'Recall', 'Accuracy'], rotation=15)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Add value labels on bars
        for bars in all_bars:
            for bar in bars:
                height = bar.get_height()
                if height > 0.02:  # Only show labels for bars that are tall enough
                    ax1.annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        # 2. L1 Per-Class F1 Scores (Top Right)
        if 'per_class_f1' in metrics_l1_filtered and metrics_l1_filtered['per_class_f1']:
            l1_classes = list(metrics_l1_filtered['per_class_f1'].keys())
            l1_f1_scores = list(metrics_l1_filtered['per_class_f1'].values())

            # Sort by F1 score for better visualization
            sorted_data = sorted(zip(l1_classes, l1_f1_scores), key=lambda x: x[1], reverse=True)
            l1_classes_sorted = [x[0] for x in sorted_data]
            l1_f1_scores_sorted = [x[1] for x in sorted_data]

            # Limit to top 10 for readability
            if len(l1_classes_sorted) > 10:
                l1_classes_sorted = l1_classes_sorted[:10]
                l1_f1_scores_sorted = l1_f1_scores_sorted[:10]

            bars = ax2.barh(range(len(l1_classes_sorted)), l1_f1_scores_sorted,
                           color=l1_color, alpha=0.7)
            ax2.set_yticks(range(len(l1_classes_sorted)))
            ax2.set_yticklabels([cls.replace('_', ' ') for cls in l1_classes_sorted])
            ax2.set_xlabel('F1 Score')
            ax2.set_title(f'L1 Per-Class F1 Scores (Top {len(l1_classes_sorted)})')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 1)

            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax2.annotate(f'{width:.3f}',
                            xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(3, 0),
                            textcoords="offset points",
                            ha='left', va='center', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No L1 per-class data available',
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('L1 Per-Class F1 Scores')

        # 3. L2 Per-Class F1 Scores (Bottom Right)
        if 'per_class_f1' in metrics_l2_filtered and metrics_l2_filtered['per_class_f1']:
            l2_classes = list(metrics_l2_filtered['per_class_f1'].keys())
            l2_f1_scores = list(metrics_l2_filtered['per_class_f1'].values())

            # Sort by F1 score for better visualization
            sorted_data = sorted(zip(l2_classes, l2_f1_scores), key=lambda x: x[1], reverse=True)
            l2_classes_sorted = [x[0] for x in sorted_data]
            l2_f1_scores_sorted = [x[1] for x in sorted_data]

            # Limit to top 10 for readability
            if len(l2_classes_sorted) > 10:
                l2_classes_sorted = l2_classes_sorted[:10]
                l2_f1_scores_sorted = l2_f1_scores_sorted[:10]

            bars = ax3.barh(range(len(l2_classes_sorted)), l2_f1_scores_sorted,
                           color=l2_color, alpha=0.7)
            ax3.set_yticks(range(len(l2_classes_sorted)))
            ax3.set_yticklabels([cls.replace('_', ' ') for cls in l2_classes_sorted])
            ax3.set_xlabel('F1 Score')
            ax3.set_title(f'L2 Per-Class F1 Scores (Top {len(l2_classes_sorted)})')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 1)

            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax3.annotate(f'{width:.3f}',
                            xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(3, 0),
                            textcoords="offset points",
                            ha='left', va='center', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'No L2 per-class data available',
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('L2 Per-Class F1 Scores')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 F1 scores plot saved: {save_path}")

        return fig

    def create_nshot_comparison_plots(self, all_results: Dict[str, Any], max_samples: int, output_dir: Path):
        """Create comprehensive comparison plots for n-shot evaluation results."""

        print("🎨 Creating n-shot comparison plots...")

        # Extract metrics for plotting
        methods = []
        l1_f1_macro_scores = []
        l1_f1_weighted_scores = []
        l1_accuracies = []
        l2_f1_macro_scores = []
        l2_f1_weighted_scores = []
        l2_accuracies = []

        # Extract n-shot values dynamically from results
        n_shots = []
        for key in all_results.keys():
            if key.endswith('_shot_text'):
                n = int(key.split('_')[0])
                if n not in n_shots:
                    n_shots.append(n)
        n_shots.sort()

        # Order: 0-shot, then n-shot text, then n-shot sensor
        method_order = ['0_shot']
        for n in n_shots:
            method_order.extend([f'{n}_shot_text', f'{n}_shot_sensor'])

        for method in method_order:
            if method in all_results:
                methods.append(method)
                result = all_results[method]
                l1_f1_macro_scores.append(result['metrics_l1'].get('f1_macro', 0))
                l1_f1_weighted_scores.append(result['metrics_l1'].get('f1_weighted', 0))
                l1_accuracies.append(result['metrics_l1'].get('accuracy', 0))
                l2_f1_macro_scores.append(result['metrics_l2'].get('f1_macro', 0))
                l2_f1_weighted_scores.append(result['metrics_l2'].get('f1_weighted', 0))
                l2_accuracies.append(result['metrics_l2'].get('accuracy', 0))

        # Create comprehensive comparison plot with 2x3 layout
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'N-Shot Evaluation Comparison ({max_samples} samples)', fontsize=16, fontweight='bold')

        # Colors for different method types
        colors = []
        for method in methods:
            if method == '0_shot':
                colors.append('#2E7D32')  # Green for 0-shot
            elif 'text' in method:
                colors.append('#1976D2')  # Blue for text-based
            elif 'sensor' in method:
                colors.append('#FF6F00')  # Orange for sensor-based
            else:
                colors.append('#757575')  # Gray for others

        # Clean method names for display
        display_names = []
        for method in methods:
            if method == '0_shot':
                display_names.append('0-shot\n(Text Prototypes)')
            elif 'text' in method:
                n = method.split('_')[0]
                display_names.append(f'{n}-shot\n(Text)')
            elif 'sensor' in method:
                n = method.split('_')[0]
                display_names.append(f'{n}-shot\n(Sensor)')
            else:
                display_names.append(method)

        x = np.arange(len(methods))

        # L1 F1-Macro Scores (Top Left)
        bars1 = ax1.bar(x, l1_f1_macro_scores, color=colors, alpha=0.8)
        ax1.set_title('L1 (Primary) F1-Macro Scores', fontweight='bold')
        ax1.set_ylabel('F1-Macro Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(display_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Add value labels
        for bar, score in zip(bars1, l1_f1_macro_scores):
            height = bar.get_height()
            ax1.annotate(f'{score:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        # L1 F1-Weighted Scores (Top Middle)
        bars2 = ax2.bar(x, l1_f1_weighted_scores, color=colors, alpha=0.8)
        ax2.set_title('L1 (Primary) F1-Weighted Scores', fontweight='bold')
        ax2.set_ylabel('F1-Weighted Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # Add value labels
        for bar, score in zip(bars2, l1_f1_weighted_scores):
            height = bar.get_height()
            ax2.annotate(f'{score:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        # L1 Accuracy (Top Right)
        bars3 = ax3.bar(x, l1_accuracies, color=colors, alpha=0.8)
        ax3.set_title('L1 (Primary) Accuracy', fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_xticks(x)
        ax3.set_xticklabels(display_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)

        # Add value labels
        for bar, score in zip(bars3, l1_accuracies):
            height = bar.get_height()
            ax3.annotate(f'{score:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        # L2 F1-Macro Scores (Bottom Left)
        bars4 = ax4.bar(x, l2_f1_macro_scores, color=colors, alpha=0.8)
        ax4.set_title('L2 (Secondary) F1-Macro Scores', fontweight='bold')
        ax4.set_ylabel('F1-Macro Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(display_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)

        # Add value labels
        for bar, score in zip(bars4, l2_f1_macro_scores):
            height = bar.get_height()
            ax4.annotate(f'{score:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        # L2 F1-Weighted Scores (Bottom Middle)
        bars5 = ax5.bar(x, l2_f1_weighted_scores, color=colors, alpha=0.8)
        ax5.set_title('L2 (Secondary) F1-Weighted Scores', fontweight='bold')
        ax5.set_ylabel('F1-Weighted Score')
        ax5.set_xticks(x)
        ax5.set_xticklabels(display_names, rotation=45, ha='right')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1)

        # Add value labels
        for bar, score in zip(bars5, l2_f1_weighted_scores):
            height = bar.get_height()
            ax5.annotate(f'{score:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        # L2 Accuracy (Bottom Right)
        bars6 = ax6.bar(x, l2_accuracies, color=colors, alpha=0.8)
        ax6.set_title('L2 (Secondary) Accuracy', fontweight='bold')
        ax6.set_ylabel('Accuracy')
        ax6.set_xticks(x)
        ax6.set_xticklabels(display_names, rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1)

        # Add value labels
        for bar, score in zip(bars6, l2_accuracies):
            height = bar.get_height()
            ax6.annotate(f'{score:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        # Save plot
        plot_path = output_dir / f'nshot_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"💾 N-shot comparison plot saved: {plot_path}")

        plt.close()

        # Create trend line plot
        self.create_nshot_trend_plot(all_results, max_samples, output_dir)

    def create_nshot_trend_plot(self, all_results: Dict[str, Any], max_samples: int, output_dir: Path):
        """Create trend line plot showing performance vs number of shots."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'N-Shot Performance Trends ({max_samples} samples)', fontsize=16, fontweight='bold')

        # Extract n-shot values dynamically from results
        n_shots_dynamic = []
        for key in all_results.keys():
            if key.endswith('_shot_text'):
                n = int(key.split('_')[0])
                if n not in n_shots_dynamic:
                    n_shots_dynamic.append(n)
        n_shots_dynamic.sort()

        # Include 0-shot at the beginning
        n_shots = [0] + n_shots_dynamic

        # L1 trends
        l1_text_f1_macro = []
        l1_text_f1_weighted = []
        l1_sensor_f1_macro = []
        l1_sensor_f1_weighted = []
        l1_text_acc = []
        l1_sensor_acc = []

        # L2 trends
        l2_text_f1_macro = []
        l2_text_f1_weighted = []
        l2_sensor_f1_macro = []
        l2_sensor_f1_weighted = []
        l2_text_acc = []
        l2_sensor_acc = []

        for n in n_shots:
            if n == 0:
                # 0-shot uses text prototypes
                if '0_shot' in all_results:
                    result = all_results['0_shot']
                    l1_text_f1_macro.append(result['metrics_l1'].get('f1_macro', 0))
                    l1_text_f1_weighted.append(result['metrics_l1'].get('f1_weighted', 0))
                    l1_sensor_f1_macro.append(result['metrics_l1'].get('f1_macro', 0))  # Same for both
                    l1_sensor_f1_weighted.append(result['metrics_l1'].get('f1_weighted', 0))
                    l1_text_acc.append(result['metrics_l1'].get('accuracy', 0))
                    l1_sensor_acc.append(result['metrics_l1'].get('accuracy', 0))

                    l2_text_f1_macro.append(result['metrics_l2'].get('f1_macro', 0))
                    l2_text_f1_weighted.append(result['metrics_l2'].get('f1_weighted', 0))
                    l2_sensor_f1_macro.append(result['metrics_l2'].get('f1_macro', 0))
                    l2_sensor_f1_weighted.append(result['metrics_l2'].get('f1_weighted', 0))
                    l2_text_acc.append(result['metrics_l2'].get('accuracy', 0))
                    l2_sensor_acc.append(result['metrics_l2'].get('accuracy', 0))
            else:
                # n-shot
                text_key = f'{n}_shot_text'
                sensor_key = f'{n}_shot_sensor'

                if text_key in all_results:
                    result = all_results[text_key]
                    l1_text_f1_macro.append(result['metrics_l1'].get('f1_macro', 0))
                    l1_text_f1_weighted.append(result['metrics_l1'].get('f1_weighted', 0))
                    l1_text_acc.append(result['metrics_l1'].get('accuracy', 0))
                    l2_text_f1_macro.append(result['metrics_l2'].get('f1_macro', 0))
                    l2_text_f1_weighted.append(result['metrics_l2'].get('f1_weighted', 0))
                    l2_text_acc.append(result['metrics_l2'].get('accuracy', 0))

                if sensor_key in all_results:
                    result = all_results[sensor_key]
                    l1_sensor_f1_macro.append(result['metrics_l1'].get('f1_macro', 0))
                    l1_sensor_f1_weighted.append(result['metrics_l1'].get('f1_weighted', 0))
                    l1_sensor_acc.append(result['metrics_l1'].get('accuracy', 0))
                    l2_sensor_f1_macro.append(result['metrics_l2'].get('f1_macro', 0))
                    l2_sensor_f1_weighted.append(result['metrics_l2'].get('f1_weighted', 0))
                    l2_sensor_acc.append(result['metrics_l2'].get('accuracy', 0))

        # Plot L1 trends
        if len(l1_text_f1_macro) == len(n_shots):
            ax1.plot(n_shots, l1_text_f1_macro, 'o-', color='#1976D2', label='Text F1-Macro', linewidth=2, markersize=8)
            ax1.plot(n_shots, l1_text_f1_weighted, 'o--', color='#1976D2', alpha=0.7, label='Text F1-Weighted', linewidth=2, markersize=6)
            ax1.plot(n_shots, l1_text_acc, 's:', color='#1976D2', alpha=0.5, label='Text Accuracy', linewidth=2, markersize=4)

        if len(l1_sensor_f1_macro) == len(n_shots):
            ax1.plot(n_shots, l1_sensor_f1_macro, 'o-', color='#FF6F00', label='Sensor F1-Macro', linewidth=2, markersize=8)
            ax1.plot(n_shots, l1_sensor_f1_weighted, 'o--', color='#FF6F00', alpha=0.7, label='Sensor F1-Weighted', linewidth=2, markersize=6)
            ax1.plot(n_shots, l1_sensor_acc, 's:', color='#FF6F00', alpha=0.5, label='Sensor Accuracy', linewidth=2, markersize=4)

        ax1.set_title('L1 (Primary) Performance Trends', fontweight='bold')
        ax1.set_xlabel('Number of Shots')
        ax1.set_ylabel('Score')
        ax1.set_xticks(n_shots)
        x_labels = ['0 (prototypes)'] + [str(n) for n in n_shots_dynamic]
        ax1.set_xticklabels(x_labels)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Plot L2 trends
        if len(l2_text_f1_macro) == len(n_shots):
            ax2.plot(n_shots, l2_text_f1_macro, 'o-', color='#1976D2', label='Text F1-Macro', linewidth=2, markersize=8)
            ax2.plot(n_shots, l2_text_f1_weighted, 'o--', color='#1976D2', alpha=0.7, label='Text F1-Weighted', linewidth=2, markersize=6)
            ax2.plot(n_shots, l2_text_acc, 's:', color='#1976D2', alpha=0.5, label='Text Accuracy', linewidth=2, markersize=4)

        if len(l2_sensor_f1_macro) == len(n_shots):
            ax2.plot(n_shots, l2_sensor_f1_macro, 'o-', color='#FF6F00', label='Sensor F1-Macro', linewidth=2, markersize=8)
            ax2.plot(n_shots, l2_sensor_f1_weighted, 'o--', color='#FF6F00', alpha=0.7, label='Sensor F1-Weighted', linewidth=2, markersize=6)
            ax2.plot(n_shots, l2_sensor_acc, 's:', color='#FF6F00', alpha=0.5, label='Sensor Accuracy', linewidth=2, markersize=4)

        ax2.set_title('L2 (Secondary) Performance Trends', fontweight='bold')
        ax2.set_xlabel('Number of Shots')
        ax2.set_ylabel('Score')
        ax2.set_xticks(n_shots)
        x_labels = ['0 (prototypes)'] + [str(n) for n in n_shots_dynamic]
        ax2.set_xticklabels(x_labels)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.tight_layout()

        # Save plot
        trend_path = output_dir / f'nshot_trends.png'
        plt.savefig(trend_path, dpi=300, bbox_inches='tight')
        print(f"💾 N-shot trend plot saved: {trend_path}")

        plt.close()

    def create_nshot_text_report(self, all_results: Dict[str, Any], output_dir: Path, max_samples: int):
        """Create comprehensive text report for n-shot evaluation."""

        print("📝 Creating n-shot text report...")

        report_path = output_dir / f'nshot_evaluation_report.txt'

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("N-SHOT EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Max Samples: {max_samples}\n")
            f.write(f"Methods Evaluated: {len(all_results)}\n\n")

            # Summary table
            f.write("SUMMARY RESULTS\n")
            f.write("-" * 110 + "\n")
            f.write(f"{'Method':<20} {'L1 F1-Macro':<12} {'L1 F1-Wtd':<12} {'L1 Accuracy':<12} {'L2 F1-Macro':<12} {'L2 F1-Wtd':<12} {'L2 Accuracy':<12}\n")
            f.write("-" * 110 + "\n")

            # Extract n-shot values dynamically from results
            n_shots = []
            for key in all_results.keys():
                if key.endswith('_shot_text'):
                    n = int(key.split('_')[0])
                    if n not in n_shots:
                        n_shots.append(n)
            n_shots.sort()

            methods = ['0_shot'] + [f'{n}_shot_text' for n in n_shots] + [f'{n}_shot_sensor' for n in n_shots]

            for method in methods:
                if method in all_results:
                    result = all_results[method]
                    l1_f1_macro = result['metrics_l1'].get('f1_macro', 0)
                    l1_f1_weighted = result['metrics_l1'].get('f1_weighted', 0)
                    l1_acc = result['metrics_l1'].get('accuracy', 0)
                    l2_f1_macro = result['metrics_l2'].get('f1_macro', 0)
                    l2_f1_weighted = result['metrics_l2'].get('f1_weighted', 0)
                    l2_acc = result['metrics_l2'].get('accuracy', 0)

                    f.write(f"{method:<20} {l1_f1_macro:<12.4f} {l1_f1_weighted:<12.4f} {l1_acc:<12.4f} {l2_f1_macro:<12.4f} {l2_f1_weighted:<12.4f} {l2_acc:<12.4f}\n")

            f.write("\n\n")

            # Detailed analysis
            f.write("DETAILED ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            for method, result in all_results.items():
                f.write(f"{method.upper().replace('_', '-')}\n")
                f.write("-" * 40 + "\n")

                # L1 metrics
                l1_metrics = result['metrics_l1']
                f.write(f"L1 (Primary) Metrics:\n")
                f.write(f"  F1-Macro:      {l1_metrics.get('f1_macro', 0):.4f}\n")
                f.write(f"  F1-Weighted:   {l1_metrics.get('f1_weighted', 0):.4f}\n")
                f.write(f"  Accuracy:      {l1_metrics.get('accuracy', 0):.4f}\n")
                f.write(f"  Precision:     {l1_metrics.get('precision_macro', 0):.4f}\n")
                f.write(f"  Recall:        {l1_metrics.get('recall_macro', 0):.4f}\n")
                f.write(f"  Classes:       {l1_metrics.get('num_classes', 0)}\n")
                f.write(f"  Samples:       {l1_metrics.get('num_samples', 0)}\n\n")

                # L2 metrics
                l2_metrics = result['metrics_l2']
                f.write(f"L2 (Secondary) Metrics:\n")
                f.write(f"  F1-Macro:      {l2_metrics.get('f1_macro', 0):.4f}\n")
                f.write(f"  F1-Weighted:   {l2_metrics.get('f1_weighted', 0):.4f}\n")
                f.write(f"  Accuracy:      {l2_metrics.get('accuracy', 0):.4f}\n")
                f.write(f"  Precision:     {l2_metrics.get('precision_macro', 0):.4f}\n")
                f.write(f"  Recall:        {l2_metrics.get('recall_macro', 0):.4f}\n")
                f.write(f"  Classes:       {l2_metrics.get('num_classes', 0)}\n")
                f.write(f"  Samples:       {l2_metrics.get('num_samples', 0)}\n\n")

                # Top performing classes
                if 'per_class_f1' in l1_metrics and l1_metrics['per_class_f1']:
                    f.write("Top L1 Classes (by F1-score):\n")
                    sorted_l1 = sorted(l1_metrics['per_class_f1'].items(), key=lambda x: x[1], reverse=True)[:5]
                    for class_name, f1_score in sorted_l1:
                        f.write(f"  {class_name:<25} {f1_score:.4f}\n")
                    f.write("\n")

                if 'per_class_f1' in l2_metrics and l2_metrics['per_class_f1']:
                    f.write("Top L2 Classes (by F1-score):\n")
                    sorted_l2 = sorted(l2_metrics['per_class_f1'].items(), key=lambda x: x[1], reverse=True)[:5]
                    for class_name, f1_score in sorted_l2:
                        f.write(f"  {class_name:<25} {f1_score:.4f}\n")
                    f.write("\n")

                f.write("\n" + "="*40 + "\n\n")

            # Key insights
            f.write("KEY INSIGHTS\n")
            f.write("=" * 80 + "\n\n")

            # Find best performing methods
            best_l1_f1 = max(all_results.items(), key=lambda x: x[1]['metrics_l1'].get('f1_macro', 0))
            best_l2_f1 = max(all_results.items(), key=lambda x: x[1]['metrics_l2'].get('f1_macro', 0))

            f.write(f"Best L1 F1-Macro: {best_l1_f1[0]} ({best_l1_f1[1]['metrics_l1'].get('f1_macro', 0):.4f})\n")
            f.write(f"Best L2 F1-Macro: {best_l2_f1[0]} ({best_l2_f1[1]['metrics_l2'].get('f1_macro', 0):.4f})\n\n")

            # Compare 0-shot vs n-shot
            if '0_shot' in all_results and '5_shot_text' in all_results:
                zero_l1 = all_results['0_shot']['metrics_l1'].get('f1_macro', 0)
                five_text_l1 = all_results['5_shot_text']['metrics_l1'].get('f1_macro', 0)
                improvement = five_text_l1 - zero_l1
                f.write(f"L1 Improvement (5-shot text vs 0-shot): {improvement:+.4f}\n")

            if '0_shot' in all_results and '5_shot_sensor' in all_results:
                zero_l1 = all_results['0_shot']['metrics_l1'].get('f1_macro', 0)
                five_sensor_l1 = all_results['5_shot_sensor']['metrics_l1'].get('f1_macro', 0)
                improvement = five_sensor_l1 - zero_l1
                f.write(f"L1 Improvement (5-shot sensor vs 0-shot): {improvement:+.4f}\n")

        print(f"💾 N-shot text report saved: {report_path}")

    def create_nshot_per_class_plots(self, all_results: Dict[str, Any], max_samples: int, output_dir: Path):
        """Create per-class F1-macro and F1-weighted plots for all methods."""

        print("🎨 Creating per-class F1 score plots...")

        # Extract n-shot values dynamically from results
        n_shots = []
        for key in all_results.keys():
            if key.endswith('_shot_text'):
                n = int(key.split('_')[0])
                if n not in n_shots:
                    n_shots.append(n)
        n_shots.sort()

        # Create method order
        method_order = ['0_shot']
        for n in n_shots:
            method_order.extend([f'{n}_shot_text', f'{n}_shot_sensor'])

        # Get all unique labels from all methods
        all_l1_labels = set()
        all_l2_labels = set()

        for method in method_order:
            if method in all_results:
                result = all_results[method]
                if 'per_class_f1' in result['metrics_l1']:
                    all_l1_labels.update(result['metrics_l1']['per_class_f1'].keys())
                if 'per_class_f1' in result['metrics_l2']:
                    all_l2_labels.update(result['metrics_l2']['per_class_f1'].keys())

        all_l1_labels = sorted(list(all_l1_labels))
        all_l2_labels = sorted(list(all_l2_labels))

        # Create 2x2 subplot layout: L1 F1-macro, L1 F1-weighted, L2 F1-macro, L2 F1-weighted
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Per-Class F1 Scores Across All Methods ({max_samples} samples)', fontsize=16, fontweight='bold')

        # Colors for different method types
        method_colors = {}
        color_palette = ['#2E7D32', '#1976D2', '#FF6F00', '#8E24AA', '#D32F2F', '#F57C00', '#388E3C', '#7B1FA2', '#C2185B']
        for i, method in enumerate(method_order):
            if method in all_results:
                method_colors[method] = color_palette[i % len(color_palette)]

        # Helper function to create per-class plot
        def plot_per_class_scores(ax, labels, level, metric_type):
            x = np.arange(len(labels))
            width = 0.8 / len([m for m in method_order if m in all_results])  # Dynamic bar width

            for i, method in enumerate(method_order):
                if method in all_results:
                    result = all_results[method]
                    metrics = result[f'metrics_{level}']

                    if metric_type == 'f1_macro':
                        # For F1-macro, use per_class_f1 scores
                        scores = []
                        per_class_f1 = metrics.get('per_class_f1', {})
                        for label in labels:
                            scores.append(per_class_f1.get(label, 0))
                    else:  # f1_weighted
                        # For F1-weighted, we need to calculate weighted scores per class
                        # Since we don't have per-class weighted F1, we'll use per_class_f1 as approximation
                        scores = []
                        per_class_f1 = metrics.get('per_class_f1', {})
                        for label in labels:
                            scores.append(per_class_f1.get(label, 0))

                    # Clean method name for legend
                    if method == '0_shot':
                        method_name = '0-shot (Prototypes)'
                    elif 'text' in method:
                        n = method.split('_')[0]
                        method_name = f'{n}-shot Text'
                    elif 'sensor' in method:
                        n = method.split('_')[0]
                        method_name = f'{n}-shot Sensor'
                    else:
                        method_name = method

                    bars = ax.bar(x + i * width, scores, width,
                                 label=method_name, color=method_colors[method], alpha=0.8)

            ax.set_xlabel('Activity Labels')
            ax.set_ylabel('F1 Score')
            ax.set_xticks(x + width * (len([m for m in method_order if m in all_results]) - 1) / 2)
            ax.set_xticklabels([label.replace('_', ' ') for label in labels], rotation=45, ha='right')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        # L1 F1-Macro (Top Left)
        if all_l1_labels:
            plot_per_class_scores(ax1, all_l1_labels[:10], 'l1', 'f1_macro')  # Limit to top 10 for readability
            ax1.set_title('L1 (Primary) Per-Class F1-Macro Scores', fontweight='bold')

        # L1 F1-Weighted approximation (Top Right)
        if all_l1_labels:
            plot_per_class_scores(ax2, all_l1_labels[:10], 'l1', 'f1_weighted')
            ax2.set_title('L1 (Primary) Per-Class F1 Scores (Approx. Weighted)', fontweight='bold')

        # L2 F1-Macro (Bottom Left)
        if all_l2_labels:
            plot_per_class_scores(ax3, all_l2_labels[:10], 'l2', 'f1_macro')
            ax3.set_title('L2 (Secondary) Per-Class F1-Macro Scores', fontweight='bold')

        # L2 F1-Weighted approximation (Bottom Right)
        if all_l2_labels:
            plot_per_class_scores(ax4, all_l2_labels[:10], 'l2', 'f1_weighted')
            ax4.set_title('L2 (Secondary) Per-Class F1 Scores (Approx. Weighted)', fontweight='bold')

        plt.tight_layout()

        # Save plot
        per_class_path = output_dir / f'nshot_per_class_f1.png'
        plt.savefig(per_class_path, dpi=300, bbox_inches='tight')
        print(f"💾 Per-class F1 plot saved: {per_class_path}")

        plt.close()

        # Create a separate heatmap visualization for better readability
        self.create_nshot_heatmap_plot(all_results, max_samples, output_dir)

    def create_nshot_heatmap_plot(self, all_results: Dict[str, Any], max_samples: int, output_dir: Path):
        """Create heatmap visualization of F1 scores across methods and classes."""

        print("🎨 Creating F1 score heatmaps...")

        # Extract n-shot values and method order
        n_shots = []
        for key in all_results.keys():
            if key.endswith('_shot_text'):
                n = int(key.split('_')[0])
                if n not in n_shots:
                    n_shots.append(n)
        n_shots.sort()

        method_order = ['0_shot']
        for n in n_shots:
            method_order.extend([f'{n}_shot_text', f'{n}_shot_sensor'])

        # Create heatmaps for L1 and L2
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        fig.suptitle(f'F1-Macro Heatmaps Across Methods and Classes ({max_samples} samples)', fontsize=16, fontweight='bold')

        for level, ax in [('l1', ax1), ('l2', ax2)]:
            # Get all labels for this level
            all_labels = set()
            for method in method_order:
                if method in all_results:
                    result = all_results[method]
                    if 'per_class_f1' in result[f'metrics_{level}']:
                        all_labels.update(result[f'metrics_{level}']['per_class_f1'].keys())

            all_labels = sorted(list(all_labels))[:15]  # Limit to top 15 for readability

            if not all_labels:
                continue

            # Create data matrix
            data_matrix = []
            method_names = []

            for method in method_order:
                if method in all_results:
                    result = all_results[method]
                    per_class_f1 = result[f'metrics_{level}'].get('per_class_f1', {})

                    row = [per_class_f1.get(label, 0) for label in all_labels]
                    data_matrix.append(row)

                    # Clean method name
                    if method == '0_shot':
                        method_names.append('0-shot\n(Prototypes)')
                    elif 'text' in method:
                        n = method.split('_')[0]
                        method_names.append(f'{n}-shot\n(Text)')
                    elif 'sensor' in method:
                        n = method.split('_')[0]
                        method_names.append(f'{n}-shot\n(Sensor)')

            # Create heatmap
            data_matrix = np.array(data_matrix)
            im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)

            # Determine font sizes based on matrix size
            num_labels = len(all_labels)
            num_methods = len(method_names)

            if num_labels <= 8 and num_methods <= 6:
                tick_fontsize = 9
                annot_fontsize = 8
            elif num_labels <= 12 and num_methods <= 8:
                tick_fontsize = 8
                annot_fontsize = 7
            else:
                tick_fontsize = 7
                annot_fontsize = 6

            # Clean and truncate labels for better display
            display_labels = []
            for label in all_labels:
                clean_label = label.replace('_', ' ')
                if len(clean_label) > 10:
                    clean_label = clean_label[:8] + '..'
                display_labels.append(clean_label)

            # Set ticks and labels
            ax.set_xticks(np.arange(len(all_labels)))
            ax.set_yticks(np.arange(len(method_names)))
            ax.set_xticklabels(display_labels, rotation=90, ha='center', fontsize=tick_fontsize)
            ax.set_yticklabels(method_names, fontsize=tick_fontsize)

            # Add text annotations with improved readability
            for i in range(len(method_names)):
                for j in range(len(all_labels)):
                    # Use different colors for better contrast
                    text_color = "white" if data_matrix[i, j] < 0.5 else "black"
                    text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                                 ha="center", va="center", color=text_color,
                                 fontsize=annot_fontsize, fontweight='bold')

            ax.set_title(f'{level.upper()} ({"Primary" if level == "l1" else "Secondary"}) F1-Macro Heatmap', fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('F1-Macro Score', rotation=270, labelpad=20)

        plt.tight_layout()

        # Save heatmap
        heatmap_path = output_dir / f'nshot_f1_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"💾 F1 heatmap saved: {heatmap_path}")

        plt.close()

    def create_tsne_scatter_plots(self, embeddings: np.ndarray,
                                 true_labels_l1: List[str], pred_labels_l1: List[str],
                                 true_labels_l2: List[str], pred_labels_l2: List[str],
                                 title: str, subtitle: str = "", save_path: str = None,
                                 max_samples_tsne: int = 5000, random_state: int = 42) -> plt.Figure:
        """Create t-SNE scatter plots comparing ground truth vs predicted labels for L1 and L2."""

        print(f"🔄 Creating t-SNE scatter plots: {title}")

        # Limit samples for t-SNE computation (it's computationally expensive)
        if len(embeddings) > max_samples_tsne:
            print(f"   Sampling {max_samples_tsne} from {len(embeddings)} samples for t-SNE")
            np.random.seed(random_state)
            indices = np.random.choice(len(embeddings), max_samples_tsne, replace=False)
            embeddings_tsne = embeddings[indices]
            true_l1_tsne = [true_labels_l1[i] for i in indices]
            pred_l1_tsne = [pred_labels_l1[i] for i in indices]
            true_l2_tsne = [true_labels_l2[i] for i in indices]
            pred_l2_tsne = [pred_labels_l2[i] for i in indices]
        else:
            embeddings_tsne = embeddings
            true_l1_tsne = true_labels_l1
            pred_l1_tsne = pred_labels_l1
            true_l2_tsne = true_labels_l2
            pred_l2_tsne = pred_labels_l2

        # Compute t-SNE
        print(f"   Computing t-SNE for {len(embeddings_tsne)} samples...")
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(embeddings_tsne)//4))
        embeddings_2d = tsne.fit_transform(embeddings_tsne)

        # Create 2x2 subplot layout with better spacing
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Set main title and subtitle with better positioning
        if subtitle:
            fig.suptitle(f't-SNE Visualization - {title}\n{subtitle}', fontsize=14, fontweight='bold', y=0.98)
        else:
            fig.suptitle(f't-SNE Visualization - {title}', fontsize=14, fontweight='bold', y=0.96)

        # Helper function to create scatter plot with proper legend handling
        def create_scatter_plot(ax, labels, title_text, level_color_base):
            # Get unique labels and assign colors
            unique_labels = sorted(list(set(labels)))

            # Limit to top 15 labels for readability
            if len(unique_labels) > 15:
                # Count label frequencies and keep top 15
                label_counts = Counter(labels)
                top_labels = [label for label, _ in label_counts.most_common(15)]

                # Map other labels to "Other"
                labels_filtered = []
                for label in labels:
                    if label in top_labels:
                        labels_filtered.append(label)
                    else:
                        labels_filtered.append("Other")
                labels = labels_filtered
                unique_labels = sorted(list(set(labels)))

            # Use colors from city metadata where available
            label_to_color = {}
            for i, label in enumerate(unique_labels):
                if level_color_base == 'blue':  # L1 labels
                    if hasattr(self, 'label_colors') and label in self.label_colors:
                        label_to_color[label] = self.label_colors[label]
                    else:
                        # Fallback to tab20 colormap for L1
                        label_to_color[label] = plt.cm.tab20(i % 20)
                else:  # L2 labels (orange)
                    if hasattr(self, 'label_colors_l2') and label in self.label_colors_l2:
                        label_to_color[label] = self.label_colors_l2[label]
                    else:
                        # Fallback to Set3 colormap for L2
                        label_to_color[label] = plt.cm.Set3(i % 12)

            # Create scatter plot
            for label in unique_labels:
                mask = np.array(labels) == label
                if np.any(mask):
                    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                             c=[label_to_color[label]], label=label.replace('_', ' '),
                             alpha=0.7, s=20)

            ax.set_title(title_text, fontweight='bold', fontsize=12)
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')  # Make subplot square-shaped

            # Return unique labels for shared legend
            return unique_labels

        # Create all scatter plots and collect unique labels
        l1_labels = create_scatter_plot(ax1, true_l1_tsne, 'L1 Primary - Ground Truth', 'blue')
        create_scatter_plot(ax2, pred_l1_tsne, 'L1 Primary - Predicted', 'blue')
        l2_labels = create_scatter_plot(ax3, true_l2_tsne, 'L2 Secondary - Ground Truth', 'orange')
        create_scatter_plot(ax4, pred_l2_tsne, 'L2 Secondary - Predicted', 'orange')

        # Create two separate legends positioned in the middle column between plots
        # L1 legend (for top plots)
        l1_handles, l1_legend_labels = ax1.get_legend_handles_labels()
        if l1_handles:
            l1_legend = fig.legend(l1_handles, l1_legend_labels, bbox_to_anchor=(0.5, 0.75),
                                  loc='center', fontsize=8, title='L1 Primary Activities',
                                  title_fontsize=9, ncol=1, frameon=True, fancybox=True,
                                  shadow=True, framealpha=0.9)

        # L2 legend (for bottom plots)
        l2_handles, l2_legend_labels = ax3.get_legend_handles_labels()
        if l2_handles:
            l2_legend = fig.legend(l2_handles, l2_legend_labels, bbox_to_anchor=(0.5, 0.25),
                                  loc='center', fontsize=8, title='L2 Secondary Activities',
                                  title_fontsize=9, ncol=1, frameon=True, fancybox=True,
                                  shadow=True, framealpha=0.9)

        # Adjust layout with more space in the middle for legends
        plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.08,
                          wspace=0.4, hspace=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 t-SNE scatter plots saved: {save_path}")

        return fig

    def run_dual_evaluation(self, max_samples: int = 10000,
                           train_split: str = 'train',
                           test_split: str = 'test',
                           k_neighbors: int = 1,
                           save_results: bool = True) -> Dict[str, Any]:
        """Run evaluation with both filtered and unfiltered data for comparison."""

        print("🚀 Starting dual embedding evaluation (filtered vs unfiltered)...")

        # Run evaluation with filtered labels
        print("\n" + "="*60)
        print("EVALUATION WITH FILTERED LABELS (no noisy labels)")
        print("="*60)
        results_filtered = self.run_evaluation(
            max_samples=max_samples,
            train_split=train_split,
            test_split=test_split,
            k_neighbors=k_neighbors,
            save_results=False,  # Don't save individual results yet
            filter_noisy_labels=True,
            compare_filtering=False
        )

        # Run evaluation with unfiltered labels
        print("\n" + "="*60)
        print("EVALUATION WITH UNFILTERED LABELS (including noisy labels)")
        print("="*60)
        results_unfiltered = self.run_evaluation(
            max_samples=max_samples,
            train_split=train_split,
            test_split=test_split,
            k_neighbors=k_neighbors,
            save_results=False,  # Don't save individual results yet
            filter_noisy_labels=False,
            compare_filtering=False
        )

        # Create combined visualizations
        if save_results:
            print("\n" + "="*60)
            print("CREATING COMBINED VISUALIZATIONS")
            print("="*60)

            output_dir = Path(self.config.get('output_dir', './embedding_evaluation'))
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract model name and test data name for chart subtitle
            model_name = Path(self.config['checkpoint_path']).parent.name if 'checkpoint_path' in self.config else 'unknown_model'
            test_data_name = Path(self.config['test_data_path']).stem if 'test_data_path' in self.config else 'unknown_data'

            # Create combined F1 scores plot
            self.create_f1_scores_plot(
                metrics_l1_filtered=results_filtered['metrics_l1'],
                metrics_l2_filtered=results_filtered['metrics_l2'],
                metrics_l1_unfiltered=results_unfiltered['metrics_l1'],
                metrics_l2_unfiltered=results_unfiltered['metrics_l2'],
                title=f'Filtered vs Unfiltered Comparison ({max_samples} samples)',
                subtitle=f'Model: {model_name} | Data: {test_data_name}',
                save_path=str(output_dir / f'f1_scores_comparison.png')
            )

            # Create combined confusion matrices plot
            self.create_combined_confusion_matrices(
                metrics_l1_filtered=results_filtered['metrics_l1'],
                metrics_l2_filtered=results_filtered['metrics_l2'],
                metrics_l1_unfiltered=results_unfiltered['metrics_l1'],
                metrics_l2_unfiltered=results_unfiltered['metrics_l2'],
                title=f'Filtered vs Unfiltered Comparison ({max_samples} samples)',
                subtitle=f'Model: {model_name} | Data: {test_data_name}',
                save_path=str(output_dir / f'confusion_matrices_comparison.png')
            )

            # Create t-SNE scatter plots for both filtered and unfiltered data
            print("\n🎨 Creating t-SNE visualizations...")

            # For filtered data
            self.create_tsne_scatter_plots(
                embeddings=results_filtered['test_embeddings'],
                true_labels_l1=results_filtered['ground_truth_l1'],
                pred_labels_l1=results_filtered['predictions_l1'],
                true_labels_l2=results_filtered['ground_truth_l2'],
                pred_labels_l2=results_filtered['predictions_l2'],
                title=f'Filtered Data ({max_samples} samples)',
                subtitle=f'Model: {model_name} | Data: {test_data_name} | Noisy labels removed',
                save_path=str(output_dir / f'tsne_scatter_filtered.png')
            )

            # For unfiltered data
            self.create_tsne_scatter_plots(
                embeddings=results_unfiltered['test_embeddings'],
                true_labels_l1=results_unfiltered['ground_truth_l1'],
                pred_labels_l1=results_unfiltered['predictions_l1'],
                true_labels_l2=results_unfiltered['ground_truth_l2'],
                pred_labels_l2=results_unfiltered['predictions_l2'],
                title=f'Unfiltered Data ({max_samples} samples)',
                subtitle=f'Model: {model_name} | Data: {test_data_name} | Including noisy labels',
                save_path=str(output_dir / f'tsne_scatter_unfiltered.png')
            )

            # Save combined results
            combined_results = {
                'filtered_results': results_filtered,
                'unfiltered_results': results_unfiltered,
                'comparison_summary': {
                    'filtered_l1_f1_macro': results_filtered['metrics_l1'].get('f1_macro', 0),
                    'unfiltered_l1_f1_macro': results_unfiltered['metrics_l1'].get('f1_macro', 0),
                    'filtered_l2_f1_macro': results_filtered['metrics_l2'].get('f1_macro', 0),
                    'unfiltered_l2_f1_macro': results_unfiltered['metrics_l2'].get('f1_macro', 0),
                }
            }

            results_file = output_dir / f'dual_evaluation_results.json'
            with open(results_file, 'w') as f:
                json.dump(combined_results, f, indent=2, default=str)
            print(f"💾 Combined results saved: {results_file}")

        return {
            'filtered': results_filtered,
            'unfiltered': results_unfiltered
        }

    def run_nshot_evaluation(self, max_samples: int = 10000,
                           train_split: str = 'train',
                           test_split: str = 'test',
                           n_shot_values: List[int] = [1, 2, 5],
                           k_neighbors: int = 1,
                           save_results: bool = True) -> Dict[str, Any]:
        """Run comprehensive n-shot evaluation including 0-shot (text prototypes) and n-shot (text and sensor)."""

        print("🚀 Starting comprehensive n-shot evaluation...")
        print(f"   N-shot values: {n_shot_values}")
        print(f"   Train split: {train_split}")
        print(f"   Test split: {test_split}")
        print(f"   Max samples: {max_samples}")

        all_results = {}

        # 1. Extract training and test data
        print("\n" + "="*60)
        print("1. EXTRACTING TRAINING AND TEST DATA")
        print("="*60)

        train_embeddings, train_labels_l1, train_labels_l2 = self.extract_embeddings_and_labels(
            train_split, max_samples
        )
        test_embeddings, test_labels_l1, test_labels_l2 = self.extract_embeddings_and_labels(
            test_split, max_samples
        )

        # Filter out noisy labels
        train_embeddings, train_labels_l1, train_labels_l2, train_original_indices = self.filter_noisy_labels(
            train_embeddings, train_labels_l1, train_labels_l2
        )
        test_embeddings, test_labels_l1, test_labels_l2, test_original_indices = self.filter_noisy_labels(
            test_embeddings, test_labels_l1, test_labels_l2
        )

        # 2. Run 0-shot evaluation (text prototypes)
        print("\n" + "="*60)
        print("2. RUNNING 0-SHOT EVALUATION (Text Prototypes)")
        print("="*60)

        prototypes_l1_0shot, counts_l1 = self.create_text_prototypes(train_labels_l1)
        prototypes_l2_0shot, counts_l2 = self.create_text_prototypes(train_labels_l2)

        pred_labels_l1_0shot = self.predict_labels_knn(test_embeddings, prototypes_l1_0shot, k_neighbors)
        pred_labels_l2_0shot = self.predict_labels_knn(test_embeddings, prototypes_l2_0shot, k_neighbors)

        metrics_l1_0shot = self.evaluate_predictions(test_labels_l1, pred_labels_l1_0shot, "L1 (0-shot)")
        metrics_l2_0shot = self.evaluate_predictions(test_labels_l2, pred_labels_l2_0shot, "L2 (0-shot)")

        all_results['0_shot'] = {
            'method': 'text_prototypes',
            'metrics_l1': metrics_l1_0shot,
            'metrics_l2': metrics_l2_0shot,
            'predictions_l1': pred_labels_l1_0shot,
            'predictions_l2': pred_labels_l2_0shot
        }

        # 3. Run n-shot evaluations
        for n_shots in n_shot_values:
            print(f"\n" + "="*60)
            print(f"3. RUNNING {n_shots}-SHOT EVALUATION")
            print("="*60)

            # Text-based n-shot
            print(f"\n--- {n_shots}-shot Text Embedding ---")
            prototypes_l1_text, prototypes_l2_text = self.create_nshot_text_prototypes(
                # test_embeddings, test_labels_l1, test_labels_l2, test_original_indices, n_shots
                train_embeddings, train_labels_l1, train_labels_l2, train_original_indices, n_shots
            )

            pred_labels_l1_text = self.predict_labels_knn(test_embeddings, prototypes_l1_text, k_neighbors)
            pred_labels_l2_text = self.predict_labels_knn(test_embeddings, prototypes_l2_text, k_neighbors)

            metrics_l1_text = self.evaluate_predictions(test_labels_l1, pred_labels_l1_text, f"L1 ({n_shots}-shot text)")
            metrics_l2_text = self.evaluate_predictions(test_labels_l2, pred_labels_l2_text, f"L2 ({n_shots}-shot text)")

            # Sensor-based n-shot
            print(f"\n--- {n_shots}-shot Sensor Embedding ---")
            prototypes_l1_sensor, prototypes_l2_sensor = self.create_nshot_sensor_prototypes(
                train_embeddings, train_labels_l1, train_labels_l2, n_shots
            )

            pred_labels_l1_sensor = self.predict_labels_knn(test_embeddings, prototypes_l1_sensor, k_neighbors)
            pred_labels_l2_sensor = self.predict_labels_knn(test_embeddings, prototypes_l2_sensor, k_neighbors)

            metrics_l1_sensor = self.evaluate_predictions(test_labels_l1, pred_labels_l1_sensor, f"L1 ({n_shots}-shot sensor)")
            metrics_l2_sensor = self.evaluate_predictions(test_labels_l2, pred_labels_l2_sensor, f"L2 ({n_shots}-shot sensor)")

            all_results[f'{n_shots}_shot_text'] = {
                'method': f'{n_shots}_shot_text',
                'metrics_l1': metrics_l1_text,
                'metrics_l2': metrics_l2_text,
                'predictions_l1': pred_labels_l1_text,
                'predictions_l2': pred_labels_l2_text
            }

            all_results[f'{n_shots}_shot_sensor'] = {
                'method': f'{n_shots}_shot_sensor',
                'metrics_l1': metrics_l1_sensor,
                'metrics_l2': metrics_l2_sensor,
                'predictions_l1': pred_labels_l1_sensor,
                'predictions_l2': pred_labels_l2_sensor
            }

        # 4. Create comprehensive visualizations and save results
        if save_results:
            print("\n" + "="*60)
            print("4. CREATING VISUALIZATIONS AND SAVING RESULTS")
            print("="*60)

            output_dir = Path(self.config.get('output_dir', './nshot_evaluation'))
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create comprehensive comparison plots
            self.create_nshot_comparison_plots(all_results, max_samples, output_dir)

            # Create per-class F1 plots
            self.create_nshot_per_class_plots(all_results, max_samples, output_dir)

            # Save detailed results
            results_summary = {
                'config': self.config,
                'evaluation_params': {
                    'max_samples': max_samples,
                    'train_split': train_split,
                    'test_split': test_split,
                    'n_shot_values': n_shot_values,
                    'k_neighbors': k_neighbors
                },
                'results': all_results
            }

            results_file = output_dir / f'nshot_evaluation_results.json'
            with open(results_file, 'w') as f:
                json.dump(results_summary, f, indent=2, default=str)
            print(f"💾 N-shot results saved: {results_file}")

            # Create text report
            self.create_nshot_text_report(all_results, output_dir, max_samples)

        # 5. Print final summary
        print("\n" + "="*60)
        print("🎯 N-SHOT EVALUATION SUMMARY")
        print("="*60)

        # Extract n-shot values dynamically from results for summary
        n_shots_summary = []
        for key in all_results.keys():
            if key.endswith('_shot_text'):
                n = int(key.split('_')[0])
                if n not in n_shots_summary:
                    n_shots_summary.append(n)
        n_shots_summary.sort()

        methods = ['0_shot'] + [f'{n}_shot_text' for n in n_shots_summary] + [f'{n}_shot_sensor' for n in n_shots_summary]

        print(f"{'Method':<20} {'L1 F1-Macro':<12} {'L1 F1-Wtd':<12} {'L1 Accuracy':<12} {'L2 F1-Macro':<12} {'L2 F1-Wtd':<12} {'L2 Accuracy':<12}")
        print("-" * 100)

        for method in methods:
            if method in all_results:
                result = all_results[method]
                l1_f1_macro = result['metrics_l1'].get('f1_macro', 0)
                l1_f1_weighted = result['metrics_l1'].get('f1_weighted', 0)
                l1_acc = result['metrics_l1'].get('accuracy', 0)
                l2_f1_macro = result['metrics_l2'].get('f1_macro', 0)
                l2_f1_weighted = result['metrics_l2'].get('f1_weighted', 0)
                l2_acc = result['metrics_l2'].get('accuracy', 0)

                print(f"{method:<20} {l1_f1_macro:<12.4f} {l1_f1_weighted:<12.4f} {l1_acc:<12.4f} {l2_f1_macro:<12.4f} {l2_f1_weighted:<12.4f} {l2_acc:<12.4f}")

        return all_results

    def run_evaluation(self, max_samples: int = 10000,
                      train_split: str = 'train',
                      test_split: str = 'test',
                      k_neighbors: int = 1,
                      save_results: bool = True,
                      filter_noisy_labels: bool = False,
                      compare_filtering: bool = True) -> Dict[str, Any]:
        """Run complete embedding evaluation pipeline."""

        print("🚀 Starting embedding evaluation...")
        print(f"   Train split: {train_split}")
        print(f"   Test split: {test_split}")
        print(f"   Max samples: {max_samples}")
        print(f"   K-neighbors: {k_neighbors}")

        results = {}

        # 1. Extract training labels to get unique activities
        print("\n" + "="*60)
        print("1. EXTRACTING TRAINING LABELS")
        print("="*60)

        # We only need labels to know which activities exist, not the embeddings
        train_embeddings, train_labels_l1, train_labels_l2 = self.extract_embeddings_and_labels(
            train_split, max_samples
        )

        # Filter out noisy labels from training data (if enabled)
        if filter_noisy_labels:
            _, train_labels_l1, train_labels_l2, _ = self.filter_noisy_labels(
                train_embeddings, train_labels_l1, train_labels_l2
            )

        # 2. Create text-based label prototypes
        print("\n" + "="*60)
        print("2. CREATING TEXT-BASED LABEL PROTOTYPES")
        print("="*60)

        prototypes_l1, counts_l1 = self.create_text_prototypes(train_labels_l1)
        prototypes_l2, counts_l2 = self.create_text_prototypes(train_labels_l2)

        results['prototypes_l1'] = prototypes_l1
        results['prototypes_l2'] = prototypes_l2
        results['prototype_counts_l1'] = counts_l1
        results['prototype_counts_l2'] = counts_l2

        # 3. Extract test embeddings
        print("\n" + "="*60)
        print("3. EXTRACTING TEST EMBEDDINGS")
        print("="*60)

        test_embeddings, test_labels_l1, test_labels_l2 = self.extract_embeddings_and_labels(
            test_split, max_samples
        )

        # Filter out noisy labels from test data (if enabled)
        if filter_noisy_labels:
            test_embeddings, test_labels_l1, test_labels_l2, _ = self.filter_noisy_labels(
                test_embeddings, test_labels_l1, test_labels_l2
            )

        # 4. Predict labels using nearest neighbors
        print("\n" + "="*60)
        print("4. PREDICTING LABELS")
        print("="*60)

        pred_labels_l1 = self.predict_labels_knn(test_embeddings, prototypes_l1, k_neighbors)
        pred_labels_l2 = self.predict_labels_knn(test_embeddings, prototypes_l2, k_neighbors)

        # 5. Evaluate predictions
        print("\n" + "="*60)
        print("5. EVALUATING PREDICTIONS")
        print("="*60)

        metrics_l1 = self.evaluate_predictions(test_labels_l1, pred_labels_l1, "L1 (Primary)")
        metrics_l2 = self.evaluate_predictions(test_labels_l2, pred_labels_l2, "L2 (Secondary)")

        results['metrics_l1'] = metrics_l1
        results['metrics_l2'] = metrics_l2
        results['predictions_l1'] = pred_labels_l1
        results['predictions_l2'] = pred_labels_l2
        results['ground_truth_l1'] = test_labels_l1
        results['ground_truth_l2'] = test_labels_l2
        results['test_embeddings'] = test_embeddings  # Store embeddings for t-SNE plotting

        # 6. Create visualizations
        if save_results and metrics_l1:
            print("\n" + "="*60)
            print("6. CREATING VISUALIZATIONS")
            print("="*60)

            output_dir = Path(self.config.get('output_dir', './embedding_evaluation'))
            output_dir.mkdir(parents=True, exist_ok=True)

            # Confusion matrices
            if 'confusion_matrix' in metrics_l1:
                self.create_confusion_matrix_plot(
                    metrics_l1['confusion_matrix'],
                    metrics_l1['unique_labels'],
                    'L1 Primary Activities',
                    str(output_dir / f'confusion_matrix_l1.png')
                )

            if 'confusion_matrix' in metrics_l2:
                self.create_confusion_matrix_plot(
                    metrics_l2['confusion_matrix'],
                    metrics_l2['unique_labels'],
                    'L2 Secondary Activities',
                    str(output_dir / f'confusion_matrix_l2.png')
                )

            # F1 Scores comprehensive plot
            if compare_filtering:
                # This would be called from run_dual_evaluation
                pass  # Plotting handled in dual evaluation
            else:
                # Single evaluation - just plot the current results
                self.create_f1_scores_plot(
                    metrics_l1_filtered=metrics_l1,
                    metrics_l2_filtered=metrics_l2,
                    title=f'Embedding Evaluation ({max_samples} samples)',
                    save_path=str(output_dir / f'f1_scores_analysis.png')
                )

                # Create t-SNE scatter plots for single evaluation
                filter_status = "filtered" if filter_noisy_labels else "unfiltered"
                model_name = Path(self.config['checkpoint_path']).parent.name if 'checkpoint_path' in self.config else 'unknown_model'
                test_data_name = Path(self.config['test_data_path']).stem if 'test_data_path' in self.config else 'unknown_data'

                self.create_tsne_scatter_plots(
                    embeddings=test_embeddings,
                    true_labels_l1=test_labels_l1,
                    pred_labels_l1=pred_labels_l1,
                    true_labels_l2=test_labels_l2,
                    pred_labels_l2=pred_labels_l2,
                    title=f'Embedding Evaluation ({max_samples} samples)',
                    subtitle=f'Model: {model_name} | Data: {test_data_name} | Labels: {filter_status}',
                    save_path=str(output_dir / f'tsne_scatter_{filter_status}.png')
                )

            # Save detailed results
            results_summary = {
                'config': self.config,
                'evaluation_params': {
                    'max_samples': max_samples,
                    'train_split': train_split,
                    'test_split': test_split,
                    'k_neighbors': k_neighbors
                },
                'metrics_summary': {
                    'l1_f1_macro': metrics_l1.get('f1_macro', 0),
                    'l1_f1_weighted': metrics_l1.get('f1_weighted', 0),
                    'l1_accuracy': metrics_l1.get('accuracy', 0),
                    'l1_classes': metrics_l1.get('num_classes', 0),
                    'l2_f1_macro': metrics_l2.get('f1_macro', 0),
                    'l2_f1_weighted': metrics_l2.get('f1_weighted', 0),
                    'l2_accuracy': metrics_l2.get('accuracy', 0),
                    'l2_classes': metrics_l2.get('num_classes', 0),
                },
                'detailed_metrics': {
                    'l1_per_class_f1': metrics_l1.get('per_class_f1', {}),
                    'l1_classification_report': metrics_l1.get('classification_report', {}),
                    'l1_unique_labels': metrics_l1.get('unique_labels', []),
                    'l2_per_class_f1': metrics_l2.get('per_class_f1', {}),
                    'l2_classification_report': metrics_l2.get('classification_report', {}),
                    'l2_unique_labels': metrics_l2.get('unique_labels', []),
                },
                'prototype_info': {
                    'l1_prototype_counts': results.get('prototype_counts_l1', {}),
                    'l2_prototype_counts': results.get('prototype_counts_l2', {}),
                }
            }

            # Save results
            results_file = output_dir / f'evaluation_results.json'
            with open(results_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = {}
                for k, v in results_summary.items():
                    if isinstance(v, np.ndarray):
                        json_results[k] = v.tolist()
                    else:
                        json_results[k] = v
                json.dump(json_results, f, indent=2, default=str)

            print(f"💾 Results saved: {results_file}")

        # 7. Print final summary
        print("\n" + "="*60)
        print("🎯 EVALUATION SUMMARY")
        print("="*60)

        if metrics_l1:
            print(f"📊 L1 (Primary Activities):")
            print(f"    F1-Score (Macro):    {metrics_l1['f1_macro']:.4f}")
            print(f"    F1-Score (Weighted): {metrics_l1['f1_weighted']:.4f}")
            print(f"    Accuracy:            {metrics_l1['accuracy']:.4f}")
            print(f"    Classes:             {metrics_l1['num_classes']}")
            print(f"    Test Samples:        {metrics_l1['num_samples']}")

        if metrics_l2:
            print(f"📊 L2 (Secondary Activities):")
            print(f"    F1-Score (Macro):    {metrics_l2['f1_macro']:.4f}")
            print(f"    F1-Score (Weighted): {metrics_l2['f1_weighted']:.4f}")
            print(f"    Accuracy:            {metrics_l2['accuracy']:.4f}")
            print(f"    Classes:             {metrics_l2['num_classes']}")
            print(f"    Test Samples:        {metrics_l2['num_samples']}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate embedding-based activity recognition')

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

    # Evaluation parameters
    parser.add_argument('--max_samples', type=int, default=10000,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--k_neighbors', type=int, default=1,
                       help='Number of neighbors for k-NN prediction')
    parser.add_argument('--output_dir', type=str, default='./embedding_evaluation',
                       help='Output directory for results')
    parser.add_argument('--filter_noisy_labels', action='store_true',
                       help='Filter out noisy labels like "Other" and "No_Activity"')
    parser.add_argument('--compare_filtering', action='store_true',
                       help='Compare filtered vs unfiltered results in a single chart')
    parser.add_argument('--run_nshot', action='store_true',
                       help='Run comprehensive n-shot evaluation (0-shot, 1-shot, 2-shot, 5-shot)')
    parser.add_argument('--n_shot_values', type=int, nargs='+', default=[1, 2, 5],
                       help='N-shot values to evaluate (default: 1 2 5)')

    args = parser.parse_args()

    # Configuration
    config = {
        'checkpoint_path': args.checkpoint,
        'train_data_path': args.train_data,
        'test_data_path': args.test_data,
        'vocab_path': args.vocab,
        'output_dir': args.output_dir,
    }

    # Run evaluation
    evaluator = EmbeddingEvaluator(config)

    if args.run_nshot:
        # Run comprehensive n-shot evaluation
        results = evaluator.run_nshot_evaluation(
            max_samples=args.max_samples,
            n_shot_values=args.n_shot_values,
            k_neighbors=args.k_neighbors,
            save_results=True
        )
        print(f"\n✅ N-shot evaluation complete! Results saved in: {args.output_dir}")
    elif args.compare_filtering:
        # Run dual evaluation (filtered vs unfiltered comparison)
        results = evaluator.run_dual_evaluation(
            max_samples=args.max_samples,
            k_neighbors=args.k_neighbors,
            save_results=True
        )
        print(f"\n✅ Dual evaluation complete! Comparison results saved in: {args.output_dir}")
    else:
        # Run single evaluation
        results = evaluator.run_evaluation(
            max_samples=args.max_samples,
            k_neighbors=args.k_neighbors,
            save_results=True,
            filter_noisy_labels=args.filter_noisy_labels
        )
        print(f"\n✅ Evaluation complete! Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
