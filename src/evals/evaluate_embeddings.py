#!/usr/bin/env python3

import sys
import os
# Add both src directory and project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
"""
Embedding-based evaluation script for CASAS activity recognition.
Computes nearest neighbor predictions in embedding space and evaluates against ground truth labels.

Supports three evaluation modes:
1. Sensor embedding evaluation (default): Evaluates sensor embeddings against text prototypes
2. Text embedding evaluation (--eval_text_embeddings): Evaluates text embeddings with/without projection
3. Comprehensive evaluation (--eval_all): All three in one - sensor + text (with/without projection)

Sample Usage (Comprehensive - RECOMMENDED):
python src/evals/evaluate_embeddings.py \
    --checkpoint trained_models/milan/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1/best_model.pt \
    --train_data data/processed/casas/milan/FD_60_p/train.json \
    --test_data data/processed/casas/milan/FD_60_p/test.json \
    --vocab data/processed/casas/milan/FD_60_p/vocab.json \
    --output_dir results/evals/milan/FD_60_p/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1 \
    --eval_all \
    --train_text_embeddings data/processed/casas/milan/FD_60_p/train_embeddings_baseline_clip.npz \
    --test_text_embeddings data/processed/casas/milan/FD_60_p/test_embeddings_baseline_clip.npz \
    --max_samples 10000 \
    --filter_noisy_labels \
    --description_style long_desc

Prototype Customization:
- --description_style: Specifies which description field to use from metadata for text prototypes
  Options: 'long_desc' (default), 'short_desc', 'zeroshot_har_desc', or any custom field
  Example: --description_style zeroshot_har_desc
- --use_multiple_prototypes: Create multiple prototypes per label (uses 'multiple_desc' field)

Output Files:
- combined_tsne_l1.png - 3-subplot t-SNE (text no proj | text proj | sensor)
- combined_confusion_matrix_l1.png - 3-subplot confusion (L1 labels)
- combined_confusion_matrix_l2.png - 3-subplot confusion (L2 labels)
- combined_f1_analysis.png - 2-subplot bar chart (L1 | L2)
- perclass_f1_weighted_l1.png - Per-class F1 weighted (L1)
- perclass_f1_weighted_l2.png - Per-class F1 weighted (L2)
- comprehensive_results.json - All metrics in JSON format
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import argparse
from typing import Dict, List, Any, Tuple, Union
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

# Retrieval metrics imports
from evals.compute_retrieval_metrics import (
    compute_label_recall_at_k,
    load_text_prototypes_from_metadata,
    encode_text_prototypes,
    compute_prototype_retrieval_metrics,
    compute_retrieval_confusion,
    compute_cosine_similarity
)


class EmbeddingEvaluator:
    """Evaluate embedding-based activity recognition using nearest neighbors."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_optimal_device()
        log_device_info(self.device)

        # Store description style (default to long_desc)
        self.description_style = config.get('description_style', 'long_desc')

        # Load models and data
        self._load_models()
        self._load_datasets()

        # Load label colors from metadata
        self._load_label_colors()

    def _load_models(self):
        """Load trained models from checkpoint."""
        print(f"🔄 Loading models from {self.config['checkpoint_path']}")

        checkpoint = torch.load(self.config['checkpoint_path'], map_location=self.device, weights_only=False)

        # Get config from checkpoint and handle dataclass/dict
        raw_config = checkpoint.get('config', {})
        # Handle both dict and dataclass config objects
        if hasattr(raw_config, '__dataclass_fields__'):
            # It's an AlignmentConfig dataclass - get encoder config from it
            model_config = getattr(raw_config, 'encoder', {}) or {}
        else:
            # It's a plain dict
            model_config = raw_config

        # Text encoder - use robust 3-tier fallback to detect correct encoder
        from evals.eval_utils import create_text_encoder_from_checkpoint
        self.text_encoder = create_text_encoder_from_checkpoint(
            checkpoint=checkpoint,
            device=self.device,
            data_path=self.config.get('train_data_path')  # For .npz fallback
        )

        if self.text_encoder is None:
            # Ultimate fallback: use default text encoder
            text_model_name = model_config.get('text_model_name', 'sentence-transformers/all-MiniLM-L6-v2')
            self.text_encoder = TextEncoder(text_model_name)
            print(f"⚠️  Using default text encoder: {text_model_name}")

        self.text_encoder.to(self.device)

        # Sensor encoder - check if it's ChronosEncoder or SensorEncoder
        self.vocab_sizes = checkpoint.get('vocab_sizes', {})

        # Check checkpoint format - AlignmentTrainer uses 'model_state_dict', AlignmentModel uses individual dicts
        if 'model_state_dict' in checkpoint:
            # Load from AlignmentModel - need to extract sensor_encoder weights
            from alignment.model import AlignmentModel
            full_model = AlignmentModel.load(
                self.config['checkpoint_path'],
                device=self.device,
                vocab_path=self.config.get('vocab_path')
            )
            self.sensor_encoder = full_model.sensor_encoder
        elif 'chronos_encoder_state_dict' in checkpoint or 'sensor_encoder_state_dict' in checkpoint:
            # Old format - individual state dicts
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
        else:
            raise ValueError("Checkpoint format not recognized - missing both 'model_state_dict' and 'sensor_encoder_state_dict'")

        self.sensor_encoder.to(self.device)
        self.sensor_encoder.eval()

        print("✅ Models loaded successfully")

    def _load_datasets(self):
        """Load datasets."""
        self.datasets = {}

        # Check if vocab_path is available - if not, we can't load datasets
        # (but this is OK for text embedding evaluation modes)
        if 'vocab_path' not in self.config or self.config.get('vocab_path') is None:
            print("ℹ️  No vocab file provided - skipping dataset loading")
            print("   (This is OK for text embedding evaluation modes)")
            return

        # Get sequence length from checkpoint config
        # This ensures we use the same sequence length as during training
        checkpoint = torch.load(self.config['checkpoint_path'], map_location=self.device, weights_only=False)
        raw_config = checkpoint.get('config', {})

        # Try to extract sequence_length from config
        if hasattr(raw_config, 'sequence_length'):
            sequence_length = raw_config.sequence_length
        elif hasattr(raw_config, 'data') and hasattr(raw_config.data, 'sequence_length'):
            sequence_length = raw_config.data.sequence_length
        elif isinstance(raw_config, dict) and 'sequence_length' in raw_config:
            sequence_length = raw_config['sequence_length']
        else:
            # Fallback to reasonable default for FD_60 data
            # Model supports up to max_seq_len=512, but 256 is good for 60-second windows
            sequence_length = 256
            print(f"⚠️  Could not find sequence_length in checkpoint, using default: {sequence_length}")
            print(f"    (Model supports up to 512, but 256 is reasonable for FD_60 data)")

        print(f"📏 Using sequence_length={sequence_length} from training config")

        if self.config['train_data_path'] and Path(self.config['train_data_path']).exists():
            self.datasets['train'] = SmartHomeDataset(
                data_path=self.config['train_data_path'],
                vocab_path=self.config['vocab_path'],
                sequence_length=sequence_length,
                max_captions=1
            )
            print(f"📊 Train dataset: {len(self.datasets['train'])} samples")

        if self.config['test_data_path'] and Path(self.config['test_data_path']).exists():
            self.datasets['test'] = SmartHomeDataset(
                data_path=self.config['test_data_path'],
                vocab_path=self.config['vocab_path'],
                sequence_length=sequence_length,
                max_captions=1
            )
            print(f"📊 Test dataset: {len(self.datasets['test'])} samples")

    def _load_label_colors(self):
        """Load label colors from city metadata."""
        try:
            metadata_path = Path(__file__).parent.parent.parent / "metadata" / "casas_metadata.json"
            with open(metadata_path, 'r') as f:
                city_metadata = json.load(f)

            # Detect which dataset we're using from the data path
            self.dataset_name = 'milan'  # default
            if 'train_data_path' in self.config:
                data_path = str(self.config['train_data_path'])
                for name in ['milan', 'cairo', 'aruba', 'tulum', 'kyoto', 'aware_home']:
                    if name in data_path.lower():
                        self.dataset_name = name
                        break

            dataset_metadata = city_metadata.get(self.dataset_name, {})

            # Load L1 colors - different keys for different datasets
            # Milan uses 'label', others use 'label_color'
            self.label_colors = dataset_metadata.get('label', dataset_metadata.get('label_color', {}))

            # Load L2 colors from label_deepcasas_color
            self.label_colors_l2 = dataset_metadata.get('label_deepcasas_color', {})

            if self.label_colors:
                print(f"🎨 Loaded {len(self.label_colors)} L1 label colors from {self.dataset_name} metadata")
            else:
                print(f"⚠️  No L1 label colors found in {self.dataset_name} metadata, using default colors")
                self.label_colors = {}

            if self.label_colors_l2:
                print(f"🎨 Loaded {len(self.label_colors_l2)} L2 label colors from {self.dataset_name} metadata")
            else:
                print(f"⚠️  No L2 label colors found in {self.dataset_name} metadata, using default colors")
                self.label_colors_l2 = {}

        except Exception as e:
            print(f"⚠️  Could not load label colors: {e}")
            self.label_colors = {}
            self.label_colors_l2 = {}

    def extract_embeddings_and_labels(self, split: str, max_samples: int = 10000) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
        """Extract embeddings, ground truth labels, and sample IDs from dataset.

        Returns:
            Tuple of (embeddings, labels_l1, labels_l2, sample_ids)
        """
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

        # Create data loader with NO SHUFFLING to maintain alignment
        data_loader = create_data_loader(
            dataset=dataset,
            text_encoder=self.text_encoder,
            span_masker=None,
            vocab_sizes=self.vocab_sizes,
            device=self.device,
            batch_size=256,  # Much larger batch size for inference (was 64)
            shuffle=False,  # NO SHUFFLING - maintain order for alignment
            num_workers=0,  # Must be 0 due to unpicklable objects
            apply_mlm=False
        )

        embeddings = []
        labels_l1 = []
        labels_l2 = []
        sample_ids = []
        samples_processed = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if samples_processed >= actual_samples:
                    break

                # Extract CLIP projected embeddings (512-dim)
                # Pack data for new encoder interface
                input_data = {
                    'categorical_features': batch['categorical_features'],
                    'coordinates': batch['coordinates'],
                    'time_deltas': batch['time_deltas']
                }
                sensor_emb = self.sensor_encoder.forward_clip(
                    input_data=input_data,
                    attention_mask=batch['mask']
                )

                embeddings.append(sensor_emb.cpu().numpy())

                # Extract labels and sample_ids for this batch
                batch_size_actual = sensor_emb.shape[0]

                # Get ground truth labels directly from the batch (much more reliable!)
                batch_labels_l1 = batch.get('activity_labels', ['Unknown'] * batch_size_actual)
                batch_labels_l2 = batch.get('activity_labels_l2', ['Unknown'] * batch_size_actual)
                batch_sample_ids = batch.get('sample_ids', [f'unknown_{i}' for i in range(batch_size_actual)])

                for i in range(batch_size_actual):
                    if samples_processed >= actual_samples:
                        break

                    # Use labels directly from the batch - this is the correct approach!
                    label_l1 = batch_labels_l1[i] if i < len(batch_labels_l1) else 'Unknown'
                    label_l2 = batch_labels_l2[i] if i < len(batch_labels_l2) else 'Unknown'
                    sample_id = batch_sample_ids[i] if i < len(batch_sample_ids) else f'unknown_{samples_processed}'

                    labels_l1.append(label_l1)
                    labels_l2.append(label_l2)
                    sample_ids.append(str(sample_id))

                    samples_processed += 1

                if batch_idx % 20 == 0:
                    print(f"  Processed {samples_processed}/{actual_samples} samples...")

        # Concatenate embeddings
        embeddings = np.vstack(embeddings)[:actual_samples]

        print(f"📈 Extracted {embeddings.shape[0]} embeddings")
        print(f"📊 L1 Labels: {len(set(labels_l1))} unique ({Counter(labels_l1).most_common(5)})")
        print(f"📊 L2 Labels: {len(set(labels_l2))} unique ({Counter(labels_l2).most_common(5)})")
        print(f"🔑 Sample IDs: {len(sample_ids)} unique sample IDs extracted")

        return embeddings, labels_l1, labels_l2, sample_ids

    def align_embeddings_by_sample_id(
        self,
        sensor_emb: np.ndarray,
        sensor_labels_l1: List[str],
        sensor_labels_l2: List[str],
        sensor_sample_ids: List[str],
        text_emb: np.ndarray,
        text_labels_l1: List[str],
        text_labels_l2: List[str],
        text_sample_ids: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
        """Align sensor and text embeddings by sample_id.

        Args:
            sensor_emb: Sensor embeddings (N_sensor, D)
            sensor_labels_l1: L1 labels for sensor embeddings
            sensor_labels_l2: L2 labels for sensor embeddings
            sensor_sample_ids: Sample IDs for sensor embeddings
            text_emb: Text embeddings (N_text, D)
            text_labels_l1: L1 labels for text embeddings
            text_labels_l2: L2 labels for text embeddings
            text_sample_ids: Sample IDs for text embeddings

        Returns:
            Tuple of (aligned_sensor_emb, aligned_text_emb, aligned_labels_l1, aligned_labels_l2, aligned_sample_ids)
        """
        print(f"\n🔗 Aligning embeddings by sample_id...")
        print(f"   Sensor: {len(sensor_sample_ids)} samples")
        print(f"   Text: {len(text_sample_ids)} samples")

        # Create dictionaries for fast lookup
        sensor_dict = {}
        for i, sample_id in enumerate(sensor_sample_ids):
            sensor_dict[sample_id] = {
                'emb': sensor_emb[i],
                'l1': sensor_labels_l1[i],
                'l2': sensor_labels_l2[i]
            }

        text_dict = {}
        for i, sample_id in enumerate(text_sample_ids):
            text_dict[sample_id] = {
                'emb': text_emb[i],
                'l1': text_labels_l1[i],
                'l2': text_labels_l2[i]
            }

        # Find common sample_ids
        sensor_ids_set = set(sensor_sample_ids)
        text_ids_set = set(text_sample_ids)
        common_ids = sensor_ids_set & text_ids_set

        print(f"   Common sample_ids: {len(common_ids)}")
        print(f"   Sensor-only: {len(sensor_ids_set - text_ids_set)}")
        print(f"   Text-only: {len(text_ids_set - sensor_ids_set)}")

        if len(common_ids) == 0:
            raise ValueError("No common sample_ids found! Cannot align embeddings.")

        # Create aligned arrays
        aligned_sensor_emb = []
        aligned_text_emb = []
        aligned_labels_l1 = []
        aligned_labels_l2 = []
        aligned_sample_ids = []

        for sample_id in sorted(common_ids):
            aligned_sensor_emb.append(sensor_dict[sample_id]['emb'])
            aligned_text_emb.append(text_dict[sample_id]['emb'])
            # Use sensor labels as ground truth (they should match text labels)
            aligned_labels_l1.append(sensor_dict[sample_id]['l1'])
            aligned_labels_l2.append(sensor_dict[sample_id]['l2'])
            aligned_sample_ids.append(sample_id)

        aligned_sensor_emb = np.array(aligned_sensor_emb)
        aligned_text_emb = np.array(aligned_text_emb)

        print(f"✅ Aligned {len(aligned_sample_ids)} samples")
        print(f"   Final L1 labels: {len(set(aligned_labels_l1))} unique")
        print(f"   Final L2 labels: {len(set(aligned_labels_l2))} unique")

        return aligned_sensor_emb, aligned_text_emb, aligned_labels_l1, aligned_labels_l2, aligned_sample_ids

    def filter_noisy_labels(self, embeddings: np.ndarray, labels_l1: List[str], labels_l2: List[str],
                           original_indices: List[int] = None) -> Tuple[np.ndarray, List[str], List[str], List[int]]:
        """Filter out noisy/uninformative labels like 'Other', 'No_Activity', etc."""

        # Define labels to exclude (case-insensitive)
        exclude_labels = {
            'other',
            'no_activity', 'No_Activity',
            'unknown', 'none', 'null', 'nan',
            'no activity', 'other activity', 'miscellaneous', 'misc'
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

    def get_labels_from_metadata(self, dataset_name: str) -> Tuple[List[str], List[str]]:
        """
        Extract all L1 and L2 labels from metadata file without loading training data.

        Args:
            dataset_name: Name of dataset (e.g., 'milan', 'aruba')

        Returns:
            Tuple of (l1_labels, l2_labels)
        """
        import json
        from pathlib import Path

        metadata_path = Path(__file__).parent.parent.parent / 'metadata' / 'casas_metadata.json'

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        if dataset_name not in metadata:
            raise ValueError(f"Dataset {dataset_name} not found in metadata. Available: {list(metadata.keys())}")

        dataset_meta = metadata[dataset_name]

        # Get L1 labels from label_l2 mapping keys
        if 'label_l2' in dataset_meta:
            l1_labels = list(dataset_meta['label_l2'].keys())
            # Get unique L2 labels from values
            l2_labels = list(set(dataset_meta['label_l2'].values()))
        else:
            # Fallback to label_to_text if label_l2 not available
            if 'label_to_text' in dataset_meta:
                l1_labels = list(dataset_meta['label_to_text'].keys())
                l2_labels = []
            else:
                raise ValueError(f"No label information found for dataset {dataset_name}")

        # Filter out empty strings
        l1_labels = [label for label in l1_labels if label and label.strip()]
        l2_labels = [label for label in l2_labels if label and label.strip()]

        print(f"📋 Extracted from metadata: {len(l1_labels)} L1 labels, {len(l2_labels)} L2 labels")

        return l1_labels, l2_labels

    def map_l1_to_l2_labels(self, l1_labels: List[str], house_name: str = "milan") -> List[str]:
        """Map L1 (primary) labels to L2 (secondary) labels using metadata mapping.

        Args:
            l1_labels: List of L1 labels to map
            house_name: Dataset name (e.g., 'milan', 'aruba')

        Returns:
            List of L2 labels corresponding to input L1 labels
        """
        try:
            metadata_path = Path(__file__).parent.parent.parent / "metadata" / "casas_metadata.json"
            with open(metadata_path, 'r') as f:
                city_metadata = json.load(f)

            dataset_metadata = city_metadata.get(house_name, {})
            label_l2_mapping = dataset_metadata.get('label_l2', {})

            # Also try label_deepcasas as alternative
            if not label_l2_mapping:
                label_l2_mapping = dataset_metadata.get('label_deepcasas', {})

            # Map each L1 label to L2
            l2_labels = []
            for l1_label in l1_labels:
                # Try direct mapping
                l2_label = label_l2_mapping.get(l1_label, l1_label)
                l2_labels.append(l2_label)

            return l2_labels
        except Exception as e:
            print(f"⚠️  Could not load L1->L2 mapping from metadata: {e}")
            # Return L1 labels as fallback
            return l1_labels

    def load_multiple_descriptions_from_metadata(self, labels: List[str], house_name: str = "milan") -> Dict[str, List[str]]:
        """Load multiple descriptions per label from metadata's multiple_desc field.

        Only includes labels that have multiple_desc field. Raises error if fewer than 5 labels found.

        Returns:
            Dict mapping label -> list of descriptions (only for labels with multiple_desc)

        Raises:
            ValueError: If fewer than 5 labels have multiple_desc field
        """
        try:
            metadata_path = Path(__file__).parent.parent.parent / "metadata" / "casas_metadata.json"
            with open(metadata_path, 'r') as f:
                city_metadata = json.load(f)

            dataset_metadata = city_metadata.get(house_name, {})
            label_to_text = dataset_metadata.get('label_to_text', {})

            label_descriptions = {}
            labels_with_multiple_desc = []
            labels_without_multiple_desc = []

            for label in labels:
                # Only use labels that have multiple_desc field
                if label in label_to_text and 'multiple_desc' in label_to_text[label]:
                    label_descriptions[label] = label_to_text[label]['multiple_desc']
                    labels_with_multiple_desc.append(label)
                else:
                    labels_without_multiple_desc.append(label)

            # Print statistics
            print(f"\n📊 Multiple descriptions availability for {house_name}:")
            print(f"   ✅ Labels WITH multiple_desc: {len(labels_with_multiple_desc)}")
            if labels_with_multiple_desc:
                print(f"   Labels: {', '.join(labels_with_multiple_desc)}")
                # Print number of descriptions per label
                for label in labels_with_multiple_desc:
                    n_desc = len(label_descriptions[label])
                    print(f"      - {label}: {n_desc} descriptions")

            if labels_without_multiple_desc:
                print(f"   ❌ Labels WITHOUT multiple_desc: {len(labels_without_multiple_desc)}")
                print(f"   Labels: {', '.join(labels_without_multiple_desc)}")

            # Enforce minimum requirement
            if len(labels_with_multiple_desc) < 5:
                raise ValueError(
                    f"Insufficient labels with multiple_desc field!\n"
                    f"   Found: {len(labels_with_multiple_desc)} labels\n"
                    f"   Required: at least 5 labels\n"
                    f"   Labels with multiple_desc: {labels_with_multiple_desc}\n"
                    f"   Dataset: {house_name}\n"
                    f"   Please ensure the metadata has multiple_desc field for at least 5 labels."
                )

            return label_descriptions

        except ValueError:
            # Re-raise ValueError (our custom error)
            raise
        except Exception as e:
            raise RuntimeError(f"Could not load descriptions from metadata: {e}")

    def create_text_prototypes(self, labels: List[str], apply_projection: bool = True, use_multiple_prototypes: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """Create text-based prototypes using the text encoder with multiple captions per label.

        Args:
            labels: List of label strings
            apply_projection: If True, uses encode_texts_clip (with projection).
                            If False, encodes without projection.
            use_multiple_prototypes: If True, create multiple prototypes per label (one per description).
                                   If False, average descriptions into single prototype per label (default).

        Returns:
            If use_multiple_prototypes=False: Dict[label -> single averaged embedding]
            If use_multiple_prototypes=True: Dict[label -> array of multiple embeddings, shape (n_descriptions, embedding_dim)]
        """

        print(f"🔄 Creating text-based label prototypes (projection={'ON' if apply_projection else 'OFF'}, multiple_prototypes={use_multiple_prototypes})...")

        # Get unique labels and their counts
        label_counts = Counter(labels)
        unique_labels = list(label_counts.keys())

        print(f"📝 Encoding {len(unique_labels)} unique labels with text encoder...")

        # Get descriptions based on mode
        if use_multiple_prototypes:
            # Load from metadata's multiple_desc field (only includes labels with multiple_desc)
            label_descriptions_dict = self.load_multiple_descriptions_from_metadata(unique_labels, self.dataset_name)
            # Filter unique_labels to only those with multiple_desc
            unique_labels = [label for label in unique_labels if label in label_descriptions_dict]
            print(f"   📋 Using {len(unique_labels)} labels with multiple_desc field")
        else:
            # Use existing convert_labels_to_text function
            label_descriptions_lists = convert_labels_to_text(unique_labels, house_name=self.dataset_name, description_style=self.description_style)
            label_descriptions_dict = {label: label_descriptions_lists[i] for i, label in enumerate(unique_labels)}

        # Create prototypes dictionary
        prototypes = {}
        total_prototypes = 0
        self.text_encoder.eval()

        with torch.no_grad():
            for label in unique_labels:
                descriptions = label_descriptions_dict[label]

                if apply_projection:
                    # Use encode_texts_clip which includes projection
                    caption_embeddings = self.text_encoder.encode_texts_clip(descriptions, self.device).cpu().numpy()
                else:
                    # Encode WITHOUT projection (raw CLIP embeddings)
                    # Load CLIP model if needed
                    if not hasattr(self.text_encoder, '_clip_model') or self.text_encoder._clip_model is None:
                        from transformers import CLIPTextModel, CLIPTokenizer
                        self.text_encoder._clip_model = CLIPTextModel.from_pretrained(self.text_encoder.model_name).to(self.device)
                        self.text_encoder._clip_tokenizer = CLIPTokenizer.from_pretrained(self.text_encoder.model_name)
                        self.text_encoder._clip_model.eval()

                    inputs = self.text_encoder._clip_tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    outputs = self.text_encoder._clip_model(**inputs)
                    embeddings = outputs.pooler_output
                    embeddings = F.normalize(embeddings, p=2, dim=-1)
                    caption_embeddings = embeddings.cpu().numpy()

                if use_multiple_prototypes:
                    # Keep all embeddings separate (shape: n_descriptions, embedding_dim)
                    prototypes[label] = caption_embeddings
                    total_prototypes += len(caption_embeddings)
                else:
                    # Average the embeddings to create a single prototype (original behavior)
                    prototype_embedding = np.mean(caption_embeddings, axis=0)
                    prototypes[label] = prototype_embedding
                    total_prototypes += 1

        if use_multiple_prototypes:
            print(f"✅ Created {total_prototypes} text-based prototypes across {len(prototypes)} labels")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                if label in prototypes:
                    n_protos = len(prototypes[label])
                    print(f"    {label}: {count} samples → {n_protos} prototypes")
        else:
            print(f"✅ Created {len(prototypes)} text-based prototypes")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                if label in label_descriptions_dict:
                    descriptions = label_descriptions_dict[label]
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
                    descriptions = convert_labels_to_text([label], house_name=self.dataset_name, description_style=self.description_style)[0]
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
                          k: int = 1,
                          use_multiple_prototypes: bool = False) -> List[str]:
        """Predict labels using k-nearest neighbors between sensor and text embeddings.

        Args:
            query_embeddings: Query embeddings to classify (n_queries, embedding_dim)
            prototypes: Dict mapping label -> prototype embedding(s)
                       If use_multiple_prototypes=False: each value is (embedding_dim,)
                       If use_multiple_prototypes=True: each value is (n_prototypes, embedding_dim)
            k: Number of nearest neighbors to consider
            use_multiple_prototypes: Whether prototypes dict contains multiple embeddings per label
        """

        print(f"🔄 Predicting labels using {k}-NN cross-modal comparison...")
        print(f"    Query embeddings: {query_embeddings.shape[0]} samples")

        if use_multiple_prototypes:
            # Flatten all prototypes and keep track of which label each belongs to
            all_prototype_embeddings = []
            all_prototype_labels = []

            for label, label_prototypes in prototypes.items():
                # label_prototypes shape: (n_prototypes, embedding_dim)
                for proto_emb in label_prototypes:
                    all_prototype_embeddings.append(proto_emb)
                    all_prototype_labels.append(label)

            prototype_embeddings = np.array(all_prototype_embeddings)
            prototype_labels = all_prototype_labels

            print(f"    Total prototypes: {len(prototype_labels)} across {len(prototypes)} labels")
        else:
            # Original behavior: one prototype per label
            prototype_labels = list(prototypes.keys())
            prototype_embeddings = np.array([prototypes[label] for label in prototype_labels])
            print(f"    Text prototypes: {len(prototypes)} activities")

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
                # Majority vote
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
                             alpha=0.6, s=20, edgecolors='white', linewidth=0.3)

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

        train_embeddings, train_labels_l1, train_labels_l2, train_sample_ids = self.extract_embeddings_and_labels(
            train_split, max_samples
        )
        test_embeddings, test_labels_l1, test_labels_l2, test_sample_ids = self.extract_embeddings_and_labels(
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

    def load_text_embeddings_from_file(self, embeddings_path: str,
                                       data_path: str,
                                       max_samples: int = None) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
        """Load text embeddings from .npz file and match with labels.

        For multi-caption embeddings (new format), uses only the FIRST caption per sample
        for consistent evaluation metrics.

        Returns:
            Tuple of (embeddings, sample_ids, labels_l1, labels_l2)
        """
        print(f"\n📖 Loading text embeddings from: {embeddings_path}")
        data = np.load(embeddings_path)
        embeddings = data['embeddings']
        sample_ids_from_emb = data['sample_ids']

        # Check if this is multi-caption format
        if 'caption_indices' in data:
            caption_indices = data['caption_indices']
            # Keep only first caption (caption_indices == 0) for evaluation
            first_caption_mask = caption_indices == 0
            embeddings = embeddings[first_caption_mask]
            sample_ids_from_emb = sample_ids_from_emb[first_caption_mask]
            print(f"   Multi-caption format detected: using first caption per sample for evaluation")
            print(f"   Filtered to {embeddings.shape[0]} unique samples (from {len(caption_indices)} total embeddings)")

        print(f"   Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
        if 'encoder_type' in data:
            print(f"   Encoder: {data.get('encoder_type', ['unknown'])[0]}")
        if 'model_name' in data:
            print(f"   Model: {data.get('model_name', ['unknown'])[0]}")

        # Load labels from data file
        print(f"\n📖 Loading labels from: {data_path}")
        with open(data_path, 'r') as f:
            data_json = json.load(f)

        samples_data = data_json.get('samples', data_json)

        label_map = {}
        for sample in samples_data:
            sample_id = sample.get('sample_id')
            if sample_id:
                metadata = sample.get('metadata', {})
                ground_truth = metadata.get('ground_truth_labels', {})
                label_l1 = ground_truth.get('primary_l1', ground_truth.get('mode', 'Unknown'))
                label_l2 = ground_truth.get('primary_l2', 'Unknown')
                label_map[sample_id] = {'label_l1': label_l1, 'label_l2': label_l2}

        print(f"   Loaded labels for {len(label_map)} samples")

        # Match embeddings with labels
        labels_l1 = []
        labels_l2 = []
        sample_ids = []
        valid_indices = []

        for i, sample_id in enumerate(sample_ids_from_emb):
            sample_id_str = str(sample_id)
            if sample_id_str in label_map:
                labels_l1.append(label_map[sample_id_str]['label_l1'])
                labels_l2.append(label_map[sample_id_str]['label_l2'])
                sample_ids.append(sample_id_str)
                valid_indices.append(i)

        embeddings = embeddings[valid_indices]
        print(f"   Matched {len(labels_l1)} samples with labels")

        # Sample if needed
        if max_samples and len(embeddings) > max_samples:
            print(f"\n🎲 Sampling {max_samples} from {len(embeddings)} samples (seed=42)")
            np.random.seed(42)
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings = embeddings[indices]
            sample_ids = [sample_ids[i] for i in indices]
            labels_l1 = [labels_l1[i] for i in indices]
            labels_l2 = [labels_l2[i] for i in indices]

        return embeddings, sample_ids, labels_l1, labels_l2

    def apply_projection_to_embeddings(self, embeddings: np.ndarray,
                                       normalize: bool = True) -> np.ndarray:
        """Apply the projection head from the checkpoint to embeddings.

        Args:
            embeddings: Input embeddings (numpy array)
            normalize: Whether to normalize after projection

        Returns:
            Projected embeddings (numpy array)
        """
        print(f"🔄 Applying projection head to {len(embeddings)} embeddings...")

        # Convert to torch tensor
        embeddings_torch = torch.from_numpy(embeddings).float().to(self.device)

        # Apply projection using the text encoder's projection head
        with torch.no_grad():
            if normalize:
                embeddings_torch = F.normalize(embeddings_torch, p=2, dim=-1)

            # Apply the clip_proj from text_encoder
            if hasattr(self.text_encoder, 'clip_proj'):
                projected = self.text_encoder.clip_proj(embeddings_torch)
            else:
                print("   ⚠️  No projection head found, returning original embeddings")
                projected = embeddings_torch

            if normalize:
                projected = F.normalize(projected, p=2, dim=-1)

        # Convert back to numpy
        projected_np = projected.cpu().numpy()
        print(f"   Projected: {embeddings.shape[1]}-dim → {projected_np.shape[1]}-dim")

        return projected_np

    def run_text_embedding_evaluation(self,
                                     train_embeddings_path: str,
                                     test_embeddings_path: str,
                                     train_data_path: str,
                                     test_data_path: str,
                                     max_samples: int = 10000,
                                     k_neighbors: int = 1,
                                     filter_noisy_labels: bool = False,
                                     save_results: bool = True) -> Dict[str, Any]:
        """Evaluate text embeddings with and without projection.

        This loads pre-computed text embeddings and evaluates them against
        label prototypes, both in the original space and after applying
        the projection head from the trained model.

        Args:
            train_embeddings_path: Path to training text embeddings .npz file
            test_embeddings_path: Path to test text embeddings .npz file
            train_data_path: Path to training data JSON file (for labels)
            test_data_path: Path to test data JSON file (for labels)
        """
        print("\n" + "="*80)
        print("TEXT EMBEDDING EVALUATION (WITH AND WITHOUT PROJECTION)")
        print("="*80)

        # Load train embeddings to get label distribution
        print("\n" + "="*60)
        print("1. LOADING TRAINING TEXT EMBEDDINGS")
        print("="*60)

        train_embeddings, train_sample_ids, train_labels_l1, train_labels_l2 = \
            self.load_text_embeddings_from_file(train_embeddings_path, train_data_path, max_samples)

        # Filter noisy labels if requested
        if filter_noisy_labels:
            train_embeddings, train_labels_l1, train_labels_l2, _ = \
                self.filter_noisy_labels(train_embeddings, train_labels_l1, train_labels_l2)

        # Load test embeddings
        print("\n" + "="*60)
        print("2. LOADING TEST TEXT EMBEDDINGS")
        print("="*60)

        test_embeddings, test_sample_ids, test_labels_l1, test_labels_l2 = \
            self.load_text_embeddings_from_file(test_embeddings_path, test_data_path, max_samples)

        # Filter noisy labels if requested
        if filter_noisy_labels:
            test_embeddings, test_labels_l1, test_labels_l2, _ = \
                self.filter_noisy_labels(test_embeddings, test_labels_l1, test_labels_l2)

        # ===== EVALUATION WITHOUT PROJECTION =====
        print("\n" + "="*60)
        print("3. EVALUATION WITHOUT PROJECTION (Original Space)")
        print("="*60)

        # Create prototypes from training embeddings (without projection)
        print("🔄 Creating prototypes from training text embeddings (no projection)...")
        prototypes_l1_orig = self.create_prototypes_from_embeddings(train_embeddings, train_labels_l1)
        prototypes_l2_orig = self.create_prototypes_from_embeddings(train_embeddings, train_labels_l2)

        # Predict using original embeddings
        pred_labels_l1_orig = self.predict_labels_knn(test_embeddings, prototypes_l1_orig, k_neighbors)
        pred_labels_l2_orig = self.predict_labels_knn(test_embeddings, prototypes_l2_orig, k_neighbors)

        # Evaluate
        metrics_l1_orig = self.evaluate_predictions(test_labels_l1, pred_labels_l1_orig, "L1 (No Projection)")
        metrics_l2_orig = self.evaluate_predictions(test_labels_l2, pred_labels_l2_orig, "L2 (No Projection)")

        # ===== EVALUATION WITH PROJECTION =====
        print("\n" + "="*60)
        print("4. EVALUATION WITH PROJECTION (Aligned Space)")
        print("="*60)

        # Apply projection to train and test embeddings
        train_embeddings_proj = self.apply_projection_to_embeddings(train_embeddings)
        test_embeddings_proj = self.apply_projection_to_embeddings(test_embeddings)

        # Create prototypes from projected training embeddings
        print("🔄 Creating prototypes from projected text embeddings...")
        prototypes_l1_proj = self.create_prototypes_from_embeddings(train_embeddings_proj, train_labels_l1)
        prototypes_l2_proj = self.create_prototypes_from_embeddings(train_embeddings_proj, train_labels_l2)

        # Predict using projected embeddings
        pred_labels_l1_proj = self.predict_labels_knn(test_embeddings_proj, prototypes_l1_proj, k_neighbors)
        pred_labels_l2_proj = self.predict_labels_knn(test_embeddings_proj, prototypes_l2_proj, k_neighbors)

        # Evaluate
        metrics_l1_proj = self.evaluate_predictions(test_labels_l1, pred_labels_l1_proj, "L1 (With Projection)")
        metrics_l2_proj = self.evaluate_predictions(test_labels_l2, pred_labels_l2_proj, "L2 (With Projection)")

        # ===== CREATE VISUALIZATIONS =====
        if save_results:
            print("\n" + "="*60)
            print("5. CREATING VISUALIZATIONS")
            print("="*60)

            output_dir = Path(self.config.get('output_dir', './text_embedding_evaluation'))
            output_dir.mkdir(parents=True, exist_ok=True)

            # Confusion matrices - without projection
            if 'confusion_matrix' in metrics_l1_orig:
                self.create_confusion_matrix_plot(
                    metrics_l1_orig['confusion_matrix'],
                    metrics_l1_orig['unique_labels'],
                    'L1 Primary Activities (No Projection)',
                    str(output_dir / 'text_confusion_matrix_l1_noproj.png')
                )

            if 'confusion_matrix' in metrics_l2_orig:
                self.create_confusion_matrix_plot(
                    metrics_l2_orig['confusion_matrix'],
                    metrics_l2_orig['unique_labels'],
                    'L2 Secondary Activities (No Projection)',
                    str(output_dir / 'text_confusion_matrix_l2_noproj.png')
                )

            # Confusion matrices - with projection
            if 'confusion_matrix' in metrics_l1_proj:
                self.create_confusion_matrix_plot(
                    metrics_l1_proj['confusion_matrix'],
                    metrics_l1_proj['unique_labels'],
                    'L1 Primary Activities (With Projection)',
                    str(output_dir / 'text_confusion_matrix_l1_proj.png')
                )

            if 'confusion_matrix' in metrics_l2_proj:
                self.create_confusion_matrix_plot(
                    metrics_l2_proj['confusion_matrix'],
                    metrics_l2_proj['unique_labels'],
                    'L2 Secondary Activities (With Projection)',
                    str(output_dir / 'text_confusion_matrix_l2_proj.png')
                )

            # t-SNE plots - without projection
            self.create_tsne_scatter_plots(
                embeddings=test_embeddings,
                true_labels_l1=test_labels_l1,
                pred_labels_l1=pred_labels_l1_orig,
                true_labels_l2=test_labels_l2,
                pred_labels_l2=pred_labels_l2_orig,
                title='Text Embeddings Evaluation (No Projection)',
                subtitle=f'Original embedding space | {len(test_embeddings)} test samples',
                save_path=str(output_dir / 'text_tsne_noproj.png')
            )

            # t-SNE plots - with projection
            self.create_tsne_scatter_plots(
                embeddings=test_embeddings_proj,
                true_labels_l1=test_labels_l1,
                pred_labels_l1=pred_labels_l1_proj,
                true_labels_l2=test_labels_l2,
                pred_labels_l2=pred_labels_l2_proj,
                title='Text Embeddings Evaluation (With Projection)',
                subtitle=f'Aligned embedding space | {len(test_embeddings_proj)} test samples',
                save_path=str(output_dir / 'text_tsne_proj.png')
            )

            # Save results
            results_summary = {
                'no_projection': {
                    'metrics_l1': {
                        'accuracy': metrics_l1_orig.get('accuracy', 0),
                        'f1_macro': metrics_l1_orig.get('f1_macro', 0),
                        'f1_weighted': metrics_l1_orig.get('f1_weighted', 0),
                        'num_classes': metrics_l1_orig.get('num_classes', 0),
                        'num_samples': metrics_l1_orig.get('num_samples', 0),
                    },
                    'metrics_l2': {
                        'accuracy': metrics_l2_orig.get('accuracy', 0),
                        'f1_macro': metrics_l2_orig.get('f1_macro', 0),
                        'f1_weighted': metrics_l2_orig.get('f1_weighted', 0),
                        'num_classes': metrics_l2_orig.get('num_classes', 0),
                        'num_samples': metrics_l2_orig.get('num_samples', 0),
                    }
                },
                'with_projection': {
                    'metrics_l1': {
                        'accuracy': metrics_l1_proj.get('accuracy', 0),
                        'f1_macro': metrics_l1_proj.get('f1_macro', 0),
                        'f1_weighted': metrics_l1_proj.get('f1_weighted', 0),
                        'num_classes': metrics_l1_proj.get('num_classes', 0),
                        'num_samples': metrics_l1_proj.get('num_samples', 0),
                    },
                    'metrics_l2': {
                        'accuracy': metrics_l2_proj.get('accuracy', 0),
                        'f1_macro': metrics_l2_proj.get('f1_macro', 0),
                        'f1_weighted': metrics_l2_proj.get('f1_weighted', 0),
                        'num_classes': metrics_l2_proj.get('num_classes', 0),
                        'num_samples': metrics_l2_proj.get('num_samples', 0),
                    }
                }
            }

            with open(output_dir / 'text_embedding_results.json', 'w') as f:
                json.dump(results_summary, f, indent=2)

            print(f"💾 Results saved: {output_dir / 'text_embedding_results.json'}")

        # ===== SUMMARY =====
        print("\n" + "="*80)
        print("🎯 TEXT EMBEDDING EVALUATION SUMMARY")
        print("="*80)

        print("\n📊 WITHOUT PROJECTION (Original Space):")
        print(f"   L1 - F1 Macro: {metrics_l1_orig.get('f1_macro', 0):.4f} | "
              f"F1 Weighted: {metrics_l1_orig.get('f1_weighted', 0):.4f} | "
              f"Accuracy: {metrics_l1_orig.get('accuracy', 0):.4f}")
        print(f"   L2 - F1 Macro: {metrics_l2_orig.get('f1_macro', 0):.4f} | "
              f"F1 Weighted: {metrics_l2_orig.get('f1_weighted', 0):.4f} | "
              f"Accuracy: {metrics_l2_orig.get('accuracy', 0):.4f}")

        print("\n📊 WITH PROJECTION (Aligned Space):")
        print(f"   L1 - F1 Macro: {metrics_l1_proj.get('f1_macro', 0):.4f} | "
              f"F1 Weighted: {metrics_l1_proj.get('f1_weighted', 0):.4f} | "
              f"Accuracy: {metrics_l1_proj.get('accuracy', 0):.4f}")
        print(f"   L2 - F1 Macro: {metrics_l2_proj.get('f1_macro', 0):.4f} | "
              f"F1 Weighted: {metrics_l2_proj.get('f1_weighted', 0):.4f} | "
              f"Accuracy: {metrics_l2_proj.get('accuracy', 0):.4f}")

        print(f"\n📈 IMPROVEMENT WITH PROJECTION:")
        l1_improvement = metrics_l1_proj.get('f1_macro', 0) - metrics_l1_orig.get('f1_macro', 0)
        l2_improvement = metrics_l2_proj.get('f1_macro', 0) - metrics_l2_orig.get('f1_macro', 0)
        print(f"   L1 F1 Macro: {l1_improvement:+.4f} ({'improved' if l1_improvement > 0 else 'degraded'})")
        print(f"   L2 F1 Macro: {l2_improvement:+.4f} ({'improved' if l2_improvement > 0 else 'degraded'})")

        return {
            'no_projection': {
                'metrics_l1': metrics_l1_orig,
                'metrics_l2': metrics_l2_orig,
            },
            'with_projection': {
                'metrics_l1': metrics_l1_proj,
                'metrics_l2': metrics_l2_proj,
            }
        }

    def create_prototypes_from_embeddings(self, embeddings: np.ndarray,
                                         labels: List[str]) -> Dict[str, np.ndarray]:
        """Create prototypes by averaging embeddings for each label.

        Args:
            embeddings: Array of embeddings
            labels: Corresponding labels

        Returns:
            Dictionary mapping label to prototype embedding
        """
        unique_labels = sorted(list(set(labels)))
        prototypes = {}

        for label in unique_labels:
            # Get all embeddings for this label
            mask = np.array(labels) == label
            label_embeddings = embeddings[mask]

            # Average to create prototype
            prototype = np.mean(label_embeddings, axis=0)
            prototypes[label] = prototype

        print(f"✅ Created {len(prototypes)} prototypes from embeddings")

        return prototypes

    def create_combined_tsne_plot(self,
                                 embeddings_dict: Dict[str, np.ndarray],
                                 labels_dict: Dict[str, List[str]],
                                 save_path: str = None,
                                 max_samples_tsne: int = 5000) -> plt.Figure:
        """Create combined t-SNE plot with 3 subplots showing different embedding types.

        Args:
            embeddings_dict: Dict with keys 'text_noproj', 'text_proj', 'sensor'
            labels_dict: Dict with same keys containing L1 ground truth labels
            save_path: Path to save plot
            max_samples_tsne: Max samples for t-SNE computation
        """
        print(f"🔄 Creating combined t-SNE visualization (3 subplots)...")

        # Sample each embedding type separately with its own labels
        embeddings_sampled = {}
        labels_sampled = {}

        for key in embeddings_dict.keys():
            embeddings = embeddings_dict[key]
            labels = labels_dict[key]

            if len(labels) > max_samples_tsne:
                print(f"   Sampling {max_samples_tsne} from {len(labels)} samples for {key}")
                np.random.seed(42)
                indices = np.random.choice(len(labels), max_samples_tsne, replace=False)
                embeddings_sampled[key] = embeddings[indices]
                labels_sampled[key] = [labels[i] for i in indices]
            else:
                embeddings_sampled[key] = embeddings
                labels_sampled[key] = labels

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
        fig.suptitle('t-SNE Visualization Comparison (L1 Ground Truth Labels)',
                     fontsize=16, fontweight='bold', y=0.98)

        # Get all unique labels across all embedding types for consistent coloring
        all_labels = []
        for labels in labels_sampled.values():
            all_labels.extend(labels)
        unique_labels_all = sorted(list(set(all_labels)))
        if len(unique_labels_all) > 15:
            label_counts = Counter(all_labels)
            unique_labels_all = [label for label, _ in label_counts.most_common(15)]

        # Create consistent color mapping
        label_to_color = {}
        for i, label in enumerate(unique_labels_all):
            if hasattr(self, 'label_colors') and label in self.label_colors:
                label_to_color[label] = self.label_colors[label]
            else:
                label_to_color[label] = plt.cm.tab20(i % 20)

        # Process each embedding type
        axes = [ax1, ax2, ax3]
        titles = ['Text Embeddings (No Projection)', 'Text Embeddings (With Projection)', 'Sensor Embeddings']
        keys = ['text_noproj', 'text_proj', 'sensor']

        for ax, title, key in zip(axes, titles, keys):
            if key not in embeddings_sampled:
                ax.text(0.5, 0.5, 'Not Available', ha='center', va='center', fontsize=14)
                ax.set_title(title, fontweight='bold', fontsize=12)
                continue

            embeddings = embeddings_sampled[key]
            labels_for_key = labels_sampled[key]

            print(f"   Computing t-SNE for {title}...")
            print(f"      Embeddings shape: {embeddings.shape}, Labels: {len(labels_for_key)}")
            print(f"      Label distribution: {Counter(labels_for_key).most_common(5)}")

            tsne = TSNE(n_components=2, random_state=42,
                       perplexity=min(30, len(embeddings)//4))
            embeddings_2d = tsne.fit_transform(embeddings)

            # Get unique labels for this specific embedding type
            unique_labels_key = sorted(list(set(labels_for_key)))
            if len(unique_labels_key) > 15:
                label_counts_key = Counter(labels_for_key)
                unique_labels_key = [label for label, _ in label_counts_key.most_common(15)]

            # Plot each label using the correct labels for this embedding type
            for label in unique_labels_key:
                mask = np.array(labels_for_key) == label
                if np.any(mask):
                    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                             c=[label_to_color[label]], label=label.replace('_', ' '),
                             alpha=0.6, s=20, edgecolors='white', linewidth=0.3)

            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_xlabel('t-SNE 1', fontsize=10)
            ax.set_ylabel('t-SNE 2', fontsize=10)
            ax.grid(True, alpha=0.3)

        # Create shared legend
        handles, legend_labels = ax1.get_legend_handles_labels()
        if handles:
            fig.legend(handles, legend_labels, bbox_to_anchor=(0.5, -0.02),
                      loc='upper center', fontsize=9, ncol=5, frameon=True,
                      fancybox=True, shadow=True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Combined t-SNE plot saved: {save_path}")

        return fig

    def create_combined_confusion_matrices(self,
                                          confusion_matrices: Dict[str, np.ndarray],
                                          unique_labels: Dict[str, List[str]],
                                          label_level: str,
                                          save_path: str = None) -> plt.Figure:
        """Create combined confusion matrix plot with 3 subplots.

        Args:
            confusion_matrices: Dict with keys 'text_noproj', 'text_proj', 'sensor'
            unique_labels: Dict with same keys containing label lists
            label_level: 'L1' or 'L2'
            save_path: Path to save plot
        """
        print(f"🔄 Creating combined confusion matrix plot ({label_level})...")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
        fig.suptitle(f'Confusion Matrix Comparison - {label_level} Labels',
                     fontsize=16, fontweight='bold')

        axes = [ax1, ax2, ax3]
        titles = ['Text (No Projection)', 'Text (With Projection)', 'Sensor']
        keys = ['text_noproj', 'text_proj', 'sensor']

        for ax, title, key in zip(axes, titles, keys):
            if key not in confusion_matrices:
                ax.text(0.5, 0.5, 'Not Available', ha='center', va='center', fontsize=14)
                ax.set_title(title, fontweight='bold', fontsize=12)
                continue

            cm = confusion_matrices[key]
            labels = unique_labels[key]

            # Limit to top 15 if too many
            if len(labels) > 15:
                cm = cm[:15, :15]
                labels = labels[:15]

            # Normalize
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

            # Plot heatmap
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=labels, yticklabels=labels, ax=ax,
                       cbar_kws={'label': 'Normalized Count'})

            ax.set_title(title, fontweight='bold', fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=10)
            ax.set_ylabel('True Label', fontsize=10)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
            plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Combined confusion matrix plot saved: {save_path}")

        return fig

    def create_combined_f1_analysis(self,
                                   metrics_dict: Dict[str, Dict],
                                   save_path: str = None) -> plt.Figure:
        """Create F1 analysis plot with 2 subplots (L1 and L2).
        Grouped by metric (each metric shows all 3 models).

        Args:
            metrics_dict: Dict with keys 'text_noproj', 'text_proj', 'sensor',
                         each containing 'metrics_l1' and 'metrics_l2'
            save_path: Path to save plot
        """
        print(f"🔄 Creating combined F1 analysis plot...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Performance Comparison Across Embedding Types',
                     fontsize=16, fontweight='bold')

        model_names = ['Text\n(No Proj)', 'Text\n(With Proj)', 'Sensor']
        keys = ['text_noproj', 'text_proj', 'sensor']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

        # Extract metrics for L1
        l1_accuracy = [metrics_dict[k]['metrics_l1'].get('accuracy', 0) for k in keys if k in metrics_dict]
        l1_f1_weighted = [metrics_dict[k]['metrics_l1'].get('f1_weighted', 0) for k in keys if k in metrics_dict]
        l1_f1_macro = [metrics_dict[k]['metrics_l1'].get('f1_macro', 0) for k in keys if k in metrics_dict]

        # Extract metrics for L2
        l2_accuracy = [metrics_dict[k]['metrics_l2'].get('accuracy', 0) for k in keys if k in metrics_dict]
        l2_f1_weighted = [metrics_dict[k]['metrics_l2'].get('f1_weighted', 0) for k in keys if k in metrics_dict]
        l2_f1_macro = [metrics_dict[k]['metrics_l2'].get('f1_macro', 0) for k in keys if k in metrics_dict]

        # Filter model names based on available data
        available_models = [model_names[i] for i, k in enumerate(keys) if k in metrics_dict]
        n_models = len(available_models)

        # Metrics grouped together
        metric_names = ['Accuracy', 'F1 Weighted', 'F1 Macro']
        x = np.arange(len(metric_names))
        width = 0.25

        # L1 subplot - each metric has bars for all models
        for i in range(n_models):
            l1_values = [l1_accuracy[i], l1_f1_weighted[i], l1_f1_macro[i]]
            bars = ax1.bar(x + (i - 1) * width, l1_values, width,
                          label=available_models[i], alpha=0.8, color=colors[i])

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0.02:
                    ax1.annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)

        ax1.set_xlabel('Metric', fontsize=11)
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('L1 Primary Activities', fontweight='bold', fontsize=13)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_names, fontsize=10)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1)

        # L2 subplot - each metric has bars for all models
        for i in range(n_models):
            l2_values = [l2_accuracy[i], l2_f1_weighted[i], l2_f1_macro[i]]
            bars = ax2.bar(x + (i - 1) * width, l2_values, width,
                          label=available_models[i], alpha=0.8, color=colors[i])

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0.02:
                    ax2.annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)

        ax2.set_xlabel('Metric', fontsize=11)
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('L2 Secondary Activities', fontweight='bold', fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_names, fontsize=10)
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Combined F1 analysis plot saved: {save_path}")

        return fig

    def create_perclass_f1_weighted_analysis(self,
                                            metrics_dict: Dict[str, Dict],
                                            label_level: str,
                                            save_path: str = None) -> plt.Figure:
        """Create per-class F1 weighted analysis plot.

        Args:
            metrics_dict: Dict with keys 'text_noproj', 'text_proj', 'sensor'
            label_level: 'L1' or 'L2'
            save_path: Path to save plot
        """
        print(f"🔄 Creating per-class F1 weighted analysis ({label_level})...")

        # Get metrics key
        metrics_key = 'metrics_l1' if label_level == 'L1' else 'metrics_l2'

        # Get all unique labels across models
        all_labels = set()
        for key in ['text_noproj', 'text_proj', 'sensor']:
            if key in metrics_dict and metrics_key in metrics_dict[key]:
                per_class_f1 = metrics_dict[key][metrics_key].get('per_class_f1', {})
                all_labels.update(per_class_f1.keys())

        labels = sorted(list(all_labels))

        # Extract F1 scores for each model
        text_noproj_f1 = []
        text_proj_f1 = []
        sensor_f1 = []

        for label in labels:
            text_noproj_f1.append(
                metrics_dict.get('text_noproj', {}).get(metrics_key, {}).get('per_class_f1', {}).get(label, 0)
            )
            text_proj_f1.append(
                metrics_dict.get('text_proj', {}).get(metrics_key, {}).get('per_class_f1', {}).get(label, 0)
            )
            sensor_f1.append(
                metrics_dict.get('sensor', {}).get(metrics_key, {}).get('per_class_f1', {}).get(label, 0)
            )

        # Create plot
        fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.8), 8))

        x = np.arange(len(labels))
        width = 0.25

        bars1 = ax.bar(x - width, text_noproj_f1, width, label='Text (No Projection)', alpha=0.8)
        bars2 = ax.bar(x, text_proj_f1, width, label='Text (With Projection)', alpha=0.8, color='orange')
        bars3 = ax.bar(x + width, sensor_f1, width, label='Sensor', alpha=0.8, color='green')

        ax.set_xlabel('Activity Class', fontsize=12)
        ax.set_ylabel('F1 Weighted Score', fontsize=12)
        ax.set_title(f'Per-Class F1 Weighted Comparison - {label_level} Labels',
                    fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([label.replace('_', ' ') for label in labels],
                          rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Per-class F1 weighted analysis saved: {save_path}")

        return fig

    def create_retrieval_metrics_visualization(self,
                                              retrieval_results: Dict[str, Dict[int, Union[float, Dict[str, float]]]],
                                              save_path: str = None,
                                              label_level: str = "L1") -> plt.Figure:
        """Create visualization for retrieval metrics across all 4 variants.

        Args:
            retrieval_results: Dict with keys 'text2sensor', 'sensor2text',
                             'prototype2sensor', 'prototype2text'
                             Values are either floats or dicts with 'macro' and 'weighted' keys
            save_path: Path to save the plot
            label_level: Label level being used (L1 or L2)
        """
        print(f"🎨 Creating retrieval metrics visualization for {label_level} labels...")

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Cross-Modal Retrieval Performance ({label_level} Labels)',
                     fontsize=18, fontweight='bold')

        # Get all K values (should be same across all directions)
        k_values_list = []
        for direction, k_results in retrieval_results.items():
            if k_results:
                k_values_list = sorted(k_results.keys())
                break

        direction_order = ['text2sensor', 'sensor2text', 'text2text', 'sensor2sensor', 'prototype2sensor', 'prototype2text']

        # Colors for different retrieval types
        colors = {
            'text2sensor': '#1f77b4',        # Blue
            'sensor2text': '#ff7f0e',        # Orange
            'text2text': '#8c564b',          # Brown
            'sensor2sensor': '#9467bd',      # Purple
            'prototype2sensor': '#2ca02c',   # Green
            'prototype2text': '#d62728'      # Red
        }

        # Labels for legend
        labels_map = {
            'text2sensor': 'Text → Sensor',
            'sensor2text': 'Sensor → Text',
            'text2text': 'Text → Text',
            'sensor2sensor': 'Sensor → Sensor',
            'prototype2sensor': 'Prototype → Sensor',
            'prototype2text': 'Prototype → Text'
        }

        # Helper function to extract metric
        def get_metric(k_result, metric_type='macro'):
            if isinstance(k_result, dict):
                return k_result.get(metric_type, 0.0)
            return k_result  # Backward compatibility

        score_label = 'Label-Precision@K'

        # Plot 1: Macro average across K values
        ax1 = axes[0, 0]
        for direction in direction_order:
            if direction in retrieval_results and retrieval_results[direction]:
                k_results = retrieval_results[direction]
                precisions = [get_metric(k_results[k], 'macro') for k in k_values_list]

                ax1.plot(k_values_list, precisions, 'o-',
                        color=colors.get(direction, '#333333'),
                        label=labels_map.get(direction, direction),
                        linewidth=2.5, markersize=8, alpha=0.8)

        ax1.set_xlabel('K (Number of Retrieved Neighbors)', fontsize=12)
        ax1.set_ylabel(score_label, fontsize=12)
        ax1.set_title('Macro Average (Equal Weight per Class)', fontweight='bold', fontsize=13)
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        ax1.set_xticks(k_values_list)

        # Plot 2: Weighted average across K values
        ax2 = axes[0, 1]
        for direction in direction_order:
            if direction in retrieval_results and retrieval_results[direction]:
                k_results = retrieval_results[direction]
                precisions = [get_metric(k_results[k], 'weighted') for k in k_values_list]

                ax2.plot(k_values_list, precisions, 'o-',
                        color=colors.get(direction, '#333333'),
                        label=labels_map.get(direction, direction),
                        linewidth=2.5, markersize=8, alpha=0.8)

        ax2.set_xlabel('K (Number of Retrieved Neighbors)', fontsize=12)
        ax2.set_ylabel(score_label, fontsize=12)
        ax2.set_title('Weighted Average (By Class Prevalence)', fontweight='bold', fontsize=13)
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        ax2.set_xticks(k_values_list)

        # Plot 3: Bar chart comparing macro at middle K
        middle_k = k_values_list[len(k_values_list)//2] if k_values_list else 10
        ax3 = axes[1, 0]

        bar_data_macro = []
        bar_labels = []
        bar_colors = []

        for direction in direction_order:
            if direction in retrieval_results and retrieval_results[direction]:
                k_results = retrieval_results[direction]
                if middle_k in k_results:
                    bar_data_macro.append(get_metric(k_results[middle_k], 'macro'))
                    bar_labels.append(labels_map.get(direction, direction))
                    bar_colors.append(colors.get(direction, '#333333'))

        if bar_data_macro:
            x = np.arange(len(bar_data_macro))
            bars = ax3.bar(x, bar_data_macro, color=bar_colors, alpha=0.8, width=0.6)

            ax3.set_ylabel(score_label, fontsize=12)
            ax3.set_title(f'Macro Comparison @ K={middle_k}', fontweight='bold', fontsize=13)
            ax3.set_xticks(x)
            ax3.set_xticklabels(bar_labels, rotation=15, ha='right', fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_ylim(0, 1)

            for bar in bars:
                height = bar.get_height()
                ax3.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Plot 4: Bar chart comparing weighted at middle K
        ax4 = axes[1, 1]

        bar_data_weighted = []
        bar_labels2 = []
        bar_colors2 = []

        for direction in direction_order:
            if direction in retrieval_results and retrieval_results[direction]:
                k_results = retrieval_results[direction]
                if middle_k in k_results:
                    bar_data_weighted.append(get_metric(k_results[middle_k], 'weighted'))
                    bar_labels2.append(labels_map.get(direction, direction))
                    bar_colors2.append(colors.get(direction, '#333333'))

        if bar_data_weighted:
            x = np.arange(len(bar_data_weighted))
            bars = ax4.bar(x, bar_data_weighted, color=bar_colors2, alpha=0.8, width=0.6)

            ax4.set_ylabel(score_label, fontsize=12)
            ax4.set_title(f'Weighted Comparison @ K={middle_k}', fontweight='bold', fontsize=13)
            ax4.set_xticks(x)
            ax4.set_xticklabels(bar_labels2, rotation=15, ha='right', fontsize=10)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_ylim(0, 1)

            for bar in bars:
                height = bar.get_height()
                ax4.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Retrieval metrics visualization saved: {save_path}")

        return fig

    def create_per_label_retrieval_heatmap_instance(self,
                                                   query_embeddings: np.ndarray,
                                                   target_embeddings: np.ndarray,
                                                   query_labels: np.ndarray,
                                                   target_labels: np.ndarray,
                                                   direction_name: str,
                                                   k: int = 10,
                                                   save_path: str = None) -> plt.Figure:
        """Create heatmap showing per-label retrieval performance for instance-to-instance.

        Args:
            query_embeddings: Query embeddings (N_q, D)
            target_embeddings: Target embeddings (N_t, D)
            query_labels: Labels for queries (N_q,)
            target_labels: Labels for targets (N_t,)
            direction_name: Name of retrieval direction
            k: K value for precision computation
            save_path: Path to save plot
        """
        from evals.compute_retrieval_metrics import compute_per_label_recall_at_k, compute_macro_and_weighted_metrics

        print(f"🎨 Creating per-label instance retrieval heatmap for {direction_name}...")

        # Compute per-label precision and counts (note: function name is legacy, it computes precision)
        per_label_precisions, per_label_counts = compute_per_label_recall_at_k(
            query_embeddings=query_embeddings,
            target_embeddings=target_embeddings,
            query_labels=query_labels,
            target_labels=target_labels,
            k=k,
            return_counts=True
        )

        # Compute macro and weighted metrics
        macro_precision, weighted_precision = compute_macro_and_weighted_metrics(
            per_label_precisions, per_label_counts
        )

        # Convert to arrays for plotting
        labels = list(per_label_precisions.keys())
        precisions = [per_label_precisions[label] for label in labels]

        # Create bar plot
        fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 8))

        # Sort by precision for better visualization
        sorted_indices = np.argsort(precisions)[::-1]
        sorted_labels = np.array(labels)[sorted_indices]
        sorted_precisions = np.array(precisions)[sorted_indices]

        # Limit to top 20 for readability
        if len(sorted_labels) > 20:
            sorted_labels = sorted_labels[:20]
            sorted_precisions = sorted_precisions[:20]

        # Create horizontal bar chart
        y_pos = np.arange(len(sorted_labels))
        colors_bars = plt.cm.RdYlGn(sorted_precisions)  # Color based on performance

        bars = ax.barh(y_pos, sorted_precisions, color=colors_bars, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([label.replace('_', ' ') for label in sorted_labels], fontsize=9)
        ax.set_xlabel(f'Label-Precision@{k}', fontsize=12)
        ax.set_title(f'Per-Label Instance Retrieval: {direction_name}',
                    fontweight='bold', fontsize=14)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, precision) in enumerate(zip(bars, sorted_precisions)):
            width = bar.get_width()
            ax.annotate(f'{precision:.3f}',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(5, 0),
                       textcoords="offset points",
                       ha='left', va='center', fontsize=8)

        # Add macro and weighted lines
        ax.axvline(macro_precision, color='blue', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'Macro: {macro_precision:.3f}')
        ax.axvline(weighted_precision, color='red', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'Weighted: {weighted_precision:.3f}')
        ax.legend(fontsize=10)

        # Add text box with metrics in corner
        textstr = f'Macro: {macro_precision:.4f}\nWeighted: {weighted_precision:.4f}\nN={len(labels)} classes'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Per-label instance retrieval heatmap saved: {save_path}")

        return fig

    def create_per_label_retrieval_heatmap(self,
                                          prototype_labels: np.ndarray,
                                          target_embeddings: np.ndarray,
                                          target_labels: np.ndarray,
                                          prototype_embeddings: np.ndarray,
                                          direction_name: str,
                                          k: int = 10,
                                          save_path: str = None) -> plt.Figure:
        """Create heatmap showing per-label retrieval performance.

        Args:
            prototype_labels: Labels for prototypes (N_prototypes,)
            target_embeddings: Target embeddings (N_targets, D)
            target_labels: Labels for targets (N_targets,)
            prototype_embeddings: Prototype embeddings (N_prototypes, D)
            direction_name: Name of retrieval direction
        k: K value for precision computation
            save_path: Path to save plot
        """
        from evals.compute_retrieval_metrics import (
            normalize_embeddings,
            compute_cosine_similarity
        )

        print(f"🎨 Creating per-label retrieval heatmap for {direction_name}...")

        # Normalize embeddings
        prototype_embeddings_norm = normalize_embeddings(prototype_embeddings)
        target_embeddings_norm = normalize_embeddings(target_embeddings)

        # Compute similarity matrix
        similarities = compute_cosine_similarity(prototype_embeddings_norm, target_embeddings_norm)

        # For each prototype, compute precision@k (truncate to available positives)
        target_label_counts = Counter(str(label) for label in target_labels)
        per_label_precisions = []
        for i, proto_label in enumerate(prototype_labels):
            proto_label_str = str(proto_label)
            proto_sims = similarities[i]
            label_total = target_label_counts.get(proto_label_str, 0)
            if label_total == 0:
                per_label_precisions.append(0.0)
                continue

            k_effective = min(k, label_total)
            top_k_indices = np.argsort(proto_sims)[-k_effective:][::-1]
            top_k_labels = target_labels[top_k_indices]
            n_matching = np.sum(top_k_labels == proto_label)
            precision = n_matching / k_effective if k_effective > 0 else 0.0
            per_label_precisions.append(precision)

        # Create bar plot
        fig, ax = plt.subplots(figsize=(max(12, len(prototype_labels) * 0.5), 8))

        # Sort by precision for better visualization
        sorted_indices = np.argsort(per_label_precisions)[::-1]
        sorted_labels = np.array(prototype_labels)[sorted_indices]
        sorted_precisions = np.array(per_label_precisions)[sorted_indices]

        # Limit to top 20 for readability
        if len(sorted_labels) > 20:
            sorted_labels = sorted_labels[:20]
            sorted_precisions = sorted_precisions[:20]

        # Create horizontal bar chart
        y_pos = np.arange(len(sorted_labels))
        colors_bars = plt.cm.RdYlGn(sorted_precisions)  # Color based on performance

        bars = ax.barh(y_pos, sorted_precisions, color=colors_bars, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([label.replace('_', ' ') for label in sorted_labels], fontsize=9)
        ax.set_xlabel(f'Label-Precision@{k}', fontsize=12)
        ax.set_title(f'Per-Label Retrieval Precision: {direction_name}',
                    fontweight='bold', fontsize=14)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, precision) in enumerate(zip(bars, sorted_precisions)):
            width = bar.get_width()
            ax.annotate(f'{precision:.3f}',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(5, 0),
                       textcoords="offset points",
                       ha='left', va='center', fontsize=8)

        # Add average line
        avg_precision = np.mean(sorted_precisions)
        ax.axvline(avg_precision, color='red', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'Average: {avg_precision:.3f}')
        ax.legend(fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Per-label retrieval heatmap saved: {save_path}")

        return fig

    def create_retrieval_confusion_heatmap(self,
                                          retrieval_confusion: Dict[str, Dict[str, float]],
                                          direction_name: str,
                                          k: int = 10,
                                          save_path: str = None) -> plt.Figure:
        """Create heatmap showing retrieval confusion (what labels are retrieved for each query label).

        This is like a confusion matrix for retrieval - shows the distribution of
        retrieved target labels for each query label.

        Args:
            retrieval_confusion: Nested dict {query_label: {target_label: proportion}}
            direction_name: Name of retrieval direction
            k: K value used
            save_path: Path to save plot
        """
        print(f"🎨 Creating retrieval confusion heatmap for {direction_name}...")

        # Get all unique query and target labels
        query_labels = sorted(list(retrieval_confusion.keys()))
        all_target_labels = set()
        for target_dist in retrieval_confusion.values():
            all_target_labels.update(target_dist.keys())
        target_labels = sorted(list(all_target_labels))

        # Create confusion matrix
        confusion_matrix_data = np.zeros((len(query_labels), len(target_labels)))

        for i, query_label in enumerate(query_labels):
            target_dist = retrieval_confusion[query_label]
            for j, target_label in enumerate(target_labels):
                confusion_matrix_data[i, j] = target_dist.get(target_label, 0)

        # Create figure
        figsize_width = max(12, len(target_labels) * 0.6)
        figsize_height = max(10, len(query_labels) * 0.6)
        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))

        # Determine font sizes based on matrix size
        n_labels = max(len(query_labels), len(target_labels))
        if n_labels <= 10:
            annot_fontsize = 8
            tick_fontsize = 9
        elif n_labels <= 15:
            annot_fontsize = 7
            tick_fontsize = 8
        else:
            annot_fontsize = 6
            tick_fontsize = 7

        # Clean labels for display
        query_labels_display = [label.replace('_', ' ') for label in query_labels]
        target_labels_display = [label.replace('_', ' ') for label in target_labels]

        # Create heatmap
        im = ax.imshow(confusion_matrix_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(len(target_labels)))
        ax.set_yticks(np.arange(len(query_labels)))
        ax.set_xticklabels(target_labels_display, rotation=90, ha='center', fontsize=tick_fontsize)
        ax.set_yticklabels(query_labels_display, fontsize=tick_fontsize)

        # Add text annotations
        for i in range(len(query_labels)):
            for j in range(len(target_labels)):
                value = confusion_matrix_data[i, j]
                if value > 0.01:  # Only show non-negligible values
                    text_color = "white" if value > 0.5 else "black"
                    text = ax.text(j, i, f'{value:.2f}',
                                 ha="center", va="center", color=text_color,
                                 fontsize=annot_fontsize)

        ax.set_xlabel('Retrieved Label (Target)', fontsize=11)
        ax.set_ylabel('Query Label', fontsize=11)
        ax.set_title(f'Retrieval Confusion @ K={k}: {direction_name}\n(Shows distribution of retrieved labels for each query)',
                    fontweight='bold', fontsize=13, pad=15)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Proportion of Retrieved Items', rotation=270, labelpad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Retrieval confusion heatmap saved: {save_path}")

        return fig

    def run_comprehensive_evaluation(self,
                                     train_text_embeddings_path: str,
                                     test_text_embeddings_path: str,
                                     max_samples: int = 10000,
                                     k_neighbors: int = 1,
                                     filter_noisy_labels: bool = False,
                                     save_results: bool = True,
                                     use_multiple_prototypes: bool = False) -> Dict[str, Any]:
        """Run comprehensive evaluation: sensor + text embeddings (with/without projection).

        Creates unified visualizations comparing all three embedding types.

        Args:
            use_multiple_prototypes: If True, use multiple prototypes per label with k-NN voting.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE EMBEDDING EVALUATION")
        print("Sensor Embeddings + Text Embeddings (with/without projection)")
        print("="*80)

        # Create output directory with subdirectory for multiple prototypes mode
        base_output_dir = Path(self.config.get('output_dir', './comprehensive_evaluation'))
        if use_multiple_prototypes:
            output_dir = base_output_dir / f"multiproto_k{k_neighbors}"
            print(f"ℹ️  Multiple prototypes mode: saving to {output_dir}")
        else:
            output_dir = base_output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create retrieval subdirectory for better organization
        retrieval_dir = output_dir / 'retrieval'
        retrieval_dir.mkdir(parents=True, exist_ok=True)

        # ===== 1. GET LABELS FROM METADATA & LOAD TEST TEXT EMBEDDINGS =====
        print("\n" + "="*60)
        print("1. LOADING LABELS AND TEST TEXT EMBEDDINGS")
        print("="*60)

        # Get labels from metadata (efficient, no need to load training data)
        train_labels_l1, train_labels_l2 = self.get_labels_from_metadata(self.dataset_name)
        print(f"✅ Got {len(train_labels_l1)} L1 labels and {len(train_labels_l2)} L2 labels from metadata")

        # Filter noisy labels from metadata if requested (BEFORE creating prototypes)
        if filter_noisy_labels:
            print("\n⚠️  Filtering noisy labels from metadata before creating prototypes...")

            # Define labels to exclude (case-insensitive)
            exclude_labels = {
                'other',
                'no_activity', 'No_Activity',
                'unknown', 'none', 'null', 'nan',
                'no activity', 'other activity', 'miscellaneous', 'misc'
            }

            # Filter L1 labels
            original_l1_count = len(train_labels_l1)
            train_labels_l1 = [label for label in train_labels_l1
                              if label.lower().strip() not in exclude_labels]

            # Filter L2 labels
            original_l2_count = len(train_labels_l2)
            train_labels_l2 = [label for label in train_labels_l2
                              if label.lower().strip() not in exclude_labels]

            print(f"   L1 labels: {original_l1_count} → {len(train_labels_l1)} (removed {original_l1_count - len(train_labels_l1)})")
            print(f"   L2 labels: {original_l2_count} → {len(train_labels_l2)} (removed {original_l2_count - len(train_labels_l2)})")

        # Only load TEST text embeddings (we don't need train embeddings)
        test_text_emb, test_text_sample_ids, test_labels_l1, test_labels_l2 = \
            self.load_text_embeddings_from_file(
                test_text_embeddings_path,
                self.config['test_data_path'],
                max_samples
            )

        # ===== 2. CREATE SYNTHETIC TEXT PROTOTYPES (L1 ONLY) =====
        print("\n" + "="*60)
        print("2. CREATING SYNTHETIC TEXT PROTOTYPES (L1 Only - L2 Derived)")
        print("="*60)
        print("ℹ️  Creating only L1 prototypes. L2 predictions will be derived from L1 using metadata mapping.")

        # Create RAW prototypes (without projection) for "no projection" comparison
        prototypes_l1_raw, _ = self.create_text_prototypes(train_labels_l1, apply_projection=False, use_multiple_prototypes=use_multiple_prototypes)

        # Create PROJECTED prototypes for "with projection" and sensor comparisons
        prototypes_l1_projected, _ = self.create_text_prototypes(train_labels_l1, apply_projection=True, use_multiple_prototypes=use_multiple_prototypes)

        # ===== 3. EVALUATE TEXT (NO PROJECTION) - DEFERRED =====
        # Note: We'll evaluate text without projection AFTER alignment with sensor embeddings
        # to ensure all three embedding types have the same samples

        # ===== 4. EVALUATE TEXT (WITH PROJECTION) - DEFERRED =====
        # Note: We'll evaluate text with projection AFTER alignment with sensor embeddings
        # to ensure all three embedding types have the same samples

        # ===== 5. EVALUATE SENSOR EMBEDDINGS =====
        print("\n" + "="*60)
        print("5. EVALUATING SENSOR EMBEDDINGS & ALIGNING WITH TEXT")
        print("="*60)

        # Extract TEST sensor embeddings only (we don't need train, we use prototypes)
        test_sensor_emb, test_sensor_l1, test_sensor_l2, test_sensor_sample_ids = self.extract_embeddings_and_labels('test', max_samples)

        # CRITICAL: Align sensor and text embeddings by sample_id BEFORE any processing
        test_sensor_emb_aligned, test_text_emb_aligned, aligned_labels_l1, aligned_labels_l2, aligned_sample_ids = \
            self.align_embeddings_by_sample_id(
                sensor_emb=test_sensor_emb,
                sensor_labels_l1=test_sensor_l1,
                sensor_labels_l2=test_sensor_l2,
                sensor_sample_ids=test_sensor_sample_ids,
                text_emb=test_text_emb,
                text_labels_l1=test_labels_l1,
                text_labels_l2=test_labels_l2,
                text_sample_ids=test_text_sample_ids
            )

        # Now use the aligned data for all subsequent processing
        test_sensor_emb = test_sensor_emb_aligned
        test_text_emb = test_text_emb_aligned
        test_sensor_l1 = aligned_labels_l1
        test_sensor_l2 = aligned_labels_l2
        test_labels_l1 = aligned_labels_l1  # Same labels for both
        test_labels_l2 = aligned_labels_l2  # Same labels for both

        # Apply projection to ALIGNED text embeddings
        test_text_emb_proj = self.apply_projection_to_embeddings(test_text_emb)

        if filter_noisy_labels:
            # Create a boolean mask for filtering based on labels
            # Use the same comprehensive exclude list as above
            exclude_labels = {
                'other',
                'no_activity', 'No_Activity',
                'unknown', 'none', 'null', 'nan',
                'no activity', 'other activity', 'miscellaneous', 'misc'
            }

            # Create mask - True means KEEP the sample
            keep_mask = []
            for l1, l2 in zip(test_sensor_l1, test_sensor_l2):
                l1_lower = l1.lower().strip()
                l2_lower = l2.lower().strip()
                keep = (l1_lower not in exclude_labels) and (l2_lower not in exclude_labels)
                keep_mask.append(keep)

            keep_mask = np.array(keep_mask)
            n_before = len(keep_mask)
            n_after = keep_mask.sum()

            print(f"🧹 Filtering noisy labels:")
            print(f"   Before: {n_before} samples")
            print(f"   After: {n_after} samples")
            print(f"   Removed: {n_before - n_after} samples")

            # Apply the same mask to ALL embeddings and labels
            test_sensor_emb = test_sensor_emb[keep_mask]
            test_text_emb = test_text_emb[keep_mask]
            test_text_emb_proj = test_text_emb_proj[keep_mask]
            test_sensor_l1 = [l for l, keep in zip(test_sensor_l1, keep_mask) if keep]
            test_sensor_l2 = [l for l, keep in zip(test_sensor_l2, keep_mask) if keep]
            test_labels_l1 = [l for l, keep in zip(test_labels_l1, keep_mask) if keep]
            test_labels_l2 = [l for l, keep in zip(test_labels_l2, keep_mask) if keep]

            # Verify all have the same length
            assert len(test_sensor_emb) == len(test_text_emb) == len(test_text_emb_proj) == len(test_sensor_l1) == len(test_labels_l1), \
                "Filtering resulted in misaligned data!"
            print(f"✅ All embeddings and labels aligned: {len(test_sensor_l1)} samples")

        # Now evaluate text embeddings WITHOUT projection (after alignment and filtering)
        print("\n" + "="*60)
        print("5a. EVALUATING TEXT EMBEDDINGS (NO PROJECTION) - After Alignment")
        print("="*60)
        # Predict L1 labels
        pred_l1_text_noproj = self.predict_labels_knn(test_text_emb, prototypes_l1_raw, k_neighbors, use_multiple_prototypes)
        # Derive L2 predictions from L1 using metadata mapping
        pred_l2_text_noproj = self.map_l1_to_l2_labels(pred_l1_text_noproj, house_name=self.dataset_name)

        metrics_l1_text_noproj = self.evaluate_predictions(test_labels_l1, pred_l1_text_noproj, "Text (No Proj) L1")
        metrics_l2_text_noproj = self.evaluate_predictions(test_labels_l2, pred_l2_text_noproj, "Text (No Proj) L2")

        # Now evaluate text embeddings WITH projection (after alignment and filtering)
        print("\n" + "="*60)
        print("5b. EVALUATING TEXT EMBEDDINGS (WITH PROJECTION) - After Alignment")
        print("="*60)
        # Predict L1 labels
        pred_l1_text_proj = self.predict_labels_knn(test_text_emb_proj, prototypes_l1_projected, k_neighbors, use_multiple_prototypes)
        # Derive L2 predictions from L1 using metadata mapping
        pred_l2_text_proj = self.map_l1_to_l2_labels(pred_l1_text_proj, house_name=self.dataset_name)

        metrics_l1_text_proj = self.evaluate_predictions(test_labels_l1, pred_l1_text_proj, "Text (With Proj) L1")
        metrics_l2_text_proj = self.evaluate_predictions(test_labels_l2, pred_l2_text_proj, "Text (With Proj) L2")

        # Evaluate sensor embeddings
        print("\n" + "="*60)
        print("5c. EVALUATING SENSOR EMBEDDINGS - After Alignment")
        print("="*60)
        # Sensor embeddings are already in projected/aligned space
        # Predict L1 labels
        pred_l1_sensor = self.predict_labels_knn(test_sensor_emb, prototypes_l1_projected, k_neighbors, use_multiple_prototypes)
        # Derive L2 predictions from L1 using metadata mapping
        pred_l2_sensor = self.map_l1_to_l2_labels(pred_l1_sensor, house_name=self.dataset_name)

        metrics_l1_sensor = self.evaluate_predictions(test_sensor_l1, pred_l1_sensor, "Sensor L1")
        metrics_l2_sensor = self.evaluate_predictions(test_sensor_l2, pred_l2_sensor, "Sensor L2")

        # ===== 6. CREATE COMBINED VISUALIZATIONS =====
        print("\n" + "="*60)
        print("6. CREATING COMBINED VISUALIZATIONS")
        print("="*60)

        # Combined t-SNE (L1 ground truth only) - ALL use the same labels after alignment
        embeddings_dict = {
            'text_noproj': test_text_emb,
            'text_proj': test_text_emb_proj,
            'sensor': test_sensor_emb
        }
        # All three embedding types should use the SAME labels (they're aligned!)
        # Using test_labels_l1 for all to ensure consistency
        labels_dict = {
            'text_noproj': test_labels_l1,
            'text_proj': test_labels_l1,
            'sensor': test_labels_l1  # Changed from test_sensor_l1 to test_labels_l1
        }

        # Sanity check: all embeddings should have same number of samples
        print(f"\n📊 t-SNE Input Sanity Check:")
        print(f"   text_noproj: {len(test_text_emb)} embeddings, {len(labels_dict['text_noproj'])} labels")
        print(f"   text_proj: {len(test_text_emb_proj)} embeddings, {len(labels_dict['text_proj'])} labels")
        print(f"   sensor: {len(test_sensor_emb)} embeddings, {len(labels_dict['sensor'])} labels")

        self.create_combined_tsne_plot(
            embeddings_dict=embeddings_dict,
            labels_dict=labels_dict,
            save_path=str(output_dir / 'combined_tsne_l1.png')
        )

        # Combined confusion matrices (L1)
        confusion_matrices_l1 = {
            'text_noproj': metrics_l1_text_noproj['confusion_matrix'],
            'text_proj': metrics_l1_text_proj['confusion_matrix'],
            'sensor': metrics_l1_sensor['confusion_matrix']
        }
        unique_labels_l1 = {
            'text_noproj': metrics_l1_text_noproj['unique_labels'],
            'text_proj': metrics_l1_text_proj['unique_labels'],
            'sensor': metrics_l1_sensor['unique_labels']
        }
        self.create_combined_confusion_matrices(
            confusion_matrices=confusion_matrices_l1,
            unique_labels=unique_labels_l1,
            label_level='L1',
            save_path=str(output_dir / 'combined_confusion_matrix_l1.png')
        )

        # Combined confusion matrices (L2)
        confusion_matrices_l2 = {
            'text_noproj': metrics_l2_text_noproj['confusion_matrix'],
            'text_proj': metrics_l2_text_proj['confusion_matrix'],
            'sensor': metrics_l2_sensor['confusion_matrix']
        }
        unique_labels_l2 = {
            'text_noproj': metrics_l2_text_noproj['unique_labels'],
            'text_proj': metrics_l2_text_proj['unique_labels'],
            'sensor': metrics_l2_sensor['unique_labels']
        }
        self.create_combined_confusion_matrices(
            confusion_matrices=confusion_matrices_l2,
            unique_labels=unique_labels_l2,
            label_level='L2',
            save_path=str(output_dir / 'combined_confusion_matrix_l2.png')
        )

        # Combined F1 analysis
        metrics_dict = {
            'text_noproj': {'metrics_l1': metrics_l1_text_noproj, 'metrics_l2': metrics_l2_text_noproj},
            'text_proj': {'metrics_l1': metrics_l1_text_proj, 'metrics_l2': metrics_l2_text_proj},
            'sensor': {'metrics_l1': metrics_l1_sensor, 'metrics_l2': metrics_l2_sensor}
        }
        self.create_combined_f1_analysis(
            metrics_dict=metrics_dict,
            save_path=str(output_dir / 'combined_f1_analysis.png')
        )

        # Per-class F1 weighted (L1)
        self.create_perclass_f1_weighted_analysis(
            metrics_dict=metrics_dict,
            label_level='L1',
            save_path=str(output_dir / 'perclass_f1_weighted_l1.png')
        )

        # Per-class F1 weighted (L2)
        self.create_perclass_f1_weighted_analysis(
            metrics_dict=metrics_dict,
            label_level='L2',
            save_path=str(output_dir / 'perclass_f1_weighted_l2.png')
        )

        # ===== 7. COMPUTE RETRIEVAL METRICS =====
        print("\n" + "="*60)
        print("7. COMPUTING RETRIEVAL METRICS (L1 and L2 Labels)")
        print("="*60)

        # Store results for both label levels
        retrieval_results_by_level = {}

        # Compute metrics for both L1 and L2 labels
        for retrieval_label_level in ['L1', 'L2']:
            print(f"\n{'='*70}")
            print(f"Computing retrieval metrics for {retrieval_label_level} labels")
            print(f"{'='*70}")

            if retrieval_label_level == 'L1':
                # Use L1 labels for all retrieval metrics
                # Note: test_labels_l1 and test_sensor_l1 are NOW THE SAME after alignment
                labels_for_retrieval = np.array(test_labels_l1)
                print(f"📋 {len(set(test_labels_l1))} unique {retrieval_label_level} labels")
            else:
                # Use L2 labels for all retrieval metrics
                # Note: test_labels_l2 and test_sensor_l2 are NOW THE SAME after alignment
                labels_for_retrieval = np.array(test_labels_l2)
                print(f"📋 {len(set(test_labels_l2))} unique {retrieval_label_level} labels")

            # Instance-to-instance retrieval (text <-> sensor)
            print("\n📊 Computing instance-to-instance retrieval...")
            cross_instance_results, cross_instance_per_label = compute_label_recall_at_k(
                sensor_embeddings=test_sensor_emb,
                text_embeddings=test_text_emb_proj,  # Use projected text embeddings
                labels=labels_for_retrieval,
                k_values=[10, 50, 100],
                directions=['text2sensor', 'sensor2text', 'sensor2sensor'],
                normalize=True,
                verbose=True,
                exclude_self=True,
                return_per_label=True
            )

            text_self_results, text_self_per_label = compute_label_recall_at_k(
                sensor_embeddings=test_sensor_emb,  # Not used for text-only direction
                text_embeddings=test_text_emb,  # Use original (non-projected) text embeddings
                labels=labels_for_retrieval,
                k_values=[10, 50, 100],
                directions=['text2text'],
                normalize=True,
                verbose=True,
                exclude_self=True,
                return_per_label=True
            )

            instance_retrieval_results = {**cross_instance_results, **text_self_results}
            instance_per_label = {**cross_instance_per_label, **text_self_per_label}

            # Initialize confusion data (will be populated later)
            instance_confusion_data = {}

            # Create instance per-label charts (always create these, even if prototype fails)
            print("\n🎨 Creating instance-to-instance per-label retrieval heatmaps...")

            # Instance-to-instance: Text -> Sensor
            self.create_per_label_retrieval_heatmap_instance(
                query_embeddings=test_text_emb_proj,
                target_embeddings=test_sensor_emb,
                query_labels=labels_for_retrieval,  # SAME labels for both (aligned!)
                target_labels=labels_for_retrieval,  # SAME labels for both (aligned!)
                direction_name=f'Text → Sensor ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'perlabel_text2sensor_{retrieval_label_level.lower()}.png')
            )

            # Instance-to-instance: Sensor -> Text
            self.create_per_label_retrieval_heatmap_instance(
                query_embeddings=test_sensor_emb,
                target_embeddings=test_text_emb_proj,
                query_labels=labels_for_retrieval,  # SAME labels for both (aligned!)
                target_labels=labels_for_retrieval,  # SAME labels for both (aligned!)
                direction_name=f'Sensor → Text ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'perlabel_sensor2text_{retrieval_label_level.lower()}.png')
            )

            # Instance-to-instance: Sensor -> Sensor
            self.create_per_label_retrieval_heatmap_instance(
                query_embeddings=test_sensor_emb,
                target_embeddings=test_sensor_emb,
                query_labels=labels_for_retrieval,
                target_labels=labels_for_retrieval,
                direction_name=f'Sensor → Sensor ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'perlabel_sensor2sensor_{retrieval_label_level.lower()}.png')
            )

            # Instance-to-instance: Text -> Text (non-projected)
            self.create_per_label_retrieval_heatmap_instance(
                query_embeddings=test_text_emb,
                target_embeddings=test_text_emb,
                query_labels=labels_for_retrieval,
                target_labels=labels_for_retrieval,
                direction_name=f'Text → Text ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'perlabel_text2text_{retrieval_label_level.lower()}.png')
            )

            # Compute and visualize retrieval confusion (error analysis)
            print("\n🎨 Creating retrieval confusion analysis (error analysis)...")

            # Text -> Sensor confusion
            text2sensor_confusion = compute_retrieval_confusion(
                query_embeddings=test_text_emb_proj,
                target_embeddings=test_sensor_emb,
                query_labels=labels_for_retrieval,  # SAME aligned labels
                target_labels=labels_for_retrieval,  # SAME aligned labels
                k=50
            )

            self.create_retrieval_confusion_heatmap(
                retrieval_confusion=text2sensor_confusion,
                direction_name=f'Text → Sensor ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'confusion_text2sensor_{retrieval_label_level.lower()}.png')
            )

            # Sensor -> Text confusion
            sensor2text_confusion = compute_retrieval_confusion(
                query_embeddings=test_sensor_emb,
                target_embeddings=test_text_emb_proj,
                query_labels=labels_for_retrieval,  # SAME aligned labels
                target_labels=labels_for_retrieval,  # SAME aligned labels
                k=50
            )

            self.create_retrieval_confusion_heatmap(
                retrieval_confusion=sensor2text_confusion,
                direction_name=f'Sensor → Text ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'confusion_sensor2text_{retrieval_label_level.lower()}.png')
            )

            # Store confusion data for JSON export
            instance_confusion_data = {
                'text2sensor': text2sensor_confusion,
                'sensor2text': sensor2text_confusion
            }

            # Sensor -> Sensor confusion
            sensor2sensor_confusion = compute_retrieval_confusion(
                query_embeddings=test_sensor_emb,
                target_embeddings=test_sensor_emb,
                query_labels=labels_for_retrieval,
                target_labels=labels_for_retrieval,
                k=50
            )

            self.create_retrieval_confusion_heatmap(
                retrieval_confusion=sensor2sensor_confusion,
                direction_name=f'Sensor → Sensor ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'confusion_sensor2sensor_{retrieval_label_level.lower()}.png')
            )

            # Text -> Text confusion
            text2text_confusion = compute_retrieval_confusion(
                query_embeddings=test_text_emb,
                target_embeddings=test_text_emb,
                query_labels=labels_for_retrieval,
                target_labels=labels_for_retrieval,
                k=50
            )

            self.create_retrieval_confusion_heatmap(
                retrieval_confusion=text2text_confusion,
                direction_name=f'Text → Text ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'confusion_text2text_{retrieval_label_level.lower()}.png')
            )

            instance_confusion_data.update({
                'sensor2sensor': sensor2sensor_confusion,
                'text2text': text2text_confusion
            })

            # Prototype-based retrieval
            print("\n📊 Computing prototype-based retrieval...")

            # Initialize prototype data (will be populated if successful)
            prototype_retrieval_results = {}
            prototype_per_label = {}
            prototype_confusion_data = {}

            # Load text prototypes from metadata
            try:
                label_to_text = load_text_prototypes_from_metadata(
                    metadata_path='metadata/casas_metadata.json',
                    dataset_name=self.dataset_name,
                    style='sourish'
                )
                print(f"✅ Loaded {len(label_to_text)} label descriptions from metadata")

                # Filter prototypes based on retrieval label level
                if retrieval_label_level == 'L1':
                    # For L1: only use L1 labels (filter to keep only labels in train_labels_l1)
                    l1_label_set = set(train_labels_l1)
                    label_to_text_filtered = {k: v for k, v in label_to_text.items() if k in l1_label_set}
                    print(f"🔍 Filtered to {len(label_to_text_filtered)} L1 prototypes (from {len(label_to_text)} total)")
                else:
                    # For L2: only use L2 labels (filter to keep only labels in train_labels_l2)
                    l2_label_set = set(train_labels_l2)
                    label_to_text_filtered = {k: v for k, v in label_to_text.items() if k in l2_label_set}
                    print(f"🔍 Filtered to {len(label_to_text_filtered)} L2 prototypes (from {len(label_to_text)} total)")

                # Encode text prototypes
                prototype_emb, prototype_labels = encode_text_prototypes(
                    label_to_text=label_to_text_filtered,
                    text_encoder=self.text_encoder,
                    device=str(self.device),
                    normalize=True
                )
                print(f"✅ Encoded {len(prototype_emb)} text prototypes for {retrieval_label_level}")

                # Compute label counts from the target dataset for weighted averaging
                label_counts_dict = {}
                for label in labels_for_retrieval:
                    label_str = str(label)
                    label_counts_dict[label_str] = label_counts_dict.get(label_str, 0) + 1

                # Compute prototype retrieval metrics
                # Note: Prototypes use their own labels from metadata, targets use consistent aligned labels
                prototype_retrieval_results, prototype_per_label = compute_prototype_retrieval_metrics(
                    prototype_embeddings=prototype_emb,
                    prototype_labels=prototype_labels,
                    sensor_embeddings=test_sensor_emb,
                    text_embeddings=test_text_emb_proj,  # Use projected text embeddings
                    target_labels=labels_for_retrieval,  # Use THE SAME aligned labels
                    label_counts=label_counts_dict,  # Pass label prevalence for weighted averaging
                    k_values=[10, 50, 100],
                    directions=['prototype2sensor', 'prototype2text'],
                    normalize=True,
                    verbose=True,
                    return_per_label=True
                )

                # Combine all retrieval results
                all_retrieval_results = {**instance_retrieval_results, **prototype_retrieval_results}

                # Create retrieval visualizations
                print(f"\n🎨 Creating retrieval metric visualizations for {retrieval_label_level}...")
                self.create_retrieval_metrics_visualization(
                    retrieval_results=all_retrieval_results,
                    save_path=str(retrieval_dir / f'metrics_comparison_{retrieval_label_level.lower()}.png'),
                    label_level=retrieval_label_level
                )

                # Create prototype per-label retrieval heatmaps
                print("\n🎨 Creating prototype per-label retrieval heatmaps...")

                # Prototype-based: Prototype -> Sensor
                self.create_per_label_retrieval_heatmap(
                    prototype_labels=prototype_labels,
                    target_embeddings=test_sensor_emb,
                    target_labels=labels_for_retrieval,  # SAME aligned labels
                    prototype_embeddings=prototype_emb,
                    direction_name=f'Prototype → Sensor ({retrieval_label_level})',
                    k=50,
                    save_path=str(retrieval_dir / f'perlabel_prototype2sensor_{retrieval_label_level.lower()}.png')
                )

                # Prototype-based: Prototype -> Text
                self.create_per_label_retrieval_heatmap(
                    prototype_labels=prototype_labels,
                    target_embeddings=test_text_emb_proj,
                    target_labels=labels_for_retrieval,  # SAME aligned labels
                    prototype_embeddings=prototype_emb,
                    direction_name=f'Prototype → Text ({retrieval_label_level})',
                    k=50,
                    save_path=str(retrieval_dir / f'perlabel_prototype2text_{retrieval_label_level.lower()}.png')
                )

                # Create prototype confusion heatmaps
                print("\n🎨 Creating prototype retrieval confusion analysis...")

                # Prototype -> Sensor confusion
                # Treat each prototype as a single query
                proto2sensor_confusion = {}
                similarities = compute_cosine_similarity(prototype_emb, test_sensor_emb)

                for i, proto_label in enumerate(prototype_labels):
                    proto_sims = similarities[i]
                    top_k_indices = np.argsort(proto_sims)[-50:][::-1]
                    top_k_labels = labels_for_retrieval[top_k_indices]

                    # Count distribution
                    label_counts = {}
                    for label in top_k_labels:
                        label_str = str(label)
                        label_counts[label_str] = label_counts.get(label_str, 0) + 1

                    # Convert to proportions
                    total = len(top_k_labels)
                    label_proportions = {
                        label: count / total for label, count in label_counts.items()
                    }
                    proto2sensor_confusion[str(proto_label)] = label_proportions

                self.create_retrieval_confusion_heatmap(
                    retrieval_confusion=proto2sensor_confusion,
                    direction_name=f'Prototype → Sensor ({retrieval_label_level})',
                    k=50,
                    save_path=str(retrieval_dir / f'confusion_prototype2sensor_{retrieval_label_level.lower()}.png')
                )

                # Prototype -> Text confusion
                proto2text_confusion = {}
                similarities = compute_cosine_similarity(prototype_emb, test_text_emb_proj)

                for i, proto_label in enumerate(prototype_labels):
                    proto_sims = similarities[i]
                    top_k_indices = np.argsort(proto_sims)[-50:][::-1]
                    top_k_labels = labels_for_retrieval[top_k_indices]

                    # Count distribution
                    label_counts = {}
                    for label in top_k_labels:
                        label_str = str(label)
                        label_counts[label_str] = label_counts.get(label_str, 0) + 1

                    # Convert to proportions
                    total = len(top_k_labels)
                    label_proportions = {
                        label: count / total for label, count in label_counts.items()
                    }
                    proto2text_confusion[str(proto_label)] = label_proportions

                self.create_retrieval_confusion_heatmap(
                    retrieval_confusion=proto2text_confusion,
                    direction_name=f'Prototype → Text ({retrieval_label_level})',
                    k=50,
                    save_path=str(retrieval_dir / f'confusion_prototype2text_{retrieval_label_level.lower()}.png')
                )

                # Store prototype confusion data for JSON export
                prototype_confusion_data = {
                    'prototype2sensor': proto2sensor_confusion,
                    'prototype2text': proto2text_confusion
                }

            except Exception as e:
                print(f"⚠️  Could not compute prototype retrieval metrics: {e}")
                import traceback
                traceback.print_exc()
                all_retrieval_results = instance_retrieval_results

                # Store results for this label level
                retrieval_results_by_level[retrieval_label_level] = {
                    'instance_to_instance': {
                        'overall': instance_retrieval_results,
                        'per_label': instance_per_label,
                        'confusion': instance_confusion_data
                    },
                    'prototype_based': {
                        'overall': {},
                        'per_label': {},
                        'confusion': {}
                    }
                }
                continue  # Skip to next label level

            # Store results for this label level (successful case)
            retrieval_results_by_level[retrieval_label_level] = {
                'instance_to_instance': {
                    'overall': instance_retrieval_results,
                    'per_label': instance_per_label,
                    'confusion': instance_confusion_data
                },
                'prototype_based': {
                    'overall': prototype_retrieval_results,
                    'per_label': prototype_per_label,
                    'confusion': prototype_confusion_data
                }
            }

            print(f"\n✅ Completed retrieval metrics for {retrieval_label_level} labels")

        # End of label level loop

        # ===== 8. SAVE RESULTS =====
        if save_results:
            results_summary = {
                'classification_metrics': {
                'text_noproj': {
                    'l1': {
                        'accuracy': metrics_l1_text_noproj.get('accuracy', 0),
                        'f1_weighted': metrics_l1_text_noproj.get('f1_weighted', 0),
                        'f1_macro': metrics_l1_text_noproj.get('f1_macro', 0),
                    },
                    'l2': {
                        'accuracy': metrics_l2_text_noproj.get('accuracy', 0),
                        'f1_weighted': metrics_l2_text_noproj.get('f1_weighted', 0),
                        'f1_macro': metrics_l2_text_noproj.get('f1_macro', 0),
                    }
                },
                'text_proj': {
                    'l1': {
                        'accuracy': metrics_l1_text_proj.get('accuracy', 0),
                        'f1_weighted': metrics_l1_text_proj.get('f1_weighted', 0),
                        'f1_macro': metrics_l1_text_proj.get('f1_macro', 0),
                    },
                    'l2': {
                        'accuracy': metrics_l2_text_proj.get('accuracy', 0),
                        'f1_weighted': metrics_l2_text_proj.get('f1_weighted', 0),
                        'f1_macro': metrics_l2_text_proj.get('f1_macro', 0),
                    }
                },
                'sensor': {
                    'l1': {
                        'accuracy': metrics_l1_sensor.get('accuracy', 0),
                        'f1_weighted': metrics_l1_sensor.get('f1_weighted', 0),
                        'f1_macro': metrics_l1_sensor.get('f1_macro', 0),
                    },
                    'l2': {
                        'accuracy': metrics_l2_sensor.get('accuracy', 0),
                        'f1_weighted': metrics_l2_sensor.get('f1_weighted', 0),
                        'f1_macro': metrics_l2_sensor.get('f1_macro', 0),
                    }
                }
                },
                'retrieval_metrics': retrieval_results_by_level
            }

            with open(output_dir / 'comprehensive_results.json', 'w') as f:
                json.dump(results_summary, f, indent=2)

            print(f"💾 Results saved: {output_dir / 'comprehensive_results.json'}")

        # ===== 9. PRINT SUMMARY WITH F1 WEIGHTED FOCUS =====
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE EVALUATION SUMMARY")
        print("="*80)
        print("Note: All evaluations use synthetic text prototypes (unprojected for 'No Proj', projected for others)")

        # Classification Metrics Summary

        # L1 Summary
        print("\n📊 L1 PRIMARY ACTIVITIES:")
        print(f"{'Model':<25} {'F1 Weighted':<15} {'Accuracy':<12} {'F1 Macro':<12}")
        print("-" * 64)

        text_noproj_l1_f1w = metrics_l1_text_noproj.get('f1_weighted', 0)
        text_proj_l1_f1w = metrics_l1_text_proj.get('f1_weighted', 0)
        sensor_l1_f1w = metrics_l1_sensor.get('f1_weighted', 0)

        print(f"{'Text (No Projection)':<25} {text_noproj_l1_f1w:<15.4f} "
              f"{metrics_l1_text_noproj.get('accuracy', 0):<12.4f} "
              f"{metrics_l1_text_noproj.get('f1_macro', 0):<12.4f}")
        print(f"{'Text (With Projection)':<25} {text_proj_l1_f1w:<15.4f} "
              f"{metrics_l1_text_proj.get('accuracy', 0):<12.4f} "
              f"{metrics_l1_text_proj.get('f1_macro', 0):<12.4f}")
        print(f"{'Sensor':<25} {sensor_l1_f1w:<15.4f} "
              f"{metrics_l1_sensor.get('accuracy', 0):<12.4f} "
              f"{metrics_l1_sensor.get('f1_macro', 0):<12.4f}")

        # L2 Summary
        print("\n📊 L2 SECONDARY ACTIVITIES:")
        print(f"{'Model':<25} {'F1 Weighted':<15} {'Accuracy':<12} {'F1 Macro':<12}")
        print("-" * 64)

        text_noproj_l2_f1w = metrics_l2_text_noproj.get('f1_weighted', 0)
        text_proj_l2_f1w = metrics_l2_text_proj.get('f1_weighted', 0)
        sensor_l2_f1w = metrics_l2_sensor.get('f1_weighted', 0)

        print(f"{'Text (No Projection)':<25} {text_noproj_l2_f1w:<15.4f} "
              f"{metrics_l2_text_noproj.get('accuracy', 0):<12.4f} "
              f"{metrics_l2_text_noproj.get('f1_macro', 0):<12.4f}")
        print(f"{'Text (With Projection)':<25} {text_proj_l2_f1w:<15.4f} "
              f"{metrics_l2_text_proj.get('accuracy', 0):<12.4f} "
              f"{metrics_l2_text_proj.get('f1_macro', 0):<12.4f}")
        print(f"{'Sensor':<25} {sensor_l2_f1w:<15.4f} "
              f"{metrics_l2_sensor.get('accuracy', 0):<12.4f} "
              f"{metrics_l2_sensor.get('f1_macro', 0):<12.4f}")

        # Deltas
        print("\n📈 F1 WEIGHTED IMPROVEMENTS:")
        print(f"{'Comparison':<50} {'L1 Delta':<12} {'L2 Delta':<12}")
        print("-" * 74)

        proj_vs_noproj_l1 = text_proj_l1_f1w - text_noproj_l1_f1w
        proj_vs_noproj_l2 = text_proj_l2_f1w - text_noproj_l2_f1w
        sensor_vs_noproj_l1 = sensor_l1_f1w - text_noproj_l1_f1w
        sensor_vs_noproj_l2 = sensor_l2_f1w - text_noproj_l2_f1w
        sensor_vs_proj_l1 = sensor_l1_f1w - text_proj_l1_f1w
        sensor_vs_proj_l2 = sensor_l2_f1w - text_proj_l2_f1w

        print(f"{'Text Projection vs No Projection':<50} {proj_vs_noproj_l1:+<12.4f} {proj_vs_noproj_l2:+<12.4f}")
        print(f"{'Sensor vs Text (No Projection)':<50} {sensor_vs_noproj_l1:+<12.4f} {sensor_vs_noproj_l2:+<12.4f}")
        print(f"{'Sensor vs Text (With Projection)':<50} {sensor_vs_proj_l1:+<12.4f} {sensor_vs_proj_l2:+<12.4f}")

        # Retrieval Metrics Summary
        print("\n" + "="*80)
        print("📊 RETRIEVAL METRICS SUMMARY (Instance Recall@K + Prototype Precision@K)")
        print("="*80)

        # Display results for both L1 and L2 label levels
        for label_level in ['L1', 'L2']:
            if label_level in retrieval_results_by_level:
                level_data = retrieval_results_by_level[label_level]

                # Combine instance and prototype results for display
                all_results = {}
                if 'instance_to_instance' in level_data and 'overall' in level_data['instance_to_instance']:
                    all_results.update(level_data['instance_to_instance']['overall'])
                if 'prototype_based' in level_data and 'overall' in level_data['prototype_based']:
                    all_results.update(level_data['prototype_based']['overall'])

                if all_results:
                    print(f"\n{label_level} Labels:")
                    print("-" * 80)

                    # Choose middle K value for summary
                    k_vals = list(next(iter(all_results.values())).keys())
                    middle_k = k_vals[len(k_vals)//2] if k_vals else 10

                    print(f"\nLabel Score @ K={middle_k} (Instance recall, Prototype precision):")
                    print(f"{'Direction':<30} {'Macro':<15} {'Weighted':<15}")
                    print("-" * 60)

                    display_map = {
                        'text2sensor': 'Text → Sensor',
                        'sensor2text': 'Sensor → Text',
                        'text2text': 'Text → Text',
                        'sensor2sensor': 'Sensor → Sensor',
                        'prototype2sensor': 'Prototype → Sensor',
                        'prototype2text': 'Prototype → Text'
                    }

                    for direction in ['text2sensor', 'sensor2text', 'text2text', 'sensor2sensor', 'prototype2sensor', 'prototype2text']:
                        if direction in all_results:
                            metrics = all_results[direction].get(middle_k, {})
                            # Handle both old format (single float) and new format (dict)
                            if isinstance(metrics, dict):
                                macro = metrics.get('macro', 0)
                                weighted = metrics.get('weighted', 0)
                            else:
                                # Backward compatibility
                                macro = weighted = metrics

                            direction_display = display_map.get(direction, direction.replace('2', ' → ').replace('prototype', 'Prototype').replace('text', 'Text').replace('sensor', 'Sensor'))
                            print(f"{direction_display:<30} {macro:<.4f} ({macro*100:>5.2f}%)  {weighted:<.4f} ({weighted*100:>5.2f}%)")

        print(f"\n✅ All results saved in: {output_dir}")

        return {
            'text_noproj': metrics_dict['text_noproj'],
            'text_proj': metrics_dict['text_proj'],
            'sensor': metrics_dict['sensor']
        }

    def run_retrieval_only_evaluation(self,
                                      train_text_embeddings_path: str,
                                      test_text_embeddings_path: str,
                                      max_samples: int = 10000,
                                      filter_noisy_labels: bool = False,
                                      save_results: bool = True) -> Dict[str, Any]:
        """Run ONLY retrieval metrics evaluation (skip classification).

        This is a lightweight version that only computes:
        - Instance-to-instance retrieval (text <-> sensor)
        - Prototype-based retrieval (prototype -> text, prototype -> sensor)
        - Retrieval visualizations

        Skips:
        - Classification metrics (accuracy, F1, confusion matrices)
        - Most visualizations (t-SNE, per-class F1 plots)
        """
        print("\n" + "="*80)
        print("RETRIEVAL-ONLY EVALUATION")
        print("Computing retrieval metrics without classification")
        print("="*80)

        output_dir = Path(self.config.get('output_dir', './retrieval_evaluation'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create retrieval subdirectory for better organization
        retrieval_dir = output_dir / 'retrieval'
        retrieval_dir.mkdir(parents=True, exist_ok=True)

        # ===== 1. LOAD DATA =====
        print("\n" + "="*60)
        print("1. LOADING LABELS AND EMBEDDINGS")
        print("="*60)

        # Get labels from metadata
        train_labels_l1, train_labels_l2 = self.get_labels_from_metadata(self.dataset_name)
        print(f"✅ Got {len(train_labels_l1)} L1 labels and {len(train_labels_l2)} L2 labels from metadata")

        # Load TEST text embeddings
        test_text_emb, test_text_sample_ids, test_labels_l1, test_labels_l2 = \
            self.load_text_embeddings_from_file(
                test_text_embeddings_path,
                self.config['test_data_path'],
                max_samples
            )

        # Extract TEST sensor embeddings
        test_sensor_emb, test_sensor_l1, test_sensor_l2, test_sensor_sample_ids = self.extract_embeddings_and_labels('test', max_samples)

        # CRITICAL: Align sensor and text embeddings by sample_id BEFORE any processing
        test_sensor_emb, test_text_emb, aligned_labels_l1, aligned_labels_l2, aligned_sample_ids = \
            self.align_embeddings_by_sample_id(
                sensor_emb=test_sensor_emb,
                sensor_labels_l1=test_sensor_l1,
                sensor_labels_l2=test_sensor_l2,
                sensor_sample_ids=test_sensor_sample_ids,
                text_emb=test_text_emb,
                text_labels_l1=test_labels_l1,
                text_labels_l2=test_labels_l2,
                text_sample_ids=test_text_sample_ids
            )

        # Now use the aligned data for all subsequent processing
        test_sensor_l1 = aligned_labels_l1
        test_sensor_l2 = aligned_labels_l2
        test_labels_l1 = aligned_labels_l1  # Same labels for both
        test_labels_l2 = aligned_labels_l2  # Same labels for both

        # Filter if needed
        if filter_noisy_labels:
            noisy_l1_labels = {'no_activity', 'No_Activity', 'Other', 'other', 'unknown', 'Unknown'}
            noisy_l2_labels = {'no_activity', 'No_Activity', 'Other', 'other', 'unknown', 'Unknown'}
            print("⚠️  Filtering noisy labels from test data...")
            test_text_emb, test_labels_l1, test_labels_l2, _ = \
                self.filter_noisy_labels(test_text_emb, test_labels_l1, test_labels_l2)
            test_sensor_emb, test_sensor_l1, test_sensor_l2, _ = \
                self.filter_noisy_labels(test_sensor_emb, test_sensor_l1, test_sensor_l2)
            train_labels_l1 = [label for label in train_labels_l1 if label not in noisy_l1_labels]
            train_labels_l2 = [label for label in train_labels_l2 if label not in noisy_l2_labels]

        # ===== 2. CREATE TEXT PROTOTYPES (PROJECTED) =====
        print("\n" + "="*60)
        print("2. CREATING TEXT PROTOTYPES (WITH PROJECTION)")
        print("="*60)

        prototypes_l1, _ = self.create_text_prototypes(train_labels_l1, apply_projection=False)
        prototypes_l2, _ = self.create_text_prototypes(train_labels_l2, apply_projection=False)

        prototypes_l1_projected, _ = self.create_text_prototypes(train_labels_l1, apply_projection=True)
        prototypes_l2_projected, _ = self.create_text_prototypes(train_labels_l2, apply_projection=True)

        # ===== 3. APPLY PROJECTION TO TEXT EMBEDDINGS =====
        print("\n" + "="*60)
        print("3. PROJECTING TEXT EMBEDDINGS")
        print("="*60)

        test_text_emb_proj = self.apply_projection_to_embeddings(test_text_emb)

        # ===== 4. COMPUTE RETRIEVAL METRICS =====
        print("\n" + "="*60)
        print("4. COMPUTING RETRIEVAL METRICS (L1 and L2 Labels)")
        print("="*60)

        retrieval_results_by_level = {}

        # Compute metrics for both L1 and L2 labels
        for retrieval_label_level in ['L1', 'L2']:
            print(f"\n{'='*70}")
            print(f"Computing retrieval metrics for {retrieval_label_level} labels")
            print(f"{'='*70}")

            if retrieval_label_level == 'L1':
                # Note: test_labels_l1 and test_sensor_l1 are NOW THE SAME after alignment
                labels_for_retrieval = np.array(test_labels_l1)
                prototypes = prototypes_l1
                prototypes_projected = prototypes_l1_projected
                print(f"📋 {len(set(test_labels_l1))} unique {retrieval_label_level} labels")
            else:
                # Note: test_labels_l2 and test_sensor_l2 are NOW THE SAME after alignment
                labels_for_retrieval = np.array(test_labels_l2)
                prototypes = prototypes_l2
                prototypes_projected = prototypes_l2_projected
                print(f"📋 {len(set(test_labels_l2))} unique {retrieval_label_level} labels")

            # Instance-to-instance retrieval (text <-> sensor)
            print("\n📊 Computing instance-to-instance retrieval...")
            instance_retrieval_results, instance_per_label = compute_label_recall_at_k(
                sensor_embeddings=test_sensor_emb,
                text_embeddings=test_text_emb_proj,
                labels=labels_for_retrieval,  # Use THE SAME labels for both (aligned!)
                k_values=[10, 50, 100],
                directions=['text2sensor', 'sensor2text', 'sensor2sensor', 'text2text'],
                exclude_self=True,
                return_per_label=True
            )

            # Prototype-based retrieval
            print("\n📊 Computing prototype-based retrieval...")
            proto_sensor_labels = None
            proto_sensor_embeddings = None
            proto_text_labels = None
            proto_text_embeddings = None
            try:
                if retrieval_label_level == 'L1':
                    proto_projected_dict = prototypes_l1_projected
                    proto_non_projected_dict = prototypes_l1
                else:
                    proto_projected_dict = prototypes_l2_projected
                    proto_non_projected_dict = prototypes_l2

                projected_prototype_labels = np.array(list(proto_projected_dict.keys()))
                projected_prototype_embeddings = np.vstack(
                    [proto_projected_dict[label] for label in projected_prototype_labels]
                )

                non_projected_prototype_labels = np.array(list(proto_non_projected_dict.keys()))
                non_projected_prototype_embeddings = np.vstack(
                    [proto_non_projected_dict[label] for label in non_projected_prototype_labels]
                )

                proto_sensor_labels = projected_prototype_labels
                proto_sensor_embeddings = projected_prototype_embeddings
                proto_text_labels = non_projected_prototype_labels
                proto_text_embeddings = non_projected_prototype_embeddings

                # Compute label counts from the target dataset for weighted averaging
                label_counts_dict = {}
                for label in labels_for_retrieval:
                    label_str = str(label)
                    label_counts_dict[label_str] = label_counts_dict.get(label_str, 0) + 1

                prototype_retrieval_results: Dict[str, Dict[int, float]] = {}
                prototype_per_label: Dict[str, Dict[int, Dict[str, float]]] = {}

                # Projected prototypes query sensor embeddings
                sensor_results, sensor_per_label = compute_prototype_retrieval_metrics(
                    prototype_embeddings=projected_prototype_embeddings,
                    prototype_labels=projected_prototype_labels,
                    sensor_embeddings=test_sensor_emb,
                    target_labels=labels_for_retrieval,
                    label_counts=label_counts_dict,
                    k_values=[10, 50, 100],
                    directions=['prototype2sensor'],
                    normalize=True,
                    verbose=True,
                    return_per_label=True
                )
                prototype_retrieval_results.update(sensor_results)
                prototype_per_label.update(sensor_per_label)

                # Non-projected prototypes query non-projected text embeddings
                text_results, text_per_label = compute_prototype_retrieval_metrics(
                    prototype_embeddings=non_projected_prototype_embeddings,
                    prototype_labels=non_projected_prototype_labels,
                    text_embeddings=test_text_emb,
                    target_labels=labels_for_retrieval,
                    label_counts=label_counts_dict,
                    k_values=[10, 50, 100],
                    directions=['prototype2text'],
                    normalize=True,
                    verbose=True,
                    return_per_label=True
                )
                prototype_retrieval_results.update(text_results)
                prototype_per_label.update(text_per_label)

                # Combine all retrieval results
                all_retrieval_results = {**instance_retrieval_results, **prototype_retrieval_results}
                all_per_label = {**instance_per_label, **prototype_per_label}

            except Exception as e:
                print(f"⚠️  Warning: Prototype retrieval failed: {e}")
                all_retrieval_results = instance_retrieval_results
                all_per_label = instance_per_label

            # Store results for this level
            retrieval_results_by_level[retrieval_label_level] = {
                'recall_at_k': all_retrieval_results,
                'per_label': all_per_label
            }

            # ===== 5. CREATE RETRIEVAL VISUALIZATIONS =====
            print(f"\n🎨 Creating retrieval visualizations for {retrieval_label_level}...")

            # Ensure retrieval directory exists (safety check)
            retrieval_dir.mkdir(parents=True, exist_ok=True)

            # Main retrieval metrics visualization
            self.create_retrieval_metrics_visualization(
                retrieval_results=all_retrieval_results,
                save_path=str(retrieval_dir / f'retrieval_metrics_{retrieval_label_level.lower()}.png'),
                label_level=retrieval_label_level
            )

            # Per-label heatmaps for instance retrieval
            print(f"\n🎨 Creating per-label retrieval heatmaps...")

            # Text -> Sensor
            self.create_per_label_retrieval_heatmap_instance(
                query_embeddings=test_text_emb_proj,
                target_embeddings=test_sensor_emb,
                query_labels=labels_for_retrieval,  # SAME aligned labels
                target_labels=labels_for_retrieval,  # SAME aligned labels
                direction_name=f'Text → Sensor ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'perlabel_text2sensor_{retrieval_label_level.lower()}.png')
            )

            # Sensor -> Text
            self.create_per_label_retrieval_heatmap_instance(
                query_embeddings=test_sensor_emb,
                target_embeddings=test_text_emb_proj,
                query_labels=labels_for_retrieval,  # SAME aligned labels
                target_labels=labels_for_retrieval,  # SAME aligned labels
                direction_name=f'Sensor → Text ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'perlabel_sensor2text_{retrieval_label_level.lower()}.png')
            )

            # Sensor -> Sensor (self-retrieval)
            self.create_per_label_retrieval_heatmap_instance(
                query_embeddings=test_sensor_emb,
                target_embeddings=test_sensor_emb,
                query_labels=labels_for_retrieval,
                target_labels=labels_for_retrieval,
                direction_name=f'Sensor → Sensor ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'perlabel_sensor2sensor_{retrieval_label_level.lower()}.png')
            )

            # Text -> Text (self-retrieval, non-projected)
            self.create_per_label_retrieval_heatmap_instance(
                query_embeddings=test_text_emb,
                target_embeddings=test_text_emb,
                query_labels=labels_for_retrieval,
                target_labels=labels_for_retrieval,
                direction_name=f'Text → Text ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'perlabel_text2text_{retrieval_label_level.lower()}.png')
            )

            # Retrieval confusion analysis (error analysis)
            print(f"\n🎨 Creating retrieval confusion analysis...")

            # Text -> Sensor confusion
            text2sensor_confusion = compute_retrieval_confusion(
                query_embeddings=test_text_emb_proj,
                target_embeddings=test_sensor_emb,
                query_labels=labels_for_retrieval,
                target_labels=labels_for_retrieval,
                k=50
            )

            self.create_retrieval_confusion_heatmap(
                retrieval_confusion=text2sensor_confusion,
                direction_name=f'Text → Sensor ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'confusion_text2sensor_{retrieval_label_level.lower()}.png')
            )

            # Sensor -> Text confusion
            sensor2text_confusion = compute_retrieval_confusion(
                query_embeddings=test_sensor_emb,
                target_embeddings=test_text_emb_proj,
                query_labels=labels_for_retrieval,
                target_labels=labels_for_retrieval,
                k=50
            )

            self.create_retrieval_confusion_heatmap(
                retrieval_confusion=sensor2text_confusion,
                direction_name=f'Sensor → Text ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'confusion_sensor2text_{retrieval_label_level.lower()}.png')
            )

            # Sensor -> Sensor confusion (self-retrieval)
            sensor2sensor_confusion = compute_retrieval_confusion(
                query_embeddings=test_sensor_emb,
                target_embeddings=test_sensor_emb,
                query_labels=labels_for_retrieval,
                target_labels=labels_for_retrieval,
                k=50
            )

            self.create_retrieval_confusion_heatmap(
                retrieval_confusion=sensor2sensor_confusion,
                direction_name=f'Sensor → Sensor ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'confusion_sensor2sensor_{retrieval_label_level.lower()}.png')
            )

            # Text -> Text confusion (self-retrieval, non-projected)
            text2text_confusion = compute_retrieval_confusion(
                query_embeddings=test_text_emb,
                target_embeddings=test_text_emb,
                query_labels=labels_for_retrieval,
                target_labels=labels_for_retrieval,
                k=50
            )

            self.create_retrieval_confusion_heatmap(
                retrieval_confusion=text2text_confusion,
                direction_name=f'Text → Text ({retrieval_label_level})',
                k=50,
                save_path=str(retrieval_dir / f'confusion_text2text_{retrieval_label_level.lower()}.png')
            )

            # Prototype-based heatmaps (if available)
            if ('prototype2sensor' in all_retrieval_results
                    and proto_sensor_labels is not None
                    and proto_sensor_embeddings is not None):
                print(f"\n🎨 Creating prototype retrieval visualizations...")

                # Prototype -> Sensor per-label heatmap
                self.create_per_label_retrieval_heatmap(
                    prototype_labels=proto_sensor_labels,
                    target_embeddings=test_sensor_emb,
                    target_labels=labels_for_retrieval,  # SAME aligned labels
                    prototype_embeddings=proto_sensor_embeddings,
                    direction_name=f'Prototype → Sensor ({retrieval_label_level})',
                    k=50,
                    save_path=str(retrieval_dir / f'perlabel_prototype2sensor_{retrieval_label_level.lower()}.png')
                )

            if ('prototype2text' in all_retrieval_results
                    and proto_text_labels is not None
                    and proto_text_embeddings is not None):
                self.create_per_label_retrieval_heatmap(
                    prototype_labels=proto_text_labels,
                    target_embeddings=test_text_emb,
                    target_labels=labels_for_retrieval,  # SAME aligned labels
                    prototype_embeddings=proto_text_embeddings,
                    direction_name=f'Prototype → Text ({retrieval_label_level})',
                    k=50,
                    save_path=str(retrieval_dir / f'perlabel_prototype2text_{retrieval_label_level.lower()}.png')
                )

            if (proto_sensor_embeddings is not None and
                    'prototype2sensor' in all_retrieval_results):
                # Prototype confusion heatmaps
                print(f"\n🎨 Creating prototype retrieval confusion analysis...")

                # Prototype -> Sensor confusion
                proto2sensor_confusion = {}
                similarities = compute_cosine_similarity(proto_sensor_embeddings, test_sensor_emb)

                for i, proto_label in enumerate(proto_sensor_labels):
                    proto_sims = similarities[i]
                    top_k_indices = np.argsort(proto_sims)[-50:][::-1]
                    top_k_labels = labels_for_retrieval[top_k_indices]

                    # Count distribution
                    label_counts = {}
                    for label in top_k_labels:
                        label_str = str(label)
                        label_counts[label_str] = label_counts.get(label_str, 0) + 1

                    # Convert to proportions
                    total = len(top_k_labels)
                    label_proportions = {
                        label: count / total for label, count in label_counts.items()
                    }
                    proto2sensor_confusion[str(proto_label)] = label_proportions

                self.create_retrieval_confusion_heatmap(
                    retrieval_confusion=proto2sensor_confusion,
                    direction_name=f'Prototype → Sensor ({retrieval_label_level})',
                    k=50,
                    save_path=str(retrieval_dir / f'confusion_prototype2sensor_{retrieval_label_level.lower()}.png')
                )

            if (proto_text_embeddings is not None and
                    'prototype2text' in all_retrieval_results):
                # Prototype -> Text confusion
                proto2text_confusion = {}
                similarities = compute_cosine_similarity(proto_text_embeddings, test_text_emb)

                for i, proto_label in enumerate(proto_text_labels):
                    proto_sims = similarities[i]
                    top_k_indices = np.argsort(proto_sims)[-50:][::-1]
                    top_k_labels = labels_for_retrieval[top_k_indices]

                    # Count distribution
                    label_counts = {}
                    for label in top_k_labels:
                        label_str = str(label)
                        label_counts[label_str] = label_counts.get(label_str, 0) + 1

                    # Convert to proportions
                    total = len(top_k_labels)
                    label_proportions = {
                        label: count / total for label, count in label_counts.items()
                    }
                    proto2text_confusion[str(proto_label)] = label_proportions

                self.create_retrieval_confusion_heatmap(
                    retrieval_confusion=proto2text_confusion,
                    direction_name=f'Prototype → Text ({retrieval_label_level})',
                    k=50,
                    save_path=str(retrieval_dir / f'confusion_prototype2text_{retrieval_label_level.lower()}.png')
                )

        # ===== 6. SAVE RESULTS =====
        if save_results:
            print("\n" + "="*60)
            print("6. SAVING RESULTS")
            print("="*60)

            # Convert numpy types to Python types for JSON serialization
            def convert_to_json_serializable(obj):
                """Recursively convert numpy types to Python types."""
                if isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_json_serializable(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            results_summary = {
                'retrieval_metrics': convert_to_json_serializable(retrieval_results_by_level),
                'config': {
                    'max_samples': max_samples,
                    'filter_noisy_labels': filter_noisy_labels,
                    'checkpoint': self.config.get('checkpoint_path', 'unknown'),
                    'test_data': self.config.get('test_data_path', 'unknown')
                }
            }

            results_file = retrieval_dir / 'retrieval_metrics.json'
            with open(results_file, 'w') as f:
                json.dump(results_summary, f, indent=2)

            print(f"💾 Results saved: {results_file}")

        # ===== 7. PRINT SUMMARY =====
        print("\n" + "="*80)
        print("🎯 RETRIEVAL-ONLY EVALUATION SUMMARY")
        print("="*80)

        middle_k = 50  # Use k=50 for summary
        for retrieval_label_level in ['L1', 'L2']:
            print(f"\n📊 {retrieval_label_level} RETRIEVAL METRICS (k={middle_k}):")
            print(f"{'Direction':<30} {'Macro Precision':<20} {'Weighted Precision':<20}")
            print("-" * 70)

            all_results = retrieval_results_by_level[retrieval_label_level]['recall_at_k']

            for direction in ['text2sensor', 'sensor2text', 'text2text', 'sensor2sensor', 'prototype2sensor', 'prototype2text']:
                if direction in all_results and all_results[direction]:
                    metrics = all_results[direction].get(middle_k, {})
                    if isinstance(metrics, dict):
                        macro = metrics.get('macro', 0)
                        weighted = metrics.get('weighted', 0)
                    else:
                        macro = weighted = metrics

                    direction_display = direction.replace('2', ' → ').replace('prototype', 'Prototype').replace('text', 'Text').replace('sensor', 'Sensor')
                    print(f"{direction_display:<30} {macro:<.4f} ({macro*100:>5.2f}%)  {weighted:<.4f} ({weighted*100:>5.2f}%)")

        print(f"\n✅ Retrieval-only evaluation complete! Results saved in: {output_dir}")

        return retrieval_results_by_level

    def run_evaluation(self, max_samples: int = 10000,
                      train_split: str = 'train',
                      test_split: str = 'test',
                      k_neighbors: int = 1,
                      save_results: bool = True,
                      filter_noisy_labels: bool = False,
                      compare_filtering: bool = False) -> Dict[str, Any]:
        """Run complete embedding evaluation pipeline."""

        print("🚀 Starting embedding evaluation...")
        print(f"   Train split: {train_split}")
        print(f"   Test split: {test_split}")
        print(f"   Max samples: {max_samples}")
        print(f"   K-neighbors: {k_neighbors}")

        results = {}

        # 1. Get labels from metadata (efficient, no need to load training data)
        print("\n" + "="*60)
        print("1. EXTRACTING LABELS FROM METADATA")
        print("="*60)

        # Get all labels from metadata without loading/processing training data
        train_labels_l1, train_labels_l2 = self.get_labels_from_metadata(self.dataset_name)

        print(f"✅ Found {len(train_labels_l1)} L1 labels and {len(train_labels_l2)} L2 labels")

        # Filter noisy labels from metadata if requested
        if filter_noisy_labels:
            print("⚠️  Filtering noisy labels from metadata before creating prototypes...")

            # Define labels to exclude (case-insensitive)
            exclude_labels = {
                'other',
                'no_activity', 'No_Activity',
                'unknown', 'none', 'null', 'nan',
                'no activity', 'other activity', 'miscellaneous', 'misc'
            }

            # Filter L1 labels
            original_l1_count = len(train_labels_l1)
            train_labels_l1 = [label for label in train_labels_l1
                              if label.lower().strip() not in exclude_labels]

            # Filter L2 labels
            original_l2_count = len(train_labels_l2)
            train_labels_l2 = [label for label in train_labels_l2
                              if label.lower().strip() not in exclude_labels]

            print(f"   L1 labels: {original_l1_count} → {len(train_labels_l1)} (removed {original_l1_count - len(train_labels_l1)})")
            print(f"   L2 labels: {original_l2_count} → {len(train_labels_l2)} (removed {original_l2_count - len(train_labels_l2)})")

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

        test_embeddings, test_labels_l1, test_labels_l2, test_sample_ids = self.extract_embeddings_and_labels(
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
    parser.add_argument('--output_dir', type=str, default='results/evals/milan/embedding_evaluation',
                       help='Output directory for results')
    parser.add_argument('--filter_noisy_labels', action='store_true',
                       help='Filter out noisy labels like "Other" and "No_Activity"')
    parser.add_argument('--compare_filtering', action='store_true',
                       help='Compare filtered vs unfiltered results in a single chart')
    parser.add_argument('--run_nshot', action='store_true',
                       help='Run comprehensive n-shot evaluation (0-shot, 1-shot, 2-shot, 5-shot)')
    parser.add_argument('--n_shot_values', type=int, nargs='+', default=[1, 2, 5],
                       help='N-shot values to evaluate (default: 1 2 5)')
    parser.add_argument('--eval_text_embeddings', action='store_true',
                       help='Evaluate text embeddings (from .npz files) instead of sensor embeddings')
    parser.add_argument('--train_text_embeddings', type=str, default=None,
                       help='Path to training text embeddings .npz file')
    parser.add_argument('--test_text_embeddings', type=str, default=None,
                       help='Path to test text embeddings .npz file')
    parser.add_argument('--eval_all', action='store_true',
                       help='Run comprehensive evaluation: sensor embeddings + text embeddings (with/without projection)')
    parser.add_argument('--only_retrieval', action='store_true',
                       help='Only compute retrieval metrics (skip classification and most visualizations)')
    parser.add_argument('--skip_classification', action='store_true',
                       help='Skip classification evaluation (only retrieval)')
    parser.add_argument('--skip_visualizations', action='store_true',
                       help='Skip visualization generation (faster evaluation)')
    parser.add_argument('--use_multiple_prototypes', action='store_true',
                       help='Use multiple prototypes per label from metadata (default: single averaged prototype)')
    parser.add_argument('--description_style', type=str, default='long_desc',
                       help='Description field to use from metadata (e.g., long_desc, short_desc, zeroshot_har_desc). Default: long_desc')

    args = parser.parse_args()

    # Configuration
    config = {
        'checkpoint_path': args.checkpoint,
        'train_data_path': args.train_data,
        'test_data_path': args.test_data,
        'output_dir': args.output_dir,
        'description_style': args.description_style,
    }

    # Only add vocab_path if the file exists
    if args.vocab and os.path.exists(args.vocab):
        config['vocab_path'] = args.vocab

    # Run evaluation
    evaluator = EmbeddingEvaluator(config)

    if args.only_retrieval:
        # Validate that text embeddings files are provided
        if not args.train_text_embeddings or not args.test_text_embeddings:
            raise ValueError("Both --train_text_embeddings and --test_text_embeddings must be provided when using --only_retrieval")

        # Run retrieval-only evaluation (skip classification)
        results = evaluator.run_retrieval_only_evaluation(
            train_text_embeddings_path=args.train_text_embeddings,
            test_text_embeddings_path=args.test_text_embeddings,
            max_samples=args.max_samples,
            filter_noisy_labels=args.filter_noisy_labels,
            save_results=True
        )
        print(f"\n✅ Retrieval-only evaluation complete! Results saved in: {args.output_dir}")
    elif args.eval_all:
        # Validate that text embeddings files are provided
        if not args.train_text_embeddings or not args.test_text_embeddings:
            raise ValueError("Both --train_text_embeddings and --test_text_embeddings must be provided when using --eval_all")

        # Run comprehensive evaluation (sensor + text with/without projection)
        results = evaluator.run_comprehensive_evaluation(
            train_text_embeddings_path=args.train_text_embeddings,
            test_text_embeddings_path=args.test_text_embeddings,
            max_samples=args.max_samples,
            k_neighbors=args.k_neighbors,
            filter_noisy_labels=args.filter_noisy_labels,
            save_results=True,
            use_multiple_prototypes=args.use_multiple_prototypes
        )
        print(f"\n✅ Comprehensive evaluation complete! Results saved in: {args.output_dir}")
    elif args.eval_text_embeddings:
        # Validate that text embeddings files are provided
        if not args.train_text_embeddings or not args.test_text_embeddings:
            raise ValueError("Both --train_text_embeddings and --test_text_embeddings must be provided when using --eval_text_embeddings")

        # Run text embedding evaluation (with and without projection)
        results = evaluator.run_text_embedding_evaluation(
            train_embeddings_path=args.train_text_embeddings,
            test_embeddings_path=args.test_text_embeddings,
            train_data_path=args.train_data,
            test_data_path=args.test_data,
            max_samples=args.max_samples,
            k_neighbors=args.k_neighbors,
            filter_noisy_labels=args.filter_noisy_labels,
            save_results=True
        )
        print(f"\n✅ Text embedding evaluation complete! Results saved in: {args.output_dir}")
    elif args.run_nshot:
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
        # Run single evaluation (sensor embeddings)
        results = evaluator.run_evaluation(
            max_samples=args.max_samples,
            k_neighbors=args.k_neighbors,
            save_results=True,
            filter_noisy_labels=args.filter_noisy_labels
        )
        print(f"\n✅ Evaluation complete! Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
