#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

"""
Comprehensive text encoder evaluation script for CASAS and MARBLE activity recognition.

This script evaluates text embeddings by:
1. Finding all embedding files in a directory
2. Creating t-SNE visualizations for all encoders (L1 and L2 labels)
3. Evaluating each encoder using text prototypes and k-NN
4. Creating confusion matrices and F1 score analysis
5. Comparing all encoders in unified plots

Sample Usage (CASAS):
  python src/evals/evaluate_text_encoder_only.py \
    --embeddings_dir data/processed/casas/milan/FL_20 \
    --captions data/processed/casas/milan/FL_20/train_captions_baseline.json \
    --data data/processed/casas/milan/FL_20/train.json \
    --output_dir results/evals/milan/FL_20 \
    --split train \
    --description_style baseline \
    --max_samples 10000 \
    --filter_noisy_labels
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import argparse
import math
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import defaultdict, Counter
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Evaluation metrics
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Transformers for CLIP
from transformers import CLIPTextModel, CLIPTokenizer

# Local imports
from utils.device_utils import get_optimal_device, log_device_info
from utils.label_utils import convert_labels_to_text
from text_encoders import (
    GTETextEncoder,
    DistilRoBERTaTextEncoder,
    LLAMATextEncoder,
    MiniLMTextEncoder,
    EmbeddingGemmaTextEncoder,
    CLIPTextEncoder,
    SigLIPTextEncoder,
    TextEncoderConfig
)


def extract_metadata_from_paths(embeddings_path: str) -> Dict[str, str]:
    """Extract metadata from file paths."""
    path = Path(embeddings_path)
    parts = path.parts
    dataset_name = 'unknown'

    for i, part in enumerate(parts):
        if part == 'casas' and i + 1 < len(parts):
            dataset_name = parts[i + 1]
            break
        elif part == 'marble':
            dataset_name = 'marble'
            break

    is_presegmented = 'presegmented' in str(path)
    filename = path.stem
    parts = filename.split('_')
    split = parts[0] if parts else 'train'

    caption_style = 'baseline'
    encoder_name = 'gte'

    if 'embeddings' in parts:
        emb_idx = parts.index('embeddings')
        if emb_idx + 1 < len(parts):
            encoder_name = parts[-1]
            if emb_idx + 2 < len(parts):
                caption_style = '_'.join(parts[emb_idx + 1:-1])
            elif emb_idx + 1 < len(parts) - 1:
                caption_style = parts[emb_idx + 1]

    return {
        'dataset_name': dataset_name,
        'split': split,
        'is_presegmented': is_presegmented,
        'caption_style': caption_style,
        'encoder_name': encoder_name
    }


def load_label_colors(dataset='milan') -> Tuple[Dict, Dict]:
    """Load label colors from metadata."""
    try:
        metadata_path = Path(__file__).parent.parent.parent / "metadata" / "casas_metadata.json"
        with open(metadata_path, 'r') as f:
            city_metadata = json.load(f)

        dataset_metadata = city_metadata.get(dataset, {})
        label_colors = dataset_metadata.get('label', dataset_metadata.get('label_color', dataset_metadata.get('lable', {})))
        label_colors_l2 = dataset_metadata.get('label_deepcasas_color', {})

        print(f"🎨 Loaded {len(label_colors)} L1 colors, {len(label_colors_l2)} L2 colors")
        return label_colors, label_colors_l2
    except Exception as e:
        print(f"⚠️  Could not load label colors: {e}")
        return {}, {}


def load_embeddings_and_labels(
    embeddings_path: str,
    captions_path: str,
    data_path: str = None,
    max_samples: int = None
) -> Tuple[np.ndarray, List[str], List[str], List[str], List[str]]:
    """Load embeddings and corresponding labels."""
    print(f"\n📖 Loading embeddings from: {embeddings_path}")
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
        print(f"   Filtered to {embeddings.shape[0]} unique samples")

    print(f"   Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    print(f"   Encoder: {data.get('encoder_type', ['unknown'])[0]}")
    print(f"   Model: {data.get('model_name', ['unknown'])[0]}")

    print(f"\n📖 Loading captions from: {captions_path}")
    with open(captions_path, 'r') as f:
        captions_data = json.load(f)

    if 'captions' in captions_data and isinstance(captions_data['captions'], list):
        samples_captions = captions_data['captions']
    elif 'samples' in captions_data:
        samples_captions = captions_data['samples']
    else:
        samples_captions = captions_data

    caption_map = {}
    for sample in samples_captions:
        sample_id = sample.get('sample_id')
        if sample_id:
            caption_map[sample_id] = sample.get('captions', [''])[0]

    if data_path is None:
        captions_path_obj = Path(captions_path)
        data_path = captions_path_obj.parent / captions_path_obj.name.replace('_captions_', '_').replace('captions_', '').replace('.json', '.json')
        if '_baseline' in str(data_path) or '_sourish' in str(data_path):
            filename = captions_path_obj.stem
            if '_captions_' in filename:
                split = filename.split('_captions_')[0]
                data_path = captions_path_obj.parent / f"{split}.json"

    print(f"\n📖 Loading labels from: {data_path}")
    with open(data_path, 'r') as f:
        data_json = json.load(f)

    if 'samples' in data_json:
        samples_data = data_json['samples']
    else:
        samples_data = data_json

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

    labels_l1 = []
    labels_l2 = []
    captions = []
    valid_indices = []

    for i, sample_id in enumerate(sample_ids_from_emb):
        sample_id_str = str(sample_id)
        if sample_id_str in label_map:
            labels_l1.append(label_map[sample_id_str]['label_l1'])
            labels_l2.append(label_map[sample_id_str]['label_l2'])
            captions.append(caption_map.get(sample_id_str, ''))
            valid_indices.append(i)

    embeddings = embeddings[valid_indices]
    sample_ids = [str(sample_ids_from_emb[i]) for i in valid_indices]

    print(f"   Matched {len(labels_l1)} samples with labels and captions")

    if max_samples and len(embeddings) > max_samples:
        print(f"\n🎲 Sampling {max_samples} from {len(embeddings)} samples (seed=42)")
        np.random.seed(42)
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        sample_ids = [sample_ids[i] for i in indices]
        labels_l1 = [labels_l1[i] for i in indices]
        labels_l2 = [labels_l2[i] for i in indices]
        captions = [captions[i] for i in indices]

    print(f"\n📊 Label distribution (L1):")
    label_counts = Counter(labels_l1)
    for label, count in label_counts.most_common(15):
        print(f"   {label}: {count}")

    return embeddings, sample_ids, labels_l1, labels_l2, captions


def filter_noisy_labels(captions: List[str], labels_l1: List[str], labels_l2: List[str],
                       embeddings: np.ndarray = None, return_indices: bool = False) -> Tuple:
    """Filter out noisy/uninformative labels.

    Args:
        return_indices: If True, returns the valid_indices array as the last element
    """
    exclude_labels = {
        'other', 'no_activity', 'unknown', 'none', 'null', 'nan',
        'no activity', 'other activity', 'miscellaneous', 'misc', 'unlabeled'
    }

    valid_indices = []
    for i, (l1, l2) in enumerate(zip(labels_l1, labels_l2)):
        l1_lower = l1.lower().strip()
        l2_lower = l2.lower().strip()
        if l1_lower not in exclude_labels and l2_lower not in exclude_labels:
            valid_indices.append(i)

    valid_indices = np.array(valid_indices)

    if len(valid_indices) == 0:
        print("⚠️  Warning: All samples filtered out!")
        if return_indices:
            return (captions, labels_l1, labels_l2, embeddings, np.arange(len(captions))) if embeddings is not None else (captions, labels_l1, labels_l2, np.arange(len(captions)))
        if embeddings is not None:
            return captions, labels_l1, labels_l2, embeddings
        return captions, labels_l1, labels_l2

    filtered_captions = [captions[i] for i in valid_indices]
    filtered_labels_l1 = [labels_l1[i] for i in valid_indices]
    filtered_labels_l2 = [labels_l2[i] for i in valid_indices]

    print(f"🧹 Filtered out noisy labels:")
    print(f"   Original: {len(captions)} → Filtered: {len(filtered_captions)} (removed: {len(captions) - len(filtered_captions)})")

    removed_l1 = Counter([labels_l1[i] for i in range(len(labels_l1)) if i not in valid_indices])
    if removed_l1:
        print(f"   Removed L1 labels: {dict(removed_l1.most_common())}")

    if embeddings is not None:
        filtered_embeddings = embeddings[valid_indices]
        if return_indices:
            return filtered_captions, filtered_labels_l1, filtered_labels_l2, filtered_embeddings, valid_indices
        return filtered_captions, filtered_labels_l1, filtered_labels_l2, filtered_embeddings

    if return_indices:
        return filtered_captions, filtered_labels_l1, filtered_labels_l2, valid_indices
    return filtered_captions, filtered_labels_l1, filtered_labels_l2


def create_text_prototypes(labels: List[str], encoder_type: str, model_name: str,
                          embedding_dim: int,
                          description_style: str = "baseline",
                          house_name: str = "milan",
                          use_projection: bool = False,
                          projection_dim: int = None) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Create text-based prototypes by encoding label descriptions."""
    print(f"🔄 Creating text-based label prototypes (style: {description_style}, encoder: {encoder_type})...")
    label_counts = Counter(labels)
    unique_labels = list(label_counts.keys())

    print(f"📝 Encoding {len(unique_labels)} unique labels with {encoder_type} encoder...")

    if house_name == 'marble':
        description_style = 'sourish'
        print(f"   Using 'sourish' description style for MARBLE labels")

    label_descriptions_lists = convert_labels_to_text(unique_labels, single_description=False,
                                                      house_name=house_name,
                                                      description_style=description_style)

    device = get_optimal_device()

    # Map encoder type to encoder class
    encoder_map = {
        'clip': CLIPTextEncoder,
        'gte': GTETextEncoder,
        'distilroberta': DistilRoBERTaTextEncoder,
        'llama': LLAMATextEncoder,
        'minilm': MiniLMTextEncoder,
        'embeddinggemma': EmbeddingGemmaTextEncoder,
        'siglip': SigLIPTextEncoder,
    }

    encoder_class = encoder_map.get(encoder_type.lower())
    if encoder_class is None:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # Create a minimal config for the encoder
    # Include projection settings if they were used during encoding
    config_kwargs = {
        'encoder_type': encoder_type,
        'model_name': model_name,
        'embedding_dim': embedding_dim,
        'batch_size': 32,
        'normalize': True
    }

    if use_projection and projection_dim is not None:
        config_kwargs['use_projection'] = use_projection
        config_kwargs['projection_dim'] = projection_dim

    config = TextEncoderConfig(**config_kwargs)

    # Initialize encoder (it will auto-detect device)
    text_encoder = encoder_class(config)
    text_encoder.model.eval()

    # Encode descriptions and create prototypes
    prototypes = {}
    for i, label in enumerate(unique_labels):
        descriptions = label_descriptions_lists[i]
        # Use the encoder's encode method
        output = text_encoder.encode(descriptions)
        embeddings = output.embeddings
        # Average all descriptions for this label
        prototype_embedding = np.mean(embeddings, axis=0)
        prototypes[label] = prototype_embedding

    embedding_dim = embeddings.shape[1]
    print(f"✅ Created {len(prototypes)} text-based prototypes ({embedding_dim}-dim)")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        descriptions = label_descriptions_lists[unique_labels.index(label)]
        print(f"    {label}: {count} samples → {len(descriptions)} descriptions")

    return prototypes, dict(label_counts)


def predict_labels_knn(query_embeddings: np.ndarray, prototypes: Dict[str, np.ndarray], k: int = 1) -> List[str]:
    """Predict labels using k-nearest neighbors."""
    print(f"🔄 Predicting labels using {k}-NN comparison...")

    prototype_labels = list(prototypes.keys())
    prototype_embeddings = np.array([prototypes[label] for label in prototype_labels])

    query_embeddings_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    prototype_embeddings_norm = prototype_embeddings / (np.linalg.norm(prototype_embeddings, axis=1, keepdims=True) + 1e-8)

    similarities = np.dot(query_embeddings_norm, prototype_embeddings_norm.T)

    if k == 1:
        nearest_indices = np.argmax(similarities, axis=1)
        predictions = [prototype_labels[idx] for idx in nearest_indices]
    else:
        top_k_indices = np.argsort(similarities, axis=1)[:, -k:]
        predictions = []
        for indices in top_k_indices:
            top_labels = [prototype_labels[idx] for idx in indices]
            prediction = Counter(top_labels).most_common(1)[0][0]
            predictions.append(prediction)

    return predictions


def evaluate_predictions(true_labels: List[str], pred_labels: List[str], label_type: str) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    print(f"🔄 Computing metrics for {label_type} labels...")

    valid_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels))
                     if true != 'Unknown' and pred != 'Unknown']

    if not valid_indices:
        print(f"⚠️  No valid predictions for {label_type}")
        return {}

    true_filtered = [true_labels[i] for i in valid_indices]
    pred_filtered = [pred_labels[i] for i in valid_indices]
    unique_labels = sorted(list(set(true_filtered + pred_filtered)))

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

    per_class_f1 = f1_score(true_filtered, pred_filtered, average=None, zero_division=0, labels=unique_labels)
    metrics['per_class_f1'] = dict(zip(unique_labels, per_class_f1))

    per_class_precision = precision_score(true_filtered, pred_filtered, average=None, zero_division=0, labels=unique_labels)
    per_class_recall = recall_score(true_filtered, pred_filtered, average=None, zero_division=0, labels=unique_labels)
    metrics['per_class_precision'] = dict(zip(unique_labels, per_class_precision))
    metrics['per_class_recall'] = dict(zip(unique_labels, per_class_recall))

    metrics['classification_report'] = classification_report(
        true_filtered, pred_filtered, target_names=unique_labels, zero_division=0, output_dict=True
    )

    metrics['confusion_matrix'] = confusion_matrix(true_filtered, pred_filtered, labels=unique_labels)

    print(f"✅ {label_type} Metrics:")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    F1 (Macro): {metrics['f1_macro']:.4f}")
    print(f"    F1 (Weighted): {metrics['f1_weighted']:.4f}")

    return metrics


def create_tsne_comparison_grid(embeddings_dir: str, captions_path: str, data_path: str,
                                output_prefix: str, label_colors: Dict, label_colors_l2: Dict,
                                sample_id_to_label_l1: Dict[str, str], sample_id_to_label_l2: Dict[str, str],
                                selected_sample_ids: List[str], split: str, max_samples: int = 10000, perplexity: int = 30):
    """Create t-SNE comparison grids for embeddings from a specific split.

    Args:
        selected_sample_ids: List of sample IDs to use (instead of indices, for proper matching across files)
    """
    print("\n" + "="*80)
    print("CREATING T-SNE COMPARISON GRID")
    print("="*80)

    embeddings_dir = Path(embeddings_dir)
    # Only get embedding files for the specified split
    embedding_files = sorted(embeddings_dir.glob(f"{split}_embeddings_*.npz"))

    if not embedding_files:
        print(f"❌ No embedding files found in {embeddings_dir}")
        return

    print(f"\n📂 Found {len(embedding_files)} embedding files:")
    for f in embedding_files:
        print(f"   - {f.name}")

    first_metadata = extract_metadata_from_paths(str(embedding_files[0]))
    dataset_name = first_metadata['dataset_name']

    n_encoders = len(embedding_files)
    n_cols = min(3, n_encoders)
    n_rows = math.ceil(n_encoders / n_cols)

    fig_l1, axes_l1 = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 7*n_rows))
    fig_l2, axes_l2 = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 7*n_rows))

    if n_encoders == 1:
        axes_l1 = [axes_l1]
        axes_l2 = [axes_l2]
    else:
        axes_l1 = axes_l1.flatten() if n_encoders > 1 else [axes_l1]
        axes_l2 = axes_l2.flatten() if n_encoders > 1 else [axes_l2]

    for idx, emb_file in enumerate(embedding_files):
        print(f"\n🔄 Processing {emb_file.name} for t-SNE...")
        metadata = extract_metadata_from_paths(str(emb_file))
        encoder_name = metadata['encoder_name'].upper()
        data = np.load(emb_file)
        embeddings = data['embeddings']
        file_sample_ids_raw = data['sample_ids']

        # Check if this is multi-caption format
        if 'caption_indices' in data:
            caption_indices = data['caption_indices']
            # Keep only first caption for evaluation
            first_caption_mask = caption_indices == 0
            embeddings = embeddings[first_caption_mask]
            file_sample_ids_raw = file_sample_ids_raw[first_caption_mask]

        file_sample_ids = [str(sid) for sid in file_sample_ids_raw]

        # Match sample_ids to get correct embeddings for this file
        file_sample_id_to_idx = {sid: i for i, sid in enumerate(file_sample_ids)}
        matching_indices = []
        matched_sample_ids = []  # Track which sample_ids we actually found

        for sid in selected_sample_ids:
            if sid in file_sample_id_to_idx:
                matching_indices.append(file_sample_id_to_idx[sid])
                matched_sample_ids.append(sid)

        matching_indices = np.array(matching_indices)
        embeddings = embeddings[matching_indices]

        # Get labels for the matched samples by looking them up by sample_id
        labels_l1_tsne = [sample_id_to_label_l1[sid] for sid in matched_sample_ids]
        labels_l2_tsne = [sample_id_to_label_l2[sid] for sid in matched_sample_ids]

        print(f"   Matched {len(matching_indices)}/{len(selected_sample_ids)} samples for {encoder_name}")
        if len(matching_indices) != len(selected_sample_ids):
            print(f"   ⚠️  Using {len(labels_l1_tsne)} labels matching the found embeddings")

        print(f"   Running t-SNE on {len(embeddings)} samples...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, verbose=0)
        projection = tsne.fit_transform(embeddings)

        # Use the filtered labels that match this encoder's embeddings
        ax_l1 = axes_l1[idx]
        unique_labels_l1 = sorted(set(labels_l1_tsne))
        for label in unique_labels_l1:
            mask = np.array(labels_l1_tsne) == label
            color = label_colors.get(label, plt.cm.tab20(len([l for l in unique_labels_l1 if l < label]) % 20))
            ax_l1.scatter(projection[mask, 0], projection[mask, 1], c=[color], label=label.replace('_', ' '),
                         alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
        ax_l1.set_title(f'{encoder_name}', fontsize=12, fontweight='bold')
        ax_l1.set_xlabel('t-SNE 1', fontsize=10)
        ax_l1.set_ylabel('t-SNE 2', fontsize=10)
        ax_l1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax_l1.grid(True, alpha=0.3)

        ax_l2 = axes_l2[idx]
        unique_labels_l2 = sorted(set(labels_l2_tsne))
        for label in unique_labels_l2:
            mask = np.array(labels_l2_tsne) == label
            color = label_colors_l2.get(label, plt.cm.tab10(len([l for l in unique_labels_l2 if l < label]) % 10))
            ax_l2.scatter(projection[mask, 0], projection[mask, 1], c=[color], label=label.replace('_', ' '),
                         alpha=0.6, s=20, edgecolors='white', linewidth=0.3)
        ax_l2.set_title(f'{encoder_name}', fontsize=12, fontweight='bold')
        ax_l2.set_xlabel('t-SNE 1', fontsize=10)
        ax_l2.set_ylabel('t-SNE 2', fontsize=10)
        ax_l2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax_l2.grid(True, alpha=0.3)

    for idx in range(n_encoders, len(axes_l1)):
        axes_l1[idx].set_visible(False)
        axes_l2[idx].set_visible(False)

    preseg_text = "presegmented" if first_metadata['is_presegmented'] else "standard"
    split = first_metadata['split']
    caption_style = first_metadata['caption_style']

    fig_l1.suptitle(f'Embedding Comparison: {dataset_name.capitalize()} ({split} - {preseg_text}), L1 Labels\n'
                    f'{caption_style.capitalize()} Captions - All Encoders',
                    fontsize=16, fontweight='bold', y=0.995)

    fig_l2.suptitle(f'Embedding Comparison: {dataset_name.capitalize()} ({split} - {preseg_text}), L2 Labels\n'
                    f'{caption_style.capitalize()} Captions - All Encoders',
                    fontsize=16, fontweight='bold', y=0.995)

    fig_l1.tight_layout()
    fig_l2.tight_layout()

    output_l1 = f"{output_prefix}_tsne_comparison_l1.png"
    output_l2 = f"{output_prefix}_tsne_comparison_l2.png"

    fig_l1.savefig(output_l1, dpi=150, bbox_inches='tight')
    fig_l2.savefig(output_l2, dpi=150, bbox_inches='tight')

    print(f"\n💾 Saved L1 t-SNE comparison to: {output_l1}")
    print(f"💾 Saved L2 t-SNE comparison to: {output_l2}")

    plt.close(fig_l1)
    plt.close(fig_l2)


def create_confusion_matrix_plot(confusion_matrix_data: np.ndarray, labels: List[str],
                                title: str, save_path: str = None):
    """Create confusion matrix visualization."""
    if len(labels) > 15:
        print(f"⚠️  Too many classes ({len(labels)}), showing top 15")
        labels = labels[:15]
        confusion_matrix_data = confusion_matrix_data[:15, :15]

    fig, ax = plt.subplots(figsize=(12, 10))
    cm_normalized = confusion_matrix_data.astype('float') / (confusion_matrix_data.sum(axis=1)[:, np.newaxis] + 1e-8)

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels,
                yticklabels=labels, ax=ax, cbar_kws={'label': 'Normalized Count'})

    ax.set_title(f'Confusion Matrix - {title}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Confusion matrix saved: {save_path}")

    plt.close(fig)


def create_encoder_comparison_plot(all_results: Dict[str, Dict], save_path: str):
    """Create comparison plot across all encoders."""
    print("\n🔄 Creating encoder comparison plot...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Text Encoder Comparison - All Encoders', fontsize=16, fontweight='bold')

    encoder_names = []
    l1_f1_macro = []
    l1_f1_weighted = []
    l1_accuracy = []
    l2_f1_macro = []
    l2_f1_weighted = []
    l2_accuracy = []

    for encoder_name, results in sorted(all_results.items()):
        encoder_names.append(encoder_name)
        l1_f1_macro.append(results['metrics_l1'].get('f1_macro', 0))
        l1_f1_weighted.append(results['metrics_l1'].get('f1_weighted', 0))
        l1_accuracy.append(results['metrics_l1'].get('accuracy', 0))
        l2_f1_macro.append(results['metrics_l2'].get('f1_macro', 0))
        l2_f1_weighted.append(results['metrics_l2'].get('f1_weighted', 0))
        l2_accuracy.append(results['metrics_l2'].get('accuracy', 0))

    x = np.arange(len(encoder_names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, l1_f1_macro, width, label='F1 Macro', alpha=0.8)
    bars2 = ax1.bar(x + width/2, l1_f1_weighted, width, label='F1 Weighted', alpha=0.8)
    ax1.set_xlabel('Encoder')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('L1 (Primary) Labels - F1 Scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels(encoder_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.02:
                ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    bars1 = ax2.bar(x - width/2, l2_f1_macro, width, label='F1 Macro', alpha=0.8)
    bars2 = ax2.bar(x + width/2, l2_f1_weighted, width, label='F1 Weighted', alpha=0.8)
    ax2.set_xlabel('Encoder')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('L2 (Secondary) Labels - F1 Scores')
    ax2.set_xticks(x)
    ax2.set_xticklabels(encoder_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.02:
                ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    bars = ax3.bar(encoder_names, l1_accuracy, alpha=0.8)
    ax3.set_xlabel('Encoder')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('L1 (Primary) Labels - Accuracy')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        if height > 0.02:
            ax3.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    bars = ax4.bar(encoder_names, l2_accuracy, alpha=0.8)
    ax4.set_xlabel('Encoder')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('L2 (Secondary) Labels - Accuracy')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        if height > 0.02:
            ax4.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Encoder comparison plot saved: {save_path}")
    plt.close(fig)


def run_comprehensive_evaluation(embeddings_dir: str, captions_path: str, data_path: str,
                                 output_dir: str, split: str = 'train', description_style: str = 'baseline',
                                 house_name: str = 'milan', max_samples: int = 10000, k_neighbors: int = 1,
                                 filter_noisy: bool = True, perplexity: int = 30, verbose: bool = False):
    """Run comprehensive evaluation for all encoders in a directory."""
    print("="*80)
    print("COMPREHENSIVE TEXT ENCODER EVALUATION")
    print("="*80)
    print(f"📂 Embeddings directory: {embeddings_dir}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📊 Split: {split}")
    print(f"🎨 Description style: {description_style}")
    print(f"🏠 House name: {house_name}")
    print(f"🔢 Max samples: {max_samples}")
    print(f"🔍 K-neighbors: {k_neighbors}")
    print(f"🧹 Filter noisy labels: {filter_noisy}")

    # Create output directory with text_only/description_style structure
    output_path = Path(output_dir) / "text_only" / description_style
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 Full output path: {output_path}")

    embeddings_dir_path = Path(embeddings_dir)
    # Look for embedding files matching the split and description style
    pattern = f"{split}_embeddings_{description_style}_*.npz"
    embedding_files = sorted(embeddings_dir_path.glob(pattern))

    if not embedding_files:
        print(f"\n❌ No embedding files found matching pattern: {pattern}")
        print(f"   Looking in: {embeddings_dir_path}")
        # Try broader pattern to see what files exist
        all_files = list(embeddings_dir_path.glob(f"{split}_embeddings_*.npz"))
        if all_files:
            print(f"\n   Found {len(all_files)} files with broader pattern:")
            for f in all_files:
                print(f"     - {f.name}")
            print(f"\n   💡 Tip: Make sure description_style matches the embedding file names")
        return

    print(f"\n📂 Found {len(embedding_files)} embedding files for split '{split}' with style '{description_style}':")
    for f in embedding_files:
        print(f"   - {f.name}")

    first_metadata = extract_metadata_from_paths(str(embedding_files[0]))
    dataset_name = first_metadata['dataset_name']

    label_colors, label_colors_l2 = load_label_colors(dataset_name)

    print("\n" + "="*80)
    print("LOADING LABELS AND CAPTIONS")
    print("="*80)

    _, sample_ids_all, labels_l1_all, labels_l2_all, captions_all = load_embeddings_and_labels(
        str(embedding_files[0]), captions_path, data_path=data_path, max_samples=None
    )

    # Track sample_ids through filtering and sampling (ensure they're strings)
    sample_ids_all = [str(sid) for sid in sample_ids_all]

    if filter_noisy:
        captions_all, labels_l1_all, labels_l2_all, filter_indices = filter_noisy_labels(
            captions_all, labels_l1_all, labels_l2_all, return_indices=True
        )
        sample_ids_all = [sample_ids_all[i] for i in filter_indices]

    if max_samples and len(labels_l1_all) > max_samples:
        print(f"\n🎲 Sampling {max_samples} from {len(labels_l1_all)} samples (seed=42)")
        np.random.seed(42)
        sample_subset_indices = np.random.choice(len(labels_l1_all), max_samples, replace=False)
        selected_sample_ids = [sample_ids_all[i] for i in sample_subset_indices]
        selected_labels_l1 = [labels_l1_all[i] for i in sample_subset_indices]
        selected_labels_l2 = [labels_l2_all[i] for i in sample_subset_indices]
        captions = [captions_all[i] for i in sample_subset_indices]
    else:
        selected_sample_ids = sample_ids_all
        selected_labels_l1 = labels_l1_all
        selected_labels_l2 = labels_l2_all
        captions = captions_all

    # Create mappings from sample_id to labels (CRITICAL for correct label assignment!)
    sample_id_to_label_l1 = dict(zip(selected_sample_ids, selected_labels_l1))
    sample_id_to_label_l2 = dict(zip(selected_sample_ids, selected_labels_l2))

    print(f"\n✅ Selected {len(selected_sample_ids)} samples for evaluation")
    print(f"   First 5 sample IDs: {selected_sample_ids[:5]}")

    all_results = {}

    print("\n" + "="*80)
    print("EVALUATING ALL ENCODERS")
    print("="*80)

    for emb_file in embedding_files:
        encoder_metadata = extract_metadata_from_paths(str(emb_file))
        encoder_name = encoder_metadata['encoder_name']

        print(f"\n{'='*80}")
        print(f"EVALUATING: {encoder_name.upper()}")
        print(f"{'='*80}")

        data = np.load(emb_file)
        embeddings = data['embeddings']
        file_sample_ids_raw = data['sample_ids']

        # Check if this is multi-caption format
        if 'caption_indices' in data:
            caption_indices = data['caption_indices']
            # Keep only first caption for evaluation
            first_caption_mask = caption_indices == 0
            embeddings = embeddings[first_caption_mask]
            file_sample_ids_raw = file_sample_ids_raw[first_caption_mask]

        file_sample_ids = [str(sid) for sid in file_sample_ids_raw]
        encoder_type = str(data.get('encoder_type', ['unknown'])[0])
        model_name = str(data.get('model_name', ['unknown'])[0])

        # Find indices in THIS file that match our selected sample_ids
        # Create a mapping from sample_id to index in this file
        file_sample_id_to_idx = {sid: idx for idx, sid in enumerate(file_sample_ids)}

        # Get indices for our selected samples AND their corresponding labels
        # CRITICAL: Look up labels by sample_id, not by position!
        matching_indices = []
        matched_sample_ids = []
        for sid in selected_sample_ids:
            if sid in file_sample_id_to_idx:
                matching_indices.append(file_sample_id_to_idx[sid])
                matched_sample_ids.append(sid)

        matching_indices = np.array(matching_indices)

        # Get labels for the matched samples by looking them up by sample_id
        labels_l1_encoder = [sample_id_to_label_l1[sid] for sid in matched_sample_ids]
        labels_l2_encoder = [sample_id_to_label_l2[sid] for sid in matched_sample_ids]

        if len(matching_indices) != len(selected_sample_ids):
            print(f"⚠️  Warning: Only found {len(matching_indices)}/{len(selected_sample_ids)} matching samples in {encoder_name}")
        else:
            print(f"✅ Matched all {len(matching_indices)} samples in {encoder_name}")

        embeddings = embeddings[matching_indices]

        # Extract projection metadata if it exists in the embeddings file
        use_projection = bool(data.get('use_projection', [False])[0]) if 'use_projection' in data else False
        projection_dim = int(data.get('projection_dim', [None])[0]) if 'projection_dim' in data else None

        # Get embedding dimension - use the native dim from metadata if projection was used
        # Otherwise use the actual embedding dimension
        if use_projection and 'embedding_dim' in data:
            # Native dimension (before projection)
            native_dim = int(data.get('embedding_dim', [embeddings.shape[1]])[0])
            embedding_dim = native_dim
        else:
            # Actual dimension (no projection)
            embedding_dim = embeddings.shape[1]

        # Try to create prototypes - this is where model loading happens
        print(f"\n📊 Creating prototypes for {encoder_name} ({model_name})...")
        if use_projection and projection_dim:
            print(f"   Using projection: {embedding_dim}-dim → {projection_dim}-dim")
        try:
            prototypes_l1, counts_l1 = create_text_prototypes(labels_l1_encoder, encoder_type, model_name,
                                                              embedding_dim, description_style, house_name,
                                                              use_projection, projection_dim)
            prototypes_l2, counts_l2 = create_text_prototypes(labels_l2_encoder, encoder_type, model_name,
                                                              embedding_dim, description_style, house_name,
                                                              use_projection, projection_dim)
        except Exception as e:
            print(f"❌ Failed to create prototypes for {encoder_name}: {e}")
            print(f"   This usually means the model type is not supported by your transformers version.")
            print(f"   Skipping {encoder_name}...")
            continue

        print(f"\n📊 Predicting labels for {encoder_name}...")

        # Check dimension compatibility
        proto_dim_l1 = next(iter(prototypes_l1.values())).shape[0]
        emb_dim = embeddings.shape[1]

        if proto_dim_l1 != emb_dim:
            print(f"⚠️  Warning: Dimension mismatch for {encoder_name}!")
            print(f"   Embeddings: {emb_dim}-dim, Prototypes: {proto_dim_l1}-dim")
            print(f"   This usually means the embeddings file was created with projection.")
            print(f"   Skipping {encoder_name}...")
            continue

        pred_labels_l1 = predict_labels_knn(embeddings, prototypes_l1, k_neighbors)
        pred_labels_l2 = predict_labels_knn(embeddings, prototypes_l2, k_neighbors)

        metrics_l1 = evaluate_predictions(labels_l1_encoder, pred_labels_l1, f"{encoder_name} L1")
        metrics_l2 = evaluate_predictions(labels_l2_encoder, pred_labels_l2, f"{encoder_name} L2")

        all_results[encoder_name] = {
            'metrics_l1': metrics_l1,
            'metrics_l2': metrics_l2,
            'predictions_l1': pred_labels_l1,
            'predictions_l2': pred_labels_l2
        }

        # Save confusion matrices with encoder prefix in the same folder (no subfolders)
        if 'confusion_matrix' in metrics_l1:
            create_confusion_matrix_plot(metrics_l1['confusion_matrix'], metrics_l1['unique_labels'],
                                        f'{encoder_name} - L1 Primary Activities',
                                        str(output_path / f'{encoder_name}_confusion_matrix_l1.png'))

        if 'confusion_matrix' in metrics_l2:
            create_confusion_matrix_plot(metrics_l2['confusion_matrix'], metrics_l2['unique_labels'],
                                        f'{encoder_name} - L2 Secondary Activities',
                                        str(output_path / f'{encoder_name}_confusion_matrix_l2.png'))

        encoder_results = {
            'encoder_name': encoder_name,
            'split': split,
            'dataset': dataset_name,
            'description_style': description_style,
            'house_name': house_name,
            'max_samples': len(embeddings),
            'k_neighbors': k_neighbors,
            'metrics_l1': {
                'accuracy': metrics_l1.get('accuracy', 0),
                'f1_macro': metrics_l1.get('f1_macro', 0),
                'f1_weighted': metrics_l1.get('f1_weighted', 0),
                'precision_macro': metrics_l1.get('precision_macro', 0),
                'recall_macro': metrics_l1.get('recall_macro', 0),
                'num_classes': metrics_l1.get('num_classes', 0),
                'num_samples': metrics_l1.get('num_samples', 0),
                'per_class_f1': metrics_l1.get('per_class_f1', {})
            },
            'metrics_l2': {
                'accuracy': metrics_l2.get('accuracy', 0),
                'f1_macro': metrics_l2.get('f1_macro', 0),
                'f1_weighted': metrics_l2.get('f1_weighted', 0),
                'precision_macro': metrics_l2.get('precision_macro', 0),
                'recall_macro': metrics_l2.get('recall_macro', 0),
                'num_classes': metrics_l2.get('num_classes', 0),
                'num_samples': metrics_l2.get('num_samples', 0),
                'per_class_f1': metrics_l2.get('per_class_f1', {})
            }
        }

        # Save results with encoder prefix in the same folder (no subfolders)
        with open(output_path / f'{encoder_name}_results.json', 'w') as f:
            json.dump(encoder_results, f, indent=2)

        print(f"💾 Saved {encoder_name} results to: {output_path / f'{encoder_name}_results.json'}")

    print("\n" + "="*80)
    print("CREATING T-SNE VISUALIZATIONS")
    print("="*80)

    create_tsne_comparison_grid(embeddings_dir=embeddings_dir, captions_path=captions_path, data_path=data_path,
                                output_prefix=str(output_path / f"{split}_all_encoders"),
                                label_colors=label_colors, label_colors_l2=label_colors_l2,
                                sample_id_to_label_l1=sample_id_to_label_l1, sample_id_to_label_l2=sample_id_to_label_l2,
                                selected_sample_ids=selected_sample_ids,
                                split=split, max_samples=max_samples, perplexity=perplexity)

    print("\n" + "="*80)
    print("CREATING ENCODER COMPARISON PLOT")
    print("="*80)

    create_encoder_comparison_plot(all_results=all_results,
                                   save_path=str(output_path / f"{split}_encoder_comparison.png"))

    summary = {
        'dataset': dataset_name,
        'split': split,
        'description_style': description_style,
        'house_name': house_name,
        'num_samples': len(selected_sample_ids),
        'num_encoders': len(all_results),
        'encoders': {}
    }

    for encoder_name, results in all_results.items():
        summary['encoders'][encoder_name] = {
            'l1_f1_macro': results['metrics_l1'].get('f1_macro', 0),
            'l1_f1_weighted': results['metrics_l1'].get('f1_weighted', 0),
            'l1_accuracy': results['metrics_l1'].get('accuracy', 0),
            'l2_f1_macro': results['metrics_l2'].get('f1_macro', 0),
            'l2_f1_weighted': results['metrics_l2'].get('f1_weighted', 0),
            'l2_accuracy': results['metrics_l2'].get('accuracy', 0),
        }

    with open(output_path / f"{split}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n💾 Saved summary to: {output_path / f'{split}_summary.json'}")

    print("\n" + "="*80)
    print("🎯 EVALUATION COMPLETE - SUMMARY")
    print("="*80)

    print(f"\n📊 Dataset: {dataset_name} ({split})")
    print(f"📊 Samples: {len(selected_sample_ids)}")
    print(f"📊 Encoders evaluated: {len(all_results)}")

    print(f"\n{'Encoder':<15} {'L1 F1-Macro':<12} {'L1 F1-Weight':<12} {'L1 Acc':<12} {'L2 F1-Macro':<12} {'L2 F1-Weight':<12} {'L2 Acc':<12}")
    print("="*90)

    for encoder_name in sorted(all_results.keys()):
        results = all_results[encoder_name]
        print(f"{encoder_name:<15} "
              f"{results['metrics_l1'].get('f1_macro', 0):<12.4f} "
              f"{results['metrics_l1'].get('f1_weighted', 0):<12.4f} "
              f"{results['metrics_l1'].get('accuracy', 0):<12.4f} "
              f"{results['metrics_l2'].get('f1_macro', 0):<12.4f} "
              f"{results['metrics_l2'].get('f1_weighted', 0):<12.4f} "
              f"{results['metrics_l2'].get('accuracy', 0):<12.4f}")

    print(f"\n✅ All results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive text encoder evaluation with t-SNE visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
  python src/evals/evaluate_text_encoder_only.py \\
    --embeddings_dir data/processed/casas/milan/FL_20 \\
    --captions data/processed/casas/milan/FL_20/train_captions_baseline.json \\
    --data data/processed/casas/milan/FL_20/train.json \\
    --output_dir results/evals/milan/FL_20 \\
    --split train \\
    --description_style baseline \\
    --max_samples 10000 \\
    --filter_noisy_labels
        '''
    )

    parser.add_argument('--embeddings_dir', type=str, required=True,
                       help='Directory containing embedding files')
    parser.add_argument('--captions', type=str, required=True,
                       help='Path to captions JSON file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data file with labels')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                       help='Which data split to evaluate')
    parser.add_argument('--description_style', type=str, default='baseline', choices=['baseline', 'sourish'],
                       help='Style of label descriptions')
    parser.add_argument('--house_name', type=str, default='milan',
                       help='House name for label descriptions')
    parser.add_argument('--max_samples', type=int, default=10000,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--k_neighbors', type=int, default=1,
                       help='Number of neighbors for k-NN prediction')
    parser.add_argument('--filter_noisy_labels', action='store_true',
                       help='Filter out noisy labels')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    run_comprehensive_evaluation(
        embeddings_dir=args.embeddings_dir,
        captions_path=args.captions,
        data_path=args.data,
        output_dir=args.output_dir,
        split=args.split,
        description_style=args.description_style,
        house_name=args.house_name,
        max_samples=args.max_samples,
        k_neighbors=args.k_neighbors,
        filter_noisy=args.filter_noisy_labels,
        perplexity=args.perplexity,
        verbose=args.verbose
    )

    print(f"\n✅ Comprehensive evaluation complete! Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()

