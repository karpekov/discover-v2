#!/usr/bin/env python3
"""
Time-of-Day (ToD) Embedding Analysis

Analyzes whether activities have different embeddings based on time of day.
Focuses on master bedroom and master bathroom activities during morning, afternoon, evening, and night.

Time-of-Day Definitions (from src/captions/rule_based/baseline.py):
- Night: hour < 5 or hour >= 20
- Morning: 5 <= hour < 12
- Afternoon: 12 <= hour < 17
- Evening: 17 <= hour < 20

Usage:
python src/evals/tod_analysis.py \
    --checkpoint trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt \
    --train_data data/processed/casas/milan/FD_60_p/train.json \
    --test_data data/processed/casas/milan/FD_60_p/test.json \
    --vocab data/processed/casas/milan/FD_60_p/vocab.json \
    --output_dir results/evals/milan/FD_60_p/tod_analysis \
    --max_samples 5000
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import argparse
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats
from scipy.spatial.distance import cdist, cosine

# Local imports
from models.sensor_encoder import SensorEncoder
from dataio.dataset import SmartHomeDataset
from torch.utils.data import DataLoader
from utils.device_utils import get_optimal_device, log_device_info


def get_time_of_day(hour: int) -> str:
    """Convert hour to time of day category."""
    if hour < 5:
        return 'night'
    elif hour < 12:
        return 'morning'
    elif hour < 17:
        return 'afternoon'
    elif hour < 20:
        return 'evening'
    else:
        return 'night'


def load_model(checkpoint_path: str, vocab_path: str, device: torch.device):
    """Load sensor encoder model from checkpoint."""
    print(f"🔄 Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load vocab sizes from checkpoint
    vocab_sizes = checkpoint.get('vocab_sizes', {})

    # Get model config
    raw_config = checkpoint.get('config', {})
    if hasattr(raw_config, '__dataclass_fields__'):
        model_config = getattr(raw_config, 'encoder', {}) or {}
    else:
        model_config = raw_config

    # Load weights based on checkpoint format
    if 'model_state_dict' in checkpoint:
        # AlignmentModel format - extract sensor_encoder weights
        from alignment.model import AlignmentModel
        full_model = AlignmentModel.load(checkpoint_path, device=device, vocab_path=vocab_path)
        sensor_encoder = full_model.sensor_encoder
    elif 'chronos_encoder_state_dict' in checkpoint or 'sensor_encoder_state_dict' in checkpoint:
        # Old format - individual state dicts
        if 'chronos_encoder_state_dict' in checkpoint:
            from encoders.chronos_encoder import ChronosEncoder
            sensor_encoder = ChronosEncoder(
                vocab_sizes=vocab_sizes,
                chronos_model_name=model_config.get('chronos_model_name', 'amazon/chronos-2'),
                projection_hidden_dim=model_config.get('projection_hidden_dim', 256),
                projection_dropout=model_config.get('projection_dropout', 0.1),
                output_dim=model_config.get('output_dim', 512),
                sequence_length=model_config.get('sequence_length', 50)
            )
            sensor_encoder.load_state_dict(checkpoint['chronos_encoder_state_dict'])
        else:
            # Standard SensorEncoder
            sensor_encoder = SensorEncoder(
                vocab_sizes=vocab_sizes,
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
            sensor_encoder.load_state_dict(checkpoint['sensor_encoder_state_dict'])
    else:
        raise ValueError("Checkpoint format not recognized - missing both 'model_state_dict' and 'sensor_encoder_state_dict'")

    sensor_encoder.to(device)
    sensor_encoder.eval()

    print(f"✅ Model loaded successfully")
    return sensor_encoder


def extract_embeddings_and_metadata(sensor_encoder, data_path, vocab_path, device, max_samples=None):
    """Extract embeddings and ToD metadata for all samples."""
    print(f"📊 Loading data from {data_path}")

    # Load dataset
    dataset = SmartHomeDataset(
        data_path=data_path,
        vocab_path=vocab_path,
        sequence_length=50,  # Will use model's default
        max_captions=1
    )

    # Limit samples if requested
    if max_samples and len(dataset) > max_samples:
        # Truncate dataset
        dataset.data = dataset.data[:max_samples]

    print(f"📊 Processing {len(dataset)} samples")

    # Create simple dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x  # Return list of samples as-is
    )

    # Extract embeddings
    print(f"🔄 Extracting embeddings...")
    all_embeddings = []

    with torch.no_grad():
        for batch_samples in dataloader:
            # Extract sensor inputs from batch samples
            # Dataset returns: categorical_features, coordinates, time_deltas, mask, captions

            # Prepare batch tensors for sensor encoder
            sensor_inputs = {}

            # Stack categorical features into a nested dict
            categorical_features = {}
            for field in dataset.categorical_fields:
                field_data = [sample['categorical_features'][field] for sample in batch_samples]
                categorical_features[field] = torch.stack(field_data).to(device)
            sensor_inputs['categorical_features'] = categorical_features

            # Stack continuous features
            coords_list = [sample['coordinates'] for sample in batch_samples]
            sensor_inputs['coordinates'] = torch.stack(coords_list).to(device)

            delta_t_list = [sample['time_deltas'] for sample in batch_samples]
            sensor_inputs['time_deltas'] = torch.stack(delta_t_list).to(device)

            # Extract embeddings
            output = sensor_encoder(sensor_inputs)
            # Handle both EncoderOutput and raw tensor
            if hasattr(output, 'embeddings'):
                embeddings = output.embeddings
            else:
                embeddings = output
            all_embeddings.append(embeddings.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)
    print(f"✅ Extracted {len(all_embeddings)} embeddings with shape {all_embeddings.shape}")

    # Now extract ToD and metadata - need to get labels from dataset samples
    print(f"🔄 Extracting time-of-day and label metadata...")
    samples_metadata = []

    for i in range(len(all_embeddings)):
        # Get the processed sample from dataset to get labels
        dataset_sample = dataset[i]
        raw_sample = dataset.data[i]

        # Get first event timestamp
        if 'sensor_sequence' in raw_sample and len(raw_sample['sensor_sequence']) > 0:
            first_event = raw_sample['sensor_sequence'][0]
            timestamp = first_event.get('datetime') or first_event.get('timestamp')

            # Default values
            tod = 'unknown'
            hour = -1

            if timestamp:
                # Parse timestamp
                try:
                    if isinstance(timestamp, str):
                        dt = pd.to_datetime(timestamp)
                    else:
                        dt = timestamp
                    hour = dt.hour
                    tod = get_time_of_day(hour)
                except:
                    pass

            # Extract labels from dataset sample (correct location)
            label_l1 = dataset_sample.get('first_activity', 'unknown')
            label_l2 = dataset_sample.get('first_activity_l2', 'unknown')

            # Use L1 label as the primary label for analysis
            primary_label = label_l1

            # Extract room from sensor sequence
            rooms = set()
            for event in raw_sample['sensor_sequence']:
                room = event.get('room_id') or event.get('room', '')
                if room:
                    rooms.add(room)

            sample_info = {
                'sample_id': raw_sample.get('sample_id', f'sample_{i}'),
                'tod': tod,
                'hour': hour,
                'label_l1': label_l1,
                'label_l2': label_l2,
                'label': primary_label,  # Primary label for analysis
                'rooms': list(rooms),
                'primary_room': list(rooms)[0] if rooms else 'unknown'
            }
            samples_metadata.append(sample_info)
        else:
            # No sensor sequence - create dummy metadata
            samples_metadata.append({
                'sample_id': f'sample_{i}',
                'tod': 'unknown',
                'hour': -1,
                'label_l1': 'unknown',
                'label_l2': 'unknown',
                'label': 'unknown',
                'rooms': [],
                'primary_room': 'unknown'
            })

    # Filter out samples with unknown ToD
    valid_indices = [i for i, s in enumerate(samples_metadata) if s['tod'] != 'unknown']
    samples_metadata = [samples_metadata[i] for i in valid_indices]
    all_embeddings = all_embeddings[valid_indices]

    print(f"✅ Loaded {len(samples_metadata)} samples with valid ToD information")

    return all_embeddings, samples_metadata


def compute_tod_statistics(samples_with_tod, embeddings):
    """Compute statistical measures of embedding differences by ToD."""
    print(f"📊 Computing ToD statistics...")

    # Group embeddings by ToD
    tod_embeddings = defaultdict(list)
    tod_labels = defaultdict(lambda: defaultdict(list))
    tod_rooms = defaultdict(lambda: defaultdict(list))

    for i, sample_info in enumerate(samples_with_tod):
        tod = sample_info['tod']
        label = sample_info['label']  # Using L1 label
        primary_room = sample_info['primary_room']

        tod_embeddings[tod].append(embeddings[i])
        tod_labels[tod][label].append(embeddings[i])
        tod_rooms[tod][primary_room].append(embeddings[i])

    # Convert to numpy arrays
    for tod in tod_embeddings:
        tod_embeddings[tod] = np.array(tod_embeddings[tod])
        for label in tod_labels[tod]:
            tod_labels[tod][label] = np.array(tod_labels[tod][label])
        for room in tod_rooms[tod]:
            tod_rooms[tod][room] = np.array(tod_rooms[tod][room])

    stats_results = {
        'overall': {},
        'by_label': {},
        'by_room': {}
    }

    # Overall ToD statistics
    tod_order = ['morning', 'afternoon', 'evening', 'night']
    tods_present = [t for t in tod_order if t in tod_embeddings]

    print("\n=== Overall ToD Statistics ===")
    for tod in tods_present:
        n_samples = len(tod_embeddings[tod])
        mean_emb = tod_embeddings[tod].mean(axis=0)
        std_emb = tod_embeddings[tod].std(axis=0).mean()
        norm = np.linalg.norm(mean_emb)

        stats_results['overall'][tod] = {
            'n_samples': int(n_samples),
            'mean_norm': float(norm),
            'mean_std': float(std_emb)
        }
        print(f"{tod.capitalize()}: {n_samples} samples, mean norm: {norm:.4f}, mean std: {std_emb:.4f}")

    # Pairwise ToD distances
    print("\n=== Pairwise ToD Mean Embedding Distances ===")
    tod_means = {tod: tod_embeddings[tod].mean(axis=0) for tod in tods_present}

    distance_matrix = []
    for tod1 in tods_present:
        row = []
        for tod2 in tods_present:
            if tod1 == tod2:
                dist = 0.0
            else:
                dist = float(cosine(tod_means[tod1], tod_means[tod2]))
            row.append(dist)
        distance_matrix.append(row)
        print(f"{tod1.capitalize()}: " + " | ".join([f"{tods_present[j][:3]}: {row[j]:.4f}" for j in range(len(row))]))

    stats_results['distance_matrix'] = {
        'tods': tods_present,
        'distances': distance_matrix
    }

    # Statistical tests between ToDs
    print("\n=== Statistical Tests (t-test) between ToDs ===")
    for i, tod1 in enumerate(tods_present):
        for tod2 in tods_present[i+1:]:
            # Flatten embeddings and perform t-test
            emb1_flat = tod_embeddings[tod1].flatten()
            emb2_flat = tod_embeddings[tod2].flatten()
            t_stat, p_value = stats.ttest_ind(emb1_flat, emb2_flat)
            print(f"{tod1.capitalize()} vs {tod2.capitalize()}: t={t_stat:.4f}, p={p_value:.4e}")

    # By label analysis (focus on activities with sufficient samples)
    print("\n=== Per-Label (L1) ToD Analysis ===")
    label_counts = Counter()
    for sample_info in samples_with_tod:
        label_counts[sample_info['label']] += 1

    top_labels = [label for label, count in label_counts.most_common(10)]

    for label in top_labels:
        label_tod_counts = {tod: len(tod_labels[tod].get(label, [])) for tod in tods_present}
        if sum(label_tod_counts.values()) < 20:  # Skip labels with too few samples
            continue

        print(f"\n{label}:")
        stats_results['by_label'][label] = {}

        for tod in tods_present:
            if label in tod_labels[tod] and len(tod_labels[tod][label]) > 0:
                n = len(tod_labels[tod][label])
                mean_norm = np.linalg.norm(tod_labels[tod][label].mean(axis=0))
                stats_results['by_label'][label][tod] = {
                    'n_samples': int(n),
                    'mean_norm': float(mean_norm)
                }
                print(f"  {tod.capitalize()}: {n} samples, mean norm: {mean_norm:.4f}")

    # Master bedroom/bathroom specific analysis
    print("\n=== Master Bedroom/Bathroom ToD Analysis ===")
    target_rooms = ['master_bedroom', 'master_bath', 'master_bathroom']

    for room in target_rooms:
        room_found = False
        for tod in tods_present:
            if room in tod_rooms[tod] and len(tod_rooms[tod][room]) > 0:
                room_found = True
                break

        if not room_found:
            continue

        print(f"\n{room.replace('_', ' ').title()}:")
        stats_results['by_room'][room] = {}

        for tod in tods_present:
            if room in tod_rooms[tod] and len(tod_rooms[tod][room]) > 0:
                n = len(tod_rooms[tod][room])
                mean_emb = tod_rooms[tod][room].mean(axis=0)
                mean_norm = np.linalg.norm(mean_emb)
                stats_results['by_room'][room][tod] = {
                    'n_samples': int(n),
                    'mean_norm': float(mean_norm)
                }
                print(f"  {tod.capitalize()}: {n} samples, mean norm: {mean_norm:.4f}")

        # Pairwise distances for this room across ToDs
        room_tod_means = {}
        for tod in tods_present:
            if room in tod_rooms[tod] and len(tod_rooms[tod][room]) > 0:
                room_tod_means[tod] = tod_rooms[tod][room].mean(axis=0)

        if len(room_tod_means) > 1:
            print(f"  Pairwise distances:")
            room_tods = list(room_tod_means.keys())
            for i, tod1 in enumerate(room_tods):
                for tod2 in room_tods[i+1:]:
                    dist = cosine(room_tod_means[tod1], room_tod_means[tod2])
                    print(f"    {tod1.capitalize()} vs {tod2.capitalize()}: {dist:.4f}")

    return stats_results, tod_embeddings, tod_labels, tod_rooms


def analyze_specific_activities(samples_with_tod, embeddings, output_dir, target_activities):
    """Deep dive analysis for specific activities."""
    print(f"\n{'='*80}")
    print(f"DEEP DIVE: Specific Activity Analysis")
    print(f"{'='*80}")

    # Print available labels for debugging
    label_counts = Counter([s['label'] for s in samples_with_tod])
    print(f"\nAvailable L1 labels (showing top 15):")
    for label, count in label_counts.most_common(15):
        print(f"  - {label}: {count} samples")

    # Filter samples for target activities with EXACT matching
    activity_data = defaultdict(lambda: {'embeddings': [], 'tods': [], 'sample_ids': [], 'matched_labels': []})

    for i, sample_info in enumerate(samples_with_tod):
        label = sample_info['label']
        for target in target_activities:
            # EXACT match only (case-insensitive, handle underscores)
            label_norm = label.lower().replace('_', '')
            target_norm = target.lower().replace('_', '')

            if label_norm == target_norm:
                activity_data[target]['embeddings'].append(embeddings[i])
                activity_data[target]['tods'].append(sample_info['tod'])
                activity_data[target]['sample_ids'].append(sample_info['sample_id'])
                activity_data[target]['matched_labels'].append(sample_info['label'])
                break

    # Convert to numpy arrays
    for activity in activity_data:
        activity_data[activity]['embeddings'] = np.array(activity_data[activity]['embeddings'])

    # Remove activities with too few samples (lower threshold for rare activities)
    activity_data = {k: v for k, v in activity_data.items() if len(v['embeddings']) > 5}

    if not activity_data:
        print("⚠️  No samples found for target activities")
        return

    print(f"\nFound {len(activity_data)} activities with sufficient samples:")
    for activity, data in activity_data.items():
        matched_labels = Counter(data['matched_labels'])
        print(f"  - {activity}: {len(data['embeddings'])} samples")
        print(f"    Matched labels: {dict(matched_labels)}")

    # Statistical analysis per activity
    stats_by_activity = {}

    for activity, data in activity_data.items():
        print(f"\n{'='*60}")
        print(f"Activity: {activity.upper()}")
        print(f"{'='*60}")

        embs = data['embeddings']
        tods = data['tods']

        # Overall statistics
        print(f"\nOverall Statistics:")
        print(f"  Total samples: {len(embs)}")
        print(f"  Mean embedding norm: {np.linalg.norm(embs.mean(axis=0)):.4f}")
        print(f"  Embedding std (mean across dims): {embs.std(axis=0).mean():.4f}")

        # By ToD
        tod_order = ['morning', 'afternoon', 'evening', 'night']
        tod_embs = {}
        tod_counts = Counter(tods)

        print(f"\nBy Time of Day:")
        for tod in tod_order:
            tod_mask = np.array([t == tod for t in tods])
            if tod_mask.sum() > 0:
                tod_embs[tod] = embs[tod_mask]
                mean_norm = np.linalg.norm(tod_embs[tod].mean(axis=0))
                print(f"  {tod.capitalize()}: {tod_mask.sum()} samples, mean norm: {mean_norm:.4f}")

        # Pairwise ToD distances
        if len(tod_embs) > 1:
            print(f"\nPairwise ToD Distances:")
            tod_means = {tod: tod_embs[tod].mean(axis=0) for tod in tod_embs}
            tods_present = list(tod_means.keys())

            for i, tod1 in enumerate(tods_present):
                for tod2 in tods_present[i+1:]:
                    dist = cosine(tod_means[tod1], tod_means[tod2])
                    print(f"    {tod1.capitalize()} vs {tod2.capitalize()}: {dist:.4f}")

        # Statistical tests
        if len(tod_embs) > 1:
            print(f"\nStatistical Tests (t-test):")
            tods_present = list(tod_embs.keys())
            for i, tod1 in enumerate(tods_present):
                for tod2 in tods_present[i+1:]:
                    if len(tod_embs[tod1]) > 1 and len(tod_embs[tod2]) > 1:
                        t_stat, p_value = stats.ttest_ind(
                            tod_embs[tod1].flatten(),
                            tod_embs[tod2].flatten()
                        )
                        sig = "✅" if p_value < 0.05 else "❌"
                        print(f"    {tod1.capitalize()} vs {tod2.capitalize()}: t={t_stat:.4f}, p={p_value:.4e} {sig}")

        stats_by_activity[activity] = {
            'total_samples': int(len(embs)),
            'tod_distribution': {tod: int(count) for tod, count in tod_counts.items()},
            'overall_mean_norm': float(np.linalg.norm(embs.mean(axis=0))),
            'tod_stats': {
                tod: {
                    'n_samples': int(len(tod_embs[tod])),
                    'mean_norm': float(np.linalg.norm(tod_embs[tod].mean(axis=0)))
                }
                for tod in tod_embs
            }
        }

    # Cross-activity comparison
    print(f"\n{'='*80}")
    print(f"CROSS-ACTIVITY COMPARISON")
    print(f"{'='*80}")

    activities = list(activity_data.keys())
    activity_means = {act: activity_data[act]['embeddings'].mean(axis=0) for act in activities}

    print(f"\nPairwise Activity Distances (Overall):")
    for i, act1 in enumerate(activities):
        for act2 in activities[i+1:]:
            dist = cosine(activity_means[act1], activity_means[act2])
            print(f"  {act1} ↔ {act2}: {dist:.4f}")

    # By ToD cross-activity distances
    for tod in ['morning', 'afternoon', 'evening', 'night']:
        tod_activity_means = {}
        for act in activities:
            tod_mask = np.array([t == tod for t in activity_data[act]['tods']])
            if tod_mask.sum() > 2:  # At least 3 samples
                tod_activity_means[act] = activity_data[act]['embeddings'][tod_mask].mean(axis=0)

        if len(tod_activity_means) > 1:
            print(f"\nPairwise Activity Distances ({tod.capitalize()}):")
            acts = list(tod_activity_means.keys())
            for i, act1 in enumerate(acts):
                for act2 in acts[i+1:]:
                    dist = cosine(tod_activity_means[act1], tod_activity_means[act2])
                    print(f"  {act1} ↔ {act2}: {dist:.4f}")

    # Visualizations
    print(f"\n📊 Creating activity-specific visualizations...")

    # Figure 1: t-SNE for specific activities
    all_activity_embs = []
    all_activity_labels = []
    all_activity_tods = []

    for activity, data in activity_data.items():
        all_activity_embs.append(data['embeddings'])
        all_activity_labels.extend([activity] * len(data['embeddings']))
        all_activity_tods.extend(data['tods'])

    all_activity_embs = np.vstack(all_activity_embs)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_activity_embs)//4))
    embs_2d = tsne.fit_transform(all_activity_embs)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Colored by activity
    ax = axes[0]
    colors = sns.color_palette('tab10', n_colors=len(activity_data))
    for i, activity in enumerate(activity_data.keys()):
        mask = np.array([label == activity for label in all_activity_labels])
        ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1],
                  c=[colors[i]], label=activity, alpha=0.7, s=30)
    ax.set_title('t-SNE: Specific Activities', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Colored by ToD
    ax = axes[1]
    tod_colors = {
        'morning': '#FDB462',
        'afternoon': '#FFFF99',
        'evening': '#BEBADA',
        'night': '#80B1D3'
    }
    for tod in ['morning', 'afternoon', 'evening', 'night']:
        mask = np.array([t == tod for t in all_activity_tods])
        if mask.sum() > 0:
            ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1],
                      c=tod_colors[tod], label=tod.capitalize(), alpha=0.7, s=30)
    ax.set_title('t-SNE: Specific Activities by ToD', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'specific_activities_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: Overall distance heatmap (across all ToDs)
    print(f"  Creating overall activity distance heatmap...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Compute distance matrix for all activities
    n = len(activities)
    dist_matrix = np.zeros((n, n))
    activity_counts = {}
    for i, act1 in enumerate(activities):
        activity_counts[act1] = len(activity_data[act1]['embeddings'])
        for j, act2 in enumerate(activities):
            if i == j:
                dist_matrix[i, j] = 0
            else:
                dist_matrix[i, j] = cosine(activity_means[act1], activity_means[act2])

    # Create labels with sample counts
    labels = [f"{act}\n(n={activity_counts[act]})" for act in activities]

    # Plot heatmap
    sns.heatmap(dist_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
               xticklabels=labels,
               yticklabels=labels,
               ax=ax, cbar_kws={'label': 'Cosine Distance'},
               square=True)
    ax.set_title('Overall Activity Embedding Distances\n(All Times of Day)',
                 fontsize=14, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'specific_activities_overall_distances.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: Distance heatmaps (by ToD)
    print(f"  Creating ToD-specific distance heatmaps...")
    n_activities = len(activities)
    tod_order = ['morning', 'afternoon', 'evening', 'night']

    # First, check which ToDs have enough data
    tods_with_data = []
    tod_activity_info = {}

    for tod in tod_order:
        tod_acts = []
        tod_means = {}
        tod_counts = {}
        for act in activities:
            tod_mask = np.array([t == tod for t in activity_data[act]['tods']])
            if tod_mask.sum() > 2:  # At least 3 samples
                tod_acts.append(act)
                tod_means[act] = activity_data[act]['embeddings'][tod_mask].mean(axis=0)
                tod_counts[act] = tod_mask.sum()

        if len(tod_acts) > 1:
            tods_with_data.append(tod)
            tod_activity_info[tod] = {
                'activities': tod_acts,
                'means': tod_means,
                'counts': tod_counts
            }
            print(f"    {tod.capitalize()}: {len(tod_acts)} activities")

    if len(tods_with_data) > 0:
        print(f"  Creating heatmaps for {len(tods_with_data)} ToDs: {tods_with_data}")

        # Create figure - use 2x2 grid for up to 4 ToDs
        fig = plt.figure(figsize=(16, 14))

        for idx, tod in enumerate(tods_with_data):
            ax = plt.subplot(2, 2, idx + 1)

            tod_acts = tod_activity_info[tod]['activities']
            tod_means = tod_activity_info[tod]['means']
            tod_counts = tod_activity_info[tod]['counts']

            print(f"    Processing {tod}: {tod_acts}")

            # Compute distance matrix
            n = len(tod_acts)
            dist_matrix = np.zeros((n, n))
            for i, act1 in enumerate(tod_acts):
                for j, act2 in enumerate(tod_acts):
                    if i == j:
                        dist_matrix[i, j] = 0
                    else:
                        dist_matrix[i, j] = cosine(tod_means[act1], tod_means[act2])

            # Create labels with sample counts
            labels = [f"{act.replace('_', ' ')}\n(n={tod_counts[act]})" for act in tod_acts]

            # Plot heatmap
            sns.heatmap(dist_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                       xticklabels=labels,
                       yticklabels=labels,
                       ax=ax,
                       cbar=True,
                       square=True,
                       vmin=0, vmax=0.7,
                       linewidths=0.5,
                       linecolor='gray')
            ax.set_title(f'{tod.capitalize()} - Activity Embedding Distances',
                        fontweight='bold', fontsize=14, pad=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

        plt.suptitle('Activity Embedding Distances by Time of Day\n(Cosine Distance)',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        save_path = output_dir / 'specific_activities_distance_heatmaps.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✅ Saved heatmap to {save_path}")
    else:
        print("⚠️  Not enough samples per ToD to create distance heatmaps")
        # Create a placeholder figure with explanation
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5,
                'Insufficient samples per time-of-day\nto create distance heatmaps\n\n' +
                'Need at least 3 samples per activity per ToD',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
        plt.savefig(output_dir / 'specific_activities_distance_heatmaps.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

    plt.tight_layout()
    plt.savefig(output_dir / 'specific_activities_distance_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Collect cross-activity distances for report
    cross_activity_distances = {
        'overall': {},
        'by_tod': {}
    }

    # Overall distances
    for i, act1 in enumerate(activities):
        for act2 in activities[i+1:]:
            dist = cosine(activity_means[act1], activity_means[act2])
            cross_activity_distances['overall'][f"{act1} ↔ {act2}"] = float(dist)

    # By ToD distances
    for tod in ['morning', 'afternoon', 'evening', 'night']:
        tod_activity_means = {}
        for act in activities:
            tod_mask = np.array([t == tod for t in activity_data[act]['tods']])
            if tod_mask.sum() > 2:
                tod_activity_means[act] = activity_data[act]['embeddings'][tod_mask].mean(axis=0)

        if len(tod_activity_means) > 1:
            cross_activity_distances['by_tod'][tod] = {}
            acts = list(tod_activity_means.keys())
            for i, act1 in enumerate(acts):
                for act2 in acts[i+1:]:
                    dist = cosine(tod_activity_means[act1], tod_activity_means[act2])
                    cross_activity_distances['by_tod'][tod][f"{act1} ↔ {act2}"] = float(dist)

    # Add to stats
    stats_by_activity['_cross_activity_distances'] = cross_activity_distances

    # Save statistics
    with open(output_dir / 'specific_activities_stats.json', 'w') as f:
        json.dump(stats_by_activity, f, indent=2)

    print(f"✅ Activity-specific analysis complete")

    return stats_by_activity


def visualize_tod_embeddings(samples_with_tod, embeddings, output_dir):
    """Create visualizations of ToD embedding differences."""
    print(f"📊 Creating visualizations...")

    # Extract ToD labels
    tod_labels = [s['tod'] for s in samples_with_tod]
    activity_labels = [s['label'] for s in samples_with_tod]  # Using L1 labels
    primary_rooms = [s['primary_room'] for s in samples_with_tod]

    # Create color maps
    tod_order = ['morning', 'afternoon', 'evening', 'night']
    tod_colors = {
        'morning': '#FDB462',  # Orange
        'afternoon': '#FFFF99',  # Yellow
        'evening': '#BEBADA',  # Purple
        'night': '#80B1D3'  # Blue
    }

    # Figure 1: t-SNE by ToD
    print("  Creating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Colored by ToD
    ax = axes[0]
    for tod in tod_order:
        mask = np.array([t == tod for t in tod_labels])
        if mask.sum() > 0:
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=tod_colors[tod], label=tod.capitalize(), alpha=0.6, s=20)
    ax.set_title('t-SNE Embeddings by Time of Day', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Colored by activity label (top 10 L1 labels)
    ax = axes[1]
    label_counts = Counter(activity_labels)
    top_labels = [label for label, _ in label_counts.most_common(10)]
    colors = sns.color_palette('tab10', n_colors=10)

    for i, label in enumerate(top_labels):
        mask = np.array([l == label for l in activity_labels])
        if mask.sum() > 0:
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[colors[i]], label=label, alpha=0.6, s=20)
    ax.set_title('t-SNE Embeddings by Activity Label (L1)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'tod_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: PCA analysis
    print("  Creating PCA visualization...")
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for tod in tod_order:
        mask = np.array([t == tod for t in tod_labels])
        if mask.sum() > 0:
            ax.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1],
                      c=tod_colors[tod], label=tod.capitalize(), alpha=0.6, s=20)
    ax.set_title('PCA Embeddings by Time of Day', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'tod_pca.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: ToD distribution and sample counts
    print("  Creating ToD distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Overall ToD distribution
    ax = axes[0, 0]
    tod_counts = Counter(tod_labels)
    tods = [tod for tod in tod_order if tod in tod_counts]
    counts = [tod_counts[tod] for tod in tods]
    colors_list = [tod_colors[tod] for tod in tods]
    ax.bar(range(len(tods)), counts, color=colors_list)
    ax.set_xticks(range(len(tods)))
    ax.set_xticklabels([t.capitalize() for t in tods])
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Distribution by Time of Day', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: ToD distribution by top labels
    ax = axes[0, 1]
    label_counts = Counter(activity_labels)
    top_labels = [label for label, _ in label_counts.most_common(5)]

    tod_label_counts = defaultdict(lambda: defaultdict(int))
    for i, sample_info in enumerate(samples_with_tod):
        if sample_info['label'] in top_labels:
            tod_label_counts[sample_info['tod']][sample_info['label']] += 1

    x = np.arange(len(tods))
    width = 0.15
    for i, label in enumerate(top_labels):
        counts = [tod_label_counts[tod][label] for tod in tods]
        ax.bar(x + i*width, counts, width, label=label)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([t.capitalize() for t in tods])
    ax.set_ylabel('Number of Samples')
    ax.set_title('ToD Distribution by Activity Label (L1)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Master bedroom/bathroom specific
    ax = axes[1, 0]
    target_rooms = ['master_bedroom', 'master_bath', 'master_bathroom']

    tod_room_counts = defaultdict(lambda: defaultdict(int))
    for sample_info in samples_with_tod:
        if sample_info['primary_room'] in target_rooms:
            room_name = sample_info['primary_room']
            tod_room_counts[sample_info['tod']][room_name] += 1

    x = np.arange(len(tods))
    width = 0.25
    for i, room in enumerate(target_rooms):
        counts = [tod_room_counts[tod][room] for tod in tods]
        if sum(counts) > 0:
            ax.bar(x + i*width, counts, width, label=room.replace('_', ' ').title())

    ax.set_xticks(x + width)
    ax.set_xticklabels([t.capitalize() for t in tods])
    ax.set_ylabel('Number of Samples')
    ax.set_title('Master Bedroom/Bathroom ToD Distribution', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Hour distribution
    ax = axes[1, 1]
    hours = [s['hour'] for s in samples_with_tod]
    ax.hist(hours, bins=24, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Distribution by Hour', fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3, axis='y')

    # Add ToD region shading
    ax.axvspan(0, 5, alpha=0.2, color=tod_colors['night'], label='Night')
    ax.axvspan(5, 12, alpha=0.2, color=tod_colors['morning'], label='Morning')
    ax.axvspan(12, 17, alpha=0.2, color=tod_colors['afternoon'], label='Afternoon')
    ax.axvspan(17, 20, alpha=0.2, color=tod_colors['evening'], label='Evening')
    ax.axvspan(20, 24, alpha=0.2, color=tod_colors['night'])

    plt.tight_layout()
    plt.savefig(output_dir / 'tod_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 4: Distance matrix heatmap
    print("  Creating distance matrix heatmap...")

    # Compute mean embeddings for each ToD
    tod_mean_embeddings = {}
    for tod in tod_order:
        mask = np.array([t == tod for t in tod_labels])
        if mask.sum() > 0:
            tod_mean_embeddings[tod] = embeddings[mask].mean(axis=0)

    # Compute pairwise cosine distances
    tods_present = list(tod_mean_embeddings.keys())
    n_tods = len(tods_present)
    distance_matrix = np.zeros((n_tods, n_tods))

    for i, tod1 in enumerate(tods_present):
        for j, tod2 in enumerate(tods_present):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                distance_matrix[i, j] = cosine(tod_mean_embeddings[tod1], tod_mean_embeddings[tod2])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(distance_matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=[t.capitalize() for t in tods_present],
                yticklabels=[t.capitalize() for t in tods_present],
                ax=ax, cbar_kws={'label': 'Cosine Distance'})
    ax.set_title('Mean Embedding Cosine Distances by ToD', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'tod_distance_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Visualizations saved to {output_dir}")


def generate_summary_report(samples_with_tod, stats_results, activity_stats, output_dir):
    """Generate a markdown summary report of the analysis."""

    from datetime import datetime

    report = []
    report.append("# Time-of-Day Embedding Analysis Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")

    # Overall statistics
    report.append("## Overall Time-of-Day Statistics\n")
    overall_stats = stats_results.get('overall', {})

    report.append("| Time of Day | Samples | Mean Norm | Mean Std |")
    report.append("|-------------|---------|-----------|----------|")
    for tod in ['morning', 'afternoon', 'evening', 'night']:
        if tod in overall_stats:
            stats = overall_stats[tod]
            report.append(f"| {tod.capitalize()} | {stats['n_samples']} | {stats['mean_norm']:.4f} | {stats['mean_std']:.4f} |")
    report.append("")

    # Distance matrix
    if 'distance_matrix' in stats_results:
        report.append("## Pairwise ToD Embedding Distances (Cosine)\n")
        dist_data = stats_results['distance_matrix']
        tods = dist_data['tods']
        distances = dist_data['distances']

        # Header row
        header = "| | " + " | ".join([t.capitalize() for t in tods]) + " |"
        separator = "|" + "|".join(["---"] * (len(tods) + 1)) + "|"
        report.append(header)
        report.append(separator)

        for i, tod1 in enumerate(tods):
            row = f"| {tod1.capitalize()} | " + " | ".join([f"{distances[i][j]:.4f}" for j in range(len(tods))]) + " |"
            report.append(row)
        report.append("")

        # Highlight key findings
        report.append("**Key Findings:**")
        max_dist = 0
        max_pair = None
        min_dist = 1.0
        min_pair = None
        for i, tod1 in enumerate(tods):
            for j, tod2 in enumerate(tods):
                if i < j:
                    dist = distances[i][j]
                    if dist > max_dist:
                        max_dist = dist
                        max_pair = (tod1, tod2)
                    if dist < min_dist:
                        min_dist = dist
                        min_pair = (tod1, tod2)

        if max_pair:
            report.append(f"- Most different: {max_pair[0].capitalize()} ↔ {max_pair[1].capitalize()} (distance: {max_dist:.4f})")
        if min_pair:
            report.append(f"- Most similar: {min_pair[0].capitalize()} ↔ {min_pair[1].capitalize()} (distance: {min_dist:.4f})")
        report.append("")

    # Top activities by ToD
    report.append("## Top Activities by Time of Day\n")

    label_counts = Counter([s['label'] for s in samples_with_tod])
    top_labels = [label for label, _ in label_counts.most_common(10)]

    by_label = stats_results.get('by_label', {})
    for label in top_labels:
        if label in by_label and len(by_label[label]) > 0:
            report.append(f"### {label}")
            report.append("")
            for tod, tod_stats in by_label[label].items():
                report.append(f"- **{tod.capitalize()}**: {tod_stats['n_samples']} samples, mean norm: {tod_stats['mean_norm']:.4f}")
            report.append("")

    # Master bedroom/bathroom analysis
    report.append("## Master Bedroom/Bathroom Analysis\n")
    by_room = stats_results.get('by_room', {})

    for room in ['master_bedroom', 'master_bath', 'master_bathroom']:
        if room in by_room:
            room_name = room.replace('_', ' ').title()
            report.append(f"### {room_name}\n")

            room_stats = by_room[room]
            for tod, tod_stats in room_stats.items():
                report.append(f"- **{tod.capitalize()}**: {tod_stats['n_samples']} samples, mean norm: {tod_stats['mean_norm']:.4f}")
            report.append("")

    # Specific activities deep dive
    if activity_stats:
        report.append("---\n")
        report.append("## Specific Activities Deep Dive\n")

        # Individual activity stats
        for activity, stats in activity_stats.items():
            # Skip internal keys
            if activity.startswith('_'):
                continue

            report.append(f"### {activity}\n")
            report.append(f"**Total samples:** {stats['total_samples']}")
            report.append(f"**Overall mean norm:** {stats['overall_mean_norm']:.4f}\n")

            report.append("**Distribution by Time of Day:**\n")
            tod_dist = stats.get('tod_distribution', {})
            for tod in ['morning', 'afternoon', 'evening', 'night']:
                if tod in tod_dist:
                    count = tod_dist[tod]
                    report.append(f"- {tod.capitalize()}: {count} samples")
            report.append("")

            report.append("**Mean Norms by ToD:**\n")
            tod_stats = stats.get('tod_stats', {})
            for tod in ['morning', 'afternoon', 'evening', 'night']:
                if tod in tod_stats:
                    norm = tod_stats[tod]['mean_norm']
                    n = tod_stats[tod]['n_samples']
                    report.append(f"- {tod.capitalize()}: {norm:.4f} (n={n})")
            report.append("")

        # Cross-activity comparison section
        report.append("---\n")
        report.append("## Cross-Activity Pairwise Distances\n")
        report.append("*Cosine distance between activity embeddings (0=identical, 1=opposite)*\n")

        # Extract cross-activity distances from stats
        if '_cross_activity_distances' in activity_stats:
            cross_distances = activity_stats['_cross_activity_distances']

            # Overall distances
            report.append("### Overall Distances (All Times of Day)\n")
            report.append("| Activity Pair | Distance | Interpretation |")
            report.append("|---------------|----------|----------------|")

            overall = cross_distances.get('overall', {})
            # Sort by distance
            sorted_pairs = sorted(overall.items(), key=lambda x: x[1])

            for pair, dist in sorted_pairs:
                if dist < 0.1:
                    interp = "Very Similar ✅"
                elif dist < 0.3:
                    interp = "Similar"
                elif dist < 0.5:
                    interp = "Moderately Different"
                else:
                    interp = "Very Different ❌"
                report.append(f"| {pair} | {dist:.4f} | {interp} |")
            report.append("")

            # By ToD distances
            by_tod = cross_distances.get('by_tod', {})
            for tod in ['morning', 'afternoon', 'evening', 'night']:
                if tod in by_tod and len(by_tod[tod]) > 0:
                    report.append(f"### Distances During {tod.capitalize()}\n")
                    report.append("| Activity Pair | Distance |")
                    report.append("|---------------|----------|")

                    sorted_tod_pairs = sorted(by_tod[tod].items(), key=lambda x: x[1])
                    for pair, dist in sorted_tod_pairs:
                        report.append(f"| {pair} | {dist:.4f} |")
                    report.append("")
        else:
            report.append("*Cross-activity distances not available*\n")

    # Files generated
    report.append("---\n")
    report.append("## Generated Files\n")
    files_list = [
        ("tod_statistics.json", "Complete statistical analysis in JSON format"),
        ("tod_tsne.png", "t-SNE visualization colored by time of day and activity labels"),
        ("tod_pca.png", "PCA visualization with explained variance"),
        ("tod_distributions.png", "Sample distribution plots across time periods"),
        ("tod_distance_matrix.png", "Heatmap of pairwise ToD distances"),
        ("specific_activities_stats.json", "Deep dive statistics for target activities"),
        ("specific_activities_tsne.png", "t-SNE visualization for specific activities"),
        ("specific_activities_overall_distances.png", "Overall activity distance heatmap"),
        ("specific_activities_distance_heatmaps.png", "Distance heatmaps by time of day"),
        ("embeddings.npy", "Raw embedding vectors (NumPy array)"),
        ("sample_metadata.csv", "Sample metadata with ToD labels")
    ]

    for filename, description in files_list:
        report.append(f"- **{filename}**: {description}")

    report.append("")
    report.append("---\n")
    report.append("*Analysis generated using tod_analysis.py*")

    # Write to file
    summary_path = output_dir / 'ANALYSIS_SUMMARY.md'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\n📄 Summary report saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Time-of-Day Embedding Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to train data JSON')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data JSON')
    parser.add_argument('--vocab', type=str, required=True,
                       help='Path to vocab JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum number of samples to analyze')
    parser.add_argument('--use_train', action='store_true',
                       help='Use train data instead of test data')
    parser.add_argument('--target_activities', nargs='+',
                       default=['Sleeping', 'Bed_to_Toilet', 'Master_Bedroom_Activity', 'Bathing'],
                       help='Specific activities to analyze in detail (L1 labels)')

    args = parser.parse_args()

    # Setup
    device = get_optimal_device()
    log_device_info(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TIME-OF-DAY EMBEDDING ANALYSIS")
    print("=" * 80)

    # Load model
    sensor_encoder = load_model(args.checkpoint, args.vocab, device)

    # Load data and extract embeddings
    data_path = args.train_data if args.use_train else args.test_data
    embeddings, samples_with_tod = extract_embeddings_and_metadata(
        sensor_encoder, data_path, args.vocab, device, max_samples=args.max_samples
    )

    # Compute statistics
    stats_results, tod_embeddings, tod_labels, tod_rooms = compute_tod_statistics(
        samples_with_tod, embeddings
    )

    # Save statistics
    print(f"\n💾 Saving statistics to {output_dir / 'tod_statistics.json'}")
    with open(output_dir / 'tod_statistics.json', 'w') as f:
        json.dump(stats_results, f, indent=2)

    # Deep dive into specific activities
    activity_stats = None
    if args.target_activities:
        activity_stats = analyze_specific_activities(
            samples_with_tod, embeddings, output_dir, args.target_activities
        )

    # Create visualizations
    visualize_tod_embeddings(samples_with_tod, embeddings, output_dir)

    # Generate summary report
    generate_summary_report(samples_with_tod, stats_results, activity_stats, output_dir)

    # Save embeddings and metadata
    print(f"💾 Saving embeddings and metadata...")
    np.save(output_dir / 'embeddings.npy', embeddings)

    metadata_df = pd.DataFrame([{
        'sample_id': s['sample_id'],
        'tod': s['tod'],
        'hour': s['hour'],
        'label_l1': s['label_l1'],
        'label_l2': s['label_l2'],
        'label': s['label'],  # Primary label used in analysis
        'primary_room': s['primary_room'],
        'rooms': '|'.join(s['rooms'])
    } for s in samples_with_tod])
    metadata_df.to_csv(output_dir / 'sample_metadata.csv', index=False)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - ANALYSIS_SUMMARY.md: Human-readable summary report")
    print("  - tod_statistics.json: Statistical analysis results")
    print("  - tod_tsne.png: t-SNE visualization")
    print("  - tod_pca.png: PCA visualization")
    print("  - tod_distributions.png: Sample distribution plots")
    print("  - tod_distance_matrix.png: Pairwise ToD distances")
    print("  - specific_activities_stats.json: Deep dive statistics for target activities")
    print("  - specific_activities_tsne.png: t-SNE for specific activities")
    print("  - specific_activities_overall_distances.png: Overall activity distance heatmap")
    print("  - specific_activities_distance_heatmaps.png: Distance heatmaps by ToD")
    print("  - embeddings.npy: Raw embeddings")
    print("  - sample_metadata.csv: Sample metadata with ToD labels")


if __name__ == '__main__':
    main()

