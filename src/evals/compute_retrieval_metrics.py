#!/usr/bin/env python3
"""
Compute Label-Recall@K metrics for cross-modal retrieval between text and sensor embeddings.

Label-Recall@K measures the proportion of retrieved neighbors that share the same label as the query.
For each query, we:
1. Rank all targets by cosine similarity
2. Take top K neighbors
3. Compute: (# neighbors with matching label) / K
4. Average across all queries

Supports:
- Text-to-Sensor retrieval (text queries → sensor targets)
- Sensor-to-Text retrieval (sensor queries → text targets)
- Both directions in one run
- Multiple K values (e.g., K = 10, 50, 100)

Example Usage:

python src/evals/compute_retrieval_metrics.py \\
    --sensor_embeddings results/evals/sensor_embeddings.npy \\
    --text_embeddings results/evals/text_embeddings.npy \\
    --labels results/evals/labels.npy \\
    --k_values 10 50 100 \\
    --directions both \\
    --normalize

Or use as a library:

from compute_retrieval_metrics import compute_label_recall_at_k

results = compute_label_recall_at_k(
    sensor_embeddings=sensor_emb,
    text_embeddings=text_emb,
    labels=labels,
    k_values=[10, 50, 100],
    directions=['text2sensor', 'sensor2text'],
    normalize=True
)
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
import json
import torch
import torch.nn.functional as F


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings along the feature dimension.

    Args:
        embeddings: Array of shape (N, D) where N is number of samples, D is dimension

    Returns:
        L2-normalized embeddings of same shape
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


def compute_cosine_similarity(queries: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between all queries and all targets.

    Args:
        queries: Query embeddings of shape (N_q, D)
        targets: Target embeddings of shape (N_t, D)

    Returns:
        Similarity matrix of shape (N_q, N_t) where entry [i, j] is similarity
        between query i and target j
    """
    # Assuming embeddings are already normalized, cosine similarity = dot product
    similarities = np.dot(queries, targets.T)
    return similarities


def compute_per_label_recall_at_k(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    query_labels: np.ndarray,
    target_labels: np.ndarray,
    k: int
) -> Dict[str, float]:
    """
    Compute per-label Label-Recall@K for instance-to-instance retrieval.

    For each unique label:
    1. Get all queries with that label
    2. For each query, find top K targets
    3. Compute recall for that query
    4. Average across all queries with that label

    Args:
        query_embeddings: Query embeddings of shape (N_q, D)
        target_embeddings: Target embeddings of shape (N_t, D)
        query_labels: Query labels of shape (N_q,)
        target_labels: Target labels of shape (N_t,)
        k: Number of top neighbors to consider

    Returns:
        Dictionary mapping label to average recall for that label
    """
    n_queries = len(query_embeddings)
    n_targets = len(target_embeddings)

    # Ensure K is valid
    k = min(k, n_targets)

    # Compute similarity matrix (N_q, N_t)
    similarities = compute_cosine_similarity(query_embeddings, target_embeddings)

    # Get unique labels
    unique_labels = sorted(list(set(query_labels)))

    # Compute recall for each label
    per_label_recalls = {}

    for label in unique_labels:
        # Get indices of queries with this label
        query_indices = np.where(query_labels == label)[0]

        if len(query_indices) == 0:
            continue

        label_recalls = []
        for i in query_indices:
            query_label = query_labels[i]
            query_sims = similarities[i]

            # Get indices of top K targets
            top_k_indices = np.argsort(query_sims)[-k:][::-1]
            top_k_labels = target_labels[top_k_indices]

            # Count how many match the query label
            n_matching = np.sum(top_k_labels == query_label)
            recall = n_matching / k
            label_recalls.append(recall)

        # Average across all queries with this label
        per_label_recalls[label] = np.mean(label_recalls)

    return per_label_recalls


def compute_label_recall_at_k_single_direction(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    query_labels: np.ndarray,
    target_labels: np.ndarray,
    k: int
) -> float:
    """
    Compute Label-Recall@K for a single retrieval direction.

    For each query:
    1. Compute similarity to all targets
    2. Rank targets by similarity (descending)
    3. Take top K targets
    4. Compute recall: (# top-K targets with matching label) / K
    5. Average across all queries

    Args:
        query_embeddings: Query embeddings of shape (N_q, D)
        target_embeddings: Target embeddings of shape (N_t, D)
        query_labels: Query labels of shape (N_q,)
        target_labels: Target labels of shape (N_t,)
        k: Number of top neighbors to consider

    Returns:
        Label-Recall@K averaged across all queries
    """
    n_queries = len(query_embeddings)
    n_targets = len(target_embeddings)

    # Ensure K is valid
    k = min(k, n_targets)

    # Compute similarity matrix (N_q, N_t)
    similarities = compute_cosine_similarity(query_embeddings, target_embeddings)

    # For each query, find top K targets
    recalls = []
    for i in range(n_queries):
        query_label = query_labels[i]
        query_sims = similarities[i]  # Shape: (N_t,)

        # Get indices of top K targets (highest similarity)
        # argsort returns ascending order, so we take the last K and reverse
        top_k_indices = np.argsort(query_sims)[-k:][::-1]

        # Get labels of top K targets
        top_k_labels = target_labels[top_k_indices]

        # Count how many match the query label
        n_matching = np.sum(top_k_labels == query_label)

        # Recall@K for this query
        recall = n_matching / k
        recalls.append(recall)

    # Average across all queries
    mean_recall = np.mean(recalls)
    return mean_recall


def compute_label_recall_at_k_with_prototypes(
    prototype_embeddings: np.ndarray,
    prototype_labels: np.ndarray,
    target_embeddings: np.ndarray,
    target_labels: np.ndarray,
    k: int
) -> float:
    """
    Compute Label-Recall@K when queries are prototypes (one per label).

    For each prototype:
    1. Compute similarity to all targets
    2. Rank targets by similarity (descending)
    3. Take top K targets
    4. Count how many of those K targets have the same label as the prototype
    5. Divide by K to get the recall ratio

    The final metric is the average across all prototypes.

    Args:
        prototype_embeddings: Prototype embeddings of shape (N_prototypes, D)
        prototype_labels: Labels for each prototype of shape (N_prototypes,)
        target_embeddings: Target embeddings of shape (N_targets, D)
        target_labels: Labels for each target of shape (N_targets,)
        k: Number of top neighbors to consider

    Returns:
        Label-Recall@K averaged across all prototypes
    """
    n_prototypes = len(prototype_embeddings)
    n_targets = len(target_embeddings)

    # Ensure K is valid
    k = min(k, n_targets)

    # Compute similarity matrix (N_prototypes, N_targets)
    similarities = compute_cosine_similarity(prototype_embeddings, target_embeddings)

    # For each prototype, find top K targets
    recalls = []
    for i in range(n_prototypes):
        proto_label = prototype_labels[i]
        proto_sims = similarities[i]  # Shape: (N_targets,)

        # Get indices of top K targets (highest similarity)
        top_k_indices = np.argsort(proto_sims)[-k:][::-1]

        # Get labels of top K targets
        top_k_labels = target_labels[top_k_indices]

        # Count how many match the prototype label
        n_matching = np.sum(top_k_labels == proto_label)

        # Recall@K for this prototype
        recall = n_matching / k
        recalls.append(recall)

    # Average across all prototypes
    mean_recall = np.mean(recalls)
    return mean_recall


def compute_label_recall_at_k(
    sensor_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    labels: np.ndarray,
    k_values: List[int] = [10, 50, 100],
    directions: List[str] = ['text2sensor', 'sensor2text'],
    normalize: bool = True,
    verbose: bool = True,
    return_per_label: bool = False
) -> Union[Dict[str, Dict[int, float]], Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, Dict[str, float]]]]]:
    """
    Compute Label-Recall@K for cross-modal retrieval.

    Args:
        sensor_embeddings: Sensor embeddings of shape (N, D_sensor)
        text_embeddings: Text embeddings of shape (N, D_text)
        labels: Labels for each example of shape (N,)
        k_values: List of K values to evaluate
        directions: List of directions to evaluate. Options:
            - 'text2sensor': Text queries → Sensor targets
            - 'sensor2text': Sensor queries → Text targets
            - 'both': Both directions (shorthand)
        normalize: Whether to L2-normalize embeddings before computing similarities
        verbose: Whether to print progress and results
        return_per_label: Whether to also return per-label metrics

    Returns:
        If return_per_label=False:
            Dictionary with structure:
            {
                'text2sensor': {k1: recall_value1, k2: recall_value2, ...},
                'sensor2text': {k1: recall_value1, k2: recall_value2, ...}
            }
        If return_per_label=True:
            Tuple of (overall_results, per_label_results) where per_label_results has structure:
            {
                'text2sensor': {k1: {label1: recall, label2: recall, ...}, k2: {...}, ...},
                'sensor2text': {k1: {label1: recall, label2: recall, ...}, k2: {...}, ...}
            }
    """
    # Validate inputs
    n_sensor, n_text, n_labels = len(sensor_embeddings), len(text_embeddings), len(labels)
    if not (n_sensor == n_text == n_labels):
        raise ValueError(
            f"Embeddings and labels must be aligned by index! "
            f"Got sensor: {n_sensor}, text: {n_text}, labels: {n_labels}"
        )

    # Handle 'both' direction shorthand
    if 'both' in directions:
        directions = ['text2sensor', 'sensor2text']

    # Normalize if requested
    if normalize:
        if verbose:
            print("[NORMALIZING] L2-normalizing embeddings...")
        sensor_embeddings = normalize_embeddings(sensor_embeddings)
        text_embeddings = normalize_embeddings(text_embeddings)

    # Store results
    results = {}

    # Compute for each direction
    for direction in directions:
        if direction not in ['text2sensor', 'sensor2text']:
            print(f"[WARNING] Unknown direction '{direction}', skipping...")
            continue

        if verbose:
            print(f"\n[COMPUTING] Label-Recall@K for {direction}...")

        # Determine query and target embeddings
        if direction == 'text2sensor':
            query_emb = text_embeddings
            target_emb = sensor_embeddings
            direction_name = "Text → Sensor"
        else:  # sensor2text
            query_emb = sensor_embeddings
            target_emb = text_embeddings
            direction_name = "Sensor → Text"

        # Compute for each K
        results[direction] = {}
        if return_per_label:
            per_label_results[direction] = {}

        for k in k_values:
            if verbose:
                print(f"  Computing for K={k}...")

            recall = compute_label_recall_at_k_single_direction(
                query_embeddings=query_emb,
                target_embeddings=target_emb,
                query_labels=labels,
                target_labels=labels,
                k=k
            )

            results[direction][k] = recall

            if verbose:
                print(f"    Label-Recall@{k}: {recall:.4f} ({recall*100:.2f}%)")

            # Compute per-label metrics if requested
            if return_per_label:
                per_label_recall = compute_per_label_recall_at_k(
                    query_embeddings=query_emb,
                    target_embeddings=target_emb,
                    query_labels=labels,
                    target_labels=labels,
                    k=k
                )
                per_label_results[direction][k] = per_label_recall

    if return_per_label:
        return results, per_label_results
    return results


def print_results_summary(results: Dict[str, Dict[int, float]]) -> None:
    """
    Print a formatted summary of retrieval results.

    Args:
        results: Results dictionary from compute_label_recall_at_k or compute_prototype_retrieval_metrics
    """
    print("\n" + "="*70)
    print("LABEL-RECALL@K RESULTS SUMMARY")
    print("="*70)

    for direction, k_results in results.items():
        # Format direction name
        if direction == 'text2sensor':
            direction_name = "Text -> Sensor"
        elif direction == 'sensor2text':
            direction_name = "Sensor -> Text"
        elif direction == 'prototype2sensor':
            direction_name = "Text Prototype -> Sensor"
        elif direction == 'prototype2text':
            direction_name = "Text Prototype -> Text"
        else:
            direction_name = direction

        print(f"\n{direction_name}:")
        print("-" * 40)

        # Sort by K value
        sorted_k = sorted(k_results.keys())
        for k in sorted_k:
            recall = k_results[k]
            print(f"  K={k:3d}  =>  Label-Recall@K = {recall:.4f} ({recall*100:.2f}%)")

    print("\n" + "="*70)


def load_text_prototypes_from_metadata(
    metadata_path: str,
    dataset_name: str = 'milan',
    style: str = 'sourish'
) -> Dict[str, str]:
    """
    Load text prototypes (label descriptions) from metadata.json.

    Args:
        metadata_path: Path to metadata JSON file
        dataset_name: Dataset name (e.g., 'milan', 'aruba')
        style: Caption style (e.g., 'sourish', 'baseline')

    Returns:
        Dictionary mapping label names to text descriptions
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    if dataset_name not in metadata:
        raise ValueError(f"Dataset '{dataset_name}' not found in metadata. Available: {list(metadata.keys())}")

    dataset_meta = metadata[dataset_name]

    # Try different possible keys for label descriptions
    label_key = f'label_to_text_{style}'
    if label_key not in dataset_meta:
        # Fallback to any label_to_text_* key
        available_keys = [k for k in dataset_meta.keys() if k.startswith('label_to_text_')]
        if available_keys:
            label_key = available_keys[0]
            print(f"[WARNING] Style '{style}' not found, using '{label_key}' instead")
        else:
            raise ValueError(f"No label descriptions found in metadata for dataset '{dataset_name}'")

    label_to_text = dataset_meta[label_key]
    return label_to_text


def encode_text_prototypes(
    label_to_text: Dict[str, str],
    text_encoder,
    device: str = 'cpu',
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode text prototypes using a text encoder.

    Args:
        label_to_text: Dictionary mapping label names to text descriptions
        text_encoder: Text encoder with encode_texts_clip method
        device: Device to use for encoding
        normalize: Whether to normalize embeddings

    Returns:
        Tuple of (prototype_embeddings, prototype_labels_as_strings)
    """
    labels = list(label_to_text.keys())
    texts = [label_to_text[label] for label in labels]

    # Encode texts
    with torch.no_grad():
        embeddings = text_encoder.encode_texts_clip(texts, device)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        embeddings = embeddings.cpu().numpy()

    return embeddings, np.array(labels)


def compute_prototype_retrieval_metrics(
    prototype_embeddings: np.ndarray,
    prototype_labels: np.ndarray,
    sensor_embeddings: Optional[np.ndarray] = None,
    text_embeddings: Optional[np.ndarray] = None,
    target_labels: Optional[np.ndarray] = None,
    k_values: List[int] = [10, 50, 100],
    directions: List[str] = ['prototype2sensor', 'prototype2text'],
    normalize: bool = True,
    verbose: bool = True,
    return_per_label: bool = False
) -> Union[Dict[str, Dict[int, float]], Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, Dict[str, float]]]]]:
    """
    Compute Label-Recall@K for prototype-based retrieval.

    Prototypes are text descriptions of labels (one per label).
    Targets are actual embeddings (sensor or text) with many examples per label.

    Args:
        prototype_embeddings: Text prototype embeddings of shape (N_prototypes, D_text)
        prototype_labels: Label names for each prototype of shape (N_prototypes,)
        sensor_embeddings: Sensor embeddings of shape (N, D_sensor) [optional]
        text_embeddings: Text embeddings of shape (N, D_text) [optional]
        target_labels: Labels for target embeddings of shape (N,) [required if any targets provided]
        k_values: List of K values to evaluate
        directions: List of directions to evaluate:
            - 'prototype2sensor': Text prototype queries → Sensor targets
            - 'prototype2text': Text prototype queries → Text targets
            - 'both': Both directions (shorthand)
        normalize: Whether to L2-normalize embeddings before computing similarities
        verbose: Whether to print progress and results
        return_per_label: Whether to also return per-label metrics

    Returns:
        If return_per_label=False:
            Dictionary with structure:
            {
                'prototype2sensor': {k1: recall_value1, k2: recall_value2, ...},
                'prototype2text': {k1: recall_value1, k2: recall_value2, ...}
            }
        If return_per_label=True:
            Tuple of (overall_results, per_label_results) where per_label_results has structure:
            {
                'prototype2sensor': {k1: {label1: recall, label2: recall, ...}, k2: {...}, ...},
                'prototype2text': {k1: {label1: recall, label2: recall, ...}, k2: {...}, ...}
            }
    """
    # Handle 'both' direction shorthand
    if 'both' in directions:
        directions = ['prototype2sensor', 'prototype2text']

    # Normalize if requested
    if normalize:
        if verbose:
            print("[NORMALIZING] L2-normalizing embeddings...")
        prototype_embeddings = normalize_embeddings(prototype_embeddings)
        if sensor_embeddings is not None:
            sensor_embeddings = normalize_embeddings(sensor_embeddings)
        if text_embeddings is not None:
            text_embeddings = normalize_embeddings(text_embeddings)

    # Store results
    results = {}

    # Compute for each direction
    for direction in directions:
        if direction not in ['prototype2sensor', 'prototype2text']:
            print(f"[WARNING] Unknown direction '{direction}', skipping...")
            continue

        # Determine target embeddings
        if direction == 'prototype2sensor':
            if sensor_embeddings is None:
                print(f"[WARNING] Sensor embeddings not provided, skipping {direction}")
                continue
            target_emb = sensor_embeddings
            direction_name = "Text Prototype -> Sensor"
        else:  # prototype2text
            if text_embeddings is None:
                print(f"[WARNING] Text embeddings not provided, skipping {direction}")
                continue
            target_emb = text_embeddings
            direction_name = "Text Prototype -> Text"

        if target_labels is None:
            raise ValueError(f"target_labels must be provided for direction '{direction}'")

        if verbose:
            print(f"\n[COMPUTING] Label-Recall@K for {direction}...")
            print(f"  {len(prototype_embeddings)} prototypes, {len(target_emb)} targets")

        # Compute for each K
        results[direction] = {}
        if return_per_label:
            per_label_results[direction] = {}

        for k in k_values:
            if verbose:
                print(f"  Computing for K={k}...")

            # Compute overall recall
            recall = compute_label_recall_at_k_with_prototypes(
                prototype_embeddings=prototype_embeddings,
                prototype_labels=prototype_labels,
                target_embeddings=target_emb,
                target_labels=target_labels,
                k=k
            )

            results[direction][k] = recall

            if verbose:
                print(f"    Label-Recall@{k}: {recall:.4f} ({recall*100:.2f}%)")

            # Compute per-label metrics if requested
            if return_per_label:
                from evals.compute_retrieval_metrics import (
                    normalize_embeddings,
                    compute_cosine_similarity
                )

                # Compute per-label recall for prototypes
                per_label_recall = {}
                similarities = compute_cosine_similarity(prototype_embeddings, target_emb)

                for i, proto_label in enumerate(prototype_labels):
                    proto_sims = similarities[i]
                    top_k_indices = np.argsort(proto_sims)[-k:][::-1]
                    top_k_labels = target_labels[top_k_indices]
                    n_matching = np.sum(top_k_labels == proto_label)
                    recall_label = n_matching / k
                    per_label_recall[str(proto_label)] = float(recall_label)

                per_label_results[direction][k] = per_label_recall

    if return_per_label:
        return results, per_label_results
    return results


def load_embeddings_and_labels(
    sensor_path: Optional[str] = None,
    text_path: Optional[str] = None,
    labels_path: Optional[str] = None,
    sensor_key: str = 'embeddings',
    text_key: str = 'embeddings',
    labels_key: str = 'labels'
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load embeddings and labels from .npy or .npz files.

    Args:
        sensor_path: Path to sensor embeddings file (.npy or .npz)
        text_path: Path to text embeddings file (.npy or .npz)
        labels_path: Path to labels file (.npy or .npz)
        sensor_key: Key to use if sensor file is .npz
        text_key: Key to use if text file is .npz
        labels_key: Key to use if labels file is .npz

    Returns:
        Tuple of (sensor_embeddings, text_embeddings, labels)
        Any component can be None if path not provided
    """
    def load_single_file(path: Optional[str], key: str) -> Optional[np.ndarray]:
        if path is None:
            return None

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix == '.npy':
            return np.load(str(path))
        elif path.suffix == '.npz':
            data = np.load(str(path))
            if key not in data:
                # Try to find the right key
                available_keys = list(data.keys())
                print(f"[WARNING] Key '{key}' not found in {path.name}")
                print(f"   Available keys: {available_keys}")
                if len(available_keys) == 1:
                    key = available_keys[0]
                    print(f"   Using key: '{key}'")
                else:
                    raise KeyError(f"Multiple keys found, please specify which to use")
            return data[key]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix} (use .npy or .npz)")

    sensor_emb = load_single_file(sensor_path, sensor_key)
    text_emb = load_single_file(text_path, text_key)
    labels = load_single_file(labels_path, labels_key)

    return sensor_emb, text_emb, labels


def save_results(results: Dict[str, Dict[int, float]], output_path: str) -> None:
    """
    Save results to a JSON file.

    Args:
        results: Results dictionary from compute_label_recall_at_k
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert int keys to strings for JSON serialization
    json_results = {}
    for direction, k_results in results.items():
        json_results[direction] = {str(k): float(v) for k, v in k_results.items()}

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n[SAVED] Results saved to: {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute Label-Recall@K for cross-modal retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic instance-to-instance retrieval
  python src/evals/compute_retrieval_metrics.py \\
      --sensor_embeddings sensor_emb.npy \\
      --text_embeddings text_emb.npy \\
      --labels labels.npy \\
      --k_values 10 50 100 \\
      --directions both

  # Prototype-based retrieval (uses label descriptions from metadata)
  python src/evals/compute_retrieval_metrics.py \\
      --sensor_embeddings sensor_emb.npy \\
      --text_embeddings text_emb.npy \\
      --labels labels.npy \\
      --use_prototypes \\
      --metadata metadata/casas_metadata.json \\
      --dataset_name milan \\
      --checkpoint trained_models/best_model.pt \\
      --k_values 10 50 100

  # With .npz files (specify keys)
  python src/evals/compute_retrieval_metrics.py \\
      --sensor_embeddings data.npz --sensor_key sensor_emb \\
      --text_embeddings data.npz --text_key text_emb \\
      --labels data.npz --labels_key labels \\
      --k_values 10 50 100 \\
      --output results.json
        """
    )

    # Input files
    parser.add_argument('--sensor_embeddings', type=str,
                        help='Path to sensor embeddings (.npy or .npz)')
    parser.add_argument('--text_embeddings', type=str,
                        help='Path to text embeddings (.npy or .npz)')
    parser.add_argument('--labels', type=str,
                        help='Path to labels (.npy or .npz)')

    # Keys for .npz files
    parser.add_argument('--sensor_key', type=str, default='embeddings',
                        help='Key for sensor embeddings if .npz file (default: embeddings)')
    parser.add_argument('--text_key', type=str, default='embeddings',
                        help='Key for text embeddings if .npz file (default: embeddings)')
    parser.add_argument('--labels_key', type=str, default='labels',
                        help='Key for labels if .npz file (default: labels)')

    # Prototype-based retrieval
    parser.add_argument('--use_prototypes', action='store_true',
                        help='Use text prototypes (label descriptions) instead of instance embeddings')
    parser.add_argument('--metadata', type=str, default='metadata/casas_metadata.json',
                        help='Path to metadata JSON file with label descriptions (for prototypes)')
    parser.add_argument('--dataset_name', type=str, default='milan',
                        help='Dataset name in metadata (e.g., milan, aruba)')
    parser.add_argument('--caption_style', type=str, default='sourish',
                        help='Caption style to use from metadata (e.g., sourish, baseline)')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to model checkpoint (required for encoding text prototypes)')

    # Evaluation parameters
    parser.add_argument('--k_values', type=int, nargs='+', default=[10, 50, 100],
                        help='List of K values to evaluate (default: 10 50 100)')
    parser.add_argument('--directions', type=str, nargs='+',
                        default=['both'],
                        help='Retrieval directions to evaluate')
    parser.add_argument('--normalize', dest='normalize', action='store_true',
                        help='L2-normalize embeddings before computing similarity (default)')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                        help='Do not normalize embeddings')
    parser.set_defaults(normalize=True)

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results as JSON (optional)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress messages')

    return parser.parse_args()


def main():
    """Main function for CLI usage."""
    args = parse_args()

    verbose = not args.quiet

    if verbose:
        print("="*70)
        print("CROSS-MODAL RETRIEVAL EVALUATION")
        print("="*70)

    # Handle prototype-based vs instance-based retrieval
    if args.use_prototypes:
        # Prototype-based retrieval
        if verbose:
            print("\n[MODE] Using text prototypes (label descriptions)")

        # Validate required arguments
        if not args.checkpoint:
            print("\n[ERROR] --checkpoint is required for prototype-based retrieval")
            return 1
        if not args.sensor_embeddings and not args.text_embeddings:
            print("\n[ERROR] At least one of --sensor_embeddings or --text_embeddings is required")
            return 1
        if not args.labels:
            print("\n[ERROR] --labels is required")
            return 1

        # Load text prototypes from metadata
        if verbose:
            print(f"\n[LOADING] Loading text prototypes from metadata...")
            print(f"   Metadata: {args.metadata}")
            print(f"   Dataset: {args.dataset_name}")
            print(f"   Style: {args.caption_style}")

        try:
            label_to_text = load_text_prototypes_from_metadata(
                metadata_path=args.metadata,
                dataset_name=args.dataset_name,
                style=args.caption_style
            )
            if verbose:
                print(f"   Found {len(label_to_text)} label descriptions")
        except Exception as e:
            print(f"\n[ERROR] Error loading text prototypes: {e}")
            return 1

        # Load text encoder from checkpoint
        if verbose:
            print(f"\n[LOADING] Loading text encoder from checkpoint...")
            print(f"   Checkpoint: {args.checkpoint}")

        try:
            # Import here to avoid dependency if not using prototypes
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from evals.eval_utils import create_text_encoder_from_checkpoint
            from utils.device_utils import get_optimal_device

            device = get_optimal_device()
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            text_encoder = create_text_encoder_from_checkpoint(checkpoint, device)

            if text_encoder is None:
                print("\n[ERROR] Could not create text encoder from checkpoint")
                return 1
        except Exception as e:
            print(f"\n[ERROR] Error loading text encoder: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # Encode text prototypes
        if verbose:
            print(f"\n[ENCODING] Encoding text prototypes...")

        try:
            prototype_emb, prototype_labels = encode_text_prototypes(
                label_to_text=label_to_text,
                text_encoder=text_encoder,
                device=str(device),
                normalize=args.normalize
            )
            if verbose:
                print(f"   Encoded {len(prototype_emb)} prototypes")
                print(f"   Prototype embeddings shape: {prototype_emb.shape}")
        except Exception as e:
            print(f"\n[ERROR] Error encoding text prototypes: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # Load target embeddings and labels
        if verbose:
            print(f"\n[LOADING] Loading target embeddings...")

        try:
            sensor_emb, text_emb, labels = load_embeddings_and_labels(
                sensor_path=args.sensor_embeddings,
                text_path=args.text_embeddings,
                labels_path=args.labels,
                sensor_key=args.sensor_key,
                text_key=args.text_key,
                labels_key=args.labels_key
            )
        except Exception as e:
            print(f"\n[ERROR] Error loading data: {e}")
            return 1

        if verbose:
            print(f"\n[SUCCESS] Data loaded successfully!")
            if sensor_emb is not None:
                print(f"   Sensor embeddings shape: {sensor_emb.shape}")
            if text_emb is not None:
                print(f"   Text embeddings shape: {text_emb.shape}")
            print(f"   Labels shape: {labels.shape}")
            print(f"   Number of examples: {len(labels)}")
            print(f"   Number of unique labels: {len(np.unique(labels))}")

        # Compute prototype-based retrieval metrics
        try:
            # Convert string labels to match prototype labels if needed
            if labels.dtype.kind in ['U', 'S', 'O']:  # String types
                target_labels = labels
            else:
                # If labels are integers, need to map them
                print("[WARNING] Target labels appear to be integers. Make sure they match prototype label format.")
                target_labels = labels

            results = compute_prototype_retrieval_metrics(
                prototype_embeddings=prototype_emb,
                prototype_labels=prototype_labels,
                sensor_embeddings=sensor_emb,
                text_embeddings=text_emb,
                target_labels=target_labels,
                k_values=args.k_values,
                directions=args.directions if args.directions != ['both'] else ['prototype2sensor', 'prototype2text'],
                normalize=args.normalize,
                verbose=verbose
            )
        except Exception as e:
            print(f"\n[ERROR] Error computing prototype metrics: {e}")
            import traceback
            traceback.print_exc()
            return 1

    else:
        # Instance-to-instance retrieval (original behavior)
        if verbose:
            print("\n[MODE] Using instance-to-instance retrieval")

        # Validate required arguments
        if not args.sensor_embeddings or not args.text_embeddings or not args.labels:
            print("\n[ERROR] --sensor_embeddings, --text_embeddings, and --labels are required")
            return 1

        if verbose:
            print(f"\n[LOADING] Loading data...")
            print(f"   Sensor embeddings: {args.sensor_embeddings}")
            print(f"   Text embeddings: {args.text_embeddings}")
            print(f"   Labels: {args.labels}")

        # Load data
        try:
            sensor_emb, text_emb, labels = load_embeddings_and_labels(
                sensor_path=args.sensor_embeddings,
                text_path=args.text_embeddings,
                labels_path=args.labels,
                sensor_key=args.sensor_key,
                text_key=args.text_key,
                labels_key=args.labels_key
            )
        except Exception as e:
            print(f"\n[ERROR] Error loading data: {e}")
            return 1

        if verbose:
            print(f"\n[SUCCESS] Data loaded successfully!")
            print(f"   Sensor embeddings shape: {sensor_emb.shape}")
            print(f"   Text embeddings shape: {text_emb.shape}")
            print(f"   Labels shape: {labels.shape}")
            print(f"   Number of examples: {len(labels)}")
            print(f"   Number of unique labels: {len(np.unique(labels))}")

        # Compute metrics
        try:
            results = compute_label_recall_at_k(
                sensor_embeddings=sensor_emb,
                text_embeddings=text_emb,
                labels=labels,
                k_values=args.k_values,
                directions=args.directions,
                normalize=args.normalize,
                verbose=verbose
            )
        except Exception as e:
            print(f"\n[ERROR] Error computing metrics: {e}")
            return 1

    # Print summary
    print_results_summary(results)

    # Save results if output path provided
    if args.output:
        try:
            save_results(results, args.output)
        except Exception as e:
            print(f"\n[WARNING] Could not save results: {e}")

    if verbose:
        print("\n[SUCCESS] Evaluation complete!")

    return 0


if __name__ == '__main__':
    exit(main())

