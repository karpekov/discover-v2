#!/usr/bin/env python3
"""
Example script demonstrating how to use compute_retrieval_metrics.py

This creates synthetic data to simulate cross-modal retrieval evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pathlib import Path

from evals.compute_retrieval_metrics import (
    compute_label_recall_at_k,
    print_results_summary,
    save_results
)


def create_synthetic_embeddings_with_structure(
    n_classes: int = 10,
    samples_per_class: int = 20,
    embedding_dim: int = 128,
    noise_level: float = 0.5,
    seed: int = 42
):
    """
    Create synthetic embeddings with some semantic structure.

    For each class, we create a "prototype" vector and add noise to create
    individual samples. This simulates real embeddings where examples of the
    same class are somewhat similar but not identical.

    Args:
        n_classes: Number of activity classes
        samples_per_class: Number of samples per class
        embedding_dim: Dimensionality of embeddings
        noise_level: Amount of noise to add (0 = all identical, 1 = random)
        seed: Random seed for reproducibility

    Returns:
        sensor_embeddings, text_embeddings, labels
    """
    np.random.seed(seed)

    n_samples = n_classes * samples_per_class

    # Create class prototypes (one per class)
    sensor_prototypes = np.random.randn(n_classes, embedding_dim)
    text_prototypes = np.random.randn(n_classes, embedding_dim)

    # Normalize prototypes
    sensor_prototypes = sensor_prototypes / np.linalg.norm(sensor_prototypes, axis=1, keepdims=True)
    text_prototypes = text_prototypes / np.linalg.norm(text_prototypes, axis=1, keepdims=True)

    # Add some correlation between modalities (to simulate cross-modal alignment)
    # Text prototypes are a mix of sensor prototypes and independent noise
    text_prototypes = 0.5 * sensor_prototypes + 0.5 * text_prototypes
    text_prototypes = text_prototypes / np.linalg.norm(text_prototypes, axis=1, keepdims=True)

    # Create samples by adding noise to prototypes
    sensor_embeddings = []
    text_embeddings = []
    labels = []

    for class_idx in range(n_classes):
        for _ in range(samples_per_class):
            # Add noise to prototype
            sensor_noise = np.random.randn(embedding_dim) * noise_level
            text_noise = np.random.randn(embedding_dim) * noise_level

            sensor_sample = sensor_prototypes[class_idx] + sensor_noise
            text_sample = text_prototypes[class_idx] + text_noise

            sensor_embeddings.append(sensor_sample)
            text_embeddings.append(text_sample)
            labels.append(class_idx)

    sensor_embeddings = np.array(sensor_embeddings)
    text_embeddings = np.array(text_embeddings)
    labels = np.array(labels)

    return sensor_embeddings, text_embeddings, labels


def main():
    print("="*70)
    print("RETRIEVAL METRICS EXAMPLE")
    print("="*70)

    # Create synthetic data with semantic structure
    print("\n[1] Creating synthetic embeddings...")
    print("    (Simulating 10 activity classes with 20 samples each)")

    sensor_emb, text_emb, labels = create_synthetic_embeddings_with_structure(
        n_classes=10,
        samples_per_class=20,
        embedding_dim=128,
        noise_level=0.3,  # Lower noise = higher similarity within class
        seed=42
    )

    print(f"    Sensor embeddings: {sensor_emb.shape}")
    print(f"    Text embeddings: {text_emb.shape}")
    print(f"    Labels: {labels.shape}")
    print(f"    Unique classes: {len(np.unique(labels))}")

    # Compute retrieval metrics
    print("\n[2] Computing Label-Recall@K metrics...")

    results = compute_label_recall_at_k(
        sensor_embeddings=sensor_emb,
        text_embeddings=text_emb,
        labels=labels,
        k_values=[5, 10, 20, 50],
        directions=['text2sensor', 'sensor2text'],
        normalize=True,
        verbose=False
    )

    # Print results
    print_results_summary(results)

    # Interpret results
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    text2sensor_10 = results['text2sensor'][10]
    sensor2text_10 = results['sensor2text'][10]

    # Random baseline would be 10% (10 samples per class out of 200 total)
    random_baseline = 10 / 200  # 0.05 = 5%

    print(f"\nRandom baseline (expected for random embeddings): {random_baseline:.2%}")
    print(f"Text -> Sensor @ K=10: {text2sensor_10:.2%}")
    print(f"Sensor -> Text @ K=10: {sensor2text_10:.2%}")

    if text2sensor_10 > 0.15:
        print("\n[SUCCESS] Embeddings show good semantic structure!")
        print("The model has learned to align text and sensor embeddings.")
    elif text2sensor_10 > 0.08:
        print("\n[MODERATE] Embeddings show some structure, but could be improved.")
    else:
        print("\n[POOR] Embeddings are close to random. Check your model training.")

    # Optional: Save results
    output_dir = Path("results/examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "example_retrieval_metrics.json"
    save_results(results, str(output_file))

    print(f"\n[3] Example complete!")
    print(f"    Results saved to: {output_file}")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()

