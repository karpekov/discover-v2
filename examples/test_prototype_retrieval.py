#!/usr/bin/env python3
"""
Example script demonstrating prototype-based retrieval metrics.

This shows how to use text prototypes (label descriptions) to retrieve
sensor and text embeddings.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pathlib import Path

from evals.compute_retrieval_metrics import (
    compute_label_recall_at_k_with_prototypes,
    print_results_summary,
    save_results
)


def create_prototype_and_instance_embeddings(
    n_classes: int = 5,
    samples_per_class: int = 30,
    embedding_dim: int = 128,
    prototype_quality: float = 0.8,
    seed: int = 42
):
    """
    Create synthetic prototypes and instance embeddings.

    Prototypes are "ideal" representations of each class.
    Instances are noisy versions of prototypes.

    Args:
        n_classes: Number of activity classes
        samples_per_class: Number of samples per class
        embedding_dim: Dimensionality of embeddings
        prototype_quality: How similar instances are to prototypes (0-1)
                          Higher = instances are closer to their class prototype
        seed: Random seed for reproducibility

    Returns:
        prototype_emb, prototype_labels, sensor_emb, text_emb, instance_labels
    """
    np.random.seed(seed)

    n_instances = n_classes * samples_per_class

    # Create class prototypes (one per class)
    # These represent the "ideal" or "canonical" representation of each class
    prototype_emb = np.random.randn(n_classes, embedding_dim)
    prototype_emb = prototype_emb / np.linalg.norm(prototype_emb, axis=1, keepdims=True)

    # Prototype labels are class indices (as strings)
    prototype_labels = np.array([f"Activity_{i}" for i in range(n_classes)])

    # Create instance embeddings by adding noise to prototypes
    sensor_emb = []
    text_emb = []
    instance_labels = []

    for class_idx in range(n_classes):
        for _ in range(samples_per_class):
            # Instances are weighted combinations of prototype + noise
            # Higher prototype_quality = more similar to prototype
            sensor_noise = np.random.randn(embedding_dim) * (1 - prototype_quality)
            text_noise = np.random.randn(embedding_dim) * (1 - prototype_quality)

            sensor_sample = prototype_quality * prototype_emb[class_idx] + sensor_noise
            text_sample = prototype_quality * prototype_emb[class_idx] + text_noise

            # Normalize
            sensor_sample = sensor_sample / np.linalg.norm(sensor_sample)
            text_sample = text_sample / np.linalg.norm(text_sample)

            sensor_emb.append(sensor_sample)
            text_emb.append(text_sample)
            instance_labels.append(f"Activity_{class_idx}")

    sensor_emb = np.array(sensor_emb)
    text_emb = np.array(text_emb)
    instance_labels = np.array(instance_labels)

    return prototype_emb, prototype_labels, sensor_emb, text_emb, instance_labels


def main():
    print("="*70)
    print("PROTOTYPE-BASED RETRIEVAL EXAMPLE")
    print("="*70)

    # Create synthetic data
    print("\n[1] Creating synthetic data...")
    print("    Scenario: 5 activity classes, 30 samples per class")
    print("    Prototypes = ideal/canonical representations")
    print("    Instances = noisy real-world examples")

    prototype_emb, prototype_labels, sensor_emb, text_emb, instance_labels = \
        create_prototype_and_instance_embeddings(
            n_classes=5,
            samples_per_class=30,
            embedding_dim=128,
            prototype_quality=0.75,  # Instances are 75% similar to prototypes
            seed=42
        )

    print(f"\n    Prototypes: {prototype_emb.shape} ({len(prototype_labels)} classes)")
    print(f"    Sensor instances: {sensor_emb.shape}")
    print(f"    Text instances: {text_emb.shape}")
    print(f"    Instance labels: {instance_labels.shape}")
    print(f"    Classes: {prototype_labels}")

    # Compute prototype -> sensor retrieval
    print("\n[2] Computing Prototype -> Sensor retrieval...")

    results_proto2sensor = {}
    for k in [5, 10, 20]:
        recall = compute_label_recall_at_k_with_prototypes(
            prototype_embeddings=prototype_emb,
            prototype_labels=prototype_labels,
            target_embeddings=sensor_emb,
            target_labels=instance_labels,
            k=k
        )
        results_proto2sensor[k] = recall
        print(f"    K={k:2d}  =>  Label-Recall@K = {recall:.4f} ({recall*100:.2f}%)")

    # Compute prototype -> text retrieval
    print("\n[3] Computing Prototype -> Text retrieval...")

    results_proto2text = {}
    for k in [5, 10, 20]:
        recall = compute_label_recall_at_k_with_prototypes(
            prototype_embeddings=prototype_emb,
            prototype_labels=prototype_labels,
            target_embeddings=text_emb,
            target_labels=instance_labels,
            k=k
        )
        results_proto2text[k] = recall
        print(f"    K={k:2d}  =>  Label-Recall@K = {recall:.4f} ({recall*100:.2f}%)")

    # Combined results
    results = {
        'prototype2sensor': results_proto2sensor,
        'prototype2text': results_proto2text
    }

    # Print summary
    print_results_summary(results)

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    proto2sensor_10 = results['prototype2sensor'][10]
    proto2text_10 = results['prototype2text'][10]

    # Random baseline
    samples_per_class = 30
    total_samples = 150
    random_baseline = samples_per_class / total_samples  # 0.2 = 20%

    print(f"\nRandom baseline: {random_baseline:.2%}")
    print(f"Prototype -> Sensor @ K=10: {proto2sensor_10:.2%}")
    print(f"Prototype -> Text @ K=10: {proto2text_10:.2%}")

    improvement_sensor = (proto2sensor_10 / random_baseline - 1) * 100
    improvement_text = (proto2text_10 / random_baseline - 1) * 100

    print(f"\nImprovement over random:")
    print(f"  Sensor: {improvement_sensor:.1f}%")
    print(f"  Text: {improvement_text:.1f}%")

    if proto2sensor_10 > 0.5:
        print("\n[SUCCESS] High prototype retrieval! Model learns semantic structure.")
    elif proto2sensor_10 > 0.3:
        print("\n[MODERATE] Decent prototype retrieval. Some semantic understanding.")
    else:
        print("\n[POOR] Low prototype retrieval. Check model training.")

    # What this means
    print("\n" + "="*70)
    print("WHAT PROTOTYPE RETRIEVAL TELLS US")
    print("="*70)
    print("""
Prototype-based retrieval measures whether:
1. The model understands class semantics (not just memorizes instances)
2. Label descriptions can effectively retrieve relevant examples
3. Embeddings generalize to canonical/prototypical representations

High prototype recall indicates:
- Model captures the essence of each class
- Embeddings are semantically meaningful
- Good zero-shot potential

This is particularly useful for:
- Testing if model understands label descriptions from metadata
- Evaluating semantic alignment (not just instance matching)
- Assessing generalization to new descriptions of same activities
    """)

    # Save results
    output_dir = Path("results/examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "prototype_retrieval_example.json"
    save_results(results, str(output_file))

    print(f"\n[4] Results saved to: {output_file}")
    print("\n" + "="*70)
    print("[SUCCESS] Example complete!")
    print("="*70)


if __name__ == '__main__':
    main()

