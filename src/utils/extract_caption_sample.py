#!/usr/bin/env python3
"""
Extract captions from any CASAS dataset and save samples grouped by activity labels.

This script can extract captions from any processed CASAS dataset (Milan, Aruba, etc.)
with any configuration (seq20, seq50, etc.) and save them grouped by L1 activity labels.

Usage Examples:
    # Extract 40 captions per label from Milan seq20 presegmented test data
    python extract_caption_sample.py \
        --data-path data/processed/casas/milan/seq20/milan_presegmented_test.json \
        --output-path data/processed/casas/milan/seq20/test_captions_40_per_label.txt \
        --captions-per-label 40

    # Extract 20 captions per label from Aruba seq50 regular test data
    python extract_caption_sample.py \
        --data-path data/processed/casas/aruba/seq50/aruba_test.json \
        --output-path data/processed/casas/aruba/seq50/test_captions_20_per_label.txt \
        --captions-per-label 20

    # Extract all available captions (no sampling) from Milan training data
    python extract_caption_sample.py \
        --data-path data/processed/casas/milan/seq20/milan_train.json \
        --output-path data/processed/casas/milan/seq20/train_captions_all.txt \
        --captions-per-label -1

    # Use default output path (auto-generated based on input path)
    python extract_caption_sample.py \
        --data-path data/processed/casas/milan/seq20/milan_presegmented_test.json \
        --captions-per-label 30

    # List available labels without extracting captions
    python extract_caption_sample.py \
        --data-path data/processed/casas/milan/seq20/milan_presegmented_test.json \
        --list-labels-only
"""

import json
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Optional
import os
from pathlib import Path


def load_dataset_labels(data_path: str) -> List[str]:
    """
    Extract unique L1 activity labels from a dataset file.

    Args:
        data_path: Path to the dataset JSON file

    Returns:
        List of unique L1 activity labels found in the dataset
    """
    print(f"Loading dataset from: {data_path}")

    # Read the JSON file
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # Try reading as JSONL (one JSON object per line)
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    print(f"Loaded {len(data)} samples from dataset")

    # Extract unique L1 labels
    labels = set()
    for sample in data:
        first_activity = sample.get('first_activity', '')
        if first_activity:
            labels.add(first_activity)

    labels = sorted(list(labels))
    print(f"Found {len(labels)} unique L1 labels:")
    for label in labels:
        print(f"  - {label}")
    print()

    return labels


def extract_captions_by_label(data_path: str, target_labels: List[str], captions_per_label: int = 40) -> Dict[str, List[str]]:
    """
    Extract captions from dataset grouped by L1 labels.

    Args:
        data_path: Path to the dataset JSON file
        target_labels: List of L1 labels to extract captions for
        captions_per_label: Number of captions to extract per label (-1 for all available)

    Returns:
        Dictionary mapping L1 labels to lists of captions
    """
    print(f"Extracting captions from: {data_path}")

    # Initialize collections for each label
    label_captions = defaultdict(list)
    label_sample_count = defaultdict(int)

    # Read the JSON file
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # Try reading as JSONL (one JSON object per line)
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    print(f"Loaded {len(data)} samples from dataset")

    # Extract captions for each sample
    for sample in data:
        # Get the first activity label (L1 label)
        first_activity = sample.get('first_activity', '')

        if not first_activity or first_activity not in target_labels:
            continue

        label_sample_count[first_activity] += 1

        # Get captions from the sample
        captions = []

        # Try different caption field names based on the dataset structure
        if 'long_captions' in sample:
            captions.extend(sample['long_captions'])
        elif 'captions' in sample:
            captions.extend(sample['captions'])
        elif 'short_captions' in sample:
            captions.extend(sample['short_captions'])

        # Add captions to the label collection
        for caption in captions:
            if caption and caption.strip():  # Skip empty captions
                label_captions[first_activity].append(caption.strip())

    print(f"\nFound samples for {len(label_captions)} labels:")
    for label, captions in label_captions.items():
        print(f"  {label}: {len(captions)} captions from {label_sample_count[label]} samples")

    # Sample the requested number of captions for each label
    sampled_captions = {}
    for label in target_labels:
        available_captions = label_captions[label]
        if len(available_captions) == 0:
            print(f"⚠️  No captions found for label: {label}")
            sampled_captions[label] = []
        elif captions_per_label == -1 or len(available_captions) <= captions_per_label:
            print(f"📝 Using all {len(available_captions)} captions for label: {label}")
            sampled_captions[label] = available_captions
        else:
            print(f"🎲 Sampling {captions_per_label} captions from {len(available_captions)} available for label: {label}")
            sampled_captions[label] = random.sample(available_captions, captions_per_label)

    return sampled_captions


def format_captions_for_file(label_captions: Dict[str, List[str]], captions_per_label: int) -> str:
    """Format captions for saving to text file."""
    output_lines = []

    # Determine header text based on sampling strategy
    if captions_per_label == -1:
        header_text = "ALL AVAILABLE CAPTIONS PER L1 LABEL"
    else:
        header_text = f"{captions_per_label} CAPTIONS PER L1 LABEL"

    output_lines.append("=" * 80)
    output_lines.append(f"DATASET CAPTIONS - {header_text}")
    output_lines.append("=" * 80)
    output_lines.append("")

    for label, captions in label_captions.items():
        output_lines.append(f"L1 LABEL: {label}")
        output_lines.append("-" * 60)

        if not captions:
            output_lines.append("   ⚠️  No captions available for this label")
        else:
            for i, caption in enumerate(captions, 1):
                output_lines.append(f"   {i:2d}. {caption}")

        output_lines.append(f"\n   📊 Total captions shown: {len(captions)}")
        output_lines.append("")

    return "\n".join(output_lines)


def save_captions_to_file(label_captions: Dict[str, List[str]], output_path: str, captions_per_label: int):
    """Save captions to text file."""
    formatted_content = format_captions_for_file(label_captions, captions_per_label)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_content)

    print(f"✅ Captions saved to: {output_path}")


def generate_default_output_path(data_path: str, captions_per_label: int) -> str:
    """Generate a default output path based on the input data path."""
    data_file = Path(data_path)

    # Create output filename based on sampling strategy
    if captions_per_label == -1:
        filename = f"{data_file.stem}_captions_all.txt"
    else:
        filename = f"{data_file.stem}_captions_{captions_per_label}_per_label.txt"

    return str(data_file.parent / filename)


def main():
    """Main function to extract and save captions by L1 label."""
    parser = argparse.ArgumentParser(
        description="Extract captions from any CASAS dataset grouped by L1 activity labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--data-path',
        required=True,
        help='Path to the dataset JSON file (e.g., data/processed/casas/milan/seq20/milan_test.json)'
    )

    parser.add_argument(
        '--output-path',
        help='Path for output text file. If not provided, auto-generated based on input path.'
    )

    parser.add_argument(
        '--captions-per-label',
        type=int,
        default=40,
        help='Number of captions to extract per label. Use -1 to extract all available captions. (default: 40)'
    )

    parser.add_argument(
        '--list-labels-only',
        action='store_true',
        help='Only list available labels without extracting captions'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducible sampling (default: 42)'
    )

    args = parser.parse_args()

    # Set random seed for reproducible sampling
    random.seed(args.random_seed)

    print("🏠 CASAS DATASET CAPTION EXTRACTION")
    print("=" * 50)

    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data file not found: {args.data_path}")
        return 1

    # Load labels from dataset
    try:
        labels = load_dataset_labels(args.data_path)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return 1

    if args.list_labels_only:
        print("✅ Label listing completed!")
        return 0

    # Generate output path if not provided
    if args.output_path is None:
        args.output_path = generate_default_output_path(args.data_path, args.captions_per_label)
        print(f"📁 Auto-generated output path: {args.output_path}")

    # Extract captions
    try:
        label_captions = extract_captions_by_label(args.data_path, labels, args.captions_per_label)
    except Exception as e:
        print(f"❌ Error extracting captions: {e}")
        return 1

    # Save to file
    try:
        save_captions_to_file(label_captions, args.output_path, args.captions_per_label)
    except Exception as e:
        print(f"❌ Error saving captions: {e}")
        return 1

    print(f"\n✅ Caption extraction completed!")
    print(f"📈 Summary:")
    total_captions = sum(len(captions) for captions in label_captions.values())
    labels_with_captions = sum(1 for captions in label_captions.values() if len(captions) > 0)
    print(f"   - Labels processed: {len(labels)}")
    print(f"   - Labels with captions: {labels_with_captions}")
    print(f"   - Total captions extracted: {total_captions}")
    print(f"   - Output file: {args.output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
