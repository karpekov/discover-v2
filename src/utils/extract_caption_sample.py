#!/usr/bin/env python3
"""
Extract captions from any CASAS dataset and save samples grouped by activity labels.

This script works with the NEW data structure where sensor data and captions are in separate files.
It can extract captions from any processed CASAS dataset (Milan, Aruba, Cairo, etc.)
with any sampling configuration (FD_60, FL_50, etc.) grouped by L1 activity labels.

Usage Examples:
    # Extract 5 captions per label from Milan FD_60_p test data (baseline style, scattered)
    python src/utils/extract_caption_sample.py \
        --data-path data/processed/casas/milan/FD_60_p/test.json \
        --captions-per-label 5

    # Extract from specific caption style (e.g., sourish, llm_gpt4)
    python src/utils/extract_caption_sample.py \
        --data-path data/processed/casas/milan/FD_60_p/test.json \
        --caption-style sourish \
        --captions-per-label 5

    # Extract from Aruba dataset
    python src/utils/extract_caption_sample.py \
        --data-path data/processed/casas/aruba/FD_60_p/train.json \
        --captions-per-label 4

    # Manually specify captions file
    python src/utils/extract_caption_sample.py \
        --data-path data/processed/casas/milan/FD_60_p/test.json \
        --captions-path data/processed/casas/milan/FD_60_p/test_captions_baseline.json \
        --captions-per-label 5

    # Extract all available captions (no sampling)
    python src/utils/extract_caption_sample.py \
        --data-path data/processed/casas/aruba/FD_60_p/train.json \
        --captions-per-label -1

    # Disable scattered sampling (sample randomly from all captions)
    python src/utils/extract_caption_sample.py \
        --data-path data/processed/casas/milan/FD_60_p/test.json \
        --captions-per-label 5 \
        --no-scatter

    # List available labels without extracting captions
    python src/utils/extract_caption_sample.py \
        --data-path data/processed/casas/milan/FD_60_p/test.json \
        --list-labels-only
"""

import json
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path


def auto_detect_captions_file(data_path: str, caption_style: str = 'baseline') -> Optional[str]:
    """
    Auto-detect the captions file based on the data file path.

    Args:
        data_path: Path to the sensor data JSON file (e.g., test.json)
        caption_style: Caption style to look for (default: 'baseline')

    Returns:
        Path to the captions file, or None if not found
    """
    data_file = Path(data_path)
    # Extract split name (train, test, val) from filename
    split_name = data_file.stem  # e.g., 'test' from 'test.json'

    # Look for caption file in the same directory
    caption_filename = f"{split_name}_captions_{caption_style}.json"
    caption_path = data_file.parent / caption_filename

    if caption_path.exists():
        return str(caption_path)

    # Try without style suffix (legacy)
    caption_filename = f"{split_name}_captions.json"
    caption_path = data_file.parent / caption_filename
    if caption_path.exists():
        return str(caption_path)

    return None


def load_dataset_labels(data_path: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Extract unique L1 activity labels from a sensor data file.

    Args:
        data_path: Path to the sensor data JSON file (NEW format with 'samples' key)

    Returns:
        Tuple of (labels, sample_id_to_label mapping)
    """
    print(f"Loading sensor data from: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data.get('samples', [])
    print(f"Loaded {len(samples)} samples from dataset")

    # Extract L1 labels and create mapping
    labels = set()
    sample_id_to_label = {}

    for sample in samples:
        sample_id = sample.get('sample_id', '')
        metadata = sample.get('metadata', {})
        ground_truth = metadata.get('ground_truth_labels', {})
        primary_l1 = ground_truth.get('primary_l1', '')

        if primary_l1:
            labels.add(primary_l1)
            sample_id_to_label[sample_id] = primary_l1

    labels = sorted(list(labels))
    print(f"Found {len(labels)} unique L1 labels:")
    for label in labels:
        count = sum(1 for l in sample_id_to_label.values() if l == label)
        print(f"  - {label}: {count} samples")
    print()

    return labels, sample_id_to_label


def extract_captions_by_label(
    captions_path: str,
    sample_id_to_label: Dict[str, str],
    target_labels: List[str],
    captions_per_label: int = 40,
    scatter_across_samples: bool = True
) -> Dict[str, List[str]]:
    """
    Extract captions from captions file grouped by L1 labels.

    Args:
        captions_path: Path to the captions JSON file (NEW format)
        sample_id_to_label: Mapping from sample_id to L1 label
        target_labels: List of L1 labels to extract captions for
        captions_per_label: Number of captions to extract per label (-1 for all available)
        scatter_across_samples: If True, sample one caption per sample to maximize scatter

    Returns:
        Dictionary mapping L1 labels to lists of captions
    """
    print(f"Extracting captions from: {captions_path}")

    # Initialize collections for each label
    # For scattered sampling: store list of (sample_id, captions) tuples per label
    label_samples = defaultdict(list)
    label_sample_count = defaultdict(int)

    # Read the captions JSON file (NEW format)
    with open(captions_path, 'r') as f:
        caption_data = json.load(f)

    caption_entries = caption_data.get('captions', [])
    print(f"Loaded {len(caption_entries)} caption entries from file")

    # Extract captions for each sample and group by L1 label
    for entry in caption_entries:
        sample_id = entry.get('sample_id', '')

        # Get the L1 label for this sample
        label = sample_id_to_label.get(sample_id, '')

        if not label or label not in target_labels:
            continue

        label_sample_count[label] += 1

        # Get all captions for this sample
        captions = entry.get('captions', [])
        captions = [c.strip() for c in captions if c and c.strip()]

        if captions:
            label_samples[label].append((sample_id, captions))

    print(f"\nFound samples for {len(label_samples)} labels:")
    for label in sorted(label_samples.keys()):
        samples = label_samples[label]
        total_captions = sum(len(caps) for _, caps in samples)
        print(f"  {label}: {total_captions} captions from {len(samples)} samples")

    # Sample the requested number of captions for each label
    sampled_captions = {}
    for label in target_labels:
        available_samples = label_samples[label]

        if len(available_samples) == 0:
            print(f"⚠️  No captions found for label: {label}")
            sampled_captions[label] = []
        elif captions_per_label == -1:
            # Use all available captions
            all_captions = [cap for _, caps in available_samples for cap in caps]
            print(f"📝 Using all {len(all_captions)} captions for label: {label}")
            sampled_captions[label] = all_captions
        elif scatter_across_samples:
            # Sample one caption per sample to maximize scatter
            if len(available_samples) <= captions_per_label:
                # Not enough samples, use one caption from each sample
                selected_captions = [random.choice(caps) for _, caps in available_samples]
                print(f"📝 Using 1 caption from each of {len(available_samples)} samples for label: {label}")
            else:
                # Enough samples, randomly select N samples and take one caption from each
                selected_samples = random.sample(available_samples, captions_per_label)
                selected_captions = [random.choice(caps) for _, caps in selected_samples]
                print(f"🎲 Sampling 1 caption from each of {captions_per_label} random samples (out of {len(available_samples)}) for label: {label}")
            sampled_captions[label] = selected_captions
        else:
            # Original behavior: sample randomly from all captions
            all_captions = [cap for _, caps in available_samples for cap in caps]
            if len(all_captions) <= captions_per_label:
                print(f"📝 Using all {len(all_captions)} captions for label: {label}")
                sampled_captions[label] = all_captions
            else:
                print(f"🎲 Sampling {captions_per_label} captions from {len(all_captions)} available for label: {label}")
                sampled_captions[label] = random.sample(all_captions, captions_per_label)

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


def generate_default_output_path(data_path: str, captions_per_label: int, caption_style: str = 'baseline') -> str:
    """Generate a default output path based on the input data path."""
    data_file = Path(data_path)

    # Create output filename based on sampling strategy
    if captions_per_label == -1:
        filename = f"{data_file.stem}_sample_{caption_style}_all.txt"
    else:
        filename = f"{data_file.stem}_sample_{caption_style}_{captions_per_label}_per_label.txt"

    return str(data_file.parent / filename)


def main():
    """Main function to extract and save captions by L1 label."""
    parser = argparse.ArgumentParser(
        description="Extract captions from any CASAS dataset grouped by L1 activity labels (NEW data structure)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--data-path',
        required=True,
        help='Path to the sensor data JSON file (e.g., data/processed/casas/milan/FD_60_p/test.json)'
    )

    parser.add_argument(
        '--captions-path',
        help='Path to the captions JSON file. If not provided, auto-detects based on data-path.'
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
        '--caption-style',
        default='baseline',
        help='Caption style to use when auto-detecting captions file (default: baseline)'
    )

    parser.add_argument(
        '--no-scatter',
        action='store_true',
        help='Disable scattered sampling (sample randomly from all captions instead of one per sample)'
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

    print("🏠 CASAS DATASET CAPTION EXTRACTION (NEW FORMAT)")
    print("=" * 60)

    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data file not found: {args.data_path}")
        return 1

    # Load labels from sensor data
    try:
        labels, sample_id_to_label = load_dataset_labels(args.data_path)
    except Exception as e:
        print(f"❌ Error loading sensor data: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if args.list_labels_only:
        print("✅ Label listing completed!")
        return 0

    # Auto-detect captions file if not provided
    if args.captions_path is None:
        args.captions_path = auto_detect_captions_file(args.data_path, args.caption_style)
        if args.captions_path is None:
            print(f"❌ Error: Could not auto-detect captions file.")
            print(f"   Please provide --captions-path explicitly.")
            return 1
        print(f"📁 Auto-detected captions file: {args.captions_path}")

    # Check if captions file exists
    if not os.path.exists(args.captions_path):
        print(f"❌ Error: Captions file not found: {args.captions_path}")
        return 1

    # Generate output path if not provided
    if args.output_path is None:
        args.output_path = generate_default_output_path(
            args.data_path, args.captions_per_label, args.caption_style
        )
        print(f"📁 Auto-generated output path: {args.output_path}")

    # Extract captions
    try:
        label_captions = extract_captions_by_label(
            args.captions_path, sample_id_to_label, labels, args.captions_per_label,
            scatter_across_samples=not args.no_scatter
        )
    except Exception as e:
        print(f"❌ Error extracting captions: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save to file
    try:
        save_captions_to_file(label_captions, args.output_path, args.captions_per_label)
    except Exception as e:
        print(f"❌ Error saving captions: {e}")
        import traceback
        traceback.print_exc()
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
