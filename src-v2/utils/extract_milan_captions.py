#!/usr/bin/env python3
"""
Extract 40 captions per L1 label from Milan training dataset and save to file.
"""

import json
import random
from collections import defaultdict
from typing import Dict, List
import os
from pathlib import Path

def load_milan_metadata():
    """Load Milan L1 labels from metadata."""
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    metadata_path = project_root / "metadata" / "city_metadata.json"

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Get Milan L1 labels from the 'lable' field (note the typo in the original file)
    milan_labels = list(metadata['milan']['lable'].keys())
    print(f"Found {len(milan_labels)} L1 labels in Milan metadata:")
    for label in milan_labels:
        print(f"  - {label}")
    print()

    return milan_labels

def extract_captions_by_label(data_path: str, target_labels: List[str], captions_per_label: int = 40) -> Dict[str, List[str]]:
    """
    Extract captions from Milan training dataset grouped by L1 labels.

    Args:
        data_path: Path to the Milan training JSON file
        target_labels: List of L1 labels to extract captions for
        captions_per_label: Number of captions to extract per label

    Returns:
        Dictionary mapping L1 labels to lists of captions
    """
    print(f"Loading Milan training dataset from: {data_path}")

    # Initialize collections for each label
    label_captions = defaultdict(list)
    label_sample_count = defaultdict(int)

    # Read the JSON file line by line to handle large files
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
        elif len(available_captions) <= captions_per_label:
            print(f"📝 Using all {len(available_captions)} captions for label: {label}")
            sampled_captions[label] = available_captions
        else:
            print(f"🎲 Sampling {captions_per_label} captions from {len(available_captions)} available for label: {label}")
            sampled_captions[label] = random.sample(available_captions, captions_per_label)

    return sampled_captions

def format_captions_for_file(label_captions: Dict[str, List[str]]) -> str:
    """Format captions for saving to text file."""
    output_lines = []

    output_lines.append("=" * 80)
    output_lines.append("MILAN TEST DATASET - 40 CAPTIONS PER L1 LABEL")
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

def save_captions_to_file(label_captions: Dict[str, List[str]], output_path: str):
    """Save captions to text file."""
    formatted_content = format_captions_for_file(label_captions)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_content)

    print(f"✅ Captions saved to: {output_path}")

def main():
    """Main function to extract and save Milan captions by L1 label."""
    print("🏠 MILAN DATASET CAPTION EXTRACTION")
    print("=" * 50)

    # Load Milan L1 labels from metadata
    milan_labels = load_milan_metadata()

    # Get project root and set paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # Path to Milan test dataset
    test_data_path = project_root / "data" / "processed" / "casas" / "milan" / "training_50" / "presegmented_test.json"

    # Output path for captions
    output_path = project_root / "data" / "processed" / "casas" / "milan" / "training_50" / "test_l1_captions_40_per_label.txt"

    # Extract captions
    label_captions = extract_captions_by_label(str(test_data_path), milan_labels, captions_per_label=40)

    # Save to file
    save_captions_to_file(label_captions, str(output_path))

    print(f"\n✅ Caption extraction completed!")
    print(f"📈 Summary:")
    total_captions = sum(len(captions) for captions in label_captions.values())
    labels_with_captions = sum(1 for captions in label_captions.values() if len(captions) > 0)
    print(f"   - Labels processed: {len(milan_labels)}")
    print(f"   - Labels with captions: {labels_with_captions}")
    print(f"   - Total captions extracted: {total_captions}")
    print(f"   - Output file: {output_path}")

if __name__ == "__main__":
    # Set random seed for reproducible sampling
    random.seed(42)
    main()
