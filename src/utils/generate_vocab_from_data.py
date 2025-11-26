#!/usr/bin/env python3
"""
Generate vocabulary JSON from sampled data.

This script extracts unique values for each categorical field from sampled data
and creates a vocab.json file.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def generate_vocab_from_samples(data_path: str, output_path: str = None,
                                include_fields: list = None, exclude_fields: list = None):
    """
    Generate vocabulary from sampled data JSON.

    Args:
        data_path: Path to train.json or test.json from sampling
        output_path: Where to save vocab.json (default: same directory)
        include_fields: List of specific fields to include (None = all fields)
        exclude_fields: List of fields to exclude (None = exclude nothing)
    """
    print(f"Loading data from: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data.get('samples', data)

    # Collect unique values for each field
    vocab = defaultdict(set)

    print(f"Processing {len(samples)} samples...")
    if include_fields:
        print(f"Including only fields: {include_fields}")
    if exclude_fields:
        print(f"Excluding fields: {exclude_fields}")

    for i, sample in enumerate(samples):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(samples)} samples...")

        sensor_sequence = sample['sensor_sequence']

        for event in sensor_sequence:
            # Collect all categorical fields
            for key, value in event.items():
                # Skip non-categorical fields
                if key in ['x', 'y', 'z', 'timestamp', 'datetime', 'time_delta']:
                    continue

                # Normalize field names
                if key in ['sensor', 'sensor_id']:
                    normalized_field = 'sensor'
                elif key in ['state', 'event_type', 'message']:
                    normalized_field = 'state'
                elif key in ['room', 'room_id', 'location']:
                    normalized_field = 'room_id'
                elif key in ['activity', 'first_activity', 'activity_l1']:
                    normalized_field = 'activity'
                elif key in ['activity_l2', 'first_activity_l2']:
                    normalized_field = 'activity_l2'
                else:
                    # Generic field
                    normalized_field = key

                # Apply include/exclude filters
                if include_fields is not None and normalized_field not in include_fields:
                    continue
                if exclude_fields is not None and normalized_field in exclude_fields:
                    continue

                vocab[normalized_field].add(str(value))

    # Convert sets to sorted lists and create mappings
    vocab_dict = {}
    for field, values in vocab.items():
        # Sort values for consistency
        sorted_values = sorted(list(values))

        # Create index mapping (0 is reserved for padding/unknown)
        vocab_dict[field] = {value: idx + 1 for idx, value in enumerate(sorted_values)}

        # Add UNK token at index 0
        vocab_dict[field]['UNK'] = 0

    # Determine output path
    if output_path is None:
        output_path = Path(data_path).parent / 'vocab.json'

    # Save vocabulary
    print(f"\nSaving vocabulary to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(vocab_dict, f, indent=2)

    # Print statistics
    print("\nVocabulary statistics:")
    for field, mapping in vocab_dict.items():
        print(f"  {field}: {len(mapping)} unique values")

    print(f"\n✅ Vocabulary saved successfully!")
    return vocab_dict


def main():
    parser = argparse.ArgumentParser(
        description='Generate vocab.json from sampled data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate vocab with all fields (default)
  python src/utils/generate_vocab_from_data.py --data data/processed/casas/milan/FD_60/train.json

  # Generate vocab without sensor field
  python src/utils/generate_vocab_from_data.py \\
      --data data/processed/casas/milan/FD_60/train.json \\
      --output data/processed/casas/milan/FD_60/vocab_no_sensor.json \\
      --exclude sensor

  # Generate vocab with only specific fields
  python src/utils/generate_vocab_from_data.py \\
      --data data/processed/casas/milan/FD_60/train.json \\
      --output data/processed/casas/milan/FD_60/vocab_state_only.json \\
      --include state

  # Generate vocab with state and room_id only
  python src/utils/generate_vocab_from_data.py \\
      --data data/processed/casas/milan/FD_60/train.json \\
      --output data/processed/casas/milan/FD_60/vocab_state_room.json \\
      --include state room_id
        """
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to train.json from sampling step'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for vocab.json (default: same directory as input)'
    )
    parser.add_argument(
        '--include',
        type=str,
        nargs='+',
        default=None,
        help='Only include these fields (e.g., --include state room_id)'
    )
    parser.add_argument(
        '--exclude',
        type=str,
        nargs='+',
        default=None,
        help='Exclude these fields (e.g., --exclude sensor)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.include and args.exclude:
        parser.error("Cannot use both --include and --exclude at the same time")

    generate_vocab_from_samples(
        args.data,
        args.output,
        include_fields=args.include,
        exclude_fields=args.exclude
    )


if __name__ == '__main__':
    main()

