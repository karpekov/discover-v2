#!/usr/bin/env python3
"""Convert raw samples to compact JSON representation.

This script converts sampled sensor data to a compact JSON format optimized
for LLM caption generation. The compact format:
- Removes the full sensor sequence (not needed for captions)
- Keeps essential metadata (duration, rooms, special sensors, time context)
- Adds derived fields (movement patterns, time of day)

Output format: JSONL (one compact JSON object per line)

Usage:
# Convert train split
python src/captions/convert_to_compact.py \
    --input data/processed/casas/milan/FD_60/train.json \
    --output data/processed/casas/milan/FD_60/train_compact.jsonl

# Convert all splits in a directory
python src/captions/convert_to_compact.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --split all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from captions.llm_based.compact_json import to_compact_caption_json, convert_samples_to_compact_jsonl


def load_samples(input_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    return data.get('samples', [])


def save_compact_jsonl(compact_samples: List[Dict[str, Any]], output_path: Path):
    """Save compact samples to JSONL file."""
    with open(output_path, 'w') as f:
        for sample in compact_samples:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved {len(compact_samples)} compact samples to {output_path}")


def save_compact_json(compact_samples: List[Dict[str, Any]], output_path: Path):
    """Save compact samples to JSON file."""
    output_data = {'samples': compact_samples}
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved {len(compact_samples)} compact samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert raw samples to compact JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/output modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--input',
        type=str,
        help='Input JSON file path'
    )
    group.add_argument(
        '--data-dir',
        type=str,
        help='Data directory containing train.json, val.json, test.json'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (for --input mode). Default: input_compact.jsonl'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (for --data-dir mode). Default: same as data-dir'
    )

    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test', 'all'],
        default='all',
        help='Which split to process when using --data-dir (default: all)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['jsonl', 'json'],
        default='jsonl',
        help='Output format (default: jsonl)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print sample compact JSON'
    )

    args = parser.parse_args()

    if args.input:
        # Single file mode
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            stem = input_path.stem
            suffix = '.jsonl' if args.format == 'jsonl' else '.json'
            output_path = input_path.parent / f'{stem}_compact{suffix}'

        print(f"Converting samples from {input_path}...")

        # Load samples
        samples = load_samples(input_path)
        print(f"Loaded {len(samples)} samples")

        # Convert to compact
        compact_samples = convert_samples_to_compact_jsonl(samples)
        print(f"Converted {len(compact_samples)} samples")

        # Save output
        if args.format == 'jsonl':
            save_compact_jsonl(compact_samples, output_path)
        else:
            save_compact_json(compact_samples, output_path)

        # Print sample if verbose
        if args.verbose and compact_samples:
            print("\nSample compact JSON:")
            print(json.dumps(compact_samples[0], indent=2))

    else:
        # Directory mode
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            sys.exit(1)

        output_dir = Path(args.output_dir) if args.output_dir else data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine splits to process
        splits_to_process = []
        if args.split == 'all':
            splits_to_process = ['train', 'val', 'test']
        else:
            splits_to_process = [args.split]

        print(f"Converting samples from {data_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Output format: {args.format}")
        print(f"Splits: {', '.join(splits_to_process)}")
        print("=" * 80)

        # Process each split
        for split in splits_to_process:
            input_path = data_dir / f'{split}.json'

            if not input_path.exists():
                print(f"\nWarning: {split}.json not found, skipping")
                continue

            print(f"\nProcessing {split} split...")

            # Load samples
            samples = load_samples(input_path)
            print(f"  Loaded {len(samples)} samples")

            if not samples:
                continue

            # Convert to compact
            compact_samples = convert_samples_to_compact_jsonl(samples)
            print(f"  Converted {len(compact_samples)} samples")

            # Save output
            suffix = '.jsonl' if args.format == 'jsonl' else '.json'
            output_path = output_dir / f'{split}_compact{suffix}'

            if args.format == 'jsonl':
                save_compact_jsonl(compact_samples, output_path)
            else:
                save_compact_json(compact_samples, output_path)

            # Print sample if verbose and first split
            if args.verbose and compact_samples and split == splits_to_process[0]:
                print("\n  Sample compact JSON:")
                sample_str = json.dumps(compact_samples[0], indent=4)
                for line in sample_str.split('\n')[:20]:  # Show first 20 lines
                    print(f"    {line}")
                if len(sample_str.split('\n')) > 20:
                    print("    ...")

        print("\n" + "=" * 80)
        print("Conversion complete!")


if __name__ == '__main__':
    main()

