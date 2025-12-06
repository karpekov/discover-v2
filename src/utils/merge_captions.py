#!/usr/bin/env python3
"""Merge multiple caption files into a single combined caption file.

This script allows you to combine captions from different sources (baseline, LLM, etc.)
with configurable ratios per source.

Usage Examples:

# Merge all captions from baseline and LLM (4+4=8 captions per sample)
python src/utils/merge_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --caption-files train_captions_baseline.json train_llm_gemini_gemini_2_5_flash.json \
    --output-style rb_and_llm \
    --splits all

# Merge with specific ratios (2 from baseline, 4 from LLM = 6 captions per sample)
python src/utils/merge_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --caption-files train_captions_baseline.json train_llm_gemini_gemini_2_5_flash.json \
    --ratios 2 4 \
    --output-style rb2_llm4 \
    --splits all

# Single split only
python src/utils/merge_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --caption-files train_captions_baseline.json train_llm_gemini_gemini_2_5_flash.json \
    --output-style rb_and_llm \
    --splits train

# Merge three sources with different ratios
python src/utils/merge_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --caption-files train_captions_baseline.json train_captions_sourish.json train_llm_gemini_gemini_2_5_flash.json \
    --ratios 2 2 4 \
    --output-style rb2_sourish2_llm4 \
    --splits all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict


def load_caption_file(file_path: Path) -> Dict[str, Any]:
    """Load captions from JSON file.

    Args:
        file_path: Path to caption file

    Returns:
        Dictionary with 'captions' key containing list of samples
    """
    print(f"  Loading: {file_path.name}")
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Normalize format - ensure we have 'captions' key with list
    if 'captions' in data and isinstance(data['captions'], list):
        samples = data['captions']
    elif 'samples' in data:
        samples = data['samples']
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError(f"Unexpected format in {file_path.name}")

    print(f"    Found {len(samples)} samples")
    return {'captions': samples}


def merge_captions(
    caption_files: List[Path],
    ratios: Optional[List[int]] = None,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Merge captions from multiple files.

    Args:
        caption_files: List of paths to caption files
        ratios: Optional list of how many captions to take from each file
                If None, takes all captions from all files
        random_seed: Random seed for reproducible sampling

    Returns:
        Merged captions dictionary in standard format
    """
    import random
    random.seed(random_seed)

    # Load all caption files
    all_data = [load_caption_file(f) for f in caption_files]

    # Get sample IDs from first file (should be consistent across all)
    first_samples = all_data[0]['captions']
    sample_ids = [s['sample_id'] for s in first_samples]

    print(f"\n  Processing {len(sample_ids)} samples...")

    # Create lookup dictionaries for each source
    source_lookups = []
    for i, data in enumerate(all_data):
        lookup = {s['sample_id']: s for s in data['captions']}
        source_lookups.append(lookup)

        # Verify all samples are present
        missing = set(sample_ids) - set(lookup.keys())
        if missing:
            print(f"    ⚠️  Warning: {len(missing)} samples missing in source {i+1}")

    # Determine ratios
    if ratios is None:
        # Use all captions from all sources
        ratios = []
        for data in all_data:
            # Get number of captions from first sample
            first_sample = data['captions'][0]
            num_captions = len(first_sample.get('captions', []))
            ratios.append(num_captions)
        print(f"  Using all captions: {ratios}")
    else:
        print(f"  Using specified ratios: {ratios}")

    if len(ratios) != len(caption_files):
        raise ValueError(f"Number of ratios ({len(ratios)}) must match number of files ({len(caption_files)})")

    # Merge captions for each sample
    merged_samples = []
    samples_with_issues = 0

    for sample_id in sample_ids:
        merged_captions = []

        # Collect captions from each source
        for source_idx, (lookup, ratio) in enumerate(zip(source_lookups, ratios)):
            if sample_id not in lookup:
                samples_with_issues += 1
                continue

            source_sample = lookup[sample_id]
            source_captions = source_sample.get('captions', [])

            # Sample or take all captions based on ratio
            if ratio >= len(source_captions):
                # Take all available
                selected_captions = source_captions
            else:
                # Randomly sample 'ratio' captions
                selected_captions = random.sample(source_captions, ratio)

            merged_captions.extend(selected_captions)

        # Create merged sample entry
        # Use metadata from first source
        base_sample = source_lookups[0].get(sample_id, {})
        merged_sample = {
            'sample_id': sample_id,
            'captions': merged_captions,
            'metadata': base_sample.get('metadata', {})
        }

        # Add merge metadata
        if 'metadata' not in merged_sample:
            merged_sample['metadata'] = {}
        merged_sample['metadata']['caption_sources'] = [f.stem for f in caption_files]
        merged_sample['metadata']['caption_ratios'] = ratios
        merged_sample['metadata']['num_captions'] = len(merged_captions)

        merged_samples.append(merged_sample)

    if samples_with_issues > 0:
        print(f"    ⚠️  {samples_with_issues} samples had missing data in some sources")

    total_captions = sum(ratios)
    print(f"  ✓ Merged {len(merged_samples)} samples, {total_captions} captions per sample")

    return {'captions': merged_samples}


def merge_splits(
    data_dir: Path,
    caption_files: List[str],
    output_style: str,
    splits: List[str],
    ratios: Optional[List[int]] = None,
    random_seed: int = 42
):
    """Merge captions for multiple splits.

    Args:
        data_dir: Directory containing caption files
        caption_files: List of caption file patterns (e.g., ['train_captions_baseline.json'])
        output_style: Style name for output files
        splits: List of splits to process ('train', 'val', 'test')
        ratios: Optional ratios for each source
        random_seed: Random seed for reproducible sampling
    """
    print("="*80)
    print("CAPTION MERGING UTILITY")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output style: {output_style}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Sources: {len(caption_files)}")
    for i, cf in enumerate(caption_files, 1):
        print(f"  {i}. {cf}")
    if ratios:
        print(f"Ratios: {ratios}")
    print("="*80)

    results_summary = []

    for split in splits:
        print(f"\n{'='*80}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*80}")

        # Build paths for this split
        split_caption_files = []
        for caption_pattern in caption_files:
            # Replace {split} placeholder or extract pattern
            # Handle patterns like "train_captions_baseline.json" -> "val_captions_baseline.json"
            # or "train_llm_gemini_gemini_2_5_flash.json" -> "val_llm_gemini_gemini_2_5_flash.json"

            if '{split}' in caption_pattern:
                caption_file = caption_pattern.replace('{split}', split)
            else:
                # Try to replace split name in the pattern
                # Detect common patterns
                for s in ['train', 'val', 'test']:
                    if caption_pattern.startswith(f'{s}_'):
                        caption_file = caption_pattern.replace(f'{s}_', f'{split}_', 1)
                        break
                else:
                    # Fallback: assume it's a filename and prepend split
                    caption_file = f"{split}_{caption_pattern}"

            file_path = data_dir / caption_file

            if not file_path.exists():
                print(f"  ⚠️  Warning: {caption_file} not found, skipping {split}")
                break

            split_caption_files.append(file_path)

        # Skip if any file is missing
        if len(split_caption_files) != len(caption_files):
            print(f"  ⚠️  Skipping {split} - not all caption files found")
            continue

        # Merge captions
        try:
            merged_data = merge_captions(split_caption_files, ratios, random_seed)

            # Save output
            output_filename = f"{split}_captions_{output_style}.json"
            output_path = data_dir / output_filename

            with open(output_path, 'w') as f:
                json.dump(merged_data, f, indent=2)

            print(f"\n  ✓ Saved to: {output_filename}")

            results_summary.append({
                'split': split,
                'output': output_filename,
                'num_samples': len(merged_data['captions']),
                'captions_per_sample': sum(ratios) if ratios else 'all'
            })

        except Exception as e:
            print(f"  ✗ Error processing {split}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if results_summary:
        print(f"✓ Successfully merged {len(results_summary)} split(s):")
        for result in results_summary:
            print(f"  - {result['split']}: {result['output']}")
            print(f"      {result['num_samples']} samples, {result['captions_per_sample']} captions/sample")
    else:
        print("✗ No splits were successfully merged")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple caption files into one',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing caption files'
    )

    parser.add_argument(
        '--caption-files',
        nargs='+',
        required=True,
        help='Caption file names or patterns to merge (e.g., train_captions_baseline.json train_llm_gemini_gemini_2_5_flash.json)'
    )

    parser.add_argument(
        '--output-style',
        type=str,
        required=True,
        help='Style name for output files (e.g., rb_and_llm, rb2_llm4)'
    )

    parser.add_argument(
        '--splits',
        nargs='+',
        default=['all'],
        help='Splits to process: train, val, test, or "all" (default: all)'
    )

    parser.add_argument(
        '--ratios',
        nargs='+',
        type=int,
        default=None,
        help='Number of captions to take from each source (default: all captions from all sources)'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducible sampling (default: 42)'
    )

    args = parser.parse_args()

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Determine splits to process
    if 'all' in args.splits or args.splits == ['all']:
        splits_to_process = ['train', 'val', 'test']
    else:
        splits_to_process = args.splits

    # Validate ratios
    if args.ratios and len(args.ratios) != len(args.caption_files):
        print(f"Error: Number of ratios ({len(args.ratios)}) must match number of caption files ({len(args.caption_files)})")
        sys.exit(1)

    # Run merging
    merge_splits(
        data_dir=data_dir,
        caption_files=args.caption_files,
        output_style=args.output_style,
        splits=splits_to_process,
        ratios=args.ratios,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()

