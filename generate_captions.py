#!/usr/bin/env python3
"""Generate captions for sampled sensor data.

This script loads sampled data (from Step 1) and generates captions (Step 3).
Captions are saved alongside the sampled data with style-specific suffixes.

Output Format: {split}_captions_{style}.json
- baseline → train_captions_baseline.json
- sourish → train_captions_sourish.json
- llm_gpt4 → train_captions_llm_gpt4.json

Usage:
    # Generate baseline captions for Milan data
    python generate_captions.py \\
        --data-dir data/processed/casas/milan/fixed_length_50 \\
        --caption-style baseline \\
        --dataset-name milan

    # Generate Sourish captions
    python generate_captions.py \\
        --data-dir data/processed/casas/milan/fixed_length_50 \\
        --caption-style sourish \\
        --dataset-name milan

    # Generate LLM captions (placeholder)
    python generate_captions.py \\
        --data-dir data/processed/casas/milan/fixed_length_50 \\
        --caption-style llm \\
        --llm-model gpt4 \\
        --dataset-name milan

    # Generate multiple styles for comparison
    python generate_captions.py --data-dir data/... --caption-style baseline --dataset-name milan
    python generate_captions.py --data-dir data/... --caption-style sourish --dataset-name milan
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from captions import (
    BaselineCaptionGenerator,
    SourishCaptionGenerator,
    LLMCaptionGenerator,
    RuleBasedCaptionConfig,
    LLMCaptionConfig
)


def load_sampled_data(data_path: Path) -> Dict[str, Any]:
    """Load sampled data from JSON file."""
    with open(data_path, 'r') as f:
        return json.load(f)


def save_captions(caption_outputs: List, output_path: Path):
    """Save caption outputs to JSON file."""
    captions_data = {
        'captions': [co.to_dict() for co in caption_outputs]
    }

    with open(output_path, 'w') as f:
        json.dump(captions_data, f, indent=2)

    print(f"Saved {len(caption_outputs)} caption outputs to {output_path}")


def generate_captions_for_split(
    data: Dict[str, Any],
    split: str,
    generator,
    output_dir: Path,
    style_suffix: str
):
    """Generate captions for a data split (train or test)."""

    samples = data.get('samples', [])

    if not samples:
        print(f"Warning: No samples found in {split} split")
        return []

    print(f"\nGenerating captions for {len(samples)} {split} samples...")

    caption_outputs = []
    for sample in tqdm(samples, desc=f"Processing {split}"):
        output = generator.generate(
            sensor_sequence=sample['sensor_sequence'],
            metadata=sample['metadata'],
            sample_id=sample['sample_id']
        )
        caption_outputs.append(output)

    # Save captions with style suffix
    output_path = output_dir / f'{split}_captions_{style_suffix}.json'
    save_captions(caption_outputs, output_path)

    # Print statistics
    stats = generator.get_statistics(caption_outputs)
    print(f"\n{split.upper()} Caption Statistics:")
    print(f"  Total samples: {stats.get('total_samples', 0)}")
    print(f"  Total captions: {stats.get('total_captions', 0)}")
    print(f"  Avg captions/sample: {stats.get('avg_captions_per_sample', 0):.2f}")
    if 'caption_length_stats' in stats:
        length_stats = stats['caption_length_stats']
        print(f"  Caption length: {length_stats.get('mean_tokens', 0):.1f} ± {length_stats.get('std_tokens', 0):.1f} tokens")
        print(f"    Min: {length_stats.get('min_tokens', 0)}, Max: {length_stats.get('max_tokens', 0)}")

    if 'sample_captions' in stats and stats['sample_captions']:
        print(f"\n  Sample captions:")
        for i, caption in enumerate(stats['sample_captions'][:2], 1):
            print(f"    {i}. {caption}")

    return caption_outputs


def main():
    parser = argparse.ArgumentParser(
        description='Generate captions for sampled sensor data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing sampled data (train.json and test.json)'
    )

    parser.add_argument(
        '--caption-style',
        type=str,
        choices=['baseline', 'sourish', 'mixed', 'llm'],
        default='baseline',
        help='Caption generation style'
    )

    parser.add_argument(
        '--llm-model',
        type=str,
        default='gpt4',
        help='LLM model name for filename (e.g., gpt4, claude, gemini) - used with --caption-style llm'
    )

    parser.add_argument(
        '--dataset-name',
        type=str,
        required=True,
        help='Dataset name (milan, aruba, cairo, etc.) - needed for some caption styles'
    )

    parser.add_argument(
        '--sensor-details',
        type=str,
        default=None,
        help='Path to sensor details JSON file (optional)'
    )

    parser.add_argument(
        '--num-captions',
        type=int,
        default=2,
        help='Number of captions to generate per sample (for baseline style)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for captions (default: same as data-dir)'
    )

    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'test', 'both'],
        default='both',
        help='Which split to process'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Caption Generation")
    print(f"=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Caption style: {args.caption_style}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output directory: {output_dir}")
    print(f"=" * 80)

    # Create caption generator
    config = RuleBasedCaptionConfig(
        caption_style=args.caption_style,
        num_captions_per_sample=args.num_captions,
        random_seed=args.random_seed,
        dataset_name=args.dataset_name,
        sensor_details_path=args.sensor_details,
        generate_long_captions=True,
        generate_short_captions=True
    )

    # Determine style suffix for filenames
    if args.caption_style == 'baseline':
        generator = BaselineCaptionGenerator(config)
        style_suffix = 'baseline'
    elif args.caption_style == 'sourish':
        generator = SourishCaptionGenerator(config)
        style_suffix = 'sourish'
    elif args.caption_style == 'mixed':
        print("Warning: Mixed style not yet implemented, using baseline")
        generator = BaselineCaptionGenerator(config)
        style_suffix = 'mixed'
    elif args.caption_style == 'llm':
        # Create LLM config
        llm_config = LLMCaptionConfig(
            caption_style='llm',
            num_captions_per_sample=args.num_captions,
            random_seed=args.random_seed,
            dataset_name=args.dataset_name,
            llm_model=args.llm_model
        )
        generator = LLMCaptionGenerator(llm_config)
        style_suffix = f'llm_{args.llm_model}'
    else:
        print(f"Error: Unknown caption style: {args.caption_style}")
        sys.exit(1)

    # Process splits
    splits_to_process = []
    if args.split in ['train', 'both']:
        splits_to_process.append('train')
    if args.split in ['test', 'both']:
        splits_to_process.append('test')

    all_outputs = {}
    for split in splits_to_process:
        data_path = data_dir / f'{split}.json'

        if not data_path.exists():
            print(f"Warning: {split}.json not found in {data_dir}, skipping")
            continue

        # Load data
        print(f"\nLoading {split} data from {data_path}...")
        data = load_sampled_data(data_path)

        # Generate captions
        outputs = generate_captions_for_split(data, split, generator, output_dir, style_suffix)
        all_outputs[split] = outputs

    print(f"\n{'=' * 80}")
    print(f"Caption generation complete!")
    print(f"Total captions generated: {sum(len(outputs) for outputs in all_outputs.values())}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()

