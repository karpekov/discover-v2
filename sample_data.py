#!/usr/bin/env python3
"""
Main script for running data sampling strategies.

Usage:
    python sample_data.py --config configs/sampling/milan_fixed_length_50.yaml
    python sample_data.py --config configs/sampling/milan_fixed_duration_60.yaml
    python sample_data.py --list-configs
"""

import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from sampling import (
    FixedLengthSampler,
    FixedDurationSampler,
    FixedLengthConfig,
    FixedDurationConfig,
    SamplingStrategy,
    load_sampling_config_from_dict
)


def load_yaml_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return config_dict


def create_sampler(config_dict: dict):
    """Create appropriate sampler based on config."""
    strategy_str = config_dict.get('strategy', 'fixed_length')

    # Convert paths to Path objects
    if 'raw_data_path' in config_dict:
        config_dict['raw_data_path'] = Path(config_dict['raw_data_path'])
    if 'output_dir' in config_dict:
        config_dict['output_dir'] = Path(config_dict['output_dir'])

    # Auto-load metadata if not provided
    if 'metadata_path' not in config_dict or config_dict['metadata_path'] is None:
        # Get the project root
        project_root = Path(__file__).parent

        # Determine metadata file based on dataset name
        dataset_name = config_dict.get('dataset_name', '')
        dataset_lower = dataset_name.lower()

        if dataset_lower in ['milan', 'aruba', 'cairo', 'kyoto']:
            metadata_path = project_root / 'metadata' / 'casas_metadata.json'
        elif dataset_lower == 'marble':
            metadata_path = project_root / 'metadata' / 'marble_metadata.json'
        else:
            metadata_path = None

        # Use the metadata file if it exists
        if metadata_path and metadata_path.exists():
            config_dict['metadata_path'] = metadata_path
            print(f"Auto-loaded sensor metadata from: {metadata_path}")
    elif 'metadata_path' in config_dict and config_dict['metadata_path'] is not None:
        # Convert to Path if it's a string
        if isinstance(config_dict['metadata_path'], str):
            config_dict['metadata_path'] = Path(config_dict['metadata_path'])

    # Convert strategy string to enum
    if isinstance(strategy_str, str):
        config_dict['strategy'] = SamplingStrategy(strategy_str)

    # Convert split_strategy string to enum if present
    if 'split_strategy' in config_dict and isinstance(config_dict['split_strategy'], str):
        from sampling.config import SplitStrategy
        config_dict['split_strategy'] = SplitStrategy(config_dict['split_strategy'])

    # Create config object
    strategy = config_dict['strategy']
    if strategy == SamplingStrategy.FIXED_LENGTH:
        config = FixedLengthConfig(**config_dict)
        sampler = FixedLengthSampler(config)
    elif strategy == SamplingStrategy.FIXED_DURATION:
        config = FixedDurationConfig(**config_dict)
        sampler = FixedDurationSampler(config)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    return sampler


def list_available_configs(config_dir: Path = None):
    """List available sampling configurations."""
    if config_dir is None:
        config_dir = Path(__file__).parent / 'configs' / 'sampling'

    if not config_dir.exists():
        print(f"No config directory found: {config_dir}")
        return

    print("\nAvailable sampling configurations:")
    print("=" * 70)

    yaml_files = sorted(config_dir.glob('*.yaml'))
    if not yaml_files:
        print("No configuration files found.")
        return

    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)

            dataset = config.get('dataset_name', 'unknown')
            strategy = config.get('strategy', 'unknown')

            # Get strategy-specific info
            if strategy == 'fixed_length':
                windows = config.get('window_sizes', [])
                detail = f"window sizes: {windows}"
            elif strategy == 'fixed_duration':
                durations = config.get('duration_seconds', [])
                detail = f"durations: {durations}s"
            else:
                detail = ""

            print(f"\n{yaml_file.stem}")
            print(f"  Dataset: {dataset}")
            print(f"  Strategy: {strategy}")
            print(f"  {detail}")
            print(f"  Path: {yaml_file}")

        except Exception as e:
            print(f"\n{yaml_file.stem} (error reading: {e})")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Sample sensor data using various strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run fixed-length sampling with 50-event windows
  python sample_data.py --config configs/sampling/milan_fixed_length_50.yaml

  # Run fixed-duration sampling with 60-second windows
  python sample_data.py --config configs/sampling/milan_fixed_duration_60.yaml

  # List all available configs
  python sample_data.py --list-configs
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--list-configs', '-l',
        action='store_true',
        help='List all available configuration files'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Override output directory from config'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (limit data processing)'
    )

    args = parser.parse_args()

    # List configs if requested
    if args.list_configs:
        list_available_configs()
        return

    # Config file required otherwise
    if not args.config:
        parser.error("--config is required (or use --list-configs)")

    # Load config
    config_path = Path(args.config)
    print(f"\nLoading configuration from: {config_path}")
    config_dict = load_yaml_config(config_path)

    # Override output dir if specified
    if args.output_dir:
        config_dict['output_dir'] = args.output_dir
        print(f"Overriding output directory: {args.output_dir}")

    # Enable debug mode if requested
    if args.debug:
        config_dict['max_lines'] = 10000
        config_dict['max_windows'] = 500
        print("Debug mode enabled: limiting data processing")

    # Print config summary
    print("\nConfiguration Summary:")
    print("=" * 70)
    print(f"Dataset: {config_dict.get('dataset_name')}")
    print(f"Strategy: {config_dict.get('strategy')}")
    print(f"Output: {config_dict.get('output_dir')}")
    print(f"Train ratio: {config_dict.get('train_ratio', 0.8)}")
    print(f"Split strategy: {config_dict.get('split_strategy', 'random')}")

    if config_dict.get('strategy') == 'fixed_length':
        print(f"Window sizes: {config_dict.get('window_sizes')}")
    elif config_dict.get('strategy') == 'fixed_duration':
        print(f"Durations: {config_dict.get('duration_seconds')}s")

    print("=" * 70)

    # Create sampler
    print("\nCreating sampler...")
    sampler = create_sampler(config_dict)

    # Run sampling
    start_time = datetime.now()
    print(f"\nStarting sampling at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        train_result, val_result, test_result = sampler.sample_dataset()

        # Save results
        output_dir = Path(config_dict['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / 'train.json'
        val_path = output_dir / 'val.json'
        test_path = output_dir / 'test.json'

        print("\nSaving results...")
        train_result.save_json(train_path)
        val_result.save_json(val_path)
        test_result.save_json(test_path)

        # Generate and save vocabulary from training data
        print("\nGenerating vocabulary from training data...")
        vocab = sampler._generate_vocabulary(train_result.samples)
        vocab_path = output_dir / 'vocab.json'

        import json
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=2)

        print(f"✅ Saved vocabulary to: {vocab_path}")
        print(f"   Vocabulary statistics:")
        for field, mapping in vocab.items():
            print(f"     - {field}: {len(mapping)} unique values")

        # Save config for reference
        config_save_path = output_dir / 'sampling_config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f"Saved config to {config_save_path}")

        # Save statistics to separate file
        stats_path = output_dir / 'statistics.json'

        # Convert strategy to string if it's an enum
        strategy_value = config_dict.get('strategy')
        if hasattr(strategy_value, 'value'):
            strategy_value = strategy_value.value

        split_strategy_value = config_dict.get('split_strategy')
        if hasattr(split_strategy_value, 'value'):
            split_strategy_value = split_strategy_value.value

        statistics_data = {
            'generated_at': datetime.now().isoformat(),
            'dataset_name': config_dict.get('dataset_name'),
            'sampling_strategy': strategy_value,
            'sampling_params': train_result.sampling_params,
            'config_file': str(config_path),
            'output_directory': str(output_dir),
            'output_files': {
                'train': str(train_path.name),
                'val': str(val_path.name),
                'test': str(test_path.name),
                'vocab': str(vocab_path.name),
                'config': str(config_save_path.name),
                'statistics': str(stats_path.name)
            },
            'split_strategy': split_strategy_value,
            'split_ratio': '70/10/20',  # train/val/test
            'random_seed': config_dict.get('random_seed'),
            'processing_duration_seconds': None,  # Will be updated below
            'train_statistics': train_result.statistics,
            'val_statistics': val_result.statistics,
            'test_statistics': test_result.statistics,
            'total_samples': len(train_result.samples) + len(val_result.samples) + len(test_result.samples),
            'train_samples_count': len(train_result.samples),
            'val_samples_count': len(val_result.samples),
            'test_samples_count': len(test_result.samples),
        }

        with open(stats_path, 'w') as f:
            json.dump(statistics_data, f, indent=2)
        print(f"Saved statistics to {stats_path}")

        # Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Update statistics file with processing duration
        statistics_data['processing_duration_seconds'] = duration
        statistics_data['completed_at'] = end_time.isoformat()
        with open(stats_path, 'w') as f:
            json.dump(statistics_data, f, indent=2)

        print("\n" + "=" * 70)
        print("SAMPLING COMPLETE")
        print("=" * 70)
        print(f"Duration: {duration:.1f} seconds")
        print(f"\nTrain samples: {len(train_result.samples)}")
        print(f"  Avg sequence length: {train_result.statistics.get('avg_sequence_length', 0):.1f}")
        print(f"  Avg duration: {train_result.statistics.get('avg_duration_seconds', 0):.1f}s")
        print(f"\nVal samples: {len(val_result.samples)}")
        print(f"  Avg sequence length: {val_result.statistics.get('avg_sequence_length', 0):.1f}")
        print(f"  Avg duration: {val_result.statistics.get('avg_duration_seconds', 0):.1f}s")
        print(f"\nTest samples: {len(test_result.samples)}")
        print(f"  Avg sequence length: {test_result.statistics.get('avg_sequence_length', 0):.1f}")
        print(f"  Avg duration: {test_result.statistics.get('avg_duration_seconds', 0):.1f}s")
        print(f"\nOutput directory: {output_dir}")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\nError during sampling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

