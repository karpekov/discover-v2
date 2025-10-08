#!/usr/bin/env python3
"""
Data generation script for CASAS dataset processing.

This script generates processed data from CASAS datasets using the dual-encoder pipeline.
It supports pre-defined configs or custom configurations for processing datasets.

Usage:
    python src-v2/data/generate_data.py --config milan_training_20
    python src-v2/data/generate_data.py --list-configs
    python src-v2/data/generate_data.py --custom --datasets milan aruba --windows 50 100
"""

import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path (we're now in src-v2/data)
current_path = Path(__file__).parent
parent_path = current_path.parent  # This is src-v2
sys.path.insert(0, str(parent_path))

from utils.process_data_configs import load_preset, build_processing_config_from_preset
# Avoid importing heavy data pipeline modules at import time.
# We'll import them lazily inside functions that actually run processing.


# -------------------------------
# Config registry utilities
# -------------------------------
from dataclasses import dataclass


@dataclass
class Config:
    name: str
    description: str
    datasets: List[str]
    window_sizes: List[int]
    expected_processing_time_hours: float
    processing_config_template: Any
    generate_presegmented: bool
    dir_name: str


def _milan_preset_path() -> Path:
    return Path(__file__).parent.parent.parent / 'configs' / 'data_generation' / 'milan'


def list_available_configs() -> Dict[str, str]:
    """List preset names and descriptions discovered across all dataset config folders."""
    available: Dict[str, str] = {}
    dataset_folders = ['milan', 'aruba', 'cairo', 'kyoto', 'tulum']

    for dataset in dataset_folders:
        presets_dir = Path(__file__).parent.parent.parent / 'configs' / 'data_generation' / dataset
        if not presets_dir.exists():
            continue

        for json_file in sorted(presets_dir.glob('*.json')):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                name = json_file.stem
                desc = f"[{dataset}] {data.get('description', name)}"
                available[name] = desc
            except Exception:
                continue
    return available


def get_config(config_name: str) -> Config:
    """Load a JSON preset (e.g., training_80) and build a Config for the appropriate dataset."""
    # Try to find the config in different dataset folders
    dataset_folders = ['milan', 'aruba', 'cairo', 'kyoto', 'tulum']
    dataset_name = None
    preset = None

    for dataset in dataset_folders:
        try:
            preset = load_preset(dataset, config_name)
            dataset_name = dataset
            break
        except FileNotFoundError:
            continue

    if preset is None:
        raise FileNotFoundError(f"Config '{config_name}' not found in any dataset folder")

    processing_cfg = build_processing_config_from_preset(preset)
    window_sizes = preset.get('window_sizes', [20])
    description = preset.get('description', config_name)
    # Presegmented dual generation default: True unless explicitly disabled
    generate_presegmented = preset.get('generate_presegmented', True)
    # Directory name default: seq{first_window_size}
    default_dir_name = f"seq{window_sizes[0] if window_sizes else 20}"
    dir_name = preset.get('dir_name', default_dir_name)

    return Config(
        name=config_name,
        description=description,
        datasets=[dataset_name],
        window_sizes=window_sizes,
        expected_processing_time_hours=max(0.1, 0.02 * sum(window_sizes)),
        processing_config_template=processing_cfg,
        generate_presegmented=generate_presegmented,
        dir_name=dir_name,
    )

def process_single_dataset(dataset_name: str, config_template, window_sizes: List[int],
                          output_dir: Path, force_reprocess: bool = False,
                          filter_labels: List[str] = None) -> Dict[str, Any]:
    """Process a single dataset with the given configuration."""

    # Lazy imports to prevent loading metadata when just listing configs
    from data.datasets import DATASET_REGISTRY
    from data.pipeline import DualEncoderPipeline

    print(f"\nProcessing {dataset_name.upper()}")
    print(f"   Window sizes: {window_sizes}")

    start_time = time.time()

    # Check if dataset exists
    if dataset_name not in DATASET_REGISTRY._configs:
        return {'status': 'error', 'message': f'Dataset {dataset_name} not found'}

    # Create dataset-specific output directory only when exporting to default location
    will_use_custom_output = (config_template.export.output_dir != "data/processed_v2")
    if not will_use_custom_output:
        dataset_output_dir = output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        dataset_output_dir = output_dir  # placeholder when custom exporter dir is used

    # Check if already processed
    if not force_reprocess:
        check_dir = dataset_output_dir if not will_use_custom_output else Path(config_template.export.output_dir)
        existing_files = list(check_dir.glob(f"{dataset_name}_*train.json")) or list(check_dir.glob(f"{dataset_name}_w*.pkl*"))
        if len(existing_files) >= 1:
            print(f"   Already processed (found {len(existing_files)} exported files)")
            return {'status': 'skipped', 'existing_files': len(existing_files)}

    try:
        # Create configuration for this dataset (make a copy to avoid modifying template)
        import copy
        config = copy.deepcopy(config_template)
        config.dataset_name = dataset_name
        config.windowing.sizes = window_sizes

        # Apply label filtering if specified
        if filter_labels:
            config.filter_labels = filter_labels
            print(f"   Filtering by labels: {filter_labels}")

        # Initialize and run pipeline
        pipeline = DualEncoderPipeline(config)
        stats = pipeline.process()

        # Export data - use custom output dir if specified in config, otherwise use default
        if config.export.output_dir != "data/processed_v2":  # Default value
            exported_files = pipeline.export_all()  # Use config's output_dir
            # Use the same directory for summary when using custom output dir
            summary_output_dir = Path(config.export.output_dir)
        else:
            exported_files = pipeline.export_all(output_dir=str(dataset_output_dir))  # Use experiment output_dir
            summary_output_dir = dataset_output_dir

        processing_time = time.time() - start_time

        result = {
            'status': 'success',
            'dataset_name': dataset_name,
            'processing_time_minutes': round(processing_time / 60, 2),
            'exported_files': exported_files,
            'statistics': stats,
            'output_directory': str(summary_output_dir)
        }

        # Save dataset summary
        summary_file = summary_output_dir / f"{dataset_name}_summary.json"
        summary_output_dir.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"   Success ({result['processing_time_minutes']:.1f}min, {len(exported_files)} files)")
        return result

    except Exception as e:
        processing_time = time.time() - start_time
        print(f"   Error: {str(e)}")
        return {
            'status': 'error',
            'dataset_name': dataset_name,
            'error_message': str(e),
            'processing_time_minutes': round(processing_time / 60, 2)
        }


def run_dual_pipeline_generation(config, output_dir: Path, force_reprocess: bool = False) -> Dict[str, Any]:
    """Run both regular and presegmented pipelines for complete dataset generation."""

    print(f"DUAL PIPELINE DATA GENERATION: {config.name}")
    print(f"Description: {config.description}")
    print(f"Datasets: {config.datasets}")
    print(f"Window sizes: {config.window_sizes}")
    print(f"Will generate both regular and presegmented datasets")
    print(f"Output: {output_dir}")

    # Set up output directory
    data_output_dir = output_dir / f"config_{config.name}"
    data_output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    total_start_time = time.time()

    # Run regular pipeline first
    print(f"\n=== RUNNING REGULAR PIPELINE ===")
    regular_config_template = create_training_config()
    regular_config_template.export.output_dir = str(data_output_dir)

    regular_result = process_single_dataset(
        dataset_name='milan',
        config_template=regular_config_template,
        window_sizes=[20],
        output_dir=data_output_dir,
        force_reprocess=force_reprocess,
        filter_labels=None  # Dual pipeline doesn't support label filtering yet
    )
    results['milan_regular'] = regular_result

    # Run presegmented pipeline
    print(f"\n=== RUNNING PRESEGMENTED PIPELINE ===")
    presegmented_config_template = create_presegmented_config()
    presegmented_config_template.export.output_dir = str(data_output_dir)

    presegmented_result = process_single_dataset(
        dataset_name='milan',
        config_template=presegmented_config_template,
        window_sizes=[20],
        output_dir=data_output_dir,
        force_reprocess=force_reprocess,
        filter_labels=None  # Dual pipeline doesn't support label filtering yet
    )
    results['milan_presegmented'] = presegmented_result

    # Generate combined processing summary
    total_time = time.time() - total_start_time
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    failed = sum(1 for r in results.values() if r['status'] == 'error')

    processing_summary = {
        'config_name': config.name,
        'config_description': config.description,
        'started_at': datetime.now().isoformat(),
        'total_time_minutes': round(total_time / 60, 2),
        'pipelines_run': ['regular', 'presegmented'],
        'datasets_generated': [
            'milan_train.json', 'milan_test.json',
            'milan_presegmented_train.json', 'milan_presegmented_test.json'
        ],
        'results_summary': {
            'successful_pipelines': successful,
            'failed_pipelines': failed
        },
        'detailed_results': results,
        'output_directory': str(data_output_dir)
    }

    # Save processing summary
    summary_file = data_output_dir / f"dual_pipeline_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(processing_summary, f, indent=2, default=str)

    print(f"\nDUAL PIPELINE RESULTS:")
    print(f"   Successful pipelines: {successful}")
    print(f"   Failed pipelines: {failed}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Summary: {summary_file}")

    # List generated datasets
    print(f"\nGENERATED DATASETS:")
    json_files = list(data_output_dir.glob("*.json"))
    for json_file in sorted(json_files):
        if json_file.name.endswith(('_train.json', '_test.json')):
            print(f"   ✅ {json_file.name}")

    return processing_summary


def run_data_generation(config_name: str, output_dir: Path,
                        force_reprocess: bool = False, filter_labels: List[str] = None) -> Dict[str, Any]:
    """Run data generation with a pre-defined configuration."""

    # Get configuration
    config = get_config(config_name)

    # Handle dual pipeline configuration specially
    if config_name == 'milan_dual_pipeline':
        return run_dual_pipeline_generation(config, output_dir, force_reprocess)

    print(f"DATA GENERATION CONFIG: {config.name}")
    print(f"Description: {config.description}")
    print(f"Datasets: {config.datasets}")
    print(f"Window sizes: {config.window_sizes}")
    print(f"Estimated time: {config.expected_processing_time_hours:.1f} hours")
    print(f"Output: {output_dir}")

    # Check if using custom output directory
    config_template = config.processing_config_template

    # If user provided filter_labels (debugging), always use their output_dir
    if filter_labels:
        data_output_dir = output_dir
        dataset_output_dir = data_output_dir
        print(f"   Debug mode: Using custom output directory: {data_output_dir}")
    elif config_template.export.output_dir != "data/processed_v2":
        # Substitute config name in output directory template
        try:
            actual_output_dir = config_template.export.output_dir.format(
                config_name=config_name,
                dir_name=config.dir_name
            )
            data_output_dir = Path(actual_output_dir)
        except (KeyError, ValueError):
            # No template substitution needed
            data_output_dir = Path(config_template.export.output_dir)
        dataset_output_dir = data_output_dir  # Same directory for dataset processing
    else:
        # Use default structure
        data_output_dir = output_dir / f"config_{config_name}"
        dataset_output_dir = data_output_dir

    data_output_dir.mkdir(parents=True, exist_ok=True)

    # Process all datasets
    results = {}
    total_start_time = time.time()

    for dataset_name in config.datasets:
        # Create a copy of the config template with the updated output directory
        import copy
        dataset_config_template = copy.deepcopy(config_template)
        if filter_labels or config_template.export.output_dir != "data/processed_v2":
            dataset_config_template.export.output_dir = str(data_output_dir)

        result = process_single_dataset(
            dataset_name=dataset_name,
            config_template=dataset_config_template,
            window_sizes=config.window_sizes,
            output_dir=dataset_output_dir,
            force_reprocess=force_reprocess,
            filter_labels=filter_labels
        )
        results[dataset_name] = result

        # Brief pause between datasets
        if result['status'] == 'success':
            time.sleep(1)

        # Optionally run presegmented generation per dataset
        if getattr(config, 'generate_presegmented', True):
            preseg_template = copy.deepcopy(dataset_config_template)
            # Lazy import for enum
            from data.data_config import WindowStrategy  # type: ignore
            preseg_template.windowing.strategy = WindowStrategy.PRESEGMENTED
            preseg_result = process_single_dataset(
                dataset_name=dataset_name,
                config_template=preseg_template,
                window_sizes=config.window_sizes,
                output_dir=dataset_output_dir,
                force_reprocess=force_reprocess,
                filter_labels=filter_labels
            )
            results[f"{dataset_name}_presegmented"] = preseg_result

    # Generate processing summary
    total_time = time.time() - total_start_time
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    skipped = sum(1 for r in results.values() if r['status'] == 'skipped')
    failed = sum(1 for r in results.values() if r['status'] == 'error')

    processing_summary = {
        'config_name': config_name,
        'config_description': config.description,
        'started_at': datetime.now().isoformat(),
        'total_time_minutes': round(total_time / 60, 2),
        'datasets_processed': config.datasets,
        'window_sizes': config.window_sizes,
        'results_summary': {
            'successful': successful,
            'skipped': skipped,
            'failed': failed
        },
        'detailed_results': results,
        'output_directory': str(data_output_dir)
    }

    # Save processing summary
    summary_file = data_output_dir / f"processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(processing_summary, f, indent=2, default=str)

    print(f"\nDATA GENERATION RESULTS:")
    print(f"   Successful: {successful}")
    print(f"   Skipped: {skipped}")
    print(f"   Failed: {failed}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Summary: {summary_file}")

    return processing_summary


def main():
    """Main function with command line interface."""

    parser = argparse.ArgumentParser(
        description="Generate processed data from CASAS datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run pre-defined configuration
    python src-v2/data/generate_data.py --config recipe_r2_full

    # Quick validation test
    python src-v2/data/generate_data.py --config quick_validation

    # List available configurations
    python src-v2/data/generate_data.py --list-configs

    # Custom configuration
    python src-v2/data/generate_data.py --custom --datasets milan aruba --windows 50 100

    # Filter by specific labels for debugging
    python src-v2/data/generate_data.py --config quick_validation --filter-labels Morning_Meds Cook
        """
    )

    parser.add_argument('--config', '-c', type=str,
                       help='Name of pre-defined configuration to run')

    parser.add_argument('--list-configs', '-l', action='store_true',
                       help='List all available configurations')

    parser.add_argument('--custom', action='store_true',
                       help='Run custom configuration (use with --datasets and --windows)')

    parser.add_argument('--datasets', nargs='+',
                       choices=['milan', 'aruba', 'cairo', 'kyoto', 'tulum'],
                       help='Datasets to process (for custom configurations)')

    parser.add_argument('--windows', nargs='+', type=int,
                       help='Window sizes to use (for custom configurations)')

    parser.add_argument('--output-dir', '-o', type=str,
                       default='data/processed_data',
                       help='Output directory for processed data')

    parser.add_argument('--force', '-f', action='store_true',
                       help='Force reprocessing even if files exist')

    parser.add_argument('--filter-labels', nargs='+', type=str,
                       help='Filter data to only include specific first_activity labels (for debugging)')

    args = parser.parse_args()

    # List configurations
    if args.list_configs:
        print("Available Data Generation Configurations:")
        print("=" * 50)
        for name, description in list_available_configs().items():
            try:
                cfg = get_config(name)
                print(f"\n{name}")
                print(f"   {description}")
                print(f"   Datasets: {cfg.datasets}")
                print(f"   Windows: {cfg.window_sizes}")
                print(f"   Est. time: {cfg.expected_processing_time_hours:.1f}h")
            except Exception as e:
                print(f"\n{name}")
                print(f"   {description}")
                print(f"   (error loading config: {e})")
        return

    # Validate arguments
    if not args.config and not args.custom:
        parser.error("Must specify either --config or --custom")

    if args.custom and (not args.datasets or not args.windows):
        parser.error("Custom configurations require --datasets and --windows")

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.config:
            # Run pre-defined configuration
            result = run_data_generation(
                config_name=args.config,
                output_dir=output_dir,
                force_reprocess=args.force,
                filter_labels=args.filter_labels
            )
        else:
            # Run custom configuration
            print(f"CUSTOM DATA GENERATION")
            print(f"Datasets: {args.datasets}")
            print(f"Window sizes: {args.windows}")

            # Create custom configuration
            custom_config = create_custom_experiment(
                name="custom",
                description="User-defined custom configuration",
                datasets=args.datasets,
                window_sizes=args.windows,
                config_type='multi_window'
            )

            # Process datasets
            results = {}
            for dataset_name in args.datasets:
                result = process_single_dataset(
                    dataset_name=dataset_name,
                    config_template=custom_config.processing_config_template,
                    window_sizes=args.windows,
                    output_dir=output_dir,
                    force_reprocess=args.force,
                    filter_labels=args.filter_labels
                )
                results[dataset_name] = result

        print(f"\nData generation completed!")
        print(f"Results available in: {output_dir}")

    except KeyboardInterrupt:
        print(f"\nData generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nData generation failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
