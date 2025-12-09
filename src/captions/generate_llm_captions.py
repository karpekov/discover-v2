#!/usr/bin/env python3
"""Generate LLM-based captions for sampled sensor data.

This script provides an end-to-end pipeline for LLM caption generation:
1. Load raw samples (train.json, val.json, test.json)
2. Convert to compact JSON representation
3. Generate captions using LLM backend
4. Save outputs with llm_captions field

Usage examples:

# DEBUG MODE - Generate captions for just 3 samples and inspect output
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend openai \
    --model gpt-4o-mini \
    --num-captions 4 \
    --debug

# Or specify custom limit
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend openai \
    --model gpt-4o-mini \
    --limit 5 \
    --verbose \
    --show-prompts

# OpenAI GPT-4 (production)
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend openai \
    --model gpt-4o-mini \
    --num-captions 4 \
    --api-key $OPENAI_API_KEY

# Google Gemini
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend gemini \
    --model gemini-1.5-flash \
    --num-captions 4 \
    --api-key $GOOGLE_API_KEY

# Local Gemma
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend gemma \
    --model google/gemma-7b \
    --num-captions 4 \
    --device cuda

# Local Llama
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend llama \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --num-captions 4 \
    --device cuda

# Use config file
python src/captions/generate_llm_captions.py \
    --config configs/captions/llm_openai.yaml \
    --data-dir data/processed/casas/milan/FD_60
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import yaml

# Load environment variables from .env file (for API keys)
try:
    from dotenv import load_dotenv
    load_dotenv()  # Automatically loads from .env in project root
except ImportError:
    # python-dotenv not installed, will use system environment variables
    pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from captions.llm_based.compact_json import to_compact_caption_json
from captions.llm_based.backends import create_backend
from captions.llm_based.prompts import build_user_prompt, build_multi_sample_prompt


def load_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_samples(data_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data.get('samples', [])


def save_compact_jsonl(compact_samples: List[Dict[str, Any]], output_path: Path):
    """Save compact JSON samples to JSONL file."""
    with open(output_path, 'w') as f:
        for sample in compact_samples:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved {len(compact_samples)} compact samples to {output_path}")


def save_captioned_json(samples: List[Dict[str, Any]], output_path: Path):
    """Save samples with LLM captions to JSON file in compact baseline format."""
    # Convert to baseline format: only sample_id, captions, minimal metadata
    compact_entries = []
    for sample in samples:
        entry = {
            'sample_id': sample['sample_id'],
            'captions': sample['llm_captions'],
            'metadata': {
                'caption_type': 'llm',
                'num_captions': len(sample['llm_captions']),
                # Optionally include compact metadata if available
                'duration_seconds': sample.get('duration_seconds'),
                'primary_room': sample.get('primary_room'),
                'num_events': sample.get('num_events')
            }
        }
        compact_entries.append(entry)

    output_data = {'captions': compact_entries}
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved {len(compact_entries)} captioned samples to {output_path}")


def generate_llm_captions(
    samples: List[Dict[str, Any]],
    backend,
    num_captions: int,
    batch_size: int = 10,
    verbose: bool = False,
    show_prompts: bool = False,
    use_multi_sample: bool = True,
    multi_sample_size: int = 5,
    checkpoint_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Generate LLM captions for samples with checkpoint recovery.

    Args:
        samples: List of raw sample dictionaries
        backend: LLM backend instance
        num_captions: Number of captions per sample
        batch_size: Number of samples to process at once
        verbose: Print generated captions
        show_prompts: Print prompts sent to LLM
        use_multi_sample: Pack multiple samples in one API call to save tokens
        multi_sample_size: Number of samples per multi-sample prompt
        checkpoint_path: Optional path to checkpoint file for crash recovery

    Returns:
        List of samples with llm_captions field added
    """
    mode_str = f"multi-sample (packing {multi_sample_size} samples/call)" if use_multi_sample else "individual"

    # Load existing checkpoint if available
    existing_captions = {}
    if checkpoint_path and checkpoint_path.exists():
        print(f"\n[Checkpoint] Found existing checkpoint: {checkpoint_path}")
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            existing_captions = {item['sample_id']: item for item in checkpoint_data.get('captions', [])}
            print(f"[Checkpoint] Loaded {len(existing_captions)} existing captions")
        except Exception as e:
            print(f"[Checkpoint] Warning: Could not load checkpoint: {e}")
            existing_captions = {}

    # Filter out samples that already have captions
    samples_to_process = []
    captioned_samples = []

    for sample in samples:
        sample_id = sample.get('sample_id', 'unknown')
        if sample_id in existing_captions:
            # Use existing caption
            captioned_samples.append(existing_captions[sample_id])
        else:
            # Need to generate
            samples_to_process.append(sample)

    if samples_to_process:
        print(f"\nGenerating LLM captions for {len(samples_to_process)} samples ({mode_str})...")
        print(f"  ({len(existing_captions)} samples already completed)")
    else:
        print(f"\n[Checkpoint] All {len(samples)} samples already have captions!")
        return captioned_samples

    for i in tqdm(range(0, len(samples_to_process), batch_size), desc="Processing batches"):
        batch = samples_to_process[i:i + batch_size]

        # Convert to compact JSON
        compact_jsons = []
        for sample in batch:
            try:
                compact_json = to_compact_caption_json(sample)
                compact_jsons.append(compact_json)
            except Exception as e:
                print(f"Warning: Failed to convert sample {sample.get('sample_id', 'unknown')}: {e}")
                compact_jsons.append({'sample_id': sample.get('sample_id', 'unknown')})

        # Note: Old multi-sample path removed in favor of structured output approach below
        if use_multi_sample and len(batch) > 1:
            # Use structured output multi-sample mode (Gemini backend)
            # Process in mini-batches to avoid token limits
            batch_captions = []

            for j in range(0, len(compact_jsons), multi_sample_size):
                mini_batch = compact_jsons[j:j + multi_sample_size]

                # Build individual prompts for each sample
                prompts = [build_user_prompt(cj, num_captions) for cj in mini_batch]
                sample_ids = [cj.get('sample_id', f'sample_{i+j+k}') for k, cj in enumerate(mini_batch)]

                # Show prompt if requested
                if show_prompts and i == 0 and j == 0:
                    print("\n" + "=" * 80)
                    print(f"MULTI-SAMPLE MODE ({len(mini_batch)} samples):")
                    print("=" * 80)
                    print(f"First prompt:\n{prompts[0]}")
                    print("=" * 80 + "\n")

                # Generate using structured output (backend will batch them)
                try:
                    import inspect
                    sig = inspect.signature(backend.generate)
                    if 'sample_ids' in sig.parameters:
                        captions_list = backend.generate(prompts=prompts, sample_ids=sample_ids)
                    else:
                        # Legacy backend without sample_ids
                        captions_list = backend.generate(prompts)
                    batch_captions.extend(captions_list)
                except Exception as e:
                    error_str = str(e).lower()
                    # Re-raise fatal errors
                    if any(x in error_str for x in ['auth', 'quota', 'api_key', 'permission', 'not found']):
                        print(f"FATAL ERROR: {e}")
                        raise

                    print(f"Warning: Multi-sample generation failed: {e}, using fallback")
                    batch_captions.extend([["Activity detected."] * num_captions for _ in mini_batch])

            caption_lists = batch_captions
        else:
            # Original: one sample per prompt (or batch multiple samples with structured output)
            prompts = [build_user_prompt(cj, num_captions) for cj in compact_jsons]
            sample_ids = [cj.get('sample_id', f'sample_{i+j}') for j, cj in enumerate(compact_jsons)]

            # Show prompts if requested
            if show_prompts and i == 0:
                print("\n" + "=" * 80)
                print("SAMPLE PROMPT (first sample):")
                print("=" * 80)
                print(prompts[0])
                print("=" * 80 + "\n")

            # Generate captions with sample_ids for structured output
            try:
                # Check if backend supports sample_ids parameter (Gemini structured output)
                import inspect
                sig = inspect.signature(backend.generate)
                if 'sample_ids' in sig.parameters:
                    caption_lists = backend.generate(prompts=prompts, sample_ids=sample_ids)
                else:
                    # Legacy backend without sample_ids
                    caption_lists = backend.generate(prompts)
            except Exception as e:
                error_str = str(e).lower()
                # Re-raise fatal errors
                if any(x in error_str for x in ['auth', 'quota', 'api_key', 'permission', 'not found']):
                    print(f"FATAL ERROR: {e}")
                    raise

                print(f"Warning: Backend generation failed for batch: {e}")
                caption_lists = [["Activity detected."] * num_captions for _ in batch]

        # Add captions to samples (keep only essential fields for output)
        for sample, captions, compact_json in zip(batch, caption_lists, compact_jsons):
            # Create compact output with only sample_id, captions, and minimal metadata
            sample_with_captions = {
                'sample_id': sample.get('sample_id', 'unknown'),
                'llm_captions': captions,
                'duration_seconds': compact_json.get('duration_seconds'),
                'primary_room': compact_json.get('primary_room'),
                'num_events': compact_json.get('num_events')
            }
            captioned_samples.append(sample_with_captions)

            # Print captions if verbose
            if verbose:
                print(f"\n{'─' * 80}")
                print(f"Sample ID: {sample.get('sample_id', 'unknown')}")
                print(f"Duration: {compact_json.get('duration_seconds', 0):.1f}s | "
                      f"Events: {compact_json.get('num_events', 0)} | "
                      f"Room: {compact_json.get('primary_room', 'unknown')}")
                if 'time_context' in compact_json:
                    tc = compact_json['time_context']
                    print(f"Time: {tc.get('day_of_week', '?')} {tc.get('period_of_day', '?')}")
                print(f"\nGenerated Captions ({len(captions)}):")
                for j, caption in enumerate(captions, 1):
                    print(f"  {j}. {caption}")

        # Save checkpoint after each batch
        if checkpoint_path:
            try:
                checkpoint_data = {'captions': captioned_samples}
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
            except Exception as e:
                print(f"[Checkpoint] Warning: Failed to save checkpoint: {e}")

    return captioned_samples


def _parse_multi_sample_response(response_text: str, num_captions: int) -> Dict[str, List[str]]:
    """Parse multi-sample response into dictionary of captions.

    Args:
        response_text: LLM response text (should be JSON object)
        num_captions: Expected number of captions per sample

    Returns:
        Dictionary mapping sample_id to list of captions
    """
    # Remove markdown code blocks
    text = response_text.strip()
    if text.startswith('```'):
        lines = text.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines)

    # Try to parse as JSON
    try:
        captions_dict = json.loads(text)
        if isinstance(captions_dict, dict):
            # Ensure all values are lists of strings
            result = {}
            for sid, caps in captions_dict.items():
                if isinstance(caps, list):
                    result[sid] = [str(c).strip() for c in caps if c]
                else:
                    result[sid] = [str(caps)]

                # Ensure correct number
                while len(result[sid]) < num_captions:
                    result[sid].append(result[sid][0] if result[sid] else "Activity detected.")
                result[sid] = result[sid][:num_captions]

            return result
    except json.JSONDecodeError:
        pass

    return {}


def main():
    parser = argparse.ArgumentParser(
        description='Generate LLM-based captions for sensor data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML config file (optional)'
    )

    # Data paths
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing sampled data (train.json, val.json, test.json)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as data-dir)'
    )

    # LLM backend settings
    parser.add_argument(
        '--backend',
        type=str,
        choices=['gemma', 'llama', 'openai', 'gemini'],
        default=None,
        help='LLM backend type'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name/identifier'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key for remote backends (OpenAI, Gemini). If not provided, reads from environment.'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device for local models (cuda, cpu, mps, or None for auto)'
    )

    # Generation settings
    parser.add_argument(
        '--num-captions',
        type=int,
        default=None,
        help='Number of captions to generate per sample (default: 4)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Sampling temperature (default: 0.9)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of samples to process in each batch (default: 10)'
    )

    parser.add_argument(
        '--multi-sample',
        action='store_true',
        help='Pack multiple samples in one API call to save tokens (experimental)'
    )

    parser.add_argument(
        '--multi-sample-size',
        type=int,
        default=5,
        help='Number of samples to pack per API call when using multi-sample (default: 5)'
    )

    # Split selection
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test', 'all'],
        default='all',
        help='Which split to process (default: all)'
    )

    # Output options
    parser.add_argument(
        '--save-compact',
        action='store_true',
        help='Save intermediate compact JSONL files'
    )

    # Debug options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode: process only first few samples and print detailed output'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of samples to process per split (useful for testing)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print generated captions to console'
    )

    parser.add_argument(
        '--show-prompts',
        action='store_true',
        help='Print the prompts sent to the LLM (debug mode)'
    )

    args = parser.parse_args()

    # Load config if provided
    config_dict = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        print(f"Loading config from: {config_path}")
        config_dict = load_config_from_yaml(config_path)

    # Merge config with args (args take precedence)
    backend = args.backend or config_dict.get('backend_type', config_dict.get('backend', 'openai'))
    model = args.model or config_dict.get('model_name', config_dict.get('model'))
    num_captions = args.num_captions if args.num_captions is not None else config_dict.get('num_captions_per_sample', 4)
    temperature = args.temperature if args.temperature is not None else config_dict.get('temperature', 0.6)
    api_key = args.api_key or config_dict.get('api_key')
    device = args.device or config_dict.get('device')

    # Debug mode: auto-set limit and verbose
    if args.debug:
        if args.limit is None:
            args.limit = 3
        args.verbose = True
        args.show_prompts = True
        print("\n🐛 DEBUG MODE ENABLED")
        print(f"  - Limiting to {args.limit} samples per split")
        print(f"  - Verbose output enabled")
        print(f"  - Showing sample prompts\n")

    # Set default model if not provided
    if model is None:
        default_models = {
            'openai': 'gpt-4o-mini',
            'gemini': 'gemini-2.5-flash',
            'gemma': 'google/gemma-7b',
            'llama': 'meta-llama/Meta-Llama-3-8B-Instruct'
        }
        model = default_models.get(backend, 'gpt-4o-mini')

    # Setup paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"LLM Caption Generation")
    print(f"=" * 80)
    print(f"Backend: {backend}")
    print(f"Model: {model}")
    print(f"Num captions: {num_captions}")
    print(f"Temperature: {temperature}")
    print(f"Batch size: {args.batch_size}")
    if args.limit:
        print(f"Sample limit: {args.limit} per split")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"=" * 80)

    # Create backend
    print(f"\nInitializing {backend} backend...")
    try:
        backend_instance = create_backend(
            backend_type=backend,
            model_name=model,
            num_captions=num_captions,
            temperature=temperature,
            api_key=api_key,
            device=device
        )
    except Exception as e:
        print(f"Error: Failed to initialize backend: {e}")
        sys.exit(1)

    # Determine splits to process
    splits_to_process = []
    if args.split == 'all':
        splits_to_process = ['train', 'val', 'test']
    else:
        splits_to_process = [args.split]

    # Process each split
    for split in splits_to_process:
        data_path = data_dir / f'{split}.json'

        if not data_path.exists():
            print(f"\nWarning: {split}.json not found, skipping")
            continue

        print(f"\n{'=' * 80}")
        print(f"Processing {split.upper()} split")
        print(f"{'=' * 80}")

        # Load samples
        print(f"Loading samples from {data_path}...")
        samples = load_samples(data_path)

        # Apply limit if specified
        if args.limit:
            original_count = len(samples)
            samples = samples[:args.limit]
            print(f"Loaded {original_count} samples, using first {len(samples)} (--limit {args.limit})")
        else:
            print(f"Loaded {len(samples)} samples")

        if not samples:
            print(f"Warning: No samples found in {split}.json, skipping")
            continue

        # Optionally save compact JSON
        if args.save_compact:
            compact_samples = []
            for sample in samples:
                try:
                    compact = to_compact_caption_json(sample)
                    compact_samples.append(compact)
                except Exception as e:
                    print(f"Warning: Failed to convert sample: {e}")

            compact_output_path = output_dir / f'{split}_compact.jsonl'
            save_compact_jsonl(compact_samples, compact_output_path)

        # Determine if using multi-sample mode (default: disabled)
        use_multi_sample = args.multi_sample

        # Create checkpoint path for crash recovery
        model_for_filename = model if model else backend
        model_short_name = model_for_filename.split('/')[-1].replace('-', '_').replace('.', '_')
        output_filename = f'{split}_llm_{backend}_{model_short_name}.json'
        checkpoint_path = output_dir / f'{output_filename}.checkpoint.tmp'

        # Generate captions with checkpoint recovery
        captioned_samples = generate_llm_captions(
            samples=samples,
            backend=backend_instance,
            num_captions=num_captions,
            batch_size=args.batch_size,
            verbose=args.verbose,
            show_prompts=args.show_prompts,
            use_multi_sample=use_multi_sample,
            multi_sample_size=args.multi_sample_size,
            checkpoint_path=checkpoint_path
        )

        # Save final output
        output_path = output_dir / output_filename
        save_captioned_json(captioned_samples, output_path)

        # Delete checkpoint file after successful completion
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                print(f"[Checkpoint] Deleted checkpoint file: {checkpoint_path.name}")
            except Exception as e:
                print(f"[Checkpoint] Warning: Could not delete checkpoint: {e}")

        # Print statistics
        print(f"\n{split.upper()} Statistics:")
        print(f"  Total samples: {len(captioned_samples)}")
        print(f"  Total captions: {len(captioned_samples) * num_captions}")

        # Show sample captions
        if captioned_samples:
            sample = captioned_samples[0]
            print(f"\n  Sample ID: {sample.get('sample_id', 'unknown')}")
            print(f"  Sample captions:")
            for i, caption in enumerate(sample.get('llm_captions', [])[:3], 1):
                print(f"    {i}. {caption}")

    print(f"\n{'=' * 80}")
    print(f"LLM caption generation complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()

