"""Example usage of the caption generation framework.

This script demonstrates how to use the caption generators programmatically.

Run from project root:
    conda activate discover-v2-env
    python src/captions/example_usage.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from captions import (
    BaselineCaptionGenerator,
    SourishCaptionGenerator,
    LLMCaptionGenerator,
    RuleBasedCaptionConfig,
    LLMCaptionConfig
)


def example_1_baseline_caption():
    """Example 1: Generate baseline captions."""
    print("\n" + "="*80)
    print("Example 1: Baseline Caption Generation")
    print("="*80)

    # Sample sensor sequence (from a typical Milan window)
    sensor_sequence = [
        {'sensor': 'M015', 'datetime': '2009-02-12 08:30:15', 'state': 'ON', 'room_id': 'kitchen'},
        {'sensor': 'M015', 'datetime': '2009-02-12 08:30:18', 'state': 'OFF', 'room_id': 'kitchen'},
        {'sensor': 'M022', 'datetime': '2009-02-12 08:30:25', 'state': 'ON', 'room_id': 'kitchen'},
        {'sensor': 'M022', 'datetime': '2009-02-12 08:30:28', 'state': 'OFF', 'room_id': 'kitchen'},
        {'sensor': 'M027', 'datetime': '2009-02-12 08:31:10', 'state': 'ON', 'room_id': 'living_room'},
        {'sensor': 'M027', 'datetime': '2009-02-12 08:31:15', 'state': 'OFF', 'room_id': 'living_room'},
    ]

    metadata = {
        'start_time': '2009-02-12 08:30:15',
        'end_time': '2009-02-12 08:31:15',
        'duration_seconds': 60.0,
        'num_events': 6,
        'rooms_visited': ['kitchen', 'living_room'],
        'tod_bucket': 'morning'
    }

    # Create generator
    config = RuleBasedCaptionConfig(
        caption_style='baseline',
        num_captions_per_sample=2,
        dataset_name='milan',
        random_seed=42
    )
    generator = BaselineCaptionGenerator(config)

    # Generate captions
    output = generator.generate(
        sensor_sequence=sensor_sequence,
        metadata=metadata,
        sample_id='example_001'
    )

    print(f"\nSample ID: {output.sample_id}")
    print(f"Number of captions: {len(output.captions)}")
    print(f"Caption type: {output.metadata.get('caption_type')}")
    print("\nGenerated captions:")
    for i, caption in enumerate(output.captions, 1):
        print(f"  {i}. {caption}")

    if 'layer_b' in output.metadata:
        print(f"\nLayer B (structured): {output.metadata['layer_b']}")

    print("\n✓ Example 1 completed successfully!")


def example_2_sourish_caption():
    """Example 2: Generate Sourish-style captions."""
    print("\n" + "="*80)
    print("Example 2: Sourish-Style Caption Generation")
    print("="*80)

    # Same sensor sequence as Example 1
    sensor_sequence = [
        {'sensor': 'M015', 'datetime': '2009-02-12 08:30:15', 'state': 'ON', 'room_id': 'kitchen'},
        {'sensor': 'M015', 'datetime': '2009-02-12 08:30:18', 'state': 'OFF', 'room_id': 'kitchen'},
        {'sensor': 'M022', 'datetime': '2009-02-12 08:30:25', 'state': 'ON', 'room_id': 'kitchen'},
        {'sensor': 'M022', 'datetime': '2009-02-12 08:30:28', 'state': 'OFF', 'room_id': 'kitchen'},
        {'sensor': 'M027', 'datetime': '2009-02-12 08:31:10', 'state': 'ON', 'room_id': 'living_room'},
        {'sensor': 'M027', 'datetime': '2009-02-12 08:31:15', 'state': 'OFF', 'room_id': 'living_room'},
    ]

    metadata = {
        'start_time': '2009-02-12 08:30:15',
        'end_time': '2009-02-12 08:31:15',
        'duration_seconds': 60.0,
        'num_events': 6,
        'rooms_visited': ['kitchen', 'living_room']
    }

    # Create generator (requires dataset name for sensor mappings)
    config = RuleBasedCaptionConfig(
        caption_style='sourish',
        num_captions_per_sample=1,
        dataset_name='milan',  # Required!
        random_seed=42
    )
    generator = SourishCaptionGenerator(config)

    # Generate captions
    output = generator.generate(
        sensor_sequence=sensor_sequence,
        metadata=metadata,
        sample_id='example_002'
    )

    print(f"\nSample ID: {output.sample_id}")
    print(f"Number of captions: {len(output.captions)}")
    print(f"Caption format: {output.metadata.get('format')}")
    print("\nGenerated caption:")
    print(f"  {output.captions[0]}")

    print("\n✓ Example 2 completed successfully!")


def example_3_batch_generation():
    """Example 3: Generate captions for multiple samples."""
    print("\n" + "="*80)
    print("Example 3: Batch Caption Generation")
    print("="*80)

    # Multiple samples
    samples = [
        {
            'sample_id': 'batch_001',
            'sensor_sequence': [
                {'sensor': 'M015', 'datetime': '2009-02-12 08:30:15', 'state': 'ON', 'room_id': 'kitchen'},
                {'sensor': 'M015', 'datetime': '2009-02-12 08:30:18', 'state': 'OFF', 'room_id': 'kitchen'},
            ],
            'metadata': {
                'duration_seconds': 3.0,
                'num_events': 2,
                'rooms_visited': ['kitchen'],
                'tod_bucket': 'morning'
            }
        },
        {
            'sample_id': 'batch_002',
            'sensor_sequence': [
                {'sensor': 'M021', 'datetime': '2009-02-12 23:15:00', 'state': 'ON', 'room_id': 'bedroom'},
                {'sensor': 'M021', 'datetime': '2009-02-12 23:15:30', 'state': 'OFF', 'room_id': 'bedroom'},
            ],
            'metadata': {
                'duration_seconds': 30.0,
                'num_events': 2,
                'rooms_visited': ['bedroom'],
                'tod_bucket': 'night'
            }
        }
    ]

    # Create generator
    config = RuleBasedCaptionConfig(
        caption_style='baseline',
        num_captions_per_sample=1,
        dataset_name='milan',
        random_seed=42,
        generate_short_captions=False  # Only long captions
    )
    generator = BaselineCaptionGenerator(config)

    # Generate captions for all samples
    outputs = generator.generate_batch(samples)

    print(f"\nGenerated captions for {len(outputs)} samples:")
    for output in outputs:
        print(f"\n  Sample: {output.sample_id}")
        print(f"    Caption: {output.captions[0]}")

    # Get statistics
    stats = generator.get_statistics(outputs)
    print(f"\nStatistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Total captions: {stats['total_captions']}")
    print(f"  Avg captions/sample: {stats['avg_captions_per_sample']:.1f}")
    print(f"  Avg caption length: {stats['caption_length_stats']['mean_tokens']:.1f} tokens")

    print("\n✓ Example 3 completed successfully!")


def example_4_llm_placeholder():
    """Example 4: LLM-based caption generation (placeholder)."""
    print("\n" + "="*80)
    print("Example 4: LLM-Based Caption Generation (Placeholder)")
    print("="*80)

    sensor_sequence = [
        {'sensor': 'M015', 'datetime': '2009-02-12 08:30:15', 'state': 'ON', 'room_id': 'kitchen'},
        {'sensor': 'M015', 'datetime': '2009-02-12 08:30:18', 'state': 'OFF', 'room_id': 'kitchen'},
    ]

    metadata = {
        'duration_seconds': 3.0,
        'num_events': 2,
        'rooms_visited': ['kitchen']
    }

    # Create LLM generator (placeholder)
    config = LLMCaptionConfig(
        llm_provider='openai',
        llm_model='gpt-4',
        temperature=0.7
    )
    generator = LLMCaptionGenerator(config)

    # Generate caption (will return placeholder)
    output = generator.generate(
        sensor_sequence=sensor_sequence,
        metadata=metadata,
        sample_id='example_llm_001'
    )

    print(f"\nSample ID: {output.sample_id}")
    print(f"LLM Provider: {output.metadata.get('llm_provider')}")
    print(f"LLM Model: {output.metadata.get('llm_model')}")
    print("\nGenerated caption (placeholder):")
    print(f"  {output.captions[0]}")

    print("\n✓ Example 4 completed (placeholder implementation)")


def example_5_comparison():
    """Example 5: Compare different caption styles."""
    print("\n" + "="*80)
    print("Example 5: Comparing Caption Styles")
    print("="*80)

    # Same sensor sequence for comparison
    sensor_sequence = [
        {'sensor': 'M015', 'datetime': '2009-02-12 08:30:15', 'state': 'ON', 'room_id': 'kitchen'},
        {'sensor': 'M015', 'datetime': '2009-02-12 08:30:18', 'state': 'OFF', 'room_id': 'kitchen'},
        {'sensor': 'M022', 'datetime': '2009-02-12 08:30:25', 'state': 'ON', 'room_id': 'kitchen'},
        {'sensor': 'M027', 'datetime': '2009-02-12 08:31:10', 'state': 'ON', 'room_id': 'living_room'},
    ]

    metadata = {
        'start_time': '2009-02-12 08:30:15',
        'end_time': '2009-02-12 08:31:15',
        'duration_seconds': 60.0,
        'num_events': 4,
        'rooms_visited': ['kitchen', 'living_room'],
        'tod_bucket': 'morning'
    }

    sample_id = 'comparison_001'

    # Baseline
    config_baseline = RuleBasedCaptionConfig(
        caption_style='baseline',
        num_captions_per_sample=1,
        dataset_name='milan',
        random_seed=42,
        generate_short_captions=False
    )
    generator_baseline = BaselineCaptionGenerator(config_baseline)
    output_baseline = generator_baseline.generate(sensor_sequence, metadata, sample_id)

    # Sourish
    config_sourish = RuleBasedCaptionConfig(
        caption_style='sourish',
        dataset_name='milan'
    )
    generator_sourish = SourishCaptionGenerator(config_sourish)
    output_sourish = generator_sourish.generate(sensor_sequence, metadata, sample_id)

    # Compare
    print(f"\nSample: {sample_id}")
    print(f"Events: {len(sensor_sequence)}")
    print(f"Duration: {metadata['duration_seconds']}s")
    print(f"Rooms: {metadata['rooms_visited']}")

    print(f"\n1. Baseline Caption (natural language):")
    print(f"   {output_baseline.captions[0]}")
    print(f"   Length: {len(output_baseline.captions[0].split())} tokens")

    print(f"\n2. Sourish Caption (structured template):")
    print(f"   {output_sourish.captions[0]}")
    print(f"   Length: {len(output_sourish.captions[0].split())} tokens")

    print("\n✓ Example 5 completed successfully!")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("Caption Generation Framework - Example Usage")
    print("="*80)

    try:
        example_1_baseline_caption()
        example_2_sourish_caption()
        example_3_batch_generation()
        example_4_llm_placeholder()
        example_5_comparison()

        print("\n" + "="*80)
        print("All examples completed successfully! ✓")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

