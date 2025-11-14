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


def generate_vocab_from_samples(data_path: str, output_path: str = None):
    """
    Generate vocabulary from sampled data JSON.
    
    Args:
        data_path: Path to train.json or test.json from sampling
        output_path: Where to save vocab.json (default: same directory)
    """
    print(f"Loading data from: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    samples = data.get('samples', data)
    
    # Collect unique values for each field
    vocab = defaultdict(set)
    
    print(f"Processing {len(samples)} samples...")
    
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
                
                # Handle different possible field names
                if key in ['sensor', 'sensor_id']:
                    vocab['sensor'].add(str(value))
                elif key in ['state', 'event_type', 'message']:
                    vocab['state'].add(str(value))
                elif key in ['room', 'room_id', 'location']:
                    vocab['room_id'].add(str(value))
                elif key in ['activity', 'first_activity', 'activity_l1']:
                    vocab['activity'].add(str(value))
                elif key in ['activity_l2', 'first_activity_l2']:
                    vocab['activity_l2'].add(str(value))
                else:
                    # Generic field
                    vocab[key].add(str(value))
    
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
    parser = argparse.ArgumentParser(description='Generate vocab.json from sampled data')
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
    
    args = parser.parse_args()
    
    generate_vocab_from_samples(args.data, args.output)


if __name__ == '__main__':
    main()

