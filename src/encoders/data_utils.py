"""
Data loading utilities for the new encoder framework.

Adapts Step 1 sampled data format to work with Step 2 encoders.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


def load_sampled_data(data_path: str, split: str = 'train', max_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    Load sampled data from Step 1 output.

    Args:
        data_path: Path to processed data directory (e.g., 'data/processed/casas/milan/fixed_duration_60sec')
        split: 'train' or 'test'
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        Dict containing dataset info and samples
    """
    data_file = Path(data_path) / f"{split}.json"

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    with open(data_file, 'r') as f:
        data = json.load(f)

    if max_samples is not None:
        data['samples'] = data['samples'][:max_samples]

    return data


def build_vocab_from_data(data: Dict[str, Any], categorical_fields: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Build vocabulary mappings from sampled data.

    Args:
        data: Loaded dataset from load_sampled_data()
        categorical_fields: List of fields to create vocabularies for

    Returns:
        vocab: Dict mapping field names to {value: index} dictionaries
    """
    vocab = {field: {} for field in categorical_fields}

    # Reserve indices for special tokens first
    for field in categorical_fields:
        vocab[field]['<PAD>'] = 0
        vocab[field]['<UNK>'] = 1

    # Collect unique values for each field
    for sample in data['samples']:
        for event in sample['sensor_sequence']:
            for field in categorical_fields:
                value = str(event.get(field, '<UNK>'))  # Convert to string
                if value not in vocab[field]:
                    vocab[field][value] = len(vocab[field])

    return vocab


def compute_time_deltas(events: List[Dict]) -> List[float]:
    """
    Compute time deltas between consecutive events.

    Args:
        events: List of event dictionaries with 'timestamp' field

    Returns:
        List of time deltas in seconds
    """
    if len(events) <= 1:
        return [0.0] * len(events)

    deltas = [0.0]  # First event has delta = 0

    for i in range(1, len(events)):
        try:
            # Parse timestamps
            curr_time = datetime.fromisoformat(events[i]['timestamp'])
            prev_time = datetime.fromisoformat(events[i-1]['timestamp'])
            delta = (curr_time - prev_time).total_seconds()
            deltas.append(max(0.0, delta))
        except:
            # Fallback if timestamp parsing fails
            deltas.append(1.0)

    return deltas


def prepare_batch_for_encoder(
    samples: List[Dict[str, Any]],
    vocab: Dict[str, Dict[str, int]],
    categorical_fields: List[str],
    max_seq_len: int = 512,
    use_coordinates: bool = False,
    use_time_deltas: bool = True,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Prepare a batch of samples for the encoder.

    Args:
        samples: List of samples from Step 1 data
        vocab: Vocabulary mappings
        categorical_fields: Fields to encode
        max_seq_len: Maximum sequence length (for padding)
        use_coordinates: Whether to include coordinate features
        use_time_deltas: Whether to include time delta features
        device: Device to place tensors on

    Returns:
        Dict containing encoder inputs:
            - categorical_features: Dict of [batch_size, seq_len] tensors
            - coordinates: [batch_size, seq_len, 2] (if use_coordinates)
            - time_deltas: [batch_size, seq_len] (if use_time_deltas)
            - attention_mask: [batch_size, seq_len] boolean mask
    """
    batch_size = len(samples)

    # Determine actual sequence lengths
    seq_lengths = [len(sample['sensor_sequence']) for sample in samples]
    actual_max_len = min(max(seq_lengths), max_seq_len)

    # Initialize tensors
    categorical_data = {
        field: torch.zeros((batch_size, actual_max_len), dtype=torch.long)
        for field in categorical_fields
    }

    if use_coordinates:
        coordinates = torch.zeros((batch_size, actual_max_len, 2), dtype=torch.float32)

    if use_time_deltas:
        time_deltas_tensor = torch.zeros((batch_size, actual_max_len), dtype=torch.float32)

    attention_mask = torch.zeros((batch_size, actual_max_len), dtype=torch.bool)

    # Fill tensors
    for i, sample in enumerate(samples):
        events = sample['sensor_sequence'][:actual_max_len]  # Truncate if needed
        seq_len = len(events)

        # Mark valid positions
        attention_mask[i, :seq_len] = True

        # Encode categorical features
        for field in categorical_fields:
            for j, event in enumerate(events):
                value = str(event.get(field, '<UNK>'))  # Convert to string
                idx = vocab[field].get(value, vocab[field]['<UNK>'])
                categorical_data[field][i, j] = idx

        # Coordinates (if available and requested)
        if use_coordinates:
            for j, event in enumerate(events):
                # Try to get coordinates from event
                x = event.get('x', 0.0)
                y = event.get('y', 0.0)
                coordinates[i, j, 0] = x
                coordinates[i, j, 1] = y

        # Time deltas (if requested)
        if use_time_deltas:
            deltas = compute_time_deltas(events)
            time_deltas_tensor[i, :seq_len] = torch.tensor(deltas, dtype=torch.float32)

    # Prepare output
    result = {
        'categorical_features': categorical_data,
        'attention_mask': attention_mask.to(device)
    }

    # Move tensors to device
    for field in categorical_fields:
        result['categorical_features'][field] = result['categorical_features'][field].to(device)

    if use_coordinates:
        result['coordinates'] = coordinates.to(device)

    if use_time_deltas:
        result['time_deltas'] = time_deltas_tensor.to(device)

    return result


def get_vocab_sizes(vocab: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    """
    Get vocabulary sizes from vocab dict.

    Args:
        vocab: Vocabulary mappings

    Returns:
        Dict mapping field names to vocabulary sizes
    """
    return {field: len(field_vocab) for field, field_vocab in vocab.items()}


def load_and_prepare_milan_data(
    data_dir: str = 'data/processed/casas/milan/fixed_duration_60sec_presegmented',
    split: str = 'train',
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    categorical_fields: Optional[List[str]] = None,
    use_coordinates: bool = False,
    use_time_deltas: bool = True,
    device: str = 'cpu'
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    Convenience function to load Milan data and prepare it for encoding.

    Args:
        data_dir: Path to processed data directory
        split: 'train' or 'test'
        batch_size: Number of samples per batch
        max_samples: Maximum number of samples to load
        categorical_fields: Fields to encode (None = auto-detect)
        use_coordinates: Whether to include coordinates
        use_time_deltas: Whether to include time deltas
        device: Device to place tensors on

    Returns:
        Tuple of (data, vocab, vocab_sizes)
    """
    # Load data
    print(f"Loading {split} data from {data_dir}...")
    data = load_sampled_data(data_dir, split=split, max_samples=max_samples)

    # Auto-detect categorical fields if not specified
    if categorical_fields is None:
        # Check first event to see what fields are available
        if data['samples']:
            first_event = data['samples'][0]['sensor_sequence'][0]
            categorical_fields = [
                field for field in ['sensor_id', 'event_type', 'room', 'sensor_type']
                if field in first_event
            ]

    print(f"Building vocabulary for fields: {categorical_fields}")
    vocab = build_vocab_from_data(data, categorical_fields)
    vocab_sizes = get_vocab_sizes(vocab)

    print(f"Vocabulary sizes: {vocab_sizes}")
    print(f"Loaded {len(data['samples'])} samples")

    return data, vocab, vocab_sizes


if __name__ == '__main__':
    # Example usage
    data, vocab, vocab_sizes = load_and_prepare_milan_data(
        max_samples=10,
        device='cpu'
    )

    print("\nPreparing batch...")
    batch_data = prepare_batch_for_encoder(
        samples=data['samples'][:8],
        vocab=vocab,
        categorical_fields=list(vocab.keys()),
        use_coordinates=False,
        use_time_deltas=True,
        device='cpu'
    )

    print(f"Batch prepared:")
    for key, value in batch_data.items():
        if key == 'categorical_features':
            for field, tensor in value.items():
                print(f"  {field}: {tensor.shape}")
        else:
            print(f"  {key}: {value.shape}")

