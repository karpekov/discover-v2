# Data Loading Guide - Step 1 → Step 2 Integration

This guide explains how to load sampled data from Step 1 and use it with Step 2 encoders.

## Overview

The new `data_utils.py` module provides utilities to bridge Step 1 (sampling) and Step 2 (encoding):
- Load sampled JSON data from Step 1
- Build vocabularies automatically
- Prepare batches for encoders
- Handle variable-length sequences

## Quick Start

```python
from src.encoders.data_utils import load_and_prepare_milan_data, prepare_batch_for_encoder
from src.encoders.config import TransformerEncoderConfig
from src.encoders.sensor.sequence import TransformerSensorEncoder

# 1. Load data and build vocab
data, vocab, vocab_sizes = load_and_prepare_milan_data(
    data_dir='data/processed/casas/milan/fixed_duration_60sec_presegmented',
    split='train',
    max_samples=100,
    categorical_fields=['sensor_id', 'event_type', 'room', 'sensor_type'],
    use_coordinates=False,
    use_time_deltas=True,
)

# 2. Create encoder with actual vocab sizes
config = TransformerEncoderConfig.base()
config.vocab_sizes = vocab_sizes
config.metadata.use_coordinates = False
config.metadata.use_time_deltas = True
config.metadata.categorical_fields = list(vocab.keys())

encoder = TransformerSensorEncoder(config)

# 3. Prepare batch
input_data = prepare_batch_for_encoder(
    samples=data['samples'][:8],
    vocab=vocab,
    categorical_fields=list(vocab.keys()),
    use_coordinates=False,
    use_time_deltas=True,
)

attention_mask = input_data.pop('attention_mask')

# 4. Encode
output = encoder(input_data, attention_mask=attention_mask)
embeddings = output.embeddings  # [batch_size, 768]
```

## Key Features

### 1. Automatic Vocabulary Building

The data loader automatically builds vocabularies from the sampled data:

```python
vocab = build_vocab_from_data(data, categorical_fields)
# Returns: {'sensor_id': {'<PAD>': 0, '<UNK>': 1, 'M021': 2, ...}, ...}
```

Special tokens are automatically added:
- `<PAD>` (index 0): Padding token
- `<UNK>` (index 1): Unknown token

### 2. Variable-Length Sequence Support

The data loader handles variable-length sequences from fixed-duration sampling:

```python
# Real Milan data example:
# Sequence lengths: [14, 3, 3, 19, 9, 2, 6, 2]
# Average length: 7.2 events per 60-second window
```

Attention masks properly mark valid vs padding positions:
- `True` = valid event
- `False` = padding

### 3. Time Delta Computation

Time deltas are automatically computed from timestamps:

```python
deltas = compute_time_deltas(events)
# Returns: [0.0, 5.2, 10.3, ...]  # Seconds between events
```

### 4. Flexible Field Selection

Choose which categorical fields to use:

```python
# Full metadata
categorical_fields = ['sensor_id', 'event_type', 'room', 'sensor_type']

# Minimal (for ablation)
categorical_fields = ['sensor_id', 'event_type']
```

## Data Format

### Step 1 Output (Sampled Data)

```json
{
  "dataset": "milan",
  "sampling_strategy": "fixed_duration",
  "samples": [
    {
      "sample_id": "milan_train_000001",
      "sensor_sequence": [
        {
          "sensor_id": "M021",
          "event_type": "ON",
          "timestamp": "2009-10-16 03:55:53.000080",
          "room": "master_bedroom",
          "sensor_type": "motion",
          "activity_l1": "Bed_to_Toilet",
          ...
        },
        // ... more events
      ],
      "metadata": {
        "duration_seconds": 60.0,
        "num_events": 14,
        ...
      }
    }
  ]
}
```

### Step 2 Input (Encoder Format)

```python
{
    'categorical_features': {
        'sensor_id': tensor([[2, 3, 4, ...]]),  # [batch_size, seq_len]
        'event_type': tensor([[1, 1, 0, ...]]),
        'room': tensor([[2, 2, 1, ...]]),
        'sensor_type': tensor([[0, 0, 1, ...]]),
    },
    'time_deltas': tensor([[0.0, 5.2, 10.3, ...]]),  # [batch_size, seq_len]
    'attention_mask': tensor([[True, True, False, ...]]),  # [batch_size, seq_len]
}
```

## Available Functions

### `load_sampled_data()`

Load Step 1 sampled data from JSON file.

```python
data = load_sampled_data(
    data_path='data/processed/casas/milan/fixed_duration_60sec',
    split='train',  # or 'test'
    max_samples=None  # None = all samples
)
```

### `build_vocab_from_data()`

Build vocabulary mappings from data.

```python
vocab = build_vocab_from_data(
    data=data,
    categorical_fields=['sensor_id', 'event_type', 'room', 'sensor_type']
)
```

### `prepare_batch_for_encoder()`

Convert samples to encoder input format.

```python
batch_data = prepare_batch_for_encoder(
    samples=data['samples'][:8],
    vocab=vocab,
    categorical_fields=list(vocab.keys()),
    max_seq_len=512,
    use_coordinates=False,
    use_time_deltas=True,
    device='cpu'
)
```

### `load_and_prepare_milan_data()`

Convenience function that does everything in one call.

```python
data, vocab, vocab_sizes = load_and_prepare_milan_data(
    data_dir='data/processed/casas/milan/fixed_duration_60sec_presegmented',
    split='train',
    batch_size=8,
    max_samples=100,
    categorical_fields=None,  # None = auto-detect
    use_coordinates=False,
    use_time_deltas=True,
    device='cpu'
)
```

## Example: Real Milan Data

The `example_usage.py` script now loads real Milan data by default:

```bash
cd /Users/alexkarpekov/code/har/discover-v2
python src/encoders/example_usage.py
```

Output:
```
Loading train data from data/processed/casas/milan/fixed_duration_60sec_presegmented...
Building vocabulary for fields: ['sensor_id', 'event_type', 'room', 'sensor_type']
Vocabulary sizes: {'sensor_id': 7, 'event_type': 4, 'room': 4, 'sensor_type': 3}
Loaded 8 samples

Input shapes (Real Milan Data):
  - sensor_id: torch.Size([8, 19])
  - event_type: torch.Size([8, 19])
  - room: torch.Size([8, 19])
  - sensor_type: torch.Size([8, 19])
  - time_deltas: torch.Size([8, 19])
  - attention_mask: torch.Size([8, 19])
  - Sequence lengths: [14, 3, 3, 19, 9, 2, 6, 2]
  - Avg length: 7.2
  - Variable length data from fixed-duration sampling!

Output embeddings shape: torch.Size([8, 768])
Embeddings are L2-normalized: True
```

## Integration with Existing Code

### With SmartHomeDataset

The existing `SmartHomeDataset` class expects a different format. To use it:

1. **Option A**: Adapt Step 1 output to SmartHomeDataset format
2. **Option B**: Use the new `data_utils.py` directly (recommended)

The new approach is simpler and designed specifically for the Step 1 → Step 2 flow.

### With DataLoader

For training, wrap the data preparation in a custom Dataset:

```python
from torch.utils.data import Dataset, DataLoader

class Step1Dataset(Dataset):
    def __init__(self, data, vocab, categorical_fields):
        self.data = data
        self.vocab = vocab
        self.categorical_fields = categorical_fields

    def __len__(self):
        return len(self.data['samples'])

    def __getitem__(self, idx):
        sample = self.data['samples'][idx]
        # Prepare single sample
        batch = prepare_batch_for_encoder(
            samples=[sample],
            vocab=self.vocab,
            categorical_fields=self.categorical_fields,
            use_coordinates=False,
            use_time_deltas=True,
        )
        # Squeeze batch dimension
        for key in batch['categorical_features']:
            batch['categorical_features'][key] = batch['categorical_features'][key].squeeze(0)
        batch['time_deltas'] = batch['time_deltas'].squeeze(0)
        batch['attention_mask'] = batch['attention_mask'].squeeze(0)
        return batch

# Create DataLoader
dataset = Step1Dataset(data, vocab, categorical_fields)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Coordinate Handling

Currently, the Step 1 sampled data doesn't include x,y coordinates. To add coordinates:

1. Load sensor metadata with coordinates
2. Join coordinates to events based on sensor_id
3. Update Step 1 samplers to include coordinates in output

Example:
```python
# Load sensor coordinates (from metadata/sensor_coordinates/milan.txt)
sensor_coords = load_sensor_coordinates('milan')

# In sampler, add coordinates to events
for event in sample['sensor_sequence']:
    sensor_id = event['sensor_id']
    event['x'], event['y'] = sensor_coords.get(sensor_id, (0.0, 0.0))
```

Then set `use_coordinates=True` in data loading.

## Troubleshooting

### Issue: Vocabulary size mismatch
**Solution**: Ensure vocab_sizes includes special tokens:
```python
vocab_sizes = get_vocab_sizes(vocab)  # Includes <PAD>, <UNK>
```

### Issue: Tensor size mismatch
**Solution**: Check that categorical_fields match between vocab building and batch preparation

### Issue: Empty sequences
**Solution**: Filter out samples with zero events before batching

### Issue: Out of memory
**Solution**: Reduce batch size or max_samples parameter

## Summary

The `data_utils.py` module provides a complete bridge between Step 1 and Step 2:
- ✅ Loads sampled JSON data
- ✅ Builds vocabularies automatically
- ✅ Handles variable-length sequences
- ✅ Computes time deltas
- ✅ Creates proper attention masks
- ✅ Tested with real Milan data

**Status**: Production-ready and integrated into `example_usage.py` ✅

## See Also

- `docs/ENCODER_GUIDE.md` - Encoder usage documentation
- `src/encoders/example_usage.py` - Working examples with real data
- `src/encoders/data_utils.py` - Data loading implementation

