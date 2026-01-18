# Special Temporal Tokens Implementation

## Summary

This document describes the implementation of special temporal tokens (`[TOD]` and `[DOW]`) for the discover-v2 pipeline. This architecture improves upon the previous approach by:
- Avoiding redundant encoding of sequence-level temporal information
- Providing explicit positions for temporal context that the transformer can attend to
- Separating event-level features from sequence-level features

## Architecture

### Token Sequence Structure

```
Position:  0        1        2       3       ...    19
Token:    [TOD]   [DOW]   Event1  Event2   ...  Event18
```

Where:
- `[TOD]`: Special token carrying time-of-day information (tod_bucket embedding only)
- `[DOW]`: Special token carrying day-of-week information (dow_bucket embedding only)
- `Event1-18`: Sensor event tokens with sensor/room/state embeddings

### Feature Encoding

**Position 0 ([TOD] token):**
- `tod_bucket`: Encoded value (1-6, representing time periods)
- `sensor`, `room_id`, `state`, `sensor_type`: 0 (padding/ignored)
- Coordinates: [0.0, 0.0]
- Time delta: 0.0
- Mask: True (always valid)

**Position 1 ([DOW] token):**
- `dow_bucket`: Encoded value (1-7, representing days of week)
- `sensor`, `room_id`, `state`, `sensor_type`: 0 (padding/ignored)
- Coordinates: [0.0, 0.0]
- Time delta: 0.0
- Mask: True (always valid)

**Positions 2-19 (Event tokens):**
- `sensor`: Actual sensor ID
- `room_id`: Actual room
- `state`: ON/OFF/value
- `sensor_type`: Motion/Door/etc.
- `tod_bucket`: 0 (not used for events)
- `dow_bucket`: 0 (not used for events)
- Coordinates: Actual (x, y) sensor locations
- Time deltas: Actual time since previous event
- Mask: True for valid events, False for padding

## Implementation Details

### Dataset Changes (`src/dataio/dataset.py`)

1. **New parameter**: `use_special_tokens: bool = True`
   - When True, prepends special tokens for tod_bucket and dow_bucket
   - When False, falls back to old behavior (temporal info repeated at every position)

2. **Separated categorical fields**:
   - `event_categorical_fields`: Applied to event tokens (`sensor`, `room_id`, `state`, `sensor_type`)
   - `sequence_categorical_fields`: Become special tokens (`tod_bucket`, `dow_bucket`)

3. **Sequence length handling**:
   - `sequence_length=20` means:
     - 2 special token positions
     - 18 event token positions
   - Variable-length sequences are padded/truncated to fit 18 event slots

4. **Extraction logic**:
   - Temporal values extracted from first event in sequence
   - Assumed constant across the 60-second window (valid assumption)

### Backward Compatibility

The implementation is **fully backward compatible**:

```python
# Old behavior (temporal info at every position)
dataset = SmartHomeDataset(
    data_path='...',
    vocab_path='...',
    use_special_tokens=False
)

# New behavior (special tokens)
dataset = SmartHomeDataset(
    data_path='...',
    vocab_path='...',
    use_special_tokens=True  # Default
)
```

## Usage

### Training with Special Tokens

```python
from dataio.dataset import SmartHomeDataset

# Create dataset with special tokens
dataset = SmartHomeDataset(
    data_path='data/processed/casas/milan/FD_60/train.json',
    vocab_path='data/processed/casas/milan/FD_60/vocab.json',
    sequence_length=20,  # 2 special + 18 events
    use_special_tokens=True
)

sample = dataset[0]
# sample['categorical_features']['tod_bucket']: [20] tensor
#   Position 0: tod value, Position 1+: zeros
# sample['categorical_features']['dow_bucket']: [20] tensor
#   Position 1: dow value, Position 0,2+: zeros
```

### Encoder Processing

The transformer encoder will process this as:

```python
# Each position gets its embedding
embeddings[0] = tod_bucket_embedding(tod_value)  # [TOD] token
embeddings[1] = dow_bucket_embedding(dow_value)  # [DOW] token
embeddings[2] = sensor_emb + room_emb + state_emb + coords  # Event 1
embeddings[3] = sensor_emb + room_emb + state_emb + coords  # Event 2
...

# Transformer self-attention allows events to attend to [TOD] and [DOW] tokens
```

## Benefits

### 1. Efficiency
- **Before**: 60 redundant temporal encodings per sequence (20 events × 3 temporal fields)
- **After**: 2 dedicated temporal tokens
- **Savings**: ~97% reduction in temporal feature redundancy

### 2. Interpretability
- Clear separation between sequence context and event sequence
- Easy to inspect what temporal info the model has access to
- Special tokens can be analyzed independently

### 3. Flexibility
- Can easily add more special tokens (e.g., `[HOUSE]`, `[DURATION]`)
- Event tokens remain clean with only event-specific info
- Attention patterns become more interpretable

### 4. Performance
- Model can learn to attend to temporal tokens when needed
- Reduces noise in event token embeddings
- Better gradient flow for sequence-level vs. event-level features

## Configuration

### Enable in Training Config

```yaml
# configs/alignment/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1.yaml
encoder:
  metadata:
    categorical_fields:
    - sensor
    - state
    - room_id
    - tod_bucket  # Becomes special token position 0
    - dow_bucket  # Becomes special token position 1
    use_coordinates: true
    use_time_deltas: false
    use_time_of_day: true
```

When the dataset loader sees `use_special_tokens=True` (default), it will:
1. Extract `tod_bucket` and `dow_bucket` from the first event
2. Create special token positions 0 and 1
3. Fill remaining positions 2-19 with event tokens

## Verification

```python
import sys
sys.path.insert(0, 'src')
from dataio.dataset import SmartHomeDataset

dataset = SmartHomeDataset(
    data_path='data/processed/casas/milan/FD_60/train.json',
    vocab_path='data/processed/casas/milan/FD_60/vocab.json',
    sequence_length=20,
    use_special_tokens=True
)

sample = dataset[0]

# All tensors should be shape [20]
for field, tensor in sample['categorical_features'].items():
    print(f'{field}: {tensor.shape}')  # torch.Size([20])

# Check special token positions
print('TOD token (position 0):', sample['categorical_features']['tod_bucket'][0].item())
print('DOW token (position 1):', sample['categorical_features']['dow_bucket'][1].item())
print('First event sensor (position 2):', sample['categorical_features']['sensor'][2].item())
```

## Future Enhancements

Potential additions:
- `[HOUSE]` token for multi-dataset training
- `[DURATION]` token for window duration info
- `[RESIDENT]` token for multi-resident datasets
- Learnable positional embeddings that distinguish special tokens from events

## Related Documentation

- `docs/TIME_OF_DAY_IMPLEMENTATION.md` - Temporal token generation
- `docs/ENCODER_GUIDE.md` - Transformer encoder architecture
- `docs/DATA_LOADING_GUIDE.md` - Data pipeline overview

