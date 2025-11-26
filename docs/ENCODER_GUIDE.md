# Sensor Encoder Guide

This guide explains the new modular encoder framework implemented in Step 2 of the pipeline restructuring.

## Overview

The encoder module (`src/encoders/`) provides a flexible, extensible framework for encoding sensor sequences into embeddings. All encoders follow a common interface defined in `BaseEncoder`, making it easy to swap architectures and experiment with different approaches.

## Key Features

### 1. Variable-Length Sequence Support
- **Padding handling**: Padding tokens are properly masked and ignored in:
  - Attention mechanisms (set to -inf before softmax)
  - Pooling operations (only valid tokens contribute to mean)
- **Flexible lengths**: Can handle sequences from 1 to `max_seq_len` events

### 2. Configurable Metadata
You can control which metadata features are used:
- **Categorical features**: sensor, state, room_id, etc.
- **Spatial features**: x,y coordinates via Fourier features
- **Temporal features**: time deltas (log-bucketed)
- **Future**: time-of-day (cyclical encoding)

### 3. CLIP Alignment Support
- All encoders provide `forward_clip()` for projected embeddings
- Projection head maps to configurable dimension (default: 512)
- L2-normalized outputs for contrastive learning

### 4. MLM Support
- `get_sequence_features()` returns per-token hidden states
- Used for masked language modeling (MLM) training
- Padding is automatically excluded from loss computation

## Architecture

```
src/encoders/
├── __init__.py              # Module exports
├── base.py                  # BaseEncoder abstract class
├── config.py                # Configuration dataclasses
└── sensor/
    ├── sequence/            # Raw sequence encoders
    │   ├── __init__.py
    │   └── transformer.py   # Transformer encoder (implemented)
    └── image/               # Image-based encoders (placeholder)
        └── __init__.py
```

## Encoder Configurations

### Available Presets

1. **Tiny** (`transformer_tiny.yaml`)
   - d_model: 256, layers: 4, heads: 4
   - Fast training, good for debugging
   - ~2M parameters

2. **Small** (`transformer_small.yaml`)
   - d_model: 512, layers: 6, heads: 8
   - Balanced for medium experiments
   - ~10M parameters

3. **Base** (`transformer_base.yaml`)
   - d_model: 768, layers: 6, heads: 8
   - Default configuration
   - ~25M parameters

4. **Minimal** (`transformer_minimal.yaml`)
   - Only sensor + state (no coords/time)
   - For ablation studies

### Configuration Options

```yaml
# Architecture
encoder_type: transformer
d_model: 768
n_layers: 6
n_heads: 8
d_ff: 3072
max_seq_len: 512
dropout: 0.1

# CLIP projection
projection_dim: 512

# Positional encoding
use_alibi: true          # ALiBi attention biases (recommended)
use_learned_pe: false    # Learned PE (if not using ALiBi)

# Metadata features
metadata:
  categorical_fields:
    - sensor
    - state
    - room_id
  use_coordinates: true    # Enable spatial features
  use_time_deltas: true    # Enable temporal features
  use_time_of_day: false   # Future: cyclical time

  # Feature normalization
  coord_norm_x_max: 10.0
  coord_norm_y_max: 10.0
  time_delta_max_seconds: 3600.0
  time_delta_bins: 100

# Pooling
pooling: cls_mean           # 'cls', 'mean', 'cls_mean'
pooling_cls_weight: 0.5     # Weight for CLS in cls_mean

# Vocabularies (set at runtime)
vocab_sizes: {}
```

## Usage Examples

### Basic Usage

```python
import torch
from src.encoders import TransformerEncoderConfig, MetadataConfig
from src.encoders.sensor.sequence import TransformerSensorEncoder

# Create config
config = TransformerEncoderConfig.base()
config.vocab_sizes = {
    'sensor': 50,
    'state': 10,
    'room_id': 20
}

# Create encoder
encoder = TransformerSensorEncoder(config)

# Prepare input data
batch_size = 32
seq_len = 50
input_data = {
    'categorical_features': {
        'sensor': torch.randint(0, 50, (batch_size, seq_len)),
        'state': torch.randint(0, 10, (batch_size, seq_len)),
        'room_id': torch.randint(0, 20, (batch_size, seq_len)),
    },
    'coordinates': torch.randn(batch_size, seq_len, 2),
    'time_deltas': torch.rand(batch_size, seq_len) * 100,
}

# Attention mask (True = valid token, False = padding)
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
attention_mask[:, 40:] = False  # Last 10 tokens are padding

# Forward pass
output = encoder(input_data, attention_mask=attention_mask)

print(output.embeddings.shape)           # [32, 768] - pooled
print(output.sequence_features.shape)    # [32, 50, 768] - per-token
print(output.projected_embeddings)       # None (use forward_clip for this)
```

### CLIP Alignment

```python
# Get projected embeddings for CLIP training
clip_embeddings = encoder.forward_clip(input_data, attention_mask=attention_mask)
print(clip_embeddings.shape)  # [32, 512] - projected and normalized
```

### MLM Training

```python
# Get sequence features for MLM
hidden_states = encoder.get_sequence_features(input_data, attention_mask=attention_mask)
print(hidden_states.shape)  # [32, 50, 768]

# Apply MLM head to predict masked tokens
# (Padding positions will be automatically excluded from loss)
mlm_head = MLMHeads(config.vocab_sizes, config.d_model)
predictions = mlm_head(hidden_states)
```

### Configuring Metadata

```python
# Minimal encoder (no spatial/temporal features)
config = TransformerEncoderConfig.tiny()
config.metadata.use_coordinates = False
config.metadata.use_time_deltas = False
config.metadata.categorical_fields = ['sensor', 'state']

encoder = TransformerSensorEncoder(config)

# Input only needs categorical features
input_data = {
    'categorical_features': {
        'sensor': torch.randint(0, 50, (batch_size, seq_len)),
        'state': torch.randint(0, 10, (batch_size, seq_len)),
    }
}
# coordinates and time_deltas are optional now
```

### Flexible Categorical Fields

You can train models with **any subset of categorical fields**. The encoder dynamically creates embeddings only for the fields specified in `categorical_fields`.

**Example: Training without sensor tokens**

```yaml
# config.yaml
encoder:
  metadata:
    categorical_fields:
      - state        # Event type (ON/OFF/OPEN/CLOSE)
      - room_id      # Room location
      # sensor field removed - no sensor tokens!
    use_coordinates: false
    use_time_deltas: false
    use_time_of_day: true
```

This allows you to:
- **Ablation studies**: Test which features are most important
- **Privacy**: Train without sensor IDs
- **Generalization**: Learn patterns based on event types and locations only
- **Domain adaptation**: Transfer models across environments with different sensors

**Important Notes:**
1. The vocabulary must match the categorical fields (generate vocab with only desired fields)
2. If `sensor` is removed from `categorical_fields`, sensor UNK filtering is automatically skipped
3. The encoder will only create embeddings for fields present in `categorical_fields`

**Example: Generating vocab for subset of fields**

```python
from src.encoders.data_utils import build_vocab_from_data

# Only build vocab for state and room_id (skip sensor)
vocab = build_vocab_from_data(data, categorical_fields=['state', 'room_id'])
```

### Loading from YAML Config

```python
import yaml
from dataclasses import asdict

# Load config
with open('configs/encoders/transformer_base.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

# Create config object
metadata_config = MetadataConfig(**config_dict['metadata'])
config = TransformerEncoderConfig(
    **{k: v for k, v in config_dict.items() if k != 'metadata'},
    metadata=metadata_config
)

# Set vocab sizes from data
config.vocab_sizes = {'sensor': 50, 'state': 10, 'room_id': 20}

encoder = TransformerSensorEncoder(config)
```

## Key Implementation Details

### Padding Handling

1. **Attention Masking**
   - Padding positions are masked with `-inf` before softmax
   - Ensures padding tokens don't influence attention weights
   - See `ALiBiAttention.forward()`

2. **Pooling**
   - Mean pooling only averages over valid tokens
   - Division by actual sequence length (excluding padding)
   - See `TransformerSensorEncoder._pool_embeddings()`

3. **MLM Loss**
   - Downstream MLM loss only computes on valid positions
   - No need to manually handle padding in loss computation

### Extensibility

The architecture is designed to be easily extended:

1. **New Sequence Encoders**
   - Inherit from `SequenceEncoder`
   - Implement required methods
   - Add to `src/encoders/sensor/sequence/`

2. **Image-Based Encoders**
   - Inherit from `ImageSequenceEncoder`
   - Process floor plan visualizations
   - Add to `src/encoders/sensor/image/`

3. **Shared Components**
   - Both encoder types can share the final transformer
   - Image encoders: vision model → per-image embeddings → transformer → pooled
   - Raw encoders: categorical/continuous → embeddings → transformer → pooled

### Relationship to Original SensorEncoder

The new `TransformerSensorEncoder` is a drop-in replacement for the original `SensorEncoder` with:
- Same architectural components (ALiBi, Fourier features, etc.)
- Improved padding handling
- Configurable metadata
- Cleaner interface
- Better separation of concerns

Original code remains in `src/models/sensor_encoder.py` for backward compatibility.

## Integration with Pipeline

### Step 1: Sampling → Step 2: Encoding

```python
# Load sampled data from Step 1
import json
with open('data/processed/casas/milan/fixed_duration_60s/train.json') as f:
    sampled_data = json.load(f)

# Extract features for encoding
samples = sampled_data['samples']
# ... prepare batch from samples ...

# Encode
output = encoder(input_data, attention_mask=attention_mask)
```

### Step 2: Encoding → Step 5: Alignment

```python
# In training loop (see Step 5: Alignment)
from src.losses.clip import CLIPLoss

clip_loss_fn = CLIPLoss()

# Get sensor embeddings
sensor_embeddings = encoder.forward_clip(sensor_data, attention_mask)

# Get text embeddings (from Step 4)
text_embeddings = text_encoder.encode(captions)

# Compute CLIP loss
loss = clip_loss_fn(sensor_embeddings, text_embeddings)
```

### Combined MLM + CLIP Training

```python
# Get sequence features for MLM
sequence_features = encoder.get_sequence_features(sensor_data, attention_mask)
mlm_predictions = mlm_head(sequence_features)
mlm_loss = mlm_head.compute_loss(mlm_predictions, targets, mask_positions)

# Get CLIP embeddings
clip_embeddings = encoder.forward_clip(sensor_data, attention_mask)
clip_loss = clip_loss_fn(clip_embeddings, text_embeddings)

# Combined loss
total_loss = 0.3 * mlm_loss + 0.7 * clip_loss
```

## Performance Considerations

1. **Model Size vs Performance**
   - Tiny: Fast training, good for debugging
   - Small: Balanced performance/speed
   - Base: Best results for most tasks
   - Large: For final models with lots of data

2. **Sequence Length**
   - ALiBi attention allows variable lengths efficiently
   - No need to pad to same length within batch
   - Longer sequences → more memory

3. **Metadata Features**
   - Coordinates: +1 FFT projection (~100K params)
   - Time deltas: +1 embedding layer (~100K params)
   - Minimal overhead, usually worth including

## Future Enhancements

1. **Image-Based Encoders**
   - Implement `src/encoders/sensor/image/clip_based.py`
   - Visualize sensor activations on floor plans
   - Use CLIP/DINO/SigLIP vision encoders

2. **Time-of-Day Encoding**
   - Cyclical hour/day-of-week embeddings
   - Capture daily/weekly activity patterns

3. **Chronos Integration**
   - Adapt existing Chronos encoder to new interface
   - Add to `src/encoders/sensor/sequence/chronos.py`

4. **Multi-Modal Fusion**
   - Combine image and sequence encoders
   - Late fusion or cross-attention

## Troubleshooting

### Issue: Padding tokens affecting results
**Solution**: Ensure `attention_mask` is properly set (True for valid, False for padding)

### Issue: OOM errors
**Solution**: Use smaller model (tiny/small) or reduce batch size

### Issue: NaN losses
**Solution**: Check that attention mask has at least 1 valid token per sample

### Issue: Want to disable certain metadata
**Solution**: Set `use_coordinates=False` or `use_time_deltas=False` in config

## Summary

The new encoder framework provides:
- ✅ Variable-length sequence support with proper padding
- ✅ Configurable metadata (spatial, temporal, categorical)
- ✅ CLIP alignment support
- ✅ MLM training support
- ✅ Multiple model sizes (tiny, small, base, large)
- ✅ Clean, extensible interface
- ✅ Ready for Step 3 (Captions) and Step 5 (Alignment)

All TODOs for Step 2 are complete! 🎉

