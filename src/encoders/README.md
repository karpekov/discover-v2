# Sensor Encoders Module

Modular encoder framework for smart home sensor sequences.

## Quick Start

```python
from src.encoders.config import TransformerEncoderConfig
from src.encoders.sensor.sequence import TransformerSensorEncoder

# Create encoder
config = TransformerEncoderConfig.base()
config.vocab_sizes = {'sensor': 50, 'state': 10, 'room_id': 20}
encoder = TransformerSensorEncoder(config)

# Prepare data
input_data = {
    'categorical_features': {
        'sensor': torch.randint(0, 50, (batch_size, seq_len)),
        'state': torch.randint(0, 10, (batch_size, seq_len)),
        'room_id': torch.randint(0, 20, (batch_size, seq_len)),
    },
    'coordinates': torch.randn(batch_size, seq_len, 2),
    'time_deltas': torch.rand(batch_size, seq_len) * 100,
}
attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

# Encode
output = encoder(input_data, attention_mask=attention_mask)
embeddings = output.embeddings  # [batch_size, d_model]
```

## Features

- **Variable-length sequences**: Properly handles padding
- **Configurable metadata**: Enable/disable coordinates, time, etc.
- **CLIP alignment**: `forward_clip()` for projected embeddings
- **MLM support**: `get_sequence_features()` for per-token features
- **Multiple sizes**: tiny, small, base, large presets

## Documentation

See `docs/ENCODER_GUIDE.md` for complete documentation.

## Example Usage

Run the example script to see all features in action:

```bash
python src/encoders/example_usage.py
```

## Configuration

Pre-configured YAML files in `configs/encoders/`:
- `transformer_tiny.yaml` - Fast, small model
- `transformer_small.yaml` - Balanced model
- `transformer_base.yaml` - Default model
- `transformer_minimal.yaml` - Ablation (no spatial/temporal)

## Architecture

```
encoders/
├── base.py              # BaseEncoder abstract class
├── config.py            # Configuration dataclasses
├── sensor/
│   ├── sequence/        # Raw sequence encoders
│   │   └── transformer.py
│   └── image/           # Image-based encoders (placeholder)
└── example_usage.py     # Usage examples
```

