# Chronos-2 Encoder for Smart Home Activity Recognition

## Overview

The Chronos-2 encoder provides an alternative to the standard Transformer-based sensor encoder by using Amazon's Chronos-2 time series foundation model. This approach treats sensor event sequences as multivariate time series and extracts embeddings using a frozen Chronos-2 model with a trainable MLP projection head.

## Architecture

### Components

1. **Chronos-2 Model (Frozen)**
   - Uses `amazon/chronos-t5-small` by default
   - Processes sensor sequences as time series
   - All parameters are frozen during training

2. **Time Series Conversion**
   - Converts sensor events to 5-dimensional time series:
     - Sensor ID (normalized)
     - Room ID (normalized)
     - X coordinate
     - Y coordinate
     - Time delta (log-scaled)

3. **Trainable MLP Projection Head**
   - Hidden dimension: 256 (configurable)
   - Output dimension: 512 (for CLIP alignment)
   - Dropout: 0.1 (configurable)
   - Only this head is trainable

## Training

### Training Script

```bash
python src/training/train_chronos_clip.py --config configs/training/milan/chronos_clip.json
```

### Configuration

Example configuration file (`configs/training/milan/chronos_clip.json`):

```json
{
  "chronos_model_name": "amazon/chronos-t5-small",
  "projection_hidden_dim": 256,
  "projection_dropout": 0.1,
  "output_dim": 512,
  "sequence_length": 50,
  "batch_size": 64,
  "learning_rate": 0.001,
  "betas": [0.9, 0.98],
  "weight_decay": 0.01,
  "warmup_ratio": 0.1,
  "max_epochs": 50,
  "grad_clip_norm": 1.0,
  "temperature_init": 0.02,
  "learnable_temperature": true,
  "use_hard_negatives": false,
  "log_interval": 50,
  "val_interval": 500,
  "save_interval": 2000,
  "use_amp": false,
  "num_workers": 0,
  "use_wandb": true,
  "wandb_project": "discover-v2",
  "wandb_name": "chronos_clip_milan",
  "wandb_tags": ["chronos", "clip", "milan"],
  "train_data_path": "data/processed/casas/milan/seq50/milan_train.json",
  "val_data_path": "data/processed/casas/milan/seq50/milan_test.json",
  "vocab_path": "data/processed/casas/milan/seq50/milan_vocab.json",
  "output_dir": "trained_models/milan/chronos_clip",
  "text_model_name": "thenlper/gte-base",
  "max_captions": 4,
  "caption_types": "long"
}
```

### Key Differences from Standard Training

1. **No MLM**: Only CLIP-style contrastive learning is used
2. **Frozen Backbone**: Chronos-2 model is completely frozen
3. **Small Trainable Component**: Only the MLP projection head is trained
4. **Simpler Training Loop**: No MLM masking or span masker needed

## Installation

To use the full Chronos-2 model (optional, falls back to statistical features if not available):

```bash
# Install Chronos-2 (recommended)
pip install "chronos-forecasting>=2.0"

# Or install from GitHub (for Chronos-1 compatibility)
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

**Note**: Chronos-2 is primarily designed for time series forecasting, not embedding extraction. The current implementation uses statistical time series features as a fallback, which works well for activity recognition. If you want to use the actual Chronos-2 encoder for embeddings, you'll need to access the underlying T5 encoder directly (see [Chronos-2 documentation](https://huggingface.co/amazon/chronos-2)).

If Chronos is not installed, the encoder will automatically fall back to statistical time series features (mean, std, max, min + downsampled sequence).

## Usage in Code

```python
from models.chronos_encoder import ChronosEncoder

# Initialize encoder
encoder = ChronosEncoder(
    vocab_sizes=vocab_sizes,
    chronos_model_name="amazon/chronos-t5-small",
    projection_hidden_dim=256,
    output_dim=512,
    sequence_length=50
)

# Forward pass (returns 512-dim CLIP embeddings)
embeddings = encoder.forward_clip(
    categorical_features=batch['categorical_features'],
    coordinates=batch['coordinates'],
    time_deltas=batch['time_deltas'],
    mask=batch['mask']
)
```

## Integration with Existing Pipeline

The Chronos encoder is fully compatible with the existing CLIP training pipeline:

- Uses the same data format as `SensorEncoder`
- Outputs 512-dim embeddings compatible with CLIP alignment
- Can be used with existing evaluation scripts
- Works with the same text encoder and loss functions

## Advantages

1. **Pre-trained Time Series Knowledge**: Leverages Chronos-2's pre-trained understanding of temporal patterns
2. **Efficient Training**: Only a small projection head needs to be trained
3. **Fast Inference**: Frozen backbone means faster forward passes
4. **Domain Transfer**: Chronos-2's general time series knowledge may transfer well to sensor data

## Limitations

1. **Model Availability**: Requires Chronos package installation for full functionality
2. **Sequence Length**: Works best with fixed-length sequences (default: 50)
3. **No MLM**: Cannot leverage masked language modeling for additional supervision

## Comparison with Standard Encoder

| Feature | Standard Encoder | Chronos Encoder |
|---------|------------------|-----------------|
| Architecture | Custom Transformer | Pre-trained Chronos-2 |
| Trainable Params | Full model | Projection head only |
| Training Objective | CLIP + MLM | CLIP only |
| Time Series Focus | Event-based | Temporal patterns |
| Pre-training | None | Chronos-2 foundation model |

## Output

Training outputs are saved to the specified `output_dir`:

- `best_model.pt` - Best model checkpoint (based on validation loss)
- `final_model.pt` - Final model checkpoint
- `checkpoint_step_*.pt` - Intermediate checkpoints
- `hyperparameters.json` - Training configuration
- `run_summary.txt` - Training summary

## Evaluation

The trained Chronos encoder can be evaluated using the same evaluation scripts as the standard encoder:

```bash
python src/evals/evaluate_embeddings.py \
    --checkpoint trained_models/milan/chronos_clip/best_model.pt \
    --train_data data/processed/casas/milan/seq50/milan_train.json \
    --test_data data/processed/casas/milan/seq50/milan_test.json \
    --vocab data/processed/casas/milan/seq50/milan_vocab.json \
    --output_dir results/evals/milan/chronos_clip
```

Note: You may need to modify the evaluation script to load `ChronosEncoder` instead of `SensorEncoder` if it hardcodes the model type.

