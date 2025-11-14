# Alignment Training Guide (Step 5)

This guide covers the alignment stage (Step 5) where we align sensor encoder outputs with text embeddings using contrastive learning (CLIP loss).

## Overview

The alignment module combines:
- **Sensor Encoder** (trainable): Encodes sensor sequences into embeddings
- **Sensor Projection** (trainable): Projects sensor embeddings into shared space
- **Text Projection** (optional, trainable): Projects text embeddings into shared space
- **CLIP Loss** (learnable temperature): Aligns sensor and text embeddings
- **MLM Loss** (optional): Masked language modeling on sensor sequences

## Quick Start

### 1. Train with Pre-computed Text Embeddings

```bash
# Train alignment model using existing data and embeddings
python train.py --config configs/alignment/milan_baseline.yaml
```

### 2. Train with On-the-Fly Caption Encoding

```yaml
# config.yaml
train_captions_path: data/processed/casas/milan/fixed_duration_60s/train_captions_baseline.json
text_encoder_config_path: configs/text_encoders/gte_base.yaml
```

```bash
python train.py --config config.yaml
```

### 3. Train Full Pipeline from Scratch

```bash
# Automatically runs: sampling → captions → text encoding → alignment
python train.py --config configs/alignment/milan_baseline.yaml --run-full-pipeline
```

## Directory Structure

```
src/alignment/
├── __init__.py         # Module exports
├── config.py           # Configuration classes
├── model.py            # AlignmentModel (combines all components)
├── trainer.py          # AlignmentTrainer (training loop)
└── dataset.py          # AlignmentDataset (data loading)

configs/alignment/
├── milan_baseline.yaml          # CLIP-only training
├── milan_with_mlm.yaml          # CLIP + MLM (50-50)
└── milan_mlp_projection.yaml   # MLP projections
```

## Configuration

### Basic Configuration

```yaml
# Experiment metadata
experiment_name: milan_baseline_alignment
output_dir: trained_models/milan/alignment_baseline

# Data paths
train_data_path: data/processed/casas/milan/fixed_duration_60s/train.json
val_data_path: data/processed/casas/milan/fixed_duration_60s/test.json
vocab_path: data/processed/casas/milan/fixed_duration_60s/vocab.json

# Text embeddings (pre-computed)
train_text_embeddings_path: data/processed/casas/milan/fixed_duration_60s/train_embeddings_baseline_gte_base.npz
val_text_embeddings_path: data/processed/casas/milan/fixed_duration_60s/test_embeddings_baseline_gte_base.npz

# Encoder configuration
encoder_config_path: configs/encoders/transformer_base.yaml
encoder_type: transformer

# Projection heads
sensor_projection:
  type: linear  # or 'mlp'
  dim: 512
  hidden_dim: 2048  # Only for MLP
  num_layers: 2     # Only for MLP (2 or 3)
  dropout: 0.1
  use_bn: false

text_projection: null  # null = frozen text embeddings, no projection

# Loss configuration
loss:
  clip_weight: 1.0
  temperature_init: 0.02
  learnable_temperature: true
  mlm_weight: 0.0  # Set > 0 to enable MLM
  use_hard_negatives: false

# Optimizer
optimizer:
  type: adamw
  learning_rate: 3.0e-4
  betas: [0.9, 0.98]
  weight_decay: 0.01
  warmup_ratio: 0.1
  grad_clip_norm: 1.0

# Training
training:
  batch_size: 128
  max_steps: 10000
  device: auto
  use_amp: true
  log_interval: 50
  val_interval: 500
  save_interval: 2000
  num_workers: 4

# WandB logging
use_wandb: true
wandb_project: discover-v2
wandb_name: milan_baseline_alignment
wandb_tags: [alignment, milan, baseline, clip-only]
```

## Usage Examples

### Example 1: Basic CLIP-Only Training

```bash
# Train with linear projection, CLIP loss only
python train.py --config configs/alignment/milan_baseline.yaml
```

**Config highlights:**
- Linear projection (768 → 512)
- CLIP loss only (no MLM)
- Frozen text embeddings
- Learnable temperature

### Example 2: CLIP + MLM Training

```bash
# Train with both CLIP and MLM losses
python train.py --config configs/alignment/milan_with_mlm.yaml
```

**Config highlights:**
- `mlm_weight: 1.0` (equal weight with CLIP)
- Learns from both alignment and reconstruction
- Better representation learning

### Example 3: MLP Projection (SimCLR-style)

```bash
# Train with 2-layer MLP projection
python train.py --config configs/alignment/milan_mlp_projection.yaml
```

**Config highlights:**
- `projection_type: mlp`
- 2-layer MLP: 768 → 2048 → 512
- Following SimCLR/MoCo best practices

### Example 4: Resume from Checkpoint

```bash
# Resume training from saved checkpoint
python train.py \
  --config configs/alignment/milan_baseline.yaml \
  --resume trained_models/milan/alignment_baseline/checkpoint_step_5000.pt
```

### Example 5: Override Output Directory

```bash
# Save to custom location
python train.py \
  --config configs/alignment/milan_baseline.yaml \
  --output-dir trained_models/milan/my_experiment
```

## Data Requirements

### Input Data Format

**1. Sensor Data** (from Step 1 - Data Sampling):
```json
{
  "dataset": "milan",
  "sampling_strategy": "fixed_duration",
  "samples": [
    {
      "sample_id": "milan_train_000001",
      "sensor_sequence": [
        {
          "sensor": "M001",
          "state": "ON",
          "timestamp": "2009-02-12 08:30:45",
          "room": "kitchen",
          "x": 3.5,
          "y": 2.1
        },
        // ... more events
      ],
      "metadata": { ... }
    }
  ]
}
```

**2. Text Embeddings** (from Step 4 - Text Encoding):
```python
# NPZ file with:
embeddings: np.ndarray  # [num_samples, embedding_dim]
sample_ids: np.ndarray  # [num_samples]
encoder_metadata: dict  # Encoder info
```

**3. OR Captions** (from Step 3 - Caption Generation):
```json
{
  "captions": [
    "Person moves from kitchen at 08:30, activating sensors M001, M015",
    // ... more captions
  ]
}
```

### Data Alignment

**Critical**: Sensor data and text data must be aligned by index:
- `sensor_data[i]` corresponds to `text_embeddings[i]` or `captions[i]`
- Both arrays must have the same length
- Alignment is preserved during shuffling (DataLoader shuffles indices, not data)

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AlignmentModel                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Sensor Path:                          Text Path:                │
│  ┌───────────────┐                     ┌──────────────┐         │
│  │ Sensor Data   │                     │ Text         │         │
│  │ [B, L, F]     │                     │ Embeddings   │         │
│  └───────┬───────┘                     │ [B, D_text]  │         │
│          │                              └──────┬───────┘         │
│          ▼                                     │                 │
│  ┌───────────────┐                            │                 │
│  │ Sensor        │                            │                 │
│  │ Encoder       │                            │                 │
│  │ (Trainable)   │                            │                 │
│  └───────┬───────┘                            │                 │
│          │                                     │                 │
│          ▼                                     ▼                 │
│  ┌───────────────┐                     ┌──────────────┐         │
│  │ Pooling       │                     │ Text         │         │
│  │ [B, D_model]  │                     │ Projection   │         │
│  └───────┬───────┘                     │ (Optional)   │         │
│          │                              └──────┬───────┘         │
│          ▼                                     │                 │
│  ┌───────────────┐                            │                 │
│  │ Sensor        │                            │                 │
│  │ Projection    │                            │                 │
│  │ (Trainable)   │                            │                 │
│  └───────┬───────┘                            │                 │
│          │                                     │                 │
│          ▼                                     ▼                 │
│  ┌───────────────┐                     ┌──────────────┐         │
│  │ L2 Normalize  │                     │ L2 Normalize │         │
│  │ [B, D_proj]   │                     │ [B, D_proj]  │         │
│  └───────┬───────┘                     └──────┬───────┘         │
│          │                                     │                 │
│          └──────────────┬──────────────────────┘                 │
│                         ▼                                        │
│                  ┌──────────────┐                                │
│                  │  CLIP Loss   │                                │
│                  │ (Learnable   │                                │
│                  │ Temperature) │                                │
│                  └──────────────┘                                │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Components

**1. Sensor Encoder** (Trainable)
- Transformer-based sequence encoder
- Processes variable-length sensor sequences
- Outputs per-token features + pooled embedding
- Params: ~10M-40M depending on size

**2. Sensor Projection** (Trainable)
- **Linear**: Single linear layer (768 → 512)
  - Params: ~393K
  - Fast, simple, stable
- **MLP**: 2 or 3-layer MLP (768 → 2048 → 512)
  - Params: ~2.6M (2-layer) or ~6.8M (3-layer)
  - Better representation learning
  - SimCLR/MoCo standard

**3. Text Projection** (Optional, Trainable)
- Same options as sensor projection
- Usually set to `null` (frozen text embeddings)
- Enable if text embeddings need adaptation

**4. CLIP Loss**
- Bidirectional InfoNCE loss
- Learnable temperature (default: init=0.02)
- Optional hard negative sampling
- In-batch negatives by default

## Training Strategies

### Strategy 1: CLIP-Only (Recommended for Start)

```yaml
loss:
  clip_weight: 1.0
  mlm_weight: 0.0
```

**Advantages:**
- Simpler, faster training
- Direct alignment objective
- Good for downstream retrieval

**When to use:**
- Initial experiments
- Retrieval-focused applications
- Limited compute

### Strategy 2: CLIP + MLM (Better Representations)

```yaml
loss:
  clip_weight: 1.0
  mlm_weight: 1.0  # Equal weighting
```

**Advantages:**
- Better sensor representations
- Learns reconstruction + alignment
- More robust features

**When to use:**
- Transfer learning needed
- Classification downstream tasks
- Plenty of compute

### Strategy 3: Hard Negative Mining

```yaml
loss:
  use_hard_negatives: true
  hard_negative_memory_size: 4096
  hard_negative_ratio: 0.5
  hard_negative_strategy: mixed
```

**Advantages:**
- Harder contrastive learning
- Better discrimination
- Improved retrieval

**When to use:**
- Large datasets
- Retrieval-critical applications
- After baseline converges

## Projection Head Choice

| Type | Params | Speed | Quality | Use Case |
|------|--------|-------|---------|----------|
| **Linear** | ~393K | Fast | Good | Initial experiments, fast inference |
| **MLP-2** | ~2.6M | Medium | Better | SimCLR-style, balanced |
| **MLP-3** | ~6.8M | Slow | Best | Maximum quality, research |

**Recommendation**: Start with linear, upgrade to MLP-2 if needed.

## Training Tips

### Hyperparameter Tuning

**Learning Rate:**
- Default: `3e-4` (good for most cases)
- Lower: `1e-4` for fine-tuning
- Higher: `5e-4` for large batches

**Batch Size:**
- Larger is better for contrastive learning
- Default: 128
- Increase to 256-512 if GPU memory allows
- Use gradient accumulation if needed

**Temperature:**
- Default init: `0.02` (lower = harder contrastive)
- Let it be learnable (default: true)
- If unstable, fix it: `learnable_temperature: false`

**Warmup:**
- Default: 10% of steps
- Increase to 20% if training unstable
- Decrease to 5% if training slow to converge

### Common Issues

**Issue: Loss not decreasing**
- Check data alignment (sensor ↔ text)
- Increase batch size
- Lower learning rate
- Verify embeddings are normalized

**Issue: Training unstable**
- Increase warmup ratio
- Lower learning rate
- Fix temperature (don't learn it)
- Enable gradient clipping

**Issue: Poor retrieval performance**
- Try MLP projection instead of linear
- Add MLM loss
- Enable hard negative sampling
- Increase training steps

**Issue: Overfitting**
- Add weight decay
- Use dropout in projections
- Enable data augmentation (TODO)
- Reduce model size

## Output and Checkpoints

### Saved Files

```
trained_models/milan/alignment_baseline/
├── config.yaml                    # Training configuration
├── best_model.pt                  # Best validation checkpoint
├── final_model.pt                 # Final checkpoint
├── checkpoint_step_2000.pt        # Intermediate checkpoints
├── checkpoint_step_4000.pt
└── ...
```

### Checkpoint Contents

```python
checkpoint = {
    'global_step': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'best_val_loss': float,
    'config': AlignmentConfig,
    'scaler_state_dict': dict,  # If using AMP
}
```

### Loading a Checkpoint

```python
from src.alignment.model import AlignmentModel
import torch

# Load checkpoint
checkpoint = torch.load('trained_models/milan/alignment_baseline/best_model.pt')

# Get config and vocab
config = checkpoint['config']
vocab_sizes = ...  # Load from data

# Create model
model = AlignmentModel(config, vocab_sizes)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Monitoring Training

### WandB Metrics

**Loss Metrics:**
- `train/total_loss`: Combined loss
- `train/clip_loss`: Contrastive loss
- `train/mlm_loss`: MLM loss (if enabled)
- `train/temperature`: Learned temperature value

**Accuracy Metrics:**
- `train/s2t_acc`: Sensor-to-text top-1 accuracy
- `train/t2s_acc`: Text-to-sensor top-1 accuracy
- Average should increase during training

**Training Metrics:**
- `train/learning_rate`: Current LR
- `train/step`: Global step
- `train/epoch`: Current epoch

**Validation Metrics:**
- `val_loss`, `val_clip_loss`, `val_s2t_acc`, `val_t2s_acc`

### Good Training Indicators

✅ **Healthy Training:**
- Loss steadily decreasing
- Accuracies increasing (both s2t and t2s)
- Temperature stabilizing (if learnable)
- Val loss tracking train loss

❌ **Problems:**
- Loss not decreasing after warmup
- Accuracies stuck near 0
- Large train/val gap (overfitting)
- NaN or Inf losses

## Integration with Other Steps

### From Step 1 (Sampling) → Step 5

```bash
# 1. Sample data
python sample_data.py --config configs/sampling/milan_fixed_duration_60.yaml

# 2. Skip to alignment (if embeddings pre-computed)
python train.py --config configs/alignment/milan_baseline.yaml
```

### From Step 3 (Captions) → Step 5

```bash
# 1. Generate captions
python generate_captions.py \
  --data data/processed/casas/milan/fixed_duration_60s/train.json \
  --config configs/captions/baseline_milan.yaml

# 2. Encode captions
python encode_captions.py \
  --captions data/processed/casas/milan/fixed_duration_60s/train_captions_baseline.json \
  --encoder configs/text_encoders/gte_base.yaml

# 3. Train alignment
python train.py --config configs/alignment/milan_baseline.yaml
```

### Full Pipeline

```bash
# All-in-one: sampling → captions → encoding → alignment
python train.py \
  --config configs/alignment/milan_baseline.yaml \
  --run-full-pipeline
```

## Advanced Usage

### Custom Encoder

Create a custom encoder config:

```yaml
# configs/encoders/my_custom.yaml
type: transformer
d_model: 512
n_layers: 4
n_heads: 8
d_ff: 2048
dropout: 0.1
max_seq_len: 256
projection_dim: 512
projection_type: mlp
projection_hidden_dim: 1024
projection_num_layers: 2
```

Use in alignment config:

```yaml
encoder_config_path: configs/encoders/my_custom.yaml
```

### Multi-GPU Training

```python
# TODO: Implement distributed training
# For now, train on single GPU/MPS device
```

### Gradient Accumulation

Simulate larger batch sizes:

```yaml
training:
  batch_size: 64  # Actual batch size
  # TODO: Add gradient_accumulation_steps: 2  # Effective batch size: 128
```

## Next Steps

After alignment training:

1. **Step 6: Retrieval** - Use aligned embeddings for activity retrieval
2. **Step 7: Clustering** - Cluster activities using SCAN
3. **Evaluation** - Benchmark on downstream tasks

## Troubleshooting

### Import Errors

```python
# Make sure src/ is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### CUDA Out of Memory

- Reduce batch size
- Use smaller encoder
- Enable gradient checkpointing (TODO)
- Use float16 (AMP enabled by default on CUDA)

### Slow Training

- Increase num_workers
- Use pre-computed embeddings (not on-the-fly)
- Use smaller encoder
- Profile with PyTorch profiler

---

**Status**: Implemented and ready to use ✅
**Last Updated**: November 14, 2025

