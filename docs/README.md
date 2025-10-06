# Smart-Home Event Sequence Alignment

A PyTorch implementation for aligning smart-home event sequences to text via CLIP-style contrastive learning.

## Overview

This project implements a dual-tower architecture that learns to align sensor event sequences with natural language descriptions:

- **Text Tower**: Frozen `thenlper/gte-base` encoder (768-d, L2-normalized)
- **Sensor Tower**: Custom Transformer with ALiBi attention, categorical embeddings, and Fourier features
- **Training**: Bidirectional InfoNCE (CLIP) loss + multi-field MLM with span masking
- **Retrieval**: FAISS-based similarity search for text-to-sensor and sensor-to-text queries

## Architecture

### Sensor Token Embedding
- Sum of categorical field embeddings: `sensor_id + room_id + event_type + sensor_type + tod_bucket + delta_t_bucket + [floor_id] + [dow]`
- Fourier features for continuous (x,y) coordinates (L=12 bands)
- Log-bucketed time delta embeddings

### Transformer Features
- ALiBi positional bias (default)
- Optional RoPE for time/space (configurable)
- Pre-LN architecture (6-8 layers, 8 heads, d=768)
- Sequence pooling: 0.5×CLS + 0.5×mean(masked tokens)

### Training Objectives
- **CLIP Loss**: Bidirectional InfoNCE with learnable temperature (init=0.05)
- **MLM Loss** (λ=0.3): Multi-field masked language modeling
  - ~25% span masking (Poisson length≈3)
  - Field-balanced priors: room(.30), event_type(.20), sensor_id(.20), tod(.15), delta_t_bucket(.10), sensor_type(.05)
  - BERT-style 80/10/10 masking

## Installation

```bash
pip install torch transformers faiss-cpu scikit-learn numpy pandas wandb
```

## Quick Start

### 1. Create Sample Data
```python
from dataio.dataset import create_sample_dataset

create_sample_dataset(
    output_path="sample_data.json",
    vocab_path="sample_vocab.json",
    num_samples=1000,
    sequence_length=20
)
```

### 2. Train Model
```bash
python train.py \
    --train_data sample_data.json \
    --vocab sample_vocab.json \
    --output_dir ./outputs \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --max_steps 10000
```

### 3. Evaluate Retrieval
```bash
python eval_retrieval.py \
    --checkpoint ./outputs/best_model.pt \
    --eval_data sample_data.json \
    --vocab sample_vocab.json \
    --run_demo \
    --save_embeddings
```

## Data Format

### Input Data Structure
```json
{
  "events": [
    {
      "sensor_id": "motion_001",
      "room_id": "kitchen",
      "event_type": "ON",
      "sensor_type": "motion",
      "tod_bucket": "hour_8",
      "delta_t_bucket": "bucket_2",
      "floor_id": "floor_1",
      "dow": "day_1",
      "x": 2.5,
      "y": 3.1,
      "timestamp": 1234567890
    }
  ],
  "captions": [
    "morning routine",
    "getting ready for work",
    "kitchen activity"
  ]
}
```

### Vocabulary Structure
```json
{
  "sensor_id": {"sensor_001": 0, "sensor_002": 1, "<UNK>": 2, "<PAD>": 3},
  "room_id": {"kitchen": 0, "bedroom": 1, "<UNK>": 2, "<PAD>": 3},
  "event_type": {"ON": 0, "OFF": 1, "<UNK>": 2, "<PAD>": 3}
}
```

## Configuration

See `config.json` for full configuration options:

- **Model**: Architecture parameters (layers, heads, dimensions)
- **Training**: Optimization settings (lr, batch size, schedules)
- **Loss**: CLIP and MLM loss weights and parameters
- **MLM**: Masking probabilities and field priors
- **Logging**: Intervals and output settings

## Retrieval Demo

The evaluation script includes a FAISS-based retrieval demo:

```python
# Text-to-sensor queries
demo_queries = [
    "night wandering",
    "morning routine",
    "cooking dinner",
    "watching TV"
]

# Returns top-k most similar sensor sequences
results = evaluator.text_to_sensor_retrieval(demo_queries, k=10)
```

## Training Features

- **Device Support**: Automatic detection of CUDA, Apple Silicon (MPS), and CPU
- **Mixed Precision**: Automatic Mixed Precision (AMP) with GradScaler (CUDA only)
- **Gradient Checkpointing**: Memory-efficient training
- **Learning Rate Schedule**: Cosine annealing with 5% warmup
- **Monitoring**: Weights & Biases integration (optional)
- **Checkpointing**: Regular model saving with best model tracking

## File Structure

```
src-v2/
├── models/
│   ├── text_encoder.py      # Frozen gte-base encoder
│   ├── sensor_encoder.py    # Transformer with ALiBi/RoPE
│   └── mlm_heads.py         # Multi-field MLM heads
├── losses/
│   └── clip.py              # CLIP + combined loss
├── dataio/
│   ├── dataset.py           # Smart-home dataset
│   └── collate.py           # Batch collation with masking
├── train.py                 # Training script
├── eval_retrieval.py        # FAISS retrieval evaluation
└── config.json              # Configuration template
```

## Metrics

The evaluation computes:
- **Precision@k** (k=1,5,10): Retrieval precision
- **nDCG@k** (k=1,5,10): Normalized discounted cumulative gain
- **Bidirectional accuracy**: Sensor↔text alignment accuracy

## Requirements

- PyTorch ≥ 1.12 (with MPS support for Apple Silicon)
- transformers ≥ 4.20
- faiss-cpu or faiss-gpu
- scikit-learn, numpy, pandas
- Optional: wandb for experiment tracking

### Device Support
- **CUDA**: Full support with AMP and pinned memory
- **Apple Silicon (MPS)**: Optimized settings, single-threaded data loading
- **CPU**: Fallback with optimized settings

## License

This implementation is provided as-is for research purposes.
