# Text Encoder Guide (Step 4)

This guide explains how to use the text encoder framework to embed captions generated in Step 3.

## Overview

Text encoders convert textual captions into fixed-dimensional vector embeddings. These embeddings are:
- **Frozen**: The text encoder weights are not updated during training
- **Pre-computed**: Embeddings are generated once and cached
- **Reusable**: Can be loaded during CLIP training without re-encoding

## Quick Start

### 1. Encode Captions from Step 3

```bash
# Basic usage with default GTE encoder
python src/text_encoders/encode_captions.py \
    --captions data/processed/casas/milan/fixed_length_20/train_captions_baseline.json \
    --output data/embeddings/text/milan/fixed_length_20/train_baseline_gte.npz

# Use a different encoder
python src/text_encoders/encode_captions.py \
    --captions data/processed/casas/milan/fixed_length_20/train_captions_baseline.json \
    --config configs/text_encoders/distilroberta_base.yaml \
    --output data/embeddings/text/milan/fixed_length_20/train_baseline_distilroberta.npz

# Use GPU for faster encoding
python src/text_encoders/encode_captions.py \
    --captions data/processed/casas/milan/fixed_length_20/train_captions_baseline.json \
    --output data/embeddings/text/milan/fixed_length_20/train_baseline_gte.npz \
    --device cuda
```

### 2. List Available Encoders

```bash
python src/text_encoders/encode_captions.py --list-configs
```

## Available Encoders

### GTE (General Text Embeddings)
- **Model**: `thenlper/gte-base`
- **Embedding dim**: 768
- **Recommended for**: Default choice, good balance of quality and speed
- **Config**: `configs/text_encoders/gte_base.yaml`

### DistilRoBERTa
- **Model**: `distilroberta-base`
- **Embedding dim**: 768
- **Recommended for**: Alternative to GTE with similar performance
- **Config**: `configs/text_encoders/distilroberta_base.yaml`

### MiniLM-L6
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding dim**: 384
- **Recommended for**: Faster encoding, smaller memory footprint
- **Config**: `configs/text_encoders/minilm_l6.yaml`

### CLIP Text
- **Model**: `openai/clip-vit-base-patch32`
- **Embedding dim**: 512
- **Recommended for**: Direct CLIP compatibility
- **Config**: `configs/text_encoders/clip_vit_base.yaml`

### SigLIP Text
- **Model**: `google/siglip-base-patch16-224`
- **Embedding dim**: 768
- **Recommended for**: Alternative to CLIP
- **Config**: `configs/text_encoders/siglip_base.yaml`

## Configuration

### YAML Config Structure

```yaml
encoder_type: gte              # Encoder type: gte, distilroberta, llama, clip, siglip
model_name: thenlper/gte-base  # HuggingFace model name
embedding_dim: 768             # Output embedding dimension
max_length: 512                # Max sequence length for tokenization
batch_size: 32                 # Batch size for encoding
normalize: true                # L2-normalize embeddings
device: cpu                    # Device: cpu, cuda, mps
cache_dir: null                # Model cache directory (optional)

# Optional projection head (for CLIP compatibility)
use_projection: false          # Whether to use projection head
projection_dim: 512            # Projection dimension if enabled
```

### Projection Head

The projection head is useful when:
1. You want to match a specific embedding dimension (e.g., 512 for CLIP)
2. You're aligning with sensor encoders that use a projection

Example config with projection:

```yaml
encoder_type: gte
model_name: thenlper/gte-base
embedding_dim: 768
use_projection: true
projection_dim: 512  # Project 768-d → 512-d
```

## Programmatic Usage

### Basic Encoding

```python
from text_encoders import GTETextEncoder, TextEncoderConfig

# Create config
config = TextEncoderConfig(
    encoder_type='gte',
    model_name='thenlper/gte-base',
    embedding_dim=768,
    normalize=True,
    device='cpu'
)

# Initialize encoder
encoder = GTETextEncoder(config)

# Encode captions
captions = [
    "Person moves from kitchen to living room",
    "Motion detected in bedroom"
]
output = encoder.encode(captions)

# Access embeddings
embeddings = output.embeddings  # Shape: [2, 768]
```

### Load from YAML Config

```python
from text_encoders import GTETextEncoder, TextEncoderConfig

# Load config
config = TextEncoderConfig.from_yaml('configs/text_encoders/gte_base.yaml')

# Initialize encoder
encoder = GTETextEncoder(config)

# Encode
output = encoder.encode(captions)
```

### Batch Encoding for Large Datasets

```python
# For memory efficiency with large datasets
output = encoder.encode_batch(captions, batch_size=32)
```

### Save and Load Embeddings

```python
# Save embeddings
encoder.save_embeddings(
    embeddings=output.embeddings,
    output_path='embeddings.npz',
    metadata={'dataset': 'milan', 'style': 'baseline'}
)

# Load embeddings
embeddings, metadata = encoder.load_embeddings('embeddings.npz')
```

## Output Format

Embeddings are saved as compressed numpy archives (`.npz`) with the following structure:

```python
{
    'embeddings': np.ndarray,        # [num_samples, embedding_dim]
    'sample_ids': np.ndarray,        # [num_samples] - sample IDs from caption file
    'encoder_type': str,             # e.g., 'gte'
    'model_name': str,               # e.g., 'thenlper/gte-base'
    'embedding_dim': int,            # e.g., 768
    'normalize': bool,               # Whether embeddings are normalized
    'use_projection': bool,          # Whether projection was used
    'projection_dim': int,           # Projection dimension (0 if not used)
}
```

### Loading in Training Code

```python
import numpy as np

# Load embeddings
data = np.load('embeddings.npz')
embeddings = data['embeddings']      # [num_samples, embedding_dim]
sample_ids = data['sample_ids']      # [num_samples]

# Use in training
# embeddings can be indexed by sample ID to get corresponding text embedding
```

## Integration with CLIP Training (Step 5)

Text embeddings will be loaded in the CLIP training collate function:

```python
def collate_fn(batch, text_embeddings_cache):
    # Get sample IDs from batch
    sample_ids = [sample['sample_id'] for sample in batch]

    # Look up pre-computed text embeddings
    text_embeddings = []
    for sample_id in sample_ids:
        embedding = text_embeddings_cache[sample_id]
        text_embeddings.append(embedding)

    text_embeddings = torch.tensor(np.stack(text_embeddings))

    # ... rest of collate function
```

## Common Workflows

### Workflow 1: Encode All Caption Styles

```bash
# Baseline captions
python src/text_encoders/encode_captions.py \
    --captions data/processed/casas/milan/fixed_length_20/train_captions_baseline.json \
    --output data/embeddings/text/milan/fixed_length_20/train_baseline_gte.npz

# Sourish captions
python src/text_encoders/encode_captions.py \
    --captions data/processed/casas/milan/fixed_length_20/train_captions_sourish.json \
    --output data/embeddings/text/milan/fixed_length_20/train_sourish_gte.npz
```

### Workflow 2: Encode with Multiple Encoders

```bash
# GTE
python src/text_encoders/encode_captions.py \
    --captions data/processed/casas/milan/fixed_length_20/train_captions_baseline.json \
    --config configs/text_encoders/gte_base.yaml \
    --output data/embeddings/text/milan/fixed_length_20/train_baseline_gte.npz

# DistilRoBERTa
python src/text_encoders/encode_captions.py \
    --captions data/processed/casas/milan/fixed_length_20/train_captions_baseline.json \
    --config configs/text_encoders/distilroberta_base.yaml \
    --output data/embeddings/text/milan/fixed_length_20/train_baseline_distilroberta.npz

# CLIP
python src/text_encoders/encode_captions.py \
    --captions data/processed/casas/milan/fixed_length_20/train_captions_baseline.json \
    --config configs/text_encoders/clip_vit_base.yaml \
    --output data/embeddings/text/milan/fixed_length_20/train_baseline_clip.npz
```

### Workflow 3: Train and Test Sets

```bash
# Encode training captions
python src/text_encoders/encode_captions.py \
    --captions data/processed/casas/milan/fixed_length_20/train_captions_baseline.json \
    --output data/embeddings/text/milan/fixed_length_20/train_baseline_gte.npz

# Encode test captions
python src/text_encoders/encode_captions.py \
    --captions data/processed/casas/milan/fixed_length_20/test_captions_baseline.json \
    --output data/embeddings/text/milan/fixed_length_20/test_baseline_gte.npz
```

## Directory Structure

Recommended output structure:

```
data/embeddings/text/
├── milan/
│   ├── fixed_length_20/
│   │   ├── train_baseline_gte.npz
│   │   ├── train_baseline_distilroberta.npz
│   │   ├── train_sourish_gte.npz
│   │   ├── test_baseline_gte.npz
│   │   └── test_sourish_gte.npz
│   └── fixed_duration_60/
│       └── train_baseline_gte.npz
└── aruba/
    └── fixed_length_20/
        └── train_baseline_gte.npz
```

## Tips and Best Practices

1. **Choose the Right Encoder**:
   - GTE: Default choice, good all-around performance
   - DistilRoBERTa: Alternative to GTE
   - MiniLM: Faster, smaller, good for prototyping
   - CLIP/SigLIP: Use if you want direct compatibility with vision models

2. **Use Projection Heads**:
   - Enable projection if your sensor encoder outputs 512-d embeddings
   - This ensures matching dimensions for CLIP loss

3. **Batch Size**:
   - Larger batches are faster but use more memory
   - Adjust based on your hardware (CPU: 32, GPU: 64-128)

4. **Device Selection**:
   - Use GPU/MPS for large datasets (10k+ captions)
   - CPU is fine for smaller datasets or prototyping

5. **File Naming**:
   - Use descriptive names: `{split}_{caption_style}_{encoder}.npz`
   - Example: `train_baseline_gte.npz`

## Troubleshooting

### Out of Memory Errors

Reduce batch size:
```bash
python src/text_encoders/encode_captions.py --batch-size 16 ...
```

### Model Download Issues

Set cache directory:
```yaml
cache_dir: /path/to/model/cache
```

### Dimension Mismatches

Check embedding dimensions:
```python
data = np.load('embeddings.npz')
print(data['embedding_dim'])  # Should match your config
```

## Next Steps

After encoding captions:
1. ✓ **Step 1**: Sample data → `train.json`, `test.json`
2. ✓ **Step 2**: Encode sensor sequences → Sensor encoder ready
3. ✓ **Step 3**: Generate captions → `train_captions_*.json`
4. ✓ **Step 4**: Encode captions → `train_*_embeddings.npz` ← **You are here**
5. ⏳ **Step 5**: Train CLIP alignment with pre-computed embeddings

See `docs/STEP4_TEXT_ENCODER_SUMMARY.md` for implementation details.

