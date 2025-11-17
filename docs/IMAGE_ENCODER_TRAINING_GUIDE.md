# Image-Based Sensor Encoder Training Guide

This guide explains how to train sensor encoders using frozen vision model embeddings (CLIP, DINOv2, SigLIP) instead of raw sensor sequences.

## Overview

The image-based encoder approach:
1. **Pre-generates** sensor activation images on floor plans
2. **Pre-embeds** them using frozen vision models (CLIP, DINOv2, etc.)
3. **Trains** a transformer to process these frozen embeddings for CLIP alignment
4. **Benefits**: Leverages powerful pre-trained vision models, spatial understanding

## Architecture

```
Sensor Activation (M001, ON)
    ↓
Frozen Image Embedding (from CLIP/DINOv2/SigLIP)
    ↓
[Frozen/Trainable] Linear Projection → d_model
    ↓
[Optional] Spatial Features (coordinates)
    ↓
[Optional] Temporal Features (time deltas)
    ↓
[Trainable] Transformer Layers (with ALiBi)
    ↓
Pooling (CLS + Mean)
    ↓
[Trainable] Projection Head → CLIP space
    ↓
Alignment with Text Embeddings
```

## Key Differences from Sequence-Based Encoder

| Aspect | Sequence-Based | Image-Based |
|--------|---------------|-------------|
| **Input** | Learnable sensor/state embeddings | Frozen vision model embeddings |
| **Input Features** | Discrete tokens (sensor IDs, states) | Rich spatial representations (512D-768D) |
| **Embedding Layer** | Trainable | Frozen (pre-computed) |
| **Transformer** | Trainable | Trainable |
| **Spatial Info** | Via Fourier features (optional) | Implicitly in image embeddings |
| **Pre-computation** | None | Images + embeddings generated once |
| **Training Speed** | Normal | Faster (no embedding lookup, frozen input) |
| **Memory** | Lower | Higher (larger input embeddings) |

## Step-by-Step Usage

### Step 1: Generate Sensor Images

Generate floor plan images with sensor activations marked:

```bash
# Generate 224×224 images (default for CLIP/SigLIP)
python -m src.encoders.sensor.image.generate_images \
    --dataset milan \
    --output-width 224 \
    --output-height 224

# Generate larger images for higher resolution models
python -m src.encoders.sensor.image.generate_images \
    --dataset milan \
    --output-width 512 \
    --output-height 512
```

**Output**: `data/processed/casas/milan/layout_embeddings/images/dim224/`

Contains images like:
- `M001_ON.png` - Motion sensor M001 active
- `M001_OFF.png` - Motion sensor M001 inactive
- `D002_CLOSE.png` - Door sensor D002 closed
- etc.

### Step 2: Embed Images with Vision Model

Process images through vision models to create frozen embeddings:

```bash
# CLIP embeddings (512D) → clip_base/
python -m src.encoders.sensor.image.embed_images \
    --dataset milan \
    --model clip \
    --batch-size 32

# DINOv2 embeddings (768D) → dinov2/
python -m src.encoders.sensor.image.embed_images \
    --dataset milan \
    --model dinov2 \
    --batch-size 32

# SigLIP embeddings (768D) → siglip_base_patch16_224/
python -m src.encoders.sensor.image.embed_images \
    --dataset milan \
    --model siglip \
    --batch-size 32
```

**Output**: `data/processed/casas/milan/layout_embeddings/embeddings/{model}/dim224/embeddings.npz`

Contains:
- `embeddings`: [num_sensors, embedding_dim] array
- `sensor_ids`: ["M001", "M002", ...]
- `states`: ["ON", "OFF", ...]
- `image_keys`: ["M001_ON", "M002_OFF", ...]

### Step 3: Prepare Text Embeddings

Generate captions and text embeddings (same as sequence-based training):

```bash
# Generate captions
python generate_captions.py \
    --config configs/captions/baseline_milan.yaml \
    --data-path data/processed/casas/milan/fixed_length_20_presegmented/train.json \
    --output-path data/processed/casas/milan/fixed_length_20_presegmented/train_captions_baseline.json

# Encode captions
python encode_captions.py \
    --config configs/text_encoders/gte_base.yaml \
    --captions-path data/processed/casas/milan/fixed_length_20_presegmented/train_captions_baseline.json \
    --output-path data/processed/casas/milan/fixed_length_20_presegmented/train_embeddings_baseline_gte_base.npz
```

### Step 4: Train Alignment Model

Train the image-based encoder:

```bash
# Using CLIP embeddings
python train.py --config configs/alignment/milan_image_clip.yaml

# Using DINOv2 embeddings
python train.py --config configs/alignment/milan_image_dinov2.yaml

# Using SigLIP embeddings
python train.py --config configs/alignment/milan_image_siglip.yaml
```

## Configuration Files

### Encoder Config Example: `configs/encoders/transformer_image_clip.yaml`

```yaml
type: transformer
encoder_type: transformer

# Image-based settings
use_image_embeddings: true
image_model_name: clip  # Options: 'clip', 'dinov2', 'siglip'
image_size: 224
freeze_input_projection: true  # Recommended

# Architecture
d_model: 768
n_layers: 6
n_heads: 8
d_ff: 3072
max_seq_len: 512
dropout: 0.1

# Projection for CLIP alignment
projection_dim: 512
projection_type: linear

# Positional encoding
use_alibi: true
use_learned_pe: false

# Metadata features (optional)
metadata:
  categorical_fields: []  # Empty - using only image embeddings
  use_coordinates: false  # Can enable for additional spatial info
  use_time_deltas: false  # Can enable for temporal dynamics

# Pooling strategy
pooling: cls_mean
pooling_cls_weight: 0.5
```

### Alignment Config Example: `configs/alignment/milan_image_clip.yaml`

```yaml
experiment_name: milan_image_clip
output_dir: trained_models/milan/alignment_image_clip

# Dataset metadata (REQUIRED for image-based encoders)
dataset: milan
dataset_type: casas

# Data paths
train_data_path: data/processed/casas/milan/fixed_length_20_presegmented/train.json
val_data_path: data/processed/casas/milan/fixed_length_20_presegmented/test.json
vocab_path: data/processed/casas/milan/fixed_length_20_presegmented/vocab.json

# Text embeddings
train_text_embeddings_path: data/processed/casas/milan/fixed_length_20_presegmented/train_embeddings_baseline_gte_base.npz
val_text_embeddings_path: data/processed/casas/milan/fixed_length_20_presegmented/test_embeddings_baseline_gte_base.npz

# Encoder config (image-based)
encoder_config_path: configs/encoders/transformer_image_clip.yaml
encoder_type: transformer

# Loss configuration
loss:
  clip_weight: 1.0
  mlm_weight: 0.0  # No MLM for image-based (no learnable embeddings)

# ... (rest of training config)
```

## Choosing a Vision Model

### CLIP (`clip`)
- **Dimensions**: 512
- **Strengths**: Well-aligned with text, contrastive pre-training
- **Use case**: Good default choice for text-vision alignment
- **Training**: Trained on 400M image-text pairs

### DINOv2 (`dinov2`)
- **Dimensions**: 768
- **Strengths**: Better spatial understanding, self-supervised
- **Use case**: When spatial relationships matter more
- **Training**: Self-supervised on diverse images

### SigLIP (`siglip`)
- **Dimensions**: 768
- **Strengths**: Improved CLIP with signature loss
- **Use case**: Alternative to CLIP with potentially better performance
- **Training**: Similar to CLIP but with improved loss function

## Configuration Options

### Image-Based Settings

```yaml
use_image_embeddings: true          # Enable image-based mode
image_model_name: clip              # Vision model: 'clip', 'dinov2', 'siglip'
image_size: 224                     # Must match generated embeddings
freeze_input_projection: true       # Freeze projection layer (recommended)
```

**Why freeze input projection?**
- Image embeddings are already high-quality representations
- Freezing prevents overfitting to the downstream task
- Faster training with fewer parameters to update
- Can set to `false` if you want to fine-tune the projection

### Metadata Features

You can optionally add spatial and temporal features on top of image embeddings:

```yaml
metadata:
  categorical_fields: []       # Empty for pure image-based
  use_coordinates: true        # Add Fourier coordinate features
  use_time_deltas: true        # Add temporal bucket features
```

**When to enable metadata features:**
- `use_coordinates: true` - If spatial relationships beyond what's in images are important
- `use_time_deltas: true` - If temporal dynamics are crucial
- Both `false` - For pure vision-based representation

### Pooling Strategy

```yaml
pooling: cls_mean              # Options: 'cls', 'mean', 'cls_mean'
pooling_cls_weight: 0.5        # Weight for CLS in cls_mean (0.0-1.0)
```

- `cls`: Use only CLS token (attention-based global representation)
- `mean`: Average all sequence tokens (uniform representation)
- `cls_mean`: Weighted combination (recommended)

## Training Tips

### 1. No MLM Loss
Image-based encoders don't use MLM loss because the input embeddings are frozen:

```yaml
loss:
  clip_weight: 1.0
  mlm_weight: 0.0  # Must be 0 for image-based
```

### 2. Batch Size
Image embeddings are larger (512D-768D), so you might need smaller batch sizes:

```yaml
training:
  batch_size: 64              # Smaller than sequence-based (128-256)
  accumulation_steps: 2       # Use gradient accumulation if needed
```

### 3. Learning Rate
Start with slightly lower learning rate:

```yaml
optimizer:
  lr: 0.0003                  # vs 0.0005 for sequence-based
```

### 4. Projection Head
Use MLP projection for better alignment:

```yaml
sensor_projection:
  type: mlp                   # vs 'linear'
  dim: 512
  hidden_dim: 2048
  num_layers: 2
```

## Programmatic Usage

### Creating Image-Based Encoder

```python
from src.encoders import build_encoder
from src.encoders.config import TransformerEncoderConfig

# Load config
config = TransformerEncoderConfig.from_dict({
    'use_image_embeddings': True,
    'image_model_name': 'clip',
    'image_size': 224,
    'freeze_input_projection': True,
    'd_model': 768,
    'projection_dim': 512,
    # ... other config
})

# Build encoder (requires dataset and vocab)
encoder = build_encoder(
    config=config,
    dataset='milan',
    dataset_type='casas',
    vocab=vocab  # Required for image-based encoders
)

# Forward pass
output = encoder(input_data, attention_mask)
projected = encoder.forward_clip(input_data, attention_mask)
```

### Checking Encoder Type

```python
from src.encoders.sensor.sequence.image_transformer import ImageTransformerSensorEncoder

if isinstance(encoder, ImageTransformerSensorEncoder):
    print(f"Image-based encoder using {encoder.image_model_name}")
    print(f"Frozen embeddings: {len(encoder.image_lookup.image_keys)} sensor-state pairs")
    print(f"Embedding dimension: {encoder.image_lookup.embedding_dim}")
```

## Troubleshooting

### Error: "Image embeddings not found"

**Cause**: Images or embeddings haven't been generated.

**Solution**:
```bash
# Step 1: Generate images
python -m src.encoders.sensor.image.generate_images --dataset milan

# Step 2: Embed images
python -m src.encoders.sensor.image.embed_images --dataset milan --model clip
```

### Error: "No image embedding found for sensor 'M001' with state 'ON'"

**Cause**: Missing sensor-state pair in embeddings.

**Solution**: Check if images were generated for all sensors:
```bash
ls data/processed/casas/milan/layout_embeddings/images/dim224/
```

### Error: "dataset parameter is required for image-based encoders"

**Cause**: Forgot to specify dataset in alignment config.

**Solution**: Add to config:
```yaml
dataset: milan
dataset_type: casas
```

### Out of Memory

**Cause**: Image embeddings are larger than learned embeddings.

**Solutions**:
1. Reduce batch size: `batch_size: 64` → `batch_size: 32`
2. Enable gradient accumulation: `accumulation_steps: 2`
3. Use mixed precision: `use_amp: true`

## Performance Comparison

### Expected Performance

| Approach | Embedding Dim | Trainable Params | Training Speed | Memory |
|----------|--------------|------------------|----------------|---------|
| Sequence-based | 768 | ~43M | 1.0x | 1.0x |
| Image (CLIP) | 512 | ~41M | 1.2x | 1.3x |
| Image (DINOv2) | 768 | ~41M | 1.2x | 1.5x |
| Image (SigLIP) | 768 | ~41M | 1.2x | 1.5x |

### Advantages of Image-Based

1. **Leverages Pre-trained Vision Models**: Benefits from models trained on millions of images
2. **Spatial Understanding**: Floor plan context implicitly encoded
3. **Transfer Learning**: Vision features may generalize better
4. **Interpretable**: Can visualize sensor activations on floor plans

### Disadvantages of Image-Based

1. **Higher Memory**: Larger input embeddings (512D-768D vs learnable)
2. **Pre-computation Required**: Must generate images and embeddings first
3. **Fixed Embeddings**: Can't fine-tune vision model during training
4. **No MLM**: Can't use masked language modeling loss

## Advanced Topics

### Custom Vision Models

To use a different vision model:

1. Add embedder class in `src/encoders/sensor/image/embed_images.py`:
```python
class MyVisionModelEmbedder(ImageEmbedder):
    def __init__(self, model_name: str = "my/vision-model", device: str = None):
        super().__init__(model_name, device)

    def load_model(self):
        # Load your model
        self.model = ...
        self.embedding_dim = ...
```

2. Update `get_embedder()` function to include your model
3. Generate embeddings: `--model my_vision_model`

### Mixing Image and Metadata Features

Combine frozen image embeddings with coordinate/time features:

```yaml
metadata:
  use_coordinates: true   # Add spatial Fourier features
  use_time_deltas: true   # Add temporal bucket features
```

This creates a hybrid representation:
- Image embedding: 512D-768D (frozen)
- Coordinate features: 768D (trainable)
- Time delta features: 768D (trainable)
- Combined: Added together in embedding space

### Unfreezing Input Projection

To fine-tune the input projection during training:

```yaml
freeze_input_projection: false
```

This allows the model to adapt the vision embeddings to the downstream task, but risks overfitting.

## Complete Example Workflow

```bash
# 1. Sample data (if not already done)
python sample_data.py \
    --config configs/sampling/milan_fixed_length_20_presegmented.yaml

# 2. Generate images
python -m src.encoders.sensor.image.generate_images \
    --dataset milan \
    --output-width 224 \
    --output-height 224

# 3. Embed images with CLIP
python -m src.encoders.sensor.image.embed_images \
    --dataset milan \
    --model clip \
    --batch-size 32

# 4. Generate captions
python generate_captions.py \
    --config configs/captions/baseline_milan.yaml \
    --data-path data/processed/casas/milan/fixed_length_20_presegmented/train.json \
    --output-path data/processed/casas/milan/fixed_length_20_presegmented/train_captions_baseline.json

# Repeat for test split
python generate_captions.py \
    --config configs/captions/baseline_milan.yaml \
    --data-path data/processed/casas/milan/fixed_length_20_presegmented/test.json \
    --output-path data/processed/casas/milan/fixed_length_20_presegmented/test_captions_baseline.json

# 5. Encode captions to text embeddings
python encode_captions.py \
    --config configs/text_encoders/gte_base.yaml \
    --captions-path data/processed/casas/milan/fixed_length_20_presegmented/train_captions_baseline.json

python encode_captions.py \
    --config configs/text_encoders/gte_base.yaml \
    --captions-path data/processed/casas/milan/fixed_length_20_presegmented/test_captions_baseline.json

# 6. Train image-based alignment model
python train.py --config configs/alignment/milan_image_clip.yaml

# 7. Monitor training
# - Check WandB: https://wandb.ai/your-entity/discover-v2
# - Check logs: logs/text/milan_image_clip/
# - Check checkpoints: trained_models/milan/alignment_image_clip/
```

## Next Steps

After training, you can:
1. **Evaluate**: Use retrieval tasks to compare with sequence-based encoders
2. **Visualize**: Inspect t-SNE plots of learned representations
3. **Compare**: Train with different vision models (CLIP vs DINOv2 vs SigLIP)
4. **Experiment**: Try different metadata feature combinations
5. **Fine-tune**: Optionally unfreeze input projection for task-specific adaptation

---

**Related Documentation:**
- `docs/IMAGE_GENERATION_GUIDE.md` - How to generate sensor images
- `docs/ENCODER_GUIDE.md` - General encoder usage
- `docs/ALIGNMENT_GUIDE.md` - Alignment training details
- `docs/TEXT_ENCODER_GUIDE.md` - Text embedding generation

