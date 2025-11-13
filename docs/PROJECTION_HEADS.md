# CLIP Projection Heads

The encoder now supports both **linear** and **MLP** projection heads for CLIP alignment.

## Quick Summary

✅ **Default**: Linear projection (768 → 512)
✅ **Option**: MLP projection (768 → 2048 → 512)
✅ **Configurable**: 2-layer or 3-layer MLP

## Usage

### Linear Projection (Default)

```python
from src.encoders.config import TransformerEncoderConfig
from src.encoders.sensor.sequence import TransformerSensorEncoder

config = TransformerEncoderConfig.base()
config.projection_type = 'linear'  # Default
config.projection_dim = 512

encoder = TransformerSensorEncoder(config)
# ~393K params in projection head
```

### MLP Projection (2-layer)

```python
config = TransformerEncoderConfig.base()
config.projection_type = 'mlp'
config.projection_dim = 512
config.projection_hidden_dim = 2048
config.projection_num_layers = 2

encoder = TransformerSensorEncoder(config)
# ~2.6M params in projection head
# Architecture: 768 → 2048 → 512
```

### MLP Projection (3-layer)

```python
config = TransformerEncoderConfig.base()
config.projection_type = 'mlp'
config.projection_dim = 512
config.projection_hidden_dim = 2048
config.projection_num_layers = 3

encoder = TransformerSensorEncoder(config)
# ~6.8M params in projection head
# Architecture: 768 → 2048 → 2048 → 512
```

## Configuration Files

### Linear (Default)
- `configs/encoders/transformer_base.yaml`

### MLP (2-layer)
- `configs/encoders/transformer_base_mlp.yaml`

### MLP (3-layer)
- `configs/encoders/transformer_base_mlp3.yaml`

## YAML Configuration

```yaml
# Projection settings
projection_dim: 512
projection_type: mlp  # 'linear' or 'mlp'
projection_hidden_dim: 2048  # Only for MLP
projection_num_layers: 2  # 2 or 3, only for MLP
```

## MLP Architecture

The MLP projection follows SimCLR/MoCo design:

**2-layer MLP:**
```
Input (768d)
  ↓
Linear(768 → 2048)
  ↓
GELU()
  ↓
Dropout(0.1)
  ↓
Linear(2048 → 512, no bias)
  ↓
L2 Normalize
  ↓
Output (512d)
```

**3-layer MLP:**
```
Input (768d)
  ↓
Linear(768 → 2048)
  ↓
GELU() + Dropout
  ↓
Linear(2048 → 2048)
  ↓
GELU() + Dropout
  ↓
Linear(2048 → 512, no bias)
  ↓
L2 Normalize
  ↓
Output (512d)
```

## When to Use Which?

### Linear Projection
**Use when:**
- Faster training
- Fewer parameters
- Simpler models
- Initial experiments

**Advantages:**
- Fast (~393K params)
- Stable training
- Less overfitting risk

### MLP Projection (2-layer)
**Use when:**
- Better alignment quality needed
- Following SimCLR/MoCo best practices
- Contrastive pre-training

**Advantages:**
- Better representation learning
- Non-linear transformation
- Standard in contrastive methods

### MLP Projection (3-layer)
**Use when:**
- Maximum alignment quality
- Large-scale training
- Following CLIP/SimCLRv2

**Advantages:**
- Deepest projection
- Best for complex mappings
- SOTA contrastive methods

## Parameter Count

For base model (768d):

| Projection Type | Params | Architecture |
|----------------|--------|--------------|
| Linear | ~393K | 768 → 512 |
| MLP (2-layer) | ~2.6M | 768 → 2048 → 512 |
| MLP (3-layer) | ~6.8M | 768 → 2048 → 2048 → 512 |

## Implementation Details

**File**: `src/encoders/sensor/sequence/projection.py`

- `LinearProjection`: Simple linear layer
- `MLPProjection`: 2 or 3-layer MLP with GELU and dropout
- `create_projection_head()`: Factory function

**Integration**: `TransformerSensorEncoder.clip_proj`

All projection heads:
- ✅ L2-normalize outputs
- ✅ Initialize with small weights (std=0.02)
- ✅ Support both forward() and forward_clip()
- ✅ Compatible with existing training code

## Backward Compatibility

✅ Default is `linear`, same as before
✅ Existing configs still work
✅ No breaking changes

## Recommendations

**For most use cases**: Start with linear, try MLP-2 if needed
**For contrastive pre-training**: Use MLP-2 (SimCLR standard)
**For production**: Linear for speed, MLP-2 for quality
**For research**: Try all three and compare

---

**Status**: Implemented and tested ✅
**Added**: November 13, 2025

