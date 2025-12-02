# Configurable Alignment Loss Types

## Overview

The alignment training pipeline now supports three different loss types for contrastive learning:

1. **InfoNCE** (default) - CLIP-style bidirectional contrastive loss
2. **Sigmoid** - SigLIP-style pairwise binary cross-entropy
3. **Focal Sigmoid** - Focal loss variant for handling hard examples

All loss types maintain the same temperature scaling and hard negative sampling functionality.

## Configuration

Add the following fields to the `loss` section of your YAML config:

```yaml
loss:
  clip_weight: 1.0
  temperature_init: 0.07
  learnable_temperature: true
  mlm_weight: 0.5
  use_hard_negatives: false
  hard_negative_memory_size: 4096
  hard_negative_ratio: 0.5
  hard_negative_strategy: mixed
  hard_negative_sampling_temperature: 0.1

  # Alignment loss type configuration
  alignment_loss_type: infonce    # options: infonce | sigmoid | focal_sigmoid
  focal_gamma: 2.0                # only used for focal_sigmoid
  focal_alpha: null               # optional class-balancing for focal_sigmoid
```

### Backward Compatibility

If `alignment_loss_type` is missing from your config, it defaults to `"infonce"`, ensuring all existing configs continue to work without modification.

## Loss Type Details

### 1. InfoNCE (CLIP-style)

**Type**: `infonce`

**Description**: Bidirectional cross-entropy loss that treats alignment as a classification problem. For a batch of B samples, each sensor embedding should match exactly one text embedding (on the diagonal).

**Formula**:
```
loss_i = CrossEntropy(sim_matrix / temperature, targets)
loss_j = CrossEntropy(sim_matrix.T / temperature, targets)
loss = 0.5 * (loss_i + loss_j)
```

**When to use**:
- Default choice for most applications
- Works well with in-batch negatives
- Supports hard negative sampling
- Standard CLIP-style contrastive learning

**Example config**:
```yaml
loss:
  alignment_loss_type: infonce
  temperature_init: 0.07
  learnable_temperature: true
```

### 2. Sigmoid (SigLIP-style)

**Type**: `sigmoid`

**Description**: Treats each pair independently using binary cross-entropy. Positives are on the diagonal (matching pairs), negatives are off-diagonal (non-matching pairs).

**Formula**:
```
labels = eye(B)  # Identity matrix
loss = BCE_with_logits(sim_matrix / temperature, labels)
```

**When to use**:
- When you want independent pairwise learning
- May be more stable than InfoNCE in some cases
- Works well with smaller batches
- Doesn't require normalization across negatives

**Example config**:
```yaml
loss:
  alignment_loss_type: sigmoid
  temperature_init: 0.07
  learnable_temperature: true
```

**Note**: Hard negatives are currently only supported with InfoNCE.

### 3. Focal Sigmoid

**Type**: `focal_sigmoid`

**Description**: Applies focal weighting to sigmoid BCE to focus learning on hard examples. The focal weight is `(1 - p_t)^gamma` where `p_t` is the predicted probability of the correct class.

**Formula**:
```
p = sigmoid(sim_matrix / temperature)
pt = labels * p + (1 - labels) * (1 - p)
focal_weight = (1 - pt)^gamma
if alpha is not None:
    focal_weight *= (labels * alpha + (1 - labels) * (1 - alpha))
loss = mean(focal_weight * BCE_with_logits(sim_matrix / temperature, labels))
```

**Parameters**:
- `focal_gamma`: Focusing parameter (higher = more focus on hard examples)
  - Default: 2.0
  - Range: [0, 5] (typical: 1-3)
- `focal_alpha`: Optional class-balancing weight
  - Default: None (no balancing)
  - Range: [0, 1] (typical: 0.25 or 0.5)

**When to use**:
- When you have class imbalance (many more negatives than positives)
- When you want to focus on hard examples
- For fine-tuning after initial training with InfoNCE
- When dealing with noisy data

**Example config**:
```yaml
loss:
  alignment_loss_type: focal_sigmoid
  focal_gamma: 2.0      # Focus on hard examples
  focal_alpha: 0.25     # Balance positives (optional)
  temperature_init: 0.07
  learnable_temperature: true
```

## Comparison

| Feature | InfoNCE | Sigmoid | Focal Sigmoid |
|---------|---------|---------|---------------|
| **Hard Negatives** | ✅ Supported | ❌ Not yet | ❌ Not yet |
| **Batch Sensitivity** | High | Low | Low |
| **Memory Usage** | Standard | Standard | Standard |
| **Gradient Stability** | Good | Very Good | Good |
| **Hard Example Focus** | Moderate | Low | Very High |
| **Typical Use Case** | General contrastive learning | Stable training, small batches | Hard examples, imbalanced data |

## Usage Examples

### Example 1: Standard Training with InfoNCE

```yaml
loss:
  clip_weight: 1.0
  temperature_init: 0.07
  learnable_temperature: true
  alignment_loss_type: infonce
```

### Example 2: Stable Training with Sigmoid

```yaml
loss:
  clip_weight: 1.0
  temperature_init: 0.1  # Slightly higher temperature for sigmoid
  learnable_temperature: true
  alignment_loss_type: sigmoid
```

### Example 3: Focal Loss for Hard Examples

```yaml
loss:
  clip_weight: 1.0
  temperature_init: 0.07
  learnable_temperature: true
  alignment_loss_type: focal_sigmoid
  focal_gamma: 2.0
  focal_alpha: 0.25
```

### Example 4: InfoNCE with Hard Negatives

```yaml
loss:
  clip_weight: 1.0
  temperature_init: 0.07
  learnable_temperature: true
  alignment_loss_type: infonce
  use_hard_negatives: true
  hard_negative_memory_size: 4096
  hard_negative_ratio: 0.5
  hard_negative_strategy: mixed
```

## Implementation Details

### Temperature Scaling

All loss types use the same temperature scaling mechanism:

```python
similarity_matrix = sensor_emb @ text_emb.T  # Cosine similarity
logits = similarity_matrix / temperature
```

The temperature is:
- Initialized from `temperature_init`
- Learnable if `learnable_temperature=true`
- Stored in log-space for numerical stability

### Similarity Matrix

The similarity matrix is computed as:
```python
sim_matrix = torch.matmul(
    F.normalize(sensor_emb, p=2, dim=-1),
    F.normalize(text_emb, p=2, dim=-1).T
)
```

This gives cosine similarities in [-1, 1], which are then divided by temperature.

### Hard Negatives (InfoNCE only)

When hard negatives are enabled with InfoNCE:
1. Memory bank stores past embeddings
2. Hard negatives are sampled based on similarity
3. Similarity matrix is extended: `[B, B+N]` where N is number of hard negatives
4. Targets remain `[0, 1, 2, ..., B-1]` (positives still on first B positions)

**Note**: Hard negatives are currently only compatible with InfoNCE loss type.

## Testing

A comprehensive test suite is provided in `test_alignment_loss.py`:

```bash
python test_alignment_loss.py
```

This tests:
- All three loss types work correctly
- Backward compatibility (missing config defaults to infonce)
- Gradient flow through all loss types
- Integration with CLIPLoss and CombinedLoss
- Invalid loss type handling

## Migration Guide

### From Existing Configs

If you have existing configs, they will continue to work without changes:
- Missing `alignment_loss_type` defaults to `"infonce"`
- All existing hyperparameters remain the same
- No breaking changes to the training pipeline

### To Try New Loss Types

To experiment with new loss types:

1. Copy your existing config
2. Add `alignment_loss_type: sigmoid` or `alignment_loss_type: focal_sigmoid`
3. (Optional) Adjust temperature for sigmoid losses (try 0.1 instead of 0.07)
4. (For focal) Set `focal_gamma` and optionally `focal_alpha`
5. Train and compare results

### Recommended Experimentation Order

1. **Baseline**: Start with InfoNCE (default)
2. **Alternative**: Try sigmoid for more stable training
3. **Advanced**: Use focal_sigmoid if you have hard examples or imbalanced data

## WandB Logging

The alignment loss type is automatically logged to WandB:
- Loss type is included in run config
- No changes needed to logging code
- Use `alignment_loss_type` field to filter runs

## References

- **InfoNCE**: [CLIP paper](https://arxiv.org/abs/2103.00020)
- **Sigmoid**: [SigLIP paper](https://arxiv.org/abs/2303.15343)
- **Focal Loss**: [Focal Loss paper](https://arxiv.org/abs/1708.02002)

## Troubleshooting

### Loss is NaN

- Reduce temperature (try 0.05 or 0.03)
- Reduce learning rate
- Enable gradient clipping
- Check for invalid embeddings

### Loss not improving

- Try different loss type (sigmoid is more stable)
- Adjust temperature
- Increase batch size (for InfoNCE)
- For focal: reduce gamma (try 1.0)

### Gradients exploding

- Lower learning rate
- Enable gradient clipping
- Reduce temperature
- For focal: reduce gamma

## Future Work

- Hard negative support for sigmoid and focal_sigmoid
- Adaptive temperature scheduling
- Mixed loss strategies (combine multiple loss types)
- Per-example focal weights based on difficulty

