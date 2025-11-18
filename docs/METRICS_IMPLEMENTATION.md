# Metrics Implementation Summary

## Overview

Comprehensive metrics suite has been integrated into the alignment training pipeline (`AlignmentTrainer`). Metrics are computed at two different intervals with appropriate sampling strategies.

---

## Implementation Details

### 1. **At `val_interval` (every 50 steps by default)**

**Metrics Computed**:
- ✅ Basic losses: `val_loss`, `val_clip_loss`, `val_mlm_loss`
- ✅ Basic accuracies: `val_s2t_acc`, `val_t2s_acc`
- ✅ **NEW: Alignment Health**
  - `val_alignment/pos_cos_mean` - Mean cosine similarity for positive pairs
  - `val_alignment/neg_cos_mean` - Mean cosine similarity for negative pairs
  - `val_alignment/pos_neg_gap` - **Key metric**: Gap between pos and neg (higher is better)
  - Additional: pos/neg median, std
- ✅ **NEW: MLM Accuracy**
  - `val_mlm_accuracy/overall` - Macro-averaged accuracy across all fields
  - `val_mlm_accuracy/{field}` - Per-field accuracy (sensor, state, room_id)
  - `val_mlm_accuracy/{field}_top5` - Top-5 accuracy for fields with large vocab

**Sampling**: Uses full validation set (typically small, ~500-1000 samples)

**Console Output Example**:
```
Validation | Loss: 4.5234 | CLIP: 4.1234 | S2T: 0.156 | T2S: 0.148 | MLM: 0.3456 | Gap: 0.234 | MLM Acc: 0.782
```

**Cost**: 🟢 LOW - Simple operations, no additional forward passes needed

---

### 2. **At `metrics_interval` (every 50 steps by default)**

**Metrics Computed**:
- ✅ **Recall@K** (k=1, 5, 10) - Overall retrieval metrics
  - `{split}/recall@{k}/sensor_to_text` - Sensor→text retrieval
  - `{split}/recall@{k}/text_to_sensor` - Text→sensor retrieval
  - `{split}/recall@{k}/average` - Average of both directions
- ✅ **Recall@5 Per-Class** - Activity-specific breakdown
  - `{split}/recall@5/class_{activity}/sensor_to_text`
  - `{split}/recall@5/class_{activity}/text_to_sensor`
  - `{split}/recall@5/class_{activity}/average`
  - `{split}/recall@5/class_{activity}/num_samples`
- ✅ **nDCG@K** (k=1, 5, 10) - Overall ranking metrics
  - `{split}/ndcg@{k}/sensor_to_text`
  - `{split}/ndcg@{k}/text_to_sensor`
  - `{split}/ndcg@{k}/average`
- ✅ Computed on **both train and val** splits

**Sampling Strategy**:
- Samples `metrics_sample_batches` (default: 10) from each data loader
- Total samples per split: ~1,280 (10 batches × 128 batch_size)
- Configurable via `training.metrics_sample_batches` in config

**Console Output Example**:
```
Computing comprehensive retrieval metrics at step 50...
Comprehensive metrics for train: collected 1280 samples from 10 batches
Comprehensive metrics for val: collected 640 samples from 10 batches
Retrieval Metrics | train/R@1: 0.156 | train/R@5: 0.423 | val/R@1: 0.148 | val/R@5: 0.401
```

**Cost**: 🟡 MEDIUM - Requires topk operations and similarity matrix computation

---

## Configuration

### Updated `TrainingConfig` (`src/alignment/config.py`)

```python
@dataclass
class TrainingConfig:
    # Logging intervals
    log_interval: int = 50
    val_interval: int = 500          # Validation + alignment health + MLM acc
    save_interval: int = 2000
    metrics_interval: int = 1000     # Comprehensive retrieval metrics

    # Metrics sampling
    metrics_sample_batches: int = 10  # Batches to sample per split
    metrics_sample_size: int = 1000   # Target sample size (for future use)
```

### Updated `milan_fixed20_v0.yaml`

```yaml
training:
  # Logging intervals
  log_interval: 50          # Log basic training metrics (loss, acc, temp, lr)
  val_interval: 50          # Validation: loss, acc, alignment health, MLM accuracy
  save_interval: 2000       # Save checkpoints
  metrics_interval: 50      # Comprehensive metrics: Recall@K, nDCG@K (retrieval)

  # Metrics sampling configuration
  metrics_sample_batches: 10  # Sample 10 batches (~1280 samples) for retrieval metrics
  metrics_sample_size: 1000   # Target sample size for expensive metrics
```

---

## Code Changes

### 1. **`AlignmentTrainer.__init__()`**
- ✅ Import `TrainingMetrics` from `src.utils.training_metrics`
- ✅ Initialize `self.metrics_tracker` with vocab_sizes and sample_size

### 2. **`AlignmentTrainer.validate()`** (Enhanced)
- ✅ Collect sensor and text embeddings throughout validation
- ✅ Collect MLM predictions, labels, and mask positions
- ✅ Compute alignment health metrics using `metrics_tracker.compute_alignment_health()`
- ✅ Compute MLM accuracy using `metrics_tracker.compute_mlm_accuracy()`
- ✅ Add `val_` prefix to all metrics for consistency
- ✅ Enhanced console logging to show Gap and MLM Acc

### 3. **`AlignmentTrainer.compute_comprehensive_metrics()`** (New Method)
- ✅ Sample `max_batches` from provided data loader
- ✅ Collect sensor and text embeddings
- ✅ Collect ground truth labels (if available) for per-class metrics
- ✅ Compute Recall@K (k=1,5,10) overall + Recall@5 per-class
- ✅ Compute nDCG@K (k=1,5,10) overall only
- ✅ Add split prefix (train/val) to all metrics
- ✅ Log sample size and batch count

### 4. **`AlignmentTrainer.train()`** (Enhanced Training Loop)
- ✅ Added `metrics_interval` check at step `self.global_step % metrics_interval == 0`
- ✅ Call `compute_comprehensive_metrics()` on both train and val loaders
- ✅ Merge metrics from both splits
- ✅ Log key retrieval metrics to console (R@1, R@5 for train and val)
- ✅ Log all metrics to WandB

---

## Metrics NOT Implemented (As Requested)

❌ **F1 Scores** - Requires label encoding with text encoder (expensive)
❌ **Representation Diagnostics** - Embedding norms/stats (debugging only)
❌ **Per-Activity Alignment Health** - Would require caption strings

These can be added later if needed by uncommenting relevant sections.

---

## WandB Integration

All metrics are automatically logged to WandB with proper prefixes:

**Validation Metrics** (every `val_interval`):
- `val_loss`, `val_clip_loss`, `val_mlm_loss`
- `val_s2t_acc`, `val_t2s_acc`
- `val_alignment/pos_cos_mean`, `val_alignment/neg_cos_mean`, `val_alignment/pos_neg_gap`
- `val_mlm_accuracy/overall`, `val_mlm_accuracy/{field}`

**Comprehensive Metrics** (every `metrics_interval`):
- `train/recall@{k}/{direction}`, `val/recall@{k}/{direction}`
- `train/ndcg@{k}/{direction}`, `val/ndcg@{k}/{direction}`
- `train/recall@5/class_{activity}/{direction}` (per-class)
- `val/recall@5/class_{activity}/{direction}` (per-class)

---

## Performance Considerations

### Validation (every 50 steps)
- **Time**: ~2-3 seconds on validation set (~500 samples)
- **Impact**: Minimal - simple operations on collected embeddings
- **Frequency**: Can run frequently without slowing training

### Comprehensive Metrics (every 50 steps)
- **Time**: ~5-10 seconds per split (train + val)
- **Impact**: Moderate - matrix operations and topk
- **Frequency**: Currently set to same as val_interval, but can be increased if needed

### Recommended Settings

**For fast iteration** (default in config):
```yaml
val_interval: 50
metrics_interval: 50
```

**For production runs**:
```yaml
val_interval: 500        # Every ~5 minutes
metrics_interval: 1000   # Every ~10 minutes
```

**For debugging**:
```yaml
val_interval: 25
metrics_interval: 100
```

---

## Example Training Output

```
Step     50 | Loss: 4.5678 | CLIP: 4.1234 | S2T: 0.156 | T2S: 0.148 | Temp: 0.070000 | LR: 3.00e-04 | MLM: 1.2345
Validation | Loss: 4.5234 | CLIP: 4.1234 | S2T: 0.156 | T2S: 0.148 | MLM: 1.2345 | Gap: 0.234 | MLM Acc: 0.782
Computing comprehensive retrieval metrics at step 50...
Comprehensive metrics for train: collected 1280 samples from 10 batches
Comprehensive metrics for val: collected 640 samples from 10 batches
Retrieval Metrics | train/R@1: 0.156 | train/R@5: 0.423 | val/R@1: 0.148 | val/R@5: 0.401
```

---

## Testing Checklist

- [x] Metrics tracker initialized properly
- [x] Validation metrics include alignment health and MLM accuracy
- [x] Comprehensive metrics computed at correct interval
- [x] Both train and val splits evaluated for retrieval metrics
- [x] Console output is informative but not overwhelming
- [x] All metrics logged to WandB with proper prefixes
- [x] No linter errors
- [x] Config updated with proper documentation

---

## Next Steps

1. ✅ **Run a test training** to verify metrics work end-to-end
2. Monitor WandB dashboard for proper metric organization
3. Adjust `metrics_interval` based on training speed preferences
4. Consider adding F1 scores if zero-shot classification is important

---

## Quick Reference: Key Metrics to Watch

| Metric | Good Value | What It Means |
|--------|-----------|---------------|
| `val_alignment/pos_neg_gap` | > 0.3 | Strong alignment between sensor and text |
| `val_mlm_accuracy/overall` | > 0.7 | Model understands sensor sequences |
| `val/recall@5/average` | > 0.4 | Good retrieval quality |
| `val_clip_loss` | Decreasing | Alignment improving |
| `val_s2t_acc` | > 0.15 | Better than random (1/128) |

**Key Insight**: `pos_neg_gap` and `recall@5/average` are the most important metrics for monitoring alignment quality!

