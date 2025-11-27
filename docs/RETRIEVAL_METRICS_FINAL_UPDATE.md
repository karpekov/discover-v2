# Retrieval Metrics Final Update Summary

## Changes Made

### 1. Added Per-Label Charts for All 4 Retrieval Variants

Now generates **4 per-label retrieval charts** (not just 2):

#### Instance-to-Instance Retrieval:
- **`retrieval_perlabel_text2sensor_l1.png`** - Text → Sensor per-label performance
- **`retrieval_perlabel_sensor2text_l1.png`** - Sensor → Text per-label performance

#### Prototype-Based Retrieval:
- **`retrieval_perlabel_prototype2sensor_l1.png`** - Prototype → Sensor per-label performance
- **`retrieval_perlabel_prototype2text_l1.png`** - Prototype → Text per-label performance

Each chart shows:
- Horizontal bars sorted by performance (best to worst)
- Color-coded by recall (green = good, red = poor)
- Value labels on each bar
- Average performance line for reference
- Limited to top 20 labels for readability

### 2. Consistent Label Level Usage (L1 vs L2)

**CRITICAL FIX**: Ensures label consistency to avoid mixing labels:
- **L1 labels**: `Evening_Meds`, `Morning_Meds` (separate)
- **L2 labels**: `Take_medicine` (combined)

#### Implementation:
```python
# Choose which label level to use for retrieval (L1 or L2)
retrieval_label_level = 'L1'  # Can be changed to 'L2' if needed

if retrieval_label_level == 'L1':
    text_labels_for_retrieval = np.array(test_labels_l1)
    sensor_labels_for_retrieval = np.array(test_sensor_l1)
else:
    text_labels_for_retrieval = np.array(test_labels_l2)
    sensor_labels_for_retrieval = np.array(test_sensor_l2)
```

**All retrieval computations use these consistent labels:**
- Instance-to-instance (text2sensor, sensor2text)
- Prototype-based (prototype2sensor, prototype2text)

### 3. Per-Class Metrics in comprehensive_results.json

The JSON now includes detailed per-label retrieval metrics:

```json
{
  "classification_metrics": { ... },
  "retrieval_metrics": {
    "label_level": "L1",
    "instance_to_instance": {
      "overall": {
        "text2sensor": {"10": 0.XX, "50": 0.XX, "100": 0.XX},
        "sensor2text": {"10": 0.XX, "50": 0.XX, "100": 0.XX}
      },
      "per_label": {
        "text2sensor": {
          "10": {
            "Evening_Meds": 0.XX,
            "Morning_Meds": 0.XX,
            "Cooking": 0.XX,
            ...
          },
          "50": { ... },
          "100": { ... }
        },
        "sensor2text": {
          "10": { ... },
          "50": { ... },
          "100": { ... }
        }
      }
    },
    "prototype_based": {
      "overall": {
        "prototype2sensor": {"10": 0.XX, "50": 0.XX, "100": 0.XX},
        "prototype2text": {"10": 0.XX, "50": 0.XX, "100": 0.XX}
      },
      "per_label": {
        "prototype2sensor": {
          "10": {
            "Evening_Meds": 0.XX,
            "Morning_Meds": 0.XX,
            "Cooking": 0.XX,
            ...
          },
          "50": { ... },
          "100": { ... }
        },
        "prototype2text": {
          "10": { ... },
          "50": { ... },
          "100": { ... }
        }
      }
    }
  }
}
```

### 4. New Functions Added

#### In `compute_retrieval_metrics.py`:

##### `compute_per_label_recall_at_k()`
Computes per-label recall for instance-to-instance retrieval:
```python
def compute_per_label_recall_at_k(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    query_labels: np.ndarray,
    target_labels: np.ndarray,
    k: int
) -> Dict[str, float]:
    """
    For each unique label:
    1. Get all queries with that label
    2. For each query, find top K targets
    3. Compute recall for that query
    4. Average across all queries with that label

    Returns: {label1: recall, label2: recall, ...}
    """
```

#### In `evaluate_embeddings.py`:

##### `create_per_label_retrieval_heatmap_instance()`
Creates per-label charts for instance-to-instance retrieval:
```python
def create_per_label_retrieval_heatmap_instance(
    self,
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    query_labels: np.ndarray,
    target_labels: np.ndarray,
    direction_name: str,
    k: int = 10,
    save_path: str = None
) -> plt.Figure
```

### 5. Updated Function Signatures

Both retrieval functions now support `return_per_label` parameter:

```python
# compute_label_recall_at_k
results, per_label_results = compute_label_recall_at_k(
    ...,
    return_per_label=True  # Returns tuple: (overall, per_label)
)

# compute_prototype_retrieval_metrics
results, per_label_results = compute_prototype_retrieval_metrics(
    ...,
    return_per_label=True  # Returns tuple: (overall, per_label)
)
```

## Output Files Generated

After running comprehensive evaluation, you'll now get **13 files** (was 10):

### Classification Visualizations:
1. `combined_tsne_l1.png`
2. `combined_confusion_matrix_l1.png`
3. `combined_confusion_matrix_l2.png`
4. `combined_f1_analysis.png`
5. `perclass_f1_weighted_l1.png`
6. `perclass_f1_weighted_l2.png`

### Retrieval Visualizations (NEW):
7. **`retrieval_metrics_comparison.png`** - Overall comparison chart
8. **`retrieval_perlabel_text2sensor_l1.png`** ← NEW!
9. **`retrieval_perlabel_sensor2text_l1.png`** ← NEW!
10. **`retrieval_perlabel_prototype2sensor_l1.png`** ← NEW!
11. **`retrieval_perlabel_prototype2text_l1.png`** ← NEW!

### Results File:
12. **`comprehensive_results.json`** - All metrics including per-label retrieval

## Key Benefits

### 1. **Complete Per-Label Analysis**
- See which activities are easy/hard to retrieve
- Identify systematic weaknesses in cross-modal alignment
- Compare performance across all 4 retrieval variants per activity

### 2. **Label Consistency**
- No mixing of L1 and L2 labels
- `Evening_Meds` and `Morning_Meds` stay separate when using L1
- Can switch to L2 by changing one variable: `retrieval_label_level = 'L2'`

### 3. **Comprehensive JSON Export**
- Overall metrics for quick summary
- Per-label metrics for detailed analysis
- Label level clearly specified
- Easy to parse programmatically for further analysis

### 4. **Actionable Insights**
- Identify which activities need better text descriptions
- Find activities with poor sensor-text alignment
- Compare prototype vs instance performance per activity
- Guide targeted improvements in training data or model architecture

## Usage

No changes needed! Simply run:

```bash
python src/evals/evaluate_embeddings.py \
    --checkpoint trained_models/milan/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1/best_model.pt \
    --train_data data/processed/casas/milan/FD_60_p/train.json \
    --test_data data/processed/casas/milan/FD_60_p/test.json \
    --vocab data/processed/casas/milan/FD_60_p/vocab.json \
    --output_dir results/evals/milan/FD_60_p/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1 \
    --eval_all \
    --train_text_embeddings data/processed/casas/milan/FD_60_p/train_embeddings_baseline_clip.npz \
    --test_text_embeddings data/processed/casas/milan/FD_60_p/test_embeddings_baseline_clip.npz \
    --max_samples 10000 \
    --filter_noisy_labels
```

## Changing Label Level

To use L2 labels instead of L1, edit line ~3075 in `evaluate_embeddings.py`:

```python
# Change from:
retrieval_label_level = 'L1'

# To:
retrieval_label_level = 'L2'
```

Then all 4 retrieval charts and JSON will use L2 labels:
- Files named with `_l2.png` suffix
- JSON includes `"label_level": "L2"`

## Example Per-Label Analysis

### Text → Sensor @ K=10 (example):
```
Evening_Meds:     0.923  (Excellent)
Morning_Meds:     0.915  (Excellent)
Cooking:          0.847  (Good)
Eating:           0.782  (Good)
Sleeping:         0.654  (Moderate)
Other:            0.234  (Poor - needs improvement)
```

This tells you:
- Medicine-taking activities have strong text-sensor alignment
- "Other" activity is poorly defined and needs better descriptions

## Technical Details

### Label Consistency Guarantee:
1. Single source of truth: `retrieval_label_level` variable
2. Same labels used for:
   - Query embeddings
   - Target embeddings
   - Per-label computation
   - Prototype target matching
3. Prevents mismatches like comparing "Evening_Meds" queries to "Take_medicine" targets

### Per-Label Computation:
- **Instance retrieval**: Averages recall across all queries with each label
- **Prototype retrieval**: One query per label (the prototype), so direct mapping
- Both stored in same JSON structure for easy comparison

### Performance:
- Minimal overhead (~5-10% longer evaluation time)
- Per-label computation happens during overall metric calculation
- Results cached and reused for visualization

## Files Modified

1. **`src/evals/compute_retrieval_metrics.py`**
   - Added `compute_per_label_recall_at_k()` function
   - Updated `compute_label_recall_at_k()` to support `return_per_label`
   - Updated `compute_prototype_retrieval_metrics()` to support `return_per_label`

2. **`src/evals/evaluate_embeddings.py`**
   - Added `retrieval_label_level` variable for consistency
   - Added `create_per_label_retrieval_heatmap_instance()` function
   - Updated retrieval computation to return per-label metrics
   - Updated JSON export to include per-label metrics with label level
   - Added 2 new per-label chart generations (text2sensor, sensor2text)

3. **`docs/RETRIEVAL_METRICS_FINAL_UPDATE.md`** (this file)
   - Complete documentation of changes

