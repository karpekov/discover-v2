# Retrieval Metrics Integration Summary

## Overview

Successfully integrated cross-modal retrieval metrics into `evaluate_embeddings.py`. The comprehensive evaluation now includes both classification metrics and retrieval metrics across all 4 retrieval variants.

## What Was Added

### 1. Imports
Added retrieval metrics imports at the top of `evaluate_embeddings.py`:
```python
from evals.compute_retrieval_metrics import (
    compute_label_recall_at_k,
    load_text_prototypes_from_metadata,
    encode_text_prototypes,
    compute_prototype_retrieval_metrics
)
```

### 2. New Visualization Functions

- **Left plot**: Line chart showing Label Score@K (instance recall + prototype precision) vs K for all 6 directions (text2sensor, sensor2text, text2text, sensor2sensor, prototype2sensor, prototype2text)
- **Right plot**: Bar chart comparing all methods at a specific K value

**Output**: `retrieval_metrics_comparison.png`

- Horizontal bar chart showing retrieval precision for each activity label
- Color-coded by performance (red to green)
- Sorted by performance for easy identification of strong/weak labels
- Shows average performance line

**Outputs**:
- `retrieval_perlabel_prototype2sensor.png`
- `retrieval_perlabel_prototype2text.png`

### 3. Integration into `run_comprehensive_evaluation()`

Added **Section 7: COMPUTE RETRIEVAL METRICS** after classification evaluation:

```python
# Instance-to-instance retrieval
instance_retrieval_results = compute_label_recall_at_k(
    sensor_embeddings=test_sensor_emb,
    text_embeddings=test_text_emb_proj,
    labels=np.array(test_labels_l1),
    k_values=[10, 50, 100],
    directions=['text2sensor', 'sensor2text', 'sensor2sensor'],
    normalize=True,
    exclude_self=True
)

# Prototype-based retrieval
label_to_text = load_text_prototypes_from_metadata(...)
prototype_emb, prototype_labels = encode_text_prototypes(...)
prototype_retrieval_results = compute_prototype_retrieval_metrics(
    prototype_embeddings=prototype_emb,
    prototype_labels=prototype_labels,
    sensor_embeddings=test_sensor_emb,
    text_embeddings=test_text_emb_proj,
    target_labels=np.array(test_labels_l1),
    k_values=[10, 50, 100],
    directions=['prototype2sensor', 'prototype2text']
)
```

### 4. Updated `comprehensive_results.json`

The results file now has two main sections:

```json
{
  "classification_metrics": {
    "text_noproj": { "l1": {...}, "l2": {...} },
    "text_proj": { "l1": {...}, "l2": {...} },
    "sensor": { "l1": {...}, "l2": {...} }
  },
  "retrieval_metrics": {
    "instance_to_instance": {
      "text2sensor": { "10": 0.XX, "50": 0.XX, "100": 0.XX },
      "sensor2text": { "10": 0.XX, "50": 0.XX, "100": 0.XX },
      "text2text": { "10": 0.XX, "50": 0.XX, "100": 0.XX },
      "sensor2sensor": { "10": 0.XX, "50": 0.XX, "100": 0.XX }
    },
    "prototype_based": {
      "prototype2sensor": { "10": 0.XX, "50": 0.XX, "100": 0.XX },
      "prototype2text": { "10": 0.XX, "50": 0.XX, "100": 0.XX }
    }
  }
}
```

### 5. Enhanced Console Output

Added detailed retrieval metrics summary to console output:

```
================================================================================
📊 RETRIEVAL METRICS SUMMARY (Instance Recall@K + Prototype Precision@K)
================================================================================

Label Score @ K=50 (Instance recall, Prototype precision):
Direction                      Score
------------------------------------------
Text → Sensor                  0.7234      (72.34%)
Sensor → Text                  0.7156      (71.56%)
Prototype → Sensor             0.8345      (83.45%)
Prototype → Text               0.8123      (81.23%)

Detailed Results (All K values):

Text → Sensor:
  K= 10  =>  0.7534 (75.34%)
  K= 50  =>  0.7234 (72.34%)
  K=100  =>  0.6923 (69.23%)
...
```
The console summary now includes both the Sensor → Sensor and Text → Text directions so you have self-retrieval checks for each modality.

## Retrieval Variants

In addition to the cross-modal combinations below, the evaluation now also reports self-retrieval sanity checks for both sensor and text embeddings.

### 1. Text → Sensor (Instance-to-Instance)
- **Query**: Text embeddings of activity descriptions
- **Target**: Sensor embeddings
- **Use case**: Can text descriptions find corresponding sensor events?

### 2. Sensor → Text (Instance-to-Instance)
- **Query**: Sensor embeddings
- **Target**: Text embeddings of activity descriptions
- **Use case**: Can sensor events find their text descriptions?

### 3. Text → Text (Instance-to-Instance)
- **Query**: Projected text embeddings
- **Target**: Projected text embeddings (excluding the query itself)
- **Use case**: Self-similarity sanity check to ensure text embeddings are not trivially identical and retrieval changes when the query is removed

### 4. Prototype → Sensor (Prototype-Based)
- **Query**: Text prototypes from metadata (canonical label descriptions)
- **Target**: Sensor embeddings
- **Use case**: Can label definitions retrieve all sensor examples of that activity?

### 5. Prototype → Text (Prototype-Based)
- **Query**: Text prototypes from metadata
- **Target**: Text embeddings of activity descriptions
- **Use case**: Can label definitions retrieve all text descriptions of that activity?

## Generated Visualizations

When running comprehensive evaluation, you'll get these new files:

1. **`retrieval_metrics_comparison.png`**
   - Overall performance comparison
   - Line chart showing trends across K values
   - Bar chart for at-a-glance comparison

2. **`retrieval_perlabel_prototype2sensor.png`**
   - Per-label retrieval performance for Prototype → Sensor
   - Identifies which activities are well/poorly retrieved

3. **`retrieval_perlabel_prototype2text.png`**
   - Per-label retrieval performance for Prototype → Text
   - Shows semantic alignment quality per activity

## Usage

No changes to command-line interface! The retrieval metrics are automatically computed when running comprehensive evaluation:

```bash
python src/evals/evaluate_embeddings.py \
    --checkpoint trained_models/best_model.pt \
    --train_data data/processed/train.json \
    --test_data data/processed/test.json \
    --vocab data/processed/vocab.json \
    --output_dir results/evals/comprehensive \
    --eval_all \
    --train_text_embeddings data/processed/train_embeddings.npz \
    --test_text_embeddings data/processed/test_embeddings.npz \
    --max_samples 10000 \
    --filter_noisy_labels
```

The script will automatically:
1. Run all classification evaluations
2. Compute all 4 retrieval variants
3. Create visualizations
4. Save everything to `comprehensive_results.json`

## Key Features

### Robust Error Handling
If prototype retrieval fails (e.g., metadata not found), the script:
- Continues with instance-to-instance retrieval
- Prints warning message
- Still saves available results

### Automatic Normalization
All embeddings are L2-normalized before computing similarities to ensure fair cosine similarity comparisons.

### Multiple K Values
Evaluates at K=[10, 50, 100] by default to show performance at different retrieval depths.

### Color-Coded Visualizations
- **Blue**: Text → Sensor
- **Orange**: Sensor → Text
- **Green**: Prototype → Sensor
- **Red**: Prototype → Text

## Performance Interpretation

### Good Performance Indicators
- **Recall > 0.6**: Strong cross-modal alignment
- **Prototype recall ≥ instance recall**: Good generalization
- **Stable across K**: Consistent ranking quality
- **Balanced directions**: Symmetric cross-modal understanding

### Areas for Improvement
- **Recall < 0.3**: Weak cross-modal alignment
- **Large drop with increasing K**: Ranking issues
- **Prototype << instance**: Overfitting to specific phrasings
- **Imbalanced directions**: One modality dominates

## Technical Details

- Uses **cosine similarity** after L2-normalization
- Loads text prototypes from `metadata/casas_metadata.json`
- Encodes prototypes using the model's text encoder with projection
- Uses L1 labels for retrieval evaluation
- Efficient numpy operations for similarity computation

## Files Modified

1. **`src/evals/evaluate_embeddings.py`**
   - Added retrieval metrics imports
   - Added 2 new visualization functions
   - Integrated retrieval computation into comprehensive evaluation
   - Updated results structure
   - Enhanced console output

2. **`src/evals/compute_retrieval_metrics.py`** (already created)
   - Core retrieval metrics implementation
   - Supports both instance and prototype modes

3. **`docs/RETRIEVAL_METRICS_INTEGRATION.md`** (this file)
   - Integration documentation

## Benefits

1. **Comprehensive Evaluation**: Classification + Retrieval in one run
2. **Multiple Perspectives**: 4 different ways to assess cross-modal quality
3. **Actionable Insights**: Per-label analysis identifies weak spots
4. **Production Ready**: Robust error handling and clear visualizations
5. **No Extra Work**: Automatic integration with existing workflow

## Future Enhancements

Possible additions:
- Mean Average Precision (MAP) metric
- Precision@K alongside Recall@K
- Cross-validation with different K values
- Retrieval metrics for L2 labels
- Confusion matrices for retrieval errors

