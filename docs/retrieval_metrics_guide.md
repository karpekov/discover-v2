# Label Score@K Retrieval Metrics Guide

## Overview

The `compute_retrieval_metrics.py` script computes Label-Precision/Score@K metrics for cross-modal retrieval between text and sensor embeddings, plus an optional sensor → sensor sanity check. These metrics measure how well embeddings from one modality can retrieve relevant examples from another modality (or itself) based on label similarity.

## What is Label-Recall@K?

For each query embedding:
1. Compute similarity to all target embeddings
2. Rank targets by similarity (descending)
3. Take the top K targets
4. Count how many of those K targets have the same label as the query
5. Divide by K to get the recall ratio

The final metric is the average across all queries.

**Example:** If a query has label "Cooking" and we retrieve K=10 neighbors, and 7 of them also have label "Cooking", then Label-Recall@10 = 7/10 = 0.7 (70%).

## Usage

### Standalone CLI

#### Instance-to-Instance Retrieval (Default)

By default the `directions` parameter includes `text2sensor` and `sensor2text`. You can add `sensor2sensor` and/or `text2text` to compute self-retrieval sanity checks where each embedding dresses its nearest neighbors without returning itself.

```bash
# Basic usage with both retrieval directions
python src/evals/compute_retrieval_metrics.py \
    --sensor_embeddings results/evals/sensor_emb.npy \
    --text_embeddings results/evals/text_emb.npy \
    --labels results/evals/labels.npy \
    --k_values 10 50 100 \
    --directions text2sensor sensor2text \
    --normalize

# Add sensor/text self-checks
python src/evals/compute_retrieval_metrics.py \
    --sensor_embeddings results/evals/sensor_emb.npy \
    --text_embeddings results/evals/text_emb.npy \
    --labels results/evals/labels.npy \
    --k_values 10 50 \
    --directions text2sensor sensor2text sensor2sensor text2text \
    --normalize

# Text-to-sensor retrieval only
python src/evals/compute_retrieval_metrics.py \
    --sensor_embeddings sensor_emb.npy \
    --text_embeddings text_emb.npy \
    --labels labels.npy \
    --k_values 10 50 \
    --directions text2sensor \
    --output results/retrieval_metrics.json
```

#### Prototype-Based Retrieval (New!)

Use text prototypes (label descriptions from metadata) as queries:

```bash
# Text prototypes -> sensor embeddings & text embeddings
python src/evals/compute_retrieval_metrics.py \
    --sensor_embeddings results/sensor_emb.npy \
    --text_embeddings results/text_emb.npy \
    --labels results/labels.npy \
    --use_prototypes \
    --metadata metadata/casas_metadata.json \
    --dataset_name milan \
    --caption_style sourish \
    --checkpoint trained_models/best_model.pt \
    --k_values 10 50 100 \
    --output results/prototype_retrieval.json

# Text prototypes -> sensor embeddings only
python src/evals/compute_retrieval_metrics.py \
    --sensor_embeddings results/sensor_emb.npy \
    --labels results/labels.npy \
    --use_prototypes \
    --metadata metadata/casas_metadata.json \
    --dataset_name milan \
    --checkpoint trained_models/best_model.pt \
    --k_values 10 50 100 \
    --directions prototype2sensor
```

#### With .npz Files

```bash
python src/evals/compute_retrieval_metrics.py \
    --sensor_embeddings data.npz --sensor_key sensor_embeddings \
    --text_embeddings data.npz --text_key text_embeddings \
    --labels data.npz --labels_key labels \
    --k_values 10 50 100
```

### As a Library

#### Instance-to-Instance Retrieval

```python
import numpy as np
from evals.compute_retrieval_metrics import (
    compute_label_recall_at_k,
    print_results_summary
)

# Load your data
sensor_embeddings = np.load('sensor_emb.npy')  # Shape: (N, D_sensor)
text_embeddings = np.load('text_emb.npy')      # Shape: (N, D_text)
labels = np.load('labels.npy')                 # Shape: (N,)

# Compute retrieval metrics
results = compute_label_recall_at_k(
    sensor_embeddings=sensor_embeddings,
    text_embeddings=text_embeddings,
    labels=labels,
    k_values=[10, 50, 100],
    directions=['text2sensor', 'sensor2text'],
    normalize=True,  # L2-normalize before computing similarity
    verbose=True
)

# Print formatted results
print_results_summary(results)

# To include sensor/text self-checks, extend `directions`:
results_self = compute_label_recall_at_k(
    sensor_embeddings=sensor_embeddings,
    text_embeddings=text_embeddings,
    labels=labels,
    k_values=[10, 50, 100],
    directions=['text2sensor', 'sensor2text', 'text2text', 'sensor2sensor'],
    normalize=True,
    verbose=True
:)
# Access results programmatically
text2sensor_recall_at_10 = results['text2sensor'][10]
sensor2text_recall_at_50 = results['sensor2text'][50]
```

#### Prototype-Based Retrieval

```python
import numpy as np
import torch
from evals.compute_retrieval_metrics import (
    load_text_prototypes_from_metadata,
    encode_text_prototypes,
    compute_prototype_retrieval_metrics,
    print_results_summary
)
from evals.eval_utils import create_text_encoder_from_checkpoint
from utils.device_utils import get_optimal_device

# Load text prototypes from metadata
label_to_text = load_text_prototypes_from_metadata(
    metadata_path='metadata/casas_metadata.json',
    dataset_name='milan',
    style='sourish'
)

# Load text encoder from checkpoint
device = get_optimal_device()
checkpoint = torch.load('trained_models/best_model.pt', map_location=device, weights_only=False)
text_encoder = create_text_encoder_from_checkpoint(checkpoint, device)

# Encode text prototypes
prototype_emb, prototype_labels = encode_text_prototypes(
    label_to_text=label_to_text,
    text_encoder=text_encoder,
    device=str(device),
    normalize=True
)

# Load target embeddings
sensor_embeddings = np.load('sensor_emb.npy')
text_embeddings = np.load('text_emb.npy')
target_labels = np.load('labels.npy')  # Should be string labels matching prototype_labels

# Compute prototype retrieval metrics
results = compute_prototype_retrieval_metrics(
    prototype_embeddings=prototype_emb,
    prototype_labels=prototype_labels,
    sensor_embeddings=sensor_embeddings,
    text_embeddings=text_embeddings,
    target_labels=target_labels,
    k_values=[10, 50, 100],
    directions=['prototype2sensor', 'prototype2text'],
    normalize=True,
    verbose=True
)

print_results_summary(results)

# Access results
proto2sensor_recall_at_10 = results['prototype2sensor'][10]
proto2text_recall_at_50 = results['prototype2text'][50]
```

## Integration with evaluate_embeddings.py

The script is designed to be easily integrated into `evaluate_embeddings.py`. Here are examples for both modes:

### Instance-to-Instance Retrieval

```python
# In evaluate_embeddings.py, after extracting embeddings:

from evals.compute_retrieval_metrics import compute_label_recall_at_k, print_results_summary

# Extract embeddings for train/test splits
train_sensor_emb, train_labels_l1, train_labels_l2 = self.extract_embeddings_and_labels('train')
test_sensor_emb, test_labels_l1, test_labels_l2 = self.extract_embeddings_and_labels('test')

# Get text embeddings (using text encoder)
train_text_labels = convert_labels_to_text(train_labels_l1, train_labels_l2)
test_text_labels = convert_labels_to_text(test_labels_l1, test_labels_l2)

train_text_emb = self.text_encoder.encode_texts_clip(train_text_labels, self.device)
test_text_emb = self.text_encoder.encode_texts_clip(test_text_labels, self.device)

train_text_emb = train_text_emb.cpu().numpy()
test_text_emb = test_text_emb.cpu().numpy()

# Compute retrieval metrics on test set
retrieval_results = compute_label_recall_at_k(
    sensor_embeddings=test_sensor_emb,
    text_embeddings=test_text_emb,
    labels=np.array(test_labels_l1),  # or test_labels_l2 for L2 labels
    k_values=[10, 50, 100],
    directions=['text2sensor', 'sensor2text'],
    normalize=True,
    verbose=True
)

print_results_summary(retrieval_results)

# Save to output directory
import json
output_path = Path(self.config['output_dir']) / 'retrieval_metrics.json'
with open(output_path, 'w') as f:
    json.dump({
        direction: {str(k): float(v) for k, v in k_results.items()}
        for direction, k_results in retrieval_results.items()
    }, f, indent=2)
```

### Prototype-Based Retrieval

```python
# In evaluate_embeddings.py:

from evals.compute_retrieval_metrics import (
    load_text_prototypes_from_metadata,
    encode_text_prototypes,
    compute_prototype_retrieval_metrics,
    print_results_summary
)

# Extract test embeddings
test_sensor_emb, test_labels_l1, test_labels_l2 = self.extract_embeddings_and_labels('test')

# Load and encode text prototypes
label_to_text = load_text_prototypes_from_metadata(
    metadata_path='metadata/casas_metadata.json',
    dataset_name='milan',
    style='sourish'
)

prototype_emb, prototype_labels = encode_text_prototypes(
    label_to_text=label_to_text,
    text_encoder=self.text_encoder,
    device=str(self.device),
    normalize=True
)

# Optionally encode test text embeddings too
test_text_labels = convert_labels_to_text(test_labels_l1, test_labels_l2)
test_text_emb = self.text_encoder.encode_texts_clip(test_text_labels, self.device)
test_text_emb = F.normalize(test_text_emb, p=2, dim=-1).cpu().numpy()

# Compute prototype retrieval metrics
prototype_results = compute_prototype_retrieval_metrics(
    prototype_embeddings=prototype_emb,
    prototype_labels=prototype_labels,
    sensor_embeddings=test_sensor_emb,
    text_embeddings=test_text_emb,
    target_labels=np.array(test_labels_l1),  # Must be string labels
    k_values=[10, 50, 100],
    directions=['prototype2sensor', 'prototype2text'],
    normalize=True,
    verbose=True
)

print_results_summary(prototype_results)

# Save results
output_path = Path(self.config['output_dir']) / 'prototype_retrieval_metrics.json'
save_results(prototype_results, str(output_path))
```

## Output Format

### Console Output

#### Instance-to-Instance

```
======================================================================
LABEL-RECALL@K RESULTS SUMMARY
======================================================================

Text -> Sensor:
----------------------------------------
  K= 10  =>  Label-Recall@K = 0.7234 (72.34%)
  K= 50  =>  Label-Recall@K = 0.6891 (68.91%)
  K=100  =>  Label-Recall@K = 0.6523 (65.23%)

Sensor -> Text:
----------------------------------------
  K= 10  =>  Label-Recall@K = 0.7156 (71.56%)
  K= 50  =>  Label-Recall@K = 0.6812 (68.12%)
  K=100  =>  Label-Recall@K = 0.6478 (64.78%)

======================================================================
```

#### Prototype-Based

```
======================================================================
LABEL-RECALL@K RESULTS SUMMARY
======================================================================

Text Prototype -> Sensor:
----------------------------------------
  K= 10  =>  Label-Recall@K = 0.8345 (83.45%)
  K= 50  =>  Label-Recall@K = 0.7923 (79.23%)
  K=100  =>  Label-Recall@K = 0.7634 (76.34%)

Text Prototype -> Text:
----------------------------------------
  K= 10  =>  Label-Recall@K = 0.8123 (81.23%)
  K= 50  =>  Label-Recall@K = 0.7845 (78.45%)
  K=100  =>  Label-Recall@K = 0.7567 (75.67%)

======================================================================
```

### JSON Output

#### Instance-to-Instance

```json
{
  "text2sensor": {
    "10": 0.7234,
    "50": 0.6891,
    "100": 0.6523
  },
  "sensor2text": {
    "10": 0.7156,
    "50": 0.6812,
    "100": 0.6478
  }
}
```

#### Prototype-Based

```json
{
  "prototype2sensor": {
    "10": 0.8345,
    "50": 0.7923,
    "100": 0.7634
  },
  "prototype2text": {
    "10": 0.8123,
    "50": 0.7845,
    "100": 0.7567
  }
}
```

## Understanding the Results

### Good Performance Indicators
- **High recall (>0.6)**: Embeddings capture semantic similarity well
- **Stable across K values**: Consistent retrieval quality
- **Balanced directions**: Both text->sensor and sensor->text perform similarly
- **Prototype recall > instance recall**: Model generalizes well to unseen descriptions

### Poor Performance Indicators
- **Low recall (<0.3)**: Embeddings may not capture semantic relationships
- **Large drop with increasing K**: Model struggles with ranking
- **Imbalanced directions**: One modality may dominate the embedding space
- **Prototype recall << instance recall**: Model overfits to specific phrasings

### Comparison with Classification
- **Classification** measures if the top-1 prediction is exactly correct
- **Label-Recall@K** measures if the label appears in top-K (more lenient)
- Useful for understanding embedding quality beyond classification

### Prototype vs Instance Retrieval
- **Instance retrieval**: Queries and targets are both actual examples (aligned)
- **Prototype retrieval**: Queries are label descriptions, targets are examples
- **Use case for prototypes**: Test zero-shot understanding of label semantics
- **Expected**: Prototype recall often higher (label descriptions are more "canonical")

## Tips

1. **Always normalize**: Set `normalize=True` to ensure fair cosine similarity
2. **Choose appropriate K values**: Start with 10, 50, 100 for balanced view
3. **Compare with random baseline**: Random chance = (# examples per class) / (total examples)
4. **Use on test set**: Evaluate on held-out data to avoid overfitting
5. **Consider label hierarchy**: Evaluate on both L1 and L2 labels separately
6. **Use prototypes for zero-shot eval**: Prototypes test if model understands label semantics
7. **Compare both modes**: Instance vs prototype retrieval gives complementary insights
8. **Check label format**: For prototypes, ensure target labels are strings matching prototype labels

## Implementation Details

- Uses **cosine similarity** after L2-normalization
- Efficient **numpy matrix operations** for similarity computation
- Two retrieval modes:
  - **Instance-to-instance**: Aligned embeddings (i-th sensor, i-th text, i-th label)
  - **Prototype-based**: One text prototype per label → many target embeddings
- Supports **multi-label evaluation** through separate runs per label level
- Loads text prototypes from **metadata JSON** (label_to_text_* fields)
- Uses checkpoint's **text encoder** to encode prototypes on-the-fly

