# Multiple Prototype Text Encoder Evaluation

## Overview

Enhanced the text encoder evaluation script to support multiple prototypes per activity label using k-NN with majority voting.

## Key Changes

### 1. L2 Labels Derived from L1 Predictions

**Before**: L2 prototypes were created separately from L2 label descriptions.

**After**:
- Only L1 prototypes are created
- L1 predictions are made first
- L2 predictions are derived by mapping L1 predictions using the `label_l2` field in metadata
- This ensures consistency since L2 labels are combinations of L1 labels (e.g., "Watch_TV" and "Read" → "Relax")

### 2. Multiple Prototypes Per Label

**New Function**: `load_multiple_descriptions_from_metadata()`
- Loads descriptions from metadata's `multiple_desc` field
- **Only** includes labels that have `multiple_desc` (no fallback)
- Prints statistics: how many labels have `multiple_desc` and lists them
- Raises error if fewer than 5 labels have `multiple_desc`

**Example**: If "Watch_TV" has 8 descriptions in `multiple_desc`, all 8 are encoded as separate prototypes.

### 3. k-NN with Majority Voting

**Updated Function**: `predict_labels_knn()`
- When `use_multiple_prototypes=True`:
  - Flattens all prototypes (e.g., 10 labels × 8 prototypes = 80 total prototypes)
  - Tracks which label each prototype belongs to
  - Finds k nearest prototypes for each data point
  - Uses majority voting to determine final label

**Example with k=5**:
- Find 5 nearest prototypes: [Watch_TV, Watch_TV, Read, Watch_TV, Sleep]
- Majority vote: Watch_TV (3 votes) → prediction = "Watch_TV"

### 4. Backward Compatibility

**Two Modes**:
1. **Single prototype mode** (default, original behavior):
   - `--use_multiple_prototypes` NOT specified
   - Uses existing `convert_labels_to_text()` function
   - Averages all descriptions into one prototype per label

2. **Multiple prototype mode** (new):
   - `--use_multiple_prototypes` flag enabled
   - Uses `multiple_desc` from metadata
   - k-NN with majority voting

### 5. Evaluation Filtering

When using multiple prototypes mode:
- Only evaluates samples whose L1 labels have prototypes
- Prints warning if some samples are filtered out
- Ensures fair evaluation (only comparing against labels we have prototypes for)

## Usage

### Original Mode (Single Prototype)

```bash
python src/evals/evaluate_text_encoder_only.py \
  --embeddings_dir data/processed/casas/milan/FD_60 \
  --captions data/processed/casas/milan/FD_60/train_captions_baseline.json \
  --data data/processed/casas/milan/FD_60/train.json \
  --output_dir results/evals/milan/FD_60 \
  --split train \
  --description_style baseline \
  --max_samples 10000 \
  --filter_noisy_labels \
  --k_neighbors 1
```

### New Mode (Multiple Prototypes with k-NN Voting)

```bash
python src/evals/evaluate_text_encoder_only.py \
  --embeddings_dir data/processed/casas/milan/FD_60 \
  --captions data/processed/casas/milan/FD_60/train_captions_baseline.json \
  --data data/processed/casas/milan/FD_60/train.json \
  --output_dir results/evals/milan/FD_60 \
  --split train \
  --description_style baseline \
  --max_samples 10000 \
  --filter_noisy_labels \
  --use_multiple_prototypes \
  --k_neighbors 5
```

**Key Parameters**:
- `--use_multiple_prototypes`: Enable multiple prototypes mode
- `--k_neighbors`: Number of nearest prototypes for voting (recommend 5 for multiple prototypes)

## Output Organization

Results are saved in subdirectories based on configuration:
- `text_only/baseline/` - Single prototype, default
- `text_only/baseline_multiproto_k5/` - Multiple prototypes with k=5
- `text_only/baseline_averaged_multiproto_k5/` - Multiple prototypes + averaged captions

## Metadata Requirements

For multiple prototypes mode to work, labels must have `multiple_desc` field in `casas_metadata.json`:

```json
"label_to_text": {
  "Watch_TV": {
    "short_desc": "watching television",
    "long_desc": ["Watch TV activity..."],
    "multiple_desc": [
      "Sitting in the TV room watching television for an extended period.",
      "Stationary presence on a couch or chair in front of the TV with minor movement.",
      "Evening leisure in the TV room, staying mostly in one seat to watch a show.",
      ...
    ]
  }
}
```

**Requirements**:
- At least 5 labels must have `multiple_desc` field
- Script will error if fewer than 5 labels have it
- Script will list which labels have/don't have `multiple_desc`

## Implementation Details

### Prototype Creation Flow

1. Load unique L1 labels from data
2. If `use_multiple_prototypes=True`:
   - Load only labels with `multiple_desc` from metadata
   - Verify at least 5 labels found
   - Encode each description separately (no averaging)
3. If False (default):
   - Use existing `convert_labels_to_text()`
   - Average descriptions per label

### Prediction Flow

1. Create flattened prototype array with label tracking
2. Normalize embeddings and prototypes
3. Compute cosine similarities
4. For each query:
   - Find k nearest prototypes
   - Majority vote on their labels
   - Assign winning label as prediction
5. Map L1 predictions → L2 using `label_l2` metadata

### Evaluation Flow

1. Filter samples to only those with L1 labels that have prototypes
2. Compute metrics (accuracy, F1, etc.) on filtered samples
3. Create confusion matrices for L1 and L2
4. Generate t-SNE visualizations
5. Create comparison plots across encoders

## Benefits

1. **Richer representations**: Multiple prototypes capture label diversity
2. **Better generalization**: k-NN voting reduces impact of outlier prototypes
3. **Consistent L2 labels**: Derived from L1 ensures proper label hierarchy
4. **Flexible k**: Can tune k parameter for optimal performance
5. **Backward compatible**: Original behavior preserved when flag not used

## Example Output

```
📊 Multiple descriptions availability for milan:
   ✅ Labels WITH multiple_desc: 12
   Labels: Kitchen_Activity, Sleep, Read, Watch_TV, Master_Bedroom_Activity, ...
      - Kitchen_Activity: 8 descriptions
      - Sleep: 8 descriptions
      - Read: 8 descriptions
      ...
   ❌ Labels WITHOUT multiple_desc: 3
   Labels: Meditate, no_activity, Other

📝 Encoding 12 unique labels with clip encoder...
   📋 Using 12 labels with multiple_desc field
✅ Created 96 text-based prototypes across 12 labels (512-dim)
    Kitchen_Activity: 450 samples → 8 prototypes
    Sleep: 380 samples → 8 prototypes
    ...

🔄 Predicting labels using 5-NN comparison...
   Total prototypes: 96 across 12 labels
   ✅ All 1000 samples have L1 labels with multiple_desc prototypes

✅ L1 Metrics:
    Accuracy: 0.8750
    F1 (Macro): 0.8456
    F1 (Weighted): 0.8734
```

