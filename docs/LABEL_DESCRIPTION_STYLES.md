# Label Description Styles

This document describes the different styles of activity label descriptions available in the system and how to use them.

## Overview

The system supports multiple styles of text descriptions for activity labels, which are used when creating text-based prototypes for zero-shot or text-only classification. These descriptions are stored in `metadata/casas_metadata.json` and accessed via `src/utils/label_utils.py`.

## Available Styles

### 1. Baseline Style (Default)

The baseline style provides rich, contextual descriptions of activities with multiple variations per label. These descriptions emphasize sensor patterns, durations, and behavioral context.

**Structure:**
- **Short descriptions**: Brief, human-readable descriptions
- **Long descriptions**: Detailed descriptions with sensor and timing context (multiple variations per label)

**Example (Milan - Sleep):**
- Short: "sleeping in bed in master bedroom, usually at night time"
- Long: ["Prolonged or intermittent bed-area motion in the master bedroom, often at night, reflecting sleep and related nighttime activity."]

**Location in metadata:** `label_to_text` field

### 2. Sourish Style

The Sourish style uses template-based, single-sentence descriptions from the research by Sourish Gunesh Dhekane on zero-shot activity recognition. These descriptions are more formulaic and emphasize duration, location, and specific actions.

**Structure:**
- Single description per label following a consistent template
- Emphasizes: time duration, location, and the action being performed

**Example (Milan - Sleep):**
- "Sleep activity takes place for hours when a person sleeps on bed in the master bedroom"

**Location in metadata:** `label_to_text_sourish` field

## Supported Datasets

The Sourish label descriptions are currently available for:
- **Milan**: All 16 activity labels
- **Aruba**: All 12 activity labels
- **Cairo**: All 14 activity labels

## Usage

### In Code

Use the `convert_labels_to_text()` function with the `description_style` parameter:

```python
from utils.label_utils import convert_labels_to_text

# Baseline style (default)
descriptions = convert_labels_to_text(
    labels=['Sleep', 'Kitchen_Activity'],
    house_name='milan',
    description_style='baseline'
)

# Sourish style
descriptions = convert_labels_to_text(
    labels=['Sleep', 'Kitchen_Activity'],
    house_name='milan',
    description_style='sourish'
)
```

### In Text Encoder Evaluation

The `evaluate_text_encoder_only.py` script accepts the `--description_style` flag:

```bash
# Using baseline descriptions (default)
python src/evals/evaluate_text_encoder_only.py \
    --train_data data/processed/casas/milan/sourish_seq20/milan_train.json \
    --test_data data/processed/casas/milan/sourish_seq20/milan_presegmented_test.json \
    --output_dir results/milan_seq20/textonly_baseline \
    --description_style baseline \
    --filter_noisy_labels

# Using Sourish descriptions
python src/evals/evaluate_text_encoder_only.py \
    --train_data data/processed/casas/milan/sourish_seq20/milan_train.json \
    --test_data data/processed/casas/milan/sourish_seq20/milan_presegmented_test.json \
    --output_dir results/milan_seq20/textonly_sourish \
    --description_style sourish \
    --filter_noisy_labels
```

### Command-line Arguments

- `--description_style`: Choice of `baseline` or `sourish` (default: `baseline`)
- `--house_name`: Dataset name for loading appropriate descriptions (default: `milan`)

## Implementation Details

### label_utils.py

The `convert_labels_to_text()` function:
1. Loads house metadata from `metadata/casas_metadata.json`
2. Selects the appropriate description source based on `description_style`
3. Returns descriptions in the requested format (single or multiple per label)

**Key behavior differences:**
- **Baseline**: Can return multiple description variants per label
- **Sourish**: Always returns a single description per label (wrapped in a list if multiple descriptions are requested)

### evaluate_text_encoder_only.py

The evaluation script:
1. Accepts `--description_style` and `--house_name` arguments
2. Passes these to `create_text_prototypes()` method
3. Uses `convert_labels_to_text()` to get appropriate descriptions
4. Creates embeddings and prototypes based on the selected style

## When to Use Each Style

### Use Baseline Style When:
- You want richer, more varied descriptions
- You're working with diverse sensor patterns
- You need multiple description variations for robustness
- Your model benefits from more contextual information

### Use Sourish Style When:
- You want to compare with Sourish's zero-shot approach
- You prefer deterministic, template-based descriptions
- You're working with the same datasets (Milan, Aruba, Cairo)
- You want descriptions that closely match the research paper format

## Adding New Description Styles

To add a new description style:

1. **Add descriptions to metadata:**
   ```json
   "label_to_text_newstyle": {
     "Activity_Name": "Description text"
   }
   ```

2. **Update `label_utils.py`:**
   - Add new condition in `convert_labels_to_text()` for the new style
   - Handle the description format appropriately

3. **Update evaluation scripts:**
   - Add the new style to `choices` in argument parser
   - Ensure it's passed through to `convert_labels_to_text()`

4. **Document the new style:**
   - Update this file with the new style description
   - Provide usage examples

## References

- Baseline descriptions: Original system design
- Sourish descriptions: From Sourish Gunesh Dhekane's zero-shot activity recognition research
  - Source: `OTHER/sourish_zero_shot/dataset_utils.py`
  - Paper references included in original research code

