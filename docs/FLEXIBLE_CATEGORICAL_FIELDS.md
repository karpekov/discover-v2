# Flexible Categorical Fields Configuration

## Overview

You can now train models with **any subset of categorical fields**, including models without sensor tokens. This enables ablation studies, privacy-preserving models, and domain adaptation experiments.

## Key Feature

**The vocab file can contain ALL fields** (sensor, state, room_id, etc.), but training only uses the fields you specify in `categorical_fields`. No need for multiple vocab files!

## Problem (Before)

Previously, removing `sensor` from `categorical_fields` would cause the training loop to break because:
- `AlignmentDataset._load_sensor_data()` had hardcoded sensor filtering logic
- The code assumed `sensor` field always existed in the vocabulary
- The dataset would try to use ALL fields from the vocab file

## Solution (Now)

The training pipeline now **automatically filters** to only use the fields you specify:
- ✅ Vocab file contains all fields (generated once during sampling)
- ✅ Config specifies which fields to use via `categorical_fields`
- ✅ Training automatically uses only the specified fields
- ✅ Sensor filtering only runs if `sensor` is in `categorical_fields`
- ✅ MLM span masker adapts to available fields
- ✅ No code changes needed - just modify your config!

## Usage

### Training Without Sensor Tokens

**Simply modify your alignment config - that's it!**

```yaml
# configs/alignment/milan_no_sensors.yaml

# Use the SAME vocab file as always
vocab_path: data/processed/casas/milan/FD_60/vocab.json

encoder:
  metadata:
    categorical_fields:
      - state        # Event type (ON/OFF/OPEN/CLOSE)
      - room_id      # Room location
      # sensor field removed - even though it's in vocab.json!
    use_coordinates: true
    use_time_deltas: false
    use_time_of_day: true

experiment_name: milan_fd60_nosensor
output_dir: trained_models/milan/milan_fd60_nosensor

# ... rest of config
```

**Train normally:**

```bash
python train.py --config configs/alignment/milan_no_sensors.yaml
```

That's it! The vocab file still contains `sensor`, but training will only use `state` and `room_id`.

### Optional: Custom Vocab Files

If you want to create a minimal vocab file (e.g., for cleaner inspection), you can:

```bash
# Generate vocab with only specific fields
python src/utils/generate_vocab_from_data.py \
    --data data/processed/casas/milan/FD_60/train.json \
    --output data/processed/casas/milan/FD_60/vocab_state_room.json \
    --include state room_id
```

But this is **optional** - the regular vocab.json works fine!

## Common Use Cases

### 1. Ablation Study: Which features matter most?

```yaml
# Test 1: State only
categorical_fields: [state]

# Test 2: State + room
categorical_fields: [state, room_id]

# Test 3: All features
categorical_fields: [sensor, state, room_id]
```

### 2. Privacy-Preserving Models

Train without sensor IDs to prevent learning sensor-specific patterns:

```yaml
categorical_fields: [state, room_id]  # No sensor IDs
use_coordinates: false                # No location coordinates
```

### 3. Domain Adaptation

Train a model that can transfer across homes with different sensor layouts:

```yaml
categorical_fields: [state]  # Only event types (ON/OFF/OPEN/CLOSE)
# No sensor IDs, no room IDs, no coordinates
# Pure temporal pattern learning
```

### 4. Minimal Feature Set

Train the simplest possible model:

```yaml
categorical_fields: [state]
use_coordinates: false
use_time_deltas: false
use_time_of_day: false
```

## What Changed (Technical Details)

### Modified Files

**1. `src/alignment/dataset.py`:**

Added `categorical_fields` parameter and field filtering:

```python
def __init__(self, ..., categorical_fields=None):
    self.full_vocab = vocab  # Keep full vocab for reference

    # Filter vocab to only used fields
    if categorical_fields is not None:
        self.vocab = {field: vocab[field] for field in categorical_fields if field in vocab}
    else:
        self.vocab = vocab  # Use all fields
```

Sensor filtering now checks filtered vocab:

```python
def _load_sensor_data(self):
    # Only filter by sensors if 'sensor' is in categorical_fields
    should_filter_sensors = 'sensor' in self.vocab
    ...
```

**2. `src/alignment/trainer.py`:**

Pass categorical_fields to datasets:

```python
categorical_fields = encoder_config.get('metadata', {}).get('categorical_fields', [])

train_dataset = AlignmentDataset(
    ...,
    vocab=vocab,  # Full vocab (all fields)
    categorical_fields=categorical_fields  # Only use these
)
```

MLM span masker now adapts to active fields:

```python
def _setup_span_masker(self):
    # Get active fields from config
    categorical_fields = set(encoder_config.get('metadata', {}).get('categorical_fields', []))

    # Filter field_priors to only active fields
    field_priors = {f: p for f, p in all_priors.items() if f in categorical_fields}

    # Only correlate fields if both are active
    if 'sensor' in categorical_fields and 'room_id' in categorical_fields:
        correlated_groups.append(['sensor', 'room_id'])
```

**3. `src/utils/generate_vocab_from_data.py`:**

Added `--include` and `--exclude` flags (optional - for creating minimal vocab files):

```bash
python src/utils/generate_vocab_from_data.py --data train.json --exclude sensor
```

### Already Flexible (No changes needed)

**`src/encoders/sensor/sequence/transformer.py`:**

The encoder already iterates over categorical fields dynamically - just works!

## Verification

To verify your config will work, check:

1. ✅ All fields in `categorical_fields` exist in the vocab file
2. ✅ Vocab can contain MORE fields than `categorical_fields` (extra fields are ignored)
3. ✅ All samples in your data have the required fields

**Quick check:**

```bash
# View what fields are in your vocabulary
python -c "import json; vocab = json.load(open('data/processed/casas/milan/FD_60/vocab.json')); print('Vocab fields:', list(vocab.keys()))"

# View what fields are in your config
python -c "import yaml; cfg = yaml.safe_load(open('configs/alignment/your_config.yaml')); print('Config fields:', cfg['encoder']['metadata']['categorical_fields'])"
```

**Verification script:**

```python
import json
import yaml

# Load config
with open('configs/alignment/your_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load vocab
with open(config['vocab_path'], 'r') as f:
    vocab = json.load(f)

# Check alignment
cat_fields = set(config['encoder']['metadata']['categorical_fields'])
vocab_fields = set(vocab.keys())

print(f"Config fields: {sorted(cat_fields)}")
print(f"Vocab fields: {sorted(vocab_fields)}")

# Check if all config fields are in vocab
missing = cat_fields - vocab_fields
extra = vocab_fields - cat_fields

if missing:
    print(f"\n❌ ERROR: Fields in config but not in vocab: {missing}")
    print("   → You need to regenerate vocab or fix config")
else:
    print("\n✅ All config fields found in vocab!")

if extra:
    print(f"ℹ️  Extra fields in vocab (will be ignored): {extra}")
    print("   → This is OK! Vocab can have more fields than needed.")
```

## Examples in Configs

See updated config:
- `configs/alignment/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_nocoords_v1.yaml`

This config already uses flexible categorical fields:

```yaml
encoder:
  metadata:
    categorical_fields:
    - sensor
    - state
    - room_id
    use_coordinates: false  # Spatial features disabled
    use_time_deltas: false  # Temporal features disabled
    use_time_of_day: true   # Cyclical time enabled
```

To train without sensor, simply remove `sensor` from the list and regenerate the vocab!

## Summary

### Before
❌ Removing `sensor` from `categorical_fields` → training crashes
❌ Vocab file must match config fields exactly
❌ Need multiple vocab files for different field combinations

### Now
✅ Single vocab file contains all fields (generated once)
✅ Config specifies which fields to use via `categorical_fields`
✅ Training automatically filters to only use specified fields
✅ No need to regenerate vocab for different experiments
✅ Removing `sensor` from `categorical_fields` → training works seamlessly

**The key insight:** Vocab file is just a reference dictionary. Training uses only what you specify in the config!

