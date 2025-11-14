# Sampling Updates Summary

## Changes Made

### 1. Automatic Vocabulary Generation
- Added `_generate_vocabulary()` method to `BaseSampler` class
- Vocabulary is now automatically generated from training data during sampling
- Creates index mappings for all categorical fields (sensor, state, room, activity, etc.)
- Index 0 reserved for UNK (unknown/padding) token
- Saves `vocab.json` alongside train/val/test splits

### 2. Three-Way Data Splits (70/10/20)
- Updated `_split_by_days()` to return **three** splits instead of two
- **70% Training** - Used for model training
- **10% Validation** - Used for hyperparameter tuning and early stopping
- **20% Test** - Reserved for final evaluation only
- All splits preserve temporal or random day-based splitting
- Generates `train.json`, `val.json`, and `test.json`

### 3. Updated Sampling Pipeline
The sampling pipeline now outputs:
```
data/processed/<dataset>/<sampling_strategy>/
├── train.json          # 70% of data
├── val.json            # 10% of data
├── test.json           # 20% of data
├── vocab.json          # Generated from train.json
├── sampling_config.yaml # Configuration used
└── statistics.json     # Detailed stats for all splits
```

## Usage

### Running Sampling
```bash
# Standard sampling (now with vocab generation)
python sample_data.py --config configs/sampling/milan_fixed_length_20.yaml

# Debug mode (limited data)
python sample_data.py --config configs/sampling/milan_fixed_length_20.yaml --debug
```

### Generating Vocab from Existing Data
If you need to generate vocab for existing sampled data:
```bash
python scripts/generate_vocab_from_data.py --data data/processed/casas/milan/fixed_length_20/train.json
```

## Benefits

1. **No Manual Steps**: Vocabulary generation is automatic - no separate script needed
2. **Proper Validation**: 10% validation split for better hyperparameter tuning
3. **Consistent Splits**: Same splitting logic across all sampling strategies
4. **Clean Pipeline**: Single command generates everything needed for training

## Migration Notes

**Existing Data**:
- Old datasets only have `train.json` and `test.json`
- To use them with the new training pipeline, either:
  1. Re-run sampling to get val.json and vocab.json
  2. Manually generate vocab using the script above
  3. Use test.json as val.json temporarily (not recommended for final evaluation)

**Configuration**:
- No changes needed to sampling configs
- `train_ratio` parameter is now ignored (fixed at 70/10/20)
- Split strategy (temporal/random) still works as before

