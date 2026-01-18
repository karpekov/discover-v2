# Temporal Token Implementation (Time-of-Day & Day-of-Week)

## Summary

This document describes the implementation of temporal bucketing features as categorical tokens in the discover-v2 pipeline:
- **Time-of-Day (ToD) tokens**: Encode what time of day an activity occurs (6 buckets)
- **Day-of-Week (DoW) tokens**: Encode which day of the week an activity occurs (7 buckets)

These temporal tokens allow the model to learn daily and weekly patterns in human activities.

## Implementation Status

✅ **COMPLETED**: All code changes have been implemented and tested.

⚠️ **ACTION REQUIRED**: Models need to be retrained with the updated configuration to use ToD tokens.

## What Was Changed

### 1. Data Sampling (`src/sampling/base.py`)

**Added temporal feature extraction:**
- Created `_add_temporal_features()` method that adds `tod_bucket`, `dow_bucket`, and `time_delta_bucket` to raw data
- Updated `_event_to_dict()` to preserve these temporal features in sensor events
- Integrated temporal feature extraction into the data loading pipeline

**ToD Bucketing Logic:**
```python
def _get_tod_bucket(hour):
    if 5 <= hour < 8:
        return 'early_morning'
    elif 8 <= hour < 12:
        return 'late_morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    elif 21 <= hour < 24:
        return 'night_before_midnight'
    else:  # 0 <= hour < 5
        return 'night_after_midnight'
```

This creates 6 distinct time-of-day buckets plus an "UNK" token (7 total vocab items).

**DoW Bucketing Logic:**
```python
def _get_dow_bucket(dow):
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    return days[dow] if 0 <= dow <= 6 else 'unknown'
```

This creates 7 distinct day-of-week buckets plus an "UNK" token (8 total vocab items).

### 2. Encoder Configuration

**Updated:** `configs/alignment/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1.yaml`

Added `tod_bucket` and `dow_bucket` to categorical fields:
```yaml
metadata:
  categorical_fields:
  - sensor
  - state
  - room_id
  - tod_bucket  # NEW - Time of day
  - dow_bucket  # NEW - Day of week
  use_coordinates: true
  use_time_deltas: false
  use_time_of_day: true
```

### 3. Data Regeneration

**Regenerated:** `data/processed/casas/milan/FD_60_p/`

The FD_60_p dataset was regenerated with temporal token support:
- ✅ `vocab.json` now includes `tod_bucket` with 7 unique values
- ✅ `vocab.json` now includes `dow_bucket` with 8 unique values
- ✅ `vocab.json` now includes `time_delta_bucket` with 7 unique values
- ✅ All sensor events now contain `tod_bucket`, `dow_bucket`, and `time_delta_bucket` fields
- ✅ Vocabulary statistics:
  - sensor: 32 unique values
  - state: 6 unique values
  - room_id: 11 unique values
  - **tod_bucket: 7 unique values** (NEW)
  - **dow_bucket: 8 unique values** (NEW)
  - **time_delta_bucket: 7 unique values** (NEW)

## Verification

### Dataset Verification
```python
# Confirmed: SmartHomeDataset loads tod_bucket and dow_bucket
sample = dataset[0]
print(sample['categorical_features'].keys())
# Output: ['sensor', 'room_id', 'state', 'sensor_type', 'tod_bucket', 'dow_bucket']
```

### Encoder Verification
```python
# Confirmed: Encoder config includes tod_bucket and dow_bucket in categorical_fields
config.metadata.categorical_fields
# Output: ['sensor', 'state', 'room_id', 'tod_bucket', 'dow_bucket']
```

### Existing Model Check
```python
# Confirmed: Existing trained model does NOT have tod_bucket or dow_bucket
# The model at trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt
# was trained WITHOUT temporal token support and needs to be retrained.
```

## How Temporal Tokens Are Used

1. **Data Loading**: Raw sensor events are enriched with `tod_bucket` and `dow_bucket` based on their timestamp
2. **Vocabulary**: Each unique bucket value gets an index in the vocabulary
   - `tod_bucket`: 7 values (6 time periods + UNK)
   - `dow_bucket`: 8 values (7 days + UNK)
3. **Embedding**: The encoder creates learned embedding layers for both fields (vocab_size × d_model=512)
4. **Token Representation**: Each sensor event's embedding includes:
   - Sensor embedding
   - State embedding
   - Room embedding
   - **ToD bucket embedding** (NEW)
   - **DoW bucket embedding** (NEW)
   - Coordinate features (x, y)
5. **Training**: The model learns to associate different embeddings with:
   - Different times of day (morning routines vs. evening routines)
   - Different days of week (weekday patterns vs. weekend patterns)

## Benefits

- **Daily Temporal Context**: Model can learn that certain activities are more likely at certain times of day
- **Weekly Temporal Context**: Model can learn weekday vs. weekend patterns (e.g., "work" on weekdays, "relax" on weekends)
- **Disambiguation**: Activities with similar sensor patterns but different times/days can be distinguished
- **Improved Retrieval**: Queries like "morning routine", "evening activities", or "weekend behavior" become more meaningful
- **Better Generalization**: The model can learn that "cooking" at 7am is breakfast while "cooking" at 6pm is dinner

## Next Steps

### To Use ToD Tokens in Training:

1. **Regenerate other datasets** (if needed):
   ```bash
   python sample_data.py --config configs/sampling/milan_FD_60.yaml
   python sample_data.py --config configs/sampling/aruba_FD_60.yaml
   # etc.
   ```

2. **Regenerate captions and embeddings** (if needed):
   ```bash
   bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60 --skip-sampling
   ```

3. **Train a new model** with the updated config:
   ```bash
   python train.py --config configs/alignment/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1.yaml
   ```

4. **Verify temporal embeddings** in the trained model:
   ```python
   checkpoint = torch.load('path/to/model.pt')
   state_dict = checkpoint['model_state_dict']
   # Should see:
   #   sensor_encoder.embeddings.tod_bucket.weight
   #   sensor_encoder.embeddings.dow_bucket.weight
   ```

## Analysis Results

The time-of-day analysis (`src/evals/tod_analysis.py`) on the existing model showed:
- **Different activities have distinct temporal patterns**
- **Embeddings vary by time of day** for the same activity type
- **Master bedroom and bathroom activities** show clear morning vs. night differences

With temporal tokens (ToD + DoW) explicitly included in training, these patterns should be even more pronounced and useful for:
- **Activity recognition**: Better distinguish "morning bathroom" vs. "night bathroom"
- **Retrieval**: More accurate matching for temporal queries
- **Classification**: Improved accuracy by leveraging temporal context
- **Pattern discovery**: Identify weekday vs. weekend routines automatically

## Files Modified

1. `src/sampling/base.py` - Added temporal feature extraction (tod_bucket, dow_bucket, time_delta_bucket)
2. `src/dataio/dataset.py` - Added dow_bucket to possible categorical fields
3. `configs/alignment/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1.yaml` - Added tod_bucket and dow_bucket to categorical_fields
4. `data/processed/casas/milan/FD_60_p/` - Regenerated with temporal token support

## Related Documentation

- `docs/ENCODER_GUIDE.md` - General encoder architecture
- `docs/DATA_LOADING_GUIDE.md` - Data loading pipeline
- `docs/SAMPLING_UPDATES.md` - Sampling strategies
- `src/evals/tod_analysis.py` - Time-of-day embedding analysis script

