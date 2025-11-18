# Data Sampling Module (Step 1)

This module implements different data sampling strategies for converting raw sensor data into training/testing samples.

## Implemented Strategies

### 1a. Fixed-Length Sampling (`FixedLengthSampler`)
Samples fixed number of sensor events per window.

**Example**: 20, 50, 100 events per window

**Features**:
- Fixed sequence length (easier for models to handle)
- Sliding window with configurable overlap
- Optional presegmentation by ground truth labels
- Preserves all metadata for caption generation

**Use Case**: Traditional sequence models that expect fixed input size

### 1b. Fixed-Duration Sampling (`FixedDurationSampler`)
Samples fixed time duration per window, resulting in variable-length sequences.

**Example**: 30s, 60s, 120s windows

**Features**:
- Variable sequence length (1 to hundreds of events)
- Time-based windowing (more realistic for real-world scenarios)
- Sliding window with configurable overlap
- Optional cap on maximum events per window
- Metadata includes target vs actual duration

**Use Case**: Models that can handle variable-length inputs; more realistic temporal modeling

### 1c. Variable-Duration Sampling (Placeholder)
Future implementation for sampling with varying durations selected from a list.

## Quick Start

### List Available Configs
```bash
python sample_data.py --list-configs
```

### Run Fixed-Length Sampling
```bash
# 50-event windows
python sample_data.py --config configs/sampling/milan_FL_50.yaml

# Multiple window sizes (20 and 50 events)
python sample_data.py --config configs/sampling/milan_FL_20_50.yaml

# With presegmentation (activity-based)
python sample_data.py --config configs/sampling/milan_fixed_length_presegmented.yaml
```

### Run Fixed-Duration Sampling
```bash
# 60-second windows
python sample_data.py --config configs/sampling/milan_FD_60.yaml

# Multiple durations (30s, 60s, 120s)
python sample_data.py --config configs/sampling/milan_FD_30_60_120.yaml
```

### Debug Mode (Test with Limited Data)
```bash
python sample_data.py --config configs/sampling/milan_FL_50.yaml --debug
```

## Module Structure

```
src/sampling/
├── __init__.py           # Module exports
├── base.py               # BaseSampler abstract class + SamplingResult
├── config.py             # Configuration dataclasses
├── utils.py              # Shared utilities
├── fixed_length.py       # Fixed-length sampler (1a)
├── fixed_duration.py     # Fixed-duration sampler (1b)
└── README.md             # This file
```

## Configuration Options

### Common Options (All Strategies)
```yaml
dataset_name: "milan"                    # Dataset to process
raw_data_path: "data/raw/casas/milan"    # Path to raw data
output_dir: "data/sampled/milan/..."     # Output directory

# Train/test split
split_strategy: "random"                 # or "temporal"
train_ratio: 0.8                         # 80/20 split
random_seed: 42

# Overlap settings
overlap_factor: 0.5                      # 50% overlap between windows

# Presegmentation (optional)
use_presegmentation: false               # Split by activity labels first
presegment_label_level: "l1"             # "l1" or "l2"
min_segment_events: 8
exclude_no_activity: true

# Filtering
filter_numeric_sensors: true             # Remove temp/humidity sensors
max_gap_minutes: 30                      # Max time gap within window

# Metadata
preserve_full_metadata: true
include_spatial_info: true
include_sensor_types: true
```

### Fixed-Length Specific
```yaml
strategy: "fixed_length"
window_sizes: [20, 50]                   # List of window sizes to generate
min_events_per_window: 8                 # Recipe R2 requirement
```

### Fixed-Duration Specific
```yaml
strategy: "fixed_duration"
duration_seconds: [30, 60, 120]          # List of durations to generate
min_events_per_window: 1                 # At least 1 event required
max_events_per_window: null              # Optional cap for dense periods
max_sequence_length: 256                 # For padding during training
```

## Output Structure

```
data/processed/casas/milan/
├── FL_50/
│   ├── train.json
│   ├── test.json
│   └── sampling_config.yaml
├── FD_60/
│   ├── train.json
│   ├── test.json
│   └── sampling_config.yaml
└── ...
```

## Output Format

Each sampler produces two JSON files (`train.json` and `test.json`) with the following structure:

```json
{
  "dataset": "milan",
  "sampling_strategy": "fixed_length",
  "sampling_params": {
    "window_sizes": [50],
    "overlap_factor": 0.5,
    "presegmented": false,
    ...
  },
  "split": "train",
  "samples": [
    {
      "sample_id": "milan_train_000001",
      "sensor_sequence": [
        {
          "sensor_id": "M001",
          "event_type": "ON",
          "timestamp": "2009-02-12 08:30:45",
          "room": "kitchen",
          "activity_l1": "cooking",
          ...
        },
        ...
      ],
      "metadata": {
        "window_id": 0,
        "start_time": "...",
        "end_time": "...",
        "duration_seconds": 485.2,
        "num_events": 50,
        "rooms_visited": ["kitchen", "living_room"],
        "primary_room": "kitchen",
        "room_transitions": 3,
        "ground_truth_labels": {
          "primary_l1": "cooking",
          "primary_l2": "Cook.Breakfast",
          "all_labels_l1": ["cooking", "eating"],
          "label_distribution": {"cooking": 0.7, "eating": 0.3}
        },
        "presegmented": false
      }
    }
  ],
  "statistics": {
    "total_samples": 10000,
    "avg_sequence_length": 50.0,
    "avg_duration_seconds": 485.8,
    "room_distribution": {...}
  }
}
```

## Usage in Code

```python
from sampling import FixedLengthSampler, FixedLengthConfig, SamplingStrategy
from pathlib import Path

# Create config
config = FixedLengthConfig(
    dataset_name="milan",
    raw_data_path=Path("data/raw/casas/milan"),
    output_dir=Path("data/processed/casas/milan/test"),
    strategy=SamplingStrategy.FIXED_LENGTH,
    window_sizes=[50],
    train_ratio=0.8,
    overlap_factor=0.5
)

# Create sampler
sampler = FixedLengthSampler(config)

# Run sampling
train_result, test_result = sampler.sample_dataset()

# Save results
train_result.save_json(Path("data/processed/casas/milan/test/train.json"))
test_result.save_json(Path("data/processed/casas/milan/test/test.json"))

# Access statistics
print(f"Train samples: {len(train_result.samples)}")
print(f"Avg length: {train_result.statistics['avg_sequence_length']}")
```

## Key Design Decisions

1. **Self-Sufficient**: Each sampler works independently with minimal dependencies
2. **Column Mapping**: Adapted to work with `casas_end_to_end_preprocess` output format
3. **Standardized Output**: All samplers produce the same JSON structure
4. **Flexible Configuration**: YAML-based configs for easy experimentation
5. **Metadata Preservation**: All information needed for caption generation is preserved

## Testing

Both samplers have been tested on Milan dataset:

**Fixed-Length (50 events)**:
- 186 train samples, 192 test samples (debug mode)
- Avg sequence length: 50 events
- Avg duration: ~485 seconds

**Fixed-Duration (60 seconds)**:
- 500 train samples, 500 test samples (debug mode)
- Avg sequence length: ~14 events (variable!)
- Avg duration: ~43 seconds

## Next Steps

- Step 2: Implement sensor encoders (transformer, chronos, image-based)
- Step 3: Implement caption generators (rule-based, LLM-based)
- Step 4: Implement text encoders
- Step 5: Implement alignment training
- Step 6: Implement retrieval
- Step 7: Implement clustering

## Contributing

When adding new sampling strategies:
1. Inherit from `BaseSampler`
2. Implement `_create_windows_for_dataframe()` and `_get_sampling_params()`
3. Create corresponding config class inheriting from `SamplingConfig`
4. Add tests and example configs
5. Update this README

