# Data Generation for CASAS Dual-Encoder Training

This directory contains the data generation pipeline for processing CASAS smart home datasets into training data for dual-encoder models.

## Overview

The data generation system processes raw CASAS sensor data into structured training examples with diversified natural language captions. It supports multiple datasets, window sizes, and splitting strategies.

## Quick Start

```bash
# Activate the conda environment
conda activate har_env

# List available configurations
python src/generate_data.py --list-configs

# Generate Milan training data (recommended)
python src/generate_data.py --config milan_training --force

# Quick validation test
python src/generate_data.py --config quick_validation
```

## Key Features

### 🔄 **Data Splitting Strategies**
- **Random Split (Default)**: Randomly selects 20% of days for testing
- **Temporal Split**: Uses last 20% of days chronologically for testing

### 📝 **Diversified Caption Generation**
- **Two Narrative Modes**:
  - **Active**: "The resident moved from kitchen to living room"
  - **Passive**: "Motion was detected in the kitchen area"
- **Enhanced Temporal Descriptions**: Time information at beginning or end
- **"Back" Terminology**: For revisited locations ("moved back", "came back")
- **Duration Variants**: From "very short span" to "substantial period"
- **Gap Detection**: Notes significant pauses between sensor activations
- **Multiple Resident Terms**: resident, occupant, dweller, person, individual

### 🏠 **Supported Datasets**
- **Milan**: Multi-room apartment with detailed sensor layout
- **Aruba**: Single-occupant smart home
- **Cairo**: Multi-sensor environment
- **Kyoto**: Traditional Japanese home
- **Tulum**: Research facility dataset

## Available Configurations

### Training Configurations
- **`milan_training`**: Milan dataset for dual-encoder training (JSON export)
- **`milan_temporal_split`**: Milan with temporal splitting for comparison

### Research Configurations
- **`recipe_r2_full`**: Full Recipe R2 implementation (all datasets, 128-event windows)
- **`recipe_r2_milan`**: Recipe R2 with Milan only
- **`multi_window_small`**: Multiple window sizes on smaller datasets

### Testing Configurations
- **`quick_validation`**: Fast test with small windows

## Command Line Interface

### Basic Usage
```bash
# Run pre-defined configuration
python src/generate_data.py --config <config_name> [--force]

# List all available configurations
python src/generate_data.py --list-configs

# Custom configuration
python src/generate_data.py --custom --datasets milan aruba --windows 20 50
```

### Arguments
- `--config, -c`: Name of pre-defined configuration
- `--list-configs, -l`: List all available configurations
- `--custom`: Run custom configuration
- `--datasets`: Datasets to process (for custom configs)
- `--windows`: Window sizes to use (for custom configs)
- `--output-dir, -o`: Output directory (default: `data/processed_data`)
- `--force, -f`: Force reprocessing even if files exist

## Output Structure

### Default Output: `data/data_for_alignment/`
```
data/data_for_alignment/
├── milan_train.json          # Training data (67K samples)
├── milan_test.json           # Test data (17K samples)
├── milan_vocab.json          # Feature vocabulary
├── milan_statistics.json    # Dataset statistics
└── processing_summary.json  # Processing metadata
```

### Data Format (JSON Lines)
Each line in the training/test files contains:
```json
{
  "window_id": 12345,
  "events": [...],           // Sensor events with features
  "captions": ["..."],       // Diversified natural language captions
  "metadata": {              // Window metadata
    "duration_sec": 120.5,
    "rooms_visited": ["kitchen", "living_room"],
    "primary_activity": "cooking",
    ...
  }
}
```

## Configuration System

### Processing Pipeline
1. **Data Loading**: Raw CASAS data preprocessing
2. **Windowing**: Fixed-length sliding windows with overlap
3. **Feature Extraction**: Spatial, temporal, and sensor features
4. **Caption Generation**: Diversified natural language descriptions
5. **Export**: JSON format for training

### Key Configuration Options
```python
# Windowing
sizes: [20, 50, 100]          # Window sizes (number of events)
overlap_ratio: 0.75           # Sliding window overlap
min_events: 8                 # Minimum events per window

# Splitting
split_strategy: "random"      # "random" or "temporal"
test_size: 0.2               # 20% for testing

# Captions
use_enhanced_captions: True   # Enable diversified captions
include_duration: True        # Include temporal information
include_room_transitions: True # Include movement descriptions
```

## Examples

### Generate Training Data
```bash
# Generate Milan training data with random splitting
python src/generate_data.py --config milan_training --force

# Generate with temporal splitting for comparison
python src/generate_data.py --config milan_temporal_split --force
```

### Custom Processing
```bash
# Process multiple datasets with different window sizes
python src/generate_data.py --custom \
  --datasets milan aruba cairo \
  --windows 20 50 100 \
  --output-dir data/my_custom_data
```

### Quick Testing
```bash
# Fast validation with small dataset
python src/generate_data.py --config quick_validation
```

## Caption Examples

### Active Mode (Resident Actions)
- "The resident moved from kitchen to living room lasting 3 minutes on Monday in December during the evening"
- "Occupant spent time in kitchen for 2 minutes, with activity near fridge on Wednesday in Jan"

### Passive Mode (Sensor Detection)
- "Motion was detected lasting 4 minutes movement from bedroom to bathroom on Friday in autumn"
- "Activity occurred in kitchen area for 1 minute with stove area motion during afternoon hours"

### Enhanced Features
- **Back Movement**: "moved back to kitchen", "returned to living room"
- **Duration Variety**: "very short span", "substantial period", "lasted quite long"
- **Temporal Flexibility**: Time at start or end of sentence
- **Gap Detection**: "with significant pauses between sensor activations"

## Performance

### Typical Processing Times
- **Quick Validation**: ~1-2 minutes
- **Milan Training**: ~3-5 minutes
- **Multi-Dataset**: ~10-30 minutes
- **Full Recipe R2**: ~2-4 hours

### Output Sizes
- **Milan Training**: ~558MB (train + test JSON)
- **Vocabulary**: ~2KB
- **Statistics**: ~9KB

## Troubleshooting

### Common Issues
1. **Missing conda environment**: Run `conda activate har_env`
2. **Memory errors**: Use smaller window sizes or datasets
3. **File permission errors**: Check output directory permissions
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode
Add `--force` flag to regenerate existing data and see full processing logs.

## File Organization

```
src/
├── generate_data.py           # Main data generation script
├── config/
│   ├── processing.py          # Processing configurations
│   └── datasets.py           # Dataset configurations
├── data/
│   ├── pipeline.py           # Main processing pipeline
│   ├── data_loader.py        # Data loading and splitting
│   ├── windowing.py          # Window generation
│   ├── features.py           # Feature extraction
│   ├── captions.py           # Caption generation
│   └── exporters.py          # Data export utilities
├── experiments/
│   └── experiment_configs.py # Pre-defined configurations
└── README_DATA_GENERATION.md # This file
```

## Next Steps

After generating training data:
1. **Training**: Use the JSON files for dual-encoder model training
2. **Evaluation**: Compare random vs temporal splitting performance
3. **Analysis**: Examine caption diversity and quality
4. **Scaling**: Process additional datasets as needed

For questions or issues, refer to the main project documentation or examine the configuration files for customization options.
