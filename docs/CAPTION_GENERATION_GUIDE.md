# Caption Generation Guide (Step 3)

This guide explains how to generate captions for sampled sensor data using the modular caption generation framework.

## Overview

The caption generation system (Step 3) converts sensor reading sequences into natural language descriptions. It supports multiple caption styles that can be easily swapped and configured.

## Architecture

### Components

```
src/captions/
├── base.py              # BaseCaptionGenerator, CaptionOutput
├── config.py            # Configuration classes
├── rule_based/          # Rule-based generators
│   ├── baseline.py      # Enhanced baseline captions
│   └── sourish.py       # Sourish-style structured captions
└── llm_based/           # LLM-based generators (placeholder)
    └── base.py          # LLMCaptionGenerator (TODO)
```

### Caption Styles

#### 1. Baseline (Enhanced)

Natural language captions with rich temporal and spatial context.

**Features:**
- Temporal context (day of week, month, time of day)
- Duration descriptions with gap analysis
- Room transitions with back-movement detection
- Sensor-specific details (bed, toilet, door, appliances)
- Multiple variations (active vs passive voice)

**Example Output:**
```
"On Monday in January during the morning, the resident moved lasting 5 minutes
from kitchen to living room then to bedroom, with activity near fridge and
movement in living room."
```

**Config:** `configs/captions/baseline_milan.yaml`

#### 2. Sourish Style

Structured template-based captions following Sourish Dhekane's research format.

**Format:** `when + duration + where + sensors`

**Features:**
- Fixed template structure
- Dataset-specific sensor-to-location mappings
- Deterministic output (no variation)
- Four-component format

**Example Output:**
```
"The activity started at seven AM morning and ended at seven AM morning.
The activity was performed for five minutes. The activity is taking place
in kitchen near fridge mainly and parts of it in living room. The most
commonly fired sensor in this activity is Motion sensor in kitchen near fridge."
```

**Config:** `configs/captions/sourish_milan.yaml`

#### 3. LLM-Based (Placeholder)

Future support for LLM-generated captions using GPT-4, Claude, Gemini, or local models.

**Status:** Not yet implemented. Placeholder exists for future integration.

## Usage

### Command Line Interface

#### Using Config Files (Recommended)

```bash
# Generate captions using config file
# All parameters (num_captions, dataset_name, etc.) come from config
python src/captions/generate_captions.py \
    --config configs/captions/baseline_aruba.yaml \
    --data-dir data/processed/casas/aruba/FD_60

# Generate captions for Milan using config
python src/captions/generate_captions.py \
    --config configs/captions/baseline_milan.yaml \
    --data-dir data/processed/casas/milan/FD_60

# Override config values with command-line args
python src/captions/generate_captions.py \
    --config configs/captions/baseline_aruba.yaml \
    --data-dir data/processed/casas/aruba/FD_60 \
    --num-captions 8

# Sourish captions from config
python src/captions/generate_captions.py \
    --config configs/captions/sourish_milan.yaml \
    --data-dir data/processed/casas/milan/FD_60
```

#### Using Command-Line Arguments Only

```bash
# Generate baseline captions for Milan data
# Output: train_captions_baseline.json, test_captions_baseline.json
python src/captions/generate_captions.py \
    --data-dir data/processed/casas/milan/fixed_length_50 \
    --caption-style baseline \
    --dataset-name milan \
    --num-captions 4

# Generate Sourish captions
# Output: train_captions_sourish.json, test_captions_sourish.json
python src/captions/generate_captions.py \
    --data-dir data/processed/casas/milan/fixed_length_50 \
    --caption-style sourish \
    --dataset-name milan

# Generate LLM captions (placeholder)
# Output: train_captions_llm_gpt4.json, test_captions_llm_gpt4.json
python src/captions/generate_captions.py \
    --data-dir data/processed/casas/milan/fixed_length_50 \
    --caption-style llm \
    --llm-model gpt4 \
    --dataset-name milan

# Generate captions only for training set
python src/captions/generate_captions.py \
    --data-dir data/processed/casas/milan/fixed_duration_60 \
    --caption-style baseline \
    --dataset-name milan \
    --split train

# Customize number of captions per sample
python src/captions/generate_captions.py \
    --data-dir data/processed/casas/milan/fixed_length_50 \
    --caption-style baseline \
    --dataset-name milan \
    --num-captions 5
```

### Programmatic Usage

```python
from pathlib import Path
import json
from captions import BaselineCaptionGenerator, RuleBasedCaptionConfig

# Load sampled data
with open('data/processed/casas/milan/fixed_length_50/train.json', 'r') as f:
    data = json.load(f)

# Create generator
config = RuleBasedCaptionConfig(
    caption_style='baseline',
    num_captions_per_sample=2,
    dataset_name='milan',
    sensor_details_path='metadata/casas_metadata.json'
)
generator = BaselineCaptionGenerator(config)

# Generate captions for samples
caption_outputs = []
for sample in data['samples']:
    output = generator.generate(
        sensor_sequence=sample['sensor_sequence'],
        metadata=sample['metadata'],
        sample_id=sample['sample_id']
    )
    caption_outputs.append(output)

# Save captions
output_data = {'captions': [co.to_dict() for co in caption_outputs]}
with open('data/processed/casas/milan/fixed_length_50/train_captions_baseline.json', 'w') as f:
    json.dump(output_data, f, indent=2)

# Print results
for output in caption_outputs[:3]:
    print(f"Sample: {output.sample_id}")
    for i, caption in enumerate(output.captions, 1):
        print(f"  Caption {i}: {caption}")
    print()

# Get statistics
stats = generator.get_statistics(caption_outputs)
print(f"Total captions: {stats['total_captions']}")
print(f"Avg length: {stats['caption_length_stats']['mean_tokens']:.1f} tokens")
```

## Output Format

### Caption Files

Captions are saved as JSON files alongside the sampled data with style-specific suffixes:

```
data/processed/casas/milan/fixed_length_50/
├── train.json                      # Sampled data
├── test.json                       # Sampled data
├── train_captions_baseline.json   # Baseline captions
├── test_captions_baseline.json    # Baseline captions
├── train_captions_sourish.json    # Sourish captions (optional)
├── test_captions_sourish.json     # Sourish captions (optional)
├── train_captions_llm_gpt4.json   # LLM captions (future)
└── test_captions_llm_gpt4.json    # LLM captions (future)
```

**Filename Format**: `{split}_captions_{style}.json`
- **Rule-based styles**: `baseline`, `sourish`, `mixed`
- **LLM styles**: `llm_{model}` (e.g., `llm_gpt4`, `llm_claude`, `llm_gemini`)

This allows you to generate and store multiple caption styles for the same sensor data.

### Caption JSON Structure

```json
{
  "captions": [
    {
      "sample_id": "milan_train_000001",
      "captions": [
        "On Monday in January...",
        "kitchen activity day"
      ],
      "metadata": {
        "caption_type": "baseline",
        "num_long": 2,
        "num_short": 2,
        "layer_b": "span=08:30–08:35; dur=5.0m; ..."
      }
    }
  ]
}
```

### Matching Captions to Sensor Data

Captions and sensor sequences are matched by `sample_id`:

```python
import json

# Load sensor data and captions
with open('train.json', 'r') as f:
    sensor_data = json.load(f)

# Load baseline captions
with open('train_captions_baseline.json', 'r') as f:
    baseline_caption_data = json.load(f)

# Load sourish captions (if available)
with open('train_captions_sourish.json', 'r') as f:
    sourish_caption_data = json.load(f)

# Create indices for fast lookup
baseline_index = {
    item['sample_id']: item['captions']
    for item in baseline_caption_data['captions']
}
sourish_index = {
    item['sample_id']: item['captions']
    for item in sourish_caption_data['captions']
}

# Match sensor sequences to captions
for sample in sensor_data['samples']:
    sample_id = sample['sample_id']
    sensor_sequence = sample['sensor_sequence']
    baseline_captions = baseline_index.get(sample_id, [])
    sourish_captions = sourish_index.get(sample_id, [])

    print(f"Sample: {sample_id}")
    print(f"  Events: {len(sensor_sequence)}")
    print(f"  Baseline captions: {baseline_captions}")
    print(f"  Sourish captions: {sourish_captions}")
```

## Configuration

### Using Config Files

Config files provide a convenient way to manage caption generation settings. They are loaded using the `--config` flag and support all generation parameters.

**Priority**: Command-line arguments override config file values.

### Baseline Configuration

```yaml
# configs/captions/baseline_milan.yaml
caption_style: baseline
num_captions_per_sample: 4  # Generate 4 captions per sample
random_seed: 42

dataset_name: milan
sensor_details_path: metadata/casas_metadata.json

# Caption generation options
generate_long_captions: true
generate_short_captions: true
include_temporal_context: true
include_duration_details: true
include_sensor_details: true
```

**Usage:**
```bash
python src/captions/generate_captions.py \
    --config configs/captions/baseline_milan.yaml \
    --data-dir data/processed/casas/milan/FD_60
```

**Override config values:**
```bash
python src/captions/generate_captions.py \
    --config configs/captions/baseline_milan.yaml \
    --data-dir data/processed/casas/milan/FD_60 \
    --num-captions 8  # Override config's num_captions_per_sample
```

### Sourish Configuration

```yaml
# configs/captions/sourish_milan.yaml
caption_style: sourish
num_captions_per_sample: 1  # Sourish is deterministic
random_seed: 42

dataset_name: milan  # REQUIRED for sensor mappings
```

**Note**: The `num_captions_per_sample` value in config files is properly respected (previously there was a bug where it was always defaulting to 2).

## Caption Statistics

The system automatically computes statistics:

```
TRAIN Caption Statistics:
  Total samples: 10000
  Total captions: 40000
  Avg captions/sample: 4.00
  Caption length: 42.3 ± 12.5 tokens
    Min: 3, Max: 87

  Sample captions:
    1. On Monday in January during the morning...
    2. kitchen activity day
```

## Extending with New Caption Styles

To add a new caption style:

1. **Create a new generator class** in `src/captions/rule_based/`:

```python
from ..base import BaseCaptionGenerator, CaptionOutput
from ..config import RuleBasedCaptionConfig

class MyCustomCaptionGenerator(BaseCaptionGenerator):
    def __init__(self, config: RuleBasedCaptionConfig):
        super().__init__(config)

    def generate(self, sensor_sequence, metadata, sample_id):
        # Your caption generation logic
        caption_text = self._my_generation_logic(sensor_sequence, metadata)

        return CaptionOutput(
            captions=[caption_text],
            sample_id=sample_id,
            metadata={'caption_type': 'my_custom'}
        )

    def generate_batch(self, samples):
        return [self.generate(**s) for s in samples]
```

2. **Register in `__init__.py`**:

```python
from .my_custom import MyCustomCaptionGenerator

__all__ = [..., 'MyCustomCaptionGenerator']
```

3. **Update `src/captions/generate_captions.py`** to include new style in choices.

## Integration with Training Pipeline

Captions will be used in Step 4 (Text Encoding) and Step 5 (Alignment):

```
Step 1: Sample Data → data/processed/.../train.json
                   ↓
Step 3: Generate Captions → data/processed/.../train_captions_{style}.json
                   ↓
Step 4: Text Encoding → text embeddings
                   ↓
Step 5: Alignment → aligned sensor + text embeddings
```

You can generate multiple caption styles and compare them during training:

```bash
# Generate both baseline and sourish captions
python src/captions/generate_captions.py --data-dir data/processed/casas/milan/fixed_length_50 \
    --caption-style baseline --dataset-name milan

python src/captions/generate_captions.py --data-dir data/processed/casas/milan/fixed_length_50 \
    --caption-style sourish --dataset-name milan

# Now you have:
# - train_captions_baseline.json
# - train_captions_sourish.json
# Both can be used separately or compared during training
```

## Notes

- **Sensor Details**: Baseline captions are richer when sensor metadata is provided
- **Dataset Name**: Required for Sourish style (uses dataset-specific mappings)
- **Reproducibility**: Use `random_seed` for consistent caption variations
- **Performance**: Caption generation is fast (~1000 samples/sec for baseline)
- **Storage**: Captions stored separately from sensor data for flexibility

## Future Enhancements

- [ ] Mixed caption strategy (random selection from multiple styles)
- [ ] LLM-based caption generation (GPT-4, Claude, Llama)
- [ ] Caption quality metrics
- [ ] Multi-language support
- [ ] Configurable templates for custom caption formats

