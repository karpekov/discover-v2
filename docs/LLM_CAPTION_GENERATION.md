# LLM-Based Caption Generation

This document describes the LLM-based caption generation system for sensor activity sequences.

## Overview

The LLM caption generator takes raw sensor sequences and generates natural language captions using Large Language Models (LLMs). It supports multiple backends:

- **Local models**: Gemma, Llama (via Hugging Face Transformers)
- **Remote APIs**: OpenAI (GPT), Google (Gemini)

## Architecture

The system follows a three-stage pipeline:

1. **Compact JSON Conversion**: Raw sensor sequences → Compact JSON metadata
2. **Prompt Generation**: Compact JSON → LLM prompts
3. **Caption Generation**: LLM prompts → Natural language captions

### Compact JSON Format

The compact JSON representation removes the full sensor sequence and keeps only essential metadata:

```json
{
  "sample_id": "milan_train_000001",
  "duration_seconds": 120.5,
  "num_events": 45,
  "primary_room": "kitchen",
  "rooms_visited": ["kitchen", "living_room"],
  "room_transitions": 2,
  "time_context": {
    "start_time": "2023-01-15T08:30:00",
    "end_time": "2023-01-15T08:32:00",
    "day_of_week": "Monday",
    "month": "January",
    "period_of_day": "morning"
  },
  "special_sensors": {
    "special_sensors_triggered": ["fridge", "stove"],
    "primary_special_sensor": "fridge",
    "special_sensor_counts": {"fridge": 5, "stove": 2},
    "frequent_special_sensors": ["fridge"]
  },
  "movement_summary": {
    "pattern": "kitchen→living_room",
    "num_rooms": 2,
    "num_unique_rooms": 2,
    "num_transitions": 2
  },
  "primary_l1": "Meal_Preparation",
  "primary_l2": "Cook",
  "all_labels_l1": ["Meal_Preparation", "Eating"]
}
```

## Usage

### Quick Start - Command Line

#### Debug Mode (Recommended for First-Time Use)

Start with debug mode to generate captions for just a few samples and inspect the results:

```bash
# Debug with OpenAI (processes 3 samples, shows prompts and captions)
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend openai \
    --model gpt-4o-mini \
    --num-captions 4 \
    --debug

# Debug with custom limit
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend gemini \
    --model gemini-1.5-flash \
    --limit 5 \
    --verbose \
    --show-prompts
```

**Debug Mode Features:**
- `--debug`: Auto-enables verbose output, shows prompts, limits to 3 samples
- `--limit N`: Process only first N samples per split
- `--verbose`: Print generated captions to console
- `--show-prompts`: Display the prompts sent to the LLM

#### OpenAI GPT

```bash
# Using config file
python src/captions/generate_llm_captions.py \
    --config configs/captions/llm_openai.yaml \
    --data-dir data/processed/casas/milan/FD_60

# Using command-line args
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend openai \
    --model gpt-4o-mini \
    --num-captions 4 \
    --api-key $OPENAI_API_KEY
```

#### Google Gemini

```bash
# Using config file
python src/captions/generate_llm_captions.py \
    --config configs/captions/llm_gemini.yaml \
    --data-dir data/processed/casas/milan/FD_60

# Using command-line args
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend gemini \
    --model gemini-1.5-flash \
    --num-captions 4 \
    --api-key $GOOGLE_API_KEY
```

#### Local Gemma

```bash
# Using config file
python src/captions/generate_llm_captions.py \
    --config configs/captions/llm_gemma.yaml \
    --data-dir data/processed/casas/milan/FD_60

# Using command-line args
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend gemma \
    --model google/gemma-7b \
    --num-captions 4 \
    --device cuda
```

#### Local Llama

```bash
# Using config file
python src/captions/generate_llm_captions.py \
    --config configs/captions/llm_llama.yaml \
    --data-dir data/processed/casas/milan/FD_60

# Using command-line args
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend llama \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --num-captions 4 \
    --device cuda
```

### Using the Existing generate_captions.py Script

The standard caption generation script also supports LLM backends:

```bash
python src/captions/generate_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --caption-style llm \
    --llm-backend openai \
    --llm-model gpt-4o-mini \
    --num-captions 4
```

### Programmatic Usage

```python
from captions.llm_based import (
    LLMCaptionGenerator,
    to_compact_caption_json,
    create_backend
)
from captions.config import LLMCaptionConfig

# Method 1: Using LLMCaptionGenerator
config = LLMCaptionConfig(
    backend_type='openai',
    model_name='gpt-4o-mini',
    num_captions_per_sample=4,
    temperature=0.9,
    api_key='sk-...'  # Or None to read from env
)

generator = LLMCaptionGenerator(config)

# Generate captions for a single sample
output = generator.generate(
    sensor_sequence=sample['sensor_sequence'],
    metadata=sample['metadata'],
    sample_id=sample['sample_id']
)

print(output.captions)
# ['Morning kitchen activity...', 'Person cooking breakfast...', ...]

# Generate captions for batch
outputs = generator.generate_batch(samples)

# Method 2: Using backend directly
backend = create_backend(
    backend_type='openai',
    model_name='gpt-4o-mini',
    num_captions=4,
    temperature=0.9,
    api_key='sk-...'
)

# Convert sample to compact JSON
compact_json = to_compact_caption_json(sample)

# Build prompt
from captions.llm_based.prompts import build_user_prompt
user_prompt = build_user_prompt(compact_json, num_captions=4)

# Generate
caption_lists = backend.generate([user_prompt])
captions = caption_lists[0]
```

## Configuration

### Config File Format (YAML)

```yaml
# Backend settings
backend_type: openai  # 'gemma', 'llama', 'openai', 'gemini'
model_name: gpt-4o-mini

# Generation settings
num_captions_per_sample: 4
temperature: 0.9

# API key (optional - can also set via env var or CLI arg)
api_key: sk-...

# Device (for local models only)
device: cuda  # or 'cpu', 'mps', null for auto

# Dataset info
dataset_name: milan

# Random seed
random_seed: 42
```

### Backend Types

| Backend | Type | Models | API Key Required |
|---------|------|--------|------------------|
| `openai` | Remote | `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo` | Yes (OPENAI_API_KEY) |
| `gemini` | Remote | `gemini-1.5-flash`, `gemini-1.5-pro` | Yes (GOOGLE_API_KEY) |
| `gemma` | Local | `google/gemma-7b`, `google/gemma-2b` | No |
| `llama` | Local | `meta-llama/Meta-Llama-3-8B-Instruct`, etc. | No |

### API Keys

API keys can be provided in three ways (in order of precedence):

1. Command-line argument: `--api-key sk-...`
2. Config file: `api_key: sk-...`
3. Environment variable: `OPENAI_API_KEY` or `GOOGLE_API_KEY`

## Output Format

### File Organization

```
data/processed/casas/milan/FD_60/
├── train.json                              # Original samples
├── test.json
├── train_llm_openai_gpt_4o_mini.json      # LLM captions (OpenAI)
├── test_llm_openai_gpt_4o_mini.json
├── train_llm_gemini_gemini_1_5_flash.json # LLM captions (Gemini)
└── test_llm_gemini_gemini_1_5_flash.json
```

### Output JSON Structure

The output files contain the original samples with an added `llm_captions` field:

```json
{
  "samples": [
    {
      "sample_id": "milan_train_000001",
      "sensor_sequence": [...],
      "metadata": {...},
      "llm_captions": [
        "On Monday morning in January, the resident spent two minutes in the kitchen preparing breakfast near the fridge and stove.",
        "Morning activity in the kitchen with fridge and stove use, lasting about 2 minutes.",
        "Person in kitchen during morning hours, interacting with fridge and cooking appliances.",
        "Kitchen meal prep activity on a Monday morning, 2 minute duration."
      ]
    }
  ]
}
```

## Prompting System

### System Prompt

The system prompt is shared across all backends and defines the caption writing guidelines:

```
You are an expert caption writer for describing how someone lives in a house
based on nearable sensor activation data. Your goal is to turn structured
metadata into several short, diverse, natural-language captions that a human
might type when searching for this segment.

The captions will be embedded with a CLIP-style text encoder, so they must
be clear, concrete, and diverse enough.

Follow these rules:
- Write 4 alternative captions per example.
- Keep them under 100 words. Create various length of captions for the same data sample.
- Focus on the main activity, place, important objects, and time context.
- Prefer concrete nouns and action verbs over abstract phrasing.
- Describe this as if we are observing a person living in a house and doing the things actively.
- It is fine to repeat important key words across captions, but vary wording, syntax, and level of detail.
- Do not include explanations or commentary.
- Output only a JSON list of strings, with each string being one caption.
```

### User Prompt Template

For each sample, a user prompt is built from the compact JSON:

```
You are given a structured description of a short sensor-based activity segment.
Your task is to generate diverse, human-style captions that follow the system instructions.

Here is the metadata for one segment:

Sample id: milan_train_000001
Duration (seconds): 120.5
Number of events: 45
Primary room: kitchen
Rooms visited: kitchen, living_room
Room transitions: 2
Time context: day of week Monday, month January, period of day morning
Activity (L1): Meal_Preparation
Special sensors triggered: fridge, stove
Frequent special sensors: fridge
Movement pattern: kitchen→living_room

Using this information, generate 4 alternative captions that a human might
use when searching for this segment in a retrieval system.

Remember to follow all formatting rules and return only a JSON list of strings.
```

## Utility Scripts

### Convert to Compact JSON

Convert raw samples to compact JSON format for inspection:

```bash
# Convert single file
python src/captions/convert_to_compact.py \
    --input data/processed/casas/milan/FD_60/train.json \
    --output train_compact.jsonl

# Convert directory (all splits)
python src/captions/convert_to_compact.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --format jsonl

# Save as JSON instead of JSONL
python src/captions/convert_to_compact.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --format json

# Show sample output
python src/captions/convert_to_compact.py \
    --input data/processed/casas/milan/FD_60/train.json \
    --verbose
```

## Dependencies

### Local Backends (Gemma, Llama)

```bash
conda activate discover-v2-env
pip install transformers torch accelerate
```

For CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Remote Backends

#### OpenAI

```bash
pip install openai
```

#### Gemini

```bash
pip install google-generativeai
```

## Performance Considerations

### Local Models

- **GPU Memory**: Gemma-7B requires ~14GB VRAM, Llama-3-8B requires ~16GB
- **Speed**: ~1-2 samples/sec on consumer GPU (RTX 4090)
- **Batch Size**: Adjust based on available VRAM

### Remote APIs

- **Rate Limits**:
  - OpenAI: ~500 requests/min (tier dependent)
  - Gemini: ~60 requests/min (free tier)
- **Speed**: ~5-10 samples/sec (network dependent)
- **Cost**:
  - GPT-4o-mini: ~$0.15 per 1K samples (4 captions each)
  - Gemini Flash: Free tier available

## Troubleshooting

### Local Models Not Loading

```python
# Check CUDA availability
import torch
print(torch.cuda.is_available())

# Use CPU if GPU unavailable
python src/captions/generate_llm_captions.py \
    --backend gemma \
    --model google/gemma-7b \
    --device cpu
```

### API Rate Limits

```bash
# Reduce batch size
python src/captions/generate_llm_captions.py \
    --backend openai \
    --batch-size 5  # Lower batch size
```

### JSON Parsing Errors

If the LLM fails to output valid JSON, the system automatically falls back to text parsing. Check the output for any warning messages.

## Best Practices

1. **Start with API backends** (OpenAI, Gemini) for initial testing
2. **Use local models** for large-scale generation to reduce costs
3. **Adjust temperature** (0.7-1.0) for caption diversity
4. **Save compact JSON** with `--save-compact` for debugging
5. **Process in batches** to handle rate limits and memory constraints

## Integration with Training Pipeline

The generated captions can be used in the text encoder training:

```python
# Load samples with LLM captions
with open('train_llm_openai_gpt_4o_mini.json', 'r') as f:
    data = json.load(f)

for sample in data['samples']:
    sensor_sequence = sample['sensor_sequence']
    llm_captions = sample['llm_captions']

    # Use for CLIP-style training
    # sensor_embedding = sensor_encoder(sensor_sequence)
    # text_embeddings = text_encoder(llm_captions)
    # loss = contrastive_loss(sensor_embedding, text_embeddings)
```

## Future Enhancements

- [ ] Support for more backends (Claude, Mistral, Cohere)
- [ ] Batch processing with async API calls
- [ ] Caption quality metrics
- [ ] Multi-language caption generation
- [ ] Fine-tuning local models on domain-specific data

---

Last updated: 2025-01-08

