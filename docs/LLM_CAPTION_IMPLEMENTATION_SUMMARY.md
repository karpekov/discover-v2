# LLM Caption Generation - Implementation Summary

This document summarizes the LLM-based caption generation system that was implemented.

## Overview

A complete LLM-based caption generation pipeline has been implemented with support for multiple backends (local and remote), pluggable architecture, and comprehensive debugging tools.

## What Was Implemented

### 1. Core Components

#### Compact JSON Representation (`src/captions/llm_based/compact_json.py`)

Converts raw sensor samples into a compact JSON format optimized for LLM caption generation:

- **Function**: `to_compact_caption_json(sample: dict) -> dict`
- **Output fields**:
  - Basic info: `sample_id`, `duration_seconds`, `num_events`
  - Spatial: `primary_room`, `rooms_visited`, `room_transitions`
  - Temporal: `time_context` (start/end time, day_of_week, month, period_of_day)
  - Sensors: `special_sensors` (triggered, primary, counts, frequent)
  - Movement: `movement_summary` (pattern, transitions)
  - Activity labels: `primary_l1`, `primary_l2`, `all_labels_l1` (if available)

- **Features**:
  - Automatically derives time context (morning/afternoon/evening/night)
  - Identifies frequent special sensors (count ≥ 3)
  - Creates movement pattern summaries (e.g., "kitchen→living_room")
  - Removes full sensor sequence (not needed for captions)

#### LLM Backends (`src/captions/llm_based/backends.py`)

Four backend implementations with unified interface:

1. **GemmaHFBackend** - Local Gemma models via Hugging Face
   - Default: `google/gemma-7b`
   - Configurable device, temperature, top_p

2. **LlamaHFBackend** - Local Llama models via Hugging Face
   - Default: `meta-llama/Meta-Llama-3-8B-Instruct`
   - Chat template support for instruct models

3. **OpenAIBackend** - OpenAI GPT models via API
   - Default: `gpt-4o-mini`
   - Supports all GPT models

4. **GeminiBackend** - Google Gemini models via API
   - Default: `gemini-1.5-flash`
   - System instruction support

**Common Interface**: `CaptionModel` protocol
```python
def generate(self, prompts: List[str]) -> List[List[str]]:
    """Generate captions for list of prompts."""
```

**Factory Function**: `create_backend(backend_type, model_name, ...)`

#### Prompting System (`src/captions/llm_based/prompts.py`)

Unified prompting across all backends:

- **SYSTEM_PROMPT**: Expert caption writer instructions
  - Clear rules for caption generation
  - Focus on concrete nouns and action verbs
  - Output as JSON list of strings

- **User Prompt Builder**: `build_user_prompt(compact_json, num_captions)`
  - Constructs structured metadata description
  - Omits missing fields gracefully
  - Requests specific number of captions

- **Features**:
  - Consistent prompting across local and remote models
  - Flexible prompt construction from compact JSON
  - Batch prompt generation

#### LLM Caption Generator (`src/captions/llm_based/base.py`)

Main generator class implementing `BaseCaptionGenerator`:

```python
class LLMCaptionGenerator(BaseCaptionGenerator):
    def __init__(self, config: LLMCaptionConfig):
        # Initialize backend
        self.backend = create_backend(...)

    def generate(self, sensor_sequence, metadata, sample_id) -> CaptionOutput:
        # 1. Convert to compact JSON
        # 2. Build prompt
        # 3. Call backend
        # 4. Return CaptionOutput

    def generate_batch(self, samples) -> List[CaptionOutput]:
        # Batch processing support
```

#### Configuration (`src/captions/config.py`)

Extended `LLMCaptionConfig`:

```python
@dataclass
class LLMCaptionConfig(CaptionConfig):
    backend_type: str = 'openai'  # gemma, llama, openai, gemini
    model_name: str = 'gpt-4o-mini'
    api_key: Optional[str] = None
    device: Optional[str] = None  # For local models
    temperature: float = 0.9
    max_tokens: int = 512
    top_p: float = 0.95
```

### 2. Scripts and Tools

#### End-to-End Caption Generation (`src/captions/generate_llm_captions.py`)

Complete pipeline script with features:

- **Config file support**: Load settings from YAML
- **Multiple backends**: Switch between local/remote models
- **Batch processing**: Configurable batch size
- **Split handling**: Process train/val/test separately or together
- **Compact JSON export**: Optional intermediate file saving

**Debug Features** (NEW):
- `--debug`: Auto-enables verbose mode, limits to 3 samples, shows prompts
- `--limit N`: Process only first N samples per split
- `--verbose`: Print generated captions to console
- `--show-prompts`: Display prompts sent to LLM

**Usage**:
```bash
# Debug mode - quick test with 3 samples
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend openai \
    --model gpt-4o-mini \
    --debug

# Production run
python src/captions/generate_llm_captions.py \
    --config configs/captions/llm_openai.yaml \
    --data-dir data/processed/casas/milan/FD_60
```

#### Compact JSON Converter (`src/captions/convert_to_compact.py`)

Standalone utility to convert samples to compact JSON:

- Single file or directory mode
- JSONL or JSON output
- Verbose mode to inspect output

**Usage**:
```bash
# Convert directory
python src/captions/convert_to_compact.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --format jsonl \
    --verbose

# Convert single file
python src/captions/convert_to_compact.py \
    --input train.json \
    --output train_compact.jsonl
```

#### Integration with Existing Script (`src/captions/generate_captions.py`)

Updated to support LLM backends:

```bash
python src/captions/generate_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --caption-style llm \
    --llm-backend openai \
    --llm-model gpt-4o-mini \
    --num-captions 4
```

### 3. Configuration Files

Four pre-configured YAML files in `configs/captions/`:

1. **llm_openai.yaml** - OpenAI GPT-4o-mini
2. **llm_gemini.yaml** - Google Gemini 1.5 Flash
3. **llm_gemma.yaml** - Local Gemma 7B
4. **llm_llama.yaml** - Local Llama 3 8B Instruct

Each config specifies:
- Backend type and model name
- Generation parameters (temperature, num_captions)
- Device settings (for local models)
- Dataset info

### 4. Examples and Tests

#### Test Script (`examples/test_llm_captions.py`)

Minimal test with synthetic sample:

```bash
# Test OpenAI
export OPENAI_API_KEY=sk-...
python examples/test_llm_captions.py --backend openai

# Test Gemini
export GOOGLE_API_KEY=...
python examples/test_llm_captions.py --backend gemini

# Test local model
python examples/test_llm_captions.py --backend gemma --device cuda
```

Features:
- Creates minimal test sample
- Shows compact JSON conversion
- Displays prompts
- Analyzes generated captions
- Checks keyword coverage

### 5. Documentation

#### Main Documentation (`docs/LLM_CAPTION_GENERATION.md`)

Comprehensive guide covering:
- Architecture overview
- Usage examples for all backends
- Configuration options
- Output format
- Prompting system details
- API key management
- Performance considerations
- Troubleshooting
- Integration with training pipeline

#### Implementation Summary (`docs/LLM_CAPTION_IMPLEMENTATION_SUMMARY.md`)

This document - overview of what was implemented.

## File Structure

```
discover-v2/
├── src/captions/
│   ├── llm_based/
│   │   ├── __init__.py              # Exports
│   │   ├── base.py                  # LLMCaptionGenerator
│   │   ├── backends.py              # Backend implementations
│   │   ├── compact_json.py          # Compact JSON conversion
│   │   └── prompts.py               # Prompt templates
│   ├── generate_llm_captions.py     # End-to-end script
│   ├── convert_to_compact.py        # Utility script
│   ├── generate_captions.py         # Updated for LLM support
│   └── config.py                    # Updated with LLMCaptionConfig
├── configs/captions/
│   ├── llm_openai.yaml              # OpenAI config
│   ├── llm_gemini.yaml              # Gemini config
│   ├── llm_gemma.yaml               # Gemma config
│   └── llm_llama.yaml               # Llama config
├── examples/
│   └── test_llm_captions.py         # Test script
└── docs/
    ├── LLM_CAPTION_GENERATION.md    # Main documentation
    └── LLM_CAPTION_IMPLEMENTATION_SUMMARY.md  # This file
```

## Key Features

### Pluggable Backend Architecture

- Single interface for all backends
- Easy to add new backends
- Consistent behavior across local/remote models
- Factory pattern for backend creation

### Compact JSON Representation

- Removes unnecessary data (full sensor sequence)
- Enriches with derived features (time context, movement patterns)
- Works for train/val/test splits
- Efficient for LLM consumption

### Debug-Friendly

- **Debug mode**: `--debug` flag for quick testing
- **Sample limiting**: `--limit N` to process subset
- **Verbose output**: See captions as they're generated
- **Prompt inspection**: `--show-prompts` to view LLM inputs
- **Test script**: Minimal example for backend verification

### Flexible Configuration

- YAML config files for common setups
- Command-line arg overrides
- Environment variable support for API keys
- Sensible defaults for all backends

### Production Ready

- Batch processing support
- Error handling with fallbacks
- Progress tracking with tqdm
- Split handling (train/val/test)
- Output file naming conventions

## Usage Examples

### Quick Test (Debug Mode)

```bash
# Test with 3 samples, see prompts and captions
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend openai \
    --debug
```

Output:
```
🐛 DEBUG MODE ENABLED
  - Limiting to 3 samples per split
  - Verbose output enabled
  - Showing sample prompts

===============================================================
SAMPLE PROMPT (first sample):
===============================================================
You are given a structured description of...

Sample id: milan_train_000001
Duration (seconds): 120.5
Primary room: kitchen
...
===============================================================

────────────────────────────────────────────────────────────
Sample ID: milan_train_000001
Duration: 120.5s | Events: 45 | Room: kitchen
Time: Monday morning

Generated Captions (4):
  1. On Monday morning in January, the resident spent...
  2. Morning kitchen activity lasting 2 minutes...
  3. Person cooking breakfast near the fridge...
  4. Kitchen meal prep on Monday, 2 minute duration.
```

### Production Run with Config

```bash
# Generate captions for all splits
python src/captions/generate_llm_captions.py \
    --config configs/captions/llm_openai.yaml \
    --data-dir data/processed/casas/milan/FD_60
```

### Custom Backend and Model

```bash
# Use specific model with custom settings
python src/captions/generate_llm_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --backend openai \
    --model gpt-4o \
    --num-captions 6 \
    --temperature 1.0 \
    --batch-size 5
```

### Inspect Compact JSON

```bash
# See what metadata the LLM receives
python src/captions/convert_to_compact.py \
    --input data/processed/casas/milan/FD_60/train.json \
    --verbose
```

## Dependencies

### Required

- Python 3.8+
- transformers (for local models)
- torch (for local models)
- openai (for OpenAI backend)
- google-generativeai (for Gemini backend)
- pyyaml
- tqdm

### Installation

```bash
conda activate discover-v2-env

# For local models
pip install transformers torch accelerate

# For OpenAI
pip install openai

# For Gemini
pip install google-generativeai
```

## Output Format

Files are saved with descriptive names:

```
data/processed/casas/milan/FD_60/
├── train_llm_openai_gpt_4o_mini.json
├── test_llm_openai_gpt_4o_mini.json
└── train_llm_gemini_gemini_1_5_flash.json
```

Each file contains samples with added `llm_captions` field:

```json
{
  "samples": [
    {
      "sample_id": "milan_train_000001",
      "sensor_sequence": [...],
      "metadata": {...},
      "llm_captions": [
        "Caption 1...",
        "Caption 2...",
        "Caption 3...",
        "Caption 4..."
      ]
    }
  ]
}
```

## Integration with Existing System

The LLM caption generator:

- ✅ Follows existing `BaseCaptionGenerator` interface
- ✅ Uses existing `CaptionConfig` structure
- ✅ Integrates with `generate_captions.py` script
- ✅ Compatible with existing caption styles (baseline, sourish)
- ✅ Outputs same `CaptionOutput` format
- ✅ Works with train/val/test splits
- ✅ Fits into existing data pipeline

## Next Steps

To use the LLM caption generator in training:

1. **Generate captions** for your dataset:
   ```bash
   python src/captions/generate_llm_captions.py \
       --config configs/captions/llm_openai.yaml \
       --data-dir data/processed/casas/milan/FD_60
   ```

2. **Load captions** in training pipeline:
   ```python
   with open('train_llm_openai_gpt_4o_mini.json') as f:
       data = json.load(f)

   for sample in data['samples']:
       llm_captions = sample['llm_captions']
       # Use for CLIP training
   ```

3. **Compare** with rule-based captions:
   - Generate baseline captions
   - Generate LLM captions
   - Train models with both
   - Compare retrieval performance

## Troubleshooting

### API Keys

Set environment variables:
```bash
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
```

### Local Models OOM

```bash
# Use smaller batch size
--batch-size 1

# Use CPU
--device cpu

# Use smaller model
--model google/gemma-2b
```

### Rate Limits

```bash
# Reduce batch size
--batch-size 5

# Process only train split
--split train
```

## Summary

A complete, production-ready LLM caption generation system has been implemented with:

- ✅ 4 backend implementations (Gemma, Llama, OpenAI, Gemini)
- ✅ Compact JSON representation
- ✅ Unified prompting system
- ✅ End-to-end generation script
- ✅ Debug mode for testing
- ✅ Config file support
- ✅ Integration with existing pipeline
- ✅ Comprehensive documentation
- ✅ Test examples
- ✅ No new dependencies on existing modules

The system is ready for use!

---

Last updated: 2025-01-08

