# Caption Style Guide

This document describes the different caption generation styles available in the system and how to use them.

## Overview

The system supports multiple caption generation styles to enable different research approaches and comparisons:

1. **Baseline** (default): Natural language captions with rich variation
2. **Sourish**: Structured template-based captions following Sourish Dhekane's zero-shot classification approach

## Caption Styles

### 1. Baseline Captions (`caption_style: "baseline"`)

**Description:** Natural, human-readable captions with rich linguistic diversity and randomized variations.

**Key Features:**
- **Natural language**: Fluent, varied sentences
- **Randomization**: Multiple templates, word choices, and sentence structures
- **Semantic depth**: Contextual information (gaps, duration patterns, resident behavior)
- **Activity-centric**: Focuses on what the resident is doing
- **Rich variation**: Active/passive modes, diverse vocabulary, temporal ordering

**Example Captions:**
```
"On Monday in January in the morning, the resident moved lasting 2 minutes movement from master bedroom to master bathroom, with bed area motion and bathroom activity."

"Activity detected in the afternoon spanning 5 minutes movement from kitchen to living room, showing activity near fridge and lounge area motion."
```

**When to Use:**
- General-purpose caption generation
- Retrieval tasks requiring natural language
- When linguistic diversity is important
- For generative models or caption-based understanding

**Implementation:** `src/data/captions.py` - `CaptionGenerator`

---

### 2. Sourish Captions (`caption_style: "sourish"`)

**Description:** Structured, template-based captions following Sourish Dhekane's approach for zero-shot activity recognition.

**Key Features:**
- **Fixed template structure**: `when + duration + where + sensors`
- **Deterministic**: Same input produces identical output
- **Sensor-centric**: Emphasizes which sensors fired and their locations
- **Dataset-specific mappings**: Hard-coded sensor-to-location mappings for each dataset
- **Mechanical descriptions**: Lists facts without interpretation

**Template Structure (4-Component Format):**

All Sourish captions follow the exact same structure with 4 required components:

```
"The activity started at [TIME] [AM/PM] [ZONE] and ended at [TIME] [AM/PM] [ZONE].
The activity was performed for [DURATION].
The activity is taking place [LOCATION1] mainly and parts of it [LOCATION2].
The [most commonly fired sensor(s)] in this activity are [SENSOR_TYPE] [SENSOR_LOCATION]."
```

**Components:**
1. **Time of Occurrence**: When the activity started and ended (with period of day)
2. **Duration**: How long the activity lasted
3. **Top-2 Locations**: Most common and second most common locations
4. **Top-2 Sensors**: Most commonly fired sensors with their context
```

**Example Caption:**
```
"The activity started at eight AM morning and ended at eight AM morning. The activity was performed for two minutes. The activity is taking place in master bedroom mainly and parts of it in aisle near master bedroom. The two most commonly fired sensors in this activity are Motion sensor in master bedroom and Motion sensor in aisle near master bedroom."
```

Note: Every caption contains all 4 components - no shortened versions.

**When to Use:**
- Zero-shot activity classification
- Comparing with Sourish's baseline results
- When systematic, structured descriptions are needed
- For sentence embedding similarity-based classification

**Implementation:** `src/data/captions_sourish.py` - `SourishCaptionGenerator`

**Dataset Support:**
- **Milan**: Full sensor mapping (28 sensors)
- **Aruba**: Full sensor mapping (32 sensors)
- **Cairo**: Full sensor mapping (27 sensors)
- Other datasets: Fallback to generic descriptions

---

## Configuration

### Setting Caption Style

In your data generation config JSON:

```json
{
  "captions": {
    "caption_style": "baseline",  // or "sourish"
    "num_captions_per_window": 4,
    "max_caption_length": 50,
    "use_enhanced_captions": true,
    "include_duration": true,
    "include_time_context": true,
    "include_room_transitions": true,
    "include_salient_sensors": true,
    "caption_types": "long",
    "random_seed": 42,
    "use_synonyms": true
  }
}
```

### Example Configs

**Baseline Captions:**
```bash
# Use existing configs (default is baseline)
python src/data/generate_data.py --config milan_baseline_seq20
```

**Sourish Captions:**
```bash
# Use sourish-specific config
python src/data/generate_data.py --config milan_sourish_test
```

---

## Comparison

| Aspect | Baseline | Sourish |
|--------|----------|---------|
| **Style** | Natural language | Template-based |
| **Variation** | High (randomized) | None (deterministic) |
| **Focus** | Activity & behavior | Sensors & locations |
| **Sensor Info** | Semantic (e.g., "bed area motion") | Technical (e.g., "Motion sensor M021 in master bedroom on bed") |
| **Location** | Room-level | Fine-grained sensor locations |
| **Complexity** | Variable sentences | Fixed structure |
| **Use Case** | General NLP tasks | Zero-shot classification |
| **Readability** | High (human-like) | Medium (mechanical) |

---

## Implementation Details

### Baseline Generator (`CaptionGenerator`)

**Features:**
- Sensor ontology extraction from metadata
- Room trajectory analysis
- Salient sensor selection
- Duration and gap analysis
- Active/passive voice variations
- Temporal context variations (day of week, month, time of day)

**Output:**
- 2 long templated captions
- 2 short creative captions (< 10 tokens)

### Sourish Generator (`SourishCaptionGenerator`)

**Features:**
- Hard-coded sensor-to-location mappings
- Two most common location extraction
- Two most common sensor extraction
- Time zone determination (morning, afternoon, evening, night)
- Number-to-text conversion using `inflect`
- Dataset-specific special cases (e.g., Milan bed-to-toilet scenarios)

**Output:**
- 1 full structured caption per window (all 4 components: when + duration + where + sensors)
- Deterministic - same window always produces the same caption

---

## Adding New Caption Styles

To add a new caption style:

1. **Create generator class:**
   ```python
   # src/data/captions_mystyle.py
   class MyStyleCaptionGenerator:
       def __init__(self, config: CaptionConfig):
           self.config = config

       def generate_captions(self, window, window_features, sensor_details):
           # Generate captions
           return {
               'long': [...],
               'short': [...],
               'all': [...]
           }
   ```

2. **Update pipeline:**
   ```python
   # src/data/pipeline.py
   from .captions_mystyle import MyStyleCaptionGenerator

   # In __init__:
   if caption_style == 'mystyle':
       self.caption_generator = MyStyleCaptionGenerator(config.captions)
   ```

3. **Update config loader:**
   ```python
   # src/utils/process_data_configs.py
   # No changes needed - caption_style is already extracted
   ```

4. **Document the style** in this file

---

## Dependencies

### Baseline
- pandas, numpy
- Standard Python libraries

### Sourish
- pandas, numpy
- **inflect** (for number-to-text conversion)

To install dependencies:
```bash
conda activate discover-v2-env
pip install inflect
```

Or add to `env.yaml`:
```yaml
dependencies:
  - pip:
    - inflect>=7.5.0
```

---

## References

### Sourish's Zero-Shot Activity Recognition
- **Location:** `OTHER/sourish_zero_shot/`
- **Files:**
  - `dataset_utils.py`: Sensor-to-location mappings
  - `utils.py`: Caption generation templates
  - `zeroshot_classify.py`: Classification approach

### Baseline Captions
- **Paper:** (Future reference to DISCOVER-v2 paper)
- **Implementation:** `src/data/captions.py`

---

## Testing

### Quick Test
```bash
# Test Sourish captions on Milan
python src/data/generate_data.py --config milan_sourish_test

# View generated captions
cat data/processed/casas/milan/milan_sourish_test/milan_test.json | python -m json.tool | grep -A 3 '"captions"' | head -20
```

### Expected Output Format
```json
{
  "captions": [
    "Full caption with all components...",
    "Full caption with all components...",
    "Short location caption.",
    "Short location caption."
  ]
}
```

---

## Future Extensions

Potential additional caption styles:
- **BERT-style**: Masked language model format
- **Concise**: Ultra-short factual descriptions
- **Narrative**: Story-like continuous descriptions
- **Multi-lingual**: Captions in multiple languages
- **Question-Answer**: Pairs of questions and answers about activities

---

Last updated: 2025-01-08

