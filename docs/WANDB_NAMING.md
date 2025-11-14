# WandB Naming Convention

This document describes the intelligent WandB naming scheme for discover-v2 experiments.

## Overview

WandB runs are automatically organized with intuitive names, groups, and tags based on your configuration. This makes it easy to:
- **Find experiments**: Search by dataset, encoder, caption style, etc.
- **Compare runs**: Group similar experiments together
- **Filter results**: Use tags to narrow down experiments

## Hierarchical Organization

```
discover-v2 (Project)
├── milan (Dataset)
│   ├── milan_fixdur60s_tf-base (Group)
│   │   ├── milan_fixdur60s_tf-base_baseline_gte_linear
│   │   ├── milan_fixdur60s_tf-base_baseline_gte_mlp2
│   │   ├── milan_fixdur60s_tf-base_sourish_gte_linear
│   │   └── milan_fixdur60s_tf-base_baseline_llama_linear
│   └── milan_fixlen50_tf-small (Group)
│       └── ...
└── aruba (Dataset)
    └── ...
```

## Run Name Format

**Format**: `{dataset}_{sampling}_{encoder}_{caption-style}_{text-encoder}_{projection}`

**Optional suffixes**: `__{loss-config}` (if not default CLIP-only)

### Components

| Component | Description | Examples |
|-----------|-------------|----------|
| **dataset** | Dataset name | `milan`, `aruba`, `cairo` |
| **sampling** | Sampling strategy | `fixdur60s`, `fixlen50`, `fixdur120s_preseg` |
| **encoder** | Sensor encoder model | `tf-base`, `tf-tiny`, `tf-small` |
| **caption-style** | Caption generation style | `baseline`, `sourish`, `adl-llm` |
| **text-encoder** | Text embedding model | `gte`, `llama`, `distilroberta`, `gemma` |
| **projection** | Projection type | `linear`, `mlp2`, `mlp3` |
| **loss-config** | Loss configuration (optional) | `clip+mlm`, `clip+hn`, `clip+mlm+hn` |

### Examples

#### Standard Configurations

```
milan_fixdur60s_tf-base_baseline_gte_linear
├─┬───────┬────┬──────┬─────────┬───┬──────
│ │       │    │      │         │   └─ Linear projection
│ │       │    │      │         └─ GTE text encoder
│ │       │    │      └─ Baseline captions
│ │       │    └─ Transformer-base encoder
│ │       └─ Fixed-duration 60s sampling
│ └─ Milan dataset
└─ Project root

milan_fixlen50_tf-small_sourish_llama_mlp2
├─┬───────┬─────────┬───────┬─────┬────
│ │       │         │       │     └─ 2-layer MLP projection
│ │       │         │       └─ LLAMA text encoder
│ │       │         └─ Sourish captions
│ │       └─ Transformer-small encoder
│ └─ Fixed-length 50 events sampling
└─ Milan dataset
```

#### With Non-Default Loss

```
milan_fixdur60s_tf-base_baseline_gte_linear__clip+mlm
                                          └──────────
                                            Suffix: CLIP + MLM loss

aruba_fixlen20_tf-tiny_baseline_gte_mlp2__clip+hn
                                         └────────
                                           Suffix: CLIP + hard negatives
```

## Group Name Format

**Format**: `{dataset}_{sampling}_{encoder}`

Groups together runs that differ only in caption/text encoder/projection choices.

### Examples

```
milan_fixdur60s_tf-base
milan_fixlen50_tf-small
aruba_fixdur120s_preseg_tf-base
```

## Tags

Tags are automatically generated based on configuration:

### Dataset Tags
- Dataset name: `milan`, `aruba`, `cairo`

### Sampling Tags
- Strategy: `fixed-duration`, `fixed-length`, `variable-duration`
- Presegmentation: `presegmented`

### Model Tags
- Encoder: `tf-base`, `tf-small`, `tf-tiny`
- Caption style: `caption-baseline`, `caption-sourish`, `caption-adl-llm`
- Text encoder: `text-gte`, `text-llama`, `text-distilroberta`
- Projection: `proj-linear`, `proj-mlp2`, `proj-mlp3`

### Loss Tags
- Loss components: `mlm`, `hard-negatives`
- Temperature: `learnable-temp`, `fixed-temp`

### Example Tag Set

```python
[
    'milan',                    # Dataset
    'fixed-duration',           # Sampling strategy
    'tf-base',                  # Encoder
    'caption-baseline',         # Caption style
    'text-gte',                 # Text encoder
    'proj-linear',              # Projection
    'learnable-temp'            # Temperature setting
]
```

With MLM:
```python
[
    'milan',
    'fixed-duration',
    'tf-base',
    'caption-baseline',
    'text-gte',
    'proj-mlp2',
    'mlm',                      # MLM enabled
    'learnable-temp'
]
```

## Configuration

### Automatic (Recommended)

Leave `wandb_name`, `wandb_group`, and `wandb_tags` as `null` or empty:

```yaml
# WandB logging
use_wandb: true
wandb_project: discover-v2
wandb_entity: your_username
wandb_name: null      # Auto-generated
wandb_tags: []        # Auto-generated
wandb_group: null     # Auto-generated
wandb_notes: "Optional description of experiment"
```

### Manual Override

You can still manually specify names if needed:

```yaml
# WandB logging
use_wandb: true
wandb_project: discover-v2
wandb_entity: your_username
wandb_name: my_custom_experiment_v1
wandb_tags: [custom, experiment, testing]
wandb_group: my_experiments
wandb_notes: "Custom experiment description"
```

## Abbreviations Reference

### Sampling Strategies
| Full Name | Abbreviation | Example |
|-----------|--------------|---------|
| Fixed duration | `fixdur{N}s` | `fixdur60s`, `fixdur120s` |
| Fixed length | `fixlen{N}` | `fixlen50`, `fixlen20` |
| Variable duration | `vardur` | `vardur` |
| Presegmented | `_preseg` | `fixdur60s_preseg` |

### Encoders
| Full Name | Abbreviation |
|-----------|--------------|
| Transformer base | `tf-base` |
| Transformer tiny | `tf-tiny` |
| Transformer small | `tf-small` |
| Transformer minimal | `tf-min` |
| Chronos | `chronos` |

### Text Encoders
| Full Name | Abbreviation |
|-----------|--------------|
| GTE-base | `gte` |
| DistilRoBERTa | `distilroberta` |
| MiniLM | `minilm` |
| EmbeddingGemma | `gemma` |
| LLAMA Embed | `llama` |
| CLIP | `clip` |
| SigLIP | `siglip` |

### Projections
| Full Name | Abbreviation |
|-----------|--------------|
| Linear | `linear` |
| MLP 2-layer | `mlp2` |
| MLP 3-layer | `mlp3` |

### Loss Configurations
| Configuration | Abbreviation |
|---------------|--------------|
| CLIP only | (no suffix) |
| CLIP + MLM | `clip+mlm` |
| CLIP + Hard Negatives | `clip+hn` |
| CLIP + MLM + HN | `clip+mlm+hn` |

## Benefits

### 1. Easy Filtering

Search for specific configurations:
```
# All Milan experiments
milan_*

# All fixed-duration 60s experiments
*_fixdur60s_*

# All experiments with baseline captions and GTE
*_baseline_gte_*

# All experiments with MLP projections
*_mlp2
*_mlp3

# All experiments with MLM
*__clip+mlm*
```

### 2. Clear Grouping

Groups organize related experiments:
- Compare different caption styles for the same encoder
- Compare different text encoders for the same caption style
- Compare different projection types for the same setup

### 3. Intuitive Tags

Filter by tags in WandB UI:
- Show only `fixed-duration` experiments
- Show only `mlm` experiments
- Show only `proj-mlp2` experiments
- Combine: `milan` + `tf-base` + `caption-baseline`

### 4. Self-Documenting

Run names tell you exactly what configuration was used:
```
milan_fixdur60s_tf-base_baseline_gte_linear__clip+mlm

You immediately know:
✓ Dataset: Milan
✓ Sampling: Fixed-duration 60 seconds
✓ Encoder: Transformer-base
✓ Captions: Baseline style
✓ Text encoder: GTE-base
✓ Projection: Linear
✓ Loss: CLIP + MLM
```

## Implementation

The naming scheme is implemented in `src/alignment/wandb_utils.py`:

```python
from src.alignment.wandb_utils import (
    generate_wandb_run_name,
    generate_wandb_group,
    generate_wandb_tags
)

# Auto-generate names from config
run_name = generate_wandb_run_name(config)
group = generate_wandb_group(config)
tags = generate_wandb_tags(config)
```

The `AlignmentTrainer` automatically uses these functions if no manual names are provided.

## Tips

### Finding Your Experiments

**By dataset:**
```
Project: discover-v2
└─ Filter tags: milan
```

**By configuration:**
```
Project: discover-v2
└─ Filter tags: milan, tf-base, caption-baseline
```

**By technique:**
```
Project: discover-v2
└─ Filter tags: mlm, hard-negatives
```

### Comparing Runs

Use groups to compare similar runs:
```
Group: milan_fixdur60s_tf-base
├─ Runs with different caption styles
├─ Runs with different text encoders
└─ Runs with different projections
```

### Organizing Experiments

Use the hierarchical structure:
```
discover-v2/
├─ milan/                          (dataset tag)
│  ├─ milan_fixdur60s_tf-base/    (group)
│  │  ├─ ...baseline_gte_linear
│  │  ├─ ...baseline_gte_mlp2
│  │  └─ ...sourish_gte_linear
│  └─ milan_fixlen50_tf-small/    (group)
│     └─ ...
└─ aruba/                          (dataset tag)
   └─ ...
```

---

**Status**: ✅ Implemented and ready to use
**Last Updated**: November 14, 2025

