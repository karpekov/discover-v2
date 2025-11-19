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

**Format**: `{dataset}_{sampling_short}_{seq/img}_{rb/llm}_text{encoder}_proj{lin/mlp}`

**All detailed metadata (loss config, temperature, etc.) goes in tags, not in the run name.**

### Components

| Component | Description | Examples |
|-----------|-------------|----------|
| **dataset** | Dataset name | `milan`, `aruba`, `cairo` |
| **sampling_short** | Sampling strategy (abbreviated) | `fd20`, `fl50`, `fd120p` |
| **seq/img** | Data encoder type | `seq` (sequence), `img` (image-based) |
| **rb/llm** | Caption style | `rb0` (baseline), `rb` (rule-based), `llm` |
| **text{encoder}** | Text embedding model | `textclip`, `textminilm`, `textgte` |
| **proj{lin/mlp}** | Projection type | `projlin`, `projmlp` |

### Examples

#### Standard Configurations

```
milan_fd20_seq_rb0_textclip_projlin
├───┬──┬──┬──┬──────┬─────
│   │  │  │  │      └─ Linear projection
│   │  │  │  └─ CLIP text encoder
│   │  │  └─ Baseline (rule-based 0) captions
│   │  └─ Sequence-based encoder
│   └─ Fixed-duration 20 seconds
└─ Milan dataset

milan_fl30_img_llm_textminilm_projmlp
├───┬──┬──┬──┬────────┬────
│   │  │  │  │        └─ MLP projection
│   │  │  │  └─ MiniLM text encoder
│   │  │  └─ LLM-generated captions
│   │  └─ Image-based encoder
│   └─ Fixed-length 30 events
└─ Milan dataset

aruba_fd60p_seq_rb_textgte_projlin
├────┬───┬──┬─┬──────┬─────
│    │   │  │ │      └─ Linear projection
│    │   │  │ └─ GTE text encoder
│    │   │  └─ Rule-based captions (non-baseline)
│    │   └─ Sequence-based encoder
│    └─ Fixed-duration 60s, presegmented
└─ Aruba dataset
```

## Group Name Format

**Format**: `{dataset}_{sampling}_{seq/img}`

Groups together runs that differ only in caption/text encoder/projection choices.

### Examples

```
milan_fd60_seq
milan_fl50_img
aruba_fd120p_seq
```

## Tags

Tags contain **all detailed metadata** for filtering and organization:

### Core Tags
- **Dataset**: `milan`, `aruba`, `cairo`
- **Sampling strategy**: `fixed-duration`, `fixed-length`, `variable-duration`
- **Sampling specific**: `fd20`, `fl30`, `fd60p` (exact sampling config)
- **Presegmentation**: `presegmented`
- **Encoder type**: `seq` (sequence-based), `img` (image-based)
- **Image model** (if img): `img-clip`, `img-dinov2`, `img-siglip`

### Caption & Text Encoder Tags
- **Caption style**: `rb0` (baseline), `rb` (rule-based), `llm`
- **Text encoder**: `text-clip`, `text-minilm`, `text-gte`, `text-llama`

### Projection Tags
- **Type**: `proj-lin`, `proj-mlp`
- **MLP layers**: `mlp-2layer`, `mlp-3layer`

### Training Configuration Tags
- **Loss components**: `mlm`, `hard-negatives`
- **MLM weight**: `mlm-weight-0.5`, `mlm-weight-1.0`
- **Temperature**: `learnable-temp`, `fixed-temp`, `temp-0070`
- **Optimizer**: `opt-adamw`, `opt-adam`
- **Learning rate**: `lr-3e-04`, `lr-1e-03`

### Example Tag Set

```python
# Sequence-based with MLM
[
    'milan',                    # Dataset
    'fixed-duration',           # Sampling type
    'fd20',                     # Exact sampling config
    'seq',                      # Encoder type
    'rb0',                      # Baseline captions
    'text-clip',                # Text encoder
    'proj-lin',                 # Projection type
    'mlm',                      # MLM enabled
    'mlm-weight-0.5',           # MLM weight
    'fixed-temp',               # Temperature mode
    'temp-0070',                # Temperature value
    'opt-adamw',                # Optimizer
    'lr-3e-04'                  # Learning rate
]

# Image-based with LLM captions
[
    'aruba',
    'fixed-length',
    'fl30',
    'img',                      # Image-based encoder
    'img-clip',                 # Using CLIP image encoder
    'llm',                      # LLM-generated captions
    'text-minilm',
    'proj-mlp',
    'mlp-2layer',
    'learnable-temp',
    'opt-adamw',
    'lr-5e-04'
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
| Fixed duration | `fd{N}` | `fd20`, `fd60`, `fd120` |
| Fixed length | `fl{N}` | `fl20`, `fl30`, `fl50` |
| Variable duration | `vd` | `vd` |
| Presegmented | `p` suffix | `fd60p`, `fl30p` |

### Encoder Types
| Type | Abbreviation |
|------|--------------|
| Sequence-based (raw data) | `seq` |
| Image-based | `img` |

### Caption Styles
| Full Name | Abbreviation |
|-----------|--------------|
| Baseline rule-based | `rb0` |
| Other rule-based | `rb` |
| LLM-generated | `llm` |

### Text Encoders
| Full Name | Abbreviation |
|-----------|--------------|
| CLIP | `textclip` |
| MiniLM | `textminilm` |
| GTE-base | `textgte` |
| DistilRoBERTa | `textdistilroberta` |
| EmbeddingGemma | `textgemma` |
| LLAMA Embed | `textllama` |
| SigLIP | `textsiglip` |

### Projections
| Full Name | Abbreviation |
|-----------|--------------|
| Linear | `projlin` |
| MLP (any layers) | `projmlp` |

## Benefits

### 1. Easy Filtering

Search for specific configurations:
```
# All Milan experiments
milan_*

# All FD_20 experiments
*_fd20_*

# All sequence-based experiments
*_seq_*

# All image-based experiments
*_img_*

# All baseline caption experiments
*_rb0_*

# All CLIP text encoder experiments
*_textclip_*

# All MLP projection experiments
*_projmlp
```

### 2. Clear Grouping

Groups organize related experiments:
- Compare different caption styles for the same data/sampling config
- Compare different text encoders for the same setup
- Compare different projection types for the same encoder

### 3. Intuitive Tags

Filter by tags in WandB UI:
- Show only `fixed-duration` experiments
- Show only `seq` or `img` experiments
- Show only `mlm` experiments
- Show only `rb0` (baseline) or `llm` caption experiments
- Combine: `milan` + `fd20` + `seq` + `mlm`

### 4. Self-Documenting

Run names tell you exactly what configuration was used:
```
milan_fd20_seq_rb0_textclip_projlin

You immediately know:
✓ Dataset: Milan
✓ Sampling: Fixed-duration 20 seconds (FD_20)
✓ Encoder: Sequence-based
✓ Captions: Baseline rule-based
✓ Text encoder: CLIP
✓ Projection: Linear

Tags provide additional details:
✓ MLM weight, temperature, optimizer, learning rate, etc.
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

**By sampling config:**
```
Project: discover-v2
└─ Filter tags: fd20  (all FD_20 runs)
└─ Filter tags: fl30  (all FL_30 runs)
```

**By encoder type:**
```
Project: discover-v2
└─ Filter tags: seq   (sequence-based)
└─ Filter tags: img   (image-based)
└─ Filter tags: img-clip  (image-based with CLIP)
```

**By caption style:**
```
Project: discover-v2
└─ Filter tags: rb0   (baseline)
└─ Filter tags: llm   (LLM-generated)
```

**By training config:**
```
Project: discover-v2
└─ Filter tags: mlm, mlm-weight-0.5
└─ Filter tags: learnable-temp
```

### Comparing Runs

Use groups to compare similar runs:
```
Group: milan_fd20_seq
├─ milan_fd20_seq_rb0_textclip_projlin
├─ milan_fd20_seq_rb0_textgte_projlin
├─ milan_fd20_seq_llm_textclip_projlin
└─ milan_fd20_seq_rb0_textclip_projmlp
```

### Organizing Experiments

Use the hierarchical structure:
```
discover-v2/
├─ milan/                              (dataset tag)
│  ├─ milan_fd20_seq/                 (group)
│  │  ├─ milan_fd20_seq_rb0_textclip_projlin
│  │  ├─ milan_fd20_seq_rb0_textgte_projlin
│  │  └─ milan_fd20_seq_llm_textclip_projmlp
│  ├─ milan_fl30_img/                 (group)
│  │  ├─ milan_fl30_img_rb0_textclip_projlin
│  │  └─ milan_fl30_img_llm_textminilm_projmlp
│  └─ ...
└─ aruba/                              (dataset tag)
   └─ ...
```

---

**Status**: ✅ Implemented and ready to use
**Last Updated**: November 19, 2025

