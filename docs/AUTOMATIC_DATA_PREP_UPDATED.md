# Automatic Data Preparation Pipeline

## Overview

The `prepare_data.sh` script automates the entire data preparation pipeline in one command:

1. **Sample Data** → Generate train/val/test splits using sampling configs
2. **Generate Captions** → Create natural language descriptions using caption configs
3. **Encode Captions** → Embed captions into vectors using text encoder configs

This replaces the manual 3-step process with a single command.

## Quick Start

```bash
# Full pipeline for Milan FD_60 (regular + presegmented)
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60

# Aruba with baseline captions and CLIP encoder (default)
bash bash_scripts/prepare_data.sh --dataset aruba --sampling FD_60

# Cairo with Sourish captions
bash bash_scripts/prepare_data.sh --dataset cairo --sampling FD_60 --caption-style sourish

# Use GTE text encoder instead of CLIP
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60 --text-encoder gte_base
```

## Command-Line Arguments

### Required

| Argument | Description | Example |
|----------|-------------|---------|
| `--dataset` | Dataset name | `milan`, `aruba`, `cairo` |
| `--sampling` | Sampling strategy | `FD_60`, `FD_30`, `FD_120`, `FL_20`, `FL_50` |

### Optional

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--caption-style` | Caption generation style | `baseline` | `baseline`, `sourish` |
| `--text-encoder` | Text encoder config | `clip_vit_base` | See below |
| `--no-presegmented` | Skip presegmented (_p) version | false | flag |
| `--skip-sampling` | Skip data sampling step | false | flag |
| `--skip-captions` | Skip caption generation | false | flag |
| `--skip-encoding` | Skip text encoding | false | flag |
| `--device` | Device for encoding | `auto` | `cpu`, `cuda`, `mps` |
| `--help` | Show help message | - | flag |

### Available Text Encoders

- `clip_vit_base` (default) - CLIP text encoder, 512-dim
- `gte_base` - GTE text embeddings, 768-dim
- `gte_base_projected` - GTE with projection to 512-dim
- `distilroberta_base` - DistilRoBERTa, 768-dim
- `minilm_l6` - MiniLM (fast), 384-dim
- `siglip_base` - SigLIP text encoder, 768-dim
- `llama_embed_8b` - LLaMA embeddings, large
- `embeddinggemma_300m` - Gemma embeddings

## Common Use Cases

### 1. Full Pipeline (Most Common)

Run all three steps for both regular and presegmented versions:

```bash
bash bash_scripts/prepare_data.sh \
    --dataset milan \
    --sampling FD_60
```

**What it does:**
- Samples: `milan_FD_60` and `milan_FD_60_p`
- Generates baseline captions for train/val/test
- Encodes captions with CLIP text encoder

### 2. Only Regular Version (No Presegmented)

```bash
bash bash_scripts/prepare_data.sh \
    --dataset aruba \
    --sampling FD_60 \
    --no-presegmented
```

**What it does:**
- Only processes `aruba_FD_60` (skips `aruba_FD_60_p`)

### 3. Different Caption Style

```bash
bash bash_scripts/prepare_data.sh \
    --dataset milan \
    --sampling FD_60 \
    --caption-style sourish
```

**What it does:**
- Uses Sourish-style captions instead of baseline
- Config file: `configs/captions/sourish_milan.yaml`

### 4. Different Text Encoder

```bash
bash bash_scripts/prepare_data.sh \
    --dataset milan \
    --sampling FD_60 \
    --text-encoder gte_base
```

**What it does:**
- Uses GTE instead of CLIP for text encoding
- Config file: `configs/text_encoders/gte_base.yaml`

### 5. Skip Sampling (Data Already Exists)

If you already have sampled data and only want to regenerate captions + embeddings:

```bash
bash bash_scripts/prepare_data.sh \
    --dataset milan \
    --sampling FD_60 \
    --skip-sampling
```

**Use case:** You changed caption generation code and want to regenerate captions

### 6. Skip Captions (Captions Already Exist)

If you have captions but want to regenerate embeddings with a different encoder:

```bash
bash bash_scripts/prepare_data.sh \
    --dataset milan \
    --sampling FD_60 \
    --skip-sampling \
    --skip-captions \
    --text-encoder gte_base
```

**Use case:** You want to try a different text encoder without regenerating everything

### 7. Multiple Sampling Strategies

To prepare multiple sampling strategies, run the script multiple times:

```bash
# FD_60
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60

# FD_30
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_30

# FD_120
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_120
```

Or create a simple loop:

```bash
for sampling in FD_30 FD_60 FD_120; do
    bash bash_scripts/prepare_data.sh \
        --dataset milan \
        --sampling $sampling
done
```

## Output Structure

After running the script, your data will be organized as:

```
data/processed/casas/milan/
├── FD_60/
│   ├── train.json                           # Sampled sensor data
│   ├── val.json
│   ├── test.json
│   ├── train_captions_baseline.json         # Generated captions
│   ├── val_captions_baseline.json
│   ├── test_captions_baseline.json
│   ├── train_embeddings_baseline_clip.npz   # Text embeddings
│   ├── val_embeddings_baseline_clip.npz
│   └── test_embeddings_baseline_clip.npz
└── FD_60_p/                                  # Presegmented version
    └── (same structure)
```

## Workflow Examples

### Complete Data Generation (New Dataset)

```bash
# Generate all data for Milan FD_60
bash bash_scripts/prepare_data.sh \
    --dataset milan \
    --sampling FD_60

# Expected runtime:
# - Sampling: ~2-5 minutes
# - Captions: ~1-2 minutes
# - Encoding: ~1-3 minutes (CPU) or ~30s (GPU)
```

### Regenerate After Code Changes

#### Changed caption generation code:

```bash
bash bash_scripts/prepare_data.sh \
    --dataset milan \
    --sampling FD_60 \
    --skip-sampling
```

#### Changed text encoder config:

```bash
bash bash_scripts/prepare_data.sh \
    --dataset milan \
    --sampling FD_60 \
    --skip-sampling \
    --skip-captions
```

### Generate Multiple Text Encodings

Compare different text encoders by generating embeddings with each:

```bash
# CLIP embeddings (default)
bash bash_scripts/prepare_data.sh \
    --dataset milan \
    --sampling FD_60

# GTE embeddings
bash bash_scripts/prepare_data.sh \
    --dataset milan \
    --sampling FD_60 \
    --skip-sampling \
    --skip-captions \
    --text-encoder gte_base

# MiniLM embeddings
bash bash_scripts/prepare_data.sh \
    --dataset milan \
    --sampling FD_60 \
    --skip-sampling \
    --skip-captions \
    --text-encoder minilm_l6
```

Now you have multiple text embeddings to compare during training!

### Generate Both Caption Styles

```bash
# Baseline captions
bash bash_scripts/prepare_data.sh \
    --dataset milan \
    --sampling FD_60

# Sourish captions (reusing sampled data)
bash bash_scripts/prepare_data.sh \
    --dataset milan \
    --sampling FD_60 \
    --skip-sampling \
    --caption-style sourish
```

## Error Handling

The script will:
- ✓ Exit immediately on any error (thanks to `set -e`)
- ✓ Check if config files exist before running
- ✓ Warn if data directories are missing
- ✓ Show clear success/failure messages for each step

### Common Errors

**"Config file not found"**
```bash
# Solution: Check that sampling config exists
ls configs/sampling/${DATASET}_${SAMPLING}.yaml
```

**"Data directory not found"**
```bash
# Solution: Don't skip sampling, or check data path
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60
# (without --skip-sampling)
```

**"Caption config not found"**
```
# Warning only - script will use command-line args instead
# Create config file if you want consistent settings:
configs/captions/${CAPTION_STYLE}_${DATASET}.yaml
```

## Integration with Training

After running this script, you're ready for training:

```bash
# 1. Prepare data (you just did this!)
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60

# 2. Train alignment model
python train.py --config configs/alignment/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1.yaml
```

## Comparison with Old Workflow

### Old Way (Manual)

```bash
# Step 1: Sample data
python sample_data.py --config configs/sampling/milan_FD_60.yaml
python sample_data.py --config configs/sampling/milan_FD_60_p.yaml

# Step 2: Generate captions
python src/captions/generate_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --caption-style baseline --dataset-name milan --split all
python src/captions/generate_captions.py \
    --data-dir data/processed/casas/milan/FD_60_p \
    --caption-style baseline --dataset-name milan --split all

# Step 3: Encode captions
python src/text_encoders/encode_captions.py \
    --data-dir data/processed/casas/milan/FD_60 \
    --caption-style baseline --split all \
    --config configs/text_encoders/clip_vit_base.yaml
python src/text_encoders/encode_captions.py \
    --data-dir data/processed/casas/milan/FD_60_p \
    --caption-style baseline --split all \
    --config configs/text_encoders/clip_vit_base.yaml
```

**Total: 6 commands!**

### New Way (Automated)

```bash
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60
```

**Total: 1 command!** 🎉

## Tips

1. **Use `--skip-sampling` liberally** - Sampling is usually the slowest step. If your data exists, skip it.

2. **Test with `--no-presegmented` first** - Process just the regular version to verify everything works.

3. **Check output files** - After each run, verify files were created:
   ```bash
   ls data/processed/casas/milan/FD_60/
   ```

4. **Use GPU for encoding** - If available, add `--device cuda` for 5-10x speedup on text encoding.

5. **Create your own wrapper** - For repeated experiments, create a simple script:
   ```bash
   #!/bin/bash
   for dataset in milan aruba cairo; do
       bash bash_scripts/prepare_data.sh \
           --dataset $dataset \
           --sampling FD_60
   done
   ```

## See Also

- `docs/CAPTION_GENERATION_GUIDE.md` - Caption generation details
- `docs/TEXT_ENCODER_GUIDE.md` - Text encoding details
- `docs/SAMPLING_UPDATES.md` - Sampling strategy details
- `bash_scripts/generate_all_data_milan.sh` - Old batch generation script

