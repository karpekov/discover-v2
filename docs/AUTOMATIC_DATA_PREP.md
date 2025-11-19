# Automatic Data Preparation

This guide explains how to use the automatic data preparation system that generates all required data for training.

## Overview

The automatic data preparation system eliminates the need to manually run separate scripts for:
1. **Data Sampling** (FL/FD strategies)
2. **Caption Generation** (baseline/sourish styles)
3. **Text Embedding Generation** (CLIP, GTE, MiniLM, etc.)

It intelligently parses your training config to determine requirements and generates any missing data automatically.

## Quick Start

### Option 1: Integrated with Training (Recommended)

Add `--prepare-data` flag to your training command:

```bash
# Automatically prepare all data then train
python train.py \
    --config configs/alignment/milan_fixed20_v0.1.yaml \
    --prepare-data
```

This will:
1. Parse your config to determine data requirements
2. Check if data exists
3. Generate any missing data automatically
4. Generate both regular and presegmented versions
5. Start training

### Option 2: Standalone Data Preparation

Prepare data separately before training:

```bash
# Prepare all required data
python src/sampling/data_prep.py --config configs/alignment/milan_fixed20_v0.1.yaml

# Then train normally
python train.py --config configs/alignment/milan_fixed20_v0.1.yaml
```

## What It Does

### 1. Config Parsing

The system automatically extracts requirements from your config:

```yaml
# From this config:
train_data_path: data/processed/casas/milan/FL_20/train.json
train_text_embeddings_path: data/processed/casas/milan/FL_20/train_embeddings_baseline_clip.npz

# It determines:
# - Dataset: milan
# - Sampling: FL_20 (Fixed Length, 20 events)
# - Caption style: baseline
# - Text encoder: clip
# - Splits: train, val, test
```

### 2. Data Generation Pipeline

For each required dataset variant:

#### Step 1: Data Sampling
```bash
# Automatically runs:
python sample_data.py --config configs/sampling/milan_FL_20.yaml
```

Generates:
- `data/processed/casas/milan/FL_20/train.json`
- `data/processed/casas/milan/FL_20/val.json`
- `data/processed/casas/milan/FL_20/test.json`
- `data/processed/casas/milan/FL_20/vocab.json`

#### Step 2: Caption Generation
```bash
# Automatically runs:
python src/captions/generate_captions.py \
    --data-dir data/processed/casas/milan/FL_20 \
    --caption-style baseline \
    --dataset-name milan \
    --split all
```

Generates:
- `train_captions_baseline.json`
- `val_captions_baseline.json`
- `test_captions_baseline.json`

#### Step 3: Text Embedding Generation
```bash
# Automatically runs:
python src/text_encoders/encode_captions.py \
    --data-dir data/processed/casas/milan/FL_20 \
    --caption-style baseline \
    --split all \
    --config configs/text_encoders/clip_vit_base.yaml
```

Generates:
- `train_embeddings_baseline_clip.npz`
- `val_embeddings_baseline_clip.npz`
- `test_embeddings_baseline_clip.npz`

### 3. Presegmented Version

By default, also generates presegmented version (FL_20_p):
- Repeats all steps for presegmented data
- Uses presegmentation config (milan_FL_20_p.yaml)
- Useful for comparing presegmented vs. non-presegmented performance

## Command Line Options

### Training with Auto-Prep

```bash
# Prepare all data (including presegmented)
python train.py --config <config> --prepare-data

# Skip presegmented version
python train.py --config <config> --prepare-data --no-presegmented
```

### Standalone Preparation

```bash
# Prepare all data (including presegmented)
python src/sampling/data_prep.py --config <config>

# Skip presegmented version
python src/sampling/data_prep.py --config <config> --no-presegmented

# Verbose logging
python src/sampling/data_prep.py --config <config> --verbose
```

## Examples

### Example 1: Milan FL_20 with CLIP

```bash
# Config uses CLIP embeddings on FL_20 data
python train.py \
    --config configs/alignment/milan_fixed20_v0.1.yaml \
    --prepare-data
```

Automatically generates:
- `milan/FL_20/` - Regular sampling
- `milan/FL_20_p/` - Presegmented sampling
- Baseline captions for both
- CLIP embeddings for both

### Example 2: Milan FD_60 with GTE

```yaml
# Your config
train_data_path: data/processed/casas/milan/FD_60/train.json
train_text_embeddings_path: data/processed/casas/milan/FD_60/train_embeddings_baseline_gte.npz
```

```bash
python train.py \
    --config configs/alignment/milan_fixeddur60_gte.yaml \
    --prepare-data
```

Automatically generates:
- `milan/FD_60/` - Fixed duration 60s
- `milan/FD_60_p/` - Presegmented version
- Baseline captions for both
- GTE embeddings for both

### Example 3: Aruba with Multiple Encoders

```bash
# Prepare data for Aruba FL_20 with CLIP
python src/sampling/data_prep.py --config configs/alignment/aruba_fixed20_clip.yaml

# Then you can manually add other encoders
python src/text_encoders/encode_captions.py \
    --data-dir data/processed/casas/aruba/FL_20 \
    --caption-style baseline \
    --split all \
    --config configs/text_encoders/minilm_l6.yaml
```

## Programmatic Usage

You can also use the data preparation system in your own scripts:

```python
from src.sampling.data_prep import DataPreparer
from src.alignment.config import AlignmentConfig

# Load your config
config = AlignmentConfig.from_yaml('configs/alignment/milan_fixed20_v0.1.yaml')

# Create preparer
preparer = DataPreparer()

# Prepare all data
success = preparer.prepare_all_data(
    config,
    include_presegmented=True  # Also generate presegmented version
)

if success:
    print("✓ All data prepared!")
else:
    print("✗ Data preparation failed")
```

### Fine-Grained Control

```python
from src.sampling.data_prep import DataPreparer

preparer = DataPreparer()

# Just sampling
preparer.prepare_sampling('milan', 'FL_20')

# Just captions
preparer.prepare_captions('milan', 'FL_20', 'baseline')

# Just embeddings
preparer.prepare_embeddings('milan', 'FL_20', 'baseline', 'clip')
```

## Smart Behavior

### 1. Skips Existing Data
If data already exists, it's not regenerated:
```
✓ Sampled data already exists: data/processed/casas/milan/FL_20
✓ Captions already exist: data/processed/casas/milan/FL_20 (baseline)
✓ Embeddings already exist: data/processed/casas/milan/FL_20 (clip)
```

### 2. Validates Configs
Checks that required sampling/encoder configs exist:
```
⚠️  Sampling config not found: configs/sampling/milan_FL_30.yaml
⚠️  Text encoder config not found: configs/text_encoders/custom_encoder.yaml
```

### 3. Detailed Logging
Shows exactly what's being generated:
```
================================================================================
PREPARING: milan_FL_20
================================================================================
Generating sampled data: milan_FL_20
Command: python sample_data.py --config configs/sampling/milan_FL_20.yaml
✓ Successfully generated: data/processed/casas/milan/FL_20

Generating baseline captions: milan_FL_20
Command: python src/captions/generate_captions.py ...
✓ Successfully generated captions: data/processed/casas/milan/FL_20

Generating clip embeddings: milan_FL_20
Command: python src/text_encoders/encode_captions.py ...
✓ Successfully generated embeddings: data/processed/casas/milan/FL_20
```

## Troubleshooting

### Issue: Config parsing failed

**Symptom:**
```
Could not parse dataset/sampling strategy from config
```

**Solution:**
Make sure your paths follow the expected format:
```yaml
train_data_path: data/processed/casas/{dataset}/{strategy}/train.json
train_text_embeddings_path: data/processed/casas/{dataset}/{strategy}/train_embeddings_{caption_style}_{encoder}.npz
```

### Issue: Sampling config not found

**Symptom:**
```
Cannot find sampling config for milan_FL_30
```

**Solution:**
Create the missing config file:
```bash
# Copy an existing config and modify
cp configs/sampling/milan_FL_20.yaml configs/sampling/milan_FL_30.yaml
# Edit window_sizes: [30]
```

### Issue: Text encoder config not found

**Symptom:**
```
Cannot find text encoder config for: custom_encoder
```

**Solution:**
Use a supported encoder name or create a custom config:
- Supported: `clip`, `gte`, `minilm`, `distilroberta`, `llama`, `gemma`, `siglip`
- Or create: `configs/text_encoders/custom_encoder.yaml`

### Issue: Command failed during execution

**Symptom:**
```
Command failed with exit code 1
STDERR: <error message>
```

**Solution:**
1. Check the full error message in the logs
2. Try running the failed command manually to debug
3. Ensure all dependencies are installed (`conda activate discover-v2-env`)

## Best Practices

### 1. Use Prepared Data for Multiple Experiments

Prepare data once, reuse for multiple training runs:
```bash
# Prepare data once
python src/sampling/data_prep.py --config configs/alignment/milan_fixed20_v0.1.yaml

# Train multiple times with different hyperparameters
python train.py --config configs/alignment/milan_fixed20_v0.1.yaml
python train.py --config configs/alignment/milan_fixed20_v0.2.yaml  # Different hyperparams
```

### 2. Generate Multiple Encoders

Prepare base data, then add encoders as needed:
```bash
# Base preparation (includes CLIP)
python src/sampling/data_prep.py --config configs/alignment/milan_fixed20_clip.yaml

# Add more encoders
cd data/processed/casas/milan/FL_20
python ../../../../src/text_encoders/encode_captions.py \
    --data-dir . --caption-style baseline --split all \
    --config ../../../../configs/text_encoders/gte_base.yaml

python ../../../../src/text_encoders/encode_captions.py \
    --data-dir . --caption-style baseline --split all \
    --config ../../../../configs/text_encoders/minilm_l6.yaml
```

### 3. Organize by Dataset

Keep data organized by dataset and strategy:
```
data/processed/casas/
├── milan/
│   ├── FL_20/
│   ├── FL_20_p/
│   ├── FL_50/
│   ├── FD_60/
│   └── FD_60_p/
└── aruba/
    ├── FL_20/
    └── FL_50/
```

### 4. Check Before Long Training Runs

For long experiments, verify data first:
```bash
# Prepare and verify data
python src/sampling/data_prep.py --config configs/alignment/milan_fixed20_v0.1.yaml

# Check that all files exist
ls -lh data/processed/casas/milan/FL_20/

# Then start training
python train.py --config configs/alignment/milan_fixed20_v0.1.yaml
```

## Integration with Training Pipeline

The data preparation integrates seamlessly with the training pipeline:

```bash
# Option 1: Automatic preparation during training
python train.py --config <config> --prepare-data

# Option 2: Manual preparation, then training
python src/sampling/data_prep.py --config <config>
python train.py --config <config>

# Both achieve the same result!
```

## Advanced Usage

### Custom Data Requirements

If you need custom data preparation logic:

```python
from src.sampling.data_prep import DataPreparer

class CustomPreparer(DataPreparer):
    def prepare_custom_data(self, dataset, strategy):
        # Your custom logic here
        pass

preparer = CustomPreparer()
preparer.prepare_all_data(config)
```

### Parallel Preparation

For large datasets or multiple configs:

```bash
# Prepare multiple datasets in parallel
python src/sampling/data_prep.py --config configs/alignment/milan_fixed20.yaml &
python src/sampling/data_prep.py --config configs/alignment/aruba_fixed20.yaml &
wait
```

## Summary

The automatic data preparation system:
- ✅ Eliminates manual data generation steps
- ✅ Intelligently parses config requirements
- ✅ Skips existing data (fast for reruns)
- ✅ Generates presegmented versions automatically
- ✅ Provides detailed logging and error messages
- ✅ Integrates with training pipeline
- ✅ Can be used standalone or programmatically

**For most users:**
```bash
python train.py --config <your_config> --prepare-data
```

That's it! 🚀

