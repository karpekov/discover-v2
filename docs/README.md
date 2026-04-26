# Smart-Home Event Sequence Alignment (discover-v2)

A PyTorch implementation for aligning smart-home event sequences to text via CLIP-style contrastive learning, implementing Recipe R2 for HAR clustering research.

## Project Overview

This project implements a dual-tower architecture that learns to align sensor event sequences with natural language descriptions:

- **Text Tower**: Frozen `thenlper/gte-base` encoder (768-d, L2-normalized)
- **Sensor Tower**: Custom Transformer with ALiBi attention, categorical embeddings, and Fourier features
- **Training**: Bidirectional InfoNCE (CLIP) loss + multi-field MLM with span masking
- **Retrieval**: FAISS-based similarity search for text-to-sensor and sensor-to-text queries

## Multi-Household Training (all_casas)

In addition to per-house training (milan / aruba / cairo), the pipeline supports
training on all three houses simultaneously using a combined dataset.

### Generating the combined dataset

```bash
conda activate discover-v2-env
python scripts/combine_casas_datasets.py
```

This produces two separate subfolders mirroring the per-house structure:

```
data/processed/casas/all_casas/
  FD_60/
    vocab.json
    train.json / val.json / test.json
    {split}_captions_baseline.json
    {split}_embeddings_baseline_clip.npz
    val_{house}.json / test_{house}.json   ← per-household eval subsets
  FD_60_p/
    (same layout)
```

Key transformations:

- **Sensor IDs are house-prefixed** (`M001` → `milan_M001`, `aruba_M001`) so
  physically different sensors with the same raw name get distinct vocab tokens.
- Each sample gains a `household` field (top-level and inside `metadata`) to
  support per-household evaluation filtering.
- Per-household test/val subsets (`test_{house}.json`, `val_{house}.json`) are
  pre-generated for easy per-house evaluation without code changes.

### Training on all_casas

```bash
# Uses FD_60 combined dataset by default
python train.py --config configs/alignment/all_casas_fd60_seq_rb1_textclip_projmlp_clipmlm_v3.yaml
```

### Evaluation

```bash
# Whole combined test set
python src/evals/evaluate_embeddings.py \
    --checkpoint trained_models/all_casas/.../best_model.pt \
    --test_data  data/processed/casas/all_casas/FD_60/test.json \
    --vocab      data/processed/casas/all_casas/FD_60/vocab.json \
    --output_dir results/evals/all_casas/combined

# Per-household (example: milan only)
python src/evals/evaluate_embeddings.py \
    --checkpoint trained_models/all_casas/.../best_model.pt \
    --test_data  data/processed/casas/all_casas/FD_60/test_milan.json \
    --vocab      data/processed/casas/all_casas/FD_60/vocab.json \
    --output_dir results/evals/all_casas/milan
```

---

## Current Project Structure (Post-Reorganization)

```
discover-v2/
├── src/                          # Main source code
│   ├── data/                     # Data processing pipeline
│   │   ├── data_load_clean.py    # Raw CASAS data processing
│   │   ├── data_loader.py        # Unified data loading
│   │   ├── generate_data.py      # Data generation scripts
│   │   └── ...
│   ├── models/                   # Neural network architectures
│   │   ├── text_encoder.py       # Frozen GTE-base text encoder
│   │   ├── sensor_encoder.py     # Custom Transformer with ALiBi
│   │   ├── chronos_encoder.py    # Chronos-2 time series encoder
│   │   ├── scan_model.py         # SCAN clustering model
│   │   └── ...
│   ├── training/                 # Training scripts
│   │   ├── train_clip.py         # Main CLIP training
│   │   ├── train_chronos_clip.py # Chronos-2 CLIP training
│   │   ├── train_scan.py         # SCAN clustering training
│   │   └── ...
│   ├── evals/                    # Evaluation scripts
│   │   ├── evaluate_embeddings.py
│   │   ├── visualize_embeddings.py
│   │   ├── scan_evaluation.py
│   │   └── ...
│   ├── utils/                    # Utility functions
│   └── losses/                   # Loss functions
├── configs/                      # Configuration files
│   ├── training/milan/           # Training configs (JSON)
│   └── data_generation/milan/   # Data generation presets
├── data/                         # Data directories
│   ├── raw/casas/               # Raw CASAS datasets
│   └── processed/casas/milan/    # Processed data
├── trained_models/milan/         # Model checkpoints
├── logs/                         # Training logs
│   ├── text/                    # Text logs
│   └── wandb/                   # WandB logs
├── results/evals/milan/          # Evaluation results
├── docs/                         # Documentation
└── AGENTS.md                     # Agent instructions
```

## Quick Start

### 1. Environment Setup
```bash
# Create conda environment
conda env create -f env.yaml
conda activate discover-v2-env
```

### 2. Data Generation
```bash
# Generate Milan training data
python src/data/generate_data.py --config training_50 --force
```

### 3. Training
```bash
# Train CLIP model
python src/training/train_clip.py --config configs/training/milan/tiny_50_oct1.json

# Train Chronos-2 encoder with CLIP alignment
python src/training/train_chronos_clip.py --config configs/training/milan/chronos_clip.json

# Train SCAN clustering model
python src/training/train_scan.py --config configs/training/milan/baseline.json
```

### 4. Evaluation
```bash
# Evaluate embeddings
python src/evals/evaluate_embeddings.py \
    --checkpoint trained_models/milan/tiny_50/best_model.pt \
    --train_data data/processed/casas/milan/training_50/train.json \
    --test_data data/processed/casas/milan/training_50/presegmented_test.json \
    --vocab data/processed/casas/milan/training_50/vocab.json \
    --output_dir results/evals/milan/tiny_50
```

## Architecture Details

### Sensor Encoder Options

**1. Transformer Encoder (sensor_encoder.py)**
- Sum of categorical field embeddings: `sensor_id + room_id + event_type + sensor_type + tod_bucket + delta_t_bucket + [floor_id] + [dow]`
- Fourier features for continuous (x,y) coordinates (L=12 bands)
- Log-bucketed time delta embeddings
- ALiBi positional bias (default)
- Optional RoPE for time/space (configurable)
- Pre-LN architecture (6-8 layers, 8 heads, d=768)
- Sequence pooling: 0.5×CLS + 0.5×mean(masked tokens)

**2. Chronos-2 Encoder (chronos_encoder.py)**
- Uses Amazon's Chronos-2 time series foundation model (frozen)
- Converts sensor sequences to multivariate time series
- Trainable MLP projection head (256-dim hidden, 512-dim output)
- Only projection head is trainable; Chronos model remains frozen
- Compatible with CLIP alignment training

### Training Objectives

**Standard Training (train_clip.py)**
- **CLIP Loss**: Bidirectional InfoNCE with learnable temperature (init=0.05)
- **MLM Loss** (λ=0.3): Multi-field masked language modeling
  - ~25% span masking (Poisson length≈3)
  - Field-balanced priors: room(.30), event_type(.20), sensor_id(.20), tod(.15), delta_t_bucket(.10), sensor_type(.05)
  - BERT-style 80/10/10 masking

**Chronos-2 Training (train_chronos_clip.py)**
- **CLIP Loss Only**: Bidirectional InfoNCE with learnable temperature
- No MLM: Only CLIP-style contrastive learning
- Only the MLP projection head is trainable; Chronos-2 model is frozen

## Configuration System

The project uses a two-tier configuration system:

### Training Configs (`configs/training/milan/`)
- JSON files defining model architecture, training parameters, and data paths
- Examples: `baseline.json`, `tiny_50_oct1.json`, `gemma_50.json`

### Data Generation Configs (`configs/data_generation/milan/`)
- JSON presets for data processing pipelines
- Examples: `training_20.json`, `training_50.json`, `presegmented.json`

## Key Features

- **Multi-Dataset Support**: Milan, Aruba, Cairo, Tulum2009, twor.2009
- **Flexible Architecture**: Configurable transformer layers, attention mechanisms
- **Comprehensive Evaluation**: Embedding quality, clustering, retrieval metrics
- **SCAN Integration**: Clustering-based activity recognition
- **WandB Integration**: Experiment tracking and visualization
- **Device Support**: CUDA, Apple Silicon (MPS), CPU

## Documentation

- `docs/README.md` - This file (project overview)
- `docs/PIPELINE_ORGANIZATION.md` - Detailed pipeline structure
- `docs/README_DATA_GENERATION.md` - Data generation guide
- `docs/CAPTION_STYLES.md` - Caption generation styles (baseline vs. Sourish)
- `docs/LABEL_DESCRIPTION_STYLES.md` - Label description styles for text-only evaluation
- `docs/SOURISH_COMPARISON_RESULTS.md` - Comparison of text encoders and label description styles
- `docs/WANDB_SETUP_GUIDE.md` - WandB configuration
- `docs/RETRIEVAL_GUIDE.md` - Retrieval system usage
- `docs/AGENT_GUIDE.md` - Guide for AI agents
- `AGENTS.md` - Quick reference for AI agents

## Recent Changes (Post-Reorganization)

- **Directory Structure**: Renamed `src-v2/` → `src/` for cleaner organization
- **Configuration Management**: Moved to `configs/` with JSON-based presets
- **Results Organization**: Centralized in `results/evals/{city_name}/`
- **Model Storage**: Organized by city in `trained_models/{city_name}/`
- **Log Management**: Separated text and WandB logs in `logs/`

## Requirements

- Python 3.8+
- PyTorch ≥ 1.12
- transformers ≥ 4.20
- faiss-cpu or faiss-gpu
- scikit-learn, numpy, pandas
- Optional: wandb for experiment tracking

See `env.yaml` for complete conda environment specification.
