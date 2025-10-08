# Smart-Home Event Sequence Alignment (discover-v2)

A PyTorch implementation for aligning smart-home event sequences to text via CLIP-style contrastive learning, implementing Recipe R2 for HAR clustering research.

## Project Overview

This project implements a dual-tower architecture that learns to align sensor event sequences with natural language descriptions:

- **Text Tower**: Frozen `thenlper/gte-base` encoder (768-d, L2-normalized)
- **Sensor Tower**: Custom Transformer with ALiBi attention, categorical embeddings, and Fourier features
- **Training**: Bidirectional InfoNCE (CLIP) loss + multi-field MLM with span masking
- **Retrieval**: FAISS-based similarity search for text-to-sensor and sensor-to-text queries

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
│   │   ├── scan_model.py         # SCAN clustering model
│   │   └── ...
│   ├── training/                 # Training scripts
│   │   ├── train_clip.py         # Main CLIP training
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

### Sensor Token Embedding
- Sum of categorical field embeddings: `sensor_id + room_id + event_type + sensor_type + tod_bucket + delta_t_bucket + [floor_id] + [dow]`
- Fourier features for continuous (x,y) coordinates (L=12 bands)
- Log-bucketed time delta embeddings

### Transformer Features
- ALiBi positional bias (default)
- Optional RoPE for time/space (configurable)
- Pre-LN architecture (6-8 layers, 8 heads, d=768)
- Sequence pooling: 0.5×CLS + 0.5×mean(masked tokens)

### Training Objectives
- **CLIP Loss**: Bidirectional InfoNCE with learnable temperature (init=0.05)
- **MLM Loss** (λ=0.3): Multi-field masked language modeling
  - ~25% span masking (Poisson length≈3)
  - Field-balanced priors: room(.30), event_type(.20), sensor_id(.20), tod(.15), delta_t_bucket(.10), sensor_type(.05)
  - BERT-style 80/10/10 masking

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
