# Agent Documentation Guide

## For Future AI Agents

This document provides essential information for AI agents working on the discover-v2 project.

## Project Context

**discover-v2** is a PyTorch implementation for aligning smart-home event sequences to text via CLIP-style contrastive learning. It implements Recipe R2 for HAR (Human Activity Recognition) clustering research.

## Key Architecture

- **Dual-Tower Architecture**: Text encoder (frozen GTE-base) + Sensor encoder (custom Transformer)
- **Training**: Bidirectional InfoNCE (CLIP) loss + multi-field MLM with span masking
- **Datasets**: CASAS smart-home datasets (Milan, Aruba, Cairo, Tulum2009, twor.2009)
- **Evaluation**: Comprehensive evaluation suite with clustering, retrieval, and alignment metrics

## Critical Files to Understand

### Core Implementation
- `src/data/generate_data.py` - Main data generation script
- `src/training/train_clip.py` - Main CLIP training script
- `src/training/train_scan.py` - SCAN clustering training
- `src/evals/evaluate_embeddings.py` - Primary evaluation script

### Configuration System
- `configs/training/milan/` - Training configurations (JSON)
- `configs/data_generation/milan/` - Data generation presets (JSON)
- `src/utils/process_data_configs.py` - Configuration helpers

### Data Structure
- `data/raw/casas/` - Raw CASAS datasets
- `data/processed/casas/milan/` - Processed data
- `trained_models/milan/` - Model checkpoints
- `results/evals/milan/` - Evaluation results

## Common Tasks

### Data Generation
```bash
conda activate discover-v2-env
python src/data/generate_data.py --config training_50 --force
```

### Training
```bash
conda activate discover-v2-env
python src/training/train_clip.py --config configs/training/milan/tiny_50_oct1.json
```

### Evaluation
```bash
conda activate discover-v2-env
python src/evals/evaluate_embeddings.py \
    --checkpoint trained_models/milan/tiny_50/best_model.pt \
    --train_data data/processed/casas/milan/training_50/train.json \
    --test_data data/processed/casas/milan/training_50/presegmented_test.json \
    --vocab data/processed/casas/milan/training_50/vocab.json \
    --output_dir results/evals/milan/tiny_50
```

## Important Notes

1. **Always use conda environment**: `conda activate discover-v2-env`
2. **Configuration files are JSON**: Located in `configs/` directory
3. **Results go to specific directories**: `results/evals/{city_name}/`
4. **Models organized by city**: `trained_models/{city_name}/`
5. **Data processing is modular**: Use `src/data/generate_data.py` for data generation

## Recent Changes (October 2024)

- Renamed `src-v2/` → `src/` for cleaner organization
- Moved configurations to `configs/` with JSON-based presets
- Centralized results in `results/evals/{city_name}/`
- Organized models by city in `trained_models/{city_name}/`
- Separated logs into `logs/text/` and `logs/wandb/`

## When Making Changes

1. **Consult existing docs**: Always check `docs/` folder first
2. **Update documentation**: Update relevant docs when making structural changes
3. **Test imports**: Verify imports work with `conda activate discover-v2-env`
4. **Follow patterns**: Use existing code patterns and organization
5. **Update configs**: Modify JSON configs rather than hardcoding values

## Troubleshooting

- **Import errors**: Ensure `conda activate discover-v2-env` is active
- **Path issues**: Check that paths match current structure (use `src/` not `src-v2/`)
- **Config errors**: Verify JSON syntax in `configs/` files
- **Data issues**: Check data exists in `data/processed/casas/milan/`

## Key Dependencies

- PyTorch ≥ 1.12
- transformers ≥ 4.20
- faiss-cpu or faiss-gpu
- scikit-learn, numpy, pandas
- Optional: wandb for experiment tracking

See `env.yaml` for complete environment specification.
