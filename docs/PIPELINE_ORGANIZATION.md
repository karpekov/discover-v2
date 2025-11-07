# Pipeline Organization

## Overview
This document describes the current structure of the smart-home event sequence alignment pipeline after major reorganization (October 2024).

## Current Directory Structure

### Core Pipeline (`src/`)
Contains the main implementation modules:

- **`data/`**: Complete data processing pipeline
  - `data_load_clean.py` - Raw CASAS data processing and cleaning
  - `data_loader.py` - Unified data loading with ProcessedDataset
  - `generate_data.py` - Data generation scripts with config presets
  - `data_config.py` - ProcessingConfig and data configuration
  - `datasets.py` - Dataset configurations and metadata
  - `pipeline.py` - End-to-end data processing pipeline

- **`models/`**: Neural network architectures
  - `text_encoder.py` - Frozen GTE-base text encoder
  - `sensor_encoder.py` - Custom Transformer with ALiBi attention
  - `chronos_encoder.py` - Chronos-2 time series encoder (frozen Chronos + trainable MLP)
  - `scan_model.py` - SCAN clustering model
  - `mlm_heads.py` - Multi-field MLM heads
  - `classification_head.py` - Classification heads
  - `text_encoders/` - Specialized text encoders (Gemma, cached)

- **`training/`**: Training scripts
  - `train_clip.py` - Main CLIP training with dual-encoder alignment
  - `train_chronos_clip.py` - Chronos-2 encoder CLIP training (CLIP only, no MLM)
  - `train_scan.py` - SCAN clustering training
  - `classification_trainer.py` - Classification training

- **`evals/`**: Evaluation and analysis scripts
  - `evaluate_embeddings.py` - Embedding quality evaluation
  - `visualize_embeddings.py` - t-SNE/UMAP visualization
  - `scan_evaluation.py` - SCAN clustering evaluation
  - `evaluate_checkpoints.py` - Checkpoint evaluation
  - `embedding_alignment_analysis.py` - Alignment analysis
  - `caption_alignment_analysis.py` - Caption alignment analysis
  - `run_all_evals.py` - Comprehensive evaluation suite

- **`utils/`**: Utility functions
  - `process_data_configs.py` - Data configuration helpers
  - `extract_milan_captions.py` - Caption extraction
  - `debug_training.py` - Training debugging utilities
  - `wandb_config.py` - WandB configuration

- **`losses/`**: Loss functions
  - `clip.py` - CLIP + combined loss implementation
  - `scan_loss.py` - SCAN clustering loss

- **`dataio/`**: Data loading and processing
  - `dataset.py` - Smart-home dataset implementation
  - `collate.py` - Batch collation with masking
  - `classification_dataset.py` - Classification datasets
  - `scan_dataset.py` - SCAN-specific datasets

### Configuration System (`configs/`)

- **`training/milan/`**: Training configuration files (JSON)
  - `baseline.json` - Standard baseline configuration
  - `tiny_50_oct1.json` - Tiny model with 50% data
  - `gemma_50.json` - Gemma text encoder configuration
  - `improved_mlm.json` - Improved MLM configuration
  - And more...

- **`data_generation/milan/`**: Data generation presets (JSON)
  - `training_20.json` - 20% data generation preset
  - `training_50.json` - 50% data generation preset
  - `presegmented.json` - Presegmented data preset
  - `quick_validation.json` - Quick validation preset

### Data Organization (`data/`)

- **`raw/casas/`**: Raw CASAS datasets
  - `milan/` - Milan dataset
  - `aruba/` - Aruba dataset
  - `cairo/` - Cairo dataset
  - `tulum2009/` - Tulum 2009 dataset
  - `twor.2009/` - twor.2009 dataset

- **`processed/casas/milan/`**: Processed data
  - `training_20/` - 20% training data
  - `training_50/` - 50% training data
  - `presegmented/` - Presegmented data
  - Various JSON files (train.json, test.json, vocab.json, etc.)

### Model Storage (`trained_models/`)

- **`milan/`**: Milan-specific model checkpoints
  - `baseline/` - Baseline model outputs
  - `tiny_50/` - Tiny model with 50% data
  - `gemma_50/` - Gemma model outputs
  - `scan_50clusters/` - SCAN clustering models
  - And more...

### Logs (`logs/`)

- **`text/`**: Text-based training logs
- **`wandb/`**: WandB experiment logs

### Results (`results/`)

- **`evals/milan/`**: Evaluation results organized by model
  - `baseline/` - Baseline evaluation results
  - `tiny_50/` - Tiny model evaluation results
  - `gemma_50/` - Gemma model evaluation results
  - Various analysis outputs (clustering, alignment, visualizations)

### Documentation (`docs/`)

- `README.md` - Main project documentation
- `PIPELINE_ORGANIZATION.md` - This file
- `README_DATA_GENERATION.md` - Data generation guide
- `WANDB_SETUP_GUIDE.md` - WandB configuration
- `RETRIEVAL_GUIDE.md` - Retrieval system usage

## Workflow Examples

### 1. Data Generation
```bash
# Generate Milan training data with 50% split
python src/data/generate_data.py --config training_50 --force

# Generate custom data configuration
python src/data/generate_data.py --custom --datasets milan --windows 20 50
```

### 2. Training
```bash
# Train CLIP model
python src/training/train_clip.py --config configs/training/milan/tiny_50_oct1.json

# Train SCAN clustering model
python src/training/train_scan.py --config configs/training/milan/baseline.json
```

### 3. Evaluation
```bash
# Evaluate embeddings
python src/evals/evaluate_embeddings.py \
    --checkpoint trained_models/milan/tiny_50/best_model.pt \
    --train_data data/processed/casas/milan/training_50/train.json \
    --test_data data/processed/casas/milan/training_50/presegmented_test.json \
    --vocab data/processed/casas/milan/training_50/vocab.json \
    --output_dir results/evals/milan/tiny_50

# Run comprehensive evaluation suite
python src/evals/run_all_evals.py \
    --checkpoint trained_models/milan/tiny_50/best_model.pt \
    --train_data data/processed/casas/milan/training_50/train.json \
    --test_data data/processed/casas/milan/training_50/presegmented_test.json \
    --vocab data/processed/casas/milan/training_50/vocab.json
```

### 4. Visualization
```bash
# Generate t-SNE visualizations
python src/evals/visualize_embeddings.py \
    --checkpoint trained_models/milan/tiny_50/best_model.pt \
    --train_data data/processed/casas/milan/training_50/train.json \
    --test_data data/processed/casas/milan/training_50/presegmented_test.json \
    --vocab data/processed/casas/milan/training_50/vocab.json \
    --output_dir results/evals/milan/tiny_50
```

## Key Benefits of Current Organization

1. **Clear Separation of Concerns**
   - Core implementation in `src/`
   - Configuration management in `configs/`
   - Data organization in `data/`
   - Results and logs properly separated

2. **Scalable Structure**
   - Easy to add new cities (just add to `configs/training/{city}/`)
   - Easy to add new model configurations
   - Centralized data processing pipeline

3. **Maintainable Codebase**
   - Related files grouped together
   - Clear dependency structure
   - JSON-based configuration system

4. **Research-Friendly**
   - All evaluation tools in `src/evals/`
   - Comprehensive result organization
   - Easy comparison of different models

## Migration History

- **October 2024**: Major reorganization from `src-v2/` to `src/`
- **Configuration System**: Moved from Python configs to JSON presets
- **Results Organization**: Centralized evaluation results in `results/evals/`
- **Model Storage**: Organized by city in `trained_models/{city}/`
- **Log Management**: Separated text and WandB logs

## Future Considerations

1. **Multi-City Support**: Easy to extend to other CASAS cities
2. **Model Variants**: Simple to add new architecture configurations
3. **Evaluation Metrics**: Comprehensive evaluation suite already in place
4. **Documentation**: Self-documenting structure with clear organization
