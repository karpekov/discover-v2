# Pipeline Organization

## Overview
This document describes the reorganized structure of the smart-home event sequence alignment pipeline after reorganization on September 24, 2025.

## Directory Structure

### Core Pipeline (`src/`)
Contains the main implementation modules:
- **`models/`**: Neural network architectures
  - `text_encoder.py` - Frozen GTE-base text encoder
  - `sensor_encoder.py` - Custom Transformer with ALiBi attention
  - `mlm_heads.py` - Multi-field MLM heads
- **`losses/`**: Loss functions
  - `clip.py` - CLIP + combined loss implementation
- **`dataio/`**: Data loading and processing
  - `dataset.py` - Smart-home dataset implementation
  - `collate.py` - Batch collation with masking
- **`data/`**: Data processing pipeline modules
- **`config/`**: Configuration management
- **`utils/`**: Utility functions
- **`experiments/`**: Experiment configurations

### Scripts (`scripts/`)
Main entry point scripts:
- **`train.py`** - Main training script with full configuration support

### Core Pipeline (`src/`) - Evaluation Module
- **`evals/`**: All evaluation-related scripts and results
  - **`eval_retrieval.py`** - FAISS retrieval evaluation with metrics
  - **`query_retrieval.py`** - Interactive query interface
  - **`evaluate_embeddings.py`** - Embedding quality evaluation
  - **`visualize_embeddings.py`** - t-SNE/UMAP visualization
  - **`summarize_evaluation.py`** - Evaluation result summarization
  - **`embedding_evaluation/`** - Evaluation results and confusion matrices
  - **`embedding_visualizations/`** - t-SNE/UMAP visualization outputs

### Data Evaluation (`evals/`) - Root Level (Git Ignored)
Contains large data files and experimental results not tracked by git:
- **`data_for_deepcasas/`** - Preprocessed data for DeepCASAS comparison
- **`data_for_tdost/`** - Preprocessed data for TDOST comparison
- **Various CSV and analysis files** - Clustering metrics and ground truth data

### Models (`models/`)
Trained model checkpoints and outputs:
- **`outputs/milan_training/`** - Training outputs
  - `best_model.pt` - Best performing checkpoint
  - `final_model.pt` - Final training checkpoint

### Logs (`logs/`)
Training and execution logs:
- **`training_log.txt`** - Main training log
- **`milan_training_train.log`** - Milan-specific training log

### Configuration (`config/`)
Configuration files:
- **`config.json`** - Main model and training configuration

### Documentation (`docs/`)
Documentation files:
- **`README.md`** - Main project documentation
- **`RETRIEVAL_GUIDE.md`** - Guide for using the retrieval system
- **`PIPELINE_ORGANIZATION.md`** - This file

## Workflow

### 1. Training
```bash
cd scripts
python train.py \
    --train_data ../data/processed/milan_train.json \
    --vocab ../data/processed/milan_vocab.json \
    --config ../config/config.json \
    --output_dir ../models/outputs/milan_training
```

### 2. Evaluation
```bash
cd src/evals
python eval_retrieval.py \
    --checkpoint ../../models/outputs/milan_training/best_model.pt \
    --eval_data ../../data/processed/milan_test.json \
    --vocab ../../data/processed/milan_vocab.json
```

### 3. Interactive Querying
```bash
cd src/evals
python query_retrieval.py --interactive
```

### 4. Visualization
```bash
cd src/evals
python visualize_embeddings.py \
    --checkpoint ../../models/outputs/milan_training/best_model.pt \
    --data ../../data/processed/milan_test.json \
    --output_dir embedding_visualizations/
```

## Key Benefits of Reorganization

1. **Clear Separation of Concerns**
   - Core implementation in `src/`
   - Entry points in `scripts/`
   - Evaluation tools in `evals/`
   - Model artifacts in `models/`

2. **Improved Maintainability**
   - Related files grouped together
   - Clear dependency structure
   - Centralized configuration

3. **Better Development Workflow**
   - Easy to find relevant files
   - Logical execution flow
   - Separated logs and outputs

4. **Research-Friendly Structure**
   - All evaluation tools in one place
   - Easy comparison of methods
   - Organized experimental results

## Migration Notes

- All evaluation scripts now include proper import path adjustments
- Training script moved to `scripts/` with updated imports
- Configuration centralized in `config/` folder
- Model outputs organized in `models/` folder
- Logs separated from source code

## Next Steps

1. Update any external scripts that reference old paths
2. Consider creating a `requirements.txt` in the root directory
3. Add integration tests for the reorganized structure
4. Update any CI/CD pipelines to use new paths
