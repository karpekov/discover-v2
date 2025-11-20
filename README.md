# DISCOVER-v2

A PyTorch implementation for learning aligned representations of smart-home sensor sequences and natural language descriptions using CLIP-style contrastive learning.

## Project Structure

```
discover-v2/
├── src/                          # Source code
│   ├── alignment/                # CLIP-style alignment training
│   ├── captions/                 # Caption generation (baseline, sourish)
│   ├── data/                     # Data loading and preprocessing
│   ├── encoders/                 # Sensor encoders (transformer, image)
│   ├── evals/                    # Evaluation scripts
│   ├── models/                   # Model architectures
│   ├── sampling/                 # Data sampling strategies (FL/FD)
│   ├── text_encoders/            # Text encoders (CLIP, GTE, etc.)
│   └── training/                 # Training loops
├── configs/                      # Configuration files
│   ├── alignment/                # Alignment training configs (YAML)
│   ├── sampling/                 # Sampling configs (YAML)
│   ├── captions/                 # Caption generation configs
│   ├── encoders/                 # Sensor encoder configs
│   └── text_encoders/            # Text encoder configs
├── data/                         # Data directory
│   ├── raw/casas/                # Raw CASAS datasets
│   └── processed/casas/          # Processed data
├── trained_models/               # Model checkpoints
├── logs/                         # Training logs (text + wandb)
├── results/                      # Evaluation results
├── metadata/                     # Sensor coordinates & metadata
├── docs/                         # Detailed documentation
└── bash_scripts/                 # Automation scripts
```

## Quick Start

### 1. Environment Setup

```bash
# Create and activate conda environment
conda env create -f env.yaml
conda activate discover-v2-env
```

### 2. Data Preparation

#### Option A: Automated (Recommended)

Generate all data for a specific dataset:

```bash
# Generate all Milan data variations (FL/FD, captions, embeddings)
bash bash_scripts/generate_all_data_milan.sh

# Or for Aruba
bash bash_scripts/generate_all_data_aruba.sh
```

#### Option B: Manual Step-by-Step

```bash
# Step 1: Sample data (FL=Fixed Length, FD=Fixed Duration)
python sample_data.py --config configs/sampling/milan_FL_50.yaml

# Step 2: Generate captions
python src/captions/generate_captions.py \
    --data-dir data/processed/casas/milan/FL_50 \
    --caption-style baseline \
    --dataset-name milan \
    --split all

# Step 3: Encode captions to embeddings
python src/text_encoders/encode_captions.py \
    --data-dir data/processed/casas/milan/FL_50 \
    --caption-style baseline \
    --split all \
    --config configs/text_encoders/clip_vit_base.yaml
```

### 3. Training

```bash
# Train alignment model
python train.py --config configs/alignment/milan_fd60_seq_rb0_textclip_projmlp_clip_v1.yaml

# Train with automatic data preparation
python train.py --config configs/alignment/milan_fd60_seq_rb0_textclip_projmlp_clip_v1.yaml --prepare-data
```

### 4. Evaluation

```bash
# Evaluate trained model
python src/evals/evaluate_embeddings.py \
    --checkpoint trained_models/milan/my_model/best_model.pt \
    --train-data data/processed/casas/milan/FD_60/train.json \
    --test-data data/processed/casas/milan/FD_60/test.json \
    --vocab data/processed/casas/milan/FD_60/vocab.json \
    --output-dir results/evals/milan/my_model
```

## Sampling Strategies

- **Fixed Length (FL)**: Sample sequences with fixed number of events
  - `FL_20`: 20 events per sequence
  - `FL_50`: 50 events per sequence
  - `FL_20_p`: 20 events, presegmented by activity labels

- **Fixed Duration (FD)**: Sample sequences with fixed time windows
  - `FD_30`: 30-second windows
  - `FD_60`: 60-second windows
  - `FD_120`: 120-second windows
  - `FD_60_p`: 60-second, presegmented

All strategies available for both Milan and Aruba datasets in `configs/sampling/`.

## Caption Styles

- **Baseline**: Natural language descriptions with temporal/spatial context
- **Sourish**: Structured template-based format (when/duration/where/sensors)

Configure via `--caption-style` flag or configs in `configs/captions/`.

## Text Encoders

Available text encoders:
- `clip_vit_base`: OpenAI CLIP (512-dim)
- `gte_base`: GTE text encoder (768-dim)
- `minilm_l6`: Sentence transformer (384-dim)
- `distilroberta_base`: DistilRoBERTa (768-dim)
- `siglip_base`: SigLIP text encoder (768-dim)

Configs in `configs/text_encoders/`.

## Common Commands

```bash
# List available sampling configs
ls configs/sampling/

# Generate data for specific sampling strategy
python sample_data.py --config configs/sampling/milan_FD_60.yaml

# Generate captions with different styles
python src/captions/generate_captions.py --data-dir data/processed/casas/milan/FD_60 --caption-style baseline --dataset-name milan
python src/captions/generate_captions.py --data-dir data/processed/casas/milan/FD_60 --caption-style sourish --dataset-name milan

# Encode with different text encoders
python src/text_encoders/encode_captions.py --data-dir data/processed/casas/milan/FD_60 --caption-style baseline --config configs/text_encoders/clip_vit_base.yaml
python src/text_encoders/encode_captions.py --data-dir data/processed/casas/milan/FD_60 --caption-style baseline --config configs/text_encoders/gte_base.yaml

# Resume training from checkpoint
python train.py --config configs/alignment/my_config.yaml --resume trained_models/milan/my_model/checkpoint_step_5000.pt
```

## Configuration

Training configs (YAML) in `configs/alignment/` specify:
- Data paths (sampled data + text embeddings)
- Encoder architecture (transformer-based)
- Projection heads (linear or MLP)
- Loss weights (CLIP + optional MLM)
- Training hyperparameters (batch size, learning rate, etc.)
- WandB logging settings

See example configs for templates.

## Documentation

For detailed guides, see `docs/`:
- `AUTOMATIC_DATA_PREP.md` - Automated data preparation
- `ALIGNMENT_GUIDE.md` - Training alignment models
- `CAPTION_GENERATION_GUIDE.md` - Caption generation details
- `TEXT_ENCODER_GUIDE.md` - Text encoder usage
- `ENCODER_GUIDE.md` - Sensor encoder architectures
- `EVAL_SCRIPTS_UPDATES.md` - Evaluation tools

## Requirements

- Python 3.8+
- PyTorch ≥ 1.12
- transformers ≥ 4.20
- faiss-cpu/faiss-gpu
- wandb (optional, for experiment tracking)

See `env.yaml` for complete environment specification.

## Datasets

Project supports CASAS smart home datasets:
- Milan (primary)
- Aruba
- Cairo
- Tulum2009
- twor.2009

Raw data should be placed in `data/raw/casas/{dataset}/`.

## License

See LICENSE file for details.

