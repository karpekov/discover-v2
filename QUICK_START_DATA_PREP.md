# Quick Start: Data Preparation

## One-Line Data Preparation 🚀

The `prepare_data.sh` script automates **Sample → Caption → Encode** in one command!

### Most Common Usage

```bash
# Full pipeline: Milan FD_60 (both regular + presegmented)
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60
```

This will:
1. ✓ Sample data: `milan_FD_60` and `milan_FD_60_p`
2. ✓ Generate baseline captions for train/val/test
3. ✓ Encode with CLIP text encoder

### Quick Examples

```bash
# Aruba FD_60
bash bash_scripts/prepare_data.sh --dataset aruba --sampling FD_60

# Cairo with Sourish captions
bash bash_scripts/prepare_data.sh --dataset cairo --sampling FD_60 --caption-style sourish

# Use GTE encoder instead of CLIP
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60 --text-encoder gte_base

# Skip presegmented version
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60 --no-presegmented

# Data exists, regenerate captions + embeddings only
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60 --skip-sampling
```

### Common Workflows

**New dataset from scratch:**
```bash
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60
```

**Changed caption generation code:**
```bash
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60 --skip-sampling
```

**Try different text encoder:**
```bash
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60 \
    --skip-sampling --skip-captions --text-encoder gte_base
```

**Multiple sampling strategies:**
```bash
for sampling in FD_30 FD_60 FD_120; do
    bash bash_scripts/prepare_data.sh --dataset milan --sampling $sampling
done
```

### Available Options

| Flag | Options | Default |
|------|---------|---------|
| `--dataset` | milan, aruba, cairo | *required* |
| `--sampling` | FD_60, FD_30, FD_120, FL_20, FL_50 | *required* |
| `--caption-style` | baseline, sourish | baseline |
| `--text-encoder` | clip_vit_base, gte_base, minilm_l6, etc. | clip_vit_base |
| `--no-presegmented` | Skip _p version | false |
| `--skip-sampling` | Skip sampling step | false |
| `--skip-captions` | Skip caption generation | false |
| `--skip-encoding` | Skip text encoding | false |
| `--device` | cpu, cuda, mps | auto |

### Get Help

```bash
bash bash_scripts/prepare_data.sh --help
```

### Output Location

```
data/processed/casas/{dataset}/{sampling}/
├── train.json                              # Sampled data
├── val.json
├── test.json
├── train_captions_baseline.json            # Captions
├── val_captions_baseline.json
├── test_captions_baseline.json
├── train_embeddings_baseline_clip.npz      # Embeddings
├── val_embeddings_baseline_clip.npz
└── test_embeddings_baseline_clip.npz
```

### Next Step: Training

```bash
# After data preparation, train your model:
python train.py --config configs/alignment/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1.yaml
```

---

**Full documentation:** `docs/AUTOMATIC_DATA_PREP_UPDATED.md`

