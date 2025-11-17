# Quick Start: Image-Based Sensor Encoder Training

Train sensor encoders using frozen vision model embeddings (CLIP, DINOv2, SigLIP) instead of raw sequences.

## What This Does

Instead of learning sensor embeddings from scratch, this approach:
1. Visualizes sensor activations on floor plans as images
2. Embeds images using pre-trained vision models (frozen)
3. Trains a transformer to process these rich representations
4. Aligns with text descriptions using CLIP loss

## Prerequisites

```bash
# Activate conda environment
conda activate discover-v2-env
```

## Complete Workflow (5 Commands)

### 1. Sample Data (if not done)

```bash
python sample_data.py \
    --config configs/sampling/milan_fixed_length_20_presegmented.yaml
```

**Output**: `data/processed/casas/milan/fixed_length_20_presegmented/train.json`

### 2. Generate Sensor Images

```bash
python -m src.encoders.sensor.image.generate_images \
    --dataset milan \
    --output-width 224 \
    --output-height 224
```

**Output**: `data/processed/casas/milan/layout_embeddings/images/dim224/`
**Time**: ~10 seconds (66 images for Milan)

### 3. Embed Images with Vision Model

```bash
# CLIP (512D) - recommended default
python -m src.encoders.sensor.image.embed_images \
    --dataset milan \
    --model clip \
    --batch-size 32
```

**Output**: `data/processed/casas/milan/layout_embeddings/embeddings/clip_base/dim224/embeddings.npz`
**Time**: ~30 seconds

**Other options**:
```bash
# DINOv2 (768D) - better spatial understanding
python -m src.encoders.sensor.image.embed_images --dataset milan --model dinov2

# SigLIP (768D) - improved CLIP
python -m src.encoders.sensor.image.embed_images --dataset milan --model siglip
```

### 4. Generate Text Embeddings

```bash
# Generate captions
python generate_captions.py \
    --config configs/captions/baseline_milan.yaml \
    --data-path data/processed/casas/milan/fixed_length_20_presegmented/train.json

python generate_captions.py \
    --config configs/captions/baseline_milan.yaml \
    --data-path data/processed/casas/milan/fixed_length_20_presegmented/test.json

# Encode captions
python encode_captions.py \
    --config configs/text_encoders/gte_base.yaml \
    --captions-path data/processed/casas/milan/fixed_length_20_presegmented/train_captions_baseline.json

python encode_captions.py \
    --config configs/text_encoders/gte_base.yaml \
    --captions-path data/processed/casas/milan/fixed_length_20_presegmented/test_captions_baseline.json
```

### 5. Train Image-Based Alignment Model

```bash
# Using CLIP embeddings
python train.py --config configs/alignment/milan_image_clip.yaml
```

**Output**: `trained_models/milan/alignment_image_clip/`
**Logs**: `logs/text/milan_image_clip/`
**WandB**: https://wandb.ai/your-entity/discover-v2

## Compare Different Vision Models

```bash
# Train with CLIP
python train.py --config configs/alignment/milan_image_clip.yaml

# Train with DINOv2
python train.py --config configs/alignment/milan_image_dinov2.yaml

# Compare results in WandB
```

## Configuration Examples

### Minimal Config (configs/encoders/transformer_image_clip.yaml)

```yaml
type: transformer
use_image_embeddings: true      # Enable image-based mode
image_model_name: clip          # Vision model: clip, dinov2, siglip
image_size: 224                 # Must match generated embeddings
freeze_input_projection: true   # Keep frozen (recommended)

d_model: 768
projection_dim: 512
```

### Full Training Config (configs/alignment/milan_image_clip.yaml)

```yaml
experiment_name: milan_image_clip
dataset: milan                  # REQUIRED for image-based
dataset_type: casas            # REQUIRED for image-based

encoder_config_path: configs/encoders/transformer_image_clip.yaml
train_data_path: data/processed/casas/milan/.../train.json
train_text_embeddings_path: data/processed/casas/milan/.../train_embeddings_baseline_gte_base.npz
# ... (rest of config)
```

## Key Differences from Sequence-Based

| Aspect | Sequence-Based | Image-Based |
|--------|---------------|-------------|
| **Input** | Learnable embeddings | Frozen vision embeddings |
| **Embedding Dim** | 256-1024 | 512 (CLIP) or 768 (DINOv2/SigLIP) |
| **Pre-compute** | None | Images + embeddings |
| **Training Speed** | 1.0x | 1.2x (faster) |
| **Spatial Info** | Via features | Implicit in embeddings |
| **MLM Loss** | Yes | No (frozen inputs) |

## Troubleshooting

### "Image embeddings not found"

**Solution**: Run steps 2 and 3 above to generate images and embeddings.

### "dataset parameter is required"

**Solution**: Add to alignment config:
```yaml
dataset: milan
dataset_type: casas
```

### Out of Memory

**Solution**: Reduce batch size:
```yaml
training:
  batch_size: 64  # Instead of 128
```

## Monitoring Training

```bash
# View logs
tail -f logs/text/milan_image_clip/training.log

# Check checkpoints
ls trained_models/milan/alignment_image_clip/

# Open WandB dashboard
# Navigate to: https://wandb.ai/your-entity/discover-v2/runs/...
```

## Results Location

After training:
```
trained_models/milan/alignment_image_clip/
├── best_model.pt           # Best validation checkpoint
├── final_model.pt          # Final checkpoint
├── checkpoint_epoch_10.pt  # Intermediate checkpoints
└── config.yaml             # Saved config

logs/text/milan_image_clip/
├── training.log            # Training logs
└── events.out.tfevents.*   # TensorBoard logs

logs/wandb/milan_image_clip/
└── latest-run/             # WandB logs
```

## Next Steps

1. **Evaluate**: Compare with sequence-based encoder on retrieval tasks
2. **Experiment**: Try different vision models (CLIP vs DINOv2 vs SigLIP)
3. **Tune**: Add metadata features for hybrid approach
4. **Visualize**: Inspect learned representations with t-SNE

## Documentation

- **Complete Guide**: `docs/IMAGE_ENCODER_TRAINING_GUIDE.md`
- **Implementation**: `docs/IMAGE_ENCODER_IMPLEMENTATION_SUMMARY.md`
- **Image Generation**: `docs/IMAGE_GENERATION_GUIDE.md`
- **Alignment Training**: `docs/ALIGNMENT_GUIDE.md`

## Example Script

```bash
# Test the implementation
python -m src.encoders.sensor.sequence.example_image_encoder
```

## Support

For issues or questions, check:
1. `docs/IMAGE_ENCODER_TRAINING_GUIDE.md` - Detailed troubleshooting
2. Error messages - Include instructions for fixing
3. Example script - Validates prerequisites

---

**That's it!** 🎉 You can now train sensor encoders using powerful pre-trained vision models.

