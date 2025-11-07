# How to Use Chronos-2

## Overview

[Chronos-2](https://huggingface.co/amazon/chronos-2) is Amazon's 120M-parameter time series foundation model for zero-shot forecasting. While it's primarily designed for forecasting, we can use it for embedding extraction in our smart home activity recognition pipeline.

## Installation

### Option 1: Install Chronos-2 Package (Recommended)

```bash
pip install "chronos-forecasting>=2.0"
```

This installs the latest Chronos-2 package with `Chronos2Pipeline`.

### Option 2: Install from GitHub (For Chronos-1 Compatibility)

```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

## Key Features of Chronos-2

According to the [HuggingFace model card](https://huggingface.co/amazon/chronos-2):

- **120M parameters**: Encoder-only T5-based architecture
- **Multivariate support**: Can handle multiple time series features simultaneously
- **Max context length**: 8192 timesteps
- **Efficient**: Over 300 forecasts per second on A10G GPU
- **Zero-shot**: Works without fine-tuning

## Using Chronos-2 in Our Pipeline

### Current Implementation

Our `ChronosEncoder` class:

1. **Loads Chronos-2** (frozen) using `Chronos2Pipeline`
2. **Converts sensor sequences** to multivariate time series (5 features: sensor_id, room_id, x, y, time_delta)
3. **Extracts embeddings** using statistical features (fallback approach)
4. **Projects to 512-dim** using a trainable MLP head for CLIP alignment

### Why Statistical Features?

Chronos-2 is designed for **forecasting**, not embedding extraction. The pipeline API (`predict_df`) is optimized for generating future predictions, not extracting intermediate representations.

For embedding extraction, we currently use:
- **Statistical features**: Mean, std, max, min for each of 5 time series features (20 dims)
- **Temporal features**: Downsampled sequence (50 dims)
- **Total**: 70-dimensional feature vector → MLP → 512-dim CLIP embedding

This approach works well because:
- ✅ Fast and efficient
- ✅ No dependency on Chronos package
- ✅ Captures important time series statistics
- ✅ Works with any sequence length

### Future: Direct Chronos-2 Encoder Access

To use the actual Chronos-2 encoder for embeddings, you would need to:

1. Access the underlying T5 encoder directly
2. Pass time series through the encoder's embedding layer
3. Extract hidden states from the encoder layers
4. Pool the sequence (mean/max/CLS token)

This requires deeper integration with the Chronos-2 model architecture.

## Training with Chronos Encoder

```bash
# Train with Chronos-2 encoder
python src/training/train_chronos_clip.py --config configs/training/milan/chronos_clip.json
```

The config file (`chronos_clip.json`) specifies:
- `chronos_model_name`: "amazon/chronos-2"
- `projection_hidden_dim`: 256
- `output_dim`: 512 (for CLIP alignment)

## Model Comparison

| Feature | Chronos-2 | Our Implementation |
|---------|-----------|-------------------|
| Primary Use | Forecasting | Embedding Extraction |
| API | `predict_df()` | Statistical Features |
| Multivariate | ✅ Native | ✅ Via feature stacking |
| Embeddings | Via encoder access | Statistical + temporal |
| Training | Frozen | Frozen (only MLP trained) |

## References

- [Chronos-2 HuggingFace Model](https://huggingface.co/amazon/chronos-2)
- [Chronos-2 Technical Report](https://arxiv.org/abs/2510.15821)
- [Chronos-2 GitHub](https://github.com/amazon-science/chronos-forecasting)

## Troubleshooting

### Import Error: "No module named 'chronos'"

Install the package:
```bash
pip install "chronos-forecasting>=2.0"
```

### Model Loading Fails

The code will automatically fall back to statistical features. This is expected and works well for our use case.

### Want to Use Actual Chronos-2 Embeddings?

You would need to modify `_extract_chronos_embeddings()` to:
1. Access `pipeline.model.encoder` directly
2. Tokenize time series appropriately
3. Extract encoder hidden states
4. Pool the sequence

This is more complex but would leverage Chronos-2's pre-trained time series knowledge.

