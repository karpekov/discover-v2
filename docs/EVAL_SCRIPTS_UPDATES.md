# Evaluation Scripts Updates

## Summary

Updated all evaluation scripts to work with the new data format and model checkpoints from the milan_fixed20_v0 training run.

## Key Changes

### 1. Data Format Compatibility (`src/dataio/dataset.py`)

**Problem**: The new data sampling pipeline generates a different data structure than the old format.

**Changes**:
- **Dynamic Field Detection**: Instead of hardcoding `sensor_id`, the dataset now dynamically determines categorical fields from the vocabulary
- **Event Sequence Key**: Handle both `events` (old format) and `sensor_sequence` (new format)
- **Data Container**: Support both direct list format and dict with `samples` key
- **Field Name Mapping**: Map between new field names (`sensor_id`, `event_type`, `room`) and vocab field names (`sensor`, `state`, `room_id`)
- **Label Extraction**: Extract ground truth labels from `metadata.ground_truth_labels.primary_l1/l2` (new) or directly from sample data (old)
- **Timestamp Parsing**: Handle both string timestamps (ISO format) and numeric timestamps, with proper parsing using `dateutil.parser`
- **Coordinate Handling**: Gracefully handle samples without coordinate information, disabling normalization if needed

### 2. Model Checkpoint Loading (`src/alignment/model.py`, evaluation scripts)

**Problem**: Checkpoints saved by `AlignmentTrainer` use a different structure than those saved by `AlignmentModel`.

**Changes**:
- **Unified Checkpoint Loading**: Handle two formats:
  - `model_state_dict` from `AlignmentTrainer` (full model state)
  - Individual state dicts (`sensor_encoder_state_dict`, etc.) from older `AlignmentModel`
- **Vocab Size Inference**: Automatically infer `vocab_sizes` from embedding layer keys in the state dict
- **Strict=False Loading**: Use `strict=False` when loading to ignore unexpected keys like `text_projection.proj.weight`
- **PyTorch Safety**: Added `weights_only=False` to all `torch.load` calls to handle dataclass configs

### 3. Text Encoder Dimension Auto-Detection (`src/evals/eval_utils.py`)

**Problem**: Different training runs use different text encoders (CLIP 512-dim, GTE 768-dim, MiniLM 384-dim), and hardcoding the dimension causes errors.

**Solution**: Created `create_text_encoder_from_checkpoint()` utility that:
- Automatically detects text embedding dimensions from `text_projection.proj.weight` in checkpoint
- Creates a CLIP-based encoder matching the training setup
- Loads the learned projection head weights from checkpoint
- Falls back to standard encoder if projection head not found

**Updated Scripts**:
- `src/evals/evaluate_embeddings.py`
- `src/evals/embedding_alignment_analysis.py`
- `src/evals/caption_alignment_analysis.py`
- `src/evals/visualize_embeddings.py`

### 4. Sensor Encoder Interface (`src/encoders/sensor/sequence/transformer.py`)

**Problem**: The new `TransformerSensorEncoder` expects a different calling convention than the old `SensorEncoder`.

**Changes**:
- **Input Data Packing**: Pack categorical features, coordinates, and time deltas into an `input_data` dict
- **Attention Mask Parameter**: Use `attention_mask` parameter instead of `mask`
- Updated all evaluation scripts to use the new interface:
  ```python
  input_data = {
      'categorical_features': batch['categorical_features'],
      'coordinates': batch['coordinates'],
      'time_deltas': batch['time_deltas']
  }
  sensor_emb = self.sensor_encoder.forward_clip(
      input_data=input_data,
      attention_mask=batch['mask']
  )
  ```

### 5. Metadata Label Colors (`src/evals/evaluate_embeddings.py`)

**Problem**: L1 label colors weren't being loaded for visualizations.

**Changes**:
- **Dataset Detection**: Automatically detect which dataset (milan, cairo, etc.) from the data path
- **Flexible Key Lookup**: Milan uses `label` key, others use `label_color`
- **L2 Colors**: Load from `label_deepcasas_color` for all datasets

## Files Modified

### Core Data Processing
- `src/dataio/dataset.py` - Data format compatibility

### Model Loading
- `src/alignment/model.py` - Checkpoint loading
- `src/alignment/trainer.py` - PyTorch safety

### Evaluation Scripts
- `src/evals/eval_utils.py` - NEW: Shared text encoder utility
- `src/evals/evaluate_embeddings.py` - Text encoder, label colors, forward_clip interface
- `src/evals/embedding_alignment_analysis.py` - Text encoder, forward_clip interface
- `src/evals/caption_alignment_analysis.py` - Text encoder, forward_clip interface
- `src/evals/visualize_embeddings.py` - Text encoder, forward_clip interface
- `src/evals/clustering_evaluation.py` - forward_clip interface
- `src/evals/query_retrieval.py` - forward_clip interface
- `src/evals/evaluate_checkpoints.py` - forward_clip interface, PyTorch safety
- `src/evals/eval_retrieval.py` - PyTorch safety

## Testing

Tested with:
```bash
python src/evals/run_all_evals.py \
    --checkpoint trained_models/milan/fixed20_v0/best_model.pt \
    --train_data data/processed/casas/milan/fixed_length_20/train.json \
    --test_data data/processed/casas/milan/fixed_length_20/test.json \
    --vocab data/processed/casas/milan/fixed_length_20/vocab.json \
    --output_dir results/evals/milan/fixed20_v0 \
    --max_samples 10000 \
    --n_clusters 50
```

## Future Improvements

1. **Config-Based Text Encoder**: Store text encoder configuration in checkpoint metadata to avoid dimension detection hacks
2. **Unified Data Format**: Standardize on a single data format to avoid compatibility checks
3. **Version Markers**: Add version markers to checkpoints to enable format-specific loading logic
4. **Integration Tests**: Add automated tests for evaluation scripts with different checkpoint formats

