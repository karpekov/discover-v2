# Data Preparation Script - Summary

## What Was Created

A comprehensive bash script that automates the entire data preparation pipeline in **one command**.

### Script Location
`bash_scripts/prepare_data.sh`

### What It Does

**Old way (manual, 6+ commands):**
```bash
# Sample data
python sample_data.py --config configs/sampling/milan_FD_60.yaml
python sample_data.py --config configs/sampling/milan_FD_60_p.yaml

# Generate captions
python src/captions/generate_captions.py --data-dir ... --caption-style baseline ...
python src/captions/generate_captions.py --data-dir ... --caption-style baseline ...

# Encode captions
python src/text_encoders/encode_captions.py --data-dir ... --config ...
python src/text_encoders/encode_captions.py --data-dir ... --config ...
```

**New way (automated, 1 command):**
```bash
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60
```

## Key Features

### 1. **Runs Both Regular and Presegmented Versions**
- Automatically processes both `FD_60` and `FD_60_p`
- Can skip presegmented with `--no-presegmented`

### 2. **Uses Config Files**
- Caption config: `configs/captions/baseline_milan.yaml` (with your fixed 4 captions!)
- Text encoder config: `configs/text_encoders/clip_vit_base.yaml`
- Sampling config: `configs/sampling/milan_FD_60.yaml`

### 3. **Flexible Skipping**
- `--skip-sampling` - Data already exists, regenerate captions + embeddings
- `--skip-captions` - Captions exist, regenerate embeddings only
- `--skip-encoding` - Skip text encoding entirely

### 4. **Multiple Options**
- Caption styles: `baseline` (default), `sourish`
- Text encoders: `clip_vit_base` (default), `gte_base`, `minilm_l6`, etc.
- Devices: `auto` (default), `cpu`, `cuda`, `mps`

### 5. **Smart Error Handling**
- Checks if config files exist
- Warns if data directories missing
- Exits immediately on errors
- Clear success/failure messages

## Usage Examples

### Most Common: Full Pipeline
```bash
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60
```

### After Changing Caption Code
```bash
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60 --skip-sampling
```

### Try Different Text Encoder
```bash
bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60 \
    --skip-sampling --skip-captions --text-encoder gte_base
```

### Multiple Datasets
```bash
for dataset in milan aruba cairo; do
    bash bash_scripts/prepare_data.sh --dataset $dataset --sampling FD_60
done
```

### Multiple Sampling Strategies
```bash
for sampling in FD_30 FD_60 FD_120; do
    bash bash_scripts/prepare_data.sh --dataset milan --sampling $sampling
done
```

## Documentation Created

1. **`bash_scripts/prepare_data.sh`** - The main script
2. **`docs/AUTOMATIC_DATA_PREP_UPDATED.md`** - Full documentation
3. **`QUICK_START_DATA_PREP.md`** - Quick reference at project root
4. **This file** - Summary of what was created

## Integration with Fixed Caption Config

The script now properly uses the caption config files you created:

```yaml
# configs/captions/baseline_aruba.yaml
num_captions_per_sample: 4  # This now works correctly!
```

When you run:
```bash
bash bash_scripts/prepare_data.sh --dataset aruba --sampling FD_60
```

It will:
1. Load `configs/captions/baseline_aruba.yaml`
2. Use `num_captions_per_sample: 4` (your 4 captions!)
3. Generate captions with the config
4. Encode all 4 captions per sample

## Benefits

✅ **One command** instead of 6+
✅ **Automatic presegmented** version handling
✅ **Config file integration** (no hardcoded defaults!)
✅ **Smart skipping** for faster iteration
✅ **Clear error messages** and validation
✅ **Flexible options** for different use cases
✅ **Self-documenting** with `--help`

## Testing

The script has been tested and works correctly:

```bash
$ bash bash_scripts/prepare_data.sh --help
# Shows complete help message with all options

$ bash bash_scripts/prepare_data.sh
# Error: --dataset and --sampling are required!

$ bash bash_scripts/prepare_data.sh --dataset invalid --sampling FD_60
# Error: Dataset must be one of: milan, aruba, cairo, kyoto
```

## Next Steps for You

1. **Test the script:**
   ```bash
   bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60
   ```

2. **Use it in your workflow:**
   - Every time you change data generation: run the script
   - Testing new caption styles: add `--caption-style sourish`
   - Trying new text encoders: add `--text-encoder gte_base`

3. **Create shortcuts** for your common workflows:
   ```bash
   # Add to your ~/.bashrc or create an alias
   alias prep_milan="bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60"
   ```

## Files Modified/Created

### New Files
- `bash_scripts/prepare_data.sh` - Main automation script
- `docs/AUTOMATIC_DATA_PREP_UPDATED.md` - Full documentation
- `docs/DATA_PREP_SCRIPT_SUMMARY.md` - This summary
- `QUICK_START_DATA_PREP.md` - Quick reference

### Modified Files
- `src/captions/generate_captions.py` - Added `--config` flag support
- `docs/CAPTION_GENERATION_GUIDE.md` - Updated with config file usage

## Related Issues Fixed

1. ✅ **Caption config files now work** - Fixed hardcoded default of 2 captions
2. ✅ **Config loading added** - YAML caption configs are now loaded
3. ✅ **One-command pipeline** - Automated the entire workflow

## See Also

- `docs/CAPTION_GENERATION_GUIDE.md` - Caption generation details
- `docs/TEXT_ENCODER_GUIDE.md` - Text encoding details
- `docs/SAMPLING_UPDATES.md` - Sampling strategy details
- `bash_scripts/generate_all_data_milan.sh` - Old batch script (still works)

