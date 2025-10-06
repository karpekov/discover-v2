# Weights & Biases Setup Guide for CASAS Clustering

This guide will help you set up Weights & Biases (W&B) to monitor your training runs for the CASAS clustering project.

## 🚀 Quick Start

### 1. Installation & Authentication
W&B is already installed in your `har_env` environment. To authenticate:

```bash
# Activate environment
conda activate har_env

# Login to W&B (you've already done this)
wandb login
# Paste your API key from https://wandb.ai/authorize
```

### 2. Basic Training with W&B
```bash
# Basic training with W&B enabled
python scripts/train.py \
  --train_data data/data_for_alignment/milan_training/train.json \
  --vocab data/data_for_alignment/milan_training/vocab.json \
  --val_data data/data_for_alignment/milan_training/val.json \
  --output_dir ./outputs/milan_baseline \
  --wandb_project discover-v2 \
  --wandb_name "milan_baseline_experiment_v1" \
  --wandb_tags milan baseline dual-encoder
```

### 3. Using Predefined Experiments
```bash
# Use predefined experiment configurations
python scripts/train.py \
  --train_data data/data_for_alignment/milan_training/train.json \
  --vocab data/data_for_alignment/milan_training/vocab.json \
  --experiment milan_baseline
```

## 📊 What Gets Logged to W&B

### Training Metrics (every 50 steps)
- `train/total_loss` - Combined CLIP + MLM loss
- `train/clip_loss` - Contrastive loss component
- `train/mlm_loss` - Masked language model loss (if enabled)
- `train/sensor_to_text_acc` - Retrieval accuracy (sensor → text)
- `train/text_to_sensor_acc` - Retrieval accuracy (text → sensor)
- `train/temperature` - Learned temperature parameter
- `train/learning_rate` - Current learning rate
- `system/gpu_memory_*` - GPU memory usage

### Validation Metrics (every 500 steps)
- `val/total_loss` - Validation loss
- `val/clip_loss` - Validation CLIP loss
- `val/sensor_to_text_acc` - Validation retrieval accuracy
- `val/text_to_sensor_acc` - Validation retrieval accuracy

### Model Metadata (Local Files Only)
- **Best Model Info**: Local path, file size, validation loss when saved
- **Final Model Info**: Local path, file size, parameter counts
- **Code Snapshot**: Current code state saved with each run
- **Note**: Model files stay local - only metadata uploaded to W&B

### Model Parameters (optional)
- Parameter histograms (every 100 steps by default)
- Gradient histograms (disabled by default - expensive)

## 🛠️ Configuration Options

### Command Line Arguments
```bash
# W&B specific arguments
--wandb_project discover-v2        # Project name
--wandb_name "my_experiment"        # Run name
--wandb_tags tag1 tag2 tag3        # Tags for organization
--wandb_notes "Experiment notes"    # Description
--no_wandb                         # Disable W&B logging
--experiment milan_baseline        # Use predefined config
```

### Configuration File
Use the baseline configuration in `config/milan_baseline_config.json`:

```bash
python scripts/train.py \
  --config config/milan_baseline_config.json \
  --train_data path/to/train.json \
  --vocab path/to/vocab.json
```

### Programmatic Configuration
```python
from config.wandb_config import WandBConfig, get_wandb_config_for_experiment

# Create W&B config manager
wandb_manager = WandBConfig(project_name="discover-v2")

# Get configuration for experiment
config = {...}  # Your base config
wandb_config = wandb_manager.get_config("my_experiment", config)

# Or use predefined experiments
wandb_config = get_wandb_config_for_experiment("milan_baseline", config)
```

## 🎯 Predefined Experiments

### Available Experiments
1. **`milan_baseline`** - Baseline dual-encoder training
2. **`milan_ablation_mlm`** - MLM weight ablation study
3. **`milan_ablation_temperature`** - Temperature initialization study
4. **`multi_dataset`** - Multi-dataset training

### Example Usage
```bash
# Run baseline experiment
python scripts/train.py \
  --train_data data/milan/train.json \
  --vocab data/milan/vocab.json \
  --experiment milan_baseline

# Run ablation study
python scripts/train.py \
  --train_data data/milan/train.json \
  --vocab data/milan/vocab.json \
  --experiment milan_ablation_mlm \
  --wandb_name "mlm_weight_0.3"
```

## 📈 W&B Dashboard Features

### Run Organization
- **Projects**: All runs grouped under `discover-v2`
- **Groups**: Experiments grouped by dataset and model size
- **Tags**: Filterable tags for easy searching
- **Notes**: Detailed experiment descriptions

### Visualizations
- **Loss Curves**: Training and validation losses over time
- **Accuracy Plots**: Retrieval accuracy metrics
- **System Metrics**: GPU memory and utilization
- **Parameter Tracking**: Model parameter distributions
- **Model Comparison**: Side-by-side run comparisons

### Model Management
- **Artifacts**: Versioned model checkpoints
- **Best Model Tracking**: Automatic best model identification
- **Code Versioning**: Code snapshots for reproducibility

## 🔧 Advanced Configuration

### Disable Expensive Logging
```python
config = {
    'wandb_log_gradients': False,    # Disable gradient histograms
    'wandb_watch_freq': 500,         # Reduce parameter logging frequency
    'wandb_log_model': False,        # Keep models local (default)
}
```

### Privacy & Data Security
- **Models**: All model files remain local - only metadata uploaded
- **Data**: No training data uploaded to W&B
- **Code**: Only code snapshots (no sensitive data)
- **Metrics**: Only training metrics and hyperparameters logged

### Custom Tags and Organization
```python
config = {
    'wandb_tags': ['milan', 'ablation', 'temperature_0.02'],
    'wandb_group': 'milan_temperature_study',
    'wandb_notes': 'Testing impact of temperature initialization on convergence'
}
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your API key is exactly 40 characters
   ```bash
   wandb login
   # Paste key from https://wandb.ai/authorize
   ```

2. **Import Error**: Make sure W&B is installed in the correct environment
   ```bash
   conda activate har_env
   pip install wandb
   ```

3. **Permission Errors**: Check your W&B entity/team permissions
   ```bash
   python -c "import wandb; print(wandb.api.viewer())"
   ```

### Disable W&B Temporarily
```bash
# Disable W&B for debugging
python scripts/train.py --no_wandb [other args...]
```

## 📚 Comparison with TensorBoard

| Feature | TensorBoard | Weights & Biases |
|---------|-------------|------------------|
| **Setup** | Local files | Cloud-based |
| **Collaboration** | Manual sharing | Built-in sharing |
| **Model Versioning** | Manual | Automatic artifacts |
| **Hyperparameter Tracking** | Limited | Full tracking |
| **Experiment Organization** | Folder-based | Tags/Groups/Projects |
| **Mobile Access** | No | Yes |
| **Code Tracking** | No | Automatic |

## 🎉 Next Steps

1. **Run your first experiment**:
   ```bash
   python scripts/train.py \
     --train_data your_data.json \
     --vocab your_vocab.json \
     --experiment milan_baseline
   ```

2. **Check your dashboard**: Visit https://wandb.ai/your-username/discover-v2

3. **Explore features**: Try different experiments, compare runs, and use the model artifacts

4. **Customize**: Modify `src-v2/config/wandb_config.py` for your specific needs

Happy training! 🚀
