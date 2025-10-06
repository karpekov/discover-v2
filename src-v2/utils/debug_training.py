#!/usr/bin/env python3
"""
Simple Debug Training Script

Just calls the existing SmartHomeTrainer with debug-friendly settings.
Set breakpoints wherever you need to inspect the training process.

Run from src-v2/utils/ directory:
    cd src-v2/utils
    python debug_training.py

Or from src-v2/ directory:
    python -m utils.debug_training
"""

import os
import sys
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
src_v2_dir = current_dir.parent
project_root = src_v2_dir.parent

# Change to project root to avoid import issues
original_cwd = os.getcwd()
os.chdir(project_root)

# Add paths for imports
# No need for scripts path anymore
sys.path.insert(0, str(src_v2_dir))

# Import the existing trainer
from training.train_clip import SmartHomeTrainer


def get_debug_config():
    """Simple debug configuration for Milan dataset."""
    # Paths relative to project root
    project_root = Path(__file__).parent.parent.parent

    return {
        # Model config
        'text_model_name': 'thenlper/gte-base',
        'd_model': 768,
        'n_layers': 6,
        'n_heads': 8,
        'd_ff': 3072,
        'dropout': 0.1,
        'fourier_bands': 12,
        'max_seq_len': 512,
        'sequence_length': 20,
        'max_captions': 3,

        # Training config - small for debugging
        'batch_size': 4,
        'learning_rate': 1e-4,
        'betas': (0.9, 0.98),
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'max_epochs': 1,
        'max_steps': 50,  # Just 50 steps for debugging
        'grad_clip_norm': 1.0,

        # Loss config
        'mlm_weight': 1.0,
        'clip_weight': 1.0,
        'temperature_init': 0.02,
        'learnable_temperature': True,

        # MLM config
        'mask_prob': 0.25,
        'mean_span_length': 3.0,

        # Logging
        'log_interval': 5,  # Log every 5 steps
        'val_interval': 20,
        'save_interval': 100,
        'output_dir': str(project_root / 'debug_output'),

        # Disable wandb for debugging
        'use_wandb': False,

        # Data paths (relative to project root)
        'train_data_path': str(project_root / 'data/data_for_alignment/milan_training_20/milan_train.json'),
        'vocab_path': str(project_root / 'data/data_for_alignment/milan_training_20/milan_vocab.json'),
        'val_data_path': None,  # No validation for debugging

        # Device config
        'use_amp': False,  # Disable AMP for easier debugging
        'num_workers': 0,  # Single-threaded for debugging
    }


def main():
    print("=== Simple Debug Training ===")
    print(f"Original directory: {original_cwd}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).parent}")

    # Get config
    config = get_debug_config()

    # Check data files exist
    train_path = Path(config['train_data_path'])
    vocab_path = Path(config['vocab_path'])

    if not train_path.exists():
        print(f"ERROR: Training data not found: {train_path}")
        print(f"Absolute path: {train_path.absolute()}")
        return 1

    if not vocab_path.exists():
        print(f"ERROR: Vocabulary not found: {vocab_path}")
        print(f"Absolute path: {vocab_path.absolute()}")
        return 1

    print(f"Training data: {train_path}")
    print(f"Vocabulary: {vocab_path}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Max steps: {config['max_steps']}")
    print()

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    try:
        # Initialize trainer - SET BREAKPOINT HERE if you want to inspect initialization
        print("Initializing trainer...")
        trainer = SmartHomeTrainer(config)

        # SET BREAKPOINT HERE to inspect trainer state before training
        print("Starting training...")

        # Call the existing train method - SET BREAKPOINT INSIDE train() method to step through
        trainer.train()

        print("Training completed!")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    # Quick test to make sure imports work
    try:
        from training.train_clip import SmartHomeTrainer
        print("✓ Successfully imported SmartHomeTrainer")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("Make sure you're running from src-v2/utils/ directory or src-v2/ directory")
        print("Current Python path:")
        for p in sys.path:
            print(f"  {p}")
        exit(1)

    exit(main())
