"""
Example usage of the alignment module.

This demonstrates how to use AlignmentModel, AlignmentTrainer, and AlignmentDataset
for sensor-text alignment training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from src.alignment.config import AlignmentConfig
from src.alignment.model import AlignmentModel
from src.alignment.trainer import AlignmentTrainer


def example_1_load_config():
    """Example 1: Load configuration from YAML file."""
    print("=" * 80)
    print("Example 1: Load Configuration")
    print("=" * 80)

    config_path = "configs/alignment/milan_baseline.yaml"

    try:
        config = AlignmentConfig.from_yaml(config_path)
        print(f"✅ Loaded config from: {config_path}")
        print(f"   Experiment: {config.experiment_name}")
        print(f"   Output dir: {config.output_dir}")
        print(f"   Encoder: {config.encoder_type}")
        print(f"   Projection: {config.sensor_projection.type} ({config.sensor_projection.dim}d)")
        print(f"   CLIP weight: {config.loss.clip_weight}")
        print(f"   MLM weight: {config.loss.mlm_weight}")
        print(f"   Batch size: {config.training.batch_size}")
        print(f"   Max steps: {config.training.max_steps}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def example_2_create_config_programmatically():
    """Example 2: Create configuration programmatically."""
    print("\n" + "=" * 80)
    print("Example 2: Create Configuration Programmatically")
    print("=" * 80)

    try:
        from src.alignment.config import (
            AlignmentConfig,
            ProjectionConfig,
            LossConfig,
            OptimizerConfig,
            TrainingConfig
        )

        config = AlignmentConfig(
            experiment_name="my_experiment",
            output_dir="trained_models/my_experiment",
            train_data_path="data/processed/casas/milan/fixed_duration_60s/train.json",
            vocab_path="data/processed/casas/milan/fixed_duration_60s/vocab.json",
            train_text_embeddings_path="data/processed/casas/milan/fixed_duration_60s/train_embeddings_baseline_gte_base.npz",
            encoder_config_path="configs/encoders/transformer_base.yaml",
            sensor_projection=ProjectionConfig(
                type='mlp',
                dim=512,
                hidden_dim=2048,
                num_layers=2
            ),
            loss=LossConfig(
                clip_weight=1.0,
                temperature_init=0.02,
                learnable_temperature=True
            ),
            optimizer=OptimizerConfig(
                learning_rate=3e-4
            ),
            training=TrainingConfig(
                batch_size=128,
                max_steps=10000
            )
        )

        print(f"✅ Created config programmatically")
        print(f"   Experiment: {config.experiment_name}")
        print(f"   Projection: {config.sensor_projection.type}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def example_3_validate_config():
    """Example 3: Validate configuration."""
    print("\n" + "=" * 80)
    print("Example 3: Validate Configuration")
    print("=" * 80)

    try:
        from src.alignment.config import AlignmentConfig

        # Create an invalid config (missing required fields)
        config = AlignmentConfig(
            experiment_name="test",
            output_dir="test_output"
        )

        # This should raise validation errors
        config.validate()
        print(f"✅ Config validated (shouldn't reach here with empty config)")
        return False
    except ValueError as e:
        print(f"✅ Config validation working correctly!")
        print(f"   Caught expected error: {str(e)[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def example_4_model_creation():
    """Example 4: Create AlignmentModel (requires data)."""
    print("\n" + "=" * 80)
    print("Example 4: Create AlignmentModel")
    print("=" * 80)

    print("ℹ️  This example requires actual data to run.")
    print("   To create a model, you need:")
    print("   1. A valid AlignmentConfig")
    print("   2. Vocabulary sizes dict")
    print("   3. Actual data files")
    print("")
    print("   Example code:")
    print("   ```python")
    print("   config = AlignmentConfig.from_yaml('configs/alignment/milan_baseline.yaml')")
    print("   vocab_sizes = {'sensor': 51, 'state': 3, 'room_id': 10}")
    print("   model = AlignmentModel(config, vocab_sizes)")
    print("   model.to(device)")
    print("   ```")

    return True


def example_5_training_loop():
    """Example 5: Training with AlignmentTrainer."""
    print("\n" + "=" * 80)
    print("Example 5: Training with AlignmentTrainer")
    print("=" * 80)

    print("ℹ️  To train an alignment model, use the unified train.py script:")
    print("")
    print("   # Basic training with existing data")
    print("   python train.py --config configs/alignment/milan_baseline.yaml")
    print("")
    print("   # Full pipeline from scratch")
    print("   python train.py --config configs/alignment/milan_baseline.yaml --run-full-pipeline")
    print("")
    print("   # Resume from checkpoint")
    print("   python train.py --config configs/alignment/milan_baseline.yaml \\")
    print("                   --resume trained_models/milan/alignment_baseline/checkpoint_step_5000.pt")
    print("")
    print("   Or use AlignmentTrainer directly:")
    print("   ```python")
    print("   from src.alignment.trainer import AlignmentTrainer")
    print("   config = AlignmentConfig.from_yaml('configs/alignment/milan_baseline.yaml')")
    print("   trainer = AlignmentTrainer(config)")
    print("   trainer.train()")
    print("   ```")

    return True


def example_6_inference():
    """Example 6: Using trained model for inference."""
    print("\n" + "=" * 80)
    print("Example 6: Using Trained Model for Inference")
    print("=" * 80)

    print("ℹ️  After training, load and use the model:")
    print("")
    print("   ```python")
    print("   import torch")
    print("   from src.alignment.model import AlignmentModel")
    print("   ")
    print("   # Load checkpoint")
    print("   checkpoint = torch.load('trained_models/milan/alignment_baseline/best_model.pt')")
    print("   config = checkpoint['config']")
    print("   vocab_sizes = {...}  # Load from data")
    print("   ")
    print("   # Create and load model")
    print("   model = AlignmentModel(config, vocab_sizes)")
    print("   model.load_state_dict(checkpoint['model_state_dict'])")
    print("   model.eval()")
    print("   ")
    print("   # Forward pass")
    print("   with torch.no_grad():")
    print("       outputs = model(sensor_data, text_embeddings, attention_mask)")
    print("       sensor_embeddings = outputs['sensor_embeddings_projected']")
    print("       text_embeddings = outputs['text_embeddings_projected']")
    print("   ```")

    return True


def run_all_examples():
    """Run all examples."""
    examples = [
        ("Load Config", example_1_load_config),
        ("Create Config Programmatically", example_2_create_config_programmatically),
        ("Validate Config", example_3_validate_config),
        ("Model Creation", example_4_model_creation),
        ("Training Loop", example_5_training_loop),
        ("Inference", example_6_inference),
    ]

    print("\n" + "=" * 80)
    print("ALIGNMENT MODULE - EXAMPLE USAGE")
    print("=" * 80)
    print("")

    results = []
    for name, example_fn in examples:
        try:
            success = example_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Example '{name}' failed with exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {passed}/{total} examples passed")

    return passed == total


if __name__ == '__main__':
    success = run_all_examples()
    sys.exit(0 if success else 1)

