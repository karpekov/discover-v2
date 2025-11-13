"""
Example usage of the new modular encoder framework.

This script demonstrates how to:
1. Create an encoder from config
2. Prepare input data with padding
3. Run forward pass
4. Use for CLIP alignment
5. Use for MLM training

Run this script to verify the encoder works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml

from src.encoders.config import TransformerEncoderConfig, MetadataConfig
from src.encoders.sensor.sequence import TransformerSensorEncoder
from src.encoders.data_utils import load_and_prepare_milan_data, prepare_batch_for_encoder


def example_basic_usage(use_real_data: bool = True):
    """Example 1: Basic encoder usage with all metadata."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage (Full Metadata)")
    print("="*60)

    if use_real_data:
        # Load real Milan data from Step 1
        print("\n--- Loading Real Milan Data ---")
        try:
            data, vocab, vocab_sizes = load_and_prepare_milan_data(
                data_dir='data/processed/casas/milan/fixed_duration_60sec_presegmented',
                split='train',
                max_samples=8,  # Load 8 samples for batch
                categorical_fields=['sensor_id', 'event_type', 'room', 'sensor_type'],
                use_coordinates=False,  # Milan data doesn't have coordinates yet
                use_time_deltas=True,
                device='cpu'
            )

            # Create config with actual vocab sizes
            config = TransformerEncoderConfig.base()
            config.vocab_sizes = vocab_sizes
            config.metadata.use_coordinates = False  # No coordinates in current data
            config.metadata.use_time_deltas = True
            config.metadata.categorical_fields = list(vocab.keys())

            print(f"\nConfig: {config.encoder_type}")
            print(f"  - d_model: {config.d_model}")
            print(f"  - n_layers: {config.n_layers}")
            print(f"  - n_heads: {config.n_heads}")
            print(f"  - projection_dim: {config.projection_dim}")
            print(f"  - use_coordinates: {config.metadata.use_coordinates}")
            print(f"  - use_time_deltas: {config.metadata.use_time_deltas}")
            print(f"  - categorical_fields: {config.metadata.categorical_fields}")

            # Create encoder
            encoder = TransformerSensorEncoder(config)
            total_params = sum(p.numel() for p in encoder.parameters())
            print(f"\nTotal parameters: {total_params:,}")

            # Prepare batch from real data
            print("\n--- Preparing Batch ---")
            input_data = prepare_batch_for_encoder(
                samples=data['samples'][:8],
                vocab=vocab,
                categorical_fields=list(vocab.keys()),
                use_coordinates=False,
                use_time_deltas=True,
                device='cpu'
            )

            attention_mask = input_data.pop('attention_mask')

            print(f"\nInput shapes (Real Milan Data):")
            for field in input_data['categorical_features']:
                print(f"  - {field}: {input_data['categorical_features'][field].shape}")
            print(f"  - time_deltas: {input_data['time_deltas'].shape}")
            print(f"  - attention_mask: {attention_mask.shape}")

            # Print sequence length statistics
            seq_lengths = attention_mask.sum(dim=1).tolist()
            print(f"  - Sequence lengths: {seq_lengths}")
            print(f"  - Avg length: {sum(seq_lengths)/len(seq_lengths):.1f}")
            print(f"  - Variable length data from fixed-duration sampling!")

        except FileNotFoundError as e:
            print(f"\n⚠️  Real data not found: {e}")
            print("Falling back to synthetic data...")
            use_real_data = False

    if not use_real_data:
        # Fallback to synthetic data
        print("\n--- Using Synthetic Data ---")
        config = TransformerEncoderConfig.base()
        config.vocab_sizes = {
            'sensor': 50,
            'state': 10,
            'room_id': 20
        }

        print(f"\nConfig: {config.encoder_type}")
        print(f"  - d_model: {config.d_model}")
        print(f"  - n_layers: {config.n_layers}")
        print(f"  - n_heads: {config.n_heads}")
        print(f"  - projection_dim: {config.projection_dim}")
        print(f"  - use_coordinates: {config.metadata.use_coordinates}")
        print(f"  - use_time_deltas: {config.metadata.use_time_deltas}")

        # Create encoder
        encoder = TransformerSensorEncoder(config)
        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"\nTotal parameters: {total_params:,}")

        # Prepare synthetic input data
        batch_size = 8
        seq_len = 50

        input_data = {
            'categorical_features': {
                'sensor': torch.randint(0, 50, (batch_size, seq_len)),
                'state': torch.randint(0, 10, (batch_size, seq_len)),
                'room_id': torch.randint(0, 20, (batch_size, seq_len)),
            },
            'coordinates': torch.randn(batch_size, seq_len, 2),
            'time_deltas': torch.rand(batch_size, seq_len) * 100,
        }

        # Create attention mask with padding
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        # Last 10 tokens are padding for first 4 samples
        attention_mask[:4, 40:] = False

        print(f"\nInput shapes (Synthetic):")
        print(f"  - sensor: {input_data['categorical_features']['sensor'].shape}")
        print(f"  - coordinates: {input_data['coordinates'].shape}")
        print(f"  - time_deltas: {input_data['time_deltas'].shape}")
        print(f"  - attention_mask: {attention_mask.shape}")
        print(f"  - Padding samples: 4/8 have padding")

    # Forward pass
    print("\n--- Forward Pass ---")
    with torch.no_grad():
        output = encoder(input_data, attention_mask=attention_mask)

    batch_size = attention_mask.shape[0]
    print(f"Output embeddings shape: {output.embeddings.shape}")
    print(f"Sequence features shape: {output.sequence_features.shape}")
    print(f"Embeddings are L2-normalized: {torch.allclose(output.embeddings.norm(dim=-1), torch.ones(batch_size))}")

    return encoder, input_data, attention_mask


def example_clip_alignment(encoder, input_data, attention_mask):
    """Example 2: Using encoder for CLIP alignment."""
    print("\n" + "="*60)
    print("Example 2: CLIP Alignment")
    print("="*60)

    with torch.no_grad():
        clip_embeddings = encoder.forward_clip(input_data, attention_mask=attention_mask)

    print(f"CLIP embeddings shape: {clip_embeddings.shape}")
    print(f"CLIP embeddings are L2-normalized: {torch.allclose(clip_embeddings.norm(dim=-1), torch.ones(clip_embeddings.shape[0]))}")

    # Simulate text embeddings
    text_embeddings = torch.randn(clip_embeddings.shape[0], clip_embeddings.shape[1])
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)

    # Compute similarity
    similarity = torch.matmul(clip_embeddings, text_embeddings.T)
    print(f"Similarity matrix shape: {similarity.shape}")
    print(f"Similarity range: [{similarity.min().item():.3f}, {similarity.max().item():.3f}]")


def example_mlm(encoder, input_data, attention_mask):
    """Example 3: Using encoder for MLM."""
    print("\n" + "="*60)
    print("Example 3: MLM Training")
    print("="*60)

    with torch.no_grad():
        sequence_features = encoder.get_sequence_features(input_data, attention_mask=attention_mask)

    print(f"Sequence features shape: {sequence_features.shape}")

    # Simulate MLM head predictions
    vocab_size = 50
    batch_size, seq_len, d_model = sequence_features.shape

    # Simple linear classifier
    mlm_head = torch.nn.Linear(d_model, vocab_size)
    with torch.no_grad():
        logits = mlm_head(sequence_features)

    print(f"MLM logits shape: {logits.shape}")

    # Create mask positions (e.g., 15% of tokens)
    mask_positions = torch.rand(batch_size, seq_len) < 0.15
    # Don't mask padding tokens
    mask_positions = mask_positions & attention_mask

    print(f"Masked positions: {mask_positions.sum().item()} / {attention_mask.sum().item()} valid tokens")

    # Compute loss only on masked positions
    # Use first categorical field as target (sensor_id or sensor)
    first_field = list(input_data['categorical_features'].keys())[0]
    targets = input_data['categorical_features'][first_field]

    if mask_positions.sum() > 0:
        loss = torch.nn.functional.cross_entropy(
            logits[mask_positions],
            targets[mask_positions]
        )
        print(f"MLM loss (example on {first_field}): {loss.item():.4f}")
    else:
        print(f"No masked positions to compute loss on")


def example_minimal_encoder():
    """Example 4: Minimal encoder (no spatial/temporal features)."""
    print("\n" + "="*60)
    print("Example 4: Minimal Encoder (Ablation)")
    print("="*60)

    # Create minimal config
    config = TransformerEncoderConfig.tiny()
    config.metadata.use_coordinates = False
    config.metadata.use_time_deltas = False
    config.metadata.categorical_fields = ['sensor', 'state']
    config.vocab_sizes = {
        'sensor': 50,
        'state': 10,
    }

    print(f"\nMinimal config:")
    print(f"  - use_coordinates: {config.metadata.use_coordinates}")
    print(f"  - use_time_deltas: {config.metadata.use_time_deltas}")
    print(f"  - categorical_fields: {config.metadata.categorical_fields}")

    encoder = TransformerSensorEncoder(config)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")

    # Input only needs categorical features
    batch_size = 8
    seq_len = 50

    input_data = {
        'categorical_features': {
            'sensor': torch.randint(0, 50, (batch_size, seq_len)),
            'state': torch.randint(0, 10, (batch_size, seq_len)),
        }
        # No coordinates or time_deltas needed!
    }

    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    print(f"\nInput shapes (minimal):")
    print(f"  - sensor: {input_data['categorical_features']['sensor'].shape}")
    print(f"  - state: {input_data['categorical_features']['state'].shape}")

    # Forward pass
    with torch.no_grad():
        output = encoder(input_data, attention_mask=attention_mask)

    print(f"Output embeddings shape: {output.embeddings.shape}")
    print(f"Successfully encoded with minimal features!")


def example_variable_length():
    """Example 5: Variable-length sequences (like from fixed_duration sampler)."""
    print("\n" + "="*60)
    print("Example 5: Variable-Length Sequences")
    print("="*60)

    config = TransformerEncoderConfig.small()
    config.vocab_sizes = {'sensor': 50, 'state': 10, 'room_id': 20}
    encoder = TransformerSensorEncoder(config)

    # Simulate variable-length sequences
    batch_size = 4
    max_seq_len = 50

    # Different actual lengths per sample
    actual_lengths = [10, 25, 40, 50]

    input_data = {
        'categorical_features': {
            'sensor': torch.randint(0, 50, (batch_size, max_seq_len)),
            'state': torch.randint(0, 10, (batch_size, max_seq_len)),
            'room_id': torch.randint(0, 20, (batch_size, max_seq_len)),
        },
        'coordinates': torch.randn(batch_size, max_seq_len, 2),
        'time_deltas': torch.rand(batch_size, max_seq_len) * 100,
    }

    # Create attention masks based on actual lengths
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    for i, length in enumerate(actual_lengths):
        attention_mask[i, :length] = True

    print(f"Variable sequence lengths: {actual_lengths}")
    print(f"Max sequence length: {max_seq_len}")
    print(f"Attention mask shape: {attention_mask.shape}")

    # Forward pass
    with torch.no_grad():
        output = encoder(input_data, attention_mask=attention_mask)

    print(f"\nOutput embeddings shape: {output.embeddings.shape}")
    print(f"All samples encoded to same dimension despite different lengths!")

    # Verify padding is properly handled
    # Change values in padding region - should not affect output
    input_data_modified = {
        k: v.clone() if torch.is_tensor(v) else {kk: vv.clone() for kk, vv in v.items()}
        for k, v in input_data.items()
    }

    # Modify padding regions (set to different valid values)
    for i, length in enumerate(actual_lengths):
        if length < max_seq_len:
            # Use valid indices for categorical features
            input_data_modified['categorical_features']['sensor'][i, length:] = 0
            input_data_modified['categorical_features']['state'][i, length:] = 0
            input_data_modified['categorical_features']['room_id'][i, length:] = 0
            # Use extreme values for continuous features
            input_data_modified['coordinates'][i, length:] = 999.0
            input_data_modified['time_deltas'][i, length:] = 999.0

    with torch.no_grad():
        output_modified = encoder(input_data_modified, attention_mask=attention_mask)

    diff = (output.embeddings - output_modified.embeddings).abs().max().item()
    print(f"Max difference after modifying padding: {diff:.6f}")
    print(f"Padding properly ignored: {diff < 1e-5}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MODULAR ENCODER FRAMEWORK - USAGE EXAMPLES")
    print("="*70)

    # Run examples
    encoder, input_data, attention_mask = example_basic_usage()
    example_clip_alignment(encoder, input_data, attention_mask)
    example_mlm(encoder, input_data, attention_mask)
    example_minimal_encoder()
    example_variable_length()

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✅")
    print("="*70)
    print("\nThe encoder framework is ready for integration with:")
    print("  - Step 1: Sampling (load data from JSON)")
    print("  - Step 3: Captions (combine with text encodings)")
    print("  - Step 5: Alignment (CLIP training)")
    print("\nSee docs/ENCODER_GUIDE.md for more details.")
    print()

