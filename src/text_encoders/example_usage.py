"""Example usage of the text encoder framework.

This script demonstrates how to use text encoders programmatically.

Run from project root:
    conda activate discover-v2-env
    python src/text_encoders/example_usage.py
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from text_encoders import (
    GTETextEncoder,
    DistilRoBERTaTextEncoder,
    LLAMATextEncoder,
    CLIPTextEncoder,
    TextEncoderConfig
)


def example_1_basic_encoding():
    """Example 1: Basic caption encoding with GTE."""
    print("\n" + "="*80)
    print("Example 1: Basic Caption Encoding (GTE)")
    print("="*80)
    
    # Sample captions
    captions = [
        "Person moves from kitchen to living room in the morning",
        "Motion detected in bedroom during night time",
        "Multiple sensor activations in bathroom"
    ]
    
    # Create config (device will be auto-detected)
    config = TextEncoderConfig(
        encoder_type='gte',
        model_name='thenlper/gte-base',
        embedding_dim=768,
        normalize=True
    )
    
    # Initialize encoder
    encoder = GTETextEncoder(config)
    
    # Encode captions
    output = encoder.encode(captions)
    
    print(f"\nInput: {len(captions)} captions")
    print(f"Output shape: {output.embeddings.shape}")
    print(f"Embedding dim: {output.metadata['embedding_dim']}")
    print(f"Model: {output.metadata['model_name']}")
    
    # Check normalization
    norms = np.linalg.norm(output.embeddings, axis=1)
    print(f"\nEmbedding norms (should be ~1.0 if normalized):")
    for i, norm in enumerate(norms):
        print(f"  Caption {i+1}: {norm:.6f}")
    
    print("\n✓ Example 1 completed successfully!")


def example_2_with_projection():
    """Example 2: Encoding with projection head (for CLIP)."""
    print("\n" + "="*80)
    print("Example 2: Encoding with Projection Head")
    print("="*80)
    
    captions = [
        "Kitchen activity in the morning",
        "Bedroom activity at night"
    ]
    
    # Config with projection (device will be auto-detected)
    config = TextEncoderConfig(
        encoder_type='gte',
        model_name='thenlper/gte-base',
        embedding_dim=768,
        use_projection=True,
        projection_dim=512,
        normalize=True
    )
    
    encoder = GTETextEncoder(config)
    output = encoder.encode(captions)
    
    print(f"\nBase embedding dim: {config.embedding_dim}")
    print(f"Projected dim: {config.projection_dim}")
    print(f"Output shape: {output.embeddings.shape}")
    
    # Verify projection worked
    assert output.embeddings.shape[1] == 512, "Projection dimension mismatch!"
    
    print("\n✓ Example 2 completed successfully!")


def example_3_batch_encoding():
    """Example 3: Batch encoding for large datasets."""
    print("\n" + "="*80)
    print("Example 3: Batch Encoding")
    print("="*80)
    
    # Simulate large number of captions
    num_captions = 100
    captions = [f"Caption {i}: Activity in room" for i in range(num_captions)]
    
    config = TextEncoderConfig(
        encoder_type='gte',
        model_name='thenlper/gte-base',
        embedding_dim=768,
        batch_size=16,  # Process 16 at a time
        normalize=True
    )
    
    encoder = GTETextEncoder(config)
    
    # Encode in batches
    output = encoder.encode_batch(captions, batch_size=16)
    
    print(f"\nTotal captions: {num_captions}")
    print(f"Batch size: 16")
    print(f"Number of batches: {output.metadata['num_batches']}")
    print(f"Output shape: {output.embeddings.shape}")
    
    print("\n✓ Example 3 completed successfully!")


def example_4_save_and_load():
    """Example 4: Save and load embeddings."""
    print("\n" + "="*80)
    print("Example 4: Save and Load Embeddings")
    print("="*80)
    
    captions = [
        "Morning activity in kitchen",
        "Evening activity in bedroom"
    ]
    
    config = TextEncoderConfig(
        encoder_type='gte',
        model_name='thenlper/gte-base',
        embedding_dim=768,
        normalize=True
    )
    
    encoder = GTETextEncoder(config)
    print(f"Using device: {encoder.device} (auto-detected)")
    
    # Encode
    output = encoder.encode(captions)
    print(f"Encoded {output.embeddings.shape[0]} captions")
    
    # Save
    temp_path = '/tmp/test_embeddings.npz'
    encoder.save_embeddings(
        output.embeddings,
        temp_path,
        metadata={'caption_style': 'baseline', 'dataset': 'test'}
    )
    print(f"Saved embeddings to: {temp_path}")
    
    # Load
    loaded_embeddings, loaded_metadata = encoder.load_embeddings(temp_path)
    print(f"Loaded embeddings shape: {loaded_embeddings.shape}")
    print(f"Loaded metadata: {loaded_metadata}")
    
    # Verify they match
    assert np.allclose(output.embeddings, loaded_embeddings), "Embeddings don't match!"
    
    print("\n✓ Example 4 completed successfully!")


def example_5_compare_encoders():
    """Example 5: Compare different encoders."""
    print("\n" + "="*80)
    print("Example 5: Compare Different Encoders")
    print("="*80)
    
    caption = "Person moves from kitchen to living room"
    
    # GTE encoder (device auto-detected)
    config_gte = TextEncoderConfig(
        encoder_type='gte',
        model_name='thenlper/gte-base',
        embedding_dim=768,
        normalize=True
    )
    encoder_gte = GTETextEncoder(config_gte)
    output_gte = encoder_gte.encode([caption])
    
    # DistilRoBERTa encoder (device auto-detected)
    config_roberta = TextEncoderConfig(
        encoder_type='distilroberta',
        model_name='distilroberta-base',
        embedding_dim=768,
        normalize=True
    )
    encoder_roberta = DistilRoBERTaTextEncoder(config_roberta)
    output_roberta = encoder_roberta.encode([caption])
    
    print(f"\nCaption: '{caption}'")
    print(f"\nGTE embedding:")
    print(f"  Shape: {output_gte.embeddings.shape}")
    print(f"  Norm: {np.linalg.norm(output_gte.embeddings[0]):.6f}")
    print(f"  First 5 values: {output_gte.embeddings[0, :5]}")
    
    print(f"\nDistilRoBERTa embedding:")
    print(f"  Shape: {output_roberta.embeddings.shape}")
    print(f"  Norm: {np.linalg.norm(output_roberta.embeddings[0]):.6f}")
    print(f"  First 5 values: {output_roberta.embeddings[0, :5]}")
    
    # Compute similarity
    similarity = np.dot(output_gte.embeddings[0], output_roberta.embeddings[0])
    print(f"\nCosine similarity between encoders: {similarity:.4f}")
    
    print("\n✓ Example 5 completed successfully!")


def example_6_from_yaml():
    """Example 6: Load encoder from YAML config."""
    print("\n" + "="*80)
    print("Example 6: Load Encoder from YAML Config")
    print("="*80)
    
    # Load config from YAML
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'text_encoders' / 'gte_base.yaml'
    
    if not config_path.exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("   Skipping this example")
        return
    
    config = TextEncoderConfig.from_yaml(str(config_path))
    
    print(f"Loaded config from: {config_path.name}")
    print(f"  Encoder: {config.encoder_type}")
    print(f"  Model: {config.model_name}")
    print(f"  Embedding dim: {config.embedding_dim}")
    
    # Initialize encoder
    encoder = GTETextEncoder(config)
    
    # Encode
    caption = "Test caption from YAML config"
    output = encoder.encode([caption])
    
    print(f"\nEncoded caption:")
    print(f"  Output shape: {output.embeddings.shape}")
    
    print("\n✓ Example 6 completed successfully!")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("Text Encoder Framework - Example Usage")
    print("="*80)
    
    try:
        example_1_basic_encoding()
        example_2_with_projection()
        example_3_batch_encoding()
        example_4_save_and_load()
        example_5_compare_encoders()
        example_6_from_yaml()
        
        print("\n" + "="*80)
        print("All examples completed successfully! ✓")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

