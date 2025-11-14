"""
Configuration dataclasses for encoders.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetadataConfig:
    """
    Configuration for which metadata features to use in encoding.

    This allows flexible control over what information is fed to the encoder:
    - categorical_fields: sensor, state, room_id, etc.
    - use_coordinates: Include x,y spatial information
    - use_time_deltas: Include temporal information
    - use_time_of_day: Include cyclical time-of-day encoding
    """
    categorical_fields: List[str] = field(default_factory=lambda: ['sensor', 'state', 'room_id'])
    use_coordinates: bool = True
    use_time_deltas: bool = True
    use_time_of_day: bool = False  # Future: cyclical hour/day encoding

    # Coordinate normalization bounds (for Fourier features)
    coord_norm_x_max: float = 10.0
    coord_norm_y_max: float = 10.0

    # Time delta bucketing
    time_delta_max_seconds: float = 3600.0  # 1 hour
    time_delta_bins: int = 100


@dataclass
class EncoderConfig:
    """Base configuration for all encoders."""
    encoder_type: str  # 'transformer', 'chronos', 'clip_image', etc.
    d_model: int = 768
    projection_dim: int = 512  # For CLIP alignment
    max_seq_len: int = 512
    dropout: float = 0.1


@dataclass
class TransformerEncoderConfig(EncoderConfig):
    """
    Configuration for transformer-based raw sequence encoder.

    This is the modular replacement for the original SensorEncoder.
    """
    encoder_type: str = 'transformer'

    # Architecture
    d_model: int = 768
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 3072
    max_seq_len: int = 512
    dropout: float = 0.1

    # Projection for CLIP
    projection_dim: int = 512
    projection_type: str = 'linear'  # 'linear' or 'mlp'
    projection_hidden_dim: int = 2048  # Hidden dimension for MLP projection
    projection_num_layers: int = 2  # Number of layers in MLP (2 or 3)

    # Positional encoding
    use_alibi: bool = True  # Use ALiBi attention biases
    use_learned_pe: bool = False  # Use learned positional embeddings (if not ALiBi)

    # Feature encoding
    fourier_bands: int = 12  # For coordinate encoding
    metadata: MetadataConfig = field(default_factory=MetadataConfig)

    # Vocabulary sizes (will be set at runtime from data)
    vocab_sizes: Dict[str, int] = field(default_factory=dict)

    # Pooling strategy
    pooling: str = 'cls_mean'  # 'cls', 'mean', 'cls_mean'
    pooling_cls_weight: float = 0.5  # Weight for CLS in cls_mean pooling

    @classmethod
    def from_dict(cls, config_dict: Dict[str, any]):
        """Create config from dictionary."""
        # Handle nested metadata config
        if 'metadata' in config_dict and isinstance(config_dict['metadata'], dict):
            config_dict = config_dict.copy()
            config_dict['metadata'] = MetadataConfig(**config_dict['metadata'])

        # Filter to only valid fields for this dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        return cls(**filtered_dict)

    # Model size presets
    @staticmethod
    def tiny():
        """Tiny model preset."""
        return TransformerEncoderConfig(
            d_model=256,
            n_layers=4,
            n_heads=4,
            d_ff=1024,
            projection_dim=256,
        )

    @staticmethod
    def small():
        """Small model preset."""
        return TransformerEncoderConfig(
            d_model=512,
            n_layers=6,
            n_heads=8,
            d_ff=2048,
            projection_dim=512,
        )

    @staticmethod
    def base():
        """Base model preset (default)."""
        return TransformerEncoderConfig(
            d_model=768,
            n_layers=6,
            n_heads=8,
            d_ff=3072,
            projection_dim=512,
        )

    @staticmethod
    def large():
        """Large model preset."""
        return TransformerEncoderConfig(
            d_model=1024,
            n_layers=12,
            n_heads=16,
            d_ff=4096,
            projection_dim=512,
        )


@dataclass
class ImageEncoderConfig(EncoderConfig):
    """
    Configuration for image-based encoders (future implementation).

    These encoders visualize sensor activations on floor plans and use
    vision models (CLIP, DINO, etc.) to encode them.
    """
    encoder_type: str = 'image_clip'

    # Image generation
    image_size: int = 224
    use_floor_plan: bool = True
    sensor_marker_size: int = 10
    color_by_sensor_type: bool = True

    # Vision model
    vision_model: str = 'clip'  # 'clip', 'siglip', 'dino', 'video'
    vision_model_name: str = 'ViT-B/32'
    freeze_vision_model: bool = True

    # Sequence aggregation
    aggregation: str = 'transformer'  # 'mean', 'attention', 'transformer'
    aggregation_layers: int = 2  # If using transformer aggregation

    d_model: int = 768
    projection_dim: int = 512

