"""
Factory functions for building encoders.
"""

from typing import Dict, Any
import yaml
from pathlib import Path

from src.encoders.config import TransformerEncoderConfig
from src.encoders.sensor.sequence import TransformerSensorEncoder


def build_encoder(config: Dict[str, Any] or str):
    """
    Build an encoder from config dictionary or YAML path.

    Args:
        config: Either a dictionary with encoder config or path to YAML file

    Returns:
        Encoder instance
    """
    # Load config from file if string path provided
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

    # Get encoder type
    encoder_type = config.get('type', 'transformer')

    if encoder_type == 'transformer':
        # Build transformer encoder
        encoder_config = TransformerEncoderConfig.from_dict(config)
        encoder = TransformerSensorEncoder(encoder_config)
    elif encoder_type == 'chronos':
        # TODO: Implement Chronos encoder
        raise NotImplementedError("Chronos encoder not yet implemented in new framework")
    elif encoder_type == 'image':
        # TODO: Implement image-based encoder
        raise NotImplementedError("Image-based encoder not yet implemented")
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return encoder


def load_encoder_config(config_path: str) -> TransformerEncoderConfig:
    """
    Load encoder config from YAML file.

    Args:
        config_path: Path to encoder config YAML

    Returns:
        EncoderConfig instance
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    encoder_type = config_dict.get('type', 'transformer')

    if encoder_type == 'transformer':
        return TransformerEncoderConfig.from_dict(config_dict)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

