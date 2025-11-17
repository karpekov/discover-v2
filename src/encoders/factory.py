"""
Factory functions for building encoders.
"""

from typing import Dict, Any, Optional
import yaml
from pathlib import Path
import logging

from src.encoders.config import TransformerEncoderConfig
from src.encoders.sensor.sequence import TransformerSensorEncoder
from src.encoders.sensor.sequence.image_transformer import ImageTransformerSensorEncoder

logger = logging.getLogger(__name__)


def build_encoder(
    config: Dict[str, Any] or str,
    dataset: Optional[str] = None,
    dataset_type: str = "casas",
    vocab: Optional[Dict[str, Dict[str, int]]] = None
):
    """
    Build an encoder from config dictionary or YAML path.

    Args:
        config: Either a dictionary with encoder config or path to YAML file
        dataset: Dataset name (required for image-based encoders, e.g., "milan")
        dataset_type: Dataset type (default: "casas")
        vocab: Vocabulary mapping (required for image-based encoders)

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

        # Check if using image embeddings
        if encoder_config.use_image_embeddings:
            # Build image-based transformer encoder
            if dataset is None:
                raise ValueError(
                    "dataset parameter is required for image-based encoders. "
                    "Please provide the dataset name (e.g., 'milan')"
                )
            if vocab is None:
                raise ValueError(
                    "vocab parameter is required for image-based encoders. "
                    "Vocabulary is needed to map sensor/state indices to strings."
                )
            if encoder_config.image_model_name is None:
                raise ValueError(
                    "image_model_name must be specified in config when use_image_embeddings=True. "
                    "Options: 'clip', 'dinov2', 'siglip'"
                )

            logger.info(
                f"Building ImageTransformerSensorEncoder with {encoder_config.image_model_name} embeddings"
            )

            encoder = ImageTransformerSensorEncoder(
                config=encoder_config,
                dataset=dataset,
                image_model_name=encoder_config.image_model_name,
                dataset_type=dataset_type,
                image_size=encoder_config.image_size,
                vocab=vocab,
                freeze_input_projection=encoder_config.freeze_input_projection
            )
        else:
            # Build standard sequence-based transformer encoder
            logger.info("Building TransformerSensorEncoder (sequence-based)")
            encoder = TransformerSensorEncoder(encoder_config)

    elif encoder_type == 'chronos':
        # TODO: Implement Chronos encoder
        raise NotImplementedError("Chronos encoder not yet implemented in new framework")
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

