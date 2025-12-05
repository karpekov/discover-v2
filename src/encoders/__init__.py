"""
Encoder module for smart home sensor sequences.

This module provides a modular encoder framework for processing sensor sequences
into embeddings. Supports multiple encoder architectures:
- Raw sequence encoders (transformers)
- Image-based encoders (vision transformers, CLIP, etc.)

All encoders follow a common interface defined in base.py.
"""

from encoders.base import BaseEncoder, EncoderOutput
from encoders.config import EncoderConfig, TransformerEncoderConfig, MetadataConfig
from encoders.factory import build_encoder, load_encoder_config

__all__ = [
    'BaseEncoder',
    'EncoderOutput',
    'EncoderConfig',
    'TransformerEncoderConfig',
    'MetadataConfig',
    'build_encoder',
    'load_encoder_config',
]

