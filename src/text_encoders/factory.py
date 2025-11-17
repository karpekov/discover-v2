"""
Factory functions for building text encoders.
"""

from typing import Dict, Any
import yaml

from .base import TextEncoderConfig
from .frozen import (
    GTETextEncoder,
    DistilRoBERTaTextEncoder,
    LLAMATextEncoder,
    MiniLMTextEncoder,
    EmbeddingGemmaTextEncoder,
    CLIPTextEncoder,
    SigLIPTextEncoder
)


def build_text_encoder(config: Dict[str, Any] or str):
    """
    Build a text encoder from config dictionary or YAML path.

    Args:
        config: Either a dictionary with text encoder config or path to YAML file

    Returns:
        Text encoder instance
    """
    # Load config from file if string path provided
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

    # Create config object
    text_config = TextEncoderConfig.from_dict(config)

    # Build encoder based on model name
    model_name = text_config.model_name.lower()

    if 'gte' in model_name:
        encoder = GTETextEncoder(text_config)
    elif 'distilroberta' in model_name or 'distilbert' in model_name:
        encoder = DistilRoBERTaTextEncoder(text_config)
    elif 'llama' in model_name and 'embed' in model_name:
        encoder = LLAMATextEncoder(text_config)
    elif 'minilm' in model_name or 'all-minilm' in model_name:
        encoder = MiniLMTextEncoder(text_config)
    elif 'gemma' in model_name and 'embedding' in model_name:
        encoder = EmbeddingGemmaTextEncoder(text_config)
    elif 'clip' in model_name and 'openai' in model_name:
        encoder = CLIPTextEncoder(text_config)
    elif 'siglip' in model_name:
        encoder = SigLIPTextEncoder(text_config)
    else:
        # Default to GTE if unknown
        print(f"Warning: Unknown model name '{model_name}', defaulting to GTETextEncoder")
        encoder = GTETextEncoder(text_config)

    return encoder

