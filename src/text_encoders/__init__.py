"""Text encoders for caption embedding.

This module provides frozen text encoders for embedding captions into fixed-dimensional
vectors. These embeddings can be pre-computed and cached since the text encoders remain
frozen during training.

Available encoders:
    - GTETextEncoder: thenlper/gte-base (768-d)
    - DistilRoBERTaTextEncoder: distilroberta-base (768-d)
    - LLAMATextEncoder: nvidia/llama-embed-nemotron-8b (4096-d)
    - MiniLMTextEncoder: sentence-transformers/all-MiniLM-L6-v2 (384-d)
    - EmbeddingGemmaTextEncoder: google/embeddinggemma-300m (768-d)
    - CLIPTextEncoder: OpenAI CLIP text encoder (512-d)
    - SigLIPTextEncoder: Google SigLIP text encoder (512-d)

Usage:
    >>> from text_encoders import GTETextEncoder, TextEncoderConfig
    >>>
    >>> config = TextEncoderConfig.from_yaml('configs/text_encoders/gte_base.yaml')
    >>> encoder = GTETextEncoder(config)
    >>>
    >>> captions = ["Person moves from kitchen to living room", "Activity in bedroom"]
    >>> embeddings = encoder.encode(captions)  # [2, 768]
"""

from .base import BaseTextEncoder, TextEncoderConfig, TextEncoderOutput
from .frozen import (
    GTETextEncoder,
    DistilRoBERTaTextEncoder,
    LLAMATextEncoder,
    MiniLMTextEncoder,
    EmbeddingGemmaTextEncoder,
    CLIPTextEncoder,
    SigLIPTextEncoder
)

__all__ = [
    # Base classes
    'BaseTextEncoder',
    'TextEncoderConfig',
    'TextEncoderOutput',
    
    # Frozen encoders
    'GTETextEncoder',
    'DistilRoBERTaTextEncoder',
    'LLAMATextEncoder',
    'MiniLMTextEncoder',
    'EmbeddingGemmaTextEncoder',
    'CLIPTextEncoder',
    'SigLIPTextEncoder',
]

