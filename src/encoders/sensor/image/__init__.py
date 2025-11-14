"""
Image-based sensor encoders.

These encoders visualize sensor activations on floor plans and use
vision models to encode them.

Components:
- generate_images: Generate sensor activation images for a dataset
- embed_images: Embed images using vision models (CLIP, SigLIP, etc.)
- Image sequence encoders (future): Process sequences of embeddings
"""

from src.encoders.sensor.image.generate_images import (
    generate_dataset_images,
    get_image_path,
    load_image_metadata,
)
from src.encoders.sensor.image.embed_images import (
    embed_dataset_images,
    load_embeddings,
    get_sensor_embedding,
    get_embedder,
)

__all__ = [
    # Image generation
    'generate_dataset_images',
    'get_image_path',
    'load_image_metadata',
    # Image embedding
    'embed_dataset_images',
    'load_embeddings',
    'get_sensor_embedding',
    'get_embedder',
]

# Placeholder - will be implemented in the future

