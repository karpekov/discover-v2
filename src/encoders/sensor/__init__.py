"""
Sensor encoders module.

Contains different types of sensor sequence encoders:
- sequence/: Raw sequence encoders (transformers, etc.)
- image/: Image-based encoders (CLIP, DINO, etc.) [placeholder]
"""

from src.encoders.sensor.sequence.transformer import TransformerSensorEncoder

__all__ = [
    'TransformerSensorEncoder',
]

