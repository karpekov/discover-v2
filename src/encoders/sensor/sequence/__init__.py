"""
Sequence-based sensor encoders.

These encoders process raw sensor sequences with categorical and continuous features.
"""

from src.encoders.sensor.sequence.transformer import TransformerSensorEncoder

__all__ = [
    'TransformerSensorEncoder',
]

