"""Rule-based caption generators."""

from .baseline import BaselineCaptionGenerator
from .sourish import SourishCaptionGenerator

__all__ = [
    'BaselineCaptionGenerator',
    'SourishCaptionGenerator',
]

