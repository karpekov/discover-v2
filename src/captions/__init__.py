"""Caption generation for sensor sequences.

This module provides rule-based and LLM-based caption generation
for sensor reading sequences.
"""

from .base import BaseCaptionGenerator, CaptionOutput
from .config import CaptionConfig, RuleBasedCaptionConfig, LLMCaptionConfig
from .rule_based.baseline import BaselineCaptionGenerator
from .rule_based.sourish import SourishCaptionGenerator
from .llm_based.base import LLMCaptionGenerator

__all__ = [
    'BaseCaptionGenerator',
    'CaptionOutput',
    'CaptionConfig',
    'RuleBasedCaptionConfig',
    'LLMCaptionConfig',
    'BaselineCaptionGenerator',
    'SourishCaptionGenerator',
    'LLMCaptionGenerator',
]

