"""LLM-based caption generation."""

from .base import LLMCaptionGenerator
from .backends import (
    CaptionModel,
    GemmaHFBackend,
    LlamaHFBackend,
    OpenAIBackend,
    GeminiBackend,
    create_backend
)
from .compact_json import to_compact_caption_json, convert_samples_to_compact_jsonl
from .prompts import SYSTEM_PROMPT, build_user_prompt, build_full_prompt, build_prompts_batch

__all__ = [
    'LLMCaptionGenerator',
    'CaptionModel',
    'GemmaHFBackend',
    'LlamaHFBackend',
    'OpenAIBackend',
    'GeminiBackend',
    'create_backend',
    'to_compact_caption_json',
    'convert_samples_to_compact_jsonl',
    'SYSTEM_PROMPT',
    'build_user_prompt',
    'build_full_prompt',
    'build_prompts_batch',
]
