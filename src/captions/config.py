"""Configuration classes for caption generation."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class CaptionConfig:
    """Base configuration for caption generation."""

    # General settings
    caption_style: str = 'baseline'  # 'baseline', 'sourish', 'mixed', 'llm'
    num_captions_per_sample: int = 2  # Number of captions to generate per sample

    # Random seed for reproducibility
    random_seed: int = 42

    # Dataset information (needed for some generators)
    dataset_name: Optional[str] = None  # 'milan', 'aruba', 'cairo', etc.

    # Metadata paths
    sensor_details_path: Optional[str] = None  # Path to sensor metadata (if needed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'caption_style': self.caption_style,
            'num_captions_per_sample': self.num_captions_per_sample,
            'random_seed': self.random_seed,
            'dataset_name': self.dataset_name,
            'sensor_details_path': self.sensor_details_path
        }


@dataclass
class RuleBasedCaptionConfig(CaptionConfig):
    """Configuration for rule-based caption generation."""

    caption_style: str = 'baseline'  # 'baseline', 'sourish', 'mixed'

    # Caption generation options
    generate_long_captions: bool = True  # Generate long detailed captions
    generate_short_captions: bool = False  # Generate short creative captions (disabled by default)

    # For baseline style
    include_temporal_context: bool = True  # Include time of day, day of week
    include_duration_details: bool = True  # Include duration descriptions
    include_sensor_details: bool = True  # Include sensor-specific details

    # For mixed style
    mix_strategies: List[str] = field(default_factory=lambda: ['baseline', 'sourish'])
    mix_probabilities: Optional[List[float]] = None  # If None, uniform distribution


@dataclass
class LLMCaptionConfig(CaptionConfig):
    """Configuration for LLM-based caption generation (placeholder)."""

    caption_style: str = 'llm'

    # LLM settings
    llm_provider: str = 'openai'  # 'openai', 'anthropic', 'google', 'local'
    llm_model: str = 'gpt-4'  # Model name

    # API settings
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None

    # Local model settings
    local_model_path: Optional[str] = None  # For local models like Llama

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 0.9

    # Prompt template
    prompt_template: Optional[str] = None

