"""Base classes for caption generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd


@dataclass
class CaptionOutput:
    """Container for generated captions.

    Attributes:
        captions: List of caption strings
        sample_id: Identifier for the sample
        metadata: Additional metadata (e.g., caption_type, layer_b for enhanced captions)
    """
    captions: List[str]
    sample_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'sample_id': self.sample_id,
            'captions': self.captions,
            'metadata': self.metadata
        }


class BaseCaptionGenerator(ABC):
    """Abstract base class for caption generators.

    All caption generators (rule-based, LLM-based) should inherit from this class.
    """

    def __init__(self, config: Any):
        """Initialize the caption generator.

        Args:
            config: Configuration object (CaptionConfig or subclass)
        """
        self.config = config

    @abstractmethod
    def generate(self,
                 sensor_sequence: List[Dict[str, Any]],
                 metadata: Dict[str, Any],
                 sample_id: str) -> CaptionOutput:
        """Generate captions for a sensor sequence.

        Args:
            sensor_sequence: List of sensor events, each with:
                - sensor_id: Sensor identifier
                - event_type/state: Event type (ON, OFF, etc.)
                - timestamp/datetime: Event timestamp
                - room/room_id: Room identifier
                - x, y: Coordinates (if available)
                - Additional metadata
            metadata: Sample metadata with:
                - start_time: Start timestamp
                - end_time: End timestamp
                - duration_seconds: Duration
                - num_events: Number of events
                - rooms_visited: List of rooms
                - ground_truth_labels: Ground truth (if available)
            sample_id: Unique identifier for the sample

        Returns:
            CaptionOutput with generated captions
        """
        pass

    @abstractmethod
    def generate_batch(self,
                      samples: List[Dict[str, Any]]) -> List[CaptionOutput]:
        """Generate captions for a batch of samples.

        Args:
            samples: List of sample dictionaries, each containing:
                - sample_id: Unique identifier
                - sensor_sequence: List of sensor events
                - metadata: Sample metadata

        Returns:
            List of CaptionOutput objects
        """
        pass

    def get_statistics(self, caption_outputs: List[CaptionOutput]) -> Dict[str, Any]:
        """Compute statistics about generated captions.

        Args:
            caption_outputs: List of CaptionOutput objects

        Returns:
            Dictionary with statistics
        """
        if not caption_outputs:
            return {}

        total_captions = sum(len(co.captions) for co in caption_outputs)
        lengths = [len(caption.split()) for co in caption_outputs for caption in co.captions]

        import numpy as np
        return {
            'total_samples': len(caption_outputs),
            'total_captions': total_captions,
            'avg_captions_per_sample': total_captions / len(caption_outputs),
            'caption_length_stats': {
                'mean_tokens': np.mean(lengths) if lengths else 0,
                'std_tokens': np.std(lengths) if lengths else 0,
                'min_tokens': np.min(lengths) if lengths else 0,
                'max_tokens': np.max(lengths) if lengths else 0
            },
            'sample_captions': caption_outputs[0].captions if caption_outputs else []
        }

