"""LLM-based caption generation (placeholder implementation).

This module will eventually support:
- OpenAI API (GPT-4, GPT-3.5)
- Anthropic API (Claude)
- Google API (Gemini)
- Local models (Llama, Mistral, etc.)
"""

from typing import List, Dict, Any
import warnings

from ..base import BaseCaptionGenerator, CaptionOutput
from ..config import LLMCaptionConfig


class LLMCaptionGenerator(BaseCaptionGenerator):
    """Generate captions using Large Language Models (placeholder).

    TODO: Implement LLM-based caption generation in future versions.
    This placeholder allows the pipeline to be designed with LLM support in mind.
    """

    def __init__(self, config: LLMCaptionConfig):
        super().__init__(config)
        warnings.warn(
            "LLMCaptionGenerator is a placeholder. "
            "LLM-based caption generation is not yet implemented.",
            FutureWarning
        )

    def generate(self,
                 sensor_sequence: List[Dict[str, Any]],
                 metadata: Dict[str, Any],
                 sample_id: str) -> CaptionOutput:
        """Generate captions using an LLM (placeholder).

        Future implementation will:
        1. Format sensor sequence into a structured prompt
        2. Call LLM API or local model
        3. Parse and validate LLM response
        4. Return captions
        """
        # Placeholder: Return a simple caption
        caption_text = self._generate_placeholder_caption(sensor_sequence, metadata)

        return CaptionOutput(
            captions=[caption_text],
            sample_id=sample_id,
            metadata={
                'caption_type': 'llm_placeholder',
                'llm_provider': self.config.llm_provider,
                'llm_model': self.config.llm_model
            }
        )

    def generate_batch(self, samples: List[Dict[str, Any]]) -> List[CaptionOutput]:
        """Generate captions for a batch of samples (placeholder)."""
        outputs = []
        for sample in samples:
            output = self.generate(
                sensor_sequence=sample['sensor_sequence'],
                metadata=sample['metadata'],
                sample_id=sample['sample_id']
            )
            outputs.append(output)
        return outputs

    def _generate_placeholder_caption(self,
                                     sensor_sequence: List[Dict[str, Any]],
                                     metadata: Dict[str, Any]) -> str:
        """Generate a simple placeholder caption."""
        num_events = len(sensor_sequence)
        duration_sec = metadata.get('duration_seconds', 0)
        duration_min = round(duration_sec / 60.0, 1)

        # Extract unique rooms
        rooms = set()
        for event in sensor_sequence:
            room = event.get('room_id', event.get('room', 'unknown'))
            if room:
                rooms.add(room)

        if len(rooms) == 1:
            room_desc = f"in {list(rooms)[0]}"
        elif len(rooms) > 1:
            room_desc = f"across {len(rooms)} rooms"
        else:
            room_desc = "in the house"

        return f"Sensor activity detected {room_desc} with {num_events} events over {duration_min} minutes. (LLM placeholder)"

    # Future methods to implement:

    def _format_prompt(self, sensor_sequence: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        """Format sensor data into LLM prompt (TODO)."""
        raise NotImplementedError("LLM prompt formatting not yet implemented")

    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API (TODO)."""
        raise NotImplementedError("OpenAI API integration not yet implemented")

    def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API (TODO)."""
        raise NotImplementedError("Anthropic API integration not yet implemented")

    def _call_google_api(self, prompt: str) -> str:
        """Call Google API (TODO)."""
        raise NotImplementedError("Google API integration not yet implemented")

    def _call_local_model(self, prompt: str) -> str:
        """Call local LLM model (TODO)."""
        raise NotImplementedError("Local model integration not yet implemented")

