"""LLM-based caption generation.

This module supports:
- OpenAI API (GPT-4, GPT-3.5)
- Google API (Gemini)
- Local models (Gemma, Llama via Hugging Face)
"""

from typing import List, Dict, Any, Optional

from ..base import BaseCaptionGenerator, CaptionOutput
from ..config import LLMCaptionConfig
from .compact_json import to_compact_caption_json
from .prompts import build_user_prompt, build_multi_sample_prompt, SYSTEM_PROMPT
from .backends import create_backend, CaptionModel
import json


class LLMCaptionGenerator(BaseCaptionGenerator):
    """Generate captions using Large Language Models.

    Supports pluggable backends:
    - Local: Gemma, Llama (via Hugging Face Transformers)
    - Remote: OpenAI (GPT), Google (Gemini)
    """

    def __init__(self, config: LLMCaptionConfig):
        """Initialize LLM caption generator.

        Args:
            config: LLMCaptionConfig with backend settings
        """
        super().__init__(config)

        # Create backend
        self.backend = create_backend(
            backend_type=config.backend_type,
            model_name=config.model_name,
            num_captions=config.num_captions_per_sample,
            temperature=config.temperature,
            api_key=config.api_key,
            device=config.device
        )

        print(f"Initialized LLMCaptionGenerator with backend: {config.backend_type}, model: {config.model_name}")

    def generate(self,
                 sensor_sequence: List[Dict[str, Any]],
                 metadata: Dict[str, Any],
                 sample_id: str) -> CaptionOutput:
        """Generate captions using an LLM.

        Args:
            sensor_sequence: List of sensor events
            metadata: Sample metadata
            sample_id: Unique sample identifier

        Returns:
            CaptionOutput with generated captions
        """
        # Convert to compact JSON
        sample_dict = {
            'sample_id': sample_id,
            'sensor_sequence': sensor_sequence,
            'metadata': metadata
        }
        compact_json = to_compact_caption_json(sample_dict)

        # Build prompt
        user_prompt = build_user_prompt(compact_json, self.config.num_captions_per_sample)

        # Generate captions using backend
        caption_lists = self.backend.generate([user_prompt])
        captions = caption_lists[0]  # Get captions for this single sample

        return CaptionOutput(
            captions=captions,
            sample_id=sample_id,
            metadata={
                'caption_type': 'llm',
                'backend_type': self.config.backend_type,
                'model_name': self.config.model_name,
                'num_requested': self.config.num_captions_per_sample,
                'num_generated': len(captions)
            }
        )

    def generate_batch(self, samples: List[Dict[str, Any]],
                      use_multi_sample: bool = True,
                      multi_sample_size: int = 5) -> List[CaptionOutput]:
        """Generate captions for a batch of samples.

        Args:
            samples: List of sample dictionaries with sensor_sequence, metadata, sample_id
            use_multi_sample: If True, pack multiple samples into single API calls to save tokens
            multi_sample_size: Number of samples per multi-sample prompt

        Returns:
            List of CaptionOutput objects
        """
        # Convert all samples to compact JSON
        compact_jsons = []
        sample_ids = []

        for sample in samples:
            compact_json = to_compact_caption_json(sample)
            compact_jsons.append(compact_json)
            sample_ids.append(sample['sample_id'])

        outputs = []

        if use_multi_sample and len(samples) > 1:
            # Process multiple samples per API call to save tokens
            for i in range(0, len(compact_jsons), multi_sample_size):
                batch_compact = compact_jsons[i:i + multi_sample_size]
                batch_ids = sample_ids[i:i + multi_sample_size]

                # Build multi-sample prompt
                multi_prompt = build_multi_sample_prompt(batch_compact, self.config.num_captions_per_sample)

                # Generate for all samples in one call
                try:
                    caption_lists = self.backend.generate([multi_prompt])
                    response_text = caption_lists[0][0] if caption_lists and caption_lists[0] else None

                    if response_text:
                        # Try to parse as JSON object mapping sample_id -> captions
                        try:
                            # Remove markdown if present
                            clean_text = response_text.strip()
                            if clean_text.startswith('```'):
                                lines = clean_text.split('\n')
                                clean_text = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

                            captions_dict = json.loads(clean_text)

                            # Extract captions for each sample
                            for sample_id in batch_ids:
                                captions = captions_dict.get(sample_id, ["Activity detected."] * self.config.num_captions_per_sample)
                                if not isinstance(captions, list):
                                    captions = [str(captions)]

                                # Ensure correct number
                                captions = captions[:self.config.num_captions_per_sample]
                                while len(captions) < self.config.num_captions_per_sample:
                                    captions.append(captions[0] if captions else "Activity detected.")

                                output = CaptionOutput(
                                    captions=captions,
                                    sample_id=sample_id,
                                    metadata={
                                        'caption_type': 'llm',
                                        'backend_type': self.config.backend_type,
                                        'model_name': self.config.model_name,
                                        'num_requested': self.config.num_captions_per_sample,
                                        'num_generated': len(captions),
                                        'multi_sample_batch': True
                                    }
                                )
                                outputs.append(output)
                        except json.JSONDecodeError:
                            # Fallback to individual generation for this batch
                            print(f"Warning: Failed to parse multi-sample response, falling back to individual generation")
                            for cj, sid in zip(batch_compact, batch_ids):
                                user_prompt = build_user_prompt(cj, self.config.num_captions_per_sample)
                                caption_lists = self.backend.generate([user_prompt])
                                captions = caption_lists[0]
                                outputs.append(CaptionOutput(
                                    captions=captions,
                                    sample_id=sid,
                                    metadata={'caption_type': 'llm', 'backend_type': self.config.backend_type,
                                            'model_name': self.config.model_name}
                                ))
                    else:
                        # No response, use fallback
                        for sid in batch_ids:
                            outputs.append(CaptionOutput(
                                captions=["Activity detected."] * self.config.num_captions_per_sample,
                                sample_id=sid,
                                metadata={'caption_type': 'llm', 'backend_type': self.config.backend_type}
                            ))
                except Exception as e:
                    print(f"Warning: Multi-sample generation failed: {e}, falling back")
                    # Fallback to individual generation
                    for cj, sid in zip(batch_compact, batch_ids):
                        user_prompt = build_user_prompt(cj, self.config.num_captions_per_sample)
                        caption_lists = self.backend.generate([user_prompt])
                        captions = caption_lists[0]
                        outputs.append(CaptionOutput(
                            captions=captions,
                            sample_id=sid,
                            metadata={'caption_type': 'llm', 'backend_type': self.config.backend_type,
                                    'model_name': self.config.model_name}
                        ))
        else:
            # Original behavior: one sample per prompt
            user_prompts = [
                build_user_prompt(cj, self.config.num_captions_per_sample)
                for cj in compact_jsons
            ]

            # Generate captions for all prompts
            all_caption_lists = self.backend.generate(user_prompts)

            # Create CaptionOutput for each sample
            for sample_id, captions in zip(sample_ids, all_caption_lists):
                output = CaptionOutput(
                    captions=captions,
                    sample_id=sample_id,
                    metadata={
                        'caption_type': 'llm',
                        'backend_type': self.config.backend_type,
                        'model_name': self.config.model_name,
                        'num_requested': self.config.num_captions_per_sample,
                        'num_generated': len(captions)
                    }
                )
                outputs.append(output)

        return outputs

