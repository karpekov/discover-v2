"""LLM backend implementations for caption generation."""

from typing import List, Protocol, Dict, Any, Optional
import json
import warnings


class CaptionModel(Protocol):
    """Protocol for LLM caption generation backends."""

    def generate(self, prompts: List[str]) -> List[List[str]]:
        """Generate captions for a list of prompts.

        Args:
            prompts: List of user prompts (one per sample)

        Returns:
            List of caption lists (one list of captions per prompt)
        """
        ...


# ============================================================================
# Hugging Face Local Backends
# ============================================================================


class GemmaHFBackend:
    """Gemma model backend via Hugging Face Transformers."""

    def __init__(self,
                 model_name: str = "google/gemma-7b",
                 num_captions: int = 4,
                 max_new_tokens: int = 128,
                 temperature: float = 0.9,
                 top_p: float = 0.95,
                 device: Optional[str] = None):
        """Initialize Gemma backend.

        Args:
            model_name: Hugging Face model identifier
            num_captions: Number of captions to generate per sample
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            device: Device to use (None = auto)
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError("Install transformers and torch: pip install transformers torch")

        self.num_captions = num_captions
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"Loading Gemma model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model with appropriate dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if device is None else device
        )
        self.model.eval()
        print(f"Model loaded on device: {self.model.device}")

    def generate(self, prompts: List[str]) -> List[List[str]]:
        """Generate captions for prompts."""
        import torch

        all_captions = []

        for prompt in prompts:
            # Generate multiple captions for this prompt
            captions = []

            for _ in range(self.num_captions):
                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                # Decode (skip input prompt)
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )

                # Try to parse as JSON list
                try:
                    parsed = self._parse_output(generated_text)
                    captions.extend(parsed)
                    break  # Successfully got captions
                except Exception:
                    # If parsing fails, use raw text as single caption
                    captions.append(generated_text.strip())

            # Ensure we have the right number
            captions = captions[:self.num_captions]
            while len(captions) < self.num_captions:
                captions.append(captions[0] if captions else "Activity detected.")

            all_captions.append(captions)

        return all_captions

    def _parse_output(self, text: str) -> List[str]:
        """Parse JSON list from model output."""
        # Try to find JSON list in text
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        raise ValueError("No JSON list found")


class LlamaHFBackend:
    """Llama model backend via Hugging Face Transformers."""

    def __init__(self,
                 model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 num_captions: int = 4,
                 max_new_tokens: int = 128,
                 temperature: float = 0.9,
                 top_p: float = 0.95,
                 device: Optional[str] = None):
        """Initialize Llama backend.

        Args:
            model_name: Hugging Face model identifier
            num_captions: Number of captions to generate per sample
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            device: Device to use (None = auto)
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError("Install transformers and torch: pip install transformers torch")

        self.num_captions = num_captions
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"Loading Llama model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if device is None else device
        )
        self.model.eval()
        print(f"Model loaded on device: {self.model.device}")

        # Check if this is a chat model
        self.is_chat_model = "instruct" in model_name.lower() or "chat" in model_name.lower()

    def generate(self, prompts: List[str]) -> List[List[str]]:
        """Generate captions for prompts."""
        import torch

        all_captions = []

        for prompt in prompts:
            captions = []

            # Use chat template if available
            if self.is_chat_model and hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                formatted_prompt = prompt

            for _ in range(self.num_captions):
                # Tokenize
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                # Decode
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )

                # Try to parse as JSON
                try:
                    parsed = self._parse_output(generated_text)
                    captions.extend(parsed)
                    break
                except Exception:
                    captions.append(generated_text.strip())

            # Ensure correct number
            captions = captions[:self.num_captions]
            while len(captions) < self.num_captions:
                captions.append(captions[0] if captions else "Activity detected.")

            all_captions.append(captions)

        return all_captions

    def _parse_output(self, text: str) -> List[str]:
        """Parse JSON list from model output."""
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        raise ValueError("No JSON list found")


# ============================================================================
# Remote API Backends
# ============================================================================


class OpenAIBackend:
    """OpenAI API backend for GPT models."""

    def __init__(self,
                 model_name: str = "gpt-4o-mini",
                 num_captions: int = 4,
                 temperature: float = 0.9,
                 api_key: Optional[str] = None):
        """Initialize OpenAI backend.

        Args:
            model_name: Model identifier (gpt-4o-mini, gpt-4o, etc.)
            num_captions: Number of captions to generate per sample
            temperature: Sampling temperature
            api_key: OpenAI API key (if None, reads from env OPENAI_API_KEY)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        self.num_captions = num_captions
        self.temperature = temperature
        self.model_name = model_name

        # Initialize client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Will read from OPENAI_API_KEY env var
            self.client = OpenAI()

        print(f"Initialized OpenAI backend with model: {model_name}")

    def generate(self, prompts: List[str]) -> List[List[str]]:
        """Generate captions for prompts."""
        all_captions = []

        system_prompt = self._get_system_prompt()

        for prompt in prompts:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=512
                )

                # Parse response
                content = response.choices[0].message.content
                captions = self._parse_output(content)

                # Ensure correct number
                captions = captions[:self.num_captions]
                while len(captions) < self.num_captions:
                    captions.append(captions[0] if captions else "Activity detected.")

                all_captions.append(captions)

            except Exception as e:
                print(f"Warning: OpenAI API error: {e}")
                # Fallback
                all_captions.append(["Activity detected."] * self.num_captions)

        return all_captions

    def _get_system_prompt(self) -> str:
        """Get the system prompt."""
        return """You are an expert caption writer for describing how someone lives in a house based on nearable sensor activation data. Your goal is to turn structured metadata into several short, diverse, natural-language captions that a human might type when searching for this segment.

The captions will be embedded with a CLIP-style text encoder, so they must be clear, concrete, and diverse enough.

Follow these rules:
- Write 4 alternative captions per example.
- Keep them under 100 words. Create various length of captions for the same data sample.
- Focus on the main activity, place, important objects, and time context.
- Prefer concrete nouns and action verbs over abstract phrasing.
- Describe this as if we are observing a person living in a house and doing the things actively.
- It is fine to repeat important key words across captions, but vary wording, syntax, and level of detail.
- Do not include explanations or commentary.
- Output only a JSON list of strings, with each string being one caption."""

    def _parse_output(self, text: str) -> List[str]:
        """Parse JSON list from output."""
        # Try to find and parse JSON
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]

        # Fallback: split by newlines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return lines if lines else ["Activity detected."]


class GeminiBackend:
    """Google Gemini API backend."""

    def __init__(self,
                 model_name: str = "gemini-2.5-flash",
                 num_captions: int = 4,
                 temperature: float = 0.9,
                 api_key: Optional[str] = None):
        """Initialize Gemini backend.

        Args:
            model_name: Model identifier (gemini-2.5-flash, gemini-1.5-flash, gemini-1.5-pro, etc.)
            num_captions: Number of captions to generate per sample
            temperature: Sampling temperature
            api_key: Google AI API key (if None, reads from env GOOGLE_API_KEY)
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Install google-genai: pip install -U google-genai")

        self.num_captions = num_captions
        self.temperature = temperature
        self.model_name = model_name
        self.types = types

        # Get API key
        if not api_key:
            import os
            api_key = os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")

        # Initialize client with new API
        self.client = genai.Client(api_key=api_key)
        self.system_prompt = self._get_system_prompt()

        print(f"Initialized Gemini backend with model: {model_name}")

    def generate(self, prompts: List[str]) -> List[List[str]]:
        """Generate captions for prompts."""
        all_captions = []

        for prompt in prompts:
            try:
                # Use new API format with proper config
                # Disable thinking for 2.5 models to save tokens and get faster responses
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        temperature=self.temperature,
                        max_output_tokens=512,
                        thinking_config=self.types.ThinkingConfig(thinking_budget=0)  # Disable thinking
                    )
                )

                # Get text from response
                content = response.text
                if not content:
                    raise ValueError("Could not extract text from response")

                captions = self._parse_output(content)

                # Ensure correct number
                captions = captions[:self.num_captions]
                while len(captions) < self.num_captions:
                    captions.append(captions[0] if captions else "Activity detected.")

                all_captions.append(captions)

            except Exception as e:
                print(f"Warning: Gemini API error: {e}")
                import traceback
                traceback.print_exc()
                # Fallback
                all_captions.append(["Activity detected."] * self.num_captions)

        return all_captions

    def _get_system_prompt(self) -> str:
        """Get the system prompt."""
        return """You are an expert caption writer for describing how someone lives in a house based on nearable sensor activation data. Your goal is to turn structured metadata into several short, diverse, natural-language captions that a human might type when searching for this segment.

The captions will be embedded with a CLIP-style text encoder, so they must be clear, concrete, and diverse enough.

Follow these rules:
- Write 4 alternative captions per example.
- Keep them under 100 words. Create various length of captions for the same data sample.
- Focus on the main activity, place, important objects, and time context.
- Prefer concrete nouns and action verbs over abstract phrasing.
- Describe this as if we are observing a person living in a house and doing the things actively.
- It is fine to repeat important key words across captions, but vary wording, syntax, and level of detail.
- Do not include explanations or commentary.
- Output only a JSON list of strings, with each string being one caption."""

    def _parse_output(self, text: str) -> List[str]:
        """Parse JSON list from output."""
        if not text:
            return ["Activity detected."]

        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith('```'):
            # Remove opening ```json or ```
            lines = text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            # Remove closing ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines)

        # Try to find and parse JSON
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    # Filter out non-string items and empty strings
                    return [str(item).strip() for item in parsed if item]
            except json.JSONDecodeError as e:
                print(f"DEBUG: JSON parse failed: {e}")
                pass

        # Fallback: split by newlines and filter
        lines = [line.strip() for line in text.split('\n') if line.strip() and not line.strip().startswith(('```', '[', ']'))]
        return lines if lines else ["Activity detected."]


# ============================================================================
# Backend Factory
# ============================================================================


def create_backend(backend_type: str,
                   model_name: str,
                   num_captions: int = 4,
                   temperature: float = 0.9,
                   api_key: Optional[str] = None,
                   device: Optional[str] = None) -> CaptionModel:
    """Factory function to create LLM backends.

    Args:
        backend_type: One of 'gemma', 'llama', 'openai', 'gemini'
        model_name: Model identifier
        num_captions: Number of captions per sample
        temperature: Sampling temperature
        api_key: API key for remote backends
        device: Device for local backends

    Returns:
        Backend instance
    """
    backend_type = backend_type.lower()

    if backend_type == 'gemma':
        return GemmaHFBackend(
            model_name=model_name,
            num_captions=num_captions,
            temperature=temperature,
            device=device
        )
    elif backend_type == 'llama':
        return LlamaHFBackend(
            model_name=model_name,
            num_captions=num_captions,
            temperature=temperature,
            device=device
        )
    elif backend_type == 'openai':
        return OpenAIBackend(
            model_name=model_name,
            num_captions=num_captions,
            temperature=temperature,
            api_key=api_key
        )
    elif backend_type == 'gemini':
        return GeminiBackend(
            model_name=model_name,
            num_captions=num_captions,
            temperature=temperature,
            api_key=api_key
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. "
                         f"Choose from: gemma, llama, openai, gemini")

