"""Utility functions for evaluation scripts."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


def create_text_encoder_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: torch.device,
    data_path: Optional[str] = None
):
    """
    Create a text encoder compatible with the checkpoint using a 3-tier fallback strategy:

    1. **Primary**: Extract from checkpoint['text_encoder_metadata'] (saved during training)
    2. **Fallback 1**: Load from .npz file metadata (if data_path provided)
    3. **Fallback 2**: Default to CLIP (with warning)

    Args:
        checkpoint: Model checkpoint dictionary
        device: Device to load models on
        data_path: Optional path to training data (to find .npz embeddings file)

    Returns:
        Text encoder module with encode_texts_clip method, or None if fallback needed

    Raises:
        ValueError: If text encoder metadata indicates mismatch or unsupported encoder
    """
    # Get projection head dimensions from checkpoint
    proj_weight = None
    if 'model_state_dict' in checkpoint and 'text_projection.proj.weight' in checkpoint['model_state_dict']:
        proj_weight = checkpoint['model_state_dict']['text_projection.proj.weight']
        proj_dim, base_dim = proj_weight.shape
        print(f"📐 Detected projection dimensions from checkpoint: {base_dim} → {proj_dim}")
    else:
        print("⚠️  No text projection head found in checkpoint")
        return None

    # ==== TIER 1: Try checkpoint metadata (saved during training) ====
    if 'text_encoder_metadata' in checkpoint:
        metadata = checkpoint['text_encoder_metadata']
        encoder_type = metadata.get('encoder_type', 'unknown')
        model_name = metadata.get('model_name', 'unknown')
        embedding_dim = metadata.get('embedding_dim', base_dim)

        print(f"✅ Found text encoder metadata in checkpoint:")
        print(f"   Encoder type: {encoder_type}")
        print(f"   Model name: {model_name}")
        print(f"   Embedding dim: {embedding_dim}")

        # Validate dimension match
        if embedding_dim != base_dim:
            raise ValueError(
                f"❌ DIMENSION MISMATCH! "
                f"Checkpoint metadata says {embedding_dim}D embeddings, "
                f"but projection head expects {base_dim}D inputs. "
                f"This checkpoint may be corrupted or from a different training run."
            )

        # Create encoder based on type
        text_encoder = _create_encoder_by_type(encoder_type, model_name, base_dim, proj_dim, device)
        if text_encoder and proj_weight is not None:
            text_encoder.clip_proj.weight.data = proj_weight.clone()
        return text_encoder

    # ==== TIER 2: Try .npz file metadata (fallback) ====
    if data_path:
        print("⚠️  No text encoder metadata in checkpoint, trying .npz file...")
        try:
            # Try to find .npz file in same directory as data
            data_dir = Path(data_path).parent
            npz_files = sorted(data_dir.glob("*_embeddings_*.npz"))  # Sort for determinism

            if npz_files:
                print(f"   Found {len(npz_files)} .npz files, checking for dimension match...")

                # Try each .npz file until we find one with matching dimensions
                matched_file = None
                for npz_path in npz_files:
                    try:
                        data = np.load(str(npz_path))
                        embedding_dim = int(data['embedding_dim'].item()) if 'embedding_dim' in data else 0

                        if embedding_dim == base_dim:
                            # Found a match!
                            matched_file = npz_path
                            encoder_type = str(data['encoder_type'].item()) if 'encoder_type' in data else 'unknown'
                            model_name = str(data['model_name'].item()) if 'model_name' in data else 'unknown'

                            print(f"   ✅ Matched: {npz_path.name} ({embedding_dim}D)")
                            print(f"   Encoder type: {encoder_type}")
                            print(f"   Model name: {model_name}")

                            # Create encoder based on type
                            text_encoder = _create_encoder_by_type(encoder_type, model_name, base_dim, proj_dim, device)
                            if text_encoder and proj_weight is not None:
                                text_encoder.clip_proj.weight.data = proj_weight.clone()
                            return text_encoder
                        else:
                            print(f"   ⏭️  Skipping {npz_path.name} (dimension mismatch: {embedding_dim}D ≠ {base_dim}D)")
                    except Exception as e:
                        print(f"   ⏭️  Skipping {npz_path.name} (error: {e})")
                        continue

                # If we get here, no matching file was found
                if not matched_file:
                    print(f"   ❌ No .npz file with matching dimension ({base_dim}D) found")

        except Exception as e:
            print(f"⚠️  Could not load from .npz file: {e}")

    # ==== TIER 3: Final fallback to CLIP (with warning) ====
    print("⚠️  WARNING: No text encoder metadata found!")
    print("   Defaulting to CLIP (openai/clip-vit-base-patch32)")
    print("   This may give incorrect results if training used a different encoder!")

    text_encoder = _create_encoder_by_type('clip', 'openai/clip-vit-base-patch32', base_dim, proj_dim, device)
    if text_encoder and proj_weight is not None:
        text_encoder.clip_proj.weight.data = proj_weight.clone()
    return text_encoder


def _create_encoder_by_type(
    encoder_type: str,
    model_name: str,
    base_dim: int,
    proj_dim: int,
    device: torch.device
) -> Optional[nn.Module]:
    """
    Create a text encoder based on type and model name.

    Supported types: clip, llama, gte, minilm
    """
    if encoder_type == 'unknown':
        print("⚠️  Unknown encoder type, defaulting to CLIP")
        encoder_type = 'clip'
        model_name = 'openai/clip-vit-base-patch32'

    if encoder_type == 'clip':
        return _create_clip_encoder(model_name, base_dim, proj_dim, device)
    elif encoder_type in ['llama', 'embedding_llama']:
        return _create_llama_encoder(model_name, base_dim, proj_dim, device)
    elif encoder_type in ['gte', 'sentence-transformers']:
        return _create_sentence_transformer_encoder(model_name, base_dim, proj_dim, device)
    else:
        raise ValueError(
            f"❌ Unsupported encoder type: {encoder_type}\n"
            f"   Supported types: clip, llama, gte, sentence-transformers\n"
            f"   Please implement support for this encoder or retrain with a supported one."
        )


def _create_clip_encoder(model_name: str, base_dim: int, proj_dim: int, device: torch.device) -> nn.Module:
    """Create a CLIP-based text encoder."""
    from transformers import CLIPTextModel, CLIPTokenizer

    class CLIPTextEncoder(nn.Module):
        def __init__(self, model_name, base_dim, proj_dim):
            super().__init__()
            self.base_dim = base_dim
            self.proj_dim = proj_dim
            self.clip_proj = nn.Linear(base_dim, proj_dim, bias=False)
            self.model_name = model_name
            self._clip_model = None
            self._clip_tokenizer = None

        def encode_texts_clip(self, texts, device):
            if self._clip_model is None:
                self._clip_model = CLIPTextModel.from_pretrained(self.model_name).to(device)
                self._clip_tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
                self._clip_model.eval()
                for param in self._clip_model.parameters():
                    param.requires_grad = False

            inputs = self._clip_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                embeddings = outputs.pooler_output
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                embeddings = self.clip_proj(embeddings)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings

    print(f"   Creating CLIP encoder: {model_name}")
    return CLIPTextEncoder(model_name, base_dim, proj_dim)


def _create_llama_encoder(model_name: str, base_dim: int, proj_dim: int, device: torch.device) -> nn.Module:
    """Create a LLaMA-based text encoder."""
    from transformers import AutoModel, AutoTokenizer

    class LLaMATextEncoder(nn.Module):
        def __init__(self, model_name, base_dim, proj_dim):
            super().__init__()
            self.base_dim = base_dim
            self.proj_dim = proj_dim
            self.clip_proj = nn.Linear(base_dim, proj_dim, bias=False)
            self.model_name = model_name
            self._model = None
            self._tokenizer = None

        def encode_texts_clip(self, texts, device):
            if self._model is None:
                self._model = AutoModel.from_pretrained(self.model_name).to(device)
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model.eval()
                for param in self._model.parameters():
                    param.requires_grad = False

            inputs = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use mean pooling over last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                embeddings = self.clip_proj(embeddings)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings

    print(f"   Creating LLaMA encoder: {model_name}")
    return LLaMATextEncoder(model_name, base_dim, proj_dim)


def _create_sentence_transformer_encoder(model_name: str, base_dim: int, proj_dim: int, device: torch.device) -> nn.Module:
    """Create a sentence-transformers-based encoder (GTE, MiniLM, etc.)."""
    from transformers import AutoModel, AutoTokenizer

    class SentenceTransformerEncoder(nn.Module):
        def __init__(self, model_name, base_dim, proj_dim):
            super().__init__()
            self.base_dim = base_dim
            self.proj_dim = proj_dim
            self.clip_proj = nn.Linear(base_dim, proj_dim, bias=False)
            self.model_name = model_name
            self._model = None
            self._tokenizer = None

        def encode_texts_clip(self, texts, device):
            if self._model is None:
                self._model = AutoModel.from_pretrained(self.model_name).to(device)
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model.eval()
                for param in self._model.parameters():
                    param.requires_grad = False

            inputs = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use CLS token (first token) for sentence embedding
                embeddings = outputs.last_hidden_state[:, 0]
                embeddings = F.normalize(embeddings, p=2, dim=-1)
                embeddings = self.clip_proj(embeddings)
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings

    print(f"   Creating Sentence-Transformer encoder: {model_name}")
    return SentenceTransformerEncoder(model_name, base_dim, proj_dim)

