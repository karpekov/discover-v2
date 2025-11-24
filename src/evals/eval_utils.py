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
    # Get projection head dimensions and type from checkpoint
    proj_state_dict = {}
    proj_type = None
    base_dim = None
    proj_dim = None
    
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        
        # Try linear projection first: text_projection.proj.weight
        if 'text_projection.proj.weight' in model_state:
            proj_type = 'linear'
            proj_weight = model_state['text_projection.proj.weight']
            proj_dim, base_dim = proj_weight.shape
            proj_state_dict = {'proj.weight': proj_weight}
            print(f"📐 Detected linear projection: {base_dim} → {proj_dim}")
            
        # Try MLP projection: text_projection.mlp.*
        elif any(k.startswith('text_projection.mlp.') for k in model_state.keys()):
            proj_type = 'mlp'
            # Extract all MLP weights
            mlp_keys = [k for k in model_state.keys() if k.startswith('text_projection.mlp.')]
            for key in mlp_keys:
                short_key = key.replace('text_projection.', '')
                proj_state_dict[short_key] = model_state[key]
            
            # Get dimensions from first layer
            first_layer_weight = model_state['text_projection.mlp.0.weight']
            hidden_dim, base_dim = first_layer_weight.shape
            
            # Get output dimension from last layer
            last_layer_keys = [k for k in mlp_keys if '.weight' in k]
            last_layer_key = sorted(last_layer_keys)[-1]
            proj_dim, _ = model_state[last_layer_key].shape
            
            print(f"📐 Detected MLP projection: {base_dim} → {hidden_dim} → {proj_dim}")
            print(f"   Found {len(mlp_keys)} MLP parameters")
            
        # Check if using pre-computed embeddings (no text projection needed)
        elif 'text_encoder_metadata' in checkpoint:
            # Get dimensions from metadata
            metadata = checkpoint['text_encoder_metadata']
            base_dim = metadata.get('embedding_dim', 512)
            proj_dim = metadata.get('projection_dim', 0)
            if proj_dim == 0:
                proj_dim = base_dim  # No projection, just use base dimension
            proj_type = 'none'
            print(f"📐 Using text encoder metadata dimensions: {base_dim} (no projection)")
        else:
            print("⚠️  No text projection head found in checkpoint")
            return None

    # ==== TIER 1: Try checkpoint metadata (saved during training) ====
    if 'text_encoder_metadata' in checkpoint:
        metadata = checkpoint['text_encoder_metadata']
        encoder_type = metadata.get('encoder_type', 'unknown')
        model_name = metadata.get('model_name', 'unknown')
        embedding_dim = metadata.get('embedding_dim', base_dim if base_dim else 512)

        print(f"✅ Found text encoder metadata in checkpoint:")
        print(f"   Encoder type: {encoder_type}")
        print(f"   Model name: {model_name}")
        print(f"   Embedding dim: {embedding_dim}")

        # Validate dimension match (only if we have base_dim from projection)
        if base_dim is not None and embedding_dim != base_dim:
            raise ValueError(
                f"❌ DIMENSION MISMATCH! "
                f"Checkpoint metadata says {embedding_dim}D embeddings, "
                f"but projection head expects {base_dim}D inputs. "
                f"This checkpoint may be corrupted or from a different training run."
            )

        # Use embedding_dim as both base and proj if no projection head
        if base_dim is None:
            base_dim = embedding_dim
        if proj_dim is None:
            proj_dim = embedding_dim

        # Create encoder based on type
        text_encoder = _create_encoder_by_type(
            encoder_type, model_name, base_dim, proj_dim, device, 
            proj_type=proj_type, proj_state_dict=proj_state_dict
        )
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
                            text_encoder = _create_encoder_by_type(
                                encoder_type, model_name, base_dim, proj_dim, device,
                                proj_type=proj_type, proj_state_dict=proj_state_dict
                            )
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

    text_encoder = _create_encoder_by_type(
        'clip', 'openai/clip-vit-base-patch32', base_dim, proj_dim, device,
        proj_type=proj_type, proj_state_dict=proj_state_dict
    )
    return text_encoder


def _create_encoder_by_type(
    encoder_type: str,
    model_name: str,
    base_dim: int,
    proj_dim: int,
    device: torch.device,
    proj_type: Optional[str] = None,
    proj_state_dict: Optional[Dict[str, torch.Tensor]] = None
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
        return _create_clip_encoder(model_name, base_dim, proj_dim, device, proj_type, proj_state_dict)
    elif encoder_type in ['llama', 'embedding_llama']:
        return _create_llama_encoder(model_name, base_dim, proj_dim, device, proj_type, proj_state_dict)
    elif encoder_type in ['gte', 'sentence-transformers']:
        return _create_sentence_transformer_encoder(model_name, base_dim, proj_dim, device, proj_type, proj_state_dict)
    else:
        raise ValueError(
            f"❌ Unsupported encoder type: {encoder_type}\n"
            f"   Supported types: clip, llama, gte, sentence-transformers\n"
            f"   Please implement support for this encoder or retrain with a supported one."
        )


def _create_clip_encoder(
    model_name: str, 
    base_dim: int, 
    proj_dim: int, 
    device: torch.device,
    proj_type: Optional[str] = None,
    proj_state_dict: Optional[Dict[str, torch.Tensor]] = None
) -> nn.Module:
    """Create a CLIP-based text encoder."""
    from transformers import CLIPTextModel, CLIPTokenizer

    class CLIPTextEncoder(nn.Module):
        def __init__(self, model_name, base_dim, proj_dim, proj_type='linear'):
            super().__init__()
            self.base_dim = base_dim
            self.proj_dim = proj_dim
            self.proj_type = proj_type
            self.model_name = model_name
            self._clip_model = None
            self._clip_tokenizer = None
            
            # Create projection head based on type
            if proj_type == 'mlp':
                # MLP projection: Linear -> GELU -> Dropout -> Linear
                self.clip_proj = nn.Sequential(
                    nn.Linear(base_dim, base_dim),  # mlp.0
                    nn.GELU(),  # mlp.1
                    nn.Dropout(0.1),  # mlp.2
                    nn.Linear(base_dim, proj_dim, bias=False)  # mlp.3
                )
            else:
                # Linear projection
                self.clip_proj = nn.Linear(base_dim, proj_dim, bias=False)

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
    print(f"   Projection type: {proj_type or 'linear'}")
    encoder = CLIPTextEncoder(model_name, base_dim, proj_dim, proj_type or 'linear')
    
    # Load projection weights if provided
    if proj_state_dict:
        try:
            encoder.clip_proj.load_state_dict(proj_state_dict, strict=False)
            print(f"   ✅ Loaded projection weights from checkpoint")
        except Exception as e:
            print(f"   ⚠️  Could not load projection weights: {e}")
    
    return encoder


def _create_llama_encoder(
    model_name: str, 
    base_dim: int, 
    proj_dim: int, 
    device: torch.device,
    proj_type: Optional[str] = None,
    proj_state_dict: Optional[Dict[str, torch.Tensor]] = None
) -> nn.Module:
    """Create a LLaMA-based text encoder."""
    from transformers import AutoModel, AutoTokenizer

    class LLaMATextEncoder(nn.Module):
        def __init__(self, model_name, base_dim, proj_dim, proj_type='linear'):
            super().__init__()
            self.base_dim = base_dim
            self.proj_dim = proj_dim
            self.proj_type = proj_type
            self.model_name = model_name
            self._model = None
            self._tokenizer = None
            
            # Create projection head based on type
            if proj_type == 'mlp':
                # MLP projection: Linear -> GELU -> Dropout -> Linear
                self.clip_proj = nn.Sequential(
                    nn.Linear(base_dim, base_dim),  # mlp.0
                    nn.GELU(),  # mlp.1
                    nn.Dropout(0.1),  # mlp.2
                    nn.Linear(base_dim, proj_dim, bias=False)  # mlp.3
                )
            else:
                # Linear projection
                self.clip_proj = nn.Linear(base_dim, proj_dim, bias=False)

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
    print(f"   Projection type: {proj_type or 'linear'}")
    encoder = LLaMATextEncoder(model_name, base_dim, proj_dim, proj_type or 'linear')
    
    # Load projection weights if provided
    if proj_state_dict:
        try:
            encoder.clip_proj.load_state_dict(proj_state_dict, strict=False)
            print(f"   ✅ Loaded projection weights from checkpoint")
        except Exception as e:
            print(f"   ⚠️  Could not load projection weights: {e}")
    
    return encoder


def _create_sentence_transformer_encoder(
    model_name: str, 
    base_dim: int, 
    proj_dim: int, 
    device: torch.device,
    proj_type: Optional[str] = None,
    proj_state_dict: Optional[Dict[str, torch.Tensor]] = None
) -> nn.Module:
    """Create a sentence-transformers-based encoder (GTE, MiniLM, etc.)."""
    from transformers import AutoModel, AutoTokenizer

    class SentenceTransformerEncoder(nn.Module):
        def __init__(self, model_name, base_dim, proj_dim, proj_type='linear'):
            super().__init__()
            self.base_dim = base_dim
            self.proj_dim = proj_dim
            self.proj_type = proj_type
            self.model_name = model_name
            self._model = None
            self._tokenizer = None
            
            # Create projection head based on type
            if proj_type == 'mlp':
                # MLP projection: Linear -> GELU -> Dropout -> Linear
                self.clip_proj = nn.Sequential(
                    nn.Linear(base_dim, base_dim),  # mlp.0
                    nn.GELU(),  # mlp.1
                    nn.Dropout(0.1),  # mlp.2
                    nn.Linear(base_dim, proj_dim, bias=False)  # mlp.3
                )
            else:
                # Linear projection
                self.clip_proj = nn.Linear(base_dim, proj_dim, bias=False)

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
    print(f"   Projection type: {proj_type or 'linear'}")
    encoder = SentenceTransformerEncoder(model_name, base_dim, proj_dim, proj_type or 'linear')
    
    # Load projection weights if provided
    if proj_state_dict:
        try:
            encoder.clip_proj.load_state_dict(proj_state_dict, strict=False)
            print(f"   ✅ Loaded projection weights from checkpoint")
        except Exception as e:
            print(f"   ⚠️  Could not load projection weights: {e}")
    
    return encoder

