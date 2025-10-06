import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from models.text_encoders.embedding_gemma import EmbeddingGemmaEncoder


class TextEncoder(nn.Module):
  """
  Frozen text encoder using thenlper/gte-base.
  Returns L2-normalized 768-d embeddings.
  """

  def __init__(self, model_name: str = "thenlper/gte-base"):
    super().__init__()
    self.model = AutoModel.from_pretrained(model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Freeze all parameters
    for param in self.model.parameters():
      param.requires_grad = False

    self.model.eval()

    # CLIP projection head (learnable)
    self.clip_proj = nn.Linear(768, 512, bias=False)

    # Initialize with near-identity (copy first 512 dims)
    with torch.no_grad():
      identity_init = torch.eye(768)[:512]  # [512, 768]
      self.clip_proj.weight.copy_(identity_init)

  def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
      input_ids: [batch_size, seq_len]
      attention_mask: [batch_size, seq_len]

    Returns:
      embeddings: [batch_size, 768] L2-normalized
    """
    with torch.no_grad():
      outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
      # Use CLS token (first token) as sentence embedding
      embeddings = outputs.last_hidden_state[:, 0]  # [batch_size, 768]

    # L2 normalize
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings

  def forward_clip(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for CLIP alignment with projection head.

    Args:
      input_ids: [batch_size, seq_len]
      attention_mask: [batch_size, seq_len]

    Returns:
      embeddings: [batch_size, 512] L2-normalized projected embeddings
    """
    # Get base embeddings (768-dim, L2-normalized)
    base_embeddings = self.forward(input_ids, attention_mask)

    # Apply projection head
    projected = self.clip_proj(base_embeddings)

    # L2 normalize projected embeddings
    projected = F.normalize(projected, p=2, dim=-1)

    return projected

  def encode_texts(self, texts: list[str], device: torch.device) -> torch.Tensor:
    """
    Convenience method to encode a list of texts.

    Args:
      texts: List of text strings
      device: Target device

    Returns:
      embeddings: [len(texts), 768] L2-normalized
    """
    # Tokenize
    encoded = self.tokenizer(
      texts,
      padding=True,
      truncation=True,
      max_length=512,
      return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    return self.forward(input_ids, attention_mask)

  def encode_texts_clip(self, texts: list[str], device: torch.device) -> torch.Tensor:
    """
    Convenience method to encode a list of texts with CLIP projection.

    Args:
      texts: List of text strings
      device: Target device

    Returns:
      embeddings: [len(texts), 512] L2-normalized projected embeddings
    """
    # Tokenize
    encoded = self.tokenizer(
      texts,
      padding=True,
      truncation=True,
      max_length=512,
      return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    return self.forward_clip(input_ids, attention_mask)


def get_text_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe config loader for text encoder settings with backward-compatible defaults.
    
    Args:
        cfg: Full training configuration dictionary
        
    Returns:
        Dictionary with text encoder specific settings
    """
    text_cfg = {
        "text_model_name": cfg.get("text_model_name", "thenlper/gte-base"),
        "text_backend": cfg.get("text_backend", "hf"),
        "text_prompt_mode": cfg.get("text_prompt_mode", "document"),
        "text_output_dim": cfg.get("text_output_dim", 512),
        "use_text_proj_head": cfg.get("use_text_proj_head", True),
        "use_cached_embeddings": cfg.get("use_cached_embeddings", False),
        "text_embeddings_cache_path": cfg.get("text_embeddings_cache_path", None),
    }
    
    # Force GTE-compatible settings when using GTE models
    model_name = text_cfg["text_model_name"]
    
    # Check if this is a GTE model (by name, not backend)
    is_gte_model = "gte" in model_name.lower() or "thenlper" in model_name.lower()
    
    if is_gte_model:
        # Force settings to match current GTE+proj behavior
        text_cfg["text_backend"] = "hf"  # Force HF backend for GTE
        text_cfg["use_text_proj_head"] = True
        text_cfg["text_output_dim"] = 512
        # text_prompt_mode is ignored for GTE models
    
    return text_cfg


def build_text_encoder(cfg: Dict[str, Any]) -> Union[TextEncoder, 'EmbeddingGemmaEncoder', 'CachedTextEncoder']:
    """
    Factory function to build appropriate text encoder based on configuration.
    
    Args:
        cfg: Training configuration dictionary
        
    Returns:
        Text encoder instance (TextEncoder, EmbeddingGemmaEncoder, or CachedTextEncoder)
    """
    tcfg = get_text_cfg(cfg)
    name = tcfg["text_model_name"]
    backend = tcfg["text_backend"]
    
    # Check if we should use cached embeddings
    if tcfg["use_cached_embeddings"] and tcfg["text_embeddings_cache_path"]:
        try:
            from models.text_encoders.cached_text_encoder import CachedTextEncoder
            cache_path = tcfg["text_embeddings_cache_path"]
            
            # Create fallback encoder if needed (for cache misses)
            fallback_encoder = None
            if tcfg.get("use_fallback_encoder", True):
                # Build the original encoder as fallback
                fallback_cfg = tcfg.copy()
                fallback_cfg["use_cached_embeddings"] = False  # Avoid recursion
                fallback_encoder = build_text_encoder({**cfg, **fallback_cfg})
            
            return CachedTextEncoder(
                cache_path=cache_path,
                fallback_encoder=fallback_encoder
            )
        except Exception as e:
            print(f"Warning: Could not load CachedTextEncoder: {e}")
            print("Falling back to regular text encoder")
            # Fall through to regular encoder selection
    
    # Check if we should use EmbeddingGemma
    if "embeddinggemma" in name.lower() or backend == "sentence_transformers":
        try:
            from models.text_encoders.embedding_gemma import EmbeddingGemmaEncoder
            return EmbeddingGemmaEncoder(
                model_name=name,
                prompt_mode=tcfg["text_prompt_mode"],
                output_dim=tcfg["text_output_dim"],
                use_proj_head=tcfg["use_text_proj_head"]
            )
        except ImportError as e:
            print(f"Warning: Could not load EmbeddingGemmaEncoder: {e}")
            print("Falling back to default TextEncoder (GTE)")
            # Fall through to default TextEncoder
    
    # Default: existing GTE encoder (unchanged behavior)
    return TextEncoder(model_name=name)


def log_text_encoder_config(cfg: Dict[str, Any], logger=None):
    """
    Log the resolved text encoder configuration at startup.
    
    Args:
        cfg: Training configuration dictionary
        logger: Optional logger instance
    """
    tcfg = get_text_cfg(cfg)
    
    log_func = logger.info if logger else print
    
    log_func("Text Encoder Configuration:")
    log_func(f"  Model: {tcfg['text_model_name']}")
    log_func(f"  Backend: {tcfg['text_backend']}")
    log_func(f"  Output Dimension: {tcfg['text_output_dim']}")
    log_func(f"  Use Projection Head: {tcfg['use_text_proj_head']}")
    
    # Note when settings are ignored
    model_name = tcfg["text_model_name"]
    backend = tcfg["text_backend"]
    is_gte_model = "gte" in model_name.lower() or "thenlper" in model_name.lower()
    
    if is_gte_model:
        log_func(f"  Prompt Mode: {tcfg['text_prompt_mode']} (ignored for GTE)")
    else:
        log_func(f"  Prompt Mode: {tcfg['text_prompt_mode']}")
    
    # Log which encoder will be used
    if tcfg["use_cached_embeddings"]:
        cache_path = tcfg.get("text_embeddings_cache_path", "None")
        log_func(f"  -> Using CachedTextEncoder with precomputed embeddings")
        log_func(f"     Cache path: {cache_path}")
    elif "embeddinggemma" in model_name.lower() or backend == "sentence_transformers":
        log_func("  -> Using EmbeddingGemma encoder with sentence_transformers backend")
    else:
        log_func("  -> Using default GTE encoder with HuggingFace transformers backend")
