import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class EmbeddingGemmaEncoder(nn.Module):
    """
    Text encoder using EmbeddingGemma with sentence_transformers backend.
    Supports prompt modes and Matryoshka Representation Learning (MRL) for dimension truncation.
    """
    
    # Prompt templates from the EmbeddingGemma documentation
    PROMPT_TEMPLATES = {
        "query": "task: search result | query: ",
        "document": "title: none | text: ",
        "clustering": "task: clustering | query: ",
        "classification": "task: classification | query: ",
        "retrieval": "task: search result | query: ",
        "similarity": "task: sentence similarity | query: ",
    }

    def __init__(
        self, 
        model_name: str = "google/embeddinggemma-300m",
        prompt_mode: str = "document",
        output_dim: int = 512,
        use_proj_head: bool = False
    ):
        super().__init__()
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence_transformers is required for EmbeddingGemma. "
                "Install with: pip install sentence-transformers>=5.0.0"
            )
        
        self.model_name = model_name
        self.prompt_mode = prompt_mode
        self.output_dim = output_dim
        self.use_proj_head = use_proj_head
        
        # Load the sentence transformer model
        self.model = SentenceTransformer(model_name)
        
        # Get the model's native output dimension
        self.native_dim = self.model.get_sentence_embedding_dimension()
        
        # Validate output dimension for MRL
        if not use_proj_head:
            if output_dim > self.native_dim:
                raise ValueError(f"output_dim ({output_dim}) cannot exceed native dimension ({self.native_dim}) without projection head")
            # EmbeddingGemma supports MRL truncation to 512, 256, 128 dims
            valid_mrl_dims = [128, 256, 512, self.native_dim]
            if output_dim not in valid_mrl_dims:
                print(f"Warning: output_dim {output_dim} may not be optimal for MRL. Recommended: {valid_mrl_dims}")
        
        # Optional projection head (similar to GTE's clip_proj)
        self.proj_head = None
        if use_proj_head:
            self.proj_head = nn.Linear(self.native_dim, output_dim, bias=False)
            # Initialize with near-identity if possible
            with torch.no_grad():
                if output_dim <= self.native_dim:
                    identity_init = torch.eye(self.native_dim)[:output_dim]  # [output_dim, native_dim]
                    self.proj_head.weight.copy_(identity_init)
                else:
                    # Xavier initialization for expanding projection
                    nn.init.xavier_uniform_(self.proj_head.weight)
        
        # Freeze the sentence transformer model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        # Validate prompt mode
        if prompt_mode not in self.PROMPT_TEMPLATES:
            print(f"Warning: Unknown prompt_mode '{prompt_mode}'. Available: {list(self.PROMPT_TEMPLATES.keys())}")
            print(f"Using 'document' mode as fallback.")
            self.prompt_mode = "document"

    def _apply_prompt(self, texts: List[str]) -> List[str]:
        """Apply prompt template to texts."""
        if self.prompt_mode in self.PROMPT_TEMPLATES:
            prompt = self.PROMPT_TEMPLATES[self.prompt_mode]
            return [prompt + text for text in texts]
        return texts

    def _truncate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply MRL truncation if no projection head is used."""
        if not self.use_proj_head and self.output_dim < embeddings.size(-1):
            return embeddings[..., :self.output_dim]
        return embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass compatible with existing TextEncoder API.
        Note: EmbeddingGemma uses sentence_transformers, so input_ids/attention_mask are not used directly.
        This method is kept for API compatibility but should not be the primary interface.
        """
        raise NotImplementedError(
            "EmbeddingGemmaEncoder uses sentence_transformers backend. "
            "Use encode_texts() or encode_texts_clip() instead of forward()."
        )

    def forward_clip(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        CLIP-style forward pass compatible with existing API.
        Note: This is kept for compatibility but should not be the primary interface.
        """
        raise NotImplementedError(
            "EmbeddingGemmaEncoder uses sentence_transformers backend. "
            "Use encode_texts_clip() instead of forward_clip()."
        )

    def encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Encode texts to embeddings (base method, returns native or truncated dimensions).
        
        Args:
            texts: List of text strings
            device: Target device
            
        Returns:
            embeddings: [len(texts), output_dim] L2-normalized
        """
        if not texts:
            return torch.empty(0, self.output_dim, device=device)
        
        # Apply prompt template
        prompted_texts = self._apply_prompt(texts)
        
        # Encode using sentence transformers (automatically handles device placement)
        with torch.no_grad():
            embeddings = self.model.encode(
                prompted_texts,
                convert_to_tensor=True,
                normalize_embeddings=True,  # L2 normalize
                device=device.type if device.type != 'mps' else 'cpu'  # sentence_transformers may not support MPS
            )
        
        # Move to target device if needed
        if embeddings.device != device:
            embeddings = embeddings.to(device)
        
        # Apply MRL truncation if no projection head
        embeddings = self._truncate_embeddings(embeddings)
        
        # Apply projection head if present
        if self.proj_head is not None:
            # Clone the embeddings to make them normal tensors for autograd
            embeddings = embeddings.clone()
            embeddings = self.proj_head(embeddings)
            # Re-normalize after projection
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings

    def encode_texts_clip(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Encode texts with CLIP-style processing (same as encode_texts for EmbeddingGemma).
        Kept for API compatibility with existing TextEncoder.
        
        Args:
            texts: List of text strings
            device: Target device
            
        Returns:
            embeddings: [len(texts), output_dim] L2-normalized
        """
        return self.encode_texts(texts, device)

    def get_output_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.output_dim

    def get_native_dim(self) -> int:
        """Get the native model embedding dimension."""
        return self.native_dim

    def get_prompt_template(self) -> str:
        """Get the current prompt template."""
        return self.PROMPT_TEMPLATES.get(self.prompt_mode, "")

    def set_prompt_mode(self, mode: str):
        """Change the prompt mode."""
        if mode in self.PROMPT_TEMPLATES:
            self.prompt_mode = mode
        else:
            print(f"Warning: Unknown prompt_mode '{mode}'. Available: {list(self.PROMPT_TEMPLATES.keys())}")

    def __repr__(self):
        return (
            f"EmbeddingGemmaEncoder(\n"
            f"  model_name={self.model_name},\n"
            f"  prompt_mode={self.prompt_mode},\n"
            f"  native_dim={self.native_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  use_proj_head={self.use_proj_head}\n"
            f")"
        )
