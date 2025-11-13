"""LLAMA embedding text encoder.

Uses LLAMA-based embedding models (e.g., sentence-transformers/all-MiniLM-L6-v2).
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List
import numpy as np

from ..base import BaseTextEncoder, TextEncoderConfig, TextEncoderOutput


class LLAMATextEncoder(BaseTextEncoder):
    """Frozen LLAMA-based text encoder.
    
    Can use various LLAMA-based embedding models.
    Embeddings are extracted via mean pooling and L2-normalized.
    """
    
    def __init__(self, config: TextEncoderConfig):
        """Initialize LLAMA encoder.
        
        Args:
            config: Text encoder configuration
        """
        super().__init__(config)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir
        )
        self.model = AutoModel.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir
        )
        
        # Device is auto-detected in parent class
        self.model.to(self.device)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        # Optional projection head
        self.projection = None
        if config.use_projection:
            self.projection = torch.nn.Linear(
                config.embedding_dim,
                config.projection_dim,
                bias=False
            )
            with torch.no_grad():
                identity_init = torch.eye(config.embedding_dim)[:config.projection_dim]
                self.projection.weight.copy_(identity_init)
            
            self.projection.to(self.device)
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling with attention mask.
        
        Args:
            token_embeddings: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            Pooled embeddings [batch_size, hidden_dim]
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def encode(self, texts: List[str]) -> TextEncoderOutput:
        """Encode texts into embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            TextEncoderOutput with embeddings [len(texts), embedding_dim]
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Mean pooling
            embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            
            # L2 normalize if requested
            if self.config.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            # Apply projection if present
            if self.projection is not None:
                embeddings = self.projection(embeddings)
                if self.config.normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy()
        
        return TextEncoderOutput(
            embeddings=embeddings_np,
            metadata={
                'num_texts': len(texts),
                'embedding_dim': embeddings_np.shape[1],
                'model_name': self.config.model_name,
            }
        )

