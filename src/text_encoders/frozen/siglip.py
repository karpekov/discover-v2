"""SigLIP text encoder.

Uses Google SigLIP text encoder for embedding captions.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List
import numpy as np

from ..base import BaseTextEncoder, TextEncoderConfig, TextEncoderOutput


class SigLIPTextEncoder(BaseTextEncoder):
    """Frozen SigLIP text encoder.
    
    Uses Google SigLIP text encoder (e.g., google/siglip-base-patch16-224).
    Embeddings are extracted from the pooled output and L2-normalized.
    """
    
    def __init__(self, config: TextEncoderConfig):
        """Initialize SigLIP text encoder.
        
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
            # Get text embeddings from the model
            outputs = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            embeddings = outputs  # [batch_size, embedding_dim]
            
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

