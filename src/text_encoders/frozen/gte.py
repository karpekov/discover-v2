"""GTE (General Text Embeddings) text encoder.

Uses thenlper/gte-base model from HuggingFace.
This is the default text encoder used in the original implementation.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List
import numpy as np

from ..base import BaseTextEncoder, TextEncoderConfig, TextEncoderOutput


class GTETextEncoder(BaseTextEncoder):
    """Frozen GTE text encoder.

    Uses thenlper/gte-base (768-d embeddings) by default.
    Embeddings are extracted from the CLS token and L2-normalized.
    """

    def __init__(self, config: TextEncoderConfig):
        """Initialize GTE encoder.

        Args:
            config: Text encoder configuration
        """
        super().__init__(config)

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
            trust_remote_code=True
        )

        # Device is auto-detected in parent class
        self.model.to(self.device)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        # Optional projection head (for CLIP compatibility)
        self.projection = None
        if config.use_projection:
            self.projection = torch.nn.Linear(
                config.embedding_dim,
                config.projection_dim,
                bias=False
            )
            # Initialize with near-identity (copy first projection_dim dimensions)
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
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Use CLS token (first token) as sentence embedding
            embeddings = outputs.last_hidden_state[:, 0]  # [batch_size, embedding_dim]

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

