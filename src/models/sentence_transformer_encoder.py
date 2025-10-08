"""
Sentence Transformer text encoder wrapper.
Uses the sentence_transformers library directly for proper pooling.
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List


class SentenceTransformerEncoder(nn.Module):
    """
    Text encoder using sentence-transformers library.
    Properly handles mean pooling for sentence-transformers models.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-distilroberta-v1"):
        super().__init__()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"Loaded SentenceTransformer: {model_name}")
        print(f"  Embedding dimension: {self.embedding_dim}")

    def encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Encode a list of texts using sentence-transformers.

        Args:
            texts: List of text strings
            device: Target device (note: sentence-transformers handles device internally)

        Returns:
            embeddings: [len(texts), embedding_dim] L2-normalized tensor
        """
        # sentence-transformers returns numpy arrays by default
        # We can request pytorch tensors and specify device
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True,  # L2 normalize
            show_progress_bar=False
        )

        return embeddings

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Forward pass for compatibility."""
        return self.encode_texts(texts, device)

