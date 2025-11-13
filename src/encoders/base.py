"""
Base encoder classes and interfaces.

Defines the common interface that all sensor encoders must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Any
import torch
import torch.nn as nn


@dataclass
class EncoderOutput:
    """
    Standard output from encoder forward pass.

    Attributes:
        embeddings: [batch_size, d_model] pooled sequence embeddings (L2-normalized)
        sequence_features: [batch_size, seq_len, d_model] per-token hidden states (for MLM)
        projected_embeddings: [batch_size, proj_dim] projected embeddings for CLIP alignment
        attention_mask: [batch_size, seq_len] attention mask used (True = valid token)
    """
    embeddings: torch.Tensor  # Pooled, normalized
    sequence_features: Optional[torch.Tensor] = None  # For MLM
    projected_embeddings: Optional[torch.Tensor] = None  # For CLIP
    attention_mask: Optional[torch.Tensor] = None


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for all sensor encoders.

    All encoders must implement:
    - forward(): Main encoding with pooling
    - get_sequence_features(): Get per-token features for MLM
    - forward_clip(): Get projected embeddings for CLIP alignment
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        input_data: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> EncoderOutput:
        """
        Main forward pass with pooling.

        Args:
            input_data: Dict containing encoder-specific inputs
            attention_mask: [batch_size, seq_len] boolean mask (True = valid)
            **kwargs: Additional encoder-specific arguments

        Returns:
            EncoderOutput with pooled embeddings
        """
        pass

    @abstractmethod
    def get_sequence_features(
        self,
        input_data: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Get per-token hidden states for MLM.

        Args:
            input_data: Dict containing encoder-specific inputs
            attention_mask: [batch_size, seq_len] boolean mask (True = valid)
            **kwargs: Additional encoder-specific arguments

        Returns:
            hidden_states: [batch_size, seq_len, d_model] per-token features
        """
        pass

    @abstractmethod
    def forward_clip(
        self,
        input_data: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with projection head for CLIP alignment.

        Args:
            input_data: Dict containing encoder-specific inputs
            attention_mask: [batch_size, seq_len] boolean mask (True = valid)
            **kwargs: Additional encoder-specific arguments

        Returns:
            projected_embeddings: [batch_size, proj_dim] L2-normalized
        """
        pass

    @property
    @abstractmethod
    def d_model(self) -> int:
        """Return the embedding dimension."""
        pass

    @property
    def output_dim(self) -> int:
        """Return the output embedding dimension (after projection if applicable)."""
        return self.config.projection_dim if hasattr(self.config, 'projection_dim') else self.d_model


class SequenceEncoder(BaseEncoder):
    """
    Base class for sequence-based encoders (transformers, etc.).

    These encoders process raw sensor sequences with categorical and continuous features.
    """

    @abstractmethod
    def _create_embeddings(
        self,
        categorical_features: Dict[str, torch.Tensor],
        continuous_features: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create initial token embeddings from input features.

        Args:
            categorical_features: Dict of [batch_size, seq_len] tensors
            continuous_features: Dict of [batch_size, seq_len, feat_dim] tensors
            attention_mask: [batch_size, seq_len] boolean mask

        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        pass

    @abstractmethod
    def _pool_embeddings(
        self,
        sequence_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence embeddings into fixed-size representation.

        Args:
            sequence_embeddings: [batch_size, seq_len, d_model]
            attention_mask: [batch_size, seq_len] boolean mask

        Returns:
            pooled: [batch_size, d_model] L2-normalized
        """
        pass


class ImageSequenceEncoder(BaseEncoder):
    """
    Base class for image-based encoders (CLIP, DINO, etc.).

    These encoders process spatial visualizations of sensor activations.
    Placeholder for future implementation.
    """

    def __init__(self, config):
        super().__init__(config)
        raise NotImplementedError("Image-based encoders not yet implemented")

    def forward(self, input_data, attention_mask=None, **kwargs):
        raise NotImplementedError("Image-based encoders not yet implemented")

    def get_sequence_features(self, input_data, attention_mask=None, **kwargs):
        raise NotImplementedError("Image-based encoders not yet implemented")

    def forward_clip(self, input_data, attention_mask=None, **kwargs):
        raise NotImplementedError("Image-based encoders not yet implemented")

    @property
    def d_model(self):
        return self.config.d_model

