"""
SCAN Clustering Model for src pipeline.
Uses pre-trained AlignmentModel sensor encoder and adds clustering head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
import os
from pathlib import Path
import yaml

# For loading AlignmentModel
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from alignment.model import AlignmentModel
from alignment.config import AlignmentConfig


class SCANClusteringModel(nn.Module):
    """
    SCAN clustering model that uses a pre-trained AlignmentModel's sensor encoder
    and adds a clustering classification head.

    This model can load from:
    1. A model directory containing best_model.pt and config.yaml
    2. A direct checkpoint path (.pt file)
    """

    def __init__(
        self,
        pretrained_model_path: str,
        num_clusters: int,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        vocab_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize SCAN clustering model.

        Args:
            pretrained_model_path: Path to model directory or checkpoint file
            num_clusters: Number of clusters
            dropout: Dropout rate for clustering head
            freeze_encoder: Whether to freeze encoder weights
            vocab_path: Path to vocabulary file (optional, auto-detected from config)
            device: Device to load model onto
        """
        super().__init__()

        self.num_clusters = num_clusters
        self.pretrained_model_path = pretrained_model_path
        self.freeze_encoder = freeze_encoder
        self.device = device or torch.device('cpu')

        # Load the pre-trained AlignmentModel
        self.alignment_model, self.alignment_config, self.vocab_sizes = self._load_pretrained_model(
            pretrained_model_path, vocab_path, device
        )

        # Extract sensor encoder
        self.sensor_encoder = self.alignment_model.sensor_encoder

        # Get the embedding dimension from the encoder
        self.d_model = self.sensor_encoder.d_model

        # Clustering head - projects embeddings to cluster logits
        self.clustering_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, num_clusters)
        )

        # Optionally freeze the encoder
        if freeze_encoder:
            for param in self.sensor_encoder.parameters():
                param.requires_grad = False

    def _load_pretrained_model(
        self,
        model_path: str,
        vocab_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[AlignmentModel, AlignmentConfig, Dict[str, int]]:
        """
        Load pre-trained AlignmentModel from checkpoint.

        Args:
            model_path: Path to model directory or checkpoint file
            vocab_path: Optional path to vocabulary file
            device: Device to load model onto

        Returns:
            Tuple of (model, config, vocab_sizes)
        """
        model_path = Path(model_path)

        # Determine if path is directory or file
        if model_path.is_dir():
            checkpoint_path = model_path / 'best_model.pt'
            config_path = model_path / 'config.yaml'

            if not checkpoint_path.exists():
                # Try final_model.pt
                checkpoint_path = model_path / 'final_model.pt'
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"No checkpoint found in {model_path}")

            # Load config to get vocab path
            if config_path.exists():
                config = AlignmentConfig.from_yaml(str(config_path))
                if vocab_path is None:
                    vocab_path = config.vocab_path
        else:
            # Direct checkpoint path
            checkpoint_path = model_path
            config = None

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        # Load model using AlignmentModel.load()
        model = AlignmentModel.load(
            str(checkpoint_path),
            device=device,
            vocab_path=vocab_path
        )
        model.eval()

        # Get config from loaded model
        config = model.config

        # Get vocab_sizes
        vocab_sizes = model.vocab_sizes

        return model, config, vocab_sizes

    def forward(
        self,
        input_data: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            input_data: Dict containing:
                - categorical_features: Dict[str, Tensor] of [batch_size, seq_len]
                - coordinates: [batch_size, seq_len, 2] coordinate features
                - time_deltas: [batch_size, seq_len] time delta features
            attention_mask: [batch_size, seq_len] attention mask
            return_embeddings: If True, return both logits and embeddings

        Returns:
            logits: [batch_size, num_clusters] cluster logits
            embeddings (optional): [batch_size, d_model] sensor embeddings
        """
        # Get base embeddings from the sensor encoder
        encoder_output = self.sensor_encoder(
            input_data=input_data,
            attention_mask=attention_mask
        )

        # Use pooled embeddings (already L2 normalized in encoder)
        seq_embeddings = encoder_output.embeddings  # [batch_size, d_model]

        # Pass through clustering head
        logits = self.clustering_head(seq_embeddings)  # [batch_size, num_clusters]

        if return_embeddings:
            return logits, seq_embeddings
        else:
            return logits

    def get_embeddings(
        self,
        input_data: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract embeddings without computing logits.

        Args:
            input_data: Dict containing encoder inputs
            attention_mask: [batch_size, seq_len] attention mask

        Returns:
            embeddings: [batch_size, d_model] sequence-level embeddings
        """
        with torch.no_grad():
            encoder_output = self.sensor_encoder(
                input_data=input_data,
                attention_mask=attention_mask
            )
        return encoder_output.embeddings

    def get_cluster_predictions(
        self,
        input_data: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get cluster predictions for input sequences.

        Returns:
            predictions: [batch_size] cluster indices
        """
        with torch.no_grad():
            logits = self.forward(input_data, attention_mask)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def train_mode(self):
        """Set model to training mode."""
        self.train()
        self.clustering_head.train()

        if not self.freeze_encoder:
            self.sensor_encoder.train()
        else:
            self.sensor_encoder.eval()  # Keep encoder in eval mode if frozen

    def eval_mode(self):
        """Set model to evaluation mode."""
        self.eval()
        self.sensor_encoder.eval()
        self.clustering_head.eval()

    def get_data_paths(self) -> Dict[str, str]:
        """
        Get data paths from the pre-trained model's config.

        Returns:
            Dict with train_data_path, val_data_path, vocab_path
        """
        return {
            'train_data_path': self.alignment_config.train_data_path,
            'val_data_path': self.alignment_config.val_data_path,
            'vocab_path': self.alignment_config.vocab_path
        }
