"""
SCAN Clustering Model for src-v2 pipeline.
Uses pre-trained sensor encoder and adds clustering head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import os
from pathlib import Path

from .sensor_encoder import SensorEncoder


class SCANClusteringModel(nn.Module):
    """
    SCAN clustering model that uses a pre-trained sensor encoder
    and adds a clustering classification head.
    """

    def __init__(
        self,
        pretrained_model_path: str,
        num_clusters: int,
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        super().__init__()

        self.num_clusters = num_clusters
        self.pretrained_model_path = pretrained_model_path
        self.freeze_encoder = freeze_encoder

        # Load the pre-trained sensor encoder
        self.sensor_encoder = self._load_pretrained_encoder(pretrained_model_path)

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

    def _load_pretrained_encoder(self, model_path: str) -> SensorEncoder:
        """Load pre-trained sensor encoder from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pre-trained model not found at: {model_path}")

        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Extract configuration to reconstruct the model
        if 'config' in checkpoint:
            config = checkpoint['config']
        elif 'hyperparameters' in checkpoint:
            config = checkpoint['hyperparameters']
        else:
            # Try to load from hyperparameters.json in the same directory
            model_dir = Path(model_path).parent
            config_path = model_dir / 'hyperparameters.json'
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError("Cannot find model configuration. Need config/hyperparameters in checkpoint or hyperparameters.json file.")

        # Update vocab sizes from checkpoint if available
        vocab_sizes = config.get('vocab_sizes', {})
        if 'vocab_sizes' in checkpoint:
            vocab_sizes = checkpoint['vocab_sizes']

        # Create sensor encoder with the same configuration
        sensor_encoder = SensorEncoder(
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            fourier_bands=config['fourier_bands'],
            max_seq_len=config['max_seq_len'],
            vocab_sizes=vocab_sizes
        )

        # Load the state dict
        if 'sensor_encoder_state_dict' in checkpoint:
            sensor_encoder.load_state_dict(checkpoint['sensor_encoder_state_dict'])
        elif 'model_state_dict' in checkpoint:
            # Try to extract sensor encoder weights from combined state dict
            sensor_encoder_state = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith('sensor_encoder.'):
                    new_key = key.replace('sensor_encoder.', '')
                    sensor_encoder_state[new_key] = value
            sensor_encoder.load_state_dict(sensor_encoder_state)
        else:
            raise ValueError("Cannot find sensor encoder weights in checkpoint")

        return sensor_encoder

    def forward(
        self,
        categorical_features: Dict[str, torch.Tensor],
        coordinates: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: torch.Tensor,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            categorical_features: Dict of categorical feature tensors
            coordinates: [batch_size, seq_len, 2] coordinate features
            time_deltas: [batch_size, seq_len] time delta features
            mask: [batch_size, seq_len] attention mask
            return_embeddings: If True, return both logits and embeddings

        Returns:
            logits: [batch_size, num_clusters] cluster logits
            embeddings (optional): [batch_size, d_model] sensor embeddings
        """
        # Get base embeddings from the sensor encoder (without CLIP projection)
        # This returns [batch_size, d_model] pooled and normalized embeddings
        seq_embeddings = self.sensor_encoder(
            categorical_features=categorical_features,
            coordinates=coordinates,
            time_deltas=time_deltas,
            mask=mask
        )

        # Pass through clustering head
        logits = self.clustering_head(seq_embeddings)  # [batch_size, num_clusters]

        if return_embeddings:
            return logits, seq_embeddings
        else:
            return logits

    def get_embeddings(
        self,
        categorical_features: Dict[str, torch.Tensor],
        coordinates: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract embeddings without computing logits.

        Returns:
            embeddings: [batch_size, d_model] sequence-level embeddings
        """
        with torch.no_grad():
            # Get base embeddings directly from sensor encoder (without CLIP projection)
            embeddings = self.sensor_encoder(
                categorical_features=categorical_features,
                coordinates=coordinates,
                time_deltas=time_deltas,
                mask=mask
            )
        return embeddings

    def update_vocab_sizes(self, vocab_sizes: Dict[str, int]):
        """Update vocabulary sizes in the sensor encoder."""
        # Note: Vocabulary sizes are already set when loading the pre-trained model
        # The sensor encoder embeddings are created with the correct vocab sizes from checkpoint
        pass

    def train_mode(self):
        """Set model to training mode."""
        self.train()
        if not self.freeze_encoder:
            self.sensor_encoder.train()
        else:
            self.sensor_encoder.eval()  # Keep encoder in eval mode if frozen

    def eval_mode(self):
        """Set model to evaluation mode."""
        self.eval()
        self.sensor_encoder.eval()
