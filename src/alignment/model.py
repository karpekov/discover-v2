"""
Alignment model that combines sensor encoder, text encoder, and CLIP loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import yaml

from src.alignment.config import AlignmentConfig
from src.encoders import build_encoder
from src.encoders.base import EncoderOutput
from src.encoders.sensor.sequence.projection import create_projection_head
from src.losses.clip import CLIPLoss
from src.text_encoders import build_text_encoder


class AlignmentModel(nn.Module):
    """
    Model for aligning sensor encoder outputs with text embeddings.

    Components:
    - Sensor encoder (trainable)
    - Sensor projection head (trainable)
    - Text projection head (optional, trainable if specified)
    - CLIP loss function (with learnable temperature)

    The text encoder itself is frozen and uses pre-computed embeddings.
    """

    def __init__(self, config: AlignmentConfig, vocab_sizes: Dict[str, int]):
        super().__init__()
        self.config = config
        self.vocab_sizes = vocab_sizes

        # Build sensor encoder
        self.sensor_encoder = self._build_sensor_encoder()

        # Build sensor projection head
        self.sensor_projection = self._build_sensor_projection()

        # Build text projection head (optional)
        self.text_projection = self._build_text_projection()

        # Build CLIP loss
        self.clip_loss = CLIPLoss(
            temperature_init=config.loss.temperature_init,
            learnable_temperature=config.loss.learnable_temperature,
            use_hard_negatives=config.loss.use_hard_negatives,
            hard_negative_config={
                'memory_size': config.loss.hard_negative_memory_size,
                'hard_negative_ratio': config.loss.hard_negative_ratio,
                'sampling_strategy': config.loss.hard_negative_strategy,
                'temperature_for_sampling': config.loss.hard_negative_sampling_temperature,
            } if config.loss.use_hard_negatives else None
        )

        # MLM heads (optional, only if mlm_weight > 0)
        self.mlm_heads = None
        if config.loss.mlm_weight > 0:
            from src.models.mlm_heads import MLMHeads
            self.mlm_heads = MLMHeads(
                d_model=self.sensor_encoder.config.d_model,
                vocab_sizes=vocab_sizes,
                dropout=self.sensor_encoder.config.dropout
            )

    def _build_sensor_encoder(self):
        """Build sensor encoder from config."""
        if self.config.encoder_config_path is None:
            raise ValueError("encoder_config_path is required")

        # Load encoder config
        with open(self.config.encoder_config_path, 'r') as f:
            encoder_config_dict = yaml.safe_load(f)

        # Add vocab sizes to encoder config
        encoder_config_dict['vocab_sizes'] = self.vocab_sizes

        # Build encoder using the factory function
        encoder = build_encoder(encoder_config_dict)

        return encoder

    def _build_sensor_projection(self):
        """Build projection head for sensor embeddings."""
        proj_config = self.config.sensor_projection
        d_model = self.sensor_encoder.config.d_model

        return create_projection_head(
            projection_type=proj_config.type,
            d_model=d_model,
            projection_dim=proj_config.dim,
            hidden_dim=proj_config.hidden_dim,
            num_layers=proj_config.num_layers,
            dropout=proj_config.dropout,
            use_bn=proj_config.use_bn
        )

    def _build_text_projection(self):
        """Build projection head for text embeddings (optional)."""
        if self.config.text_projection is None:
            return None

        # We'll defer initialization until we see the first batch
        # to infer the text embedding dimension automatically
        return None  # Will be initialized in forward pass

    def forward(
        self,
        sensor_data: Dict[str, torch.Tensor],
        text_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_encoder_output: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for alignment.

        Args:
            sensor_data: Dictionary with sensor input features
                - categorical_features: [batch, seq_len, num_fields]
                - coordinates: [batch, seq_len, 2]
                - time_deltas: [batch, seq_len]
            text_embeddings: [batch, text_dim] pre-computed text embeddings
            attention_mask: [batch, seq_len] attention mask (True = valid)
            return_encoder_output: Whether to return full encoder output

        Returns:
            Dictionary with:
                - sensor_embeddings_projected: [batch, proj_dim]
                - text_embeddings_projected: [batch, proj_dim]
                - encoder_output: EncoderOutput (if return_encoder_output=True)
        """
        # Encode sensor data
        encoder_output = self.sensor_encoder(sensor_data, attention_mask)

        # Project sensor embeddings
        sensor_embeddings_projected = self.sensor_projection(encoder_output.embeddings)
        sensor_embeddings_projected = F.normalize(sensor_embeddings_projected, p=2, dim=-1)

        # Initialize text projection on first forward pass if needed
        if self.config.text_projection is not None and self.text_projection is None:
            proj_config = self.config.text_projection
            text_dim = text_embeddings.shape[-1]  # Infer from actual embeddings
            self.text_projection = create_projection_head(
                projection_type=proj_config.type,
                d_model=text_dim,
                projection_dim=proj_config.dim,
                hidden_dim=proj_config.hidden_dim,
                num_layers=proj_config.num_layers,
                dropout=proj_config.dropout,
                use_bn=proj_config.use_bn
            ).to(text_embeddings.device)

        # Project text embeddings (if projection head exists)
        if self.text_projection is not None:
            text_embeddings_projected = self.text_projection(text_embeddings)
            text_embeddings_projected = F.normalize(text_embeddings_projected, p=2, dim=-1)
        else:
            # Use text embeddings as-is (should already be normalized)
            text_embeddings_projected = F.normalize(text_embeddings, p=2, dim=-1)

        result = {
            'sensor_embeddings_projected': sensor_embeddings_projected,
            'text_embeddings_projected': text_embeddings_projected,
        }

        # Generate MLM predictions if MLM heads exist
        if self.mlm_heads is not None:
            # Get token-level embeddings from encoder output
            token_embeddings = encoder_output.sequence_features  # [batch, seq_len, d_model]
            mlm_predictions = self.mlm_heads(token_embeddings)  # Dict[field, [batch, seq_len, vocab_size]]
            result['mlm_predictions'] = mlm_predictions

        if return_encoder_output:
            result['encoder_output'] = encoder_output

        return result

    def compute_loss(
        self,
        sensor_embeddings_projected: torch.Tensor,
        text_embeddings_projected: torch.Tensor,
        mlm_predictions: Optional[Dict[str, torch.Tensor]] = None,
        mlm_labels: Optional[Dict[str, torch.Tensor]] = None,
        mlm_mask_positions: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute alignment loss.

        Args:
            sensor_embeddings_projected: [batch, proj_dim] projected sensor embeddings
            text_embeddings_projected: [batch, proj_dim] projected text embeddings
            mlm_predictions: Optional MLM predictions for each field
            mlm_labels: Optional MLM labels for each field
            mlm_mask_positions: Optional MLM mask positions for each field

        Returns:
            total_loss: Scalar loss tensor
            loss_dict: Dictionary with loss components
        """
        # Compute CLIP loss
        clip_loss, _ = self.clip_loss(
            sensor_embeddings_projected,
            text_embeddings_projected,
            return_similarity_matrix=False
        )

        # Compute accuracies
        s2t_acc, t2s_acc = self.clip_loss.get_accuracy(
            sensor_embeddings_projected,
            text_embeddings_projected
        )

        # Initialize total loss
        total_loss = self.config.loss.clip_weight * clip_loss

        loss_dict = {
            'clip_loss': clip_loss.item(),
            'sensor_to_text_acc': s2t_acc,
            'text_to_sensor_acc': t2s_acc,
            'temperature': self.clip_loss.temperature.item()
        }

        # Add MLM loss if specified
        if self.config.loss.mlm_weight > 0 and mlm_predictions is not None:
            mlm_loss = self._compute_mlm_loss(mlm_predictions, mlm_labels, mlm_mask_positions)
            if mlm_loss is not None:
                total_loss += self.config.loss.mlm_weight * mlm_loss
                loss_dict['mlm_loss'] = mlm_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def _compute_mlm_loss(
        self,
        mlm_predictions: Dict[str, torch.Tensor],
        mlm_labels: Dict[str, torch.Tensor],
        mlm_mask_positions: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Compute MLM loss for masked fields.

        Note: Padding tokens are automatically excluded because SpanMasker
        respects the attention_mask and only masks valid (non-padded) positions.
        """
        if mlm_labels is None or mlm_mask_positions is None:
            return None

        total_mlm_loss = 0.0
        num_fields = 0

        for field, predictions in mlm_predictions.items():
            if field in mlm_labels and field in mlm_mask_positions:
                labels = mlm_labels[field]
                mask_pos = mlm_mask_positions[field]

                # Only compute loss for masked positions (excludes padding automatically)
                if mask_pos.sum() > 0:
                    masked_predictions = predictions[mask_pos]
                    masked_labels = labels[mask_pos]

                    field_loss = F.cross_entropy(masked_predictions, masked_labels, ignore_index=-100)
                    total_mlm_loss += field_loss
                    num_fields += 1

        if num_fields > 0:
            return total_mlm_loss / num_fields
        else:
            return None

    def get_trainable_parameters(self):
        """Get all trainable parameters for optimizer."""
        params = []

        # Sensor encoder parameters
        params.extend(list(self.sensor_encoder.parameters()))

        # Sensor projection parameters
        params.extend(list(self.sensor_projection.parameters()))

        # Text projection parameters (if exists)
        if self.text_projection is not None:
            params.extend(list(self.text_projection.parameters()))

        # CLIP loss parameters (learnable temperature)
        params.extend(list(self.clip_loss.parameters()))

        # MLM heads parameters (if exists)
        if self.mlm_heads is not None:
            params.extend(list(self.mlm_heads.parameters()))

        return params

    def save(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'config': self.config,
            'vocab_sizes': self.vocab_sizes,
            'sensor_encoder_state_dict': self.sensor_encoder.state_dict(),
            'sensor_projection_state_dict': self.sensor_projection.state_dict(),
            'clip_loss_state_dict': self.clip_loss.state_dict(),
        }

        if self.text_projection is not None:
            checkpoint['text_projection_state_dict'] = self.text_projection.state_dict()

        if self.mlm_heads is not None:
            checkpoint['mlm_heads_state_dict'] = self.mlm_heads.state_dict()

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device or 'cpu')

        config = checkpoint['config']
        vocab_sizes = checkpoint['vocab_sizes']

        # Create model
        model = cls(config, vocab_sizes)

        # Load state dicts
        model.sensor_encoder.load_state_dict(checkpoint['sensor_encoder_state_dict'])
        model.sensor_projection.load_state_dict(checkpoint['sensor_projection_state_dict'])
        model.clip_loss.load_state_dict(checkpoint['clip_loss_state_dict'])

        if 'text_projection_state_dict' in checkpoint:
            model.text_projection.load_state_dict(checkpoint['text_projection_state_dict'])

        if 'mlm_heads_state_dict' in checkpoint:
            model.mlm_heads.load_state_dict(checkpoint['mlm_heads_state_dict'])

        if device is not None:
            model = model.to(device)

        return model

