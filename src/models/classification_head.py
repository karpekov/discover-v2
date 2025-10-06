import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from .sensor_encoder import SensorEncoder


class SensorClassificationModel(nn.Module):
    """
    Classification model that uses a pre-trained SensorEncoder backbone
    with a frozen transformer and trains only a classification head.
    """

    def __init__(
        self,
        pretrained_sensor_encoder: SensorEncoder,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        freeze_backbone: bool = True
    ):
        """
        Args:
            pretrained_sensor_encoder: Pre-trained SensorEncoder model
            num_classes: Number of classification classes
            hidden_dim: Hidden dimension for MLP head (if None, uses linear classifier)
            dropout: Dropout probability for classification head
            freeze_backbone: Whether to freeze the sensor encoder backbone
        """
        super().__init__()

        self.sensor_encoder = pretrained_sensor_encoder
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        # Freeze the backbone if requested
        if freeze_backbone:
            for param in self.sensor_encoder.parameters():
                param.requires_grad = False
            # Ensure the backbone is in eval mode
            self.sensor_encoder.eval()

        # Get the output dimension from the sensor encoder
        # The pooled output (before CLIP projection) is d_model dimensional
        encoder_output_dim = self.sensor_encoder.d_model

        # Create classification head
        if hidden_dim is not None:
            # MLP head: encoder_output -> hidden -> num_classes
            self.classifier = nn.Sequential(
                nn.Linear(encoder_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            # Simple linear classifier
            self.classifier = nn.Linear(encoder_output_dim, num_classes)

        # Initialize classification head
        self._initialize_classifier()

    def _initialize_classifier(self):
        """Initialize the classification head parameters."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        categorical_features: Dict[str, torch.Tensor],
        coordinates: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        blackout_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            categorical_features: Dict of [batch_size, seq_len] tensors
            coordinates: [batch_size, seq_len, 2] normalized (x,y) coordinates
            time_deltas: [batch_size, seq_len] time since previous event (seconds)
            mask: [batch_size, seq_len] boolean mask (True = valid)
            blackout_masks: Optional dict of blackout masks for MLM-style masking

        Returns:
            logits: [batch_size, num_classes] classification logits
        """
        # Ensure backbone is in appropriate mode
        if self.freeze_backbone:
            self.sensor_encoder.eval()

        # Get embeddings from the sensor encoder (before CLIP projection)
        # This uses the pooled representation: 0.5 * CLS + 0.5 * mean pooling
        with torch.set_grad_enabled(not self.freeze_backbone):
            embeddings = self.sensor_encoder.forward(
                categorical_features=categorical_features,
                coordinates=coordinates,
                time_deltas=time_deltas,
                mask=mask,
                blackout_masks=blackout_masks
            )

        # Apply classification head
        logits = self.classifier(embeddings)

        return logits

    def predict_proba(
        self,
        categorical_features: Dict[str, torch.Tensor],
        coordinates: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        blackout_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Get class probabilities.

        Returns:
            probs: [batch_size, num_classes] class probabilities
        """
        logits = self.forward(
            categorical_features=categorical_features,
            coordinates=coordinates,
            time_deltas=time_deltas,
            mask=mask,
            blackout_masks=blackout_masks
        )
        return F.softmax(logits, dim=-1)

    def predict(
        self,
        categorical_features: Dict[str, torch.Tensor],
        coordinates: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        blackout_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Get predicted class indices.

        Returns:
            predictions: [batch_size] predicted class indices
        """
        logits = self.forward(
            categorical_features=categorical_features,
            coordinates=coordinates,
            time_deltas=time_deltas,
            mask=mask,
            blackout_masks=blackout_masks
        )
        return torch.argmax(logits, dim=-1)

    def get_embeddings(
        self,
        categorical_features: Dict[str, torch.Tensor],
        coordinates: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        blackout_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Get the learned embeddings from the backbone.

        Returns:
            embeddings: [batch_size, d_model] learned embeddings
        """
        with torch.set_grad_enabled(not self.freeze_backbone):
            embeddings = self.sensor_encoder.forward(
                categorical_features=categorical_features,
                coordinates=coordinates,
                time_deltas=time_deltas,
                mask=mask,
                blackout_masks=blackout_masks
            )
        return embeddings

    def unfreeze_backbone(self):
        """Unfreeze the backbone for end-to-end fine-tuning."""
        self.freeze_backbone = False
        for param in self.sensor_encoder.parameters():
            param.requires_grad = True
        self.sensor_encoder.train()

    def freeze_backbone_layers(self):
        """Freeze the backbone layers."""
        self.freeze_backbone = True
        for param in self.sensor_encoder.parameters():
            param.requires_grad = False
        self.sensor_encoder.eval()


def load_pretrained_sensor_encoder(
    checkpoint_path: str,
    vocab_sizes: Optional[Dict[str, int]] = None,
    device: torch.device = torch.device('cpu')
) -> SensorEncoder:
    """
    Load a pre-trained SensorEncoder from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint
        vocab_sizes: Vocabulary sizes for the model
        device: Device to load the model on

    Returns:
        sensor_encoder: Loaded SensorEncoder model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Use vocabulary sizes from checkpoint if available, otherwise use provided ones
    if 'vocab_sizes' in checkpoint:
        vocab_sizes = checkpoint['vocab_sizes']
    elif vocab_sizes is None:
        raise ValueError("vocab_sizes must be provided if not available in checkpoint")

    # Extract model configuration from checkpoint if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_args = {
            'd_model': config.get('d_model', 768),
            'n_layers': config.get('n_layers', 6),
            'n_heads': config.get('n_heads', 8),
            'd_ff': config.get('d_ff', 3072),
            'max_seq_len': config.get('max_seq_len', 512),
            'dropout': config.get('dropout', 0.1),
            'fourier_bands': config.get('fourier_bands', 12),
            'use_rope_time': config.get('use_rope_time', False),
            'use_rope_2d': config.get('use_rope_2d', False)
        }
    else:
        # Use default parameters for Milan baseline model
        model_args = {
            'd_model': 512,  # Milan baseline uses 512
            'n_layers': 4,   # Milan baseline uses 4 layers
            'n_heads': 8,
            'd_ff': 2048,    # Milan baseline uses 2048
            'max_seq_len': 512,
            'dropout': 0.3,  # Milan baseline uses 0.3
            'fourier_bands': 12,
            'use_rope_time': False,
            'use_rope_2d': False
        }

    # Create model
    sensor_encoder = SensorEncoder(
        vocab_sizes=vocab_sizes,
        **model_args
    )

    # Load state dict
    if 'sensor_encoder_state_dict' in checkpoint:
        sensor_encoder.load_state_dict(checkpoint['sensor_encoder_state_dict'])
    elif 'sensor_encoder' in checkpoint:
        sensor_encoder.load_state_dict(checkpoint['sensor_encoder'])
    elif 'model_state_dict' in checkpoint:
        # Extract sensor encoder from full model state dict
        sensor_encoder_state = {
            k[len('sensor_encoder.'):]: v
            for k, v in checkpoint['model_state_dict'].items()
            if k.startswith('sensor_encoder.')
        }
        sensor_encoder.load_state_dict(sensor_encoder_state)
    else:
        # Assume the checkpoint is the sensor encoder state dict
        sensor_encoder.load_state_dict(checkpoint)

    sensor_encoder.to(device)
    sensor_encoder.eval()

    return sensor_encoder


def create_l2_label_mapping(metadata_path: str = None) -> Dict[str, int]:
    """
    Create label to index mapping for L2 activity labels.
    Excludes 'No_Activity' but keeps 'Other' as requested.

    Args:
        metadata_path: Path to metadata file (optional, uses hardcoded mapping if None)

    Returns:
        label_to_idx: Dictionary mapping L2 labels to indices
    """
    # Based on the metadata analysis, the L2 labels are:
    l2_labels = [
        'Cook',           # 0
        'Sleep',          # 1
        'Work',           # 2
        'Eat',            # 3
        'Relax',          # 4
        'Bathing',        # 5
        'Other',          # 6 - Keep as requested
        'Bed_to_toilet',  # 7
        'Take_medicine',  # 8
        'Leave_Home',     # 9
        'Enter_Home'      # 10
        # Exclude 'No_Activity' as requested
    ]

    label_to_idx = {label: idx for idx, label in enumerate(l2_labels)}

    return label_to_idx, l2_labels
