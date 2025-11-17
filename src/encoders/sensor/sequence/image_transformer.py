"""
Image-based Transformer Sensor Encoder.

This encoder uses pre-computed vision model embeddings (CLIP, DINOv2, SigLIP, etc.)
as frozen input features instead of learnable embeddings. The transformer layers
remain trainable for MLM and CLIP alignment tasks.

Key differences from TransformerSensorEncoder:
- Loads frozen image embeddings for all sensor-state pairs
- Maps sensor activations to pre-computed embeddings (frozen)
- No learnable sensor/state embeddings
- Transformer processes the frozen image features (trainable)
- Projection layer for input dimension matching (frozen or trainable)

Usage:
    1. Generate images: python -m src.encoders.sensor.image.generate_images --dataset milan
    2. Embed images: python -m src.encoders.sensor.image.embed_images --dataset milan --model clip
    3. Train: Use this encoder in alignment training with image_embeddings_config
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from src.encoders.base import SequenceEncoder, EncoderOutput
from src.encoders.config import TransformerEncoderConfig
from src.encoders.sensor.sequence.projection import create_projection_head
from src.encoders.sensor.sequence.transformer import (
    FourierFeatures,
    TransformerLayer
)

logger = logging.getLogger(__name__)


class ImageEmbeddingLookup(nn.Module):
    """
    Lookup table for frozen image embeddings.

    Loads pre-computed vision model embeddings for all sensor-state pairs
    and provides fast lookup during training.
    """

    def __init__(
        self,
        dataset: str,
        model_name: str,
        dataset_type: str = "casas",
        image_size: int = 224,
        project_root: Optional[Path] = None
    ):
        """
        Args:
            dataset: Dataset name (e.g., "milan")
            model_name: Vision model name (e.g., "clip", "dinov2", "siglip")
            dataset_type: Dataset type (default: "casas")
            image_size: Image dimension used for embedding (default: 224)
            project_root: Project root directory (auto-detected if None)
        """
        super().__init__()

        self.dataset = dataset
        self.model_name = model_name
        self.dataset_type = dataset_type
        self.image_size = image_size

        # Load embeddings
        embeddings_data = self._load_embeddings(project_root)

        # Create lookup dictionary: sensor_id + state -> embedding
        self.embedding_dict = {}
        for i, key in enumerate(embeddings_data['image_keys']):
            # key format: "M001_ON", "D002_CLOSE", etc.
            self.embedding_dict[key] = embeddings_data['embeddings'][i]

        # Store embeddings as frozen tensor
        embeddings_array = embeddings_data['embeddings']
        self.embedding_dim = embeddings_array.shape[1]

        # Register as buffer (frozen, but moved to device with model)
        self.register_buffer(
            'embeddings_tensor',
            torch.from_numpy(embeddings_array).float()
        )

        # Create fast lookup indices
        self.sensor_ids = list(embeddings_data['sensor_ids'])
        self.states = list(embeddings_data['states'])
        self.image_keys = list(embeddings_data['image_keys'])

        # Create mapping for fast indexing
        self.key_to_idx = {key: i for i, key in enumerate(self.image_keys)}

        logger.info(
            f"Loaded {len(self.image_keys)} frozen image embeddings "
            f"({self.embedding_dim}D) from {model_name}"
        )

    def _load_embeddings(self, project_root: Optional[Path] = None) -> Dict:
        """Load embeddings from NPZ file."""
        if project_root is None:
            # Auto-detect project root
            project_root = Path(__file__).parent.parent.parent.parent.parent

        # Construct path to embeddings
        dim_folder = f"dim{self.image_size}"

        # Simplified model naming (same as in embed_images.py)
        model_name_map = {
            "clip": "clip_base",
            "dinov2": "dinov2",
            "siglip": "siglip_base_patch16_224"
        }

        clean_model_name = model_name_map.get(
            self.model_name,
            self.model_name.replace("/", "_").replace("-", "_")
        )

        embeddings_file = (
            project_root / "data" / "processed" / self.dataset_type / self.dataset /
            "layout_embeddings" / "embeddings" / clean_model_name / dim_folder / "embeddings.npz"
        )

        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Image embeddings not found: {embeddings_file}\n\n"
                f"Please generate them first:\n"
                f"  1. Generate images: python -m src.encoders.sensor.image.generate_images "
                f"--dataset {self.dataset}\n"
                f"  2. Embed images: python -m src.encoders.sensor.image.embed_images "
                f"--dataset {self.dataset} --model {self.model_name}"
            )

        logger.info(f"Loading image embeddings from: {embeddings_file}")
        data = np.load(embeddings_file, allow_pickle=True)

        return {
            'embeddings': data['embeddings'],
            'sensor_ids': data['sensor_ids'],
            'states': data['states'],
            'image_keys': data['image_keys']
        }

    def get_embedding(self, sensor_id: str, state: str) -> torch.Tensor:
        """
        Get embedding for a single sensor-state pair.

        Args:
            sensor_id: Sensor identifier (e.g., "M001")
            state: State (e.g., "ON", "OFF", "CLOSE")

        Returns:
            Frozen embedding tensor [embedding_dim]
        """
        key = f"{sensor_id}_{state}"

        if key not in self.key_to_idx:
            raise ValueError(
                f"No image embedding found for sensor '{sensor_id}' with state '{state}'. "
                f"Available sensors: {set(self.sensor_ids[:10])}... (showing first 10)"
            )

        idx = self.key_to_idx[key]
        return self.embeddings_tensor[idx]

    def _normalize_state(self, state_str: str) -> str:
        """
        Normalize noisy state values to clean states.
        
        Handles data quality issues like 'ON`' -> 'ON', 'ON0' -> 'ON', 'O' -> 'ON'
        """
        # Handle common typos and variations
        state_str = state_str.upper().strip()
        
        # Map noisy states to clean states
        normalization_map = {
            'ON`': 'ON',   # Backtick typo
            'ON0': 'ON',   # Zero typo
            'O': 'ON',     # Abbreviation
            'UNK': 'OFF',  # Default unknown to OFF
        }
        
        return normalization_map.get(state_str, state_str)
    
    def get_embeddings_batch(
        self,
        sensor_ids: torch.Tensor,
        states: torch.Tensor,
        vocab: Dict[str, Dict[str, int]]
    ) -> torch.Tensor:
        """
        Get embeddings for a batch of sensor activations.
        
        Args:
            sensor_ids: Tensor of sensor ID indices [batch_size, seq_len]
            states: Tensor of state indices [batch_size, seq_len]
            vocab: Vocabulary mapping (indices -> strings)
        
        Returns:
            Batch of embeddings [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = sensor_ids.shape
        device = sensor_ids.device
        
        # Create output tensor
        embeddings = torch.zeros(
            batch_size, seq_len, self.embedding_dim,
            device=device, dtype=self.embeddings_tensor.dtype
        )
        
        # Reverse vocabulary lookups
        idx_to_sensor = {v: k for k, v in vocab['sensor'].items()}
        idx_to_state = {v: k for k, v in vocab['state'].items()}
        
        # Track missing keys globally (class attribute)
        if not hasattr(self, '_missing_keys_logged'):
            self._missing_keys_logged = set()
            self._missing_count = 0
        
        missing_keys = set()
        
        # Fill in embeddings
        for b in range(batch_size):
            for t in range(seq_len):
                sensor_idx = sensor_ids[b, t].item()
                state_idx = states[b, t].item()
                
                # Convert indices to strings
                sensor_id_str = idx_to_sensor.get(sensor_idx, f"UNK_{sensor_idx}")
                state_str = idx_to_state.get(state_idx, f"UNK_{state_idx}")
                
                # Normalize state to handle noisy data
                state_str_normalized = self._normalize_state(state_str)
                
                # Lookup embedding
                key = f"{sensor_id_str}_{state_str_normalized}"
                
                if key in self.key_to_idx:
                    idx = self.key_to_idx[key]
                    embeddings[b, t] = self.embeddings_tensor[idx]
                else:
                    # Track missing key
                    missing_keys.add(key)
                    self._missing_count += 1
        
        # Only log warnings for new missing keys or if count is significant
        new_missing_keys = missing_keys - self._missing_keys_logged
        if new_missing_keys:
            self._missing_keys_logged.update(new_missing_keys)
            # Only warn if this is a new key or if we've seen too many misses
            if self._missing_count <= 100:  # Log first 100 occurrences
                logger.debug(
                    f"Image embeddings not found for {len(new_missing_keys)} new sensor-state pairs: "
                    f"{list(new_missing_keys)[:3]}. Using zero vectors. "
                    f"(Total missing: {self._missing_count})"
                )
            elif self._missing_count == 101:
                logger.warning(
                    f"Suppressing further missing embedding warnings. "
                    f"Total unique missing pairs: {len(self._missing_keys_logged)}"
                )
        
        return embeddings

    def forward(self, sensor_ids: torch.Tensor, states: torch.Tensor, vocab: Dict) -> torch.Tensor:
        """Forward pass - calls get_embeddings_batch."""
        return self.get_embeddings_batch(sensor_ids, states, vocab)


class ImageTransformerSensorEncoder(SequenceEncoder):
    """
    Image-based Transformer Sensor Encoder.

    Uses frozen vision model embeddings (CLIP, DINOv2, SigLIP) as input features
    and processes them with a trainable transformer for MLM and CLIP alignment.

    Architecture:
        1. Frozen image embeddings (pre-computed) for each sensor-state pair
        2. Optional projection to match d_model (frozen or trainable)
        3. Optional coordinate and time features (trainable)
        4. Trainable transformer layers with ALiBi attention
        5. Pooling and projection for CLIP alignment (trainable)

    Key Features:
        - Image embeddings remain frozen during training
        - Transformer and projections are trainable
        - Supports MLM loss on transformer outputs
        - Supports CLIP alignment loss
        - Handles variable-length sequences with padding
    """

    def __init__(
        self,
        config: TransformerEncoderConfig,
        dataset: str,
        image_model_name: str,
        dataset_type: str = "casas",
        image_size: int = 224,
        vocab: Optional[Dict[str, Dict[str, int]]] = None,
        freeze_input_projection: bool = True
    ):
        """
        Args:
            config: Transformer encoder configuration
            dataset: Dataset name (e.g., "milan")
            image_model_name: Vision model name (e.g., "clip", "dinov2", "siglip")
            dataset_type: Dataset type (default: "casas")
            image_size: Image dimension used (default: 224)
            vocab: Vocabulary mapping for sensor IDs and states (required for lookup)
            freeze_input_projection: Whether to freeze the input projection layer
        """
        super().__init__(config)
        self.config = config
        self.dataset = dataset
        self.image_model_name = image_model_name
        self.vocab = vocab

        # Load frozen image embeddings
        self.image_lookup = ImageEmbeddingLookup(
            dataset=dataset,
            model_name=image_model_name,
            dataset_type=dataset_type,
            image_size=image_size
        )

        # Input projection: image_dim -> d_model
        self.input_projection = nn.Linear(self.image_lookup.embedding_dim, config.d_model)

        # Initialize input projection to be close to identity if dimensions match
        if self.image_lookup.embedding_dim == config.d_model:
            with torch.no_grad():
                self.input_projection.weight.copy_(torch.eye(config.d_model))
                self.input_projection.bias.zero_()

        # Optionally freeze input projection
        if freeze_input_projection:
            for param in self.input_projection.parameters():
                param.requires_grad = False
            logger.info("Input projection layer is FROZEN")
        else:
            logger.info("Input projection layer is TRAINABLE")

        # Optional metadata features (trainable)
        self.use_metadata_features = False
        if config.metadata.use_coordinates:
            self.fourier_features = FourierFeatures(config.d_model, config.fourier_bands)
            self.use_metadata_features = True

        if config.metadata.use_time_deltas:
            self.time_delta_embedding = nn.Embedding(
                config.metadata.time_delta_bins, config.d_model
            )
            self.use_metadata_features = True

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        # Positional encoding (if not using ALiBi)
        if not config.use_alibi and config.use_learned_pe:
            self.pos_embedding = nn.Parameter(torch.randn(1, config.max_seq_len + 1, config.d_model))
        else:
            self.pos_embedding = None

        # Transformer layers (trainable)
        self.layers = nn.ModuleList([
            TransformerLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Pooling projection
        self.pool_proj = nn.Linear(config.d_model, config.d_model)

        # CLIP projection head (trainable)
        self.clip_proj = create_projection_head(
            projection_type=config.projection_type,
            d_model=config.d_model,
            projection_dim=config.projection_dim,
            hidden_dim=config.projection_hidden_dim,
            num_layers=config.projection_num_layers,
            dropout=config.dropout
        )

        self.dropout = nn.Dropout(config.dropout)

        logger.info(
            f"ImageTransformerSensorEncoder initialized:\n"
            f"  - Vision model: {image_model_name}\n"
            f"  - Image embedding dim: {self.image_lookup.embedding_dim}\n"
            f"  - Transformer d_model: {config.d_model}\n"
            f"  - Projection dim: {config.projection_dim}\n"
            f"  - Num sensors: {len(self.image_lookup.image_keys)}\n"
            f"  - Input projection: {'FROZEN' if freeze_input_projection else 'TRAINABLE'}\n"
            f"  - Metadata features: {self.use_metadata_features}"
        )

    @property
    def d_model(self) -> int:
        return self.config.d_model

    def _log_bucket_time_delta(self, time_deltas: torch.Tensor) -> torch.Tensor:
        """Convert time deltas to log-bucketed indices."""
        time_deltas = torch.clamp(time_deltas, min=1e-6)
        log_deltas = torch.log(time_deltas + 1)
        max_log = math.log(self.config.metadata.time_delta_max_seconds + 1)
        bucket_indices = (log_deltas / max_log * (self.config.metadata.time_delta_bins - 1)).long()
        bucket_indices = torch.clamp(bucket_indices, 0, self.config.metadata.time_delta_bins - 1)
        return bucket_indices

    def _create_embeddings(
        self,
        categorical_features: Dict[str, torch.Tensor],
        continuous_features: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create initial token embeddings from frozen image features + optional metadata.

        Args:
            categorical_features: Dict with 'sensor' [B, L] and 'state' [B, L] tensors
            continuous_features: Dict with 'coordinates' [B, L, 2], 'time_deltas' [B, L]
            attention_mask: [batch_size, seq_len] boolean mask (True = valid)

        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        sensor_ids = categorical_features['sensor']
        states = categorical_features['state']
        batch_size, seq_len = sensor_ids.shape
        device = sensor_ids.device

        # Get frozen image embeddings
        image_embeddings = self.image_lookup(sensor_ids, states, self.vocab)
        # image_embeddings: [batch_size, seq_len, image_embedding_dim]

        # Project to d_model (frozen or trainable)
        token_embeddings = self.input_projection(image_embeddings)
        # token_embeddings: [batch_size, seq_len, d_model]

        # Add optional metadata features (if enabled)
        if self.use_metadata_features:
            # Add coordinate features
            if self.config.metadata.use_coordinates and 'coordinates' in continuous_features:
                coord_features = self.fourier_features(continuous_features['coordinates'])
                token_embeddings = token_embeddings + coord_features

            # Add time delta features
            if self.config.metadata.use_time_deltas and 'time_deltas' in continuous_features:
                time_bucket_indices = self._log_bucket_time_delta(continuous_features['time_deltas'])
                time_embeddings = self.time_delta_embedding(time_bucket_indices)
                token_embeddings = token_embeddings + time_embeddings

        return token_embeddings

    def _pool_embeddings(
        self,
        sequence_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence embeddings into fixed-size representation.

        Args:
            sequence_embeddings: [batch_size, seq_len+1, d_model] (with CLS)
            attention_mask: [batch_size, seq_len] boolean mask (True = valid)

        Returns:
            pooled: [batch_size, d_model] L2-normalized
        """
        # Extract CLS token
        cls_embedding = sequence_embeddings[:, 0]

        if self.config.pooling == 'cls':
            pooled = cls_embedding
        elif self.config.pooling == 'mean':
            token_embeddings = sequence_embeddings[:, 1:]
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings)
                masked_embeddings = token_embeddings * mask_expanded
                pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = token_embeddings.mean(dim=1)
        elif self.config.pooling == 'cls_mean':
            token_embeddings = sequence_embeddings[:, 1:]
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings)
                masked_embeddings = token_embeddings * mask_expanded
                mean_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                mean_embedding = token_embeddings.mean(dim=1)

            w = self.config.pooling_cls_weight
            pooled = w * cls_embedding + (1 - w) * mean_embedding
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling}")

        # Project and normalize
        pooled = self.pool_proj(pooled)
        pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled

    def forward(
        self,
        input_data: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> EncoderOutput:
        """
        Main forward pass with pooling.

        Args:
            input_data: Dict containing:
                - categorical_features: Dict with 'sensor' and 'state' [batch_size, seq_len]
                - coordinates: [batch_size, seq_len, 2] (optional)
                - time_deltas: [batch_size, seq_len] (optional)
            attention_mask: [batch_size, seq_len] boolean mask (True = valid, False = padding)
            **kwargs: Additional arguments

        Returns:
            EncoderOutput with pooled embeddings
        """
        categorical_features = input_data['categorical_features']
        continuous_features = {
            k: v for k, v in input_data.items()
            if k in ['coordinates', 'time_deltas']
        }

        batch_size = list(categorical_features.values())[0].shape[0]
        seq_len = list(categorical_features.values())[0].shape[1]

        # Create initial embeddings (frozen image features + metadata)
        token_embeddings = self._create_embeddings(
            categorical_features,
            continuous_features,
            attention_mask
        )

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, token_embeddings], dim=1)  # [B, L+1, D]

        # Add positional encoding (if not using ALiBi)
        if self.pos_embedding is not None:
            embeddings = embeddings + self.pos_embedding[:, :seq_len+1]

        embeddings = self.dropout(embeddings)

        # Extend mask for CLS token
        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            extended_mask = torch.cat([cls_mask, attention_mask], dim=1)
        else:
            extended_mask = None

        # Apply transformer layers
        for layer in self.layers:
            embeddings = layer(embeddings, extended_mask)

        embeddings = self.ln_f(embeddings)

        # Pool embeddings
        pooled = self._pool_embeddings(embeddings, attention_mask)

        return EncoderOutput(
            embeddings=pooled,
            sequence_features=embeddings[:, 1:],  # Exclude CLS for MLM
            attention_mask=attention_mask
        )

    def get_sequence_features(
        self,
        input_data: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Get per-token hidden states for MLM.

        Args:
            input_data: Dict containing encoder inputs
            attention_mask: [batch_size, seq_len] boolean mask
            **kwargs: Additional arguments

        Returns:
            hidden_states: [batch_size, seq_len, d_model] (without CLS)
        """
        output = self.forward(input_data, attention_mask, **kwargs)
        return output.sequence_features

    def forward_clip(
        self,
        input_data: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with projection head for CLIP alignment.

        Args:
            input_data: Dict containing encoder inputs
            attention_mask: [batch_size, seq_len] boolean mask
            **kwargs: Additional arguments

        Returns:
            projected_embeddings: [batch_size, projection_dim] L2-normalized
        """
        # Get base embeddings
        output = self.forward(input_data, attention_mask, **kwargs)
        base_embeddings = output.embeddings

        # Apply projection head
        projected = self.clip_proj(base_embeddings)

        # L2 normalize
        projected = F.normalize(projected, p=2, dim=-1)

        return projected

