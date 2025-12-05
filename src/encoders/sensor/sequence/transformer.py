"""
Transformer-based sensor encoder.

A modular, improved version of the original SensorEncoder with:
- Proper padding handling (padding ignored in attention/pooling)
- Configurable metadata features (coordinates, time, etc.)
- Support for variable-length sequences
- Clean interface for CLIP alignment
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from encoders.base import SequenceEncoder, EncoderOutput
from encoders.config import TransformerEncoderConfig
from encoders.sensor.sequence.projection import create_projection_head


class FourierFeatures(nn.Module):
    """Fourier features for continuous coordinates."""

    def __init__(self, d_model: int, num_bands: int = 12):
        super().__init__()
        self.num_bands = num_bands
        self.proj = nn.Linear(4 * num_bands, d_model)  # 2 coords × 2 functions × num_bands

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [batch_size, seq_len, 2] normalized (x,y) coordinates

        Returns:
            features: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = coords.shape

        # Create frequency bands
        freqs = torch.arange(self.num_bands, device=coords.device, dtype=coords.dtype)
        freqs = 2.0 ** freqs  # [num_bands]

        # coords: [batch_size, seq_len, 2]
        # freqs: [num_bands] -> [1, 1, 2, num_bands]
        coords_expanded = coords.unsqueeze(-1)  # [batch_size, seq_len, 2, 1]
        freqs_expanded = freqs.view(1, 1, 1, -1)  # [1, 1, 1, num_bands]

        # Compute sin/cos features
        angles = coords_expanded * freqs_expanded * math.pi  # [batch_size, seq_len, 2, num_bands]
        sin_features = torch.sin(angles)  # [batch_size, seq_len, 2, num_bands]
        cos_features = torch.cos(angles)  # [batch_size, seq_len, 2, num_bands]

        # Concatenate and reshape
        fourier_features = torch.cat([sin_features, cos_features], dim=-1)  # [batch_size, seq_len, 2, 2*num_bands]
        fourier_features = fourier_features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 4*num_bands]

        return self.proj(fourier_features)


class ALiBiAttention(nn.Module):
    """Multi-head attention with ALiBi positional bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # ALiBi slopes
        self.register_buffer("slopes", self._get_alibi_slopes(n_heads))

    def _get_alibi_slopes(self, n_heads: int) -> torch.Tensor:
        """Get ALiBi slopes for each head."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(n_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(n_heads))
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes.extend(get_slopes_power_of_2(2*closest_power_of_2)[0::2][:n_heads-closest_power_of_2])
            return torch.tensor(slopes)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] boolean mask (True = valid, False = padding)

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add ALiBi bias
        position_bias = self._get_alibi_bias(seq_len, device=x.device)
        scores = scores + position_bias

        # Apply mask - mask out padding positions
        if mask is not None:
            # Convert boolean mask to attention mask
            # mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attn_mask = attn_mask.expand(-1, self.n_heads, seq_len, -1)
            # Set padding positions to -inf so they get 0 weight after softmax
            scores = scores.masked_fill(~attn_mask, float('-inf'))

        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.out_proj(out)

    def _get_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get ALiBi positional bias matrix."""
        # Create distance matrix
        positions = torch.arange(seq_len, device=device)
        distances = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]
        distances = distances.abs()

        # Apply slopes
        bias = -distances.unsqueeze(0) * self.slopes.unsqueeze(-1).unsqueeze(-1)  # [n_heads, seq_len, seq_len]
        return bias.unsqueeze(0)  # [1, n_heads, seq_len, seq_len]


class TransformerLayer(nn.Module):
    """Pre-LN Transformer layer with ALiBi attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ALiBiAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] boolean mask (True = valid)
        """
        # Pre-LN: normalize before attention
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerSensorEncoder(SequenceEncoder):
    """
    Transformer-based sensor encoder with configurable metadata features.

    This is the modular, improved version of the original SensorEncoder.
    Key improvements:
    - Properly handles variable-length sequences with padding
    - Configurable metadata (can enable/disable coordinates, time, etc.)
    - Clean interface for CLIP alignment
    - Better separation of concerns
    """

    def __init__(self, config: TransformerEncoderConfig):
        super().__init__(config)
        self.config = config

        # Categorical embeddings (sensor, state, room, etc.)
        self.embeddings = nn.ModuleDict()
        for field, vocab_size in config.vocab_sizes.items():
            if field in config.metadata.categorical_fields:
                self.embeddings[field] = nn.Embedding(vocab_size, config.d_model)

        # Continuous features
        if config.metadata.use_coordinates:
            self.fourier_features = FourierFeatures(config.d_model, config.fourier_bands)

        if config.metadata.use_time_deltas:
            self.time_delta_embedding = nn.Embedding(config.metadata.time_delta_bins, config.d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        # Positional encoding (if not using ALiBi)
        if not config.use_alibi and config.use_learned_pe:
            self.pos_embedding = nn.Parameter(torch.randn(1, config.max_seq_len + 1, config.d_model))
        else:
            self.pos_embedding = None

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Pooling projection
        self.pool_proj = nn.Linear(config.d_model, config.d_model)

        # CLIP projection head (linear or MLP)
        self.clip_proj = create_projection_head(
            projection_type=config.projection_type,
            d_model=config.d_model,
            projection_dim=config.projection_dim,
            hidden_dim=config.projection_hidden_dim,
            num_layers=config.projection_num_layers,
            dropout=config.dropout
        )

        self.dropout = nn.Dropout(config.dropout)

    @property
    def d_model(self) -> int:
        return self.config.d_model

    def _log_bucket_time_delta(self, time_deltas: torch.Tensor) -> torch.Tensor:
        """Convert time deltas to log-bucketed indices."""
        # Clamp to avoid log(0)
        time_deltas = torch.clamp(time_deltas, min=1e-6)

        # Log scale buckets
        log_deltas = torch.log(time_deltas + 1)  # +1 to handle 0 case
        max_log = math.log(self.config.metadata.time_delta_max_seconds + 1)

        # Normalize to [0, time_delta_bins-1]
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
        Create initial token embeddings from input features.

        Args:
            categorical_features: Dict of [batch_size, seq_len] tensors
            continuous_features: Dict with 'coordinates' [B, L, 2], 'time_deltas' [B, L]
            attention_mask: [batch_size, seq_len] boolean mask (True = valid)

        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        batch_size = attention_mask.shape[0] if attention_mask is not None else list(categorical_features.values())[0].shape[0]
        seq_len = list(categorical_features.values())[0].shape[1]
        device = list(categorical_features.values())[0].device

        # Initialize token embeddings
        token_embeddings = torch.zeros(batch_size, seq_len, self.config.d_model, device=device)

        # Add categorical embeddings
        for field, indices in categorical_features.items():
            if field in self.embeddings:
                token_embeddings += self.embeddings[field](indices)

        # Add coordinate features (if enabled)
        if self.config.metadata.use_coordinates and 'coordinates' in continuous_features:
            coord_features = self.fourier_features(continuous_features['coordinates'])
            token_embeddings += coord_features

        # Add time delta features (if enabled)
        if self.config.metadata.use_time_deltas and 'time_deltas' in continuous_features:
            time_bucket_indices = self._log_bucket_time_delta(continuous_features['time_deltas'])
            time_embeddings = self.time_delta_embedding(time_bucket_indices)
            token_embeddings += time_embeddings

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
        cls_embedding = sequence_embeddings[:, 0]  # [batch_size, d_model]

        if self.config.pooling == 'cls':
            pooled = cls_embedding
        elif self.config.pooling == 'mean':
            # Mean pool over valid tokens (excluding CLS and padding)
            token_embeddings = sequence_embeddings[:, 1:]  # [batch_size, seq_len, d_model]
            if attention_mask is not None:
                # Mask out padding for mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings)
                masked_embeddings = token_embeddings * mask_expanded
                pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = token_embeddings.mean(dim=1)
        elif self.config.pooling == 'cls_mean':
            # Weighted combination of CLS and mean
            token_embeddings = sequence_embeddings[:, 1:]  # [batch_size, seq_len, d_model]
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
                - categorical_features: Dict of [batch_size, seq_len] tensors
                - coordinates: [batch_size, seq_len, 2] (if use_coordinates)
                - time_deltas: [batch_size, seq_len] (if use_time_deltas)
            attention_mask: [batch_size, seq_len] boolean mask (True = valid, False = padding)
            **kwargs: Additional arguments (e.g., blackout_masks for MLM)

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

        # Create initial embeddings
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
        # Run full forward pass
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

