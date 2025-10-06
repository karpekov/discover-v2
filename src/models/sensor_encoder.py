import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


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
      mask: [batch_size, seq_len] boolean mask (True = valid)

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

    # Apply mask
    if mask is not None:
      # Convert boolean mask to attention mask
      attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
      attn_mask = attn_mask.expand(-1, self.n_heads, seq_len, -1)
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
    # Pre-LN: normalize before attention
    x = x + self.attn(self.ln1(x), mask)
    x = x + self.ffn(self.ln2(x))
    return x


class SensorEncoder(nn.Module):
  """
  Sensor tower: Transformer encoder for smart-home event sequences.
  """

  def __init__(
    self,
    vocab_sizes: Dict[str, int],
    d_model: int = 768,
    n_layers: int = 6,
    n_heads: int = 8,
    d_ff: int = 3072,
    max_seq_len: int = 512,
    dropout: float = 0.1,
    fourier_bands: int = 12,
    use_rope_time: bool = False,
    use_rope_2d: bool = False
  ):
    super().__init__()
    self.d_model = d_model
    self.use_rope_time = use_rope_time
    self.use_rope_2d = use_rope_2d

    # Categorical embeddings
    self.embeddings = nn.ModuleDict()
    for field, vocab_size in vocab_sizes.items():
      self.embeddings[field] = nn.Embedding(vocab_size, d_model)

    # Continuous features
    self.fourier_features = FourierFeatures(d_model, fourier_bands)

    # Time delta embedding (log-bucketed)
    self.time_delta_bins = 100  # Configurable
    self.time_delta_embedding = nn.Embedding(self.time_delta_bins, d_model)

    # CLS token
    self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    # Positional encoding (if not using RoPE)
    if not (use_rope_time or use_rope_2d):
      self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len + 1, d_model))

    # Transformer layers
    self.layers = nn.ModuleList([
      TransformerLayer(d_model, n_heads, d_ff, dropout)
      for _ in range(n_layers)
    ])

    # Final layer norm
    self.ln_f = nn.LayerNorm(d_model)

    # Pooling projection
    self.pool_proj = nn.Linear(d_model, d_model)

    # CLIP projection head
    self.clip_proj = nn.Linear(d_model, 512, bias=False)

    # Initialize projection head with small normal
    nn.init.normal_(self.clip_proj.weight, std=0.02)

    self.dropout = nn.Dropout(dropout)

  def _log_bucket_time_delta(self, time_deltas: torch.Tensor) -> torch.Tensor:
    """Convert time deltas to log-bucketed indices."""
    # Clamp to avoid log(0)
    time_deltas = torch.clamp(time_deltas, min=1e-6)

    # Log scale buckets
    log_deltas = torch.log(time_deltas + 1)  # +1 to handle 0 case
    max_log = math.log(3600 + 1)  # 1 hour max

    # Normalize to [0, time_delta_bins-1]
    bucket_indices = (log_deltas / max_log * (self.time_delta_bins - 1)).long()
    bucket_indices = torch.clamp(bucket_indices, 0, self.time_delta_bins - 1)

    return bucket_indices

  def forward(
    self,
    categorical_features: Dict[str, torch.Tensor],
    coordinates: torch.Tensor,
    time_deltas: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    blackout_masks: Optional[Dict[str, torch.Tensor]] = None
  ) -> torch.Tensor:
    """
    Args:
      categorical_features: Dict of [batch_size, seq_len] tensors
      coordinates: [batch_size, seq_len, 2] normalized (x,y) coordinates
      time_deltas: [batch_size, seq_len] time since previous event (seconds)
      mask: [batch_size, seq_len] boolean mask (True = valid)
      blackout_masks: Optional dict of [batch_size, seq_len] boolean masks for zeroing embeddings

    Returns:
      embeddings: [batch_size, d_model] L2-normalized
    """
    batch_size, seq_len = coordinates.shape[:2]

    # Sum categorical embeddings with blackout support
    token_embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=coordinates.device)
    for field, indices in categorical_features.items():
      if field in self.embeddings:
        field_embeddings = self.embeddings[field](indices)
        # Apply blackout mask if provided
        if blackout_masks and field in blackout_masks:
          blackout_mask = blackout_masks[field].unsqueeze(-1)  # [batch_size, seq_len, 1]
          field_embeddings = field_embeddings * (~blackout_mask).float()
        token_embeddings += field_embeddings

    # Add Fourier features for coordinates with blackout support
    coord_features = self.fourier_features(coordinates)
    if blackout_masks and 'coordinates' in blackout_masks:
      coord_blackout_mask = blackout_masks['coordinates'].unsqueeze(-1)  # [batch_size, seq_len, 1]
      coord_features = coord_features * (~coord_blackout_mask).float()
    token_embeddings += coord_features

    # Add time delta embeddings with blackout support
    time_bucket_indices = self._log_bucket_time_delta(time_deltas)
    time_embeddings = self.time_delta_embedding(time_bucket_indices)
    if blackout_masks and 'time_deltas' in blackout_masks:
      time_blackout_mask = blackout_masks['time_deltas'].unsqueeze(-1)  # [batch_size, seq_len, 1]
      time_embeddings = time_embeddings * (~time_blackout_mask).float()
    token_embeddings += time_embeddings

    # Add CLS token
    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    embeddings = torch.cat([cls_tokens, token_embeddings], dim=1)  # [batch_size, seq_len+1, d_model]

    # Add positional encoding (if not using RoPE)
    if not (self.use_rope_time or self.use_rope_2d):
      embeddings += self.pos_embedding[:, :seq_len+1]

    embeddings = self.dropout(embeddings)

    # Extend mask for CLS token
    if mask is not None:
      cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
      extended_mask = torch.cat([cls_mask, mask], dim=1)
    else:
      extended_mask = None

    # Apply transformer layers
    for layer in self.layers:
      embeddings = layer(embeddings, extended_mask)

    embeddings = self.ln_f(embeddings)

    # Pooling: 0.5 * CLS + 0.5 * mean of valid tokens
    cls_embedding = embeddings[:, 0]  # [batch_size, d_model]

    if mask is not None:
      # Mask out invalid tokens for mean pooling
      token_embeddings = embeddings[:, 1:]  # [batch_size, seq_len, d_model]
      mask_expanded = mask.unsqueeze(-1).expand_as(token_embeddings)
      masked_embeddings = token_embeddings * mask_expanded
      mean_embedding = masked_embeddings.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
    else:
      mean_embedding = embeddings[:, 1:].mean(dim=1)

    # Combine CLS and mean
    pooled = 0.5 * cls_embedding + 0.5 * mean_embedding

    # Final projection and L2 normalization
    pooled = self.pool_proj(pooled)
    pooled = F.normalize(pooled, p=2, dim=-1)

    return pooled

  def forward_clip(
    self,
    categorical_features: Dict[str, torch.Tensor],
    coordinates: torch.Tensor,
    time_deltas: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    blackout_masks: Optional[Dict[str, torch.Tensor]] = None
  ) -> torch.Tensor:
    """
    Forward pass for CLIP alignment with projection head.

    Args:
      categorical_features: Dict of [batch_size, seq_len] tensors
      coordinates: [batch_size, seq_len, 2] normalized (x,y) coordinates
      time_deltas: [batch_size, seq_len] time since previous event (seconds)
      mask: [batch_size, seq_len] boolean mask (True = valid)

    Returns:
      embeddings: [batch_size, 512] L2-normalized projected embeddings
    """
    # Get base embeddings (768-dim, L2-normalized)
    base_embeddings = self.forward(categorical_features, coordinates, time_deltas, mask, blackout_masks)

    # Apply projection head
    projected = self.clip_proj(base_embeddings)

    # L2 normalize projected embeddings
    projected = F.normalize(projected, p=2, dim=-1)

    return projected

  def get_sequence_representations(
    self,
    categorical_features: Dict[str, torch.Tensor],
    coordinates: torch.Tensor,
    time_deltas: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    blackout_masks: Optional[Dict[str, torch.Tensor]] = None
  ) -> torch.Tensor:
    """
    Get sequence-level hidden states before pooling (for MLM).

    Args:
      categorical_features: Dict of [batch_size, seq_len] tensors
      coordinates: [batch_size, seq_len, 2] normalized (x,y) coordinates
      time_deltas: [batch_size, seq_len] time since previous event (seconds)
      mask: [batch_size, seq_len] boolean mask (True = valid)

    Returns:
      hidden_states: [batch_size, seq_len, d_model] sequence representations
    """
    batch_size, seq_len = coordinates.shape[:2]

    # Sum categorical embeddings with blackout support
    token_embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=coordinates.device)
    for field, indices in categorical_features.items():
      if field in self.embeddings:
        field_embeddings = self.embeddings[field](indices)
        # Apply blackout mask if provided
        if blackout_masks and field in blackout_masks:
          blackout_mask = blackout_masks[field].unsqueeze(-1)  # [batch_size, seq_len, 1]
          field_embeddings = field_embeddings * (~blackout_mask).float()
        token_embeddings += field_embeddings

    # Add Fourier features for coordinates with blackout support
    coord_features = self.fourier_features(coordinates)
    if blackout_masks and 'coordinates' in blackout_masks:
      coord_blackout_mask = blackout_masks['coordinates'].unsqueeze(-1)  # [batch_size, seq_len, 1]
      coord_features = coord_features * (~coord_blackout_mask).float()
    token_embeddings += coord_features

    # Add time delta embeddings with blackout support
    time_bucket_indices = self._log_bucket_time_delta(time_deltas)
    time_embeddings = self.time_delta_embedding(time_bucket_indices)
    if blackout_masks and 'time_deltas' in blackout_masks:
      time_blackout_mask = blackout_masks['time_deltas'].unsqueeze(-1)  # [batch_size, seq_len, 1]
      time_embeddings = time_embeddings * (~time_blackout_mask).float()
    token_embeddings += time_embeddings

    # Add CLS token
    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    embeddings = torch.cat([cls_tokens, token_embeddings], dim=1)  # [batch_size, seq_len+1, d_model]

    # Add positional encoding (if not using RoPE)
    if not (self.use_rope_time or self.use_rope_2d):
      embeddings += self.pos_embedding[:, :seq_len+1]

    embeddings = self.dropout(embeddings)

    # Extend mask for CLS token
    if mask is not None:
      cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
      extended_mask = torch.cat([cls_mask, mask], dim=1)
    else:
      extended_mask = None

    # Apply transformer layers
    for layer in self.layers:
      embeddings = layer(embeddings, extended_mask)

    embeddings = self.ln_f(embeddings)

    # Return only the token representations (exclude CLS token)
    return embeddings[:, 1:]  # [batch_size, seq_len, d_model]
