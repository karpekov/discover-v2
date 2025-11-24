import torch
import torch.nn as nn
from typing import Dict, Optional


class MLMHeads(nn.Module):
  """
  Multi-field MLM heads for masked language modeling on sensor events.
  Each field has its own classifier head.
  """

  def __init__(self, d_model: int, vocab_sizes: Dict[str, int], dropout: float = 0.1):
    super().__init__()
    self.d_model = d_model
    self.vocab_sizes = vocab_sizes

    # Create a classifier head for each field
    self.heads = nn.ModuleDict()
    for field, vocab_size in vocab_sizes.items():
      self.heads[field] = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.LayerNorm(d_model),
        nn.Linear(d_model, vocab_size)
      )

  def forward(
    self,
    hidden_states: torch.Tensor,
    mask_positions: Optional[Dict[str, torch.Tensor]] = None
  ) -> Dict[str, torch.Tensor]:
    """
    Args:
      hidden_states: [batch_size, seq_len, d_model] transformer outputs
      mask_positions: Dict of [batch_size, seq_len] boolean masks for each field
                     (True = position is masked and should be predicted)

    Returns:
      predictions: Dict of [batch_size, seq_len, vocab_size] logits for each field
    """
    predictions = {}

    for field, head in self.heads.items():
      logits = head(hidden_states)  # [batch_size, seq_len, vocab_size]
      predictions[field] = logits

    return predictions

  def compute_loss(
    self,
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    mask_positions: Dict[str, torch.Tensor]
  ) -> torch.Tensor:
    """
    Compute cross-entropy loss only on masked positions.

    Args:
      predictions: Dict of [batch_size, seq_len, vocab_size] logits
      targets: Dict of [batch_size, seq_len] target indices
      mask_positions: Dict of [batch_size, seq_len] boolean masks

    Returns:
      loss: Scalar loss tensor
    """
    total_loss = 0.0
    total_masked_tokens = 0

    for field in predictions.keys():
      if field not in targets or field not in mask_positions:
        continue

      pred_logits = predictions[field]  # [batch_size, seq_len, vocab_size]
      target_ids = targets[field]  # [batch_size, seq_len]
      field_mask = mask_positions[field]  # [batch_size, seq_len]

      # Only compute loss on masked positions
      if field_mask.sum() == 0:
        continue

      # Flatten and select masked positions
      pred_flat = pred_logits.view(-1, pred_logits.size(-1))  # [batch_size * seq_len, vocab_size]
      target_flat = target_ids.view(-1)  # [batch_size * seq_len]
      mask_flat = field_mask.view(-1)  # [batch_size * seq_len]

      # Select only masked positions
      masked_pred = pred_flat[mask_flat]  # [num_masked, vocab_size]
      masked_target = target_flat[mask_flat]  # [num_masked]

      if masked_pred.size(0) > 0:
        field_loss = nn.functional.cross_entropy(masked_pred, masked_target, reduction='sum')
        total_loss += field_loss
        total_masked_tokens += masked_pred.size(0)

    if total_masked_tokens > 0:
      return total_loss / total_masked_tokens
    else:
      return torch.tensor(0.0, device=predictions[next(iter(predictions))].device)


class SpanMasker(nn.Module):
  """
  Enhanced span masking for MLM with field-balanced priors, BERT-style masking, correlated field masking,
  field blackout, transition-span masking, strict correlated masking, and adaptive masking.
  """

  def __init__(
    self,
    mask_prob: float = 0.25,
    field_priors: Optional[Dict[str, float]] = None,
    bert_prob_mask: float = 0.8,
    bert_prob_random: float = 0.1,
    bert_prob_keep: float = 0.1,
    mean_span_length: float = 3.0,
    correlated_field_prob: float = 0.5,
    # New enhanced masking parameters
    enable_field_blackout: bool = True,
    p_transition_seed: float = 0.3,
    strict_corr_mask: bool = True,
    adaptive_mask: bool = True,
    mask_acc_threshold: float = 0.97,
    adaptive_window_steps: int = 100
  ):
    super().__init__()
    self.mask_prob = mask_prob
    self.bert_prob_mask = bert_prob_mask
    self.bert_prob_random = bert_prob_random
    self.bert_prob_keep = bert_prob_keep
    self.mean_span_length = mean_span_length
    self.correlated_field_prob = correlated_field_prob

    # Enhanced masking parameters
    self.enable_field_blackout = enable_field_blackout
    self.p_transition_seed = p_transition_seed
    self.strict_corr_mask = strict_corr_mask
    self.adaptive_mask = adaptive_mask
    self.mask_acc_threshold = mask_acc_threshold
    self.adaptive_window_steps = adaptive_window_steps

    # Default field-balanced masking priors
    if field_priors is None:
      field_priors = {
        'room_id': 0.30,
        'event_type': 0.20,
        'sensor_id': 0.20,
        'tod_bucket': 0.15,
        'delta_t_bucket': 0.10,
        'sensor_type': 0.05,
        'floor_id': 0.05,  # Optional
        'dow': 0.05        # Optional
      }

    self.field_priors = field_priors

    # Normalize priors
    total_prior = sum(field_priors.values())
    self.field_priors = {k: v / total_prior for k, v in field_priors.items()}

    # Define correlated field groups
    self.correlated_groups = [
      ['room_id', 'sensor_id', 'sensor_type'],  # Location-related fields
      ['tod_bucket', 'delta_t_bucket'],         # Time-related fields
    ]

    # Field blackout mapping - which embeddings to zero when field is masked
    self.blackout_mapping = {
      'room_id': ['sensor_id', 'sensor_type', 'coordinates'],
      'sensor_id': ['room_id', 'sensor_type', 'coordinates'],
      'sensor_type': ['room_id', 'sensor_id', 'coordinates'],
      'tod_bucket': ['delta_t_bucket', 'time_deltas'],
      'delta_t_bucket': ['tod_bucket', 'time_deltas']
    }

    # Adaptive masking state - track per-field accuracy
    self.field_accuracies = {}  # field -> list of recent accuracies
    self.adaptive_multipliers = {}  # field -> current mask rate multiplier
    self.step_count = 0

  def _sample_span_lengths(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Sample span lengths from Poisson distribution."""
    # Use exponential distribution to approximate Poisson
    exp_dist = torch.distributions.Exponential(1.0 / self.mean_span_length)
    spans = exp_dist.sample((batch_size * seq_len,)).to(device)
    spans = torch.clamp(spans.round().long(), min=1, max=seq_len)
    return spans.view(batch_size, seq_len)

  def _apply_correlated_masking(self, field_masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Apply correlated masking to related fields."""
    batch_size, seq_len = next(iter(field_masks.values())).shape
    device = next(iter(field_masks.values())).device

    # Create a copy of field masks to modify
    correlated_masks = {field: mask.clone() for field, mask in field_masks.items()}

    for group in self.correlated_groups:
      # Find fields in this group that exist in the data
      available_fields = [field for field in group if field in field_masks]
      if len(available_fields) < 2:
        continue

      # For each position, decide whether to apply correlated masking
      correlation_decisions = torch.rand(batch_size, seq_len, device=device) < self.correlated_field_prob

      for field in available_fields:
        # If any field in the group is masked at a position, apply correlation
        group_mask_union = torch.zeros_like(field_masks[field])
        for other_field in available_fields:
          group_mask_union |= field_masks[other_field]

        # Apply correlated masking: if correlation is triggered and any field in group is masked,
        # mask this field too at those positions
        correlated_positions = correlation_decisions & group_mask_union
        correlated_masks[field] |= correlated_positions

    return correlated_masks

  def _detect_transitions(self, activity_labels: Optional[torch.Tensor], room_labels: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Detect activity and room transitions for transition-span masking.

    Args:
      activity_labels: [batch_size, seq_len] activity labels
      room_labels: [batch_size, seq_len] room labels
      device: torch device

    Returns:
      transition_mask: [batch_size, seq_len] boolean mask (True = transition position)
    """
    if activity_labels is None and room_labels is None:
      return torch.zeros(1, 1, device=device, dtype=torch.bool)

    batch_size, seq_len = (activity_labels if activity_labels is not None else room_labels).shape
    transition_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)

    # Detect activity transitions
    if activity_labels is not None:
      activity_transitions = torch.zeros_like(activity_labels, dtype=torch.bool)
      activity_transitions[:, 1:] = activity_labels[:, 1:] != activity_labels[:, :-1]
      transition_mask |= activity_transitions

    # Detect room transitions
    if room_labels is not None:
      room_transitions = torch.zeros_like(room_labels, dtype=torch.bool)
      room_transitions[:, 1:] = room_labels[:, 1:] != room_labels[:, :-1]
      transition_mask |= room_transitions

    return transition_mask

  def _apply_strict_correlated_masking(self, field_masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Apply strict correlated masking - within a span, mask ALL fields in correlated groups.

    Args:
      field_masks: Dict of [batch_size, seq_len] boolean masks

    Returns:
      strict_masks: Dict of [batch_size, seq_len] boolean masks with strict correlation
    """
    batch_size, seq_len = next(iter(field_masks.values())).shape
    strict_masks = {field: mask.clone() for field, mask in field_masks.items()}

    for group in self.correlated_groups:
      # Find fields in this group that exist in the data
      available_fields = [field for field in group if field in field_masks]
      if len(available_fields) < 2:
        continue

      # Create union mask for the group
      group_union_mask = torch.zeros(batch_size, seq_len, device=next(iter(field_masks.values())).device, dtype=torch.bool)
      for field in available_fields:
        group_union_mask |= field_masks[field]

      # Apply union mask to all fields in the group
      for field in available_fields:
        strict_masks[field] = group_union_mask

    return strict_masks

  def _compute_blackout_masks(self, field_masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Compute blackout masks for embedding zeroing based on field masking.

    Args:
      field_masks: Dict of [batch_size, seq_len] boolean masks (True = masked)

    Returns:
      blackout_masks: Dict mapping embedding types to [batch_size, seq_len] boolean masks
    """
    batch_size, seq_len = next(iter(field_masks.values())).shape
    device = next(iter(field_masks.values())).device

    blackout_masks = {
      'coordinates': torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool),
      'time_deltas': torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
    }

    # Add categorical field blackouts
    for field in field_masks.keys():
      blackout_masks[field] = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)

    # Apply blackout mapping
    for masked_field, mask in field_masks.items():
      if masked_field in self.blackout_mapping:
        for target_field in self.blackout_mapping[masked_field]:
          if target_field in blackout_masks:
            blackout_masks[target_field] |= mask

    return blackout_masks

  def update_adaptive_masks(self, field_accuracies: Dict[str, float]):
    """
    Update adaptive masking multipliers based on recent field accuracies.

    Args:
      field_accuracies: Dict mapping field names to current MLM accuracies
    """
    if not self.adaptive_mask:
      return

    self.step_count += 1

    for field, accuracy in field_accuracies.items():
      # Initialize tracking for new fields
      if field not in self.field_accuracies:
        self.field_accuracies[field] = []
        self.adaptive_multipliers[field] = 1.0

      # Add current accuracy
      self.field_accuracies[field].append(accuracy)

      # Keep only recent accuracies (sliding window)
      if len(self.field_accuracies[field]) > self.adaptive_window_steps:
        self.field_accuracies[field] = self.field_accuracies[field][-self.adaptive_window_steps:]

      # Update multiplier if we have enough data
      if len(self.field_accuracies[field]) >= min(10, self.adaptive_window_steps // 2):
        recent_avg_acc = sum(self.field_accuracies[field][-10:]) / len(self.field_accuracies[field][-10:])

        if recent_avg_acc > self.mask_acc_threshold:
          # Accuracy too high - increase masking difficulty
          self.adaptive_multipliers[field] = min(2.0, self.adaptive_multipliers[field] * 1.1)
        elif recent_avg_acc < self.mask_acc_threshold - 0.05:  # 5% buffer
          # Accuracy too low - decrease masking difficulty
          self.adaptive_multipliers[field] = max(0.5, self.adaptive_multipliers[field] * 0.95)

  def forward(
    self,
    categorical_features: Dict[str, torch.Tensor],
    vocab_sizes: Dict[str, int],
    activity_labels: Optional[torch.Tensor] = None,
    room_labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None
  ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Apply enhanced span masking to categorical features.

    Args:
      categorical_features: Dict of [batch_size, seq_len] input features
      vocab_sizes: Dict mapping field names to vocabulary sizes
      activity_labels: Optional [batch_size, seq_len] activity labels for transition detection
      room_labels: Optional [batch_size, seq_len] room labels for transition detection
      attention_mask: Optional [batch_size, seq_len] attention mask (True = valid, False = padding)

    Returns:
      masked_features: Dict of [batch_size, seq_len] masked input features
      original_features: Dict of [batch_size, seq_len] original features (for targets)
      mask_positions: Dict of [batch_size, seq_len] boolean masks (True = masked)
      blackout_masks: Dict of [batch_size, seq_len] boolean masks for embedding blackout
    """
    batch_size, seq_len = next(iter(categorical_features.values())).shape
    device = next(iter(categorical_features.values())).device

    # Create attention mask if not provided (assume all positions are valid)
    if attention_mask is None:
      attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)

    # 1. Transition-span masking: seed masks at activity/room transitions
    transition_mask = self._detect_transitions(activity_labels, room_labels, device) if (activity_labels is not None or room_labels is not None) else torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)

    # Sample which positions to mask (base probability + transition seeding)
    # Only consider valid (non-padded) positions
    base_mask_decisions = (torch.rand(batch_size, seq_len, device=device) < self.mask_prob) & attention_mask
    transition_seed_decisions = (torch.rand(batch_size, seq_len, device=device) < self.p_transition_seed) & attention_mask
    mask_decisions = base_mask_decisions | (transition_mask & transition_seed_decisions)

    # Sample span lengths
    span_lengths = self._sample_span_lengths(batch_size, seq_len, device)

    # Apply span masking (only to valid positions)
    final_mask = torch.zeros_like(mask_decisions)
    for b in range(batch_size):
      for s in range(seq_len):
        if mask_decisions[b, s]:
          span_len = min(span_lengths[b, s].item(), seq_len - s)
          # Only mask positions within attention mask
          for i in range(span_len):
            if s + i < seq_len and attention_mask[b, s + i]:
              final_mask[b, s + i] = True

    # 2. Compute independent field masks with adaptive multipliers
    independent_field_masks = {}
    for field, features in categorical_features.items():
      if field not in self.field_priors:
        # Don't mask fields not in priors
        independent_field_masks[field] = torch.zeros_like(features, dtype=torch.bool)
        continue

      # Apply field-specific masking probability with adaptive adjustment
      base_field_prior = self.field_priors[field]
      adaptive_multiplier = self.adaptive_multipliers.get(field, 1.0) if self.adaptive_mask else 1.0
      adjusted_prior = min(0.8, base_field_prior * adaptive_multiplier)  # Cap at 80%

      field_mask_prob = torch.rand_like(features.float()) < adjusted_prior
      field_mask = final_mask & field_mask_prob
      independent_field_masks[field] = field_mask

    # 3. Apply correlated masking (strict or probabilistic)
    if self.strict_corr_mask:
      correlated_field_masks = self._apply_strict_correlated_masking(independent_field_masks)
    else:
      correlated_field_masks = self._apply_correlated_masking(independent_field_masks)

    # Now apply BERT-style masking with the correlated masks
    masked_features = {}
    original_features = {}
    mask_positions = {}

    for field, features in categorical_features.items():
      field_mask = correlated_field_masks.get(field, torch.zeros_like(features, dtype=torch.bool))

      # BERT-style masking: 80% [MASK], 10% random, 10% keep
      bert_decisions = torch.rand_like(features.float())

      masked_input = features.clone()
      vocab_size = vocab_sizes.get(field, features.max().item() + 1)

      # 80% probability: replace with mask token (use vocab_size-1 as mask token)
      # Note: vocab_size includes +1 for the mask token, so valid indices are 0 to vocab_size-1
      mask_token_positions = field_mask & (bert_decisions < self.bert_prob_mask)
      masked_input[mask_token_positions] = vocab_size - 1  # Mask token ID (last valid index)

      # 10% probability: replace with random token (excluding mask token)
      random_positions = field_mask & (bert_decisions >= self.bert_prob_mask) & (bert_decisions < self.bert_prob_mask + self.bert_prob_random)
      if random_positions.sum() > 0:
        # Generate random tokens from original vocabulary (0 to vocab_size-2), excluding mask token
        random_tokens = torch.randint(0, vocab_size - 1, size=(random_positions.sum().item(),), device=device)
        masked_input[random_positions] = random_tokens

      # 10% probability: keep original (no change needed)

      masked_features[field] = masked_input
      original_features[field] = features.clone()
      mask_positions[field] = field_mask

    # 4. Compute blackout masks for embedding zeroing
    blackout_masks = self._compute_blackout_masks(correlated_field_masks) if self.enable_field_blackout else {}

    return masked_features, original_features, mask_positions, blackout_masks
