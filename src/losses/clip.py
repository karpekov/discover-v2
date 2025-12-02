import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Callable
import random

from .alignment_loss import build_alignment_loss


class HardNegativeSampler:
    """
    Hard negative sampler for improved contrastive learning.

    Maintains memory banks and implements various hard negative sampling strategies.
    """

    def __init__(
        self,
        memory_size: int = 4096,
        hard_negative_ratio: float = 0.5,
        sampling_strategy: str = "mixed",  # "memory_bank", "cross_batch", "mixed"
        temperature_for_sampling: float = 0.1
    ):
        self.memory_size = memory_size
        self.hard_negative_ratio = hard_negative_ratio
        self.sampling_strategy = sampling_strategy
        self.temperature_for_sampling = temperature_for_sampling

        # Memory banks for embeddings
        self.sensor_memory = None
        self.text_memory = None
        self.memory_ptr = 0
        self.memory_filled = False

    def update_memory(self, sensor_embeddings: torch.Tensor, text_embeddings: torch.Tensor):
        """Update memory banks with new embeddings."""
        batch_size = sensor_embeddings.size(0)
        embed_dim = sensor_embeddings.size(1)
        device = sensor_embeddings.device

        # Initialize memory banks if needed
        if self.sensor_memory is None:
            self.sensor_memory = torch.randn(self.memory_size, embed_dim, device=device)
            self.text_memory = torch.randn(self.memory_size, embed_dim, device=device)

        # Update memory with current batch
        end_ptr = self.memory_ptr + batch_size
        if end_ptr <= self.memory_size:
            # Normal case: batch fits in remaining memory
            self.sensor_memory[self.memory_ptr:end_ptr] = sensor_embeddings.detach()
            self.text_memory[self.memory_ptr:end_ptr] = text_embeddings.detach()
            self.memory_ptr = end_ptr
        else:
            # Wrap around case
            remaining = self.memory_size - self.memory_ptr
            self.sensor_memory[self.memory_ptr:] = sensor_embeddings[:remaining].detach()
            self.text_memory[self.memory_ptr:] = text_embeddings[:remaining].detach()

            overflow = batch_size - remaining
            if overflow > 0:
                self.sensor_memory[:overflow] = sensor_embeddings[remaining:].detach()
                self.text_memory[:overflow] = text_embeddings[remaining:].detach()

            self.memory_ptr = overflow
            self.memory_filled = True

    def sample_hard_negatives(
        self,
        sensor_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_hard_negatives: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hard negatives for the current batch.

        Returns:
            hard_sensor_negatives: [num_hard_negatives, embed_dim]
            hard_text_negatives: [num_hard_negatives, embed_dim]
        """
        if not self.memory_filled and self.memory_ptr < num_hard_negatives:
            # Not enough data in memory yet, return random samples from current batch
            batch_size = sensor_embeddings.size(0)
            if batch_size >= num_hard_negatives:
                indices = torch.randperm(batch_size)[:num_hard_negatives]
                return sensor_embeddings[indices], text_embeddings[indices]
            else:
                # Repeat samples if batch too small
                indices = torch.randint(0, batch_size, (num_hard_negatives,))
                return sensor_embeddings[indices], text_embeddings[indices]

        available_size = self.memory_size if self.memory_filled else self.memory_ptr

        if self.sampling_strategy == "memory_bank":
            # Sample randomly from memory bank
            indices = torch.randperm(available_size)[:num_hard_negatives]
            return self.sensor_memory[indices], self.text_memory[indices]

        elif self.sampling_strategy == "cross_batch":
            # Sample hard negatives based on similarity to current batch
            return self._sample_similarity_based_negatives(
                sensor_embeddings, text_embeddings, num_hard_negatives, available_size
            )

        elif self.sampling_strategy == "mixed":
            # Mix of random and similarity-based sampling
            num_random = num_hard_negatives // 2
            num_hard = num_hard_negatives - num_random

            # Random negatives
            random_indices = torch.randperm(available_size)[:num_random]
            random_sensor = self.sensor_memory[random_indices]
            random_text = self.text_memory[random_indices]

            # Hard negatives
            hard_sensor, hard_text = self._sample_similarity_based_negatives(
                sensor_embeddings, text_embeddings, num_hard, available_size
            )

            # Combine
            combined_sensor = torch.cat([random_sensor, hard_sensor], dim=0)
            combined_text = torch.cat([random_text, hard_text], dim=0)

            return combined_sensor, combined_text

        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _sample_similarity_based_negatives(
        self,
        sensor_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_negatives: int,
        available_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample negatives based on similarity to current batch."""
        batch_size = sensor_embeddings.size(0)

        # Compute similarities between current batch and memory
        # For sensor embeddings
        sensor_similarities = torch.matmul(
            sensor_embeddings,
            self.sensor_memory[:available_size].t()
        )  # [batch_size, available_size]

        # For text embeddings
        text_similarities = torch.matmul(
            text_embeddings,
            self.text_memory[:available_size].t()
        )  # [batch_size, available_size]

        # Average similarities across batch
        avg_sensor_sim = sensor_similarities.mean(dim=0)  # [available_size]
        avg_text_sim = text_similarities.mean(dim=0)  # [available_size]

        # Sample based on similarity (higher similarity = more likely to be selected as hard negative)
        # Use temperature to control hardness
        sensor_probs = F.softmax(avg_sensor_sim / self.temperature_for_sampling, dim=0)
        text_probs = F.softmax(avg_text_sim / self.temperature_for_sampling, dim=0)

        # Sample indices
        sensor_indices = torch.multinomial(sensor_probs, num_negatives, replacement=True)
        text_indices = torch.multinomial(text_probs, num_negatives, replacement=True)

        return self.sensor_memory[sensor_indices], self.text_memory[text_indices]


class CLIPLoss(nn.Module):
  """
  Configurable contrastive alignment loss with learnable temperature.

  Supports multiple alignment loss types:
  - infonce: Bidirectional InfoNCE (CLIP-style)
  - sigmoid: Sigmoid BCE (SigLIP-style)
  - focal_sigmoid: Focal sigmoid BCE

  Uses in-batch negatives and optional hard negatives for contrastive learning.
  """

  def __init__(
    self,
    temperature_init: float = 0.05,
    learnable_temperature: bool = True,
    use_hard_negatives: bool = False,
    hard_negative_config: Optional[Dict[str, Any]] = None,
    alignment_loss_type: str = 'infonce',
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None
  ):
    super().__init__()

    if learnable_temperature:
      # Learnable temperature parameter initialized from temperature_init
      self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature_init)))
    else:
      # Fixed temperature
      self.register_buffer("log_temperature", torch.log(torch.tensor(temperature_init)))

    # Hard negative sampling
    self.use_hard_negatives = use_hard_negatives
    if use_hard_negatives:
      config = hard_negative_config or {}
      self.hard_negative_sampler = HardNegativeSampler(
        memory_size=config.get('memory_size', 4096),
        hard_negative_ratio=config.get('hard_negative_ratio', 0.5),
        sampling_strategy=config.get('sampling_strategy', 'mixed'),
        temperature_for_sampling=config.get('temperature_for_sampling', 0.1)
      )
    else:
      self.hard_negative_sampler = None

    # Build alignment loss function
    self.alignment_loss_fn = build_alignment_loss(
      loss_type=alignment_loss_type,
      focal_gamma=focal_gamma,
      focal_alpha=focal_alpha
    )
    self.alignment_loss_type = alignment_loss_type

  @property
  def temperature(self) -> torch.Tensor:
    """Get current temperature value."""
    return torch.exp(self.log_temperature)

  def forward(
    self,
    sensor_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    return_similarity_matrix: bool = False
  ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute alignment loss with optional hard negatives.

    Args:
      sensor_embeddings: [batch_size, d_model] L2-normalized sensor embeddings
      text_embeddings: [batch_size, d_model] L2-normalized text embeddings
      return_similarity_matrix: Whether to return the similarity matrix

    Returns:
      loss: Scalar loss tensor
      similarity_matrix: [batch_size, extended_size] similarity matrix (optional)
    """
    batch_size = sensor_embeddings.size(0)
    device = sensor_embeddings.device

    # Ensure embeddings are L2-normalized
    sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    # Handle hard negatives if enabled
    # Note: Hard negatives only work with InfoNCE loss type currently
    if self.use_hard_negatives and self.hard_negative_sampler is not None and self.alignment_loss_type == 'infonce':
      # Update memory bank
      self.hard_negative_sampler.update_memory(sensor_embeddings, text_embeddings)

      # Sample hard negatives
      num_hard_negatives = int(batch_size * self.hard_negative_sampler.hard_negative_ratio)
      if num_hard_negatives > 0:
        hard_sensor_negs, hard_text_negs = self.hard_negative_sampler.sample_hard_negatives(
          sensor_embeddings, text_embeddings, num_hard_negatives
        )

        # Extend embeddings with hard negatives
        extended_sensor_embeddings = torch.cat([sensor_embeddings, hard_sensor_negs], dim=0)
        extended_text_embeddings = torch.cat([text_embeddings, hard_text_negs], dim=0)

        # Compute extended similarity matrix
        similarity_matrix = torch.matmul(sensor_embeddings, extended_text_embeddings.t())
        similarity_matrix_t = torch.matmul(text_embeddings, extended_sensor_embeddings.t())

        # Scale by temperature
        logits = similarity_matrix / self.temperature
        logits_t = similarity_matrix_t / self.temperature

        # Labels: positive pairs are still at diagonal positions (first batch_size positions)
        labels = torch.arange(batch_size, device=device)

        # Compute losses with extended negatives (only InfoNCE supports this currently)
        sensor_to_text_loss = F.cross_entropy(logits, labels)
        text_to_sensor_loss = F.cross_entropy(logits_t, labels)
        total_loss = (sensor_to_text_loss + text_to_sensor_loss) / 2.0

      else:
        # Fallback to standard in-batch negatives
        similarity_matrix = torch.matmul(sensor_embeddings, text_embeddings.t())
        logits = similarity_matrix / self.temperature
        total_loss = self.alignment_loss_fn(logits)
    else:
      # Standard in-batch negatives only
      similarity_matrix = torch.matmul(sensor_embeddings, text_embeddings.t())

      # Scale by temperature
      logits = similarity_matrix / self.temperature

      # Apply alignment loss function
      total_loss = self.alignment_loss_fn(logits)

    if return_similarity_matrix:
      return total_loss, similarity_matrix
    else:
      return total_loss, None

  def get_accuracy(
    self,
    sensor_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    top_k: int = 1
  ) -> tuple[float, float]:
    """
    Compute top-k accuracy for both directions.

    Args:
      sensor_embeddings: [batch_size, d_model] L2-normalized sensor embeddings
      text_embeddings: [batch_size, d_model] L2-normalized text embeddings
      top_k: Number of top predictions to consider

    Returns:
      sensor_to_text_acc: Sensor-to-text top-k accuracy
      text_to_sensor_acc: Text-to-sensor top-k accuracy
    """
    batch_size = sensor_embeddings.size(0)
    device = sensor_embeddings.device

    # Ensure embeddings are L2-normalized
    sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(sensor_embeddings, text_embeddings.t())

    # True labels (diagonal)
    labels = torch.arange(batch_size, device=device)

    # Sensor-to-text accuracy
    _, sensor_to_text_pred = torch.topk(similarity_matrix, k=top_k, dim=1)
    sensor_to_text_correct = (sensor_to_text_pred == labels.unsqueeze(1)).any(dim=1)
    sensor_to_text_acc = sensor_to_text_correct.float().mean().item()

    # Text-to-sensor accuracy
    _, text_to_sensor_pred = torch.topk(similarity_matrix.t(), k=top_k, dim=1)
    text_to_sensor_correct = (text_to_sensor_pred == labels.unsqueeze(1)).any(dim=1)
    text_to_sensor_acc = text_to_sensor_correct.float().mean().item()

    return sensor_to_text_acc, text_to_sensor_acc


class CombinedLoss(nn.Module):
  """
  Combined alignment + MLM loss with configurable weighting and optional hard negatives.

  Supports multiple alignment loss types through the CLIPLoss module.
  """

  def __init__(
    self,
    mlm_weight: float = 0.3,
    clip_weight: float = 1.0,
    temperature_init: float = 0.05,
    learnable_temperature: bool = True,
    use_hard_negatives: bool = False,
    hard_negative_config: Optional[Dict[str, Any]] = None,
    alignment_loss_type: str = 'infonce',
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None
  ):
    super().__init__()
    self.mlm_weight = mlm_weight
    self.clip_weight = clip_weight

    self.clip_loss = CLIPLoss(
      temperature_init=temperature_init,
      learnable_temperature=learnable_temperature,
      use_hard_negatives=use_hard_negatives,
      hard_negative_config=hard_negative_config,
      alignment_loss_type=alignment_loss_type,
      focal_gamma=focal_gamma,
      focal_alpha=focal_alpha
    )

  def forward(
    self,
    sensor_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    mlm_loss: Optional[torch.Tensor] = None
  ) -> tuple[torch.Tensor, dict]:
    """
    Compute combined loss.

    Args:
      sensor_embeddings: [batch_size, d_model] sensor embeddings
      text_embeddings: [batch_size, d_model] text embeddings
      mlm_loss: Optional MLM loss tensor

    Returns:
      total_loss: Combined loss
      loss_dict: Dictionary with individual loss components
    """
    # Compute CLIP loss
    clip_loss, similarity_matrix = self.clip_loss(
      sensor_embeddings, text_embeddings, return_similarity_matrix=True
    )

    # Compute accuracies
    s2t_acc, t2s_acc = self.clip_loss.get_accuracy(sensor_embeddings, text_embeddings)

    # Total loss
    total_loss = self.clip_weight * clip_loss

    loss_dict = {
      'clip_loss': clip_loss.item(),
      'sensor_to_text_acc': s2t_acc,
      'text_to_sensor_acc': t2s_acc,
      'temperature': self.clip_loss.temperature.item()
    }

    # Add MLM loss if provided
    if mlm_loss is not None:
      total_loss += self.mlm_weight * mlm_loss
      loss_dict['mlm_loss'] = mlm_loss.item()

    loss_dict['total_loss'] = total_loss.item()

    return total_loss, loss_dict
