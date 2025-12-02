"""
Configurable alignment loss module with multiple loss types.

Supports:
- InfoNCE (CLIP-style bidirectional contrastive loss)
- Sigmoid BCE (SigLIP-style pairwise loss)
- Focal Sigmoid (focal BCE with optional class balancing)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable


def infonce_loss(sim_matrix: torch.Tensor) -> torch.Tensor:
    """
    InfoNCE loss (CLIP-style bidirectional contrastive loss).

    Treats alignment as a classification problem where each sample should
    match itself in the other modality (diagonal entries are positives).

    Args:
        sim_matrix: [B, B] similarity matrix (temperature-scaled)

    Returns:
        Scalar loss tensor
    """
    B = sim_matrix.size(0)
    targets = torch.arange(B, device=sim_matrix.device)

    # Row-wise: sensor → text
    loss_i = F.cross_entropy(sim_matrix, targets)

    # Column-wise: text → sensor
    loss_j = F.cross_entropy(sim_matrix.t(), targets)

    return 0.5 * (loss_i + loss_j)


def sigmoid_loss(sim_matrix: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid BCE loss (SigLIP-style pairwise loss).

    Treats each pair independently using binary cross-entropy.
    Positives are on the diagonal, negatives are off-diagonal.

    Args:
        sim_matrix: [B, B] similarity matrix (temperature-scaled logits)

    Returns:
        Scalar loss tensor
    """
    B = sim_matrix.size(0)
    # Labels: identity matrix (1 for positives on diagonal, 0 elsewhere)
    labels = torch.eye(B, device=sim_matrix.device)

    # Binary cross-entropy with logits
    loss = F.binary_cross_entropy_with_logits(sim_matrix, labels, reduction='mean')
    return loss


def focal_sigmoid_loss(
    sim_matrix: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = None
) -> torch.Tensor:
    """
    Focal sigmoid loss (pairwise focal BCE).

    Applies focal weighting to focus learning on hard examples.
    The focal weight is (1 - p_t)^gamma where p_t is the predicted probability
    of the correct class.

    Args:
        sim_matrix: [B, B] similarity matrix (temperature-scaled logits)
        gamma: Focal loss focusing parameter (higher = more focus on hard examples)
        alpha: Optional class-balancing weight for positives (None = no balancing)

    Returns:
        Scalar loss tensor
    """
    B = sim_matrix.size(0)
    logits = sim_matrix
    labels = torch.eye(B, device=logits.device)

    # Compute probabilities
    p = torch.sigmoid(logits)

    # Compute pt: p if y=1, (1-p) if y=0
    pt = labels * p + (1 - labels) * (1 - p)

    # Focal weight: (1 - pt)^gamma
    focal_weight = (1.0 - pt).pow(gamma)

    # Optional alpha-balancing between positives and negatives
    if alpha is not None:
        alpha_t = labels * alpha + (1 - labels) * (1 - alpha)
        focal_weight = focal_weight * alpha_t

    # Binary cross-entropy per element (no reduction)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

    # Apply focal weight and average
    loss = (focal_weight * bce).mean()
    return loss


def build_alignment_loss(
    loss_type: str = 'infonce',
    focal_gamma: float = 2.0,
    focal_alpha: Optional[float] = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Factory function to build alignment loss function.

    Args:
        loss_type: Type of alignment loss ('infonce', 'sigmoid', 'focal_sigmoid')
        focal_gamma: Gamma parameter for focal loss (only used if loss_type='focal_sigmoid')
        focal_alpha: Alpha parameter for focal loss (only used if loss_type='focal_sigmoid')

    Returns:
        Loss function that takes a similarity matrix and returns a scalar loss

    Examples:
        >>> loss_fn = build_alignment_loss('infonce')
        >>> sim_matrix = torch.randn(32, 32)  # [batch_size, batch_size]
        >>> loss = loss_fn(sim_matrix)

        >>> loss_fn = build_alignment_loss('focal_sigmoid', focal_gamma=2.0)
        >>> loss = loss_fn(sim_matrix)
    """
    loss_type = loss_type.lower()

    if loss_type == 'infonce':
        def fn(sim_matrix: torch.Tensor) -> torch.Tensor:
            return infonce_loss(sim_matrix)

    elif loss_type == 'sigmoid':
        def fn(sim_matrix: torch.Tensor) -> torch.Tensor:
            return sigmoid_loss(sim_matrix)

    elif loss_type == 'focal_sigmoid':
        def fn(sim_matrix: torch.Tensor) -> torch.Tensor:
            return focal_sigmoid_loss(sim_matrix, gamma=focal_gamma, alpha=focal_alpha)

    else:
        raise ValueError(
            f"Unknown alignment_loss_type: '{loss_type}'. "
            f"Supported types: 'infonce', 'sigmoid', 'focal_sigmoid'"
        )

    return fn

