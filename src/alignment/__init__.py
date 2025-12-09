"""
Alignment module for sensor-text alignment training.

This module provides components for aligning sensor encoder outputs
with text embeddings using contrastive learning (CLIP loss).
"""

from alignment.config import AlignmentConfig
from alignment.model import AlignmentModel
from alignment.trainer import AlignmentTrainer
from alignment.wandb_utils import (
    generate_wandb_run_name,
    generate_wandb_group,
    generate_wandb_tags
)

__all__ = [
    'AlignmentConfig',
    'AlignmentModel',
    'AlignmentTrainer',
    'generate_wandb_run_name',
    'generate_wandb_group',
    'generate_wandb_tags',
]

