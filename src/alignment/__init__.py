"""
Alignment module for sensor-text alignment training.

This module provides components for aligning sensor encoder outputs
with text embeddings using contrastive learning (CLIP loss).
"""

from src.alignment.config import AlignmentConfig
from src.alignment.model import AlignmentModel
from src.alignment.trainer import AlignmentTrainer
from src.alignment.wandb_utils import (
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

