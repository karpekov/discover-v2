#!/usr/bin/env python3
"""
Weights & Biases configuration for smart-home training experiments.
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path


class WandBConfig:
    """Configuration manager for Weights & Biases integration."""

    def __init__(self, project_name: str = "discover-v2", entity: Optional[str] = None):
        self.project_name = project_name
        self.entity = entity

    def get_config(self, experiment_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get W&B configuration for a specific experiment."""

        # Base W&B settings
        wandb_config = {
            'use_wandb': True,
            'wandb_project': self.project_name,
            'wandb_entity': self.entity,
            'wandb_name': experiment_name,
            'wandb_tags': self._get_tags(config),
            'wandb_notes': self._get_notes(config),
            'wandb_group': self._get_group(config),
        }

        # Add W&B-specific logging settings
        wandb_config.update({
            'wandb_log_model': False,  # Don't upload model files, only metadata
            'wandb_log_gradients': False,  # Log gradient histograms (expensive)
            'wandb_log_parameters': True,  # Log parameter histograms
            'wandb_watch_freq': 100,  # How often to log gradients/parameters
            'wandb_log_graph': True,  # Log model graph
        })

        return wandb_config

    def _get_tags(self, config: Dict[str, Any]) -> list:
        """Generate tags based on configuration."""
        tags = []

        # Dataset tags
        if 'train_data_path' in config:
            data_path = Path(config['train_data_path'])
            if 'milan' in str(data_path).lower():
                tags.append('milan')
            elif 'aruba' in str(data_path).lower():
                tags.append('aruba')
            elif 'cairo' in str(data_path).lower():
                tags.append('cairo')

        # Model architecture tags
        tags.extend([
            f"d_model_{config.get('d_model', 768)}",
            f"layers_{config.get('n_layers', 6)}",
            f"heads_{config.get('n_heads', 8)}"
        ])

        # Training strategy tags
        if config.get('mlm_weight', 0) > 0:
            tags.append('mlm')
        if config.get('clip_weight', 0) > 0:
            tags.append('clip')

        # Loss configuration
        mlm_weight = config.get('mlm_weight', 1.0)
        clip_weight = config.get('clip_weight', 1.0)
        if mlm_weight == clip_weight:
            tags.append("balanced-loss")
        else:
            tags.append(f"mlm_{mlm_weight}_clip_{clip_weight}")

        return tags

    def _get_notes(self, config: Dict[str, Any]) -> str:
        """Generate experiment notes."""
        notes_parts = []

        # Model info
        notes_parts.append(f"Dual-encoder with {config.get('d_model', 768)}d embeddings")

        # Loss info
        mlm_weight = config.get('mlm_weight', 1.0)
        clip_weight = config.get('clip_weight', 1.0)
        if mlm_weight == clip_weight:
            notes_parts.append(f"Loss: Balanced {mlm_weight:.1f}×MLM + {clip_weight:.1f}×CLIP")
        else:
            notes_parts.append(f"Loss: {mlm_weight:.1f}×MLM + {clip_weight:.1f}×CLIP")

        # Training info
        batch_size = config.get('batch_size', 32)
        lr = config.get('learning_rate', 2e-4)
        notes_parts.append(f"Training: BS={batch_size}, LR={lr}")

        return " | ".join(notes_parts)

    def _get_group(self, config: Dict[str, Any]) -> str:
        """Generate experiment group name."""
        # Group by dataset and major architecture choices
        group_parts = []

        # Dataset
        if 'train_data_path' in config:
            data_path = Path(config['train_data_path'])
            if 'milan' in str(data_path).lower():
                group_parts.append('milan')
            elif 'aruba' in str(data_path).lower():
                group_parts.append('aruba')
            elif 'cairo' in str(data_path).lower():
                group_parts.append('cairo')
            else:
                group_parts.append('unknown_dataset')

        # Model size
        d_model = config.get('d_model', 768)
        if d_model <= 512:
            group_parts.append('small')
        elif d_model <= 768:
            group_parts.append('base')
        else:
            group_parts.append('large')

        return '_'.join(group_parts)


# Predefined experiment configurations
EXPERIMENT_CONFIGS = {
    'milan_baseline': {
        'experiment_name': 'milan_baseline_v1',
        'description': 'Baseline dual-encoder training on Milan dataset',
        'tags': ['milan', 'baseline', 'dual-encoder'],
    },

    'milan_ablation_mlm': {
        'experiment_name': 'milan_ablation_mlm_v1',
        'description': 'Ablation study: MLM weight variations',
        'tags': ['milan', 'ablation', 'mlm'],
    },

    'milan_ablation_temperature': {
        'experiment_name': 'milan_ablation_temp_v1',
        'description': 'Ablation study: Temperature initialization',
        'tags': ['milan', 'ablation', 'temperature'],
    },

    'multi_dataset': {
        'experiment_name': 'multi_dataset_v1',
        'description': 'Training across multiple CASAS datasets',
        'tags': ['multi-dataset', 'generalization'],
    }
}


def get_wandb_config_for_experiment(experiment_key: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Get W&B configuration for a predefined experiment."""
    if experiment_key not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment_key}. Available: {list(EXPERIMENT_CONFIGS.keys())}")

    exp_config = EXPERIMENT_CONFIGS[experiment_key]
    wandb_manager = WandBConfig()

    # Get base W&B config
    wandb_config = wandb_manager.get_config(exp_config['experiment_name'], base_config)

    # Override with experiment-specific settings
    wandb_config.update({
        'wandb_name': exp_config['experiment_name'],
        'wandb_notes': exp_config['description'],
        'wandb_tags': exp_config['tags'] + wandb_config.get('wandb_tags', []),
    })

    return wandb_config


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    example_config = {
        'train_data_path': 'data/milan_training/train.json',
        'd_model': 768,
        'n_layers': 6,
        'n_heads': 8,
        'batch_size': 128,
        'learning_rate': 3e-4,
        'mlm_weight': 0.5,
        'clip_weight': 1.0,
    }

    # Test configuration generation
    wandb_manager = WandBConfig()
    wandb_config = wandb_manager.get_config('test_experiment', example_config)

    print("Generated W&B Configuration:")
    for key, value in wandb_config.items():
        print(f"  {key}: {value}")

    print("\nPredefined experiment configs:")
    for key, config in EXPERIMENT_CONFIGS.items():
        print(f"  {key}: {config['description']}")
