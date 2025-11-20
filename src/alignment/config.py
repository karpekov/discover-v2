"""
Configuration classes for alignment training.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
from pathlib import Path
import yaml


@dataclass
class ProjectionConfig:
    """Configuration for projection heads."""
    type: Literal['linear', 'mlp'] = 'linear'
    dim: int = 512
    hidden_dim: int = 2048  # For MLP only
    num_layers: int = 2  # For MLP only (2 or 3)
    dropout: float = 0.1  # For MLP only
    use_bn: bool = False  # For MLP only


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    # CLIP loss settings
    clip_weight: float = 1.0
    temperature_init: float = 0.02
    learnable_temperature: bool = True

    # MLM loss settings (optional)
    mlm_weight: float = 0.0
    mask_prob: float = 0.25
    mean_span_length: float = 5.0
    enable_field_blackout: bool = True
    p_transition_seed: float = 0.3
    strict_corr_mask: bool = True

    # Hard negative sampling (optional)
    use_hard_negatives: bool = False
    hard_negative_memory_size: int = 4096
    hard_negative_ratio: float = 0.5
    hard_negative_strategy: Literal['memory_bank', 'cross_batch', 'mixed'] = 'mixed'
    hard_negative_sampling_temperature: float = 0.1


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    type: Literal['adam', 'adamw', 'sgd'] = 'adamw'
    learning_rate: float = 3e-4
    betas: tuple = (0.9, 0.98)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip_norm: Optional[float] = 1.0


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 128
    max_epochs: Optional[int] = None
    max_steps: Optional[int] = 10000

    # Device settings
    device: str = 'auto'  # 'auto', 'cuda', 'mps', 'cpu'
    use_amp: bool = True  # Automatic mixed precision (CUDA only)

    # Logging and checkpointing
    log_interval: int = 50
    val_interval: int = 500
    save_interval: int = 2000
    metrics_interval: int = 1000  # Interval for comprehensive retrieval metrics

    # Metrics configuration
    metrics_sample_batches: int = 10  # Number of batches to sample for comprehensive metrics
    metrics_sample_size: int = 1000  # Target sample size for expensive metrics

    # Data settings
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True


@dataclass
class AlignmentConfig:
    """
    Main configuration for alignment training.

    This config orchestrates the alignment of sensor encoder outputs
    with text embeddings using contrastive learning.
    """

    # Experiment metadata
    experiment_name: str = 'alignment_default'
    output_dir: str = 'trained_models/alignment/default'

    # Data paths (sensor data)
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    vocab_path: Optional[str] = None

    # Dataset metadata (required for image-based encoders)
    dataset: Optional[str] = None  # e.g., 'milan', 'aruba'
    dataset_type: str = 'casas'  # e.g., 'casas', 'marble'

    # Text embeddings paths (pre-computed)
    train_text_embeddings_path: Optional[str] = None
    val_text_embeddings_path: Optional[str] = None

    # OR: Caption paths (will compute embeddings on-the-fly)
    train_captions_path: Optional[str] = None
    val_captions_path: Optional[str] = None
    text_encoder_config_path: Optional[str] = None

    # Encoder config (data encoder)
    encoder_config_path: Optional[str] = None  # Path to encoder YAML (old way)
    encoder: Optional[dict] = None  # Inline encoder config (new way)
    encoder_type: Literal['transformer', 'chronos', 'image'] = 'transformer'

    # Projection heads
    sensor_projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    text_projection: Optional[ProjectionConfig] = None  # None means frozen text embeddings

    # Loss configuration
    loss: LossConfig = field(default_factory=LossConfig)

    # Optimizer configuration
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # WandB logging
    use_wandb: bool = True
    wandb_project: str = 'discover-v2'
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: list = field(default_factory=list)
    wandb_notes: str = ''
    wandb_group: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'AlignmentConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Parse nested configs
        if 'sensor_projection' in config_dict:
            config_dict['sensor_projection'] = ProjectionConfig(**config_dict['sensor_projection'])

        if 'text_projection' in config_dict and config_dict['text_projection'] is not None:
            config_dict['text_projection'] = ProjectionConfig(**config_dict['text_projection'])

        # Handle mlm section - merge into loss config
        if 'mlm' in config_dict:
            mlm_config = config_dict.pop('mlm')
            if 'loss' not in config_dict:
                config_dict['loss'] = {}
            # Merge mlm params into loss config
            config_dict['loss'].update(mlm_config)

        if 'loss' in config_dict:
            config_dict['loss'] = LossConfig(**config_dict['loss'])

        if 'optimizer' in config_dict:
            config_dict['optimizer'] = OptimizerConfig(**config_dict['optimizer'])

        if 'training' in config_dict:
            config_dict['training'] = TrainingConfig(**config_dict['training'])

        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        import dataclasses

        def asdict_recursive(obj):
            """Convert dataclass to dict recursively."""
            if dataclasses.is_dataclass(obj):
                return {k: asdict_recursive(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        config_dict = asdict_recursive(self)

        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> bool:
        """Validate configuration."""
        errors = []

        # Check data paths
        if self.train_data_path is None:
            errors.append("train_data_path is required")

        if self.vocab_path is None:
            errors.append("vocab_path is required")

        # Check text embeddings or captions
        has_train_embeddings = self.train_text_embeddings_path is not None
        has_train_captions = self.train_captions_path is not None and self.text_encoder_config_path is not None

        if not (has_train_embeddings or has_train_captions):
            errors.append("Either train_text_embeddings_path OR (train_captions_path + text_encoder_config_path) is required")

        # Check encoder config
        # Must have either encoder_config_path or inline encoder config
        if self.encoder_config_path is None and self.encoder is None:
            errors.append("Either encoder_config_path or inline encoder config is required")

        # Check training steps/epochs
        if self.training.max_steps is None and self.training.max_epochs is None:
            errors.append("Either max_steps or max_epochs must be specified")

        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        return True

