"""
Utilities for WandB logging organization.

Generates intuitive run names based on configuration.
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional


def extract_dataset_name(data_path: str) -> str:
    """
    Extract dataset name from data path.

    Examples:
        data/processed/casas/milan/... → milan
        data/processed/marble/house1/... → house1
    """
    path = Path(data_path)
    parts = path.parts

    # Try to find dataset name in path
    for i, part in enumerate(parts):
        if part in ['casas', 'marble', 'orange4home']:
            # Next part is the dataset name
            if i + 1 < len(parts):
                return parts[i + 1]

    # Fallback: use parent directory name
    return path.parent.name


def extract_sampling_strategy(data_path: str, config: Any) -> str:
    """
    Extract sampling strategy abbreviation.

    Examples:
        FD_60 → fd60
        FL_50 → fl50
        FD_120_p → fd120_p
    """
    path_str = str(data_path)

    # Check for presegmentation
    preseg_suffix = '_preseg' if 'presegmented' in path_str else ''

    # Extract strategy from path
    if 'fixed_duration' in path_str:
        # Extract duration number
        match = re.search(r'fixed_duration[_-]?(\d+)[s]?', path_str)
        if match:
            duration = match.group(1)
            return f'fixdur{duration}s{preseg_suffix}'
        return f'fixdur{preseg_suffix}'

    elif 'fixed_length' in path_str:
        # Extract length number
        match = re.search(r'fixed_length[_-]?(\d+)', path_str)
        if match:
            length = match.group(1)
            return f'fixlen{length}{preseg_suffix}'
        return f'fixlen{preseg_suffix}'

    elif 'variable_duration' in path_str:
        return f'vardur{preseg_suffix}'

    # Fallback
    return 'unknown'


def extract_encoder_name(encoder_config_path: str) -> str:
    """
    Extract encoder model name abbreviation.

    Examples:
        configs/encoders/transformer_base.yaml → tf-base
        configs/encoders/transformer_tiny.yaml → tf-tiny
        configs/encoders/transformer_small.yaml → tf-small
    """
    if not encoder_config_path:
        return 'unknown'

    path = Path(encoder_config_path)
    stem = path.stem  # e.g., 'transformer_base'

    # Map common names
    name_map = {
        'transformer_base': 'tf-base',
        'transformer_tiny': 'tf-tiny',
        'transformer_small': 'tf-small',
        'transformer_minimal': 'tf-min',
        'transformer_base_mlp': 'tf-base-mlp',
        'transformer_base_mlp3': 'tf-base-mlp3',
        'chronos': 'chronos',
        'image': 'img',
    }

    return name_map.get(stem, stem.replace('transformer_', 'tf-'))


def extract_caption_style(embeddings_path: Optional[str] = None, captions_path: Optional[str] = None) -> str:
    """
    Extract caption style from embeddings or captions path.

    Examples:
        train_embeddings_baseline_gte_base.npz → baseline
        train_captions_sourish.json → sourish
        train_embeddings_adl-llm_distilroberta.npz → adl-llm
    """
    path_str = embeddings_path or captions_path or ''

    # Check for common caption styles
    if 'baseline' in path_str:
        return 'baseline'
    elif 'sourish' in path_str:
        return 'sourish'
    elif 'adl-llm' in path_str or 'adl_llm' in path_str:
        return 'adl-llm'
    elif 'llm' in path_str:
        return 'llm'
    elif 'mixed' in path_str:
        return 'mixed'

    # Fallback
    return 'unknown'


def extract_text_encoder_name(embeddings_path: Optional[str] = None, text_encoder_config: Optional[str] = None) -> str:
    """
    Extract text encoder name abbreviation.

    Examples:
        train_embeddings_baseline_gte_base.npz → gte
        configs/text_encoders/distilroberta_base.yaml → distilroberta
        train_embeddings_sourish_llama_embed.npz → llama
    """
    path_str = embeddings_path or text_encoder_config or ''

    # Map common encoder names
    encoder_map = {
        'gte': 'gte',
        'distilroberta': 'distilroberta',
        'minilm': 'minilm',
        'embeddinggemma': 'gemma',
        'llama': 'llama',
        'clip': 'clip',
        'siglip': 'siglip',
    }

    path_lower = path_str.lower()
    for full_name, short_name in encoder_map.items():
        if full_name in path_lower:
            return short_name

    # Fallback
    return 'unknown'


def extract_projection_type(config: Any) -> str:
    """
    Extract projection type abbreviation.

    Examples:
        linear → linear
        mlp (2-layer) → mlp2
        mlp (3-layer) → mlp3
    """
    proj_config = config.sensor_projection

    if proj_config.type == 'linear':
        return 'linear'
    elif proj_config.type == 'mlp':
        num_layers = proj_config.num_layers
        return f'mlp{num_layers}'

    return proj_config.type


def extract_loss_config(config: Any) -> str:
    """
    Extract loss configuration abbreviation.

    Examples:
        CLIP only → clip
        CLIP + MLM → clip+mlm
        CLIP + hard negatives → clip+hn
        CLIP + MLM + hard negatives → clip+mlm+hn
    """
    loss_config = config.loss

    components = ['clip']

    if loss_config.mlm_weight > 0:
        components.append('mlm')

    if loss_config.use_hard_negatives:
        components.append('hn')

    return '+'.join(components)


def generate_wandb_run_name(config: Any) -> str:
    """
    Generate intuitive WandB run name from configuration.

    Format: {dataset}_{sampling}_{encoder}_{caption-style}_{text-encoder}_{projection}

    Example: milan_fixdur60s_tf-base_baseline_gte_linear

    Optional suffixes for non-default configs:
        - Loss config if not CLIP-only
        - Other important hyperparams

    Args:
        config: AlignmentConfig instance

    Returns:
        Formatted run name
    """
    # Extract components
    dataset = extract_dataset_name(config.train_data_path)
    sampling = extract_sampling_strategy(config.train_data_path, config)
    encoder = extract_encoder_name(config.encoder_config_path)
    caption_style = extract_caption_style(config.train_text_embeddings_path, config.train_captions_path)
    text_encoder = extract_text_encoder_name(config.train_text_embeddings_path, config.text_encoder_config_path)
    projection = extract_projection_type(config)

    # Base name
    base_components = [
        dataset,
        sampling,
        encoder,
        caption_style,
        text_encoder,
        projection,
    ]

    base_name = '_'.join(base_components)

    # Optional suffixes for non-default configurations
    suffixes = []

    # Loss configuration (if not default CLIP-only)
    loss_config = extract_loss_config(config)
    if loss_config != 'clip':
        suffixes.append(loss_config)

    # Add temperature info if fixed (non-learnable)
    if not config.loss.learnable_temperature:
        temp = config.loss.temperature_init
        suffixes.append(f'temp{temp:.3f}'.replace('.', ''))

    # Combine
    if suffixes:
        return f"{base_name}__{'-'.join(suffixes)}"
    else:
        return base_name


def generate_wandb_group(config: Any) -> str:
    """
    Generate WandB group name for organizing related runs.

    Groups runs by: {dataset}_{sampling}_{encoder}

    Example: milan_fixdur60s_tf-base

    This groups together runs that differ only in caption/text encoder/projection.

    Args:
        config: AlignmentConfig instance

    Returns:
        Group name
    """
    dataset = extract_dataset_name(config.train_data_path)
    sampling = extract_sampling_strategy(config.train_data_path, config)
    encoder = extract_encoder_name(config.encoder_config_path)

    return f"{dataset}_{sampling}_{encoder}"


def generate_wandb_tags(config: Any) -> list:
    """
    Generate WandB tags for filtering and organization.

    Args:
        config: AlignmentConfig instance

    Returns:
        List of tags
    """
    tags = []

    # Dataset
    dataset = extract_dataset_name(config.train_data_path)
    tags.append(dataset)

    # Sampling strategy
    sampling = extract_sampling_strategy(config.train_data_path, config)
    if 'fixdur' in sampling:
        tags.append('fixed-duration')
    elif 'fixlen' in sampling:
        tags.append('fixed-length')
    if 'preseg' in sampling:
        tags.append('presegmented')

    # Encoder type
    encoder = extract_encoder_name(config.encoder_config_path)
    tags.append(encoder)

    # Caption style
    caption_style = extract_caption_style(config.train_text_embeddings_path, config.train_captions_path)
    tags.append(f'caption-{caption_style}')

    # Text encoder
    text_encoder = extract_text_encoder_name(config.train_text_embeddings_path, config.text_encoder_config_path)
    tags.append(f'text-{text_encoder}')

    # Projection
    projection = extract_projection_type(config)
    tags.append(f'proj-{projection}')

    # Loss components
    if config.loss.mlm_weight > 0:
        tags.append('mlm')
    if config.loss.use_hard_negatives:
        tags.append('hard-negatives')

    # Temperature
    if config.loss.learnable_temperature:
        tags.append('learnable-temp')
    else:
        tags.append('fixed-temp')

    return tags


if __name__ == '__main__':
    # Test the functions
    print("Testing WandB naming utilities...")

    class MockConfig:
        def __init__(self):
            self.train_data_path = "data/processed/casas/milan/FD_60/train.json"
            self.train_text_embeddings_path = "data/processed/casas/milan/FD_60/train_embeddings_baseline_gte_base.npz"
            self.train_captions_path = None
            self.text_encoder_config_path = "configs/text_encoders/gte_base.yaml"
            self.encoder_config_path = "configs/encoders/transformer_base.yaml"

            class SensorProjection:
                type = 'linear'
                num_layers = 2

            class Loss:
                mlm_weight = 0.0
                use_hard_negatives = False
                learnable_temperature = True
                temperature_init = 0.02

            self.sensor_projection = SensorProjection()
            self.loss = Loss()

    config = MockConfig()

    print(f"\nRun name: {generate_wandb_run_name(config)}")
    print(f"Group: {generate_wandb_group(config)}")
    print(f"Tags: {generate_wandb_tags(config)}")

    # Test with MLM
    config.loss.mlm_weight = 1.0
    print(f"\nWith MLM:")
    print(f"Run name: {generate_wandb_run_name(config)}")
    print(f"Tags: {generate_wandb_tags(config)}")

