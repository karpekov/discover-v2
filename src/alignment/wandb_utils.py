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
        FL_20 → fl20
        FD_60 → fd60
        FD_120_p → fd120p
    """
    path_str = str(data_path)

    # Look for FL_XX or FL_XX_p patterns (Fixed Length)
    match = re.search(r'/FL_(\d+)(_p)?/', path_str)
    if match:
        length = match.group(1)
        preseg = 'p' if match.group(2) else ''
        return f'fl{length}{preseg}'

    # Look for FD_XX or FD_XX_p patterns (Fixed Duration)
    match = re.search(r'/FD_(\d+)(_p)?/', path_str)
    if match:
        duration = match.group(1)
        preseg = 'p' if match.group(2) else ''
        return f'fd{duration}{preseg}'

    # Legacy: full name patterns
    if 'fixed_duration' in path_str:
        match = re.search(r'fixed_duration[_-]?(\d+)[s]?', path_str)
        if match:
            duration = match.group(1)
            return f'fd{duration}'
        return 'fd'

    elif 'fixed_length' in path_str:
        match = re.search(r'fixed_length[_-]?(\d+)', path_str)
        if match:
            length = match.group(1)
            return f'fl{length}'
        return 'fl'

    elif 'variable_duration' in path_str:
        return 'vd'

    # Fallback
    return 'unknown'


def extract_encoder_name(config: Any) -> str:
    """
    Extract encoder type: seq (sequence-based) or img (image-based).

    Examples:
        transformer_base.yaml → seq
        transformer_image_clip.yaml → img
        encoder_type: 'image' → img
    """
    # Check encoder_type field first
    if hasattr(config, 'encoder_type'):
        if config.encoder_type == 'image':
            return 'img'

    # Check encoder_config_path for image indicators
    if hasattr(config, 'encoder_config_path') and config.encoder_config_path:
        path_str = str(config.encoder_config_path).lower()
        if 'image' in path_str:
            return 'img'

    # Check if inline encoder config has image settings
    if hasattr(config, 'encoder') and config.encoder:
        if isinstance(config.encoder, dict):
            if config.encoder.get('use_image_embeddings', False):
                return 'img'

    # Default to sequence-based
    return 'seq'


def extract_caption_style(embeddings_path: Optional[str] = None, captions_path: Optional[str] = None) -> str:
    """
    Extract caption style: rb0 (baseline rule-based), rb (other rule-based), or llm.

    Examples:
        train_embeddings_baseline_clip.npz → rb0
        train_captions_sourish.json → rb
        train_embeddings_llm_gte.npz → llm
    """
    path_str = embeddings_path or captions_path or ''

    # Check for baseline (rb0)
    if 'baseline' in path_str:
        return 'rb0'

    # Check for LLM-generated
    elif 'llm' in path_str or 'gpt' in path_str or 'claude' in path_str:
        return 'llm'

    # Check for other rule-based styles (sourish, mixed, etc.)
    elif 'sourish' in path_str or 'mixed' in path_str:
        return 'rb'

    # Fallback
    return 'rb'


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
    Extract projection type: lin or mlp.

    Examples:
        linear → lin
        mlp (2 or 3 layers) → mlp
    """
    proj_config = config.sensor_projection

    if proj_config.type == 'linear':
        return 'lin'
    elif proj_config.type == 'mlp':
        return 'mlp'

    return 'lin'  # fallback


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

    Format: {dataset}_{sampling_short}_{seq/img}_{rb/llm}_text{encoder}_proj{lin/mlp}

    Examples:
        milan_fd20_seq_rb0_textclip_projlin
        milan_fl30_img_llm_textminilm_projmlp

    Args:
        config: AlignmentConfig instance

    Returns:
        Formatted run name
    """
    # Extract components
    dataset = extract_dataset_name(config.train_data_path)
    sampling = extract_sampling_strategy(config.train_data_path, config)
    encoder_type = extract_encoder_name(config)  # seq or img
    caption_style = extract_caption_style(config.train_text_embeddings_path, config.train_captions_path)
    text_encoder = extract_text_encoder_name(config.train_text_embeddings_path, config.text_encoder_config_path)
    projection = extract_projection_type(config)

    # Build name: {dataset}_{sampling}_{seq/img}_{rb/llm}_text{encoder}_proj{lin/mlp}
    name = f"{dataset}_{sampling}_{encoder_type}_{caption_style}_text{text_encoder}_proj{projection}"

    return name


def generate_wandb_group(config: Any) -> str:
    """
    Generate WandB group name for organizing related runs.

    Groups runs by: {dataset}_{sampling}_{seq/img}

    Example: milan_fd20_seq

    This groups together runs that differ only in caption/text encoder/projection.

    Args:
        config: AlignmentConfig instance

    Returns:
        Group name
    """
    dataset = extract_dataset_name(config.train_data_path)
    sampling = extract_sampling_strategy(config.train_data_path, config)
    encoder_type = extract_encoder_name(config)

    return f"{dataset}_{sampling}_{encoder_type}"


def generate_wandb_tags(config: Any) -> list:
    """
    Generate WandB tags for filtering and organization.

    All detailed metadata goes in tags now (not in run name).

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
    if 'fd' in sampling:
        tags.append('fixed-duration')
    elif 'fl' in sampling:
        tags.append('fixed-length')
    elif 'vd' in sampling:
        tags.append('variable-duration')
    if sampling.endswith('p'):
        tags.append('presegmented')

    # Add full sampling tag (e.g., 'fd20', 'fl30p')
    tags.append(sampling)

    # Encoder type
    encoder_type = extract_encoder_name(config)
    tags.append(encoder_type)  # 'seq' or 'img'

    # If image-based, try to detect which image model
    if encoder_type == 'img' and hasattr(config, 'encoder_config_path') and config.encoder_config_path:
        path_str = str(config.encoder_config_path).lower()
        if 'clip' in path_str:
            tags.append('img-clip')
        elif 'dinov2' in path_str:
            tags.append('img-dinov2')
        elif 'siglip' in path_str:
            tags.append('img-siglip')

    # Caption style
    caption_style = extract_caption_style(config.train_text_embeddings_path, config.train_captions_path)
    tags.append(caption_style)  # 'rb0', 'rb', or 'llm'

    # Text encoder
    text_encoder = extract_text_encoder_name(config.train_text_embeddings_path, config.text_encoder_config_path)
    tags.append(f'text-{text_encoder}')

    # Projection
    projection = extract_projection_type(config)
    tags.append(f'proj-{projection}')

    # If MLP, add num layers
    if projection == 'mlp' and hasattr(config, 'sensor_projection'):
        if hasattr(config.sensor_projection, 'num_layers'):
            tags.append(f'mlp-{config.sensor_projection.num_layers}layer')

    # Loss components
    if config.loss.mlm_weight > 0:
        tags.append('mlm')
        tags.append(f'mlm-weight-{config.loss.mlm_weight}')

    if config.loss.use_hard_negatives:
        tags.append('hard-negatives')

    # Temperature
    if config.loss.learnable_temperature:
        tags.append('learnable-temp')
    else:
        tags.append('fixed-temp')
        temp = config.loss.temperature_init
        tags.append(f'temp-{temp:.3f}'.replace('.', ''))

    # Optimizer
    if hasattr(config, 'optimizer') and hasattr(config.optimizer, 'type'):
        tags.append(f'opt-{config.optimizer.type}')

    # Learning rate
    if hasattr(config, 'optimizer') and hasattr(config.optimizer, 'lr'):
        lr = config.optimizer.lr
        tags.append(f'lr-{lr:.0e}')

    return tags


if __name__ == '__main__':
    # Test the functions
    print("Testing WandB naming utilities...")

    class MockConfig:
        def __init__(self):
            self.train_data_path = "data/processed/casas/milan/FD_20/train.json"
            self.train_text_embeddings_path = "data/processed/casas/milan/FD_20/train_embeddings_baseline_clip.npz"
            self.train_captions_path = None
            self.text_encoder_config_path = "configs/text_encoders/clip_vit_base.yaml"
            self.encoder_config_path = None
            self.encoder_type = 'transformer'

            class Optimizer:
                type = 'adamw'
                lr = 3e-4

            class SensorProjection:
                type = 'linear'
                num_layers = 2

            class Loss:
                mlm_weight = 0.5
                use_hard_negatives = False
                learnable_temperature = False
                temperature_init = 0.07

            self.sensor_projection = SensorProjection()
            self.loss = Loss()
            self.optimizer = Optimizer()

    # Test 1: Sequence-based with baseline captions
    print("\n=== Test 1: Sequence + Baseline + CLIP ===")
    config = MockConfig()
    print(f"Expected: milan_fd20_seq_rb0_textclip_projlin")
    print(f"Got:      {generate_wandb_run_name(config)}")
    print(f"Group:    {generate_wandb_group(config)}")
    print(f"Tags:     {generate_wandb_tags(config)}")

    # Test 2: Image-based with LLM captions
    print("\n=== Test 2: Image + LLM + MiniLM ===")
    config.train_data_path = "data/processed/casas/milan/FL_30/train.json"
    config.train_text_embeddings_path = "data/processed/casas/milan/FL_30/train_embeddings_llm_minilm.npz"
    config.encoder_config_path = "configs/encoders/transformer_image_clip.yaml"
    config.sensor_projection.type = 'mlp'
    config.sensor_projection.num_layers = 2
    print(f"Expected: milan_fl30_img_llm_textminilm_projmlp")
    print(f"Got:      {generate_wandb_run_name(config)}")
    print(f"Group:    {generate_wandb_group(config)}")

    # Test 3: Presegmented
    print("\n=== Test 3: Presegmented ===")
    config.train_data_path = "data/processed/casas/aruba/FD_60_p/train.json"
    config.train_text_embeddings_path = "data/processed/casas/aruba/FD_60_p/train_embeddings_baseline_gte.npz"
    config.encoder_config_path = None
    config.encoder_type = 'transformer'
    config.sensor_projection.type = 'linear'
    print(f"Expected: aruba_fd60p_seq_rb0_textgte_projlin")
    print(f"Got:      {generate_wandb_run_name(config)}")

