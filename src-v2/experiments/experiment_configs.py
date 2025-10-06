"""
Experiment configurations for dual-encoder alignment experiments.

This module defines pre-configured experiments with different combinations
of datasets, window sizes, and processing parameters.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from data.data_config import (
    ProcessingConfig, WindowingConfig, FeatureConfig,
    CaptionConfig, ExportConfig, WindowStrategy, SplitStrategy
)


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment."""
    name: str
    description: str
    datasets: List[str]
    window_sizes: List[int]
    processing_config_template: ProcessingConfig
    expected_processing_time_hours: float = 1.0
    data_types: List[str] = None  # ['sliding', 'presegmented', 'both'] - defaults to ['sliding']


def create_recipe_r2_config() -> ProcessingConfig:
    """Create Recipe R2 configuration with full features."""
    return ProcessingConfig(
        dataset_name="",  # Will be set per dataset
        windowing=WindowingConfig(
            sizes=[128],  # Recipe R2 default
            strategy=WindowStrategy.SLIDING,
            overlap_ratio=0.75,
            min_events=8,
            max_gap_minutes=30
        ),
        features=FeatureConfig(
            use_coordinates=True,
            use_fourier_coords=True,
            include_house_tokens=True,
            include_event_type=True,
            include_sensor_type=True,
            include_time_features=True,
            include_delta_time=True,
            time_bucket_minutes=30,
            delta_time_log_scale=True
        ),
        captions=CaptionConfig(
            num_captions_per_window=2,
            max_caption_length=30,
            use_enhanced_captions=True,
            include_duration=True,
            include_time_context=True,
            include_room_transitions=True,
            include_salient_sensors=True
        ),
        export=ExportConfig(
            formats=['pickle'],
            compress=True,
            include_raw_events=False,
            include_statistics=True
        )
    )


def create_training_config() -> ProcessingConfig:
    """Create configuration for training data export (JSON format)."""
    return ProcessingConfig(
        dataset_name="",  # Will be set per dataset
        windowing=WindowingConfig(
            sizes=[20],  # Match training sequence length
            strategy=WindowStrategy.SLIDING,
            overlap_ratio=0.75,
            min_events=8,
            max_gap_minutes=30
        ),
        features=FeatureConfig(
            use_coordinates=True,
            use_fourier_coords=True,
            include_house_tokens=True,
            include_event_type=True,
            include_sensor_type=True,
            include_time_features=True,
            include_delta_time=True,
            time_bucket_minutes=30,
            delta_time_log_scale=True
        ),
        captions=CaptionConfig(
            num_captions_per_window=4,  # Multiple diverse captions per window for better training
            max_caption_length=50,
            use_enhanced_captions=True,
            include_duration=True,
            include_time_context=True,
            include_room_transitions=True,
            include_salient_sensors=True
        ),
        export=ExportConfig(
            formats=['json'],  # JSON format for training
            output_dir="data/data_for_alignment/{config_name}",  # Save in data/data_for_alignment/{config_name}/
            compress=False,
            include_raw_events=False,
            include_statistics=True
        )
    )


def create_multi_window_config() -> ProcessingConfig:
    """Create configuration for multiple window sizes."""
    return ProcessingConfig(
        dataset_name="",  # Will be set per dataset
        windowing=WindowingConfig(
            sizes=[20, 50, 100],
            strategy=WindowStrategy.SLIDING,
            overlap_ratio=0.75,
            min_events=8,
            max_gap_minutes=30
        ),
        features=FeatureConfig(
            use_coordinates=True,
            use_fourier_coords=True,
            include_house_tokens=True,
            include_event_type=True,
            include_sensor_type=True,
            include_time_features=True,
            include_delta_time=True,
            time_bucket_minutes=30,
            delta_time_log_scale=True
        ),
        captions=CaptionConfig(
            num_captions_per_window=2,
            max_caption_length=30,
            use_enhanced_captions=True,
            include_duration=True,
            include_time_context=True,
            include_room_transitions=True,
            include_salient_sensors=True
        ),
        export=ExportConfig(
            formats=['pickle'],
            compress=True,
            include_raw_events=False,
            include_statistics=True
        )
    )


def create_temporal_split_config() -> ProcessingConfig:
    """Create configuration with temporal splitting (for comparison with random splitting)."""
    config = create_training_config()
    config.split_strategy = SplitStrategy.TEMPORAL  # Override default random splitting
    return config


def create_presegmented_config() -> ProcessingConfig:
    """Create configuration for presegmented (clean) windows."""
    return ProcessingConfig(
        dataset_name="",  # Will be set per dataset
        windowing=WindowingConfig(
            sizes=[20],  # Match training sequence length
            strategy=WindowStrategy.PRESEGMENTED,  # Use presegmented strategy
            overlap_ratio=0.75,  # Still use overlap within segments
            min_events=8,
            max_gap_minutes=30,
            # Presegmented-specific options
            drop_incomplete=True,
            presegment_activity_level="l1",  # Use L1 activities for segmentation
            min_segment_events=8,
            exclude_no_activity=True
        ),
        features=FeatureConfig(
            use_coordinates=True,
            use_fourier_coords=True,
            include_house_tokens=True,
            include_event_type=True,
            include_sensor_type=True,
            include_time_features=True,
            include_delta_time=True,
            time_bucket_minutes=30,
            delta_time_log_scale=True
        ),
        captions=CaptionConfig(
            num_captions_per_window=4,  # Multiple diverse captions per window
            max_caption_length=50,
            use_enhanced_captions=True,
            include_duration=True,
            include_time_context=True,
            include_room_transitions=True,
            include_salient_sensors=True
        ),
        export=ExportConfig(
            formats=['json'],  # JSON format for training
            output_dir="data/data_for_alignment/{config_name}",
            compress=False,
            include_raw_events=False,
            include_statistics=True
        )
    )


# Pre-defined experiment configurations
EXPERIMENT_CONFIGS = {

    'milan_training_20': ExperimentConfig(
        name='milan_training_20',
        description='Milan dataset for training pipeline (JSON export)',
        datasets=['milan'],
        window_sizes=[20],
        processing_config_template=create_training_config(),
        expected_processing_time_hours=0.5,
        data_types=['both']
    ),

    'milan_training_50': ExperimentConfig(
        name='milan_training_50',
        description='Milan dataset for training pipeline (JSON export)',
        datasets=['milan'],
        window_sizes=[50],
        processing_config_template=create_training_config(),
        expected_processing_time_hours=0.5,
        data_types=['both']
    ),


    'milan_training_50_vOct1': ExperimentConfig(
        name='milan_training_50_vOct1',
        description='Milan dataset for training pipeline (JSON export)',
        datasets=['milan'],
        window_sizes=[50],
        processing_config_template=create_training_config(),
        expected_processing_time_hours=0.5,
        data_types=['both']
    ),

    'milan_training_20_vOct1': ExperimentConfig(
        name='milan_training_20_vOct1',
        description='Milan dataset for training pipeline (JSON export)',
        datasets=['milan'],
        window_sizes=[20],
        processing_config_template=create_training_config(),
        expected_processing_time_hours=0.5,
        data_types=['both']
    ),

    'milan_training_100': ExperimentConfig(
        name='milan_training_100',
        description='Milan dataset for training pipeline (JSON export)',
        datasets=['milan'],
        window_sizes=[100],
        processing_config_template=create_training_config(),
        expected_processing_time_hours=0.5,
        data_types=['both']
    ),

    'milan_training_20_50': ExperimentConfig(
        name='milan_training_20_50',
        description='Milan dataset for training pipeline (JSON export)',
        datasets=['milan'],
        window_sizes=[20, 50],
        processing_config_template=create_training_config(),
        expected_processing_time_hours=0.5
    ),

    'milan_temporal_split': ExperimentConfig(
        name='milan_temporal_split',
        description='Milan dataset with temporal splitting (last 20% of days for testing)',
        datasets=['milan'],
        window_sizes=[20],
        processing_config_template=create_temporal_split_config(),
        expected_processing_time_hours=0.5
    ),

    'milan_presegmented': ExperimentConfig(
        name='milan_presegmented',
        description='Milan dataset with presegmented clean windows (single activity per window)',
        datasets=['milan'],
        window_sizes=[20],
        processing_config_template=create_presegmented_config(),
        expected_processing_time_hours=0.5,
        data_types=['presegmented']
    ),
}


def get_experiment_config(name: str) -> ExperimentConfig:
    """Get experiment configuration by name."""
    if name not in EXPERIMENT_CONFIGS:
        available = list(EXPERIMENT_CONFIGS.keys())
        raise ValueError(f"Unknown experiment '{name}'. Available: {available}")

    return EXPERIMENT_CONFIGS[name]


def list_available_experiments() -> Dict[str, str]:
    """List all available experiment configurations."""
    return {name: config.description for name, config in EXPERIMENT_CONFIGS.items()}


def create_custom_experiment(name: str, description: str, datasets: List[str],
                           window_sizes: List[int], config_type: str = 'multi_window') -> ExperimentConfig:
    """Create a custom experiment configuration."""

    if config_type == 'recipe_r2':
        config_template = create_recipe_r2_config()
    elif config_type == 'quick_validation':
        config_template = create_quick_validation_config()
    else:  # multi_window
        config_template = create_multi_window_config()

    # Update window sizes
    config_template.windowing.sizes = window_sizes

    # Estimate processing time (rough heuristic)
    time_per_dataset_hour = 0.5 if max(window_sizes) <= 50 else 1.0
    estimated_time = len(datasets) * len(window_sizes) * time_per_dataset_hour

    return ExperimentConfig(
        name=name,
        description=description,
        datasets=datasets,
        window_sizes=window_sizes,
        processing_config_template=config_template,
        expected_processing_time_hours=estimated_time
    )