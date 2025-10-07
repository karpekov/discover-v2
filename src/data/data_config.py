"""
Processing configuration for the dual-encoder pipeline.

This module defines all configuration dataclasses for the data processing pipeline,
including windowing, features, captions, and export settings.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class WindowStrategy(Enum):
    """Window generation strategy."""
    SLIDING = "sliding"
    NON_OVERLAPPING = "non_overlapping"
    PRESEGMENTED = "presegmented"  # Activity-based segmentation first, then overlapping windows


class TodBucketing(Enum):
    """Time-of-day bucketing strategy."""
    PARTS_OF_DAY = "parts_of_day"
    HOURLY = "hourly"
    BI_HOURLY = "bi_hourly"
    QUAD_HOURLY = "quad_hourly"


class SplitStrategy(Enum):
    """Train/test split strategy."""
    RANDOM = "random"  # Random selection of days
    TEMPORAL = "temporal"  # Last X% of days chronologically


@dataclass
class WindowingConfig:
    """Configuration for window generation."""
    sizes: List[int]  # Window sizes in number of events
    strategy: WindowStrategy = WindowStrategy.SLIDING
    overlap_ratio: float = 0.75  # For sliding windows
    min_events: int = 8  # Minimum events per window
    max_gap_minutes: int = 30  # Maximum gap between events in a window

    # Presegmented window options
    drop_incomplete: bool = True  # Drop windows that are too small
    presegment_activity_level: str = "l1"  # "l1" or "l2" - which activity level to use for segmentation
    min_segment_events: int = 8  # Minimum events per activity segment to consider
    exclude_no_activity: bool = True  # Exclude segments with no_activity/No_Activity labels


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    # Spatial features
    use_coordinates: bool = True
    use_fourier_coords: bool = True
    fourier_coord_dims: int = 32

    # House layout features
    include_house_tokens: bool = True

    # Event features
    include_event_type: bool = True
    include_sensor_type: bool = True

    # Temporal features
    include_time_features: bool = True
    include_delta_time: bool = True
    time_bucket_minutes: int = 30
    delta_time_log_scale: bool = True
    tod_bucketing: TodBucketing = TodBucketing.PARTS_OF_DAY
    max_delta_t_minutes: int = 60  # Maximum delta time to consider

    # Optional features
    include_day_of_week: bool = True
    include_sequence_position: bool = False


@dataclass
class CaptionConfig:
    """Configuration for caption generation."""
    num_captions_per_window: int = 4  # 2 templated + 2 short creative
    max_caption_length: int = 30  # In tokens

    # Enhanced caption features
    use_enhanced_captions: bool = True
    include_duration: bool = True
    include_time_context: bool = True
    include_room_transitions: bool = True
    include_salient_sensors: bool = True

    # Caption type selection
    caption_types: str = 'long'  # 'long', 'short', 'both' - which caption types to include

    # Randomization
    random_seed: Optional[int] = 42
    use_synonyms: bool = True


@dataclass
class ExportConfig:
    """Configuration for data export."""
    formats: List[str]  # ['pickle', 'numpy', 'hdf5', 'txt']
    output_dir: str = "data/processed_v2"
    compress: bool = True
    include_raw_events: bool = False
    include_statistics: bool = True

    # Output structure
    separate_files_per_window_size: bool = True
    include_metadata: bool = True


@dataclass
class ProcessingConfig:
    """Main processing configuration combining all sub-configs."""
    dataset_name: str

    # Sub-configurations
    windowing: WindowingConfig
    features: FeatureConfig
    captions: CaptionConfig
    export: ExportConfig

    # Global settings
    train_test_split_by_days: bool = True
    test_ratio: float = 0.2
    test_size: float = 0.2  # Same as test_ratio, for compatibility
    split_strategy: SplitStrategy = SplitStrategy.RANDOM
    random_seed: int = 42

    # Processing options
    use_pre_segmentation: bool = False  # Align with existing labels
    filter_numeric_sensors: bool = True  # Filter out temperature/numeric sensors
    min_sequence_length: int = 1  # Minimum events in a sequence
    max_workers: Optional[int] = None  # For parallel processing

    # Label filtering for debugging
    filter_labels: Optional[List[str]] = None  # If provided, only keep data with these first_activity labels

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.windowing.overlap_ratio < 0 or self.windowing.overlap_ratio >= 1:
            raise ValueError("overlap_ratio must be in [0, 1)")

        if self.test_ratio <= 0 or self.test_ratio >= 1:
            raise ValueError("test_ratio must be in (0, 1)")

        if self.windowing.min_events <= 0:
            raise ValueError("min_events must be positive")

        valid_formats = {'pickle', 'numpy', 'hdf5', 'txt', 'json'}
        invalid_formats = set(self.export.formats) - valid_formats
        if invalid_formats:
            raise ValueError(f"Invalid export formats: {invalid_formats}. Valid: {valid_formats}")


def create_default_config(dataset_name: str) -> ProcessingConfig:
    """Create a default processing configuration for a dataset."""
    return ProcessingConfig(
        dataset_name=dataset_name,
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
