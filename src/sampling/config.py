"""
Configuration dataclasses for data sampling strategies.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from pathlib import Path


class SamplingStrategy(Enum):
    """Supported sampling strategies."""
    FIXED_LENGTH = "fixed_length"
    FIXED_DURATION = "fixed_duration"
    VARIABLE_DURATION = "variable_duration"


class SplitStrategy(Enum):
    """Train/test split strategy."""
    RANDOM = "random"  # Random selection of days
    TEMPORAL = "temporal"  # Last X% of days chronologically


@dataclass
class SamplingConfig:
    """Base configuration for all sampling strategies."""

    # Dataset information
    dataset_name: str  # e.g., "milan", "aruba", "cairo"
    raw_data_path: Path  # Path to raw sensor data
    output_dir: Path  # Where to save sampled data

    # Sampling strategy
    strategy: SamplingStrategy

    # Train/test split
    split_strategy: SplitStrategy = SplitStrategy.RANDOM
    train_ratio: float = 0.8  # 80% train, 20% test
    random_seed: int = 42

    # Overlap settings
    overlap_factor: float = 0.5  # 0.5 = 50% overlap (default)

    # Presegmentation options
    use_presegmentation: bool = False  # Split by ground truth labels first
    presegment_label_level: str = "l1"  # "l1" or "l2"
    min_segment_events: int = 1  # Minimum events per segment
    exclude_no_activity: bool = True  # Skip "No_Activity" segments

    # Filtering
    filter_numeric_sensors: bool = True  # Filter out temp/humidity sensors
    max_gap_minutes: Optional[int] = 30  # Max time gap within a window (None = no limit)

    # Processing limits (for debugging)
    max_lines: Optional[int] = None  # Limit raw data lines
    max_windows: Optional[int] = None  # Limit total windows

    # Metadata preservation
    preserve_full_metadata: bool = True  # Include all metadata for captions
    include_spatial_info: bool = True  # x, y coordinates
    include_sensor_types: bool = True  # Sensor type information
    metadata_path: Optional[Path] = None  # Path to metadata JSON file (casas_metadata.json or marble_metadata.json)

    # Multi-resident options
    multiresident_flatten: bool = False  # Load pre-flattened CSV (data_processed_flattened.csv) instead of raw data
    multiresident_split_by_resident: bool = False  # Split by resident_info before windowing so no window crosses resident boundaries

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.overlap_factor < 1:
            raise ValueError(f"overlap_factor must be in [0, 1), got {self.overlap_factor}")
        if not 0 < self.train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {self.train_ratio}")


@dataclass
class FixedLengthConfig(SamplingConfig):
    """Configuration for fixed-length sampling (Step 1a).

    Samples fixed number of sensor events per window.
    Example: 20, 50, 100 events per window.
    """

    strategy: SamplingStrategy = SamplingStrategy.FIXED_LENGTH

    # Fixed-length specific settings
    window_sizes: List[int] = field(default_factory=lambda: [20, 50])
    min_events_per_window: int = 8  # Recipe R2 requirement

    def __post_init__(self):
        super().__post_init__()
        if not self.window_sizes:
            raise ValueError("window_sizes cannot be empty")
        if any(size < self.min_events_per_window for size in self.window_sizes):
            raise ValueError(f"All window sizes must be >= {self.min_events_per_window}")


@dataclass
class FixedDurationConfig(SamplingConfig):
    """Configuration for fixed-duration sampling (Step 1b).

    Samples fixed time duration per window, resulting in variable-length sequences.
    Example: 30s, 60s, 120s windows (each can contain different number of events).
    """

    strategy: SamplingStrategy = SamplingStrategy.FIXED_DURATION

    # Fixed-duration specific settings
    duration_seconds: List[int] = field(default_factory=lambda: [30, 60])
    min_events_per_window: int = 1  # At least 1 event required
    max_events_per_window: Optional[int] = None  # Optional cap (for very dense periods)

    # Padding/truncation (for model input later)
    max_sequence_length: Optional[int] = 256  # Will be padded/truncated during training

    def __post_init__(self):
        super().__post_init__()
        if not self.duration_seconds:
            raise ValueError("duration_seconds cannot be empty")
        if any(d <= 0 for d in self.duration_seconds):
            raise ValueError("All durations must be positive")


@dataclass
class VariableDurationConfig(SamplingConfig):
    """Configuration for variable-duration sampling (Step 1c).

    Samples windows with varying durations selected from a list.
    Example: Randomly pick from [10s, 30s, 60s, 120s] for each window.

    This is a PLACEHOLDER for future implementation.
    """

    strategy: SamplingStrategy = SamplingStrategy.VARIABLE_DURATION

    # Variable-duration specific settings
    duration_options: List[int] = field(default_factory=lambda: [10, 30, 60, 120])
    selection_strategy: str = "uniform_random"  # "uniform_random", "weighted", etc.
    duration_weights: Optional[List[float]] = None  # For weighted selection
    min_events_per_window: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not self.duration_options:
            raise ValueError("duration_options cannot be empty")
        if self.selection_strategy == "weighted" and not self.duration_weights:
            raise ValueError("duration_weights required for weighted selection")


def load_sampling_config_from_dict(config_dict: Dict[str, Any]) -> SamplingConfig:
    """Load sampling config from dictionary (e.g., from YAML)."""
    strategy_str = config_dict.get("strategy", "fixed_length")
    strategy = SamplingStrategy(strategy_str)

    if strategy == SamplingStrategy.FIXED_LENGTH:
        return FixedLengthConfig(**config_dict)
    elif strategy == SamplingStrategy.FIXED_DURATION:
        return FixedDurationConfig(**config_dict)
    elif strategy == SamplingStrategy.VARIABLE_DURATION:
        return VariableDurationConfig(**config_dict)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy_str}")

