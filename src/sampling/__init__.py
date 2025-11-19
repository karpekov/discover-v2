"""
Data sampling module for HAR discover-v2 project.

This module provides different sampling strategies for converting raw sensor data
into training/testing samples:
- Fixed-length sampling (1a): Fixed number of events per window
- Fixed-duration sampling (1b): Fixed time duration per window
- Variable-duration sampling (1c): Variable time durations per window (future)

Each sampler is self-sufficient and produces standardized JSON output.

Also includes automatic data preparation utility for training pipeline.
"""

from .base import BaseSampler, SamplingResult
from .config import (
    SamplingConfig,
    FixedLengthConfig,
    FixedDurationConfig,
    VariableDurationConfig,
    SamplingStrategy,
    load_sampling_config_from_dict,
)
from .fixed_length import FixedLengthSampler
from .fixed_duration import FixedDurationSampler
from .utils import standardize_column_names, get_column_name, print_column_info
from .data_prep import DataPreparer, prepare_data_for_config  # Can also be run as script

__all__ = [
    # Base classes
    'BaseSampler',
    'SamplingResult',

    # Config classes
    'SamplingConfig',
    'FixedLengthConfig',
    'FixedDurationConfig',
    'VariableDurationConfig',
    'SamplingStrategy',
    'load_sampling_config_from_dict',

    # Sampler implementations
    'FixedLengthSampler',
    'FixedDurationSampler',

    # Data utilities
    'standardize_column_names',
    'get_column_name',
    'print_column_info',

    # Data preparation
    'DataPreparer',
    'prepare_data_for_config',
]

