"""Data processing pipeline components."""

from .data_loader import DataLoader, ProcessedDataset
from .windowing import WindowProcessor, ProcessedWindow, WindowMetadata
from .features import FeatureExtractor, EventFeatures, WindowFeatures
from .captions import CaptionGenerator, Caption
from .exporters import DataExporter
from .pipeline import DualEncoderPipeline

# Configuration classes moved from config/
from .data_config import ProcessingConfig, WindowingConfig, FeatureConfig, CaptionConfig, ExportConfig
from .datasets import get_dataset_config, DATASET_REGISTRY

__all__ = [
    'DataLoader', 'ProcessedDataset',
    'WindowProcessor', 'ProcessedWindow', 'WindowMetadata',
    'FeatureExtractor', 'EventFeatures', 'WindowFeatures',
    'CaptionGenerator', 'Caption',
    'DataExporter',
    'DualEncoderPipeline',
    'ProcessingConfig', 'WindowingConfig', 'FeatureConfig', 'CaptionConfig', 'ExportConfig',
    'get_dataset_config', 'DATASET_REGISTRY'
]
