from pathlib import Path
from typing import Dict, Any
import json

from data.data_config import (
    ProcessingConfig, WindowingConfig, FeatureConfig,
    CaptionConfig, ExportConfig, WindowStrategy, SplitStrategy
)


def load_preset(house: str, name: str) -> Dict[str, Any]:
    base = Path(__file__).parent.parent.parent / 'configs' / 'data_generation' / house
    with open(base / f'{name}.json', 'r') as f:
        return json.load(f)


def build_processing_config_from_preset(preset: Dict[str, Any]) -> ProcessingConfig:
    presegmented = preset.get('presegmented', False)
    split = preset.get('split_strategy')

    window_strategy = WindowStrategy.PRESEGMENTED if presegmented else WindowStrategy.SLIDING
    split_strategy = SplitStrategy.TEMPORAL if split == 'temporal' else SplitStrategy.RANDOM

    return ProcessingConfig(
        dataset_name="",
        windowing=WindowingConfig(
            sizes=preset.get('window_sizes', [20]),
            strategy=window_strategy,
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
            num_captions_per_window=4 if not presegmented else 4,
            max_caption_length=50 if not presegmented else 50,
            use_enhanced_captions=True,
            include_duration=True,
            include_time_context=True,
            include_room_transitions=True,
            include_salient_sensors=True
        ),
        export=ExportConfig(
            formats=['json'],
            output_dir=preset.get('export_dir', 'data/processed/casas'),
            compress=False,
            include_raw_events=False,
            include_statistics=True
        )
    )
