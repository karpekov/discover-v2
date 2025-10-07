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
    """Build ProcessingConfig from a JSON-like preset with extensive overrides.

    Supported top-level keys (all optional):
      - window_sizes: [int]
      - presegmented: bool (alias for windowing.strategy)
      - split_strategy: 'temporal' | 'random'
      - test_ratio: float in (0,1)
      - random_seed: int
      - filter_labels: [str]
      - export_dir: str (alias for export.output_dir, supports {config_name} and {dir_name})
      - dir_name: str (consumed by the caller when formatting export_dir)
      - generate_presegmented: bool (consumed by caller)

    Nested sections to override defaults:
      - windowing: dict of WindowingConfig fields
      - features: dict of FeatureConfig fields
      - captions: dict of CaptionConfig fields
      - export: dict of ExportConfig fields
    """

    # Derive strategies from simple aliases
    presegmented_alias = preset.get('presegmented', False)
    split = preset.get('split_strategy')

    window_strategy = WindowStrategy.PRESEGMENTED if presegmented_alias else WindowStrategy.SLIDING
    split_strategy = SplitStrategy.TEMPORAL if split == 'temporal' else SplitStrategy.RANDOM

    # Base defaults
    window_defaults = {
        'sizes': preset.get('window_sizes', [20]),
        'strategy': window_strategy,
        'overlap_ratio': 0.75,
        'min_events': 8,
        'max_gap_minutes': 30,
    }
    features_defaults = {
        'use_coordinates': True,
        'use_fourier_coords': True,
        'include_house_tokens': True,
        'include_event_type': True,
        'include_sensor_type': True,
        'include_time_features': True,
        'include_delta_time': True,
        'time_bucket_minutes': 30,
        'delta_time_log_scale': True,
    }
    captions_defaults = {
        'num_captions_per_window': 4,
        'max_caption_length': 50,
        'use_enhanced_captions': True,
        'include_duration': True,
        'include_time_context': True,
        'include_room_transitions': True,
        'include_salient_sensors': True,
    }
    export_defaults = {
        'formats': ['json'],
        'output_dir': preset.get('export_dir', 'data/processed/casas'),
        'compress': False,
        'include_raw_events': False,
        'include_statistics': True,
    }

    # Apply nested overrides if present
    window_overrides = preset.get('windowing', {}) or {}
    features_overrides = preset.get('features', {}) or {}
    captions_overrides = preset.get('captions', {}) or {}
    export_overrides = preset.get('export', {}) or {}

    # Build sub-configs
    windowing = WindowingConfig(
        sizes=window_overrides.get('sizes', window_defaults['sizes']),
        strategy=WindowStrategy(window_overrides.get('strategy', window_defaults['strategy'])),
        overlap_ratio=window_overrides.get('overlap_ratio', window_defaults['overlap_ratio']),
        min_events=window_overrides.get('min_events', window_defaults['min_events']),
        max_gap_minutes=window_overrides.get('max_gap_minutes', window_defaults['max_gap_minutes']),
        drop_incomplete=window_overrides.get('drop_incomplete', True),
        presegment_activity_level=window_overrides.get('presegment_activity_level', 'l1'),
        min_segment_events=window_overrides.get('min_segment_events', 8),
        exclude_no_activity=window_overrides.get('exclude_no_activity', True),
    )

    # Coerce tod_bucketing to enum if provided as string
    _tod = features_overrides.get('tod_bucketing', features_defaults.get('tod_bucketing', None))
    try:
        tod_enum = _tod if _tod is None or hasattr(_tod, 'value') else TodBucketing(_tod)
    except Exception:
        tod_enum = FeatureConfig.__dataclass_fields__['tod_bucketing'].default

    features = FeatureConfig(
        use_coordinates=features_overrides.get('use_coordinates', features_defaults['use_coordinates']),
        use_fourier_coords=features_overrides.get('use_fourier_coords', features_defaults['use_fourier_coords']),
        fourier_coord_dims=features_overrides.get('fourier_coord_dims', 32),
        include_house_tokens=features_overrides.get('include_house_tokens', features_defaults['include_house_tokens']),
        include_event_type=features_overrides.get('include_event_type', features_defaults['include_event_type']),
        include_sensor_type=features_overrides.get('include_sensor_type', features_defaults['include_sensor_type']),
        include_time_features=features_overrides.get('include_time_features', features_defaults['include_time_features']),
        include_delta_time=features_overrides.get('include_delta_time', features_defaults['include_delta_time']),
        time_bucket_minutes=features_overrides.get('time_bucket_minutes', features_defaults['time_bucket_minutes']),
        delta_time_log_scale=features_overrides.get('delta_time_log_scale', features_defaults['delta_time_log_scale']),
        tod_bucketing=tod_enum,
        max_delta_t_minutes=features_overrides.get('max_delta_t_minutes', 60),
        include_day_of_week=features_overrides.get('include_day_of_week', True),
        include_sequence_position=features_overrides.get('include_sequence_position', False),
    )

    captions = CaptionConfig(
        num_captions_per_window=captions_overrides.get('num_captions_per_window', captions_defaults['num_captions_per_window']),
        max_caption_length=captions_overrides.get('max_caption_length', captions_defaults['max_caption_length']),
        use_enhanced_captions=captions_overrides.get('use_enhanced_captions', captions_defaults['use_enhanced_captions']),
        include_duration=captions_overrides.get('include_duration', captions_defaults['include_duration']),
        include_time_context=captions_overrides.get('include_time_context', captions_defaults['include_time_context']),
        include_room_transitions=captions_overrides.get('include_room_transitions', captions_defaults['include_room_transitions']),
        include_salient_sensors=captions_overrides.get('include_salient_sensors', captions_defaults['include_salient_sensors']),
        caption_types=captions_overrides.get('caption_types', 'long'),
        random_seed=captions_overrides.get('random_seed', 42),
        use_synonyms=captions_overrides.get('use_synonyms', True),
    )

    export = ExportConfig(
        formats=export_overrides.get('formats', export_defaults['formats']),
        output_dir=export_overrides.get('output_dir', export_defaults['output_dir']),
        compress=export_overrides.get('compress', export_defaults['compress']),
        include_raw_events=export_overrides.get('include_raw_events', export_defaults['include_raw_events']),
        include_statistics=export_overrides.get('include_statistics', export_defaults['include_statistics']),
        separate_files_per_window_size=export_overrides.get('separate_files_per_window_size', True),
        include_metadata=export_overrides.get('include_metadata', True),
    )

    # Build main config with global overrides
    return ProcessingConfig(
        dataset_name="",
        windowing=windowing,
        features=features,
        captions=captions,
        export=export,
        train_test_split_by_days=True,
        test_ratio=float(preset.get('test_ratio', 0.2)),
        test_size=float(preset.get('test_ratio', 0.2)),
        split_strategy=split_strategy,
        random_seed=int(preset.get('random_seed', 42)),
        use_pre_segmentation=bool(preset.get('use_pre_segmentation', False)),
        filter_numeric_sensors=bool(preset.get('filter_numeric_sensors', True)),
        min_sequence_length=int(preset.get('min_sequence_length', 1)),
        max_workers=preset.get('max_workers', None),
        filter_labels=preset.get('filter_labels', None),
    )
