"""Data export utilities for processed sensor data."""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import gzip
from dataclasses import asdict

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

from .windowing import ProcessedWindow
from .features import WindowFeatures, EventFeatures
from .captions import Caption
from .data_config import ExportConfig


class DataExporter:
    """Export processed data in multiple formats."""

    def __init__(self, config: ExportConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all_formats(self, dataset_name: str, window_size: int,
                          windows: List[ProcessedWindow],
                          window_features: List[WindowFeatures],
                          captions: List[Caption],
                          feature_vocab: Dict[str, List[str]],
                          feature_mappings: Dict[str, Dict[str, int]],
                          metadata: Dict[str, Any],
                          is_presegmented: bool = False,
                          debug_mode: bool = False) -> Dict[str, str]:
        """Export data in all requested formats.

        Returns:
            Dictionary mapping format -> file_path
        """
        exported_files = {}

        for format_name in self.config.formats:
            if format_name == 'numpy':
                file_path = self._export_numpy(
                    dataset_name, window_size, windows, window_features,
                    captions, feature_vocab, feature_mappings, metadata, is_presegmented
                )
            elif format_name == 'pickle':
                file_path = self._export_pickle(
                    dataset_name, window_size, windows, window_features,
                    captions, feature_vocab, feature_mappings, metadata, is_presegmented
                )
            elif format_name == 'txt':
                file_path = self._export_txt(
                    dataset_name, window_size, windows, captions, is_presegmented
                )
            elif format_name == 'hdf5':
                if not HAS_HDF5:
                    print(f"Warning: Skipping HDF5 export - h5py not installed")
                    continue
                file_path = self._export_hdf5(
                    dataset_name, window_size, windows, window_features,
                    captions, feature_vocab, feature_mappings, metadata, is_presegmented
                )
            elif format_name == 'json':
                if debug_mode:
                    file_path = self._export_debug_json(
                        dataset_name, window_size, windows, window_features,
                        captions, feature_vocab, feature_mappings, metadata, is_presegmented
                    )
                else:
                    file_path = self._export_json(
                        dataset_name, window_size, windows, window_features,
                        captions, feature_vocab, feature_mappings, metadata, is_presegmented
                    )
            else:
                print(f"Warning: Unknown export format: {format_name}")
                continue

            exported_files[format_name] = str(file_path)
            print(f"  Exported {format_name}: {file_path}")

        return exported_files

    def _get_base_name(self, dataset_name: str, window_size: int, is_presegmented: bool = False) -> str:
        """Generate base filename with optional presegmented suffix."""
        base_name = f"{dataset_name}_w{window_size}"
        if is_presegmented:
            base_name += "_presegmented"
        return base_name

    def _export_numpy(self, dataset_name: str, window_size: int,
                     windows: List[ProcessedWindow],
                     window_features: List[WindowFeatures],
                     captions: List[Caption],
                     feature_vocab: Dict[str, List[str]],
                     feature_mappings: Dict[str, Dict[str, int]],
                     metadata: Dict[str, Any], is_presegmented: bool = False) -> Path:
        """Export as numpy arrays."""

        base_name = self._get_base_name(dataset_name, window_size, is_presegmented)
        output_file = self.output_dir / f"{base_name}.npz"

        # Convert windows to structured arrays
        window_data = self._windows_to_numpy(windows, window_features, feature_mappings)

        # Convert captions to arrays
        caption_texts = [c.text for c in captions]
        caption_window_ids = [c.window_id for c in captions]
        caption_types = [c.caption_type for c in captions]

        # Save as compressed numpy archive
        save_dict = {
            'window_ids': window_data['window_ids'],
            'event_features': window_data['event_features'],
            'window_metadata': window_data['window_metadata'],
            'caption_texts': np.array(caption_texts, dtype=object),
            'caption_window_ids': np.array(caption_window_ids),
            'caption_types': np.array(caption_types, dtype=object),
            'feature_vocab': np.array([feature_vocab], dtype=object)[0],
            'feature_mappings': np.array([feature_mappings], dtype=object)[0],
            'metadata': np.array([metadata], dtype=object)[0]
        }

        if self.config.compress:
            np.savez_compressed(output_file, **save_dict)
        else:
            np.savez(output_file, **save_dict)

        return output_file

    def _export_pickle(self, dataset_name: str, window_size: int,
                      windows: List[ProcessedWindow],
                      window_features: List[WindowFeatures],
                      captions: List[Caption],
                      feature_vocab: Dict[str, List[str]],
                      feature_mappings: Dict[str, Dict[str, int]],
                      metadata: Dict[str, Any], is_presegmented: bool = False) -> Path:
        """Export as pickle file."""

        base_name = self._get_base_name(dataset_name, window_size, is_presegmented)
        output_file = self.output_dir / f"{base_name}.pkl"

        data = {
            'windows': windows,
            'window_features': window_features,
            'captions': captions,
            'feature_vocab': feature_vocab,
            'feature_mappings': feature_mappings,
            'metadata': metadata,
            'dataset_name': dataset_name,
            'window_size': window_size
        }

        if self.config.compress:
            output_file = output_file.with_suffix('.pkl.gz')
            with gzip.open(output_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(output_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return output_file

    def _export_txt(self, dataset_name: str, window_size: int,
                   windows: List[ProcessedWindow],
                   captions: List[Caption], is_presegmented: bool = False) -> Path:
        """Export captions as text files."""

        base_name = self._get_base_name(dataset_name, window_size, is_presegmented)
        output_file = self.output_dir / f"{base_name}_captions.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Captions for {dataset_name} (window_size={window_size})\n")
            f.write(f"# Format: window_id | caption_type | text\n\n")

            for caption in captions:
                f.write(f"{caption.window_id} | {caption.caption_type} | {caption.text}\n")

        return output_file

    def _export_hdf5(self, dataset_name: str, window_size: int,
                    windows: List[ProcessedWindow],
                    window_features: List[WindowFeatures],
                    captions: List[Caption],
                    feature_vocab: Dict[str, List[str]],
                    feature_mappings: Dict[str, Dict[str, int]],
                    metadata: Dict[str, Any], is_presegmented: bool = False) -> Path:
        """Export as HDF5 file."""

        if not HAS_HDF5:
            raise ImportError("h5py is required for HDF5 export. Install with: pip install h5py")

        base_name = self._get_base_name(dataset_name, window_size, is_presegmented)
        output_file = self.output_dir / f"{base_name}.h5"

        with h5py.File(output_file, 'w') as f:
            # Window data
            windows_group = f.create_group('windows')

            for i, (window, features) in enumerate(zip(windows, window_features)):
                window_group = windows_group.create_group(f'window_{window.metadata.window_id}')

                # Window metadata
                meta_group = window_group.create_group('metadata')
                meta_dict = asdict(window.metadata)
                for key, value in meta_dict.items():
                    if value is not None:
                        if isinstance(value, (list, tuple)):
                            meta_group.create_dataset(key, data=np.array(value, dtype=object))
                        else:
                            meta_group.create_dataset(key, data=value)

                # Event data
                events_df = window.events
                events_group = window_group.create_group('events')
                for col in events_df.columns:
                    events_group.create_dataset(col, data=events_df[col].values)

            # Captions
            captions_group = f.create_group('captions')
            caption_texts = [c.text for c in captions]
            caption_window_ids = [c.window_id for c in captions]
            caption_types = [c.caption_type for c in captions]

            captions_group.create_dataset('texts', data=np.array(caption_texts, dtype='S'))
            captions_group.create_dataset('window_ids', data=caption_window_ids)
            captions_group.create_dataset('types', data=np.array(caption_types, dtype='S'))

            # Metadata
            meta_group = f.create_group('metadata')
            for key, value in metadata.items():
                if isinstance(value, dict):
                    subgroup = meta_group.create_group(key)
                    for subkey, subvalue in value.items():
                        subgroup.create_dataset(subkey, data=subvalue)
                elif isinstance(value, (list, tuple)):
                    meta_group.create_dataset(key, data=np.array(value, dtype=object))
                else:
                    meta_group.create_dataset(key, data=value)

        return output_file

    def _windows_to_numpy(self, windows: List[ProcessedWindow],
                         window_features: List[WindowFeatures],
                         feature_mappings: Dict[str, Dict[str, int]]) -> Dict[str, np.ndarray]:
        """Convert windows to numpy arrays."""

        window_ids = []
        all_event_features = []
        window_metadata = []

        for window, features in zip(windows, window_features):
            window_ids.append(window.metadata.window_id)

            # Convert window metadata to dict
            meta_dict = asdict(window.metadata)
            # Convert non-serializable types
            for key, value in meta_dict.items():
                if hasattr(value, 'isoformat'):  # datetime
                    meta_dict[key] = value.isoformat()
                elif isinstance(value, (list, tuple)):
                    meta_dict[key] = str(value)
            window_metadata.append(meta_dict)

            # Convert events to feature arrays
            window_events = []
            for _, event in window.events.iterrows():
                event_features = {
                    'sensor_id': feature_mappings['sensor_id'].get(event.get('sensor', ''), 0),
                    'room_id': feature_mappings['room_id'].get(event.get('room_id', ''), 0),
                    'event_type': feature_mappings['event_type'].get(event.get('event_type', ''), 0),
                    'sensor_type': feature_mappings['sensor_type'].get(event.get('sensor_type', ''), 0),
                    'tod_bucket': feature_mappings['tod_bucket'].get(event.get('tod_bucket', ''), 0),
                    'delta_t_bucket': feature_mappings['delta_t_bucket'].get(event.get('time_delta_bucket', ''), 0),
                    'hour': event.get('hour', 0),
                    'day_of_week': event.get('day_of_week', 0),
                    'x_coord_norm': event.get('x_coord_norm', 0.0),
                    'y_coord_norm': event.get('y_coord_norm', 0.0)
                }
                window_events.append(event_features)

            all_event_features.append(window_events)

        return {
            'window_ids': np.array(window_ids),
            'event_features': np.array(all_event_features, dtype=object),
            'window_metadata': np.array(window_metadata, dtype=object)
        }

    def _export_json(self, dataset_name: str, window_size: int,
                    windows: List[ProcessedWindow],
                    window_features: List[WindowFeatures],
                    captions: List[Caption],
                    feature_vocab: Dict[str, Any],
                    feature_mappings: Dict[str, Dict[str, int]],
                    metadata: Dict[str, Any], is_presegmented: bool = False) -> Path:
        """Export data in JSON format for training."""

        # Create training data format
        training_data = []

        # Group captions by window_id to handle multiple captions per window
        captions_by_window = {}
        for caption in captions:
            window_id = caption.window_id
            if window_id not in captions_by_window:
                captions_by_window[window_id] = {'long': [], 'short': [], 'all': []}

            # Categorize caption based on its type (set during generation)
            if caption.caption_type == 'short_creative':
                captions_by_window[window_id]['short'].append(caption.text)
            else:
                # 'enhanced' and 'basic' are considered long captions
                captions_by_window[window_id]['long'].append(caption.text)
            captions_by_window[window_id]['all'].append(caption.text)

        for window, window_features_obj in zip(windows, window_features):
            # Extract events from window
            events = []

            for i, event_features in enumerate(window_features_obj.events):
                # Build event dict
                event_dict = {
                    'sensor_id': event_features.sensor_id,
                    'room_id': event_features.room_id,
                    'event_type': event_features.event_type,
                    'sensor_type': event_features.sensor_type,
                    'tod_bucket': event_features.tod_bucket,
                    'delta_t_bucket': event_features.delta_t_bucket,
                    'x': float(event_features.x_coord) if event_features.x_coord is not None else 0.0,
                    'y': float(event_features.y_coord) if event_features.y_coord is not None else 0.0,
                    'timestamp': float(i),  # Use index as timestamp for now
                    'delta_t_since_prev': 1.0  # Default time delta
                }

                # Add optional fields
                if hasattr(event_features, 'floor_id'):
                    event_dict['floor_id'] = event_features.floor_id
                if hasattr(event_features, 'dow'):
                    event_dict['dow'] = event_features.dow

                events.append(event_dict)

            # Get all captions for this window
            window_id = window.metadata.window_id
            window_caption_data = captions_by_window.get(window_id, {'long': [''], 'short': [''], 'all': ['']})

            # Create training sample with separated caption types
            sample = {
                'events': events,
                'captions': window_caption_data['all'],  # All captions (for backward compatibility)
                'long_captions': window_caption_data['long'],  # Long captions only
                'short_captions': window_caption_data['short'],  # Short captions only
                'first_activity': window.metadata.primary_activity,
                'first_activity_l2': window.metadata.primary_activity_l2
            }

            training_data.append(sample)

        # Split into train/test (80/20)
        split_idx = int(len(training_data) * 0.8)
        train_data = training_data[:split_idx]
        test_data = training_data[split_idx:]

        # Generate base names with presegmented suffix if needed
        base_name = self._get_base_name(dataset_name, window_size, is_presegmented)
        train_base = base_name.replace(f"_w{window_size}", "")

        # Export training data
        train_file = self.output_dir / f"{train_base}_train.json"
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2, default=str)

        # Export test data
        test_file = self.output_dir / f"{train_base}_test.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2, default=str)

        # Create vocabulary for training
        vocab = {}
        for field, field_vocab in feature_vocab.items():
            if hasattr(field_vocab, 'vocab'):
                vocab[field] = field_vocab.vocab
            elif isinstance(field_vocab, dict):
                vocab[field] = field_vocab
            else:
                # Create enumeration from list
                vocab[field] = {str(val): i for i, val in enumerate(field_vocab)}

            # Add special tokens
            vocab[field]['<UNK>'] = len(vocab[field])
            vocab[field]['<PAD>'] = len(vocab[field])

        # Export vocabulary
        vocab_file = self.output_dir / f"{train_base}_vocab.json"
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f, indent=2)

        print(f"    Training data: {len(train_data)} samples -> {train_file}")
        print(f"    Test data: {len(test_data)} samples -> {test_file}")
        print(f"    Vocabulary: {vocab_file}")

        return train_file  # Return the main training file

    def export_summary_statistics(self, dataset_name: str,
                                 processing_stats: Dict[str, Any],
                                 window_stats: Dict[int, Dict[str, Any]],
                                 caption_stats: Dict[str, Any]) -> Path:
        """Export summary statistics as JSON."""

        output_file = self.output_dir / f"{dataset_name}_statistics.json"

        summary = {
            'dataset_name': dataset_name,
            'processing_stats': processing_stats,
            'window_stats': window_stats,
            'caption_stats': caption_stats,
            'export_timestamp': pd.Timestamp.now().isoformat()
        }

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"  Exported statistics: {output_file}")
        return output_file

    def _export_debug_json(self, dataset_name: str, window_size: int,
                          windows: List[ProcessedWindow],
                          window_features: List[WindowFeatures],
                          captions: List[Caption],
                          feature_vocab: Dict[str, Any],
                          feature_mappings: Dict[str, Dict[str, int]],
                          metadata: Dict[str, Any], is_presegmented: bool = False) -> Path:
        """Export lightweight debug data in JSON format - only captions and metadata."""

        # Create lightweight debug data format
        debug_data = []

        # Group captions by window_id
        captions_by_window = {}
        for caption in captions:
            window_id = caption.window_id
            if window_id not in captions_by_window:
                captions_by_window[window_id] = {'long': [], 'short': [], 'all': []}

            # Categorize caption based on its type
            if caption.caption_type == 'short_creative':
                captions_by_window[window_id]['short'].append(caption.text)
            else:
                captions_by_window[window_id]['long'].append(caption.text)
            captions_by_window[window_id]['all'].append(caption.text)

        for window in windows:
            window_id = window.metadata.window_id
            window_caption_data = captions_by_window.get(window_id, {'long': [''], 'short': [''], 'all': ['']})

            # Create lightweight debug sample - only essential info
            sample = {
                'window_id': window_id,
                'split': window.metadata.split,
                'captions': window_caption_data['all'],
                'long_captions': window_caption_data['long'],
                'short_captions': window_caption_data['short'],
                'first_activity': window.metadata.primary_activity,
                'first_activity_l2': window.metadata.primary_activity_l2,
                'duration_sec': window.metadata.duration_sec,
                'num_events': len(window.events),
                'rooms_visited': list(window.metadata.rooms_visited) if window.metadata.rooms_visited else [],
                'start_time': str(window.events.iloc[0]['datetime']) if not window.events.empty else '',
                'end_time': str(window.events.iloc[-1]['datetime']) if not window.events.empty else '',
                'date': str(window.events.iloc[0]['datetime'].date()) if not window.events.empty else '',
                'hour': window.events.iloc[0]['datetime'].hour if not window.events.empty else 0,
                'dow': window.events.iloc[0]['datetime'].strftime('%A') if not window.events.empty else ''
            }

            debug_data.append(sample)

        # Split into train/test (80/20)
        split_idx = int(len(debug_data) * 0.8)
        train_data = debug_data[:split_idx]
        test_data = debug_data[split_idx:]

        # Generate base names with debug prefix
        base_name = self._get_base_name(dataset_name, window_size, is_presegmented)
        train_base = base_name.replace(f"_w{window_size}", "")

        # Add debug prefix
        debug_prefix = "debug_"

        # Export debug training data
        train_file = self.output_dir / f"{debug_prefix}{train_base}_train.json"
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2, default=str)

        # Export debug test data
        test_file = self.output_dir / f"{debug_prefix}{train_base}_test.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2, default=str)

        # Create summary with label distribution
        label_counts = {}
        for sample in debug_data:
            label = sample['first_activity']
            label_counts[label] = label_counts.get(label, 0) + 1

        debug_summary = {
            'dataset_name': dataset_name,
            'filter_applied': True,
            'total_samples': len(debug_data),
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'label_distribution': label_counts,
            'sample_captions': [sample['captions'][0] for sample in debug_data[:5] if sample['captions']],
            'export_timestamp': pd.Timestamp.now().isoformat()
        }

        summary_file = self.output_dir / f"{debug_prefix}{train_base}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(debug_summary, f, indent=2, default=str)

        print(f"    Debug training data: {len(train_data)} samples -> {train_file}")
        print(f"    Debug test data: {len(test_data)} samples -> {test_file}")
        print(f"    Debug summary: {summary_file}")

        return train_file
