"""Main processing pipeline for dual-encoder alignment."""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from dataclasses import asdict
from tqdm import tqdm

from .data_config import ProcessingConfig
from .datasets import get_dataset_config
from .data_loader import DataLoader, ProcessedDataset
from .windowing import WindowProcessor, ProcessedWindow
from .features import FeatureExtractor, WindowFeatures, EventFeatures
from .captions import CaptionGenerator, Caption
from .captions_sourish import SourishCaptionGenerator
from .exporters import DataExporter


class DualEncoderPipeline:
    """Main processing pipeline for dual-encoder alignment."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.dataset_config = get_dataset_config(config.dataset_name)

        # Initialize components
        self.data_loader = DataLoader(config)
        self.window_processor = WindowProcessor(config.windowing)
        self.feature_extractor = FeatureExtractor(config.features, self.dataset_config)

        # Select caption generator based on style
        caption_style = config.captions.caption_style if hasattr(config.captions, 'caption_style') else 'baseline'
        if caption_style == 'sourish':
            # Pass dataset name to Sourish generator for dataset-specific rules
            caption_config = config.captions
            self.caption_generator = SourishCaptionGenerator(caption_config)
            self.caption_generator.dataset_name = config.dataset_name
            print(f"  Using Sourish-style caption generator")
        else:
            self.caption_generator = CaptionGenerator(config.captions)
            print(f"  Using baseline caption generator")

        self.exporter = DataExporter(config.export)

        # Results storage
        self.dataset = None
        self.windows_by_size = {}
        self.features_by_size = {}
        self.captions_by_size = {}
        self.feature_vocab = None
        self.feature_mappings = None

    def process(self) -> Dict[str, Any]:
        """Run the complete processing pipeline."""
        print(f"Starting dual-encoder processing for {self.config.dataset_name}")

        # Step 1: Load and preprocess data
        print("\n1. Loading and preprocessing data...")
        self.dataset = self.data_loader.load_and_process()

        # Step 2: Create windows
        print("\n2. Creating windows...")
        self.windows_by_size = self.window_processor.process_dataset(
            self.dataset.train_df, self.dataset.test_df,
            max_windows=self.config.max_windows
        )

        # Step 3: Extract features
        print("\n3. Extracting features...")
        self.features_by_size, self.feature_vocab, self.feature_mappings = self._extract_all_features()

        # Step 4: Generate captions
        print("\n4. Generating captions...")
        self.captions_by_size = self._generate_all_captions()

        # Step 5: Compute statistics
        print("\n5. Computing statistics...")
        processing_stats = self._compute_processing_statistics()

        print(f"\nProcessing complete for {self.config.dataset_name}")
        return processing_stats

    def export_all(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Export all processed data."""
        if output_dir:
            self.config.export.output_dir = output_dir

        self.exporter = DataExporter(self.config.export)

        print(f"\n📁 Exporting data to {self.config.export.output_dir}")

        exported_files = {}

        # Export each window size separately
        for window_size in self.config.windowing.sizes:
            print(f"\nExporting window_size={window_size}...")

            if window_size not in self.windows_by_size:
                print(f"  Skipping window_size={window_size} (no data)")
                continue

            windows = self.windows_by_size[window_size]
            features = self.features_by_size[window_size]
            captions = self.captions_by_size[window_size]

            # Check if this is presegmented data
            is_presegmented = (self.config.windowing.strategy.value == "presegmented")

            # Check if we're in debug mode (label filtering)
            debug_mode = hasattr(self.config, 'filter_labels') and self.config.filter_labels is not None

            # Export in requested formats
            size_files = self.exporter.export_all_formats(
                dataset_name=self.config.dataset_name,
                window_size=window_size,
                windows=windows,
                window_features=features,
                captions=captions,
                feature_vocab=self.feature_vocab,
                feature_mappings=self.feature_mappings,
                metadata=self.dataset.metadata,
                is_presegmented=is_presegmented,
                debug_mode=debug_mode
            )

            exported_files[window_size] = size_files

        # Export summary statistics
        processing_stats = self._compute_processing_statistics()
        window_stats = self.window_processor.get_statistics(self.windows_by_size)
        caption_stats = self._compute_caption_statistics()

        stats_file = self.exporter.export_summary_statistics(
            self.config.dataset_name, processing_stats, window_stats, caption_stats
        )

        exported_files['statistics'] = str(stats_file)

        print(f"\nExport complete!")
        return exported_files

    def _extract_all_features(self) -> Tuple[Dict[int, List[WindowFeatures]],
                                           Dict[str, List[str]],
                                           Dict[str, Dict[str, int]]]:
        """Extract features for all window sizes."""
        features_by_size = {}
        all_event_features = []

        # Extract features for each window size
        for window_size, windows in self.windows_by_size.items():
            print(f"  Extracting features for window_size={window_size} ({len(windows)} windows)")

            window_features = []
            for window in tqdm(windows, desc=f"Extracting features (size={window_size})", leave=False):
                features = self.feature_extractor.extract_window_features(
                    window.metadata.window_id, window.events
                )
                window_features.append(features)

                # Collect all event features for vocabulary building
                all_event_features.extend(features.events)

            features_by_size[window_size] = window_features

        # Build feature vocabulary and mappings
        print("  Building feature vocabulary...")
        feature_vocab = self.feature_extractor.create_discrete_feature_vocab(all_event_features)
        feature_mappings = self.feature_extractor.create_feature_mappings(feature_vocab)

        vocab_sizes = {k: len(v) for k, v in feature_vocab.items()}
        print(f"  Feature vocabulary sizes: {vocab_sizes}")

        return features_by_size, feature_vocab, feature_mappings

    def _generate_all_captions(self) -> Dict[int, List[Caption]]:
        """Generate captions for all window sizes."""
        captions_by_size = {}

        for window_size in self.config.windowing.sizes:
            if window_size not in self.windows_by_size:
                continue

            windows = self.windows_by_size[window_size]
            features = self.features_by_size[window_size]

            print(f"  Generating captions for window_size={window_size} ({len(windows)} windows)")

            all_captions = []

            for window, window_features in tqdm(zip(windows, features), desc=f"Generating captions (size={window_size})", leave=False, total=len(windows)):
                # Pass sensor details if available, otherwise None for basic captions
                sensor_details = self.dataset_config.sensor_details if self.dataset_config.sensor_details else None
                caption_dict = self.caption_generator.generate_captions(window, window_features, sensor_details)

                # Always store all captions (both long and short) for proper export
                # The dataset loader will filter based on caption_types during training
                all_captions.extend(caption_dict['all'])

            captions_by_size[window_size] = all_captions
            print(f"    Generated {len(all_captions)} captions")

            # Print random sample captions for inspection to show diversification
            print(f"    Sample captions for window_size={window_size}:")
            import random
            num_samples = min(5, len(all_captions))
            random.seed(42)  # For reproducible examples
            sample_indices = random.sample(range(len(all_captions)), num_samples)

            for i, idx in enumerate(sample_indices):
                caption = all_captions[idx]
                print(f"      [{i+1}] {caption.text}")
                if hasattr(caption, 'layer_b') and caption.layer_b:
                    print(f"          Layer B: {caption.layer_b[:100]}...")
            print()

        return captions_by_size

    def _compute_processing_statistics(self) -> Dict[str, Any]:
        """Compute overall processing statistics."""
        stats = {
            'dataset_name': self.config.dataset_name,
            'config': asdict(self.config),
            'dataset_metadata': self.dataset.metadata if self.dataset else {},
            'window_sizes_processed': list(self.windows_by_size.keys()),
            'total_windows': sum(len(windows) for windows in self.windows_by_size.values()),
            'total_captions': sum(len(captions) for captions in self.captions_by_size.values())
        }

        # Add per-size statistics
        for window_size in self.config.windowing.sizes:
            if window_size in self.windows_by_size:
                windows = self.windows_by_size[window_size]
                captions = self.captions_by_size.get(window_size, [])

                # Split statistics
                train_windows = [w for w in windows if w.metadata.split == 'train']
                test_windows = [w for w in windows if w.metadata.split == 'test']

                stats[f'window_size_{window_size}'] = {
                    'total_windows': len(windows),
                    'train_windows': len(train_windows),
                    'test_windows': len(test_windows),
                    'total_captions': len(captions),
                    'avg_captions_per_window': len(captions) / len(windows) if windows else 0
                }

        return stats

    def _compute_caption_statistics(self) -> Dict[str, Any]:
        """Compute caption statistics across all window sizes."""
        all_captions = []
        for captions in self.captions_by_size.values():
            all_captions.extend(captions)

        if not all_captions:
            return {}

        return self.caption_generator.get_caption_statistics(all_captions)

    def get_sample_data(self, window_size: int = None, num_samples: int = 5) -> Dict[str, Any]:
        """Get sample data for inspection."""
        if window_size is None:
            window_size = self.config.windowing.sizes[0]

        if window_size not in self.windows_by_size:
            return {}

        windows = self.windows_by_size[window_size][:num_samples]
        features = self.features_by_size[window_size][:num_samples]
        captions = [c for c in self.captions_by_size[window_size]
                   if c.window_id in [w.metadata.window_id for w in windows]]

        samples = []
        for window, window_features in zip(windows, features):
            window_captions = [c for c in captions if c.window_id == window.metadata.window_id]

            sample = {
                'window_id': window.metadata.window_id,
                'split': window.metadata.split,
                'size': window.size,
                'duration_sec': window.metadata.duration_sec,
                'primary_activity': window.metadata.primary_activity_l2,
                'rooms_visited': window.metadata.rooms_visited,
                'num_events': len(window.events),
                'captions': [c.text for c in window_captions],
                'sample_events': window.events[['datetime', 'sensor', 'room_id', 'state', 'first_activity_l2']].head(3).to_dict('records')
            }
            samples.append(sample)

        return {
            'window_size': window_size,
            'num_samples': len(samples),
            'samples': samples
        }
