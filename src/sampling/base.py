"""
Base sampler class for all data sampling strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class Sample:
    """A single training/testing sample."""
    sample_id: str
    sensor_sequence: List[Dict[str, Any]]  # List of sensor events
    metadata: Dict[str, Any]  # Window metadata (duration, rooms, labels, etc.)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'sample_id': self.sample_id,
            'sensor_sequence': self.sensor_sequence,
            'metadata': self.metadata
        }


@dataclass
class SamplingResult:
    """Result of a sampling operation."""
    dataset_name: str
    sampling_strategy: str
    sampling_params: Dict[str, Any]
    split: str  # 'train' or 'test'
    samples: List[Sample]
    statistics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'dataset': self.dataset_name,
            'sampling_strategy': self.sampling_strategy,
            'sampling_params': self.sampling_params,
            'split': self.split,
            'samples': [s.to_dict() for s in self.samples],
            'statistics': self.statistics
        }

    def save_json(self, output_path: Path):
        """Save to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Saved {len(self.samples)} samples to {output_path}")


class BaseSampler(ABC):
    """Abstract base class for all sampling strategies."""

    def __init__(self, config):
        """Initialize sampler with configuration.

        Args:
            config: A SamplingConfig (or subclass) instance
        """
        self.config = config
        self._sample_id_counter = 0
        self.sensor_details_map = None  # Will be loaded from metadata if available

        # Load metadata if provided
        if config.metadata_path and config.metadata_path.exists():
            self._load_metadata()

    def sample_dataset(self) -> Tuple[SamplingResult, SamplingResult, SamplingResult]:
        """Main entry point for sampling a dataset.

        Returns:
            Tuple of (train_result, val_result, test_result)
        """
        print(f"\n{'='*60}")
        print(f"Starting {self.config.strategy.value} sampling")
        print(f"Dataset: {self.config.dataset_name}")
        print(f"{'='*60}\n")

        # Step 1: Load raw data
        print("Step 1: Loading raw data...")
        df = self._load_raw_data()
        print(f"  Loaded {len(df)} events")

        # Step 2: Train/val/test split by days (70/10/20)
        print("\nStep 2: Splitting into train/val/test...")
        train_df, val_df, test_df = self._split_by_days(df)
        print(f"  Train: {len(train_df)} events")
        print(f"  Val: {len(val_df)} events")
        print(f"  Test: {len(test_df)} events")

        # Step 3: Apply sampling strategy
        print("\nStep 3: Creating windows...")
        train_samples = self._create_samples(train_df, split='train')
        print(f"  Created {len(train_samples)} train samples")

        self._sample_id_counter = 0  # Reset for val set
        val_samples = self._create_samples(val_df, split='val')
        print(f"  Created {len(val_samples)} val samples")

        self._sample_id_counter = 0  # Reset for test set
        test_samples = self._create_samples(test_df, split='test')
        print(f"  Created {len(test_samples)} test samples")

        # Step 4: Compute statistics
        print("\nStep 4: Computing statistics...")
        train_stats = self._compute_statistics(train_samples)
        val_stats = self._compute_statistics(val_samples)
        test_stats = self._compute_statistics(test_samples)

        # Step 5: Create results
        sampling_params = self._get_sampling_params()

        train_result = SamplingResult(
            dataset_name=self.config.dataset_name,
            sampling_strategy=self.config.strategy.value,
            sampling_params=sampling_params,
            split='train',
            samples=train_samples,
            statistics=train_stats
        )

        val_result = SamplingResult(
            dataset_name=self.config.dataset_name,
            sampling_strategy=self.config.strategy.value,
            sampling_params=sampling_params,
            split='val',
            samples=val_samples,
            statistics=val_stats
        )

        test_result = SamplingResult(
            dataset_name=self.config.dataset_name,
            sampling_strategy=self.config.strategy.value,
            sampling_params=sampling_params,
            split='test',
            samples=test_samples,
            statistics=test_stats
        )

        print(f"\n{'='*60}")
        print("Sampling complete!")
        print(f"{'='*60}\n")

        return train_result, val_result, test_result

    @abstractmethod
    def _create_windows_for_dataframe(self, df: pd.DataFrame, split: str) -> List[Dict[str, Any]]:
        """Create windows from a dataframe using specific sampling strategy.

        This is the core method that each sampler must implement.

        Args:
            df: DataFrame with sensor events
            split: 'train' or 'test'

        Returns:
            List of window dictionaries with 'events' and 'metadata' keys
        """
        pass

    @abstractmethod
    def _get_sampling_params(self) -> Dict[str, Any]:
        """Get sampling parameters for this strategy."""
        pass

    def _load_metadata(self):
        """Load sensor metadata from JSON file."""
        import json

        print(f"  Loading metadata from: {self.config.metadata_path}")
        with open(self.config.metadata_path, 'r') as f:
            metadata = json.load(f)

        # Extract sensor_details for the dataset
        dataset_metadata = metadata.get(self.config.dataset_name, {})
        sensor_location = dataset_metadata.get('sensor_location', {})
        sensor_details = dataset_metadata.get('sensor_details', {})

        # Build map: sensor_id -> sensor_detail (only if detail is different from location and not null)
        self.sensor_details_map = {}
        for sensor_id, detail in sensor_details.items():
            # Include if detail exists, is not null, and is different from location
            if detail and pd.notna(detail):
                location = sensor_location.get(sensor_id, '')
                if detail != location:
                    self.sensor_details_map[sensor_id] = detail

        print(f"  Found {len(self.sensor_details_map)} sensors with special details")

    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw sensor data from file.

        Returns:
            DataFrame with standardized columns: ['timestamp', 'sensor_id', 'event_type', 'room', ...]
        """
        import sys
        from pathlib import Path as ImportPath
        # Add parent directory for imports
        parent_dir = ImportPath(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from data.data_load_clean import casas_end_to_end_preprocess
        from .utils import standardize_column_names

        # Use existing data loading infrastructure
        df = casas_end_to_end_preprocess(
            dataset_name=self.config.dataset_name,
            max_lines=self.config.max_lines,
            save_to_csv=False  # Don't save intermediate files
        )

        # Standardize column names for consistent usage downstream
        df = standardize_column_names(df)

        # Add sensor_details column if metadata is available
        if self.sensor_details_map is not None:
            df['sensor_detail'] = df['sensor_id'].map(self.sensor_details_map)
            num_with_details = df['sensor_detail'].notna().sum()
            print(f"  Added sensor_detail column ({num_with_details} events with special sensors)")

        # Filter numeric sensors if requested
        if self.config.filter_numeric_sensors:
            df = self._filter_numeric_sensors(df)

        return df

    def _filter_numeric_sensors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out temperature and other numeric sensors."""
        # Use is_numeric column (standardized from data_load_clean)
        if 'is_numeric' in df.columns:
            filtered_df = df[~df['is_numeric']].copy()
            if len(filtered_df) < len(df):
                print(f"  Filtered {len(df) - len(filtered_df)} numeric sensor events")
            return filtered_df

        return df

    def _split_by_days(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train/val/test by days (70/10/20)."""
        from .config import SplitStrategy

        # Get unique dates (timestamp column is now standardized)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        unique_dates = sorted(df['date'].unique())

        n_dates = len(unique_dates)
        n_train = int(n_dates * 0.7)  # 70% for training
        n_val = int(n_dates * 0.1)    # 10% for validation
        # Remaining ~20% for test

        if self.config.split_strategy == SplitStrategy.RANDOM:
            # Random split
            np.random.seed(self.config.random_seed)
            shuffled_dates = np.random.choice(unique_dates, size=n_dates, replace=False)
            train_dates = shuffled_dates[:n_train]
            val_dates = shuffled_dates[n_train:n_train+n_val]
            test_dates = shuffled_dates[n_train+n_val:]
        else:
            # Temporal split (first 70% for train, next 10% for val, last 20% for test)
            train_dates = unique_dates[:n_train]
            val_dates = unique_dates[n_train:n_train+n_val]
            test_dates = unique_dates[n_train+n_val:]

        train_df = df[df['date'].isin(train_dates)].copy()
        val_df = df[df['date'].isin(val_dates)].copy()
        test_df = df[df['date'].isin(test_dates)].copy()

        # Remove temporary date column
        train_df = train_df.drop('date', axis=1)
        val_df = val_df.drop('date', axis=1)
        test_df = test_df.drop('date', axis=1)

        return train_df, val_df, test_df

    def _create_samples(self, df: pd.DataFrame, split: str) -> List[Sample]:
        """Create samples from dataframe using the specific windowing strategy."""
        # Call the abstract method to get windows
        windows = self._create_windows_for_dataframe(df, split)

        # Convert windows to Sample objects
        samples = []
        for window in windows:
            sample_id = f"{self.config.dataset_name}_{split}_{self._sample_id_counter:06d}"
            self._sample_id_counter += 1

            sample = Sample(
                sample_id=sample_id,
                sensor_sequence=window['events'],
                metadata=window['metadata']
            )
            samples.append(sample)

        # Apply max_windows limit if specified
        if self.config.max_windows and len(samples) > self.config.max_windows:
            samples = samples[:self.config.max_windows]

        return samples

    def _compute_statistics(self, samples: List[Sample]) -> Dict[str, Any]:
        """Compute statistics for a list of samples."""
        if not samples:
            return {}

        durations = [s.metadata.get('duration_seconds', 0) for s in samples]
        num_events = [len(s.sensor_sequence) for s in samples]

        stats = {
            'total_samples': len(samples),
            'avg_sequence_length': float(np.mean(num_events)),
            'std_sequence_length': float(np.std(num_events)),
            'min_sequence_length': int(np.min(num_events)),
            'max_sequence_length': int(np.max(num_events)),
            'avg_duration_seconds': float(np.mean(durations)),
            'std_duration_seconds': float(np.std(durations)),
            'min_duration_seconds': float(np.min(durations)),
            'max_duration_seconds': float(np.max(durations)),
        }

        # Collect room statistics
        all_rooms = []
        for s in samples:
            rooms = s.metadata.get('rooms_visited', [])
            all_rooms.extend(rooms)

        if all_rooms:
            from collections import Counter
            room_counts = Counter(all_rooms)
            stats['room_distribution'] = dict(room_counts.most_common(10))

        return stats

    def _event_to_dict(self, row: pd.Series, preserve_full: bool = True) -> Dict[str, Any]:
        """Convert a DataFrame row to an event dictionary."""
        # Use standardized column names
        event = {
            'sensor_id': row.get('sensor_id', 'unknown'),
            'event_type': row.get('event_type', 'unknown'),
            'timestamp': str(row.get('timestamp', '')),
        }

        if preserve_full:
            # Add all available metadata (column names are now standardized)
            optional_fields = [
                'room', 'sensor_type', 'sensor_detail', 'x', 'y', 'floor',
                'activity_l1', 'activity_l2', 'activity_full'
            ]

            for field in optional_fields:
                if field in row and pd.notna(row[field]):
                    event[field] = row[field]

        return event

    def _generate_vocabulary(self, samples: List[Sample]) -> Dict[str, Dict[str, int]]:
        """
        Generate vocabulary from samples.

        Extracts unique values for each categorical field and creates index mappings.
        Index 0 is reserved for UNK (unknown/padding).

        Args:
            samples: List of Sample objects

        Returns:
            Dictionary mapping field names to {value: index} dictionaries
        """
        from collections import defaultdict

        # Collect unique values for each field
        vocab_sets = defaultdict(set)

        for sample in samples:
            sensor_sequence = sample.sensor_sequence

            for event in sensor_sequence:
                for key, value in event.items():
                    # Skip non-categorical fields
                    if key in ['x', 'y', 'z', 'timestamp', 'datetime', 'time_delta']:
                        continue

                    # Normalize field names and collect values
                    if key in ['sensor', 'sensor_id']:
                        vocab_sets['sensor'].add(str(value))
                    elif key in ['state', 'event_type', 'message']:
                        vocab_sets['state'].add(str(value))
                    elif key in ['room', 'room_id', 'location']:
                        vocab_sets['room_id'].add(str(value))
                    elif key in ['activity', 'first_activity', 'activity_l1']:
                        vocab_sets['activity'].add(str(value))
                    elif key in ['activity_l2', 'first_activity_l2']:
                        vocab_sets['activity_l2'].add(str(value))
                    elif key in ['sensor_type', 'type']:
                        vocab_sets['sensor_type'].add(str(value))
                    else:
                        # Generic field
                        vocab_sets[key].add(str(value))

        # Convert sets to sorted lists and create index mappings
        vocab_dict = {}
        for field, values in vocab_sets.items():
            # Sort values for consistency
            sorted_values = sorted(list(values))

            # Create index mapping (0 is reserved for UNK)
            vocab_dict[field] = {value: idx + 1 for idx, value in enumerate(sorted_values)}

            # Add UNK token at index 0
            vocab_dict[field]['UNK'] = 0

        return vocab_dict

