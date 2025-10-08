"""Unified data loader for CASAS datasets."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import datetime
from sklearn.model_selection import train_test_split

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from .data_load_clean import casas_end_to_end_preprocess

from .datasets import get_dataset_config, DatasetConfig
from .data_config import ProcessingConfig, SplitStrategy


@dataclass
class ProcessedDataset:
    """Container for processed dataset with metadata."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    config: DatasetConfig
    processing_config: ProcessingConfig
    metadata: Dict[str, Any]

    @property
    def full_df(self) -> pd.DataFrame:
        """Get combined train + test dataframe."""
        return pd.concat([self.train_df, self.test_df], ignore_index=True)


class DataLoader:
    """Unified data loader for CASAS datasets."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.dataset_config = get_dataset_config(config.dataset_name)

    def load_and_process(self) -> ProcessedDataset:
        """Load and process a CASAS dataset."""
        print(f"Loading dataset: {self.config.dataset_name}")
        if self.config.max_lines is not None:
            print(f"🔍 DataLoader config has max_lines={self.config.max_lines}")

        # Load raw data using existing pipeline
        df = casas_end_to_end_preprocess(
            self.config.dataset_name,
            save_to_csv=False,
            force_download=False,
            max_lines=self.config.max_lines
        )

        # Basic cleaning and filtering
        df = self._clean_data(df)

        # Add enhanced metadata
        df = self._add_metadata(df)

        # Split into train/test
        train_df, test_df = self._train_test_split(df)

        # Compute dataset metadata
        metadata = self._compute_metadata(df, train_df, test_df)

        return ProcessedDataset(
            train_df=train_df,
            test_df=test_df,
            config=self.dataset_config,
            processing_config=self.config,
            metadata=metadata
        )

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter the raw data."""
        print(f"  Original data shape: {df.shape}")

        # Filter numeric sensors if requested
        if self.config.filter_numeric_sensors:
            df = df[~df['is_numeric']].copy()
            print(f"  After filtering numeric sensors: {df.shape}")

        # Filter out records without datetime
        df = df[df['datetime'].notna()].copy()

        # Filter by specific labels if requested (for debugging)
        if self.config.filter_labels:
            print(f"  Filtering by labels: {self.config.filter_labels}")
            if 'first_activity' in df.columns:
                original_shape = df.shape
                df = df[df['first_activity'].isin(self.config.filter_labels)].copy()
                print(f"  After label filtering: {df.shape} (removed {original_shape[0] - df.shape[0]} rows)")

                # Show label distribution after filtering
                if not df.empty:
                    label_counts = df['first_activity'].value_counts()
                    print(f"  Label distribution after filtering:")
                    for label, count in label_counts.items():
                        print(f"    {label}: {count}")
                else:
                    print(f"  WARNING: No data remaining after filtering by labels {self.config.filter_labels}")
            else:
                print(f"  WARNING: first_activity column not found, skipping label filtering")

        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)

        # Filter very short sequences if needed
        if self.config.min_sequence_length > 1:
            # This is a placeholder - actual implementation would need more logic
            pass

        print(f"  Final cleaned data shape: {df.shape}")
        return df

    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced metadata columns."""
        print("  Adding enhanced metadata...")

        df = df.copy()

        # Time-based features
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_week_name'] = df['datetime'].dt.day_name()

        # Time of day bucketing
        df['tod_bucket'] = df['hour'].apply(self._get_tod_bucket)

        # Time gaps between readings
        df['time_delta_sec'] = df['datetime'].diff().dt.total_seconds().fillna(0)
        df['time_delta_bucket'] = df['time_delta_sec'].apply(self._get_delta_t_bucket)

        # Sensor type (first character: M, D, T, etc.)
        df['sensor_type'] = df['sensor'].str[0]

        # Event type (ON/OFF or numeric value)
        df['event_type'] = df['state']

        # Room information
        if self.dataset_config.sensor_location:
            df['room_id'] = df['sensor'].map(self.dataset_config.sensor_location)
        else:
            df['room_id'] = 'unknown'

        # Sensor details (if available)
        if self.dataset_config.sensor_details:
            df['sensor_detail'] = df['sensor'].map(self.dataset_config.sensor_details)
        else:
            df['sensor_detail'] = ''

        # Spatial coordinates
        if self.config.features.use_coordinates and self.dataset_config.sensor_coordinates:
            coord_map = self.dataset_config.sensor_coordinates
            df['x_coord'] = df['sensor'].map(lambda s: coord_map.get(s, (0, 0))[0] if coord_map else 0)
            df['y_coord'] = df['sensor'].map(lambda s: coord_map.get(s, (0, 0))[1] if coord_map else 0)
        else:
            df['x_coord'] = 0
            df['y_coord'] = 0

            # Coordinates will be normalized later in feature extraction

        # Activity segmentation information
        if self.config.use_pre_segmentation:
            df = self._add_segmentation_info(df)

        return df

    def _add_segmentation_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add activity segmentation information."""
        # Create segment IDs based on activity changes
        df['activity_changed'] = (df['first_activity'] != df['first_activity'].shift(1))
        df['segment_id'] = df['activity_changed'].cumsum()

        # Add segment metadata
        segment_info = df.groupby('segment_id').agg({
            'first_activity': 'first',
            'first_activity_l2': 'first',
            'datetime': ['first', 'last', 'count']
        }).reset_index()

        segment_info.columns = [
            'segment_id', 'segment_activity', 'segment_activity_l2',
            'segment_start', 'segment_end', 'segment_length'
        ]

        # Merge back
        df = df.merge(segment_info, on='segment_id', how='left')

        return df

    def _get_tod_bucket(self, hour: int) -> str:
        """Convert hour to time-of-day bucket."""
        if self.config.features.tod_bucketing.value == "parts_of_day":
            if 5 <= hour < 8:
                return 'early_morning'
            elif 8 <= hour < 12:
                return 'late_morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 21:
                return 'evening'
            elif 21 <= hour < 24:
                return 'night_before_midnight'
            else:  # 0 <= hour < 5
                return 'night_after_midnight'
        elif self.config.features.tod_bucketing.value == "hourly":
            return f"h{hour:02d}"
        elif self.config.features.tod_bucketing.value == "bi_hourly":
            return f"h{hour//2*2:02d}-{hour//2*2+1:02d}"
        elif self.config.features.tod_bucketing.value == "quad_hourly":
            return f"h{hour//4*4:02d}-{hour//4*4+3:02d}"
        else:
            return f"h{hour:02d}"

    def _get_delta_t_bucket(self, delta_sec: float) -> str:
        """Convert time delta to bucket."""
        if pd.isna(delta_sec) or delta_sec <= 0:
            return 'dt_0'

        # Convert to minutes
        delta_min = delta_sec / 60

        # Cap at max_delta_t_minutes
        delta_min = min(delta_min, self.config.features.max_delta_t_minutes)

        # Log-spaced buckets
        if delta_min <= 1:
            return 'dt_lt1min'
        elif delta_min <= 2:
            return 'dt_1-2min'
        elif delta_min <= 5:
            return 'dt_2-5min'
        elif delta_min <= 10:
            return 'dt_5-10min'
        elif delta_min <= 30:
            return 'dt_10-30min'
        else:
            return 'dt_gt30min'

    def _train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        print(f"  Splitting data (test_size={self.config.test_size}, strategy={self.config.split_strategy})")

        if self.config.split_strategy == SplitStrategy.TEMPORAL:
            # Temporal split: use first X% for training, last Y% for testing
            split_date = df['date'].quantile(1 - self.config.test_size)
            train_df = df[df['date'] <= split_date].copy()
            test_df = df[df['date'] > split_date].copy()

        elif self.config.split_strategy == SplitStrategy.RANDOM:
            # Random split by days: randomly select X% of days for testing
            unique_dates = df['date'].unique()
            train_dates, test_dates = train_test_split(
                unique_dates,
                test_size=self.config.test_size,
                random_state=self.config.random_seed
            )
            train_df = df[df['date'].isin(train_dates)].copy()
            test_df = df[df['date'].isin(test_dates)].copy()

        else:
            raise ValueError(f"Unknown split strategy: {self.config.split_strategy}")

        # Reset indices
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        print(f"  Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        print(f"  Train date range: {train_df['date'].min()} to {train_df['date'].max()}")
        print(f"  Test date range: {test_df['date'].min()} to {test_df['date'].max()}")

        return train_df, test_df

    def _compute_metadata(self, df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute dataset metadata."""
        metadata = {
            'dataset_name': self.config.dataset_name,
            'total_events': len(df),
            'train_events': len(train_df),
            'test_events': len(test_df),
            'date_range': (df['date'].min(), df['date'].max()),
            'unique_dates': len(df['date'].unique()),
            'unique_sensors': len(df['sensor'].unique()),
            'unique_rooms': len(df['room_id'].unique()),
            'unique_activities': len(df['first_activity'].unique()),
            'unique_activities_l2': len(df['first_activity_l2'].unique()),
            'sensor_types': sorted(df['sensor_type'].unique()),
            'rooms': sorted([r for r in df['room_id'].unique() if pd.notna(r)]),
            'activities': sorted(df['first_activity'].unique()),
            'activities_l2': sorted(df['first_activity_l2'].unique()),
            'tod_buckets': sorted(df['tod_bucket'].unique()),
            'delta_t_buckets': sorted(df['time_delta_bucket'].unique()),
        }

        # Activity distributions
        metadata['activity_distribution'] = df['first_activity_l2'].value_counts().to_dict()
        metadata['train_activity_distribution'] = train_df['first_activity_l2'].value_counts().to_dict()
        metadata['test_activity_distribution'] = test_df['first_activity_l2'].value_counts().to_dict()

        return metadata
