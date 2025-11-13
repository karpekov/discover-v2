"""
Fixed-duration sampler (Step 1b).

Samples fixed time duration per window, resulting in variable-length sequences.
Example: 30s, 60s, 120s windows (each can contain different number of events).

This is a NEW implementation for time-based windowing.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import timedelta
from tqdm import tqdm

from .base import BaseSampler
from .config import FixedDurationConfig
from .utils import (
    compute_window_metadata,
    create_duration_windows,
    presegment_by_activity,
    check_time_gap
)


class FixedDurationSampler(BaseSampler):
    """Sampler for fixed-duration windows (fixed time duration)."""

    def __init__(self, config: FixedDurationConfig):
        """Initialize fixed-duration sampler.

        Args:
            config: FixedDurationConfig instance
        """
        if not isinstance(config, FixedDurationConfig):
            raise TypeError(f"Expected FixedDurationConfig, got {type(config)}")
        super().__init__(config)
        self.config: FixedDurationConfig = config  # Type hint for IDE

    def _get_sampling_params(self) -> Dict[str, Any]:
        """Get sampling parameters for fixed-duration strategy."""
        return {
            'duration_seconds': self.config.duration_seconds,
            'overlap_factor': self.config.overlap_factor,
            'min_events_per_window': self.config.min_events_per_window,
            'max_events_per_window': self.config.max_events_per_window,
            'max_sequence_length': self.config.max_sequence_length,
            'presegmented': self.config.use_presegmentation,
            'presegment_label_level': self.config.presegment_label_level if self.config.use_presegmentation else None,
        }

    def _create_windows_for_dataframe(self, df: pd.DataFrame, split: str) -> List[Dict[str, Any]]:
        """Create fixed-duration windows from dataframe.

        Args:
            df: DataFrame with sensor events
            split: 'train' or 'test'

        Returns:
            List of window dictionaries with 'events' and 'metadata' keys
        """
        all_windows = []

        # Process each duration
        for duration_sec in self.config.duration_seconds:
            print(f"  Creating windows of {duration_sec}s duration...")

            if self.config.use_presegmentation:
                windows = self._create_presegmented_duration_windows(df, duration_sec, split)
            else:
                windows = self._create_duration_windows(df, duration_sec, split)

            all_windows.extend(windows)
            print(f"    Created {len(windows)} windows")

        return all_windows

    def _create_duration_windows(self, df: pd.DataFrame, duration_sec: int, split: str) -> List[Dict[str, Any]]:
        """Create time-duration-based sliding windows."""
        windows = []

        if len(df) < self.config.min_events_per_window:
            return windows

        # Ensure timestamp is datetime and sorted
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['date'] = df['timestamp'].dt.date

        # Group by date to avoid cross-day windows
        for date, date_df in tqdm(df.groupby('date'), desc=f"  Processing dates", leave=False):
            date_df = date_df.reset_index(drop=True)

            if len(date_df) < self.config.min_events_per_window:
                continue

            # Create duration-based windows using utility function
            window_indices = create_duration_windows(
                date_df,
                duration_seconds=duration_sec,
                overlap_factor=self.config.overlap_factor,
                min_events=self.config.min_events_per_window
            )

            # Create window dictionaries
            for start_idx, end_idx in window_indices:
                window_events_df = date_df.iloc[start_idx:end_idx].copy()

                # Skip if too few events
                if len(window_events_df) < self.config.min_events_per_window:
                    continue

                # Optionally cap at max events (for very dense periods)
                if self.config.max_events_per_window and len(window_events_df) > self.config.max_events_per_window:
                    window_events_df = window_events_df.iloc[:self.config.max_events_per_window]

                # Check time gap if specified
                if self.config.max_gap_minutes:
                    if not check_time_gap(window_events_df, self.config.max_gap_minutes):
                        continue

                # Convert events to list of dictionaries
                events = self._dataframe_to_events(window_events_df)

                # Compute metadata
                metadata = compute_window_metadata(
                    window_events_df,
                    window_id=len(windows),
                    presegmented=False
                )
                metadata['target_duration_seconds'] = duration_sec
                metadata['actual_duration_seconds'] = metadata['duration_seconds']
                metadata['variable_length'] = True  # Flag for variable-length sequences

                windows.append({
                    'events': events,
                    'metadata': metadata
                })

        return windows

    def _create_presegmented_duration_windows(self, df: pd.DataFrame, duration_sec: int, split: str) -> List[Dict[str, Any]]:
        """Create duration-based windows from activity segments."""
        windows = []

        if len(df) < self.config.min_events_per_window:
            return windows

        # Ensure timestamp is datetime and sorted
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['date'] = df['timestamp'].dt.date

        # Group by date to avoid cross-day segments
        for date, date_df in tqdm(df.groupby('date'), desc=f"  Processing dates", leave=False):
            date_df = date_df.reset_index(drop=True)

            if len(date_df) < self.config.min_events_per_window:
                continue

            # Segment by activity
            segments = presegment_by_activity(
                date_df,
                label_level=self.config.presegment_label_level,
                min_events=self.config.min_segment_events,
                exclude_no_activity=self.config.exclude_no_activity
            )

            # Create duration windows within each segment
            for segment_df in segments:
                segment_windows = self._create_segment_duration_windows(segment_df, duration_sec)
                windows.extend(segment_windows)

        return windows

    def _create_segment_duration_windows(self, segment_df: pd.DataFrame, duration_sec: int) -> List[Dict[str, Any]]:
        """Create duration-based windows within a single activity segment."""
        windows = []

        if len(segment_df) < self.config.min_events_per_window:
            return windows

        # Check segment duration
        timestamps = pd.to_datetime(segment_df['timestamp'])
        segment_duration = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()

        # If segment is shorter than or equal to target duration, create single window
        if segment_duration <= duration_sec:
            if len(segment_df) >= self.config.min_events_per_window:
                events = self._dataframe_to_events(segment_df)
                metadata = compute_window_metadata(segment_df, window_id=0, presegmented=True)
                metadata['target_duration_seconds'] = duration_sec
                metadata['actual_duration_seconds'] = segment_duration
                metadata['is_complete_segment'] = True
                metadata['variable_length'] = True
                windows.append({'events': events, 'metadata': metadata})
            return windows

        # Create duration-based sliding windows within segment
        window_indices = create_duration_windows(
            segment_df,
            duration_seconds=duration_sec,
            overlap_factor=self.config.overlap_factor,
            min_events=self.config.min_events_per_window
        )

        for start_idx, end_idx in window_indices:
            window_events_df = segment_df.iloc[start_idx:end_idx].copy()

            # Skip if too few events
            if len(window_events_df) < self.config.min_events_per_window:
                continue

            # Optionally cap at max events
            if self.config.max_events_per_window and len(window_events_df) > self.config.max_events_per_window:
                window_events_df = window_events_df.iloc[:self.config.max_events_per_window]

            # Check time gap if specified
            if self.config.max_gap_minutes:
                if not check_time_gap(window_events_df, self.config.max_gap_minutes):
                    continue

            events = self._dataframe_to_events(window_events_df)
            metadata = compute_window_metadata(window_events_df, window_id=len(windows), presegmented=True)
            metadata['target_duration_seconds'] = duration_sec
            metadata['actual_duration_seconds'] = metadata['duration_seconds']
            metadata['is_complete_segment'] = False
            metadata['variable_length'] = True

            windows.append({'events': events, 'metadata': metadata})

        return windows

    def _dataframe_to_events(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to list of event dictionaries."""
        events = []
        for _, row in df.iterrows():
            event = self._event_to_dict(row, preserve_full=self.config.preserve_full_metadata)
            events.append(event)
        return events

