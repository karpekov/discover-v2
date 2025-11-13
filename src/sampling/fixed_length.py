"""
Fixed-length sampler (Step 1a).

Samples fixed number of sensor events per window.
Example: 20, 50, 100 events per window.

This is adapted from the existing windowing.py logic to be self-sufficient.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

from .base import BaseSampler
from .config import FixedLengthConfig
from .utils import (
    compute_window_metadata,
    create_sliding_windows,
    presegment_by_activity
)


class FixedLengthSampler(BaseSampler):
    """Sampler for fixed-length windows (fixed number of events)."""

    def __init__(self, config: FixedLengthConfig):
        """Initialize fixed-length sampler.

        Args:
            config: FixedLengthConfig instance
        """
        if not isinstance(config, FixedLengthConfig):
            raise TypeError(f"Expected FixedLengthConfig, got {type(config)}")
        super().__init__(config)
        self.config: FixedLengthConfig = config  # Type hint for IDE

    def _get_sampling_params(self) -> Dict[str, Any]:
        """Get sampling parameters for fixed-length strategy."""
        return {
            'window_sizes': self.config.window_sizes,
            'overlap_factor': self.config.overlap_factor,
            'min_events_per_window': self.config.min_events_per_window,
            'presegmented': self.config.use_presegmentation,
            'presegment_label_level': self.config.presegment_label_level if self.config.use_presegmentation else None,
        }

    def _create_windows_for_dataframe(self, df: pd.DataFrame, split: str) -> List[Dict[str, Any]]:
        """Create fixed-length windows from dataframe.

        Args:
            df: DataFrame with sensor events
            split: 'train' or 'test'

        Returns:
            List of window dictionaries with 'events' and 'metadata' keys
        """
        all_windows = []

        # Process each window size
        for window_size in self.config.window_sizes:
            print(f"  Creating windows of size {window_size}...")

            if self.config.use_presegmentation:
                windows = self._create_presegmented_windows(df, window_size, split)
            else:
                windows = self._create_sliding_windows(df, window_size, split)

            all_windows.extend(windows)
            print(f"    Created {len(windows)} windows")

        return all_windows

    def _create_sliding_windows(self, df: pd.DataFrame, window_size: int, split: str) -> List[Dict[str, Any]]:
        """Create sliding windows with overlap."""
        windows = []

        if len(df) < self.config.min_events_per_window:
            return windows

        # Add date column for grouping
        df = df.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date

        # Group by date to avoid cross-day windows
        for date, date_df in tqdm(df.groupby('date'), desc=f"  Processing dates", leave=False):
            date_df = date_df.reset_index(drop=True)

            if len(date_df) < self.config.min_events_per_window:
                continue

            # Create sliding windows using utility function
            window_indices = create_sliding_windows(
                date_df,
                window_size=window_size,
                overlap_factor=self.config.overlap_factor,
                min_events=self.config.min_events_per_window
            )

            # Create window dictionaries
            for start_idx, end_idx in window_indices:
                window_events_df = date_df.iloc[start_idx:end_idx].copy()

                # Check time gap if specified
                if self.config.max_gap_minutes:
                    from .utils import check_time_gap
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

                windows.append({
                    'events': events,
                    'metadata': metadata
                })

        return windows

    def _create_presegmented_windows(self, df: pd.DataFrame, window_size: int, split: str) -> List[Dict[str, Any]]:
        """Create windows from activity-based segments with overlap within segments."""
        windows = []

        if len(df) < self.config.min_events_per_window:
            return windows

        # Add date column for grouping
        df = df.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date

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

            # Create overlapping windows within each segment
            for segment_df in segments:
                segment_windows = self._create_segment_windows(segment_df, window_size)
                windows.extend(segment_windows)

        return windows

    def _create_segment_windows(self, segment_df: pd.DataFrame, window_size: int) -> List[Dict[str, Any]]:
        """Create overlapping windows within a single activity segment."""
        windows = []

        if len(segment_df) < self.config.min_events_per_window:
            return windows

        # If segment is exactly window_size or smaller, create single window
        if len(segment_df) <= window_size:
            if len(segment_df) >= self.config.min_events_per_window:
                events = self._dataframe_to_events(segment_df)
                metadata = compute_window_metadata(segment_df, window_id=0, presegmented=True)
                metadata['is_complete_segment'] = True
                windows.append({'events': events, 'metadata': metadata})
            return windows

        # Create sliding windows within segment
        window_indices = create_sliding_windows(
            segment_df,
            window_size=window_size,
            overlap_factor=self.config.overlap_factor,
            min_events=self.config.min_events_per_window
        )

        for start_idx, end_idx in window_indices:
            window_events_df = segment_df.iloc[start_idx:end_idx].copy()

            # Check time gap if specified
            if self.config.max_gap_minutes:
                from .utils import check_time_gap
                if not check_time_gap(window_events_df, self.config.max_gap_minutes):
                    continue

            events = self._dataframe_to_events(window_events_df)
            metadata = compute_window_metadata(window_events_df, window_id=len(windows), presegmented=True)
            metadata['is_complete_segment'] = (start_idx == 0 and end_idx == len(segment_df))

            windows.append({'events': events, 'metadata': metadata})

        return windows

    def _dataframe_to_events(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to list of event dictionaries."""
        events = []
        for _, row in df.iterrows():
            event = self._event_to_dict(row, preserve_full=self.config.preserve_full_metadata)
            events.append(event)
        return events

