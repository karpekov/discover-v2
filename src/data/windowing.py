"""Fixed-length windowing with overlap for dual-encoder training."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Iterator, Tuple, Optional
from dataclasses import dataclass
import warnings
from tqdm import tqdm

from .data_config import WindowingConfig, WindowStrategy


@dataclass
class WindowMetadata:
    """Metadata for a single window."""
    window_id: int
    start_idx: int
    end_idx: int
    size: int
    window_length: int  # Target length for this window type

    # Time information
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_sec: float

    # Activity information (if presegmented)
    primary_activity: str
    primary_activity_l2: str
    activity_distribution: Dict[str, int]

    # Spatial information
    rooms_visited: List[str]
    primary_room: str
    room_transitions: int

    # Split information
    split: str  # 'train' or 'test'
    date: pd.Timestamp

    # Segment information (if presegmented)
    segment_ids: List[int] = None
    is_complete_segment: bool = False


@dataclass
class ProcessedWindow:
    """A processed window ready for model training."""
    metadata: WindowMetadata
    events: pd.DataFrame  # The actual event sequence

    @property
    def size(self) -> int:
        return len(self.events)

    @property
    def is_valid(self) -> bool:
        """Check if window meets minimum requirements."""
        return self.size >= 8  # Recipe R2 requirement


class WindowProcessor:
    """Process data into fixed-length windows with overlap."""

    def __init__(self, config: WindowingConfig):
        self.config = config
        self._window_id_counter = 0

    def process_dataset(self, train_df: pd.DataFrame, test_df: pd.DataFrame, max_windows: Optional[int] = None) -> Dict[int, List[ProcessedWindow]]:
        """Process train and test dataframes into windows of different sizes.

        Returns:
            Dict mapping window_size -> List[ProcessedWindow]
        """
        print("Creating windows...")

        results = {}

        for window_size in self.config.sizes:
            # Process train data
            train_windows = self._create_windows(train_df, window_size, split='train')

            # Process test data
            test_windows = self._create_windows(test_df, window_size, split='test')

            # Combine and filter
            all_windows = train_windows + test_windows

            # Filter out invalid windows
            valid_windows = [w for w in all_windows if w.is_valid]

            # Apply max_windows limit if specified
            if max_windows is not None and len(valid_windows) > max_windows:
                print(f"    Limiting to first {max_windows} windows (was {len(valid_windows)})")
                valid_windows = valid_windows[:max_windows]

            print(f"    Window size {window_size}: {len(all_windows)} total, {len(valid_windows)} valid (>={self.config.min_events} events)")

            results[window_size] = valid_windows

        return results

    def _create_windows(self, df: pd.DataFrame, window_size: int, split: str) -> List[ProcessedWindow]:
        """Create windows from a dataframe."""
        if len(df) == 0:
            return []

        windows = []

        if self.config.strategy == WindowStrategy.SLIDING:
            windows = self._create_sliding_windows(df, window_size, split)
        elif self.config.strategy == WindowStrategy.NON_OVERLAPPING:
            windows = self._create_non_overlapping_windows(df, window_size, split)
        elif self.config.strategy == WindowStrategy.PRESEGMENTED:
            windows = self._create_presegmented_windows(df, window_size, split)
        else:
            raise ValueError(f"Unknown window strategy: {self.config.strategy}")

        return windows

    def _create_sliding_windows(self, df: pd.DataFrame, window_size: int, split: str) -> List[ProcessedWindow]:
        """Create sliding windows with overlap."""
        windows = []

        if len(df) < self.config.min_events:
            return windows

        # Calculate step size based on overlap ratio
        step_size = max(1, int(window_size * (1 - self.config.overlap_ratio)))

        # Group by date to avoid cross-day windows
        for date, date_df in tqdm(df.groupby('date'), desc=f"Processing dates ({split})", leave=False):
            date_df = date_df.reset_index(drop=True)

            if len(date_df) < self.config.min_events:
                continue

            # Create sliding windows for this date
            start_indices = range(0, len(date_df) - self.config.min_events + 1, step_size)
            for start_idx in tqdm(start_indices, desc=f"Creating windows for {date.strftime('%Y-%m-%d')}", leave=False):
                end_idx = min(start_idx + window_size, len(date_df))

                # Skip if window is too small
                if end_idx - start_idx < self.config.min_events:
                    if self.config.drop_incomplete:
                        continue

                window_events = date_df.iloc[start_idx:end_idx].copy()
                metadata = self._create_window_metadata(
                    window_events, start_idx, end_idx, window_size, split, date
                )

                windows.append(ProcessedWindow(metadata=metadata, events=window_events))

        return windows

    def _create_non_overlapping_windows(self, df: pd.DataFrame, window_size: int, split: str) -> List[ProcessedWindow]:
        """Create non-overlapping windows."""
        windows = []

        # Group by date to avoid cross-day windows
        for date, date_df in df.groupby('date'):
            date_df = date_df.reset_index(drop=True)

            if len(date_df) < self.config.min_events:
                continue

            # Create non-overlapping windows for this date
            for start_idx in range(0, len(date_df), window_size):
                end_idx = min(start_idx + window_size, len(date_df))

                # Skip if window is too small
                if end_idx - start_idx < self.config.min_events:
                    if self.config.drop_incomplete:
                        continue

                window_events = date_df.iloc[start_idx:end_idx].copy()
                metadata = self._create_window_metadata(
                    window_events, start_idx, end_idx, window_size, split, date
                )

                windows.append(ProcessedWindow(metadata=metadata, events=window_events))

        return windows

    def _create_presegmented_windows(self, df: pd.DataFrame, window_size: int, split: str) -> List[ProcessedWindow]:
        """Create windows from activity-based segments with overlap within segments."""
        windows = []

        if len(df) < self.config.min_events:
            return windows

        # Determine which activity column to use for segmentation
        if self.config.presegment_activity_level == "l2":
            activity_col = "first_activity_l2"
        else:
            activity_col = "first_activity"

        if activity_col not in df.columns:
            print(f"Warning: {activity_col} column not found. Falling back to sliding windows.")
            return self._create_sliding_windows(df, window_size, split)

        # Group by date to avoid cross-day segments
        for date, date_df in tqdm(df.groupby('date'), desc=f"Processing dates ({split})", leave=False):
            date_df = date_df.reset_index(drop=True)

            if len(date_df) < self.config.min_events:
                continue

            # Create activity segments within this date
            segments = self._segment_by_activity(date_df, activity_col)

            # Create overlapping windows within each segment
            for segment in segments:
                segment_windows = self._create_segment_windows(segment, window_size, split, date)
                windows.extend(segment_windows)

        return windows

    def _segment_by_activity(self, df: pd.DataFrame, activity_col: str) -> List[pd.DataFrame]:
        """Segment dataframe by activity changes."""
        segments = []

        # Find activity change points
        activity_changes = (df[activity_col] != df[activity_col].shift(1)).cumsum()

        # Group by activity segments
        for segment_id, segment_df in df.groupby(activity_changes):
            segment_df = segment_df.reset_index(drop=True)

            # Skip segments that are too short
            if len(segment_df) < self.config.min_segment_events:
                continue

            # Skip no-activity segments if configured
            if self.config.exclude_no_activity:
                activity_value = segment_df[activity_col].iloc[0]
                if activity_value in ['no_activity', 'No_Activity']:
                    continue

            segments.append(segment_df)

        return segments

    def _create_segment_windows(self, segment_df: pd.DataFrame, window_size: int,
                               split: str, date: pd.Timestamp) -> List[ProcessedWindow]:
        """Create overlapping windows within a single activity segment."""
        windows = []

        if len(segment_df) < self.config.min_events:
            return windows

        # Calculate step size based on overlap ratio
        step_size = max(1, int(window_size * (1 - self.config.overlap_ratio)))

        # Create sliding windows within this segment
        start_indices = range(0, len(segment_df) - self.config.min_events + 1, step_size)

        for start_idx in start_indices:
            end_idx = min(start_idx + window_size, len(segment_df))

            # Skip if window is too small
            if end_idx - start_idx < self.config.min_events:
                if self.config.drop_incomplete:
                    continue

            window_events = segment_df.iloc[start_idx:end_idx].copy()
            metadata = self._create_window_metadata(
                window_events, start_idx, end_idx, window_size, split, date
            )

            # Mark this as a presegmented window
            metadata.is_complete_segment = (start_idx == 0 and end_idx == len(segment_df))

            windows.append(ProcessedWindow(metadata=metadata, events=window_events))

        return windows

    def _create_window_metadata(self, events: pd.DataFrame, start_idx: int, end_idx: int,
                               window_length: int, split: str, date: pd.Timestamp) -> WindowMetadata:
        """Create metadata for a window."""

        # Activity information
        activities = events['first_activity'].dropna()
        activities_l2 = events['first_activity_l2'].dropna()

        primary_activity = activities.mode().iloc[0] if len(activities) > 0 else 'no_activity'
        primary_activity_l2 = activities_l2.mode().iloc[0] if len(activities_l2) > 0 else 'No_Activity'

        activity_dist = activities_l2.value_counts().to_dict()

        # Spatial information
        rooms = events['room_id'].dropna()
        rooms_visited = rooms.unique().tolist()
        primary_room = rooms.mode().iloc[0] if len(rooms) > 0 else 'unknown'

        # Count room transitions
        room_transitions = (rooms != rooms.shift(1)).sum() - 1 if len(rooms) > 1 else 0

        # Segmentation information (if available)
        segment_ids = None
        is_complete_segment = False

        if 'segment_id' in events.columns:
            segment_ids = events['segment_id'].unique().tolist()
            # Check if this window represents a complete segment
            if len(segment_ids) == 1:
                segment_events = events[events['segment_id'] == segment_ids[0]]
                is_complete_segment = len(segment_events) == len(events)

        metadata = WindowMetadata(
            window_id=self._get_next_window_id(),
            start_idx=start_idx,
            end_idx=end_idx,
            size=len(events),
            window_length=window_length,
            start_time=events['datetime'].iloc[0],
            end_time=events['datetime'].iloc[-1],
            duration_sec=(events['datetime'].iloc[-1] - events['datetime'].iloc[0]).total_seconds(),
            primary_activity=primary_activity,
            primary_activity_l2=primary_activity_l2,
            activity_distribution=activity_dist,
            rooms_visited=rooms_visited,
            primary_room=primary_room,
            room_transitions=room_transitions,
            split=split,
            date=date,
            segment_ids=segment_ids,
            is_complete_segment=is_complete_segment
        )

        return metadata

    def _get_next_window_id(self) -> int:
        """Get next unique window ID."""
        window_id = self._window_id_counter
        self._window_id_counter += 1
        return window_id

    def get_statistics(self, windows_by_size: Dict[int, List[ProcessedWindow]]) -> Dict[str, Any]:
        """Compute statistics about the generated windows."""
        stats = {}

        for window_size, windows in windows_by_size.items():
            if not windows:
                continue

            # Basic counts
            train_windows = [w for w in windows if w.metadata.split == 'train']
            test_windows = [w for w in windows if w.metadata.split == 'test']

            # Size distribution
            sizes = [w.size for w in windows]

            # Duration distribution
            durations = [w.metadata.duration_sec for w in windows]

            # Activity distribution
            activities = [w.metadata.primary_activity_l2 for w in windows]
            activity_counts = pd.Series(activities).value_counts()

            # Room distribution
            rooms = [w.metadata.primary_room for w in windows]
            room_counts = pd.Series(rooms).value_counts()

            # Segmentation stats (if available)
            complete_segments = [w for w in windows if w.metadata.is_complete_segment]

            stats[window_size] = {
                'total_windows': len(windows),
                'train_windows': len(train_windows),
                'test_windows': len(test_windows),
                'size_stats': {
                    'mean': np.mean(sizes),
                    'std': np.std(sizes),
                    'min': np.min(sizes),
                    'max': np.max(sizes)
                },
                'duration_stats': {
                    'mean_sec': np.mean(durations),
                    'std_sec': np.std(durations),
                    'min_sec': np.min(durations),
                    'max_sec': np.max(durations)
                },
                'top_activities': activity_counts.head(10).to_dict(),
                'top_rooms': room_counts.head(10).to_dict(),
                'complete_segments': len(complete_segments),
                'complete_segment_ratio': len(complete_segments) / len(windows) if windows else 0
            }

        return stats
