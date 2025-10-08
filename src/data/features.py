"""Per-event feature engineering for dual-encoder training."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math

from .data_config import FeatureConfig
from .datasets import DatasetConfig
from utils.spatial_utils import (
    normalize_coordinates, fourier_features,
    create_2d_grid_encoding
)


@dataclass
class EventFeatures:
    """Container for per-event features."""
    # Core identifiers
    sensor_id: str
    room_id: str
    event_type: str  # ON/OFF/value
    sensor_type: str  # M/D/T/etc.

    # Temporal features
    tod_bucket: str
    delta_t_bucket: str
    hour: int
    day_of_week: int

    # Optional fields with defaults
    sensor_detail: Optional[str] = None

    # Spatial features
    x_coord: Optional[float] = None
    y_coord: Optional[float] = None
    x_coord_norm: Optional[float] = None
    y_coord_norm: Optional[float] = None

    # Fourier coordinates (if enabled)
    fourier_x_sin: Optional[float] = None
    fourier_x_cos: Optional[float] = None
    fourier_y_sin: Optional[float] = None
    fourier_y_cos: Optional[float] = None

    # Grid coordinates (if enabled)
    grid_x: Optional[int] = None
    grid_y: Optional[int] = None

    # House-level features
    house_token: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class WindowFeatures:
    """Container for window-level features."""
    window_id: int
    events: List[EventFeatures]

    # Aggregate spatial features
    trajectory_features: Dict[str, Any]

    # Window-level metadata
    duration_sec: float
    num_events: int
    unique_sensors: int
    unique_rooms: int
    primary_room: str
    room_sequence: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'window_id': self.window_id,
            'events': [event.to_dict() for event in self.events],
            'trajectory_features': self.trajectory_features,
            'duration_sec': self.duration_sec,
            'num_events': self.num_events,
            'unique_sensors': self.unique_sensors,
            'unique_rooms': self.unique_rooms,
            'primary_room': self.primary_room,
            'room_sequence': self.room_sequence
        }


class FeatureExtractor:
    """Extract per-event and window-level features."""

    def __init__(self, config: FeatureConfig, dataset_config: DatasetConfig):
        self.config = config
        self.dataset_config = dataset_config

        # Precompute spatial mappings
        self._setup_spatial_features()

    def _setup_spatial_features(self):
        """Setup spatial feature mappings."""
        self.normalized_coords = None
        self.fourier_coords = None
        self.grid_coords = None

        if self.config.use_coordinates and self.dataset_config.sensor_coordinates:
            # Normalize coordinates
            self.normalized_coords = normalize_coordinates(
                self.dataset_config.sensor_coordinates
            )

            # Compute Fourier features if enabled (placeholder for now)
            if self.config.use_fourier_coords:
                self.fourier_coords = {}

            # Create grid encoding
            self.grid_coords = create_2d_grid_encoding(
                self.normalized_coords,
                grid_size=32
            )

    def extract_event_features(self, event: pd.Series) -> EventFeatures:
        """Extract features for a single event."""

        # Core identifiers
        sensor_id = event.get('sensor', '')
        room_id = event.get('room_id', 'unknown')
        sensor_detail = event.get('sensor_detail', None)
        event_type = event.get('event_type', event.get('state', ''))
        sensor_type = sensor_id[0] if sensor_id else 'U'  # Unknown

        # Temporal features
        tod_bucket = event.get('tod_bucket', 'unknown')
        delta_t_bucket = event.get('time_delta_bucket', 'dt_0')
        hour = event.get('hour', 0)
        day_of_week = event.get('day_of_week', 0)

        # Initialize spatial features
        x_coord = None
        y_coord = None
        x_coord_norm = None
        y_coord_norm = None
        fourier_x_sin = None
        fourier_x_cos = None
        fourier_y_sin = None
        fourier_y_cos = None
        grid_x = None
        grid_y = None

        # Extract spatial features if available
        if self.config.use_coordinates and sensor_id in self.dataset_config.sensor_coordinates:
            x_coord, y_coord = self.dataset_config.sensor_coordinates[sensor_id]

            if self.normalized_coords and sensor_id in self.normalized_coords:
                x_coord_norm, y_coord_norm = self.normalized_coords[sensor_id]

            if self.config.use_fourier_coords and self.fourier_coords and sensor_id in self.fourier_coords:
                fourier_features = self.fourier_coords[sensor_id]
                fourier_x_sin, fourier_x_cos, fourier_y_sin, fourier_y_cos = fourier_features

            if self.grid_coords and sensor_id in self.grid_coords:
                grid_x, grid_y = self.grid_coords[sensor_id]

        # House token (if enabled)
        house_token = None
        if self.config.include_house_tokens:
            house_token = f"[HOUSE={self.dataset_config.name.upper()}]"

        return EventFeatures(
            sensor_id=sensor_id,
            room_id=room_id,
            sensor_detail=sensor_detail,
            event_type=event_type,
            sensor_type=sensor_type,
            tod_bucket=tod_bucket,
            delta_t_bucket=delta_t_bucket,
            hour=hour,
            day_of_week=day_of_week,
            x_coord=x_coord,
            y_coord=y_coord,
            x_coord_norm=x_coord_norm,
            y_coord_norm=y_coord_norm,
            fourier_x_sin=fourier_x_sin,
            fourier_x_cos=fourier_x_cos,
            fourier_y_sin=fourier_y_sin,
            fourier_y_cos=fourier_y_cos,
            grid_x=grid_x,
            grid_y=grid_y,
            house_token=house_token
        )

    def extract_window_features(self, window_id: int, events_df: pd.DataFrame) -> WindowFeatures:
        """Extract features for a window of events."""

        # Extract per-event features
        event_features = []
        for _, event in events_df.iterrows():
            features = self.extract_event_features(event)
            event_features.append(features)

        # Compute trajectory features (placeholder for now)
        trajectory_features = {}

        # Window-level aggregates
        duration_sec = 0.0
        if len(events_df) > 1:
            start_time = events_df['datetime'].iloc[0]
            end_time = events_df['datetime'].iloc[-1]
            duration_sec = (end_time - start_time).total_seconds()

        num_events = len(events_df)
        unique_sensors = len(events_df['sensor'].unique())
        unique_rooms = len(events_df['room_id'].unique())

        # Primary room (most frequent)
        primary_room = events_df['room_id'].mode().iloc[0] if len(events_df) > 0 else 'unknown'

        # Room sequence (ordered list of rooms visited)
        room_sequence = events_df['room_id'].tolist()

        return WindowFeatures(
            window_id=window_id,
            events=event_features,
            trajectory_features=trajectory_features,
            duration_sec=duration_sec,
            num_events=num_events,
            unique_sensors=unique_sensors,
            unique_rooms=unique_rooms,
            primary_room=primary_room,
            room_sequence=room_sequence
        )

    def create_discrete_feature_vocab(self, all_events: List[EventFeatures]) -> Dict[str, List[str]]:
        """Create vocabulary for discrete features."""
        vocab = {
            'sensor_id': [],
            'room_id': [],
            'event_type': [],
            'sensor_type': [],
            'tod_bucket': [],
            'delta_t_bucket': []
        }

        # Collect all unique values
        for event in all_events:
            vocab['sensor_id'].append(event.sensor_id)
            vocab['room_id'].append(event.room_id)
            vocab['event_type'].append(event.event_type)
            vocab['sensor_type'].append(event.sensor_type)
            vocab['tod_bucket'].append(event.tod_bucket)
            vocab['delta_t_bucket'].append(event.delta_t_bucket)

        # Remove duplicates and sort
        for key in vocab:
            # Convert all values to strings and filter out None/NaN values
            unique_values = set()
            for value in vocab[key]:
                if value is not None and not (isinstance(value, float) and pd.isna(value)):
                    unique_values.add(str(value))
            vocab[key] = sorted(list(unique_values))

        # Add special tokens
        for key in vocab:
            vocab[key] = ['[PAD]', '[UNK]'] + vocab[key]

        return vocab

    def create_feature_mappings(self, vocab: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
        """Create feature value to index mappings."""
        mappings = {}
        for feature_name, values in vocab.items():
            mappings[feature_name] = {value: idx for idx, value in enumerate(values)}
        return mappings

    def encode_event_features(self, event: EventFeatures, mappings: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """Encode event features using mappings."""
        encoded = {}

        # Discrete features
        encoded['sensor_id'] = mappings['sensor_id'].get(event.sensor_id, mappings['sensor_id']['[UNK]'])
        encoded['room_id'] = mappings['room_id'].get(event.room_id, mappings['room_id']['[UNK]'])
        encoded['event_type'] = mappings['event_type'].get(event.event_type, mappings['event_type']['[UNK]'])
        encoded['sensor_type'] = mappings['sensor_type'].get(event.sensor_type, mappings['sensor_type']['[UNK]'])
        encoded['tod_bucket'] = mappings['tod_bucket'].get(event.tod_bucket, mappings['tod_bucket']['[UNK]'])
        encoded['delta_t_bucket'] = mappings['delta_t_bucket'].get(event.delta_t_bucket, mappings['delta_t_bucket']['[UNK]'])

        # Continuous features
        if event.x_coord_norm is not None:
            encoded['x_coord_norm'] = event.x_coord_norm
        if event.y_coord_norm is not None:
            encoded['y_coord_norm'] = event.y_coord_norm

        # Fourier features
        if event.fourier_x_sin is not None:
            encoded['fourier_x_sin'] = event.fourier_x_sin
            encoded['fourier_x_cos'] = event.fourier_x_cos
            encoded['fourier_y_sin'] = event.fourier_y_sin
            encoded['fourier_y_cos'] = event.fourier_y_cos

        # Grid features
        if event.grid_x is not None:
            encoded['grid_x'] = event.grid_x
            encoded['grid_y'] = event.grid_y

        # Temporal features
        encoded['hour'] = event.hour
        encoded['day_of_week'] = event.day_of_week

        return encoded
