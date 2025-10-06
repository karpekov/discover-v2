"""
Spatial utilities for coordinate processing and house layout encoding.

This module provides utilities for processing spatial information including
coordinate normalization, Fourier feature generation, and spatial encoding.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class SpatialFeatures:
    """Container for spatial features."""
    coordinates: Optional[Tuple[float, float]] = None
    fourier_features: Optional[np.ndarray] = None
    room_embedding: Optional[np.ndarray] = None
    house_tokens: Optional[List[str]] = None


class SpatialEncoder:
    """Encoder for spatial information and house layout."""

    def __init__(self, fourier_dims: int = 32):
        self.fourier_dims = fourier_dims
        self.coordinate_bounds = {}  # Will be set during fit
        self.room_to_id = {}
        self.sensor_to_room = {}

    def fit(self, sensor_coordinates: Dict[str, Tuple[float, float]],
            sensor_to_room: Dict[str, str]):
        """Fit the encoder on the spatial data."""
        self.sensor_to_room = sensor_to_room

        # Compute coordinate bounds for normalization
        if sensor_coordinates:
            coords = list(sensor_coordinates.values())
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]

            self.coordinate_bounds = {
                'x_min': min(x_coords), 'x_max': max(x_coords),
                'y_min': min(y_coords), 'y_max': max(y_coords)
            }

        # Create room vocabulary
        unique_rooms = set(sensor_to_room.values())
        self.room_to_id = {room: i for i, room in enumerate(sorted(unique_rooms))}

    def encode_sensor(self, sensor_id: str,
                     sensor_coordinates: Optional[Dict[str, Tuple[float, float]]] = None) -> SpatialFeatures:
        """Encode spatial features for a sensor."""
        features = SpatialFeatures()

        # Get coordinates if available
        if sensor_coordinates and sensor_id in sensor_coordinates:
            raw_coords = sensor_coordinates[sensor_id]
            features.coordinates = self.normalize_coordinates(raw_coords)

            # Generate Fourier features
            if features.coordinates:
                features.fourier_features = self.fourier_features(features.coordinates)

        # Get room information
        if sensor_id in self.sensor_to_room:
            room = self.sensor_to_room[sensor_id]
            if room in self.room_to_id:
                # Simple room embedding (one-hot for now)
                room_embedding = np.zeros(len(self.room_to_id))
                room_embedding[self.room_to_id[room]] = 1.0
                features.room_embedding = room_embedding

        return features

    def normalize_coordinates(self, coords: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Normalize coordinates to [0, 1] range."""
        if not self.coordinate_bounds:
            return coords

        x, y = coords
        x_range = self.coordinate_bounds['x_max'] - self.coordinate_bounds['x_min']
        y_range = self.coordinate_bounds['y_max'] - self.coordinate_bounds['y_min']

        if x_range == 0 or y_range == 0:
            return coords

        norm_x = (x - self.coordinate_bounds['x_min']) / x_range
        norm_y = (y - self.coordinate_bounds['y_min']) / y_range

        return (norm_x, norm_y)

    def fourier_features(self, coords: Tuple[float, float]) -> np.ndarray:
        """Generate Fourier features from coordinates."""
        x, y = coords

        # Generate frequency scales
        freqs = np.arange(1, self.fourier_dims // 4 + 1)

        # Compute Fourier features
        features = []
        for freq in freqs:
            features.extend([
                np.sin(2 * np.pi * freq * x),
                np.cos(2 * np.pi * freq * x),
                np.sin(2 * np.pi * freq * y),
                np.cos(2 * np.pi * freq * y)
            ])

        return np.array(features[:self.fourier_dims])


def normalize_coordinates(coordinates: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Normalize a dictionary of coordinates to [0, 1] range."""
    if not coordinates:
        return coordinates

    coords = list(coordinates.values())
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    normalized = {}
    for sensor_id, (x, y) in coordinates.items():
        norm_x = (x - x_min) / x_range
        norm_y = (y - y_min) / y_range
        normalized[sensor_id] = (norm_x, norm_y)

    return normalized


def fourier_features(coords: Tuple[float, float], n_dims: int = 32) -> np.ndarray:
    """Generate Fourier features from 2D coordinates."""
    x, y = coords

    # Generate frequency scales
    freqs = np.arange(1, n_dims // 4 + 1)

    # Compute Fourier features
    features = []
    for freq in freqs:
        features.extend([
            np.sin(2 * np.pi * freq * x),
            np.cos(2 * np.pi * freq * x),
            np.sin(2 * np.pi * freq * y),
            np.cos(2 * np.pi * freq * y)
        ])

    return np.array(features[:n_dims])


def compute_spatial_distance_matrix(coordinates: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
    """Compute pairwise Euclidean distances between sensors."""
    distance_matrix = {}

    for sensor1, coords1 in coordinates.items():
        distance_matrix[sensor1] = {}
        for sensor2, coords2 in coordinates.items():
            if sensor1 == sensor2:
                distance_matrix[sensor1][sensor2] = 0.0
            else:
                x1, y1 = coords1
                x2, y2 = coords2
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distance_matrix[sensor1][sensor2] = distance

    return distance_matrix


def create_2d_grid_encoding(coordinates: Dict[str, Tuple[float, float]],
                          grid_size: int = 10) -> Dict[str, Tuple[int, int]]:
    """Create discrete 2D grid encoding of sensor positions."""
    if not coordinates:
        return {}

    # Normalize coordinates first
    normalized = normalize_coordinates(coordinates)

    # Convert to grid positions
    grid_positions = {}
    for sensor_id, (norm_x, norm_y) in normalized.items():
        grid_x = int(norm_x * (grid_size - 1))
        grid_y = int(norm_y * (grid_size - 1))
        grid_positions[sensor_id] = (grid_x, grid_y)

    return grid_positions
