"""
Dataset configuration and metadata management.

This module handles loading dataset-specific configurations from the metadata files
and provides a registry for accessing dataset information.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class DatasetConfig:
    """Configuration for a single CASAS dataset."""
    name: str
    display_name: str
    data_path: str
    casas_url: str
    data_raw_filename: str

    # Spatial information
    sensor_coordinates: Optional[Dict[str, tuple]] = None
    room_coordinates: Optional[Dict[str, tuple]] = None
    house_layout: Optional[Dict[str, Any]] = None

    # Dataset-specific metadata
    sensor_details: Optional[Dict[str, str]] = None
    sensor_location: Optional[Dict[str, str]] = None  # Sensor to room mapping
    total_events: Optional[int] = None
    date_range: Optional[tuple] = None

    # Processing hints
    recommended_window_sizes: Optional[List[int]] = None
    known_issues: Optional[List[str]] = None


class DatasetRegistry:
    """Registry for managing dataset configurations."""

    def __init__(self, metadata_path: str = "metadata/house_metadata.json"):
        self.metadata_path = Path(metadata_path)
        self._configs: Dict[str, DatasetConfig] = {}
        self._load_configs()

    def _load_configs(self):
        """Load dataset configurations from metadata file."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        for city_name, config_data in metadata.items():
            # Skip datasets that don't have required fields or are known to be problematic
            if city_name in ['aware_home', 'cepid_10']:
                continue

            try:
                # Basic required fields
                dataset_config = DatasetConfig(
                    name=city_name,
                    display_name=config_data.get('display_name', city_name.title()),
                    data_path=f"data/{city_name}",
                    casas_url=config_data.get('casas_url', ''),
                    data_raw_filename=config_data.get('data_raw_filename', 'data'),

                    # Optional sensor details if available
                    sensor_details=config_data.get('sensor_details'),
                    sensor_location=config_data.get('sensor_location'),

                    # Optional spatial information
                    sensor_coordinates=config_data.get('sensor_coordinates'),
                    room_coordinates=config_data.get('room_coordinates'),

                    # Processing hints
                    recommended_window_sizes=config_data.get('recommended_window_sizes', [20, 50, 100]),
                    known_issues=config_data.get('known_issues', [])
                )

                self._configs[city_name] = dataset_config

            except KeyError as e:
                print(f"Warning: Skipping dataset {city_name} due to missing field: {e}")
                continue

    def get(self, dataset_name: str) -> DatasetConfig:
        """Get dataset configuration by name."""
        if dataset_name not in self._configs:
            available = list(self._configs.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")

        return self._configs[dataset_name]

    def list_available(self) -> List[str]:
        """List all available dataset names."""
        return list(self._configs.keys())

    def list_with_coordinates(self) -> List[str]:
        """List datasets that have spatial coordinate information."""
        return [
            name for name, config in self._configs.items()
            if config.sensor_coordinates is not None
        ]

    def list_with_sensor_details(self) -> List[str]:
        """List datasets that have detailed sensor descriptions."""
        return [
            name for name, config in self._configs.items()
            if config.sensor_details is not None
        ]


# Global dataset registry instance
DATASET_REGISTRY = DatasetRegistry()


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get dataset configuration by name (convenience function)."""
    return DATASET_REGISTRY.get(dataset_name)


def list_available_datasets() -> List[str]:
    """List all available datasets (convenience function)."""
    return DATASET_REGISTRY.list_available()


def load_sensor_coordinates(dataset_name: str) -> Optional[Dict[str, tuple]]:
    """Load sensor coordinates for a dataset from coordinate files."""
    coord_file = Path(f"metadata/sensor_coordinates/{dataset_name}.txt")

    if not coord_file.exists():
        return None

    coordinates = {}
    try:
        with open(coord_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        sensor_id = parts[0]
                        x, y = float(parts[1]), float(parts[2])
                        coordinates[sensor_id] = (x, y)

        return coordinates if coordinates else None

    except Exception as e:
        print(f"Warning: Could not load coordinates for {dataset_name}: {e}")
        return None