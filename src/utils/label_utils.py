"""
Utility functions for label processing and conversion in CASAS activity recognition.
"""

import json
import os
from typing import List, Union, Dict, Any


def load_house_metadata(house_name: str = "milan") -> Dict[str, Any]:
    """Load house metadata from house_metadata.json file.

    Args:
        house_name: Name of the house to load metadata for

    Returns:
        Dictionary containing house metadata
    """
    # Get the path to the metadata file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    metadata_path = os.path.join(project_root, "metadata", "house_metadata.json")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata.get(house_name, {})


def convert_labels_to_text(labels: List[str], single_description: bool = False, house_name: str = "milan") -> Union[List[List[str]], List[str]]:
    """Convert CASAS activity labels to natural language descriptions.

    Args:
        labels: List of CASAS activity labels
        single_description: If True, return single description per label. If False, return multiple descriptions per label.
        house_name: Name of the house to get label descriptions for

    Returns:
        List of descriptions (single or multiple per label)
    """

    # Load house metadata
    house_metadata = load_house_metadata(house_name)
    label_to_text = house_metadata.get("label_to_text", {})

    descriptions = []

    if single_description:
        # Return single description per label
        for label in labels:
            if label in label_to_text:
                descriptions.append(label_to_text[label]["short_desc"])
            else:
                # Fallback: convert label to readable text
                readable = label.replace('_', ' ').lower()
                descriptions.append(f"{readable} activity")
    else:
        # Return multiple descriptions per label
        for label in labels:
            if label in label_to_text:
                descriptions.append(label_to_text[label]["long_desc"])
            else:
                # Fallback: convert label to readable text with multiple variations
                readable = label.replace('_', ' ').lower()
                fallback_descriptions = [
                    f"{readable} activity",
                    f"general {readable} behavior",
                    f"{readable} related tasks"
                ]
                descriptions.append(fallback_descriptions)

    return descriptions
