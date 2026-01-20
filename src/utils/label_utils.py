"""
Utility functions for label processing and conversion in CASAS activity recognition.
"""

import json
import os
from typing import List, Union, Dict, Any


def load_house_metadata(house_name: str = "milan") -> Dict[str, Any]:
    """Load house metadata from casas_metadata.json or marble_metadata.json file.

    Args:
        house_name: Name of the house to load metadata for

    Returns:
        Dictionary containing house metadata
    """
    # Get the path to the metadata file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # Check if it's MARBLE dataset
    if house_name == "marble":
        metadata_path = os.path.join(project_root, "metadata", "marble_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata.get("marble", {})

    # Default to casas_metadata.json for CASAS datasets
    metadata_path = os.path.join(project_root, "metadata", "casas_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata.get(house_name, {})


def convert_labels_to_text(labels: List[str], single_description: bool = False, house_name: str = "milan",
                           description_style: str = "long_desc") -> Union[List[List[str]], List[str]]:
    """Convert CASAS activity labels to natural language descriptions.

    Args:
        labels: List of CASAS activity labels
        single_description: If True, return single description per label. If False, return multiple descriptions per label.
        house_name: Name of the house to get label descriptions for
        description_style: Field from label_to_text to use - "long_desc" (default), "short_desc", "zeroshot_har_desc", etc.

    Returns:
        List of descriptions (single or multiple per label)
    """

    # Load house metadata
    house_metadata = load_house_metadata(house_name)
    label_to_text = house_metadata.get("label_to_text", {})
    descriptions = []

    # Handle description style
    for label in labels:
        if label in label_to_text:
            label_data = label_to_text[label]

            # Try to get the requested description style
            if description_style in label_data:
                desc = label_data[description_style]
                # If desc is a list, return as-is or single item
                if isinstance(desc, list):
                    descriptions.append(desc[0] if single_description else desc)
                else:
                    # Single string description
                    descriptions.append(desc if single_description else [desc])
            else:
                # Fallback to long_desc if requested style not available
                if "long_desc" in label_data:
                    desc = label_data["long_desc"]
                    descriptions.append(desc[0] if single_description else desc)
                elif "short_desc" in label_data:
                    desc = label_data["short_desc"]
                    descriptions.append(desc if single_description else [desc])
                else:
                    # Ultimate fallback
                    readable = label.replace('_', ' ').lower()
                    fallback = f"{readable} activity"
                    descriptions.append(fallback if single_description else [fallback])
        else:
            # Label not in metadata - create fallback
            readable = label.replace('_', ' ').lower()
            if single_description:
                descriptions.append(f"{readable} activity")
            else:
                fallback_descriptions = [
                    f"{readable} activity",
                    f"general {readable} behavior",
                    f"{readable} related tasks"
                ]
                descriptions.append(fallback_descriptions)

    return descriptions
