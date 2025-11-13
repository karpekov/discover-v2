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
                           description_style: str = "baseline") -> Union[List[List[str]], List[str]]:
    """Convert CASAS activity labels to natural language descriptions.

    Args:
        labels: List of CASAS activity labels
        single_description: If True, return single description per label. If False, return multiple descriptions per label.
        house_name: Name of the house to get label descriptions for
        description_style: Style of descriptions to use - "baseline" (default) or "sourish"

    Returns:
        List of descriptions (single or multiple per label)
    """

    # Load house metadata
    house_metadata = load_house_metadata(house_name)

    # Select the appropriate label-to-text mapping based on description style
    if description_style == "sourish":
        label_to_text_sourish = house_metadata.get("label_to_text_sourish", {})
        # For sourish style, always return single descriptions (they don't have multiple versions)
        descriptions = []
        for label in labels:
            if label in label_to_text_sourish:
                desc = label_to_text_sourish[label]
                # If requesting multiple descriptions but sourish has only one, return as list with single item
                descriptions.append([desc] if not single_description else desc)
            else:
                # Fallback: convert label to readable text
                readable = label.replace('_', ' ').lower()
                fallback = f"{readable} activity"
                descriptions.append([fallback] if not single_description else fallback)
        return descriptions
    else:
        # Original baseline behavior
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
