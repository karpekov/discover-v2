"""Sourish-style caption generation using structured templates.

This module implements caption generation following Sourish Dhekane's approach
from zero-shot activity recognition research. The captions are more structured
and template-based compared to the baseline natural language captions.

Key differences from baseline captions:
- Fixed template structure: when + duration + where + sensors
- More mechanical/systematic descriptions
- Focuses on sensor-centric information
- Deterministic output (less variation)
- Dataset-specific hard-coded sensor-to-location mappings
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
import inflect

from .data_config import CaptionConfig
from .windowing import ProcessedWindow
from .features import WindowFeatures


@dataclass
class Caption:
    """Container for captions (compatible with baseline Caption class)."""
    text: str  # Full caption text
    window_id: int
    caption_type: str = 'sourish'

    # Additional fields for compatibility
    layer_b: Optional[str] = None

    @property
    def tokens(self) -> List[str]:
        """Tokenized caption."""
        return self.text.split()

    @property
    def is_short(self) -> bool:
        """Check if this is a short caption (< 10 tokens)."""
        return len(self.tokens) < 10


class SourishCaptionGenerator:
    """Generate Sourish-style captions using structured templates.

    This implementation follows Sourish Dhekane's approach with:
    - Hard-coded sensor-to-location mappings for each dataset
    - Structured template: when + duration + where + sensors
    - Dataset-specific special cases
    """

    # Milan sensor to location mappings (from Sourish's code)
    MILAN_SENSOR_CONTEXT = {
        "M001": "at home entrance",
        "M002": "between home entrance and living room",
        "M003": "in dinning room",
        "M004": "in living room on sofa",
        "M005": "in living room near slider door",
        "M006": "between living room and workspace and TV room",
        "M007": "near desk in workspace and TV room",
        "M008": "in workspace and TV room on sofa",
        "M009": "in aisle near washer and dryer",
        "M010": "in aisle between dinning area and kitchen",
        "M011": "in aisle between kitchen and guest bathroom toilet sink",
        "M012": "between dinning area and kitchen",
        "M013": "in master bathroom toilet",
        "M014": "in kitchen near the medicine cabinet door",
        "M015": "in kitchen near fridge",
        "M016": "in kitchen near guest bedroom and bathroom toilet aisle",
        "M017": "in guest bathroom toilet sink",
        "M018": "in toilet and shower in guest bathroom toilet",
        "M019": "in aisle near master bedroom",
        "M020": "in master bedroom near the master bathroom toilet",
        "M021": "in master bedroom on bed",
        "M022": "in kitchen near stove",
        "M023": "in kitchen",
        "M024": "in guest bedroom",
        "M025": "between master bedroom and master bathroom toilet",
        "M026": "in workspace and TV room",
        "M027": "in living room",
        "M028": "in master bedroom",
        "T001": "in kitchen",
        "T002": "between aisle and bathroom sink",
        "D001": "on home entrance door",
        "D002": "near home entrance door",
        "D003": "in kitchen on garage door",
    }

    # Aruba sensor to location mappings (from Sourish's code)
    ARUBA_SENSOR_CONTEXT = {
        "T003": "in the kitchen near the counter",
        "M015": "in the kitchen near the counter",
        "M019": "in the middle of the kitchen",
        "M017": "in the middle of the kitchen",
        "M016": "in the kitchen near the back door",
        "D002": "on the back door in the kitchen",
        "M018": "between the kitchen and the dining area",
        "M014": "in the dining area",
        "M013": "between the dining area and the living room",
        "T002": "in the living room",
        "M012": "in the living room",
        "M020": "in the living room",
        "M009": "in the living room",
        "M010": "in the living room",
        "M011": "in the home entrance aisle near the front door",
        "M008": "between the home entrance aisle and the bedroom entrance",
        "D001": "on front door near aisle",
        "M001": "in the bedroom near the closet",
        "M002": "in the bedroom on the bed near the nightstand",
        "M003": "in the bedroom on the bed near the nightstand close to the entrance",
        "T001": "in the bedroom near the bed and the nightstand close to the entrance",
        "M005": "in the bedroom between the bed and the toilet and bathroom",
        "M007": "in the bedroom between the bed and the toilet and bathroom",
        "M006": "in the bedroom near the home entrance aisle",
        "M004": "in the bedroom close to the toilet and bathroom",
        "M021": "in the aisle between bathroom and dining area",
        "T004": "in the aisle between bathroom and dining area",
        "M031": "in the bathroom",
        "D003": "on the bathroom door near the aisle",
        "M022": "in the aisle between the bathroom and the bedroom",
        "M024": "in the bedroom",
        "M023": "in the bedroom near the aisle",
        "M029": "in the aisle near the bathroom and the garage door",
        "M030": "in the aisle near the bathroom and the garage door",
        "D004": "on the garage door",
        "M028": "between the office and the aisle",
        "M027": "in the office",
        "M026": "in the office",
        "M025": "in the office",
        "T005": "in the office",
    }

    # Cairo sensor to location mappings (from Sourish's code)
    CAIRO_SENSOR_CONTEXT = {
        "M001": "in work area office",
        "M002": "in aisle near bedroom and guest bedroom",
        "M003": "in aisle near stairs",
        "M004": "in guest bedroom",
        "M005": "in bedroom",
        "M006": "in bedroom entrance",
        "M007": "between bed and toilet",
        "M008": "in bedroom on bed",
        "M009": "in bedroom on bed",
        "M010": "near top of stairs",
        "M011": "in living room near bottom of stairs",
        "M012": "in kitchen",
        "M013": "in living room on couch",
        "M014": "between living room and stairs",
        "M015": "between living room and outside door",
        "M016": "in living room",
        "M017": "in living room on couch",
        "M018": "in living room",
        "M019": "in kitchen",
        "M020": "in food table and kitchen",
        "M021": "on medicine cabinet",
        "M022": "in kitchen",
        "M023": "in living room",
        "M024": "in kitchen",
        "M025": "in basement near laundry room",
        "M026": "in laundry room",
        "M027": "near garage door",
        "T001": "in bedroom",
        "T002": "in work area office",
        "T003": "in living room near stairs",
        "T004": "in kitchen",
        "T005": "in living room",
    }

    def __init__(self, config: CaptionConfig):
        self.config = config
        self.inflect_engine = inflect.engine()
        self.dataset_name = None  # Will be set by pipeline

    def _get_sensor_context_map(self) -> Dict[str, str]:
        """Get the appropriate sensor context mapping for the dataset."""
        if self.dataset_name == "milan":
            return self.MILAN_SENSOR_CONTEXT
        elif self.dataset_name == "aruba":
            return self.ARUBA_SENSOR_CONTEXT
        elif self.dataset_name == "cairo":
            return self.CAIRO_SENSOR_CONTEXT
        else:
            # Fallback: empty map
            return {}

    def generate_captions(self, window: ProcessedWindow,
                         window_features: WindowFeatures,
                         sensor_details: Optional[Dict[str, str]] = None) -> Dict[str, List[Caption]]:
        """Generate captions and return them separated into long and short arrays.

        For Sourish-style captions:
        - Long captions: Full structured descriptions (when + duration + where + sensors)
        - Short captions: Where sentence only (location-focused)
        """

        long_captions = []
        short_captions = []

        # Generate 2 full structured captions
        for _ in range(2):
            caption = self._generate_full_caption(window, window_features, sensor_details)
            if caption:
                long_captions.append(caption)

        # Generate 2 short location-focused captions
        for _ in range(2):
            caption = self._generate_short_caption(window, window_features, sensor_details)
            if caption:
                short_captions.append(caption)

        return {
            'long': long_captions,
            'short': short_captions,
            'all': long_captions + short_captions
        }

    def _generate_full_caption(self, window: ProcessedWindow,
                              window_features: WindowFeatures,
                              sensor_details: Optional[Dict[str, str]] = None) -> Optional[Caption]:
        """Generate full structured caption: when + duration + where + sensors."""

        # Get time information
        first_event = window.events.iloc[0]
        last_event = window.events.iloc[-1]

        time_seq = window.events['datetime'].tolist()
        sensor_seq = window.events['sensor'].tolist() if 'sensor' in window.events.columns else []

        # Generate components
        start_time, end_time = self._get_start_and_end_time(time_seq)
        when_sentence = self._generate_when_sentence(start_time, end_time, first_event)
        time_diff = self._get_time_diff(start_time, end_time)
        duration_sentence = self._generate_duration_sentence(time_diff)
        where_sentence = self._generate_where_sentence(sensor_seq, window_features, sensor_details)
        sensors_sentence = self._generate_sensors_sentence(sensor_seq, window_features, sensor_details)

        # Combine into full caption
        caption_text = f"{when_sentence} {duration_sentence} {where_sentence} {sensors_sentence}"

        # Apply dataset-specific rules (Milan special cases from Sourish's code)
        if self.dataset_name == "milan":
            if "guest" in caption_text.lower():
                caption_text = where_sentence
            elif "master bedroom" in caption_text.lower() and "master bathroom" in caption_text.lower():
                caption_text = where_sentence

        return Caption(
            text=caption_text,
            window_id=window.metadata.window_id,
            caption_type='sourish_full'
        )

    def _generate_short_caption(self, window: ProcessedWindow,
                               window_features: WindowFeatures,
                               sensor_details: Optional[Dict[str, str]] = None) -> Optional[Caption]:
        """Generate short location-focused caption."""

        sensor_seq = window.events['sensor'].tolist() if 'sensor' in window.events.columns else []
        where_sentence = self._generate_where_sentence(sensor_seq, window_features, sensor_details)

        return Caption(
            text=where_sentence,
            window_id=window.metadata.window_id,
            caption_type='sourish_short'
        )

    def _get_start_and_end_time(self, time_seq: List) -> Tuple[str, str]:
        """Extract start and end time strings."""
        if not time_seq:
            return "00:00:00", "00:00:00"

        # Convert to datetime if needed
        if isinstance(time_seq[0], int):
            time_seq = [datetime.fromtimestamp(ts / 1000.0) for ts in time_seq]

        start_time = time_seq[0].strftime("%H:%M:%S") if hasattr(time_seq[0], 'strftime') else "00:00:00"
        end_time = time_seq[-1].strftime("%H:%M:%S") if hasattr(time_seq[-1], 'strftime') else "00:00:00"

        return start_time, end_time

    def _get_hour_from_time(self, time_str: str) -> str:
        """Extract hour from time string."""
        try:
            return time_str.split(":")[0]
        except:
            return "0"

    def _determine_zone(self, hour: int, dataset: str = None) -> str:
        """Determine time zone (morning, afternoon, etc.) from hour.

        Uses dataset-specific definitions from Sourish's code.
        """
        if dataset == "aras":
            if hour < 5:
                return "past midnight"
            elif 5 <= hour < 7:
                return "early morning"
            elif 7 <= hour < 14:
                return "morning"
            elif 14 <= hour < 18:
                return "afternoon"
            elif 18 <= hour < 21:
                return "evening"
            else:
                return "night"
        elif dataset == "cairo":
            if hour < 5:
                return "past midnight"
            elif 5 <= hour < 9:
                return "early morning"
            elif 9 <= hour < 11:
                return "morning"
            elif 11 <= hour < 17:
                return "late afternoon"
            elif 17 <= hour < 21:
                return "evening"
            else:
                return "night"
        else:
            # Default for milan, aruba, kyoto7
            if hour < 5:
                return "past midnight"
            elif 5 <= hour < 7:
                return "early morning"
            elif 7 <= hour < 12:
                return "morning"
            elif 12 <= hour < 17:
                return "afternoon"
            elif 17 <= hour < 20:
                return "evening"
            else:
                return "night"

    def _convert_num_to_text(self, num: int) -> str:
        """Convert number to text (e.g., 7 -> 'seven')."""
        try:
            return self.inflect_engine.number_to_words(num)
        except:
            return str(num)

    def _generate_when_sentence(self, start_time: str, end_time: str, first_event: pd.Series) -> str:
        """Generate 'when' sentence describing time of activity."""

        start_hour = int(self._get_hour_from_time(start_time))
        end_hour = int(self._get_hour_from_time(end_time))

        start_zone = self._determine_zone(start_hour, self.dataset_name)
        end_zone = self._determine_zone(end_hour, self.dataset_name)

        # Determine AM/PM
        start_am_pm = "AM " if start_hour < 12 else "PM "
        end_am_pm = "AM " if end_hour < 12 else "PM "

        # Dataset-specific formatting (from Sourish's code)
        if self.dataset_name == "aras":
            # ARAS doesn't include time numbers, just zones
            when_sentence = f"The activity started at {start_zone} and ended at {end_zone}."
        else:
            # Default format for milan, aruba, cairo, kyoto7
            when_sentence = (f"The activity started at {self._convert_num_to_text(start_hour)} "
                            f"{start_am_pm}{start_zone} and ended at {self._convert_num_to_text(end_hour)} "
                            f"{end_am_pm}{end_zone}.")

        return when_sentence

    def _get_time_diff(self, start_time: str, end_time: str) -> int:
        """Calculate time difference in seconds."""
        try:
            start_dt = datetime.strptime(start_time, "%H:%M:%S")
            end_dt = datetime.strptime(end_time, "%H:%M:%S")
            diff = (end_dt - start_dt).total_seconds()

            # Handle cases where end time is before start time (crosses midnight)
            if diff < 0:
                diff += 86400  # Add 24 hours

            # Round up to nearest second (following Sourish's math.ceil)
            import math
            return math.ceil(diff)
        except:
            return 0

    def _get_time_context(self, time_diff: int) -> str:
        """Convert time difference to text description."""
        if time_diff < 60:
            return f"{self._convert_num_to_text(time_diff)} seconds"
        elif time_diff < 3600:
            minutes = int(time_diff / 60)
            return f"{self._convert_num_to_text(minutes)} minutes"
        else:
            hours = int(time_diff / 3600)
            return f"{self._convert_num_to_text(hours)} hours"

    def _generate_duration_sentence(self, time_diff: int) -> str:
        """Generate duration sentence."""
        return f"The activity was performed for {self._get_time_context(time_diff)}."

    def _get_sensor_global_context(self, sensor_id: str) -> str:
        """Get location context for a sensor using dataset-specific mappings."""
        sensor_map = self._get_sensor_context_map()
        return sensor_map.get(sensor_id, "in the house ")

    def _generate_where_sentence(self, sensor_seq: List[str],
                                 window_features: WindowFeatures,
                                 sensor_details: Optional[Dict[str, str]] = None) -> str:
        """Generate 'where' sentence describing location.

        Uses Sourish's approach: extract location from sensor IDs using hard-coded mappings,
        then find two most common locations.
        """

        if not sensor_seq:
            return "The activity is taking place in the house."

        # Get location context for each sensor
        sensor_contexts = []
        for sensor_id in sensor_seq:
            context = self._get_sensor_global_context(sensor_id)
            sensor_contexts.append(context)

        if not sensor_contexts:
            return "The activity is taking place in the house."

        # Get two most common locations
        location_counts = Counter(sensor_contexts)
        most_common = location_counts.most_common(2)

        most_common_location = most_common[0][0]

        if len(most_common) == 1:
            second_most_common_location = most_common_location
        else:
            second_most_common_location = most_common[1][0]

        # Format the where sentence
        where_sentence = f"The activity is taking place {most_common_location} mainly and parts of it {second_most_common_location}."

        return where_sentence

    def _convert_sensor_type_to_text(self, sensor_id: str) -> str:
        """Convert sensor ID prefix to text description."""
        if not sensor_id:
            return "Motion sensor "

        prefix = sensor_id[0] if len(sensor_id) > 0 else ""

        # Handle multi-character prefixes for some datasets
        if len(sensor_id) >= 2:
            two_char_prefix = sensor_id[:2]
            multi_char_types = {
                "Ph": "Photocell sensor ",
                "Ir": "Infrared sensor ",
                "Fo": "Force sensor ",
                "Di": "Distance sensor ",
                "Co": "Contact sensor ",
                "So": "Sonar Distance sensor ",
                "Te": "Temperature sensor ",
            }
            if two_char_prefix in multi_char_types:
                return multi_char_types[two_char_prefix]

        # Single character prefixes
        sensor_types = {
            "M": "Motion sensor ",
            "D": "Door sensor ",
            "T": "Temperature sensor ",
            "P": "Pressure Mat sensor ",
            "R": "Magnetic sensor ",
            "E": "Smart Plug sensor ",
            "S": "Smart Phone sensor ",
            "I": "Item sensor ",
            "L": "Light switch sensor ",
            "A": "Burner Hot Water or Cold Water sensor "
        }

        return sensor_types.get(prefix, "Motion sensor ")

    def _generate_sensors_sentence(self, sensor_seq: List[str],
                                   window_features: WindowFeatures,
                                   sensor_details: Optional[Dict[str, str]] = None) -> str:
        """Generate sentence about most commonly fired sensors."""

        if not sensor_seq:
            return "Sensors were activated during this activity."

        # Get two most common sensors
        sensor_counts = Counter(sensor_seq)
        most_common = sensor_counts.most_common(2)

        if len(most_common) == 1:
            sensor_id = most_common[0][0]
            sensor_type = self._convert_sensor_type_to_text(sensor_id)
            sensor_context = self._get_sensor_global_context(sensor_id)

            sensors_sentence = f"The most commonly fired sensor in this activity is {sensor_type}{sensor_context}"
        else:
            sensor1_id = most_common[0][0]
            sensor2_id = most_common[1][0]

            sensor1_type = self._convert_sensor_type_to_text(sensor1_id)
            sensor2_type = self._convert_sensor_type_to_text(sensor2_id)

            sensor1_context = self._get_sensor_global_context(sensor1_id)
            sensor2_context = self._get_sensor_global_context(sensor2_id)

            sensors_sentence = (f"The two most commonly fired sensors in this activity are "
                              f"{sensor1_type}{sensor1_context} and "
                              f"{sensor2_type}{sensor2_context}.")

        return sensors_sentence

    def get_caption_statistics(self, all_captions: List[Caption]) -> Dict[str, Any]:
        """Compute statistics about generated captions."""
        if not all_captions:
            return {}

        # Length statistics
        lengths = [len(caption.tokens) for caption in all_captions]

        # Separate by type
        full_captions = [c for c in all_captions if c.caption_type == 'sourish_full']
        short_captions = [c for c in all_captions if c.caption_type == 'sourish_short']

        stats = {
            'total_captions': len(all_captions),
            'full_captions': len(full_captions),
            'short_captions': len(short_captions),
            'stats': {
                'mean_tokens': np.mean(lengths) if lengths else 0,
                'std_tokens': np.std(lengths) if lengths else 0,
                'min_tokens': np.min(lengths) if lengths else 0,
                'max_tokens': np.max(lengths) if lengths else 0
            },
            'sample_captions': [caption.text for caption in all_captions[:5]]
        }

        return stats
