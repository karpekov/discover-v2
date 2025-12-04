"""Baseline (enhanced) caption generation using sensor ontology and aggregation.

Ported from src/data/captions.py with adaptations for the new modular structure.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
import random
from datetime import datetime
from collections import Counter

from ..base import BaseCaptionGenerator, CaptionOutput
from ..config import RuleBasedCaptionConfig


class BaselineCaptionGenerator(BaseCaptionGenerator):
    """Generate enhanced baseline captions using sensor ontology and aggregation.

    This generator creates natural language captions with:
    - Temporal context (day of week, month, time of day)
    - Duration descriptions (with gap analysis)
    - Room transitions (with back-movement detection)
    - Sensor-specific details (bed, toilet, door, appliances)
    - Multiple caption variations (active vs passive voice)
    """

    def __init__(self, config: RuleBasedCaptionConfig):
        super().__init__(config)
        self.random = random.Random(config.random_seed)

    def generate(self,
                 sensor_sequence: List[Dict[str, Any]],
                 metadata: Dict[str, Any],
                 sample_id: str) -> CaptionOutput:
        """Generate captions for a sensor sequence."""

        # Convert sensor sequence to DataFrame for easier processing
        df = pd.DataFrame(sensor_sequence)

        # Normalize column names (handle both old and new formats)
        if 'timestamp' in df.columns and 'datetime' not in df.columns:
            df['datetime'] = df['timestamp']
        if 'sensor_id' in df.columns and 'sensor' not in df.columns:
            df['sensor'] = df['sensor_id']
        if 'room' in df.columns and 'room_id' not in df.columns:
            df['room_id'] = df['room']

        # Convert datetime strings to datetime objects if needed
        if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')

        captions = []
        caption_metadata = {}

        # Load sensor details if provided
        sensor_details = self._load_sensor_details() if self.config.sensor_details_path else None

        # Generate long captions
        if self.config.generate_long_captions:
            for i in range(self.config.num_captions_per_sample):
                caption_text, layer_b = self._generate_enhanced_caption(df, metadata, sensor_details)
                captions.append(caption_text)
                if layer_b and i == 0:  # Save layer_b only for first caption
                    caption_metadata['layer_b'] = layer_b

        # Generate short creative captions
        if self.config.generate_short_captions:
            short_captions = self._generate_short_creative_captions(df, metadata, sensor_details)
            captions.extend(short_captions)

        caption_metadata['caption_type'] = 'baseline'
        caption_metadata['num_long'] = self.config.num_captions_per_sample if self.config.generate_long_captions else 0
        caption_metadata['num_short'] = len(short_captions) if self.config.generate_short_captions else 0

        return CaptionOutput(
            captions=captions,
            sample_id=sample_id,
            metadata=caption_metadata
        )

    def generate_batch(self, samples: List[Dict[str, Any]]) -> List[CaptionOutput]:
        """Generate captions for a batch of samples."""
        outputs = []
        for sample in samples:
            output = self.generate(
                sensor_sequence=sample['sensor_sequence'],
                metadata=sample['metadata'],
                sample_id=sample['sample_id']
            )
            outputs.append(output)
        return outputs

    def _load_sensor_details(self) -> Optional[Dict[str, str]]:
        """Load sensor details from file if provided."""
        if not self.config.sensor_details_path:
            return None

        import json
        import os

        if not os.path.exists(self.config.sensor_details_path):
            print(f"Warning: Sensor details file not found: {self.config.sensor_details_path}")
            return None

        try:
            with open(self.config.sensor_details_path, 'r') as f:
                metadata = json.load(f)
            # Extract sensor details based on dataset
            if self.config.dataset_name:
                sensor_details = metadata.get(self.config.dataset_name, {}).get('sensor_details', {})
                # Convert None values to empty strings
                return {k: (v if v is not None else '') for k, v in sensor_details.items()}
            return {}
        except Exception as e:
            print(f"Warning: Could not load sensor details: {e}")
            return None

    def _generate_enhanced_caption(self,
                                   df: pd.DataFrame,
                                   metadata: Dict[str, Any],
                                   sensor_details: Optional[Dict[str, str]]) -> Tuple[str, str]:
        """Generate enhanced caption with Layer A (human-readable) and Layer B (structured)."""

        if len(df) == 0:
            return "No sensor activity detected.", ""

        # Extract temporal information
        first_event = df.iloc[0]
        last_event = df.iloc[-1]

        # Day of week
        dow = first_event['datetime'].strftime('%A')

        # Month description
        month_desc = self._generate_month_description(first_event['datetime'])

        # Time of day
        tod = metadata.get('tod_bucket', None)
        if tod is None:
            # Compute time of day from timestamp if not provided
            hour = first_event['datetime'].hour
            if hour < 5:
                tod = 'night'
            elif hour < 12:
                tod = 'morning'
            elif hour < 17:
                tod = 'afternoon'
            elif hour < 20:
                tod = 'evening'
            else:
                tod = 'night'
        if isinstance(tod, (int, float)):
            tod = str(tod)
        tod = tod.replace('_', ' ').replace('after midnight', 'night')

        # Duration in minutes
        dur = round(metadata['duration_seconds'] / 60.0, 1)

        # Extract room sequence
        room_sequence = self._extract_room_sequence(df)
        unique_rooms = self._get_unique_consecutive_rooms(room_sequence)

        # Extract precomputed special sensors from metadata (if available)
        special_sensors = metadata.get('special_sensors', None)

        # Aggregate sensors for Layer B (still needed for full detail)
        agg, ids_per_room = self._aggregate_sensors(df, sensor_details)

        # Generate Layer A caption with special sensors (or None if not available)
        layer_a = self._generate_natural_caption(
            dow, month_desc, tod, dur, unique_rooms, df, metadata,
            special_sensors=special_sensors
        )

        # Generate Layer B (structured evidence) with special sensors
        layer_b = self._generate_layer_b(
            agg, metadata, first_event, last_event, dow, tod, dur,
            special_sensors=special_sensors
        )

        return layer_a, layer_b

    def _extract_room_sequence(self, df: pd.DataFrame) -> List[str]:
        """Extract room sequence from dataframe."""
        if 'room_id' in df.columns:
            return df['room_id'].tolist()
        elif 'room' in df.columns:
            return df['room'].tolist()
        return ['unknown'] * len(df)

    def _get_unique_consecutive_rooms(self, room_sequence: List[str]) -> List[str]:
        """Get list of unique consecutive rooms (remove adjacent duplicates)."""
        unique_rooms = []
        for room in room_sequence:
            # Convert to string if not already
            if not isinstance(room, str):
                room = str(room) if room is not None and (not isinstance(room, float) or not pd.isna(room)) else 'unknown'
            if not unique_rooms or room != unique_rooms[-1]:
                unique_rooms.append(room)
        return unique_rooms

    def _aggregate_sensors(self,
                          df: pd.DataFrame,
                          sensor_details: Optional[Dict[str, str]]) -> Tuple[Dict[str, Dict], Dict[str, Set[str]]]:
        """Aggregate sensor activations by room and type."""

        agg = {}
        ids_per_room = {}

        for _, event in df.iterrows():
            sensor_id = event.get('sensor', '')
            room = event.get('room_id', event.get('room', 'unknown'))

            # Get sensor type and tags
            sensor_type, _, tags = self._sensor_ontology(sensor_id, sensor_details)

            # Initialize room in aggregation
            if room not in agg:
                agg[room] = {}

            # Aggregate by sensor type
            if sensor_type not in agg[room]:
                agg[room][sensor_type] = set()
            agg[room][sensor_type].add(sensor_id)

            # Count area tags
            for tag in tags:
                tag_key = f"{tag}_count"
                if tag_key not in agg[room]:
                    agg[room][tag_key] = 0
                agg[room][tag_key] += 1

            # Track sensor IDs per room
            if room not in ids_per_room:
                ids_per_room[room] = set()
            ids_per_room[room].add(sensor_id)

        return agg, ids_per_room

    def _sensor_ontology(self,
                        sensor_id: str,
                        sensor_details: Optional[Dict[str, str]]) -> Tuple[str, str, List[str]]:
        """Extract sensor type, room, and area tags from sensor description."""

        # Sensor type by prefix
        if sensor_id.startswith("M"):
            sensor_type = "motion"
        elif sensor_id.startswith("D"):
            sensor_type = "door"
        elif sensor_id.startswith("T"):
            sensor_type = "temp"
        else:
            sensor_type = "other"

        desc = sensor_details.get(sensor_id, '') if sensor_details else ''
        desc_lower = desc.lower()

        # Check for bed sensor (special case)
        if "bed" in desc_lower:
            sensor_type = "bed"

        # Room extraction
        room_keywords = [
            "master bedroom", "master bathroom", "kitchen", "entryway", "dining room",
            "living room", "workspace", "tv room", "hallway", "guest bathroom",
            "guest bedroom", "bathroom", "bedroom", "laundry"
        ]
        room = next((r for r in room_keywords if r in desc_lower), "unknown")

        # Area/object tags
        area_keywords = [
            "bed", "fridge", "stove", "toilet", "sink", "cabinet", "desk",
            "armchair", "chair", "island", "medicine", "bathtub", "shower", "table", "pill",
            "entrance", "doorway"
        ]
        tags = [k for k in area_keywords if k in desc_lower]

        return sensor_type, room, tags

    def _select_salient_sensors(self,
                               unique_rooms: List[str],
                               agg: Dict[str, Dict]) -> List[str]:
        """Select salient sensors for Layer A caption."""

        salient = []
        seen_rooms = set()

        # High-priority cues (bed, toilet, door)
        for room, feats in agg.items():
            if "bed_count" in feats and feats["bed_count"] > 0:
                salient.append("bed sensor")
            if "toilet_count" in feats and feats["toilet_count"] > 0:
                salient.append("toilet area PIR")
            if "door" in feats and len(feats["door"]) > 0:
                salient.append("entry door")

        # Cover trajectory with one cue per room
        for room in unique_rooms:
            if room in seen_rooms:
                continue

            room_key = room.replace(' ', '_')
            feats = agg.get(room_key, {})

            # Look for area tags
            area_priorities = ["fridge", "stove", "sink", "desk", "armchair", "chair", "cabinet", "bath", "bathtub", "medicine", "pill"]
            area = next((a for a in area_priorities if f"{a}_count" in feats and feats[f"{a}_count"] > 0), None)

            if area:
                room_clean = room.replace('_', ' ')
                room_short = room_clean.split()[0] if room != "unknown" else "room"
                salient.append(f"{room_short} motion ({area} area)")
            elif "motion" in feats:
                room_clean = room.replace('_', ' ')
                room_short = room_clean.split()[0] if room != "unknown" else "room"
                salient.append(f"{room_short} motion")

            seen_rooms.add(room)

            if len(salient) >= 3:
                break

        return salient[:3]

    def _generate_natural_caption(self,
                                 dow: str,
                                 month_desc: str,
                                 tod: str,
                                 dur: float,
                                 unique_rooms: List[str],
                                 df: pd.DataFrame,
                                 metadata: Dict[str, Any],
                                 special_sensors: Optional[Dict[str, Any]] = None) -> str:
        """Generate natural sentence-like caption."""

        # Generate duration description
        duration_desc = self._generate_duration_description(dur, df)

        # Get time phrase
        time_phrase = self._generate_time_phrase(tod)

        # Generate room description
        room_desc = self._generate_room_description(unique_rooms)

        # Generate sensor description (with special sensors if available)
        sensor_desc = self._generate_sensor_description(special_sensors)

        # Choose between active and passive mode
        use_active_mode = self.random.choice([True, False])

        # Time context
        temporal_at_start = self.random.choice([True, False])
        time_context = f"on {dow} in {month_desc} {time_phrase}"

        if use_active_mode:
            resident_terms = ['resident', 'dweller', 'occupant', 'person', 'individual']
            resident_term = self.random.choice(resident_terms)
            return self._generate_active_caption(
                time_context, temporal_at_start, resident_term,
                duration_desc, room_desc, sensor_desc, unique_rooms
            )
        else:
            return self._generate_passive_caption(
                time_context, temporal_at_start,
                duration_desc, room_desc, sensor_desc
            )

    def _generate_active_caption(self,
                                time_context: str,
                                temporal_at_start: bool,
                                resident_term: str,
                                duration_desc: str,
                                room_desc: str,
                                sensor_desc: Optional[str],
                                unique_rooms: List[str]) -> str:
        """Generate caption with active resident actions."""

        if len(unique_rooms) > 1:
            action_verbs = ['moved', 'transitioned', 'traveled', 'went', 'proceeded', 'navigated']
        else:
            action_verbs = ['was active', 'spent time', 'was present', 'remained', 'stayed']

        action_verb = self.random.choice(action_verbs)

        # Generate templates with or without sensor description
        if sensor_desc:
            if temporal_at_start:
                templates = [
                    f"{time_context.capitalize()}, the {resident_term} {action_verb} {duration_desc} {room_desc}, with {sensor_desc}.",
                    f"{time_context.capitalize()}, {resident_term} {action_verb} {duration_desc} {room_desc}, showing {sensor_desc}.",
                    f"The {resident_term} {action_verb} {time_context} {duration_desc} {room_desc}, indicating {sensor_desc}."
                ]
            else:
                templates = [
                    f"The {resident_term} {action_verb} {duration_desc} {room_desc}, with {sensor_desc} {time_context}.",
                    f"{resident_term.capitalize()} {action_verb} {duration_desc} {room_desc}, showing {sensor_desc} {time_context}.",
                    f"{resident_term.capitalize()} was {action_verb.replace('was ', '')} {duration_desc} {room_desc}, indicating {sensor_desc} {time_context}."
                ]
        else:
            # No special sensors - simpler caption
            if temporal_at_start:
                templates = [
                    f"{time_context.capitalize()}, the {resident_term} {action_verb} {duration_desc} {room_desc}.",
                    f"{time_context.capitalize()}, {resident_term} {action_verb} {duration_desc} {room_desc}.",
                    f"The {resident_term} {action_verb} {time_context} {duration_desc} {room_desc}."
                ]
            else:
                templates = [
                    f"The {resident_term} {action_verb} {duration_desc} {room_desc} {time_context}.",
                    f"{resident_term.capitalize()} {action_verb} {duration_desc} {room_desc} {time_context}.",
                    f"{resident_term.capitalize()} was {action_verb.replace('was ', '')} {duration_desc} {room_desc} {time_context}."
                ]

        return self.random.choice(templates)

    def _generate_passive_caption(self,
                                 time_context: str,
                                 temporal_at_start: bool,
                                 duration_desc: str,
                                 room_desc: str,
                                 sensor_desc: Optional[str]) -> str:
        """Generate caption with passive detection language."""

        action_words = ['Activity', 'Motion', 'Movement']
        detection_words = ['detected', 'recorded', 'observed', 'captured', 'occurred', 'happened', 'took place']

        action_word = self.random.choice(action_words)
        detection_word = self.random.choice(detection_words)

        # Generate templates with or without sensor description
        if sensor_desc:
            if temporal_at_start:
                templates = [
                    f"{time_context.capitalize()}, {action_word.lower()} {detection_word} {duration_desc} {room_desc}, with {sensor_desc}.",
                    f"{action_word} {detection_word} {time_context} {duration_desc} {room_desc}, showing {sensor_desc}.",
                    f"{time_context.capitalize()} saw {action_word.lower()} {duration_desc} {room_desc} with {sensor_desc}."
                ]
            else:
                templates = [
                    f"{action_word} {detection_word} {duration_desc} {room_desc} with {sensor_desc} {time_context}.",
                    f"Sensors {detection_word} {action_word.lower()} {duration_desc} {room_desc}, showing {sensor_desc} {time_context}.",
                    f"{action_word} was {detection_word} {duration_desc} {room_desc} with {sensor_desc} {time_context}."
                ]
        else:
            # No special sensors - simpler caption
            if temporal_at_start:
                templates = [
                    f"{time_context.capitalize()}, {action_word.lower()} {detection_word} {duration_desc} {room_desc}.",
                    f"{action_word} {detection_word} {time_context} {duration_desc} {room_desc}.",
                    f"{time_context.capitalize()} saw {action_word.lower()} {duration_desc} {room_desc}."
                ]
            else:
                templates = [
                    f"{action_word} {detection_word} {duration_desc} {room_desc} {time_context}.",
                    f"Sensors {detection_word} {action_word.lower()} {duration_desc} {room_desc} {time_context}.",
                    f"{action_word} was {detection_word} {duration_desc} {room_desc} {time_context}."
                ]

        return self.random.choice(templates)

    def _generate_duration_description(self, dur: float, df: pd.DataFrame) -> str:
        """Generate natural duration description."""
        minutes = round(dur)

        # Check for long gaps
        has_long_gaps = False
        if 'time_delta_sec' in df.columns:
            max_gap_sec = df['time_delta_sec'].max()
            has_long_gaps = max_gap_sec > 300  # 5 minutes

        # Base descriptions
        if minutes < 1:
            base_descriptions = [
                "lasting less than a minute",
                "in under a minute",
                "for mere seconds",
                "briefly"
            ]
        elif minutes == 1:
            base_descriptions = [
                "lasting 1 minute",
                "over a 1-minute span",
                "for about a minute"
            ]
        elif minutes < 5:
            base_descriptions = [
                f"lasting {minutes} minutes",
                f"over a {minutes}-minute span",
                f"for {minutes} minutes"
            ]
        elif minutes < 10:
            base_descriptions = [
                f"extending {minutes} minutes",
                f"spanning {minutes} minutes",
                f"over a {minutes}-minute period"
            ]
        else:
            base_descriptions = [
                f"over a substantial period of time ({minutes} minutes)",
                f"sustained for {minutes} minutes",
                f"lasting {minutes} minutes"
            ]

        base_desc = self.random.choice(base_descriptions)

        # Add gap information if applicable
        if has_long_gaps and self.random.choice([True, False]):
            gap_additions = [
                ", with significant pauses between sensor activations",
                ", including periods where a lot of time passed before the next sensor got activated"
            ]
            base_desc += self.random.choice(gap_additions)

        return base_desc

    def _generate_time_phrase(self, tod: str) -> str:
        """Generate time of day phrase."""
        time_phrases = {
            'morning': ['in the morning', 'during morning hours'],
            'afternoon': ['in the afternoon', 'during the day'],
            'evening': ['in the evening', 'during evening hours'],
            'night': ['during the night', 'late at night']
        }

        tod_key = 'night' if 'night' in tod else tod.split()[0] if tod != 'unknown' else 'evening'
        phrases = time_phrases.get(tod_key, time_phrases['evening'])
        return self.random.choice(phrases)

    def _generate_month_description(self, timestamp: datetime) -> str:
        """Generate diverse month descriptions."""
        month_num = timestamp.month

        full_names = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        abbrev_names = [
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ]

        style = self.random.choice(['full', 'abbrev'])

        if style == 'full':
            return full_names[month_num - 1]
        else:
            return abbrev_names[month_num - 1]

    def _generate_room_description(self, unique_rooms: List[str]) -> str:
        """Generate natural room transition description."""
        # Clean room names: replace underscores with spaces
        clean_rooms = [room.replace('_', ' ') for room in unique_rooms]

        if len(clean_rooms) == 1:
            return f"in {clean_rooms[0]}"
        elif len(clean_rooms) == 2:
            transition_words = ['movement from', 'transition from', 'activity from']
            return f"{self.random.choice(transition_words)} {clean_rooms[0]} to {clean_rooms[1]}"
        else:
            # Check for back movements
            if len(clean_rooms) == 3 and clean_rooms[0] == clean_rooms[2]:
                return f"movement from {clean_rooms[0]} to {clean_rooms[1]} and back to {clean_rooms[0]}"
            else:
                return f"movement from {clean_rooms[0]} to {clean_rooms[1]} then to {clean_rooms[2]}"

    def _generate_sensor_description(self, special_sensors: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Generate sensor activity description, incorporating special sensors if available.

        Returns None if no special sensors are available.
        """

        # Use precomputed special sensors if available
        if not special_sensors:
            return None

        special_list = special_sensors.get('special_sensors_triggered', [])
        primary_special = special_sensors.get('primary_special_sensor', None)

        if not special_list:
            return None

        descriptions = []

        # First, add primary special sensor with emphasis if it exists
        if primary_special:
            descriptions.append(self._format_special_sensor_desc(primary_special, is_primary=True))

        # Then add other special sensors WITHOUT emphasis (excluding primary)
        for sensor_detail in special_list:
            if sensor_detail != primary_special:
                descriptions.append(self._format_special_sensor_desc(sensor_detail, is_primary=False))

        # Format the descriptions
        if len(descriptions) == 1:
            return descriptions[0]
        elif len(descriptions) == 2:
            return f"{descriptions[0]} and {descriptions[1]}"
        elif len(descriptions) <= 4:
            return f"{', '.join(descriptions[:-1])}, and {descriptions[-1]}"
        else:
            # Too many special sensors, just mention top ones
            return f"{', '.join(descriptions[:3])}, and {len(descriptions) - 3} other special sensors"

    def _format_special_sensor_desc(self, sensor_detail: str, is_primary: bool = False) -> str:
        """Format a special sensor detail into a natural description."""
        detail_lower = sensor_detail.lower()

        # Map sensor details to natural descriptions
        if 'bed' in detail_lower:
            options = ['activity near bed', 'bed sensor activation', 'movement in sleeping area', 'lying on bed']
        elif 'medicine' in detail_lower or 'pill' in detail_lower:
            options = ['medicine cabinet access', 'activity at medicine cabinet', 'taking medicine']
        elif 'fridge' in detail_lower or 'refrigerator' in detail_lower:
            options = ['fridge access', 'refrigerator activity', 'movement by fridge', 'opening fridge']
        elif 'stove' in detail_lower or 'cooking' in detail_lower:
            options = ['stove activity', 'cooking area motion', 'activity near stove', 'using stove']
        elif 'desk' in detail_lower:
            options = ['desk activity', 'workspace motion', 'activity at desk', 'working at desk', 'using desk', 'sitting at desk']
        elif 'table' in detail_lower and 'dining' in detail_lower:
            options = ['dining table activity', 'movement at dining table', 'eating at dining table', 'sitting at dining table']
        elif 'table' in detail_lower:
            options = ['table activity', 'movement near table', 'using table', 'sitting at table']
        elif 'toilet' in detail_lower:
            options = ['toilet area activity', 'movement near toilet', 'using toilet']
        elif 'sink' in detail_lower:
            options = ['sink area activity', 'movement by sink']
        elif 'cabinet' in detail_lower:
            options = ['cabinet access', 'movement by cabinet']
        elif 'door' in detail_lower and ('entry' in detail_lower or 'entrance' in detail_lower):
            options = ['entry door activity', 'entrance movement', 'opening entry door', 'closing entry door', 'entering the house', 'exiting the house']
        elif 'door' in detail_lower:
            options = ['door activity', 'door sensor activation']
        elif 'couch' in detail_lower or 'sofa' in detail_lower:
            options = ['couch activity', 'movement on sofa', 'sitting on sofa', 'using sofa']
        elif 'armchair' in detail_lower or 'chair' in detail_lower:
            options = ['chair activity', 'seating area motion', 'sitting on chair', 'using chair']
        else:
            # Generic description from the sensor detail itself
            options = [f'activity at {sensor_detail}', f'{sensor_detail} activity']

        desc = self.random.choice(options)

        # Add emphasis for primary special sensor
        if is_primary:
            emphasis = ['frequent', 'repeated', 'significant', 'notable']
            return f"{self.random.choice(emphasis)} {desc}"

        return desc

    def _generate_layer_b(self,
                         agg: Dict[str, Dict],
                         metadata: Dict[str, Any],
                         first_event: pd.Series,
                         last_event: pd.Series,
                         dow: str,
                         tod: str,
                         dur: float,
                         special_sensors: Optional[Dict[str, Any]] = None) -> str:
        """Generate Layer B: structured evidence with special sensor information."""

        # Build compact dict
        detail = {}
        for room, feats in agg.items():
            detail[room] = {}
            for k, v in feats.items():
                if k in ["motion", "door", "temp", "bed"] and isinstance(v, set):
                    detail[room][k] = sorted(list(v))
                elif isinstance(v, (int, float)) and v > 0:
                    detail[room][k] = v

        start_time = first_event['datetime'].strftime('%H:%M')
        end_time = last_event['datetime'].strftime('%H:%M')

        # Build base layer B
        layer_b = (f"span={start_time}-{end_time}; dur={dur}m; dow={dow}; "
                  f"month={first_event['datetime'].month}; tod={tod}; "
                  f"rooms={list(detail.keys())}; sensors={detail}")

        # Add special sensor information if available
        if special_sensors:
            special_list = special_sensors.get('special_sensors_triggered', [])
            primary_special = special_sensors.get('primary_special_sensor', None)
            sensor_counts = special_sensors.get('special_sensor_counts', {})

            if special_list:
                layer_b += f"; special_sensors={special_list}"

                if primary_special:
                    layer_b += f"; primary_special={primary_special}"

                # Add counts for sensors triggered multiple times
                frequent_sensors = {k: v for k, v in sensor_counts.items() if v >= 2}
                if frequent_sensors:
                    layer_b += f"; special_counts={frequent_sensors}"
            else:
                # No special sensors triggered in this window
                layer_b += "; special_sensors=[]"
        else:
            # Special sensor metadata not available
            layer_b += "; special_sensors=not_available"

        return layer_b

    def _generate_short_creative_captions(self,
                                         df: pd.DataFrame,
                                         metadata: Dict[str, Any],
                                         sensor_details: Optional[Dict[str, str]]) -> List[str]:
        """Generate 2 short creative captions (under 5 tokens each)."""

        if len(df) == 0:
            return []

        captions = []

        # Extract context
        unique_rooms = self._get_unique_consecutive_rooms(self._extract_room_sequence(df))
        first_event = df.iloc[0]
        tod = metadata.get('tod_bucket', None)
        if tod is None:
            # Compute time of day from timestamp if not provided
            hour = first_event['datetime'].hour
            if hour < 5:
                tod = 'night'
            elif hour < 12:
                tod = 'morning'
            elif hour < 17:
                tod = 'afternoon'
            elif hour < 20:
                tod = 'evening'
            else:
                tod = 'night'
        is_night = 'night' in tod or 'after midnight' in tod
        is_day = 'morning' in tod or 'afternoon' in tod

        # Generate first short caption (location-focused)
        if len(unique_rooms) == 1:
            room = unique_rooms[0].replace('_', ' ')
            if 'kitchen' in room.lower():
                options = ["kitchen activity night" if is_night else "kitchen activity day",
                          "late kitchen motion" if is_night else "daytime kitchen use"]
            elif 'bedroom' in room.lower():
                options = ["bedroom night activity" if is_night else "bedroom day activity",
                          "sleeping area motion" if is_night else "bedroom movement"]
            elif 'bathroom' in room.lower():
                options = ["bathroom activity", "restroom motion"]
            elif 'living' in room.lower():
                options = ["living room activity", "lounge area motion"]
            else:
                options = [f"{room} activity", f"nighttime {room}" if is_night else f"daytime {room}"]
        else:
            if len(unique_rooms) == 2:
                options = ["room transitions", "multiple rooms"]
            else:
                options = ["multiple room transitions", "complex movement"]

        captions.append(self.random.choice(options))

        # Generate second short caption (activity-focused)
        if len(unique_rooms) == 1:
            room = unique_rooms[0].replace('_', ' ')
            if 'kitchen' in room.lower():
                options2 = ["food preparation", "culinary activity"]
            elif 'bedroom' in room.lower():
                options2 = ["sleep preparation" if is_night else "bedroom tasks", "bedtime routine" if is_night else "personal space"]
            elif 'bathroom' in room.lower():
                options2 = ["personal hygiene", "bathroom routine"]
            elif 'living' in room.lower():
                options2 = ["relaxation time", "leisure activity"]
            else:
                options2 = ["daily activity", "routine motion"]
        else:
            if any('kitchen' in r.lower() for r in unique_rooms):
                options2 = ["meal related", "food activity"]
            elif any('bathroom' in r.lower() for r in unique_rooms):
                options2 = ["hygiene routine", "personal care"]
            else:
                options2 = ["daily routine", "home navigation"]

        captions.append(self.random.choice(options2))

        return captions

