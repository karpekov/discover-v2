"""Enhanced caption generation using sensor ontology and aggregation."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import random
from datetime import datetime

from .data_config import CaptionConfig
from .windowing import ProcessedWindow
from .features import WindowFeatures


@dataclass
class Caption:
    """Container for enhanced captions with layers."""
    text: str  # Layer A: Compact, human-readable
    window_id: int
    caption_type: str = 'enhanced'  # 'enhanced', 'basic', 'short_creative'

    # Additional fields for enhanced captions
    layer_b: Optional[str] = None  # Layer B: Detailed, structured evidence

    @property
    def tokens(self) -> List[str]:
        """Tokenized Layer A caption."""
        return self.text.split()

    @property
    def is_short(self) -> bool:
        """Check if this is a short caption (< 10 tokens)."""
        return len(self.tokens) < 10


class CaptionGenerator:
    """Generate enhanced captions using sensor ontology and aggregation."""

    def __init__(self, config: CaptionConfig):
        self.config = config
        self.random = random.Random(42)

    def generate_captions(self, window: ProcessedWindow,
                         window_features: WindowFeatures,
                         sensor_details: Optional[Dict[str, str]] = None) -> Dict[str, List[Caption]]:
        """Generate captions and return them separated into long and short arrays."""

        long_captions = []
        short_captions = []

        # Generate 2 long templated captions
        if sensor_details:
            long_captions.extend(self._generate_enhanced_captions(window, window_features, sensor_details))
            long_captions.extend(self._generate_enhanced_captions(window, window_features, sensor_details))
        else:
            long_captions.extend(self._generate_basic_captions(window, window_features))
            long_captions.extend(self._generate_basic_captions(window, window_features))

        # Generate 2 short creative captions
        short_captions.extend(self._generate_short_creative_captions(window, window_features, sensor_details))

        # Return as separate arrays
        return {
            'long': long_captions,
            'short': short_captions,
            'all': long_captions + short_captions
        }

    def _generate_enhanced_captions(self, window: ProcessedWindow,
                                   window_features: WindowFeatures,
                                   sensor_details: Dict[str, str]) -> List[Caption]:
        """Generate enhanced captions using sensor ontology."""

        layer_a, layer_b = self._make_enhanced_captions(window, window_features, sensor_details)

        caption = Caption(
            text=layer_a,
            window_id=window.metadata.window_id,
            caption_type='enhanced',
            layer_b=layer_b
        )

        return [caption]

    def _generate_basic_captions(self, window: ProcessedWindow,
                                window_features: WindowFeatures) -> List[Caption]:
        """Generate basic captions for datasets without sensor details."""

        # Simple trajectory-based caption
        room_sequence = window_features.room_sequence
        unique_rooms = []
        for room in room_sequence:
            # Convert to string if it's not already (handles NaN, floats, etc.)
            if not isinstance(room, str):
                room = str(room) if room is not None and (not isinstance(room, float) or not pd.isna(room)) else 'unknown'
            if not unique_rooms or room != unique_rooms[-1]:
                unique_rooms.append(room)

        if len(unique_rooms) == 1:
            text = f"Activity in {unique_rooms[0].replace('_', ' ')}."
        elif len(unique_rooms) == 2:
            text = f"Movement from {unique_rooms[0].replace('_', ' ')} to {unique_rooms[1].replace('_', ' ')}."
        else:
            rooms_str = " → ".join([r.replace('_', ' ') for r in unique_rooms[:3]])
            text = f"Activity across {rooms_str}."

        caption = Caption(
            text=text,
            window_id=window.metadata.window_id,
            caption_type='basic'
        )

        return [caption]

    def _sensor_ontology(self, sensor_id: str, desc: str) -> Tuple[str, str, List[str]]:
        """Extract sensor type, room, and area tags from sensor description."""

        # Sensor type by prefix
        if sensor_id.startswith("M"):
            sensor_type = "motion"
            # Special case for bed sensor
            if "bed" in desc.lower():
                sensor_type = "bed"
        elif sensor_id.startswith("D"):
            sensor_type = "door"
        elif sensor_id.startswith("T"):
            sensor_type = "temp"
        else:
            sensor_type = "other"

        # Room extraction - look for room names in description
        room_keywords = [
            "master bedroom", "master bathroom", "kitchen", "entryway", "dining room",
            "living room", "workspace", "tv room", "hallway", "guest bathroom",
            "guest bedroom", "bathroom", "bedroom", "laundry"
        ]

        desc_lower = desc.lower()
        room = next((r for r in room_keywords if r in desc_lower), "unknown")

        # Area/object tags
        area_keywords = [
            "bed", "fridge", "stove", "toilet", "sink", "cabinet", "desk",
            "armchair", "chair", "island", "medicine", "bathtub", "shower", "table", "pill",
            "entrance", "doorway"
        ]

        tags = [k for k in area_keywords if k in desc_lower]

        return sensor_type, room, tags

    def _aggregate(self, window: ProcessedWindow,
                  sensor_details: Dict[str, str]) -> Tuple[Dict[str, Dict], Dict[str, Set[str]]]:
        """Aggregate sensor activations by room and type."""

        agg = {}
        ids_per_room = {}

        for _, event in window.events.iterrows():
            sensor_id = event.get('sensor', '')
            desc = sensor_details.get(sensor_id, '')

            # Use room_id from the event data instead of extracting from description
            room = event.get('room_id', 'unknown')

            # Get sensor type and tags from description
            sensor_type, _, tags = self._sensor_ontology(sensor_id, desc)

            # Aggregate by room and type
            if room not in agg:
                agg[room] = {}
            if sensor_type not in agg[room]:
                agg[room][sensor_type] = set()
            agg[room][sensor_type].add(sensor_id)

            # Count area tags
            for tag in tags:
                tag_key = f"{tag}_count"  # Use different key to avoid conflict with sensor sets
                if tag_key not in agg[room]:
                    agg[room][tag_key] = 0
                agg[room][tag_key] += 1

            # Track sensor IDs per room
            if room not in ids_per_room:
                ids_per_room[room] = set()
            ids_per_room[room].add(sensor_id)

        return agg, ids_per_room

    def _select_salient(self, traj_rooms: List[str], agg: Dict[str, Dict]) -> List[str]:
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

        # Cover trajectory with one cue per room (prefer area tags)
        for room in traj_rooms:
            if room in seen_rooms:
                continue

            # Convert room name back to underscore format to match aggregation keys
            room_key = room.replace(' ', '_')
            feats = agg.get(room_key, {})

            # Look for area tags in priority order
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

            # Limit to 3 salient items
            if len(salient) >= 3:
                break

        return salient[:3]

    def _make_enhanced_captions(self, window: ProcessedWindow,
                               window_features: WindowFeatures,
                               sensor_details: Dict[str, str]) -> Tuple[str, str]:
        """Generate Layer A and Layer B captions."""

        # Extract temporal information
        first_event = window.events.iloc[0]
        last_event = window.events.iloc[-1]

        # Day of week (full name)
        dow = first_event['datetime'].strftime('%A')  # Monday, Tuesday, etc.

        # Month information (diverse descriptions)
        month_desc = self._generate_month_description(first_event['datetime'])

        # Time of day (clean up)
        tod = first_event.get('tod_bucket', 'unknown')
        if isinstance(tod, (int, float)):
            tod = str(tod)
        tod = tod.replace('_', ' ').replace('after midnight', 'night')

        # Duration in minutes
        dur = round(window.metadata.duration_sec / 60.0, 1)

        # Clean room names (replace underscores with spaces)
        room_sequence = window_features.room_sequence
        unique_rooms = []
        for room in room_sequence:
            # Convert to string if it's not already (handles NaN, floats, etc.)
            if not isinstance(room, str):
                room = str(room) if room is not None and (not isinstance(room, float) or not pd.isna(room)) else 'unknown'
            clean_room = room.replace('_', ' ')
            if not unique_rooms or clean_room != unique_rooms[-1]:
                unique_rooms.append(clean_room)

        # Aggregate and select salient sensors
        agg, ids_per_room = self._aggregate(window, sensor_details)
        salient = self._select_salient(unique_rooms, agg)

        # Generate natural sentence-like caption
        layer_a = self._generate_natural_caption(dow, month_desc, tod, dur, unique_rooms, salient, window)

        # Layer B: Detailed, structured evidence
        # Build compact dict: room: type[list IDs]
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

        layer_b = (f"span={start_time}–{end_time}; dur={dur}m; dow={dow}; "
                  f"month={first_event['datetime'].month}; tod={tod}; "
                  f"rooms={list(detail.keys())}; sensors={detail}")

        return layer_a, layer_b

    def _generate_natural_caption(self, dow: str, month_desc: str, tod: str, dur: float,
                                 unique_rooms: List[str], salient: List[str], window: ProcessedWindow) -> str:
        """Generate natural sentence-like captions with variations."""

        # Generate duration description with gap analysis
        duration_desc = self._generate_duration_description(dur, window)

        # Get time phrase
        time_phrase = self._generate_time_phrase(tod)

        # Generate room transition description with resident terms and back movements
        room_desc = self._generate_room_description(unique_rooms)

        # Generate sensor description
        sensor_desc = self._generate_sensor_description(salient)

        # Choose between two modes: passive detection or active resident actions
        use_active_mode = self.random.choice([True, False])

        # Resident terms for room movements
        resident_terms = ['resident', 'dweller', 'occupant', 'person', 'individual']
        resident_term = self.random.choice(resident_terms)

        # Randomize temporal order (beginning vs end)
        temporal_at_start = self.random.choice([True, False])
        time_context = f"on {dow} in {month_desc} {time_phrase}"

        if use_active_mode:
            # ACTIVE MODE: Resident performs actions
            return self._generate_active_caption(time_context, temporal_at_start, resident_term, duration_desc, room_desc, sensor_desc, unique_rooms)
        else:
            # PASSIVE MODE: Motion/activity is detected
            return self._generate_passive_caption(time_context, temporal_at_start, duration_desc, room_desc, sensor_desc)

    def _generate_active_caption(self, time_context: str, temporal_at_start: bool, resident_term: str,
                                duration_desc: str, room_desc: str, sensor_desc: str, unique_rooms: List[str]) -> str:
        """Generate captions with active resident actions."""

        # Active verbs for residents
        if len(unique_rooms) > 1:
            action_verbs = ['moved', 'transitioned', 'traveled', 'went', 'proceeded', 'navigated']
        else:
            action_verbs = ['was active', 'spent time', 'was present', 'remained', 'stayed']

        action_verb = self.random.choice(action_verbs)

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

        return self.random.choice(templates)

    def _generate_passive_caption(self, time_context: str, temporal_at_start: bool,
                                 duration_desc: str, room_desc: str, sensor_desc: str) -> str:
        """Generate captions with passive detection language."""

        # Passive detection words
        action_words = ['Activity', 'Motion', 'Movement']
        detection_words = ['detected', 'recorded', 'observed', 'captured', 'occurred', 'happened', 'took place']

        action_word = self.random.choice(action_words)
        detection_word = self.random.choice(detection_words)

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

        return self.random.choice(templates)

    def _generate_duration_description(self, dur: float, window: ProcessedWindow) -> str:
        """Generate natural duration descriptions with actual minutes, enhanced variants, and gap analysis."""
        minutes = round(dur)

        # Analyze gaps between sensor readings (if time_delta_sec column exists)
        has_long_gaps = False
        if 'time_delta_sec' in window.events.columns:
            # Check for gaps longer than 5 minutes (300 seconds)
            max_gap_sec = window.events['time_delta_sec'].max()
            has_long_gaps = max_gap_sec > 300

        # Base duration descriptions
        base_descriptions = []

        if minutes < 1:
            base_descriptions = [
                "lasting less than a minute",
                "in under a minute",
                "for mere seconds",
                "briefly",
                "in a very short span of time",
                "over just a few seconds",
                "in an extremely brief period"
            ]
        elif minutes == 1:
            base_descriptions = [
                "lasting 1 minute",
                "over a 1-minute span",
                "for about a minute",
                "in a single minute",
                "in a very short span of time"
            ]
        elif minutes < 5:
            base_descriptions = [
                f"lasting {minutes} minutes",
                f"over a {minutes}-minute span",
                f"for {minutes} minutes",
                f"across {minutes} minutes"
            ]
        elif minutes < 10:
            base_descriptions = [
                f"extending {minutes} minutes",
                f"spanning {minutes} minutes",
                f"over a {minutes}-minute period",
                f"for {minutes} minutes"
            ]
        elif minutes < 15:
            base_descriptions = [
                f"over a substantial {minutes}-minute period",
                f"spanning {minutes} minutes",
                f"over a {minutes}-minute period",
                f"for {minutes} minutes",
                f"the activity lasted quite long ({minutes} minutes)"
            ]
        else:
            base_descriptions = [
                f"over a substantial period of time ({minutes} minutes)",
                f"over a prolonged period of time ({minutes} minutes)",
                f"sustained for {minutes} minutes",
                f"extending across {minutes} minutes",
                f"lasting {minutes} minutes",
                f"the activity lasted quite long ({minutes} minutes)"
            ]

        # Select base description
        base_desc = self.random.choice(base_descriptions)

        # Add gap information if there were long gaps
        if has_long_gaps:
            gap_additions = [
                ", with significant pauses between sensor activations",
                ", including periods where a lot of time passed before the next sensor got activated",
                ", with notable gaps in sensor readings",
                ", including extended intervals between sensor events"
            ]
            # Sometimes add gap information (50% chance)
            if self.random.choice([True, False]):
                base_desc += self.random.choice(gap_additions)

        return base_desc

    def _generate_time_phrase(self, tod: str) -> str:
        """Generate simple time of day phrases."""
        time_phrases = {
            'morning': [
                'in the morning', 'during morning hours', 'in the early morning'
            ],
            'afternoon': [
                'in the afternoon', 'during the day', 'during afternoon hours'
            ],
            'evening': [
                'in the evening', 'during evening hours', 'in the late evening'
            ],
            'night': [
                'during the night', 'late at night', 'in the middle of the night',
                'during night hours'
            ]
        }

        tod_key = 'night' if 'night' in tod else tod.split()[0] if tod != 'unknown' else 'evening'
        return self.random.choice(time_phrases.get(tod_key, time_phrases['evening']))

    def _generate_month_description(self, timestamp: datetime) -> str:
        """Generate diverse month descriptions (Jan, January, second winter month, etc.)."""
        month_num = timestamp.month

        # Full month names
        full_names = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]

        # Abbreviated month names
        abbrev_names = [
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ]

        # Seasonal descriptions
        seasonal_descriptions = {
            1: ['the second winter month', 'deep winter', 'mid-winter'],
            2: ['the third winter month', 'late winter', 'the end of winter', 'winter\'s end'],
            3: ['early spring', 'the first spring month', 'spring\'s beginning', 'late winter/early spring'],
            4: ['mid-spring', 'the second spring month', 'spring season', 'full spring'],
            5: ['late spring', 'the third spring month', 'spring\'s peak', 'the final spring month'],
            6: ['early summer', 'the first summer month', 'summer\'s beginning', 'late spring/early summer'],
            7: ['mid-summer', 'the second summer month', 'peak summer', 'the height of summer'],
            8: ['late summer', 'the third summer month', 'summer\'s end', 'the final summer month'],
            9: ['early autumn', 'the first fall month', 'autumn\'s beginning', 'late summer/early fall'],
            10: ['mid-autumn', 'the second fall month', 'peak autumn', 'full fall season'],
            11: ['late autumn', 'the third fall month', 'autumn\'s end', 'late fall'],
            12: ['early winter', 'the first winter month', 'winter\'s beginning', 'the holiday month']
        }

        # Choose description style randomly
        styles = ['full', 'abbrev', 'seasonal']
        style = self.random.choice(styles)

        if style == 'full':
            return full_names[month_num - 1]
        elif style == 'abbrev':
            return abbrev_names[month_num - 1]
        else:  # seasonal
            return self.random.choice(seasonal_descriptions[month_num])

    def _generate_room_description(self, unique_rooms: List[str]) -> str:
        """Generate natural room transition description with 'back' terminology for revisited rooms."""
        if len(unique_rooms) == 1:
            return f"in {unique_rooms[0]}"
        elif len(unique_rooms) == 2:
            transition_words = ['movement from', 'transition from', 'activity from']
            return f"{self.random.choice(transition_words)} {unique_rooms[0]} to {unique_rooms[1]}"
        else:
            # Check for revisited rooms to add "back" terminology
            room_counts = {}
            for room in unique_rooms:
                room_counts[room] = room_counts.get(room, 0) + 1

            has_revisited = any(count > 1 for count in room_counts.values())

            if len(unique_rooms) == 3:
                # Check if last room is a return to first room
                if unique_rooms[0] == unique_rooms[2]:
                    back_phrases = [
                        f"movement from {unique_rooms[0]} to {unique_rooms[1]} and back to {unique_rooms[0]}",
                        f"activity from {unique_rooms[0]} to {unique_rooms[1]} then back to {unique_rooms[0]}",
                        f"transition from {unique_rooms[0]} to {unique_rooms[1]} and returning to {unique_rooms[0]}",
                        f"movement from {unique_rooms[0]} to {unique_rooms[1]} before coming back to {unique_rooms[0]}"
                    ]
                    return self.random.choice(back_phrases)
                elif has_revisited:
                    # Some room was revisited but not necessarily a simple back-and-forth
                    return f"movement from {unique_rooms[0]} to {unique_rooms[1]} then back to {unique_rooms[2]}"
                else:
                    return f"movement from {unique_rooms[0]} to {unique_rooms[1]} then to {unique_rooms[2]}"
            else:
                # More than 3 rooms - look for back patterns
                if has_revisited:
                    multi_phrases = [
                        f"complex movement across multiple rooms including {unique_rooms[0]}, {unique_rooms[1]}, and {unique_rooms[2]}, with some areas revisited",
                        f"movement across multiple areas starting from {unique_rooms[0]} through {unique_rooms[1]} to {unique_rooms[2]}, including returns to previous locations",
                        f"activity spanning {unique_rooms[0]}, {unique_rooms[1]}, {unique_rooms[2]} and other areas, with back-and-forth movements"
                    ]
                else:
                    multi_phrases = [
                        f"activating sensors in multiple rooms including {unique_rooms[0]}, {unique_rooms[1]} and {unique_rooms[2]}",
                        f"movement across multiple areas starting from {unique_rooms[0]} through {unique_rooms[1]} to {unique_rooms[2]}",
                        f"activity spanning {unique_rooms[0]}, {unique_rooms[1]}, {unique_rooms[2]} and other areas"
                    ]
                return self.random.choice(multi_phrases)

    def _generate_sensor_description(self, salient: List[str]) -> str:
        """Generate simple sensor activity descriptions."""
        if not salient:
            return self.random.choice(["motion detected", "activity sensed", "movement registered"])

        # Process salient sensors into simple descriptions
        descriptions = []
        for sensor in salient:
            if 'bed sensor' in sensor:
                bed_phrases = [
                    'activity near bed', 'movement in sleeping area',
                    'bed area motion', 'motion detected on bed'
                ]
                descriptions.append(self.random.choice(bed_phrases))
            elif 'toilet' in sensor:
                toilet_phrases = [
                    'movement near toilet', 'bathroom activity',
                    'restroom motion', 'activity in bathroom area'
                ]
                descriptions.append(self.random.choice(toilet_phrases))
            elif 'entry door' in sensor:
                door_phrases = [
                    'door activity', 'entrance movement',
                    'entry/exit motion', 'doorway activity'
                ]
                descriptions.append(self.random.choice(door_phrases))
            elif 'kitchen motion' in sensor:
                if 'fridge area' in sensor:
                    fridge_phrases = [
                        'activity near fridge', 'movement by refrigerator',
                        'fridge area activity'
                    ]
                    descriptions.append(self.random.choice(fridge_phrases))
                elif 'stove area' in sensor:
                    stove_phrases = [
                        'movement near stove', 'cooking area activity',
                        'activity by stove'
                    ]
                    descriptions.append(self.random.choice(stove_phrases))
                else:
                    kitchen_phrases = [
                        'kitchen area movement', 'cooking area motion',
                        'kitchen activity'
                    ]
                    descriptions.append(self.random.choice(kitchen_phrases))
            elif 'workspace motion' in sensor or 'desk area' in sensor or ('motion' in sensor and 'desk' in sensor):
                desk_phrases = [
                    'activity at desk', 'movement near desk',
                    'desk area activity', 'workspace motion'
                ]
                descriptions.append(self.random.choice(desk_phrases))
            elif 'motion' in sensor:
                # Extract room from sensor description
                if '(' in sensor:
                    area = sensor.split('(')[1].split(' area')[0]
                    descriptions.append(f'activity near {area}')
                else:
                    room = sensor.split(' motion')[0]
                    descriptions.append(f'movement in {room}')

        # Combine descriptions
        if len(descriptions) == 1:
            return descriptions[0]
        elif len(descriptions) == 2:
            return f"{descriptions[0]} and {descriptions[1]}"
        else:
            return f"{', '.join(descriptions[:-1])}, and {descriptions[-1]}"

    def get_caption_statistics(self, all_captions: List[Caption]) -> Dict[str, Any]:
        """Compute statistics about generated captions."""
        if not all_captions:
            return {}

        # Length statistics for Layer A
        layer_a_lengths = [len(caption.tokens) for caption in all_captions]

        # Separate enhanced and basic captions
        enhanced_captions = [c for c in all_captions if c.caption_type == 'enhanced']
        basic_captions = [c for c in all_captions if c.caption_type == 'basic']

        stats = {
            'total_captions': len(all_captions),
            'enhanced_captions': len(enhanced_captions),
            'basic_captions': len(basic_captions),
            'layer_a_stats': {
                'mean_tokens': np.mean(layer_a_lengths),
                'std_tokens': np.std(layer_a_lengths),
                'min_tokens': np.min(layer_a_lengths),
                'max_tokens': np.max(layer_a_lengths)
            },
            'sample_captions': [caption.text for caption in all_captions[:5]]
        }

        # Layer B statistics for enhanced captions
        if enhanced_captions:
            layer_b_lengths = [len(c.layer_b) for c in enhanced_captions if c.layer_b]
            if layer_b_lengths:
                stats['layer_b_stats'] = {
                    'mean_chars': np.mean(layer_b_lengths),
                    'std_chars': np.std(layer_b_lengths),
                    'min_chars': np.min(layer_b_lengths),
                    'max_chars': np.max(layer_b_lengths)
                }
                stats['sample_layer_b'] = [c.layer_b for c in enhanced_captions[:3] if c.layer_b]

        return stats

    def _generate_short_creative_captions(self, window: ProcessedWindow,
                                         window_features: WindowFeatures,
                                         sensor_details: Optional[Dict[str, str]] = None) -> List[Caption]:
        """Generate 2 short creative captions (under 5 tokens each)."""

        captions = []

        # Analyze the window to determine context
        unique_rooms = []
        room_sequence = window_features.room_sequence
        for room in room_sequence:
            # Convert to string if it's not already (handles NaN, floats, etc.)
            if not isinstance(room, str):
                room = str(room) if room is not None and (not isinstance(room, float) or not pd.isna(room)) else 'unknown'
            if not unique_rooms or room != unique_rooms[-1]:
                unique_rooms.append(room.replace('_', ' '))

        # Get time context
        first_event = window.events.iloc[0]
        tod = first_event.get('tod_bucket', 'unknown')
        is_night = 'night' in tod or 'after midnight' in tod
        is_day = 'morning' in tod or 'afternoon' in tod

        # Generate first short caption
        if len(unique_rooms) == 1:
            # Single room activity
            room = unique_rooms[0]
            if 'kitchen' in room.lower():
                if is_night:
                    options = ["kitchen activity night", "late kitchen motion", "nighttime kitchen use", "kitchen at night"]
                else:
                    options = ["kitchen activity day", "cooking area motion", "kitchen movement", "daytime kitchen use"]
            elif 'bedroom' in room.lower():
                if is_night:
                    options = ["bedroom night activity", "sleeping area motion", "nighttime bedroom", "bedroom at night"]
                else:
                    options = ["bedroom day activity", "bedroom movement", "daytime bedroom", "bedroom motion"]
            elif 'bathroom' in room.lower():
                options = ["bathroom activity", "restroom motion", "bathroom use", "toilet area"]
            elif 'living' in room.lower():
                options = ["living room activity", "lounge area motion", "living space", "main room"]
            else:
                if is_night:
                    options = [f"{room} night", f"nighttime {room}", f"{room} activity", f"night motion"]
                else:
                    options = [f"{room} activity", f"daytime {room}", f"{room} motion", f"day activity"]
        else:
            # Multiple rooms - transitions
            if len(unique_rooms) == 2:
                room1, room2 = unique_rooms[0], unique_rooms[1]
                # Focus on key room pairs
                if 'kitchen' in room1.lower() and 'living' in room2.lower():
                    if is_day:
                        options = ["kitchen living day", "cooking to lounge", "kitchen living motion", "day room transitions"]
                    else:
                        options = ["kitchen living night", "evening transitions", "night room movement", "kitchen living"]
                elif 'bedroom' in room1.lower() and 'bathroom' in room2.lower():
                    options = ["bedroom bathroom", "bed to toilet", "bathroom transition", "bedroom restroom"]
                else:
                    options = ["room transitions", "multiple rooms", "area movement", "room changes"]
            else:
                options = ["multiple room transitions", "complex movement", "room navigation", "area changes"]

        # Add sensor-specific context if available
        if sensor_details:
            # Look for specific sensors
            sensor_types = set()
            for _, event in window.events.iterrows():
                sensor_id = event.get('sensor', '')
                desc = sensor_details.get(sensor_id, '').lower()
                if 'fridge' in desc:
                    sensor_types.add('fridge')
                elif 'stove' in desc or 'oven' in desc:
                    sensor_types.add('oven')
                elif 'door' in desc:
                    sensor_types.add('door')
                elif 'bed' in desc:
                    sensor_types.add('bed')

            # Override options if specific sensors detected
            if 'fridge' in sensor_types and 'oven' in sensor_types:
                options = ["fridge oven sensors", "cooking appliances", "kitchen appliance use", "fridge and stove"]
            elif 'fridge' in sensor_types:
                options = ["fridge sensor activity", "refrigerator use", "fridge area", "fridge motion"]
            elif 'oven' in sensor_types:
                options = ["oven sensor activity", "stove area", "cooking appliance", "oven use"]

        # Select first caption
        caption1_text = self.random.choice(options)
        captions.append(Caption(
            text=caption1_text,
            window_id=window.metadata.window_id,
            caption_type='short_creative'
        ))

        # Generate second short caption with different style
        # More abstract/activity focused
        if len(unique_rooms) == 1:
            room = unique_rooms[0]
            if 'kitchen' in room.lower():
                options2 = ["food preparation", "culinary activity", "meal prep", "cooking motion"]
            elif 'bedroom' in room.lower():
                if is_night:
                    options2 = ["sleep preparation", "bedtime routine", "rest activity", "sleeping"]
                else:
                    options2 = ["bedroom tasks", "personal space", "private activity", "room activity"]
            elif 'bathroom' in room.lower():
                options2 = ["personal hygiene", "bathroom routine", "restroom use", "hygiene activity"]
            elif 'living' in room.lower():
                options2 = ["relaxation time", "leisure activity", "social space", "lounge time"]
            else:
                options2 = ["daily activity", "routine motion", "home activity", "personal task"]
        else:
            # Multiple rooms - focus on activity type
            if any('kitchen' in r.lower() for r in unique_rooms):
                options2 = ["meal related", "food activity", "kitchen routine", "culinary tasks"]
            elif any('bathroom' in r.lower() for r in unique_rooms):
                options2 = ["hygiene routine", "personal care", "bathroom routine", "daily hygiene"]
            else:
                options2 = ["daily routine", "home navigation", "household activity", "routine tasks"]

        caption2_text = self.random.choice(options2)
        captions.append(Caption(
            text=caption2_text,
            window_id=window.metadata.window_id,
            caption_type='short_creative'
        ))

        return captions