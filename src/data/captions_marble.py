"""Window2Text-style caption generation for MARBLE dataset."""

import json
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd


class MarbleCaptionGenerator:
    """Generate Window2Text-style captions for MARBLE windows."""

    def __init__(self, metadata_path='metadata/marble_metadata.json'):
        """Initialize with MARBLE metadata."""
        self.metadata = self._load_metadata(metadata_path)
        self.sensor_location = self.metadata.get('sensor_location', {})
        self.sensor_details = self.metadata.get('sensor_details', {})

        # Sensor type mappings for action generation
        self.sensor_type_map = {
            'R': 'magnetic',
            'E': 'smart_plug',
            'P': 'pressure_mat'
        }

    def _load_metadata(self, metadata_path: str) -> Dict:
        """Load MARBLE metadata from JSON file."""
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)

        return all_metadata.get('marble', {})

    def generate_caption(self, window: Dict, window_length_sec: float) -> str:
        """Generate a Window2Text-style caption for a window.

        Args:
            window: Window dictionary with states and metadata
            window_length_sec: Window length in seconds

        Returns:
            Caption string
        """
        parts = []

        # 1. Starting location
        starting_location = self._determine_starting_location(window)
        if starting_location:
            parts.append(f"The subject is in the {starting_location}.")

        # 2. Already-active states
        already_active_desc = self._describe_already_active_states(window)
        if already_active_desc:
            parts.append(already_active_desc)

        # 3. Actions inside window (inner + persistent states)
        actions_desc = self._describe_window_actions(window)
        if actions_desc:
            parts.append(actions_desc)

        return " ".join(parts)

    def _generate_temporal_context(self, window: Dict, window_length_sec: float) -> str:
        """Generate temporal context: window length and time of day."""
        window_start = pd.to_datetime(window['window_start_ms'], unit='ms')

        # Window length
        length_desc = f"This window lasts {int(window_length_sec)} seconds"

        # Time of day in hh:mm am/pm format
        time_str = window_start.strftime("%I:%M %p").lstrip('0')

        return f"{length_desc}, starting at approximately {time_str}."

    def _determine_starting_location(self, window: Dict) -> Optional[str]:
        """Determine starting location from already-active or persistent states."""
        already_active = window.get('already_active_states', [])
        persistent = window.get('persistent_states', [])

        # Check already-active states first
        for state in already_active:
            sensor_id = self._extract_sensor_id(state.get('sensor_or_bin', ''))
            location = self.sensor_location.get(sensor_id)
            if location:
                return location.replace('_', ' ')

        # Check persistent states
        for state in persistent:
            sensor_id = self._extract_sensor_id(state.get('sensor_or_bin', ''))
            location = self.sensor_location.get(sensor_id)
            if location:
                return location.replace('_', ' ')

        return None

    def _extract_sensor_id(self, sensor_or_bin: str) -> str:
        """Extract sensor ID from sensor_or_bin (handles bin suffixes)."""
        # Remove _binX suffix if present
        if '_bin' in sensor_or_bin:
            return sensor_or_bin.split('_bin')[0]
        return sensor_or_bin

    def _describe_already_active_states(self, window: Dict) -> str:
        """Describe already-active states."""
        already_active = window.get('already_active_states', [])
        if not already_active:
            return ""

        descriptions = []
        grouped_by_location = {}

        # Group by location
        for state in already_active:
            sensor_id = self._extract_sensor_id(state.get('sensor_or_bin', ''))
            location = self.sensor_location.get(sensor_id, 'unknown')

            if location not in grouped_by_location:
                grouped_by_location[location] = []
            grouped_by_location[location].append((sensor_id, state))

        # Generate descriptions
        for location, states in grouped_by_location.items():
            location_desc = location.replace('_', ' ')
            sensor_descriptions = []

            for sensor_id, state in states:
                action = self._state_to_action(sensor_id, state, is_already_active=True)
                if action:
                    sensor_descriptions.append(action)

            if sensor_descriptions:
                combined = ", ".join(sensor_descriptions)
                descriptions.append(f"The subject is already {combined} in the {location_desc}.")

        return " ".join(descriptions)

    def _describe_window_actions(self, window: Dict) -> str:
        """Describe actions inside the window (inner + persistent states)."""
        inner_states = window.get('inner_states', [])
        persistent_states = window.get('persistent_states', [])

        # Combine and sort by start time
        all_states = []
        for state in inner_states:
            all_states.append(('inner', state))
        for state in persistent_states:
            all_states.append(('persistent', state))

        if not all_states:
            return ""

        # Sort by start time
        all_states.sort(key=lambda x: x[1].get('t_start', 0))

        # Group by room transitions
        descriptions = []
        current_location = None
        location_states = []

        for state_type, state in all_states:
            sensor_id = self._extract_sensor_id(state.get('sensor_or_bin', ''))
            location = self.sensor_location.get(sensor_id, 'unknown')
            location = location.replace('_', ' ')

            # Check for room transition
            if current_location is None:
                current_location = location
                location_states = [(state_type, state)]
            elif location != current_location:
                # Process previous location
                if location_states:
                    location_desc = self._describe_location_actions(
                        current_location, location_states
                    )
                    if location_desc:
                        descriptions.append(location_desc)

                # Announce transition
                descriptions.append(f"Then, they move to the {location}.")
                current_location = location
                location_states = [(state_type, state)]
            else:
                location_states.append((state_type, state))

        # Process last location
        if location_states:
            location_desc = self._describe_location_actions(
                current_location, location_states
            )
            if location_desc:
                descriptions.append(location_desc)

        return " ".join(descriptions)

    def _describe_location_actions(self, location: str, states: List[Tuple[str, Dict]]) -> str:
        """Describe actions in a specific location."""
        actions = []

        # Sort states by start time for chronological order
        states_sorted = sorted(states, key=lambda x: x[1].get('t_start', 0))

        for state_type, state in states_sorted:
            sensor_id = self._extract_sensor_id(state.get('sensor_or_bin', ''))
            action = self._state_to_action(sensor_id, state, is_already_active=False)

            if action:
                # For persistent states, add "stays near" language
                if state_type == 'persistent':
                    duration_sec = (state.get('t_end', 0) - state.get('t_start', 0)) / 1000.0
                    if duration_sec > 0:
                        duration_desc = self._format_duration(duration_sec)
                        # For persistent, use "stays near" language
                        sensor_detail = self.sensor_details.get(sensor_id, '')
                        if 'chair' in sensor_detail.lower() or 'couch' in sensor_detail.lower():
                            action = f"stays seated for {duration_desc}"
                        else:
                            action = f"stays near the {location} for {duration_desc}"

                actions.append(action)

        if not actions:
            return ""

        # Combine actions with temporal connectors
        if len(actions) == 1:
            return f"The subject {actions[0]}."
        else:
            # Use temporal connectors
            combined = f"The subject {actions[0]}"
            for i, action in enumerate(actions[1:], 1):
                if i == len(actions) - 1:
                    combined += f", and then {action}."
                else:
                    combined += f", then {action}"
            return combined

    def _state_to_action(self, sensor_id: str, state: Dict, is_already_active: bool) -> Optional[str]:
        """Convert a state to a natural language action."""
        if not sensor_id:
            return None

        sensor_type = sensor_id[0] if len(sensor_id) > 0 else 'U'
        sensor_detail = self.sensor_details.get(sensor_id, '')

        # Extract action from sensor details
        action = self._extract_action_from_detail(sensor_id, sensor_detail, sensor_type)

        if is_already_active:
            return f"{action}"
        else:
            # For inner states, this is an ON->OFF interval
            duration_sec = (state.get('t_end', 0) - state.get('t_start', 0)) / 1000.0

            if duration_sec > 0:
                duration_desc = self._format_duration(duration_sec)
                # Format as "turned on X and turned it off after Y seconds"
                if 'turns on' in action or 'opens' in action:
                    # Extract the object
                    if 'turns on the' in action:
                        obj = action.replace('turns on the', '').strip()
                        return f"turned on the {obj} and turned it off after {duration_desc}"
                    elif 'opens the' in action:
                        obj = action.replace('opens the', '').strip()
                        return f"opened the {obj} and closed it after {duration_desc}"
                    elif 'opens' in action:
                        return f"{action} and closed it after {duration_desc}"
                return f"{action} for {duration_desc}"
            else:
                return action

    def _extract_action_from_detail(self, sensor_id: str, sensor_detail: str, sensor_type: str) -> str:
        """Extract action description from sensor details."""
        detail_lower = sensor_detail.lower()

        # Magnetic sensors (R)
        if sensor_type == 'R':
            if 'fridge' in detail_lower:
                return "opens the fridge"
            elif 'drawer' in detail_lower:
                if 'cutlery' in detail_lower:
                    return "opens the cutlery drawer"
                elif 'pots' in detail_lower:
                    return "opens the pots drawer"
            elif 'cabinet' in detail_lower or 'medicines' in detail_lower:
                return "opens the medicines cabinet"
            elif 'pantry' in detail_lower:
                return "opens the pantry"
            else:
                return "opens a drawer or cabinet"

        # Smart Plug sensors (E)
        elif sensor_type == 'E':
            if 'stove' in detail_lower:
                return "turns on the stove"
            elif 'television' in detail_lower or 'tv' in detail_lower:
                return "turns on the television"
            else:
                return "turns on an appliance"

        # Pressure Mat sensors (P)
        elif sensor_type == 'P':
            if 'chair' in detail_lower:
                if 'dining' in detail_lower:
                    return "sits on a dining room chair"
                elif 'office' in detail_lower:
                    return "sits on the office chair"
                else:
                    return "sits on a chair"
            elif 'couch' in detail_lower:
                return "sits on the couch"
            else:
                return "sits down"

        return "activates a sensor"

    def _format_duration(self, seconds: float) -> str:
        """Format duration in qualitative language."""
        if seconds < 1:
            return "less than a second"
        elif seconds < 2:
            return "about a second"
        elif seconds < 5:
            return f"about {int(seconds)} seconds"
        elif seconds < 10:
            return f"about {int(seconds)} seconds"
        elif seconds < 60:
            return f"about {int(seconds)} seconds"
        else:
            minutes = int(seconds / 60)
            if minutes == 1:
                return "about a minute"
            else:
                return f"about {minutes} minutes"

