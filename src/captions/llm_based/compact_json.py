"""Convert raw samples to compact JSON representation for LLM caption generation."""

from typing import Dict, Any, List
from datetime import datetime
from collections import Counter


def to_compact_caption_json(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw sample to a compact JSON object for caption generation.

    Args:
        sample: Raw sample dictionary with:
            - sample_id: Unique identifier
            - sensor_sequence: List of sensor events
            - metadata: Sample metadata

    Returns:
        Compact JSON object with fields optimized for LLM caption generation
    """
    sample_id = sample['sample_id']
    sensor_sequence = sample['sensor_sequence']
    metadata = sample['metadata']

    # Extract duration (prefer actual_duration_seconds if available)
    duration_seconds = metadata.get('actual_duration_seconds', metadata.get('duration_seconds', 0))

    # Extract basic info
    num_events = metadata.get('num_events', len(sensor_sequence))
    primary_room = metadata.get('primary_room', 'unknown')
    rooms_visited = metadata.get('rooms_visited', [])
    room_transitions = metadata.get('room_transitions', 0)

    # Time context
    start_time = metadata.get('start_time', '')
    end_time = metadata.get('end_time', '')

    time_context = _extract_time_context(start_time, end_time)

    # Special sensors
    special_sensors_data = metadata.get('special_sensors', {})
    special_sensors = _extract_special_sensors(special_sensors_data)

    # Movement summary
    movement_summary = _extract_movement_summary(rooms_visited, room_transitions, primary_room)

    # Activity labels (if available)
    ground_truth = metadata.get('ground_truth_labels', {})
    primary_l1 = ground_truth.get('primary_l1', None)
    primary_l2 = ground_truth.get('primary_l2', None)
    all_labels_l1 = ground_truth.get('all_labels_l1', [])

    # Build compact object
    compact = {
        'sample_id': sample_id,
        'duration_seconds': duration_seconds,
        'num_events': num_events,
        'primary_room': primary_room,
        'rooms_visited': rooms_visited,
        'room_transitions': room_transitions,
        'time_context': time_context,
        'special_sensors': special_sensors,
        'movement_summary': movement_summary
    }

    # Add activity labels if available
    if primary_l1:
        compact['primary_l1'] = primary_l1
    if primary_l2:
        compact['primary_l2'] = primary_l2
    if all_labels_l1:
        compact['all_labels_l1'] = all_labels_l1

    return compact


def _extract_time_context(start_time: str, end_time: str) -> Dict[str, Any]:
    """Extract time context from start and end times.

    Args:
        start_time: ISO timestamp string
        end_time: ISO timestamp string

    Returns:
        Dictionary with time context info
    """
    if not start_time:
        return {}

    try:
        # Parse start time
        dt = datetime.fromisoformat(start_time.replace(' ', 'T'))

        # Day of week
        day_of_week = dt.strftime('%A')

        # Month
        month = dt.strftime('%B')

        # Period of day
        hour = dt.hour
        if 5 <= hour < 12:
            period_of_day = 'morning'
        elif 12 <= hour < 17:
            period_of_day = 'afternoon'
        elif 17 <= hour < 21:
            period_of_day = 'evening'
        else:
            period_of_day = 'night'

        return {
            'start_time': start_time,
            'end_time': end_time,
            'day_of_week': day_of_week,
            'month': month,
            'period_of_day': period_of_day
        }
    except Exception as e:
        # If parsing fails, return minimal info
        return {
            'start_time': start_time,
            'end_time': end_time
        }


def _extract_special_sensors(special_sensors_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and process special sensor information.

    Args:
        special_sensors_data: Special sensors dict from metadata

    Returns:
        Processed special sensors dict
    """
    if not special_sensors_data:
        return {}

    special_sensors_triggered = special_sensors_data.get('special_sensors_triggered', [])
    primary_special_sensor = special_sensors_data.get('primary_special_sensor', None)
    special_sensor_counts = special_sensors_data.get('special_sensor_counts', {})

    # Compute frequent special sensors (count >= 3 or top-k)
    frequent_special_sensors = []
    if special_sensor_counts:
        # Get sensors with count >= 3
        frequent = [sensor for sensor, count in special_sensor_counts.items() if count >= 3]
        if not frequent:
            # If none with count >= 3, get top 3
            sorted_sensors = sorted(special_sensor_counts.items(), key=lambda x: x[1], reverse=True)
            frequent = [sensor for sensor, _ in sorted_sensors[:3]]
        frequent_special_sensors = frequent

    result = {
        'special_sensors_triggered': special_sensors_triggered,
        'special_sensor_counts': special_sensor_counts
    }

    if primary_special_sensor:
        result['primary_special_sensor'] = primary_special_sensor

    if frequent_special_sensors:
        result['frequent_special_sensors'] = frequent_special_sensors

    return result


def _extract_movement_summary(rooms_visited: List[str],
                               room_transitions: int,
                               primary_room: str) -> Dict[str, Any]:
    """Extract a simple movement pattern summary.

    Args:
        rooms_visited: List of rooms in order
        room_transitions: Number of transitions
        primary_room: Primary room

    Returns:
        Dictionary with movement pattern info
    """
    if not rooms_visited:
        return {'pattern': 'unknown'}

    # Single room stay
    if len(rooms_visited) == 1:
        return {
            'pattern': 'single_room_stay',
            'room': rooms_visited[0]
        }

    # Multiple rooms - create sequence string
    # Deduplicate consecutive rooms for pattern
    deduplicated = []
    for room in rooms_visited:
        if not deduplicated or deduplicated[-1] != room:
            deduplicated.append(room)

    if len(deduplicated) <= 4:
        # Short sequence - show full pattern
        pattern = '→'.join(deduplicated)
    else:
        # Long sequence - show abbreviated
        pattern = f"{deduplicated[0]}→{deduplicated[1]}→...→{deduplicated[-1]}"

    return {
        'pattern': pattern,
        'num_rooms': len(rooms_visited),
        'num_unique_rooms': len(set(rooms_visited)),
        'num_transitions': room_transitions
    }


def convert_samples_to_compact_jsonl(input_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a list of raw samples to compact JSON format.

    Args:
        input_samples: List of raw sample dictionaries

    Returns:
        List of compact JSON objects
    """
    compact_samples = []
    for sample in input_samples:
        try:
            compact = to_compact_caption_json(sample)
            compact_samples.append(compact)
        except Exception as e:
            print(f"Warning: Failed to convert sample {sample.get('sample_id', 'unknown')}: {e}")
            continue

    return compact_samples

