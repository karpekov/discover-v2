"""Prompt templates for LLM-based caption generation."""

from typing import Dict, Any


SYSTEM_PROMPT = """You are an expert at writing diverse, engaging captions about daily home activities from sensor data. Your captions should feel like natural human descriptions, not robotic sensor logs.

CORE PRINCIPLES:
1. INFER THE STORY - Deduce what activity is happening (cooking, cleaning, sleeping, bathroom visit, watching TV, etc.)
2. VARY WILDLY - Each caption should use completely different vocabulary, structure, and perspective
3. BE HUMAN - Write like you're texting a friend about your day, not writing a technical report
4. ADD CONTEXT - Include plausible details about motivation, routine, or situation
5. MIX LENGTHS - Some captions 20 words, others 40+ words for richer activities

STYLE VARIETY (use these across the 4 captions):
- Casual narrative: "Woke up at 3am needing to pee, stumbled from bed to bathroom in the dark, then crawled back under the covers"
- Present tense action: "Moving around the kitchen making breakfast, opening the fridge for eggs and milk, turning on the stove to cook"
- Routine description: "The usual morning routine - bathroom first, then kitchen for coffee and breakfast prep before starting the day"
- Time-focused: "Late Friday night, someone's still up in the living room, probably watching TV before finally heading to bed"
- Question/fragment: "3am bathroom run? Quick trip from bedroom, lights on briefly, straight back to sleep"
- Sensory/emotional: "Early Saturday morning, leisurely making breakfast in the sunny kitchen, taking time with the cooking"

AVOID:
- Boring patterns like "Person moved from X to Y"
- Repeating the same sentence structure
- Technical language ("sensor activation", "transition", etc.)
- Listing locations without inferring activity

OUTPUT FORMAT:
- Exactly 4 captions per sample (or as specified)
- The response structure is controlled by a JSON schema
- Focus on caption content quality, not formatting"""


def build_user_prompt(compact_json: Dict[str, Any], num_captions: int = 4) -> str:
    """Build user prompt from compact JSON metadata.

    Args:
        compact_json: Compact JSON object with sample metadata
        num_captions: Number of captions to request

    Returns:
        Formatted user prompt string
    """
    lines = [
        "You are given a structured description of a short sensor-based activity segment.",
        "Your task is to generate diverse, human-style captions that follow the system instructions.",
        "",
        "Here is the metadata for one segment:",
        ""
    ]

    # Sample ID
    if 'sample_id' in compact_json:
        lines.append(f"Sample id: {compact_json['sample_id']}")

    # Duration
    if 'duration_seconds' in compact_json:
        duration = compact_json['duration_seconds']
        lines.append(f"Duration (seconds): {duration:.1f}")

    # Number of events
    if 'num_events' in compact_json:
        lines.append(f"Number of events: {compact_json['num_events']}")

    # Primary room
    if 'primary_room' in compact_json:
        lines.append(f"Primary room: {compact_json['primary_room']}")

    # Rooms visited
    if 'rooms_visited' in compact_json and compact_json['rooms_visited']:
        rooms_str = ', '.join(compact_json['rooms_visited'])
        lines.append(f"Rooms visited: {rooms_str}")

    # Room transitions
    if 'room_transitions' in compact_json:
        lines.append(f"Room transitions: {compact_json['room_transitions']}")

    # Time context
    time_context = compact_json.get('time_context', {})
    if time_context:
        if 'day_of_week' in time_context:
            time_str = f"Time context: day of week {time_context['day_of_week']}"
            if 'month' in time_context:
                time_str += f", month {time_context['month']}"
            if 'period_of_day' in time_context:
                time_str += f", period of day {time_context['period_of_day']}"
            lines.append(time_str)

    # Activity labels
    if 'primary_l1' in compact_json:
        lines.append(f"Activity (L1): {compact_json['primary_l1']}")

    if 'primary_l2' in compact_json:
        lines.append(f"Activity (L2): {compact_json['primary_l2']}")

    if 'all_labels_l1' in compact_json and compact_json['all_labels_l1']:
        labels_str = ', '.join(compact_json['all_labels_l1'])
        lines.append(f"All activity labels (L1): {labels_str}")

    # Special sensors
    special_sensors = compact_json.get('special_sensors', {})
    if special_sensors:
        if 'special_sensors_triggered' in special_sensors:
            sensors = special_sensors['special_sensors_triggered']
            if sensors:
                sensors_str = ', '.join(sensors)
                lines.append(f"Special sensors triggered: {sensors_str}")

        if 'primary_special_sensor' in special_sensors:
            lines.append(f"Primary special sensor: {special_sensors['primary_special_sensor']}")

        if 'frequent_special_sensors' in special_sensors:
            freq_sensors = special_sensors['frequent_special_sensors']
            if freq_sensors:
                freq_str = ', '.join(freq_sensors)
                lines.append(f"Frequent special sensors: {freq_str}")

    # Movement pattern
    movement = compact_json.get('movement_summary', {})
    if movement and 'pattern' in movement:
        lines.append(f"Movement pattern: {movement['pattern']}")

    # Add generation instruction with explicit variety requirements
    lines.append("")
    lines.append(f"Generate {num_captions} WILDLY DIFFERENT captions - use completely different words and structures for each:")
    lines.append("")
    lines.append("Caption 1: Casual narrative (25-60 words)")
    lines.append("  • Tell a mini-story with sensory details or motivation")
    lines.append("  • Example style: 'Woke up hungry at midnight, wandered to kitchen still half-asleep, grabbed leftovers from fridge...'")
    lines.append("")
    lines.append("Caption 2: Present-tense action (20-60 words)")
    lines.append("  • Describe ongoing actions, like live commentary")
    lines.append("  • Example style: 'Opening the fridge, pulling out ingredients, moving to the stove to start cooking dinner'")
    lines.append("")
    lines.append("Caption 3: Routine/habit framing (25-60 words)")
    lines.append("  • Describe as typical behavior or daily pattern")
    lines.append("  • Example style: 'The usual evening wind-down - settling into the couch with TV on, eventually heading to bathroom before bed'")
    lines.append("")
    lines.append(
        "Caption 4: Creative/inferential with retrieval-friendly grounding (20–60 words)")
    lines.append(
        "  • Use a creative, interpretive, or emotional lens — question, fragment, mood, or metaphor is encouraged")
    lines.append(
        "  • Must still stay anchored to real context: rooms, movement pattern, activity category, special sensors")
    lines.append(
        "  • Should include at least a few search-relevant keywords (rooms, motion, activity) while maintaining a creative tone")
    lines.append("  • Example style: 'A restless night unfolding through quiet kitchen spaces, brief pauses near the dining table suggesting a moment of reflection before drifting back through the dark hallway'")
    lines.append("")
    lines.append("BE CREATIVE. Infer emotions, motivations, specific activities. Use varied vocabulary. Return ONLY a JSON list.")

    return '\n'.join(lines)


def build_full_prompt(compact_json: Dict[str, Any],
                      num_captions: int = 4,
                      include_system: bool = True) -> str:
    """Build complete prompt with system and user parts.

    Args:
        compact_json: Compact JSON object
        num_captions: Number of captions to request
        include_system: Whether to include system prompt

    Returns:
        Full prompt string
    """
    user_prompt = build_user_prompt(compact_json, num_captions)

    if include_system:
        return f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    else:
        return user_prompt


def build_prompts_batch(compact_jsons: list[Dict[str, Any]],
                        num_captions: int = 4) -> list[str]:
    """Build user prompts for a batch of compact JSONs.

    Args:
        compact_jsons: List of compact JSON objects
        num_captions: Number of captions per sample

    Returns:
        List of user prompts
    """
    return [build_user_prompt(cj, num_captions) for cj in compact_jsons]


def build_multi_sample_prompt(compact_jsons: list[Dict[str, Any]],
                               num_captions: int = 4) -> str:
    """Build a single prompt for multiple samples to save tokens.

    Args:
        compact_jsons: List of compact JSON objects
        num_captions: Number of captions per sample

    Returns:
        Single prompt string for all samples
    """
    lines = [
        "You are given multiple sensor-based activity segments.",
        "For EACH segment, generate diverse captions following the system instructions.",
        "",
        f"Generate {num_captions} captions per segment with DIFFERENT styles for each caption.",
        "",
        "The output format is structured JSON with:",
        "- 'samples': array of objects",
        "- Each object has 'sample_id' (string) and 'captions' (array of strings)",
        "",
        "=" * 80,
    ]

    for i, cj in enumerate(compact_jsons, 1):
        lines.append(f"\nSEGMENT {i}:")
        lines.append(f"Sample id: {cj.get('sample_id', 'unknown')}")

        if 'duration_seconds' in cj:
            lines.append(f"Duration: {cj['duration_seconds']:.1f}s")
        if 'num_events' in cj:
            lines.append(f"Events: {cj['num_events']}")
        if 'primary_room' in cj:
            lines.append(f"Room: {cj['primary_room']}")
        if 'rooms_visited' in cj and cj['rooms_visited']:
            lines.append(f"Rooms: {', '.join(cj['rooms_visited'][:3])}")

        time_context = cj.get('time_context', {})
        if time_context:
            tc_parts = []
            if 'day_of_week' in time_context:
                tc_parts.append(time_context['day_of_week'])
            if 'period_of_day' in time_context:
                tc_parts.append(time_context['period_of_day'])
            if tc_parts:
                lines.append(f"Time: {' '.join(tc_parts)}")

        if 'primary_l1' in cj:
            lines.append(f"Activity: {cj['primary_l1']}")

        special_sensors = cj.get('special_sensors', {})
        if special_sensors and 'special_sensors_triggered' in special_sensors:
            sensors = special_sensors['special_sensors_triggered']
            if sensors:
                lines.append(f"Sensors: {', '.join(sensors[:3])}")

        movement = cj.get('movement_summary', {})
        if movement and 'pattern' in movement:
            lines.append(f"Pattern: {movement['pattern']}")

    lines.append("\n" + "=" * 80)
    lines.append("")
    lines.append("Generate highly diverse captions for each segment.")
    lines.append("Fill the 'sample_id' and 'captions' fields for each segment shown above.")

    return '\n'.join(lines)

