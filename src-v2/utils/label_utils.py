"""
Utility functions for label processing and conversion in CASAS activity recognition.
"""

from typing import List, Union


def convert_labels_to_text(labels: List[str], single_description: bool = False) -> Union[List[List[str]], List[str]]:
    """Convert CASAS activity labels to natural language descriptions.

    Args:
        labels: List of CASAS activity labels
        single_description: If True, return single description per label. If False, return multiple descriptions per label.

    Returns:
        List of descriptions (single or multiple per label)
    """

    # Mapping for single descriptions (used by alignment analysis)
    single_label_to_text = {
        'Kitchen_Activity': 'cooking and kitchen activities',
        'Sleep': 'sleeping in bed in master bedroom, usually at night time',
        'Read': 'reading a book or newspaper in a static position',
        'Watch_TV': 'watching television',
        'Master_Bedroom_Activity': 'activities and motion in the master bedroom',
        'Master_Bathroom': 'using the master bathroom, shower, or bathtub',
        'Guest_Bathroom': 'using the guest bathroom, shower, or bathtub',
        'Dining_Rm_Activity': 'activity and motion in dining room',
        'Desk_Activity': 'working at desk in workspace',
        'Leave_Home': 'leaving or entering the house through the door',
        'Chores': 'doing household chores',
        'Meditate': 'meditation',
        'no_activity': 'no specific activity or idle time',
        'Bed_to_Toilet': 'going from bed to toilet at night',
        'Morning_Meds': 'taking morning medication',
        'Eve_Meds': 'taking evening medication',
        # L2 labels
        'Cook': 'cooking and preparing food',
        'No_Activity': 'no specific activity or idle time',
        'Work': 'working at desk or workspace',
        'Eat': 'eating meals',
        'Relax': 'relaxing, watching TV or reading',
        'Bathing': 'bathing or using bathroom',
        'Other': 'other miscellaneous activities',
        'Take_medicine': 'taking medication',
        'Bed_to_toilet': 'going from bed to toilet at night'
    }

    # Mapping from CASAS labels to multiple natural language descriptions
    label_to_text = {
        # L1 labels - Primary activities
        'Kitchen_Activity': [
            'Kitchen activity with motion by refrigerator, cooking area activity, and brief movements near stove or within the kitchen.',
        ],
        'Sleep': [
            'Prolonged or intermittent bed-area motion in the master bedroom, often at night, reflecting sleep and related nighttime activity.',
        ],
        'Read': [
            'Reading activity with presence in living room near armchair ranging from brief moments to extended periods.',
        ],
        'Watch_TV': [
            'Watch TV activity with time spent in tv room, ranging from brief moments to extended periods',
        ],
        'Master_Bedroom_Activity': [
            'General master bedroom bedroom activity during day or evening.',
        ],
        'Master_Bathroom': [
            'Activity or motion in the master bathroom, can be related to sleep or hygiene.',
        ],
        'Guest_Bathroom': [
            'Guest bathroom activity with short transitions through nearby common areas.',
        ],
        'Dining_Rm_Activity': [
            'Dining room activity with motion across living room and adjacent areas, back-and-forth transitions, and related movements in surrounding areas.',
        ],
        'Desk_Activity': [
            'Desk activity with brief to moderate durations in workspace, centered on motion detection and presence near the desk area.',
        ],
        'Leave_Home': [
            'Leave home activity with short bursts of entry and exit and doorway motion, as well as longer periods in house entrance and nearby areas',
        ],
        'Chores': [
            'Chores activity with both short and occasional longer periods in master bathroom and master bedroom, often showing restroom motion, activity near sink, bed area motion, and movement in master,',
        ],
        'Meditate': [
            'Meditation activity with short and long periods in guest bedroom showing area motion',
        ],
        'no_activity': [
            'No specific activity or idle time.',
            'Minimal sensor activation across the home.',
            'Quiet period with little to no movement.'
        ],
        'Bed_to_toilet': [
            'Leaving bed during night and moving to bathroom, including short transition from master bedroom bed to toilet',
        ],
        'Morning_Meds': [
            'Medication activity with very short spans in kitchen, typically under a minute, showing movement by medicine cabinet and doorway in the morning time',
        ],
        'Eve_Meds': [
            'Medication activity with very short spans in kitchen, typically under a minute, showing movement by medicine cabinet and doorway in the evening and night time',
        ],

        # L2 labels - Secondary activities
        'Cook': [
            'Kitchen activity with motion by refrigerator, cooking area activity, and brief movements near stove or within the kitchen.',
        ],
        'No_Activity': [
            'No specific activity or idle time.',
            'Minimal sensor activation across the home.',
            'Quiet period with little to no movement.',
            'Inactive state with no significant motion.'
        ],
        'Work': [
            'Desk activity with brief to moderate durations in workspace, centered on motion detection and presence near the desk area.',
        ],
        'Eat': [
            'Activity spanning short and long periods in dining room, including movement around the dining room table.',
        ],
        'Relax': [
            'Reading or watch TV activity with time spent in living room near armchair or in tv room, ranging from brief moments to extended periods.',
        ],
        'Bathing': [
            'Bathroom activity in guest or master bathroom, with short transitions through nearby areas and motions related to hygiene.',
        ],
        'Other': [
            'Activity often happening in guest bedroom or master bedroom only.',
        ],
        'Take_medicine': [
            'Medication activity with very short spans in kitchen, typically under a minute, showing movement by medicine cabinet and doorway in the evening and night time',
        ],
        'Bed_to_toilet': [
            'Leaving bed during night and moving to bathroom, including short transition from master bedroom bed to toilet',
        ]
    }

    descriptions = []

    if single_description:
        # Return single description per label
        for label in labels:
            if label in single_label_to_text:
                descriptions.append(single_label_to_text[label])
            else:
                # Fallback: convert label to readable text
                readable = label.replace('_', ' ').lower()
                descriptions.append(f"{readable} activity")
    else:
        # Return multiple descriptions per label
        for label in labels:
            if label in label_to_text:
                descriptions.append(label_to_text[label])
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
