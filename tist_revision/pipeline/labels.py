"""
Evaluation label set: raw CASAS activities merged into the classes used by DeepCASAS
(Liciotti et al.) and TDOST (`translateDict` in ~/code/har/TDOST/TDOST/utils.py), e.g. Milan
Read + Watch_TV -> Relax, Cairo Breakfast/Lunch/Dinner -> Eat, Cairo R1/R2 wake -> Other.

Keys are normalised raw (l1) names: lower-case, spaces -> underscores.
"""

TDOST_CLASSES = ('Other', 'Relax', 'Cook', 'Leave_Home', 'Enter_Home', 'Sleep', 'Eat', 'Work',
                 'Bed_to_Toilet', 'Bathing', 'Take_Medicine', 'Personal_Hygiene')

_RAW_TO_TDOST = {
    # Milan
    '': 'Other', 'other': 'Other', 'master_bedroom_activity': 'Other', 'meditate': 'Other',
    'read': 'Relax', 'watch_tv': 'Relax',
    'kitchen_activity': 'Cook', 'dining_rm_activity': 'Eat',
    'desk_activity': 'Work', 'chores': 'Work',
    'master_bathroom': 'Bathing', 'guest_bathroom': 'Bathing',
    'morning_meds': 'Take_Medicine', 'eve_meds': 'Take_Medicine', 'evening_meds': 'Take_Medicine',
    'sleep': 'Sleep', 'bed_to_toilet': 'Bed_to_Toilet', 'leave_home': 'Leave_Home',
    # Aruba
    'meal_preparation': 'Cook', 'eating': 'Eat', 'relax': 'Relax', 'sleeping': 'Sleep',
    'work': 'Work', 'wash_dishes': 'Work', 'housekeeping': 'Work',
    'respirate': 'Other', 'resperate': 'Other', 'enter_home': 'Enter_Home',
    # Cairo
    'breakfast': 'Eat', 'lunch': 'Eat', 'dinner': 'Eat',
    'r1_sleep': 'Sleep', 'r2_sleep': 'Sleep',
    'r1_wake': 'Other', 'r2_wake': 'Other', 'night_wandering': 'Other',
    'r1_work_in_office': 'Work', 'laundry': 'Work',
    'r2_take_medicine': 'Take_Medicine', 'r2_take_meds': 'Take_Medicine',
    # Kyoto7 / twor.2009 (TDOST keeps Personal_Hygiene as its own class); with and without the resident id
    'clean': 'Work', 'study': 'Other', 'wash_bathtub': 'Other',
    'personal_hygiene': 'Personal_Hygiene', 'r1_personal_hygiene': 'Personal_Hygiene',
    'r2_personal_hygiene': 'Personal_Hygiene',
    'r1_bed_to_toilet': 'Bed_to_Toilet', 'r2_bed_to_toilet': 'Bed_to_Toilet',
    'r1_work': 'Work', 'r2_work': 'Work',
    # Continuous data only (absent from pre-segmented windows)
    'no_activity': 'No_Activity',
}


def tdost_label(raw: str) -> str:
    """Map a raw CASAS activity name (any city) to its TDOST / DeepCASAS class."""
    if raw and '+' in raw:  # several activities at once (Kyoto7 household-state labels): map each part
        return '+'.join(sorted(tdost_label(part) for part in raw.split('+')))
    key = (raw or '').strip().lower().replace(' ', '_')
    if key not in _RAW_TO_TDOST:
        raise KeyError(f'No TDOST mapping for activity {raw!r}')
    return _RAW_TO_TDOST[key]
