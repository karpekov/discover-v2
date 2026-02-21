"""
Utility functions for data sampling.

Includes column standardization, window creation, metadata computation, and more.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


# ============================================================================
# Column Standardization
# ============================================================================

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names from casas_end_to_end_preprocess output.

    Maps the actual column names to expected names used throughout the sampling pipeline:
    - 'sensor' → 'sensor_id'
    - 'state' → 'event_type'
    - 'datetime' → 'timestamp' (keep both for compatibility)
    - 'sensor_location' → 'room'
    - 'first_activity' → 'activity_l1'
    - 'first_activity_l2' → 'activity_l2'
    - 'sensor_data_type' → 'sensor_type'

    Args:
        df: DataFrame from casas_end_to_end_preprocess

    Returns:
        DataFrame with standardized column names (original columns preserved)
    """
    df = df.copy()

    # Column mapping: old_name -> new_name
    column_mapping = {
        'sensor': 'sensor_id',
        'state': 'event_type',
        'sensor_location': 'room',
        'first_activity': 'activity_l1',
        'first_activity_l2': 'activity_l2',
        'sensor_data_type': 'sensor_type',
    }

    # Create new columns (don't drop old ones for backward compatibility)
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    # Ensure 'timestamp' column exists (alias for 'datetime')
    if 'datetime' in df.columns and 'timestamp' not in df.columns:
        df['timestamp'] = df['datetime']

    # Ensure datetime is in the right format
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

    return df


def get_column_name(df: pd.DataFrame, *alternatives: str) -> str:
    """
    Get the first available column name from a list of alternatives.

    Useful for backward compatibility when column names might vary.

    Args:
        df: DataFrame to check
        *alternatives: Column name alternatives in order of preference

    Returns:
        First column name that exists in the dataframe

    Raises:
        ValueError: If none of the alternatives exist

    Example:
        >>> time_col = get_column_name(df, 'timestamp', 'datetime', 'time')
    """
    for col in alternatives:
        if col in df.columns:
            return col

    raise ValueError(f"None of the column alternatives found: {alternatives}")


def get_casas_column_info(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get information about which column names are being used.

    Useful for debugging column name issues.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary mapping standard names to actual column names
    """
    info = {}

    column_checks = {
        'sensor_id': ['sensor_id', 'sensor'],
        'event_type': ['event_type', 'state'],
        'timestamp': ['timestamp', 'datetime'],
        'room': ['room', 'room_id', 'sensor_location'],
        'activity_l1': ['activity_l1', 'first_activity'],
        'activity_l2': ['activity_l2', 'first_activity_l2'],
        'sensor_type': ['sensor_type', 'sensor_data_type'],
    }

    for standard_name, alternatives in column_checks.items():
        found = None
        for alt in alternatives:
            if alt in df.columns:
                found = alt
                break
        info[standard_name] = found if found else 'NOT FOUND'

    return info


def print_column_info(df: pd.DataFrame):
    """Print column mapping information for debugging."""
    info = get_casas_column_info(df)
    print("\n" + "="*60)
    print("Column Mapping Information:")
    print("="*60)
    for standard, actual in info.items():
        status = "✓" if actual != 'NOT FOUND' else "✗"
        print(f"{status} {standard:15s} -> {actual}")
    print("="*60 + "\n")


# ============================================================================
# Window Metadata
# ============================================================================


def compute_window_metadata(events_df: pd.DataFrame,
                            window_id: int,
                            presegmented: bool = False) -> Dict[str, Any]:
    """Compute metadata for a window of events.

    Args:
        events_df: DataFrame of events in this window
        window_id: Unique window identifier
        presegmented: Whether this came from presegmented data

    Returns:
        Dictionary of metadata
    """
    if len(events_df) == 0:
        return {}

    # Time information (columns are now standardized)
    timestamps = pd.to_datetime(events_df['timestamp'])
    start_time = timestamps.iloc[0]
    end_time = timestamps.iloc[-1]
    duration_sec = (end_time - start_time).total_seconds()

    # Spatial information (columns are now standardized)
    rooms = events_df['room'].dropna().tolist() if 'room' in events_df.columns else []
    rooms_visited = list(dict.fromkeys(rooms))  # Preserve order, remove duplicates
    primary_room = Counter(rooms).most_common(1)[0][0] if rooms else "unknown"
    room_transitions = count_room_transitions(events_df)

    # Activity information (columns are now standardized)
    activity_l1_list = events_df['activity_l1'].dropna().tolist() if 'activity_l1' in events_df.columns else []
    activity_l2_list = events_df['activity_l2'].dropna().tolist() if 'activity_l2' in events_df.columns else []

    primary_activity_l1 = Counter(activity_l1_list).most_common(1)[0][0] if activity_l1_list else "unknown"
    primary_activity_l2 = Counter(activity_l2_list).most_common(1)[0][0] if activity_l2_list else "unknown"

    # Activity distribution
    activity_dist = {}
    if activity_l1_list:
        total = len(activity_l1_list)
        for activity, count in Counter(activity_l1_list).items():
            activity_dist[activity] = count / total

    # Multi-resident info: capture the dominant resident_info for this window
    resident_info = None
    if 'resident_info' in events_df.columns:
        resident_values = events_df['resident_info'].dropna()
        if len(resident_values) > 0:
            resident_info = Counter(resident_values).most_common(1)[0][0]

    # Special sensor information (sensors with details, not just room location)
    special_sensors_info = {}
    if 'sensor_detail' in events_df.columns:
        # Get all special sensors triggered in this window
        special_sensors = events_df['sensor_detail'].dropna().tolist()

        if special_sensors:
            # All special sensors (unique, preserving order)
            special_sensors_list = list(dict.fromkeys(special_sensors))

            # Count frequencies
            sensor_counts = Counter(special_sensors)

            # Find most frequent special sensor (if triggered at least 2 times)
            most_common = sensor_counts.most_common(1)[0]
            primary_special_sensor = most_common[0] if most_common[1] >= 2 else None

            special_sensors_info = {
                'special_sensors_triggered': special_sensors_list,
                'num_special_sensors': len(special_sensors_list),
                'primary_special_sensor': primary_special_sensor,
                'special_sensor_counts': dict(sensor_counts)
            }

    metadata = {
        'window_id': window_id,
        'start_time': str(start_time),
        'end_time': str(end_time),
        'duration_seconds': duration_sec,
        'num_events': len(events_df),
        'rooms_visited': rooms_visited,
        'primary_room': primary_room,
        'room_transitions': room_transitions,
        'ground_truth_labels': {
            'primary_l1': primary_activity_l1,
            'primary_l2': primary_activity_l2,
            'all_labels_l1': list(set(activity_l1_list)),
            'all_labels_l2': list(set(activity_l2_list)),
            'label_distribution': activity_dist,
            'num_unique_labels_l1': len(set(activity_l1_list)),
            'num_unique_labels_l2': len(set(activity_l2_list))
        },
        'presegmented': presegmented
    }

    # Add resident_info if available (multi-resident datasets only)
    if resident_info is not None:
        metadata['resident_info'] = resident_info

    # Add special sensor info if available
    if special_sensors_info:
        metadata['special_sensors'] = special_sensors_info

    return metadata


def count_room_transitions(df: pd.DataFrame) -> int:
    """Count number of room transitions in a sequence.

    Only counts transitions for ON signals (not OFF signals), as OFF signals
    don't represent actual movement/presence in a location.
    """
    if 'room' not in df.columns or len(df) < 2:
        return 0

    # Filter to only ON events (standardized column name is 'event_type')
    if 'event_type' in df.columns:
        df_on = df[df['event_type'] == 'ON'].copy()
    else:
        df_on = df.copy()  # Fallback if event_type not available

    if len(df_on) < 2:
        return 0

    rooms = df_on['room'].dropna().tolist()
    if len(rooms) < 2:
        return 0

    transitions = sum(1 for i in range(1, len(rooms)) if rooms[i] != rooms[i-1])
    return transitions


def check_time_gap(df: pd.DataFrame, max_gap_minutes: Optional[int]) -> bool:
    """Check if any time gap in the sequence exceeds the maximum.

    Args:
        df: DataFrame of events (with standardized 'timestamp' column)
        max_gap_minutes: Maximum allowed gap in minutes (None = no limit)

    Returns:
        True if all gaps are acceptable, False otherwise
    """
    if max_gap_minutes is None or len(df) < 2:
        return True

    timestamps = pd.to_datetime(df['timestamp'])
    gaps = timestamps.diff()[1:]  # Skip first NaT
    max_gap = gaps.max()

    return max_gap <= timedelta(minutes=max_gap_minutes)


def create_sliding_windows(df: pd.DataFrame,
                          window_size: int,
                          overlap_factor: float,
                          min_events: int = 1) -> List[Tuple[int, int]]:
    """Create sliding window indices.

    Args:
        df: DataFrame of events
        window_size: Number of events per window
        overlap_factor: Overlap ratio (0.5 = 50% overlap)
        min_events: Minimum events required in a window

    Returns:
        List of (start_idx, end_idx) tuples
    """
    if len(df) < min_events:
        return []

    windows = []
    stride = max(1, int(window_size * (1 - overlap_factor)))

    start_idx = 0
    while start_idx + min_events <= len(df):
        end_idx = min(start_idx + window_size, len(df))
        windows.append((start_idx, end_idx))

        # Move to next window
        if end_idx >= len(df):
            break
        start_idx += stride

    return windows


def create_duration_windows(df: pd.DataFrame,
                           duration_seconds: int,
                           overlap_factor: float,
                           min_events: int = 1) -> List[Tuple[int, int]]:
    """Create time-duration-based window indices.

    Args:
        df: DataFrame of events (with standardized 'timestamp' column)
        duration_seconds: Duration of each window in seconds
        overlap_factor: Overlap ratio (0.5 = 50% overlap)
        min_events: Minimum events required in a window

    Returns:
        List of (start_idx, end_idx) tuples
    """
    if len(df) < min_events:
        return []

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    windows = []
    stride_seconds = max(1, int(duration_seconds * (1 - overlap_factor)))

    start_idx = 0
    while start_idx < len(df):
        start_time = df.loc[start_idx, 'timestamp']
        end_time = start_time + timedelta(seconds=duration_seconds)

        # Find all events within this time window
        mask = (df['timestamp'] >= start_time) & (df['timestamp'] < end_time)
        event_indices = df[mask].index.tolist()

        if len(event_indices) >= min_events:
            end_idx = event_indices[-1] + 1  # Inclusive end
            windows.append((start_idx, end_idx))

        # Move to next window by time stride
        next_start_time = start_time + timedelta(seconds=stride_seconds)
        next_indices = df[df['timestamp'] >= next_start_time].index

        if len(next_indices) == 0:
            break

        next_start_idx = next_indices[0]
        if next_start_idx <= start_idx:
            # Avoid infinite loop
            start_idx += 1
        else:
            start_idx = next_start_idx

    return windows


def presegment_by_activity(df: pd.DataFrame,
                           label_level: str = 'l1',
                           min_events: int = 1,
                           exclude_no_activity: bool = True) -> List[pd.DataFrame]:
    """Split dataframe into segments based on ground truth activity labels.

    Args:
        df: DataFrame with standardized activity columns ('activity_l1', 'activity_l2')
        label_level: 'l1' or 'l2'
        min_events: Minimum events per segment
        exclude_no_activity: Skip segments with no activity

    Returns:
        List of DataFrames, one per activity segment
    """
    if len(df) == 0:
        return []

    # Use standardized column names
    label_col = f'activity_{label_level}'

    if label_col not in df.columns:
        print(f"Warning: {label_col} not found, returning full dataframe")
        return [df]

    segments = []
    current_segment = []
    current_label = None

    for idx, row in df.iterrows():
        label = row[label_col]

        if pd.isna(label):
            # Continue current segment
            if current_segment:
                current_segment.append(idx)
        elif label != current_label:
            # New activity - save previous segment
            if current_segment and len(current_segment) >= min_events:
                segment_df = df.loc[current_segment].copy()

                # Check if we should exclude no-activity segments
                if exclude_no_activity:
                    primary_label = segment_df[label_col].mode()[0] if len(segment_df) > 0 else None
                    if primary_label and 'no_activity' not in primary_label.lower():
                        segments.append(segment_df)
                else:
                    segments.append(segment_df)

            # Start new segment
            current_segment = [idx]
            current_label = label
        else:
            # Same activity, continue segment
            current_segment.append(idx)

    # Don't forget the last segment
    if current_segment and len(current_segment) >= min_events:
        segment_df = df.loc[current_segment].copy()
        if exclude_no_activity:
            primary_label = segment_df[label_col].mode()[0] if len(segment_df) > 0 else None
            if primary_label and 'no_activity' not in primary_label.lower():
                segments.append(segment_df)
        else:
            segments.append(segment_df)

    return segments


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    from pathlib import Path

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# ============================================================================
# Statistics Comparison and Analysis
# ============================================================================

def analyze_sampling_statistics(
    dataset_name: str,
    base_dir: str = "data/processed/casas",
    output_dir: str = "results/eda/datasources",
    include_plots: bool = True
):
    """
    Analyze and compare sampling statistics across all sampling strategies for a dataset.

    Collects statistics from all statistics.json files in the processed data folders,
    creates comparison tables and visualizations.

    Args:
        dataset_name: Name of dataset (e.g., 'milan', 'aruba', 'marble')
        base_dir: Base directory where processed data is stored
        output_dir: Where to save the output tables and charts
        include_plots: Whether to generate visualization plots

    Returns:
        DataFrame with comparison statistics
    """
    import json
    import glob
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from datetime import datetime

    print(f"\n{'='*70}")
    print(f"Analyzing Sampling Statistics for {dataset_name.upper()}")
    print(f"{'='*70}\n")

    # Find all statistics files
    if dataset_name.lower() == 'marble':
        search_pattern = f"data/processed/marble/**/statistics.json"
    else:
        search_pattern = f"{base_dir}/{dataset_name}/*/statistics.json"

    stats_files = glob.glob(search_pattern, recursive=True)

    if not stats_files:
        print(f"No statistics files found for {dataset_name} in {search_pattern}")
        return None

    print(f"Found {len(stats_files)} statistics files:")
    for f in stats_files:
        print(f"  - {f}")

    # Load all statistics
    all_stats = []
    for stats_file in stats_files:
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        # Extract folder name for strategy identification
        folder_name = Path(stats_file).parent.name

        # Parse the statistics
        row = {
            'strategy_folder': folder_name,
            'sampling_strategy': stats.get('sampling_strategy', 'unknown'),
            'generated_at': stats.get('generated_at', ''),
            'processing_duration_sec': stats.get('processing_duration_seconds', 0),

            # Split info
            'split_strategy': stats.get('split_strategy', ''),
            'train_ratio': stats.get('train_ratio', 0),
            'random_seed': stats.get('random_seed', 42),

            # Sample counts
            'total_samples': stats.get('total_samples', 0),
            'train_samples': stats.get('train_samples_count', 0),
            'test_samples': stats.get('test_samples_count', 0),
        }

        # Train statistics
        train_stats = stats.get('train_statistics', {})
        row.update({
            'train_avg_seq_length': train_stats.get('avg_sequence_length', 0),
            'train_std_seq_length': train_stats.get('std_sequence_length', 0),
            'train_min_seq_length': train_stats.get('min_sequence_length', 0),
            'train_max_seq_length': train_stats.get('max_sequence_length', 0),
            'train_avg_duration': train_stats.get('avg_duration_seconds', 0),
            'train_min_duration': train_stats.get('min_duration_seconds', 0),
            'train_max_duration': train_stats.get('max_duration_seconds', 0),
        })

        # Test statistics
        test_stats = stats.get('test_statistics', {})
        row.update({
            'test_avg_seq_length': test_stats.get('avg_sequence_length', 0),
            'test_std_seq_length': test_stats.get('std_sequence_length', 0),
            'test_min_seq_length': test_stats.get('min_sequence_length', 0),
            'test_max_seq_length': test_stats.get('max_sequence_length', 0),
            'test_avg_duration': test_stats.get('avg_duration_seconds', 0),
            'test_min_duration': test_stats.get('min_duration_seconds', 0),
            'test_max_duration': test_stats.get('max_duration_seconds', 0),
        })

        # Extract sampling parameters
        sampling_params = stats.get('sampling_params', {})
        if 'duration_seconds' in sampling_params:
            row['duration_seconds'] = sampling_params.get('duration_seconds', [None])[0]
        if 'window_sizes' in sampling_params:
            row['window_size'] = sampling_params.get('window_sizes', [None])[0]

        row['overlap_factor'] = sampling_params.get('overlap_factor', None)
        row['use_presegmentation'] = sampling_params.get('use_presegmentation', False)

        all_stats.append(row)

    # Create DataFrame
    df_stats = pd.DataFrame(all_stats)

    # Sort by strategy type and size (numerically)
    # Create a sort key that handles duration and window size numerically
    def get_sort_key(row):
        folder = row['strategy_folder']
        strategy = row['sampling_strategy']

        # Extract numeric value for sorting
        if 'duration' in folder or 'FD_' in folder:
            # Extract duration number (e.g., "30" from "FD_30")
            import re
            match = re.search(r'(?:duration_|FD_)(\d+)', folder)
            duration_val = int(match.group(1)) if match else 0
            # Sort: fixed_duration first, then by duration value, then presegmented last
            preseg = 1 if 'preseg' in folder else 0
            return (0, duration_val, preseg)
        elif 'length' in folder or 'FL_' in folder:
            # Extract length number (e.g., "20" from "FL_20")
            import re
            match = re.search(r'(?:length_|FL_)(\d+)', folder)
            length_val = int(match.group(1)) if match else 0
            # Sort: fixed_length second, then by length value, then presegmented last
            preseg = 1 if 'preseg' in folder else 0
            return (1, length_val, preseg)
        else:
            return (2, 0, 0)

    df_stats['_sort_key'] = df_stats.apply(get_sort_key, axis=1)
    df_stats = df_stats.sort_values('_sort_key').drop('_sort_key', axis=1)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save full statistics table
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_path / f'{dataset_name}_sampling_statistics_{timestamp}.csv'
    df_stats.to_csv(csv_path, index=False)
    print(f"\n✓ Saved detailed statistics to: {csv_path}")

    # Create summary table
    summary_cols = [
        'strategy_folder', 'sampling_strategy',
        'total_samples', 'train_samples', 'test_samples',
        'train_avg_seq_length', 'train_avg_duration',
        'processing_duration_sec', 'use_presegmentation'
    ]

    if 'duration_seconds' in df_stats.columns:
        summary_cols.insert(2, 'duration_seconds')
    if 'window_size' in df_stats.columns:
        summary_cols.insert(2, 'window_size')

    df_summary = df_stats[summary_cols].copy()

    # Format for display
    df_summary['train_avg_seq_length'] = df_summary['train_avg_seq_length'].round(1)
    df_summary['train_avg_duration'] = df_summary['train_avg_duration'].round(1)
    df_summary['processing_duration_sec'] = df_summary['processing_duration_sec'].round(1)

    # Save summary table
    summary_path = output_path / f'{dataset_name}_sampling_summary.csv'
    df_summary.to_csv(summary_path, index=False)
    print(f"✓ Saved summary table to: {summary_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY TABLE - {dataset_name.upper()}")
    print(f"{'='*70}\n")
    print(df_summary.to_string(index=False))
    print()

    # Generate visualizations
    if include_plots and len(df_stats) > 0:
        print("\nGenerating visualizations...")
        _generate_sampling_comparison_plots(df_stats, dataset_name, output_path)

    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}\n")

    return df_stats


def _generate_sampling_comparison_plots(df_stats: pd.DataFrame, dataset_name: str, output_dir: Path):
    """Generate comparison plots for sampling statistics."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))

    # 1. Total Samples by Strategy
    ax1 = plt.subplot(2, 3, 1)
    strategies = df_stats['strategy_folder'].values
    samples = df_stats['total_samples'].values
    colors = ['#3498db' if 'preseg' not in s else '#e74c3c' for s in strategies]

    bars = ax1.barh(range(len(strategies)), samples, color=colors)
    ax1.set_yticks(range(len(strategies)))
    ax1.set_yticklabels([s.replace('fixed_', '').replace('_', '\n') for s in strategies], fontsize=8)
    ax1.set_xlabel('Total Samples', fontweight='bold')
    ax1.set_title(f'Sample Counts by Strategy - {dataset_name.upper()}', fontweight='bold', fontsize=12)
    ax1.invert_yaxis()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, samples)):
        ax1.text(val, bar.get_y() + bar.get_height()/2, f' {int(val):,}',
                va='center', fontsize=8)

    # 2. Average Sequence Length
    ax2 = plt.subplot(2, 3, 2)
    train_seq = df_stats['train_avg_seq_length'].values
    test_seq = df_stats['test_avg_seq_length'].values

    x = np.arange(len(strategies))
    width = 0.35

    ax2.bar(x - width/2, train_seq, width, label='Train', color='#2ecc71')
    ax2.bar(x + width/2, test_seq, width, label='Test', color='#f39c12')

    ax2.set_ylabel('Avg Sequence Length', fontweight='bold')
    ax2.set_title(f'Sequence Lengths - {dataset_name.upper()}', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace('fixed_', '').replace('_', '\n') for s in strategies],
                         rotation=45, ha='right', fontsize=7)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Average Duration (if applicable)
    ax3 = plt.subplot(2, 3, 3)
    if df_stats['train_avg_duration'].max() > 0:
        durations = df_stats['train_avg_duration'].values
        colors_dur = ['#9b59b6' if 'duration' in s else '#1abc9c' for s in strategies]

        bars = ax3.barh(range(len(strategies)), durations, color=colors_dur)
        ax3.set_yticks(range(len(strategies)))
        ax3.set_yticklabels([s.replace('fixed_', '').replace('_', '\n') for s in strategies], fontsize=8)
        ax3.set_xlabel('Avg Duration (seconds)', fontweight='bold')
        ax3.set_title(f'Window Durations - {dataset_name.upper()}', fontweight='bold', fontsize=12)
        ax3.invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars, durations)):
            if val > 0:
                ax3.text(val, bar.get_y() + bar.get_height()/2, f' {val:.1f}s',
                        va='center', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'Duration not applicable\n(fixed-length strategies)',
                ha='center', va='center', transform=ax3.transAxes, fontsize=10)
        ax3.axis('off')

    # 4. Processing Time
    ax4 = plt.subplot(2, 3, 4)
    proc_times = df_stats['processing_duration_sec'].values
    colors_time = ['#e67e22' if t > 100 else '#27ae60' for t in proc_times]

    bars = ax4.barh(range(len(strategies)), proc_times, color=colors_time)
    ax4.set_yticks(range(len(strategies)))
    ax4.set_yticklabels([s.replace('fixed_', '').replace('_', '\n') for s in strategies], fontsize=8)
    ax4.set_xlabel('Processing Time (seconds)', fontweight='bold')
    ax4.set_title(f'Processing Time - {dataset_name.upper()}', fontweight='bold', fontsize=12)
    ax4.invert_yaxis()

    for i, (bar, val) in enumerate(zip(bars, proc_times)):
        ax4.text(val, bar.get_y() + bar.get_height()/2, f' {val:.1f}s',
                va='center', fontsize=8)

    # 5. Train/Test Split Comparison
    ax5 = plt.subplot(2, 3, 5)
    train_counts = df_stats['train_samples'].values
    test_counts = df_stats['test_samples'].values

    x = np.arange(len(strategies))
    width = 0.35

    ax5.bar(x - width/2, train_counts, width, label='Train', color='#3498db', alpha=0.8)
    ax5.bar(x + width/2, test_counts, width, label='Test', color='#e74c3c', alpha=0.8)

    ax5.set_ylabel('Sample Count', fontweight='bold')
    ax5.set_title(f'Train/Test Split - {dataset_name.upper()}', fontweight='bold', fontsize=12)
    ax5.set_xticks(x)
    ax5.set_xticklabels([s.replace('fixed_', '').replace('_', '\n') for s in strategies],
                         rotation=45, ha='right', fontsize=7)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    # 6. Presegmentation vs Standard
    ax6 = plt.subplot(2, 3, 6)

    # Group by presegmentation
    df_stats['is_presegmented'] = df_stats['use_presegmentation']

    if df_stats['is_presegmented'].any():
        preseg_counts = df_stats.groupby('is_presegmented')['total_samples'].sum()
        preseg_avg_seq = df_stats.groupby('is_presegmented')['train_avg_seq_length'].mean()

        categories = ['Standard', 'Presegmented']
        counts = [preseg_counts.get(False, 0), preseg_counts.get(True, 0)]
        avg_seqs = [preseg_avg_seq.get(False, 0), preseg_avg_seq.get(True, 0)]

        x = np.arange(len(categories))
        width = 0.35

        ax6_twin = ax6.twinx()

        bars1 = ax6.bar(x - width/2, counts, width, label='Total Samples', color='#3498db', alpha=0.7)
        bars2 = ax6_twin.bar(x + width/2, avg_seqs, width, label='Avg Seq Length', color='#e74c3c', alpha=0.7)

        ax6.set_xlabel('Strategy Type', fontweight='bold')
        ax6.set_ylabel('Total Samples', color='#3498db', fontweight='bold')
        ax6_twin.set_ylabel('Avg Sequence Length', color='#e74c3c', fontweight='bold')
        ax6.set_title(f'Presegmentation Impact - {dataset_name.upper()}', fontweight='bold', fontsize=12)
        ax6.set_xticks(x)
        ax6.set_xticklabels(categories)

        # Add value labels
        for bar, val in zip(bars1, counts):
            ax6.text(bar.get_x() + bar.get_width()/2, val, f'{int(val):,}',
                    ha='center', va='bottom', fontsize=9, color='#3498db', fontweight='bold')

        for bar, val in zip(bars2, avg_seqs):
            ax6_twin.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}',
                         ha='center', va='bottom', fontsize=9, color='#e74c3c', fontweight='bold')

        ax6.tick_params(axis='y', labelcolor='#3498db')
        ax6_twin.tick_params(axis='y', labelcolor='#e74c3c')
    else:
        ax6.text(0.5, 0.5, 'No presegmented strategies available',
                ha='center', va='center', transform=ax6.transAxes, fontsize=10)
        ax6.axis('off')

    plt.tight_layout()

    # Save figure
    output_file = output_dir / f'{dataset_name}_sampling_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved comparison plots to: {output_file}")


if __name__ == '__main__':
    """Run statistics analysis from command line."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Collect and analyze sampling statistics across all strategies for a dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze Milan dataset sampling statistics
    python src/sampling/utils.py --collect-statistics milan

    # Analyze Aruba dataset
    python src/sampling/utils.py --collect-statistics aruba

    # Analyze Marble dataset
    python src/sampling/utils.py --collect-statistics marble

    # Specify custom output directory
    python src/sampling/utils.py --collect-statistics milan --output-dir my_results

    # Skip generating plots (faster)
    python src/sampling/utils.py --collect-statistics milan --no-plots
        """
    )

    parser.add_argument(
        '--collect-statistics', '--collect', '--analyze',
        dest='dataset',
        type=str,
        required=True,
        metavar='DATASET',
        help='Collect and analyze sampling statistics for DATASET (e.g., milan, aruba, marble)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results/eda/datasources',
        help='Output directory for results (default: results/eda/datasources)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating visualization plots'
    )

    args = parser.parse_args()

    # Run analysis
    analyze_sampling_statistics(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        include_plots=not args.no_plots
    )

