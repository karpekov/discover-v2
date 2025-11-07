"""
Exploratory Data Analysis (EDA) for MARBLE dataset.

This script generates comprehensive visualizations and statistics for the MARBLE dataset.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def load_marble_data(data_path: str = 'data/processed/marble/marble_environment_single_resident.csv'):
    """Load the processed MARBLE dataset."""
    df = pd.read_csv(data_path)

    # Convert timestamps to datetime
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df['activity_start_dt'] = pd.to_datetime(df['activity_start'], unit='ms', errors='coerce')
    df['activity_end_dt'] = pd.to_datetime(df['activity_end'], unit='ms', errors='coerce')

    # Extract time features
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['time_of_day_seconds'] = df['hour'] * 3600 + df['minute'] * 60

    return df


def load_marble_metadata(metadata_path: str = 'metadata/marble_metadata.json'):
    """Load MARBLE metadata."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata['marble']


def create_output_dir(output_dir: str = 'results/eda/marble'):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_activity_timeline(df: pd.DataFrame, metadata: Dict, output_dir: str,
                           sample_size: Optional[int] = None):
    """Create timeline visualization of activities during the day.

    x-axis: time during the day
    1D line with colored bars (no y-axis sequence)
    Creates separate subplots for each scenario+instance combination.
    Splits long timelines (>3000 mins) into subcharts.
    """
    print("Creating activity timeline visualization...")

    # Sample data if requested
    if sample_size:
        df_plot = df.head(sample_size).copy()
    else:
        df_plot = df.copy()

    # Get unique scenario+instance combinations
    df_plot['scenario_instance'] = df_plot['scenario'] + '_inst' + df_plot['instance'].astype(str)
    scenario_instances = sorted(df_plot['scenario_instance'].unique())

    # Get activity colors
    label_colors = metadata.get('label_color', {})

    # Collect all subplots (may need multiple for long timelines)
    all_subplots = []

    for scenario_instance in scenario_instances:
        df_combo = df_plot[df_plot['scenario_instance'] == scenario_instance].copy()
        scenario = df_combo['scenario'].iloc[0] if len(df_combo) > 0 else ''
        instance = df_combo['instance'].iloc[0] if len(df_combo) > 0 else ''
        time_of_day = df_combo['time_of_day'].iloc[0] if len(df_combo) > 0 else ''
        tod_name = {'m': 'Morning', 'a': 'Afternoon', 'e': 'Evening'}.get(time_of_day, time_of_day)

        # Sort by timestamp
        df_combo = df_combo.sort_values('ts').reset_index(drop=True)

        # Group by activity periods
        activity_periods = []
        for idx, row in df_combo.iterrows():
            if pd.notna(row['activity']) and pd.notna(row['activity_start']) and pd.notna(row['activity_end']):
                activity_periods.append({
                    'start': row['activity_start'],
                    'end': row['activity_end'],
                    'activity': row['activity'],
                    'color': label_colors.get(row['activity'], '#878787')
                })

        if len(activity_periods) == 0:
            continue

        # Remove duplicates and sort
        activity_periods = pd.DataFrame(activity_periods).drop_duplicates(['start', 'end', 'activity'])
        activity_periods = activity_periods.sort_values('start').reset_index(drop=True)

        # Get base time and total duration
        base_time = activity_periods.iloc[0]['start']
        last_time = activity_periods.iloc[-1]['end']
        total_duration = (last_time - base_time) / 1000 / 60  # minutes

        # Split into subcharts if duration > 3000 minutes
        if total_duration > 3000:
            # Split into chunks of ~2000 minutes
            chunk_size = 2000  # minutes
            n_chunks = int(np.ceil(total_duration / chunk_size))

            for chunk_idx in range(n_chunks):
                chunk_start_min = chunk_idx * chunk_size
                chunk_end_min = (chunk_idx + 1) * chunk_size
                chunk_start_ts = base_time + chunk_start_min * 60 * 1000
                chunk_end_ts = base_time + chunk_end_min * 60 * 1000

                # Filter periods in this chunk
                chunk_periods = activity_periods[
                    (activity_periods['end'] >= chunk_start_ts) &
                    (activity_periods['start'] <= chunk_end_ts)
                ].copy()

                if len(chunk_periods) > 0:
                    all_subplots.append({
                        'periods': chunk_periods,
                        'base_time': chunk_start_ts,
                        'scenario': scenario,
                        'instance': instance,
                        'time_of_day': tod_name,
                        'chunk': chunk_idx + 1,
                        'n_chunks': n_chunks
                    })
        else:
            all_subplots.append({
                'periods': activity_periods,
                'base_time': base_time,
                'scenario': scenario,
                'instance': instance,
                'time_of_day': tod_name,
                'chunk': None,
                'n_chunks': None
            })

    # Collect all unique activities for legend
    all_activities = set()
    for subplot_data in all_subplots:
        for _, period in subplot_data['periods'].iterrows():
            all_activities.add((period['activity'], period['color']))

    # Create figure with subplots (increased height for better spacing)
    n_subplots = len(all_subplots)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(16, 1.5 * n_subplots))

    if n_subplots == 1:
        axes = [axes]

    # Add more spacing between subplots for subheaders
    plt.subplots_adjust(hspace=2.0)

    # Track which activities we've added to legend (to avoid duplicates)
    legend_handles = {}

    for subplot_idx, subplot_data in enumerate(all_subplots):
        ax = axes[subplot_idx]
        periods = subplot_data['periods']
        base_time = subplot_data['base_time']

        # Plot as 1D timeline (single horizontal bar, no y-axis sequence)
        y_pos = 0.5  # Fixed y position for 1D timeline

        for idx, period in periods.iterrows():
            start_ts = period['start']
            end_ts = period['end']
            duration = (end_ts - start_ts) / 1000 / 60  # Convert to minutes

            start_rel = (start_ts - base_time) / 1000 / 60  # minutes from start
            end_rel = (end_ts - base_time) / 1000 / 60

            # Plot horizontal bar
            bar = ax.barh(y_pos, end_rel - start_rel, left=start_rel, height=0.8,
                    color=period['color'], alpha=0.8, edgecolor='black', linewidth=1,
                    label=period['activity'])

            # Track for legend (only add once per activity)
            if period['activity'] not in legend_handles:
                from matplotlib.patches import Patch
                legend_handles[period['activity']] = Patch(facecolor=period['color'],
                                                          edgecolor='black',
                                                          label=period['activity'])

        # Set labels and title
        title = f"Scenario {subplot_data['scenario']} - Instance {subplot_data['instance']} ({subplot_data['time_of_day']})"
        if subplot_data['chunk'] is not None:
            title += f" - Part {subplot_data['chunk']}/{subplot_data['n_chunks']}"

        ax.set_xlabel('Time (minutes from start)', fontsize=10)
        ax.set_ylabel('', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([])  # Remove y-axis ticks
        ax.grid(True, alpha=0.3, axis='x')

    # Add legend to the figure (on the right side)
    if legend_handles:
        # Sort legend by activity name for consistency
        sorted_handles = [legend_handles[act] for act in sorted(legend_handles.keys())]
        fig.legend(handles=sorted_handles, loc='center right', bbox_to_anchor=(1.0, 0.5),
                  fontsize=9, title='Activities', title_fontsize=10, framealpha=0.9)

    plt.suptitle('Activity Timeline - Activities of Daily Living',
                 fontsize=14, fontweight='bold', y=0.995)
    # Adjust layout: leave space for legend on right, preserve hspace for subheaders
    plt.subplots_adjust(left=0.05, right=0.82, top=0.95, bottom=0.05, hspace=2.0)
    plt.savefig(os.path.join(output_dir, 'activity_timeline.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  ✓ Saved: activity_timeline.png")


def plot_sensor_activity_chart(df: pd.DataFrame, metadata: Dict, output_dir: str):
    """Create sensor activity chart.

    y-axis: sensor_id
    bars colored by activity label
    """
    print("Creating sensor activity chart...")

    # Get activity colors
    label_colors = metadata.get('label_color', {})

    # Count events per sensor and activity
    sensor_activity_counts = df.groupby(['sensor_id', 'activity']).size().reset_index(name='count')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Get unique sensors and activities
    sensors = sorted(df['sensor_id'].unique())
    activities = sorted(df['activity'].dropna().unique())

    # Create stacked bar chart
    bottom = np.zeros(len(sensors))

    for activity in activities:
        counts = []
        for sensor in sensors:
            count = sensor_activity_counts[
                (sensor_activity_counts['sensor_id'] == sensor) &
                (sensor_activity_counts['activity'] == activity)
            ]['count'].sum()
            counts.append(count)

        color = label_colors.get(activity, '#878787')
        ax.barh(sensors, counts, left=bottom, label=activity, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        bottom += np.array(counts)

    ax.set_xlabel('Number of Events', fontsize=12)
    ax.set_ylabel('Sensor ID', fontsize=12)
    ax.set_title('Sensor Activity by Activity Label', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensor_activity_chart.png'), bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: sensor_activity_chart.png")


def generate_summary_statistics(df: pd.DataFrame, metadata: Dict, output_dir: str):
    """Generate and save summary statistics."""
    print("Generating summary statistics...")

    stats = []

    # Basic statistics
    stats.append("=" * 80)
    stats.append("MARBLE Dataset Summary Statistics")
    stats.append("=" * 80)
    stats.append("")

    stats.append(f"Total number of events: {len(df):,}")
    stats.append(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    stats.append(f"Number of unique scenarios: {df['scenario'].nunique()}")
    stats.append(f"Scenarios: {', '.join(sorted(df['scenario'].unique()))}")
    stats.append(f"Number of unique instances: {df['instance'].nunique()}")
    stats.append(f"Number of unique subjects: {df['subject'].nunique()}")
    stats.append(f"Number of unique sensors: {df['sensor_id'].nunique()}")
    stats.append(f"Number of unique activities: {df['activity'].nunique()}")
    stats.append("")

    # Activity distribution
    stats.append("-" * 80)
    stats.append("Activity Distribution")
    stats.append("-" * 80)
    activity_counts = df['activity'].value_counts()
    for activity, count in activity_counts.items():
        pct = (count / len(df)) * 100
        stats.append(f"  {activity:30s}: {count:6d} ({pct:5.2f}%)")
    stats.append("")

    # Sensor distribution
    stats.append("-" * 80)
    stats.append("Sensor Distribution")
    stats.append("-" * 80)
    sensor_counts = df['sensor_id'].value_counts()
    for sensor, count in sensor_counts.items():
        pct = (count / len(df)) * 100
        stats.append(f"  {sensor:10s}: {count:6d} ({pct:5.2f}%)")
    stats.append("")

    # Time of day distribution
    stats.append("-" * 80)
    stats.append("Time of Day Distribution")
    stats.append("-" * 80)
    tod_counts = df['time_of_day'].value_counts()
    for tod, count in tod_counts.items():
        pct = (count / len(df)) * 100
        tod_name = {'m': 'Morning', 'a': 'Afternoon', 'e': 'Evening'}.get(tod, tod)
        stats.append(f"  {tod_name:15s}: {count:6d} ({pct:5.2f}%)")
    stats.append("")

    # Scenario distribution
    stats.append("-" * 80)
    stats.append("Scenario Distribution")
    stats.append("-" * 80)
    scenario_counts = df['scenario'].value_counts()
    for scenario, count in scenario_counts.items():
        pct = (count / len(df)) * 100
        stats.append(f"  {scenario:10s}: {count:6d} ({pct:5.2f}%)")
    stats.append("")

    # Activity duration statistics
    stats.append("-" * 80)
    stats.append("Activity Duration Statistics (minutes)")
    stats.append("-" * 80)
    df_with_duration = df[df['activity'].notna() & df['activity_start'].notna() & df['activity_end'].notna()].copy()
    df_with_duration['duration_min'] = (df_with_duration['activity_end'] - df_with_duration['activity_start']) / 1000 / 60

    duration_stats = df_with_duration.groupby('activity')['duration_min'].agg(['mean', 'median', 'std', 'min', 'max'])
    for activity in duration_stats.index:
        mean_dur = duration_stats.loc[activity, 'mean']
        median_dur = duration_stats.loc[activity, 'median']
        stats.append(f"  {activity:30s}: Mean={mean_dur:6.2f}, Median={median_dur:6.2f}")
    stats.append("")

    # Save statistics
    stats_text = '\n'.join(stats)
    with open(os.path.join(output_dir, 'summary_statistics.txt'), 'w') as f:
        f.write(stats_text)

    print(f"  ✓ Saved: summary_statistics.txt")

    # Also create a summary DataFrame
    summary_df = pd.DataFrame({
        'Metric': [
            'Total Events', 'Unique Scenarios', 'Unique Instances',
            'Unique Subjects', 'Unique Sensors', 'Unique Activities',
            'Date Range Start', 'Date Range End'
        ],
        'Value': [
            len(df),
            df['scenario'].nunique(),
            df['instance'].nunique(),
            df['subject'].nunique(),
            df['sensor_id'].nunique(),
            df['activity'].nunique(),
            str(df['datetime'].min()),
            str(df['datetime'].max())
        ]
    })
    summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
    print(f"  ✓ Saved: summary_statistics.csv")


def plot_activity_heatmaps(df: pd.DataFrame, metadata: Dict, output_dir: str):
    """Create KDE heatmaps for each activity on the floor plan.
    Shows all activities in a single figure with subplots.
    Background is 70% transparent (30% opacity).
    """
    print("Creating activity heatmaps on floor plan (KDE plots)...")

    # Check if floor plan image exists
    floor_plan_paths = [
        'metadata/floor_plans_augmented/marble.png',
        'metadata/floor_plans_augmented/marble.jpg',
        'charts/house_layouts/marble.png',
        'charts/house_layouts/marble.jpg'
    ]

    floor_plan_path = None
    for path in floor_plan_paths:
        if os.path.exists(path):
            floor_plan_path = path
            break

    if floor_plan_path is None:
        print("  ⚠️  Floor plan image not found. Creating heatmaps without background image.")
        use_floor_plan = False
    else:
        print(f"  Using floor plan: {floor_plan_path}")
        use_floor_plan = True

    # Get sensor coordinates
    sensor_coords = metadata.get('sensor_coordinates', {})
    if not sensor_coords:
        print("  ⚠️  No sensor coordinates found in metadata. Skipping heatmaps.")
        return

    # Get unique activities (excluding NaN and TRANSITION)
    activities = sorted([a for a in df['activity'].dropna().unique() if a != 'TRANSITION'])

    if len(activities) == 0:
        print("  ⚠️  No activities found. Skipping heatmaps.")
        return

    # Load floor plan image once
    img = None
    img_height, img_width = 600, 800  # Default dimensions
    if use_floor_plan:
        try:
            img = plt.imread(floor_plan_path)
            img_height, img_width = img.shape[:2]
        except Exception as e:
            print(f"    Warning: Could not load floor plan image: {e}")
            img = None
            use_floor_plan = False

    # Calculate grid dimensions for subplots
    n_activities = len(activities)
    n_cols = 4
    n_rows = (n_activities + n_cols - 1) // n_cols

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_activities == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, activity in enumerate(activities):
        print(f"  Processing: {activity}")
        ax = axes[idx]

        # Filter data for this activity
        df_activity = df[df['activity'] == activity].copy()

        if len(df_activity) == 0:
            ax.axis('off')
            continue

        # Get sensor coordinates for events in this activity
        x_coords = []
        y_coords = []

        for sensor_id in df_activity['sensor_id'].unique():
            if sensor_id in sensor_coords:
                x, y = sensor_coords[sensor_id]
                # Repeat coordinates based on event count for this sensor
                count = len(df_activity[df_activity['sensor_id'] == sensor_id])
                x_coords.extend([x] * count)
                y_coords.extend([y] * count)

        if len(x_coords) == 0:
            ax.axis('off')
            continue

        # Display floor plan background (60% transparent = 40% opacity)
        if img is not None:
            ax.imshow(img, extent=[0, img_width, img_height, 0], alpha=0.4, aspect='auto', zorder=0)

        # Create KDE heatmap
        try:
            from scipy.stats import gaussian_kde

            # Need at least 2 unique points for KDE
            unique_points = len(set(zip(x_coords, y_coords)))
            if unique_points < 2:
                # Use 2D histogram for single point
                hist, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=20)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                im = ax.imshow(hist.T, origin='lower', extent=extent, cmap='YlOrRd',
                              alpha=0.6, aspect='auto', zorder=1)
            else:
                # Create KDE with proper data structure
                values = np.vstack([x_coords, y_coords])

                # Add small jitter if all points are too close (helps with singular matrix)
                if unique_points < len(x_coords) * 0.5:
                    np.random.seed(42)
                    jitter = np.random.normal(0, 2, values.shape)
                    values = values + jitter

                # Create KDE with adjusted bandwidth
                kde = gaussian_kde(values)

                # Adjust bandwidth if needed (multiply by factor to make it more spread out)
                kde.set_bandwidth(kde.factor * 1.5)

                # Create grid for KDE evaluation
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                margin = 50
                x_grid = np.linspace(max(0, x_min - margin), min(img_width, x_max + margin), 100)
                y_grid = np.linspace(max(0, y_min - margin), min(img_height, y_max + margin), 100)
                X, Y = np.meshgrid(x_grid, y_grid)
                positions = np.vstack([X.ravel(), Y.ravel()])

                # Evaluate KDE
                Z = kde(positions)
                Z = Z.reshape(X.shape)

                # Plot KDE heatmap
                im = ax.contourf(X, Y, Z, levels=20, cmap='YlOrRd', alpha=0.6, zorder=1)
                ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.2, linewidths=0.5, zorder=1)

        except ImportError:
            # Fallback to 2D histogram if scipy not available
            hist, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=30)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax.imshow(hist.T, origin='lower', extent=extent, cmap='YlOrRd',
                          alpha=0.6, aspect='auto', zorder=1)
        except Exception as e:
            # Fallback to 2D histogram if KDE fails
            try:
                hist, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=30)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                im = ax.imshow(hist.T, origin='lower', extent=extent, cmap='YlOrRd',
                              alpha=0.6, aspect='auto', zorder=1)
            except:
                # Final fallback to scatter plot
                ax.scatter(x_coords, y_coords, s=100, alpha=0.6, c='red', zorder=1, edgecolors='black', linewidths=1)

        # Count unique sensors for this activity
        unique_sensors = df_activity['sensor_id'].unique()
        n_unique_sensors = len(unique_sensors)

        # Add sensor labels and circles
        for sensor_id, (x, y) in sensor_coords.items():
            if sensor_id in df_activity['sensor_id'].values:
                # If 1-2 sensors only, draw large red circle
                if n_unique_sensors <= 2:
                    circle = plt.Circle((x, y), 30, color='red', fill=False, linewidth=3, zorder=3)
                    ax.add_patch(circle)

                ax.annotate(sensor_id, (x, y), fontsize=7, ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='black'),
                           zorder=4, fontweight='bold')

        # Set limits
        if img is not None:
            ax.set_xlim(0, img_width)
            ax.set_ylim(img_height, 0)  # Inverted for image coordinates
        else:
            if x_coords and y_coords:
                margin = 50
                ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
                ax.set_ylim(max(y_coords) + margin, min(y_coords) - margin)

        ax.set_title(activity, fontsize=10, fontweight='bold')
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for idx in range(n_activities, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Activity Heatmaps - Spatial Distribution by Activity (KDE)',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(output_dir, 'activity_heatmaps_all.png'),
               bbox_inches='tight', dpi=150)
    plt.close()

    print(f"  ✓ Saved: activity_heatmaps_all.png")


def plot_additional_eda_charts(df: pd.DataFrame, metadata: Dict, output_dir: str):
    """Create additional helpful EDA charts."""
    print("Creating additional EDA charts...")

    # 1. Activity distribution pie chart
    print("  Creating activity distribution pie chart...")
    fig, ax = plt.subplots(figsize=(12, 8))
    activity_counts = df['activity'].value_counts()
    label_colors = metadata.get('label_color', {})
    colors = [label_colors.get(act, '#878787') for act in activity_counts.index]

    ax.pie(activity_counts.values, labels=activity_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax.set_title('Activity Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activity_distribution_pie.png'), bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: activity_distribution_pie.png")

    # 2. Sensor usage by time of day
    print("  Creating sensor usage by time of day chart...")
    fig, ax = plt.subplots(figsize=(14, 8))
    sensor_tod = pd.crosstab(df['sensor_id'], df['time_of_day'])
    sensor_tod.plot(kind='bar', ax=ax, stacked=True, colormap='Set2')
    ax.set_xlabel('Sensor ID', fontsize=12)
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.set_title('Sensor Usage by Time of Day', fontsize=14, fontweight='bold')
    ax.legend(title='Time of Day', labels=['Morning', 'Afternoon', 'Evening'])
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensor_usage_by_tod.png'), bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: sensor_usage_by_tod.png")

    # 3. Activity duration distribution
    print("  Creating activity duration distribution...")
    df_with_duration = df[df['activity'].notna() & df['activity_start'].notna() &
                         df['activity_end'].notna()].copy()
    df_with_duration['duration_min'] = (df_with_duration['activity_end'] -
                                        df_with_duration['activity_start']) / 1000 / 60

    fig, ax = plt.subplots(figsize=(14, 8))
    activities = sorted(df_with_duration['activity'].unique())
    durations = [df_with_duration[df_with_duration['activity'] == act]['duration_min'].values
                for act in activities]

    bp = ax.boxplot(durations, labels=activities, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#7fbc41')
        patch.set_alpha(0.7)

    ax.set_xlabel('Activity', fontsize=12)
    ax.set_ylabel('Duration (minutes)', fontsize=12)
    ax.set_title('Activity Duration Distribution', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activity_duration_distribution.png'), bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: activity_duration_distribution.png")

    # 4. Sensor status distribution
    print("  Creating sensor status distribution...")
    fig, ax = plt.subplots(figsize=(12, 6))
    status_counts = df['sensor_status'].value_counts()
    status_counts.plot(kind='bar', ax=ax, color=['#7fbc41', '#EF8636'])
    ax.set_xlabel('Sensor Status', fontsize=12)
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.set_title('Sensor Status Distribution', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensor_status_distribution.png'), bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: sensor_status_distribution.png")

    # 5. Activity by scenario
    print("  Creating activity by scenario chart...")
    fig, ax = plt.subplots(figsize=(14, 8))
    scenario_activity = pd.crosstab(df['scenario'], df['activity'])
    scenario_activity.plot(kind='bar', ax=ax, stacked=True, colormap='tab20')
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.set_title('Activity Distribution by Scenario', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activity_by_scenario.png'), bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: activity_by_scenario.png")


def main():
    """Main function to run all EDA analyses."""
    print("=" * 80)
    print("MARBLE Dataset - Exploratory Data Analysis")
    print("=" * 80)
    print()

    # Load data and metadata
    print("Loading data and metadata...")
    df = load_marble_data()
    metadata = load_marble_metadata()
    print(f"  ✓ Loaded {len(df):,} events")
    print()

    # Create output directory
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}")
    print()

    # Generate visualizations
    plot_activity_timeline(df, metadata, output_dir)
    print()

    plot_sensor_activity_chart(df, metadata, output_dir)
    print()

    generate_summary_statistics(df, metadata, output_dir)
    print()

    plot_activity_heatmaps(df, metadata, output_dir)
    print()

    plot_additional_eda_charts(df, metadata, output_dir)
    print()

    print("=" * 80)
    print("✓ EDA Analysis Complete!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

