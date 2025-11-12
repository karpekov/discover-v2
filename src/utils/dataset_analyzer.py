"""
Dataset Analyzer for CASAS and MARBLE datasets.

This script analyzes sensor data from various smart home datasets to:
1. Count unique sensors and labels
2. Determine sensor value types (binary ON/OFF, numeric, etc.)
3. Generate visualizations and reports
4. Help classify sensor types (motion, door, temperature, etc.)

Datasets analyzed: aruba, milan, cairo, tulum2009, marble

Usage:
    conda activate discover-v2-env
    python src/utils/dataset_analyzer.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from collections import defaultdict, Counter
from termcolor import colored
import re

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.data_load_clean import (
    casas_end_to_end_preprocess,
    process_marble_environmental_data,
    CASAS_METADATA
)


class DatasetAnalyzer:
    """Analyze smart home datasets to extract sensor characteristics."""
    
    def __init__(self, output_dir='results/eda/datasources'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets = {
            'aruba': 'casas',
            'milan': 'casas',
            'cairo': 'casas',
            'tulum': 'casas',
            'marble': 'marble'
        }
        
        self.analysis_results = {}
        
    def load_casas_dataset(self, dataset_name, force_download=False):
        """Load a CASAS dataset."""
        print(colored(f"\n{'='*60}", 'cyan'))
        print(colored(f"Loading {dataset_name.upper()} dataset...", 'cyan'))
        print(colored(f"{'='*60}", 'cyan'))
        
        try:
            # Process the dataset (downloads if needed)
            df = casas_end_to_end_preprocess(
                dataset_name=dataset_name,
                save_to_csv=True,
                force_download=force_download
            )
            print(colored(f"✓ Successfully loaded {dataset_name}", 'green'))
            return df
        except Exception as e:
            print(colored(f"✗ Failed to load {dataset_name}: {e}", 'red'))
            return None
    
    def load_marble_dataset(self):
        """Load the MARBLE dataset."""
        print(colored(f"\n{'='*60}", 'cyan'))
        print(colored(f"Loading MARBLE dataset...", 'cyan'))
        print(colored(f"{'='*60}", 'cyan'))
        
        try:
            # First process environmental data
            df_env = process_marble_environmental_data()
            print(colored(f"✓ Successfully loaded marble environmental data", 'green'))
            return df_env
        except Exception as e:
            print(colored(f"✗ Failed to load marble: {e}", 'red'))
            return None
    
    def classify_sensor_type(self, sensor_id, sensor_location=None):
        """
        Classify sensor type based on sensor ID and location.
        
        Returns: sensor_type (motion, door, temperature, etc.)
        """
        sensor_id = str(sensor_id).upper()
        
        # Pattern matching for sensor types
        if sensor_id.startswith('M'):
            return 'motion'
        elif sensor_id.startswith('D'):
            return 'door'
        elif sensor_id.startswith('T'):
            return 'temperature'
        elif sensor_id.startswith('L'):
            return 'light'
        elif sensor_id.startswith('I'):
            return 'item'
        elif sensor_id.startswith('E'):
            return 'electricity'
        elif sensor_id.startswith('AD'):
            return 'analog'
        elif sensor_id.startswith('P') and len(sensor_id) <= 3:
            # P sensors in MARBLE dataset are pressure sensors
            return 'pressure'
        elif sensor_id.startswith('R') and len(sensor_id) <= 3:
            # R sensors in MARBLE dataset are magnetic sensors (not burners)
            return 'magnetic'
        elif sensor_id.startswith('F'):
            return 'fan'
        elif 'MOTION' in sensor_id or '_MOTION_' in sensor_id:
            return 'motion'
        elif 'DOOR' in sensor_id or 'ENTRY' in sensor_id or '_ENTRY_' in sensor_id:
            return 'door'
        elif 'TEMP' in sensor_id or 'TEMPERATURE' in sensor_id:
            return 'temperature'
        elif 'POWER' in sensor_id or 'CABLE' in sensor_id or '_POWER_' in sensor_id:
            return 'electricity'
        elif 'LIGHT' in sensor_id or '_LIGHT_' in sensor_id:
            return 'light'
        elif 'VIBRATION' in sensor_id or 'TOILETING' in sensor_id or '_TOILETING_' in sensor_id:
            return 'vibration'
        elif 'BED' in sensor_id or 'INBED' in sensor_id or 'ASLEEP' in sensor_id or 'BEDEXIT' in sensor_id:
            return 'bed'
        elif 'MEDBOX' in sensor_id or '_MEDBOX_' in sensor_id:
            return 'medication'
        elif 'HUMIDITY' in sensor_id:
            return 'humidity'
        else:
            return 'unknown'
    
    def analyze_sensor_values(self, df, sensor_col='sensor_id', value_col='sensor_status'):
        """
        Analyze the types of values sensors produce.
        
        Returns:
            dict: Statistics about sensor value types
        """
        results = {
            'binary_sensors': [],      # Sensors with ON/OFF or OPEN/CLOSE
            'numeric_sensors': [],      # Sensors with numeric values
            'multi_state_sensors': [],  # Sensors with multiple discrete states
            'value_distribution': defaultdict(list)
        }
        
        # Group by sensor and analyze values
        for sensor_id in df[sensor_col].unique():
            sensor_data = df[df[sensor_col] == sensor_id]
            unique_values = sensor_data[value_col].unique()
            
            # Check if numeric
            numeric_values = []
            for val in unique_values:
                try:
                    numeric_values.append(float(val))
                except (ValueError, TypeError):
                    pass
            
            # Classify sensor based on values
            if len(numeric_values) == len(unique_values) and len(unique_values) > 2:
                # All values are numeric and more than 2 unique values
                results['numeric_sensors'].append({
                    'sensor_id': sensor_id,
                    'min': min(numeric_values),
                    'max': max(numeric_values),
                    'mean': np.mean(numeric_values),
                    'unique_count': len(unique_values)
                })
            elif set([str(v).lower() for v in unique_values]).issubset({'on', 'off', 'open', 'close', 'present', 'absent', '1', '0', 'true', 'false'}):
                # Binary sensor
                results['binary_sensors'].append({
                    'sensor_id': sensor_id,
                    'values': list(unique_values)
                })
            else:
                # Multi-state or other
                results['multi_state_sensors'].append({
                    'sensor_id': sensor_id,
                    'values': list(unique_values),
                    'unique_count': len(unique_values)
                })
            
            # Store value distribution
            value_counts = sensor_data[value_col].value_counts()
            for val, count in value_counts.items():
                results['value_distribution'][sensor_id].append({
                    'value': val,
                    'count': int(count),
                    'percentage': float(count / len(sensor_data) * 100)
                })
        
        return results
    
    def analyze_dataset(self, dataset_name, df):
        """Perform comprehensive analysis on a dataset."""
        if df is None or len(df) == 0:
            print(colored(f"Skipping analysis for {dataset_name} (no data)", 'yellow'))
            return None
        
        print(colored(f"\nAnalyzing {dataset_name}...", 'cyan'))
        
        results = {
            'dataset_name': dataset_name,
            'total_rows': len(df),
            'date_range': None,
            'sensors': {},
            'labels': {},
            'sensor_value_analysis': {},
            'sensor_type_classification': {}
        }
        
        # Determine column names (datasets have different schemas)
        if 'sensor_id' in df.columns:
            sensor_col = 'sensor_id'
            value_col = 'sensor_status'
        elif 'Sensor' in df.columns:
            sensor_col = 'Sensor'
            value_col = 'Status'
        else:
            # For marble or other datasets with different schema
            sensor_cols = [c for c in df.columns if 'sensor' in c.lower()]
            if sensor_cols:
                sensor_col = sensor_cols[0]
                # Find value column
                value_cols = [c for c in df.columns if 'status' in c.lower() or 'value' in c.lower()]
                value_col = value_cols[0] if value_cols else df.columns[1]
            else:
                print(colored(f"Could not identify sensor columns in {dataset_name}", 'yellow'))
                return results
        
        # Date range
        date_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
        if date_cols:
            try:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                results['date_range'] = {
                    'start': str(df[date_cols[0]].min()),
                    'end': str(df[date_cols[0]].max()),
                    'duration_days': (df[date_cols[0]].max() - df[date_cols[0]].min()).days
                }
            except:
                pass
        
        # Sensor analysis
        unique_sensors = df[sensor_col].unique()
        results['sensors']['total_count'] = len(unique_sensors)
        results['sensors']['sensor_list'] = list(unique_sensors)
        
        # Classify each sensor
        sensor_types = defaultdict(list)
        for sensor_id in unique_sensors:
            sensor_type = self.classify_sensor_type(sensor_id)
            sensor_types[sensor_type].append(sensor_id)
        
        results['sensor_type_classification'] = {
            k: {'count': len(v), 'sensors': v}
            for k, v in sensor_types.items()
        }
        
        # Analyze sensor values
        results['sensor_value_analysis'] = self.analyze_sensor_values(df, sensor_col, value_col)
        
        # Label analysis
        label_cols = [c for c in df.columns if 'label' in c.lower() or 'activity' in c.lower()]
        if label_cols:
            label_col = label_cols[0]
            unique_labels = df[label_col].unique()
            label_counts = df[label_col].value_counts()
            
            results['labels'] = {
                'total_count': len(unique_labels),
                'label_list': list(unique_labels),
                'label_distribution': {
                    str(label): int(count)
                    for label, count in label_counts.items()
                }
            }
        
        # Sensor activation statistics
        sensor_activation_counts = df[sensor_col].value_counts()
        results['sensors']['activation_stats'] = {
            'most_active': {
                'sensor_id': str(sensor_activation_counts.index[0]),
                'count': int(sensor_activation_counts.iloc[0])
            },
            'least_active': {
                'sensor_id': str(sensor_activation_counts.index[-1]),
                'count': int(sensor_activation_counts.iloc[-1])
            },
            'mean_activations': float(sensor_activation_counts.mean()),
            'median_activations': float(sensor_activation_counts.median())
        }
        
        return results
    
    def generate_visualizations(self, dataset_name, results):
        """Generate visualization charts for the dataset."""
        if results is None:
            return
        
        print(colored(f"Generating visualizations for {dataset_name}...", 'cyan'))
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Sensor type distribution
        ax1 = plt.subplot(2, 3, 1)
        sensor_types = results['sensor_type_classification']
        type_names = list(sensor_types.keys())
        type_counts = [sensor_types[t]['count'] for t in type_names]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(type_names)))
        ax1.pie(type_counts, labels=type_names, autopct='%1.1f%%', colors=colors)
        ax1.set_title(f'Sensor Types Distribution - {dataset_name}', fontsize=12, fontweight='bold')
        
        # 2. Sensor value types
        ax2 = plt.subplot(2, 3, 2)
        value_analysis = results['sensor_value_analysis']
        value_type_counts = {
            'Binary': len(value_analysis.get('binary_sensors', [])),
            'Numeric': len(value_analysis.get('numeric_sensors', [])),
            'Multi-state': len(value_analysis.get('multi_state_sensors', []))
        }
        
        bars = ax2.bar(value_type_counts.keys(), value_type_counts.values(), color=['#66c2a5', '#fc8d62', '#8da0cb'])
        ax2.set_title(f'Sensor Value Types - {dataset_name}', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Sensors')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # 3. Label distribution (if available)
        if results.get('labels') and results['labels'].get('label_distribution'):
            ax3 = plt.subplot(2, 3, 3)
            labels_dist = results['labels']['label_distribution']
            
            # Sort by count
            sorted_labels = sorted(labels_dist.items(), key=lambda x: x[1], reverse=True)
            labels_names = [l[0][:20] for l in sorted_labels[:15]]  # Top 15, truncate names
            labels_counts = [l[1] for l in sorted_labels[:15]]
            
            ax3.barh(range(len(labels_names)), labels_counts, color='#e78ac3')
            ax3.set_yticks(range(len(labels_names)))
            ax3.set_yticklabels(labels_names, fontsize=8)
            ax3.set_xlabel('Count')
            ax3.set_title(f'Activity Label Distribution (Top 15) - {dataset_name}', fontsize=12, fontweight='bold')
            ax3.invert_yaxis()
        
        # 4. Sensor type breakdown with counts
        ax4 = plt.subplot(2, 3, 4)
        sensor_types_sorted = sorted(sensor_types.items(), key=lambda x: x[1]['count'], reverse=True)
        type_names_sorted = [t[0] for t in sensor_types_sorted]
        type_counts_sorted = [t[1]['count'] for t in sensor_types_sorted]
        
        bars = ax4.barh(range(len(type_names_sorted)), type_counts_sorted, color='#a6d854')
        ax4.set_yticks(range(len(type_names_sorted)))
        ax4.set_yticklabels(type_names_sorted, fontsize=9)
        ax4.set_xlabel('Number of Sensors')
        ax4.set_title(f'Sensor Types Breakdown - {dataset_name}', fontsize=12, fontweight='bold')
        ax4.invert_yaxis()
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, type_counts_sorted)):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(count)}',
                    ha='left', va='center', fontsize=8, fontweight='bold')
        
        # 5. Dataset summary statistics
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        summary_text = f"""
        Dataset: {dataset_name.upper()}
        
        Total Rows: {results['total_rows']:,}
        Total Sensors: {results['sensors']['total_count']}
        Total Labels: {results['labels'].get('total_count', 'N/A')}
        
        Date Range:
        """
        
        if results['date_range']:
            summary_text += f"""  Start: {results['date_range']['start'][:10]}
          End: {results['date_range']['end'][:10]}
          Duration: {results['date_range']['duration_days']} days
        """
        
        summary_text += f"""
        Sensor Activations:
          Most Active: {results['sensors']['activation_stats']['most_active']['sensor_id']}
            ({results['sensors']['activation_stats']['most_active']['count']:,} times)
          Mean: {results['sensors']['activation_stats']['mean_activations']:.1f}
          Median: {results['sensors']['activation_stats']['median_activations']:.1f}
        """
        
        ax5.text(0.1, 0.9, summary_text, fontsize=9, verticalalignment='top', 
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 6. Numeric sensor ranges (if any)
        ax6 = plt.subplot(2, 3, 6)
        numeric_sensors = value_analysis.get('numeric_sensors', [])
        
        if numeric_sensors:
            sensor_ids = [s['sensor_id'][:15] for s in numeric_sensors[:10]]  # Top 10
            mins = [s['min'] for s in numeric_sensors[:10]]
            maxs = [s['max'] for s in numeric_sensors[:10]]
            means = [s['mean'] for s in numeric_sensors[:10]]
            
            x = np.arange(len(sensor_ids))
            width = 0.25
            
            ax6.bar(x - width, mins, width, label='Min', color='#b3e2cd')
            ax6.bar(x, means, width, label='Mean', color='#fdcdac')
            ax6.bar(x + width, maxs, width, label='Max', color='#cbd5e8')
            
            ax6.set_ylabel('Value')
            ax6.set_title(f'Numeric Sensor Ranges - {dataset_name}', fontsize=12, fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels(sensor_ids, rotation=45, ha='right', fontsize=8)
            ax6.legend()
        else:
            ax6.text(0.5, 0.5, 'No numeric sensors found', 
                    ha='center', va='center', fontsize=12)
            ax6.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / f'{dataset_name}_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(colored(f"✓ Saved visualization to {output_file}", 'green'))
    
    def generate_report(self):
        """Generate a comprehensive text report."""
        report_file = self.output_dir / 'dataset_analysis_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SMART HOME DATASET ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            for dataset_name, results in self.analysis_results.items():
                if results is None:
                    continue
                
                f.write(f"\n{'='*80}\n")
                f.write(f"DATASET: {dataset_name.upper()}\n")
                f.write(f"{'='*80}\n\n")
                
                # Basic statistics
                f.write(f"Basic Statistics:\n")
                f.write(f"  Total sensor readings: {results['total_rows']:,}\n")
                f.write(f"  Number of unique sensors: {results['sensors']['total_count']}\n")
                f.write(f"  Number of activity labels: {results['labels'].get('total_count', 'N/A')}\n")
                
                if results['date_range']:
                    f.write(f"  Date range: {results['date_range']['start'][:10]} to {results['date_range']['end'][:10]}\n")
                    f.write(f"  Duration: {results['date_range']['duration_days']} days\n")
                
                # Sensor types
                f.write(f"\nSensor Type Classification:\n")
                sensor_types = results['sensor_type_classification']
                for sensor_type, info in sorted(sensor_types.items(), key=lambda x: x[1]['count'], reverse=True):
                    f.write(f"  {sensor_type.capitalize()}: {info['count']} sensors\n")
                    f.write(f"    Examples: {', '.join(info['sensors'][:5])}\n")
                
                # Sensor value types
                f.write(f"\nSensor Value Types:\n")
                value_analysis = results['sensor_value_analysis']
                
                binary_count = len(value_analysis.get('binary_sensors', []))
                f.write(f"  Binary sensors (ON/OFF, OPEN/CLOSE): {binary_count}\n")
                if binary_count > 0:
                    f.write(f"    Examples: {', '.join([s['sensor_id'] for s in value_analysis['binary_sensors'][:5]])}\n")
                
                numeric_count = len(value_analysis.get('numeric_sensors', []))
                f.write(f"  Numeric sensors: {numeric_count}\n")
                if numeric_count > 0:
                    f.write(f"    Examples: {', '.join([s['sensor_id'] for s in value_analysis['numeric_sensors'][:5]])}\n")
                    for sensor in value_analysis['numeric_sensors'][:3]:
                        f.write(f"      {sensor['sensor_id']}: range [{sensor['min']:.2f}, {sensor['max']:.2f}], mean={sensor['mean']:.2f}\n")
                
                multi_count = len(value_analysis.get('multi_state_sensors', []))
                f.write(f"  Multi-state sensors: {multi_count}\n")
                if multi_count > 0:
                    f.write(f"    Examples: {', '.join([s['sensor_id'] for s in value_analysis['multi_state_sensors'][:5]])}\n")
                
                # Activity labels
                if results.get('labels') and results['labels'].get('label_distribution'):
                    f.write(f"\nActivity Labels (Top 10):\n")
                    labels_dist = results['labels']['label_distribution']
                    sorted_labels = sorted(labels_dist.items(), key=lambda x: x[1], reverse=True)
                    for label, count in sorted_labels[:10]:
                        percentage = (count / results['total_rows']) * 100
                        f.write(f"  {label}: {count:,} ({percentage:.1f}%)\n")
                
                # Sensor activation statistics
                f.write(f"\nSensor Activation Statistics:\n")
                stats = results['sensors']['activation_stats']
                f.write(f"  Most active sensor: {stats['most_active']['sensor_id']} ({stats['most_active']['count']:,} activations)\n")
                f.write(f"  Least active sensor: {stats['least_active']['sensor_id']} ({stats['least_active']['count']:,} activations)\n")
                f.write(f"  Mean activations per sensor: {stats['mean_activations']:.1f}\n")
                f.write(f"  Median activations per sensor: {stats['median_activations']:.1f}\n")
                
                f.write("\n")
        
        print(colored(f"\n✓ Report saved to {report_file}", 'green'))
    
    def save_sensor_type_mapping(self):
        """Save sensor type classification as JSON for future use."""
        mapping_file = self.output_dir / 'sensor_type_mapping.json'
        
        sensor_mapping = {}
        for dataset_name, results in self.analysis_results.items():
            if results is None:
                continue
            
            dataset_mapping = {}
            sensor_types = results['sensor_type_classification']
            
            for sensor_type, info in sensor_types.items():
                for sensor_id in info['sensors']:
                    dataset_mapping[sensor_id] = sensor_type
            
            sensor_mapping[dataset_name] = dataset_mapping
        
        with open(mapping_file, 'w') as f:
            json.dump(sensor_mapping, f, indent=2)
        
        print(colored(f"✓ Sensor type mapping saved to {mapping_file}", 'green'))
    
    def generate_comparison_visualization(self):
        """Generate a comparison visualization across all datasets."""
        print(colored("\nGenerating comparison visualization...", 'cyan'))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data for comparison
        dataset_names = []
        total_sensors = []
        total_labels = []
        total_rows = []
        sensor_type_data = defaultdict(list)
        
        for dataset_name, results in self.analysis_results.items():
            if results is None:
                continue
            
            dataset_names.append(dataset_name)
            total_sensors.append(results['sensors']['total_count'])
            total_labels.append(results['labels'].get('total_count', 0))
            total_rows.append(results['total_rows'])
            
            # Collect sensor type counts
            for sensor_type, info in results['sensor_type_classification'].items():
                sensor_type_data[sensor_type].append(info['count'])
        
        # 1. Total sensors comparison
        ax1 = axes[0, 0]
        bars = ax1.bar(dataset_names, total_sensors, color='#8dd3c7')
        ax1.set_title('Total Sensors by Dataset', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Sensors')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # 2. Total labels comparison
        ax2 = axes[0, 1]
        bars = ax2.bar(dataset_names, total_labels, color='#fdb462')
        ax2.set_title('Total Activity Labels by Dataset', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Labels')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
        
        # 3. Dataset sizes
        ax3 = axes[1, 0]
        bars = ax3.bar(dataset_names, total_rows, color='#bebada')
        ax3.set_title('Dataset Size (Total Readings)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Readings')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_yscale('log')
        
        for bar, rows in zip(bars, total_rows):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rows:,}',
                    ha='center', va='bottom', fontsize=8)
        
        # 4. Sensor type distribution heatmap
        ax4 = axes[1, 1]
        
        # Create matrix for heatmap
        sensor_types = list(sensor_type_data.keys())
        matrix_data = []
        
        for sensor_type in sensor_types:
            row = []
            for dataset_name in dataset_names:
                # Find count for this sensor type in this dataset
                results = self.analysis_results[dataset_name]
                if results and sensor_type in results['sensor_type_classification']:
                    row.append(results['sensor_type_classification'][sensor_type]['count'])
                else:
                    row.append(0)
            matrix_data.append(row)
        
        if matrix_data:
            sns.heatmap(matrix_data, annot=True, fmt='d', cmap='YlOrRd',
                       xticklabels=dataset_names, yticklabels=sensor_types,
                       ax=ax4, cbar_kws={'label': 'Count'})
            ax4.set_title('Sensor Type Distribution by Dataset', fontsize=12, fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / 'dataset_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(colored(f"✓ Saved comparison visualization to {output_file}", 'green'))
    
    def run_analysis(self, force_download=False):
        """Run complete analysis pipeline."""
        print(colored("\n" + "="*80, 'cyan'))
        print(colored("SMART HOME DATASET ANALYZER", 'cyan', attrs=['bold']))
        print(colored("="*80 + "\n", 'cyan'))
        
        # Analyze each dataset
        for dataset_name, dataset_type in self.datasets.items():
            try:
                if dataset_type == 'casas':
                    df = self.load_casas_dataset(dataset_name, force_download)
                elif dataset_type == 'marble':
                    df = self.load_marble_dataset()
                else:
                    continue
                
                if df is not None:
                    results = self.analyze_dataset(dataset_name, df)
                    self.analysis_results[dataset_name] = results
                    
                    if results:
                        self.generate_visualizations(dataset_name, results)
            
            except Exception as e:
                print(colored(f"Error processing {dataset_name}: {e}", 'red'))
                import traceback
                traceback.print_exc()
        
        # Generate cross-dataset comparisons and reports
        if self.analysis_results:
            self.generate_comparison_visualization()
            self.generate_report()
            self.save_sensor_type_mapping()
        
        print(colored("\n" + "="*80, 'green'))
        print(colored("✓ ANALYSIS COMPLETE!", 'green', attrs=['bold']))
        print(colored(f"✓ Results saved to: {self.output_dir}", 'green'))
        print(colored("="*80 + "\n", 'green'))


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze smart home datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--force_download',
        action='store_true',
        help='Force re-download of datasets'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/eda/datasources',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = DatasetAnalyzer(output_dir=args.output_dir)
    analyzer.run_analysis(force_download=args.force_download)


if __name__ == '__main__':
    main()

