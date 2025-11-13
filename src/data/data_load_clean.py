""" Functions to download and process raw CASAS data.

All relevant metadata is stored in `metadata/casas_metadata.json` file.

Main function:
- casas_end_to_end_preprocess(city_name: str)

Helper functions are:
- download_and_unzip()
- read_and_process_data()
- process_raw_data()

The module contains several helper functions that start with _ (underscore).

Print color: `magenta`

Example usage:
  ```python
  from data_process import casas_end_to_end_preprocess

  df_aruba = casas_end_to_end_preprocess("aruba")
  ```
"""

import os
import shutil
from pyparsing import col
import requests
import zipfile
import pandas as pd
import numpy as np
import datetime
from io import StringIO
from termcolor import colored
import json
import argparse
from sklearn.model_selection import train_test_split


def _get_json_metadata(file_path='metadata/casas_metadata.json'):
  with open(file_path, 'r') as file:
    metadata = json.load(file)
  return metadata
CASAS_METADATA = _get_json_metadata()


def download_and_unzip(
    url='https://casas.wsu.edu/datasets/tulum.zip',
    delete_zip=True
):
  """Download a zip file from a URL and extract its contents.

  Args:
    url (str): The URL to download the zip file from.
    delete_zip (bool):
      Whether to delete the zip file after extracting its contents.
  """
  # Store raw CASAS archives under unified raw directory
  data_dir = 'data/raw/casas'
  os.makedirs(data_dir, exist_ok=True)

  filename = url.split('/')[-1]
  zip_path = os.path.join(data_dir, filename)
  flag_path = os.path.join(data_dir, f"{filename}.downloaded")

  # Check if the dataset was already downloaded by looking for the flag file
  if os.path.exists(flag_path):
    print(colored(f"{filename} was already downloaded.", 'magenta'))
    if not os.path.exists(zip_path):
      print(colored("Zip file does not exist, assuming contents are already extracted.", 'magenta'))
      return
  else:
    # Download the zip file
    print(colored(f"Downloading {filename}...", 'magenta'))
    response = requests.get(url)
    with open(zip_path, 'wb') as zip_file:
      zip_file.write(response.content)

    # Unzip the file
    print(colored(f"Unzipping {filename}...", 'magenta'))
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
      # Special case for the Aruba dataset: extract to a subfolder.
      if filename == 'aruba.zip':
        os.makedirs(f'{data_dir}/aruba', exist_ok=True)
        zip_ref.extractall(f'{data_dir}/aruba')
      # Other datasets alredy have a subfolder within the zip file.
      else:
        zip_ref.extractall(data_dir)

    # Assume name of the contents is the same as the zip file name
    unzipped_name = os.path.splitext(filename)[0]
    unzipped_dir = next(
      (dir for dir in os.listdir(data_dir)
       if unzipped_name in dir and not dir.endswith('.zip'))
    )
    # Copy the PNG file to the 'charts/layouts' directory

    png_files = (
      file for file in os.listdir(os.path.join(data_dir, unzipped_dir))
      if file.endswith('.png')
      or file.endswith('.jpg')
      or file.endswith('.jpeg')
    )
    png_file = next(png_files)

    # Special case for the Kyoto dataset image file
    if filename == 'twor.2009.zip':
      png_file = next(png_files) # Skip the first image file
      new_filename = 'kyoto.png'
      new_file_path = os.path.join(data_dir, unzipped_dir, new_filename)
      if not os.path.exists(new_file_path):
        os.rename(os.path.join(data_dir, unzipped_dir, png_file), new_file_path)
      png_file = new_filename

    # Special case for the Cairo dataset image file
    if filename == 'cairo.zip':
      if not os.path.exists(os.path.join(data_dir, unzipped_dir, png_file.lower())):
        os.rename(
          os.path.join(data_dir, unzipped_dir, png_file),
          os.path.join(data_dir, unzipped_dir, png_file.lower())
        )
      png_file = png_file.lower()

    layouts_dir = os.path.join('charts', 'house_layouts')
    os.makedirs(layouts_dir, exist_ok=True)
    shutil.copy(
      os.path.join(data_dir, unzipped_dir, png_file),
      layouts_dir
    )
    print(colored("Layout Image file copied to 'charts/house_layouts' directory.", 'magenta'))

    # Create a flag file to indicate the zip was downloaded and extracted
    with open(flag_path, 'w') as f:
      f.write('Downloaded and extracted.')

  # Optionally delete the zip file
  if delete_zip:
    os.remove(zip_path)
    print(colored(f"Deleted {filename}", 'magenta'))
  else:
    print(colored(f"Downloaded and extracted {filename} without deleting the zip file.", 'magenta'))


def read_and_process_data(
    file_path='data/raw/casas/tulum2009/data.txt',
    columns=['date', 'sensor', 'state', 'activity_full'],
    max_lines=None
):
  """Read and process the data from a text file.

  Args:
    file_path (str): The path to the text file to read.
    columns (list of str): The column names for the data.
    annotate_data (bool): Whether to annotate the data with additional columns.

  Returns:
    pd.DataFrame: The processed data with 3 new columns:
      - datetime (formatted date)
      - activity (should be something like 'Sleeping', 'Showering', etc.)
      - marker (should be 'begin', 'end', or NaN).
  """
  if file_path in ['data/raw/casas/milan/data', 'data/raw/casas/aruba/data', 'data/raw/casas/twor.2009/data']:
    return _read_irregular_data(file_path, max_lines=max_lines)

  df_raw = (pd
    .read_csv(
      file_path,
      delimiter='\t',
      names=columns,
      header=None,
      on_bad_lines='warn'
    )
    .sort_values(by='date')
    .assign(
      datetime=lambda df: pd.to_datetime(df.date, errors='coerce'),
      # Split from the right side, only once
      activity=lambda df: df.activity_full.str.rsplit(' ', expand=True, n=1)[0],
      marker=lambda df: df.activity_full.str.rsplit(' ', expand=True, n=1)[1],
    )
  )
  print(colored(f'Raw data shape: {df_raw.shape}', 'magenta'))

  return df_raw

def _read_irregular_data(
    file_path='data/raw/casas/aruba/data',
    columns=['date', 'time', 'sensor', 'state', 'activity', 'marker'],
    max_lines=None
):
  """Read and process the data from a text file with irregular spacing."""
  with open(file_path, 'r') as file:
      lines = file.read().splitlines()

  total_lines = len(lines)

  # Limit lines for testing if specified
  if max_lines is not None:
      lines = lines[:max_lines]
      print(colored(f'⚠️  LIMITED TO FIRST {max_lines} LINES FOR TESTING (original file had {total_lines} lines)', 'yellow'))

  # Remove extra spaces in each line and format delimiters as single space
  lines = [" ".join(line.split()) for line in lines]

  df_raw = (pd
    .read_csv(
      StringIO('\n'.join(lines)),
      delimiter=' ',
      names=columns,
      header=None,
      on_bad_lines='warn',
    )
    .sort_values(by=['date', 'time'])
    .assign(
      date=lambda df:
        df.date + ' ' + df.time
    )
    .assign(
      date=lambda df: df.date.apply(_add_missing_microseconds),
      datetime=lambda df:
        pd.to_datetime(df.date, errors='coerce'),
      activity_full=lambda df:
        df.activity + ' ' + df.marker,
    )
    .drop(columns=['time']) # is included in the date column
  )
  print(colored(f'Raw data shape: {df_raw.shape}', 'magenta'))

  return df_raw


def process_raw_data(df_raw):
  """Process the raw data into a clean, annotated DataFrame.

  Args:
    df_raw (pd.DataFrame): The raw data to process and annotate.

  Returns:
    pd.DataFrame: The processed data with the following new columns:
      - activity_list: A list of activities that happened at that sensor reading.
      - num_activities: The number of activities for each record.
      - activity_list_str: A string representation of the activity list.
      - first_activity: The first activity in the activity list.
      - sensor_data_type: The type of data for each sensor.
      - is_numeric: Whether the sensor data is numeric data (float).
  """
  # Create a dictionary of sensor types
  sensor_type_dict = _create_sensor_type_dict(df_raw)

  df = (pd
    .DataFrame(
      # Convert the raw DF to a list of dicts and then annotate with activities.
      _annotate_activity_list(df_raw.to_dict('records'))
    )
    .assign(
      num_activities=lambda df: df.activity_list.apply(len),
      activity_list_str=lambda df: df.activity_list.apply(
        lambda x: ', '.join(x)),
      first_activity=lambda df: df.activity_list.apply(
        lambda x: x[0] if x else 'no_activity'),
      sensor_data_type=lambda df: df.sensor.map(sensor_type_dict),
      is_numeric=lambda df: df.sensor_data_type == 'numeric'
    )
  )

  return df

def process_multiresident_home(df, num_residents=2):

  for res_num in range(1, num_residents + 1):
    df[f'activity_list_R{res_num}'] = df.activity_list.apply(lambda x: sorted([w for w in x if w.startswith(f'R{res_num}')]))
    df[f'first_activity_R{res_num}'] = df[f'activity_list_R{res_num}'].apply(lambda x: x[0] if x else 'no_activity')

  return df

def get_df_dated_sample(df, start_date_str, end_date_str=None):
  """Get a sample of the dataframe based on the date range provided.

  If only the start date is provided, the function will return only
  the rows for that date. If both start and end dates are provided, the
  function will return the rows for the range of dates.

  Args:
  - df (pd.DataFrame): The dataframe to sample from.
  - start_date_str (str): The start date in the format 'YYYY-MM-DD'.
  - end_date_str (str) [Optional]: The end date in the format 'YYYY-MM-DD'.

  Returns:
  - df_sample (pd.DataFrame): The sampled dataframe.
  """
  start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()

  if end_date_str:
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
    df_sample = (
        df[
            (df.datetime.dt.date >= start_date) &
            (df.datetime.dt.date <= end_date)
        ]
        .reset_index(drop=True)
    )
  else:
    df_sample = (
        df[
            df.datetime.dt.date == start_date
        ]
        .reset_index(drop=True)
    )
  return df_sample

def casas_end_to_end_preprocess(dataset_name, save_to_csv=True, force_download=False, custom_train_test_split=False, max_lines=None):
  """ Process the CASAS dataset.

  Given the `metadata/casas_metadata.json` file, this function downloads the
  raw data from CASAS website, processes it, and returns a clean DataFrame with
  labels and other annotations.

  If `custom_train_test_split` is set to True, the function will split the data
  into train and test sets and return tow DataFrames instead of one.
  """
  if max_lines is not None:
    print(colored(f'🔍 casas_end_to_end_preprocess called with max_lines={max_lines}', 'cyan'))

  if dataset_name == 'aware_home':
    raise ValueError(
      'Use the `awaerehome_end_to_end_preprocess()` function for the Aware Home dataset.'
    )
  elif dataset_name not in CASAS_METADATA:
    raise ValueError(
      f'Dataset name "{dataset_name}" not in `casas_metadata.json`. Must be one of: {list(CASAS_METADATA.keys())}'
    )
  metadata = CASAS_METADATA.get(dataset_name)

  casas_url = metadata['casas_url']
  data_processed_filename = metadata['data_processed_filename']
  data_raw_filename = metadata['data_raw_filename']
  folder_name = metadata['data_raw_filename'].split('/')[0]

  # Prefer local raw path under data/raw/casas/{folder_name}/ if present
  local_raw_dir = f'data/raw/casas/{folder_name}'
  local_raw_file = f'{local_raw_dir}/data' if folder_name in ['milan', 'aruba', 'twor.2009'] else f'{local_raw_dir}/data.txt'

  # Check if processed data already exists:
  if os.path.exists(f'data/{data_processed_filename}') and not force_download:
    if custom_train_test_split:
      df_train = pd.read_csv(f'data/{folder_name}/data_processed_train.csv')
      df_test = pd.read_csv(f'data/{folder_name}/data_processed_test.csv')

      df_train['datetime'] = pd.to_datetime(df_train['datetime'])
      df_test['datetime'] = pd.to_datetime(df_test['datetime'])
      print(colored(f'Train and test data loaded from CSV. Train shape: {df_train.shape}, Test shape: {df_test.shape}', 'magenta'))

      return df_train, df_test
    else:
      df = pd.read_csv(f'data/{data_processed_filename}')
      df['datetime'] = pd.to_datetime(df['datetime'])
      print(colored(f'Processed data loaded from CSV. Shape: {df.shape}', 'magenta'))

      return df
  # Else, locate local raw data or download as fallback
  if os.path.exists(local_raw_file):
    print(colored(f'Using local raw data at {local_raw_file}', 'magenta'))
    df_raw = read_and_process_data(file_path=local_raw_file, max_lines=max_lines)
  else:
    download_and_unzip(casas_url)
    df_raw = read_and_process_data(file_path=f'data/raw/casas/{data_raw_filename}', max_lines=max_lines)
  df = process_raw_data(df_raw)
  # df = tulum_specific_preprocess(df)

  df['first_activity_l2'] = df.first_activity.map(metadata['label_l2'])
  # Add a column with sensor locations
  df['sensor_location'] = df.sensor.map(metadata['sensor_location'])
  # Add a column with sensor types
  df['sensor_type'] = df.sensor.map(metadata.get('sensor_type', {}))

  if metadata['num_residents'] > 1:
    df = process_multiresident_home(df)

  # Tulum specific fix
  if dataset_name == 'tulum':
    # Fix the 'OFF5' state to 'OFF'
    df['state'] = df.state.replace('OFF5', 'OFF')

  # Apply max_lines limit after processing if specified
  if max_lines is not None and len(df) > max_lines:
    print(colored(f'Limiting processed data to first {max_lines} rows (was {len(df)})', 'yellow'))
    df = df.head(max_lines).copy()

  print(colored(f'Processed data loaded. Shape: {df.shape}', 'magenta'))
  print(colored(f'Data date range: {df["datetime"].min()} to {df["datetime"].max()}', 'cyan'))
  # Save the data to a CSV file:
  if save_to_csv:
    # Save processed under data/processed to keep outputs small and consistent
    processed_out_path = f'data/{data_processed_filename}'
    os.makedirs(os.path.dirname(processed_out_path), exist_ok=True)
    df.to_csv(processed_out_path, index=False)
    print(colored(f'Processed data saved to "{processed_out_path}"', 'magenta'))

  # Custom train-test split for Milan and Aruba datasets and save to disk.
  if custom_train_test_split:
    _, _ = _custom_casas_train_test_split(
        df, dataset_name, save_to_disk=save_to_csv)
  return df

def awaerehome_end_to_end_preprocess(save_to_csv=False):
  """ Process the Aware Home dataset.

  Current implementation expects the data to be in 'data/aware_home/data.csv'
  and have the following columns:
    [
      'date', 'sensor', 'state', 'sensor_type',
      'sensor_location', 'spec_location', 'activity', 'status'
    ]

  The output data format is the same as CASAS datasets, with some additional
  columns from the original csv file.
  """
  df_raw = pd.read_csv('data/aware_home/data.csv')

  df_raw = (
    df_raw
    .assign(
      datetime=lambda df: pd.to_datetime(df.stamp),
      state=lambda df: df.value.map({1: 'ON', 0: 'OFF'}),
      marker = lambda df: df.status
    )
    .rename(columns={
      'itemname': 'sensor',
    })
  )

  df = process_raw_data(df_raw)

  print(colored(f'Processed data loaded. Shape: {df.shape}', 'magenta'))
  # Save the data to a CSV file:
  if save_to_csv:
    df.to_csv('data/aware_home/data_processed.csv', index=False)
    print(colored('Processed data saved to "data/aware_home/data_processed.csv"', 'magenta'))
  return df

def _check_prev_marker_dict(prev_marker_dict):
  """Check if any values in the dict are equal to `begin`, return list with those activities

  Args:
    prev_marker_dict: dict of form:
      {
        'Cook_Breakfast': 'end',
        'Eat_Lunch': 'begin',
        'Watch_TV': end,
        ...
      }

  Returns:
    List of activities that have `begin` as value. E.g.:
      ['Eat_Lunch']
  """

  output_list = []

  for activity, marker in prev_marker_dict.items():
    if marker == 'begin':
      output_list.append(activity)

  return sorted(output_list)


def _annotate_activity_list(data):
  """ Annotate each timestamp with a list of activities in between `begin`
      and `end` markers.

  Args:
    data: list of dicts
      Must have the following fields:
      - `activity`: ['Activity_1', 'Activity_2', ..., np.nan]
      - `marker`: ['begin', 'end', np.nan]

  Returns:
    data with an additional field `activity_list`
  """
  unique_activities_list = list(
    {rec['activity'] for rec in data if rec['activity'] is not np.nan}
  )
  # Create a dict of all activities annotated with the default `end` value.
  prev_marker_dict = {key: 'end' for key in unique_activities_list}

  for record in data:
    current_activity = record['activity']
    current_marker = record['marker']
    if current_marker == 'begin':
      # If this marker is begin, update the dictionary.
      prev_marker_dict[current_activity] = current_marker
      # Extract all keys in dict that have `begin` recorded
      record['activity_list'] = _check_prev_marker_dict(prev_marker_dict)

    # If marker is other than `begin`, check the dictionary if it has any `begin`
    # values from previous records. If not, this will create an empty list.
    record['activity_list'] = _check_prev_marker_dict(prev_marker_dict)

    # If marker is end, update the dictionary, but only after that activity was
    # added to the `activity_list`
    if current_marker == 'end':
      prev_marker_dict[current_activity] = current_marker

  return data


def _custom_casas_train_test_split(df, city, test_size=0.2, split_type='daily', save_to_disk=False):
  """Custom temporal train-test split for CASAS datasets.

  The split is based on the `datetime` column, with the test set containing
  the last `test_size` fraction of the data.

  split_type: `temporal` or `daily`:
    - temporal: test set is contained in the last fraction of the data.
    - daily:    train and test set are split based on the date, but days are
                shuffled randomly.
  """
  df['date_col'] = df['datetime'].dt.date

  num_unique_dates = len(df['date_col'].unique())

  if split_type == 'temporal':
    train_split_date_idx = int(num_unique_dates * (1 - test_size))
    train_dates = sorted(df.date_col.unique())[:train_split_date_idx]
    test_dates = sorted(df.date_col.unique())[train_split_date_idx:]
    # Get the row indices for train and validation sets.
    train_ids = df[df.date_col.isin(train_dates)].index
    test_ids = df[df.date_col.isin(test_dates)].index
  elif split_type == 'daily':
    # Shuffle the unique dates and split them into train and test sets.
    unique_dates = np.random.permutation(df.date_col.unique())
    train_dates = unique_dates[:int(num_unique_dates * (1 - test_size))]
    test_dates = unique_dates[int(num_unique_dates * (1 - test_size)):]
    train_ids = df[df.date_col.isin(train_dates)].index
    test_ids = df[df.date_col.isin(test_dates)].index

  df_train = df.loc[train_ids].reset_index(drop=True)
  df_test = df.loc[test_ids].reset_index(drop=True)

  if save_to_disk:
    folder_name = CASAS_METADATA[city]['data_raw_filename'].split('/')[0]
    df_train.to_csv(f'data/{folder_name}/data_processed_train.csv', index=False)
    df_test.to_csv(f'data/{folder_name}/data_processed_test.csv', index=False)

  return df_train, df_test


def _check_reading_type(readings):
  """Check if the readings are all numeric or all strings.

  Args:
    readings (pd.Series): The readings to check.

  Returns:
    str: 'numeric' if all readings are floats, 'str' otherwise.
  """
  all_strings = True
  for reading in readings:
    try:
      # Attempt to convert reading to a float
      float(reading)
      # If conversion is successful, it's a numeric reading
      all_strings = False
      break
    except ValueError:
      # If conversion fails, continue checking
      continue
  return 'str' if all_strings else 'numeric'

def _create_sensor_type_dict(
  df,
  sensor_column='sensor',
  state_column='state'
):
  """Create a dictionary of state data types that are per sensor.

  Args:
    df (pd.DataFrame): The data to process.
    sensor_column (str): The name of the column with sensor names.
    state_column (str): The name of the column with state data.

  Returns:
    dict: A dictionary with sensor names as keys and state data types as values.

  Sample output:
    {
      'M001': 'str',
      'M002': 'str',
      'T001': 'numeric',
      ...
    }
  """
  sensor_type_dict = (df
    .groupby([sensor_column])
    .agg({state_column: _check_reading_type})
    .to_dict()
      [state_column]
  )
  return sensor_type_dict


def _add_missing_microseconds(date_str):
  if '.' not in date_str:
    return date_str + '.000000'  # Add .000000 if no microseconds are present
  return date_str


def process_marble_environmental_data(
    base_path='data/raw/marble/MARBLE/dataset',
    output_path='data/processed/marble/marble_environment_single_resident.csv',
    save_to_csv=True
):
  """Process MARBLE dataset environmental sensors and labels for single resident scenarios.

  Processes only single resident scenarios (A1*, B1*, C1*, D1*) and merges
  environmental.csv with labels.csv from each subject folder.

  Args:
    base_path (str): Base path to MARBLE dataset directory.
    output_path (str): Path to save the processed CSV file.
    save_to_csv (bool): Whether to save the processed data to CSV.

  Returns:
    pd.DataFrame: Processed dataframe with columns:
      - scenario: Scenario ID (A1, B1, C1, D1)
      - time_of_day: Time period code (m=morning, a=afternoon, e=evening)
      - instance: Instance number
      - subject: Subject ID
      - sensor_id: Environmental sensor ID
      - sensor_status: Sensor status (ON/OFF)
      - ts: Timestamp (UNIX milliseconds)
      - subject_id: Subject ID from environmental.csv (may be same as subject)
      - activity: Activity label from labels.csv (None if no activity at that time)
      - activity_start: Start timestamp of activity interval
      - activity_end: End timestamp of activity interval
  """
  import glob

  print(colored("Processing MARBLE dataset (single resident scenarios)...", 'magenta'))

  all_data = []

  # Process only single resident scenarios: A1, B1, C1, D1
  single_resident_scenarios = ['A1', 'B1', 'C1', 'D1']

  for scenario_prefix in single_resident_scenarios:
    # Find all scenario folders matching the pattern (e.g., A1a, A1m, A1e)
    scenario_pattern = os.path.join(base_path, f'{scenario_prefix}*')
    scenario_dirs = glob.glob(scenario_pattern)

    for scenario_dir in scenario_dirs:
      scenario_name = os.path.basename(scenario_dir)
      # Extract scenario (A1, B1, etc.) and time_of_day (m, a, e)
      scenario = scenario_name[:2]  # A1, B1, C1, D1
      time_of_day = scenario_name[2] if len(scenario_name) > 2 else None

      if time_of_day not in ['m', 'a', 'e']:
        print(colored(f"⚠️  Skipping scenario {scenario_name} (unknown time_of_day)", 'yellow'))
        continue

      print(colored(f"Processing scenario: {scenario_name}", 'cyan'))

      # Find all instance folders
      instance_dirs = glob.glob(os.path.join(scenario_dir, 'instance*'))

      for instance_dir in instance_dirs:
        instance_name = os.path.basename(instance_dir)
        # Extract instance number (e.g., "instance1" -> "1")
        instance_num = instance_name.replace('instance', '')

        # Check for environmental.csv
        env_file = os.path.join(instance_dir, 'environmental.csv')
        if not os.path.exists(env_file):
          print(colored(f"⚠️  environmental.csv not found in {instance_dir}", 'yellow'))
          continue

        # Read environmental.csv
        try:
          df_env = pd.read_csv(env_file)
          if df_env.empty:
            continue
          # Convert subject_id to numeric for consistent comparison
          df_env['subject_id'] = pd.to_numeric(df_env['subject_id'], errors='coerce')
        except Exception as e:
          print(colored(f"⚠️  Error reading {env_file}: {e}", 'yellow'))
          continue

        # Find all subject folders
        subject_dirs = glob.glob(os.path.join(instance_dir, 'subject-*'))

        for subject_dir in subject_dirs:
          subject_id_str = os.path.basename(subject_dir).replace('subject-', '')
          # Convert to int to match the type in environmental.csv
          try:
            subject_id = int(subject_id_str)
          except ValueError:
            # If conversion fails, try string comparison
            subject_id = subject_id_str

          # Check for labels.csv
          labels_file = os.path.join(subject_dir, 'labels.csv')
          if not os.path.exists(labels_file):
            print(colored(f"⚠️  labels.csv not found in {subject_dir}", 'yellow'))
            continue

          # Read labels.csv
          try:
            df_labels = pd.read_csv(labels_file)
            if df_labels.empty:
              continue
          except Exception as e:
            print(colored(f"⚠️  Error reading {labels_file}: {e}", 'yellow'))
            continue

          # Filter environmental data for this subject
          df_env_subject = df_env[df_env['subject_id'] == subject_id].copy()

          if df_env_subject.empty:
            continue

          # Merge environmental data with labels based on timestamps
          # For each environmental event, find the activity that was active at that time
          df_env_subject['activity'] = None
          df_env_subject['activity_start'] = None
          df_env_subject['activity_end'] = None

          for idx, row in df_env_subject.iterrows():
            event_ts = row['ts']
            # Find activity that was active at this timestamp
            # Activity is active if ts_start <= event_ts <= ts_end
            active_activities = df_labels[
              (df_labels['ts_start'] <= event_ts) &
              (df_labels['ts_end'] >= event_ts)
            ]

            if not active_activities.empty:
              # Take the first matching activity (should be only one)
              activity_row = active_activities.iloc[0]
              df_env_subject.at[idx, 'activity'] = activity_row['act']
              df_env_subject.at[idx, 'activity_start'] = activity_row['ts_start']
              df_env_subject.at[idx, 'activity_end'] = activity_row['ts_end']

          # Add metadata columns
          df_env_subject['scenario'] = scenario
          df_env_subject['time_of_day'] = time_of_day
          df_env_subject['instance'] = instance_num
          df_env_subject['subject'] = subject_id

          # Reorder columns
          column_order = [
            'scenario', 'time_of_day', 'instance', 'subject',
            'sensor_id', 'sensor_status', 'ts', 'subject_id',
            'activity', 'activity_start', 'activity_end'
          ]
          # Only include columns that exist
          column_order = [col for col in column_order if col in df_env_subject.columns]
          df_env_subject = df_env_subject[column_order]

          all_data.append(df_env_subject)

          print(colored(
            f"  ✓ Processed {scenario_name}/{instance_name}/subject-{subject_id}: "
            f"{len(df_env_subject)} environmental events", 'green'
          ))

  if not all_data:
    print(colored("⚠️  No data found to process", 'yellow'))
    return pd.DataFrame()

  # Concatenate all data
  df_combined = pd.concat(all_data, ignore_index=True)

  # Add sensor type mapping from MARBLE metadata
  marble_metadata_path = 'metadata/marble_metadata.json'
  if os.path.exists(marble_metadata_path):
    with open(marble_metadata_path, 'r') as f:
      marble_metadata = json.load(f)
    sensor_type_map = marble_metadata.get('marble', {}).get('sensor_type', {})
    df_combined['sensor_type'] = df_combined['sensor_id'].map(sensor_type_map)

  print(colored(f"Processed data shape: {df_combined.shape}", 'magenta'))
  print(colored(
    f"Scenarios: {df_combined['scenario'].unique()}, "
    f"Instances: {df_combined['instance'].nunique()}, "
    f"Subjects: {df_combined['subject'].nunique()}", 'cyan'
  ))

  # Save to CSV
  if save_to_csv:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    print(colored(f"Processed data saved to '{output_path}'", 'magenta'))

  return df_combined


def process_marble_to_trainable_data(
    csv_path='data/processed/marble/marble_environment_single_resident.csv',
    window_length_sec=16.0,
    overlap_fraction=0.5,
    power_threshold=10.0,
    temperature_bins=5,
    train_split=0.8,
    output_dir='data/processed/marble',
    save_to_json=True
):
  """Process MARBLE environmental CSV to trainable data following ADL-LLM preprocessing.

  This function implements the complete ADL-LLM preprocessing pipeline:
  1. Build event stream (ON/OFF events from sensors)
  2. Convert events to sensor states
  3. Create fixed-length sliding windows
  4. Classify states inside windows
  5. Assign activity labels to windows
  6. Split into train/test sets

  Args:
    csv_path (str): Path to the input CSV file.
    window_length_sec (float): Window length in seconds (τ).
    overlap_fraction (float): Overlap fraction (o) between windows.
    power_threshold (float): Threshold for power sensors (above = ON, below = OFF).
    temperature_bins (int): Number of bins for temperature discretization.
    train_split (float): Fraction of data for training (default 0.8).
    output_dir (str): Directory to save output files.
    save_to_json (bool): Whether to save processed data to JSON.

  Returns:
    tuple: (df_train, df_test) DataFrames with windowed data.
  """
  print(colored("Processing MARBLE data for ADL-LLM preprocessing...", 'magenta'))

  # Step 1: Check if CSV exists, if not, try to generate it
  if not os.path.exists(csv_path):
    print(colored(f"CSV file not found at {csv_path}", 'yellow'))
    print(colored("Attempting to generate it from raw MARBLE data...", 'cyan'))

    # Default base path for raw MARBLE data
    default_base_path = 'data/raw/marble/MARBLE/dataset'

    # Check if raw data exists
    if not os.path.exists(default_base_path):
      print(colored("\n" + "="*80, 'red'))
      print(colored("ERROR: Raw MARBLE dataset not found!", 'red'))
      print(colored("="*80, 'red'))
      print(colored("\nPlease download and extract the MARBLE dataset first:", 'yellow'))
      print(colored(f"  1. Download MARBLE.rar from:", 'yellow'))
      print(colored("     https://dataverse.unimi.it/dataset.xhtml?persistentId=doi:10.13130/RD_UNIMI/VGLD0Y", 'cyan'))
      print(colored(f"  2. Extract the archive to: {os.path.dirname(default_base_path)}", 'yellow'))
      print(colored(f"  3. Expected structure: {default_base_path}/", 'yellow'))
      print(colored("     (should contain scenario folders like A1a, A1m, A1e, etc.)", 'yellow'))
      print(colored("\nAfter extraction, run this command again.", 'yellow'))
      print(colored("="*80 + "\n", 'red'))
      raise FileNotFoundError(
        f"Raw MARBLE dataset not found at {default_base_path}. "
        f"Please download and extract MARBLE.rar first."
      )

    # Try to generate the CSV
    try:
      print(colored(f"Processing raw MARBLE data from {default_base_path}...", 'cyan'))
      df_env = process_marble_environmental_data(
        base_path=default_base_path,
        output_path=csv_path,
        save_to_csv=True
      )
      if df_env.empty:
        raise ValueError("Generated CSV file is empty. Please check the raw data.")
      print(colored(f"Successfully generated CSV file with {len(df_env)} rows", 'green'))
    except Exception as e:
      print(colored(f"\nFailed to generate CSV file: {e}", 'red'))
      print(colored("Please ensure the raw MARBLE data is properly extracted.", 'yellow'))
      raise

  # Read the CSV file
  df = pd.read_csv(csv_path)
  if df.empty:
    raise ValueError(f"CSV file is empty: {csv_path}")
  print(colored(f"Loaded CSV with {len(df)} rows", 'cyan'))

  # Convert timestamp from milliseconds to datetime
  df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
  df = df.sort_values('datetime').reset_index(drop=True)

  # Step 2: Build event stream
  print(colored("Building event stream...", 'cyan'))
  events = _build_event_stream(df, power_threshold, temperature_bins)
  print(colored(f"Created {len(events)} events", 'cyan'))

  # Step 3: Convert events to sensor states
  print(colored("Converting events to sensor states...", 'cyan'))
  states = _convert_events_to_states(events)
  print(colored(f"Created {len(states)} states", 'cyan'))

  # Step 4: Create fixed-length sliding windows
  print(colored("Creating sliding windows...", 'cyan'))
  windows = _create_sliding_windows(
    states,
    window_length_sec,
    overlap_fraction,
    df  # Pass original df for activity labels
  )
  print(colored(f"Created {len(windows)} windows", 'cyan'))

  # Step 5: Classify states and assign activity labels
  print(colored("Classifying states and assigning activity labels...", 'cyan'))
  windowed_data = _process_windows(windows, window_length_sec, df)

  # Step 5b: Generate captions for windows
  print(colored("Generating captions for windows...", 'cyan'))
  import sys
  # Add src to path if needed
  current_dir = os.path.dirname(os.path.abspath(__file__))
  src_dir = os.path.dirname(current_dir)
  if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

  from data.captions_marble import MarbleCaptionGenerator
  caption_generator = MarbleCaptionGenerator()
  for window in windowed_data:
    window['caption'] = caption_generator.generate_caption(window, window_length_sec)
  print(colored(f"Generated captions for {len(windowed_data)} windows", 'cyan'))

  # Convert to DataFrame (serialize nested structures)
  df_windows = _convert_windows_to_dataframe(windowed_data)

  # Filter out windows with no states
  num_before_states = len(df_windows)
  df_windows = df_windows[df_windows['num_states'] > 0].reset_index(drop=True)
  num_after_states = len(df_windows)
  num_filtered_states = num_before_states - num_after_states
  if num_filtered_states > 0:
    print(colored(f"Filtered out {num_filtered_states} windows with no states ({num_filtered_states/num_before_states*100:.1f}%)", 'yellow'))

  # Filter out unlabeled windows (including None/NaN values)
  num_before = len(df_windows)
  df_windows = df_windows[
    (df_windows['activity_label'] != 'unlabeled') &
    (df_windows['activity_label'].notna())
  ].reset_index(drop=True)
  num_after = len(df_windows)
  num_filtered = num_before - num_after
  if num_filtered > 0:
    print(colored(f"Filtered out {num_filtered} unlabeled windows ({num_filtered/num_before*100:.1f}%)", 'yellow'))

  if num_filtered_states > 0 or num_filtered > 0:
    print(colored(f"Remaining windows: {num_after}", 'cyan'))

  # Step 6: Split into train/test (80/20)
  print(colored("Splitting into train/test sets...", 'cyan'))
  df_train, df_test = _split_train_test(df_windows, train_split)

  print(colored(f"Train windows: {len(df_train)}, Test windows: {len(df_test)}", 'magenta'))

  # Save to JSON
  if save_to_json:
    # Create subfolder based on window length (e.g., sec60, sec16)
    window_length_str = f"sec{int(window_length_sec)}"
    output_subdir = os.path.join(output_dir, window_length_str)
    os.makedirs(output_subdir, exist_ok=True)

    train_path = os.path.join(output_subdir, 'marble_trainable_train.json')
    test_path = os.path.join(output_subdir, 'marble_trainable_test.json')
    _save_windows_to_json(df_train, train_path)
    _save_windows_to_json(df_test, test_path)
    print(colored(f"Saved train data to '{train_path}'", 'green'))
    print(colored(f"Saved test data to '{test_path}'", 'green'))

    # Generate and save statistics
    stats_path = os.path.join(output_subdir, 'marble_trainable_stats.md')
    _generate_dataset_stats(df_train, df_test, stats_path, window_length_sec, overlap_fraction)
    print(colored(f"Saved dataset statistics to '{stats_path}'", 'green'))

  return df_train, df_test


def _build_event_stream(df, power_threshold, temperature_bins):
  """Build event stream from sensor data.

  Binary sensors: emit ON when activated, OFF when deactivated.
  Power sensors: apply threshold; above = ON, below = OFF.
  Temperature: discretize into bins; emit OFF for old bin and ON for new bin.

  Returns:
    List of (sensor_or_bin, ON/OFF, timestamp_ms) tuples, sorted by time.
  """
  events = []

  # Group by sensor to process each sensor's readings
  for sensor_id, sensor_df in df.groupby('sensor_id'):
    sensor_df = sensor_df.sort_values('ts').reset_index(drop=True)

    # Determine sensor type from sensor_id
    sensor_type = sensor_id[0] if len(sensor_id) > 0 else 'U'

    # Check if sensor_status is numeric (power/temperature) or binary
    first_status = sensor_df['sensor_status'].iloc[0]
    is_numeric = isinstance(first_status, (int, float)) or (
      isinstance(first_status, str) and first_status.replace('.', '').replace('-', '').isdigit()
    )

    if sensor_type == 'E' or (is_numeric and sensor_type not in ['T', 'E']):
      # Power sensor or numeric sensor: apply threshold
      for _, row in sensor_df.iterrows():
        try:
          value = float(row['sensor_status'])
          event_type = 'ON' if value >= power_threshold else 'OFF'
          events.append((sensor_id, event_type, row['ts']))
        except (ValueError, TypeError):
          # If conversion fails, treat as binary
          event_type = 'ON' if str(row['sensor_status']).upper() in ['ON', '1', 'TRUE'] else 'OFF'
          events.append((sensor_id, event_type, row['ts']))

    elif sensor_type == 'T' or (is_numeric and sensor_type == 'T'):
      # Temperature sensor: discretize into bins
      values = []
      for _, row in sensor_df.iterrows():
        try:
          values.append(float(row['sensor_status']))
        except (ValueError, TypeError):
          continue

      if len(values) > 0:
        min_val, max_val = min(values), max(values)
        bin_width = (max_val - min_val) / temperature_bins if max_val > min_val else 1.0

        prev_bin = None
        for _, row in sensor_df.iterrows():
          try:
            value = float(row['sensor_status'])
            current_bin = int((value - min_val) / bin_width) if bin_width > 0 else 0
            current_bin = min(current_bin, temperature_bins - 1)
            bin_id = f"{sensor_id}_bin{current_bin}"

            if prev_bin is not None and prev_bin != current_bin:
              # Emit OFF for old bin
              old_bin_id = f"{sensor_id}_bin{prev_bin}"
              events.append((old_bin_id, 'OFF', row['ts']))

            # Emit ON for new bin
            events.append((bin_id, 'ON', row['ts']))
            prev_bin = current_bin
          except (ValueError, TypeError):
            continue

    else:
      # Binary sensor: use sensor_status directly
      for _, row in sensor_df.iterrows():
        status = str(row['sensor_status']).upper()
        event_type = 'ON' if status in ['ON', '1', 'TRUE', 'OPEN'] else 'OFF'
        events.append((sensor_id, event_type, row['ts']))

  # Sort by timestamp
  events.sort(key=lambda x: x[2])
  return events


def _convert_events_to_states(events):
  """Convert events to sensor states.

  Pair each ON with the next OFF for the same sensor/bin.
  Each state is (sensor_or_bin, t_start, t_end).
  For repeated ONs, close the previous interval at the new ON time.

  Returns:
    List of (sensor_or_bin, t_start_ms, t_end_ms) tuples.
  """
  states = []
  active_states = {}  # sensor_or_bin -> (t_start, last_event_time)

  for sensor_or_bin, event_type, timestamp in events:
    if event_type == 'ON':
      # If already active, close previous state and start new one
      if sensor_or_bin in active_states:
        prev_start, _ = active_states[sensor_or_bin]
        states.append((sensor_or_bin, prev_start, timestamp))
      # Start new state
      active_states[sensor_or_bin] = (timestamp, timestamp)
    elif event_type == 'OFF':
      # Close the active state
      if sensor_or_bin in active_states:
        t_start, _ = active_states[sensor_or_bin]
        states.append((sensor_or_bin, t_start, timestamp))
        del active_states[sensor_or_bin]

  # Close any remaining active states (use last timestamp)
  if events:
    last_timestamp = events[-1][2]
    for sensor_or_bin, (t_start, _) in active_states.items():
      states.append((sensor_or_bin, t_start, last_timestamp))

  return states


def _create_sliding_windows(states, window_length_sec, overlap_fraction, df_original):
  """Create fixed-length sliding windows.

  Window length = τ seconds.
  Overlap fraction = o.
  Step size = τ * (1 - o).

  For each window [t, t+τ], collect states that intersect:
    s_start ≤ t+τ and s_end ≥ t

  Returns:
    List of dicts, each containing window info and intersecting states.
  """
  if not states:
    return []

  # Get time range from states
  all_times = []
  for _, t_start, t_end in states:
    all_times.extend([t_start, t_end])

  min_time = min(all_times)
  max_time = max(all_times)

  # Convert window_length_sec to milliseconds
  window_length_ms = window_length_sec * 1000
  step_size_ms = window_length_ms * (1 - overlap_fraction)

  windows = []
  window_id = 0

  # Create windows
  window_start = min_time
  while window_start + window_length_ms <= max_time:
    window_end = window_start + window_length_ms

    # Find states that intersect this window
    # State intersects if: s_start ≤ window_end and s_end ≥ window_start
    intersecting_states = [
      (sensor_or_bin, t_start, t_end)
      for sensor_or_bin, t_start, t_end in states
      if t_start <= window_end and t_end >= window_start
    ]

    windows.append({
      'window_id': window_id,
      'window_start_ms': window_start,
      'window_end_ms': window_end,
      'states': intersecting_states
    })

    window_id += 1
    window_start += step_size_ms

  return windows


def _process_windows(windows, window_length_sec, df_original):
  """Process windows: classify states and assign activity labels.

  For each state in each window:
    - Inner: fully inside (s_start ≥ t and s_end ≤ t+τ)
    - Already-active: started before window (s_start < t)
    - Persistent: ends after window (s_end > t+τ)

  Assign activity label based on maximum overlap duration.
  """
  window_length_ms = window_length_sec * 1000
  processed_windows = []

  for window in windows:
    window_start = window['window_start_ms']
    window_end = window['window_end_ms']
    states = window['states']

    # Classify states
    inner_states = []
    already_active_states = []
    persistent_states = []

    for sensor_or_bin, t_start, t_end in states:
      is_inner = t_start >= window_start and t_end <= window_end
      is_already_active = t_start < window_start
      is_persistent = t_end > window_end

      state_info = {
        'sensor_or_bin': sensor_or_bin,
        't_start': t_start,
        't_end': t_end,
        'is_inner': is_inner,
        'is_already_active': is_already_active,
        'is_persistent': is_persistent
      }

      if is_inner:
        inner_states.append(state_info)
      if is_already_active:
        already_active_states.append(state_info)
      if is_persistent:
        persistent_states.append(state_info)

    # Assign activity label based on overlap duration
    activity_label, num_overlapping_activities = _assign_activity_label(window_start, window_end, df_original)

    processed_windows.append({
      'window_id': window['window_id'],
      'window_start_ms': window_start,
      'window_end_ms': window_end,
      'window_start_datetime': pd.to_datetime(window_start, unit='ms'),
      'window_end_datetime': pd.to_datetime(window_end, unit='ms'),
      'num_states': len(states),
      'num_inner_states': len(inner_states),
      'num_already_active_states': len(already_active_states),
      'num_persistent_states': len(persistent_states),
      'inner_states': inner_states,
      'already_active_states': already_active_states,
      'persistent_states': persistent_states,
      'activity_label': activity_label,
      'num_overlapping_activities': num_overlapping_activities
    })

  return processed_windows


def _assign_activity_label(window_start_ms, window_end_ms, df_original):
  """Assign activity label to window based on maximum overlap duration.

  Consider all ground-truth activity intervals intersecting the window.
  Compute total duration of overlap for each activity label.
  Window's label is the activity with maximum overlap time.

  Returns:
    tuple: (activity_label, num_overlapping_activities)
  """
  # Filter activities that intersect the window
  # Activity intersects if: activity_start <= window_end and activity_end >= window_start
  intersecting_activities = df_original[
    (df_original['activity_start'].notna()) &
    (df_original['activity_end'].notna()) &
    (df_original['activity_start'] <= window_end_ms) &
    (df_original['activity_end'] >= window_start_ms)
  ]

  if intersecting_activities.empty:
    return 'unlabeled', 0

  # Compute overlap duration for each activity
  activity_overlaps = {}
  for _, row in intersecting_activities.iterrows():
    activity = row['activity']
    if pd.isna(activity):
      continue

    activity_start = row['activity_start']
    activity_end = row['activity_end']

    # Calculate overlap
    overlap_start = max(window_start_ms, activity_start)
    overlap_end = min(window_end_ms, activity_end)
    overlap_duration = max(0, overlap_end - overlap_start)

    if activity not in activity_overlaps:
      activity_overlaps[activity] = 0
    activity_overlaps[activity] += overlap_duration

  if not activity_overlaps:
    return 'unlabeled', 0

  # Count unique overlapping activities
  num_overlapping_activities = len(activity_overlaps)

  # Return activity with maximum overlap and count of overlapping activities
  return max(activity_overlaps.items(), key=lambda x: x[1])[0], num_overlapping_activities


def _convert_windows_to_dataframe(windowed_data):
  """Convert windowed data to DataFrame, keeping nested structures as lists/dicts."""
  rows = []
  for window in windowed_data:
    row = {
      'window_id': window['window_id'],
      'window_start_ms': window['window_start_ms'],
      'window_end_ms': window['window_end_ms'],
      'window_start_datetime': window['window_start_datetime'],
      'window_end_datetime': window['window_end_datetime'],
      'num_states': window['num_states'],
      'num_inner_states': window['num_inner_states'],
      'num_already_active_states': window['num_already_active_states'],
      'num_persistent_states': window['num_persistent_states'],
      'activity_label': window['activity_label'],
      'num_overlapping_activities': window.get('num_overlapping_activities', 0),
      'caption': window.get('caption', ''),
      # Keep nested structures as lists/dicts (not JSON strings)
      'inner_states': window['inner_states'],
      'already_active_states': window['already_active_states'],
      'persistent_states': window['persistent_states']
    }
    rows.append(row)

  df = pd.DataFrame(rows)
  # Convert datetime columns to strings for serialization
  if 'window_start_datetime' in df.columns:
    df['window_start_datetime'] = df['window_start_datetime'].astype(str)
  if 'window_end_datetime' in df.columns:
    df['window_end_datetime'] = df['window_end_datetime'].astype(str)

  return df


def _save_windows_to_json(df, json_path):
  """Save windowed data to JSON file with proper nested structure."""
  import json

  # Convert DataFrame to list of dictionaries
  records = df.to_dict('records')

  # Convert datetime strings back to ISO format if needed
  for record in records:
    # Ensure nested structures are properly formatted
    for key in ['inner_states', 'already_active_states', 'persistent_states']:
      if key in record and isinstance(record[key], str):
        # If it's a JSON string, parse it
        try:
          record[key] = json.loads(record[key])
        except (json.JSONDecodeError, TypeError):
          pass

  # Save as JSON with indentation for readability
  with open(json_path, 'w') as f:
    json.dump(records, f, indent=2, default=str)


def _generate_dataset_stats(df_train, df_test, stats_path, window_length_sec, overlap_fraction):
  """Generate and save dataset statistics to a markdown file.

  Args:
    df_train (pd.DataFrame): Training set windows.
    df_test (pd.DataFrame): Test set windows.
    stats_path (str): Path to save the statistics file.
    window_length_sec (float): Window length in seconds.
    overlap_fraction (float): Overlap fraction between windows.
  """
  stats_lines = []
  stats_lines.append("# MARBLE Trainable Dataset Statistics\n")
  stats_lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

  # Processing parameters
  stats_lines.append("## Processing Parameters\n")
  stats_lines.append(f"- Window length: {window_length_sec} seconds\n")
  stats_lines.append(f"- Overlap fraction: {overlap_fraction}\n")
  stats_lines.append(f"- Step size: {window_length_sec * (1 - overlap_fraction):.2f} seconds\n\n")

  # Overall statistics
  stats_lines.append("## Overall Statistics\n")
  stats_lines.append(f"- **Total windows**: {len(df_train) + len(df_test):,}\n")
  stats_lines.append(f"- **Train windows**: {len(df_train):,} ({len(df_train)/(len(df_train)+len(df_test))*100:.1f}%)\n")
  stats_lines.append(f"- **Test windows**: {len(df_test):,} ({len(df_test)/(len(df_train)+len(df_test))*100:.1f}%)\n\n")

  # Events/States statistics
  stats_lines.append("## Window Statistics\n")

  # Average events per window
  avg_states_train = df_train['num_states'].mean() if 'num_states' in df_train.columns else 0
  avg_states_test = df_test['num_states'].mean() if 'num_states' in df_test.columns else 0
  avg_states_total = (df_train['num_states'].sum() + df_test['num_states'].sum()) / (len(df_train) + len(df_test))

  stats_lines.append(f"- **Average states per window (train)**: {avg_states_train:.2f}\n")
  stats_lines.append(f"- **Average states per window (test)**: {avg_states_test:.2f}\n")
  stats_lines.append(f"- **Average states per window (total)**: {avg_states_total:.2f}\n")
  stats_lines.append(f"- **Min states per window**: {min(df_train['num_states'].min(), df_test['num_states'].min())}\n")
  stats_lines.append(f"- **Max states per window**: {max(df_train['num_states'].max(), df_test['num_states'].max())}\n\n")

  # State type statistics
  if 'num_inner_states' in df_train.columns:
    avg_inner_train = df_train['num_inner_states'].mean()
    avg_inner_test = df_test['num_inner_states'].mean()
    avg_already_active_train = df_train['num_already_active_states'].mean()
    avg_already_active_test = df_test['num_already_active_states'].mean()
    avg_persistent_train = df_train['num_persistent_states'].mean()
    avg_persistent_test = df_test['num_persistent_states'].mean()

    stats_lines.append("### State Type Distribution\n")
    stats_lines.append(f"- **Inner states** (train): {avg_inner_train:.2f} avg, (test): {avg_inner_test:.2f} avg\n")
    stats_lines.append(f"- **Already-active states** (train): {avg_already_active_train:.2f} avg, (test): {avg_already_active_test:.2f} avg\n")
    stats_lines.append(f"- **Persistent states** (train): {avg_persistent_train:.2f} avg, (test): {avg_persistent_test:.2f} avg\n\n")

  # Overlapping activities statistics
  if 'num_overlapping_activities' in df_train.columns:
    avg_overlapping_train = df_train['num_overlapping_activities'].mean()
    avg_overlapping_test = df_test['num_overlapping_activities'].mean()
    avg_overlapping_total = (df_train['num_overlapping_activities'].sum() + df_test['num_overlapping_activities'].sum()) / (len(df_train) + len(df_test))
    max_overlapping = max(df_train['num_overlapping_activities'].max(), df_test['num_overlapping_activities'].max())

    # Count windows with multiple overlapping activities
    multi_overlap_train = (df_train['num_overlapping_activities'] > 1).sum()
    multi_overlap_test = (df_test['num_overlapping_activities'] > 1).sum()
    multi_overlap_total = multi_overlap_train + multi_overlap_test
    total_windows = len(df_train) + len(df_test)
    multi_overlap_pct = (multi_overlap_total / total_windows * 100) if total_windows > 0 else 0

    stats_lines.append("### Activity Overlap Statistics\n")
    stats_lines.append(f"- **Average overlapping activities per window (train)**: {avg_overlapping_train:.2f}\n")
    stats_lines.append(f"- **Average overlapping activities per window (test)**: {avg_overlapping_test:.2f}\n")
    stats_lines.append(f"- **Average overlapping activities per window (total)**: {avg_overlapping_total:.2f}\n")
    stats_lines.append(f"- **Max overlapping activities in a window**: {max_overlapping}\n")
    stats_lines.append(f"- **Windows with multiple overlapping activities**: {multi_overlap_total:,} ({multi_overlap_pct:.1f}%)\n")
    stats_lines.append(f"  - Train: {multi_overlap_train:,}, Test: {multi_overlap_test:,}\n\n")

  # Label statistics
  stats_lines.append("## Label Distribution\n\n")

  # Count labels in train and test
  train_label_counts = df_train['activity_label'].value_counts().sort_index()
  test_label_counts = df_test['activity_label'].value_counts().sort_index()
  all_labels = sorted(set(df_train['activity_label'].unique()) | set(df_test['activity_label'].unique()))

  stats_lines.append("### Windows per Label\n\n")
  stats_lines.append("| Label | Train | Test | Total | Train % | Test % |\n")
  stats_lines.append("|-------|-------|------|-------|---------|--------|\n")

  for label in all_labels:
    train_count = train_label_counts.get(label, 0)
    test_count = test_label_counts.get(label, 0)
    total_count = train_count + test_count
    train_pct = (train_count / len(df_train) * 100) if len(df_train) > 0 else 0
    test_pct = (test_count / len(df_test) * 100) if len(df_test) > 0 else 0
    stats_lines.append(f"| {label} | {train_count:,} | {test_count:,} | {total_count:,} | {train_pct:.1f}% | {test_pct:.1f}% |\n")

  stats_lines.append("\n")

  # Average events per label
  stats_lines.append("### Average States per Window by Label\n\n")
  stats_lines.append("| Label | Train Avg | Test Avg | Overall Avg |\n")
  stats_lines.append("|-------|-----------|----------|-------------|\n")

  for label in all_labels:
    train_label_windows = df_train[df_train['activity_label'] == label]
    test_label_windows = df_test[df_test['activity_label'] == label]

    if len(train_label_windows) > 0 and len(test_label_windows) > 0:
      train_avg = train_label_windows['num_states'].mean()
      test_avg = test_label_windows['num_states'].mean()
      overall_avg = (train_label_windows['num_states'].sum() + test_label_windows['num_states'].sum()) / (len(train_label_windows) + len(test_label_windows))
      stats_lines.append(f"| {label} | {train_avg:.2f} | {test_avg:.2f} | {overall_avg:.2f} |\n")
    elif len(train_label_windows) > 0:
      train_avg = train_label_windows['num_states'].mean()
      stats_lines.append(f"| {label} | {train_avg:.2f} | - | {train_avg:.2f} |\n")
    elif len(test_label_windows) > 0:
      test_avg = test_label_windows['num_states'].mean()
      stats_lines.append(f"| {label} | - | {test_avg:.2f} | {test_avg:.2f} |\n")

  stats_lines.append("\n")

  # Time range
  if 'window_start_ms' in df_train.columns and 'window_end_ms' in df_train.columns:
    all_start_times = pd.concat([df_train['window_start_ms'], df_test['window_start_ms']])
    all_end_times = pd.concat([df_train['window_end_ms'], df_test['window_end_ms']])

    min_time = pd.to_datetime(all_start_times.min(), unit='ms')
    max_time = pd.to_datetime(all_end_times.max(), unit='ms')
    duration = max_time - min_time

    stats_lines.append("## Time Range\n")
    stats_lines.append(f"- **Start time**: {min_time}\n")
    stats_lines.append(f"- **End time**: {max_time}\n")
    stats_lines.append(f"- **Total duration**: {duration}\n")
    stats_lines.append(f"- **Duration in hours**: {duration.total_seconds() / 3600:.2f}\n\n")

  # Label coverage check
  stats_lines.append("## Label Coverage\n")
  train_labels_set = set(df_train['activity_label'].unique())
  test_labels_set = set(df_test['activity_label'].unique())
  all_labels_set = set(all_labels)

  missing_in_test = train_labels_set - test_labels_set
  missing_in_train = test_labels_set - train_labels_set

  if missing_in_test:
    stats_lines.append(f"⚠️ **Labels in train but not in test**: {', '.join(sorted(missing_in_test))}\n")
  if missing_in_train:
    stats_lines.append(f"⚠️ **Labels in test but not in train**: {', '.join(sorted(missing_in_train))}\n")
  if not missing_in_test and not missing_in_train:
    stats_lines.append("✅ **All labels present in both train and test sets**\n")

  stats_lines.append(f"\n- **Unique labels in train**: {len(train_labels_set)}\n")
  stats_lines.append(f"- **Unique labels in test**: {len(test_labels_set)}\n")
  stats_lines.append(f"- **Total unique labels**: {len(all_labels_set)}\n")

  # Write to file
  with open(stats_path, 'w') as f:
    f.writelines(stats_lines)


def _split_train_test(df_windows, train_split):
  """Split windows into train and test sets using stratified random sampling.

  Ensures each activity label is present in both train and test sets.
  Uses random sampling (not temporal ordering).

  Args:
    df_windows (pd.DataFrame): DataFrame with windowed data.
    train_split (float): Fraction of data for training (default 0.8).

  Returns:
    tuple: (df_train, df_test) DataFrames with stratified split.
  """
  # Reset index to ensure clean indexing
  df_windows = df_windows.reset_index(drop=True)

  # Check if we have enough samples per label for stratification
  label_counts = df_windows['activity_label'].value_counts()
  min_samples_per_label = label_counts.min()

  # Calculate test size
  test_size = 1.0 - train_split

  # If any label has fewer than 2 samples, we can't stratify properly
  # In that case, we'll do a simple random split but try to ensure each label appears in both sets
  if min_samples_per_label < 2:
    print(colored(
      f"⚠️  Warning: Some labels have < 2 samples. Using random split with label preservation.",
      'yellow'
    ))
    # Simple random split, but manually ensure each label appears in both sets
    df_train, df_test = train_test_split(
      df_windows,
      test_size=test_size,
      random_state=42,
      shuffle=True
    )

    # Ensure each label appears in both sets
    train_labels = set(df_train['activity_label'].unique())
    test_labels = set(df_test['activity_label'].unique())
    missing_in_test = train_labels - test_labels
    missing_in_train = test_labels - train_labels

    # Move one sample from train to test for labels missing in test
    for label in missing_in_test:
      label_samples = df_train[df_train['activity_label'] == label]
      if len(label_samples) > 1:
        # Get the first row as a DataFrame
        sample_to_move = label_samples.iloc[[0]]
        df_test = pd.concat([df_test, sample_to_move], ignore_index=True)
        df_train = df_train.drop(label_samples.index[0])

    # Move one sample from test to train for labels missing in train
    for label in missing_in_train:
      label_samples = df_test[df_test['activity_label'] == label]
      if len(label_samples) > 1:
        # Get the first row as a DataFrame
        sample_to_move = label_samples.iloc[[0]]
        df_train = pd.concat([df_train, sample_to_move], ignore_index=True)
        df_test = df_test.drop(label_samples.index[0])

    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

  # Use stratified split if we have enough samples
  df_train, df_test = train_test_split(
    df_windows,
    test_size=test_size,
    stratify=df_windows['activity_label'],
    random_state=42,
    shuffle=True
  )

  # Verify stratification worked
  train_labels = set(df_train['activity_label'].unique())
  test_labels = set(df_test['activity_label'].unique())
  all_labels = set(df_windows['activity_label'].unique())

  if train_labels != all_labels or test_labels != all_labels:
    print(colored(
      f"⚠️  Warning: Not all labels present in both sets. Train: {len(train_labels)}, Test: {len(test_labels)}, All: {len(all_labels)}",
      'yellow'
    ))

  return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Process CASAS or MARBLE datasets',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Process MARBLE dataset (environmental data)
  python src/data/data_load_clean.py --dataset marble --function process_marble_environmental_data

  # Process MARBLE dataset to trainable data (ADL-LLM preprocessing)
  python src/data/data_load_clean.py --dataset marble --function process_marble_to_trainable_data

  # Process CASAS dataset (milan)
  python src/data/data_load_clean.py --dataset milan

  # Process CASAS dataset with custom train-test split
  python src/data/data_load_clean.py --dataset aruba --custom_train_test_split
    """
  )

  parser.add_argument(
    '--dataset',
    type=str,
    help='Dataset name (e.g., milan, aruba, cairo, marble)'
  )

  parser.add_argument(
    '--function',
    type=str,
    default=None,
    help='Function to run (for marble: process_marble_environmental_data, process_marble_to_trainable_data)'
  )

  parser.add_argument(
    '--force_download',
    action='store_true',
    help='Force re-download of raw data (for CASAS datasets)'
  )

  parser.add_argument(
    '--custom_train_test_split',
    action='store_true',
    help='Create custom train-test split (for CASAS datasets)'
  )

  parser.add_argument(
    '--max_lines',
    type=int,
    default=None,
    help='Limit number of lines to process (for testing)'
  )

  parser.add_argument(
    '--base_path',
    type=str,
    default='data/raw/marble/MARBLE/dataset',
    help='Base path to MARBLE dataset (for marble dataset only)'
  )

  parser.add_argument(
    '--output_path',
    type=str,
    default='data/processed/marble/marble_environment_single_resident.csv',
    help='Output path for processed CSV (for marble dataset only)'
  )

  parser.add_argument(
    '--csv_path',
    type=str,
    default='data/processed/marble/marble_environment_single_resident.csv',
    help='Path to input CSV for process_marble_to_trainable_data'
  )

  parser.add_argument(
    '--window_length_sec',
    type=float,
    default=16.0,
    help='Window length in seconds (for process_marble_to_trainable_data)'
  )

  parser.add_argument(
    '--overlap_fraction',
    type=float,
    default=0.5,
    help='Overlap fraction between windows (for process_marble_to_trainable_data)'
  )

  args = parser.parse_args()

  # Process MARBLE dataset
  if args.dataset == 'marble':
    if args.function == 'process_marble_environmental_data' or args.function is None:
      df = process_marble_environmental_data(
        base_path=args.base_path,
        output_path=args.output_path,
        save_to_csv=True
      )
      print(colored(f"\n✓ Processing complete! Shape: {df.shape}", 'green'))
    elif args.function == 'process_marble_to_trainable_data':
      df_train, df_test = process_marble_to_trainable_data(
        csv_path=args.csv_path,
        window_length_sec=args.window_length_sec,
        overlap_fraction=args.overlap_fraction,
        save_to_json=True
      )
      print(colored(f"\n✓ Processing complete! Train: {df_train.shape}, Test: {df_test.shape}", 'green'))
    else:
      print(colored(f"⚠️  Unknown function '{args.function}' for marble dataset", 'yellow'))
      print(colored("Available functions: process_marble_environmental_data, process_marble_to_trainable_data", 'yellow'))

  # Process CASAS datasets
  elif args.dataset in CASAS_METADATA or args.dataset == 'aware_home':
    if args.dataset == 'aware_home':
      df = awaerehome_end_to_end_preprocess(save_to_csv=True)
    else:
      # Use explicit flag if provided, otherwise default based on dataset
      if args.custom_train_test_split:
        custom_train_test_split = True
      elif args.dataset in ['milan', 'aruba', 'kyoto', 'cairo']:
        custom_train_test_split = True
      else:
        custom_train_test_split = False

      df = casas_end_to_end_preprocess(
        args.dataset,
        save_to_csv=True,
        force_download=args.force_download,
        custom_train_test_split=custom_train_test_split,
        max_lines=args.max_lines
      )
      print(df.groupby(['first_activity_l2', 'first_activity']).agg(
          num=('sensor', 'count')).assign(perc=lambda df: df.num/df.num.sum()))
    print(colored(f"\n✓ Processing complete! Shape: {df.shape}", 'green'))

  # Default: process multiple CASAS datasets (original behavior)
  elif args.dataset is None:
    print(colored("No dataset specified. Processing default CASAS datasets...", 'magenta'))
    df_list = []
    for city in ['milan', 'aruba', 'cairo']:
      print(colored(f"Processing {city} dataset...", 'magenta'))
      if city == 'aware_home':
        df = awaerehome_end_to_end_preprocess()
      else:
        if city in ['milan', 'aruba', 'kyoto', 'cairo']:
          custom_train_test_split = True
        else:
          custom_train_test_split = False
        df = casas_end_to_end_preprocess(
          city,
          force_download=True,
          custom_train_test_split=custom_train_test_split
        )
        print(df.groupby(['first_activity_l2', 'first_activity']).agg(
            num=('sensor', 'count')).assign(perc=lambda df: df.num/df.num.sum()))
        df_list.append(df)

  else:
    print(colored(f"⚠️  Unknown dataset '{args.dataset}'", 'yellow'))
    print(colored(f"Available datasets: {list(CASAS_METADATA.keys())}, marble, aware_home", 'yellow'))
    parser.print_help()