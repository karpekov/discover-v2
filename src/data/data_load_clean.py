""" Functions to download and process raw CASAS data.

All relevant metadata is stored in `metadata/house_metadata.json` file.

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


def _get_json_metadata(file_path='metadata/house_metadata.json'):
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
    columns=['date', 'sensor', 'state', 'activity_full']
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
    return _read_irregular_data(file_path)

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
):
  """Read and process the data from a text file with irregular spacing."""
  with open(file_path, 'r') as file:
      lines = file.read().splitlines()

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

def casas_end_to_end_preprocess(dataset_name, save_to_csv=True, force_download=False, custom_train_test_split=False):
  """ Process the CASAS dataset.

  Given the `metadata/house_metadata.json` file, this function downloads the
  raw data from CASAS website, processes it, and returns a clean DataFrame with
  labels and other annotations.

  If `custom_train_test_split` is set to True, the function will split the data
  into train and test sets and return tow DataFrames instead of one.
  """
  if dataset_name == 'aware_home':
    raise ValueError(
      'Use the `awaerehome_end_to_end_preprocess()` function for the Aware Home dataset.'
    )
  elif dataset_name not in CASAS_METADATA:
    raise ValueError(
      f'Dataset name "{dataset_name}" not in `house_metadata.json`. Must be one of: {list(CASAS_METADATA.keys())}'
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
    df_raw = read_and_process_data(file_path=local_raw_file)
  else:
    download_and_unzip(casas_url)
    df_raw = read_and_process_data(file_path=f'data/raw/casas/{data_raw_filename}')
  df = process_raw_data(df_raw)
  # df = tulum_specific_preprocess(df)

  df['first_activity_l2'] = df.first_activity.map(metadata['label_l2'])
  # Add a column with sensor locations
  df['sensor_location'] = df.sensor.map(metadata['sensor_location'])

  if metadata['num_residents'] > 1:
    df = process_multiresident_home(df)

  # Tulum specific fix
  if dataset_name == 'tulum':
    # Fix the 'OFF5' state to 'OFF'
    df['state'] = df.state.replace('OFF5', 'OFF')

  print(colored(f'Processed data loaded. Shape: {df.shape}', 'magenta'))
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

if __name__ == '__main__':

  df_list = []
  # for city in CASAS_METADATA:
  for city in ['milan', 'aruba', 'cairo']:
    print(
      colored(f"Processing {city} dataset...", 'magenta')
    )
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

  # df = casas_end_to_end_preprocess('milan')
  # print(df.shape)

  # df_ah = awaerehome_end_to_end_preprocess()
  # print(df_ah.shape)

  # df_milan = df_list[0]
  # df_aruba = df_list[1]
  # df_cairo = df_list[2]
  # print('Milan: num unique days: ', len(set(df_milan.date_col)))
  # print('Aruba: num unique days: ', len(set(df_aruba.date_col)))
  # print('Cairo: num unique days: ', len(set(df_cairo.date_col)))

  # df_milan_gr = df_milan.groupby(['first_activity_l2']).agg(
  #     num=('sensor', 'count')).assign(perc=lambda df: df.num/df.num.sum())

  # df_aruba_gr = df_aruba.groupby(['first_activity_l2']).agg(
  #     num=('sensor', 'count')).assign(perc=lambda df: df.num/df.num.sum())

  # df_cairo_gr = df_cairo.groupby(['first_activity_l2']).agg(
  #     num=('sensor', 'count')).assign(perc=lambda df: df.num/df.num.sum())

  # # join the three dataframes
  # df_all = df_milan_gr.join(df_aruba_gr, lsuffix='_milan', rsuffix='_aruba')
  # df_all = df_all.join(df_cairo_gr, rsuffix='_cairo')
  # print(df_all)