import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import json


class SmartHomeDataset(Dataset):
  """
  Dataset for smart-home event sequences with text captions.

  Expected data format:
  - Each sample contains a sequence of T events (e.g., T=20)
  - Per-event categorical features: sensor_id, room_id, event_type, sensor_type,
    tod_bucket, delta_t_bucket, floor_id (optional), dow (optional)
  - Per-event continuous features: normalized (x,y) coordinates, delta_t_since_prev
  - 1-3 short captions per window describing the activity
  """

  def __init__(
    self,
    data_path: str,
    vocab_path: str,
    sequence_length: int = 20,
    max_captions: int = 3,
    normalize_coords: bool = True,
    caption_types: str = 'long'  # 'long', 'short', 'both'
  ):
    """
    Args:
      data_path: Path to the dataset file (JSON or parquet)
      vocab_path: Path to vocabulary mappings (JSON)
      sequence_length: Fixed sequence length T
      max_captions: Maximum number of captions per sequence
      normalize_coords: Whether to normalize coordinates to [0,1]
      caption_types: Which caption types to use ('long', 'short', 'both')
    """
    self.sequence_length = sequence_length
    self.max_captions = max_captions
    self.caption_types = caption_types
    self.normalize_coords = normalize_coords

    # Load vocabulary mappings
    with open(vocab_path, 'r') as f:
      self.vocab = json.load(f)

    # Load data
    self.data = self._load_data(data_path)

    # Categorical field names - use only fields that exist in vocab
    # Common categorical fields across different data formats
    possible_fields = [
      'sensor_id', 'sensor', 'room_id', 'event_type', 'state', 'sensor_type',
      'tod_bucket', 'delta_t_bucket', 'floor_id', 'dow'
    ]
    
    self.categorical_fields = [field for field in possible_fields if field in self.vocab]

    # Get vocabulary sizes
    self.vocab_sizes = {field: len(self.vocab[field]) for field in self.categorical_fields}

    # Coordinate normalization bounds (computed from data if needed)
    if self.normalize_coords:
      self._compute_coord_bounds()

  def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
    """Load dataset from file."""
    if data_path.endswith('.json'):
      with open(data_path, 'r') as f:
        data = json.load(f)
        # Handle both formats: direct list or dict with 'samples' key
        if isinstance(data, dict) and 'samples' in data:
          return data['samples']
        elif isinstance(data, list):
          return data
        else:
          raise ValueError(f"Unexpected data format in {data_path}")
    elif data_path.endswith('.parquet'):
      df = pd.read_parquet(data_path)
      return df.to_dict('records')
    else:
      raise ValueError(f"Unsupported file format: {data_path}")

  def _compute_coord_bounds(self):
    """Compute coordinate bounds for normalization."""
    all_x, all_y = [], []

    for sample in self.data:
      # Handle both old format (events) and new format (sensor_sequence)
      if 'events' in sample:
        events = sample['events']
        for event in events:
          all_x.append(event['x'])
          all_y.append(event['y'])
      elif 'sensor_sequence' in sample:
        # New format - coordinates might be in sensor_sequence
        for event in sample['sensor_sequence']:
          if 'x' in event and 'y' in event:
            all_x.append(event['x'])
            all_y.append(event['y'])
      # If no coordinate data found, skip normalization
      
    if not all_x or not all_y:
      # No coordinates found - disable normalization
      self.normalize_coords = False
      self.x_min, self.x_max = 0, 1
      self.y_min, self.y_max = 0, 1
      self.x_range, self.y_range = 1, 1
      return

    self.x_min, self.x_max = min(all_x), max(all_x)
    self.y_min, self.y_max = min(all_y), max(all_y)

    # Add small epsilon to avoid division by zero
    self.x_range = max(self.x_max - self.x_min, 1e-6)
    self.y_range = max(self.y_max - self.y_min, 1e-6)

  def _normalize_coordinates(self, x: float, y: float) -> Tuple[float, float]:
    """Normalize coordinates to [0, 1] range."""
    if self.normalize_coords:
      x_norm = (x - self.x_min) / self.x_range
      y_norm = (y - self.y_min) / self.y_range
      return x_norm, y_norm
    else:
      return x, y

  def _encode_categorical(self, field: str, value: str) -> int:
    """Encode categorical value to integer index."""
    if field not in self.vocab:
      return 0  # Unknown token

    field_vocab = self.vocab[field]
    if value in field_vocab:
      return field_vocab[value]
    else:
      # Return unknown token index (assume it's the last index)
      return len(field_vocab) - 1

  def _pad_or_truncate_sequence(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pad or truncate event sequence to fixed length."""
    if len(events) >= self.sequence_length:
      # Truncate
      return events[:self.sequence_length]
    else:
      # Pad with the last event (or create dummy events)
      padded_events = events.copy()
      last_event = events[-1] if events else self._create_dummy_event()

      while len(padded_events) < self.sequence_length:
        padded_events.append(last_event.copy())

      return padded_events

  def _create_dummy_event(self) -> Dict[str, Any]:
    """Create a dummy event for padding."""
    return {
      'sensor_id': '<PAD>',
      'room_id': '<PAD>',
      'event_type': '<PAD>',
      'sensor_type': '<PAD>',
      'tod_bucket': '<PAD>',
      'delta_t_bucket': '<PAD>',
      'floor_id': '<PAD>',
      'dow': '<PAD>',
      'x': 0.0,
      'y': 0.0,
      'delta_t_since_prev': 0.0
    }

  def _compute_delta_times(self, events: List[Dict[str, Any]]) -> List[float]:
    """Compute time deltas between consecutive events."""
    if len(events) <= 1:
      return [0.0] * len(events)

    delta_times = [0.0]  # First event has delta_t = 0

    for i in range(1, len(events)):
      # Priority 1: Use pre-computed delta_t_since_prev (most accurate, in seconds)
      if 'delta_t_since_prev' in events[i]:
        delta_t = events[i]['delta_t_since_prev']
      # Priority 2: Compute from timestamps (could be Unix ms or normalized index)
      elif 'timestamp' in events[i] and 'timestamp' in events[i-1]:
        curr_time = events[i]['timestamp']
        prev_time = events[i-1]['timestamp']
        
        # Handle different timestamp formats
        if isinstance(curr_time, str):
          # String timestamp - parse it
          from dateutil import parser as date_parser
          try:
            curr_dt = date_parser.parse(curr_time)
            prev_dt = date_parser.parse(prev_time)
            delta_t = max(0.0, (curr_dt - prev_dt).total_seconds())
          except Exception:
            # Fallback if parsing fails
            delta_t = 1.0
        else:
          # Numeric timestamp
          delta_t = max(0.0, curr_time - prev_time)
          # If timestamps are in milliseconds (Unix time), convert to seconds
          if curr_time > 10000:  # Heuristic: Unix timestamps are large
            delta_t = delta_t / 1000.0
      else:
        # Fallback to default
        delta_t = 1.0

      delta_times.append(delta_t)

    return delta_times

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    """
    Get a single sample.

    Returns:
      sample: Dict containing:
        - categorical_features: Dict of [seq_len] tensors for each categorical field
        - coordinates: [seq_len, 2] tensor of normalized (x,y) coordinates
        - time_deltas: [seq_len] tensor of time deltas since previous event
        - mask: [seq_len] boolean tensor (True = valid event, False = padding)
        - captions: List of text captions (1-3 strings)
    """
    sample_data = self.data[idx]

    # Get events - handle both old format (events) and new format (sensor_sequence)
    if 'sensor_sequence' in sample_data:
      events = sample_data['sensor_sequence']
    elif 'events' in sample_data:
      events = sample_data['events']
    else:
      raise KeyError(f"Sample {idx} has neither 'events' nor 'sensor_sequence' key. Keys: {list(sample_data.keys())}")
    
    # Normalize field names to match vocab expectations
    # New format uses: sensor_id, event_type, room
    # Old format uses: sensor, state, room_id
    # We need to map to whatever is in the vocab
    normalized_events = []
    for event in events:
      norm_event = event.copy()
      # Map sensor_id → sensor if vocab has 'sensor'
      if 'sensor' in self.categorical_fields and 'sensor_id' in event:
        norm_event['sensor'] = event['sensor_id']
      # Map event_type → state if vocab has 'state'
      if 'state' in self.categorical_fields and 'event_type' in event:
        norm_event['state'] = event['event_type']
      # Map room → room_id if vocab has 'room_id'
      if 'room_id' in self.categorical_fields and 'room' in event:
        norm_event['room_id'] = event['room']
      normalized_events.append(norm_event)
    
    events = normalized_events

    # Select appropriate caption type
    if self.caption_types == 'long' and 'long_captions' in sample_data:
      captions = sample_data.get('long_captions', [])
    elif self.caption_types == 'short' and 'short_captions' in sample_data:
      captions = sample_data.get('short_captions', [])
    elif self.caption_types == 'both':
      long_caps = sample_data.get('long_captions', [])
      short_caps = sample_data.get('short_captions', [])
      captions = long_caps + short_caps
    else:
      # Fallback: filter existing captions by length for backward compatibility
      all_captions = sample_data.get('captions', [''])
      if self.caption_types == 'long':
        # Filter for long captions (>= 10 tokens)
        captions = [cap for cap in all_captions if len(cap.split()) >= 10]
      elif self.caption_types == 'short':
        # Filter for short captions (< 10 tokens)
        captions = [cap for cap in all_captions if len(cap.split()) < 10]
      else:
        # Use all captions
        captions = all_captions

    # Ensure we have at least one caption
    if not captions:
      captions = ['']

    # Limit number of captions
    captions = captions[:self.max_captions]

    # Pad or truncate sequence
    events = self._pad_or_truncate_sequence(events)

    # Compute time deltas
    time_deltas = self._compute_delta_times(events)

    # Create mask (True for valid events, False for padding)
    # Use the original events list (before padding/truncation)
    if 'sensor_sequence' in sample_data:
      original_events = sample_data['sensor_sequence']
    elif 'events' in sample_data:
      original_events = sample_data['events']
    else:
      original_events = []
    original_length = min(len(original_events), self.sequence_length)
    mask = [True] * original_length + [False] * (self.sequence_length - original_length)

    # Process categorical features
    categorical_features = {}
    for field in self.categorical_fields:
      field_values = []
      for event in events:
        value = event.get(field, '<UNK>')
        encoded_value = self._encode_categorical(field, str(value))
        field_values.append(encoded_value)
      categorical_features[field] = torch.tensor(field_values, dtype=torch.long)

    # Process coordinates
    coordinates = []
    for event in events:
      x, y = event.get('x', 0.0), event.get('y', 0.0)
      x_norm, y_norm = self._normalize_coordinates(x, y)
      coordinates.append([x_norm, y_norm])
    coordinates = torch.tensor(coordinates, dtype=torch.float32)

    # Process time deltas
    time_deltas = torch.tensor(time_deltas, dtype=torch.float32)

    # Create mask tensor
    mask = torch.tensor(mask, dtype=torch.bool)

    # Include ground truth labels if available
    result = {
      'categorical_features': categorical_features,
      'coordinates': coordinates,
      'time_deltas': time_deltas,
      'mask': mask,
      'captions': captions
    }

    # Add ground truth labels if present
    # New format: labels are in metadata.ground_truth_labels
    # Old format: labels are directly in sample_data
    if 'metadata' in sample_data and 'ground_truth_labels' in sample_data['metadata']:
      gt_labels = sample_data['metadata']['ground_truth_labels']
      if 'primary_l1' in gt_labels:
        result['first_activity'] = gt_labels['primary_l1']
      if 'primary_l2' in gt_labels:
        result['first_activity_l2'] = gt_labels['primary_l2']
    else:
      # Old format fallback
      if 'first_activity' in sample_data:
        result['first_activity'] = sample_data['first_activity']
      if 'first_activity_l2' in sample_data:
        result['first_activity_l2'] = sample_data['first_activity_l2']

    return result


def create_sample_dataset(
  output_path: str,
  vocab_path: str,
  num_samples: int = 1000,
  sequence_length: int = 20
):
  """
  Create a sample dataset for testing purposes.

  Args:
    output_path: Path to save the sample dataset
    vocab_path: Path to save vocabulary mappings
    num_samples: Number of sample sequences to generate
    sequence_length: Length of each sequence
  """
  import random

  # Create vocabulary
  vocab = {
    'sensor_id': {f'sensor_{i}': i for i in range(50)},
    'room_id': {'kitchen': 0, 'bedroom': 1, 'bathroom': 2, 'living_room': 3, 'hallway': 4},
    'event_type': {'ON': 0, 'OFF': 1, 'OPEN': 2, 'CLOSE': 3},
    'sensor_type': {'motion': 0, 'door': 1, 'light': 2, 'temperature': 3},
    'tod_bucket': {f'hour_{i}': i for i in range(24)},
    'delta_t_bucket': {f'bucket_{i}': i for i in range(20)},
    'floor_id': {'floor_1': 0, 'floor_2': 1},
    'dow': {f'day_{i}': i for i in range(7)}
  }

  # Add special tokens
  for field in vocab:
    vocab[field]['<UNK>'] = len(vocab[field])
    vocab[field]['<PAD>'] = len(vocab[field])

  # Save vocabulary
  with open(vocab_path, 'w') as f:
    json.dump(vocab, f, indent=2)

  # Generate sample data
  data = []

  # Sample captions for different activities
  activity_captions = [
    ["morning routine", "getting ready for work", "bathroom and kitchen activity"],
    ["cooking dinner", "preparing meal", "kitchen activity"],
    ["watching TV", "relaxing in living room", "evening entertainment"],
    ["going to bed", "nighttime routine", "bedroom activity"],
    ["night wandering", "restless movement", "late night activity"],
    ["cleaning house", "household chores", "moving between rooms"],
    ["working from home", "office activity", "focused work session"]
  ]

  for i in range(num_samples):
    # Random sequence length (with some variation)
    seq_len = random.randint(sequence_length // 2, sequence_length + 5)

    events = []
    current_time = 0.0

    for j in range(seq_len):
      # Generate random event
      event = {
        'sensor_id': f'sensor_{random.randint(0, 49)}',
        'room_id': random.choice(list(vocab['room_id'].keys())[:-2]),  # Exclude special tokens
        'event_type': random.choice(list(vocab['event_type'].keys())[:-2]),
        'sensor_type': random.choice(list(vocab['sensor_type'].keys())[:-2]),
        'tod_bucket': f'hour_{random.randint(0, 23)}',
        'delta_t_bucket': f'bucket_{random.randint(0, 19)}',
        'floor_id': random.choice(['floor_1', 'floor_2']),
        'dow': f'day_{random.randint(0, 6)}',
        'x': random.uniform(0, 10),  # Room coordinates
        'y': random.uniform(0, 10),
        'timestamp': current_time
      }

      events.append(event)
      current_time += np.random.exponential(30.0)  # Average 30 seconds between events

    # Select random captions
    caption_set = random.choice(activity_captions)
    num_captions = random.randint(1, min(3, len(caption_set)))
    captions = random.sample(caption_set, num_captions)

    data.append({
      'events': events,
      'captions': captions
    })

  # Save data
  with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)

  print(f"Created sample dataset with {num_samples} samples")
  print(f"Data saved to: {output_path}")
  print(f"Vocabulary saved to: {vocab_path}")


if __name__ == "__main__":
  # Create sample dataset for testing
  create_sample_dataset(
    output_path="/tmp/sample_smarthome_data.json",
    vocab_path="/tmp/sample_vocab.json",
    num_samples=100,
    sequence_length=20
  )
