"""
Dataset for alignment training.

Loads sensor data and text embeddings (pre-computed or on-the-fly).
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any
from pathlib import Path


class AlignmentDataset(Dataset):
    """
    Dataset for sensor-text alignment training.

    Loads:
    - Sensor sequences from JSON (Step 1 output)
    - Text embeddings from NPZ (Step 4 output) OR
    - Captions from JSON (Step 3 output) + text encoder for on-the-fly encoding
    """

    def __init__(
        self,
        data_path: str,
        text_embeddings_path: Optional[str] = None,
        captions_path: Optional[str] = None,
        text_encoder_config_path: Optional[str] = None,
        vocab: Optional[Dict[str, Dict[str, int]]] = None,
        device: Optional[torch.device] = None,
        span_masker: Optional[Any] = None,
        vocab_sizes: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            data_path: Path to sampled sensor data JSON (Step 1 output)
            text_embeddings_path: Path to pre-computed text embeddings NPZ (Step 4 output)
            captions_path: Path to captions JSON (Step 3 output) - alternative to text_embeddings_path
            text_encoder_config_path: Path to text encoder config YAML - required if using captions_path
            vocab: Vocabulary mapping for categorical features
            device: Device for tensors
            span_masker: Optional SpanMasker for MLM (if mlm_weight > 0)
            vocab_sizes: Vocabulary sizes for each field (required for MLM)
        """
        self.data_path = data_path
        self.text_embeddings_path = text_embeddings_path
        self.captions_path = captions_path
        self.text_encoder_config_path = text_encoder_config_path
        self.vocab = vocab
        self.device = device or torch.device('cpu')
        self.span_masker = span_masker
        self.vocab_sizes = vocab_sizes

        # Load sensor data
        self.sensor_data = self._load_sensor_data()

        # Load text embeddings (pre-computed or on-the-fly)
        if text_embeddings_path:
            self.text_embeddings = self._load_text_embeddings()
            self.text_encoder = None
        elif captions_path and text_encoder_config_path:
            self.captions = self._load_captions()
            self.text_encoder = self._load_text_encoder()
            self.text_embeddings = None
        else:
            raise ValueError("Either text_embeddings_path OR (captions_path + text_encoder_config_path) must be provided")

        # Validate data alignment
        self._validate_data()

    def _load_sensor_data(self) -> List[Dict[str, Any]]:
        """Load sensor data from JSON."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        # Extract samples
        if 'samples' in data:
            samples = data['samples']
        else:
            samples = data

        return samples

    def _load_text_embeddings(self) -> np.ndarray:
        """Load pre-computed text embeddings from NPZ."""
        data = np.load(self.text_embeddings_path)
        embeddings = data['embeddings']

        return embeddings

    def _load_captions(self) -> List[str]:
        """Load captions from JSON."""
        with open(self.captions_path, 'r') as f:
            data = json.load(f)

        # Extract captions
        if 'captions' in data:
            captions = data['captions']
        elif isinstance(data, list):
            captions = [item['caption'] if isinstance(item, dict) else item for item in data]
        else:
            raise ValueError(f"Unexpected captions format in {self.captions_path}")

        return captions

    def _load_text_encoder(self):
        """Load text encoder for on-the-fly encoding."""
        from src.text_encoders import build_text_encoder
        import yaml

        with open(self.text_encoder_config_path, 'r') as f:
            config = yaml.safe_load(f)

        encoder = build_text_encoder(config)
        encoder.to(self.device)
        encoder.eval()

        return encoder

    def _validate_data(self):
        """
        Validate that sensor data and text data are aligned.

        This checks that the lengths match. The alignment is preserved during shuffling
        because DataLoader shuffles indices (not the underlying arrays), and we use
        the same index to access both sensor_data[idx] and text_embeddings[idx].
        """
        num_sensor_samples = len(self.sensor_data)

        if self.text_embeddings is not None:
            num_text_samples = len(self.text_embeddings)
        else:
            num_text_samples = len(self.captions)

        if num_sensor_samples != num_text_samples:
            raise ValueError(
                f"Mismatch between sensor samples ({num_sensor_samples}) "
                f"and text samples ({num_text_samples}). "
                f"Sensor data and text data must have the same number of samples "
                f"and be ordered consistently."
            )

    def __len__(self) -> int:
        return len(self.sensor_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Note: When DataLoader shuffles, it shuffles the indices passed to this method,
        not the underlying data. Since we use the same idx for both sensor_data[idx]
        and text_embeddings[idx] (or captions[idx]), the alignment is preserved.
        """
        # Get sensor data
        sensor_sample = self.sensor_data[idx]
        sensor_sequence = sensor_sample['sensor_sequence']

        # Convert sensor sequence to tensors
        # Initialize categorical features as separate lists for each field
        categorical_features = {field: [] for field in self.vocab.keys()}
        coordinates = []
        time_deltas = []

        for i, event in enumerate(sensor_sequence):
            # Encode categorical features - one list per field
            for field in self.vocab.keys():
                # Map field names to event keys
                field_mapping = {
                    'sensor': 'sensor',
                    'sensor_id': 'sensor_id',
                    'state': 'state',
                    'event_type': 'event_type',
                    'room_id': 'room_id',
                    'room': 'room',
                }

                event_key = field_mapping.get(field, field)
                value = event.get(event_key, 'UNK')

                # Get vocab index
                if value in self.vocab[field]:
                    idx_val = self.vocab[field][value]
                else:
                    idx_val = self.vocab[field].get('UNK', 0)

                categorical_features[field].append(idx_val)

            # Get coordinates
            x = event.get('x', 0.0)
            y = event.get('y', 0.0)
            coordinates.append([x, y])

            # Get time delta
            if i == 0:
                time_delta = 0.0
            else:
                # Compute time delta from timestamps
                from datetime import datetime
                curr_time = datetime.fromisoformat(event.get('timestamp', event.get('datetime')))
                prev_time = datetime.fromisoformat(sensor_sequence[i-1].get('timestamp', sensor_sequence[i-1].get('datetime')))
                time_delta = (curr_time - prev_time).total_seconds()

            time_deltas.append(time_delta)

        # Convert to tensors - categorical_features is a dict of tensors
        categorical_features = {
            field: torch.tensor(values, dtype=torch.long)
            for field, values in categorical_features.items()
        }
        coordinates = torch.tensor(coordinates, dtype=torch.float32)
        time_deltas = torch.tensor(time_deltas, dtype=torch.float32)

        # Get text embedding or caption (using same idx ensures alignment)
        if self.text_embeddings is not None:
            text_embedding = torch.tensor(self.text_embeddings[idx], dtype=torch.float32)
            caption = None
        else:
            text_embedding = None
            caption = self.captions[idx]

        return {
            'categorical_features': categorical_features,
            'coordinates': coordinates,
            'time_deltas': time_deltas,
            'text_embedding': text_embedding,
            'caption': caption,
            'sample_id': sensor_sample.get('sample_id', f'sample_{idx}')
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with padding.

        Handles variable-length sequences by padding to max length in batch.
        """
        batch_size = len(batch)

        # Find max sequence length in batch (check one of the categorical fields)
        first_field = list(batch[0]['categorical_features'].keys())[0]
        max_len = max(item['categorical_features'][first_field].size(0) for item in batch)

        # Get all categorical field names
        cat_fields = list(batch[0]['categorical_features'].keys())

        # Initialize padded tensors for each categorical field
        categorical_features = {
            field: torch.zeros((batch_size, max_len), dtype=torch.long)
            for field in cat_fields
        }
        coordinates = torch.zeros((batch_size, max_len, 2), dtype=torch.float32)
        time_deltas = torch.zeros((batch_size, max_len), dtype=torch.float32)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        # Fill in data
        for i, item in enumerate(batch):
            # Get sequence length from one of the fields
            seq_len = item['categorical_features'][first_field].size(0)

            # Pad categorical features (each field separately)
            for field in cat_fields:
                categorical_features[field][i, :seq_len] = item['categorical_features'][field]

            coordinates[i, :seq_len] = item['coordinates']
            time_deltas[i, :seq_len] = item['time_deltas']
            attention_mask[i, :seq_len] = True

        # Handle text embeddings or captions
        if batch[0]['text_embedding'] is not None:
            # Pre-computed embeddings
            text_embeddings = torch.stack([item['text_embedding'] for item in batch])
        else:
            # On-the-fly encoding
            captions = [item['caption'] for item in batch]
            with torch.no_grad():
                text_embeddings = self.text_encoder.encode_batch(captions, device=self.device)

        # Move to device
        categorical_features = {
            field: tensor.to(self.device)
            for field, tensor in categorical_features.items()
        }
        coordinates = coordinates.to(self.device)
        time_deltas = time_deltas.to(self.device)
        attention_mask = attention_mask.to(self.device)
        text_embeddings = text_embeddings.to(self.device)

        result = {
            'sensor_data': {
                'categorical_features': categorical_features,
                'coordinates': coordinates,
                'time_deltas': time_deltas,
            },
            'text_embeddings': text_embeddings,
            'attention_mask': attention_mask,
            'sample_ids': [item['sample_id'] for item in batch]
        }

        # Apply MLM masking if span_masker is provided
        if self.span_masker is not None and self.vocab_sizes is not None:
            masked_features, original_features, mask_positions, blackout_masks = self.span_masker(
                categorical_features,
                self.vocab_sizes,
                activity_labels=None,  # Could add if needed
                room_labels=categorical_features.get('room_id'),
                attention_mask=attention_mask  # Pass attention mask to prevent masking padding
            )

            # Replace categorical features with masked versions
            result['sensor_data']['categorical_features'] = masked_features
            result['mlm_labels'] = original_features
            result['mlm_mask_positions'] = mask_positions
            # Note: mlm_predictions will be generated by the model during forward pass

        return result

