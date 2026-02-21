#!/usr/bin/env python3
"""
General retrieval script for querying the trained model with arbitrary text queries.
This is the main evaluation tool for the trained dual-encoder model.

The script automatically:
- Detects the text encoder type (CLIP, SigLIP, GTE, etc.) from checkpoint config
- Loads text projection head if present in checkpoint
- Applies projections during inference for proper alignment

Sample Usage:
  # Interactive mode
  python src/evals/query_retrieval.py \
    --checkpoint trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt \
    --test_data data/processed/casas/milan/FD_60/test.json \
    --vocab data/processed/casas/milan/FD_60/vocab.json \
    --captions data/processed/casas/milan/FD_60/test_captions_baseline.json \
    --interactive \
    --max_samples 5000

  # Single query mode
  python src/evals/query_retrieval.py \
    --checkpoint trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt \
    --test_data data/processed/casas/milan/FD_60/test.json \
    --vocab data/processed/casas/milan/FD_60/vocab.json \
    --captions data/processed/casas/milan/FD_60/test_captions_baseline.json \
    --query "cooking in kitchen" \
    --top_k 5

  # Demo mode (runs predefined queries)
  python src/evals/query_retrieval.py \
    --checkpoint trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt \
    --test_data data/processed/casas/milan/FD_60/test.json \
    --vocab data/processed/casas/milan/FD_60/vocab.json \
    --captions data/processed/casas/milan/FD_60/test_captions_baseline.json

Note: You'll need the processed data files (test.json, vocab.json, test_captions_baseline.json)
      which should match the data used during training. Generate them using the data generation
      pipeline or copy from the training server.
"""

import torch
import numpy as np
import faiss
import json
import argparse
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.text_encoder import TextEncoder, build_text_encoder
from models.sensor_encoder import SensorEncoder
from dataio.dataset import SmartHomeDataset
from dataio.collate import create_data_loader
from utils.device_utils import get_optimal_device


class SmartHomeRetrieval:
    """General retrieval system for smart-home sensor sequences."""

    def __init__(self, checkpoint_path: str, vocab_path: str, test_data_path: str,
                 captions_path: str = None, max_samples: int = 5000, debug: bool = False,
                 metadata_path: str = None):
        self.device = get_optimal_device()
        self.checkpoint_path = checkpoint_path
        self.vocab_path = vocab_path
        self.test_data_path = test_data_path
        self.captions_path = captions_path
        self.max_samples = max_samples
        self.captions_dict = {}
        self.debug = debug
        self.metadata_path = metadata_path
        self.sensor_details = {}

        # Load models and data
        self._load_models()
        self._load_vocab()
        self._load_metadata()
        self._load_captions()
        self._load_data()
        self._extract_embeddings()
        self._build_index()

    def _load_models(self):
        """Load trained models from checkpoint."""
        print(f"🔄 Loading models from {self.checkpoint_path}")

        # Set up module namespace for backwards compatibility with checkpoints saved with 'src.' prefix
        import sys
        import types

        if 'src' not in sys.modules:
            src_module = types.ModuleType('src')
            src_module.__path__ = []
            sys.modules['src'] = src_module

        # Import and map all necessary modules
        import encoders
        import encoders.base
        import encoders.config
        import alignment
        import alignment.config
        import alignment.model
        import losses
        import losses.clip

        # Create mappings
        for module_name in list(sys.modules.keys()):
            if not module_name.startswith('src.') and not module_name.startswith('_'):
                if any(module_name.startswith(pkg) for pkg in ['encoders', 'alignment', 'losses', 'models', 'dataio', 'utils']):
                    sys.modules[f'src.{module_name}'] = sys.modules[module_name]
                    # Also set as attribute on src module
                    parts = module_name.split('.')
                    if len(parts) == 1:
                        setattr(sys.modules['src'], parts[0], sys.modules[module_name])

        # Now load the model
        from alignment.model import AlignmentModel

        full_model = AlignmentModel.load(
            self.checkpoint_path,
            device=self.device,
            vocab_path=self.vocab_path
        )

        # Extract components
        self.sensor_encoder = full_model.sensor_encoder
        self.text_projection = full_model.text_projection
        self.config = full_model.config
        self.vocab_sizes = full_model.vocab_sizes

        # Get config dict for compatibility - handle both dict and dataclass
        if hasattr(self.config, '__dataclass_fields__'):
            # It's an AlignmentConfig dataclass
            encoder_config = self.config.encoder
            if hasattr(encoder_config, '__dataclass_fields__'):
                # Encoder config is also a dataclass
                config = {
                    'd_model': encoder_config.d_model,
                    'n_layers': encoder_config.n_layers,
                    'n_heads': encoder_config.n_heads,
                    'd_ff': encoder_config.d_ff,
                    'max_seq_len': encoder_config.max_seq_len,
                    'dropout': encoder_config.dropout,
                    'fourier_bands': getattr(encoder_config, 'fourier_bands', 12),
                    'train_text_embeddings_path': self.config.train_text_embeddings_path,
                }
            else:
                # Encoder config is a dict
                config = dict(encoder_config)
                config['train_text_embeddings_path'] = self.config.train_text_embeddings_path
        else:
            # Config is already a dict
            config = self.config

        # Sensor encoder is already loaded and on device
        self.sensor_encoder.eval()

        # Load the checkpoint again to pass to text encoder creator
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Use the same robust text encoder detection as evaluate_embeddings.py
        from evals.eval_utils import create_text_encoder_from_checkpoint
        self.text_encoder = create_text_encoder_from_checkpoint(
            checkpoint=checkpoint,
            device=self.device,
            data_path=config.get('train_text_embeddings_path')
        )

        if self.text_encoder is None:
            # Ultimate fallback
            from models.text_encoder import TextEncoder
            text_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            self.text_encoder = TextEncoder(text_model_name)
            print(f"⚠️  Using default text encoder: {text_model_name}")

        self.text_encoder.to(self.device)

        # Text projection is already loaded
        if self.text_projection is not None:
            self.text_projection.eval()
            print(f"✅ Text projection loaded from checkpoint")

        print(f"✅ Models loaded successfully (d_model={config['d_model']}, n_layers={config['n_layers']})")

    def _load_vocab(self):
        """Load vocabulary for decoding sensor sequences."""
        with open(self.vocab_path, 'r') as f:
            self.vocab = json.load(f)

        # Create reverse vocabulary for decoding
        self.reverse_vocab = {}
        for field, field_vocab in self.vocab.items():
            self.reverse_vocab[field] = {v: k for k, v in field_vocab.items()}

        print(f"📚 Vocabulary loaded: {list(self.vocab.keys())}")

    def _load_metadata(self):
        """Load CASAS metadata for sensor details."""
        if self.metadata_path is None:
            # Try to infer metadata path from vocab path
            # Assuming structure: data/processed/casas/{home}/vocab.json
            # Metadata is at: metadata/casas_metadata.json
            vocab_dir = os.path.dirname(self.vocab_path)
            if 'milan' in vocab_dir.lower():
                self.metadata_path = 'metadata/casas_metadata.json'
            elif 'aruba' in vocab_dir.lower():
                self.metadata_path = 'metadata/casas_metadata.json'
            else:
                print(f"⚠️  No metadata path provided and could not infer from vocab path")
                return

        if self.metadata_path and os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Extract sensor_details for the appropriate home
                # The metadata file has structure: {home: {sensor_details: {...}}}
                for home_name, home_data in metadata.items():
                    if isinstance(home_data, dict) and 'sensor_details' in home_data:
                        # Merge all sensor details (handling multiple homes)
                        details = home_data['sensor_details']
                        for sensor_id, description in details.items():
                            if description is not None:  # Only store sensors with actual details
                                self.sensor_details[sensor_id] = description

                special_count = len(self.sensor_details)
                print(f"🗂️  Loaded metadata: {special_count} sensors with special details")

            except Exception as e:
                print(f"⚠️  Could not load metadata from {self.metadata_path}: {e}")
                self.sensor_details = {}
        else:
            print(f"⚠️  Metadata file not found: {self.metadata_path}")

    def _load_captions(self):
        """Load captions from separate JSON file if provided."""
        if self.captions_path is None:
            print(f"⚠️  No captions file provided, will use sample indices as captions")
            return

        try:
            with open(self.captions_path, 'r') as f:
                captions_data = json.load(f)

            # Handle different caption file formats
            if isinstance(captions_data, dict):
                # Check if this is a wrapper dict with "captions" key containing the actual list
                if 'captions' in captions_data and isinstance(captions_data['captions'], list):
                    # Format: {"captions": [{"sample_id": ..., "captions": [...], ...}, ...]}
                    captions_list = captions_data['captions']
                    for item in captions_list:
                        if isinstance(item, dict) and 'sample_id' in item:
                            self.captions_dict[item['sample_id']] = {
                                'captions': item.get('captions', []),
                                'layer_b': item.get('metadata', {}).get('layer_b', '') or item.get('layer_b', ''),
                                'caption_type': item.get('metadata', {}).get('caption_type', '') or item.get('caption_type', ''),
                            }
                else:
                    # Format 1: Dictionary with sample_ids as keys
                    for sample_id, data in captions_data.items():
                        if isinstance(data, dict):
                            self.captions_dict[sample_id] = {
                                'captions': data.get('captions', []),
                                'layer_b': data.get('layer_b', ''),
                                'caption_type': data.get('caption_type', ''),
                            }
                        elif isinstance(data, list):
                            # Direct list of captions
                            self.captions_dict[sample_id] = {
                                'captions': data,
                                'layer_b': '',
                                'caption_type': '',
                            }
            elif isinstance(captions_data, list):
                # Format 2: List of items with sample_id field
                for item in captions_data:
                    if isinstance(item, dict) and 'sample_id' in item:
                        self.captions_dict[item['sample_id']] = {
                            'captions': item.get('captions', []),
                            'layer_b': item.get('metadata', {}).get('layer_b', '') or item.get('layer_b', ''),
                            'caption_type': item.get('metadata', {}).get('caption_type', '') or item.get('caption_type', ''),
                        }

            print(f"📖 Loaded {len(self.captions_dict)} captions from {self.captions_path}")

            # Debug: Show sample of loaded captions
            if len(self.captions_dict) > 0:
                sample_id = list(self.captions_dict.keys())[0]
                sample_data = self.captions_dict[sample_id]
                num_caps = len(sample_data.get('captions', []))
                if num_caps > 0:
                    first_cap = sample_data['captions'][0]
                    print(f"   Sample: {sample_id} has {num_caps} caption(s)")
                    print(f"   First caption: {first_cap[:80]}..." if len(first_cap) > 80 else f"   First caption: {first_cap}")
                else:
                    print(f"   Sample: {sample_id} -> {num_caps} captions")
            else:
                print(f"   ⚠️  Warning: No captions found in file. File might be empty or in wrong format.")
                print(f"   File structure: {list(captions_data.keys()) if isinstance(captions_data, dict) else 'list'}")

        except Exception as e:
            print(f"⚠️  Could not load captions from {self.captions_path}: {e}")
            import traceback
            traceback.print_exc()
            self.captions_dict = {}

    def _load_data(self):
        """Load test dataset."""
        self.test_dataset = SmartHomeDataset(
            data_path=self.test_data_path,
            vocab_path=self.vocab_path,
            sequence_length=20,
            max_captions=1
        )

        self.test_loader = create_data_loader(
            dataset=self.test_dataset,
            text_encoder=self.text_encoder,
            span_masker=None,
            vocab_sizes=self.vocab_sizes,
            device=self.device,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            apply_mlm=False
        )

        print(f"📊 Test dataset loaded: {len(self.test_dataset)} samples")

    def _extract_embeddings(self):
        """Extract embeddings from test dataset."""
        print(f"🔄 Extracting embeddings (max {self.max_samples} samples)...")

        sensor_embeddings = []
        text_embeddings = []
        self.raw_batches = []
        self.sample_indices = []

        samples_processed = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                if samples_processed >= self.max_samples:
                    break

                # Pack data for new encoder interface
                input_data = {
                    'categorical_features': batch['categorical_features'],
                    'coordinates': batch['coordinates'],
                    'time_deltas': batch['time_deltas']
                }
                sensor_emb = self.sensor_encoder.forward_clip(
                    input_data=input_data,
                    attention_mask=batch['mask']
                )

                text_emb = batch['text_embeddings']

                # Apply text projection if it exists
                if self.text_projection is not None:
                    text_emb = self.text_projection(text_emb)
                    text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=-1)

                sensor_embeddings.append(sensor_emb.cpu().numpy())
                text_embeddings.append(text_emb.cpu().numpy())
                self.raw_batches.append(batch)

                # Track sample indices
                batch_size = sensor_emb.shape[0]
                for i in range(batch_size):
                    if samples_processed >= self.max_samples:
                        break
                    self.sample_indices.append((batch_idx, i))
                    samples_processed += 1

        # Concatenate embeddings
        self.sensor_embeddings = np.concatenate(sensor_embeddings, axis=0)[:self.max_samples]
        self.text_embeddings = np.concatenate(text_embeddings, axis=0)[:self.max_samples]

        print(f"📈 Extracted {self.sensor_embeddings.shape[0]} embeddings")

    def _build_index(self):
        """Build FAISS index for fast retrieval."""
        d_model = self.sensor_embeddings.shape[1]
        self.sensor_index = faiss.IndexFlatIP(d_model)
        self.sensor_index.add(self.sensor_embeddings.astype(np.float32))

        print(f"🔍 Built FAISS index with {self.sensor_index.ntotal} vectors")

    def _decode_sequence(self, batch_idx: int, sample_idx: int, max_events: int = 10) -> tuple[List[Dict], Dict[str, Any]]:
        """Decode a sensor sequence from batch data."""
        batch = self.raw_batches[batch_idx]

        # Extract categorical features for this sample
        categorical_features = {}
        for field in batch['categorical_features']:
            categorical_features[field] = batch['categorical_features'][field][sample_idx]

        coordinates = batch['coordinates'][sample_idx]
        time_deltas = batch['time_deltas'][sample_idx]
        mask = batch['mask'][sample_idx]

        # Extract metadata
        metadata = {
            'valid_events': mask.sum().item(),
            'sequence_length': len(mask),
            'total_time_span': time_deltas[mask].sum().item() if mask.any() else 0.0
        }

        # Add day of week and other fields if available
        if 'delta_t_bucket' in categorical_features:
            delta_t_idx = categorical_features['delta_t_bucket'][0].item()
            metadata['delta_t_bucket'] = self.reverse_vocab.get('delta_t_bucket', {}).get(delta_t_idx, f"UNK_{delta_t_idx}")

        # Try to extract day of week if available in vocabulary
        if 'dow' in self.vocab:  # day of week
            if 'dow' in categorical_features:
                dow_idx = categorical_features['dow'][0].item()
                metadata['day_of_week'] = self.reverse_vocab['dow'].get(dow_idx, f"UNK_{dow_idx}")

        # Extract room sequence and transitions
        room_sequence = []
        for i in range(min(mask.sum().item(), 20)):
            if mask[i] and 'room_id' in categorical_features:
                room_idx = categorical_features['room_id'][i].item()
                room_name = self.reverse_vocab['room_id'].get(room_idx, f"UNK_{room_idx}")
                room_sequence.append(room_name)

        unique_rooms = list(set(room_sequence))
        metadata['rooms_visited'] = unique_rooms
        metadata['num_rooms'] = len(unique_rooms)
        metadata['room_sequence'] = room_sequence[:5]  # First 5 rooms

        # Extract time patterns
        tod_sequence = []
        for i in range(min(mask.sum().item(), 20)):
            if mask[i] and 'tod_bucket' in categorical_features:
                tod_idx = categorical_features['tod_bucket'][i].item()
                tod_name = self.reverse_vocab['tod_bucket'].get(tod_idx, f"UNK_{tod_idx}")
                tod_sequence.append(tod_name)

        unique_tods = list(set(tod_sequence))
        metadata['time_periods'] = unique_tods
        metadata['primary_time_period'] = tod_sequence[0] if tod_sequence else "unknown"

        # Decode events
        events = []
        valid_events = mask.sum().item()

        for i in range(min(valid_events, max_events)):
            if not mask[i]:
                break

            # Decode categorical features
            event = {}
            # Map common field names (handle both old and new naming conventions)
            field_mapping = {
                'sensor': 'sensor_id',
                'state': 'event_type',
                'room_id': 'room_id',
                'sensor_type': 'sensor_type',
                'sensor_detail': 'sensor_detail',
                'tod_bucket': 'tod_bucket',
                'time_delta_bucket': 'delta_t_bucket',
                'dow_bucket': 'day_of_week'
            }

            for vocab_field, event_field in field_mapping.items():
                if vocab_field in categorical_features:
                    idx_val = categorical_features[vocab_field][i].item()
                    decoded_val = self.reverse_vocab[vocab_field].get(idx_val, f"UNK_{idx_val}")
                    event[event_field] = decoded_val

            # Legacy support for old field names
            for field in ['sensor_id', 'room_id', 'event_type', 'sensor_type', 'tod_bucket', 'delta_t_bucket']:
                if field in categorical_features and field not in event:
                    idx_val = categorical_features[field][i].item()
                    decoded_val = self.reverse_vocab[field].get(idx_val, f"UNK_{idx_val}")
                    event[field] = decoded_val

            # Add day of week if available (handle both naming conventions)
            if 'dow_bucket' not in event and 'dow' in categorical_features:
                dow_idx = categorical_features['dow'][i].item()
                event['day_of_week'] = self.reverse_vocab.get('dow', {}).get(dow_idx, f"UNK_{dow_idx}")

            # Add coordinates and time
            event['x'] = coordinates[i][0].item()
            event['y'] = coordinates[i][1].item()
            event['time_delta'] = time_deltas[i].item()
            event['event_index'] = i

            events.append(event)

        return events, metadata

    def _get_caption_and_labels(self, batch_idx: int, sample_idx: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Get caption data and labels for a sample."""
        dataset_idx = batch_idx * 64 + sample_idx
        labels_info = {}
        caption_data = {'captions': [], 'layer_b': '', 'primary_caption': ''}

        if dataset_idx < len(self.test_dataset):
            try:
                original_sample = self.test_dataset.data[dataset_idx]

                # Get sample_id
                sample_id = original_sample.get('sample_id', f"sample_{dataset_idx}")
                labels_info['sample_id'] = sample_id

                # Get captions from the loaded captions dictionary
                if sample_id in self.captions_dict:
                    caption_info = self.captions_dict[sample_id]
                    caption_data['captions'] = caption_info.get('captions', [])
                    caption_data['layer_b'] = caption_info.get('layer_b', '')
                    caption_data['caption_type'] = caption_info.get('caption_type', '')
                    # Use first caption as primary
                    if caption_data['captions']:
                        caption_data['primary_caption'] = caption_data['captions'][0]

                # Fallback: check if captions are in the original sample data
                if not caption_data['captions'] and 'captions' in original_sample:
                    caption_data['captions'] = original_sample['captions']
                    if caption_data['captions']:
                        caption_data['primary_caption'] = caption_data['captions'][0]

                # Another fallback: check for 'caption' (singular) field
                if not caption_data['captions'] and 'caption' in original_sample:
                    caption_text = original_sample['caption']
                    if caption_text:
                        caption_data['captions'] = [caption_text]
                        caption_data['primary_caption'] = caption_text

                if not caption_data['primary_caption']:
                    caption_data['primary_caption'] = f"Sample {sample_id}"

                # Extract metadata from the sample
                if 'metadata' in original_sample:
                    metadata = original_sample['metadata']
                    labels_info['window_id'] = metadata.get('window_id')
                    labels_info['start_time'] = metadata.get('start_time')
                    labels_info['end_time'] = metadata.get('end_time')
                    labels_info['duration_seconds'] = metadata.get('duration_seconds')
                    labels_info['num_events'] = metadata.get('num_events')
                    labels_info['rooms_visited'] = metadata.get('rooms_visited', [])
                    labels_info['primary_room'] = metadata.get('primary_room')
                    labels_info['room_transitions'] = metadata.get('room_transitions')

                    # Extract ground truth labels from metadata
                    if 'ground_truth_labels' in metadata:
                        gt_labels = metadata['ground_truth_labels']
                        labels_info['activity_l1'] = gt_labels.get('primary_l1')
                        labels_info['activity_l2'] = gt_labels.get('primary_l2')
                        labels_info['all_labels_l1'] = gt_labels.get('all_labels_l1', [])
                        labels_info['all_labels_l2'] = gt_labels.get('all_labels_l2', [])

                # Fallback: try old format for backward compatibility
                if 'activity_l1' not in labels_info:
                    if 'activity' in original_sample:
                        labels_info['activity_l1'] = original_sample['activity']
                    elif 'first_activity' in original_sample:
                        labels_info['activity_l1'] = original_sample['first_activity']

                if 'activity_l2' not in labels_info:
                    if 'activity_l2' in original_sample:
                        labels_info['activity_l2'] = original_sample['activity_l2']
                    elif 'first_activity_l2' in original_sample:
                        labels_info['activity_l2'] = original_sample['first_activity_l2']

                # Fallback to inferred labels if ground truth not available
                if 'activity_l1' not in labels_info or not labels_info['activity_l1']:
                    cap = caption_data['primary_caption'].lower()
                    if "kitchen" in cap and ("cook" in cap or "stove" in cap):
                        labels_info['activity_l1'] = "cooking"
                    elif "bathroom" in cap or "toilet" in cap:
                        labels_info['activity_l1'] = "bathroom"
                    elif "bedroom" in cap or "bed" in cap:
                        labels_info['activity_l1'] = "bedroom"
                    elif "night" in cap and ("wander" in cap or "restless" in cap):
                        labels_info['activity_l1'] = "night_wandering"
                    elif "morning" in cap:
                        labels_info['activity_l1'] = "morning_routine"
                    else:
                        labels_info['activity_l1'] = "unknown"

                return caption_data, labels_info

            except Exception as e:
                error_caption = {'captions': [], 'layer_b': '', 'primary_caption': f"Sample {dataset_idx}"}
                return error_caption, {'error': str(e)}

        empty_caption = {'captions': [], 'layer_b': '', 'primary_caption': f"Batch {batch_idx}, Sample {sample_idx}"}
        return empty_caption, {}

    def query(self, query_text: str, top_k: int = 5, show_details: bool = True) -> List[Dict]:
        """
        Query the retrieval system with arbitrary text.

        Args:
            query_text: Natural language query (e.g., "cooking", "night wandering")
            top_k: Number of top results to return
            show_details: Whether to show detailed sensor sequences

        Returns:
            List of retrieval results with scores and details
        """
        print(f"\n🔍 QUERY: '{query_text}'")
        print("=" * 60)

        # Encode query
        with torch.no_grad():
            query_emb = self.text_encoder.encode_texts_clip([query_text], self.device)

            # Apply text projection if it exists
            if self.text_projection is not None:
                query_emb = self.text_projection(query_emb)
                query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=-1)

        query_emb = query_emb.cpu().numpy().astype(np.float32)

        # Search
        scores, indices = self.sensor_index.search(query_emb, k=top_k)

        results = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            # Get batch and sample info
            batch_idx, sample_idx = self.sample_indices[idx]

            # Get caption data and labels
            caption_data, labels_info = self._get_caption_and_labels(batch_idx, sample_idx)

            # Debug: Show caption and event data for first result
            if rank == 0 and self.debug:
                print(f"[DEBUG] Caption data: {caption_data}")
                print(f"[DEBUG] Labels info keys: {labels_info.keys()}")
                print(f"[DEBUG] First event fields: {events[0].keys() if events else 'No events'}")
                if events:
                    print(f"[DEBUG] First 3 events: {events[:3]}")

            # Decode sequence and get metadata
            events, metadata = self._decode_sequence(batch_idx, sample_idx)

            result = {
                'rank': rank + 1,
                'score': float(score),
                'caption_data': caption_data,
                'events': events,
                'metadata': metadata,
                'labels': labels_info,
                'batch_idx': batch_idx,
                'sample_idx': sample_idx
            }

            results.append(result)

            # Display result - COMPACT FORMAT
            # Rank line with sample ID
            sample_id = labels_info.get('sample_id', f"sample_{idx}")
            print(f"\n🏆 RANK {rank + 1} | Score: {score:.4f} | {sample_id}")
            print("-" * 80)

            # CASAS Labels (Ground Truth)
            if labels_info:
                labels_parts = []
                if 'activity_l1' in labels_info and labels_info['activity_l1']:
                    labels_parts.append(f"L1: {labels_info['activity_l1']}")
                if 'activity_l2' in labels_info and labels_info['activity_l2']:
                    labels_parts.append(f"L2: {labels_info['activity_l2']}")
                if labels_parts:
                    print(f"🏷️  CASAS Labels: {' | '.join(labels_parts)}")
                    if 'all_labels_l1' in labels_info and len(labels_info['all_labels_l1']) > 1:
                        print(f"   All L1: {', '.join(labels_info['all_labels_l1'])}")

            # Time info (duration, time of day, day of week)
            time_parts = []
            if 'num_events' in labels_info:
                time_parts.append(f"{labels_info['num_events']} events")
            if 'duration_seconds' in labels_info:
                time_parts.append(f"{labels_info['duration_seconds']:.1f}s duration")

            # Get time of day and day of week from events
            tod_str = None
            dow_str = None
            if events:
                first_event = events[0]
                if 'tod_bucket' in first_event:
                    tod_str = first_event['tod_bucket']
                if 'day_of_week' in first_event:
                    dow_str = first_event['day_of_week']

            if tod_str and tod_str != 'UNK':
                time_parts.append(f"ToD: {tod_str}")
            if dow_str and dow_str not in ['UNK', 'unknown']:
                time_parts.append(f"DoW: {dow_str}")

            # Format start time to just hours and minutes
            if 'start_time' in labels_info and labels_info['start_time']:
                start_time_str = labels_info['start_time']
                # Parse and format time (handle format like "2009-12-15 15:05:24.000033")
                try:
                    from datetime import datetime
                    # Try parsing with microseconds
                    if '.' in start_time_str:
                        dt = datetime.strptime(start_time_str.split('.')[0], '%Y-%m-%d %H:%M:%S')
                    else:
                        dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                    time_parts.append(f"Start: {dt.strftime('%Y-%m-%d %H:%M')}")
                except:
                    # Fallback if parsing fails
                    time_parts.append(f"Start: {start_time_str}")

            if time_parts:
                print(f"⏰ {' | '.join(time_parts)}")

            # Location info
            location_parts = []
            if 'primary_room' in labels_info and labels_info['primary_room']:
                location_parts.append(f"Primary: {labels_info['primary_room']}")

            if 'rooms_visited' in labels_info and labels_info['rooms_visited']:
                rooms = labels_info['rooms_visited']
                if len(rooms) <= 3:
                    location_parts.append(f"Visited: {', '.join(rooms)}")
                else:
                    location_parts.append(f"Visited: {', '.join(rooms[:3])} +{len(rooms)-3} more")

            if 'room_transitions' in labels_info:
                location_parts.append(f"{labels_info['room_transitions']} transitions")

            if location_parts:
                print(f"📍 {' | '.join(location_parts)}")

            # Special sensors (only show sensors with metadata details)
            special_sensors = []
            sensor_seen = set()

            for event in events:
                # Try multiple field names for sensor ID
                sensor_id = event.get('sensor_id') or event.get('sensor_detail') or event.get('sensor')
                event_type = event.get('event_type') or event.get('state')

                # Only show sensors that have special details in metadata
                if sensor_id and sensor_id not in sensor_seen and sensor_id != 'UNK':
                    if sensor_id in self.sensor_details:
                        sensor_seen.add(sensor_id)
                        # Get the description
                        description = self.sensor_details[sensor_id]

                        # Include state if meaningful (ON/OPEN/PRESENT)
                        if event_type and event_type.upper() in ['ON', 'OPEN', 'PRESENT']:
                            special_sensors.append(f"{sensor_id} ({description})")
                        else:
                            special_sensors.append(f"{sensor_id} ({description})")

            if special_sensors:
                # Show up to 5 special sensors
                special_list = special_sensors[:5]
                print(f"🔧 Special sensors: {', '.join(special_list)}")

            # Caption at the end - show actual caption text if available
            caption_text = None
            if caption_data.get('captions') and len(caption_data['captions']) > 0:
                # Get first non-empty caption
                for cap in caption_data['captions']:
                    if cap and cap.strip() and not cap.startswith('Sample '):
                        caption_text = cap
                        break

            # Fallback to layer_b if no caption found
            if not caption_text and caption_data.get('layer_b'):
                caption_text = caption_data['layer_b']

            # Always show caption if we have something meaningful
            if caption_text:
                # Truncate long captions
                if len(caption_text) > 150:
                    caption_text = caption_text[:147] + "..."
                print(f"📝 Caption: {caption_text}")

        return results

    def analyze_query_results(self, query_text: str, results: List[Dict]):
        """Analyze patterns in query results."""
        print(f"\n📊 ANALYSIS FOR '{query_text}':")
        print("-" * 40)

        # Room analysis
        all_rooms = []
        all_tod = []
        all_sensors = []

        for result in results:
            for event in result['events']:
                if 'room_id' in event:
                    all_rooms.append(event['room_id'])
                if 'tod_bucket' in event:
                    all_tod.append(event['tod_bucket'])
                # Handle both sensor_id and sensor_detail fields
                sensor = event.get('sensor_id') or event.get('sensor_detail')
                if sensor:
                    all_sensors.append(sensor)

        # Top patterns
        from collections import Counter

        room_counts = Counter(all_rooms)
        tod_counts = Counter(all_tod)
        sensor_counts = Counter(all_sensors)

        print(f"  Top rooms: {dict(room_counts.most_common(3))}")
        print(f"  Time periods: {dict(tod_counts.most_common(3))}")
        print(f"  Top sensors: {dict(sensor_counts.most_common(3))}")
        print(f"  Score range: {min(r['score'] for r in results):.3f} - {max(r['score'] for r in results):.3f}")


def main():
    parser = argparse.ArgumentParser(description='Query smart-home retrieval system')
    parser.add_argument('--checkpoint', type=str,
                       default='trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str,
                       default='data/processed/casas/milan/FD_60/test.json',
                       help='Path to test data')
    parser.add_argument('--vocab', type=str,
                       default='data/processed/casas/milan/FD_60/vocab.json',
                       help='Path to vocabulary file')
    parser.add_argument('--captions', type=str,
                       default='data/processed/casas/milan/FD_60/test_captions_baseline.json',
                       help='Path to captions JSON file')
    parser.add_argument('--metadata', type=str,
                       default='metadata/casas_metadata.json',
                       help='Path to CASAS metadata file with sensor details')
    parser.add_argument('--query', type=str, help='Query text (e.g., "cooking", "night wandering")')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results')
    parser.add_argument('--max_samples', type=int, default=5000, help='Max samples to process')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    # Initialize retrieval system
    print("🚀 Initializing Smart-Home Retrieval System")
    retrieval = SmartHomeRetrieval(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        test_data_path=args.test_data,
        captions_path=args.captions,
        max_samples=args.max_samples,
        debug=args.debug,
        metadata_path=args.metadata
    )

    print(f"\n✅ System ready! Loaded {retrieval.sensor_index.ntotal} sequences")

    # Interactive mode
    if args.interactive:
        print("\n🎯 INTERACTIVE MODE")
        print("Enter queries like: 'cooking', 'night wandering', 'bathroom activity', 'morning routine'")
        print("Type 'quit' to exit")

        while True:
            query = input("\n🔍 Query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue

            results = retrieval.query(query, top_k=args.top_k)
            retrieval.analyze_query_results(query, results)

    # Single query mode
    elif args.query:
        results = retrieval.query(args.query, top_k=args.top_k)
        retrieval.analyze_query_results(args.query, results)

    # Demo mode
    else:
        print("\n🎯 DEMO MODE - Testing common queries")
        demo_queries = [
            "cooking",
            "night wandering",
            "bathroom activity",
            "morning routine",
            "kitchen activity",
            "bedroom activity",
            "restless movement"
        ]

        for query in demo_queries:
            results = retrieval.query(query, top_k=5, show_details=False)
            retrieval.analyze_query_results(query, results)
            print()


if __name__ == "__main__":
    main()
