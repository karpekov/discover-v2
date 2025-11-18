#!/usr/bin/env python3
"""
General retrieval script for querying the trained model with arbitrary text queries.
This is the main evaluation tool for the trained dual-encoder model.
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

    def __init__(self, checkpoint_path: str, vocab_path: str, test_data_path: str, max_samples: int = 5000):
        self.device = get_optimal_device()
        self.checkpoint_path = checkpoint_path
        self.vocab_path = vocab_path
        self.test_data_path = test_data_path
        self.max_samples = max_samples

        # Load models and data
        self._load_models()
        self._load_vocab()
        self._load_data()
        self._extract_embeddings()
        self._build_index()

    def _load_models(self):
        """Load trained models from checkpoint."""
        print(f"🔄 Loading models from {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Get config from checkpoint
        config = checkpoint['config']
        self.vocab_sizes = checkpoint['vocab_sizes']

        # Text encoder (frozen)
        # Use text encoder factory with default GTE

        eval_config = {"text_model_name": "thenlper/gte-base", "use_cached_embeddings": False}

        self.text_encoder = build_text_encoder(eval_config)
        self.text_encoder.to(self.device)

        # Sensor encoder with correct parameters from checkpoint
        self.sensor_encoder = SensorEncoder(
            vocab_sizes=self.vocab_sizes,
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            max_seq_len=config.get('max_seq_len', 512),
            dropout=config.get('dropout', 0.1),
            fourier_bands=config.get('fourier_bands', 12),
            use_rope_time=config.get('use_rope_time', False),
            use_rope_2d=config.get('use_rope_2d', False)
        )
        self.sensor_encoder.load_state_dict(checkpoint['sensor_encoder_state_dict'])
        self.sensor_encoder.to(self.device)
        self.sensor_encoder.eval()

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
            for field in ['sensor_id', 'room_id', 'event_type', 'sensor_type', 'tod_bucket', 'delta_t_bucket']:
                if field in categorical_features:
                    idx_val = categorical_features[field][i].item()
                    decoded_val = self.reverse_vocab[field].get(idx_val, f"UNK_{idx_val}")
                    event[field] = decoded_val

            # Add day of week if available
            if 'dow' in categorical_features:
                dow_idx = categorical_features['dow'][i].item()
                event['day_of_week'] = self.reverse_vocab.get('dow', {}).get(dow_idx, f"UNK_{dow_idx}")

            # Add coordinates and time
            event['x'] = coordinates[i][0].item()
            event['y'] = coordinates[i][1].item()
            event['time_delta'] = time_deltas[i].item()
            event['event_index'] = i

            events.append(event)

        return events, metadata

    def _get_caption_and_labels(self, batch_idx: int, sample_idx: int) -> tuple[str, Dict[str, Any]]:
        """Get caption and any available labels for a sample."""
        dataset_idx = batch_idx * 64 + sample_idx
        labels_info = {}

        if dataset_idx < len(self.test_dataset):
            try:
                original_sample = self.test_dataset.data[dataset_idx]
                caption = original_sample['captions'][0] if original_sample['captions'] else "No caption"

                # Extract ground truth labels if available
                if 'first_activity' in original_sample:
                    labels_info['ground_truth_activity'] = original_sample['first_activity']

                if 'first_activity_l2' in original_sample:
                    labels_info['ground_truth_activity_l2'] = original_sample['first_activity_l2']

                # Fallback to inferred labels if ground truth not available
                if 'ground_truth_activity' not in labels_info:
                    if "kitchen" in caption.lower() and ("cook" in caption.lower() or "stove" in caption.lower()):
                        labels_info['inferred_activity'] = "cooking"
                    elif "bathroom" in caption.lower() or "toilet" in caption.lower():
                        labels_info['inferred_activity'] = "bathroom"
                    elif "bedroom" in caption.lower() or "bed" in caption.lower():
                        labels_info['inferred_activity'] = "bedroom"
                    elif "night" in caption.lower() and ("wander" in caption.lower() or "restless" in caption.lower()):
                        labels_info['inferred_activity'] = "night_wandering"
                    elif "morning" in caption.lower():
                        labels_info['inferred_activity'] = "morning_routine"
                    else:
                        labels_info['inferred_activity'] = "general_activity"

                # Extract day and time from caption if available
                if "Monday" in caption:
                    labels_info['caption_day'] = "Monday"
                elif "Tuesday" in caption:
                    labels_info['caption_day'] = "Tuesday"
                elif "Wednesday" in caption:
                    labels_info['caption_day'] = "Wednesday"
                elif "Thursday" in caption:
                    labels_info['caption_day'] = "Thursday"
                elif "Friday" in caption:
                    labels_info['caption_day'] = "Friday"
                elif "Saturday" in caption:
                    labels_info['caption_day'] = "Saturday"
                elif "Sunday" in caption:
                    labels_info['caption_day'] = "Sunday"

                # Extract time period from caption
                if "morning" in caption.lower():
                    labels_info['caption_time'] = "morning"
                elif "evening" in caption.lower():
                    labels_info['caption_time'] = "evening"
                elif "night" in caption.lower():
                    labels_info['caption_time'] = "night"
                elif "afternoon" in caption.lower():
                    labels_info['caption_time'] = "afternoon"

                return caption, labels_info

            except Exception as e:
                return f"Sample {dataset_idx}", {'error': str(e)}

        return f"Batch {batch_idx}, Sample {sample_idx}", {}

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
        query_emb = query_emb.cpu().numpy().astype(np.float32)

        # Search
        scores, indices = self.sensor_index.search(query_emb, k=top_k)

        results = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            # Get batch and sample info
            batch_idx, sample_idx = self.sample_indices[idx]

            # Get caption and labels
            caption, labels_info = self._get_caption_and_labels(batch_idx, sample_idx)

            # Decode sequence and get metadata
            events, metadata = self._decode_sequence(batch_idx, sample_idx)

            result = {
                'rank': rank + 1,
                'score': float(score),
                'caption': caption,
                'events': events,
                'metadata': metadata,
                'labels': labels_info,
                'batch_idx': batch_idx,
                'sample_idx': sample_idx
            }

            results.append(result)

            # Display result
            print(f"\n🏆 RANK {rank + 1} | Score: {score:.4f}")
            print("-" * 60)
            print(f"📝 Caption: {caption}")

            # Display labels and activity info
            if labels_info:
                print(f"🏷️  Labels:")
                if 'ground_truth_activity' in labels_info:
                    print(f"    Ground Truth Activity: {labels_info['ground_truth_activity']}")
                if 'ground_truth_activity_l2' in labels_info:
                    print(f"    Ground Truth Activity L2: {labels_info['ground_truth_activity_l2']}")
                if 'inferred_activity' in labels_info:
                    print(f"    Inferred Activity: {labels_info['inferred_activity']}")
                if 'caption_day' in labels_info:
                    print(f"    Day (from caption): {labels_info['caption_day']}")
                if 'caption_time' in labels_info:
                    print(f"    Time (from caption): {labels_info['caption_time']}")
                if 'error' in labels_info:
                    print(f"    Error: {labels_info['error']}")

            # Display metadata
            print(f"📊 Metadata:")
            print(f"    Events: {metadata['valid_events']}/{metadata['sequence_length']}")
            print(f"    Duration: {metadata['total_time_span']:.1f}s")
            print(f"    Rooms: {metadata['rooms_visited']} ({metadata['num_rooms']} total)")
            print(f"    Time period: {metadata['primary_time_period']}")
            print(f"    Room path: {' → '.join(metadata['room_sequence'])}")

            if 'day_of_week' in metadata:
                print(f"    Day of week (sensor): {metadata['day_of_week']}")
            if 'delta_t_bucket' in metadata:
                print(f"    Time delta: {metadata['delta_t_bucket']}")

            if show_details:
                print(f"🔧 Sensor Sequence ({len(events)} events):")
                for i, event in enumerate(events):
                    parts = []
                    for field in ['sensor_id', 'room_id', 'event_type', 'tod_bucket']:
                        if field in event:
                            parts.append(f"{field}={event[field]}")

                    # Add day of week if available
                    if 'day_of_week' in event:
                        parts.append(f"dow={event['day_of_week']}")

                    coords = f"({event['x']:.2f},{event['y']:.2f})"
                    parts.append(f"coords={coords}")

                    if 'time_delta' in event:
                        parts.append(f"Δt={event['time_delta']:.1f}s")

                    print(f"    Event {i+1:2d}: {' | '.join(parts)}")

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
                if 'sensor_id' in event:
                    all_sensors.append(event['sensor_id'])

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
    parser.add_argument('--checkpoint', type=str, default='./outputs/milan_training/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str,
                       default='../data/processed_experiments/experiment_milan_training/milan/milan_test.json',
                       help='Path to test data')
    parser.add_argument('--vocab', type=str,
                       default='../data/processed_experiments/experiment_milan_training/milan/milan_vocab.json',
                       help='Path to vocabulary file')
    parser.add_argument('--query', type=str, help='Query text (e.g., "cooking", "night wandering")')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top results')
    parser.add_argument('--max_samples', type=int, default=5000, help='Max samples to process')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    # Initialize retrieval system
    print("🚀 Initializing Smart-Home Retrieval System")
    retrieval = SmartHomeRetrieval(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        test_data_path=args.test_data,
        max_samples=args.max_samples
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
            results = retrieval.query(query, top_k=2, show_details=False)
            retrieval.analyze_query_results(query, results)
            print()


if __name__ == "__main__":
    main()
