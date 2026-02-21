#!/usr/bin/env python3

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
import faiss
from sklearn.metrics import ndcg_score

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.text_encoder import TextEncoder, build_text_encoder
from models.sensor_encoder import SensorEncoder
from models.mlm_heads import MLMHeads
from dataio.dataset import SmartHomeDataset
from dataio.collate import create_data_loader


class RetrievalEvaluator:
  """
  Evaluator for text-to-sensor and sensor-to-text retrieval.
  Uses FAISS for efficient similarity search.
  """

  def __init__(self, config: Dict[str, Any]):
    self.config = config
    self.device = torch.device(config['device'])

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    self.logger = logging.getLogger(__name__)

    # Load models
    self.load_models()

    # Setup data
    self.setup_data()

    # Initialize FAISS indices
    self.sensor_index = None
    self.text_index = None

  def load_models(self):
    """Load trained models from checkpoint."""
    checkpoint_path = self.config['checkpoint_path']
    self.logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Set up module namespace for backwards compatibility
    import sys
    import types

    if 'src' not in sys.modules:
      src_module = types.ModuleType('src')
      src_module.__path__ = []
      sys.modules['src'] = src_module

    # Import and map necessary modules
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

    # Load the full alignment model
    from alignment.model import AlignmentModel

    full_model = AlignmentModel.load(
      checkpoint_path,
      device=self.device,
      vocab_path=self.config['vocab_path']
    )

    # Extract components
    self.sensor_encoder = full_model.sensor_encoder
    self.text_projection = full_model.text_projection
    self.config_obj = full_model.config
    self.vocab_sizes = full_model.vocab_sizes

    # Sensor encoder is already loaded and on device
    self.sensor_encoder.eval()

    # Load checkpoint again for text encoder
    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

    # Use robust text encoder detection
    from evals.eval_utils import create_text_encoder_from_checkpoint
    train_data_path = None
    if hasattr(self.config_obj, 'train_text_embeddings_path'):
      train_data_path = self.config_obj.train_text_embeddings_path

    self.text_encoder = create_text_encoder_from_checkpoint(
      checkpoint=checkpoint,
      device=self.device,
      data_path=train_data_path
    )

    if self.text_encoder is None:
      # Ultimate fallback
      from models.text_encoder import TextEncoder
      text_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
      self.text_encoder = TextEncoder(text_model_name)
      self.logger.warning(f"Using default text encoder: {text_model_name}")

    self.text_encoder.to(self.device)
    self.text_encoder.eval()

    # Text projection
    if self.text_projection is not None:
      self.text_projection.eval()
      self.logger.info("Text projection loaded from checkpoint")

    # Get d_model from encoder config
    if hasattr(self.config_obj, 'encoder'):
      encoder_config = self.config_obj.encoder
      if hasattr(encoder_config, 'd_model'):
        d_model = encoder_config.d_model
        n_layers = encoder_config.n_layers
      else:
        d_model = encoder_config.get('d_model', 768)
        n_layers = encoder_config.get('n_layers', 6)
    else:
      d_model = self.config.get('d_model', 768)
      n_layers = self.config.get('n_layers', 6)

    self.logger.info(f"Models loaded successfully (d_model={d_model}, n_layers={n_layers})")

  def setup_data(self):
    """Setup evaluation dataset."""
    dataset = SmartHomeDataset(
      data_path=self.config['eval_data_path'],
      vocab_path=self.config['vocab_path'],
      sequence_length=self.config.get('sequence_length', 20),
      max_captions=self.config.get('max_captions', 3)
    )

    # Create data loader without MLM masking
    self.eval_loader = create_data_loader(
      dataset=dataset,
      text_encoder=self.text_encoder,
      span_masker=None,  # No masking during evaluation
      vocab_sizes=self.vocab_sizes,
      device=self.device,
      batch_size=self.config.get('batch_size', 64),
      shuffle=False,
      num_workers=self.config.get('num_workers', 4),
      apply_mlm=False
    )

    self.logger.info(f"Setup evaluation data: {len(self.eval_loader)} batches")

  def extract_embeddings(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
    """
    Extract sensor and text embeddings from the evaluation dataset.

    Returns:
      sensor_embeddings: [N, d_model] numpy array
      text_embeddings: [M, d_model] numpy array
      captions: List of caption strings
      metadata: List of metadata dictionaries for each sensor sequence
    """
    sensor_embeddings = []
    text_embeddings = []
    captions = []
    metadata = []

    self.logger.info("Extracting embeddings...")

    with torch.no_grad():
      for batch_idx, batch in enumerate(self.eval_loader):
        # Get sensor embeddings using forward_clip (applies projection + normalization)
        input_data = {
          'categorical_features': batch['categorical_features'],
          'coordinates': batch['coordinates'],
          'time_deltas': batch['time_deltas']
        }
        sensor_emb = self.sensor_encoder.forward_clip(
          input_data=input_data,
          attention_mask=batch['mask']
        )

        # Get text embeddings (already computed in collate)
        text_emb = batch['text_embeddings']

        # Apply text projection if it exists
        if self.text_projection is not None:
          text_emb = self.text_projection(text_emb)
          text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=-1)

        # Convert to numpy and store
        sensor_embeddings.append(sensor_emb.cpu().numpy())
        text_embeddings.append(text_emb.cpu().numpy())

        # Store captions and metadata
        batch_size = sensor_emb.shape[0]
        for i in range(batch_size):
          # For now, we'll create simple metadata
          # In a real implementation, you'd extract this from the dataset
          metadata.append({
            'batch_idx': batch_idx,
            'sample_idx': i,
            'sequence_length': batch['mask'][i].sum().item()
          })

        # Note: We're assuming one caption per sample for simplicity
        # In practice, you might need to handle multiple captions per sample

        if batch_idx % 10 == 0:
          self.logger.info(f"Processed {batch_idx + 1}/{len(self.eval_loader)} batches")

    # Concatenate all embeddings
    sensor_embeddings = np.concatenate(sensor_embeddings, axis=0)
    text_embeddings = np.concatenate(text_embeddings, axis=0)

    # For now, create dummy captions since we don't have direct access
    # In a real implementation, you'd extract these from the dataset
    captions = [f"caption_{i}" for i in range(text_embeddings.shape[0])]

    self.logger.info(f"Extracted {sensor_embeddings.shape[0]} sensor embeddings and {text_embeddings.shape[0]} text embeddings")

    return sensor_embeddings, text_embeddings, captions, metadata

  def build_faiss_indices(self, sensor_embeddings: np.ndarray, text_embeddings: np.ndarray):
    """Build FAISS indices for efficient retrieval."""
    d_model = sensor_embeddings.shape[1]

    # Build sensor index
    self.sensor_index = faiss.IndexFlatIP(d_model)  # Inner product for cosine similarity
    self.sensor_index.add(sensor_embeddings.astype(np.float32))

    # Build text index
    self.text_index = faiss.IndexFlatIP(d_model)
    self.text_index.add(text_embeddings.astype(np.float32))

    self.logger.info(f"Built FAISS indices: sensor={self.sensor_index.ntotal}, text={self.text_index.ntotal}")

  def text_to_sensor_retrieval(
    self,
    query_texts: List[str],
    sensor_embeddings: np.ndarray,
    metadata: List[Dict],
    k: int = 10
  ) -> List[List[Dict]]:
    """
    Retrieve sensor sequences given text queries.

    Args:
      query_texts: List of query text strings
      sensor_embeddings: Sensor embeddings array
      metadata: Metadata for each sensor sequence
      k: Number of results to retrieve

    Returns:
      results: List of lists, each containing top-k retrieved sensor sequences
    """
    # Encode query texts using the CLIP method
    query_embeddings = self.text_encoder.encode_texts_clip(query_texts, self.device)

    # Apply text projection if it exists
    if self.text_projection is not None:
      query_embeddings = self.text_projection(query_embeddings)
      query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)

    query_embeddings = query_embeddings.cpu().numpy().astype(np.float32)

    # Search in sensor index
    scores, indices = self.sensor_index.search(query_embeddings, k)

    results = []
    for i, (query_scores, query_indices) in enumerate(zip(scores, indices)):
      query_results = []
      for score, idx in zip(query_scores, query_indices):
        if idx < len(metadata):  # Valid index
          result = {
            'score': float(score),
            'metadata': metadata[idx],
            'sensor_embedding': sensor_embeddings[idx]
          }
          query_results.append(result)
      results.append(query_results)

    return results

  def sensor_to_text_retrieval(
    self,
    sensor_indices: List[int],
    text_embeddings: np.ndarray,
    captions: List[str],
    k: int = 10
  ) -> List[List[Dict]]:
    """
    Retrieve text captions given sensor sequence indices.

    Args:
      sensor_indices: List of sensor sequence indices to query
      text_embeddings: Text embeddings array
      captions: List of caption strings
      k: Number of results to retrieve

    Returns:
      results: List of lists, each containing top-k retrieved captions
    """
    # Get sensor embeddings for the specified indices
    sensor_embs = []
    for idx in sensor_indices:
      if 0 <= idx < self.sensor_index.ntotal:
        # Reconstruct sensor embedding from FAISS index
        sensor_emb = self.sensor_index.reconstruct(idx)
        sensor_embs.append(sensor_emb)
      else:
        # Invalid index - create zero embedding
        sensor_embs.append(np.zeros(text_embeddings.shape[1], dtype=np.float32))

    sensor_embs = np.array(sensor_embs)

    # Search in text index
    scores, indices = self.text_index.search(sensor_embs, k)

    results = []
    for i, (query_scores, query_indices) in enumerate(zip(scores, indices)):
      query_results = []
      for score, idx in zip(query_scores, query_indices):
        if idx < len(captions):  # Valid index
          result = {
            'score': float(score),
            'caption': captions[idx],
            'text_embedding': text_embeddings[idx]
          }
          query_results.append(result)
      results.append(query_results)

    return results

  def compute_retrieval_metrics(
    self,
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    ground_truth_indices: List[int],
    k_values: List[int] = [1, 5, 10]
  ) -> Dict[str, float]:
    """
    Compute retrieval metrics (Precision@k, nDCG@k).

    Args:
      query_embeddings: Query embeddings [N, d_model]
      target_embeddings: Target embeddings [M, d_model]
      ground_truth_indices: Ground truth target indices for each query
      k_values: List of k values to compute metrics for

    Returns:
      metrics: Dictionary of metric values
    """
    # Compute similarity matrix
    similarity_matrix = np.dot(query_embeddings, target_embeddings.T)

    metrics = {}

    for k in k_values:
      precisions = []
      ndcg_scores = []

      for i, gt_idx in enumerate(ground_truth_indices):
        # Get top-k predictions
        top_k_indices = np.argsort(similarity_matrix[i])[-k:][::-1]

        # Precision@k
        precision = 1.0 if gt_idx in top_k_indices else 0.0
        precisions.append(precision)

        # nDCG@k
        relevance_scores = np.zeros(len(target_embeddings))
        relevance_scores[gt_idx] = 1.0
        predicted_relevance = relevance_scores[top_k_indices]

        if len(predicted_relevance) > 0:
          ndcg = ndcg_score([relevance_scores], [similarity_matrix[i]], k=k)
          ndcg_scores.append(ndcg)

      metrics[f'precision@{k}'] = np.mean(precisions)
      if ndcg_scores:
        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)

    return metrics

  def demo_retrieval(self, sensor_embeddings: np.ndarray, text_embeddings: np.ndarray, captions: List[str], metadata: List[Dict]):
    """Run a demo of the retrieval system."""
    self.logger.info("Running retrieval demo...")

    # Sample query texts
    demo_queries = [
      "night wandering",
      "morning routine",
      "cooking dinner",
      "watching TV",
      "bathroom activity",
      "restless movement"
    ]

    # Text-to-sensor retrieval
    self.logger.info("\n=== Text-to-Sensor Retrieval ===")
    results = self.text_to_sensor_retrieval(demo_queries, sensor_embeddings, metadata, k=5)

    for i, (query, query_results) in enumerate(zip(demo_queries, results)):
      self.logger.info(f"\n🔍 Query: '{query}'")
      self.logger.info("=" * 80)
      for j, result in enumerate(query_results[:5]):  # Show top 5
        self.logger.info(f"\n🏆 RANK {j+1} | Score: {result['score']:.4f}")
        self.logger.info("-" * 80)

        meta = result['metadata']
        self.logger.info(f"📊 Sample {meta.get('batch_idx', '?')}:{meta.get('sample_idx', '?')}")

        # Metadata info
        meta_parts = []
        if 'sequence_length' in meta:
          meta_parts.append(f"{meta['sequence_length']} valid events")
        if 'total_time_span' in meta:
          meta_parts.append(f"{meta['total_time_span']:.1f}s duration")
        if 'num_rooms' in meta:
          meta_parts.append(f"{meta['num_rooms']} rooms")

        if meta_parts:
          self.logger.info(f"⏰ {' | '.join(meta_parts)}")

        # Location
        if 'rooms_visited' in meta and meta['rooms_visited']:
          rooms = meta['rooms_visited']
          if len(rooms) <= 3:
            self.logger.info(f"📍 Rooms: {', '.join(rooms)}")
          else:
            self.logger.info(f"📍 Rooms: {', '.join(rooms[:3])} +{len(rooms)-3} more")

        # Time period
        if 'time_periods' in meta and meta['time_periods']:
          tod = [t for t in meta['time_periods'] if t != 'UNK']
          if tod:
            self.logger.info(f"🕐 Time of day: {', '.join(tod)}")

    # Sensor-to-text retrieval (using random sensor indices)
    self.logger.info("\n\n=== Sensor-to-Text Retrieval ===")
    random_sensor_indices = np.random.choice(len(metadata), size=3, replace=False).tolist()
    results = self.sensor_to_text_retrieval(random_sensor_indices, text_embeddings, captions, k=5)

    for i, (sensor_idx, query_results) in enumerate(zip(random_sensor_indices, results)):
      self.logger.info(f"\n🔍 Sensor sequence {sensor_idx}:")
      self.logger.info("=" * 80)
      for j, result in enumerate(query_results[:5]):  # Show top 5
        self.logger.info(f"  {j+1}. Score: {result['score']:.4f} | Caption: '{result['caption']}'")

  def evaluate(self):
    """Run full evaluation."""
    # Extract embeddings
    sensor_embeddings, text_embeddings, captions, metadata = self.extract_embeddings()

    # Build FAISS indices
    self.build_faiss_indices(sensor_embeddings, text_embeddings)

    # Run demo
    if self.config.get('run_demo', True):
      self.demo_retrieval(sensor_embeddings, text_embeddings, captions, metadata)

    # Compute metrics (simplified - assumes paired data)
    if len(sensor_embeddings) == len(text_embeddings):
      self.logger.info("\n=== Retrieval Metrics ===")

      # Text-to-sensor metrics
      ground_truth = list(range(len(sensor_embeddings)))
      t2s_metrics = self.compute_retrieval_metrics(
        text_embeddings, sensor_embeddings, ground_truth
      )

      # Sensor-to-text metrics
      s2t_metrics = self.compute_retrieval_metrics(
        sensor_embeddings, text_embeddings, ground_truth
      )

      self.logger.info("Text-to-Sensor:")
      for metric, value in t2s_metrics.items():
        self.logger.info(f"  {metric}: {value:.4f}")

      self.logger.info("Sensor-to-Text:")
      for metric, value in s2t_metrics.items():
        self.logger.info(f"  {metric}: {value:.4f}")

      # Save metrics
      if self.config.get('output_dir'):
        metrics_path = os.path.join(self.config['output_dir'], 'retrieval_metrics.json')
        all_metrics = {
          'text_to_sensor': t2s_metrics,
          'sensor_to_text': s2t_metrics
        }
        with open(metrics_path, 'w') as f:
          json.dump(all_metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {metrics_path}")

    # Save embeddings
    if self.config.get('save_embeddings', False):
      output_dir = self.config.get('output_dir', './outputs')
      os.makedirs(output_dir, exist_ok=True)

      np.save(os.path.join(output_dir, 'sensor_embeddings.npy'), sensor_embeddings)
      np.save(os.path.join(output_dir, 'text_embeddings.npy'), text_embeddings)

      with open(os.path.join(output_dir, 'captions.json'), 'w') as f:
        json.dump(captions, f, indent=2)

      with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

      self.logger.info(f"Embeddings saved to {output_dir}")


def main():
  parser = argparse.ArgumentParser(description='Evaluate retrieval performance')
  parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
  parser.add_argument('--eval_data', type=str, required=True, help='Path to evaluation data')
  parser.add_argument('--vocab', type=str, required=True, help='Path to vocabulary file')
  parser.add_argument('--output_dir', type=str, default='./eval_outputs', help='Output directory')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
  parser.add_argument('--save_embeddings', action='store_true', help='Save extracted embeddings')
  parser.add_argument('--run_demo', action='store_true', help='Run retrieval demo')

  args = parser.parse_args()

  config = {
    'checkpoint_path': args.checkpoint,
    'eval_data_path': args.eval_data,
    'vocab_path': args.vocab,
    'output_dir': args.output_dir,
    'batch_size': args.batch_size,
    'save_embeddings': args.save_embeddings,
    'run_demo': args.run_demo,
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'num_workers': 0  # Use 0 for MPS compatibility
  }

  evaluator = RetrievalEvaluator(config)
  evaluator.evaluate()


if __name__ == "__main__":
  main()
