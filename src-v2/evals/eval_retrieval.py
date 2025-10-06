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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src-v2'))

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

    checkpoint = torch.load(checkpoint_path, map_location=self.device)

    # Initialize text encoder
    # Use text encoder factory to handle different encoder types

    eval_config = model_config.copy() if "model_config" in locals() else {"text_model_name": self.config.get('text_model_name', 'thenlper/gte-base'}

    eval_config["use_cached_embeddings"] = False  # Compute embeddings on-the-fly for eval

    self.text_encoder = build_text_encoder(eval_config))
    self.text_encoder.to(self.device)

    # Initialize sensor encoder
    vocab_sizes = checkpoint['vocab_sizes']
    self.sensor_encoder = SensorEncoder(
      vocab_sizes=vocab_sizes,
      d_model=self.config.get('d_model', 768),
      n_layers=self.config.get('n_layers', 6),
      n_heads=self.config.get('n_heads', 8),
      d_ff=self.config.get('d_ff', 3072),
      max_seq_len=self.config.get('max_seq_len', 512),
      dropout=0.0,  # No dropout during evaluation
      fourier_bands=self.config.get('fourier_bands', 12),
      use_rope_time=self.config.get('use_rope_time', False),
      use_rope_2d=self.config.get('use_rope_2d', False)
    )
    self.sensor_encoder.load_state_dict(checkpoint['sensor_encoder_state_dict'])
    self.sensor_encoder.to(self.device)
    self.sensor_encoder.eval()

    self.vocab_sizes = vocab_sizes
    self.logger.info("Models loaded successfully")

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
        # Get sensor embeddings
        sensor_emb = self.sensor_encoder(
          categorical_features=batch['categorical_features'],
          coordinates=batch['coordinates'],
          time_deltas=batch['time_deltas'],
          mask=batch['mask']
        )

        # Get text embeddings (already computed in collate)
        text_emb = batch['text_embeddings']

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
    # Encode query texts
    query_embeddings = self.text_encoder.encode_texts(query_texts, self.device)
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
      self.logger.info(f"\nQuery: '{query}'")
      for j, result in enumerate(query_results[:3]):  # Show top 3
        self.logger.info(f"  {j+1}. Score: {result['score']:.4f}, Metadata: {result['metadata']}")

    # Sensor-to-text retrieval (using random sensor indices)
    self.logger.info("\n=== Sensor-to-Text Retrieval ===")
    random_sensor_indices = np.random.choice(len(metadata), size=3, replace=False).tolist()
    results = self.sensor_to_text_retrieval(random_sensor_indices, text_embeddings, captions, k=5)

    for i, (sensor_idx, query_results) in enumerate(zip(random_sensor_indices, results)):
      self.logger.info(f"\nSensor sequence {sensor_idx}:")
      for j, result in enumerate(query_results[:3]):  # Show top 3
        self.logger.info(f"  {j+1}. Score: {result['score']:.4f}, Caption: '{result['caption']}'")

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
