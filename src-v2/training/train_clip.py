#!/usr/bin/env python3

"""
Usage:
  python src-v2/training/train_clip.py --config src-v2/config/milan_tiny_50_oct1.json
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
# Removed tqdm import - no longer needed
try:
  import wandb
  WANDB_AVAILABLE = True
except ImportError:
  WANDB_AVAILABLE = False
  wandb = None

# Add parent directory to path since we're in training/ subdirectory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.text_encoder import TextEncoder, build_text_encoder, log_text_encoder_config
from models.sensor_encoder import SensorEncoder
from models.mlm_heads import MLMHeads, SpanMasker
from losses.clip import CombinedLoss
from dataio.dataset import SmartHomeDataset
from dataio.collate import create_data_loader
from utils.device_utils import get_optimal_device, get_device_config, log_device_info, optimize_for_device
from utils.training_metrics import TrainingMetrics
from utils.wandb_config import WandBConfig, get_wandb_config_for_experiment


def compute_training_schedule(config: Dict[str, Any], dataset_size: int) -> Dict[str, Any]:
  """
  Compute training schedule based on either epochs or steps specification.

  Args:
    config: Training configuration
    dataset_size: Number of samples in the training dataset

  Returns:
    Updated config with both max_epochs and max_steps computed
  """
  batch_size = config['batch_size']
  steps_per_epoch = max(1, dataset_size // batch_size)

  # Check which parameter is specified
  has_epochs = 'max_epochs' in config and config['max_epochs'] is not None
  has_steps = 'max_steps' in config and config['max_steps'] is not None

  if has_epochs and has_steps:
    # Both specified - use as is but log a warning
    print(f"Warning: Both max_epochs ({config['max_epochs']}) and max_steps ({config['max_steps']}) specified. Using both.")
    # Still set steps_per_epoch for reference
    config['steps_per_epoch'] = steps_per_epoch
    return config
  elif has_epochs and not has_steps:
    # Compute steps from epochs
    config['max_steps'] = config['max_epochs'] * steps_per_epoch
    print(f"Computed max_steps={config['max_steps']} from max_epochs={config['max_epochs']} (steps_per_epoch={steps_per_epoch})")
  elif has_steps and not has_epochs:
    # Compute epochs from steps
    config['max_epochs'] = max(1, config['max_steps'] // steps_per_epoch)
    print(f"Computed max_epochs={config['max_epochs']} from max_steps={config['max_steps']} (steps_per_epoch={steps_per_epoch})")
  else:
    # Neither specified - use defaults
    if 'max_epochs' not in config:
      config['max_epochs'] = 50
    if 'max_steps' not in config:
      config['max_steps'] = config['max_epochs'] * steps_per_epoch
    print(f"Using default max_epochs={config['max_epochs']}, computed max_steps={config['max_steps']}")

  # Store steps_per_epoch for reference
  config['steps_per_epoch'] = steps_per_epoch

  return config


class SmartHomeTrainer:
  """
  Trainer for smart-home event sequence alignment.
  """

  def __init__(self, config: Dict[str, Any]):
    # Optimize config for device
    if isinstance(config.get('device'), str):
      device = torch.device(config['device'])
    else:
      device = get_optimal_device()

    self.config = optimize_for_device(config, device)
    self.device = device

    # Create output directory first
    os.makedirs(self.config['output_dir'], exist_ok=True)

    # Save hyperparameters and run info to the output directory
    self._save_run_info()

    # Log device information
    log_device_info(self.device)

    # Setup logging
    self.setup_logging()

    # Initialize models
    self.setup_models()

    # Log text encoder configuration (after models are initialized)
    log_text_encoder_config(self.config, self.logger)

    # Initialize loss function
    self.setup_loss()

    # Initialize data (this will compute training schedule)
    self.setup_data()

    # Initialize optimizer and scheduler (after training schedule is computed)
    self.setup_optimizer()

    # Initialize training utilities
    self.scaler = GradScaler() if config['use_amp'] and self.device.type == 'cuda' else None
    self.global_step = 0
    self.best_loss = float('inf')

    # Initialize metrics tracker
    self.metrics_tracker = TrainingMetrics(
      vocab_sizes=self.vocab_sizes,
      sample_size=config.get('metrics_sample_size', 1000),
      text_encoder=self.text_encoder
    )

  def _save_run_info(self):
    """Save hyperparameters and run information to the output directory."""
    from datetime import datetime
    import json

    # Create run info dictionary
    run_info = {
      'timestamp': datetime.now().isoformat(),
      'config': self.config.copy(),  # Full hyperparameters
      'device': str(self.device),
      'pytorch_version': torch.__version__,
    }

    # Save hyperparameters as JSON
    hyperparams_path = os.path.join(self.config['output_dir'], 'hyperparameters.json')
    with open(hyperparams_path, 'w') as f:
      json.dump(run_info, f, indent=2, default=str)

    # Save a simple text summary for quick reference
    summary_path = os.path.join(self.config['output_dir'], 'run_summary.txt')
    with open(summary_path, 'w') as f:
      f.write(f"Training Run Summary\n")
      f.write(f"{'='*50}\n")
      f.write(f"Timestamp: {run_info['timestamp']}\n")
      f.write(f"Device: {run_info['device']}\n")
      f.write(f"PyTorch Version: {run_info['pytorch_version']}\n")
      f.write(f"Output Directory: {self.config['output_dir']}\n\n")

      f.write(f"Key Hyperparameters:\n")
      f.write(f"  - Model: d_model={self.config.get('d_model', 'N/A')}, n_layers={self.config.get('n_layers', 'N/A')}\n")
      f.write(f"  - Training: batch_size={self.config.get('batch_size', 'N/A')}, lr={self.config.get('learning_rate', 'N/A')}\n")
      f.write(f"  - Loss: mlm_weight={self.config.get('mlm_weight', 'N/A')}, clip_weight={self.config.get('clip_weight', 'N/A')}\n")
      f.write(f"  - Temperature: init={self.config.get('temperature_init', 'N/A')}, learnable={self.config.get('learnable_temperature', 'N/A')}\n")
      f.write(f"  - Steps: max_steps={self.config.get('max_steps', 'N/A')}, save_interval={self.config.get('save_interval', 'N/A')}\n\n")

      f.write(f"Checkpoint Files (will be saved in this directory):\n")
      f.write(f"  - best_model.pt (saved when validation improves)\n")
      f.write(f"  - checkpoint_step_*.pt (saved every {self.config.get('save_interval', 'N/A')} steps)\n")
      f.write(f"  - final_model.pt (saved at end of training)\n")

  def setup_logging(self):
    """Setup logging configuration."""
    # Create logs directory at repo root
    log_dir = os.path.join('logs', 'text')
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = self.config.get('wandb_name', 'training')
    if not run_name or run_name == 'None':
      run_name = 'training'

    log_file = os.path.join(log_dir, f'{run_name}_{timestamp}.log')

    logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s',
      handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
      ]
    )
    self.logger = logging.getLogger(__name__)
    self.logger.info(f"Logging to: {log_file}")

    # Initialize wandb if enabled and available
    if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
      # Set W&B directory to integrate with project structure
      wandb_dir = os.path.join('logs', 'wandb')
      os.makedirs(wandb_dir, exist_ok=True)

      # Enhanced W&B initialization with better organization
      wandb.init(
        project=self.config.get('wandb_project', 'discover-v2'),
        entity=self.config.get('wandb_entity', None),
        name=self.config.get('wandb_name', self.config.get('run_name', None)),
        config=self.config,
        tags=self.config.get('wandb_tags', []),
        notes=self.config.get('wandb_notes', ''),
        group=self.config.get('wandb_group', None),
        job_type='train',
        dir=wandb_dir,  # Store W&B files in logs/wandb
        reinit=True,
        save_code=True  # Save the current code state
      )
      self.logger.info(f"W&B logging to: {wandb_dir}")


  def setup_models(self):
    """Initialize all models."""
    # Text encoder (frozen) - use factory function for flexible backend support
    self.text_encoder = build_text_encoder(self.config)
    self.text_encoder.to(self.device)

    # Load vocabulary
    with open(self.config['vocab_path'], 'r') as f:
      vocab = json.load(f)

    # Get vocabulary sizes (add 1 for mask token)
    vocab_sizes = {field: len(field_vocab) + 1 for field, field_vocab in vocab.items()}

    # Sensor encoder
    self.sensor_encoder = SensorEncoder(
      vocab_sizes=vocab_sizes,
      d_model=self.config['d_model'],
      n_layers=self.config['n_layers'],
      n_heads=self.config['n_heads'],
      d_ff=self.config['d_ff'],
      max_seq_len=self.config['max_seq_len'],
      dropout=self.config['dropout'],
      fourier_bands=self.config['fourier_bands'],
      use_rope_time=self.config.get('use_rope_time', False),
      use_rope_2d=self.config.get('use_rope_2d', False)
    )
    self.sensor_encoder.to(self.device)

    # MLM heads
    self.mlm_heads = MLMHeads(
      d_model=self.config['d_model'],
      vocab_sizes=vocab_sizes,
      dropout=self.config['dropout']
    )
    self.mlm_heads.to(self.device)

    # Span masker with enhanced features
    self.span_masker = SpanMasker(
      mask_prob=self.config['mask_prob'],
      field_priors=self.config.get('field_priors', None),
      mean_span_length=self.config['mean_span_length'],
      # Enhanced masking parameters
      enable_field_blackout=self.config.get('enable_field_blackout', True),
      p_transition_seed=self.config.get('p_transition_seed', 0.3),
      strict_corr_mask=self.config.get('strict_corr_mask', True),
      adaptive_mask=self.config.get('adaptive_mask', True),
      mask_acc_threshold=self.config.get('mask_acc_threshold', 0.97),
      adaptive_window_steps=self.config.get('adaptive_window_steps', 100)
    )

    self.vocab_sizes = vocab_sizes

    # Enable gradient checkpointing if requested and available
    if self.config.get('gradient_checkpointing', False):
      if hasattr(self.sensor_encoder, 'gradient_checkpointing_enable'):
        self.sensor_encoder.gradient_checkpointing_enable()
      else:
        self.logger.warning("Gradient checkpointing not available for custom SensorEncoder")

    self.logger.info(f"Initialized models with {sum(p.numel() for p in self.sensor_encoder.parameters() if p.requires_grad):,} trainable parameters")

    # Setup W&B model watching after models are initialized
    if self.config.get('use_wandb', False) and WANDB_AVAILABLE and self.config.get('wandb_watch_model', True):
      wandb.watch(
        models=[self.sensor_encoder, self.mlm_heads],
        criterion=None,  # We'll log loss manually
        log='all' if self.config.get('wandb_log_gradients', False) else 'parameters',
        log_freq=self.config.get('wandb_watch_freq', 100),
        idx=None
      )

  def _get_text_encoder_trainable_params(self):
    """Get trainable parameters from text encoder (projection head if it exists)."""
    if hasattr(self.text_encoder, 'clip_proj') and self.text_encoder.clip_proj is not None:
      # GTE encoder with clip_proj
      return list(self.text_encoder.clip_proj.parameters())
    elif hasattr(self.text_encoder, 'proj_head') and self.text_encoder.proj_head is not None:
      # EmbeddingGemma encoder with proj_head
      return list(self.text_encoder.proj_head.parameters())
    else:
      # No trainable projection head
      return []

  def setup_optimizer(self):
    """Setup optimizer and learning rate scheduler."""
    # Only optimize sensor encoder, MLM heads, text encoder projection (if any), and learnable temperature
    trainable_params = (list(self.sensor_encoder.parameters()) +
                       list(self.mlm_heads.parameters()) +
                       list(self.loss_fn.parameters()) +  # Include loss function parameters (learnable temperature)
                       self._get_text_encoder_trainable_params())

    self.optimizer = torch.optim.AdamW(
      trainable_params,
      lr=self.config['learning_rate'],
      betas=self.config['betas'],
      weight_decay=self.config['weight_decay']
    )

    # Cosine annealing scheduler with warmup
    total_steps = self.config['max_steps']
    warmup_steps = int(self.config['warmup_ratio'] * total_steps)

    def lr_lambda(step):
      if step < warmup_steps:
        return step / warmup_steps
      else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))

    self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    self.logger.info(f"Setup optimizer with {len(trainable_params):,} parameters")

  def setup_loss(self):
    """Setup loss function."""
    # Configure hard negatives if enabled
    hard_negative_config = None
    if self.config.get('use_hard_negatives', False):
      hard_negative_config = {
        'memory_size': self.config.get('hard_negative_memory_size', 4096),
        'hard_negative_ratio': self.config.get('hard_negative_ratio', 0.5),
        'sampling_strategy': self.config.get('hard_negative_strategy', 'mixed'),
        'temperature_for_sampling': self.config.get('hard_negative_sampling_temperature', 0.1)
      }

    self.loss_fn = CombinedLoss(
      mlm_weight=self.config['mlm_weight'],
      clip_weight=self.config['clip_weight'],
      temperature_init=self.config['temperature_init'],
      learnable_temperature=self.config['learnable_temperature'],
      use_hard_negatives=self.config.get('use_hard_negatives', False),
      hard_negative_config=hard_negative_config
    )
    self.loss_fn.to(self.device)

  def setup_data(self):
    """Setup datasets and data loaders."""
    # Training dataset
    train_dataset = SmartHomeDataset(
      data_path=self.config['train_data_path'],
      vocab_path=self.config['vocab_path'],
      sequence_length=self.config['sequence_length'],
      max_captions=self.config['max_captions'],
      caption_types=self.config.get('caption_types', 'long')
    )

    self.train_loader = create_data_loader(
      dataset=train_dataset,
      text_encoder=self.text_encoder,
      span_masker=self.span_masker,
      vocab_sizes=self.vocab_sizes,
      device=self.device,
      batch_size=self.config['batch_size'],
      shuffle=True,
      num_workers=self.config['num_workers'],
      apply_mlm=True
    )

    # Validation dataset (optional)
    if self.config.get('val_data_path'):
      val_dataset = SmartHomeDataset(
        data_path=self.config['val_data_path'],
        vocab_path=self.config['vocab_path'],
        sequence_length=self.config['sequence_length'],
        max_captions=self.config['max_captions'],
        caption_types=self.config.get('caption_types', 'long')
      )

      self.val_loader = create_data_loader(
        dataset=val_dataset,
        text_encoder=self.text_encoder,
        span_masker=self.span_masker,
        vocab_sizes=self.vocab_sizes,
        device=self.device,
        batch_size=self.config['batch_size'],
        shuffle=False,
        num_workers=self.config['num_workers'],
        apply_mlm=True
      )
    else:
      self.val_loader = None

    # Setup presegmented data loaders for F1 evaluations (if available)
    self.train_presegmented_loader = None
    self.val_presegmented_loader = None

    if self.config.get('train_presegmented_data_path'):
      train_presegmented_dataset = SmartHomeDataset(
        data_path=self.config['train_presegmented_data_path'],
        vocab_path=self.config.get('presegmented_vocab_path', self.config['vocab_path']),
        sequence_length=self.config['sequence_length'],
        max_captions=self.config['max_captions'],
        caption_types=self.config.get('caption_types', 'long')
      )

      self.train_presegmented_loader = create_data_loader(
        dataset=train_presegmented_dataset,
        text_encoder=self.text_encoder,
        span_masker=self.span_masker,
        vocab_sizes=self.vocab_sizes,
        device=self.device,
        batch_size=self.config['batch_size'],
        shuffle=True,
        num_workers=self.config['num_workers'],
        apply_mlm=False  # No MLM for F1 evaluation
      )

    if self.config.get('val_presegmented_data_path'):
      val_presegmented_dataset = SmartHomeDataset(
        data_path=self.config['val_presegmented_data_path'],
        vocab_path=self.config.get('presegmented_vocab_path', self.config['vocab_path']),
        sequence_length=self.config['sequence_length'],
        max_captions=self.config['max_captions'],
        caption_types=self.config.get('caption_types', 'long')
      )

      self.val_presegmented_loader = create_data_loader(
        dataset=val_presegmented_dataset,
        text_encoder=self.text_encoder,
        span_masker=self.span_masker,
        vocab_sizes=self.vocab_sizes,
        device=self.device,
        batch_size=self.config['batch_size'],
        shuffle=False,
        num_workers=self.config['num_workers'],
        apply_mlm=False  # No MLM for F1 evaluation
      )

    # Log data loader setup
    log_msg = f"Setup data loaders: train={len(self.train_loader)} batches"
    if self.val_loader:
      log_msg += f", val={len(self.val_loader)} batches"
    if self.train_presegmented_loader:
      log_msg += f", train_presegmented={len(self.train_presegmented_loader)} batches"
    if self.val_presegmented_loader:
      log_msg += f", val_presegmented={len(self.val_presegmented_loader)} batches"
    self.logger.info(log_msg)

    # Compute training schedule based on dataset size
    dataset_size = len(train_dataset)
    self.config = compute_training_schedule(self.config, dataset_size)
    steps_per_epoch = self.config.get('steps_per_epoch', 'unknown')
    self.logger.info(f"Training schedule: {self.config['max_epochs']} epochs, {self.config['max_steps']} steps, {steps_per_epoch} steps/epoch")

  def forward_pass(self, batch: Dict[str, Any], apply_mlm: bool = True, return_raw_embeddings: bool = False):
    """Forward pass through the model."""
    # Get sensor embeddings
    if apply_mlm and 'masked_categorical_features' in batch:
      # Use masked features for MLM training
      categorical_features = batch['masked_categorical_features']
    else:
      categorical_features = batch['categorical_features']

    # Get blackout masks if available
    blackout_masks = batch.get('blackout_masks', None)

    # Get sensor embeddings with CLIP projection (512-dim)
    sensor_embeddings = self.sensor_encoder.forward_clip(
      categorical_features=categorical_features,
      coordinates=batch['coordinates'],
      time_deltas=batch['time_deltas'],
      mask=batch['mask'],
      blackout_masks=blackout_masks
    )

    # Get text embeddings with CLIP projection (already computed in collate function)
    text_embeddings = batch['text_embeddings']

    results = {
      'sensor_embeddings': sensor_embeddings,
      'text_embeddings': text_embeddings
    }

    # Get raw embeddings for representation diagnostics if requested
    if return_raw_embeddings:
      # Get raw sensor embeddings (before CLIP projection)
      sensor_raw = self.sensor_encoder.get_sequence_representations(
        categorical_features=categorical_features,
        coordinates=batch['coordinates'],
        time_deltas=batch['time_deltas'],
        mask=batch['mask'],
        blackout_masks=blackout_masks
      )
      # Pool the sequence to get single representation per sample
      sensor_raw_pooled = sensor_raw.mean(dim=1)  # [batch_size, d_model]

      # Get raw text embeddings (before CLIP projection)
      captions = batch.get('captions', [])
      if captions:
        text_raw = self.text_encoder.encode_texts(captions, self.device)
      else:
        # Fallback: use the base embeddings from the batch if available
        text_raw = batch.get('text_embeddings_raw', torch.randn_like(text_embeddings))

      results['sensor_embeddings_raw'] = sensor_raw_pooled
      results['text_embeddings_raw'] = text_raw

    # Compute MLM predictions if masking was applied
    if apply_mlm and 'mlm_mask_positions' in batch:
      # Get hidden states from sensor encoder (before pooling)
      # We need the sequence representations before pooling for MLM
      sequence_hidden_states = self.sensor_encoder.get_sequence_representations(
        categorical_features=categorical_features,
        coordinates=batch['coordinates'],
        time_deltas=batch['time_deltas'],
        mask=batch['mask'],
        blackout_masks=blackout_masks
      )

      # Compute MLM predictions
      mlm_predictions = self.mlm_heads(sequence_hidden_states)
      results['mlm_predictions'] = mlm_predictions

    return results

  def compute_loss(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]):
    """Compute total loss."""
    sensor_embeddings = model_outputs['sensor_embeddings']
    text_embeddings = model_outputs['text_embeddings']

    # Compute MLM loss if MLM predictions are available
    mlm_loss = None
    if 'mlm_predictions' in model_outputs and 'mlm_mask_positions' in batch and 'mlm_labels' in batch:
      mlm_predictions = model_outputs['mlm_predictions']
      mlm_labels = batch['mlm_labels']
      mlm_mask_positions = batch['mlm_mask_positions']

      # Compute MLM loss for each field
      total_mlm_loss = 0.0
      num_fields = 0

      for field, predictions in mlm_predictions.items():
        if field in mlm_labels and field in mlm_mask_positions:
          labels = mlm_labels[field]
          mask_pos = mlm_mask_positions[field]

          # Only compute loss for masked positions
          if mask_pos.sum() > 0:  # Check if there are any masked positions
            masked_predictions = predictions[mask_pos]
            masked_labels = labels[mask_pos]

            field_loss = F.cross_entropy(masked_predictions, masked_labels, ignore_index=-100)
            total_mlm_loss += field_loss
            num_fields += 1

      if num_fields > 0:
        mlm_loss = total_mlm_loss / num_fields

    # Compute CLIP loss
    total_loss, loss_dict = self.loss_fn(
      sensor_embeddings=sensor_embeddings,
      text_embeddings=text_embeddings,
      mlm_loss=mlm_loss
    )

    return total_loss, loss_dict

  def train_step(self, batch: Dict[str, Any]):
    """Single training step."""
    self.optimizer.zero_grad()

    if self.config['use_amp'] and self.device.type == 'cuda':
      with autocast():
        model_outputs = self.forward_pass(batch, apply_mlm=True)
        loss, loss_dict = self.compute_loss(batch, model_outputs)

      self.scaler.scale(loss).backward()

      if self.config.get('grad_clip_norm'):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
          list(self.sensor_encoder.parameters()) + list(self.mlm_heads.parameters()) + list(self.loss_fn.parameters()) + self._get_text_encoder_trainable_params(),
          self.config['grad_clip_norm']
        )

      self.scaler.step(self.optimizer)
      self.scaler.update()
    else:
      # For MPS and CPU, use regular precision
      model_outputs = self.forward_pass(batch, apply_mlm=True)
      loss, loss_dict = self.compute_loss(batch, model_outputs)
      loss.backward()

      if self.config.get('grad_clip_norm'):
        torch.nn.utils.clip_grad_norm_(
          list(self.sensor_encoder.parameters()) + list(self.mlm_heads.parameters()) + list(self.loss_fn.parameters()) + self._get_text_encoder_trainable_params(),
          self.config['grad_clip_norm']
        )

      self.optimizer.step()

    self.scheduler.step()
    self.global_step += 1

    return loss_dict

  def compute_comprehensive_metrics(self, data_loader, split_name: str = "train", max_batches: int = 10):
    """
    Compute comprehensive training metrics on a subset of data.
    Collects all samples first, then computes metrics on the full sample for better coverage.

    Args:
      data_loader: DataLoader to evaluate
      split_name: Name of the split (train/val)
      max_batches: Maximum number of batches to evaluate

    Returns:
      Dictionary of computed metrics
    """
    self.sensor_encoder.eval()
    self.mlm_heads.eval()

    # Collect all data from multiple batches first
    all_sensor_embeddings = []
    all_text_embeddings = []
    all_sensor_embeddings_raw = []
    all_text_embeddings_raw = []
    all_captions = []
    all_ground_truth_labels_l1 = []
    all_ground_truth_labels_l2 = []
    all_mlm_predictions = {}
    all_mlm_labels = {}
    all_mlm_mask_positions = {}

    batch_count = 0

    with torch.no_grad():
      for batch in data_loader:
        if batch_count >= max_batches:
          break

        # Forward pass with raw embeddings
        model_outputs = self.forward_pass(batch, apply_mlm=True, return_raw_embeddings=True)

        # Collect embeddings
        all_sensor_embeddings.append(model_outputs['sensor_embeddings'])
        all_text_embeddings.append(model_outputs['text_embeddings'])

        if 'sensor_embeddings_raw' in model_outputs:
          all_sensor_embeddings_raw.append(model_outputs['sensor_embeddings_raw'])
        if 'text_embeddings_raw' in model_outputs:
          all_text_embeddings_raw.append(model_outputs['text_embeddings_raw'])

        # Collect labels and captions
        captions = batch.get('captions', None)
        if captions:
          all_captions.extend(captions)

        ground_truth_labels_l1 = batch.get('activity_labels', None)
        if ground_truth_labels_l1:
          all_ground_truth_labels_l1.extend(ground_truth_labels_l1)

        ground_truth_labels_l2 = batch.get('activity_labels_l2', None)
        if ground_truth_labels_l2:
          all_ground_truth_labels_l2.extend(ground_truth_labels_l2)

        # Collect MLM outputs for batch-wise metrics
        if 'mlm_predictions' in model_outputs:
          for field, preds in model_outputs['mlm_predictions'].items():
            if field not in all_mlm_predictions:
              all_mlm_predictions[field] = []
            all_mlm_predictions[field].append(preds)

        if 'mlm_labels' in model_outputs:
          for field, labels in model_outputs['mlm_labels'].items():
            if field not in all_mlm_labels:
              all_mlm_labels[field] = []
            all_mlm_labels[field].append(labels)

        if 'mlm_mask_positions' in model_outputs:
          for field, positions in model_outputs['mlm_mask_positions'].items():
            if field not in all_mlm_mask_positions:
              all_mlm_mask_positions[field] = []
            all_mlm_mask_positions[field].append(positions)

        batch_count += 1

    # Concatenate all collected data
    if all_sensor_embeddings:
      combined_sensor_embeddings = torch.cat(all_sensor_embeddings, dim=0)
      combined_text_embeddings = torch.cat(all_text_embeddings, dim=0)

      combined_sensor_embeddings_raw = None
      if all_sensor_embeddings_raw:
        combined_sensor_embeddings_raw = torch.cat(all_sensor_embeddings_raw, dim=0)

      combined_text_embeddings_raw = None
      if all_text_embeddings_raw:
        combined_text_embeddings_raw = torch.cat(all_text_embeddings_raw, dim=0)

      # Combine MLM outputs
      combined_mlm_outputs = {}
      if all_mlm_predictions:
        combined_mlm_outputs['mlm_predictions'] = {}
        for field, pred_list in all_mlm_predictions.items():
          combined_mlm_outputs['mlm_predictions'][field] = torch.cat(pred_list, dim=0)

      if all_mlm_labels:
        combined_mlm_outputs['mlm_labels'] = {}
        for field, label_list in all_mlm_labels.items():
          combined_mlm_outputs['mlm_labels'][field] = torch.cat(label_list, dim=0)

      if all_mlm_mask_positions:
        combined_mlm_outputs['mlm_mask_positions'] = {}
        for field, pos_list in all_mlm_mask_positions.items():
          combined_mlm_outputs['mlm_mask_positions'][field] = torch.cat(pos_list, dim=0)

      # Create combined model outputs
      combined_model_outputs = {
        'sensor_embeddings': combined_sensor_embeddings,
        'text_embeddings': combined_text_embeddings,
        **combined_mlm_outputs
      }

      if combined_sensor_embeddings_raw is not None:
        combined_model_outputs['sensor_embeddings_raw'] = combined_sensor_embeddings_raw
      if combined_text_embeddings_raw is not None:
        combined_model_outputs['text_embeddings_raw'] = combined_text_embeddings_raw

      # Log the actual sample size being used
      total_samples = combined_sensor_embeddings.size(0)
      target_sample_size = self.config.get('metrics_sample_size', 1000)
      actual_sample_size = min(total_samples, target_sample_size)

      self.logger.info(f"Comprehensive metrics for {split_name}: collected {total_samples} samples, "
                      f"using {actual_sample_size} for computation (target: {target_sample_size})")

      # Compute metrics on the full combined sample with error handling
      try:
        comprehensive_metrics = self.metrics_tracker.compute_all_metrics(
          batch=None,  # Not needed for combined computation
          model_outputs=combined_model_outputs,
          sensor_embeddings_raw=combined_sensor_embeddings_raw,
          text_embeddings_raw=combined_text_embeddings_raw,
          ground_truth_labels=all_ground_truth_labels_l1 if all_ground_truth_labels_l1 else None,
          ground_truth_labels_l2=all_ground_truth_labels_l2 if all_ground_truth_labels_l2 else None,
          captions=all_captions if all_captions else None,
          sample_for_expensive=True  # Will sample from the combined data if needed
        )
      except Exception as e:
        self.logger.error(f"Error computing comprehensive metrics: {e}")
        comprehensive_metrics = {'error': str(e)}

      # Add actual sample size to metrics for monitoring
      comprehensive_metrics['actual_sample_size'] = actual_sample_size

      # Add split prefix to all metrics
      final_metrics = {}
      for key, value in comprehensive_metrics.items():
        final_metrics[f'{split_name}/{key}'] = value

    else:
      final_metrics = {}

    # Return to training mode
    self.sensor_encoder.train()
    self.mlm_heads.train()

    return final_metrics

  def _organize_metrics_for_wandb(self, metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Organize metrics into categories for better W&B visualization.

    Args:
      metrics: Dictionary of metrics with keys like 'train/alignment/pos_cos_mean'

    Returns:
      Dictionary with reorganized metric keys
    """
    organized = {}

    for key, value in metrics.items():
      # Split the key to understand the structure
      parts = key.split('/')

      if len(parts) < 2:
        organized[key] = value
        continue

      split_name = parts[0]  # train or val
      metric_name = '/'.join(parts[1:])  # everything after split

      # Categorize metrics
      if 'alignment' in metric_name:
        category = 'alignment'
      elif 'mlm_accuracy' in metric_name:
        category = 'mlm_accuracy'
      elif 'repr' in metric_name:
        category = 'representation'
      elif 'f1' in metric_name:
        category = 'classification'
      elif 'recall@' in metric_name or 'ndcg@' in metric_name:
        if 'class_' in metric_name:
          category = 'retrieval_per_class'
        else:
          category = 'retrieval'
      elif any(x in metric_name for x in ['loss', 'contribution']):
        category = 'loss'
      elif any(x in metric_name for x in ['acc', 'accuracy']):
        category = 'classification'
      else:
        category = 'other'

      # Create organized key: category/split/metric
      new_key = f"{category}/{split_name}/{metric_name}"
      organized[new_key] = value

    return organized

  def _log_metrics_to_file(self, metrics: Dict[str, float], phase: str):
    """
    Log metrics to the log file in a structured format.

    Args:
      metrics: Dictionary of metrics to log
      phase: Phase name (e.g., 'train', 'validation', 'comprehensive')
    """
    # Group metrics by category for cleaner logging
    categories = {}
    for key, value in metrics.items():
      category = key.split('/')[0] if '/' in key else 'general'
      if category not in categories:
        categories[category] = {}
      categories[category][key] = value

    # Log each category
    for category, cat_metrics in categories.items():
      metrics_str = " | ".join([f"{k.split('/')[-1]}: {v:.4f}" if isinstance(v, (int, float)) else f"{k.split('/')[-1]}: {v}" for k, v in cat_metrics.items()])
      self.logger.info(f"{phase.upper()} {category.upper()} | Step {self.global_step} | {metrics_str}")

  def _log_metrics_to_file_silent(self, metrics: Dict[str, float], phase: str):
    """
    Log metrics to the log file only (no terminal output) in a structured format.

    Args:
      metrics: Dictionary of metrics to log
      phase: Phase name (e.g., 'train', 'validation', 'comprehensive')
    """
    # Group metrics by category for cleaner logging
    categories = {}
    for key, value in metrics.items():
      category = key.split('/')[0] if '/' in key else 'general'
      if category not in categories:
        categories[category] = {}
      categories[category][key] = value

    # Log each category to file only using debug level
    for category, cat_metrics in categories.items():
      metrics_str = " | ".join([f"{k.split('/')[-1]}: {v:.4f}" if isinstance(v, (int, float)) else f"{k.split('/')[-1]}: {v}" for k, v in cat_metrics.items()])
      self.logger.debug(f"{phase.upper()} {category.upper()} | Step {self.global_step} | {metrics_str}")

  def validate(self):
    """Validation step."""
    if self.val_loader is None:
      return {}

    self.sensor_encoder.eval()
    self.mlm_heads.eval()

    total_loss = 0.0
    total_clip_loss = 0.0
    total_s2t_acc = 0.0
    total_t2s_acc = 0.0
    num_batches = 0

    with torch.no_grad():
      for batch in self.val_loader:
        model_outputs = self.forward_pass(batch, apply_mlm=True)
        loss, loss_dict = self.compute_loss(batch, model_outputs)

        total_loss += loss_dict['total_loss']
        total_clip_loss += loss_dict['clip_loss']
        total_s2t_acc += loss_dict['sensor_to_text_acc']
        total_t2s_acc += loss_dict['text_to_sensor_acc']
        num_batches += 1

    # Return to training mode
    self.sensor_encoder.train()
    self.mlm_heads.train()

    if num_batches > 0:
      return {
        'val_loss': total_loss / num_batches,
        'val_clip_loss': total_clip_loss / num_batches,
        'val_s2t_acc': total_s2t_acc / num_batches,
        'val_t2s_acc': total_t2s_acc / num_batches
      }
    else:
      return {}

  def save_checkpoint(self, path: str):
    """Save model checkpoint."""
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
      'global_step': self.global_step,
      'sensor_encoder_state_dict': self.sensor_encoder.state_dict(),
      'mlm_heads_state_dict': self.mlm_heads.state_dict(),
      'loss_fn_state_dict': self.loss_fn.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'scheduler_state_dict': self.scheduler.state_dict(),
      'config': self.config,
      'vocab_sizes': self.vocab_sizes,
      'best_loss': self.best_loss
    }

    if self.scaler:
      checkpoint['scaler_state_dict'] = self.scaler.state_dict()

    torch.save(checkpoint, path)

    # Enhanced logging with file size and directory info
    file_size_mb = os.path.getsize(path) / (1024**2) if os.path.exists(path) else 0
    checkpoint_dir = os.path.dirname(path)
    checkpoint_name = os.path.basename(path)
    self.logger.info(f"Saved checkpoint '{checkpoint_name}' ({file_size_mb:.1f}MB) to directory: {checkpoint_dir}")

  def load_checkpoint(self, checkpoint_path: str):
    """Load model checkpoint and resume training."""
    if not os.path.exists(checkpoint_path):
      raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=self.device)

    # Load model states
    self.sensor_encoder.load_state_dict(checkpoint['sensor_encoder_state_dict'])
    self.mlm_heads.load_state_dict(checkpoint['mlm_heads_state_dict'])
    self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])

    # Load optimizer and scheduler states
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load training state
    self.global_step = checkpoint['global_step']
    self.best_loss = checkpoint.get('best_loss', float('inf'))

    # Load scaler state if available
    if self.scaler and 'scaler_state_dict' in checkpoint:
      self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Verify vocab sizes match
    if 'vocab_sizes' in checkpoint:
      checkpoint_vocab_sizes = checkpoint['vocab_sizes']
      if checkpoint_vocab_sizes != self.vocab_sizes:
        self.logger.warning(f"Vocabulary sizes mismatch! Checkpoint: {checkpoint_vocab_sizes}, Current: {self.vocab_sizes}")

    file_size_mb = os.path.getsize(checkpoint_path) / (1024**2)
    self.logger.info(f"Successfully loaded checkpoint from step {self.global_step} ({file_size_mb:.1f}MB)")
    self.logger.info(f"Resuming training from step {self.global_step + 1} with best loss: {self.best_loss:.4f}")

    return checkpoint

  def train(self):
    """Main training loop."""
    # Log checkpoint directory information
    abs_output_dir = os.path.abspath(self.config['output_dir'])
    self.logger.info(f"All checkpoints will be saved to: {abs_output_dir}")
    self.logger.info(f"  - Best model: {abs_output_dir}/best_model.pt")
    self.logger.info(f"  - Intermediate: {abs_output_dir}/checkpoint_step_*.pt (every {self.config.get('save_interval', 'N/A')} steps)")
    self.logger.info(f"  - Final model: {abs_output_dir}/final_model.pt")
    self.logger.info("Starting training...")

    self.sensor_encoder.train()
    self.mlm_heads.train()

    for epoch in range(self.config['max_epochs']):
      epoch_loss = 0.0
      num_batches = 0

      for batch in self.train_loader:
        # Training step
        loss_dict = self.train_step(batch)
        epoch_loss += loss_dict['total_loss']
        num_batches += 1

        # Compute quick MLM accuracies every few steps for immediate feedback
        if self.global_step % (self.config['log_interval'] * 4) == 0:  # Every 100 steps (4 * 25)
          with torch.no_grad():
            model_outputs = self.forward_pass(batch, apply_mlm=True, return_raw_embeddings=False)
            if 'mlm_predictions' in model_outputs:
              quick_mlm_metrics = self.metrics_tracker.compute_mlm_accuracy(
                model_outputs['mlm_predictions'],
                batch.get('mlm_labels', {}),
                batch.get('mlm_mask_positions', {})
              )

              # Update adaptive masking based on field accuracies
              if quick_mlm_metrics and self.config.get('adaptive_mask', True):
                field_accuracies = {}
                for key, value in quick_mlm_metrics.items():
                  if key.startswith('mlm_accuracy/') and not key.endswith('_top5') and key != 'mlm_accuracy/overall':
                    field_name = key.replace('mlm_accuracy/', '')
                    field_accuracies[field_name] = value

                if field_accuracies:
                  self.span_masker.update_adaptive_masks(field_accuracies)

              # Log to W&B immediately
              if self.config.get('use_wandb', False) and WANDB_AVAILABLE and quick_mlm_metrics:
                quick_wandb_metrics = {f'quick_mlm/{k}': v for k, v in quick_mlm_metrics.items()}
                wandb.log(quick_wandb_metrics, step=self.global_step)

                # Log quick MLM metrics to file only (no terminal output)
                self._log_metrics_to_file(quick_wandb_metrics, "quick_mlm")

        # Logging
        if self.global_step % self.config['log_interval'] == 0:
          lr = self.scheduler.get_last_lr()[0]
          # Simplified logging - only losses
          loss_items = [
            f"total_loss: {loss_dict['total_loss']:.4f}",
            f"clip_loss: {loss_dict['clip_loss']:.4f}",
          ]

          if 'mlm_loss' in loss_dict:
            loss_items.append(f"mlm_loss: {loss_dict['mlm_loss']:.4f}")

          loss_items.append(f"temp: {loss_dict['temperature']:.6f}")
          loss_items.append(f"lr: {lr:.2e}")

          log_msg = f"Step {self.global_step:6d} | " + " | ".join(loss_items)
          self.logger.info(log_msg)

          # Full W&B logging (including accuracies for detailed tracking)
          log_dict = {
            'train/step': self.global_step,
            'train/epoch': epoch,
            'train/learning_rate': lr,
            'train/total_loss': loss_dict['total_loss'],
            'train/clip_loss': loss_dict['clip_loss'],
            'train/sensor_to_text_acc': loss_dict['sensor_to_text_acc'],
            'train/text_to_sensor_acc': loss_dict['text_to_sensor_acc'],
            'train/temperature': loss_dict['temperature'],
          }

          # Add MLM loss if available
          if 'mlm_loss' in loss_dict:
            log_dict['train/mlm_loss'] = loss_dict['mlm_loss']

          # Add a simple test metric to verify W&B logging is working
          log_dict['debug/step_counter'] = self.global_step
          log_dict['debug/batch_size'] = len(batch['categorical_features']) if 'categorical_features' in batch else 0

          if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            # Add system metrics
            log_dict.update({
              'system/gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
              'system/gpu_memory_cached': torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0,
            })

            # Add loss configuration tracking
            log_dict.update({
              'config/mlm_weight': self.config.get('mlm_weight', 1.0),
              'config/clip_weight': self.config.get('clip_weight', 1.0),
              'config/temperature_init': self.config.get('temperature_init', 0.02),
              'config/batch_size': self.config.get('batch_size', 32),
            })

            # Add learning rate schedule tracking
            total_steps = self.config.get('max_steps', 150000)
            warmup_steps = int(self.config.get('warmup_ratio', 0.1) * total_steps)
            log_dict.update({
              'schedule/progress_pct': (self.global_step / total_steps) * 100,
              'schedule/warmup_complete': self.global_step >= warmup_steps,
              'schedule/warmup_progress': min(self.global_step / warmup_steps, 1.0) * 100 if warmup_steps > 0 else 100.0,
              'schedule/cosine_phase': max(0, self.global_step - warmup_steps) / max(1, total_steps - warmup_steps) if total_steps > warmup_steps else 0,
            })

            # Add individual loss contributions (before weighting)
            if 'mlm_loss' in loss_dict and 'clip_loss' in loss_dict:
              mlm_contribution = self.config.get('mlm_weight', 1.0) * loss_dict['mlm_loss']
              clip_contribution = self.config.get('clip_weight', 1.0) * loss_dict['clip_loss']
              log_dict.update({
                'train/mlm_contribution': mlm_contribution,
                'train/clip_contribution': clip_contribution,
                'train/loss_ratio_mlm_clip': mlm_contribution / (clip_contribution + 1e-8),
              })

            # Add quick alignment metrics (cosine similarities) from the current training step
            # Get embeddings from the loss computation
            if 'sensor_to_text_acc' in loss_dict and 'text_to_sensor_acc' in loss_dict:
              # Use the accuracies as a proxy for alignment quality
              avg_acc = (loss_dict['sensor_to_text_acc'] + loss_dict['text_to_sensor_acc']) / 2
              log_dict.update({
                'train/alignment_avg_accuracy': avg_acc,
                'train/alignment_s2t_acc': loss_dict['sensor_to_text_acc'],
                'train/alignment_t2s_acc': loss_dict['text_to_sensor_acc'],
              })

            wandb.log(log_dict, step=self.global_step)

            # Log all W&B metrics to file only (no terminal output for detailed metrics)
            self._log_metrics_to_file_silent(log_dict, "train")

        # Validation
        if self.global_step % self.config['val_interval'] == 0:
          val_metrics = self.validate()

          # Note: Comprehensive validation metrics are now only computed during metrics_interval
          # to avoid double computation and long delays during regular validation

          if val_metrics:
            # Rename validation metrics with proper prefixes
            wandb_val_metrics = {
              'val/total_loss': val_metrics['val_loss'],
              'val/clip_loss': val_metrics['val_clip_loss'],
              'val/sensor_to_text_acc': val_metrics['val_s2t_acc'],
              'val/text_to_sensor_acc': val_metrics['val_t2s_acc'],
            }

            # Simple validation message for terminal
            val_msg = f"Validation | loss: {val_metrics['val_loss']:.4f} | clip: {val_metrics['val_clip_loss']:.4f}"
            if 'val_mlm_loss' in val_metrics:
              val_msg += f" | mlm: {val_metrics['val_mlm_loss']:.4f}"
            self.logger.info(val_msg)

            if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
              # Add validation loss contributions if MLM validation is available
              if 'val_mlm_loss' in val_metrics:
                mlm_val_contribution = self.config.get('mlm_weight', 1.0) * val_metrics['val_mlm_loss']
                clip_val_contribution = self.config.get('clip_weight', 1.0) * val_metrics['val_clip_loss']
                wandb_val_metrics.update({
                  'val/mlm_loss': val_metrics['val_mlm_loss'],
                  'val/mlm_contribution': mlm_val_contribution,
                  'val/clip_contribution': clip_val_contribution,
                  'val/loss_ratio_mlm_clip': mlm_val_contribution / (clip_val_contribution + 1e-8),
                })

              wandb.log(wandb_val_metrics, step=self.global_step)

              # Log validation metrics to file only (detailed)
              self._log_metrics_to_file_silent(wandb_val_metrics, "validation")

            # Save best model
            if val_metrics['val_loss'] < self.best_loss:
              self.best_loss = val_metrics['val_loss']
              best_path = os.path.join(self.config['output_dir'], 'best_model.pt')
              self.save_checkpoint(best_path)

              # Log best model metadata to W&B (no file upload)
              if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
                best_model_metrics = {
                  'best_model/step': self.global_step,
                  'best_model/val_loss': self.best_loss,
                  'best_model/local_path': best_path,
                  'best_model/file_size_mb': os.path.getsize(best_path) / (1024**2) if os.path.exists(best_path) else 0,
                  'best_model/timestamp': self.global_step,
                }
                wandb.log(best_model_metrics, step=self.global_step)

                # Log best model metrics to file as well
                self._log_metrics_to_file(best_model_metrics, "best_model")

        # Comprehensive metrics
        if self.global_step % self.config.get('metrics_interval', 1000) == 0 and self.global_step > 0:
          # Calculate batches needed to reach metrics_sample_size
          # Use a conservative approach: ensure we get AT LEAST the desired sample size
          sample_size = self.config.get('metrics_sample_size', 1000)

          # Use a conservative batch size estimate to ensure we get enough samples
          # Even if actual batches are smaller, this ensures we collect enough batches
          conservative_batch_size = min(self.config.get('batch_size', 32), 16)  # Assume smaller actual batches
          batches_needed = max(10, (sample_size + conservative_batch_size - 1) // conservative_batch_size)

          self.logger.info(f"Computing comprehensive metrics: target_samples={sample_size}, batches_needed={batches_needed}")

          # Compute training metrics - use presegmented data if available for better F1 scores
          try:
            if self.train_presegmented_loader:
              self.logger.info("Using presegmented training data for comprehensive metrics")
              train_metrics = self.compute_comprehensive_metrics(
                self.train_presegmented_loader, split_name="train", max_batches=batches_needed
              )
            else:
              train_metrics = self.compute_comprehensive_metrics(
                self.train_loader, split_name="train", max_batches=batches_needed
              )
          except Exception as e:
            self.logger.error(f"Error computing training metrics: {e}")
            train_metrics = {}

          # Compute validation metrics if available - use presegmented data if available for better F1 scores
          val_metrics = {}
          if self.val_presegmented_loader:
            try:
              self.logger.info("Using presegmented validation data for comprehensive metrics")
              val_metrics = self.compute_comprehensive_metrics(
                self.val_presegmented_loader, split_name="val", max_batches=batches_needed
              )
            except Exception as e:
              self.logger.error(f"Error computing validation metrics: {e}")
              val_metrics = {}
          elif self.val_loader:
            try:
              val_metrics = self.compute_comprehensive_metrics(
                self.val_loader, split_name="val", max_batches=batches_needed
              )
            except Exception as e:
              self.logger.error(f"Error computing validation metrics: {e}")
              val_metrics = {}

          # Combine all metrics and organize them
          comprehensive_metrics = {**train_metrics, **val_metrics}

          if comprehensive_metrics:
            # Organize metrics into categories for W&B
            organized_metrics = self._organize_metrics_for_wandb(comprehensive_metrics)

            if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
              wandb.log(organized_metrics, step=self.global_step)

              # Log comprehensive metrics to file only
              self._log_metrics_to_file_silent(organized_metrics, "comprehensive")

            # Log key metrics summary to terminal (simplified)
            key_metrics = []
            for key, value in comprehensive_metrics.items():
              if any(x in key for x in ['pos_neg_gap', 'mlm_accuracy/overall']):
                key_metrics.append(f"{key.split('/')[-1]}: {value:.4f}")

            if key_metrics:
              self.logger.info(f"Step {self.global_step} | " + " | ".join(key_metrics[:4]))

        # Save checkpoint
        if self.global_step % self.config['save_interval'] == 0:
          checkpoint_path = os.path.join(self.config['output_dir'], f'checkpoint_step_{self.global_step}.pt')
          self.save_checkpoint(checkpoint_path)

        # Check if we've reached max steps
        if self.global_step >= self.config['max_steps']:
          break

      if self.global_step >= self.config['max_steps']:
        break

      # End of epoch logging
      avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
      self.logger.info(f"Epoch {epoch} completed | Average loss: {avg_epoch_loss:.4f}")

    # Save final checkpoint
    final_path = os.path.join(self.config['output_dir'], 'final_model.pt')
    self.save_checkpoint(final_path)

    # Log final model metadata to W&B (no file upload)
    if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
      # Log training summary with model metadata
      final_summary_metrics = {
        'training/final_step': self.global_step,
        'training/best_val_loss': self.best_loss,
        'training/status': 'completed',
        'final_model/local_path': final_path,
        'final_model/file_size_mb': os.path.getsize(final_path) / (1024**2) if os.path.exists(final_path) else 0,
        'final_model/total_parameters': sum(p.numel() for p in self.sensor_encoder.parameters()) + sum(p.numel() for p in self.mlm_heads.parameters()),
        'final_model/trainable_parameters': sum(p.numel() for p in self.sensor_encoder.parameters() if p.requires_grad) + sum(p.numel() for p in self.mlm_heads.parameters() if p.requires_grad),
      }
      wandb.log(final_summary_metrics, step=self.global_step)

      # Log final summary metrics to file as well
      self._log_metrics_to_file(final_summary_metrics, "final_summary")

      # Finish W&B run
      wandb.finish()

    self.logger.info("Training completed!")


def get_default_config():
  """Get default training configuration."""
  device = get_optimal_device()
  device_config = get_device_config(device)

  config = {
    # Model configuration
    'text_model_name': 'thenlper/gte-base',
    'd_model': 768,
    'n_layers': 6,
    'n_heads': 8,
    'd_ff': 3072,
    'dropout': 0.1,
    'fourier_bands': 12,
    'max_seq_len': 512,
    'sequence_length': 20,
    'max_captions': 3,

    # Training configuration - IMPROVED
    'batch_size': 128,  # Increased for better contrastive learning
    'learning_rate': 3e-4,  # Slightly higher for faster convergence
    'betas': (0.9, 0.98),
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,  # Longer warmup for stability
    'max_epochs': 1,  # Default epochs (will be used if neither epochs nor steps specified)
    'max_steps': None,  # Will be computed from epochs if not specified
    'grad_clip_norm': 1.0,

    # Loss configuration - BALANCED CLIP+MLM (50-50)
    'mlm_weight': 1.0,  # Equal weighting for MLM
    'clip_weight': 1.0,  # Equal weighting for CLIP
    'temperature_init': 0.02,  # Lower temperature for better alignment
    'learnable_temperature': True,

    # MLM configuration - Enhanced masking strategy
    'mask_prob': 0.25,
    'mean_span_length': 5.0,  # Longer spans for fewer but more challenging masks

    # Enhanced masking features (all configurable)
    'enable_field_blackout': True,      # Zero out correlated embeddings when fields are masked
    'p_transition_seed': 0.3,           # Probability to seed masks at activity/room transitions
    'strict_corr_mask': True,           # Mask all fields in correlated groups within spans
    'adaptive_mask': True,              # Track per-field accuracy and adjust mask rates
    'mask_acc_threshold': 0.97,         # Accuracy threshold for adaptive masking (97%)
    'adaptive_window_steps': 100,       # Window size for tracking recent accuracies

    # Training utilities
    'gradient_checkpointing': False,  # Not implemented for custom SensorEncoder

    # Logging and saving
    'log_interval': 50,  # More frequent logging
    'val_interval': 500,  # More frequent validation
    'save_interval': 2000,  # More frequent saving
    'metrics_interval': 1000,  # Comprehensive metrics every 1K steps
    'metrics_sample_size': 1000,  # Sample size for expensive metrics
    'output_dir': './models/outputs/milan_training',  # Use existing model directory structure

    # W&B Configuration
    'use_wandb': True,  # Enable W&B by default
    'wandb_project': 'discover-v2',
    'wandb_entity': None,  # Set to your W&B username/team
    'wandb_name': None,  # Will be auto-generated
    'wandb_tags': ['dual-encoder', 'smart-home', 'clip-mlm', '50-50-loss'],
    'wandb_notes': 'Dual-encoder training with balanced CLIP + MLM losses (50-50)',
    'wandb_group': None,  # Will be auto-generated
    'wandb_log_model': False,  # Don't upload model files, only metadata
    'wandb_log_gradients': False,  # Log gradient histograms (expensive)
    'wandb_watch_model': True,  # Watch model parameters
    'wandb_watch_freq': 100,  # How often to log parameters

    # Data paths (to be specified)
    'train_data_path': None,
    'val_data_path': None,
    'vocab_path': None
  }

  # Merge with device-specific settings
  config.update(device_config)
  config['device'] = device.type

  return config


def find_latest_checkpoint(checkpoint_dir: str) -> str:
  """Find the latest checkpoint in a directory."""
  import glob
  import re

  if not os.path.exists(checkpoint_dir):
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

  # Look for checkpoint files
  checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_step_*.pt")
  checkpoint_files = glob.glob(checkpoint_pattern)

  if not checkpoint_files:
    raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")

  # Extract step numbers and find the latest
  latest_step = -1
  latest_checkpoint = None

  for checkpoint_file in checkpoint_files:
    filename = os.path.basename(checkpoint_file)
    match = re.search(r'checkpoint_step_(\d+)\.pt', filename)
    if match:
      step = int(match.group(1))
      if step > latest_step:
        latest_step = step
        latest_checkpoint = checkpoint_file

  if latest_checkpoint is None:
    raise FileNotFoundError(f"No valid checkpoint files found in: {checkpoint_dir}")

  print(f"Found latest checkpoint: {latest_checkpoint} (step {latest_step})")
  return latest_checkpoint


def auto_detect_presegmented_paths(config):
  """Auto-detect presegmented data paths based on regular data paths."""
  import os

  # Try to detect presegmented training data
  if 'train_data_path' in config and not config.get('train_presegmented_data_path'):
    train_path = config['train_data_path']
    if train_path.endswith('_train.json'):
      presegmented_train_path = train_path.replace('_train.json', '_presegmented_train.json')
      if os.path.exists(presegmented_train_path):
        config['train_presegmented_data_path'] = presegmented_train_path
        print(f"Auto-detected presegmented training data: {presegmented_train_path}")

  # Try to detect presegmented validation data
  if 'val_data_path' in config and not config.get('val_presegmented_data_path'):
    val_path = config['val_data_path']
    if val_path.endswith('_test.json'):
      presegmented_val_path = val_path.replace('_test.json', '_presegmented_test.json')
      if os.path.exists(presegmented_val_path):
        config['val_presegmented_data_path'] = presegmented_val_path
        print(f"Auto-detected presegmented validation data: {presegmented_val_path}")

  # Try to detect presegmented vocabulary
  if 'vocab_path' in config and not config.get('presegmented_vocab_path'):
    vocab_path = config['vocab_path']
    if vocab_path.endswith('_vocab.json'):
      presegmented_vocab_path = vocab_path.replace('_vocab.json', '_presegmented_vocab.json')
      if os.path.exists(presegmented_vocab_path):
        config['presegmented_vocab_path'] = presegmented_vocab_path
        print(f"Auto-detected presegmented vocabulary: {presegmented_vocab_path}")

  return config


def main():
  parser = argparse.ArgumentParser(description='Train smart-home sequence alignment model')
  parser.add_argument('--config', type=str, help='Path to config file')
  parser.add_argument('--train_data', type=str, help='Path to training data')
  parser.add_argument('--vocab', type=str, help='Path to vocabulary file')
  parser.add_argument('--val_data', type=str, help='Path to validation data')
  parser.add_argument('--output_dir', type=str, default='src-v2/trained_models/default_run', help='Output directory')
  parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
  parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
  parser.add_argument('--max_steps', type=int, help='Maximum training steps (alternative to max_epochs)')
  parser.add_argument('--max_epochs', type=int, help='Maximum training epochs (alternative to max_steps)')
  parser.add_argument('--log_interval', type=int, default=50, help='Logging interval (steps)')
  parser.add_argument('--val_interval', type=int, default=500, help='Validation interval (steps)')
  parser.add_argument('--save_interval', type=int, default=2000, help='Checkpoint save interval (steps)')
  parser.add_argument('--metrics_interval', type=int, default=1000, help='Comprehensive metrics interval (steps)')

  # W&B arguments
  parser.add_argument('--wandb_project', type=str, default='discover-v2', help='W&B project name')
  parser.add_argument('--wandb_name', type=str, help='W&B run name')
  parser.add_argument('--wandb_tags', type=str, nargs='+', help='W&B tags')
  parser.add_argument('--wandb_notes', type=str, help='W&B run notes')
  parser.add_argument('--no_wandb', action='store_true', help='Disable W&B logging')
  parser.add_argument('--experiment', type=str, choices=['milan_baseline', 'milan_ablation_mlm', 'milan_ablation_temperature', 'multi_dataset'],
                      help='Use predefined experiment configuration')

  # Text encoder arguments (all optional, fallback to config or defaults)
  parser.add_argument('--text_model_name', type=str, help='Text encoder model name (e.g., google/embeddinggemma-300m)')
  parser.add_argument('--text_backend', type=str, choices=['hf', 'sentence_transformers'], help='Text encoder backend')
  parser.add_argument('--text_prompt_mode', type=str, choices=['query', 'document', 'clustering', 'classification', 'retrieval', 'similarity'],
                      help='Prompt mode for EmbeddingGemma (ignored for GTE)')
  parser.add_argument('--text_output_dim', type=int, help='Text encoder output dimension')
  parser.add_argument('--use_text_proj_head', type=lambda x: x.lower() == 'true', help='Use projection head (true/false)')

  # Resume training arguments
  parser.add_argument('--resume', type=str, help='Path to checkpoint file to resume from')
  parser.add_argument('--resume_from_dir', type=str, help='Resume from latest checkpoint in directory')

  args = parser.parse_args()

  # Load config
  if args.config:
    with open(args.config, 'r') as f:
      config = json.load(f)
  else:
    config = get_default_config()

  # Apply predefined experiment configuration if specified
  if args.experiment:
    try:
      wandb_config = get_wandb_config_for_experiment(args.experiment, config)
      config.update(wandb_config)
    except ValueError as e:
      print(f"Error loading experiment config: {e}")
      return

  # Override with command line arguments (only if provided)
  if args.train_data:
    config['train_data_path'] = args.train_data
  if args.vocab:
    config['vocab_path'] = args.vocab
  if args.val_data:
    config['val_data_path'] = args.val_data

  # Handle output_dir with smart versioning
  if args.output_dir != 'src-v2/trained_models/default_run':  # If explicitly provided
    config['output_dir'] = args.output_dir
  elif 'output_dir' not in config or not config['output_dir']:  # If not in config
    # Generate output_dir from config file name
    if args.config:
      config_name = Path(args.config).stem  # Get filename without extension
      config['output_dir'] = f'src-v2/trained_models/{config_name}'
    else:
      config['output_dir'] = args.output_dir  # Fall back to default

  # Add versioning if directory already exists
  base_output_dir = config['output_dir']
  if os.path.exists(base_output_dir):
    version = 1
    while os.path.exists(f"{base_output_dir}_v{version}"):
      version += 1
    config['output_dir'] = f"{base_output_dir}_v{version}"
    print(f"Output directory {base_output_dir} exists, using versioned: {config['output_dir']}")

  # Only override these if explicitly provided (different from parser defaults)
  parser_defaults = parser.parse_args([])  # Get default values
  if args.batch_size != parser_defaults.batch_size:
    config['batch_size'] = args.batch_size
  if args.learning_rate != parser_defaults.learning_rate:
    config['learning_rate'] = args.learning_rate
  if args.log_interval != parser_defaults.log_interval:
    config['log_interval'] = args.log_interval
  if args.val_interval != parser_defaults.val_interval:
    config['val_interval'] = args.val_interval
  if args.save_interval != parser_defaults.save_interval:
    config['save_interval'] = args.save_interval

  # Only override metrics_interval if explicitly provided
  if hasattr(args, 'metrics_interval') and args.metrics_interval != 1000:  # 1000 is the default
    config['metrics_interval'] = args.metrics_interval

  # Validate required paths
  if not config.get('train_data_path'):
    print("Error: train_data_path must be specified either in config file or via --train_data")
    return
  if not config.get('vocab_path'):
    print("Error: vocab_path must be specified either in config file or via --vocab")
    return

  # Auto-detect presegmented data paths for F1 evaluations
  config = auto_detect_presegmented_paths(config)

  # Handle epochs/steps specification from command line
  if args.max_steps is not None:
    config['max_steps'] = args.max_steps
  if args.max_epochs is not None:
    config['max_epochs'] = args.max_epochs

  # Override W&B settings from command line
  if args.no_wandb:
    config['use_wandb'] = False
  else:
    config['wandb_project'] = args.wandb_project
    if args.wandb_name:
      config['wandb_name'] = args.wandb_name
    if args.wandb_tags:
      config['wandb_tags'] = args.wandb_tags
    if args.wandb_notes:
      config['wandb_notes'] = args.wandb_notes

  # Override text encoder settings from command line (only if provided)
  if args.text_model_name:
    config['text_model_name'] = args.text_model_name
  if args.text_backend:
    config['text_backend'] = args.text_backend
  if args.text_prompt_mode:
    config['text_prompt_mode'] = args.text_prompt_mode
  if args.text_output_dim:
    config['text_output_dim'] = args.text_output_dim
  if args.use_text_proj_head is not None:  # Handle boolean explicitly
    config['use_text_proj_head'] = args.use_text_proj_head

  # Determine checkpoint path for resuming
  resume_checkpoint_path = None
  if args.resume:
    resume_checkpoint_path = args.resume
    if not os.path.exists(resume_checkpoint_path):
      print(f"Error: Checkpoint file not found: {resume_checkpoint_path}")
      return
  elif args.resume_from_dir:
    try:
      resume_checkpoint_path = find_latest_checkpoint(args.resume_from_dir)
    except FileNotFoundError as e:
      print(f"Error: {e}")
      return

  # Create trainer and start training
  trainer = SmartHomeTrainer(config)

  # Load checkpoint if resuming
  if resume_checkpoint_path:
    try:
      trainer.load_checkpoint(resume_checkpoint_path)
    except Exception as e:
      print(f"Error loading checkpoint: {e}")
      return

  trainer.train()


if __name__ == "__main__":
  main()
