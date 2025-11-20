#!/usr/bin/env python3

"""
Training script for Chronos-2 encoder with CLIP alignment.

This script trains a Chronos-2 based encoder for smart home activity recognition
using only CLIP-style contrastive learning (no MLM).

Usage:
  python src/training/train_chronos_clip.py --config configs/training/milan/chronos_clip.json
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.chronos_encoder import ChronosEncoder
from models.text_encoder import TextEncoder, build_text_encoder, log_text_encoder_config
from losses.clip import CLIPLoss
from dataio.dataset import SmartHomeDataset
from dataio.collate import create_data_loader
from utils.device_utils import get_optimal_device, get_device_config, log_device_info, optimize_for_device
from utils.training_metrics import TrainingMetrics
from utils.wandb_config import WandBConfig, get_wandb_config_for_experiment

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def compute_training_schedule(config: Dict[str, Any], dataset_size: int) -> Dict[str, Any]:
    """Compute training schedule based on epochs or steps."""
    batch_size = config['batch_size']
    steps_per_epoch = max(1, dataset_size // batch_size)

    has_epochs = 'max_epochs' in config and config['max_epochs'] is not None
    has_steps = 'max_steps' in config and config['max_steps'] is not None

    if has_epochs and not has_steps:
        config['max_steps'] = config['max_epochs'] * steps_per_epoch
        print(f"Computed max_steps={config['max_steps']} from max_epochs={config['max_epochs']}")
    elif has_steps and not has_epochs:
        config['max_epochs'] = max(1, config['max_steps'] // steps_per_epoch)
        print(f"Computed max_epochs={config['max_epochs']} from max_steps={config['max_steps']}")
    else:
        if 'max_epochs' not in config:
            config['max_epochs'] = 50
        if 'max_steps' not in config:
            config['max_steps'] = config['max_epochs'] * steps_per_epoch

    config['steps_per_epoch'] = steps_per_epoch
    return config


class ChronosCLIPTrainer:
    """Trainer for Chronos-2 encoder with CLIP alignment."""

    def __init__(self, config: Dict[str, Any]):
        # Optimize config for device
        if isinstance(config.get('device'), str):
            device = torch.device(config['device'])
        else:
            device = get_optimal_device()

        self.config = optimize_for_device(config, device)
        self.device = device

        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self._save_run_info()

        # Log device information
        log_device_info(self.device)

        # Setup logging
        self.setup_logging()

        # Initialize models
        self.setup_models()

        # Log text encoder configuration
        log_text_encoder_config(self.config, self.logger)

        # Initialize loss function
        self.setup_loss()

        # Initialize data
        self.setup_data()

        # Initialize optimizer
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
        """Save hyperparameters and run information."""
        from datetime import datetime

        run_info = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.copy(),
            'device': str(self.device),
            'pytorch_version': torch.__version__,
        }

        hyperparams_path = os.path.join(self.config['output_dir'], 'hyperparameters.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(run_info, f, indent=2, default=str)

        summary_path = os.path.join(self.config['output_dir'], 'run_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Chronos-2 CLIP Training Run Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Timestamp: {run_info['timestamp']}\n")
            f.write(f"Device: {run_info['device']}\n")
            f.write(f"Output Directory: {self.config['output_dir']}\n\n")
            f.write(f"Key Hyperparameters:\n")
            f.write(f"  - Chronos Model: {self.config.get('chronos_model_name', 'N/A')}\n")
            f.write(f"  - Projection Hidden Dim: {self.config.get('projection_hidden_dim', 'N/A')}\n")
            f.write(f"  - Training: batch_size={self.config.get('batch_size', 'N/A')}, lr={self.config.get('learning_rate', 'N/A')}\n")
            f.write(f"  - Temperature: init={self.config.get('temperature_init', 'N/A')}, learnable={self.config.get('learnable_temperature', 'N/A')}\n")

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = os.path.join('logs', 'text')
        os.makedirs(log_dir, exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = self.config.get('wandb_name', 'chronos_clip_training')
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

        # Initialize wandb if enabled
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb_dir = os.path.join('logs', 'wandb')
            os.makedirs(wandb_dir, exist_ok=True)

            wandb.init(
                project=self.config.get('wandb_project', 'discover-v2'),
                entity=self.config.get('wandb_entity', None),
                name=self.config.get('wandb_name', self.config.get('run_name', None)),
                config=self.config,
                tags=self.config.get('wandb_tags', []),
                notes=self.config.get('wandb_notes', ''),
                group=self.config.get('wandb_group', None),
                job_type='train',
                dir=wandb_dir,
                reinit=True,
                save_code=True
            )
            self.logger.info(f"W&B logging to: {wandb_dir}")

    def setup_models(self):
        """Initialize all models."""
        # Text encoder (frozen)
        self.text_encoder = build_text_encoder(self.config)
        self.text_encoder.to(self.device)

        # Load vocabulary
        with open(self.config['vocab_path'], 'r') as f:
            vocab = json.load(f)

        # Get vocabulary sizes
        # Vocab size must be max_index + 1 to accommodate all indices
        vocab_sizes = {}
        for field, field_vocab in vocab.items():
            if field_vocab:
                max_idx = max(field_vocab.values())
                vocab_sizes[field] = max_idx + 1
            else:
                vocab_sizes[field] = 0
        self.vocab_sizes = vocab_sizes

        # Chronos encoder (Chronos frozen, only projection head trainable)
        self.chronos_encoder = ChronosEncoder(
            vocab_sizes=vocab_sizes,
            chronos_model_name=self.config.get('chronos_model_name', 'amazon/chronos-t5-small'),
            projection_hidden_dim=self.config.get('projection_hidden_dim', 256),
            projection_dropout=self.config.get('projection_dropout', 0.1),
            output_dim=self.config.get('output_dim', 512),
            sequence_length=self.config.get('sequence_length', 50)
        )
        self.chronos_encoder.to(self.device)

        # Freeze Chronos model (projection head remains trainable)
        for name, param in self.chronos_encoder.named_parameters():
            if 'chronos_model' in name:
                param.requires_grad = False

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.chronos_encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.chronos_encoder.parameters())
        self.logger.info(f"Chronos encoder: {trainable_params:,} trainable / {total_params:,} total parameters")

        # Setup W&B model watching
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE and self.config.get('wandb_watch_model', True):
            wandb.watch(
                models=[self.chronos_encoder],
                criterion=None,
                log='all' if self.config.get('wandb_log_gradients', False) else 'parameters',
                log_freq=self.config.get('wandb_watch_freq', 100),
                idx=None
            )

    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Only optimize projection head and text encoder projection (if any)
        trainable_params = list(self.chronos_encoder.projection_head.parameters())

        # Add text encoder projection if it exists
        if hasattr(self.text_encoder, 'clip_proj') and self.text_encoder.clip_proj is not None:
            trainable_params.extend(list(self.text_encoder.clip_proj.parameters()))
        elif hasattr(self.text_encoder, 'proj_head') and self.text_encoder.proj_head is not None:
            trainable_params.extend(list(self.text_encoder.proj_head.parameters()))

        # Add loss function parameters (learnable temperature)
        trainable_params.extend(list(self.loss_fn.parameters()))

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

        self.logger.info(f"Setup optimizer with {len(trainable_params):,} trainable parameters")

    def setup_loss(self):
        """Setup loss function (CLIP only, no MLM)."""
        hard_negative_config = None
        if self.config.get('use_hard_negatives', False):
            hard_negative_config = {
                'memory_size': self.config.get('hard_negative_memory_size', 4096),
                'hard_negative_ratio': self.config.get('hard_negative_ratio', 0.5),
                'sampling_strategy': self.config.get('hard_negative_strategy', 'mixed'),
                'temperature_for_sampling': self.config.get('hard_negative_sampling_temperature', 0.1)
            }

        self.loss_fn = CLIPLoss(
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

        # Create data loader (no MLM, just CLIP alignment)
        self.train_loader = create_data_loader(
            dataset=train_dataset,
            text_encoder=self.text_encoder,
            span_masker=None,  # No MLM
            vocab_sizes=self.vocab_sizes,
            device=self.device,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            apply_mlm=False  # No MLM for Chronos training
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
                span_masker=None,
                vocab_sizes=self.vocab_sizes,
                device=self.device,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                apply_mlm=False
            )
        else:
            self.val_loader = None

        self.logger.info(f"Setup data loaders: train={len(self.train_loader)} batches")
        if self.val_loader:
            self.logger.info(f"  val={len(self.val_loader)} batches")

        # Compute training schedule
        dataset_size = len(train_dataset)
        self.config = compute_training_schedule(self.config, dataset_size)
        self.logger.info(f"Training schedule: {self.config['max_epochs']} epochs, {self.config['max_steps']} steps")

    def forward_pass(self, batch: Dict[str, Any]):
        """Forward pass through the model."""
        # Get sensor embeddings from Chronos encoder
        sensor_embeddings = self.chronos_encoder.forward_clip(
            categorical_features=batch['categorical_features'],
            coordinates=batch['coordinates'],
            time_deltas=batch['time_deltas'],
            mask=batch['mask']
        )

        # Get text embeddings (already computed in collate function)
        text_embeddings = batch['text_embeddings']

        return {
            'sensor_embeddings': sensor_embeddings,
            'text_embeddings': text_embeddings
        }

    def compute_loss(self, model_outputs: Dict[str, Any]):
        """Compute CLIP loss."""
        sensor_embeddings = model_outputs['sensor_embeddings']
        text_embeddings = model_outputs['text_embeddings']

        # Compute CLIP loss
        loss, _ = self.loss_fn(sensor_embeddings, text_embeddings)

        # Get accuracies
        s2t_acc, t2s_acc = self.loss_fn.get_accuracy(sensor_embeddings, text_embeddings)

        loss_dict = {
            'total_loss': loss.item(),
            'clip_loss': loss.item(),
            'sensor_to_text_acc': s2t_acc,
            'text_to_sensor_acc': t2s_acc,
            'temperature': self.loss_fn.temperature.item()
        }

        return loss, loss_dict

    def train_step(self, batch: Dict[str, Any]):
        """Single training step."""
        self.optimizer.zero_grad()

        if self.config['use_amp'] and self.device.type == 'cuda':
            with autocast():
                model_outputs = self.forward_pass(batch)
                loss, loss_dict = self.compute_loss(model_outputs)

            self.scaler.scale(loss).backward()

            if self.config.get('grad_clip_norm'):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.chronos_encoder.projection_head.parameters()) + list(self.loss_fn.parameters()),
                    self.config['grad_clip_norm']
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            model_outputs = self.forward_pass(batch)
            loss, loss_dict = self.compute_loss(model_outputs)
            loss.backward()

            if self.config.get('grad_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    list(self.chronos_encoder.projection_head.parameters()) + list(self.loss_fn.parameters()),
                    self.config['grad_clip_norm']
                )

            self.optimizer.step()

        self.scheduler.step()
        self.global_step += 1

        return loss_dict

    def validate(self):
        """Validation step."""
        if self.val_loader is None:
            return {}

        self.chronos_encoder.eval()

        total_loss = 0.0
        total_s2t_acc = 0.0
        total_t2s_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                model_outputs = self.forward_pass(batch)
                loss, loss_dict = self.compute_loss(model_outputs)

                total_loss += loss_dict['clip_loss']
                total_s2t_acc += loss_dict['sensor_to_text_acc']
                total_t2s_acc += loss_dict['text_to_sensor_acc']
                num_batches += 1

        self.chronos_encoder.train()

        if num_batches > 0:
            return {
                'val_loss': total_loss / num_batches,
                'val_s2t_acc': total_s2t_acc / num_batches,
                'val_t2s_acc': total_t2s_acc / num_batches
            }
        else:
            return {}

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'global_step': self.global_step,
            'chronos_encoder_state_dict': self.chronos_encoder.state_dict(),
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
        self.logger.info(f"Saved checkpoint to: {path}")

    def train(self):
        """Main training loop."""
        self.logger.info("Starting Chronos-2 CLIP training...")
        self.chronos_encoder.train()

        for epoch in range(self.config['max_epochs']):
            epoch_loss = 0.0
            num_batches = 0

            for batch in self.train_loader:
                loss_dict = self.train_step(batch)
                epoch_loss += loss_dict['total_loss']
                num_batches += 1

                # Logging
                if self.global_step % self.config['log_interval'] == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    log_msg = (
                        f"Step {self.global_step:6d} | "
                        f"loss: {loss_dict['total_loss']:.4f} | "
                        f"s2t_acc: {loss_dict['sensor_to_text_acc']:.4f} | "
                        f"t2s_acc: {loss_dict['text_to_sensor_acc']:.4f} | "
                        f"temp: {loss_dict['temperature']:.6f} | "
                        f"lr: {lr:.2e}"
                    )
                    self.logger.info(log_msg)

                    if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
                        wandb.log({
                            'train/step': self.global_step,
                            'train/epoch': epoch,
                            'train/learning_rate': lr,
                            'train/total_loss': loss_dict['total_loss'],
                            'train/clip_loss': loss_dict['clip_loss'],
                            'train/sensor_to_text_acc': loss_dict['sensor_to_text_acc'],
                            'train/text_to_sensor_acc': loss_dict['text_to_sensor_acc'],
                            'train/temperature': loss_dict['temperature'],
                        }, step=self.global_step)

                # Validation
                if self.global_step % self.config['val_interval'] == 0:
                    val_metrics = self.validate()
                    if val_metrics:
                        self.logger.info(
                            f"Validation | loss: {val_metrics['val_loss']:.4f} | "
                            f"s2t_acc: {val_metrics['val_s2t_acc']:.4f} | "
                            f"t2s_acc: {val_metrics['val_t2s_acc']:.4f}"
                        )

                        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
                            wandb.log({
                                'val/total_loss': val_metrics['val_loss'],
                                'val/sensor_to_text_acc': val_metrics['val_s2t_acc'],
                                'val/text_to_sensor_acc': val_metrics['val_t2s_acc'],
                            }, step=self.global_step)

                        # Save best model
                        if val_metrics['val_loss'] < self.best_loss:
                            self.best_loss = val_metrics['val_loss']
                            best_path = os.path.join(self.config['output_dir'], 'best_model.pt')
                            self.save_checkpoint(best_path)

                # Save checkpoint
                if self.global_step % self.config['save_interval'] == 0:
                    checkpoint_path = os.path.join(
                        self.config['output_dir'], f'checkpoint_step_{self.global_step}.pt'
                    )
                    self.save_checkpoint(checkpoint_path)

                # Check if we've reached max steps
                if self.global_step >= self.config['max_steps']:
                    break

            if self.global_step >= self.config['max_steps']:
                break

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            self.logger.info(f"Epoch {epoch} completed | Average loss: {avg_epoch_loss:.4f}")

        # Save final checkpoint
        final_path = os.path.join(self.config['output_dir'], 'final_model.pt')
        self.save_checkpoint(final_path)

        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.finish()

        self.logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train Chronos-2 encoder with CLIP alignment')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--train_data', type=str, help='Path to training data')
    parser.add_argument('--vocab', type=str, help='Path to vocabulary file')
    parser.add_argument('--val_data', type=str, help='Path to validation data')
    parser.add_argument('--output_dir', type=str, help='Output directory')

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        device = get_optimal_device()
        device_config = get_device_config(device)

        config = {
            'chronos_model_name': 'amazon/chronos-t5-small',
            'projection_hidden_dim': 256,
            'projection_dropout': 0.1,
            'output_dim': 512,
            'sequence_length': 50,
            'batch_size': 64,
            'learning_rate': 0.001,
            'betas': (0.9, 0.98),
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'max_epochs': 50,
            'grad_clip_norm': 1.0,
            'temperature_init': 0.02,
            'learnable_temperature': True,
            'use_hard_negatives': False,
            'log_interval': 50,
            'val_interval': 500,
            'save_interval': 2000,
            'use_amp': False,
            'num_workers': 0,
            'use_wandb': True,
            'wandb_project': 'discover-v2',
            'wandb_name': 'chronos_clip',
            'wandb_tags': ['chronos', 'clip', 'milan'],
        }
        config.update(device_config)
        config['device'] = device.type

    # Override with command line arguments
    if args.train_data:
        config['train_data_path'] = args.train_data
    if args.vocab:
        config['vocab_path'] = args.vocab
    if args.val_data:
        config['val_data_path'] = args.val_data
    if args.output_dir:
        config['output_dir'] = args.output_dir

    # Validate required paths
    if not config.get('train_data_path'):
        print("Error: train_data_path must be specified")
        return
    if not config.get('vocab_path'):
        print("Error: vocab_path must be specified")
        return

    # Create trainer and start training
    trainer = ChronosCLIPTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

