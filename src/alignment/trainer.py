"""
Trainer for alignment models.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.alignment.config import AlignmentConfig
from src.alignment.model import AlignmentModel
from src.alignment.dataset import AlignmentDataset
from src.alignment.wandb_utils import (
    generate_wandb_run_name,
    generate_wandb_group,
    generate_wandb_tags
)
from src.utils.device_utils import get_optimal_device


class AlignmentTrainer:
    """
    Trainer for sensor-text alignment.

    Handles:
    - Data loading (sensor data + text embeddings)
    - Training loop with CLIP (+ optional MLM) loss
    - Validation and metrics tracking
    - Checkpointing and logging
    - WandB integration
    """

    def __init__(self, config: AlignmentConfig):
        self.config = config

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup logging (must be before device setup since device setup logs)
        self.logger = self._setup_logging()

        # Setup device
        self.device = self._setup_device()

        # Save config
        self._save_config()

        # Setup data
        self.train_loader, self.val_loader, self.vocab_sizes = self._setup_data()

        # Compute training schedule
        self._compute_training_schedule()

        # Setup model
        self.model = AlignmentModel(config, self.vocab_sizes)
        self.model.to(self.device)

        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} total parameters")
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # Setup span masker for MLM (if enabled)
        self.span_masker = self._setup_span_masker() if config.loss.mlm_weight > 0 else None
        if self.span_masker is not None:
            self.logger.info(f"MLM enabled with mask_prob={config.loss.mask_prob}, mean_span_length={config.loss.mean_span_length}")
            # Update datasets with span_masker and vocab_sizes
            self.train_loader.dataset.span_masker = self.span_masker
            self.train_loader.dataset.vocab_sizes = self.vocab_sizes
            if self.val_loader is not None:
                self.val_loader.dataset.span_masker = self.span_masker
                self.val_loader.dataset.vocab_sizes = self.vocab_sizes

        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = self._setup_optimizer()

        # Setup training utilities
        self.scaler = GradScaler() if config.training.use_amp and self.device.type == 'cuda' else None
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Setup WandB
        if config.use_wandb and WANDB_AVAILABLE:
            self._setup_wandb()

    def _setup_device(self) -> torch.device:
        """Setup training device."""
        device_str = self.config.training.device

        if device_str == 'auto':
            device = get_optimal_device()
        else:
            device = torch.device(device_str)

        self.logger.info(f"Using device: {device}")

        return device

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        log_dir = Path('logs/text')
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = self.config.experiment_name
        log_file = log_dir / f'{exp_name}_{timestamp}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Logging to: {log_file}")

        return logger

    def _save_config(self):
        """Save configuration to output directory."""
        config_path = Path(self.config.output_dir) / 'config.yaml'
        self.config.to_yaml(str(config_path))
        self.logger.info(f"Saved config to: {config_path}")

    def _setup_data(self):
        """Setup datasets and data loaders."""
        # Load vocabulary
        import json
        with open(self.config.vocab_path, 'r') as f:
            vocab = json.load(f)

        vocab_sizes = {field: len(field_vocab) + 1 for field, field_vocab in vocab.items()}

        # Create training dataset
        train_dataset = AlignmentDataset(
            data_path=self.config.train_data_path,
            text_embeddings_path=self.config.train_text_embeddings_path,
            captions_path=self.config.train_captions_path,
            text_encoder_config_path=self.config.text_encoder_config_path,
            vocab=vocab,
            device=self.device
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.config.training.shuffle,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            collate_fn=train_dataset.collate_fn
        )

        # Create validation dataset (if specified)
        val_loader = None
        if self.config.val_data_path:
            val_dataset = AlignmentDataset(
                data_path=self.config.val_data_path,
                text_embeddings_path=self.config.val_text_embeddings_path,
                captions_path=self.config.val_captions_path,
                text_encoder_config_path=self.config.text_encoder_config_path,
                vocab=vocab,
                device=self.device
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=self.config.training.num_workers,
                pin_memory=self.config.training.pin_memory,
                collate_fn=val_dataset.collate_fn
            )

        self.logger.info(f"Training dataset: {len(train_dataset)} samples")
        if val_loader:
            self.logger.info(f"Validation dataset: {len(val_dataset)} samples")

        return train_loader, val_loader, vocab_sizes

    def _compute_training_schedule(self):
        """Compute training schedule (epochs/steps)."""
        dataset_size = len(self.train_loader.dataset)
        steps_per_epoch = len(self.train_loader)

        if self.config.training.max_steps is not None and self.config.training.max_epochs is not None:
            # Both specified - use as-is
            pass
        elif self.config.training.max_epochs is not None:
            # Compute steps from epochs
            self.config.training.max_steps = self.config.training.max_epochs * steps_per_epoch
        elif self.config.training.max_steps is not None:
            # Compute epochs from steps
            self.config.training.max_epochs = max(1, self.config.training.max_steps // steps_per_epoch)
        else:
            # Default: 10 epochs
            self.config.training.max_epochs = 10
            self.config.training.max_steps = 10 * steps_per_epoch

        self.logger.info(f"Training schedule: {self.config.training.max_epochs} epochs, {self.config.training.max_steps} steps")
        self.logger.info(f"Steps per epoch: {steps_per_epoch}")

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Get trainable parameters
        params = self.model.get_trainable_parameters()

        # Create optimizer
        opt_config = self.config.optimizer

        if opt_config.type == 'adamw':
            optimizer = torch.optim.AdamW(
                params,
                lr=opt_config.learning_rate,
                betas=opt_config.betas,
                weight_decay=opt_config.weight_decay
            )
        elif opt_config.type == 'adam':
            optimizer = torch.optim.Adam(
                params,
                lr=opt_config.learning_rate,
                betas=opt_config.betas,
                weight_decay=opt_config.weight_decay
            )
        elif opt_config.type == 'sgd':
            optimizer = torch.optim.SGD(
                params,
                lr=opt_config.learning_rate,
                momentum=0.9,
                weight_decay=opt_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config.type}")

        # Create scheduler (cosine annealing with warmup)
        total_steps = self.config.training.max_steps
        warmup_steps = int(opt_config.warmup_ratio * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        self.logger.info(f"Optimizer: {opt_config.type}, LR: {opt_config.learning_rate:.2e}")
        self.logger.info(f"Warmup steps: {warmup_steps}")

        return optimizer, scheduler

    def _setup_span_masker(self):
        """Setup span masker for MLM."""
        from src.models.mlm_heads import SpanMasker

        span_masker = SpanMasker(
            mask_prob=self.config.loss.mask_prob,
            mean_span_length=self.config.loss.mean_span_length,
            # Use defaults for other advanced features
            enable_field_blackout=False,  # Disabled for now
            p_transition_seed=0.0,  # Disabled for now
            strict_corr_mask=False,  # Disabled for now
            adaptive_mask=False  # Disabled for now
        )

        return span_masker

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        wandb_dir = Path('logs/wandb')
        wandb_dir.mkdir(parents=True, exist_ok=True)

        # Generate intelligent run name, group, and tags if not provided
        run_name = self.config.wandb_name
        if not run_name:
            run_name = generate_wandb_run_name(self.config)
            self.logger.info(f"Auto-generated WandB run name: {run_name}")

        group = self.config.wandb_group
        if not group:
            group = generate_wandb_group(self.config)
            self.logger.info(f"Auto-generated WandB group: {group}")

        tags = self.config.wandb_tags
        if not tags:
            tags = generate_wandb_tags(self.config)
            self.logger.info(f"Auto-generated WandB tags: {tags}")

        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=run_name,
            config=self.config.__dict__,
            tags=tags,
            notes=self.config.wandb_notes,
            group=group,
            dir=str(wandb_dir),
            reinit=True,
            save_code=True
        )

        # Watch model
        wandb.watch(
            self.model,
            log='all',
            log_freq=self.config.training.log_interval
        )

        self.logger.info(f"WandB logging to: {wandb_dir}")
        self.logger.info(f"WandB URL: https://wandb.ai/{self.config.wandb_entity or 'user'}/{self.config.wandb_project}/{wandb.run.id}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        if self.config.training.use_amp and self.device.type == 'cuda':
            with autocast():
                outputs = self.model(
                    sensor_data=batch['sensor_data'],
                    text_embeddings=batch['text_embeddings'],
                    attention_mask=batch.get('attention_mask'),
                    return_encoder_output=True
                )

                loss, loss_dict = self.model.compute_loss(
                    sensor_embeddings_projected=outputs['sensor_embeddings_projected'],
                    text_embeddings_projected=outputs['text_embeddings_projected'],
                    mlm_predictions=outputs.get('mlm_predictions'),  # From forward pass, not batch
                    mlm_labels=batch.get('mlm_labels'),
                    mlm_mask_positions=batch.get('mlm_mask_positions')
                )

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            if self.config.optimizer.grad_clip_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.grad_clip_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular precision
            outputs = self.model(
                sensor_data=batch['sensor_data'],
                text_embeddings=batch['text_embeddings'],
                attention_mask=batch.get('attention_mask'),
                return_encoder_output=True
            )

            loss, loss_dict = self.model.compute_loss(
                sensor_embeddings_projected=outputs['sensor_embeddings_projected'],
                text_embeddings_projected=outputs['text_embeddings_projected'],
                mlm_predictions=outputs.get('mlm_predictions'),  # From forward pass, not batch
                mlm_labels=batch.get('mlm_labels'),
                mlm_mask_positions=batch.get('mlm_mask_positions')
            )

            # Backward pass
            loss.backward()

            if self.config.optimizer.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.grad_clip_norm
                )

            self.optimizer.step()

        # Update scheduler
        self.scheduler.step()
        self.global_step += 1

        return loss_dict

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation step."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_clip_loss = 0.0
        total_s2t_acc = 0.0
        total_t2s_acc = 0.0
        total_mlm_loss = 0.0
        num_batches = 0
        has_mlm = False

        for batch in self.val_loader:
            outputs = self.model(
                sensor_data=batch['sensor_data'],
                text_embeddings=batch['text_embeddings'],
                attention_mask=batch.get('attention_mask'),
                return_encoder_output=True
            )

            loss, loss_dict = self.model.compute_loss(
                sensor_embeddings_projected=outputs['sensor_embeddings_projected'],
                text_embeddings_projected=outputs['text_embeddings_projected'],
                mlm_predictions=outputs.get('mlm_predictions'),  # From forward pass, not batch
                mlm_labels=batch.get('mlm_labels'),
                mlm_mask_positions=batch.get('mlm_mask_positions')
            )

            total_loss += loss_dict['total_loss']
            total_clip_loss += loss_dict['clip_loss']
            total_s2t_acc += loss_dict['sensor_to_text_acc']
            total_t2s_acc += loss_dict['text_to_sensor_acc']

            if 'mlm_loss' in loss_dict:
                total_mlm_loss += loss_dict['mlm_loss']
                has_mlm = True

            num_batches += 1

        metrics = {
            'val_loss': total_loss / num_batches,
            'val_clip_loss': total_clip_loss / num_batches,
            'val_s2t_acc': total_s2t_acc / num_batches,
            'val_t2s_acc': total_t2s_acc / num_batches,
        }

        if has_mlm:
            metrics['val_mlm_loss'] = total_mlm_loss / num_batches

        return metrics

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

        self.logger.info(f"Saved checkpoint to: {path}")

        if is_best:
            best_path = Path(self.config.output_dir) / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to: {best_path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.logger.info(f"Loaded checkpoint from: {path} (step {self.global_step})")

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Output directory: {self.config.output_dir}")

        for epoch in range(self.config.training.max_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in self.train_loader:
                # Training step
                loss_dict = self.train_step(batch)
                epoch_loss += loss_dict['total_loss']
                num_batches += 1

                # Logging
                if self.global_step % self.config.training.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]

                    log_msg = (
                        f"Step {self.global_step:6d} | "
                        f"Loss: {loss_dict['total_loss']:.4f} | "
                        f"CLIP: {loss_dict['clip_loss']:.4f} | "
                        f"S2T: {loss_dict['sensor_to_text_acc']:.3f} | "
                        f"T2S: {loss_dict['text_to_sensor_acc']:.3f} | "
                        f"Temp: {loss_dict['temperature']:.6f} | "
                        f"LR: {lr:.2e}"
                    )

                    if 'mlm_loss' in loss_dict:
                        log_msg += f" | MLM: {loss_dict['mlm_loss']:.4f}"

                    self.logger.info(log_msg)

                    # WandB logging
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb_dict = {
                            'train/total_loss': loss_dict['total_loss'],
                            'train/clip_loss': loss_dict['clip_loss'],
                            'train/s2t_acc': loss_dict['sensor_to_text_acc'],
                            'train/t2s_acc': loss_dict['text_to_sensor_acc'],
                            'train/temperature': loss_dict['temperature'],
                            'train/learning_rate': lr,
                            'train/step': self.global_step,
                            'train/epoch': epoch,
                        }

                        if 'mlm_loss' in loss_dict:
                            wandb_dict['train/mlm_loss'] = loss_dict['mlm_loss']

                        wandb.log(wandb_dict, step=self.global_step)

                # Validation
                if self.global_step % self.config.training.val_interval == 0:
                    val_metrics = self.validate()

                    if val_metrics:
                        val_msg = (
                            f"Validation | "
                            f"Loss: {val_metrics['val_loss']:.4f} | "
                            f"CLIP: {val_metrics['val_clip_loss']:.4f} | "
                            f"S2T: {val_metrics['val_s2t_acc']:.3f} | "
                            f"T2S: {val_metrics['val_t2s_acc']:.3f}"
                        )

                        if 'val_mlm_loss' in val_metrics:
                            val_msg += f" | MLM: {val_metrics['val_mlm_loss']:.4f}"

                        self.logger.info(val_msg)

                        # WandB logging
                        if self.config.use_wandb and WANDB_AVAILABLE:
                            wandb.log(val_metrics, step=self.global_step)

                        # Save best model
                        if val_metrics['val_loss'] < self.best_val_loss:
                            self.best_val_loss = val_metrics['val_loss']
                            self.save_checkpoint(
                                str(Path(self.config.output_dir) / 'best_model.pt'),
                                is_best=True
                            )

                # Save checkpoint
                if self.global_step % self.config.training.save_interval == 0:
                    checkpoint_path = Path(self.config.output_dir) / f'checkpoint_step_{self.global_step}.pt'
                    self.save_checkpoint(str(checkpoint_path))

                # Check max steps
                if self.global_step >= self.config.training.max_steps:
                    break

            if self.global_step >= self.config.training.max_steps:
                break

            # End of epoch
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            self.logger.info(f"Epoch {epoch} completed | Average loss: {avg_loss:.4f}")

        # Save final checkpoint
        final_path = Path(self.config.output_dir) / 'final_model.pt'
        self.save_checkpoint(str(final_path))

        self.logger.info("Training completed!")

        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

