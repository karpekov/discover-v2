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
from src.utils.training_metrics import TrainingMetrics


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
        self.train_loader, self.val_loader, self.vocab_sizes, self.vocab = self._setup_data()

        # Compute training schedule
        self._compute_training_schedule()

        # Setup model (pass vocab for image-based encoders)
        self.model = AlignmentModel(config, self.vocab_sizes, vocab=self.vocab)
        self.model.to(self.device)

        self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} total parameters")
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # Setup span masker for MLM (if enabled)
        self.span_masker = self._setup_span_masker() if config.loss.mlm_weight > 0 else None
        if self.span_masker is not None:
            self.logger.info(f"MLM enabled with mask_prob={config.loss.mask_prob}, mean_span_length={config.loss.mean_span_length}")
            # Update datasets with span_masker and vocab_sizes (filtered to exclude activity labels)
            self.train_loader.dataset.span_masker = self.span_masker
            self.train_loader.dataset.vocab_sizes = self.vocab_sizes  # Already filtered to sensor fields only
            if self.val_loader is not None:
                self.val_loader.dataset.span_masker = self.span_masker
                self.val_loader.dataset.vocab_sizes = self.vocab_sizes  # Already filtered to sensor fields only

        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = self._setup_optimizer()

        # Setup training utilities
        self.scaler = GradScaler() if config.training.use_amp and self.device.type == 'cuda' else None
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Setup metrics tracker
        self.metrics_tracker = TrainingMetrics(
            vocab_sizes=self.vocab_sizes,
            sample_size=config.training.metrics_sample_size if hasattr(config.training, 'metrics_sample_size') else 1000,
            text_encoder=None  # We don't need text encoder for non-F1 metrics
        )

        # Extract text encoder metadata for checkpoint (critical for evaluation!)
        self.text_encoder_metadata = self._extract_text_encoder_metadata()
        self.logger.info(f"Text encoder metadata: {self.text_encoder_metadata}")

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
        import yaml
        with open(self.config.vocab_path, 'r') as f:
            vocab = json.load(f)

        vocab_sizes = {field: len(field_vocab) + 1 for field, field_vocab in vocab.items()}

        # IMPORTANT: Filter vocab_sizes to only include fields used by encoder
        # Activity labels should NOT be used for self-supervised MLM training!
        # Get encoder config (either inline or from file)
        if self.config.encoder is not None:
            encoder_config = self.config.encoder
        else:
            with open(self.config.encoder_config_path, 'r') as f:
                encoder_config = yaml.safe_load(f)

        categorical_fields = encoder_config.get('metadata', {}).get('categorical_fields', [])

        # Filter vocab_sizes to only fields that are actually embedded by the encoder
        # This prevents activity labels from being used as MLM prediction targets
        mlm_vocab_sizes = {
            field: size for field, size in vocab_sizes.items()
            if field in categorical_fields
        }

        self.logger.info(f"Loaded vocabulary with {len(vocab_sizes)} fields")
        self.logger.info(f"Using {len(mlm_vocab_sizes)} fields for MLM: {list(mlm_vocab_sizes.keys())}")
        self.logger.info(f"Excluded fields (for evaluation only): {[f for f in vocab_sizes if f not in mlm_vocab_sizes]}")

        # Create training dataset
        train_dataset = AlignmentDataset(
            data_path=self.config.train_data_path,
            text_embeddings_path=self.config.train_text_embeddings_path,
            captions_path=self.config.train_captions_path,
            text_encoder_config_path=self.config.text_encoder_config_path,
            vocab=vocab,
            device=self.device,
            categorical_fields=categorical_fields  # Pass filtered fields
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
                device=self.device,
                categorical_fields=categorical_fields  # Pass filtered fields
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

        # Return mlm_vocab_sizes (filtered) for model, but keep full vocab for reference
        return train_loader, val_loader, mlm_vocab_sizes, vocab

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

    def _extract_text_encoder_metadata(self) -> Dict[str, Any]:
        """
        Extract text encoder metadata from .npz embeddings file.

        This is critical for evaluation - we need to know which text encoder
        was used during training so evaluation can use the same one.

        Returns:
            Dict with: encoder_type, model_name, embedding_dim, projection_dim
        """
        import numpy as np

        # Try to load from training embeddings path
        if hasattr(self.config, 'train_text_embeddings_path') and self.config.train_text_embeddings_path:
            try:
                data = np.load(self.config.train_text_embeddings_path)
                metadata = {
                    'encoder_type': str(data['encoder_type'].item()) if 'encoder_type' in data else 'unknown',
                    'model_name': str(data['model_name'].item()) if 'model_name' in data else 'unknown',
                    'embedding_dim': int(data['embedding_dim'].item()) if 'embedding_dim' in data else data['embeddings'].shape[1],
                    'projection_dim': int(data['projection_dim'].item()) if 'projection_dim' in data else 0,
                    'normalize': bool(data['normalize'].item()) if 'normalize' in data else True,
                    'source': 'npz_file'
                }
                self.logger.info(f"Extracted text encoder metadata from {self.config.train_text_embeddings_path}")
                return metadata
            except Exception as e:
                self.logger.warning(f"Could not extract text encoder metadata from .npz file: {e}")

        # Fallback: Return unknown (evaluation will use CLIP by default)
        self.logger.warning("No text encoder metadata found - evaluation will default to CLIP")
        return {
            'encoder_type': 'unknown',
            'model_name': 'unknown',
            'embedding_dim': 512,
            'projection_dim': 0,
            'normalize': True,
            'source': 'default'
        }

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
        """Setup span masker for MLM with flexible field configuration."""
        from src.models.mlm_heads import SpanMasker

        # Get active categorical fields from encoder config
        if self.config.encoder is not None:
            encoder_config = self.config.encoder
        else:
            import yaml
            with open(self.config.encoder_config_path, 'r') as f:
                encoder_config = yaml.safe_load(f)

        categorical_fields = set(encoder_config.get('metadata', {}).get('categorical_fields', []))

        # Define default field-specific masking probabilities
        all_field_priors = {
            'sensor': 0.20,    # Lower - often correlated with room_id
            'room_id': 0.20,   # Lower - often correlated with sensor
            'state': 0.35,     # Higher - independent temporal field, most informative
        }

        # Filter to only active fields
        field_priors = {
            field: prob for field, prob in all_field_priors.items()
            if field in categorical_fields
        }

        # If no priors defined for a field, use default
        for field in categorical_fields:
            if field not in field_priors:
                field_priors[field] = 0.25  # Default

        span_masker = SpanMasker(
            mask_prob=self.config.loss.mask_prob,
            mean_span_length=self.config.loss.mean_span_length,
            field_priors=field_priors,
            strict_corr_mask=True,
            correlated_field_prob=0.95,
            enable_field_blackout=True,
            p_transition_seed=0.3,
            adaptive_mask=False
        )

        # Define correlated fields (only if both are active)
        # sensor ↔ room_id are perfectly correlated (same sensor always in same room)
        correlated_groups = []
        if 'sensor' in categorical_fields and 'room_id' in categorical_fields:
            correlated_groups.append(['sensor', 'room_id'])

        span_masker.correlated_groups = correlated_groups

        # Update blackout mapping (only include active fields)
        blackout_mapping = {}
        if 'room_id' in categorical_fields:
            blackout_targets = []
            if 'sensor' in categorical_fields:
                blackout_targets.append('sensor')
            blackout_targets.append('coordinates')  # Always blackout coordinates with room
            blackout_mapping['room_id'] = blackout_targets

        if 'sensor' in categorical_fields:
            blackout_targets = []
            if 'room_id' in categorical_fields:
                blackout_targets.append('room_id')
            blackout_targets.append('coordinates')  # Always blackout coordinates with sensor
            blackout_mapping['sensor'] = blackout_targets

        span_masker.blackout_mapping = blackout_mapping

        self.logger.info(f"MLM span masker configured for fields: {sorted(categorical_fields)}")
        if correlated_groups:
            self.logger.info(f"  Correlated groups: {correlated_groups}")
        if blackout_mapping:
            self.logger.info(f"  Blackout mapping: {blackout_mapping}")

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

    def move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        # Move sensor data
        batch['sensor_data']['categorical_features'] = {
            field: tensor.to(self.device)
            for field, tensor in batch['sensor_data']['categorical_features'].items()
        }
        batch['sensor_data']['coordinates'] = batch['sensor_data']['coordinates'].to(self.device)
        batch['sensor_data']['time_deltas'] = batch['sensor_data']['time_deltas'].to(self.device)

        # Move text embeddings and masks
        batch['text_embeddings'] = batch['text_embeddings'].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)

        # Move MLM data if present (both are dicts of tensors)
        if 'mlm_labels' in batch:
            batch['mlm_labels'] = {
                field: tensor.to(self.device)
                for field, tensor in batch['mlm_labels'].items()
            }
        if 'mlm_mask_positions' in batch:
            batch['mlm_mask_positions'] = {
                field: tensor.to(self.device)
                for field, tensor in batch['mlm_mask_positions'].items()
            }

        return batch

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Move batch to device
        batch = self.move_batch_to_device(batch)

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
        """
        Validation step with comprehensive metrics.

        Computes:
        - Basic losses (total, CLIP, MLM)
        - Basic accuracies (S2T, T2S)
        - Alignment health (pos/neg cosine similarities, gap)
        - MLM accuracy (per-field and overall)
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        # Basic loss tracking
        total_loss = 0.0
        total_clip_loss = 0.0
        total_s2t_acc = 0.0
        total_t2s_acc = 0.0
        total_mlm_loss = 0.0
        num_batches = 0
        has_mlm = False

        # Collect embeddings and outputs for comprehensive metrics
        all_sensor_embeddings = []
        all_text_embeddings = []
        all_mlm_predictions = {}
        all_mlm_labels = {}
        all_mlm_mask_positions = {}

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = self.move_batch_to_device(batch)

                outputs = self.model(
                    sensor_data=batch['sensor_data'],
                    text_embeddings=batch['text_embeddings'],
                    attention_mask=batch.get('attention_mask'),
                    return_encoder_output=True
                )

                loss, loss_dict = self.model.compute_loss(
                    sensor_embeddings_projected=outputs['sensor_embeddings_projected'],
                    text_embeddings_projected=outputs['text_embeddings_projected'],
                    mlm_predictions=outputs.get('mlm_predictions'),
                    mlm_labels=batch.get('mlm_labels'),
                    mlm_mask_positions=batch.get('mlm_mask_positions')
                )

                # Accumulate basic metrics
                total_loss += loss_dict['total_loss']
                total_clip_loss += loss_dict['clip_loss']
                total_s2t_acc += loss_dict['sensor_to_text_acc']
                total_t2s_acc += loss_dict['text_to_sensor_acc']

                if 'mlm_loss' in loss_dict:
                    total_mlm_loss += loss_dict['mlm_loss']
                    has_mlm = True

                # Collect embeddings for alignment health
                all_sensor_embeddings.append(outputs['sensor_embeddings_projected'])
                all_text_embeddings.append(outputs['text_embeddings_projected'])

                # Collect MLM outputs
                if outputs.get('mlm_predictions') is not None:
                    for field, preds in outputs['mlm_predictions'].items():
                        if field not in all_mlm_predictions:
                            all_mlm_predictions[field] = []
                        all_mlm_predictions[field].append(preds)

                    for field, labels in batch.get('mlm_labels', {}).items():
                        if field not in all_mlm_labels:
                            all_mlm_labels[field] = []
                        all_mlm_labels[field].append(labels)

                    for field, mask_pos in batch.get('mlm_mask_positions', {}).items():
                        if field not in all_mlm_mask_positions:
                            all_mlm_mask_positions[field] = []
                        all_mlm_mask_positions[field].append(mask_pos)

                num_batches += 1

        # Basic metrics
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_clip_loss': total_clip_loss / num_batches,
            'val_s2t_acc': total_s2t_acc / num_batches,
            'val_t2s_acc': total_t2s_acc / num_batches,
        }

        if has_mlm:
            metrics['val_mlm_loss'] = total_mlm_loss / num_batches

        # Compute alignment health metrics
        if all_sensor_embeddings:
            combined_sensor_emb = torch.cat(all_sensor_embeddings, dim=0)
            combined_text_emb = torch.cat(all_text_embeddings, dim=0)

            try:
                alignment_metrics = self.metrics_tracker.compute_alignment_health(
                    combined_sensor_emb,
                    combined_text_emb,
                    captions=None  # Skip per-activity breakdown for speed
                )
                # Add 'val_' prefix to alignment metrics
                for key, value in alignment_metrics.items():
                    metrics[f'val_{key}'] = value
            except Exception as e:
                self.logger.warning(f"Error computing alignment health: {e}")

        # Compute MLM accuracy metrics
        if all_mlm_predictions and all_mlm_labels and all_mlm_mask_positions:
            try:
                # Concatenate MLM outputs (handle per-field errors gracefully)
                combined_mlm_preds = {}
                combined_mlm_labels = {}
                combined_mlm_masks = {}

                for field in all_mlm_predictions.keys():
                    try:
                        pred_list = all_mlm_predictions[field]
                        label_list = all_mlm_labels.get(field, [])
                        mask_list = all_mlm_mask_positions.get(field, [])

                        if pred_list and label_list and mask_list:
                            # Concatenate along batch dimension
                            combined_mlm_preds[field] = torch.cat(pred_list, dim=0)
                            combined_mlm_labels[field] = torch.cat(label_list, dim=0)
                            combined_mlm_masks[field] = torch.cat(mask_list, dim=0)
                    except Exception as field_error:
                        # Skip this field if concatenation fails - log as warning to make visible
                        self.logger.warning(f"Skipping MLM accuracy for field '{field}': {field_error}")
                        continue

                # Compute accuracy only for successfully concatenated fields
                if combined_mlm_preds and combined_mlm_labels and combined_mlm_masks:
                    mlm_metrics = self.metrics_tracker.compute_mlm_accuracy(
                        combined_mlm_preds,
                        combined_mlm_labels,
                        combined_mlm_masks
                    )
                    # Add 'val_' prefix to MLM metrics
                    for key, value in mlm_metrics.items():
                        metrics[f'val_{key}'] = value

                    # Log which MLM metrics were computed
                    mlm_fields = [k.replace('mlm_accuracy/', '') for k in mlm_metrics.keys() if k.startswith('mlm_accuracy/') and not k.endswith('_top5')]
                    self.logger.debug(f"Computed MLM accuracy for fields: {mlm_fields}")
                else:
                    self.logger.warning("No MLM predictions/labels/masks available for accuracy computation")
            except Exception as e:
                self.logger.warning(f"Error computing MLM accuracy: {e}")
                import traceback
                self.logger.warning(f"Traceback: {traceback.format_exc()}")

        self.model.train()
        return metrics

    @torch.no_grad()
    def compute_comprehensive_metrics(
        self,
        data_loader: DataLoader,
        split_name: str = "train",
        max_batches: int = 10
    ) -> Dict[str, float]:
        """
        Compute comprehensive retrieval metrics on a subset of data.

        Computes:
        - Recall@1, 5, 10 (overall for both sensor→text and text→sensor)
        - nDCG@1, 5, 10 (overall for both directions)
        - Recall@5 per-class breakdown

        Args:
            data_loader: DataLoader to evaluate
            split_name: Name of split ('train' or 'val')
            max_batches: Maximum number of batches to evaluate

        Returns:
            Dictionary of computed metrics with split prefix
        """
        self.model.eval()

        # Collect embeddings and labels from multiple batches
        all_sensor_embeddings = []
        all_text_embeddings = []
        all_ground_truth_labels = []
        all_l1_labels = []

        batch_count = 0
        for batch in data_loader:
            if batch_count >= max_batches:
                break

            # Move batch to device
            batch = self.move_batch_to_device(batch)

            outputs = self.model(
                sensor_data=batch['sensor_data'],
                text_embeddings=batch['text_embeddings'],
                attention_mask=batch.get('attention_mask'),
                return_encoder_output=True
            )

            # Collect projected embeddings
            all_sensor_embeddings.append(outputs['sensor_embeddings_projected'])
            all_text_embeddings.append(outputs['text_embeddings_projected'])

            # Collect ground truth labels if available (for per-class metrics)
            if 'activity_label' in batch:
                all_ground_truth_labels.extend(batch['activity_label'])

            # Collect L1 labels for label-recall metric (collect from all batches, even if some are None)
            if 'activity_label_l1' in batch:
                if batch['activity_label_l1'] is not None:
                    all_l1_labels.extend(batch['activity_label_l1'])
                else:
                    # Batch has no labels - add None for each sample in batch
                    batch_size = outputs['sensor_embeddings_projected'].size(0)
                    all_l1_labels.extend([None] * batch_size)

            batch_count += 1

        if not all_sensor_embeddings:
            self.logger.warning(f"No data collected for {split_name} comprehensive metrics")
            return {}

        # Concatenate all embeddings
        combined_sensor_emb = torch.cat(all_sensor_embeddings, dim=0)
        combined_text_emb = torch.cat(all_text_embeddings, dim=0)

        metrics = {}

        # Compute Recall@K (overall)
        try:
            recall_metrics = self.metrics_tracker.compute_recall_at_k(
                combined_sensor_emb,
                combined_text_emb,
                k_values=[1, 5, 10],
                ground_truth_labels=all_ground_truth_labels if all_ground_truth_labels else None
            )

            # Filter to keep only overall metrics and per-class Recall@5
            for key, value in recall_metrics.items():
                # Keep overall metrics for all k values
                if any(x in key for x in ['/sensor_to_text', '/text_to_sensor', '/average']) and 'class_' not in key:
                    metrics[key] = value
                # Keep per-class metrics only for k=5
                elif 'recall@5/class_' in key:
                    metrics[key] = value

        except Exception as e:
            self.logger.warning(f"Error computing Recall@K metrics: {e}")

        # Compute nDCG@K (overall only, no per-class)
        try:
            ndcg_metrics = self.metrics_tracker.compute_ndcg_at_k(
                combined_sensor_emb,
                combined_text_emb,
                k_values=[1, 5, 10],
                ground_truth_labels=None  # Skip per-class for nDCG
            )

            # Keep only overall metrics (no per-class)
            for key, value in ndcg_metrics.items():
                if 'class_' not in key:
                    metrics[key] = value

        except Exception as e:
            self.logger.warning(f"Error computing nDCG@K metrics: {e}")

        # Compute Label-Recall@10 for L1 labels (lightweight)
        # Only compute if we have labels and the count matches (function will filter None values)
        if all_l1_labels and len(all_l1_labels) == combined_sensor_emb.size(0):
            try:
                label_recall_metrics = self.metrics_tracker.compute_label_recall_at_10(
                    combined_sensor_emb,
                    combined_text_emb,
                    labels=all_l1_labels
                )
                # Add label-recall metrics (only if computation succeeded)
                if label_recall_metrics:
                    for key, value in label_recall_metrics.items():
                        metrics[key] = value
            except Exception as e:
                self.logger.debug(f"Error computing label-recall@10 (skipping): {e}")

        # Add split prefix to all metrics
        prefixed_metrics = {}
        for key, value in metrics.items():
            prefixed_metrics[f'{split_name}/{key}'] = value

        # Log sample size
        total_samples = combined_sensor_emb.size(0)
        prefixed_metrics[f'{split_name}/num_samples'] = total_samples

        self.logger.info(
            f"Comprehensive metrics for {split_name}: "
            f"collected {total_samples} samples from {batch_count} batches"
        )

        self.model.train()
        return prefixed_metrics

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'text_encoder_metadata': self.text_encoder_metadata,  # CRITICAL for evaluation!
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

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
                        # Basic metrics
                        val_msg = (
                            f"Validation | "
                            f"Loss: {val_metrics['val_loss']:.4f} | "
                            f"CLIP: {val_metrics['val_clip_loss']:.4f} | "
                            f"S2T: {val_metrics['val_s2t_acc']:.3f} | "
                            f"T2S: {val_metrics['val_t2s_acc']:.3f}"
                        )

                        if 'val_mlm_loss' in val_metrics:
                            val_msg += f" | MLM: {val_metrics['val_mlm_loss']:.4f}"

                        # Add alignment health (pos_neg_gap is the key metric)
                        if 'val_alignment/pos_neg_gap' in val_metrics:
                            val_msg += f" | Gap: {val_metrics['val_alignment/pos_neg_gap']:.3f}"

                        # Add overall MLM accuracy
                        if 'val_mlm_accuracy/overall' in val_metrics:
                            val_msg += f" | MLM Acc: {val_metrics['val_mlm_accuracy/overall']:.3f}"

                            # Add per-field MLM accuracy for visibility
                            field_accs = []
                            for field in self.vocab_sizes.keys():  # Only check fields that actually have MLM
                                key = f'val_mlm_accuracy/{field}'
                                if key in val_metrics:
                                    field_accs.append(f"{field}={val_metrics[key]:.3f}")
                            if field_accs:
                                val_msg += f" ({', '.join(field_accs)})"

                        self.logger.info(val_msg)

                        # WandB logging
                        if self.config.use_wandb and WANDB_AVAILABLE:
                            wandb.log(val_metrics, step=self.global_step)

                        # Track best validation loss (but don't save yet - wait for save_interval)
                        if val_metrics['val_loss'] < self.best_val_loss:
                            self.best_val_loss = val_metrics['val_loss']
                            self.logger.info(f"New best validation loss: {self.best_val_loss:.4f}")

                # Comprehensive metrics (retrieval metrics)
                if self.global_step % self.config.training.metrics_interval == 0 and self.global_step > 0:
                    self.logger.info(f"Computing comprehensive retrieval metrics at step {self.global_step}...")

                    # Determine how many batches to sample
                    metrics_sample_batches = getattr(self.config.training, 'metrics_sample_batches', 10)

                    # Compute metrics on training data
                    train_metrics = self.compute_comprehensive_metrics(
                        self.train_loader,
                        split_name='train',
                        max_batches=metrics_sample_batches
                    )

                    # Compute metrics on validation data if available
                    val_comprehensive_metrics = {}
                    if self.val_loader is not None:
                        val_comprehensive_metrics = self.compute_comprehensive_metrics(
                            self.val_loader,
                            split_name='val',
                            max_batches=metrics_sample_batches
                        )

                    # Combine all metrics
                    all_comprehensive_metrics = {**train_metrics, **val_comprehensive_metrics}

                    # Log key metrics to console (simplified)
                    if train_metrics:
                        key_metrics = []
                        for key in ['train/recall@1/average', 'train/recall@5/average',
                                   'val/recall@1/average', 'val/recall@5/average']:
                            if key in all_comprehensive_metrics:
                                short_key = key.split('/')[-1]  # Just get 'average'
                                split = key.split('/')[0]  # Get 'train' or 'val'
                                k_val = key.split('/')[1].replace('recall@', 'R@')  # Get 'R@1'
                                key_metrics.append(f"{split}/{k_val}: {all_comprehensive_metrics[key]:.3f}")

                        # Add label-recall@10 metrics if available
                        for key in ['train/label_recall@10/average', 'val/label_recall@10/average']:
                            if key in all_comprehensive_metrics:
                                split = key.split('/')[0]
                                key_metrics.append(f"{split}/LR@10: {all_comprehensive_metrics[key]:.3f}")

                        if key_metrics:
                            self.logger.info(f"Retrieval Metrics | {' | '.join(key_metrics)}")

                    # WandB logging
                    if self.config.use_wandb and WANDB_AVAILABLE and all_comprehensive_metrics:
                        wandb.log(all_comprehensive_metrics, step=self.global_step)

                # Save checkpoint (including best model if applicable)
                if self.global_step % self.config.training.save_interval == 0:
                    # Save regular checkpoint
                    checkpoint_path = Path(self.config.output_dir) / f'checkpoint_step_{self.global_step}.pt'
                    self.save_checkpoint(str(checkpoint_path))

                    # Also save best model checkpoint if this is the best so far
                    # Check if we have a recent validation run
                    if hasattr(self, 'best_val_loss') and self.best_val_loss != float('inf'):
                        best_path = Path(self.config.output_dir) / 'best_model.pt'
                        self.save_checkpoint(str(best_path), is_best=True)
                        self.logger.info(f"Saved best model (val_loss: {self.best_val_loss:.4f})")

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

        # Also save best model one final time
        if hasattr(self, 'best_val_loss') and self.best_val_loss != float('inf'):
            best_path = Path(self.config.output_dir) / 'best_model.pt'
            self.save_checkpoint(str(best_path), is_best=True)
            self.logger.info(f"Final best model saved (val_loss: {self.best_val_loss:.4f})")

        self.logger.info("Training completed!")

        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

