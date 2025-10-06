#!/usr/bin/env python3

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Add parent directory to path since we're in training/ subdirectory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.classification_head import SensorClassificationModel, load_pretrained_sensor_encoder, create_l2_label_mapping
from dataio.classification_dataset import SmartHomeClassificationDataset, create_classification_data_loader
from utils.device_utils import get_optimal_device, get_device_config, log_device_info, optimize_for_device


class ClassificationTrainer:
    """
    Trainer for activity classification using pre-trained sensor encoder.
    """

    def __init__(self, config: Dict[str, Any]):
        # Setup device
        if isinstance(config.get('device'), str) and config['device'] not in ['auto', None]:
            device = torch.device(config['device'])
        else:
            device = get_optimal_device()

        self.config = optimize_for_device(config, device)
        self.device = device

        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Log device info
        log_device_info(self.device)

        # Initialize datasets
        self._setup_datasets()

        # Initialize model
        self._setup_model()

        # Initialize optimizer and scheduler
        self._setup_optimizer()

        # Initialize training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.best_val_f1 = 0.0

        # Setup wandb if enabled
        self._setup_wandb()

        # Training metrics tracking
        self.train_metrics = {
            'losses': [],
            'accuracies': [],
            'learning_rates': []
        }

        self.val_metrics = {
            'losses': [],
            'accuracies': [],
            'f1_scores': [],
            'precisions': [],
            'recalls': []
        }

        logger.info("Classification trainer initialized successfully")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.config['output_dir'], 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        global logger
        logger = logging.getLogger(__name__)

    def _setup_datasets(self):
        """Initialize training and validation datasets."""
        logger.info("Setting up datasets...")

        # Load vocabulary for model initialization
        with open(self.config['vocab_path'], 'r') as f:
            vocab = json.load(f)

        # Get vocabulary sizes
        categorical_fields = ['sensor_id', 'room_id', 'event_type', 'sensor_type', 'tod_bucket', 'delta_t_bucket']
        if 'floor_id' in vocab:
            categorical_fields.append('floor_id')
        if 'dow' in vocab:
            categorical_fields.append('dow')

        self.vocab_sizes = {field: len(vocab[field]) for field in categorical_fields}

        # Create L2 label mapping
        self.l2_label_to_idx, self.l2_labels = create_l2_label_mapping()
        self.num_classes = len(self.l2_label_to_idx)

        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Classes: {self.l2_labels}")

        # Create training dataset
        self.train_dataset = SmartHomeClassificationDataset(
            data_path=self.config['train_data_path'],
            vocab_path=self.config['vocab_path'],
            sequence_length=self.config['sequence_length'],
            max_captions=self.config.get('max_captions', 4),
            normalize_coords=True,
            caption_types=self.config.get('caption_types', 'long'),
            exclude_no_activity=self.config.get('exclude_no_activity', True),
            exclude_other=self.config.get('exclude_other', False),
            l2_label_mapping=self.l2_label_to_idx
        )

        # Create validation dataset
        self.val_dataset = SmartHomeClassificationDataset(
            data_path=self.config['val_data_path'],
            vocab_path=self.config['vocab_path'],
            sequence_length=self.config['sequence_length'],
            max_captions=self.config.get('max_captions', 4),
            normalize_coords=True,
            caption_types=self.config.get('caption_types', 'long'),
            exclude_no_activity=self.config.get('exclude_no_activity', True),
            exclude_other=self.config.get('exclude_other', False),
            l2_label_mapping=self.l2_label_to_idx
        )

        # Print dataset info
        self.train_dataset.print_dataset_info()
        logger.info(f"Validation dataset size: {len(self.val_dataset)}")

        # Create data loaders
        self.train_loader = create_classification_data_loader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=self.config.get('pin_memory', False)
        )

        self.val_loader = create_classification_data_loader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 0),
            pin_memory=self.config.get('pin_memory', False)
        )

        logger.info(f"Training batches per epoch: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")

    def _setup_model(self):
        """Initialize the classification model."""
        logger.info("Setting up model...")

        # Load pre-trained sensor encoder
        pretrained_encoder = load_pretrained_sensor_encoder(
            checkpoint_path=self.config['pretrained_model_path'],
            vocab_sizes=self.vocab_sizes,
            device=self.device
        )

        logger.info(f"Loaded pre-trained encoder from: {self.config['pretrained_model_path']}")

        # Create classification model
        self.model = SensorClassificationModel(
            pretrained_sensor_encoder=pretrained_encoder,
            num_classes=self.num_classes,
            hidden_dim=self.config.get('classifier_hidden_dim', None),
            dropout=self.config.get('classifier_dropout', 0.1),
            freeze_backbone=self.config.get('freeze_backbone', True)
        )

        self.model.to(self.device)

        # Setup loss function
        if self.config.get('use_class_weights', True):
            class_weights = self.train_dataset.get_class_weights()
            class_weights = class_weights.to(self.device)
            logger.info(f"Using class weights: {class_weights}")
        else:
            class_weights = None

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    def _setup_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config['learning_rate'],
            betas=self.config.get('betas', [0.9, 0.999]),
            weight_decay=self.config.get('weight_decay', 0.01)
        )

        # Setup learning rate scheduler
        scheduler_type = self.config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['max_epochs'],
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=self.config.get('scheduler_patience', 3),
                verbose=True
            )
        else:
            self.scheduler = None

        logger.info(f"Optimizer: AdamW with lr={self.config['learning_rate']}")
        logger.info(f"Scheduler: {scheduler_type}")

    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        # Check both flat and nested wandb config formats
        wandb_config = self.config.get('wandb', {})
        use_wandb = self.config.get('use_wandb', False) or wandb_config.get('enabled', False)

        if not WANDB_AVAILABLE or not use_wandb:
            logger.info("W&B logging disabled")
            return

        # Get wandb settings from either flat or nested config
        project = self.config.get('wandb_project') or wandb_config.get('project', 'sensor-classification')
        name = self.config.get('wandb_name') or wandb_config.get('run_name', 'classification-experiment')
        tags = self.config.get('wandb_tags') or wandb_config.get('tags', ['classification', 'sensor-encoder'])
        notes = self.config.get('wandb_notes') or wandb_config.get('notes', 'Activity classification fine-tuning')
        entity = wandb_config.get('entity', None)

        wandb_init_kwargs = {
            'project': project,
            'name': name,
            'config': self.config,
            'tags': tags,
            'notes': notes
        }

        if entity:
            wandb_init_kwargs['entity'] = entity

        wandb.init(**wandb_init_kwargs)

        # Log model architecture
        wandb.watch(self.model, log_freq=100)

        logger.info("W&B logging initialized")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            categorical_features = {k: v.to(self.device) for k, v in batch['categorical_features'].items()}
            coordinates = batch['coordinates'].to(self.device)
            time_deltas = batch['time_deltas'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['l2_label_indices'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.config.get('use_amp', False):
                with autocast():
                    logits = self.model(categorical_features, coordinates, time_deltas, mask)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(categorical_features, coordinates, time_deltas, mask)
                loss = self.criterion(logits, labels)

                loss.backward()

                # Gradient clipping
                if self.config.get('grad_clip_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip_norm']
                    )

                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            self.global_step += 1

            # Log training progress
            if batch_idx % self.config.get('log_interval', 50) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                accuracy = total_correct / total_samples * 100

                logger.info(
                    f"Epoch {self.current_epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} | Acc: {accuracy:.2f}% | LR: {current_lr:.2e}"
                )

                # Log to wandb
                use_wandb = self.config.get('use_wandb', False) or self.config.get('wandb', {}).get('enabled', False)
                if WANDB_AVAILABLE and use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/accuracy': accuracy,
                        'train/learning_rate': current_lr,
                        'global_step': self.global_step
                    })

            # Compute detailed metrics every n steps
            if self.global_step % self.config.get('metrics_interval', 500) == 0 and self.global_step > 0:
                logger.info(f"Computing detailed metrics at step {self.global_step}...")
                metrics = self.compute_detailed_metrics(subset_size=self.config.get('metrics_sample_size', 1000))

                logger.info(f"Step {self.global_step} Metrics:")
                logger.info(f"  Train Acc: {metrics['train_accuracy']:.2f}%")
                logger.info(f"  Train F1 (macro): {metrics['train_f1_macro']:.2f}%")
                logger.info(f"  Train F1 (weighted): {metrics['train_f1_weighted']:.2f}%")
                logger.info(f"  Val Acc: {metrics['val_accuracy']:.2f}%")
                logger.info(f"  Val F1 (macro): {metrics['val_f1_macro']:.2f}%")
                logger.info(f"  Val F1 (weighted): {metrics['val_f1_weighted']:.2f}%")

                # Log to wandb
                use_wandb = self.config.get('use_wandb', False) or self.config.get('wandb', {}).get('enabled', False)
                if WANDB_AVAILABLE and use_wandb:
                    wandb.log({
                        'metrics/train_accuracy': metrics['train_accuracy'],
                        'metrics/train_f1_macro': metrics['train_f1_macro'],
                        'metrics/train_f1_weighted': metrics['train_f1_weighted'],
                        'metrics/train_precision': metrics['train_precision'],
                        'metrics/train_recall': metrics['train_recall'],
                        'metrics/val_accuracy': metrics['val_accuracy'],
                        'metrics/val_f1_macro': metrics['val_f1_macro'],
                        'metrics/val_f1_weighted': metrics['val_f1_weighted'],
                        'metrics/val_precision': metrics['val_precision'],
                        'metrics/val_recall': metrics['val_recall'],
                        'global_step': self.global_step
                    })

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples * 100

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    def compute_detailed_metrics(self, subset_size: int = 1000) -> Dict[str, float]:
        """
        Compute detailed metrics on train and validation subsets.

        Args:
            subset_size: Number of samples to use for metric computation

        Returns:
            Dictionary with detailed metrics
        """
        self.model.eval()

        def evaluate_subset(dataloader, max_samples):
            all_predictions = []
            all_labels = []
            samples_processed = 0

            with torch.no_grad():
                for batch in dataloader:
                    if samples_processed >= max_samples:
                        break

                    # Move batch to device
                    categorical_features = {k: v.to(self.device) for k, v in batch['categorical_features'].items()}
                    coordinates = batch['coordinates'].to(self.device)
                    time_deltas = batch['time_deltas'].to(self.device)
                    mask = batch['mask'].to(self.device)
                    labels = batch['l2_label_indices'].to(self.device)

                    # Forward pass
                    logits = self.model(categorical_features, coordinates, time_deltas, mask)
                    predictions = torch.argmax(logits, dim=-1)

                    # Collect predictions and labels
                    batch_size = min(labels.size(0), max_samples - samples_processed)
                    all_predictions.extend(predictions[:batch_size].cpu().numpy())
                    all_labels.extend(labels[:batch_size].cpu().numpy())
                    samples_processed += batch_size

            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_predictions) * 100
            precision, recall, f1_weighted, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted', zero_division=0
            )
            _, _, f1_macro, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='macro', zero_division=0
            )

            return {
                'accuracy': accuracy,
                'precision': precision * 100,
                'recall': recall * 100,
                'f1_weighted': f1_weighted * 100,
                'f1_macro': f1_macro * 100,
                'predictions': all_predictions,
                'labels': all_labels
            }

        # Evaluate on training subset
        train_metrics = evaluate_subset(self.train_loader, subset_size)

        # Evaluate on validation subset
        val_metrics = evaluate_subset(self.val_loader, subset_size)

        # Return combined metrics
        return {
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1_weighted': train_metrics['f1_weighted'],
            'train_f1_macro': train_metrics['f1_macro'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1_weighted': val_metrics['f1_weighted'],
            'val_f1_macro': val_metrics['f1_macro']
        }

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                categorical_features = {k: v.to(self.device) for k, v in batch['categorical_features'].items()}
                coordinates = batch['coordinates'].to(self.device)
                time_deltas = batch['time_deltas'].to(self.device)
                mask = batch['mask'].to(self.device)
                labels = batch['l2_label_indices'].to(self.device)

                # Forward pass
                logits = self.model(categorical_features, coordinates, time_deltas, mask)
                loss = self.criterion(logits, labels)

                # Collect predictions and labels
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                total_loss += loss.item()

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions) * 100

        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )

        # Calculate macro-averaged F1
        _, _, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro', zero_division=0
        )

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_weighted': f1 * 100,
            'f1_macro': macro_f1 * 100,
            'predictions': all_predictions,
            'labels': all_labels
        }

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_f1': self.best_val_f1,
            'config': self.config,
            'l2_label_to_idx': self.l2_label_to_idx,
            'l2_labels': self.l2_labels
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['output_dir'], f'checkpoint_epoch_{self.current_epoch}.pt')
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.config['output_dir'], 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with accuracy: {self.best_val_accuracy:.2f}%")

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def print_classification_report(self, predictions: List[int], labels: List[int]):
        """Print detailed classification report."""
        # Get unique labels in the actual data
        unique_labels = sorted(list(set(labels + predictions)))
        target_names = [self.l2_labels[i] for i in unique_labels]

        report = classification_report(
            labels, predictions,
            labels=unique_labels,
            target_names=target_names,
            zero_division=0
        )
        logger.info("Classification Report:")
        logger.info("\n" + report)

        # Confusion matrix
        cm = confusion_matrix(labels, predictions, labels=unique_labels)
        logger.info("Confusion Matrix:")
        logger.info(f"Classes: {target_names}")
        logger.info(f"\n{cm}")

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")

        # Initialize AMP scaler if using mixed precision
        if self.config.get('use_amp', False):
            self.scaler = GradScaler()

        start_time = time.time()

        for epoch in range(self.config['max_epochs']):
            self.current_epoch = epoch

            logger.info(f"\nEpoch {epoch + 1}/{self.config['max_epochs']}")
            logger.info("-" * 50)

            # Training
            train_metrics = self.train_epoch()
            self.train_metrics['losses'].append(train_metrics['loss'])
            self.train_metrics['accuracies'].append(train_metrics['accuracy'])
            self.train_metrics['learning_rates'].append(train_metrics['learning_rate'])

            # Validation
            if (epoch + 1) % self.config.get('val_interval', 1) == 0:
                val_metrics = self.validate()
                self.val_metrics['losses'].append(val_metrics['loss'])
                self.val_metrics['accuracies'].append(val_metrics['accuracy'])
                self.val_metrics['f1_scores'].append(val_metrics['f1_weighted'])
                self.val_metrics['precisions'].append(val_metrics['precision'])
                self.val_metrics['recalls'].append(val_metrics['recall'])

                logger.info(f"Validation Results:")
                logger.info(f"  Loss: {val_metrics['loss']:.4f}")
                logger.info(f"  Accuracy: {val_metrics['accuracy']:.2f}%")
                logger.info(f"  F1 (weighted): {val_metrics['f1_weighted']:.2f}%")
                logger.info(f"  F1 (macro): {val_metrics['f1_macro']:.2f}%")

                # Check if this is the best model
                is_best = val_metrics['accuracy'] > self.best_val_accuracy
                if is_best:
                    self.best_val_accuracy = val_metrics['accuracy']
                    self.best_val_f1 = val_metrics['f1_weighted']

                # Save checkpoint
                if (epoch + 1) % self.config.get('save_interval', 5) == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)

                # Print detailed classification report for best model
                if is_best:
                    self.print_classification_report(
                        val_metrics['predictions'],
                        val_metrics['labels']
                    )

                # Log to wandb
                use_wandb = self.config.get('use_wandb', False) or self.config.get('wandb', {}).get('enabled', False)
                if WANDB_AVAILABLE and use_wandb:
                    wandb.log({
                        'val/loss': val_metrics['loss'],
                        'val/accuracy': val_metrics['accuracy'],
                        'val/f1_weighted': val_metrics['f1_weighted'],
                        'val/f1_macro': val_metrics['f1_macro'],
                        'val/precision': val_metrics['precision'],
                        'val/recall': val_metrics['recall'],
                        'epoch': epoch
                    })

            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()

            logger.info(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")

        # Final save
        self.save_checkpoint()

        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time/3600:.2f} hours")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
        logger.info(f"Best validation F1: {self.best_val_f1:.2f}%")

        # Save training summary
        summary = {
            'config': self.config,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_f1': self.best_val_f1,
            'training_time_hours': total_time / 3600,
            'total_epochs': self.config['max_epochs'],
            'total_steps': self.global_step,
            'l2_labels': self.l2_labels,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }

        summary_path = os.path.join(self.config['output_dir'], 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training summary saved: {summary_path}")

        # Final evaluation on full test set
        logger.info("\n" + "="*60)
        logger.info("FINAL TEST EVALUATION")
        logger.info("="*60)

        final_test_metrics = self.validate()
        self.print_classification_report(
            final_test_metrics['predictions'],
            final_test_metrics['labels']
        )

        logger.info(f"\nFINAL TEST METRICS:")
        logger.info(f"  Accuracy: {final_test_metrics['accuracy']:.2f}%")
        logger.info(f"  F1 (weighted): {final_test_metrics['f1_weighted']:.2f}%")
        logger.info(f"  F1 (macro): {final_test_metrics['f1_macro']:.2f}%")
        logger.info(f"  Precision: {final_test_metrics['precision']:.2f}%")
        logger.info(f"  Recall: {final_test_metrics['recall']:.2f}%")

        # Log final metrics to wandb
        use_wandb = self.config.get('use_wandb', False) or self.config.get('wandb', {}).get('enabled', False)
        if WANDB_AVAILABLE and use_wandb:
            wandb.log({
                'final_test/accuracy': final_test_metrics['accuracy'],
                'final_test/f1_weighted': final_test_metrics['f1_weighted'],
                'final_test/f1_macro': final_test_metrics['f1_macro'],
                'final_test/precision': final_test_metrics['precision'],
                'final_test/recall': final_test_metrics['recall']
            })

        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train activity classification model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--pretrained_model', type=str, help='Override pretrained model path')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Apply command line overrides
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.pretrained_model:
        config['pretrained_model_path'] = args.pretrained_model

    # Create trainer and start training
    trainer = ClassificationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
