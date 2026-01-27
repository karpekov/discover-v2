#!/usr/bin/env python3

"""
SCAN Clustering Training Script for src pipeline.
Performs clustering training on pre-trained AlignmentModel embeddings.

Usage:
    # Basic usage with pre-trained model (auto-detects data paths from model config)
    python src/training/train_scan.py \
        --pretrained_model trained_models/milan/milan_fl20_seq_rb0_textclip_projmlp_clipmlm_v1 \
        --num_clusters 20 \
        --output_dir trained_models/milan/milan_fl20_scan_20cl

    # Full control with explicit paths
    python src/training/train_scan.py \
        --pretrained_model trained_models/milan/milan_fl20_seq_rb0_textclip_projmlp_clipmlm_v1 \
        --train_data data/processed/casas/milan/FL_20/train.json \
        --vocab data/processed/casas/milan/FL_20/vocab.json \
        --num_clusters 20 \
        --output_dir trained_models/milan/milan_fl20_scan_20cl

    # With custom wandb settings
    python src/training/train_scan.py \
        --pretrained_model trained_models/milan/milan_fl20_seq_discover_v1_mlm_only \
        --num_clusters 20 \
        --output_dir trained_models/milan/scan_20cl_discover_v1 \
        --wandb_project discover-v2-dv1-scan \
        --max_epochs 50
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Add parent directory to path since we're in training/ subdirectory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.scan_model import SCANClusteringModel
from losses.scan_loss import SCANLoss
from dataio.dataset import SmartHomeDataset
from dataio.scan_dataset import create_scan_data_loader
from utils.device_utils import get_optimal_device, log_device_info


class SCANTrainer:
    """
    Trainer for SCAN clustering using pre-trained AlignmentModel encoders.

    Automatically handles:
    - Loading pre-trained model
    - Extracting data paths from pre-trained model config
    - Creating SCAN dataset with KNN mining
    - Training with SCAN loss
    - Logging to wandb
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_directories()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')

        # Models and optimizers (initialized later)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Data loaders (initialized later)
        self.train_loader = None

    def setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory
        log_dir = Path('logs/text')
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = self.config.get('run_name', 'scan_training')
        log_file = log_dir / f'{run_name}_{timestamp}.log'

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
            wandb_dir = Path('logs/wandb')
            wandb_dir.mkdir(parents=True, exist_ok=True)

            wandb.init(
                project=self.config.get('wandb_project', 'discover-v2-dv1-scan'),
                entity=self.config.get('wandb_entity', None),
                name=self.config.get('wandb_name', run_name),
                config=self.config,
                tags=self.config.get('wandb_tags', ['scan', 'clustering']),
                notes=self.config.get('wandb_notes', 'SCAN clustering training on pre-trained AlignmentModel'),
                group=self.config.get('wandb_group', None),
                job_type='scan_clustering',
                dir=str(wandb_dir),
                reinit=True,
                save_code=True
            )
            self.logger.info(f"W&B logging to: {wandb_dir}")

    def setup_device(self):
        """Setup compute device."""
        self.device = get_optimal_device()
        self.config['device'] = self.device.type
        log_device_info(self.device)
        self.logger.info(f"Using device: {self.device}")

    def setup_directories(self):
        """Create output directories."""
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save hyperparameters
        with open(self.output_dir / 'hyperparameters.json', 'w') as f:
            # Convert config to JSON-serializable format
            serializable_config = {k: v for k, v in self.config.items()
                                   if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
            json.dump(serializable_config, f, indent=2)

        self.logger.info(f"Output directory: {self.output_dir}")

    def setup_model(self):
        """Initialize SCAN model from pre-trained AlignmentModel."""
        self.logger.info("Setting up SCAN clustering model...")
        self.logger.info(f"Loading pre-trained model from: {self.config['pretrained_model_path']}")

        # Create model - it will load the pre-trained AlignmentModel internally
        self.model = SCANClusteringModel(
            pretrained_model_path=self.config['pretrained_model_path'],
            num_clusters=self.config['num_clusters'],
            dropout=self.config.get('dropout', 0.1),
            freeze_encoder=self.config.get('freeze_encoder', False),
            vocab_path=self.config.get('vocab_path'),
            device=self.device
        )

        # Move to device
        self.model = self.model.to(self.device)

        # Get data paths from pre-trained model if not provided
        if self.config.get('train_data_path') is None:
            data_paths = self.model.get_data_paths()
            self.config['train_data_path'] = data_paths['train_data_path']
            self.config['val_data_path'] = data_paths.get('val_data_path')
            if self.config.get('vocab_path') is None:
                self.config['vocab_path'] = data_paths['vocab_path']
            self.logger.info(f"Auto-detected data paths from pre-trained model:")
            self.logger.info(f"  train_data_path: {self.config['train_data_path']}")
            self.logger.info(f"  vocab_path: {self.config['vocab_path']}")

        # Setup optimizer (only for clustering head and optionally unfrozen encoder)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            trainable_params,
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=self.config.get('betas', (0.9, 0.999))
        )

        # Setup loss function
        self.criterion = SCANLoss(
            entropy_weight=self.config.get('entropy_weight', 2.0)
        ).to(self.device)

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params_count:,} trainable")
        self.logger.info(f"Encoder d_model: {self.model.d_model}")
        self.logger.info(f"Number of clusters: {self.config['num_clusters']}")
        self.logger.info(f"Freeze encoder: {self.config.get('freeze_encoder', False)}")

    def extract_embeddings(self, dataset: SmartHomeDataset) -> np.ndarray:
        """
        Extract embeddings from the pre-trained model for KNN mining.
        """
        self.logger.info("Extracting embeddings for KNN mining...")

        # Get categorical fields from the model config
        encoder_config = self.model.alignment_config.encoder
        if encoder_config and 'metadata' in encoder_config:
            categorical_fields = encoder_config['metadata'].get('categorical_fields', ['sensor', 'state', 'room_id'])
        else:
            categorical_fields = ['sensor', 'state', 'room_id']

        # Create a simple collator that handles encoder input format
        def simple_collate_fn(batch):
            """Simple collator for sensor data."""
            batch_size = len(batch)

            # Extract components
            all_categorical = [sample['categorical_features'] for sample in batch]
            all_coordinates = [sample['coordinates'] for sample in batch]
            all_time_deltas = [sample['time_deltas'] for sample in batch]
            all_masks = [sample['mask'] for sample in batch]

            # Stack tensors
            coordinates = torch.stack(all_coordinates).to(self.device)
            time_deltas = torch.stack(all_time_deltas).to(self.device)
            masks = torch.stack(all_masks).to(self.device)

            # Stack categorical features
            categorical_features = {}
            for field in all_categorical[0].keys():
                field_tensors = [sample[field] for sample in all_categorical]
                categorical_features[field] = torch.stack(field_tensors).to(self.device)

            # Format input_data for encoder (same format as AlignmentModel expects)
            input_data = {
                'categorical_features': categorical_features,
                'coordinates': coordinates,
                'time_deltas': time_deltas
            }

            return {
                'input_data': input_data,
                'attention_mask': masks
            }

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.get('embedding_batch_size', 128),
            shuffle=False,
            num_workers=0,
            collate_fn=simple_collate_fn
        )

        embeddings = []
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                batch_embeddings = self.model.get_embeddings(
                    input_data=batch['input_data'],
                    attention_mask=batch['attention_mask']
                )
                embeddings.append(batch_embeddings.cpu().numpy())

        embeddings = np.vstack(embeddings)
        self.logger.info(f"Extracted embeddings shape: {embeddings.shape}")

        return embeddings

    def get_ground_truth_labels(self, dataset: SmartHomeDataset) -> Optional[List[str]]:
        """Extract ground truth labels from dataset for evaluation."""
        labels = []
        for i in range(len(dataset)):
            # Access raw sample data via dataset.data (not .samples)
            sample = dataset.data[i]

            # Try to get activity label from metadata
            if 'metadata' in sample and 'ground_truth_labels' in sample['metadata']:
                label = sample['metadata']['ground_truth_labels'].get('primary_l1', 'Unknown')
            elif 'first_activity' in sample:
                label = sample.get('first_activity', 'Unknown')
            elif 'activity' in sample:
                label = sample.get('activity', 'Unknown')
            else:
                label = 'Unknown'
            labels.append(label)

        return labels if any(l != 'Unknown' for l in labels) else None

    def setup_data(self):
        """Setup data loaders with SCAN dataset creation."""
        self.logger.info("Setting up data loaders...")

        # Setup model first to get data paths
        self.setup_model()

        # Load datasets
        train_dataset = SmartHomeDataset(
            data_path=self.config['train_data_path'],
            vocab_path=self.config['vocab_path'],
            sequence_length=self.config.get('sequence_length', 50),
            max_captions=1,  # We don't need captions for SCAN
            caption_types='long'
        )

        # Get ground truth labels for neighbor accuracy evaluation
        labels = self.get_ground_truth_labels(train_dataset)

        # Extract embeddings for KNN mining
        embeddings = self.extract_embeddings(train_dataset)

        # Create SCAN data loader with KNN mining
        self.train_loader, neighbor_accuracy = create_scan_data_loader(
            base_dataset=train_dataset,
            embeddings=embeddings,
            vocab_sizes=self.model.vocab_sizes,
            device=self.device,
            topk=self.config.get('num_neighbors', 20),
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 0),
            labels=labels
        )

        self.logger.info(f"Training data: {len(train_dataset)} samples")
        if neighbor_accuracy >= 0:
            self.logger.info(f"Neighbor accuracy (same ground truth label): {neighbor_accuracy:.4f}")

            # Log neighbor accuracy to wandb
            if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
                wandb.log({'data/neighbor_accuracy': neighbor_accuracy})

    def compute_cluster_utilization(self) -> Dict[str, float]:
        """Compute cluster utilization metrics on training data."""
        self.model.eval()

        cluster_predictions = []
        num_samples_evaluated = 0
        max_samples_for_eval = 2000  # Limit samples for efficiency

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_loader):
                if num_samples_evaluated >= max_samples_for_eval:
                    break

                # Forward pass on anchors only
                anchor_logits = self.model(
                    input_data=batch['anchor']['input_data'],
                    attention_mask=batch['anchor']['mask']
                )

                # Get cluster predictions
                batch_predictions = torch.argmax(anchor_logits, dim=1)
                cluster_predictions.extend(batch_predictions.cpu().numpy())
                num_samples_evaluated += len(batch_predictions)

        if cluster_predictions:
            cluster_predictions = np.array(cluster_predictions)
            unique_clusters = np.unique(cluster_predictions)

            from collections import Counter
            cluster_counts = Counter(cluster_predictions)

            # Find most and least populated clusters
            most_common = cluster_counts.most_common(1)[0]
            least_common = min(cluster_counts.items(), key=lambda x: x[1])

            metrics = {
                'active_clusters': len(unique_clusters),
                'cluster_utilization': len(unique_clusters) / self.config['num_clusters'],
                'samples_evaluated': len(cluster_predictions),
                'total_clusters': self.config['num_clusters'],
                'max_cluster_size': most_common[1],
                'min_cluster_size': least_common[1],
                'cluster_size_ratio': most_common[1] / least_common[1] if least_common[1] > 0 else float('inf')
            }

            return metrics
        else:
            return {
                'active_clusters': 0,
                'cluster_utilization': 0.0,
                'samples_evaluated': 0,
                'total_clusters': self.config['num_clusters'],
                'max_cluster_size': 0,
                'min_cluster_size': 0,
                'cluster_size_ratio': 0.0
            }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train_mode()

        epoch_metrics = {
            'total_loss': 0.0,
            'consistency_loss': 0.0,
            'entropy_loss': 0.0,
            'num_batches': 0
        }

        for batch_idx, batch in enumerate(self.train_loader):
            # Forward pass on anchors and neighbors
            # The batch format from SCANCollator has nested structure
            anchor_logits = self.model(
                input_data=batch['anchor']['input_data'],
                attention_mask=batch['anchor']['mask']
            )

            neighbor_logits = self.model(
                input_data=batch['neighbor']['input_data'],
                attention_mask=batch['neighbor']['mask']
            )

            # Compute SCAN loss
            total_loss, consistency_loss, entropy_loss = self.criterion(
                anchor_logits, neighbor_logits
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping if specified
            if self.config.get('grad_clip_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip_norm']
                )

            self.optimizer.step()

            # Update metrics
            epoch_metrics['total_loss'] += total_loss.item()
            epoch_metrics['consistency_loss'] += consistency_loss.item()
            epoch_metrics['entropy_loss'] += entropy_loss.item()
            epoch_metrics['num_batches'] += 1

            self.global_step += 1

            # Logging
            if batch_idx % self.config.get('log_interval', 50) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch {self.current_epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Total Loss: {total_loss.item():.4f} | "
                    f"Consistency: {consistency_loss.item():.4f} | "
                    f"Entropy: {entropy_loss.item():.4f} | "
                    f"LR: {current_lr:.2e}"
                )

                # Log to wandb
                if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
                    log_dict = {
                        'train/total_loss': total_loss.item(),
                        'train/consistency_loss': consistency_loss.item(),
                        'train/entropy_loss': entropy_loss.item(),
                        'train/learning_rate': current_lr,
                        'global_step': self.global_step
                    }

                    # Add cluster utilization tracking every few log intervals
                    if batch_idx % (self.config.get('log_interval', 50) * 5) == 0:
                        step_cluster_metrics = self.compute_cluster_utilization()
                        log_dict.update({
                            'train_step/active_clusters': step_cluster_metrics['active_clusters'],
                            'train_step/cluster_utilization': step_cluster_metrics['cluster_utilization'],
                        })

                    wandb.log(log_dict)

            # Save checkpoint
            if self.global_step % self.config.get('save_interval', 2000) == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')

        # Compute average metrics
        for key in ['total_loss', 'consistency_loss', 'entropy_loss']:
            if epoch_metrics['num_batches'] > 0:
                epoch_metrics[key] /= epoch_metrics['num_batches']

        return epoch_metrics

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / filename

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'config': self.config,
            'best_loss': self.best_loss,
            'num_clusters': self.config['num_clusters'],
            'd_model': self.model.d_model,
            'vocab_sizes': self.model.vocab_sizes,
            # Store pre-trained model path for reference
            'pretrained_model_path': self.config['pretrained_model_path'],
            # Store alignment config for reference
            'alignment_config_dict': {
                'train_data_path': self.model.alignment_config.train_data_path,
                'vocab_path': self.model.alignment_config.vocab_path,
                'encoder': self.model.alignment_config.encoder
            }
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def train(self):
        """Main training loop."""
        self.logger.info("Starting SCAN clustering training...")

        # Setup data and model
        self.setup_data()

        # Training loop
        for epoch in range(self.config['max_epochs']):
            self.current_epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{self.config['max_epochs']}")

            # Train epoch
            epoch_metrics = self.train_epoch()

            # Compute cluster utilization metrics
            cluster_metrics = self.compute_cluster_utilization()

            # Log epoch metrics
            avg_total_loss = epoch_metrics['total_loss']
            self.logger.info(
                f"Epoch {epoch + 1} completed | "
                f"Avg Total Loss: {avg_total_loss:.4f} | "
                f"Avg Consistency: {epoch_metrics['consistency_loss']:.4f} | "
                f"Avg Entropy: {epoch_metrics['entropy_loss']:.4f} | "
                f"Active Clusters: {cluster_metrics['active_clusters']}/{cluster_metrics['total_clusters']} "
                f"({cluster_metrics['cluster_utilization']:.1%})"
            )

            # Log to wandb
            if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
                wandb.log({
                    'epoch/total_loss': avg_total_loss,
                    'epoch/consistency_loss': epoch_metrics['consistency_loss'],
                    'epoch/entropy_loss': epoch_metrics['entropy_loss'],
                    'epoch': epoch + 1,

                    # Cluster utilization metrics
                    'clusters/active_clusters': cluster_metrics['active_clusters'],
                    'clusters/cluster_utilization': cluster_metrics['cluster_utilization'],
                    'clusters/total_clusters': cluster_metrics['total_clusters'],
                    'clusters/max_cluster_size': cluster_metrics['max_cluster_size'],
                    'clusters/min_cluster_size': cluster_metrics['min_cluster_size'],
                    'clusters/cluster_size_ratio': cluster_metrics['cluster_size_ratio'],
                    'clusters/samples_evaluated': cluster_metrics['samples_evaluated']
                })

            # Save best model
            if avg_total_loss < self.best_loss:
                self.best_loss = avg_total_loss
                self.save_checkpoint('best_model.pt')
                self.logger.info(f"New best model saved with loss: {self.best_loss:.4f}")

            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch + 1}.pt')

        # Save final model
        self.save_checkpoint('final_model.pt')

        # Save config.yaml for reference
        self._save_config_yaml()

        # Compute final cluster utilization
        final_cluster_metrics = self.compute_cluster_utilization()

        self.logger.info(
            f"Training completed! Final cluster utilization: "
            f"{final_cluster_metrics['active_clusters']}/{final_cluster_metrics['total_clusters']} "
            f"({final_cluster_metrics['cluster_utilization']:.1%})"
        )

        # Finish wandb run
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.log({
                'training/final_step': self.global_step,
                'training/best_loss': self.best_loss,
                'training/status': 'completed',

                # Final cluster metrics
                'training/final_active_clusters': final_cluster_metrics['active_clusters'],
                'training/final_cluster_utilization': final_cluster_metrics['cluster_utilization'],
                'training/final_max_cluster_size': final_cluster_metrics['max_cluster_size'],
                'training/final_min_cluster_size': final_cluster_metrics['min_cluster_size'],
                'training/final_cluster_size_ratio': final_cluster_metrics['cluster_size_ratio']
            })
            wandb.finish()

        self.logger.info("SCAN clustering training completed!")

    def _save_config_yaml(self):
        """Save configuration as YAML for reference."""
        import yaml

        config_to_save = {
            'pretrained_model_path': self.config['pretrained_model_path'],
            'train_data_path': self.config['train_data_path'],
            'vocab_path': self.config['vocab_path'],
            'num_clusters': self.config['num_clusters'],
            'learning_rate': self.config['learning_rate'],
            'batch_size': self.config['batch_size'],
            'max_epochs': self.config['max_epochs'],
            'freeze_encoder': self.config.get('freeze_encoder', False),
            'entropy_weight': self.config.get('entropy_weight', 2.0),
            'num_neighbors': self.config.get('num_neighbors', 20),
            'wandb_project': self.config.get('wandb_project', 'discover-v2-dv1-scan')
        }

        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False)


def get_default_config() -> Dict[str, Any]:
    """Get default SCAN training configuration."""
    return {
        # Model settings
        'num_clusters': 20,
        'dropout': 0.1,
        'freeze_encoder': False,

        # Training settings
        'batch_size': 64,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999),
        'max_epochs': 40,
        'grad_clip_norm': 1.0,

        # SCAN settings
        'entropy_weight': 2.0,
        'num_neighbors': 20,

        # Data settings
        'sequence_length': 50,
        'embedding_batch_size': 128,
        'num_workers': 0,

        # Logging and saving
        'log_interval': 50,
        'save_interval': 2000,

        # Wandb settings
        'use_wandb': True,
        'wandb_project': 'discover-v2-dv1-scan',
        'wandb_tags': ['scan', 'clustering'],
        'wandb_notes': 'SCAN clustering training on pre-trained AlignmentModel'
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train SCAN clustering model on pre-trained AlignmentModel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (auto-detects data paths from pre-trained model)
    python src/training/train_scan.py \\
        --pretrained_model trained_models/milan/milan_fl20_seq_rb0_textclip_projmlp_clipmlm_v1 \\
        --num_clusters 20 \\
        --output_dir trained_models/milan/scan_20cl_v1

    # With explicit data paths
    python src/training/train_scan.py \\
        --pretrained_model trained_models/milan/model_dir \\
        --train_data data/processed/casas/milan/FL_20/train.json \\
        --vocab data/processed/casas/milan/FL_20/vocab.json \\
        --num_clusters 20 \\
        --output_dir trained_models/milan/scan_20cl_v1

    # Frozen encoder (only train clustering head)
    python src/training/train_scan.py \\
        --pretrained_model trained_models/milan/model_dir \\
        --num_clusters 20 \\
        --freeze_encoder \\
        --output_dir trained_models/milan/scan_frozen_20cl
        """
    )

    # Required arguments
    parser.add_argument('--pretrained_model', type=str, required=True,
                       help='Path to pre-trained AlignmentModel directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for trained model')

    # Optional data paths (auto-detected from pre-trained model if not provided)
    parser.add_argument('--train_data', type=str, default=None,
                       help='Path to training data (auto-detected if not provided)')
    parser.add_argument('--vocab', type=str, default=None,
                       help='Path to vocabulary file (auto-detected if not provided)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON config file (overrides other arguments)')

    # Model settings
    parser.add_argument('--num_clusters', type=int, default=20,
                       help='Number of clusters')
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze encoder weights (only train clustering head)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate for clustering head')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=40,
                       help='Maximum training epochs')
    parser.add_argument('--num_neighbors', type=int, default=20,
                       help='Number of nearest neighbors for SCAN')
    parser.add_argument('--entropy_weight', type=float, default=2.0,
                       help='Entropy weight in SCAN loss')

    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='discover-v2-dv1-scan',
                       help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='W&B run name')
    parser.add_argument('--wandb_tags', type=str, nargs='+',
                       default=['scan', 'clustering'], help='W&B tags')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable W&B logging')

    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()

    # Override with command line arguments
    config.update({
        'pretrained_model_path': args.pretrained_model,
        'output_dir': args.output_dir,
        'use_wandb': not args.no_wandb,
    })

    # Optional paths
    if args.train_data:
        config['train_data_path'] = args.train_data
    if args.vocab:
        config['vocab_path'] = args.vocab

    # Model settings
    config['num_clusters'] = args.num_clusters
    config['freeze_encoder'] = args.freeze_encoder
    config['dropout'] = args.dropout

    # Training settings
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['max_epochs'] = args.max_epochs
    config['num_neighbors'] = args.num_neighbors
    config['entropy_weight'] = args.entropy_weight

    # Wandb settings
    config['wandb_project'] = args.wandb_project
    if args.wandb_name:
        config['wandb_name'] = args.wandb_name
    config['wandb_tags'] = args.wandb_tags

    # Generate run name if not provided
    if not config.get('wandb_name'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = Path(args.pretrained_model).name
        config['wandb_name'] = f"scan_{args.num_clusters}cl_{model_name}_{timestamp}"
        config['run_name'] = config['wandb_name']
    else:
        config['run_name'] = config['wandb_name']

    # Create trainer and start training
    trainer = SCANTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
