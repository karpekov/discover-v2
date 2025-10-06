#!/usr/bin/env python3

"""
SCAN Clustering Training Script for src-v2 pipeline.
Performs clustering training on pre-trained sensor encoder embeddings.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

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

from models.scan_model import SCANClusteringModel
from losses.scan_loss import SCANLoss
from dataio.dataset import SmartHomeDataset
from dataio.scan_dataset import create_scan_data_loader
from utils.device_utils import get_optimal_device, log_device_info
from utils.training_metrics import TrainingMetrics


class SCANTrainer:
    """
    Trainer for SCAN clustering using pre-trained sensor encoders.
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
        self.val_loader = None

        # Metrics tracking (initialized later after data loading)
        self.metrics = None

    def setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory
        log_dir = Path('src-v2/logs')
        log_dir.mkdir(exist_ok=True)

        # Generate log filename with timestamp
        from datetime import datetime
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
            wandb_dir = log_dir / 'wandb'
            wandb_dir.mkdir(exist_ok=True)

            wandb.init(
                project=self.config.get('wandb_project', 'discover-scan'),
                entity=self.config.get('wandb_entity', None),
                name=self.config.get('wandb_name', run_name),
                config=self.config,
                tags=self.config.get('wandb_tags', ['scan', 'clustering']),
                notes=self.config.get('wandb_notes', 'SCAN clustering training'),
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
            json.dump(self.config, f, indent=2)

        self.logger.info(f"Output directory: {self.output_dir}")

    def setup_model(self, vocab_sizes: Dict[str, int]):
        """Initialize SCAN model."""
        self.logger.info("Setting up SCAN clustering model...")

        # Create model
        self.model = SCANClusteringModel(
            pretrained_model_path=self.config['pretrained_model_path'],
            num_clusters=self.config['num_clusters'],
            dropout=self.config.get('dropout', 0.1),
            freeze_encoder=self.config.get('freeze_encoder', False)
        )

        # Update vocabulary sizes
        self.model.update_vocab_sizes(vocab_sizes)

        # Move to device
        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=self.config.get('betas', [0.9, 0.999])
        )

        # Setup loss function
        self.criterion = SCANLoss(
            entropy_weight=self.config.get('entropy_weight', 2.0)
        ).to(self.device)

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    def extract_embeddings(self, dataset: SmartHomeDataset) -> np.ndarray:
        """
        Extract embeddings from the pre-trained model for KNN mining.
        """
        self.logger.info("Extracting embeddings for KNN mining...")

        # Create a simple collator that only handles sensor data
        def simple_collate_fn(batch):
            """Simple collator for sensor data only."""
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

            return {
                'categorical_features': categorical_features,
                'coordinates': coordinates,
                'time_deltas': time_deltas,
                'mask': masks
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
                    categorical_features=batch['categorical_features'],
                    coordinates=batch['coordinates'],
                    time_deltas=batch['time_deltas'],
                    mask=batch['mask']
                )
                embeddings.append(batch_embeddings.cpu().numpy())

        embeddings = np.vstack(embeddings)
        self.logger.info(f"Extracted embeddings shape: {embeddings.shape}")

        return embeddings

    def setup_data(self):
        """Setup data loaders."""
        self.logger.info("Setting up data loaders...")

        # Load datasets
        train_dataset = SmartHomeDataset(
            data_path=self.config['train_data_path'],
            vocab_path=self.config['vocab_path'],
            sequence_length=self.config.get('sequence_length', 50),
            max_captions=1,  # We don't need captions for SCAN
            caption_types='long'
        )

        # Setup model with vocabulary sizes first
        self.setup_model(train_dataset.vocab_sizes)

        # Extract embeddings for KNN mining using the model
        embeddings = self.extract_embeddings(train_dataset)

        # Create SCAN data loader
        self.train_loader, neighbor_accuracy = create_scan_data_loader(
            base_dataset=train_dataset,
            embeddings=embeddings,
            vocab_sizes=train_dataset.vocab_sizes,
            device=self.device,
            topk=self.config.get('num_neighbors', 20),
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 0),
            labels=None  # Could add ground truth labels here for evaluation
        )

        self.logger.info(f"Training data: {len(train_dataset)} samples")
        self.logger.info(f"Neighbor accuracy: {neighbor_accuracy:.4f}")

        # Log neighbor accuracy to wandb
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE:
            wandb.log({'data/neighbor_accuracy': neighbor_accuracy})

        # Initialize metrics tracking
        self.metrics = TrainingMetrics(train_dataset.vocab_sizes)

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

                # Forward pass on anchors only (more efficient)
                anchor_logits = self.model(
                    categorical_features=batch['anchor']['categorical_features'],
                    coordinates=batch['anchor']['coordinates'],
                    time_deltas=batch['anchor']['time_deltas'],
                    mask=batch['anchor']['mask']
                )

                # Get cluster predictions
                batch_predictions = torch.argmax(anchor_logits, dim=1)
                cluster_predictions.extend(batch_predictions.cpu().numpy())
                num_samples_evaluated += len(batch_predictions)

        if cluster_predictions:
            import numpy as np
            cluster_predictions = np.array(cluster_predictions)
            unique_clusters = np.unique(cluster_predictions)

            metrics = {
                'active_clusters': len(unique_clusters),
                'cluster_utilization': len(unique_clusters) / self.config['num_clusters'],
                'samples_evaluated': len(cluster_predictions),
                'total_clusters': self.config['num_clusters']
            }

            # Log cluster distribution for debugging
            from collections import Counter
            cluster_counts = Counter(cluster_predictions)

            # Find most and least populated clusters
            most_common = cluster_counts.most_common(1)[0]
            least_common = min(cluster_counts.items(), key=lambda x: x[1])

            metrics.update({
                'max_cluster_size': most_common[1],
                'min_cluster_size': least_common[1],
                'cluster_size_ratio': most_common[1] / least_common[1] if least_common[1] > 0 else float('inf')
            })

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
            anchor_logits = self.model(
                categorical_features=batch['anchor']['categorical_features'],
                coordinates=batch['anchor']['coordinates'],
                time_deltas=batch['anchor']['time_deltas'],
                mask=batch['anchor']['mask']
            )

            neighbor_logits = self.model(
                categorical_features=batch['neighbor']['categorical_features'],
                coordinates=batch['neighbor']['coordinates'],
                time_deltas=batch['neighbor']['time_deltas'],
                mask=batch['neighbor']['mask']
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

                    # Add cluster utilization tracking every few log intervals for efficiency
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
            'best_loss': self.best_loss
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
        'betas': [0.9, 0.999],
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
        'wandb_project': 'discover-scan',
        'wandb_tags': ['scan', 'clustering'],
        'wandb_notes': 'SCAN clustering training on pre-trained sensor encoder'
    }


def main():
    parser = argparse.ArgumentParser(description='Train SCAN clustering model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--pretrained_model', type=str, required=True,
                       help='Path to pre-trained sensor encoder model')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--vocab', type=str, required=True,
                       help='Path to vocabulary file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for models and logs')
    parser.add_argument('--num_clusters', type=int, default=20,
                       help='Number of clusters')
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
    parser.add_argument('--wandb_project', type=str, default='discover-scan',
                       help='W&B project name')
    parser.add_argument('--wandb_name', type=str, help='W&B run name')
    parser.add_argument('--wandb_tags', type=str, nargs='+',
                       default=['scan', 'clustering'], help='W&B tags')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable W&B logging')

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()

    # Override with command line arguments (only if provided)
    # Always override required paths
    config.update({
        'pretrained_model_path': args.pretrained_model,
        'train_data_path': args.train_data,
        'vocab_path': args.vocab,
        'output_dir': args.output_dir,
        'use_wandb': not args.no_wandb,
    })

    # Only override optional arguments if they differ from defaults
    parser_defaults = {
        'num_clusters': 20,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'max_epochs': 40,
        'num_neighbors': 20,
        'entropy_weight': 2.0,
        'wandb_project': 'discover-scan',
        'wandb_tags': ['scan', 'clustering']
    }

    # Override only if different from parser default
    if args.num_clusters != parser_defaults['num_clusters']:
        config['num_clusters'] = args.num_clusters
    if args.batch_size != parser_defaults['batch_size']:
        config['batch_size'] = args.batch_size
    if args.learning_rate != parser_defaults['learning_rate']:
        config['learning_rate'] = args.learning_rate
    if args.max_epochs != parser_defaults['max_epochs']:
        config['max_epochs'] = args.max_epochs
    if args.num_neighbors != parser_defaults['num_neighbors']:
        config['num_neighbors'] = args.num_neighbors
    if args.entropy_weight != parser_defaults['entropy_weight']:
        config['entropy_weight'] = args.entropy_weight
    if args.wandb_project != parser_defaults['wandb_project']:
        config['wandb_project'] = args.wandb_project
    if args.wandb_name is not None:
        config['wandb_name'] = args.wandb_name
    if args.wandb_tags != parser_defaults['wandb_tags']:
        config['wandb_tags'] = args.wandb_tags

    # Generate run name if not provided
    if not config.get('wandb_name'):
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config['wandb_name'] = f"scan_{config['num_clusters']}cl_{timestamp}"
        config['run_name'] = config['wandb_name']

    # Create trainer and start training
    trainer = SCANTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
