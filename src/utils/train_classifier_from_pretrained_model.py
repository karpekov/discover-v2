#!/usr/bin/env python3
"""
Train a linear or MLP classifier on top of frozen pretrained embeddings for activity classification.

This script loads a pretrained sensor encoder, freezes it, and trains a simple classifier head
to predict activity labels. Useful for evaluating embedding quality and downstream task performance.

Usage Examples:

    # Basic: Linear probe on L1 labels (default, data-dir auto-detected)
    python src/utils/train_classifier_from_pretrained_model.py \
        --model trained_models/milan/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1

    # MLP classifier with custom epochs
    python src/utils/train_classifier_from_pretrained_model.py \
        --model trained_models/milan/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1 \
        --classifier mlp \
        --epochs 30

    # Override data directory (if needed)
    python src/utils/train_classifier_from_pretrained_model.py \
        --model trained_models/milan/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1 \
        --data-dir data/processed/casas/cairo/FD_30_p \
        --classifier mlp

    # L2 (secondary) labels with tuned learning rate
    python src/utils/train_classifier_from_pretrained_model.py \
        --model trained_models/milan/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1 \
        --label-level l2 \
        --lr 5e-4

    # Full custom configuration
    python src/utils/train_classifier_from_pretrained_model.py \
        --model trained_models/milan/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1 \
        --classifier mlp \
        --label-level l2 \
        --epochs 50 \
        --batch-size 128 \
        --lr 2e-3

Arguments:
    --model: Path to pretrained model directory (required)
    --data-dir: Path to data directory with train.json/test.json (optional, auto-detected from model name)
    --classifier: Classifier type - 'linear' or 'mlp' (default: linear)
    --label-level: Label granularity - 'l1' (primary) or 'l2' (secondary) (default: l1)
    --epochs: Number of training epochs (default: 30)
    --batch-size: Batch size for training (default: 64)
    --lr: Learning rate (default: 1e-3)
    --unfreeze-layers: Number of encoder layers to fine-tune (default: 0)
    --use-class-weights: Enable class weighting for imbalanced datasets
    --no-scheduler: Disable learning rate scheduler
    --gradient-clip: Gradient clipping value (default: 1.0)

Output:
    - Training progress with loss, accuracy, and F1 scores per epoch
    - Best validation accuracy and F1 score
    - Results saved to: results/evals/{dataset}/{config}/{model}/clf_probing/
      * results_{classifier}_{epochs}_{label_level}.json - Machine-readable metrics
      * results_{classifier}_{epochs}_{label_level}.txt - Human-readable report
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import time
from datetime import datetime, timedelta
from sklearn.metrics import f1_score

# Add project root to path (go up 2 levels from src/utils/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.alignment.model import AlignmentModel
from src.alignment.config import AlignmentConfig


class ActivityDataset(Dataset):
    """Simple dataset that returns sensor sequences and activity labels."""

    def __init__(self, data_path, vocab_path, label_level='l1'):
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.samples = data['samples']
        self.label_level = label_level

        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)

        # Filter samples that have activity labels (check both old and new formats)
        filtered_samples = []
        no_activity_count = 0

        for s in self.samples:
            # For L2 labels, check L1 first and exclude No_Activity samples
            if label_level == 'l2':
                l1_label = None
                # Get L1 label to check if it's No_Activity
                if 'metadata' in s and 'ground_truth_labels' in s['metadata']:
                    l1_label = s['metadata']['ground_truth_labels'].get('primary_l1', '')
                else:
                    l1_label = s.get('first_activity') or s.get('activity', '')

                # Skip if L1 is No_Activity (case-insensitive check)
                if l1_label and l1_label.lower().replace('_', '').replace('-', '') in ['noactivity', 'no activity']:
                    no_activity_count += 1
                    continue

            # New format: metadata.ground_truth_labels.primary_l1/primary_l2
            if 'metadata' in s and 'ground_truth_labels' in s['metadata']:
                if label_level == 'l1' and 'primary_l1' in s['metadata']['ground_truth_labels']:
                    filtered_samples.append(s)
                elif label_level == 'l2' and 'primary_l2' in s['metadata']['ground_truth_labels']:
                    filtered_samples.append(s)
            # Old format: first_activity (L1) or first_activity_l2 (L2)
            elif label_level == 'l1' and (s.get('first_activity') or s.get('activity')):
                filtered_samples.append(s)
            elif label_level == 'l2' and s.get('first_activity_l2'):
                filtered_samples.append(s)

        self.samples = filtered_samples

        if label_level == 'l2' and no_activity_count > 0:
            print(f"Filtered out {no_activity_count} samples with L1 'No_Activity' label")
        print(f"Loaded {len(self.samples)} samples with {label_level.upper()} labels")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get activity label based on label_level
        activity = None
        if 'metadata' in sample and 'ground_truth_labels' in sample['metadata']:
            gt_labels = sample['metadata']['ground_truth_labels']
            if self.label_level == 'l1':
                activity = gt_labels.get('primary_l1')
            else:  # l2
                activity = gt_labels.get('primary_l2')

        # Fallback to old format
        if not activity:
            if self.label_level == 'l1':
                activity = sample.get('first_activity') or sample.get('activity') or 'Other_Activity'
            else:  # l2
                activity = sample.get('first_activity_l2') or 'Other_Activity'

        return {
            'sample': sample,
            'activity': activity
        }


def collate_fn(batch, dataset, model_config):
    """Prepare batch for the pretrained model."""
    # Use the alignment dataset's collate logic but without text embeddings
    sensor_sequences = [item['sample']['sensor_sequence'] for item in batch]
    activities = [item['activity'] for item in batch]

    # Prepare sensor data similar to AlignmentDataset
    batch_size = len(sensor_sequences)
    max_len = max(len(seq) for seq in sensor_sequences)

    # Get categorical fields from config - handle both dict and config object
    if hasattr(model_config, 'metadata') and hasattr(model_config.metadata, 'categorical_fields'):
        categorical_fields = model_config.metadata.categorical_fields
    elif isinstance(model_config, dict):
        categorical_fields = model_config.get('metadata', {}).get('categorical_fields', ['sensor', 'state', 'room_id'])
    else:
        categorical_fields = ['sensor', 'state', 'room_id']  # Default fallback

    # Initialize tensors
    categorical_features = {
        field: torch.zeros((batch_size, max_len), dtype=torch.long)
        for field in categorical_fields
    }
    coordinates = torch.zeros((batch_size, max_len, 2), dtype=torch.float32)
    time_deltas = torch.zeros((batch_size, max_len), dtype=torch.float32)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    # Field mapping
    field_mapping = {
        'sensor': 'sensor_id',
        'sensor_id': 'sensor_id',
        'state': 'event_type',
        'event_type': 'event_type',
        'room_id': 'room',
        'room': 'room',
    }

    # Fill tensors
    for i, seq in enumerate(sensor_sequences):
        seq_len = min(len(seq), max_len)
        attention_mask[i, :seq_len] = True

        for j, event in enumerate(seq[:max_len]):
            # Categorical features
            for field in categorical_fields:
                event_key = field_mapping.get(field, field)
                value = event.get(event_key, 'UNK')
                idx_val = dataset.vocab[field].get(str(value), dataset.vocab[field].get('UNK', 0))
                categorical_features[field][i, j] = idx_val

            # Coordinates
            coordinates[i, j, 0] = event.get('x', 0.0)
            coordinates[i, j, 1] = event.get('y', 0.0)

            # Time delta (simplified)
            if j > 0:
                time_deltas[i, j] = 1.0  # Placeholder

    return {
        'sensor_data': {
            'categorical_features': categorical_features,
            'coordinates': coordinates,
            'time_deltas': time_deltas,
        },
        'attention_mask': attention_mask,
        'activities': activities
    }


class ActivityClassifier(nn.Module):
    """Simple classifier on top of frozen embeddings."""

    def __init__(self, pretrained_model, num_classes, classifier_type='linear', hidden_dim=256,
                 unfreeze_layers=0):
        super().__init__()
        self.encoder = pretrained_model.sensor_encoder

        # Freeze encoder by default
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Optionally unfreeze the last N transformer layers
        if unfreeze_layers > 0 and hasattr(self.encoder, 'transformer'):
            total_layers = len(self.encoder.transformer.layers)
            layers_to_unfreeze = min(unfreeze_layers, total_layers)
            print(f"Unfreezing last {layers_to_unfreeze} transformer layers (out of {total_layers})")

            for layer in self.encoder.transformer.layers[-layers_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True

            # Also unfreeze layer norm if present
            if hasattr(self.encoder.transformer, 'norm'):
                for param in self.encoder.transformer.norm.parameters():
                    param.requires_grad = True

        # Get embedding dimension from CLIP projection (not d_model)
        # The forward_clip method returns projected embeddings
        embed_dim = self.encoder.config.projection_dim

        # Classification head
        if classifier_type == 'linear':
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:  # mlp
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes)
            )

        print(f"Created {classifier_type} classifier: {embed_dim} -> {num_classes}")

    def forward(self, input_data, attention_mask):
        # Get CLIP projected embeddings from encoder
        # Use forward_clip which returns projected embeddings directly
        # Note: If encoder layers are unfrozen, gradients will flow through them
        embeddings = self.encoder.forward_clip(
            input_data=input_data,
            attention_mask=attention_mask
        )  # [batch_size, projection_dim]

        # Classify
        logits = self.classifier(embeddings)
        return logits


def train_classifier(model, train_loader, val_loader, device, epochs=10, lr=1e-3,
                    class_weights=None, use_scheduler=True, gradient_clip=1.0):
    """Train the classifier."""
    # Collect all trainable parameters (classifier + any unfrozen encoder layers)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    # Optional: learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)

    # Loss function with optional class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0
    best_val_f1 = 0.0

    # Track training start time
    training_start_time = time.time()
    training_start_datetime = datetime.now()
    print(f"\n{'='*60}")
    print(f"Training started at: {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Train
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            # Pack input_data dict (same format as evaluate_embeddings.py)
            input_data = {}
            for k, v in batch['sensor_data'].items():
                if k == 'categorical_features':
                    # Handle nested dict of tensors
                    input_data['categorical_features'] = {
                        field: tensor.to(device) for field, tensor in v.items()
                    }
                elif isinstance(v, torch.Tensor):
                    input_data[k] = v.to(device)

            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label_ids'].to(device)

            optimizer.zero_grad()
            logits = model(input_data, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping for stability
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = 100. * sum([p == l for p, l in zip(train_preds, train_labels)]) / len(train_labels)
        train_f1_macro = f1_score(train_labels, train_preds, average='macro', zero_division=0)
        train_f1_weighted = f1_score(train_labels, train_preds, average='weighted', zero_division=0)

        # Validate
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                # Pack input_data dict (same format as evaluate_embeddings.py)
                input_data = {}
                for k, v in batch['sensor_data'].items():
                    if k == 'categorical_features':
                        # Handle nested dict of tensors
                        input_data['categorical_features'] = {
                            field: tensor.to(device) for field, tensor in v.items()
                        }
                    elif isinstance(v, torch.Tensor):
                        input_data[k] = v.to(device)

                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label_ids'].to(device)

                logits = model(input_data, attention_mask)
                _, predicted = logits.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = 100. * sum([p == l for p, l in zip(val_preds, val_labels)]) / len(val_labels)
        val_f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        val_f1_weighted = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_f1_weighted > best_val_f1:
            best_val_f1 = val_f1_weighted

        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        epoch_duration_str = str(timedelta(seconds=int(epoch_duration)))

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | Train F1: {train_f1_weighted:.4f} | "
              f"Val Acc: {val_acc:.2f}% | Val F1-M: {val_f1_macro:.4f} | Val F1-W: {val_f1_weighted:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_duration_str}")

        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step()

    # Calculate total training time
    training_end_time = time.time()
    training_end_datetime = datetime.now()
    total_training_time = training_end_time - training_start_time
    total_time_str = str(timedelta(seconds=int(total_training_time)))

    print(f"\n{'='*60}")
    print(f"Training completed at: {training_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {total_time_str}")
    print(f"Average time per epoch: {str(timedelta(seconds=int(total_training_time/epochs)))}")
    print(f"{'='*60}")
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation F1 (weighted): {best_val_f1:.4f}")

    # Return comprehensive metrics (from last epoch)
    return {
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'final_train_acc': train_acc,
        'final_train_f1_macro': train_f1_macro,
        'final_train_f1_weighted': train_f1_weighted,
        'final_val_acc': val_acc,
        'final_val_f1_macro': val_f1_macro,
        'final_val_f1_weighted': val_f1_weighted,
        'total_training_time': total_time_str,
        'avg_time_per_epoch': str(timedelta(seconds=int(total_training_time/epochs)))
    }


def save_results(model_path, metrics, hyperparams, output_dir):
    """Save evaluation results to text and JSON files with descriptive naming."""

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename suffix based on hyperparams
    classifier_suffix = 'lin' if hyperparams['classifier_type'] == 'linear' else 'mlp'
    epoch_suffix = f"e{hyperparams['epochs']}"
    label_suffix = f"l{hyperparams['label_level']}"

    filename_base = f"results_{classifier_suffix}_{epoch_suffix}_{label_suffix}"

    # Prepare comprehensive results dictionary
    results = {
        'model_path': str(model_path),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hyperparameters': hyperparams,
        'metrics': {
            'train': {
                'accuracy': metrics['final_train_acc'],
                'f1_macro': metrics['final_train_f1_macro'],
                'f1_weighted': metrics['final_train_f1_weighted']
            },
            'validation': {
                'accuracy': metrics['final_val_acc'],
                'f1_macro': metrics['final_val_f1_macro'],
                'f1_weighted': metrics['final_val_f1_weighted']
            },
            'best': {
                'val_accuracy': metrics['best_val_acc'],
                'val_f1_weighted': metrics['best_val_f1']
            }
        },
        'training_time': {
            'total': metrics['total_training_time'],
            'avg_per_epoch': metrics['avg_time_per_epoch']
        }
    }

    # Save JSON
    json_path = output_dir / f"{filename_base}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save text report
    txt_path = output_dir / f"{filename_base}.txt"
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ACTIVITY CLASSIFICATION EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")

        f.write(f"Model: {model_path}\n")
        f.write(f"Timestamp: {results['timestamp']}\n\n")

        f.write("-" * 70 + "\n")
        f.write("HYPERPARAMETERS\n")
        f.write("-" * 70 + "\n")
        for key, value in hyperparams.items():
            f.write(f"  {key}: {value}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("FINAL METRICS (Last Epoch)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Train Accuracy:      {metrics['final_train_acc']:.2f}%\n")
        f.write(f"Train F1 (macro):    {metrics['final_train_f1_macro']:.4f}\n")
        f.write(f"Train F1 (weighted): {metrics['final_train_f1_weighted']:.4f}\n\n")
        f.write(f"Val Accuracy:        {metrics['final_val_acc']:.2f}%\n")
        f.write(f"Val F1 (macro):      {metrics['final_val_f1_macro']:.4f}\n")
        f.write(f"Val F1 (weighted):   {metrics['final_val_f1_weighted']:.4f}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("BEST METRICS (Across All Epochs)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best Val Accuracy:      {metrics['best_val_acc']:.2f}%\n")
        f.write(f"Best Val F1 (weighted): {metrics['best_val_f1']:.4f}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("TRAINING TIME\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total:           {metrics['total_training_time']}\n")
        f.write(f"Avg per epoch:   {metrics['avg_time_per_epoch']}\n")

        f.write("\n" + "="*70 + "\n")

    print(f"\nResults saved to:")
    print(f"  - {json_path}")
    print(f"  - {txt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to data directory containing train.json and test.json (auto-detected from model name if not provided)')
    parser.add_argument('--classifier', type=str, default='linear', choices=['linear', 'mlp'],
                       help='Classifier type: linear or mlp')
    parser.add_argument('--label-level', type=str, default='l1', choices=['l1', 'l2'],
                       help='Activity label level: l1 (primary) or l2 (secondary)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--unfreeze-layers', type=int, default=0,
                       help='Number of encoder layers to unfreeze for fine-tuning (0=frozen encoder)')
    parser.add_argument('--use-class-weights', action='store_true',
                       help='Use class weights to handle imbalanced data')
    parser.add_argument('--no-scheduler', action='store_true',
                       help='Disable learning rate scheduler')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping value (0 to disable)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Label level: {args.label_level.upper()}")

    # Load pretrained model using AlignmentModel.load() method (same as evaluate_embeddings.py)
    model_dir = Path(args.model)
    checkpoint_path = model_dir / 'best_model.pt'
    config_path = model_dir / 'config.yaml'

    print(f"\nLoading pretrained model from {model_dir}")

    # Load config to get vocab path
    config = AlignmentConfig.from_yaml(str(config_path))

    # Use AlignmentModel.load() which handles all the complexity
    pretrained_model = AlignmentModel.load(
        str(checkpoint_path),
        device=device,
        vocab_path=config.vocab_path
    )
    pretrained_model.eval()

    print("Loaded pretrained model successfully")

    # Auto-detect data directory from model name if not provided
    if args.data_dir is None:
        model_name = model_dir.name
        # Parse model name to extract dataset and config
        # e.g., milan_fd60_... -> milan, FD_60 or milan_fl20_... -> milan, FL_20
        parts = model_name.split('_')

        # First part is typically the dataset name
        dataset_name = parts[0]  # e.g., 'milan'

        # Find data config (e.g., 'fd60' -> 'FD_60', 'fl20' -> 'FL_20')
        data_config = None
        for part in parts:
            # Handle Fixed Duration (fd) or Fixed Length (fl)
            if (part.startswith('fd') or part.startswith('fl')) and len(part) > 2 and part[2:].isdigit():
                config_type = part[:2].upper()  # 'fd' -> 'FD', 'fl' -> 'FL'
                value = part[2:]  # '60', '20', etc.
                data_config = f'{config_type}_{value}'
                break

        if data_config is None:
            raise ValueError(f"Could not auto-detect data config from model name: {model_name}. Please provide --data-dir explicitly.")

        data_dir = Path(f'data/processed/casas/{dataset_name}/{data_config}_p')
        print(f"Auto-detected data directory: {data_dir}")
    else:
        data_dir = Path(args.data_dir)
        print(f"Using provided data directory: {data_dir}")

    # Load activity data (use vocab path from config)
    train_dataset = ActivityDataset(data_dir / 'train.json', config.vocab_path, label_level=args.label_level)
    val_dataset = ActivityDataset(data_dir / 'test.json', config.vocab_path, label_level=args.label_level)

    # Build activity label mapping based on label level
    all_activities = set()
    for sample in train_dataset.samples + val_dataset.samples:
        # Handle both old and new formats
        if 'metadata' in sample and 'ground_truth_labels' in sample['metadata']:
            if args.label_level == 'l1':
                activity = sample['metadata']['ground_truth_labels'].get('primary_l1')
            else:  # l2
                activity = sample['metadata']['ground_truth_labels'].get('primary_l2')
        else:
            if args.label_level == 'l1':
                activity = sample.get('first_activity') or sample.get('activity')
            else:  # l2
                activity = sample.get('first_activity_l2')
        if activity:
            all_activities.add(activity)

    activity_to_idx = {act: idx for idx, act in enumerate(sorted(all_activities))}
    num_classes = len(activity_to_idx)
    print(f"\n{num_classes} {args.label_level.upper()} activity classes: {list(activity_to_idx.keys())[:5]}...")

    # Create dataloaders with label IDs
    def collate_with_labels(batch):
        result = collate_fn(batch, train_dataset, pretrained_model.config.encoder)
        result['label_ids'] = torch.tensor([activity_to_idx[item['activity']] for item in batch])
        return result

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_with_labels)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_with_labels)

    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        from collections import Counter
        train_labels = []
        for sample in train_dataset.samples:
            # Handle both old and new formats
            if 'metadata' in sample and 'ground_truth_labels' in sample['metadata']:
                if args.label_level == 'l1':
                    activity = sample['metadata']['ground_truth_labels'].get('primary_l1')
                else:
                    activity = sample['metadata']['ground_truth_labels'].get('primary_l2')
            else:
                if args.label_level == 'l1':
                    activity = sample.get('first_activity') or sample.get('activity')
                else:
                    activity = sample.get('first_activity_l2')
            if activity:
                train_labels.append(activity_to_idx[activity])

        label_counts = Counter(train_labels)
        total_samples = len(train_labels)
        weights = [total_samples / (num_classes * label_counts[i]) for i in range(num_classes)]
        class_weights = torch.FloatTensor(weights).to(device)
        print(f"\nUsing class weights (min: {min(weights):.3f}, max: {max(weights):.3f})")

    # Create classifier
    classifier = ActivityClassifier(
        pretrained_model,
        num_classes,
        classifier_type=args.classifier,
        hidden_dim=256,
        unfreeze_layers=args.unfreeze_layers
    )
    classifier.to(device)

    # Train
    if args.unfreeze_layers > 0:
        print(f"\nTraining {args.classifier} classifier + {args.unfreeze_layers} encoder layers for {args.epochs} epochs...")
    else:
        print(f"\nTraining {args.classifier} classifier for {args.epochs} epochs...")

    metrics = train_classifier(
        classifier, train_loader, val_loader, device,
        epochs=args.epochs,
        lr=args.lr,
        class_weights=class_weights,
        use_scheduler=not args.no_scheduler,
        gradient_clip=args.gradient_clip
    )

    print("\n" + "=" * 60)
    print(f"Final Results:")
    print(f"  Model: {model_dir.name}")
    print(f"  Classifier: {args.classifier}")
    print(f"  Label Level: {args.label_level.upper()}")
    print(f"  Num Classes: {num_classes}")
    print(f"  Best Val Accuracy: {metrics['best_val_acc']:.2f}%")
    print(f"  Best Val F1 (weighted): {metrics['best_val_f1']:.4f}")
    print("=" * 60)

    # Determine output directory based on model path and data_dir
    # e.g., trained_models/milan/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1
    # -> results/evals/milan/FD_60/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1/clf_probing
    model_name = model_dir.name

    # Extract dataset info from data_dir path
    # e.g., data/processed/casas/milan/FD_60_p -> dataset: milan, config: FD_60
    dataset_parts = data_dir.parts
    dataset_name = 'unknown'
    data_config = 'unknown'

    # Try to find dataset name (e.g., 'milan', 'cairo', etc.)
    for i, part in enumerate(dataset_parts):
        if part == 'casas' and i + 1 < len(dataset_parts):
            dataset_name = dataset_parts[i + 1]
            if i + 2 < len(dataset_parts):
                data_config = dataset_parts[i + 2].replace('_p', '')  # 'FD_60_p' -> 'FD_60'
            break

    output_dir = Path('results/evals') / dataset_name / data_config / model_name / 'clf_probing'

    # Prepare hyperparameters dict
    hyperparams = {
        'classifier_type': args.classifier,
        'label_level': args.label_level,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'unfreeze_layers': args.unfreeze_layers,
        'use_class_weights': args.use_class_weights,
        'use_scheduler': not args.no_scheduler,
        'gradient_clip': args.gradient_clip,
        'num_classes': num_classes,
        'data_config': data_config
    }

    # Save results
    save_results(model_dir, metrics, hyperparams, output_dir)


if __name__ == '__main__':
    main()

