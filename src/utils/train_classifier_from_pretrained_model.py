#!/usr/bin/env python3
"""
Train a linear or MLP classifier on top of frozen pretrained embeddings for activity classification.

This script loads a pretrained sensor encoder, freezes it, and trains a simple classifier head
to predict activity labels. Useful for evaluating embedding quality and downstream task performance.

Usage Examples:

    # Basic: Linear probe on L1 labels (default)
    python src/utils/train_classifier_from_pretrained_model.py \
        --model trained_models/milan/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1

    # MLP classifier with custom epochs
    python src/utils/train_classifier_from_pretrained_model.py \
        --model trained_models/milan/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1 \
        --classifier mlp \
        --epochs 30

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
    --classifier: Classifier type - 'linear' or 'mlp' (default: linear)
    --label-level: Label granularity - 'l1' (primary) or 'l2' (secondary) (default: l1)
    --epochs: Number of training epochs (default: 30)
    --batch-size: Batch size for training (default: 64)
    --lr: Learning rate (default: 1e-3)

Output:
    - Training progress with loss, accuracy, and F1 scores per epoch
    - Best validation accuracy and F1 score
    - Per-class performance metrics
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
        for s in self.samples:
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

    def __init__(self, pretrained_model, num_classes, classifier_type='linear', hidden_dim=256):
        super().__init__()
        self.encoder = pretrained_model.sensor_encoder

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

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
        # Get CLIP projected embeddings from frozen encoder
        # Use forward_clip which returns projected embeddings directly
        with torch.no_grad():
            embeddings = self.encoder.forward_clip(
                input_data=input_data,
                attention_mask=attention_mask
            )  # [batch_size, projection_dim]

        # Classify
        logits = self.classifier(embeddings)
        return logits


def train_classifier(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    """Train the classifier."""
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | Train F1: {train_f1_weighted:.4f} | "
              f"Val Acc: {val_acc:.2f}% | Val F1-M: {val_f1_macro:.4f} | Val F1-W: {val_f1_weighted:.4f} | "
              f"Time: {epoch_duration_str}")

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
    return best_val_acc, best_val_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model directory')
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

    # Load activity data (use vocab path from config)
    data_dir = Path('data/processed/casas/milan/FD_60_p')
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

    # Create classifier
    classifier = ActivityClassifier(
        pretrained_model,
        num_classes,
        classifier_type=args.classifier,
        hidden_dim=256
    )
    classifier.to(device)

    # Train
    print(f"\nTraining {args.classifier} classifier for {args.epochs} epochs...")
    best_acc, best_f1 = train_classifier(classifier, train_loader, val_loader, device,
                                          epochs=args.epochs, lr=args.lr)

    print("\n" + "=" * 60)
    print(f"Final Results:")
    print(f"  Model: {model_dir.name}")
    print(f"  Classifier: {args.classifier}")
    print(f"  Label Level: {args.label_level.upper()}")
    print(f"  Num Classes: {num_classes}")
    print(f"  Best Val Accuracy: {best_acc:.2f}%")
    print(f"  Best Val F1 (weighted): {best_f1:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

