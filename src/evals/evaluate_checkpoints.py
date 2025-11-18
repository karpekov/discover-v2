#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

"""
Checkpoint evaluation script for CASAS activity recognition.
Evaluates F1 performance across all available checkpoints to track training progress.

Sample Usage:
python src/evals/evaluate_checkpoints.py \
    --model_dir trained_models/milan/tiny_20_oct1 \
    --train_data data/processed/casas/milan/training_20/train.json \
    --test_data data/processed/casas/milan/training_20/presegmented_test.json \
    --vocab data/processed/casas/milan/training_20/vocab.json \
    --output_dir results/evals/milan/tiny_20_oct1_checkpoints \
    --max_samples 5000 \
    --filter_noisy_labels
"""

import torch
import numpy as np
import pandas as pd
import json
import argparse
import re
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Evaluation metrics
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from models.text_encoder import TextEncoder
from models.sensor_encoder import SensorEncoder
from dataio.dataset import SmartHomeDataset
from dataio.collate import create_data_loader
from utils.device_utils import get_optimal_device, log_device_info
from utils.label_utils import convert_labels_to_text


class CheckpointEvaluator:
    """Evaluate F1 performance across multiple checkpoints to track training progress."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_optimal_device()
        log_device_info(self.device)

        # Find all checkpoint files
        self._find_checkpoints()

        # Load datasets once (they're the same for all checkpoints)
        self._load_datasets()

    def _find_checkpoints(self):
        """Find all checkpoint files in the model directory."""
        model_dir = Path(self.config['model_dir'])

        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")

        # Find all checkpoint files
        checkpoint_files = []

        # Look for checkpoint_step_*.pt files
        for file_path in model_dir.glob("checkpoint_step_*.pt"):
            checkpoint_files.append(file_path)

        # Also look for best_model.pt and final_model.pt
        for special_file in ["best_model.pt", "final_model.pt"]:
            special_path = model_dir / special_file
            if special_path.exists():
                checkpoint_files.append(special_path)

        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {model_dir}")

        # Sort checkpoints by step number (extract from filename)
        def extract_step(file_path):
            if "checkpoint_step_" in file_path.name:
                match = re.search(r'checkpoint_step_(\d+)\.pt', file_path.name)
                return int(match.group(1)) if match else float('inf')
            elif "best_model.pt" in file_path.name:
                return float('inf') - 1  # Second to last
            elif "final_model.pt" in file_path.name:
                return float('inf')  # Last
            else:
                return float('inf')

        self.checkpoint_files = sorted(checkpoint_files, key=extract_step)

        print(f"🔍 Found {len(self.checkpoint_files)} checkpoint files:")
        for i, checkpoint in enumerate(self.checkpoint_files):
            print(f"   {i+1}. {checkpoint.name}")

    def _load_datasets(self):
        """Load datasets once for all evaluations."""
        self.datasets = {}

        if self.config['train_data_path'] and Path(self.config['train_data_path']).exists():
            self.datasets['train'] = SmartHomeDataset(
                data_path=self.config['train_data_path'],
                vocab_path=self.config['vocab_path'],
                sequence_length=20,
                max_captions=1
            )
            print(f"📊 Train dataset: {len(self.datasets['train'])} samples")

        if self.config['test_data_path'] and Path(self.config['test_data_path']).exists():
            self.datasets['test'] = SmartHomeDataset(
                data_path=self.config['test_data_path'],
                vocab_path=self.config['vocab_path'],
                sequence_length=20,
                max_captions=1
            )
            print(f"📊 Test dataset: {len(self.datasets['test'])} samples")

    def _load_models_from_checkpoint(self, checkpoint_path: Path):
        """Load models from a specific checkpoint."""
        print(f"🔄 Loading models from {checkpoint_path.name}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Text encoder - use config from checkpoint
        model_config = checkpoint.get('config', {})
        text_model_name = model_config.get('text_model_name', 'thenlper/gte-base')
        text_encoder = TextEncoder(text_model_name)
        text_encoder.to(self.device)

        # Sensor encoder - use config from checkpoint
        vocab_sizes = checkpoint['vocab_sizes']
        sensor_encoder = SensorEncoder(
            vocab_sizes=vocab_sizes,
            d_model=model_config.get('d_model', 768),
            n_layers=model_config.get('n_layers', 6),
            n_heads=model_config.get('n_heads', 8),
            d_ff=model_config.get('d_ff', 3072),
            max_seq_len=model_config.get('max_seq_len', 512),
            dropout=model_config.get('dropout', 0.1),
            fourier_bands=model_config.get('fourier_bands', 12),
            use_rope_time=model_config.get('use_rope_time', False),
            use_rope_2d=model_config.get('use_rope_2d', False)
        )
        sensor_encoder.load_state_dict(checkpoint['sensor_encoder_state_dict'])
        sensor_encoder.to(self.device)
        sensor_encoder.eval()

        return text_encoder, sensor_encoder, vocab_sizes

    def extract_embeddings_and_labels(self, text_encoder, sensor_encoder, vocab_sizes,
                                    split: str, max_samples: int = 5000) -> Tuple[np.ndarray, List[str], List[str]]:
        """Extract embeddings and ground truth labels from dataset."""
        import random
        from torch.utils.data import Subset

        if split not in self.datasets:
            raise ValueError(f"Split '{split}' not available. Available: {list(self.datasets.keys())}")

        dataset = self.datasets[split]
        actual_samples = min(max_samples, len(dataset))

        # Create a random subset if needed
        if actual_samples < len(dataset):
            # Set consistent seed for reproducible sampling across checkpoints
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

            all_indices = list(range(len(dataset)))
            random.shuffle(all_indices)
            selected_indices = all_indices[:actual_samples]
            dataset = Subset(dataset, selected_indices)

        # Create data loader
        data_loader = create_data_loader(
            dataset=dataset,
            text_encoder=text_encoder,
            span_masker=None,
            vocab_sizes=vocab_sizes,
            device=self.device,
            batch_size=64,
            shuffle=False,  # Don't shuffle for consistent evaluation
            num_workers=0,
            apply_mlm=False
        )

        embeddings = []
        labels_l1 = []
        labels_l2 = []
        samples_processed = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if samples_processed >= actual_samples:
                    break

                # Extract CLIP projected embeddings (512-dim)
                # Pack data for new encoder interface
                input_data = {
                    'categorical_features': batch['categorical_features'],
                    'coordinates': batch['coordinates'],
                    'time_deltas': batch['time_deltas']
                }
                sensor_emb = sensor_encoder.forward_clip(
                    input_data=input_data,
                    attention_mask=batch['mask']
                )

                embeddings.append(sensor_emb.cpu().numpy())

                # Extract labels for this batch
                batch_size_actual = sensor_emb.shape[0]
                batch_labels_l1 = batch.get('activity_labels', ['Unknown'] * batch_size_actual)
                batch_labels_l2 = batch.get('activity_labels_l2', ['Unknown'] * batch_size_actual)

                for i in range(batch_size_actual):
                    if samples_processed >= actual_samples:
                        break

                    label_l1 = batch_labels_l1[i] if i < len(batch_labels_l1) else 'Unknown'
                    label_l2 = batch_labels_l2[i] if i < len(batch_labels_l2) else 'Unknown'

                    labels_l1.append(label_l1)
                    labels_l2.append(label_l2)
                    samples_processed += 1

        # Concatenate embeddings
        embeddings = np.vstack(embeddings)[:actual_samples]
        return embeddings, labels_l1, labels_l2

    def filter_noisy_labels(self, embeddings: np.ndarray, labels_l1: List[str], labels_l2: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
        """Filter out noisy/uninformative labels."""
        exclude_labels = {
            'other', 'no_activity', 'unknown', 'none', 'null', 'nan',
            'no activity', 'other activity', 'miscellaneous', 'misc'
        }

        valid_indices = []
        for i, (l1, l2) in enumerate(zip(labels_l1, labels_l2)):
            l1_lower = l1.lower().strip()
            l2_lower = l2.lower().strip()

            if l1_lower not in exclude_labels and l2_lower not in exclude_labels:
                valid_indices.append(i)

        if not valid_indices:
            return embeddings, labels_l1, labels_l2

        filtered_embeddings = embeddings[valid_indices]
        filtered_labels_l1 = [labels_l1[i] for i in valid_indices]
        filtered_labels_l2 = [labels_l2[i] for i in valid_indices]

        return filtered_embeddings, filtered_labels_l1, filtered_labels_l2

    def create_text_prototypes(self, text_encoder, labels: List[str]) -> Dict[str, np.ndarray]:
        """Create text-based prototypes using the text encoder."""
        unique_labels = list(set(labels))
        label_descriptions_lists = convert_labels_to_text(unique_labels)

        prototypes = {}
        text_encoder.eval()

        with torch.no_grad():
            for i, label in enumerate(unique_labels):
                descriptions = label_descriptions_lists[i]
                caption_embeddings = text_encoder.encode_texts_clip(descriptions, self.device).cpu().numpy()
                prototype_embedding = np.mean(caption_embeddings, axis=0)
                prototypes[label] = prototype_embedding

        return prototypes

    def predict_labels_knn(self, query_embeddings: np.ndarray, prototypes: Dict[str, np.ndarray]) -> List[str]:
        """Predict labels using k-nearest neighbors."""
        prototype_labels = list(prototypes.keys())
        prototype_embeddings = np.array([prototypes[label] for label in prototype_labels])

        # Normalize embeddings for cosine similarity
        query_embeddings_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        prototype_embeddings_norm = prototype_embeddings / (np.linalg.norm(prototype_embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarities
        similarities = np.dot(query_embeddings_norm, prototype_embeddings_norm.T)
        nearest_indices = np.argmax(similarities, axis=1)
        predictions = [prototype_labels[idx] for idx in nearest_indices]

        return predictions

    def evaluate_predictions(self, true_labels: List[str], pred_labels: List[str]) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        # Filter out unknown labels
        valid_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels))
                        if true != 'Unknown' and pred != 'Unknown']

        if not valid_indices:
            return {}

        true_filtered = [true_labels[i] for i in valid_indices]
        pred_filtered = [pred_labels[i] for i in valid_indices]

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(true_filtered, pred_filtered),
            'f1_macro': f1_score(true_filtered, pred_filtered, average='macro', zero_division=0),
            'f1_weighted': f1_score(true_filtered, pred_filtered, average='weighted', zero_division=0),
            'precision_macro': precision_score(true_filtered, pred_filtered, average='macro', zero_division=0),
            'recall_macro': recall_score(true_filtered, pred_filtered, average='macro', zero_division=0),
            'num_samples': len(true_filtered),
            'num_classes': len(set(true_filtered + pred_filtered))
        }

        return metrics

    def evaluate_single_checkpoint(self, checkpoint_path: Path, max_samples: int, filter_noisy: bool) -> Dict[str, Any]:
        """Evaluate a single checkpoint and return metrics."""
        # Load models for this checkpoint
        text_encoder, sensor_encoder, vocab_sizes = self._load_models_from_checkpoint(checkpoint_path)

        # Extract training labels for prototypes
        train_embeddings, train_labels_l1, train_labels_l2 = self.extract_embeddings_and_labels(
            text_encoder, sensor_encoder, vocab_sizes, 'train', max_samples
        )

        if filter_noisy:
            train_embeddings, train_labels_l1, train_labels_l2 = self.filter_noisy_labels(
                train_embeddings, train_labels_l1, train_labels_l2
            )

        # Create prototypes
        prototypes_l1 = self.create_text_prototypes(text_encoder, train_labels_l1)
        prototypes_l2 = self.create_text_prototypes(text_encoder, train_labels_l2)

        # Extract test data
        test_embeddings, test_labels_l1, test_labels_l2 = self.extract_embeddings_and_labels(
            text_encoder, sensor_encoder, vocab_sizes, 'test', max_samples
        )

        if filter_noisy:
            test_embeddings, test_labels_l1, test_labels_l2 = self.filter_noisy_labels(
                test_embeddings, test_labels_l1, test_labels_l2
            )

        # Predict labels
        pred_labels_l1 = self.predict_labels_knn(test_embeddings, prototypes_l1)
        pred_labels_l2 = self.predict_labels_knn(test_embeddings, prototypes_l2)

        # Evaluate predictions
        metrics_l1 = self.evaluate_predictions(test_labels_l1, pred_labels_l1)
        metrics_l2 = self.evaluate_predictions(test_labels_l2, pred_labels_l2)

        # Extract step number from checkpoint name
        step = self._extract_step_number(checkpoint_path)

        return {
            'checkpoint': checkpoint_path.name,
            'step': step,
            'metrics_l1': metrics_l1,
            'metrics_l2': metrics_l2
        }

    def _extract_step_number(self, checkpoint_path: Path) -> int:
        """Extract step number from checkpoint filename."""
        if "checkpoint_step_" in checkpoint_path.name:
            match = re.search(r'checkpoint_step_(\d+)\.pt', checkpoint_path.name)
            return int(match.group(1)) if match else -1
        elif "best_model.pt" in checkpoint_path.name:
            return -2  # Special marker for best model
        elif "final_model.pt" in checkpoint_path.name:
            return -1  # Special marker for final model
        else:
            return -3

    def create_performance_chart(self, results: List[Dict], output_dir: Path):
        """Create performance chart showing F1 scores across checkpoints."""
        # Prepare data for plotting
        steps = []
        l1_f1_macro = []
        l1_f1_weighted = []
        l2_f1_macro = []
        l2_f1_weighted = []
        checkpoint_names = []

        for result in results:
            step = result['step']
            if step < 0:  # Special checkpoints
                continue

            steps.append(step)
            l1_f1_macro.append(result['metrics_l1'].get('f1_macro', 0))
            l1_f1_weighted.append(result['metrics_l1'].get('f1_weighted', 0))
            l2_f1_macro.append(result['metrics_l2'].get('f1_macro', 0))
            l2_f1_weighted.append(result['metrics_l2'].get('f1_weighted', 0))
            checkpoint_names.append(result['checkpoint'])

        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('F1 Score Performance Across Training Checkpoints', fontsize=16, fontweight='bold')

        # L1 Performance
        ax1.plot(steps, l1_f1_macro, 'o-', color='#1976D2', label='F1-Macro', linewidth=2, markersize=6)
        ax1.plot(steps, l1_f1_weighted, 's--', color='#FF6F00', label='F1-Weighted', linewidth=2, markersize=6)
        ax1.set_title('L1 (Primary) Activities Performance', fontweight='bold')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('F1 Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # L2 Performance
        ax2.plot(steps, l2_f1_macro, 'o-', color='#1976D2', label='F1-Macro', linewidth=2, markersize=6)
        ax2.plot(steps, l2_f1_weighted, 's--', color='#FF6F00', label='F1-Weighted', linewidth=2, markersize=6)
        ax2.set_title('L2 (Secondary) Activities Performance', fontweight='bold')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.tight_layout()

        # Save plot
        plot_path = output_dir / 'checkpoint_performance_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"💾 Performance chart saved: {plot_path}")

        plt.close()

    def create_summary_table(self, results: List[Dict], output_dir: Path):
        """Create summary table of all checkpoint results."""
        # Create DataFrame
        data = []
        for result in results:
            row = {
                'Checkpoint': result['checkpoint'],
                'Step': result['step'] if result['step'] >= 0 else result['checkpoint'],
                'L1_F1_Macro': result['metrics_l1'].get('f1_macro', 0),
                'L1_F1_Weighted': result['metrics_l1'].get('f1_weighted', 0),
                'L1_Accuracy': result['metrics_l1'].get('accuracy', 0),
                'L2_F1_Macro': result['metrics_l2'].get('f1_macro', 0),
                'L2_F1_Weighted': result['metrics_l2'].get('f1_weighted', 0),
                'L2_Accuracy': result['metrics_l2'].get('accuracy', 0),
                'L1_Classes': result['metrics_l1'].get('num_classes', 0),
                'L2_Classes': result['metrics_l2'].get('num_classes', 0),
                'Samples': result['metrics_l1'].get('num_samples', 0)
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Save as CSV
        csv_path = output_dir / 'checkpoint_performance_summary.csv'
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"💾 Summary table saved: {csv_path}")

        # Save as formatted text
        txt_path = output_dir / 'checkpoint_performance_summary.txt'
        with open(txt_path, 'w') as f:
            f.write("CHECKPOINT PERFORMANCE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model Directory: {self.config['model_dir']}\n")
            f.write(f"Test Data: {self.config['test_data_path']}\n")
            f.write(f"Max Samples: {self.config.get('max_samples', 'N/A')}\n")
            f.write(f"Filter Noisy Labels: {self.config.get('filter_noisy_labels', False)}\n\n")

            # Find best performing checkpoints
            regular_results = [r for r in results if r['step'] >= 0]
            if regular_results:
                best_l1_macro = max(regular_results, key=lambda x: x['metrics_l1'].get('f1_macro', 0))
                best_l1_weighted = max(regular_results, key=lambda x: x['metrics_l1'].get('f1_weighted', 0))
                best_l2_macro = max(regular_results, key=lambda x: x['metrics_l2'].get('f1_macro', 0))
                best_l2_weighted = max(regular_results, key=lambda x: x['metrics_l2'].get('f1_weighted', 0))

                f.write("BEST PERFORMING CHECKPOINTS\n")
                f.write("-" * 40 + "\n")
                f.write(f"L1 F1-Macro:    {best_l1_macro['checkpoint']} (step {best_l1_macro['step']}) - {best_l1_macro['metrics_l1'].get('f1_macro', 0):.4f}\n")
                f.write(f"L1 F1-Weighted: {best_l1_weighted['checkpoint']} (step {best_l1_weighted['step']}) - {best_l1_weighted['metrics_l1'].get('f1_weighted', 0):.4f}\n")
                f.write(f"L2 F1-Macro:    {best_l2_macro['checkpoint']} (step {best_l2_macro['step']}) - {best_l2_macro['metrics_l2'].get('f1_macro', 0):.4f}\n")
                f.write(f"L2 F1-Weighted: {best_l2_weighted['checkpoint']} (step {best_l2_weighted['step']}) - {best_l2_weighted['metrics_l2'].get('f1_weighted', 0):.4f}\n\n")

            f.write("DETAILED RESULTS\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Checkpoint':<25} {'Step':<8} {'L1_F1_Macro':<12} {'L1_F1_Wtd':<12} {'L1_Acc':<8} {'L2_F1_Macro':<12} {'L2_F1_Wtd':<12} {'L2_Acc':<8} {'Samples':<8}\n")
            f.write("-" * 120 + "\n")

            for _, row in df.iterrows():
                f.write(f"{row['Checkpoint']:<25} {str(row['Step']):<8} {row['L1_F1_Macro']:<12.4f} {row['L1_F1_Weighted']:<12.4f} {row['L1_Accuracy']:<8.4f} {row['L2_F1_Macro']:<12.4f} {row['L2_F1_Weighted']:<12.4f} {row['L2_Accuracy']:<8.4f} {row['Samples']:<8}\n")

        print(f"💾 Detailed summary saved: {txt_path}")

    def run_evaluation(self, max_samples: int = 5000, filter_noisy_labels: bool = True) -> List[Dict[str, Any]]:
        """Run evaluation across all checkpoints."""
        print("🚀 Starting checkpoint evaluation...")
        print(f"   Max samples: {max_samples}")
        print(f"   Filter noisy labels: {filter_noisy_labels}")
        print(f"   Checkpoints to evaluate: {len(self.checkpoint_files)}")

        results = []

        for i, checkpoint_path in enumerate(self.checkpoint_files):
            print(f"\n📊 Evaluating checkpoint {i+1}/{len(self.checkpoint_files)}: {checkpoint_path.name}")

            try:
                result = self.evaluate_single_checkpoint(checkpoint_path, max_samples, filter_noisy_labels)
                results.append(result)

                # Print quick summary
                l1_f1_macro = result['metrics_l1'].get('f1_macro', 0)
                l1_f1_weighted = result['metrics_l1'].get('f1_weighted', 0)
                l2_f1_macro = result['metrics_l2'].get('f1_macro', 0)
                l2_f1_weighted = result['metrics_l2'].get('f1_weighted', 0)

                print(f"   L1: F1-Macro={l1_f1_macro:.4f}, F1-Weighted={l1_f1_weighted:.4f}")
                print(f"   L2: F1-Macro={l2_f1_macro:.4f}, F1-Weighted={l2_f1_weighted:.4f}")

            except Exception as e:
                print(f"   ❌ Error evaluating {checkpoint_path.name}: {e}")
                continue

        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate F1 performance across training checkpoints')

    # Model and data paths
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing checkpoint files (e.g., trained_models/milan/tiny_20_oct1)')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--vocab', type=str, required=True,
                       help='Path to vocabulary file')

    # Evaluation parameters
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum number of samples to evaluate (default: 5000)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--filter_noisy_labels', action='store_true',
                       help='Filter out noisy labels like "Other" and "No_Activity"')

    args = parser.parse_args()

    # Configuration
    config = {
        'model_dir': args.model_dir,
        'train_data_path': args.train_data,
        'test_data_path': args.test_data,
        'vocab_path': args.vocab,
        'max_samples': args.max_samples,
        'filter_noisy_labels': args.filter_noisy_labels
    }

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    evaluator = CheckpointEvaluator(config)
    results = evaluator.run_evaluation(
        max_samples=args.max_samples,
        filter_noisy_labels=args.filter_noisy_labels
    )

    if results:
        # Create visualizations and summaries
        print("\n📊 Creating performance chart...")
        evaluator.create_performance_chart(results, output_dir)

        print("📋 Creating summary table...")
        evaluator.create_summary_table(results, output_dir)

        # Save detailed results as JSON
        results_file = output_dir / 'checkpoint_evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'config': config,
                'results': results
            }, f, indent=2, default=str)
        print(f"💾 Detailed results saved: {results_file}")

        print(f"\n✅ Checkpoint evaluation complete! Results saved in: {args.output_dir}")
    else:
        print("\n❌ No successful evaluations completed.")


if __name__ == "__main__":
    main()
