#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

"""
Text encoder only evaluation script for CASAS activity recognition.
Evaluates text captions using raw text encoder embeddings (768-dim GTE) without CLIP projection.
Compares captions against text-based activity prototypes using cosine similarity.

Sample Usage:
  python src/evals/evaluate_text_encoder_only.py \
    --train_data data/processed/casas/milan/training_50/train.json \
    --test_data data/processed/casas/milan/training_50/presegmented_test.json \
    --output_dir results/evals/milan/50_textonly_new \
    --max_samples 10000 \
    --filter_noisy_labels
"""

import torch
import numpy as np
import pandas as pd
import json
import argparse
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
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt

# Local imports
from models.text_encoder import TextEncoder
from utils.device_utils import get_optimal_device, log_device_info
from utils.label_utils import convert_labels_to_text


class TextEncoderOnlyEvaluator:
    """Evaluate text encoder only activity recognition using caption-to-prototype matching."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_optimal_device()
        log_device_info(self.device)

        # Load text encoder (raw, without CLIP projection)
        self._load_text_encoder()

        # Load dataset
        self._load_dataset()

    def _load_text_encoder(self):
        """Load text encoder for raw embedding extraction."""
        model_name = self.config.get('text_model_name', 'thenlper/gte-base')
        print(f"🔄 Loading text encoder: {model_name}")

        # Use SentenceTransformer for sentence-transformers models
        if 'sentence-transformers' in model_name or model_name.startswith('all-'):
            print(f"   Using SentenceTransformer encoder (proper mean pooling)")
            # Lazy import to avoid Python version issues
            from models.sentence_transformer_encoder import SentenceTransformerEncoder
            self.text_encoder = SentenceTransformerEncoder(model_name)
            self.text_encoder.to(self.device)
            self.text_encoder.eval()
            embedding_dim = self.text_encoder.embedding_dim
        else:
            print(f"   Using standard TextEncoder (CLS token pooling)")
            self.text_encoder = TextEncoder(model_name)
            self.text_encoder.to(self.device)
            self.text_encoder.eval()
            embedding_dim = 768

        print(f"✅ Text encoder loaded successfully ({embedding_dim}-dim raw embeddings)")

    def _load_dataset(self):
        """Load Milan dataset with captions."""
        train_path = self.config['train_data_path']
        test_path = self.config['test_data_path']

        print(f"🔄 Loading datasets...")
        print(f"   Train: {train_path}")
        print(f"   Test: {test_path}")

        # Load train data
        with open(train_path, 'r') as f:
            self.train_data = json.load(f)

        # Load test data
        with open(test_path, 'r') as f:
            self.test_data = json.load(f)

        print(f"📊 Train samples: {len(self.train_data)}")
        print(f"📊 Test samples: {len(self.test_data)}")

    def extract_captions_and_labels(self, data: List[Dict], max_samples: int = None, split_name: str = "data") -> Tuple[List[str], List[str], List[str]]:
        """Extract captions and labels from dataset with optional sampling."""
        print(f"🔄 Extracting captions and labels from {split_name}...")

        captions = []
        labels_l1 = []
        labels_l2 = []

        # Apply sampling if requested
        if max_samples and len(data) > max_samples:
            print(f"🎲 Randomly sampling {max_samples} from {len(data)} total samples (seed=42)")
            import random
            random.seed(42)
            data = random.sample(data, max_samples)

        for sample in data:
            # Get long captions (more detailed and informative than short captions)
            long_captions = sample.get('long_captions', [])

            # Get labels
            activity_l1 = sample.get('first_activity', 'Unknown')
            activity_l2 = sample.get('first_activity_l2', 'Unknown')

            # Use only the first long caption from each sample
            if long_captions:
                first_caption = long_captions[0]
                if isinstance(first_caption, str):
                    captions.append(first_caption)
                elif isinstance(first_caption, list) and len(first_caption) > 0:
                    # Handle case where caption is [caption_text, metadata]
                    captions.append(first_caption[0])
                else:
                    continue

                labels_l1.append(activity_l1)
                labels_l2.append(activity_l2)

        print(f"📈 Extracted {len(captions)} captions")
        print(f"📊 L1 Labels: {len(set(labels_l1))} unique ({Counter(labels_l1).most_common(5)})")
        print(f"📊 L2 Labels: {len(set(labels_l2))} unique ({Counter(labels_l2).most_common(5)})")

        return captions, labels_l1, labels_l2

    def filter_noisy_labels(self, captions: List[str], labels_l1: List[str], labels_l2: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Filter out noisy/uninformative labels like 'Other', 'No_Activity', etc."""

        # Define labels to exclude (case-insensitive)
        exclude_labels = {
            'other', 'no_activity', 'unknown', 'none', 'null', 'nan',
            'no activity', 'other activity', 'miscellaneous', 'misc'
        }

        # Find valid indices (keep samples that don't have excluded labels in either L1 or L2)
        valid_indices = []
        for i, (l1, l2) in enumerate(zip(labels_l1, labels_l2)):
            l1_lower = l1.lower().strip()
            l2_lower = l2.lower().strip()

            # Keep sample if neither L1 nor L2 labels are in exclude list
            if l1_lower not in exclude_labels and l2_lower not in exclude_labels:
                valid_indices.append(i)

        if not valid_indices:
            print("⚠️  Warning: All samples filtered out!")
            return captions, labels_l1, labels_l2

        # Filter arrays and lists
        filtered_captions = [captions[i] for i in valid_indices]
        filtered_labels_l1 = [labels_l1[i] for i in valid_indices]
        filtered_labels_l2 = [labels_l2[i] for i in valid_indices]

        print(f"🧹 Filtered out noisy labels:")
        print(f"   Original captions: {len(captions)}")
        print(f"   Filtered captions: {len(filtered_captions)}")
        print(f"   Removed: {len(captions) - len(filtered_captions)} samples")

        # Show what was filtered out
        removed_l1 = Counter([labels_l1[i] for i in range(len(labels_l1)) if i not in valid_indices])
        removed_l2 = Counter([labels_l2[i] for i in range(len(labels_l2)) if i not in valid_indices])

        if removed_l1:
            print(f"   Removed L1 labels: {dict(removed_l1.most_common())}")
        if removed_l2:
            print(f"   Removed L2 labels: {dict(removed_l2.most_common())}")

        return filtered_captions, filtered_labels_l1, filtered_labels_l2

    def embed_captions(self, captions: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed captions using text encoder."""
        print(f"🔄 Embedding {len(captions)} captions...")

        embeddings = []

        # Check if using SentenceTransformer
        is_sentence_transformer = hasattr(self.text_encoder, 'model') and hasattr(self.text_encoder.model, 'encode')

        # Process in batches
        for i in range(0, len(captions), batch_size):
            batch_captions = captions[i:i + batch_size]

            # Use the raw forward method (no CLIP projection)
            with torch.no_grad():
                batch_embeddings = self.text_encoder.encode_texts(batch_captions, self.device)
                embeddings.append(batch_embeddings.cpu().numpy())

            if i % (batch_size * 10) == 0:
                print(f"  Processed {min(i + batch_size, len(captions))}/{len(captions)} captions...")

        # Concatenate all embeddings
        embeddings = np.vstack(embeddings)
        print(f"📈 Generated embeddings shape: {embeddings.shape}")

        return embeddings

    def create_text_prototypes(self, labels: List[str], description_style: str = "baseline") -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """Create text-based prototypes using the text encoder with multiple captions per label."""

        print(f"🔄 Creating text-based label prototypes (style: {description_style})...")

        # Get unique labels and their counts
        label_counts = Counter(labels)
        unique_labels = list(label_counts.keys())

        print(f"📝 Encoding {len(unique_labels)} unique labels with text encoder...")

        # Convert labels to multiple natural language descriptions
        # Get house_name from config or default to milan
        house_name = self.config.get('house_name', 'milan')
        label_descriptions_lists = convert_labels_to_text(unique_labels, single_description=False,
                                                          house_name=house_name,
                                                          description_style=description_style)

        # Create prototypes dictionary by averaging embeddings for multiple captions
        prototypes = {}

        with torch.no_grad():
            for i, label in enumerate(unique_labels):
                descriptions = label_descriptions_lists[i]

                # Encode all descriptions for this label using raw embeddings
                caption_embeddings = self.text_encoder.encode_texts(descriptions, self.device).cpu().numpy()

                # Average the embeddings to create a single prototype
                prototype_embedding = np.mean(caption_embeddings, axis=0)
                prototypes[label] = prototype_embedding

        print(f"✅ Created {len(prototypes)} text-based prototypes")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            descriptions = label_descriptions_lists[unique_labels.index(label)]
            print(f"    {label}: {count} samples → {len(descriptions)} captions (e.g., '{descriptions[0]}')")

        return prototypes, dict(label_counts)


    def predict_labels_knn(self, query_embeddings: np.ndarray,
                          prototypes: Dict[str, np.ndarray],
                          k: int = 1) -> List[str]:
        """Predict labels using k-nearest neighbors between caption and prototype embeddings."""

        print(f"🔄 Predicting labels using {k}-NN comparison...")
        print(f"    Caption embeddings: {query_embeddings.shape[0]} samples")
        print(f"    Text prototypes: {len(prototypes)} activities")

        # Convert prototypes to arrays
        prototype_labels = list(prototypes.keys())
        prototype_embeddings = np.array([prototypes[label] for label in prototype_labels])

        # Normalize embeddings for cosine similarity
        query_embeddings_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        prototype_embeddings_norm = prototype_embeddings / (np.linalg.norm(prototype_embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarities
        similarities = np.dot(query_embeddings_norm, prototype_embeddings_norm.T)

        if k == 1:
            # Simple nearest neighbor
            nearest_indices = np.argmax(similarities, axis=1)
            predictions = [prototype_labels[idx] for idx in nearest_indices]
        else:
            # k-NN with majority voting
            top_k_indices = np.argsort(similarities, axis=1)[:, -k:]
            predictions = []

            for indices in top_k_indices:
                # Get top-k labels and vote
                top_labels = [prototype_labels[idx] for idx in indices]
                # Simple majority vote (could be weighted by similarity)
                prediction = Counter(top_labels).most_common(1)[0][0]
                predictions.append(prediction)

        return predictions

    def evaluate_predictions(self, true_labels: List[str], pred_labels: List[str],
                           label_type: str, verbose: bool = False) -> Dict[str, Any]:
        """Compute evaluation metrics."""

        print(f"🔄 Computing metrics for {label_type} labels...")

        # Filter out unknown labels for fair evaluation
        valid_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels))
                        if true != 'Unknown' and pred != 'Unknown']

        if not valid_indices:
            print(f"⚠️  No valid predictions for {label_type}")
            return {}

        true_filtered = [true_labels[i] for i in valid_indices]
        pred_filtered = [pred_labels[i] for i in valid_indices]

        # Get unique labels
        unique_labels = sorted(list(set(true_filtered + pred_filtered)))

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(true_filtered, pred_filtered),
            'f1_macro': f1_score(true_filtered, pred_filtered, average='macro', zero_division=0),
            'f1_micro': f1_score(true_filtered, pred_filtered, average='micro', zero_division=0),
            'f1_weighted': f1_score(true_filtered, pred_filtered, average='weighted', zero_division=0),
            'precision_macro': precision_score(true_filtered, pred_filtered, average='macro', zero_division=0),
            'recall_macro': recall_score(true_filtered, pred_filtered, average='macro', zero_division=0),
            'num_samples': len(true_filtered),
            'num_classes': len(unique_labels),
            'unique_labels': unique_labels
        }

        # Per-class F1 scores
        per_class_f1 = f1_score(true_filtered, pred_filtered, average=None, zero_division=0, labels=unique_labels)
        metrics['per_class_f1'] = dict(zip(unique_labels, per_class_f1))

        # Per-class precision and recall
        per_class_precision = precision_score(true_filtered, pred_filtered, average=None, zero_division=0, labels=unique_labels)
        per_class_recall = recall_score(true_filtered, pred_filtered, average=None, zero_division=0, labels=unique_labels)
        metrics['per_class_precision'] = dict(zip(unique_labels, per_class_precision))
        metrics['per_class_recall'] = dict(zip(unique_labels, per_class_recall))

        # Classification report
        metrics['classification_report'] = classification_report(
            true_filtered, pred_filtered,
            target_names=unique_labels,
            zero_division=0,
            output_dict=True
        )

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(true_filtered, pred_filtered, labels=unique_labels)

        print(f"✅ {label_type} Metrics:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 (Macro): {metrics['f1_macro']:.4f}")
        print(f"    F1 (Micro): {metrics['f1_micro']:.4f}")
        print(f"    F1 (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"    Classes: {metrics['num_classes']}")
        print(f"    Samples: {metrics['num_samples']}")

        # Verbose output: Per-label statistics
        if verbose:
            print(f"\n📊 Per-Label Performance ({label_type}):")
            print(f"{'Label':<30} {'Support':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
            print("-" * 70)

            label_counts = Counter(true_filtered)
            for label in sorted(unique_labels, key=lambda x: label_counts.get(x, 0), reverse=True):
                support = label_counts.get(label, 0)
                prec = metrics['per_class_precision'][label]
                rec = metrics['per_class_recall'][label]
                f1 = metrics['per_class_f1'][label]
                print(f"{label:<30} {support:>8} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

        return metrics

    def show_classification_examples(self, captions: List[str], true_labels: List[str],
                                     pred_labels: List[str], embeddings: np.ndarray,
                                     prototypes: Dict[str, np.ndarray], label_type: str,
                                     num_examples: int = 3):
        """Show examples of correct and incorrect classifications with similarity scores."""

        print(f"\n🔍 Classification Examples ({label_type}):")
        print("=" * 100)

        # Find correct and incorrect predictions
        correct_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels))
                          if true == pred and true != 'Unknown']
        incorrect_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels))
                            if true != pred and true != 'Unknown' and pred != 'Unknown']

        # Show correct classifications
        if correct_indices:
            print(f"\n✅ CORRECT Classifications (showing up to {num_examples} per label):")
            print("-" * 100)

            # Group by label
            correct_by_label = defaultdict(list)
            for idx in correct_indices:
                correct_by_label[true_labels[idx]].append(idx)

            for label in sorted(correct_by_label.keys())[:5]:  # Show top 5 labels
                indices = correct_by_label[label][:num_examples]
                print(f"\n  Label: {label} ({len(correct_by_label[label])} correct)")
                for idx in indices:
                    # Compute similarity to predicted prototype
                    emb = embeddings[idx]
                    proto = prototypes[pred_labels[idx]]
                    similarity = np.dot(emb, proto) / (np.linalg.norm(emb) * np.linalg.norm(proto))

                    # Get top 3 similar labels
                    similarities = {}
                    for lbl, proto_emb in prototypes.items():
                        sim = np.dot(emb, proto_emb) / (np.linalg.norm(emb) * np.linalg.norm(proto_emb))
                        similarities[lbl] = sim
                    top_3 = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]

                    print(f"    Caption: \"{captions[idx][:80]}...\"" if len(captions[idx]) > 80 else f"    Caption: \"{captions[idx]}\"")
                    print(f"    Similarity: {similarity:.4f} | Top-3: {', '.join([f'{l}({s:.3f})' for l, s in top_3])}")

        # Show incorrect classifications
        if incorrect_indices:
            print(f"\n\n❌ INCORRECT Classifications (showing up to {num_examples * 2}):")
            print("-" * 100)

            # Show diverse errors
            shown = 0
            for idx in incorrect_indices[:num_examples * 2]:
                true_label = true_labels[idx]
                pred_label = pred_labels[idx]

                # Compute similarities
                emb = embeddings[idx]
                true_proto = prototypes.get(true_label)
                pred_proto = prototypes.get(pred_label)

                if true_proto is not None and pred_proto is not None:
                    true_sim = np.dot(emb, true_proto) / (np.linalg.norm(emb) * np.linalg.norm(true_proto))
                    pred_sim = np.dot(emb, pred_proto) / (np.linalg.norm(emb) * np.linalg.norm(pred_proto))

                    print(f"\n  True: {true_label} | Predicted: {pred_label}")
                    print(f"    Caption: \"{captions[idx][:80]}...\"" if len(captions[idx]) > 80 else f"    Caption: \"{captions[idx]}\"")
                    print(f"    True similarity: {true_sim:.4f} | Predicted similarity: {pred_sim:.4f} | Δ: {pred_sim - true_sim:.4f}")
                    shown += 1

                if shown >= num_examples * 2:
                    break

        print("\n" + "=" * 100)

    def _save_verbose_report(self, report_path: Path, 
                            metrics_l1: Dict, metrics_l2: Dict,
                            captions: List[str], true_labels_l1: List[str], true_labels_l2: List[str],
                            pred_labels_l1: List[str], pred_labels_l2: List[str],
                            embeddings: np.ndarray, prototypes_l1: Dict, prototypes_l2: Dict,
                            description_style: str):
        """Save verbose evaluation report to markdown file."""
        
        with open(report_path, 'w') as f:
            f.write("# Verbose Evaluation Report\n\n")
            f.write(f"**Description Style**: {description_style}\n")
            f.write(f"**Total Test Samples**: {len(captions)}\n\n")
            
            # L1 Performance Table
            f.write("## L1 (Primary) Label Performance\n\n")
            f.write(f"**Overall Metrics**:\n")
            f.write(f"- Accuracy: {metrics_l1['accuracy']:.4f}\n")
            f.write(f"- F1-Macro: {metrics_l1['f1_macro']:.4f}\n")
            f.write(f"- F1-Weighted: {metrics_l1['f1_weighted']:.4f}\n\n")
            
            f.write("### Per-Label Statistics\n\n")
            f.write("| Label | Support | Precision | Recall | F1-Score |\n")
            f.write("|-------|---------|-----------|--------|----------|\n")
            
            label_counts = Counter(true_labels_l1)
            unique_labels_l1 = metrics_l1['unique_labels']
            for label in sorted(unique_labels_l1, key=lambda x: label_counts.get(x, 0), reverse=True):
                support = label_counts.get(label, 0)
                prec = metrics_l1['per_class_precision'][label]
                rec = metrics_l1['per_class_recall'][label]
                f1 = metrics_l1['per_class_f1'][label]
                f.write(f"| {label} | {support} | {prec:.4f} | {rec:.4f} | {f1:.4f} |\n")
            
            # L2 Performance Table
            f.write("\n## L2 (Secondary) Label Performance\n\n")
            f.write(f"**Overall Metrics**:\n")
            f.write(f"- Accuracy: {metrics_l2['accuracy']:.4f}\n")
            f.write(f"- F1-Macro: {metrics_l2['f1_macro']:.4f}\n")
            f.write(f"- F1-Weighted: {metrics_l2['f1_weighted']:.4f}\n\n")
            
            f.write("### Per-Label Statistics\n\n")
            f.write("| Label | Support | Precision | Recall | F1-Score |\n")
            f.write("|-------|---------|-----------|--------|----------|\n")
            
            label_counts_l2 = Counter(true_labels_l2)
            unique_labels_l2 = metrics_l2['unique_labels']
            for label in sorted(unique_labels_l2, key=lambda x: label_counts_l2.get(x, 0), reverse=True):
                support = label_counts_l2.get(label, 0)
                prec = metrics_l2['per_class_precision'][label]
                rec = metrics_l2['per_class_recall'][label]
                f1 = metrics_l2['per_class_f1'][label]
                f.write(f"| {label} | {support} | {prec:.4f} | {rec:.4f} | {f1:.4f} |\n")
            
            # Classification Examples for L1
            f.write("\n## L1 Classification Examples\n\n")
            self._write_classification_examples(f, captions, true_labels_l1, pred_labels_l1, 
                                               embeddings, prototypes_l1, "L1", num_examples=3)
            
            # Classification Examples for L2
            f.write("\n## L2 Classification Examples\n\n")
            self._write_classification_examples(f, captions, true_labels_l2, pred_labels_l2, 
                                               embeddings, prototypes_l2, "L2", num_examples=3)

    def _write_classification_examples(self, f, captions: List[str], true_labels: List[str], 
                                       pred_labels: List[str], embeddings: np.ndarray,
                                       prototypes: Dict[str, np.ndarray], label_type: str,
                                       num_examples: int = 3):
        """Write classification examples to file."""
        
        # Find correct and incorrect predictions
        correct_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels)) 
                          if true == pred and true != 'Unknown']
        incorrect_indices = [i for i, (true, pred) in enumerate(zip(true_labels, pred_labels)) 
                            if true != pred and true != 'Unknown' and pred != 'Unknown']
        
        # Correct classifications
        if correct_indices:
            f.write("### ✅ Correct Classifications\n\n")
            
            correct_by_label = defaultdict(list)
            for idx in correct_indices:
                correct_by_label[true_labels[idx]].append(idx)
            
            for label in sorted(correct_by_label.keys())[:5]:
                indices = correct_by_label[label][:num_examples]
                f.write(f"\n**{label}** ({len(correct_by_label[label])} correct):\n\n")
                
                for i, idx in enumerate(indices, 1):
                    emb = embeddings[idx]
                    proto = prototypes[pred_labels[idx]]
                    similarity = np.dot(emb, proto) / (np.linalg.norm(emb) * np.linalg.norm(proto))
                    
                    # Get top 3
                    similarities = {}
                    for lbl, proto_emb in prototypes.items():
                        sim = np.dot(emb, proto_emb) / (np.linalg.norm(emb) * np.linalg.norm(proto_emb))
                        similarities[lbl] = sim
                    top_3 = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    f.write(f"{i}. Caption: \"{captions[idx]}\"\n")
                    f.write(f"   - Similarity: {similarity:.4f}\n")
                    f.write(f"   - Top-3: {', '.join([f'{l} ({s:.3f})' for l, s in top_3])}\n\n")
        
        # Incorrect classifications
        if incorrect_indices:
            f.write("\n### ❌ Incorrect Classifications\n\n")
            
            shown = 0
            for idx in incorrect_indices[:num_examples * 2]:
                true_label = true_labels[idx]
                pred_label = pred_labels[idx]
                
                emb = embeddings[idx]
                true_proto = prototypes.get(true_label)
                pred_proto = prototypes.get(pred_label)
                
                if true_proto is not None and pred_proto is not None:
                    true_sim = np.dot(emb, true_proto) / (np.linalg.norm(emb) * np.linalg.norm(true_proto))
                    pred_sim = np.dot(emb, pred_proto) / (np.linalg.norm(emb) * np.linalg.norm(pred_proto))
                    
                    f.write(f"{shown + 1}. **True**: {true_label} | **Predicted**: {pred_label}\n")
                    f.write(f"   - Caption: \"{captions[idx]}\"\n")
                    f.write(f"   - True similarity: {true_sim:.4f} | Predicted similarity: {pred_sim:.4f} | Δ: {pred_sim - true_sim:.4f}\n\n")
                    shown += 1
                
                if shown >= num_examples * 2:
                    break

    def create_confusion_matrix_plot(self, confusion_matrix: np.ndarray,
                                   labels: List[str],
                                   title: str,
                                   save_path: str = None) -> plt.Figure:
        """Create confusion matrix visualization."""

        # Limit to top classes for readability
        if len(labels) > 15:
            print(f"⚠️  Too many classes ({len(labels)}), showing top 15 by frequency")
            labels = labels[:15]
            confusion_matrix = confusion_matrix[:15, :15]

        fig, ax = plt.subplots(figsize=(12, 10))

        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-8)

        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Normalized Count'}
        )

        ax.set_title(f'Confusion Matrix - {title}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Confusion matrix saved: {save_path}")

        return fig

    def create_f1_scores_plot(self, metrics_l1: Dict[str, Any], metrics_l2: Dict[str, Any],
                             title: str = "", save_path: str = None) -> plt.Figure:
        """Create F1 scores visualization."""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'F1 Score Analysis - {title}', fontsize=16, fontweight='bold')

        # Colors
        l1_color = '#1976D2'  # Blue
        l2_color = '#FF6F00'  # Orange

        # 1. Overall Performance Metrics (Top Left)
        overall_metrics = ['f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'accuracy']
        l1_scores = [metrics_l1.get(metric, 0) for metric in overall_metrics]
        l2_scores = [metrics_l2.get(metric, 0) for metric in overall_metrics]

        x = np.arange(len(overall_metrics))
        width = 0.35

        bars1 = ax1.bar(x - width/2, l1_scores, width, label='L1 (Primary)',
                       color=l1_color, alpha=0.8)
        bars2 = ax1.bar(x + width/2, l2_scores, width, label='L2 (Secondary)',
                       color=l2_color, alpha=0.8)

        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Overall Performance Metrics')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['F1 Macro', 'F1 Weighted', 'Precision', 'Recall', 'Accuracy'], rotation=15)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.02:
                    ax1.annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        # 2. L1 Per-Class F1 Scores (Top Right)
        if 'per_class_f1' in metrics_l1 and metrics_l1['per_class_f1']:
            l1_classes = list(metrics_l1['per_class_f1'].keys())
            l1_f1_scores = list(metrics_l1['per_class_f1'].values())

            # Sort by F1 score for better visualization
            sorted_data = sorted(zip(l1_classes, l1_f1_scores), key=lambda x: x[1], reverse=True)
            l1_classes_sorted = [x[0] for x in sorted_data]
            l1_f1_scores_sorted = [x[1] for x in sorted_data]

            # Limit to top 10 for readability
            if len(l1_classes_sorted) > 10:
                l1_classes_sorted = l1_classes_sorted[:10]
                l1_f1_scores_sorted = l1_f1_scores_sorted[:10]

            bars = ax2.barh(range(len(l1_classes_sorted)), l1_f1_scores_sorted,
                           color=l1_color, alpha=0.7)
            ax2.set_yticks(range(len(l1_classes_sorted)))
            ax2.set_yticklabels([cls.replace('_', ' ') for cls in l1_classes_sorted])
            ax2.set_xlabel('F1 Score')
            ax2.set_title(f'L1 Per-Class F1 Scores (Top {len(l1_classes_sorted)})')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 1)

            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax2.annotate(f'{width:.3f}',
                            xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(3, 0),
                            textcoords="offset points",
                            ha='left', va='center', fontsize=9)
        else:
            ax2.text(0.5, 0.5, 'No L1 per-class data available',
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('L1 Per-Class F1 Scores')

        # 3. L2 Per-Class F1 Scores (Bottom Left)
        if 'per_class_f1' in metrics_l2 and metrics_l2['per_class_f1']:
            l2_classes = list(metrics_l2['per_class_f1'].keys())
            l2_f1_scores = list(metrics_l2['per_class_f1'].values())

            # Sort by F1 score for better visualization
            sorted_data = sorted(zip(l2_classes, l2_f1_scores), key=lambda x: x[1], reverse=True)
            l2_classes_sorted = [x[0] for x in sorted_data]
            l2_f1_scores_sorted = [x[1] for x in sorted_data]

            # Limit to top 10 for readability
            if len(l2_classes_sorted) > 10:
                l2_classes_sorted = l2_classes_sorted[:10]
                l2_f1_scores_sorted = l2_f1_scores_sorted[:10]

            bars = ax3.barh(range(len(l2_classes_sorted)), l2_f1_scores_sorted,
                           color=l2_color, alpha=0.7)
            ax3.set_yticks(range(len(l2_classes_sorted)))
            ax3.set_yticklabels([cls.replace('_', ' ') for cls in l2_classes_sorted])
            ax3.set_xlabel('F1 Score')
            ax3.set_title(f'L2 Per-Class F1 Scores (Top {len(l2_classes_sorted)})')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 1)

            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax3.annotate(f'{width:.3f}',
                            xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(3, 0),
                            textcoords="offset points",
                            ha='left', va='center', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'No L2 per-class data available',
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('L2 Per-Class F1 Scores')

        # 4. Sample distribution (Bottom Right)
        if metrics_l1.get('num_samples') and metrics_l2.get('num_samples'):
            sample_data = ['L1 Samples', 'L2 Samples', 'L1 Classes', 'L2 Classes']
            sample_counts = [
                metrics_l1.get('num_samples', 0),
                metrics_l2.get('num_samples', 0),
                metrics_l1.get('num_classes', 0),
                metrics_l2.get('num_classes', 0)
            ]

            colors = [l1_color, l2_color, l1_color, l2_color]

            bars = ax4.bar(sample_data, sample_counts, color=colors, alpha=0.7)
            ax4.set_ylabel('Count')
            ax4.set_title('Dataset Statistics')
            ax4.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'No sample statistics available',
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Dataset Statistics')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 F1 scores plot saved: {save_path}")

        return fig

    def run_evaluation(self, max_samples: int = 10000,
                      k_neighbors: int = 1,
                      filter_noisy_labels: bool = True,
                      save_results: bool = True,
                      description_style: str = "baseline",
                      verbose: bool = False) -> Dict[str, Any]:
        """Run complete text encoder only evaluation pipeline."""

        print("🚀 Starting text encoder only evaluation...")
        print(f"   Max samples: {max_samples}")
        print(f"   K-neighbors: {k_neighbors}")
        print(f"   Verbose mode: {verbose}")
        print(f"   Filter noisy labels: {filter_noisy_labels}")

        results = {}

        # 1. Extract training labels to create prototypes
        print("\n" + "="*60)
        print("1. EXTRACTING TRAINING CAPTIONS AND LABELS")
        print("="*60)

        train_captions, train_labels_l1, train_labels_l2 = self.extract_captions_and_labels(
            self.train_data, max_samples, "train"
        )

        # Filter out noisy labels from training data (if enabled)
        if filter_noisy_labels:
            train_captions, train_labels_l1, train_labels_l2 = self.filter_noisy_labels(
                train_captions, train_labels_l1, train_labels_l2
            )

        # 2. Create text-based label prototypes
        print("\n" + "="*60)
        print("2. CREATING TEXT-BASED LABEL PROTOTYPES")
        print("="*60)

        prototypes_l1, counts_l1 = self.create_text_prototypes(train_labels_l1, description_style)
        prototypes_l2, counts_l2 = self.create_text_prototypes(train_labels_l2, description_style)

        results['prototypes_l1'] = prototypes_l1
        results['prototypes_l2'] = prototypes_l2
        results['prototype_counts_l1'] = counts_l1
        results['prototype_counts_l2'] = counts_l2

        # 3. Extract test captions and labels
        print("\n" + "="*60)
        print("3. EXTRACTING TEST CAPTIONS AND LABELS")
        print("="*60)

        test_captions, test_labels_l1, test_labels_l2 = self.extract_captions_and_labels(
            self.test_data, max_samples, "test"
        )

        # Filter out noisy labels from test data (if enabled)
        if filter_noisy_labels:
            test_captions, test_labels_l1, test_labels_l2 = self.filter_noisy_labels(
                test_captions, test_labels_l1, test_labels_l2
            )

        # 4. Embed test captions
        print("\n" + "="*60)
        print("4. EMBEDDING TEST CAPTIONS")
        print("="*60)

        test_embeddings = self.embed_captions(test_captions)

        # 5. Predict labels using nearest neighbors
        print("\n" + "="*60)
        print("5. PREDICTING LABELS")
        print("="*60)

        pred_labels_l1 = self.predict_labels_knn(test_embeddings, prototypes_l1, k_neighbors)
        pred_labels_l2 = self.predict_labels_knn(test_embeddings, prototypes_l2, k_neighbors)

        # 6. Evaluate predictions
        print("\n" + "="*60)
        print("6. EVALUATING PREDICTIONS")
        print("="*60)

        metrics_l1 = self.evaluate_predictions(test_labels_l1, pred_labels_l1, "L1 (Primary)", verbose=verbose)
        metrics_l2 = self.evaluate_predictions(test_labels_l2, pred_labels_l2, "L2 (Secondary)", verbose=verbose)

        results['metrics_l1'] = metrics_l1
        results['metrics_l2'] = metrics_l2
        results['predictions_l1'] = pred_labels_l1
        results['predictions_l2'] = pred_labels_l2
        results['ground_truth_l1'] = test_labels_l1
        results['ground_truth_l2'] = test_labels_l2

        # Show classification examples if verbose
        if verbose and test_embeddings is not None:
            self.show_classification_examples(
                test_captions, test_labels_l1, pred_labels_l1,
                test_embeddings, prototypes_l1, "L1 (Primary)", num_examples=3
            )
            self.show_classification_examples(
                test_captions, test_labels_l2, pred_labels_l2,
                test_embeddings, prototypes_l2, "L2 (Secondary)", num_examples=3
            )

        # 7. Save verbose report if enabled
        if verbose and save_results:
            output_dir = Path(self.config.get('output_dir', './results/evals/milan/50_textonly'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            verbose_report_path = output_dir / 'verbose_evaluation_report.md'
            self._save_verbose_report(
                verbose_report_path,
                metrics_l1, metrics_l2,
                test_captions, test_labels_l1, test_labels_l2,
                pred_labels_l1, pred_labels_l2,
                test_embeddings, prototypes_l1, prototypes_l2,
                description_style
            )
            print(f"\n💾 Verbose report saved: {verbose_report_path}")

        # 8. Create visualizations
        if save_results and metrics_l1:
            print("\n" + "="*60)
            print("8. CREATING VISUALIZATIONS")
            print("="*60)

            output_dir = Path(self.config.get('output_dir', './results/evals/milan/50_textonly'))
            output_dir.mkdir(parents=True, exist_ok=True)

            # Confusion matrices
            if 'confusion_matrix' in metrics_l1:
                self.create_confusion_matrix_plot(
                    metrics_l1['confusion_matrix'],
                    metrics_l1['unique_labels'],
                    'L1 Primary Activities',
                    str(output_dir / f'confusion_matrix_l1.png')
                )

            if 'confusion_matrix' in metrics_l2:
                self.create_confusion_matrix_plot(
                    metrics_l2['confusion_matrix'],
                    metrics_l2['unique_labels'],
                    'L2 Secondary Activities',
                    str(output_dir / f'confusion_matrix_l2.png')
                )

            # F1 Scores comprehensive plot
            self.create_f1_scores_plot(
                metrics_l1=metrics_l1,
                metrics_l2=metrics_l2,
                title=f'Text Encoder Only Evaluation ({max_samples} samples)',
                save_path=str(output_dir / f'f1_scores_analysis.png')
            )

            # Save detailed results
            results_summary = {
                'config': self.config,
                'evaluation_params': {
                    'max_samples': max_samples,
                    'k_neighbors': k_neighbors,
                    'filter_noisy_labels': filter_noisy_labels
                },
                'metrics_summary': {
                    'l1_f1_macro': metrics_l1.get('f1_macro', 0),
                    'l1_f1_weighted': metrics_l1.get('f1_weighted', 0),
                    'l1_accuracy': metrics_l1.get('accuracy', 0),
                    'l1_classes': metrics_l1.get('num_classes', 0),
                    'l2_f1_macro': metrics_l2.get('f1_macro', 0),
                    'l2_f1_weighted': metrics_l2.get('f1_weighted', 0),
                    'l2_accuracy': metrics_l2.get('accuracy', 0),
                    'l2_classes': metrics_l2.get('num_classes', 0),
                },
                'detailed_metrics': {
                    'l1_per_class_f1': metrics_l1.get('per_class_f1', {}),
                    'l1_classification_report': metrics_l1.get('classification_report', {}),
                    'l1_unique_labels': metrics_l1.get('unique_labels', []),
                    'l2_per_class_f1': metrics_l2.get('per_class_f1', {}),
                    'l2_classification_report': metrics_l2.get('classification_report', {}),
                    'l2_unique_labels': metrics_l2.get('unique_labels', []),
                },
                'prototype_info': {
                    'l1_prototype_counts': results.get('prototype_counts_l1', {}),
                    'l2_prototype_counts': results.get('prototype_counts_l2', {}),
                }
            }

            # Save results
            results_file = output_dir / f'text_only_evaluation_results.json'
            with open(results_file, 'w') as f:
                json.dump(results_summary, f, indent=2, default=str)

            print(f"💾 Results saved: {results_file}")

        # 8. Print final summary
        print("\n" + "="*60)
        print("🎯 TEXT ENCODER ONLY EVALUATION SUMMARY")
        print("="*60)

        if metrics_l1:
            print(f"📊 L1 (Primary Activities):")
            print(f"    F1-Score (Macro):    {metrics_l1['f1_macro']:.4f}")
            print(f"    F1-Score (Weighted): {metrics_l1['f1_weighted']:.4f}")
            print(f"    Accuracy:            {metrics_l1['accuracy']:.4f}")
            print(f"    Classes:             {metrics_l1['num_classes']}")
            print(f"    Test Samples:        {metrics_l1['num_samples']}")

        if metrics_l2:
            print(f"📊 L2 (Secondary Activities):")
            print(f"    F1-Score (Macro):    {metrics_l2['f1_macro']:.4f}")
            print(f"    F1-Score (Weighted): {metrics_l2['f1_weighted']:.4f}")
            print(f"    Accuracy:            {metrics_l2['accuracy']:.4f}")
            print(f"    Classes:             {metrics_l2['num_classes']}")
            print(f"    Test Samples:        {metrics_l2['num_samples']}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate text encoder only activity recognition',
        epilog='''
Example usage:
  python src/evals/evaluate_text_encoder_only.py \\
    --train_data data/processed/casas/milan/training_50/train.json \\
    --test_data data/processed/casas/milan/training_50/test.json \\
    --output_dir results/evals/milan/50_textonly \\
    --max_samples 1000 \\
    --filter_noisy_labels
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data paths (required arguments)
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data (e.g., data/processed/casas/milan/training_50/train.json)')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data (e.g., data/processed/casas/milan/training_50/test.json)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results (e.g., results/evals/milan/50_textonly)')

    # Model parameters
    parser.add_argument('--text_model_name', type=str, default='thenlper/gte-base',
                       help='Text encoder model name (default: thenlper/gte-base)')

    # Evaluation parameters
    parser.add_argument('--max_samples', type=int, default=10000,
                       help='Maximum number of samples to evaluate (default: 10000)')
    parser.add_argument('--k_neighbors', type=int, default=1,
                       help='Number of neighbors for k-NN prediction (default: 1)')
    parser.add_argument('--filter_noisy_labels', action='store_true',
                       help='Filter out noisy labels like "Other" and "No_Activity"')
    parser.add_argument('--description_style', type=str, default='baseline', choices=['baseline', 'sourish'],
                       help='Style of label descriptions to use: "baseline" or "sourish" (default: baseline)')
    parser.add_argument('--house_name', type=str, default='milan',
                       help='House name for label descriptions (default: milan)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output with per-label stats and classification examples')

    args = parser.parse_args()

    # Configuration
    config = {
        'train_data_path': args.train_data,
        'test_data_path': args.test_data,
        'text_model_name': args.text_model_name,
        'output_dir': args.output_dir,
        'house_name': args.house_name,
    }

    # Run evaluation
    evaluator = TextEncoderOnlyEvaluator(config)
    results = evaluator.run_evaluation(
        max_samples=args.max_samples,
        k_neighbors=args.k_neighbors,
        filter_noisy_labels=args.filter_noisy_labels,
        save_results=True,
        description_style=args.description_style,
        verbose=args.verbose
    )

    print(f"\n✅ Text encoder only evaluation complete! Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
