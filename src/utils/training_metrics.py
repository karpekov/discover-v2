#!/usr/bin/env python3
"""
Training metrics for monitoring alignment health, MLM accuracy, and representation quality.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
import logging

# Import evaluation functions from evaluate_embeddings
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report
)

logger = logging.getLogger(__name__)

# Import FAISS for efficient nearest neighbor search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, falling back to manual matrix multiplication")


class TrainingMetrics:
    """
    Comprehensive training metrics tracker for smart-home sequence alignment.
    """

    def __init__(self, vocab_sizes: Dict[str, int], sample_size: int = 1000, text_encoder=None):
        """
        Initialize metrics tracker.

        Args:
            vocab_sizes: Vocabulary sizes for each field
            sample_size: Number of samples to use for expensive metrics
            text_encoder: Text encoder for creating label prototypes
        """
        self.vocab_sizes = vocab_sizes
        self.sample_size = sample_size
        self.field_names = list(vocab_sizes.keys())
        self.text_encoder = text_encoder

    def compute_alignment_health(
        self,
        sensor_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        captions: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute alignment health metrics: positive vs negative cosine similarities.

        Args:
            sensor_embeddings: [batch_size, embed_dim]
            text_embeddings: [batch_size, embed_dim]
            captions: Optional list of caption strings for label analysis

        Returns:
            Dictionary of alignment metrics
        """
        batch_size = sensor_embeddings.size(0)

        # Normalize embeddings
        sensor_norm = F.normalize(sensor_embeddings, p=2, dim=1)
        text_norm = F.normalize(text_embeddings, p=2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.mm(sensor_norm, text_norm.t())  # [batch_size, batch_size]

        # Positive pairs (diagonal)
        pos_sims = torch.diag(sim_matrix)

        # Negative pairs (off-diagonal)
        mask = torch.eye(batch_size, device=sim_matrix.device).bool()
        neg_sims = sim_matrix.masked_select(~mask)

        metrics = {
            'alignment/pos_cos_mean': pos_sims.mean().item(),
            'alignment/pos_cos_median': pos_sims.median().item(),
            'alignment/pos_cos_std': pos_sims.std().item(),
            'alignment/neg_cos_mean': neg_sims.mean().item(),
            'alignment/neg_cos_median': neg_sims.median().item(),
            'alignment/neg_cos_std': neg_sims.std().item(),
            'alignment/pos_neg_gap': (pos_sims.mean() - neg_sims.mean()).item(),
        }

        # Activity-specific alignment if captions are provided
        if captions:
            activity_metrics = self._compute_activity_specific_alignment(
                sim_matrix, pos_sims, neg_sims, captions
            )
            metrics.update(activity_metrics)

        return metrics

    def _compute_activity_specific_alignment(
        self,
        sim_matrix: torch.Tensor,
        pos_sims: torch.Tensor,
        neg_sims: torch.Tensor,
        captions: List[str]
    ) -> Dict[str, float]:
        """Compute alignment metrics split by activity labels."""
        batch_size = sim_matrix.size(0)

        # Extract activity labels from captions
        activity_labels = self._extract_activity_labels(captions)

        # Ensure we don't have more labels than batch size
        if len(activity_labels) > batch_size:
            activity_labels = activity_labels[:batch_size]

        # Group by activity
        activity_groups = defaultdict(list)
        for i, label in enumerate(activity_labels):
            if i < batch_size and label != 'other':  # Ensure valid index and skip generic labels
                activity_groups[label].append(i)

        metrics = {}
        for activity, indices in activity_groups.items():
            if len(indices) < 2:  # Need at least 2 samples
                continue

            # Filter indices to ensure they're within bounds
            valid_indices = [i for i in indices if i < batch_size]
            if len(valid_indices) < 2:
                continue

            activity_pos_sims = pos_sims[valid_indices]

            # For negatives, use all other samples in batch
            activity_neg_indices = []
            for i in valid_indices:
                for j in range(batch_size):
                    if j != i:  # Exclude self
                        activity_neg_indices.append((i, j))

            if activity_neg_indices:
                activity_neg_sims = torch.tensor([
                    sim_matrix[i, j].item() for i, j in activity_neg_indices
                ], device=sim_matrix.device)

                metrics[f'alignment/{activity}_pos_cos_mean'] = activity_pos_sims.mean().item()
                metrics[f'alignment/{activity}_neg_cos_mean'] = activity_neg_sims.mean().item()
                metrics[f'alignment/{activity}_pos_neg_gap'] = (
                    activity_pos_sims.mean() - activity_neg_sims.mean()
                ).item()

        return metrics

    def _extract_activity_labels(self, captions: List[str]) -> List[str]:
        """Extract activity labels from captions."""
        # Common activity keywords for CASAS datasets
        activity_keywords = {
            'cooking': ['cook', 'kitchen', 'meal', 'food', 'eat', 'prepare'],
            'relaxing': ['relax', 'rest', 'sit', 'watch', 'tv', 'living'],
            'sleeping': ['sleep', 'bed', 'bedroom', 'night'],
            'hygiene': ['shower', 'bath', 'bathroom', 'wash', 'toilet'],
            'work': ['work', 'computer', 'desk', 'office', 'study'],
            'cleaning': ['clean', 'laundry', 'vacuum', 'tidy'],
        }

        labels = []
        for caption in captions:
            caption_lower = caption.lower()
            found_activity = 'other'

            for activity, keywords in activity_keywords.items():
                if any(keyword in caption_lower for keyword in keywords):
                    found_activity = activity
                    break

            labels.append(found_activity)

        return labels

    def _convert_labels_to_text(self, labels: List[str]) -> List[str]:
        """Convert CASAS activity labels to natural language descriptions."""
        # Mapping from CASAS labels to natural language (same as evaluate_embeddings.py)
        label_to_text = {
            'Kitchen_Activity': 'cooking and kitchen activities',
            'Sleep': 'sleeping in bed in master bedroom, usually at night time',
            'Read': 'reading a book or newspaper in a static position',
            'Watch_TV': 'watching television',
            'Master_Bedroom_Activity': 'activities and motion in the master bedroom',
            'Master_Bathroom': 'using the master bathroom, shower, or bathtub',
            'Guest_Bathroom': 'using the guest bathroom, shower, or bathtub',
            'Dining_Rm_Activity': 'activity and motion in dining room',
            'Desk_Activity': 'working at desk in workspace',
            'Leave_Home': 'leaving or entering the house through the door',
            'Chores': 'doing household chores',
            'Meditate': 'meditation',
            'no_activity': 'no specific activity or idle time',
            'Bed_to_Toilet': 'going from bed to toilet at night',
            'Morning_Meds': 'taking morning medication',
            'Eve_Meds': 'taking evening medication',

            # L2 labels
            'Cook': 'cooking and meal preparation, usually in the kitchen next to the stove or fridge',
            'No_Activity': 'no specific activity or idle time',
            'Work': 'working or studying at a desk',
            'Eat': 'eating, usually at the dining room table or in the living room area',
            'Relax': 'relaxing and leisure activities like watching TV, reading, or meditating',
            'Bathing': 'using bathroom, shower, or bathtub',
            'Other': 'other miscellaneous activities',
            'Bed_to_toilet': 'going from bed to using bathroom, shower, or bathtub at night time',
            'Take_medicine': 'taking medication from the medicine cabinet',
            'Leave_Home': 'leaving or entering the house through the door',
        }

        descriptions = []
        for label in labels:
            if label in label_to_text:
                descriptions.append(label_to_text[label])
            else:
                # Fallback: convert label to readable text
                readable = label.replace('_', ' ').lower()
                descriptions.append(f"{readable} activity")

        return descriptions

    def compute_mlm_accuracy(
        self,
        mlm_predictions: Dict[str, torch.Tensor],
        mlm_labels: Dict[str, torch.Tensor],
        mlm_mask_positions: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute MLM accuracy per field.

        Args:
            mlm_predictions: Predictions for each field [batch_size, seq_len, vocab_size]
            mlm_labels: Ground truth labels [batch_size, seq_len]
            mlm_mask_positions: Mask positions [batch_size, seq_len]

        Returns:
            Dictionary of MLM accuracy metrics
        """
        metrics = {}

        for field in self.field_names:
            if field not in mlm_predictions or field not in mlm_labels:
                continue

            predictions = mlm_predictions[field]  # [batch_size, seq_len, vocab_size]
            labels = mlm_labels[field]  # [batch_size, seq_len]
            mask_pos = mlm_mask_positions.get(field)

            if mask_pos is None or mask_pos.sum() == 0:
                continue

            # Get predictions and labels for masked positions only
            masked_predictions = predictions[mask_pos]  # [num_masked, vocab_size]
            masked_labels = labels[mask_pos]  # [num_masked]

            # Compute accuracy
            pred_classes = masked_predictions.argmax(dim=-1)
            correct = (pred_classes == masked_labels).float()
            accuracy = correct.mean().item()

            metrics[f'mlm_accuracy/{field}'] = accuracy

            # Also compute top-k accuracy for larger vocabularies
            if masked_predictions.size(-1) > 5:
                top5_correct = torch.any(
                    masked_predictions.topk(5, dim=-1)[1] == masked_labels.unsqueeze(-1),
                    dim=-1
                ).float()
                metrics[f'mlm_accuracy/{field}_top5'] = top5_correct.mean().item()

        # Overall MLM accuracy (macro average)
        field_accuracies = [v for k, v in metrics.items() if k.endswith('_top5') == False]
        if field_accuracies:
            metrics['mlm_accuracy/overall'] = np.mean(field_accuracies)

        return metrics

    def compute_representation_diagnostics(
        self,
        sensor_embeddings_raw: torch.Tensor,
        sensor_embeddings_proj: torch.Tensor,
        text_embeddings_raw: torch.Tensor,
        text_embeddings_proj: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute representation diagnostics: norms, means, stds.

        Args:
            sensor_embeddings_raw: Raw sensor embeddings before projection
            sensor_embeddings_proj: Projected sensor embeddings
            text_embeddings_raw: Raw text embeddings before projection
            text_embeddings_proj: Projected text embeddings

        Returns:
            Dictionary of representation diagnostics
        """
        metrics = {}

        # Sensor embeddings diagnostics
        sensor_raw_norms = torch.norm(sensor_embeddings_raw, p=2, dim=1)
        sensor_proj_norms = torch.norm(sensor_embeddings_proj, p=2, dim=1)

        metrics.update({
            'repr/sensor_raw_norm_mean': sensor_raw_norms.mean().item(),
            'repr/sensor_raw_norm_std': sensor_raw_norms.std().item(),
            'repr/sensor_proj_norm_mean': sensor_proj_norms.mean().item(),
            'repr/sensor_proj_norm_std': sensor_proj_norms.std().item(),
            'repr/sensor_raw_mean': sensor_embeddings_raw.mean().item(),
            'repr/sensor_raw_std': sensor_embeddings_raw.std().item(),
            'repr/sensor_proj_mean': sensor_embeddings_proj.mean().item(),
            'repr/sensor_proj_std': sensor_embeddings_proj.std().item(),
        })

        # Text embeddings diagnostics
        text_raw_norms = torch.norm(text_embeddings_raw, p=2, dim=1)
        text_proj_norms = torch.norm(text_embeddings_proj, p=2, dim=1)

        metrics.update({
            'repr/text_raw_norm_mean': text_raw_norms.mean().item(),
            'repr/text_raw_norm_std': text_raw_norms.std().item(),
            'repr/text_proj_norm_mean': text_proj_norms.mean().item(),
            'repr/text_proj_norm_std': text_proj_norms.std().item(),
            'repr/text_raw_mean': text_embeddings_raw.mean().item(),
            'repr/text_raw_std': text_embeddings_raw.std().item(),
            'repr/text_proj_mean': text_embeddings_proj.mean().item(),
            'repr/text_proj_std': text_embeddings_proj.std().item(),
        })

        return metrics

    def compute_f1_scores(
        self,
        sensor_embeddings: torch.Tensor,
        ground_truth_labels: List[str],
        exclude_labels: Optional[List[str]] = None,
        label_suffix: str = ""
    ) -> Dict[str, float]:
        """
        Compute F1 scores using ground truth activity labels and text prototypes.
        Uses the same approach as evaluate_embeddings.py.

        Args:
            sensor_embeddings: Sensor embeddings [batch_size, embed_dim]
            ground_truth_labels: Ground truth activity labels
            exclude_labels: Labels to exclude (e.g., 'other', 'no_activity')
            label_suffix: Suffix to add to metric names (e.g., "_l1", "_l2")

        Returns:
            Dictionary of F1 scores
        """
        if self.text_encoder is None:
            return {f'f1{label_suffix}/no_text_encoder': 0.0}

        if exclude_labels is None:
            exclude_labels = ['other', 'no_activity', 'unknown', '']

        # Filter out excluded labels
        valid_indices = []
        valid_labels = []
        for i, label in enumerate(ground_truth_labels):
            if label.lower() not in [ex.lower() for ex in exclude_labels]:
                valid_indices.append(i)
                valid_labels.append(label)

        if len(valid_indices) < 2:
            return {f'f1{label_suffix}/insufficient_data': 0.0}

        try:
            # Get embeddings for valid samples
            valid_embeddings = sensor_embeddings[valid_indices]

            # Create text-based prototypes for unique labels
            unique_labels = sorted(list(set(valid_labels)))
            label_descriptions = self._convert_labels_to_text(unique_labels)

            # Encode labels with text encoder (CLIP projected)
            device = sensor_embeddings.device
            with torch.no_grad():
                text_embeddings = self.text_encoder.encode_texts_clip(label_descriptions, device)

            # Create prototypes dictionary
            prototypes = {}
            for i, label in enumerate(unique_labels):
                prototypes[label] = text_embeddings[i].cpu().numpy()

            # Predict labels using nearest neighbors (same as evaluate_embeddings.py)
            predicted_labels = self._predict_labels_knn(valid_embeddings.cpu().numpy(), prototypes)

            # Filter out unknown labels for fair evaluation
            final_valid_indices = [i for i, (true, pred) in enumerate(zip(valid_labels, predicted_labels))
                                 if true != 'Unknown' and pred != 'Unknown']

            if not final_valid_indices:
                return {f'f1{label_suffix}/no_valid_predictions': 0.0}

            true_filtered = [valid_labels[i] for i in final_valid_indices]
            pred_filtered = [predicted_labels[i] for i in final_valid_indices]

            # Get unique labels for consistent labeling
            all_unique_labels = sorted(list(set(true_filtered + pred_filtered)))

            # Compute metrics
            metrics = {
                f'f1{label_suffix}/accuracy': accuracy_score(true_filtered, pred_filtered),
                f'f1{label_suffix}/f1_macro': f1_score(true_filtered, pred_filtered, average='macro', zero_division=0),
                f'f1{label_suffix}/f1_micro': f1_score(true_filtered, pred_filtered, average='micro', zero_division=0),
                f'f1{label_suffix}/f1_weighted': f1_score(true_filtered, pred_filtered, average='weighted', zero_division=0),
                f'f1{label_suffix}/precision_macro': precision_score(true_filtered, pred_filtered, average='macro', zero_division=0),
                f'f1{label_suffix}/recall_macro': recall_score(true_filtered, pred_filtered, average='macro', zero_division=0),
                f'f1{label_suffix}/num_samples': len(true_filtered),
                f'f1{label_suffix}/num_classes': len(all_unique_labels)
            }

            return metrics

        except Exception as e:
            logger.warning(f"Error computing F1 scores: {e}")
            return {f'f1{label_suffix}/error': 0.0}

    def _predict_labels_knn(self, query_embeddings: np.ndarray, prototypes: Dict[str, np.ndarray], k: int = 1) -> List[str]:
        """Predict labels using k-nearest neighbors between sensor and text embeddings."""
        # Convert prototypes to arrays
        prototype_labels = list(prototypes.keys())
        prototype_embeddings = np.array([prototypes[label] for label in prototype_labels])

        # Normalize embeddings for cosine similarity
        query_embeddings_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        prototype_embeddings_norm = prototype_embeddings / (np.linalg.norm(prototype_embeddings, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarities
        similarities = np.dot(query_embeddings_norm, prototype_embeddings_norm.T)

        # Simple nearest neighbor
        nearest_indices = np.argmax(similarities, axis=1)
        predictions = [prototype_labels[idx] for idx in nearest_indices]

        return predictions

    def compute_recall_at_k(
        self,
        sensor_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        k_values: List[int] = [1, 5, 10],
        ground_truth_labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute Recall@K for both sensor-to-text and text-to-sensor retrieval.

        Args:
            sensor_embeddings: [batch_size, embed_dim] L2-normalized sensor embeddings
            text_embeddings: [batch_size, embed_dim] L2-normalized text embeddings
            k_values: List of K values to compute recall for
            ground_truth_labels: List of ground truth activity labels for per-class metrics

        Returns:
            Dictionary of Recall@K metrics (overall and per-class)
        """
        batch_size = sensor_embeddings.size(0)
        device = sensor_embeddings.device

        # Ensure embeddings are L2-normalized
        sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        metrics = {}

        # Use FAISS for efficient nearest neighbor search when available and beneficial
        # Disable FAISS on MPS due to compatibility issues causing segmentation faults
        use_faiss = (FAISS_AVAILABLE and batch_size > 50 and
                    sensor_embeddings.device.type != 'mps')

        if use_faiss:
            # Convert to numpy for FAISS
            sensor_np = sensor_embeddings.cpu().numpy().astype(np.float32)
            text_np = text_embeddings.cpu().numpy().astype(np.float32)
            d_model = sensor_np.shape[1]

            # Build FAISS indices
            text_index = faiss.IndexFlatIP(d_model)  # Inner product for cosine similarity
            text_index.add(text_np)

            sensor_index = faiss.IndexFlatIP(d_model)
            sensor_index.add(sensor_np)

            # True labels (diagonal indices)
            true_labels_np = np.arange(batch_size)

            for k in k_values:
                k_actual = min(k, batch_size)

                # Sensor-to-text retrieval using FAISS
                _, s2t_top_k_indices = text_index.search(sensor_np, k_actual)  # [batch_size, k]

                # Check if true match is in top-k for each query
                s2t_hits = np.any(s2t_top_k_indices == true_labels_np.reshape(-1, 1), axis=1)
                s2t_recall_at_k = float(s2t_hits.mean())

                # Text-to-sensor retrieval using FAISS
                _, t2s_top_k_indices = sensor_index.search(text_np, k_actual)  # [batch_size, k]

                # Check if true match is in top-k for each query
                t2s_hits = np.any(t2s_top_k_indices == true_labels_np.reshape(-1, 1), axis=1)
                t2s_recall_at_k = float(t2s_hits.mean())

                # Convert hits back to torch for consistency with per-class computation
                s2t_hits = torch.from_numpy(s2t_hits).to(device)
                t2s_hits = torch.from_numpy(t2s_hits).to(device)

                # Average of both directions
                avg_recall_at_k = (s2t_recall_at_k + t2s_recall_at_k) / 2.0

                # Overall metrics
                metrics[f'recall@{k}/sensor_to_text'] = s2t_recall_at_k
                metrics[f'recall@{k}/text_to_sensor'] = t2s_recall_at_k
                metrics[f'recall@{k}/average'] = avg_recall_at_k

        else:
            # Fallback to manual computation for small batches or when FAISS is not available
            # Compute similarity matrix
            similarity_matrix = torch.matmul(sensor_embeddings, text_embeddings.t())  # [batch_size, batch_size]

            # True labels (diagonal indices)
            true_labels = torch.arange(batch_size, device=device)

            for k in k_values:
                # Sensor-to-text retrieval
                # For each sensor embedding, find top-k most similar text embeddings
                _, s2t_top_k_indices = torch.topk(similarity_matrix, k=min(k, batch_size), dim=1)  # [batch_size, k]

                # Check if true match is in top-k for each query
                s2t_hits = torch.any(s2t_top_k_indices == true_labels.unsqueeze(1), dim=1)  # [batch_size]
                s2t_recall_at_k = s2t_hits.float().mean().item()

                # Text-to-sensor retrieval
                # For each text embedding, find top-k most similar sensor embeddings
                _, t2s_top_k_indices = torch.topk(similarity_matrix.t(), k=min(k, batch_size), dim=1)  # [batch_size, k]

                # Check if true match is in top-k for each query
                t2s_hits = torch.any(t2s_top_k_indices == true_labels.unsqueeze(1), dim=1)  # [batch_size]
                t2s_recall_at_k = t2s_hits.float().mean().item()

                # Average of both directions
                avg_recall_at_k = (s2t_recall_at_k + t2s_recall_at_k) / 2.0

                # Overall metrics
                metrics[f'recall@{k}/sensor_to_text'] = s2t_recall_at_k
                metrics[f'recall@{k}/text_to_sensor'] = t2s_recall_at_k
                metrics[f'recall@{k}/average'] = avg_recall_at_k

        # Per-class metrics computation (common for both FAISS and manual approaches)
        if ground_truth_labels is not None and len(ground_truth_labels) == batch_size:
            for k in k_values:
                try:
                    # Group hits by class for sensor-to-text
                    class_s2t_hits = defaultdict(list)
                    class_t2s_hits = defaultdict(list)

                    for i, label in enumerate(ground_truth_labels):
                        if label and label != 'other':  # Skip empty/generic labels
                            class_s2t_hits[label].append(s2t_hits[i].item())
                            class_t2s_hits[label].append(t2s_hits[i].item())

                    # Compute per-class recall@k
                    for class_name, hits in class_s2t_hits.items():
                        if len(hits) >= 5:  # Only compute for classes with enough samples
                            class_recall_s2t = np.mean(hits)
                            class_recall_t2s = np.mean(class_t2s_hits[class_name])
                            class_recall_avg = (class_recall_s2t + class_recall_t2s) / 2.0

                            # Clean class name for metric keys (remove spaces, special chars)
                            clean_class = class_name.replace(' ', '_').replace('-', '_').lower()
                            metrics[f'recall@{k}/class_{clean_class}/sensor_to_text'] = class_recall_s2t
                            metrics[f'recall@{k}/class_{clean_class}/text_to_sensor'] = class_recall_t2s
                            metrics[f'recall@{k}/class_{clean_class}/average'] = class_recall_avg
                            metrics[f'recall@{k}/class_{clean_class}/num_samples'] = len(hits)

                except Exception as e:
                    logger.warning(f"Error computing per-class recall@{k}: {e}")

        return metrics

    def compute_ndcg_at_k(
        self,
        sensor_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        k_values: List[int] = [1, 5, 10],
        ground_truth_labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute nDCG@K for both sensor-to-text and text-to-sensor retrieval.

        Args:
            sensor_embeddings: [batch_size, embed_dim] L2-normalized sensor embeddings
            text_embeddings: [batch_size, embed_dim] L2-normalized text embeddings
            k_values: List of K values to compute nDCG for
            ground_truth_labels: List of ground truth activity labels for per-class metrics

        Returns:
            Dictionary of nDCG@K metrics (overall and per-class)
        """
        batch_size = sensor_embeddings.size(0)
        device = sensor_embeddings.device

        # Ensure embeddings are L2-normalized
        sensor_embeddings = F.normalize(sensor_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(sensor_embeddings, text_embeddings.t())  # [batch_size, batch_size]

        # True labels (diagonal indices)
        true_labels = torch.arange(batch_size, device=device)

        metrics = {}

        for k in k_values:
            k_actual = min(k, batch_size)

            # Sensor-to-text retrieval
            _, s2t_top_k_indices = torch.topk(similarity_matrix, k=k_actual, dim=1)  # [batch_size, k]

            # Compute DCG@K for each query
            s2t_dcg_scores = []
            for i in range(batch_size):
                dcg = 0.0
                for rank, retrieved_idx in enumerate(s2t_top_k_indices[i]):
                    # Relevance score: 1 if it's the true match, 0 otherwise
                    relevance = 1.0 if retrieved_idx == true_labels[i] else 0.0
                    # DCG formula: rel_i / log2(i + 2) (rank is 0-indexed, so +2)
                    dcg += relevance / np.log2(rank + 2)
                s2t_dcg_scores.append(dcg)

            # IDCG@K (Ideal DCG) - best possible DCG when true match is at rank 1
            idcg_at_k = 1.0 / np.log2(2)  # True match at rank 1

            # nDCG@K = DCG@K / IDCG@K
            s2t_ndcg_scores = [dcg / idcg_at_k if idcg_at_k > 0 else 0.0 for dcg in s2t_dcg_scores]
            s2t_ndcg_at_k = np.mean(s2t_ndcg_scores)

            # Text-to-sensor retrieval
            _, t2s_top_k_indices = torch.topk(similarity_matrix.t(), k=k_actual, dim=1)  # [batch_size, k]

            # Compute DCG@K for each query
            t2s_dcg_scores = []
            for i in range(batch_size):
                dcg = 0.0
                for rank, retrieved_idx in enumerate(t2s_top_k_indices[i]):
                    # Relevance score: 1 if it's the true match, 0 otherwise
                    relevance = 1.0 if retrieved_idx == true_labels[i] else 0.0
                    # DCG formula: rel_i / log2(i + 2)
                    dcg += relevance / np.log2(rank + 2)
                t2s_dcg_scores.append(dcg)

            # nDCG@K = DCG@K / IDCG@K
            t2s_ndcg_scores = [dcg / idcg_at_k if idcg_at_k > 0 else 0.0 for dcg in t2s_dcg_scores]
            t2s_ndcg_at_k = np.mean(t2s_ndcg_scores)

            # Average of both directions
            avg_ndcg_at_k = (s2t_ndcg_at_k + t2s_ndcg_at_k) / 2.0

            # Overall metrics
            metrics[f'ndcg@{k}/sensor_to_text'] = s2t_ndcg_at_k
            metrics[f'ndcg@{k}/text_to_sensor'] = t2s_ndcg_at_k
            metrics[f'ndcg@{k}/average'] = avg_ndcg_at_k

            # Per-class metrics if ground truth labels are provided
            if ground_truth_labels is not None and len(ground_truth_labels) == batch_size:
                try:
                    # Group NDCG scores by class
                    class_s2t_ndcg = defaultdict(list)
                    class_t2s_ndcg = defaultdict(list)

                    for i, label in enumerate(ground_truth_labels):
                        if label and label != 'other':  # Skip empty/generic labels
                            class_s2t_ndcg[label].append(s2t_ndcg_scores[i])
                            class_t2s_ndcg[label].append(t2s_ndcg_scores[i])

                    # Compute per-class nDCG@k
                    for class_name, scores in class_s2t_ndcg.items():
                        if len(scores) >= 5:  # Only compute for classes with enough samples
                            class_ndcg_s2t = np.mean(scores)
                            class_ndcg_t2s = np.mean(class_t2s_ndcg[class_name])
                            class_ndcg_avg = (class_ndcg_s2t + class_ndcg_t2s) / 2.0

                            # Clean class name for metric keys
                            clean_class = class_name.replace(' ', '_').replace('-', '_').lower()
                            metrics[f'ndcg@{k}/class_{clean_class}/sensor_to_text'] = class_ndcg_s2t
                            metrics[f'ndcg@{k}/class_{clean_class}/text_to_sensor'] = class_ndcg_t2s
                            metrics[f'ndcg@{k}/class_{clean_class}/average'] = class_ndcg_avg
                            metrics[f'ndcg@{k}/class_{clean_class}/num_samples'] = len(scores)

                except Exception as e:
                    logger.warning(f"Error computing per-class nDCG@{k}: {e}")

        return metrics

    def compute_all_metrics(
        self,
        batch: Dict[str, Any],
        model_outputs: Dict[str, Any],
        sensor_embeddings_raw: Optional[torch.Tensor] = None,
        text_embeddings_raw: Optional[torch.Tensor] = None,
        ground_truth_labels: Optional[List[str]] = None,
        ground_truth_labels_l2: Optional[List[str]] = None,
        captions: Optional[List[str]] = None,
        sample_for_expensive: bool = True
    ) -> Dict[str, float]:
        """
        Compute all training metrics.

        Args:
            batch: Input batch
            model_outputs: Model outputs
            sensor_embeddings_raw: Raw sensor embeddings (before projection)
            text_embeddings_raw: Raw text embeddings (before projection)
            ground_truth_labels: Ground truth activity labels
            captions: Caption strings
            sample_for_expensive: Whether to sample for expensive metrics

        Returns:
            Dictionary of all metrics
        """
        all_metrics = {}

        # Sample if needed for expensive metrics
        batch_size = len(model_outputs['sensor_embeddings'])
        if sample_for_expensive and batch_size > self.sample_size:
            indices = torch.randperm(batch_size)[:self.sample_size]

            # Sample embeddings
            sensor_emb_sample = model_outputs['sensor_embeddings'][indices]
            text_emb_sample = model_outputs['text_embeddings'][indices]

            # Sample other inputs if provided
            if sensor_embeddings_raw is not None:
                sensor_raw_sample = sensor_embeddings_raw[indices]
            else:
                sensor_raw_sample = None

            if text_embeddings_raw is not None:
                text_raw_sample = text_embeddings_raw[indices]
            else:
                text_raw_sample = None

            if ground_truth_labels is not None:
                gt_labels_sample = [ground_truth_labels[i.item()] for i in indices]
            else:
                gt_labels_sample = None

            if ground_truth_labels_l2 is not None:
                gt_labels_l2_sample = [ground_truth_labels_l2[i.item()] for i in indices]
            else:
                gt_labels_l2_sample = None

            if captions is not None:
                captions_sample = [captions[i.item()] for i in indices]
            else:
                captions_sample = None
        else:
            sensor_emb_sample = model_outputs['sensor_embeddings']
            text_emb_sample = model_outputs['text_embeddings']
            sensor_raw_sample = sensor_embeddings_raw
            text_raw_sample = text_embeddings_raw
            gt_labels_sample = ground_truth_labels
            gt_labels_l2_sample = ground_truth_labels_l2
            captions_sample = captions

        # 1. Alignment health
        try:
            alignment_metrics = self.compute_alignment_health(
                sensor_emb_sample, text_emb_sample, captions_sample
            )
            all_metrics.update(alignment_metrics)
        except Exception as e:
            logger.warning(f"Error computing alignment health: {e}")
            all_metrics.update({'alignment/error': 0.0})

        # 2. MLM accuracy
        try:
            if 'mlm_predictions' in model_outputs:
                mlm_metrics = self.compute_mlm_accuracy(
                    model_outputs['mlm_predictions'],
                    batch.get('mlm_labels', {}),
                    batch.get('mlm_mask_positions', {})
                )
                all_metrics.update(mlm_metrics)
        except Exception as e:
            logger.warning(f"Error computing MLM accuracy: {e}")
            all_metrics.update({'mlm_accuracy/error': 0.0})

        # 3. Representation diagnostics
        try:
            if sensor_raw_sample is not None and text_raw_sample is not None:
                repr_metrics = self.compute_representation_diagnostics(
                    sensor_raw_sample, sensor_emb_sample,
                    text_raw_sample, text_emb_sample
                )
                all_metrics.update(repr_metrics)
        except Exception as e:
            logger.warning(f"Error computing representation diagnostics: {e}")
            all_metrics.update({'repr/error': 0.0})

        # 4. F1 scores (L1 and L2, with and without filtering)
        try:
            if gt_labels_sample is not None:
                # L1 F1 scores with filtering (exclude no_activity, etc.)
                f1_l1_filtered = self.compute_f1_scores(
                    sensor_emb_sample, gt_labels_sample,
                    exclude_labels=['other', 'no_activity', 'unknown', ''],
                    label_suffix="_l1_filtered"
                )
                all_metrics.update(f1_l1_filtered)

                # L1 F1 scores without filtering (include all labels)
                f1_l1_all = self.compute_f1_scores(
                    sensor_emb_sample, gt_labels_sample,
                    exclude_labels=[],
                    label_suffix="_l1_all"
                )
                all_metrics.update(f1_l1_all)

            if gt_labels_l2_sample is not None:
                # L2 F1 scores with filtering
                f1_l2_filtered = self.compute_f1_scores(
                    sensor_emb_sample, gt_labels_l2_sample,
                    exclude_labels=['other', 'no_activity', 'unknown', ''],
                    label_suffix="_l2_filtered"
                )
                all_metrics.update(f1_l2_filtered)

                # L2 F1 scores without filtering
                f1_l2_all = self.compute_f1_scores(
                    sensor_emb_sample, gt_labels_l2_sample,
                    exclude_labels=[],
                    label_suffix="_l2_all"
                )
                all_metrics.update(f1_l2_all)
        except Exception as e:
            logger.warning(f"Error computing F1 scores: {e}")
            all_metrics.update({'f1/error': 0.0})

        # 5. Recall@K metrics
        try:
            recall_metrics = self.compute_recall_at_k(
                sensor_emb_sample, text_emb_sample, k_values=[1, 5, 10],
                ground_truth_labels=gt_labels_sample
            )
            all_metrics.update(recall_metrics)
        except Exception as e:
            logger.warning(f"Error computing Recall@K metrics: {e}")
            all_metrics.update({'recall/error': 0.0})

        # 6. nDCG@K metrics
        try:
            ndcg_metrics = self.compute_ndcg_at_k(
                sensor_emb_sample, text_emb_sample, k_values=[1, 5, 10],
                ground_truth_labels=gt_labels_sample
            )
            all_metrics.update(ndcg_metrics)
        except Exception as e:
            logger.warning(f"Error computing nDCG@K metrics: {e}")
            all_metrics.update({'ndcg/error': 0.0})

        return all_metrics
