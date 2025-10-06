"""
SCAN Dataset and DataLoader implementation for clustering training.
Includes KNN mining and neighbor-based sampling.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Any, Optional, Tuple
import logging

from .dataset import SmartHomeDataset


class SCANDataset(Dataset):
    """
    Dataset for SCAN clustering training that pairs each sample with its nearest neighbors.
    """

    def __init__(
        self,
        base_dataset: SmartHomeDataset,
        embeddings: np.ndarray,
        knn_indices: np.ndarray,
        num_neighbors: int = 20,
        labels: Optional[List[str]] = None
    ):
        """
        Args:
            base_dataset: Base SmartHomeDataset
            embeddings: Pre-computed embeddings [num_samples, embedding_dim]
            knn_indices: KNN indices [num_samples, num_neighbors]
            num_neighbors: Number of neighbors to consider
            labels: Optional ground truth labels for evaluation
        """
        self.base_dataset = base_dataset
        self.embeddings = embeddings
        self.knn_indices = knn_indices
        self.num_neighbors = num_neighbors
        self.labels = labels

        assert len(base_dataset) == len(embeddings), "Dataset and embeddings size mismatch"
        assert knn_indices.shape[0] == len(embeddings), "KNN indices size mismatch"
        assert knn_indices.shape[1] >= num_neighbors, "Not enough neighbors in KNN indices"

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Created SCAN dataset with {len(self)} samples and {num_neighbors} neighbors each")

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample with its anchor and neighbor.

        Returns:
            Dictionary containing anchor and neighbor samples
        """
        # Get anchor sample
        anchor_sample = self.base_dataset[idx]

        # Randomly select one of the k nearest neighbors
        neighbor_idx = np.random.choice(self.knn_indices[idx][:self.num_neighbors])
        neighbor_sample = self.base_dataset[neighbor_idx]

        return {
            'anchor': anchor_sample,
            'neighbor': neighbor_sample,
            'anchor_idx': idx,
            'neighbor_idx': neighbor_idx
        }


def mine_nearest_neighbors(
    embeddings: np.ndarray,
    topk: int = 20,
    labels: Optional[List[str]] = None,
    metric: str = 'cosine'
) -> Tuple[np.ndarray, float]:
    """
    Mine nearest neighbors for each sample using sklearn's NearestNeighbors.

    Args:
        embeddings: [num_samples, embedding_dim] embeddings
        topk: Number of neighbors to find
        labels: Optional ground truth labels for computing accuracy
        metric: Distance metric ('cosine', 'euclidean', etc.)

    Returns:
        knn_indices: [num_samples, topk] indices of nearest neighbors
        accuracy: Neighbor accuracy if labels are provided, else -1.0
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Mining {topk} nearest neighbors for {len(embeddings)} samples using {metric} metric")

    # Initialize NearestNeighbors
    # Note: topk+1 because the first neighbor is the sample itself
    nbrs = NearestNeighbors(n_neighbors=topk + 1, metric=metric, n_jobs=-1)
    nbrs.fit(embeddings)

    # Find neighbors
    distances, indices = nbrs.kneighbors(embeddings)

    # Remove the first column (self) and keep topk neighbors
    knn_indices = indices[:, 1:topk + 1]

    # Compute neighbor accuracy if labels are provided
    accuracy = -1.0
    if labels is not None:
        labels_array = np.array(labels)
        correct = 0
        total = 0

        for i in range(len(embeddings)):
            anchor_label = labels_array[i]
            neighbor_labels = labels_array[knn_indices[i]]

            # Count how many neighbors have the same label as anchor
            same_label_count = np.sum(neighbor_labels == anchor_label)
            correct += same_label_count
            total += topk

        accuracy = correct / total
        logger.info(f"Neighbor accuracy: {accuracy:.4f}")

    logger.info(f"KNN mining completed. Shape: {knn_indices.shape}")
    return knn_indices, accuracy


class SCANCollator:
    """
    Collate function for SCAN training that handles anchor-neighbor pairs.
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        device: torch.device
    ):
        self.vocab_sizes = vocab_sizes
        self.device = device

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of anchor-neighbor pairs.

        Args:
            batch: List of samples from SCANDataset

        Returns:
            Collated batch with anchor and neighbor data
        """
        # Separate anchor and neighbor samples
        anchor_samples = [item['anchor'] for item in batch]
        neighbor_samples = [item['neighbor'] for item in batch]

        # Collate anchors and neighbors separately
        anchors_batch = self._collate_samples(anchor_samples)
        neighbors_batch = self._collate_samples(neighbor_samples)

        # Also include indices for debugging/evaluation
        anchor_indices = torch.tensor([item['anchor_idx'] for item in batch], device=self.device)
        neighbor_indices = torch.tensor([item['neighbor_idx'] for item in batch], device=self.device)

        return {
            'anchor': anchors_batch,
            'neighbor': neighbors_batch,
            'anchor_indices': anchor_indices,
            'neighbor_indices': neighbor_indices
        }

    def _collate_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples into batched tensors.
        """
        batch_size = len(samples)

        # Extract components
        all_categorical = [sample['categorical_features'] for sample in samples]
        all_coordinates = [sample['coordinates'] for sample in samples]
        all_time_deltas = [sample['time_deltas'] for sample in samples]
        all_masks = [sample['mask'] for sample in samples]

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


def create_scan_data_loader(
    base_dataset: SmartHomeDataset,
    embeddings: np.ndarray,
    vocab_sizes: Dict[str, int],
    device: torch.device,
    topk: int = 20,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    labels: Optional[List[str]] = None
) -> Tuple[DataLoader, float]:
    """
    Create a DataLoader for SCAN training.

    Args:
        base_dataset: Base SmartHomeDataset
        embeddings: Pre-computed embeddings for KNN mining
        vocab_sizes: Vocabulary sizes for categorical features
        device: Target device
        topk: Number of nearest neighbors
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        labels: Optional ground truth labels

    Returns:
        DataLoader and neighbor accuracy
    """
    logger = logging.getLogger(__name__)

    # Mine nearest neighbors
    knn_indices, accuracy = mine_nearest_neighbors(
        embeddings=embeddings,
        topk=topk,
        labels=labels,
        metric='cosine'
    )

    # Create SCAN dataset
    scan_dataset = SCANDataset(
        base_dataset=base_dataset,
        embeddings=embeddings,
        knn_indices=knn_indices,
        num_neighbors=topk,
        labels=labels
    )

    # Create collator
    collator = SCANCollator(vocab_sizes=vocab_sizes, device=device)

    # Create DataLoader
    dataloader = DataLoader(
        dataset=scan_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True if device.type == 'cuda' else False
    )

    logger.info(f"Created SCAN DataLoader with {len(scan_dataset)} samples, batch_size={batch_size}")

    return dataloader, accuracy
