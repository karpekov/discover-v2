import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from .dataset import SmartHomeDataset

logger = logging.getLogger(__name__)


class SmartHomeClassificationDataset(SmartHomeDataset):
    """
    Dataset for smart-home activity classification using L2 labels.

    Extends SmartHomeDataset to filter out 'no_activity' samples and
    provide L2 labels for classification.
    """

    def __init__(
        self,
        data_path: str,
        vocab_path: str,
        sequence_length: int = 20,
        max_captions: int = 3,
        normalize_coords: bool = True,
        caption_types: str = 'long',
        exclude_no_activity: bool = True,
        exclude_other: bool = False,
        l2_label_mapping: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            data_path: Path to the dataset file (JSON or parquet)
            vocab_path: Path to vocabulary mappings (JSON)
            sequence_length: Fixed sequence length T
            max_captions: Maximum number of captions per sequence
            normalize_coords: Whether to normalize coordinates to [0,1]
            caption_types: Which caption types to use ('long', 'short', 'both')
            exclude_no_activity: Whether to exclude samples with 'No_Activity' L2 label
            l2_label_mapping: Optional custom L2 label to index mapping
        """
        # Initialize parent class
        super().__init__(
            data_path=data_path,
            vocab_path=vocab_path,
            sequence_length=sequence_length,
            max_captions=max_captions,
            normalize_coords=normalize_coords,
            caption_types=caption_types
        )

        self.exclude_no_activity = exclude_no_activity
        self.exclude_other = exclude_other

        # Set up L2 label mapping
        if l2_label_mapping is not None:
            self.l2_label_to_idx = l2_label_mapping
        else:
            self.l2_label_to_idx = self._create_default_l2_mapping()

        self.idx_to_l2_label = {idx: label for label, idx in self.l2_label_to_idx.items()}
        self.num_classes = len(self.l2_label_to_idx)

        # Filter data and collect label statistics
        self.filtered_data, self.label_stats = self._filter_and_analyze_data()

        logger.info(f"Classification dataset created:")
        logger.info(f"  Total samples after filtering: {len(self.filtered_data)}")
        logger.info(f"  Number of classes: {self.num_classes}")
        logger.info(f"  Label distribution: {self.label_stats}")

        # Update data to use filtered version
        self.data = self.filtered_data

    def _create_default_l2_mapping(self) -> Dict[str, int]:
        """Create default L2 label to index mapping."""
        # Based on the metadata analysis, excluding 'No_Activity'
        l2_labels = [
            'Cook',           # 0
            'Sleep',          # 1
            'Work',           # 2
            'Eat',            # 3
            'Relax',          # 4
            'Bathing',        # 5
            'Bed_to_toilet',  # 6
            'Take_medicine',  # 7
            'Leave_Home',     # 8
            'Enter_Home'      # 9
        ]

        # Add 'Other' if not excluded
        if not self.exclude_other:
            l2_labels.insert(6, 'Other')  # Insert at position 6

        return {label: idx for idx, label in enumerate(l2_labels)}

    def _filter_and_analyze_data(self) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Filter data to exclude unwanted samples and analyze label distribution.

        Returns:
            filtered_data: List of filtered samples
            label_stats: Dictionary with label counts
        """
        filtered_data = []
        label_counts = {}
        excluded_count = 0
        missing_label_count = 0

        for sample in self.data:
            # Get L2 label
            l2_label = sample.get('first_activity_l2', None)

            if l2_label is None:
                missing_label_count += 1
                continue

            # Exclude 'No_Activity' if requested
            if self.exclude_no_activity and l2_label == 'No_Activity':
                excluded_count += 1
                continue

            # Exclude 'Other' if requested
            if self.exclude_other and l2_label == 'Other':
                excluded_count += 1
                continue

            # Check if label is in our mapping
            if l2_label not in self.l2_label_to_idx:
                logger.warning(f"Unknown L2 label '{l2_label}' encountered, skipping sample")
                excluded_count += 1
                continue

            # Add to filtered data
            filtered_data.append(sample)

            # Update label counts
            if l2_label not in label_counts:
                label_counts[l2_label] = 0
            label_counts[l2_label] += 1

        logger.info(f"Data filtering results:")
        logger.info(f"  Original samples: {len(self.data)}")
        logger.info(f"  Samples with missing L2 labels: {missing_label_count}")
        logger.info(f"  Samples excluded (no_activity or unknown): {excluded_count}")
        logger.info(f"  Final samples: {len(filtered_data)}")

        return filtered_data, label_counts

    def __len__(self) -> int:
        return len(self.filtered_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample with L2 label.

        Returns:
            sample: Dict containing all the original fields plus:
                - l2_label: String L2 label
                - l2_label_idx: Integer index for the L2 label
        """
        # Get the base sample from parent class
        sample_data = self.filtered_data[idx]

        # Create a temporary data list for parent's __getitem__
        original_data = self.data
        self.data = [sample_data]  # Temporarily replace with single sample

        try:
            # Get processed sample from parent
            sample = super().__getitem__(0)
        finally:
            # Restore original data
            self.data = original_data

        # Add L2 label information
        l2_label = sample_data['first_activity_l2']
        l2_label_idx = self.l2_label_to_idx[l2_label]

        sample['l2_label'] = l2_label
        sample['l2_label_idx'] = torch.tensor(l2_label_idx, dtype=torch.long)

        return sample

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling class imbalance.

        Returns:
            weights: [num_classes] tensor with inverse frequency weights
        """
        total_samples = sum(self.label_stats.values())
        weights = torch.zeros(self.num_classes)

        for label, count in self.label_stats.items():
            if label in self.l2_label_to_idx:
                idx = self.l2_label_to_idx[label]
                weights[idx] = total_samples / (self.num_classes * count)

        return weights

    def get_label_distribution(self) -> Dict[str, float]:
        """Get the relative distribution of labels as percentages."""
        total = sum(self.label_stats.values())
        return {label: (count / total) * 100 for label, count in self.label_stats.items()}

    def print_dataset_info(self):
        """Print detailed dataset information."""
        print("\n" + "="*50)
        print("CLASSIFICATION DATASET INFO")
        print("="*50)
        print(f"Dataset path: {self.data_path if hasattr(self, 'data_path') else 'N/A'}")
        print(f"Total samples: {len(self)}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Exclude no_activity: {self.exclude_no_activity}")
        print(f"Exclude other: {self.exclude_other}")

        print("\nClass mapping:")
        for label, idx in sorted(self.l2_label_to_idx.items(), key=lambda x: x[1]):
            count = self.label_stats.get(label, 0)
            percentage = (count / len(self)) * 100 if len(self) > 0 else 0
            print(f"  {idx:2d}: {label:<15} ({count:4d} samples, {percentage:5.1f}%)")

        print("\nCategorical features:")
        for field, vocab_size in self.vocab_sizes.items():
            print(f"  {field}: {vocab_size} unique values")

        print("="*50 + "\n")


def create_classification_data_loader(
    dataset: SmartHomeClassificationDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for classification dataset.

    Args:
        dataset: SmartHomeClassificationDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory

    Returns:
        dataloader: DataLoader instance
    """
    def collate_fn(batch):
        """Custom collate function for classification batches."""
        # Separate the different components
        categorical_features = {}
        coordinates = []
        time_deltas = []
        masks = []
        captions = []
        l2_labels = []
        l2_label_indices = []

        # Get all categorical field names from the first sample
        if batch:
            categorical_fields = list(batch[0]['categorical_features'].keys())
        else:
            categorical_fields = []

        # Initialize categorical features dict
        for field in categorical_fields:
            categorical_features[field] = []

        # Collect data from batch
        for sample in batch:
            # Categorical features
            for field in categorical_fields:
                categorical_features[field].append(sample['categorical_features'][field])

            # Other features
            coordinates.append(sample['coordinates'])
            time_deltas.append(sample['time_deltas'])
            masks.append(sample['mask'])
            captions.extend(sample['captions'])  # Flatten captions
            l2_labels.append(sample['l2_label'])
            l2_label_indices.append(sample['l2_label_idx'])

        # Stack tensors
        for field in categorical_fields:
            categorical_features[field] = torch.stack(categorical_features[field])

        coordinates = torch.stack(coordinates)
        time_deltas = torch.stack(time_deltas)
        masks = torch.stack(masks)
        l2_label_indices = torch.stack(l2_label_indices)

        return {
            'categorical_features': categorical_features,
            'coordinates': coordinates,
            'time_deltas': time_deltas,
            'mask': masks,
            'captions': captions,
            'l2_labels': l2_labels,
            'l2_label_indices': l2_label_indices
        }

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    # Example usage and testing
    try:
        # Create dataset
        dataset = SmartHomeClassificationDataset(
            data_path="data/processed/casas/milan/training_20/train.json",
            vocab_path="data/processed/casas/milan/training_20/vocab.json",
            sequence_length=20,
            exclude_no_activity=True
        )

        # Print info
        dataset.print_dataset_info()

        # Test getting a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"L2 label: {sample['l2_label']}")
        print(f"L2 label idx: {sample['l2_label_idx']}")

        # Test data loader
        dataloader = create_classification_data_loader(dataset, batch_size=4)
        batch = next(iter(dataloader))
        print(f"Batch keys: {batch.keys()}")
        print(f"Batch size: {len(batch['l2_labels'])}")
        print(f"Label indices shape: {batch['l2_label_indices'].shape}")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
