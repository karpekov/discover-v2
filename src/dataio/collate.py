import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any
import random


def collate_fn(
  batch: List[Dict[str, Any]],
  text_encoder,
  span_masker,
  vocab_sizes: Dict[str, int],
  device: torch.device,
  apply_mlm: bool = True
) -> Dict[str, Any]:
  """
  Collate function for smart-home dataset with MLM masking and text encoding.

  Args:
    batch: List of samples from SmartHomeDataset
    text_encoder: TextEncoder instance for encoding captions
    span_masker: SpanMasker instance for MLM
    vocab_sizes: Dictionary mapping field names to vocabulary sizes
    device: Target device
    apply_mlm: Whether to apply MLM masking (set to False during inference)

  Returns:
    collated_batch: Dictionary containing batched and processed data
  """
  batch_size = len(batch)

  # Extract components
  all_categorical = []
  all_coordinates = []
  all_time_deltas = []
  all_masks = []
  all_captions = []
  activity_labels_l1 = []
  activity_labels_l2 = []

  for sample in batch:
    all_categorical.append(sample['categorical_features'])
    all_coordinates.append(sample['coordinates'])
    all_time_deltas.append(sample['time_deltas'])
    all_masks.append(sample['mask'])
    all_captions.extend(sample['captions'])  # Flatten captions

    # Extract ground truth activity labels if available
    activity_labels_l1.append(sample.get('first_activity', 'Unknown'))
    activity_labels_l2.append(sample.get('first_activity_l2', 'Unknown'))

  # Stack tensors
  coordinates = torch.stack(all_coordinates).to(device)  # [batch_size, seq_len, 2]
  time_deltas = torch.stack(all_time_deltas).to(device)  # [batch_size, seq_len]
  masks = torch.stack(all_masks).to(device)  # [batch_size, seq_len]

  # Stack categorical features
  categorical_features = {}
  for field in all_categorical[0].keys():
    field_tensors = [sample[field] for sample in all_categorical]
    categorical_features[field] = torch.stack(field_tensors).to(device)  # [batch_size, seq_len]

  # Encode text captions
  if all_captions:
    # Remove empty captions
    all_captions = [cap for cap in all_captions if cap.strip()]

    if all_captions:
      text_embeddings = text_encoder.encode_texts_clip(all_captions, device)  # [num_captions, 512] with CLIP projection

      # If we have multiple captions per sample, we need to handle this
      # For simplicity, let's take one random caption per sample during training
      # and average during evaluation
      if len(all_captions) > batch_size:
        # Multiple captions per sample - sample one per batch item
        captions_per_sample = len(all_captions) // batch_size
        selected_indices = []
        for i in range(batch_size):
          start_idx = i * captions_per_sample
          end_idx = start_idx + captions_per_sample
          # Randomly select one caption for this sample
          selected_idx = random.randint(start_idx, end_idx - 1)
          selected_indices.append(selected_idx)

        text_embeddings = text_embeddings[selected_indices]  # [batch_size, d_model]
      elif len(all_captions) < batch_size:
        # Pad with repeated captions if needed
        while len(all_captions) < batch_size:
          all_captions.append(all_captions[-1])
        text_embeddings = text_encoder.encode_texts_clip(all_captions, device)
    else:
      # No valid captions - create dummy embeddings (512-dim for CLIP projection)
      text_embeddings = torch.zeros(batch_size, 512, device=device)
  else:
    # No captions - create dummy embeddings (512-dim for CLIP projection)
    text_embeddings = torch.zeros(batch_size, 512, device=device)

  # Prepare result dictionary
  result = {
    'categorical_features': categorical_features,
    'coordinates': coordinates,
    'time_deltas': time_deltas,
    'mask': masks,
    'text_embeddings': text_embeddings,
    'activity_labels': activity_labels_l1,  # Primary activity labels for evaluation
    'activity_labels_l2': activity_labels_l2,  # Secondary activity labels for evaluation
    'captions': all_captions,  # Include captions for evaluation
    'batch_size': batch_size
  }

  # Apply MLM masking if requested
  if apply_mlm and span_masker is not None:
    # Extract activity and room labels for transition detection
    activity_labels_tensor = None
    room_labels_tensor = None

    # Try to extract room labels from categorical features if available
    if 'room_id' in categorical_features:
      room_labels_tensor = categorical_features['room_id']

    # For activity labels, we'd need them in the dataset - for now use None
    # This could be enhanced to include activity labels from the dataset

    masked_features, original_features, mask_positions, blackout_masks = span_masker(
      categorical_features, vocab_sizes, activity_labels_tensor, room_labels_tensor
    )

    # Use masked features as input and original features as labels
    result.update({
      'categorical_features': masked_features,  # Use masked features for input
      'mlm_labels': original_features,          # Original features are the labels
      'mlm_mask_positions': mask_positions,     # Mask positions for loss computation
      'blackout_masks': blackout_masks          # Blackout masks for embedding zeroing
    })

  return result


class SmartHomeCollator:
  """
  Collator class that can be used with DataLoader.
  """

  def __init__(
    self,
    text_encoder,
    span_masker,
    vocab_sizes: Dict[str, int],
    device: torch.device,
    apply_mlm: bool = True
  ):
    self.text_encoder = text_encoder
    self.span_masker = span_masker
    self.vocab_sizes = vocab_sizes
    self.device = device
    self.apply_mlm = apply_mlm

  def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return collate_fn(
      batch=batch,
      text_encoder=self.text_encoder,
      span_masker=self.span_masker,
      vocab_sizes=self.vocab_sizes,
      device=self.device,
      apply_mlm=self.apply_mlm
    )

  def set_mlm_mode(self, apply_mlm: bool):
    """Toggle MLM masking on/off."""
    self.apply_mlm = apply_mlm


def create_data_loader(
  dataset,
  text_encoder,
  span_masker,
  vocab_sizes: Dict[str, int],
  device: torch.device,
  batch_size: int = 32,
  shuffle: bool = True,
  num_workers: int = 0,
  apply_mlm: bool = True
):
  """
  Create a DataLoader with the appropriate collate function.

  Args:
    dataset: SmartHomeDataset instance
    text_encoder: TextEncoder instance
    span_masker: SpanMasker instance
    vocab_sizes: Dictionary mapping field names to vocabulary sizes
    device: Target device
    batch_size: Batch size
    shuffle: Whether to shuffle the data
    num_workers: Number of worker processes
    apply_mlm: Whether to apply MLM masking

  Returns:
    DataLoader instance
  """
  from torch.utils.data import DataLoader

  collator = SmartHomeCollator(
    text_encoder=text_encoder,
    span_masker=span_masker,
    vocab_sizes=vocab_sizes,
    device=device,
    apply_mlm=apply_mlm
  )

  return DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    collate_fn=collator,
    pin_memory=True if device.type == 'cuda' else False
  )
