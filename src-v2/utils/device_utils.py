"""
Device utilities for handling CUDA, MPS, and CPU devices.
"""

import torch
from typing import Dict, Any


def get_optimal_device() -> torch.device:
  """
  Get the optimal device for training/inference.
  Priority: CUDA > MPS > CPU

  Returns:
    torch.device: The best available device
  """
  if torch.cuda.is_available():
    return torch.device('cuda')
  elif torch.backends.mps.is_available():
    return torch.device('mps')
  else:
    return torch.device('cpu')


def get_device_config(device: torch.device) -> Dict[str, Any]:
  """
  Get device-specific configuration settings.

  Args:
    device: Target device

  Returns:
    Dict with device-specific settings
  """
  config = {
    'use_amp': False,
    'pin_memory': False,
    'num_workers': 4
  }

  if device.type == 'cuda':
    config.update({
      'use_amp': True,
      'pin_memory': True,
      'num_workers': 4
    })
  elif device.type == 'mps':
    config.update({
      'use_amp': False,  # MPS AMP support is limited
      'pin_memory': False,
      'num_workers': 0  # MPS works better with single-threaded data loading
    })
  else:  # CPU
    config.update({
      'use_amp': False,
      'pin_memory': False,
      'num_workers': 4
    })

  return config


def log_device_info(device: torch.device):
  """
  Log information about the device being used.

  Args:
    device: Device to log info about
  """
  print(f"Using device: {device}")

  if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    print(f"  CUDA version: {torch.version.cuda}")
  elif device.type == 'mps':
    print("  Apple Silicon GPU (MPS)")
    print("  Note: Using regular precision (no AMP)")
  else:
    print("  CPU device")

  print(f"  PyTorch version: {torch.__version__}")


def optimize_for_device(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
  """
  Optimize configuration for the given device.

  Args:
    config: Base configuration
    device: Target device

  Returns:
    Optimized configuration
  """
  device_config = get_device_config(device)

  # Update config with device-specific settings
  optimized_config = config.copy()
  optimized_config.update(device_config)
  optimized_config['device'] = device.type

  return optimized_config
