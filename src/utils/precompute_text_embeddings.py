#!/usr/bin/env python3
"""
Pre-compute text embeddings for faster training.
Since the text encoder is frozen, we can compute embeddings once and cache them.
"""

import json
import pickle
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.text_encoder import build_text_encoder, get_text_cfg
from dataio.dataset import SmartHomeDataset


def load_all_unique_captions(data_paths: List[str], caption_types: str = 'long') -> List[str]:
    """
    Load all unique captions from the dataset files.
    
    Args:
        data_paths: List of paths to dataset JSON files
        caption_types: Type of captions to use ('long', 'short', etc.)
        
    Returns:
        List of unique caption strings
    """
    all_captions = set()
    
    for data_path in data_paths:
        print(f"Loading captions from: {data_path}")
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        for item in tqdm(data, desc=f"Processing {Path(data_path).name}"):
            captions = item.get('captions', [])
            
            # Handle both dictionary and list formats
            if isinstance(captions, dict):
                if caption_types == 'long':
                    caption_list = captions.get('long', [])
                elif caption_types == 'short':
                    caption_list = captions.get('short', [])
                elif caption_types == 'all':
                    caption_list = captions.get('long', []) + captions.get('short', [])
                else:
                    caption_list = captions.get(caption_types, [])
            elif isinstance(captions, list):
                # Captions are already a list (most common format)
                caption_list = captions
            else:
                caption_list = []
            
            for caption in caption_list:
                if caption and isinstance(caption, str):
                    all_captions.add(caption.strip())
    
    unique_captions = sorted(list(all_captions))
    print(f"Found {len(unique_captions)} unique captions")
    
    return unique_captions


def precompute_embeddings(
    config: Dict[str, Any],
    data_paths: List[str],
    output_path: str,
    batch_size: int = 32,
    device: str = 'auto'
):
    """
    Pre-compute text embeddings and save them to disk.
    
    Args:
        config: Training configuration with text encoder settings
        data_paths: List of dataset file paths
        output_path: Path to save the precomputed embeddings
        batch_size: Batch size for embedding computation
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
    """
    
    # Setup device
    if device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Build text encoder
    print("Building text encoder...")
    text_encoder = build_text_encoder(config)
    text_encoder.to(device)
    text_encoder.eval()
    
    # Log text encoder config
    from models.text_encoder import log_text_encoder_config
    log_text_encoder_config(config)
    
    # Load all unique captions
    caption_types = config.get('caption_types', 'long')
    unique_captions = load_all_unique_captions(data_paths, caption_types)
    
    if not unique_captions:
        raise ValueError("No captions found in the dataset files!")
    
    print(f"Pre-computing embeddings for {len(unique_captions)} unique captions...")
    
    # Pre-compute embeddings in batches
    caption_to_embedding = {}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_captions), batch_size), desc="Computing embeddings"):
            batch_captions = unique_captions[i:i + batch_size]
            
            # Get base embeddings (before projection)
            base_embeddings = text_encoder.encode_texts(batch_captions, device)
            
            # Also get projected embeddings (for CLIP)
            if hasattr(text_encoder, 'encode_texts_clip'):
                projected_embeddings = text_encoder.encode_texts_clip(batch_captions, device)
            else:
                projected_embeddings = base_embeddings
            
            # Store both base and projected embeddings
            for j, caption in enumerate(batch_captions):
                caption_to_embedding[caption] = {
                    'base': base_embeddings[j].cpu(),
                    'projected': projected_embeddings[j].cpu()
                }
    
    # Save the embeddings cache
    cache_data = {
        'caption_to_embedding': caption_to_embedding,
        'config': config,
        'text_encoder_config': get_text_cfg(config),
        'num_captions': len(unique_captions),
        'embedding_dim_base': base_embeddings.shape[-1],
        'embedding_dim_projected': projected_embeddings.shape[-1],
        'device_used': str(device)
    }
    
    print(f"Saving embeddings cache to: {output_path}")
    os.makedirs(Path(output_path).parent, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"✅ Successfully cached {len(unique_captions)} text embeddings")
    print(f"   Base embedding dim: {base_embeddings.shape[-1]}")
    print(f"   Projected embedding dim: {projected_embeddings.shape[-1]}")
    print(f"   Cache size: {Path(output_path).stat().st_size / (1024**2):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute text embeddings for faster training')
    parser.add_argument('--config', type=str, required=True, help='Path to training config file')
    parser.add_argument('--output', type=str, required=True, help='Path to save embeddings cache')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding computation')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Get data paths from config
    data_paths = []
    if config.get('train_data_path'):
        data_paths.append(config['train_data_path'])
    if config.get('val_data_path'):
        data_paths.append(config['val_data_path'])
    if config.get('train_presegmented_data_path'):
        data_paths.append(config['train_presegmented_data_path'])
    if config.get('val_presegmented_data_path'):
        data_paths.append(config['val_presegmented_data_path'])
    
    if not data_paths:
        raise ValueError("No data paths found in config file!")
    
    # Filter to existing paths
    existing_paths = []
    for path in data_paths:
        if Path(path).exists():
            existing_paths.append(path)
        else:
            print(f"Warning: Data path does not exist: {path}")
    
    if not existing_paths:
        raise ValueError("No existing data paths found!")
    
    print(f"Will process data from {len(existing_paths)} files:")
    for path in existing_paths:
        print(f"  - {path}")
    
    # Pre-compute embeddings
    precompute_embeddings(
        config=config,
        data_paths=existing_paths,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    import os
    main()
