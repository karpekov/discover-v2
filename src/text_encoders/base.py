"""Base classes for text encoders.

This module defines the interface that all text encoders must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import yaml
import torch


def get_best_device() -> str:
    """Automatically detect best available device.

    Checks in order: mps, cuda, cpu

    Returns:
        Device string ('mps', 'cuda', or 'cpu')
    """
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


@dataclass
class TextEncoderOutput:
    """Output from text encoder.

    Attributes:
        embeddings: Text embeddings [num_texts, embedding_dim]
        metadata: Optional metadata about the encoding process
    """
    embeddings: np.ndarray  # [num_texts, embedding_dim]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextEncoderConfig:
    """Configuration for text encoders.

    Attributes:
        encoder_type: Type of encoder ('gte', 'distilroberta', 'llama', 'clip', 'siglip')
        model_name: HuggingFace model name or path
        embedding_dim: Output embedding dimension
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for encoding (for memory efficiency)
        normalize: Whether to L2-normalize embeddings
        cache_dir: Optional cache directory for model downloads
        use_projection: Whether to use a projection head (for CLIP compatibility)
        projection_dim: Projection dimension if use_projection=True
        output_path_template: Optional template for output path (e.g., "data/embeddings/text/{dataset}/{strategy}/{split}_{style}_{encoder}.npz")
    """
    encoder_type: str
    model_name: str
    embedding_dim: int
    max_length: int = 512
    batch_size: int = 32
    normalize: bool = True
    cache_dir: Optional[str] = None
    use_projection: bool = False
    projection_dim: int = 512
    output_path_template: Optional[str] = None

    def __post_init__(self):
        """Set default output path template if not provided."""
        if self.output_path_template is None:
            # By default, save embeddings in the same folder as captions
            self.output_path_template = None  # Will be computed from caption path

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'encoder_type': self.encoder_type,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'normalize': self.normalize,
            'cache_dir': self.cache_dir,
            'use_projection': self.use_projection,
            'projection_dim': self.projection_dim,
            'output_path_template': self.output_path_template,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TextEncoderConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TextEncoderConfig':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def save_yaml(self, yaml_path: str):
        """Save config to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


class BaseTextEncoder(ABC):
    """Abstract base class for text encoders.

    All text encoders must implement the encode() method.
    """

    def __init__(self, config: TextEncoderConfig):
        """Initialize text encoder.

        Args:
            config: Text encoder configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None

        # Auto-detect device
        self.device = get_best_device()

    @abstractmethod
    def encode(self, texts: List[str]) -> TextEncoderOutput:
        """Encode a list of texts into embeddings.

        Args:
            texts: List of text strings

        Returns:
            TextEncoderOutput with embeddings and metadata
        """
        pass

    def encode_batch(self, texts: List[str], batch_size: Optional[int] = None) -> TextEncoderOutput:
        """Encode texts in batches for memory efficiency.

        Args:
            texts: List of text strings
            batch_size: Batch size (defaults to config.batch_size)

        Returns:
            TextEncoderOutput with all embeddings concatenated
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            output = self.encode(batch)
            all_embeddings.append(output.embeddings)

        embeddings = np.concatenate(all_embeddings, axis=0)

        return TextEncoderOutput(
            embeddings=embeddings,
            metadata={'num_batches': len(all_embeddings)}
        )

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save embeddings to disk.

        Args:
            embeddings: Embeddings array [num_samples, embedding_dim]
            output_path: Path to save embeddings (will save as .npz)
            metadata: Optional metadata to save alongside embeddings
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as compressed numpy archive
        save_dict = {'embeddings': embeddings}
        if metadata:
            # Save metadata as string arrays (npz doesn't support dicts directly)
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    save_dict[f'meta_{key}'] = np.array([value])

        np.savez_compressed(output_path, **save_dict)

    @staticmethod
    def load_embeddings(embeddings_path: str) -> tuple[np.ndarray, Dict[str, Any]]:
        """Load embeddings from disk.

        Args:
            embeddings_path: Path to embeddings file (.npz)

        Returns:
            Tuple of (embeddings, metadata)
        """
        data = np.load(embeddings_path, allow_pickle=False)
        embeddings = data['embeddings']

        # Extract metadata
        metadata = {}
        for key in data.keys():
            if key.startswith('meta_'):
                meta_key = key[5:]  # Remove 'meta_' prefix
                metadata[meta_key] = data[key].item()

        return embeddings, metadata

    def get_info(self) -> Dict[str, Any]:
        """Get information about the encoder."""
        return {
            'encoder_type': self.config.encoder_type,
            'model_name': self.config.model_name,
            'embedding_dim': self.config.embedding_dim,
            'normalize': self.config.normalize,
            'use_projection': self.config.use_projection,
            'projection_dim': self.config.projection_dim if self.config.use_projection else None,
            'device': self.device,
        }

    def get_output_path(self, captions_path: str, output_dir: Optional[str] = None) -> str:
        """Generate output path from caption file path.

        By default, saves embeddings in the same directory as the captions file.

        Args:
            captions_path: Path to input captions file
            output_dir: Optional override for output directory

        Returns:
            Generated output path
        """
        captions_path = Path(captions_path)

        # Extract split and style from filename
        # Format: {split}_captions_{style}.json
        filename = captions_path.stem  # Remove .json
        if '_captions_' in filename:
            split, style = filename.split('_captions_')
        else:
            split = 'train'
            style = 'baseline'

        # Generate output filename
        output_filename = f"{split}_embeddings_{style}_{self.config.encoder_type}.npz"

        # Determine output directory
        if output_dir:
            # Use specified output directory
            output_path = Path(output_dir) / output_filename
        elif self.config.output_path_template:
            # Use template from config
            parts = captions_path.parts
            try:
                processed_idx = parts.index('processed')
                dataset_type = parts[processed_idx + 1]  # e.g., 'casas'
                dataset = parts[processed_idx + 2]        # e.g., 'milan'
                strategy = parts[processed_idx + 3]       # e.g., 'fixed_length_20'
            except (ValueError, IndexError):
                dataset = 'unknown'
                strategy = 'unknown'

            output_path = self.config.output_path_template.format(
                dataset=dataset,
                strategy=strategy,
                split=split,
                style=style,
                encoder=self.config.encoder_type
            )
            output_path = Path(output_path)
        else:
            # Default: save in same directory as captions
            output_path = captions_path.parent / output_filename

        return str(output_path)

