"""
Embed sensor activation images using vision models.

This module loads generated sensor images and processes them through
vision models (CLIP, DINOv2, SigLIP) to create embeddings that can be
used by image-based sequence encoders.

Supported Models:
    - CLIP (openai/clip-vit-base-patch32): 512-dim contrastive vision-language embeddings
    - DINOv2 (facebook/dinov2-base): 768-dim self-supervised vision embeddings
    - SigLIP (google/siglip-base-patch16-224): 768-dim signature loss embeddings

Output format:
    data/processed/{dataset_type}/{dataset}/layout_embeddings/embeddings/
        {model_name}/
            dim224/
                embeddings.npz  # Contains: embeddings, sensor_ids, states, image_keys
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent.parent


class ImageEmbedder:
    """Base class for image embedding models."""

    def __init__(self, model_name: str, device: str = None):
        """
        Initialize the image embedder.

        Args:
            model_name: Name of the vision model
            device: Device to run on (cuda, mps, cpu). Auto-detected if None.
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"Using device: {self.device}")

        self.model = None
        self.preprocess = None
        self.embedding_dim = None

    def load_model(self):
        """Load the vision model. Must be implemented by subclasses."""
        raise NotImplementedError

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Embed a single image.

        Args:
            image: PIL Image

        Returns:
            Embedding as numpy array
        """
        raise NotImplementedError

    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Embed a batch of images.

        Args:
            images: List of PIL Images

        Returns:
            Embeddings as numpy array [batch_size, embedding_dim]
        """
        raise NotImplementedError


class CLIPImageEmbedder(ImageEmbedder):
    """CLIP vision encoder using Hugging Face transformers."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        # Simple naming for common models
        if model_name == "openai/clip-vit-base-patch32":
            simple_name = "clip_base"
        elif model_name == "openai/clip-vit-large-patch14":
            simple_name = "clip_large"
        else:
            # For custom models, use a cleaned version
            simple_name = f"clip_{model_name.split('/')[-1].replace('-', '_')}"
        
        super().__init__(simple_name, device)
        self.hf_model_name = model_name
        self.load_model()

    def load_model(self):
        """Load CLIP model from Hugging Face."""
        try:
            from transformers import CLIPProcessor, CLIPModel
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

        logger.info(f"Loading CLIP model from Hugging Face: {self.hf_model_name}")
        self.processor = CLIPProcessor.from_pretrained(self.hf_model_name)
        self.model = CLIPModel.from_pretrained(self.hf_model_name).to(self.device)
        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.config.vision_config.hidden_size
        logger.info(f"CLIP embedding dimension: {self.embedding_dim}")

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a single image with CLIP."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize embedding
            embedding = F.normalize(image_features, p=2, dim=-1)

        return embedding.cpu().numpy()[0]

    def embed_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of images with CLIP."""
        embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)

            # Encode
            with torch.no_grad():
                batch_features = self.model.get_image_features(**inputs)
                # Normalize
                batch_embeddings = F.normalize(batch_features, p=2, dim=-1)

            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)


class SigLIPImageEmbedder(ImageEmbedder):
    """SigLIP vision encoder."""

    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: str = None):
        super().__init__(f"siglip_{model_name.split('/')[-1]}", device)
        self.hf_model_name = model_name
        self.load_model()

    def load_model(self):
        """Load SigLIP model from HuggingFace."""
        try:
            from transformers import AutoProcessor, AutoModel
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

        logger.info(f"Loading SigLIP model: {self.hf_model_name}")
        self.processor = AutoProcessor.from_pretrained(self.hf_model_name)
        self.model = AutoModel.from_pretrained(self.hf_model_name).to(self.device)
        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.config.vision_config.hidden_size
        logger.info(f"SigLIP embedding dimension: {self.embedding_dim}")

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a single image with SigLIP."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            # Normalize
            embedding = F.normalize(outputs, p=2, dim=-1)

        return embedding.cpu().numpy()[0]

    def embed_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of images with SigLIP."""
        embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # Normalize
                batch_embeddings = F.normalize(outputs, p=2, dim=-1)

            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)


class DINOv2ImageEmbedder(ImageEmbedder):
    """DINOv2 vision encoder (facebook/dinov2-base)."""

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = None):
        # Simple naming for common models
        if model_name == "facebook/dinov2-base":
            simple_name = "dinov2"
        elif model_name == "facebook/dinov2-large":
            simple_name = "dinov2_large"
        elif model_name == "facebook/dinov2-giant":
            simple_name = "dinov2_giant"
        else:
            # For custom models, use a cleaned version
            simple_name = f"dinov2_{model_name.split('/')[-1].replace('-', '_')}"
        
        super().__init__(simple_name, device)
        self.hf_model_name = model_name
        self.load_model()

    def load_model(self):
        """Load DINOv2 model from HuggingFace."""
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

        logger.info(f"Loading DINOv2 model: {self.hf_model_name}")
        self.processor = AutoImageProcessor.from_pretrained(self.hf_model_name)
        self.model = AutoModel.from_pretrained(self.hf_model_name).to(self.device)
        self.model.eval()

        # Get embedding dimension from config
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"DINOv2 embedding dimension: {self.embedding_dim}")

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Embed a single image with DINOv2.

        Uses the [CLS] token from the last hidden state as the image embedding.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Take the [CLS] token (first token) from last_hidden_state
            cls_embedding = outputs.last_hidden_state[:, 0]
            # Normalize
            embedding = F.normalize(cls_embedding, p=2, dim=-1)

        return embedding.cpu().numpy()[0]

    def embed_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of images with DINOv2."""
        embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Take the [CLS] token (first token) from last_hidden_state
                cls_embeddings = outputs.last_hidden_state[:, 0]
                # Normalize
                batch_embeddings = F.normalize(cls_embeddings, p=2, dim=-1)

            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)


def get_embedder(model_name: str, device: str = None) -> ImageEmbedder:
    """
    Factory function to get the appropriate embedder.

    Args:
        model_name: Name of the model ("clip", "siglip", "dinov2", or Hugging Face model path)
        device: Device to run on

    Returns:
        ImageEmbedder instance

    Examples:
        get_embedder("clip")  # Uses openai/clip-vit-base-patch32
        get_embedder("openai/clip-vit-large-patch14")  # Specific CLIP variant
        get_embedder("siglip")  # Uses google/siglip-base-patch16-224
        get_embedder("dinov2")  # Uses facebook/dinov2-base
        get_embedder("facebook/dinov2-large")  # Specific DINOv2 variant
    """
    model_name_lower = model_name.lower()

    if "clip" in model_name_lower:
        # If just "clip", use default HF model
        if model_name.lower() == "clip":
            return CLIPImageEmbedder(model_name="openai/clip-vit-base-patch32", device=device)
        # Otherwise use the provided model name
        else:
            return CLIPImageEmbedder(model_name=model_name, device=device)

    elif "siglip" in model_name_lower:
        # If just "siglip", use default
        if model_name.lower() == "siglip":
            return SigLIPImageEmbedder(model_name="google/siglip-base-patch16-224", device=device)
        # Otherwise use the provided model name
        else:
            return SigLIPImageEmbedder(model_name=model_name, device=device)

    elif "dinov2" in model_name_lower or "dino" in model_name_lower:
        # If just "dinov2" or "dino", use default
        if model_name.lower() in ["dinov2", "dino"]:
            return DINOv2ImageEmbedder(model_name="facebook/dinov2-base", device=device)
        # Otherwise use the provided model name
        else:
            return DINOv2ImageEmbedder(model_name=model_name, device=device)

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported: 'clip', 'siglip', 'dinov2', or Hugging Face model paths like "
            f"'openai/clip-vit-base-patch32', 'facebook/dinov2-base'"
        )


def load_images_from_directory(
    image_dir: Path,
    image_keys: Optional[List[str]] = None
) -> Tuple[List[Image.Image], List[str], List[str], List[str]]:
    """
    Load images from directory.

    Args:
        image_dir: Directory containing images
        image_keys: Optional list of specific image keys to load

    Returns:
        Tuple of (images, sensor_ids, states, image_keys)
    """
    images = []
    sensor_ids = []
    states = []
    keys = []

    # Load metadata if available
    metadata_file = image_dir / "image_metadata.json"
    if metadata_file.exists() and image_keys is None:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            image_keys = metadata.get('image_keys', [])

    # If no image keys provided, scan directory
    if image_keys is None:
        image_files = sorted(image_dir.glob("*.png"))
        image_keys = [f.stem for f in image_files]

    logger.info(f"Loading {len(image_keys)} images from {image_dir}")

    for key in tqdm(image_keys, desc="Loading images"):
        # Parse sensor_id and state from key (e.g., "M001_ON")
        parts = key.rsplit('_', 1)
        if len(parts) != 2:
            logger.warning(f"Skipping invalid image key: {key}")
            continue

        sensor_id, state = parts
        image_path = image_dir / f"{key}.png"

        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue

        # Load image
        img = Image.open(image_path).convert('RGB')

        images.append(img)
        sensor_ids.append(sensor_id)
        states.append(state)
        keys.append(key)

    logger.info(f"Loaded {len(images)} images successfully")
    return images, sensor_ids, states, keys


def embed_dataset_images(
    dataset: str,
    model_name: str = "clip",
    dataset_type: str = "casas",
    output_size: Tuple[int, int] = (224, 224),
    device: str = None,
    batch_size: int = 32,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Embed all sensor images for a dataset.

    Args:
        dataset: Dataset name (e.g., "milan")
        model_name: Vision model to use ("clip", "siglip", etc.)
        dataset_type: Dataset type (default: "casas")
        output_size: Image size that was used (default: (224, 224))
        device: Device to run on (auto-detected if None)
        batch_size: Batch size for processing
        output_dir: Output directory (auto-generated if None)

    Returns:
        Path to saved embeddings file
    """
    # Get image directory
    project_root = get_project_root()
    dim_folder = f"dim{output_size[0]}"
    image_dir = (
        project_root / "data" / "processed" / dataset_type / dataset /
        "layout_embeddings" / "images" / dim_folder
    )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}\n"
            f"Please generate images first using generate_images.py"
        )

    # Load embedder
    embedder = get_embedder(model_name, device=device)

    # Load images
    images, sensor_ids, states, image_keys = load_images_from_directory(image_dir)

    # Embed images
    logger.info(f"Embedding {len(images)} images with {embedder.model_name}...")
    embeddings = embedder.embed_batch(images, batch_size=batch_size)
    logger.info(f"Generated embeddings shape: {embeddings.shape}")

    # Set up output directory
    if output_dir is None:
        output_dir = (
            project_root / "data" / "processed" / dataset_type / dataset /
            "layout_embeddings" / "embeddings" / embedder.model_name / dim_folder
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    output_file = output_dir / "embeddings.npz"
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        sensor_ids=np.array(sensor_ids),
        states=np.array(states),
        image_keys=np.array(image_keys),
        embedding_dim=embedder.embedding_dim,
        model_name=embedder.model_name
    )

    logger.info(f"Saved embeddings to: {output_file}")

    # Save metadata
    metadata_file = output_dir / "embedding_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "dataset": dataset,
            "dataset_type": dataset_type,
            "model_name": embedder.model_name,
            "embedding_dim": int(embedder.embedding_dim),
            "num_embeddings": len(embeddings),
            "output_size": list(output_size),
            "sensor_ids": sensor_ids,
            "states": states,
            "image_keys": image_keys
        }, f, indent=2)

    logger.info(f"Saved metadata to: {metadata_file}")

    return output_file


def load_embeddings(
    dataset: str,
    model_name: str = "clip",
    dataset_type: str = "casas",
    output_size: Tuple[int, int] = (224, 224)
) -> Dict[str, Any]:
    """
    Load pre-computed embeddings.

    Args:
        dataset: Dataset name
        model_name: Model name used for embedding (e.g., "clip", "openai/clip-vit-base-patch32")
        dataset_type: Dataset type
        output_size: Image size used

    Returns:
        Dictionary with embeddings and metadata
    """
    project_root = get_project_root()
    dim_folder = f"dim{output_size[0]}"

    # Normalize model name for directory lookup (must match naming in embedder classes)
    if model_name.lower() == "clip" or model_name == "openai/clip-vit-base-patch32":
        clean_name = "clip_base"
    elif model_name == "openai/clip-vit-large-patch14":
        clean_name = "clip_large"
    elif model_name.lower() in ["dinov2", "dino"] or model_name == "facebook/dinov2-base":
        clean_name = "dinov2"
    elif model_name == "facebook/dinov2-large":
        clean_name = "dinov2_large"
    elif model_name == "facebook/dinov2-giant":
        clean_name = "dinov2_giant"
    elif model_name.lower() == "siglip" or model_name == "google/siglip-base-patch16-224":
        clean_name = "siglip_base_patch16_224"
    elif "/" in model_name:
        # For custom Hugging Face models, use last part of path
        model_lower = model_name.lower()
        if "clip" in model_lower:
            clean_name = f"clip_{model_name.split('/')[-1].replace('-', '_')}"
        elif "dinov2" in model_lower or "dino" in model_lower:
            clean_name = f"dinov2_{model_name.split('/')[-1].replace('-', '_')}"
        elif "siglip" in model_lower:
            clean_name = f"siglip_{model_name.split('/')[-1].replace('-', '_')}"
        else:
            clean_name = model_name.split('/')[-1].replace('-', '_')
    else:
        clean_name = model_name.replace("/", "_").replace("-", "_")

    embeddings_file = (
        project_root / "data" / "processed" / dataset_type / dataset /
        "layout_embeddings" / "embeddings" / clean_name / dim_folder / "embeddings.npz"
    )

    if not embeddings_file.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {embeddings_file}\n"
            f"Please generate embeddings first using embed_dataset_images()"
        )

    data = np.load(embeddings_file, allow_pickle=True)

    return {
        'embeddings': data['embeddings'],
        'sensor_ids': data['sensor_ids'],
        'states': data['states'],
        'image_keys': data['image_keys'],
        'embedding_dim': int(data['embedding_dim']),
        'model_name': str(data['model_name'])
    }


def get_sensor_embedding(
    dataset: str,
    sensor_id: str,
    state: str,
    model_name: str = "clip",
    dataset_type: str = "casas",
    output_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Get embedding for a specific sensor activation.

    Args:
        dataset: Dataset name
        sensor_id: Sensor identifier (e.g., "M001")
        state: Sensor state (e.g., "ON", "OFF")
        model_name: Model name
        dataset_type: Dataset type
        output_size: Image size

    Returns:
        Embedding vector
    """
    data = load_embeddings(dataset, model_name, dataset_type, output_size)

    # Find the embedding
    image_key = f"{sensor_id}_{state}"
    idx = np.where(data['image_keys'] == image_key)[0]

    if len(idx) == 0:
        raise ValueError(f"Embedding not found for {image_key}")

    return data['embeddings'][idx[0]]


if __name__ == "__main__":
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Embed sensor activation images using vision models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed Milan images with CLIP (224×224)
  python -m src.encoders.sensor.image.embed_images --dataset milan --model clip

  # Embed with DINOv2 (224×224)
  python -m src.encoders.sensor.image.embed_images --dataset milan --model dinov2

  # Embed with SigLIP (512×512)
  python -m src.encoders.sensor.image.embed_images --dataset milan --model siglip --output-width 512 --output-height 512

  # Use specific model variant from Hugging Face
  python -m src.encoders.sensor.image.embed_images --dataset milan --model "facebook/dinov2-large"

  # Specify device
  python -m src.encoders.sensor.image.embed_images --dataset milan --model clip --device cuda
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., milan, aruba)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="clip",
        help="Vision model: 'clip', 'dinov2', 'siglip', or Hugging Face model path (default: clip)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="casas",
        help="Dataset type (default: casas)"
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=224,
        help="Image width that was used (default: 224)"
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=224,
        help="Image height that was used (default: 224)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, mps, cpu (auto-detected if not specified)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )

    args = parser.parse_args()

    logger.info(f"Embedding images for dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Image size: {args.output_width}×{args.output_height}")

    try:
        output_file = embed_dataset_images(
            dataset=args.dataset,
            model_name=args.model,
            dataset_type=args.dataset_type,
            output_size=(args.output_width, args.output_height),
            device=args.device,
            batch_size=args.batch_size
        )

        logger.info(f"✓ Successfully embedded images")
        logger.info(f"✓ Embeddings saved to: {output_file}")

    except Exception as e:
        logger.error(f"✗ Failed to embed images: {e}")
        raise

