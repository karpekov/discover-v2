"""
Visualize image embeddings using dimensionality reduction.

Creates 2D visualizations of sensor activation image embeddings using t-SNE or UMAP,
with color coding by sensor type and state.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_metadata(dataset: str, dataset_type: str = "casas") -> Dict:
    """Load dataset metadata for sensor types."""
    metadata_path = get_project_root() / "metadata" / f"{dataset_type}_metadata.json"
    with open(metadata_path, 'r') as f:
        all_metadata = json.load(f)
    return all_metadata[dataset]


def load_embeddings(
    dataset: str,
    model_name: str = "clip",
    dataset_type: str = "casas",
    output_size: Tuple[int, int] = (224, 224)
) -> Dict:
    """Load pre-computed image embeddings."""
    project_root = get_project_root()
    dim_folder = f"dim{output_size[0]}"

    # Normalize model name for directory lookup (must match naming in embed_images.py)
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
        raise FileNotFoundError(f"Embeddings not found: {embeddings_file}")

    data = np.load(embeddings_file, allow_pickle=True)

    return {
        'embeddings': data['embeddings'],
        'sensor_ids': data['sensor_ids'],
        'states': data['states'],
        'image_keys': data['image_keys'],
        'embedding_dim': int(data['embedding_dim']),
        'model_name': str(data['model_name'])
    }


def reduce_dimensions(embeddings: np.ndarray, method: str = "tsne", n_components: int = 2) -> np.ndarray:
    """
    Reduce embedding dimensions for visualization.

    Args:
        embeddings: High-dimensional embeddings
        method: 'tsne', 'umap', or 'pca'
        n_components: Number of output dimensions (typically 2)

    Returns:
        Reduced embeddings
    """
    logger.info(f"Reducing dimensions using {method.upper()}...")

    if method.lower() == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings)-1))
        reduced = reducer.fit_transform(embeddings)

    elif method.lower() == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("UMAP not installed, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(embeddings)

    elif method.lower() == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(embeddings)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne', 'umap', or 'pca'")

    return reduced


def get_sensor_type_from_metadata(sensor_id: str, metadata: Dict) -> str:
    """Get sensor type from metadata."""
    sensor_types = metadata.get('sensor_type', {})
    return sensor_types.get(sensor_id, 'unknown')


def get_sensor_room_from_metadata(sensor_id: str, metadata: Dict) -> str:
    """Get sensor room location from metadata."""
    sensor_locations = metadata.get('sensor_location', {})
    return sensor_locations.get(sensor_id, 'unknown')


def create_visualization(
    embeddings_2d: np.ndarray,
    sensor_ids: np.ndarray,
    states: np.ndarray,
    metadata: Dict,
    method: str = "tsne",
    show_labels: bool = False,
    output_path: Optional[Path] = None,
    model_name: str = "unknown",
    embedding_dim: int = 0
):
    """
    Create visualization of image embeddings.

    Args:
        embeddings_2d: 2D embeddings
        sensor_ids: Array of sensor IDs
        states: Array of states
        metadata: Dataset metadata
        method: Dimensionality reduction method used
        show_labels: Whether to show point labels
        output_path: Path to save figure
        model_name: Name of the vision model used
        embedding_dim: Dimension of the embeddings
    """
    # Get sensor types and rooms
    sensor_types = [get_sensor_type_from_metadata(sid, metadata) for sid in sensor_ids]
    sensor_rooms = [get_sensor_room_from_metadata(sid, metadata) for sid in sensor_ids]

    # Create color scheme
    # Color by sensor type, marker by state
    unique_types = sorted(set(sensor_types))
    type_colors = dict(zip(unique_types, sns.color_palette("tab10", len(unique_types))))

    # Room colors
    unique_rooms = sorted(set(sensor_rooms))
    room_colors = dict(zip(unique_rooms, sns.color_palette("husl", len(unique_rooms))))

    # State markers
    state_markers = {'ON': 'o', 'OFF': 'X', 'OPEN': 's', 'CLOSE': '^'}

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Plot 1: Color by sensor type, marker by state
    ax1 = axes[0]
    for stype in unique_types:
        for state in set(states):
            mask = (np.array(sensor_types) == stype) & (states == state)
            if mask.any():
                marker = state_markers.get(state, 'o')
                label = f"{stype} - {state}"
                ax1.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[type_colors[stype]],
                    marker=marker,
                    s=100,
                    alpha=0.7,
                    label=label,
                    edgecolors='black',
                    linewidth=0.5
                )

    if show_labels:
        for i, (x, y, sid, state) in enumerate(zip(
            embeddings_2d[:, 0], embeddings_2d[:, 1], sensor_ids, states
        )):
            ax1.annotate(
                f"{sid}_{state}",
                (x, y),
                fontsize=6,
                alpha=0.7,
                xytext=(3, 3),
                textcoords='offset points'
            )

    ax1.set_xlabel(f"{method.upper()} Dimension 1", fontsize=12)
    ax1.set_ylabel(f"{method.upper()} Dimension 2", fontsize=12)
    ax1.set_title(f"By Sensor Type and State\n({model_name}, {embedding_dim}D)", fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(alpha=0.3)

    # Plot 2: Color by state only
    ax2 = axes[1]
    unique_states = sorted(set(states))
    state_colors = dict(zip(unique_states, sns.color_palette("Set1", len(unique_states))))

    for state in unique_states:
        mask = states == state
        marker = state_markers.get(state, 'o')
        ax2.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[state_colors[state]],
            marker=marker,
            s=100,
            alpha=0.7,
            label=state,
            edgecolors='black',
            linewidth=1
        )

    ax2.set_xlabel(f"{method.upper()} Dimension 1", fontsize=12)
    ax2.set_ylabel(f"{method.upper()} Dimension 2", fontsize=12)
    ax2.set_title(f"By State\n({model_name}, {embedding_dim}D)", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # Plot 3: Color by room location
    ax3 = axes[2]
    for room in unique_rooms:
        for state in set(states):
            mask = (np.array(sensor_rooms) == room) & (states == state)
            if mask.any():
                marker = state_markers.get(state, 'o')
                label = f"{room.replace('_', ' ').title()} - {state}"
                ax3.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[room_colors[room]],
                    marker=marker,
                    s=100,
                    alpha=0.7,
                    label=label,
                    edgecolors='black',
                    linewidth=0.5
                )

    ax3.set_xlabel(f"{method.upper()} Dimension 1", fontsize=12)
    ax3.set_ylabel(f"{method.upper()} Dimension 2", fontsize=12)
    ax3.set_title(f"By Room Location\n({model_name}, {embedding_dim}D)", fontsize=14, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_image_embeddings(
    dataset: str,
    model_name: str = "clip",
    dataset_type: str = "casas",
    output_size: Tuple[int, int] = (224, 224),
    method: str = "tsne",
    show_labels: bool = False,
    output_dir: Optional[Path] = None
):
    """
    Main function to visualize image embeddings.

    Args:
        dataset: Dataset name
        model_name: Model name
        dataset_type: Dataset type
        output_size: Image size
        method: Dimensionality reduction method
        show_labels: Whether to show labels on plot
        output_dir: Output directory for saving plot
    """
    # Load embeddings
    logger.info(f"Loading embeddings for {dataset}...")
    data = load_embeddings(dataset, model_name, dataset_type, output_size)

    logger.info(f"Loaded {len(data['embeddings'])} embeddings")
    logger.info(f"Embedding dimension: {data['embedding_dim']}")
    logger.info(f"Model: {data['model_name']}")

    # Load metadata
    metadata = load_metadata(dataset, dataset_type)

    # Reduce dimensions
    embeddings_2d = reduce_dimensions(data['embeddings'], method=method)

    # Set output path - save in same folder as embeddings
    if output_dir is None:
        # Normalize model name for directory lookup (must match naming in embed_images.py)
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

        dim_folder = f"dim{output_size[0]}"
        output_dir = (
            get_project_root() / "data" / "processed" / dataset_type / dataset /
            "layout_embeddings" / "embeddings" / clean_name / dim_folder
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"visualization_{method}_{data['model_name']}.png"

    # Create visualization
    create_visualization(
        embeddings_2d,
        data['sensor_ids'],
        data['states'],
        metadata,
        method=method,
        show_labels=show_labels,
        output_path=output_path,
        model_name=data['model_name'],
        embedding_dim=data['embedding_dim']
    )

    # Print statistics
    sensor_rooms = [get_sensor_room_from_metadata(sid, metadata) for sid in data['sensor_ids']]
    logger.info("\n=== Statistics ===")
    logger.info(f"Total embeddings: {len(data['embeddings'])}")
    logger.info(f"Unique sensors: {len(set(data['sensor_ids']))}")
    logger.info(f"States: {set(data['states'])}")
    logger.info(f"Unique rooms: {len(set(sensor_rooms))}")
    logger.info(f"Rooms: {sorted(set(sensor_rooms))}")

    # Compute distances
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(data['embeddings'], metric='cosine'))

    # Same sensor, different states
    same_sensor_dists = []
    for i in range(len(data['sensor_ids'])):
        for j in range(i+1, len(data['sensor_ids'])):
            if (data['sensor_ids'][i] == data['sensor_ids'][j] and
                data['states'][i] != data['states'][j]):
                same_sensor_dists.append(distances[i, j])

    if same_sensor_dists:
        logger.info(f"\nSame sensor, different state - Avg distance: {np.mean(same_sensor_dists):.4f}")

    # Same state, different sensors
    same_state_dists = []
    for i in range(len(data['states'])):
        for j in range(i+1, len(data['states'])):
            if data['states'][i] == data['states'][j]:
                same_state_dists.append(distances[i, j])

    if same_state_dists:
        logger.info(f"Same state, different sensor - Avg distance: {np.mean(same_state_dists):.4f}")

    # Same room, different sensors
    same_room_dists = []
    for i in range(len(sensor_rooms)):
        for j in range(i+1, len(sensor_rooms)):
            if sensor_rooms[i] == sensor_rooms[j]:
                same_room_dists.append(distances[i, j])

    if same_room_dists:
        logger.info(f"Same room, different sensor - Avg distance: {np.mean(same_room_dists):.4f}")

    logger.info(f"\n✓ Visualization complete!")
    logger.info(f"✓ Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize image embeddings using dimensionality reduction"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., milan)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="clip",
        help="Model name: 'clip', 'dinov2', 'siglip', or HF model path (default: clip)"
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
        help="Image width (default: 224)"
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=224,
        help="Image height (default: 224)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tsne",
        choices=["tsne", "umap", "pca"],
        help="Dimensionality reduction method (default: tsne)"
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Show point labels on the plot"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for saving plot"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    visualize_image_embeddings(
        dataset=args.dataset,
        model_name=args.model,
        dataset_type=args.dataset_type,
        output_size=(args.output_width, args.output_height),
        method=args.method,
        show_labels=args.show_labels,
        output_dir=output_dir
    )

