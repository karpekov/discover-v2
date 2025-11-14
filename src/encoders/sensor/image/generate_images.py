"""
Generate sensor activation images for image-based sequence encoders.

This module creates images of individual sensor activations on a house layout,
which can then be embedded using vision models (CLIP, DINO, etc.) and used
for image-based sequence encoding.

For each sensor and state combination, we create an image showing:
- The house layout (currently a blank canvas with boundaries)
- A colored circle at the sensor's location
- Color coding based on sensor type and state

Output directory structure:
    data/processed/{dataset_type}/{dataset}/layout_embeddings/
        images/
            {sensor_id}_{state}.png
        embeddings/
            {model_name}/
                embeddings.npz
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger(__name__)


# Color mappings for different sensor types and states
# Format: (R, G, B)
COLOR_MAP = {
    "motion": {
        "ON": (34, 139, 34),      # Forest green
        "OFF": (220, 20, 60),     # Crimson red
        "ABSENT": (220, 20, 60),  # Same as OFF
        "PRESENT": (34, 139, 34), # Same as ON
    },
    "door": {
        "OPEN": (139, 69, 19),    # Saddle brown
        "CLOSE": (128, 128, 128), # Gray
        "ON": (139, 69, 19),      # Same as OPEN
        "OFF": (128, 128, 128),   # Same as CLOSE
    },
    "temperature": {
        "HIGH": (255, 140, 0),    # Dark orange
        "LOW": (70, 130, 180),    # Steel blue
        "NORMAL": (255, 215, 0),  # Gold
        # Temperature sensors may have continuous values, but we'll create discrete bins
        "ON": (255, 215, 0),
        "OFF": (128, 128, 128),
    },
    "light": {
        "ON": (255, 255, 0),      # Bright yellow
        "OFF": (64, 64, 64),      # Dark gray
    },
    "default": {
        "ON": (100, 100, 255),    # Blue
        "OFF": (150, 150, 150),   # Light gray
    }
}


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from src/encoders/sensor/image/ to project root
    return Path(__file__).parent.parent.parent.parent.parent


def load_metadata(dataset_type: str = "casas") -> Dict:
    """Load metadata JSON for the dataset type."""
    metadata_path = get_project_root() / "metadata" / f"{dataset_type}_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    logger.info(f"Loaded metadata from {metadata_path}")
    return metadata


def get_sensor_states(sensor_type: str) -> List[str]:
    """Get possible states for a sensor type."""
    state_map = {
        "motion": ["ON", "OFF"],
        "door": ["OPEN", "CLOSE"],
        "temperature": ["ON", "OFF"],  # Simplified for now
        "light": ["ON", "OFF"],
    }
    return state_map.get(sensor_type, ["ON", "OFF"])


def get_color(sensor_type: str, state: str) -> Tuple[int, int, int]:
    """Get RGB color for sensor type and state."""
    sensor_colors = COLOR_MAP.get(sensor_type, COLOR_MAP["default"])
    return sensor_colors.get(state, sensor_colors.get("ON", (100, 100, 255)))


def load_floor_plan(dataset: str, dataset_type: str = "casas") -> Optional[Image.Image]:
    """
    Load floor plan image for a dataset.

    Args:
        dataset: Dataset name (e.g., "milan", "aruba")
        dataset_type: Dataset type (default: "casas")

    Returns:
        PIL Image of floor plan, or None if not found
    """
    project_root = get_project_root()
    floor_plan_path = project_root / "metadata" / "floor_plans_augmented" / f"{dataset}.png"

    if floor_plan_path.exists():
        logger.info(f"Loading floor plan from {floor_plan_path}")
        return Image.open(floor_plan_path).convert('RGB')
    else:
        logger.warning(f"Floor plan not found at {floor_plan_path}")
        return None


def calculate_canvas_size(
    coordinates: Dict[str, List[int]],
    padding: int = 100,
    floor_plan: Optional[Image.Image] = None
) -> Tuple[int, int]:
    """
    Calculate canvas size based on floor plan or sensor coordinates.

    Args:
        coordinates: Sensor coordinates dictionary
        padding: Padding around the edges (only used if no floor plan)
        floor_plan: Floor plan image (if available)

    Returns:
        (width, height) tuple
    """
    if floor_plan is not None:
        # Use floor plan dimensions
        return floor_plan.size
    else:
        # Calculate from coordinates with padding
        all_x = [coord[0] for coord in coordinates.values()]
        all_y = [coord[1] for coord in coordinates.values()]

        width = max(all_x) + 2 * padding
        height = max(all_y) + 2 * padding

        return width, height


def draw_house_background(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    padding: int = 100
) -> None:
    """Draw a simple house background (boundary box) - only used if no floor plan."""
    # Draw outer boundary
    draw.rectangle(
        [(padding // 2, padding // 2), (width - padding // 2, height - padding // 2)],
        outline=(200, 200, 200),
        width=3
    )


def resize_with_padding(img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Resize image to target size with padding to maintain aspect ratio.

    Args:
        img: Input image
        target_size: Target (width, height), default is 224x224 for CLIP

    Returns:
        Resized and padded image
    """
    target_w, target_h = target_size
    original_w, original_h = img.size

    # Calculate scaling factor to fit within target size
    scale = min(target_w / original_w, target_h / original_h)

    # Calculate new dimensions
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    # Resize the image
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create a new image with target size and white background
    result = Image.new('RGB', target_size, color=(255, 255, 255))

    # Calculate position to paste (center the image)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2

    # Paste the resized image onto the padded canvas
    result.paste(img_resized, (paste_x, paste_y))

    return result


def create_sensor_image(
    sensor_id: str,
    state: str,
    coordinate: Tuple[int, int],
    sensor_type: str,
    canvas_size: Tuple[int, int],
    padding: int = 100,
    circle_radius: int = 50,
    show_label: bool = False,
    output_size: Optional[Tuple[int, int]] = (224, 224),
    floor_plan: Optional[Image.Image] = None
) -> Image.Image:
    """
    Create an image showing a single sensor activation.

    Args:
        sensor_id: Sensor identifier (e.g., "M001", "D002")
        state: Sensor state (e.g., "ON", "OFF", "OPEN", "CLOSE")
        coordinate: (x, y) position of the sensor
        sensor_type: Type of sensor (motion, door, temperature, etc.)
        canvas_size: (width, height) of the canvas
        padding: Padding around the edges (only used if no floor plan)
        circle_radius: Radius of the sensor marker circle (default: 50)
        show_label: Whether to show sensor ID label on the image
        output_size: Final output size (width, height), default 224x224 for CLIP.
                    Set to None to keep original size.
        floor_plan: Floor plan image to use as background (if available)

    Returns:
        PIL Image object
    """
    # Use floor plan as background if available, otherwise create blank canvas
    if floor_plan is not None:
        # Copy the floor plan to avoid modifying the original
        img = floor_plan.copy()
        x = coordinate[0]
        y = coordinate[1]
    else:
        # Create blank white canvas
        img = Image.new('RGB', canvas_size, color=(255, 255, 255))
        # Adjust coordinates with padding
        x = coordinate[0] + padding // 2
        y = coordinate[1] + padding // 2

    draw = ImageDraw.ImageDraw(img)

    # Draw house background if no floor plan
    if floor_plan is None:
        draw_house_background(draw, canvas_size[0], canvas_size[1], padding)

    # Get color for this sensor type and state
    color = get_color(sensor_type, state)

    # Draw the sensor as a filled circle (no outline)
    draw.ellipse(
        [(x - circle_radius, y - circle_radius),
         (x + circle_radius, y + circle_radius)],
        fill=color,
        outline=None
    )

    # Optionally add label
    if show_label:
        try:
            # Try to use a font, fall back to default if not available
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()

        label = f"{sensor_id}"
        # Get text bounding box
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw text below the circle
        text_x = x - text_width // 2
        text_y = y + circle_radius + 5
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    # Resize with padding if output_size is specified
    if output_size is not None:
        img = resize_with_padding(img, output_size)

    return img


def generate_dataset_images(
    dataset: str,
    dataset_type: str = "casas",
    output_dir: Optional[Path] = None,
    circle_radius: int = 200,
    show_labels: bool = False,
    include_off_states: bool = True,
    output_size: Optional[Tuple[int, int]] = (224, 224)
) -> Dict[str, Path]:
    """
    Generate all sensor activation images for a dataset.

    Args:
        dataset: Dataset name (e.g., "milan", "aruba")
        dataset_type: Dataset type (e.g., "casas", "marble")
        output_dir: Output directory (default: data/processed/{dataset_type}/{dataset}/layout_embeddings/images)
        circle_radius: Radius of sensor markers (default: 50 pixels)
        show_labels: Whether to show sensor IDs on images
        include_off_states: Whether to generate images for OFF states
        output_size: Final output size (width, height), default 224x224 for CLIP.
                    Set to None to keep original size without resizing.

    Returns:
        Dictionary mapping image keys to file paths
    """
    # Load metadata
    metadata = load_metadata(dataset_type)

    if dataset not in metadata:
        raise ValueError(f"Dataset '{dataset}' not found in metadata. Available: {list(metadata.keys())}")

    dataset_meta = metadata[dataset]

    # Extract sensor information
    sensor_coords = dataset_meta.get("sensor_coordinates", {})
    sensor_types = dataset_meta.get("sensor_type", {})

    if not sensor_coords:
        raise ValueError(f"No sensor coordinates found for dataset '{dataset}'")

    logger.info(f"Found {len(sensor_coords)} sensors for {dataset}")

    # Load floor plan if available
    floor_plan = load_floor_plan(dataset, dataset_type)

    # Calculate canvas size (from floor plan or coordinates)
    canvas_size = calculate_canvas_size(sensor_coords, floor_plan=floor_plan)
    logger.info(f"Canvas size: {canvas_size}")

    # Set up output directory
    if output_dir is None:
        base_dir = get_project_root() / "data" / "processed" / dataset_type / dataset / "layout_embeddings" / "images"
        # Add dimension-specific subfolder
        if output_size is not None:
            dim_folder = f"dim{output_size[0]}"  # e.g., "dim224", "dim512"
            output_dir = base_dir / dim_folder
        else:
            output_dir = base_dir / "original"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Generate images
    image_paths = {}
    total_images = 0

    for sensor_id, coord in sensor_coords.items():
        sensor_type = sensor_types.get(sensor_id, "default").lower()
        states = get_sensor_states(sensor_type)

        if not include_off_states:
            # Only generate ON/OPEN/PRESENT states
            states = [s for s in states if s in ["ON", "OPEN", "PRESENT", "HIGH"]]

        for state in states:
            # Create image
            img = create_sensor_image(
                sensor_id=sensor_id,
                state=state,
                coordinate=coord,
                sensor_type=sensor_type,
                canvas_size=canvas_size,
                circle_radius=circle_radius,
                show_label=show_labels,
                output_size=output_size,
                floor_plan=floor_plan
            )

            # Save image
            filename = f"{sensor_id}_{state}.png"
            filepath = output_dir / filename
            img.save(filepath)

            # Store path
            image_key = f"{sensor_id}_{state}"
            image_paths[image_key] = filepath
            total_images += 1

            if total_images % 20 == 0:
                logger.info(f"Generated {total_images} images...")

    logger.info(f"Successfully generated {total_images} images for {dataset}")

    # Save metadata about generated images
    metadata_file = output_dir / "image_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            "dataset": dataset,
            "dataset_type": dataset_type,
            "num_images": total_images,
            "canvas_size": canvas_size,
            "output_size": output_size,
            "circle_radius": circle_radius,
            "show_labels": show_labels,
            "include_off_states": include_off_states,
            "sensors": list(sensor_coords.keys()),
            "image_keys": list(image_paths.keys())
        }, f, indent=2)

    logger.info(f"Saved image metadata to {metadata_file}")

    return image_paths


def get_image_path(
    dataset: str,
    sensor_id: str,
    state: str,
    dataset_type: str = "casas",
    output_size: Optional[Tuple[int, int]] = (224, 224)
) -> Path:
    """
    Get the path to a specific sensor activation image.

    Args:
        dataset: Dataset name (e.g., "milan")
        sensor_id: Sensor identifier (e.g., "M001")
        state: Sensor state (e.g., "ON", "OFF")
        dataset_type: Dataset type (default: "casas")
        output_size: Output size to look for (default: (224, 224))

    Returns:
        Path to the image file
    """
    project_root = get_project_root()
    base_dir = project_root / "data" / "processed" / dataset_type / dataset / "layout_embeddings" / "images"

    # Determine dimension subfolder
    if output_size is not None:
        dim_folder = f"dim{output_size[0]}"
        image_dir = base_dir / dim_folder
    else:
        image_dir = base_dir / "original"

    return image_dir / f"{sensor_id}_{state}.png"


def load_image_metadata(
    dataset: str,
    dataset_type: str = "casas",
    output_size: Optional[Tuple[int, int]] = (224, 224)
) -> Dict:
    """
    Load metadata about generated images for a dataset.

    Args:
        dataset: Dataset name
        dataset_type: Dataset type
        output_size: Output size to look for (default: (224, 224))

    Returns:
        Dictionary with image metadata
    """
    project_root = get_project_root()
    base_dir = project_root / "data" / "processed" / dataset_type / dataset / "layout_embeddings" / "images"

    # Determine dimension subfolder
    if output_size is not None:
        dim_folder = f"dim{output_size[0]}"
        metadata_file = base_dir / dim_folder / "image_metadata.json"
    else:
        metadata_file = base_dir / "original" / "image_metadata.json"

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Image metadata not found at {metadata_file}. "
            f"Please generate images first using generate_dataset_images()."
        )

    with open(metadata_file, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    # Configure logging for command-line usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Generate sensor activation images for image-based encoders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 224x224 images for Milan dataset (default)
  python -m src.encoders.sensor.image.generate_images --dataset milan

  # Generate with labels shown
  python -m src.encoders.sensor.image.generate_images --dataset milan --show-labels

  # Generate only ON states (skip OFF states)
  python -m src.encoders.sensor.image.generate_images --dataset aruba --no-off-states

  # Keep original size (don't resize to 224x224)
  python -m src.encoders.sensor.image.generate_images --dataset milan --no-resize

  # Custom output size
  python -m src.encoders.sensor.image.generate_images --dataset cairo --output-width 512 --output-height 512

  # Larger sensor markers
  python -m src.encoders.sensor.image.generate_images --dataset milan --circle-radius 50
        """
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., milan, aruba, cairo)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="casas",
        help="Dataset type (default: casas)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/processed/{dataset_type}/{dataset}/layout_embeddings/images)"
    )
    parser.add_argument(
        "--circle-radius",
        type=int,
        default=100,
        help="Radius of sensor marker circles in pixels (default: 100)"
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Show sensor ID labels on images"
    )
    parser.add_argument(
        "--no-off-states",
        action="store_true",
        help="Don't generate images for OFF/CLOSE states (only ON/OPEN)"
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Don't resize images (keep original canvas size)"
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=224,
        help="Output image width in pixels (default: 224)"
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=224,
        help="Output image height in pixels (default: 224)"
    )

    args = parser.parse_args()

    # Determine output size
    if args.no_resize:
        output_size = None
    else:
        output_size = (args.output_width, args.output_height)

    output_dir = Path(args.output_dir) if args.output_dir else None

    logger.info(f"Generating sensor images for dataset: {args.dataset}")
    logger.info(f"Dataset type: {args.dataset_type}")
    logger.info(f"Output size: {output_size if output_size else 'original'}")
    logger.info(f"Circle radius: {args.circle_radius} pixels")
    logger.info(f"Show labels: {args.show_labels}")
    logger.info(f"Include OFF states: {not args.no_off_states}")

    # Generate images
    try:
        image_paths = generate_dataset_images(
            dataset=args.dataset,
            dataset_type=args.dataset_type,
            output_dir=output_dir,
            circle_radius=args.circle_radius,
            show_labels=args.show_labels,
            include_off_states=not args.no_off_states,
            output_size=output_size
        )

        logger.info(f"✓ Successfully generated {len(image_paths)} images")
        logger.info(f"✓ Images saved to: {list(image_paths.values())[0].parent}")

    except Exception as e:
        logger.error(f"✗ Failed to generate images: {e}")
        raise

