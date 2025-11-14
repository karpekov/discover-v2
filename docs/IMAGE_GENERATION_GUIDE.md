# Sensor Image Generation Guide

This guide explains how to generate sensor activation images for image-based sequence encoders (Step 2b).

## Overview

The image generation module creates visual representations of individual sensor activations on a house layout. Each sensor-state combination produces a single image showing:
- House boundary
- A colored circle at the sensor's location
- Color coding based on sensor type and state

These images can then be embedded using vision models (CLIP, DINO, SigLIP, etc.) and used for image-based sequence encoding.

## Directory Structure

```
data/processed/{dataset_type}/{dataset}/layout_embeddings/
├── images/
│   ├── M001_ON.png
│   ├── M001_OFF.png
│   ├── D001_OPEN.png
│   ├── D001_CLOSE.png
│   └── image_metadata.json
└── embeddings/
    ├── clip/
    │   └── embeddings.npz
    ├── dino/
    │   └── embeddings.npz
    └── siglip/
        └── embeddings.npz
```

## Color Coding

### Motion Sensors
- **ON/PRESENT**: Forest green (34, 139, 34)
- **OFF/ABSENT**: Crimson red (220, 20, 60)

### Door Sensors
- **OPEN**: Saddle brown (139, 69, 19)
- **CLOSE**: Gray (128, 128, 128)

### Temperature Sensors
- **ON/NORMAL**: Gold (255, 215, 0)
- **OFF**: Gray (128, 128, 128)
- **HIGH**: Dark orange (255, 140, 0)
- **LOW**: Steel blue (70, 130, 180)

### Light Sensors
- **ON**: Bright yellow (255, 255, 0)
- **OFF**: Dark gray (64, 64, 64)

## Command-Line Usage

### Basic Usage

```bash
# Generate images for Milan dataset
python generate_sensor_images.py --dataset milan

# Generate images for Aruba dataset
python generate_sensor_images.py --dataset aruba
```

### With Options

```bash
# Show sensor ID labels on images
python generate_sensor_images.py --dataset milan --show-labels

# Only generate ON states (skip OFF states)
python generate_sensor_images.py --dataset aruba --no-off-states

# Custom circle radius (larger markers)
python generate_sensor_images.py --dataset cairo --circle-radius 50

# Custom output directory
python generate_sensor_images.py --dataset milan --output-dir /custom/path
```

### All Options

- `--dataset`: Dataset name (required) - e.g., milan, aruba, cairo
- `--dataset-type`: Dataset type (default: casas)
- `--output-dir`: Custom output directory
- `--circle-radius`: Radius of sensor markers in pixels (default: 30)
- `--show-labels`: Show sensor ID labels on images
- `--no-off-states`: Only generate ON/OPEN states

## Programmatic Usage

### Import and Generate

```python
from src.encoders.sensor.image import generate_dataset_images

# Generate all images for a dataset
image_paths = generate_dataset_images(
    dataset="milan",
    dataset_type="casas",
    circle_radius=30,
    show_labels=False,
    include_off_states=True
)

# Returns: Dict[str, Path] mapping "SENSOR_STATE" to file path
# Example: {"M001_ON": Path(...), "M001_OFF": Path(...), ...}
```

### Get Path to Specific Image

```python
from src.encoders.sensor.image import get_image_path

# Get path to a specific sensor-state image
image_path = get_image_path(
    dataset="milan",
    sensor_id="M001",
    state="ON",
    dataset_type="casas"
)
```

### Load Image Metadata

```python
from src.encoders.sensor.image import load_image_metadata

# Load metadata about generated images
metadata = load_image_metadata(dataset="milan", dataset_type="casas")

print(f"Total images: {metadata['num_images']}")
print(f"Canvas size: {metadata['canvas_size']}")
print(f"Sensors: {metadata['sensors']}")
print(f"Image keys: {metadata['image_keys']}")
```

## Integration with Training Pipeline

The image generation is designed to be called automatically by training scripts when needed:

```python
from src.encoders.sensor.image import generate_dataset_images, load_image_metadata
from pathlib import Path

def get_or_generate_images(dataset: str, dataset_type: str = "casas"):
    """Get existing images or generate them if they don't exist."""
    try:
        # Try to load existing metadata
        metadata = load_image_metadata(dataset, dataset_type)
        print(f"Found {metadata['num_images']} existing images")
        return metadata
    except FileNotFoundError:
        # Generate images if they don't exist
        print(f"Generating images for {dataset}...")
        image_paths = generate_dataset_images(dataset, dataset_type)
        return load_image_metadata(dataset, dataset_type)

# Use in training script
metadata = get_or_generate_images("milan")
```

## Examples

### Example 1: Generate for Multiple Datasets

```python
from src.encoders.sensor.image import generate_dataset_images

datasets = ["milan", "aruba", "cairo"]

for dataset in datasets:
    print(f"Generating images for {dataset}...")
    image_paths = generate_dataset_images(
        dataset=dataset,
        dataset_type="casas",
        include_off_states=True
    )
    print(f"  Generated {len(image_paths)} images")
```

### Example 2: Custom Output Location

```python
from pathlib import Path
from src.encoders.sensor.image import generate_dataset_images

custom_dir = Path("/tmp/my_sensor_images")

image_paths = generate_dataset_images(
    dataset="milan",
    output_dir=custom_dir,
    show_labels=True
)
```

### Example 3: Larger Markers with Labels

```python
from src.encoders.sensor.image import generate_dataset_images

# Generate high-visibility images for visualization
image_paths = generate_dataset_images(
    dataset="milan",
    circle_radius=50,
    show_labels=True
)
```

## Output Format

### Image Files
- **Format**: PNG
- **Size**: Variable (depends on house layout + padding)
- **Naming**: `{sensor_id}_{state}.png`
  - Examples: `M001_ON.png`, `D002_OPEN.png`, `T001_OFF.png`

### Metadata File (`image_metadata.json`)

```json
{
  "dataset": "milan",
  "dataset_type": "casas",
  "num_images": 60,
  "canvas_size": [2653, 2707],
  "circle_radius": 30,
  "show_labels": false,
  "include_off_states": true,
  "sensors": ["D001", "D002", "M001", ...],
  "image_keys": ["D001_OPEN", "D001_CLOSE", "M001_ON", ...]
}
```

## Typical Counts

For reference, here are approximate image counts for common datasets:

- **Milan**: ~60 images (30 sensors × 2 states)
- **Aruba**: ~74 images (37 sensors × 2 states)
- **Cairo**: ~54 images (27 sensors × 2 states)

## Next Steps

Once images are generated, the next step is to:

1. **Embed images** using vision models (CLIP, DINO, etc.)
   - See `src/encoders/sensor/image/embed_images.py` (coming soon)

2. **Use in image-based encoder**
   - The encoder will look up pre-computed embeddings for each sensor activation
   - Feed embeddings into a transformer for sequence processing
   - See `src/encoders/sensor/image/encoder.py` (coming soon)

3. **Train alignment model**
   - Use the image-based encoder in place of the sequence encoder
   - Follow the same alignment training process as Step 5

## Troubleshooting

### FileNotFoundError: Metadata file not found
- Make sure the metadata JSON file exists at `metadata/casas_metadata.json`
- Check that the dataset name matches exactly (case-sensitive)

### No sensor coordinates found
- Verify the dataset has `sensor_coordinates` in the metadata JSON
- Some datasets may not have coordinate information available

### Images look too small/large
- Adjust `--circle-radius` to change marker size
- Default is 30 pixels, try 20-50 for different sizes

### Want to regenerate images
- Simply delete the `layout_embeddings/images/` directory and run again
- Or specify a different output directory

## Related Documentation

- [Encoder Guide](ENCODER_GUIDE.md) - Overview of all encoder types
- [Step 2 Encoder Summary](STEP2_ENCODER_SUMMARY.md) - Implementation details
- [Alignment Guide](ALIGNMENT_GUIDE.md) - Training with encoders

