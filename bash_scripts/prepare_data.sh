#!/bin/bash
# Comprehensive Data Preparation Pipeline
# Sample → Caption → Encode in one command
#
# This script automates the entire data preparation pipeline:
#   1. Sample data using sampling configs
#   2. Generate captions using caption configs
#   3. Encode captions using text encoder configs
#
# Usage:
#   bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60
#   bash bash_scripts/prepare_data.sh --dataset aruba --sampling FL_20 --caption-style sourish
#   bash bash_scripts/prepare_data.sh --dataset cairo --sampling FD_60 --text-encoder gte_base
#   bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60 --skip-sampling
#   bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60 --no-presegmented
#
# Arguments:
#   --dataset         Dataset name (milan, aruba, cairo) [REQUIRED]
#   --sampling        Sampling strategy (FD_60, FL_20, etc.) [REQUIRED]
#   --caption-style   Caption style (baseline, sourish) [default: baseline]
#   --text-encoder    Text encoder (clip_vit_base, gte_base, minilm_l6, etc.) [default: clip_vit_base]
#   --no-presegmented Skip the presegmented (_p) version
#   --skip-sampling   Skip data sampling step (if data already exists)
#   --skip-captions   Skip caption generation step (if captions already exist)
#   --skip-encoding   Skip text encoding step (if embeddings already exist)
#   --device          Device for text encoding (cpu, cuda, mps) [default: auto]
#   --help            Show this help message

set -e  # Exit on error

# ============================================================================
# Parse Arguments
# ============================================================================

DATASET=""
SAMPLING=""
CAPTION_STYLE="baseline"
TEXT_ENCODER="clip_vit_base"
PRESEGMENTED=true
SKIP_SAMPLING=false
SKIP_CAPTIONS=false
SKIP_ENCODING=false
DEVICE="auto"

show_help() {
    echo "Usage: bash bash_scripts/prepare_data.sh [OPTIONS]"
    echo ""
    echo "Required Arguments:"
    echo "  --dataset DATASET         Dataset name (milan, aruba, cairo)"
    echo "  --sampling SAMPLING       Sampling strategy (FD_60, FL_20, FD_30, FD_120, FL_50)"
    echo ""
    echo "Optional Arguments:"
    echo "  --caption-style STYLE     Caption style (baseline, sourish) [default: baseline]"
    echo "  --text-encoder ENCODER    Text encoder config name [default: clip_vit_base]"
    echo "                            Options: clip_vit_base, gte_base, gte_base_projected,"
    echo "                                     distilroberta_base, minilm_l6, siglip_base,"
    echo "                                     llama_embed_8b, embeddinggemma_300m"
    echo "  --no-presegmented         Skip presegmented (_p) version"
    echo "  --skip-sampling           Skip data sampling (if data already exists)"
    echo "  --skip-captions           Skip caption generation (if captions already exist)"
    echo "  --skip-encoding           Skip text encoding (if embeddings already exist)"
    echo "  --device DEVICE           Device for encoding (cpu, cuda, mps) [default: auto]"
    echo "  --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Full pipeline for Milan FD_60"
    echo "  bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60"
    echo ""
    echo "  # Aruba with Sourish captions"
    echo "  bash bash_scripts/prepare_data.sh --dataset aruba --sampling FD_60 --caption-style sourish"
    echo ""
    echo "  # Use GTE text encoder instead of CLIP"
    echo "  bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60 --text-encoder gte_base"
    echo ""
    echo "  # Skip sampling (data already exists), only regenerate captions and embeddings"
    echo "  bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60 --skip-sampling"
    echo ""
    echo "  # Only do regular version, not presegmented"
    echo "  bash bash_scripts/prepare_data.sh --dataset milan --sampling FD_60 --no-presegmented"
    echo ""
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --sampling)
            SAMPLING="$2"
            shift 2
            ;;
        --caption-style)
            CAPTION_STYLE="$2"
            shift 2
            ;;
        --text-encoder)
            TEXT_ENCODER="$2"
            shift 2
            ;;
        --no-presegmented)
            PRESEGMENTED=false
            shift
            ;;
        --skip-sampling)
            SKIP_SAMPLING=true
            shift
            ;;
        --skip-captions)
            SKIP_CAPTIONS=true
            shift
            ;;
        --skip-encoding)
            SKIP_ENCODING=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATASET" ] || [ -z "$SAMPLING" ]; then
    echo "Error: --dataset and --sampling are required!"
    echo ""
    show_help
fi

# Validate dataset
if [[ ! "$DATASET" =~ ^(milan|aruba|cairo|kyoto)$ ]]; then
    echo "Error: Dataset must be one of: milan, aruba, cairo, kyoto"
    exit 1
fi

# Validate caption style
if [[ ! "$CAPTION_STYLE" =~ ^(baseline|sourish)$ ]]; then
    echo "Error: Caption style must be one of: baseline, sourish"
    exit 1
fi

# ============================================================================
# Setup
# ============================================================================

echo "==================================================================="
echo "Data Preparation Pipeline"
echo "==================================================================="
echo "Dataset:        $DATASET"
echo "Sampling:       $SAMPLING"
echo "Caption Style:  $CAPTION_STYLE"
echo "Text Encoder:   $TEXT_ENCODER"
echo "Presegmented:   $PRESEGMENTED"
echo "Skip Sampling:  $SKIP_SAMPLING"
echo "Skip Captions:  $SKIP_CAPTIONS"
echo "Skip Encoding:  $SKIP_ENCODING"
echo "Device:         $DEVICE"
echo "==================================================================="
echo ""

# Activate conda environment
echo "Activating conda environment: discover-v2-env"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate discover-v2-env

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# Determine which configs to run
CONFIGS=("${DATASET}_${SAMPLING}")
if [ "$PRESEGMENTED" = true ]; then
    CONFIGS+=("${DATASET}_${SAMPLING}_p")
fi

echo "Will process: ${CONFIGS[@]}"
echo ""

# ============================================================================
# STEP 1: Data Sampling
# ============================================================================

if [ "$SKIP_SAMPLING" = false ]; then
    echo "==================================================================="
    echo "STEP 1: Data Sampling"
    echo "==================================================================="

    for config in "${CONFIGS[@]}"; do
        config_file="configs/sampling/${config}.yaml"

        if [ ! -f "$config_file" ]; then
            echo "⚠️  Warning: Config file not found: $config_file"
            echo "   Skipping $config"
            continue
        fi

        echo ""
        echo "-------------------------------------------------------------------"
        echo "Sampling: $config"
        echo "Config: $config_file"
        echo "-------------------------------------------------------------------"

        python sample_data.py --config "$config_file"

        if [ $? -eq 0 ]; then
            echo "✓ Sampling completed: $config"
        else
            echo "✗ Sampling failed: $config"
            exit 1
        fi
    done

    echo ""
    echo "✓ All sampling completed!"
else
    echo "==================================================================="
    echo "STEP 1: Data Sampling [SKIPPED]"
    echo "==================================================================="
fi

# ============================================================================
# STEP 2: Caption Generation
# ============================================================================

if [ "$SKIP_CAPTIONS" = false ]; then
    echo ""
    echo "==================================================================="
    echo "STEP 2: Caption Generation"
    echo "==================================================================="

    # Determine caption config file
    CAPTION_CONFIG="configs/captions/${CAPTION_STYLE}_${DATASET}.yaml"

    if [ ! -f "$CAPTION_CONFIG" ]; then
        echo "⚠️  Warning: Caption config not found: $CAPTION_CONFIG"
        echo "   Will use command-line arguments instead"
        USE_CAPTION_CONFIG=false
    else
        echo "Using caption config: $CAPTION_CONFIG"
        USE_CAPTION_CONFIG=true
    fi

    for config in "${CONFIGS[@]}"; do
        # Extract directory name (e.g., FD_60, FD_60_p)
        if [[ "$config" == *"_p" ]]; then
            dir="${SAMPLING}_p"
        else
            dir="${SAMPLING}"
        fi

        data_dir="data/processed/casas/${DATASET}/${dir}"

        # Check if data directory exists
        if [ ! -d "$data_dir" ]; then
            echo "⚠️  Warning: Data directory not found: $data_dir"
            echo "   Skipping caption generation for $config"
            continue
        fi

        echo ""
        echo "-------------------------------------------------------------------"
        echo "Generating captions for: $config"
        echo "Data dir: $data_dir"
        echo "-------------------------------------------------------------------"

        if [ "$USE_CAPTION_CONFIG" = true ]; then
            # Use config file
            python src/captions/generate_captions.py \
                --config "$CAPTION_CONFIG" \
                --data-dir "$data_dir" \
                --split all
        else
            # Use command-line arguments
            python src/captions/generate_captions.py \
                --data-dir "$data_dir" \
                --caption-style "$CAPTION_STYLE" \
                --dataset-name "$DATASET" \
                --split all
        fi

        if [ $? -eq 0 ]; then
            echo "✓ Caption generation completed: $config"
        else
            echo "✗ Caption generation failed: $config"
            exit 1
        fi
    done

    echo ""
    echo "✓ All caption generation completed!"
else
    echo ""
    echo "==================================================================="
    echo "STEP 2: Caption Generation [SKIPPED]"
    echo "==================================================================="
fi

# ============================================================================
# STEP 3: Text Encoding
# ============================================================================

if [ "$SKIP_ENCODING" = false ]; then
    echo ""
    echo "==================================================================="
    echo "STEP 3: Text Encoding"
    echo "==================================================================="

    TEXT_ENCODER_CONFIG="configs/text_encoders/${TEXT_ENCODER}.yaml"

    if [ ! -f "$TEXT_ENCODER_CONFIG" ]; then
        echo "Error: Text encoder config not found: $TEXT_ENCODER_CONFIG"
        exit 1
    fi

    echo "Using text encoder config: $TEXT_ENCODER_CONFIG"

    for config in "${CONFIGS[@]}"; do
        # Extract directory name
        if [[ "$config" == *"_p" ]]; then
            dir="${SAMPLING}_p"
        else
            dir="${SAMPLING}"
        fi

        data_dir="data/processed/casas/${DATASET}/${dir}"

        # Check if data directory exists
        if [ ! -d "$data_dir" ]; then
            echo "⚠️  Warning: Data directory not found: $data_dir"
            echo "   Skipping text encoding for $config"
            continue
        fi

        echo ""
        echo "-------------------------------------------------------------------"
        echo "Encoding captions: $config with $TEXT_ENCODER"
        echo "Data dir: $data_dir"
        echo "-------------------------------------------------------------------"

        # Build command
        CMD="python src/text_encoders/encode_captions.py \
            --data-dir \"$data_dir\" \
            --caption-style \"$CAPTION_STYLE\" \
            --split all \
            --config \"$TEXT_ENCODER_CONFIG\""

        # Add device if specified
        if [ "$DEVICE" != "auto" ]; then
            CMD="$CMD --device $DEVICE"
        fi

        # Execute command
        eval $CMD

        if [ $? -eq 0 ]; then
            echo "✓ Text encoding completed: $config"
        else
            echo "✗ Text encoding failed: $config"
            exit 1
        fi
    done

    echo ""
    echo "✓ All text encoding completed!"
else
    echo ""
    echo "==================================================================="
    echo "STEP 3: Text Encoding [SKIPPED]"
    echo "==================================================================="
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "==================================================================="
echo "DATA PREPARATION PIPELINE COMPLETED! 🎉"
echo "==================================================================="
echo ""
echo "Summary:"
echo "  Dataset:        $DATASET"
echo "  Sampling:       $SAMPLING"
if [ "$PRESEGMENTED" = true ]; then
    echo "  Versions:       ${SAMPLING}, ${SAMPLING}_p"
else
    echo "  Versions:       ${SAMPLING}"
fi
echo "  Caption Style:  $CAPTION_STYLE"
echo "  Text Encoder:   $TEXT_ENCODER"
echo ""
echo "Data location:"
echo "  Sampled data:   data/processed/casas/${DATASET}/${SAMPLING}/"
echo "  Captions:       data/processed/casas/${DATASET}/${SAMPLING}/*_captions_${CAPTION_STYLE}.json"
echo "  Embeddings:     data/processed/casas/${DATASET}/${SAMPLING}/*_embeddings_${CAPTION_STYLE}_*.npz"
echo ""
echo "Next steps:"
echo "  - Train sensor encoder (Step 2)"
echo "  - Train alignment model (Step 5)"
echo "==================================================================="

