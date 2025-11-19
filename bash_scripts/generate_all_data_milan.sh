#!/bin/bash
# Script to generate all Milan dataset variations
# - Sampling: FL_20, FL_20_p, FL_50, FL_50_p, FD_30, FD_30_p, FD_60, FD_60_p, FD_120, FD_120_p
# - Captions: baseline style for train, val, test
# - Text embeddings: CLIP and MiniLM models

set -e  # Exit on error

# Activate conda environment
echo "==================================================================="
echo "Activating conda environment: discover-v2-env"
echo "==================================================================="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate discover-v2-env

# Navigate to project root
cd "$(dirname "$0")/.."

# ============================================================================
# STEP 1: Data Sampling
# ============================================================================
echo ""
echo "==================================================================="
echo "STEP 1: Generating sampled data for Milan"
echo "==================================================================="

# Fixed-Length sampling configs
FL_CONFIGS=("milan_FL_20" "milan_FL_20_p" "milan_FL_50" "milan_FL_50_p")

# Fixed-Duration sampling configs
FD_CONFIGS=("milan_FD_30" "milan_FD_30_p" "milan_FD_60" "milan_FD_60_p" "milan_FD_120" "milan_FD_120_p")

# Combine all configs
ALL_CONFIGS=("${FL_CONFIGS[@]}" "${FD_CONFIGS[@]}")

# Run sampling for each config
for config in "${ALL_CONFIGS[@]}"; do
    echo ""
    echo "-------------------------------------------------------------------"
    echo "Sampling: $config"
    echo "-------------------------------------------------------------------"
    python sample_data.py --config "configs/sampling/${config}.yaml"
done

echo ""
echo "✓ All sampling completed!"

# ============================================================================
# STEP 2: Caption Generation (Baseline style)
# ============================================================================
echo ""
echo "==================================================================="
echo "STEP 2: Generating baseline captions for all datasets"
echo "==================================================================="

# Map configs to their output directories
declare -A CONFIG_TO_DIR=(
    ["milan_FL_20"]="FL_20"
    ["milan_FL_20_p"]="FL_20_p"
    ["milan_FL_50"]="FL_50"
    ["milan_FL_50_p"]="FL_50_p"
    ["milan_FD_30"]="FD_30"
    ["milan_FD_30_p"]="FD_30_p"
    ["milan_FD_60"]="FD_60"
    ["milan_FD_60_p"]="FD_60_p"
    ["milan_FD_120"]="FD_120"
    ["milan_FD_120_p"]="FD_120_p"
)

# Generate captions for each dataset (train, val, test)
for config in "${ALL_CONFIGS[@]}"; do
    dir="${CONFIG_TO_DIR[$config]}"
    data_dir="data/processed/casas/milan/${dir}"

    echo ""
    echo "-------------------------------------------------------------------"
    echo "Generating captions for: $config ($data_dir)"
    echo "-------------------------------------------------------------------"
    python src/captions/generate_captions.py \
        --data-dir "$data_dir" \
        --caption-style baseline \
        --dataset-name milan \
        --split all
done

echo ""
echo "✓ All caption generation completed!"

# ============================================================================
# STEP 3: Text Embeddings (CLIP and MiniLM)
# ============================================================================
echo ""
echo "==================================================================="
echo "STEP 3: Generating text embeddings (CLIP and MiniLM)"
echo "==================================================================="

# Text encoder configs
TEXT_ENCODERS=("clip_vit_base" "minilm_l6")

# Generate embeddings for each dataset and encoder
for config in "${ALL_CONFIGS[@]}"; do
    dir="${CONFIG_TO_DIR[$config]}"
    data_dir="data/processed/casas/milan/${dir}"

    for encoder in "${TEXT_ENCODERS[@]}"; do
        echo ""
        echo "-------------------------------------------------------------------"
        echo "Encoding: $config with $encoder"
        echo "-------------------------------------------------------------------"
        python src/text_encoders/encode_captions.py \
            --data-dir "$data_dir" \
            --caption-style baseline \
            --split all \
            --config "configs/text_encoders/${encoder}.yaml"
    done
done

echo ""
echo "✓ All text embedding generation completed!"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "==================================================================="
echo "ALL DATA GENERATION COMPLETED!"
echo "==================================================================="
echo ""
echo "Generated data for:"
echo "  - 10 sampling strategies (FL_20, FL_20_p, FL_50, FL_50_p, FD_30, FD_30_p, FD_60, FD_60_p, FD_120, FD_120_p)"
echo "  - Baseline captions (train, val, test splits)"
echo "  - Text embeddings (CLIP and MiniLM models)"
echo ""
echo "Data location: data/processed/casas/milan/"
echo "==================================================================="

