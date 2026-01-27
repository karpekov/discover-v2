#!/bin/bash
#SBATCH -o slurm/scan_output_%j.txt
#SBATCH -e slurm/scan_error_%j.txt
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node 1
#SBATCH -J dv2-scan
#SBATCH -p rail-lab

# SCAN Clustering Training Script for Skynet
#
# Usage:
#   sbatch bash_scripts/skynet_run_scan.sh <pretrained_model> <output_dir> [num_clusters] [max_epochs]
#
# Examples:
#   # Basic usage (20 clusters, 40 epochs by default)
#   sbatch bash_scripts/skynet_run_scan.sh \
#       trained_models/milan/milan_fl20_seq_discover_v1_mlm_only \
#       trained_models/milan/scan_fl20_20cl_discover_v1
#
#   # With custom number of clusters
#   sbatch bash_scripts/skynet_run_scan.sh \
#       trained_models/milan/milan_fl20_seq_discover_v1_mlm_only \
#       trained_models/milan/scan_fl20_50cl_discover_v1 \
#       50
#
#   # With custom clusters and epochs
#   sbatch bash_scripts/skynet_run_scan.sh \
#       trained_models/milan/milan_fl20_seq_discover_v1_mlm_only \
#       trained_models/milan/scan_fl20_20cl_discover_v1 \
#       20 \
#       60

# Arguments
PRETRAINED_MODEL=$1
OUTPUT_DIR=$2
NUM_CLUSTERS=${3:-20}
MAX_EPOCHS=${4:-40}

# Validate arguments
if [ -z "$PRETRAINED_MODEL" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: sbatch skynet_run_scan.sh <pretrained_model> <output_dir> [num_clusters] [max_epochs]"
    echo ""
    echo "Arguments:"
    echo "  pretrained_model  Path to pre-trained AlignmentModel directory"
    echo "  output_dir        Output directory for SCAN model"
    echo "  num_clusters      Number of clusters (default: 20)"
    echo "  max_epochs        Maximum training epochs (default: 40)"
    exit 1
fi

echo "========================================"
echo "SCAN Clustering Training"
echo "========================================"
echo "Pretrained model: $PRETRAINED_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Number of clusters: $NUM_CLUSTERS"
echo "Max epochs: $MAX_EPOCHS"
echo "========================================"

/coc/flash5/akarpekov3/anaconda3/envs/discover-v2-env/bin/python ./src/training/train_scan.py \
    --pretrained_model "$PRETRAINED_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_clusters "$NUM_CLUSTERS" \
    --max_epochs "$MAX_EPOCHS" \
    --wandb_project discover-v2-dv1-scan

echo "SCAN training completed!"
echo "Model saved to: $OUTPUT_DIR"

