#!/bin/bash
#
# Run evaluations for all trained models
#

# Activate conda environment
source /coc/flash5/akarpekov3/anaconda3/bin/activate discover-v2-env

# Set UTF-8 encoding to handle unicode characters
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# Base paths
REPO_ROOT="/coc/flash5/akarpekov3/discover-v2"
TRAIN_DATA="data/processed/casas/milan/FL_20_p/train.json"
TEST_DATA="data/processed/casas/milan/FL_20_p/test.json"
VOCAB="data/processed/casas/milan/FL_20_p/vocab.json"
MAX_SAMPLES=10000

cd "$REPO_ROOT"

# Loop through all trained models
for model_dir in trained_models/milan/milan_fl20_img_*_v1; do
    model_name=$(basename "$model_dir")
    checkpoint="$model_dir/best_model.pt"
    output_dir="results/evals/$model_name"

    echo ""
    echo "========================================"
    echo "Running evaluation for: $model_name"
    echo "========================================"

    # Check if checkpoint exists
    if [ ! -f "$checkpoint" ]; then
        echo "WARNING: Checkpoint not found: $checkpoint"
        echo "Skipping..."
        continue
    fi

    # Run evaluation
    python src/evals/run_all_evals.py \
        --checkpoint "$checkpoint" \
        --train_data "$TRAIN_DATA" \
        --test_data "$TEST_DATA" \
        --vocab "$VOCAB" \
        --output_dir "$output_dir" \
        --max_samples "$MAX_SAMPLES"

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "✓ Evaluation completed successfully for $model_name"
    else
        echo "✗ Evaluation failed for $model_name (exit code: $exit_code)"
    fi

    echo ""
done

echo ""
echo "========================================"
echo "All evaluations completed!"
echo "========================================"

