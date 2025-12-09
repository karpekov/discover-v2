#!/bin/bash

# Script to run comprehensive evaluations on all Milan models ending with _v1
# Automatically detects models and maps them to corresponding datasets

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Milan Model Evaluation Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Base directories
MODELS_DIR="trained_models/milan"
DATA_DIR="data/processed/casas/milan"
RESULTS_DIR="results/evals/milan"

# Ask user which type of models to process
echo -e "${YELLOW}Which models do you want to evaluate?${NC}"
echo "1) Sequence models only (seq)"
echo "2) Image models only (img)"
echo "3) Both sequence and image models"
read -p "Enter your choice (1-3): " -n 1 -r model_type_choice
echo ""
echo ""

# Set filter based on user choice
case $model_type_choice in
    1)
        MODEL_FILTER="_seq_"
        echo -e "${GREEN}✓ Processing sequence models only${NC}"
        ;;
    2)
        MODEL_FILTER="_img_"
        echo -e "${GREEN}✓ Processing image models only${NC}"
        ;;
    3)
        MODEL_FILTER=""
        echo -e "${GREEN}✓ Processing all models${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac
echo ""

# Find all model directories ending with _v1
echo -e "${YELLOW}Discovering models...${NC}"
model_count=0

for model_path in ${MODELS_DIR}/*_v1; do
    if [ -d "$model_path" ]; then
        model_name=$(basename "$model_path")

        # Filter by model type if specified
        if [ -n "$MODEL_FILTER" ] && [[ ! "$model_name" == *"$MODEL_FILTER"* ]]; then
            continue
        fi

        # Extract dataset identifier (fd60, fl20, fl50, fd120, etc.)
        # Pattern: milan_{dataset}_seq_... or milan_{dataset}_img_...
        dataset_key=$(echo "$model_name" | sed -n 's/milan_\([^_]*\)_.*/\1/p')

        if [ -z "$dataset_key" ]; then
            echo -e "${RED}⚠️  Could not extract dataset from: $model_name${NC}"
            continue
        fi

        # Convert dataset key to folder name
        # fd60 -> FD_60_p, fl20 -> FL_20_p, fd120 -> FD_120_p
        dataset_type=$(echo "$dataset_key" | sed 's/[0-9].*//g' | tr '[:lower:]' '[:upper:]')  # fd -> FD, fl -> FL
        dataset_num=$(echo "$dataset_key" | sed 's/[^0-9]*//g')  # Extract numbers: 60, 20, 120
        dataset_folder="${dataset_type}_${dataset_num}_p"

        # Check if dataset folder exists
        dataset_path="${DATA_DIR}/${dataset_folder}"
        if [ ! -d "$dataset_path" ]; then
            echo -e "${RED}⚠️  Dataset not found: $dataset_path (for model: $model_name)${NC}"
            continue
        fi

        # Check if checkpoint exists
        checkpoint_path="${model_path}/best_model.pt"
        if [ ! -f "$checkpoint_path" ]; then
            echo -e "${RED}⚠️  Checkpoint not found: $checkpoint_path${NC}"
            continue
        fi

        # Check if embeddings exist
        train_embeddings="${dataset_path}/train_embeddings_baseline_clip.npz"
        test_embeddings="${dataset_path}/test_embeddings_baseline_clip.npz"
        if [ ! -f "$train_embeddings" ] || [ ! -f "$test_embeddings" ]; then
            echo -e "${RED}⚠️  Embeddings not found for dataset: $dataset_folder${NC}"
            continue
        fi

        model_count=$((model_count + 1))

        echo -e "${GREEN}[$model_count] Found: $model_name${NC}"
        echo -e "    Dataset: ${dataset_folder}"
        echo -e "    Checkpoint: ${checkpoint_path}"
        echo ""
    fi
done

if [ $model_count -eq 0 ]; then
    echo -e "${RED}No valid models found!${NC}"
    exit 1
fi

echo -e "${BLUE}Found ${model_count} models to evaluate${NC}"
echo ""
read -p "Continue with evaluation? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Run evaluations
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Evaluations${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

current=0
for model_path in ${MODELS_DIR}/*_v1; do
    if [ -d "$model_path" ]; then
        model_name=$(basename "$model_path")

        # Filter by model type if specified
        if [ -n "$MODEL_FILTER" ] && [[ ! "$model_name" == *"$MODEL_FILTER"* ]]; then
            continue
        fi

        # Extract dataset identifier
        dataset_key=$(echo "$model_name" | sed -n 's/milan_\([^_]*\)_.*/\1/p')

        if [ -z "$dataset_key" ]; then
            continue
        fi

        # Convert dataset key to folder name
        dataset_type=$(echo "$dataset_key" | sed 's/[0-9].*//g' | tr '[:lower:]' '[:upper:]')
        dataset_num=$(echo "$dataset_key" | sed 's/[^0-9]*//g')
        dataset_folder="${dataset_type}_${dataset_num}_p"

        # Paths
        dataset_path="${DATA_DIR}/${dataset_folder}"
        checkpoint_path="${model_path}/best_model.pt"
        train_embeddings="${dataset_path}/train_embeddings_baseline_clip.npz"
        test_embeddings="${dataset_path}/test_embeddings_baseline_clip.npz"
        output_dir="${RESULTS_DIR}/${dataset_folder}/${model_name}"

        # Check all paths exist
        if [ ! -d "$dataset_path" ] || [ ! -f "$checkpoint_path" ] || \
           [ ! -f "$train_embeddings" ] || [ ! -f "$test_embeddings" ]; then
            continue
        fi

        current=$((current + 1))

        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}Evaluation ${current}/${model_count}${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo -e "${GREEN}Model:${NC} $model_name"
        echo -e "${GREEN}Dataset:${NC} $dataset_folder"
        echo -e "${GREEN}Output:${NC} $output_dir"
        echo ""

        # Run evaluation
        python src/evals/evaluate_embeddings.py \
            --checkpoint "$checkpoint_path" \
            --train_data "${dataset_path}/train.json" \
            --test_data "${dataset_path}/test.json" \
            --vocab "${dataset_path}/vocab.json" \
            --output_dir "$output_dir" \
            --eval_all \
            --train_text_embeddings "$train_embeddings" \
            --test_text_embeddings "$test_embeddings" \
            --max_samples 10000 \
            --filter_noisy_labels

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Evaluation completed: $model_name${NC}"
        else
            echo -e "${RED}❌ Evaluation failed: $model_name${NC}"
        fi
        echo ""
    fi
done

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All evaluations completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Results saved to: ${RESULTS_DIR}"

