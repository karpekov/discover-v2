#!/bin/bash
# General script to merge evaluation results for any dataset

set -e

echo "=================================================="
echo "Evaluation Results Merger"
echo "=================================================="
echo ""

# Step 1: Select dataset
echo "Select dataset:"
echo "1) milan"
echo "2) aruba"
echo "3) marble"
echo "4) cairo"
echo "5) tulum2009"
echo "6) twor2009"
echo "7) Custom dataset name"
echo ""
read -p "Enter your choice (1-7): " -n 1 -r dataset_choice
echo ""
echo ""

case $dataset_choice in
    1)
        DATASET="milan"
        ;;
    2)
        DATASET="aruba"
        ;;
    3)
        DATASET="marble"
        ;;
    4)
        DATASET="cairo"
        ;;
    5)
        DATASET="tulum2009"
        ;;
    6)
        DATASET="twor2009"
        ;;
    7)
        echo "Enter custom dataset name:"
        read -r DATASET
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "✓ Dataset: $DATASET"
echo ""

# Step 2: Select model regex pattern
echo "Select which models to merge:"
echo "1) All v1 models"
echo "2) Sequence models only (_seq_)"
echo "3) Image models only (_img_)"
echo "4) FD_30 models only"
echo "5) FD_60 models only"
echo "6) FD_120 models only"
echo "7) FL_20 models only"
echo "8) FL_50 models only"
echo "9) Models with linear projection (projlin)"
echo "10) Models with MLP projection (projmlp)"
echo "11) Models with CLIP loss only"
echo "12) Models with CLIP+MLM loss"
echo "13) Custom regex"
echo ""
read -p "Enter your choice (1-13): " choice_num
echo ""

case $choice_num in
    1)
        REGEX=".*_v1"
        OUTPUT_NAME="all_v1_models"
        echo "✓ Merging all v1 models"
        ;;
    2)
        REGEX=".*_seq_.*_v1"
        OUTPUT_NAME="sequence_models"
        echo "✓ Merging sequence models only"
        ;;
    3)
        REGEX=".*_img_.*_v1"
        OUTPUT_NAME="image_models"
        echo "✓ Merging image models only"
        ;;
    4)
        REGEX=".*fd30.*_v1"
        OUTPUT_NAME="fd30_models"
        echo "✓ Merging FD_30 models"
        ;;
    5)
        REGEX=".*fd60.*_v1"
        OUTPUT_NAME="fd60_models"
        echo "✓ Merging FD_60 models"
        ;;
    6)
        REGEX=".*fd120.*_v1"
        OUTPUT_NAME="fd120_models"
        echo "✓ Merging FD_120 models"
        ;;
    7)
        REGEX=".*fl20.*_v1"
        OUTPUT_NAME="fl20_models"
        echo "✓ Merging FL_20 models"
        ;;
    8)
        REGEX=".*fl50.*_v1"
        OUTPUT_NAME="fl50_models"
        echo "✓ Merging FL_50 models"
        ;;
    9)
        REGEX=".*projlin.*_v1"
        OUTPUT_NAME="linear_projection_models"
        echo "✓ Merging models with linear projection"
        ;;
    10)
        REGEX=".*projmlp.*_v1"
        OUTPUT_NAME="mlp_projection_models"
        echo "✓ Merging models with MLP projection"
        ;;
    11)
        REGEX=".*_clip_v1"
        OUTPUT_NAME="clip_only_models"
        echo "✓ Merging CLIP-only models"
        ;;
    12)
        REGEX=".*clipmlm.*_v1"
        OUTPUT_NAME="clip_mlm_models"
        echo "✓ Merging CLIP+MLM models"
        ;;
    13)
        echo "Enter custom regex pattern:"
        read -r REGEX
        echo "Enter output name:"
        read -r OUTPUT_NAME
        echo "✓ Using custom regex: $REGEX"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Running merger..."
echo "  Dataset: $DATASET"
echo "  Regex: $REGEX"
echo "  Output: $OUTPUT_NAME"
echo ""

python src/evals/merge_embedding_evals.py \
    --dataset "$DATASET" \
    --model-regex "$REGEX" \
    --output-name "$OUTPUT_NAME"

echo ""
echo "=================================================="
echo "✅ Merge complete!"
echo "=================================================="
echo ""
echo "Results saved to: results/evals/$DATASET/$OUTPUT_NAME/"
echo ""
echo "Files created:"
echo "  - detailed_results.csv / .json (all data)"
echo "  - summary_results.csv / .json (summary table)"
echo "  - RESULTS_REPORT.md (markdown report)"
echo "  - comprehensive_table_L1_f1_weighted.csv (+ heatmap)"
echo "  - comprehensive_table_L1_f1_macro.csv (+ heatmap)"
echo "  - comprehensive_table_L2_f1_weighted.csv (+ heatmap)"
echo "  - comprehensive_table_L2_f1_macro.csv (+ heatmap)"
echo "  - charts/ (11 comparison visualizations)"
echo ""
echo "📊 Comprehensive Tables:"
echo "  Columns: FL_20, FL_50, FD_60, FD_120 (each with Sensor/Text/Text+Proj)"
echo "  Rows: Seq-Linear-CLIP, Seq-Linear-CLIP+MLM, Seq-MLP-CLIP, Seq-MLP-CLIP+MLM,"
echo "        Img-Linear-CLIP, Img-MLP-CLIP"
echo ""
echo "Quick view of top 3 models:"
head -n 5 "results/evals/$DATASET/$OUTPUT_NAME/summary_results.csv" | column -t -s,
echo ""

