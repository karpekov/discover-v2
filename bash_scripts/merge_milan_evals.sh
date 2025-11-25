#!/bin/bash
# Helper script to merge Milan evaluation results
# (Convenience wrapper that pre-selects Milan dataset)

set -e

echo "=================================================="
echo "Milan Evaluation Results Merger"
echo "=================================================="
echo ""
echo "Dataset: milan (pre-selected)"
echo ""
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
read -p "Enter your choice (1-13): " choice
echo ""
echo ""

case $choice in
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
        REGEX=".*fd60.*_v1"
        OUTPUT_NAME="fd60_models"
        echo "✓ Merging FD_60 models"
        ;;
    5)
        REGEX=".*fd120.*_v1"
        OUTPUT_NAME="fd120_models"
        echo "✓ Merging FD_120 models"
        ;;
    6)
        REGEX=".*fl20.*_v1"
        OUTPUT_NAME="fl20_models"
        echo "✓ Merging FL_20 models"
        ;;
    7)
        REGEX=".*fl50.*_v1"
        OUTPUT_NAME="fl50_models"
        echo "✓ Merging FL_50 models"
        ;;
    8)
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
echo ""

python src/evals/merge_embedding_evals.py \
    --dataset milan \
    --model-regex "$REGEX" \
    --output-name "$OUTPUT_NAME"

echo ""
echo "=================================================="
echo "✅ Merge complete!"
echo "=================================================="
echo ""
echo "Results saved to: results/evals/milan/$OUTPUT_NAME/"
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

