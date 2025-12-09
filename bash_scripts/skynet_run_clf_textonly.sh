#! /bin/bash
#SBATCH -o slurm/clf_textonly_output_%j.txt
#SBATCH -e slurm/clf_textonly_error_%j.txt
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node 1
#SBATCH -J dv2-clf-text
#SBATCH -p rail-lab

# Usage: sbatch bash_scripts/skynet_run_clf_textonly.sh <dataset> <data_config> <caption_style> <encoder_type> <classifier> <label_level> <epochs>
# Example: sbatch bash_scripts/skynet_run_clf_textonly.sh milan FD_60_p baseline clip linear l1 50
# Example: sbatch bash_scripts/skynet_run_clf_textonly.sh milan FD_60_p baseline clip mlp l2 30

DATASET=${1:-milan}
DATA_CONFIG=${2:-FD_60_p}
CAPTION_STYLE=${3:-baseline}
ENCODER_TYPE=${4:-clip}
CLASSIFIER=${5:-linear}
LABEL_LEVEL=${6:-l2}
EPOCHS=${7:-50}

DATA_DIR=data/processed/casas/$DATASET/$DATA_CONFIG

echo "========================================"
echo "RUNNING TEXT-ONLY CLF PROBING"
echo "========================================"
echo "Dataset:       $DATASET"
echo "Config:        $DATA_CONFIG"
echo "Caption Style: $CAPTION_STYLE"
echo "Encoder:       $ENCODER_TYPE"
echo "Classifier:    $CLASSIFIER"
echo "Label Level:   $LABEL_LEVEL"
echo "Epochs:        $EPOCHS"
echo "Data Dir:      $DATA_DIR"
echo "========================================"

/coc/flash5/akarpekov3/anaconda3/envs/discover-v2-env/bin/python ./src/utils/train_classifier_from_pretrained_model.py \
  --embeddings-dir $DATA_DIR \
  --caption-style $CAPTION_STYLE \
  --encoder-type $ENCODER_TYPE \
  --data-dir $DATA_DIR \
  --classifier $CLASSIFIER \
  --label-level $LABEL_LEVEL \
  --epochs $EPOCHS \
  --batch-size 128 \
  --lr 5e-4

echo ""
echo "========================================"
echo "CLF completed for ${CAPTION_STYLE}_${ENCODER_TYPE}"
echo "Results saved to: results/evals/$DATASET/${DATA_CONFIG/_p/}/text_only/clf_probing/"
echo "========================================"

