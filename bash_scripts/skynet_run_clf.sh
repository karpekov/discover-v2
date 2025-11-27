#! /bin/bash
#SBATCH -o slurm/clf_output_%j.txt
#SBATCH -e slurm/clf_error_%j.txt
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node 1
#SBATCH -J dv2-train
#SBATCH -p overcap

MODEL_DIR=trained_models/milan/$1

echo "RUNNING CLF for $MODEL_DIR"

/coc/flash5/akarpekov3/anaconda3/envs/discover-v2-env/bin/python ./src/utils/train_classifier_from_pretrained_model.py \
  --model $MODEL_DIR \
  --epochs 50 \
  --label-level l2 \
  --batch-size 128 \
  --classifier linear \
  --lr 5e-4

echo "CLF completed for $MODEL_DIR"