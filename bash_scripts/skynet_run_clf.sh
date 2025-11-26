#! /bin/bash
#SBATCH -o slurm/clf_output_%j.txt
#SBATCH -e slurm/clf_error_%j.txt
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node 1
#SBATCH -J dv2-train
#SBATCH -p rail-lab

echo "RUNNING CLF for milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1"

/coc/flash5/akarpekov3/anaconda3/envs/discover-v2-env/bin/python ./src/utils/train_classifier_from_pretrained_model.py \
  --model trained_models/milan/milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1 \
  --epochs 30 \
  --label-level l1 \
  --batch-size 128 \
  --classifier mlp

echo "CLF completed for milan_fd60_seq_rb0_textclip_projmlp_clipmlm_v1"

