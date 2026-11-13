#! /bin/bash
# Usage (from the repo root on skynet):
#   sbatch [-p rail-lab] tist_revision/skynet/train_config.sh \
#       tist_revision/configs/alignment/<config>.yaml [--seed N --output-dir DIR ...]
# Default GRES is a40 (>=24GB). Override at submit: sbatch --gres=gpu:l40s:1 ...
#SBATCH -o slurm/tist_%x_%j.txt
#SBATCH -e slurm/tist_%x_%j.err
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node 1
#SBATCH -J tist-train
#SBATCH -p overcap
#SBATCH --requeue

set -euo pipefail
mkdir -p slurm
echo "Running: train.py --config $* on $(hostname) at $(date)"
/coc/flash5/akarpekov3/anaconda3/envs/discover-v2-env/bin/python ./train.py --config "$@"
echo "Finished: $* at $(date)"
