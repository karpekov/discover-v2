#! /bin/bash
# Usage (from the repo root on skynet):
#   sbatch [-p rail-lab] tist_revision/skynet/train_config.sh tist_revision/configs/alignment/<config>.yaml
#SBATCH -o slurm/tist_%x_%j.txt
#SBATCH -e slurm/tist_%x_%j.err
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node 1
#SBATCH -J tist-train
#SBATCH -p overcap

set -euo pipefail
mkdir -p slurm
echo "Running config: $1 on $(hostname) at $(date)"
/coc/flash5/akarpekov3/anaconda3/envs/discover-v2-env/bin/python ./train.py --config "$1"
echo "Finished: $1 at $(date)"
