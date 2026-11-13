#! /bin/bash
# Usage (from the repo root on skynet):
#   sbatch [-p rail-lab] tist_revision/skynet/train_config.sh \
#       tist_revision/configs/alignment/<config>.yaml [--seed N --output-dir DIR ...]
# Prefer launch_final.sh for the revision grid. Generic gpu:1 + ≥24GB check (see run_tasks.sh).
#SBATCH -o slurm/tist_%x_%j.txt
#SBATCH -e slurm/tist_%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes 1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node 1
#SBATCH -J tist-train
#SBATCH -p overcap
#SBATCH --requeue

set -euo pipefail
mkdir -p slurm
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk 'NR==1{print int($1)}')
if (( ${mem:-0} < 23000 )); then
  echo "Need >=24GB GPU, got ${mem} MiB on $(hostname); exiting for requeue"
  exit 99
fi
echo "Running: train.py --config $* on $(hostname) gpu=${mem}MiB at $(date)"
/coc/flash5/akarpekov3/anaconda3/envs/discover-v2-env/bin/python ./train.py --config "$@"
echo "Finished: $* at $(date)"
