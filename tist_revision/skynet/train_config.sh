#! /bin/bash
# Usage (from the repo root on skynet):
#   sbatch [-p rail-lab] tist_revision/skynet/train_config.sh \
#       tist_revision/configs/alignment/<config>.yaml [--seed N --output-dir DIR ...]
# Any GPU with ≥ 24 GB (a40 / l40s / rtx_6000 / a5000); small-GPU hosts are excluded at submit time
# via run_tasks.sh / launch_final.sh. This script also refuses <24 GB if landed there.
#SBATCH -o slurm/tist_%x_%j.txt
#SBATCH -e slurm/tist_%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12
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
