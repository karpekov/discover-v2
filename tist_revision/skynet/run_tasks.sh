#! /bin/bash
# Runs line $SLURM_ARRAY_TASK_ID (0-based) of a task file; each line is one shell command run from the repo root.
# Usage (repo root on skynet; %10 = at most 10 tasks at once):
#   bash tist_revision/skynet/launch_final.sh <task_stem>
# GRES is generic gpu:1; launch_final excludes 2080 Ti / Titan X hosts so we land on ≥24GB GPUs
# (a40 / l40s / rtx_6000 / a5000). Script also refuses <24GB if somehow scheduled there.
#SBATCH -o slurm/tist_%x_%A_%a.txt
#SBATCH -e slurm/tist_%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes 1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node 1
#SBATCH -J tist-tasks
#SBATCH -p overcap
#SBATCH --requeue

set -eo pipefail
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk 'NR==1{print int($1)}')
if (( ${mem:-0} < 23000 )); then
  echo "Need >=24GB GPU, got ${mem} MiB on $(hostname); exiting for requeue"
  exit 99
fi
export PATH=/coc/flash5/akarpekov3/anaconda3/envs/discover-v2-env/bin:$PATH
cmd=$(grep . "$1" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
echo "[$(hostname) ${mem}MiB $(date)] task $SLURM_ARRAY_TASK_ID: $cmd"
eval "$cmd"
echo "[$(date)] done"
