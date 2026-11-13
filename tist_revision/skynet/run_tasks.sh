#! /bin/bash
# Runs line $SLURM_ARRAY_TASK_ID (0-based) of a task file; each line is one shell command run from the repo root.
# Usage (repo root on skynet; %3 = at most 3 tasks at once):
#   n=$(grep -c . tist_revision/skynet/tasks/<file>.txt)
#   sbatch -p overcap --array=0-$((n-1))%3 tist_revision/skynet/run_tasks.sh tist_revision/skynet/tasks/<file>.txt
#SBATCH -o slurm/tist_%x_%A_%a.txt
#SBATCH -e slurm/tist_%x_%A_%a.err
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node 1
#SBATCH -J tist-tasks
#SBATCH -p overcap

set -eo pipefail
export PATH=/coc/flash5/akarpekov3/anaconda3/envs/discover-v2-env/bin:$PATH
cmd=$(grep . "$1" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
echo "[$(hostname) $(date)] task $SLURM_ARRAY_TASK_ID: $cmd"
eval "$cmd"
echo "[$(date)] done"
