#! /bin/bash
# Runs line $SLURM_ARRAY_TASK_ID (0-based) of a task file; each line is one shell command run from the repo root.
# Usage (repo root on skynet; %10 = at most 10 tasks at once):
#   n=$(grep -c . tist_revision/skynet/tasks/<file>.txt)
#   EXCL=$(source /etc/profile.d/Z01-slurm.sh; sinfo -p overcap -h -o '%n %G' | awk '/2080_ti|titan_x/{print $1}' | paste -sd,)
#   sbatch -p overcap --exclude="$EXCL" --array=0-$((n-1))%10 \
#       tist_revision/skynet/run_tasks.sh tist_revision/skynet/tasks/<file>.txt
# Any GPU with ≥ 24 GB; refuses <24 GB (exit 99) so --requeue can try another node.
#SBATCH -o slurm/tist_%x_%A_%a.txt
#SBATCH -e slurm/tist_%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12
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
