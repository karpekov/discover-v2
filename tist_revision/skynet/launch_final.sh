#! /bin/bash
# Submit the final-run task arrays on skynet (from the repo root). Throttle %10; exclude <24GB GPU hosts.
#   bash tist_revision/skynet/launch_final.sh              # submit all stages
#   bash tist_revision/skynet/launch_final.sh encoder_tf    # one stage only
set -euo pipefail
source /etc/profile.d/Z01-slurm.sh 2>/dev/null || true
EXCL=$(sinfo -p overcap -h -o '%n %G' | awk '/2080_ti|titan_x/{print $1}' | paste -sd,)
echo "excluding small-GPU hosts: $EXCL"

submit() {
  local name=$1 file=$2
  local n; n=$(grep -c . "$file")
  local jid
  jid=$(sbatch --parsable -p overcap -J "tist-$name" --exclude="$EXCL" \
    --array=0-$((n - 1))%10 tist_revision/skynet/run_tasks.sh "$file")
  echo "submitted $name: job $jid ($n tasks) <- $file"
}

cd "$(git rev-parse --show-toplevel)"
stages=${*:-encoder_bert encoder_tf tdost_text baselines}
for s in $stages; do
  submit "$s" "tist_revision/skynet/tasks/${s}.txt"
done
