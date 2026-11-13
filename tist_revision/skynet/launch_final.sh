#! /bin/bash
# Submit the final-run task arrays on skynet (from the repo root). Throttle %10; default GRES a40 (>=24GB).
#   bash tist_revision/skynet/launch_final.sh              # submit all stages
#   bash tist_revision/skynet/launch_final.sh encoder_tf    # one stage only
# Override GPU type: GRES=gpu:l40s:1 bash tist_revision/skynet/launch_final.sh baselines
set -euo pipefail
source /etc/profile.d/Z01-slurm.sh 2>/dev/null || true
GRES=${GRES:-gpu:a40:1}

submit() {
  local name=$1 file=$2
  local n; n=$(grep -c . "$file")
  local jid
  jid=$(sbatch --parsable -p overcap -J "tist-$name" --gres="$GRES" \
    --array=0-$((n - 1))%10 tist_revision/skynet/run_tasks.sh "$file")
  echo "submitted $name: job $jid ($n tasks, $GRES) <- $file"
}

cd "$(git rev-parse --show-toplevel)"
stages=${*:-encoder_bert encoder_tf tdost_text baselines}
for s in $stages; do
  submit "$s" "tist_revision/skynet/tasks/${s}.txt"
done
