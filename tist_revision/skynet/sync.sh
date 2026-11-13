#! /bin/bash
# Run locally from the repo root.
#   tist_revision/skynet/sync.sh push-data                 # upload pre-segmented FL_20_p (FL_20 is already on skynet)
#   tist_revision/skynet/sync.sh push-tdost                # upload artifacts/tdost/{milan,aruba,cairo}.npz
#   tist_revision/skynet/sync.sh pull-model <city> <dir>   # fetch best_model.pt + config.yaml of trained_models/<city>/<dir>
#   tist_revision/skynet/sync.sh pull-results              # fetch results CSVs + artifacts (embeddings, assignments, baselines)
set -euo pipefail
REMOTE=${REMOTE:-akarpekov3@sky2.cc.gatech.edu}
REMOTE_ROOT=${REMOTE_ROOT:-/coc/flash5/akarpekov3/discover-v2}

case "${1:-}" in
  push-data)
    for city in milan aruba cairo; do
      rsync -av --include 'train.json' --include 'val.json' --include 'test.json' --include 'vocab.json' --include 'statistics.json' --exclude '*' \
        "data/processed/casas/$city/FL_20_p/" "$REMOTE:$REMOTE_ROOT/data/processed/casas/$city/FL_20_p/"
    done ;;
  push-tdost)
    rsync -av tist_revision/artifacts/tdost/*.npz "$REMOTE:$REMOTE_ROOT/tist_revision/artifacts/tdost/" ;;
  pull-model)
    city=$2; dir=$3
    mkdir -p "trained_models/$city/$dir"
    rsync -av "$REMOTE:$REMOTE_ROOT/trained_models/$city/$dir/best_model.pt" \
              "$REMOTE:$REMOTE_ROOT/trained_models/$city/$dir/config.yaml" "trained_models/$city/$dir/" ;;
  pull-results)
    # Only folders produced on skynet; -u never overwrites newer local files.
    rsync -avu "$REMOTE:$REMOTE_ROOT/tist_revision/results/stage3_scan/" tist_revision/results/stage3_scan/
    rsync -avu "$REMOTE:$REMOTE_ROOT/tist_revision/results/stage1_5_mlm_time/" tist_revision/results/stage1_5_mlm_time/
    rsync -avu "$REMOTE:$REMOTE_ROOT/tist_revision/results/final/" tist_revision/results/final/
    rsync -avu "$REMOTE:$REMOTE_ROOT/tist_revision/results/baselines/" tist_revision/results/baselines/
    rsync -avu --include '*/' --include '*.npz' --exclude '*' \
      "$REMOTE:$REMOTE_ROOT/tist_revision/artifacts/" tist_revision/artifacts/ ;;
  *)
    echo "usage: $0 push-data | push-tdost | pull-model <city> <model_dir> | pull-results"; exit 1 ;;
esac
