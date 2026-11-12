#! /bin/bash
# Run locally from the repo root.
#   tist_revision/skynet/sync.sh push-data                 # upload pre-segmented FL_20_p (FL_20 is already on skynet)
#   tist_revision/skynet/sync.sh pull-model <city> <dir>   # fetch best_model.pt + config.yaml of trained_models/<city>/<dir>
set -euo pipefail
REMOTE=${REMOTE:-akarpekov3@sky2.cc.gatech.edu}
REMOTE_ROOT=${REMOTE_ROOT:-/coc/flash5/akarpekov3/discover-v2}

case "${1:-}" in
  push-data)
    for city in milan aruba cairo; do
      rsync -av --include 'train.json' --include 'val.json' --include 'test.json' --include 'vocab.json' --include 'statistics.json' --exclude '*' \
        "data/processed/casas/$city/FL_20_p/" "$REMOTE:$REMOTE_ROOT/data/processed/casas/$city/FL_20_p/"
    done ;;
  pull-model)
    city=$2; dir=$3
    mkdir -p "trained_models/$city/$dir"
    rsync -av "$REMOTE:$REMOTE_ROOT/trained_models/$city/$dir/best_model.pt" \
              "$REMOTE:$REMOTE_ROOT/trained_models/$city/$dir/config.yaml" "trained_models/$city/$dir/" ;;
  *)
    echo "usage: $0 push-data | pull-model <city> <model_dir>"; exit 1 ;;
esac
