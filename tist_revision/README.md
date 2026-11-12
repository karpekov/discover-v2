# TIST revision experiments (DISCOVER v1 paper)

Self-contained home for every new experiment requested by the TIST reviews. Review tracking and the
response letter live in `OTHER/discover-v1-tist-reviews/TRACKING.md` (gitignored). The original v1
models no longer exist; everything here is a re-implementation on discover-v2 code.

## Layout
- `pipeline/`: evaluation code (run from the repo root as `python -m tist_revision.pipeline.<module>`)
  - `labels.py`: raw CASAS activity → TDOST / DeepCASAS merged class (e.g. Milan Read + Watch_TV → Relax).
  - `mapping_eval.py`: shared protocol: data loading, top-m cluster → label mapping, F1 + day-block bootstrap CIs.
  - `histogram_baselines.py`: Stage 1, sensor/room/time-of-day histograms with k-means and HDBSCAN.
  - `embedding_diagnostics.py`: Stage 1.5, frozen-encoder embeddings vs histograms (kNN purity, k-means, linear probe, SCAN head).
  - `visualize.py`: confusion matrices, cluster composition, t-SNE.
- `configs/alignment/`: new encoder training configs (e.g. `*_mlm_time.yaml`: MLM with tod/dow tokens and time deltas).
- `skynet/`: `train_config.sh` (sbatch wrapper for `train.py`), `sync.sh` (push FL_20_p data, pull trained models).
- `results/`: committed CSVs (`stage1_histograms/`, `stage1_5_embeddings/`) and `figures/`.
- `artifacts/` (gitignored): data caches, encoder embeddings, cluster assignments used by `visualize.py`.

Trained models still go to `trained_models/<city>/<experiment_name>/` (repo convention, gitignored).

## Evaluation protocol
1. Clusters are fit on continuous, unlabelled `FL_20` train windows (20 events, 50% overlap).
2. All labels come from pre-segmented `FL_20_p` windows (one activity per window, same 70/10/20 day split, seed 42).
   Pool = `FL_20_p` train; for each cluster the top-m pool windows by confidence vote its label; m ∈ {5, 10, 50, all}.
3. Score on `FL_20_p` test (unseen days): weighted F1, macro F1 (over classes present in test), accuracy;
   95% CIs from a bootstrap over test days.
4. Labels: TDOST / DeepCASAS class set (`labels.py`); identical to `activity_l2` in the processed data
   (the `label_l2` dict in `metadata/casas_metadata.json` is outdated and unused here).
   Classes: Milan 10, Aruba 9, Cairo 7.
5. Variants: `all` (TDOST-comparable, keeps Other) and `excl_other`.

Encoders are fed inputs exactly as in MLM pre-training (`src/alignment/dataset.py`). `src/training/train_scan.py`
still uses `src/dataio/dataset.py` (adds 2 UNK pseudo-events, pads to 50); fix before training new SCAN models.

## Running
Local (conda env `discover-v2-env`, repo root):
```
python -m tist_revision.pipeline.histogram_baselines
python -m tist_revision.pipeline.embedding_diagnostics --cities milan aruba cairo \
    [--extra_encoders trained_models/{city}/{city}_fl20_seq_discover_v1_mlm_time --only_extra --no_histograms]
python -m tist_revision.pipeline.visualize
```

Skynet (`akarpekov3@sky2.cc.gatech.edu`, repo `/coc/flash5/akarpekov3/discover-v2`, 1×A40, partitions `overcap` / `rail-lab`):
```
tist_revision/skynet/sync.sh push-data                         # local: FL_20_p is not on skynet yet
ssh sky2 'cd /coc/flash5/akarpekov3/discover-v2 && git pull'    # skynet copy has a local edit to bash_scripts/skynet_run_config.sh
sbatch tist_revision/skynet/train_config.sh tist_revision/configs/alignment/milan_fl20_seq_discover_v1_mlm_time.yaml
tist_revision/skynet/sync.sh pull-model milan milan_fl20_seq_discover_v1_mlm_time   # local, after the job
```
Server MLM runs take 0.1–0.4 A40-hours per city; SCAN (100 epochs) 0.7–3.3 h.

## Stages
S1 histogram baselines (done) → S1.5 embedding diagnostics (done; gate) → time-feature MLM (configs ready)
→ S2 BERT vs custom transformer → S3 SCAN vs DEC vs k-means → S4 winner × 5 seeds × 3 cities.
Still to build: seed control in `train.py` / `train_scan.py`, alignment input path in `train_scan.py`,
BERT backbone, DEC, SLURM array + aggregation.
