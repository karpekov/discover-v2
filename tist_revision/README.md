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
  - `encoding.py`: encoder loading + inputs encoded exactly as in MLM pre-training (shared by everything below).
  - `scan.py`: SCAN (replaces `src/training/train_scan.py` for the revision). Frozen encoder by default
    (heads train on cached embeddings; cheap), or `--freeze_epochs N --encoder_lr 1e-5` to unfreeze after N
    epochs. Trains `--num_heads` heads and keeps the lowest-loss one. Seeded.
  - `dec.py`: DEC on the frozen encoder (identity-initialised adapter + centroids), second deep-clustering baseline.
  - `rare_classes.py`: K × m sweep, agreement / confidence thresholds, 1-NN label propagation (k-means, frozen features).
  - `summarize.py`: merges all clustering results into `results/summary_{variant}.csv`.
  - `visualize.py`: confusion matrices, cluster composition, t-SNE.
- `configs/alignment/`: new encoder training configs (e.g. `*_mlm_time.yaml`: MLM with tod/dow tokens and time deltas).
  **Decision (2026-11-12): all future encoder (re)runs use per-event inputs sensor, state, room only — no x-y
  coordinates.** Use `*_fl20_seq_nocoord_mlm.yaml` (no time features) and `*_fl20_seq_nocoord_mlm_time.yaml`
  (+ tod/dow tokens, time deltas). Not trained yet; existing results use the coordinate encoders
  (`*_discover_v1_mlm_only`, `*_discover_v1_mlm_time`). Note: `use_time_of_day` in configs is a no-op.
- `skynet/`: `train_config.sh` (sbatch wrapper for `train.py`), `run_tasks.sh` (SLURM array: one shell command
  per line of `skynet/tasks/*.txt`, throttled with `%N`), `sync.sh` (push FL_20_p data, pull models / results).
- `results/`: committed CSVs (`stage1_histograms/`, `stage1_5_*/`, `rare_classes/`, `summary_*.csv`) and `figures/`
  (png, gitignored by the repo). Per-run SCAN/DEC outputs (`stage3_scan/runs/`) are written on skynet and
  fetched with `sync.sh pull-results` (not committed; `summarize.py` merges them).
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

Encoders are fed inputs exactly as in MLM pre-training (`src/alignment/dataset.py`). The old
`src/training/train_scan.py` uses `src/dataio/dataset.py` (adds 2 UNK pseudo-events, pads to 50) and is not used
for the revision; `pipeline/scan.py` replaces it.

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
sbatch -p overcap --array=0-8%3 tist_revision/skynet/run_tasks.sh tist_revision/skynet/tasks/scan_mlm_only.txt
tist_revision/skynet/sync.sh pull-results && python -m tist_revision.pipeline.summarize   # local
```
Rules: check `squeue -p rail-lab` first (use `overcap` if labmates are queued); throttle arrays (`%2`–`%3`);
pack several cheap frozen runs per task line; single seed until the setup is chosen. Slurm logs: `slurm/tist_*`
on skynet (training progress goes to the `.err` file).
Server MLM runs take 0.1–0.4 A40-hours per city; SCAN (100 epochs) 0.7–3.3 h.

## Findings so far
**Current consolidated report (supersedes the bullets below where they differ): `results/report/REPORT.md`**
(L1 + L2 labels, rare-class masking, k-means / SCAN / DEC / embedding + time-of-day; `pipeline/label_levels.py`,
`pipeline/report.py`). Key update: k-means on MLM embedding + time-of-day histogram (K=100) is the most
consistent setup; SCAN's Aruba macro gain came from one rare class.

Earlier notes (single seed 42, 2026-11-12; details in `results/summary_*.csv`, `results/rare_classes/`):
F1 = weighted / macro, variant `all` unless noted; m = labelled windows per cluster.
- **Representations (Stage 1.5).** Frozen MLM encoder ≈ sensor histograms. The time-feature encoder
  (`*_mlm_time`) has the best kNN purity and linear probes (full-label probe macro: Cairo 0.65 vs 0.52 MLM,
  0.59 hist+tod; Aruba 0.82 vs 0.79), but k-means on it is not better than on MLM.
- **Number of clusters dominates.** Going from K=20 to K=50/100 raises macro F1 by about 0.05–0.15 for every method;
  rare classes start to get their own clusters. At equal labelling budget, 100 clusters × 1–3 labels beats
  20 × 5. m=3 ≈ m=5.
- **SCAN (frozen encoder, standardised embeddings, 5 heads).** Best settings K=100 (or 50), entropy weight 2–5:
  m=5: Milan 0.80/0.47, Aruba 0.88/0.58, Cairo 0.81/0.41–0.46. Versus k-means on the same encoder and K,
  macro F1 is higher on Aruba (+0.06–0.09), equal on Milan, mixed on Cairo (−0.03 on MLM, +0.13 on the
  time-feature encoder); weighted F1 is similar. Above DEC at K ≥ 50 in all houses.
  Entropy weight 1/2/5 and neighbours 5/20/50 give inconsistent ±0.05 differences (single seed).
- **Unfreezing the encoder does not help.** Frozen for 10 epochs, then encoder lr 1e-5: equal or worse than
  frozen, at 5–10× the cost (9–37 min vs 1–6 min on an A40); same at K=50 (Milan 0.80/0.46 vs 0.79/0.45 frozen,
  Cairo 0.80/0.39 vs 0.79/0.42). Use the frozen encoder.
- **DEC ≈ k-means.**
- **Rare classes (SCAN K=100 vs k-means K=20, m=3, MLM encoder).** Recall rises for Milan Leave_Home 0→0.62 and
  Eat 0.30→0.53, Cairo Work 0.17→0.62, Aruba Bed_to_Toilet 0→1.0; some mid-size classes drop (Aruba Eat 0.66→0.23).
- **Thresholds** (cluster agreement / distance to centroid) trade coverage for precision (Aruba 0.97/0.91 at
  62% coverage) but do not recover rare classes; 1-NN propagation from the annotated windows ≈ cluster vote.
- **Cairo**: k-means on sensor+time-of-day histograms (K=100, m=5: 0.87/0.50) is still the best weighted F1.
- Never recovered by any method: Take_Medicine (all houses), Aruba Enter_Home / Leave_Home (4 test windows each).
- Bug fixed on the way: unstandardised (L2-normalised) embeddings let SCAN heads stall at near-uniform outputs
  and argmax-collapse to one cluster; all earlier SCAN runs were discarded and re-run.

## Stages
S1 histogram baselines (done) → S1.5 embedding diagnostics (done) → time-feature MLM (trained on skynet)
→ S3 SCAN vs DEC vs k-means on frozen encoders (single-seed grid done: `skynet/tasks/grid_v2.txt`)
→ S2 BERT vs custom transformer (BERT backbone not built yet) → S4 chosen setup × 5 seeds × 3 cities.
Still to build: BERT backbone, seed control in `train.py` (MLM pre-training; `scan.py` / `dec.py` are seeded).
