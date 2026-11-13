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

## Decisions for the final runs (Alex, 2026-11-12)
- **Narrative:** the learned embeddings are for later multi-house training. Say plainly that at the single-house
  level they bring little over hand-crafted sensor + time-of-day histograms.
- **Encoder:** retrain the MLM encoder without x-y coordinates. Per-event inputs: sensor, state, room. Always
  include time-of-day encoding, i.e. config `*_fl20_seq_nocoord_mlm_time.yaml`; no run without time.
- **Clustering:** k-means, SCAN (frozen encoder) and DEC, at K = 20 (as in the current paper) and K = 50.
- **Labels:** L1 (per-house activities, resident id removed) is primary; L2 (TDOST classes) only for comparability.
  Rare classes are masked from the headline macro F1 (≥ 50 pool and ≥ 20 test windows).
- **Seeds:** 5 per configuration. The encoder is retrained per seed (15 MLM runs), with clustering seeded the same.
- **Clustering inputs:** MLM-time embedding and MLM-time + ToD histogram, for k-means, SCAN and DEC. The
  Sensor hist. and Sensor+ToD hist. k-means baselines stay in the tables.
- **CRITICAL: no pre-segmented data in training or validation, for any method.** Only the test set is pre-segmented
  (FL_20_p test, so test labels are unambiguous).
  - Baselines train and validate on continuous FL_20 train / val windows.
  - Our cluster labelling (the annotator's pool) and the linear probes must also move from FL_20_p train to
    continuous FL_20 train. All current ablation numbers use the pre-segmented pool and must be re-scored.
- **Baselines:** TDOST, DeepCASAS and TreeCNN, run on the same data and splits, compared with our best method.
  - Label budgets: all labels, 100 and 250 random windows (matching 20 or 50 clusters × 5 labels).
  - Label levels: L1 and L2.
  - TDOST uses rule-based descriptions from the official repo (`~/code/har/TDOST`, `cbalab/TDOST`, script
    `TDOST/generate_embeddings_v1*.py`, sentence encoder `all-distilroberta-v1`). Embeddings are regenerated
    for our windows; none exist yet.
  - TreeCNN is implemented from Cao et al. (2020); no prior code exists.
- **No coordinates anywhere.** Existing (coordinate) encoders are not used for any reported number. Until the
  no-coordinate encoders exist, local re-scores cover histogram baselines only.
- **Encoder choice gate (R2-2.9(b)):** our transformer vs pretrained `bert-base-uncased` (event words added as new
  tokens), both no-coordinate + time of day, single seed, 3 houses, scored with k-means / SCAN / DEC at K = 20, 50.
  Run first; the winner gets the 5-seed runs. Our transformer wins if results are comparable (smaller model).
- Status: plan drafted (below). Nothing launched until Alex says go.

## Launch plan (agreed 2026-11-12; waiting for Alex's go)
Window labels for training, validation and annotation: last event's activity, with No_Activity kept as a label.
TDOST variant: rule-based v1-1 (with time between readings).

**To implement before launch** (local):
0. BERT backbone: port v1 `MLMTrainer` (pretrained `bert-base-uncased`, new event tokens), no coordinates,
   time of day, same encode interface as our transformer.
1. Seed control in `train.py` (MLM pre-training), plus a task file for the 15 encoder runs.
2. `encode_embeddings` step: encoder → `artifacts/embeddings/{city}_nocoord_mlm_time_s{seed}_embeddings.npz`.
3. Seeded stage-B runner: k-means, frozen SCAN and DEC (`frozen_deep.py`) on MLM-time and MLM-time + ToD,
   plus histogram k-means, K ∈ {20, 50}, saved with suffix `_s{seed}`. Make `label_levels.py` / `report.py`
   aggregate mean ± std over seeds.
4. Baselines in `tist_revision/baselines/`, trained on continuous FL_20 train / val and tested on FL_20_p test:
   - DeepCASAS: port from `~/code/har/TDOST/DeepCASAS`.
   - TDOST: rule-based v1-1 sentences + `all-distilroberta-v1`, classifier from `~/code/har/TDOST/HAR`.
   - TreeCNN: new implementation from Cao et al. (2020).
   Settings: L1 and L2; budgets 100, 250 and all random windows; 5 seeds.
5. Switch skynet jobs from A40-only to any GPU with ≥ 24 GB (A40, L40S, RTX 6000, A5000).

| Stage | What | Runs | Skynet jobs | Est. per job |
|---|---|---|---|---|
| A0 | Encoder gate: our transformer vs BERT, 3 houses × 1 seed, then stage B on each | 6 (+ B) | 6 | 20–90 min |
| A | MLM encoder (gate winner), 3 houses × 5 seeds (seed 1 reused from A0) | 15 | 12 | 20–60 min |
| B | encode, then k-means / SCAN / DEC × 2 inputs × K ∈ {20, 50}, plus histogram k-means; one job per (house, seed) | ~150 | 15 | 20–40 min |
| C | TDOST embedding generation (once per house) | 3 | 3 (or local) | ~10 min |
| D | Baselines: 3 methods × 3 houses × 5 seeds × {L1, L2} × {100, 250, all}; one job per (method, house, seed) | 270 | 45 | 30–60 min |
| E | Aggregate: mean ± std over seeds; tables + PDF | – | local | – |
- Order: A0 with C and D in parallel; Alex picks the encoder from A0; then A, then B as each encoder finishes.
- Throttle: `%10` per array on overcap. The QoS cap is 48 GPUs per user; jobs are preemptible and requeued.
- Total: about 80 GPU jobs and 55 GPU-hours, roughly 6–9 h wall-clock.

## Findings so far
**Current consolidated report (supersedes the bullets below where they differ): `results/report/REPORT.md`**
(L1 with resident id removed + L2 labels, rare-class masking, k-means / SCAN / DEC / embedding + time-of-day,
linear probes at matched label budgets; `pipeline/label_levels.py`, `pipeline/probes.py`, `pipeline/report.py`).
Key update: the representation (with explicit time of day) is the bottleneck, not the clustering algorithm;
k-means ≈ linear probe at equal labels; K=50 ≈ K=100; learned encoder ≈ Sensor+ToD histograms.

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
