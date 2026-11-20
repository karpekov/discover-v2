# Explorations log (TIST revision)

Append-only, newest at the bottom, messy is fine. Every exploration gets a dated entry: question, what was run
(command / script), where outputs live (`results/temp/...` unless promoted), result, verdict. Curated current
state lives in `FINDINGS.md`; paper-grade numbers in `results/final/`.

---

## 2026-11-12 — single-seed ablations (superseded protocol)
Pre-segmented FL_20_p annotation pool, coordinate encoders, seed 42. Outputs: `results/temp/{stage1_histograms,
stage1_5_*, stage3_scan, rare_classes, label_levels, probes, summary_*.csv, figures}`, report
`results/temp/report_v0/` (`REPORT.md`, `report_tables.pdf`).
F1 = weighted / macro, variant `all` unless noted; m = labelled windows per cluster.
- **Representations (Stage 1.5).** Frozen MLM encoder ≈ sensor histograms. The time-feature encoder
  (`*_mlm_time`) has the best kNN purity and linear probes (full-label probe macro: Cairo 0.65 vs 0.52 MLM,
  0.59 hist+tod; Aruba 0.82 vs 0.79), but k-means on it is not better than on MLM.
- **Number of clusters dominates.** K=20 → 50/100 raises macro F1 by about 0.05–0.15 for every method; rare classes
  start to get their own clusters. At equal labelling budget, 100 clusters × 1–3 labels beats 20 × 5. m=3 ≈ m=5.
- **SCAN (frozen encoder, standardised embeddings, 5 heads).** Best K=100 (or 50), entropy weight 2–5: m=5 Milan
  0.80/0.47, Aruba 0.88/0.58, Cairo 0.81/0.41–0.46. Vs k-means on the same encoder: macro higher on Aruba
  (+0.06–0.09), equal on Milan, mixed on Cairo. Above DEC at K ≥ 50. Entropy weight 1/2/5 and neighbours 5/20/50
  give inconsistent ±0.05 differences (single seed).
- **Unfreezing the encoder does not help** (frozen 10 epochs, then lr 1e-5): equal or worse at 5–10× the cost.
- **DEC ≈ k-means.**
- **Rare classes (SCAN K=100 vs k-means K=20, m=3).** Recall up for Milan Leave_Home 0→0.62, Eat 0.30→0.53, Cairo
  Work 0.17→0.62, Aruba Bed_to_Toilet 0→1.0; some mid-size classes drop (Aruba Eat 0.66→0.23).
- **Thresholds** (cluster agreement / distance) trade coverage for precision but do not recover rare classes;
  1-NN propagation ≈ cluster vote.
- **Hierarchical merging** (Ward over 100 centroids → 20–50 groups ≈ k-means at 20–50; adaptive splitting labels
  46–126 groups and is not better than K=50). No gain.
- **Time features at K=20 hurt Aruba** (0.80/0.35 vs 0.89/0.66): clusters split Relax / Cook by hour.
- Bug fixed: unstandardised (L2-normalised) embeddings made SCAN heads collapse to one cluster; all earlier SCAN
  runs discarded.
- report_v0 assessment: representation is the bottleneck, not the clustering algorithm; time of day must be
  explicit; learned encoder ≈ Sensor+ToD histograms (biggest risk, R2-2.9(a)); recommend K=50.

## 2026-11-12 — protocol switch
Continuous FL_20 pool for annotation (no pre-segmented data in train/val for any method), no coordinates.
With No_Activity predictable, Cairo had 16/20 clusters mapped to No_Activity (L1 masked macro Sensor+ToD hist K=20:
0.31 / 0.23 / 0.20). Fix (as in the paper): No_Activity never predicted on test → K=20 0.35 / 0.37 / 0.52,
K=50 0.42 / 0.57 / 0.62. Label budgets count activity labels only.

## 2026-11-13 — final seeded runs (skynet)
15 TF encoders (3 houses × 5 seeds), 3 BERT (1 seed), k-means / SCAN / DEC at K=20/50; DeepCASAS / TDOST /
TreeCNN at L1/L2 × 100/250/all × 5 seeds. Scored by `pipeline/report_final.py` → `results/final/` (promoted).
Earlier preview scorer `final_scores.py` → `results/temp/final_scores/` (superseded).

## 2026-11-13 — matched-budget tables, seed std
Tables 16/17 rebuilt with one column per budget (≈100 / ≈250 / all), clustering + probes + baselines filled, bold
over the whole table; grey [std over seeds] under each number. Result: at ≈250, clustering 0.82/0.56 and hist probe
0.82/0.58 vs DeepCASAS 0.63/0.38, TDOST 0.60/0.31; all labels: TDOST 0.85/0.61, probe 0.85/0.66, clustering
0.82/0.57–0.59. m=all ≈ m=5. Seed std: clustering ≈0.01–0.02, supervised @100/250 ≈0.05–0.09.

## 2026-11-13 — TreeCNN investigation → dropped
Port predicts one class for every test window (Milan Kitchen_Activity, Cairo Dinner, Aruba Cook); val curve flat
from epoch 1; also collapses on preseg Cairo (0.53 W = majority). Untrained forward pass: logit spread across
inputs ≈2e-6 vs bias 6e-3 → input signal vanishes through 19 tree layers × 4 convs, no normalisation, convs over
the one-hot vocab axis. v1 reported 0.65 / 0.77 / 0.76 W; v1 code (former lab member) is lost, none public.
Decision (Alex): drop; see README. Response-letter paragraph drafted in chat.

## 2026-11-13 — confusion matrices (`pipeline/confusion.py` → `results/final/confusion/`)
TDOST, k-means hist / TF+ToD (K=50), TF+ToD probe; ≈250 and all; 5 seeds pooled. Patterns in `FINDINGS.md`.

## 2026-11-13 — order-sensitive classes: probe hist vs TF (`results/temp/per_class/probe_hist_vs_tf_per_class.csv`)
Hypothesis: sequence encoder beats histograms on order-dependent classes. Refuted for probes (3 seeds, all labels):
Milan Leave_Home hist 0.79 vs TF 0.52; Aruba Enter_Home 0.53 vs 0.10; Aruba Bed_to_Toilet 0.92 vs 0.20; Cairo Night
wandering 0.76 vs 0.64. TF wins only Cairo Bed to toilet (+0.24), Milan Meditate (+0.13). Door-sensor counts
identify leaving / entering.

## 2026-11-13 — per-class F1 from cluster assignments (`results/temp/per_class/clustering_per_class_f1.csv`)
No training. K=20/50, m=5/all, 5 seeds, k-means / SCAN / DEC on TF(+ToD) and k-means hist.
- SCAN is the only clustering that recovers Milan Leave_Home (0.45–0.66 vs k-means 0.00–0.09; k-means TF without
  ToD K=20 0.24). Also best on Milan Guest_Bathroom (0.68–0.72) and Cairo work in office (0.50–0.56).
- Appending ToD hurts k-means on Aruba at K=20 (Sleeping 0.97 → 0.41, Eating 0.72 → 0.10, Work 0.73 → 0.29); Aruba
  k-means TF (no ToD) K=50 gets Bed_to_Toilet 0.66.
- SCAN TF+ToD in Cairo looks like no-ToD: Breakfast 0.29–0.56, Lunch 0.21–0.59, Night wandering 0.14–0.47 vs
  k-means 0.76 / 0.86 / 0.79 → hypothesis: SCAN under-uses the ToD block.

## 2026-11-13 — SCAN checks (done, local MPS)
1. ToD weight for SCAN TF+ToD on Cairo: `frozen_deep --tod_weight w --suffix _todw{w}`, w ∈ {1,2,4,8}, seeds 1–5,
   K=20/50. Weight scales the ToD block for neighbour mining and after the head's standardisation.
   Local w=1 repro of skynet seed 1 K=50: 0.57/0.48 vs 0.54/0.46 (GPU nondeterminism).
2. SCAN on Sensor+ToD histograms, 3 houses × 5 seeds × K=20/50 (separates "SCAN" from "embedding" for Leave_Home).
Score: `python -m tist_revision.pipeline.scan_checks` → `results/temp/scan_checks/`. Results: TBD.
Early read (Cairo seed 1 only, L1 masked W/M, m=5): K=50 w=1 0.57/0.48 → w=2 0.71/0.50 → w=4 0.73/0.50 →
w=8 0.79/0.51 (k-means 0.81/0.58); Lunch 0.22 → 0.85, but Night wandering 0.00 at every w. K=20 erratic
(w=8 0.61/0.33). ToD under-use confirmed as the main cause; up-weighting closes most of the weighted-F1 gap,
not macro.
Final (5 seeds, `results/temp/scan_checks/`), L1 masked W/M:
- ToD weight, Cairo: K=50 m=5 w=1 0.59/0.48 → w=8 0.79/0.53 (k-means 0.82/0.59); m=all w=8 0.81/0.60 ≈ k-means
  0.81/0.59. K=20 w=8 still 0.68/0.43 (std 0.09) vs k-means 0.80/0.52. Partial fix; not tested on Milan/Aruba.
- SCAN on Sensor+ToD hist (no embeddings): Milan Leave_Home 0.51 at K=20 (= SCAN TF 0.51; k-means ≤0.02), Guest_Bathroom
  0.50, K=20 macro Milan 0.44 / Aruba 0.54 vs k-means 0.34 / 0.38. → the rare-routine gain comes from SCAN's balanced
  clusters, not from the encoder. Cairo SCAN hist K=20 0.68/0.44 (degrades less than SCAN TF 0.41/0.34).
- At K=50 SCAN ≈ k-means on Milan / Aruba (±0.03), worse on Cairo. Verdict: k-means default; SCAN = small-K option
  for rare personal routines (needs ToD up-weighting). Encoder still gets no unique credit.

## 2026-11-13 — SCAN main method: label-free ToD weighting (running, local MPS)
Decision: main method = TF + ToD + SCAN (README). ×8 was picked on Cairo test scores, so we use a label-free rule
instead: `frozen_deep --tod_weight balanced` gives both blocks equal total variance, computed on fit windows.
Head (after per-dimension standardisation) sqrt(512/6) = 9.24; neighbour mining sqrt(var_emb / var_tod) ≈ 0.50
in all three houses (the raw unit-norm ToD block already has ~4× the embedding variance, so mining was not the
ToD bottleneck; the head was). Runs: 3 houses × K=20/50/100 × seeds 1–5, suffix `_todwbal`.
Score: `python -m tist_revision.pipeline.scan_checks` → `results/temp/scan_checks/todw_summary.csv` (w=`bal`).
K=100 (all 45 runs done): m=5 three-house 0.82/0.60 (Milan 0.78/0.57, Aruba 0.87/0.62, Cairo 0.81/0.61);
m=all 0.84/0.62. No k-means K=100 runs on these seeds for comparison.
Result (K=20/50, 5 seeds, L1 masked W/M):
- Cairo fixed: K=50 m=5 0.79/0.54 (= ×8 0.78/0.53; unweighted 0.65/0.53; k-means 0.82/0.59). K=20 still 0.65/0.41
  (k-means 0.79/0.52).
- Costs Milan/Aruba some small-K gain: K=20 m=5 macro Milan 0.50 → 0.46, Aruba 0.56 → 0.50 (k-means 0.36 / 0.40);
  Milan Leave_Home K=20 0.51 → 0.33 (K=50 0.47; k-means ≤0.02).
- Three-house mean, SCAN-bal vs k-means: ≈100 0.73/0.46 vs 0.76/0.43; ≈250 0.80/0.55 vs 0.81/0.56; all 0.82/0.56
  vs 0.82/0.57. Verdict: SCAN (balanced) ≈ k-means; +0.03 macro at the smallest budget. Used as the paper's
  main method. Promoted 2026-11-13: `report_final.py` maps `_todwbal` → `scan`, unweighted SCAN on `+tod` →
  `scan_unw`; `confusion.py` uses SCAN; paper Section 6 / appendix numbers updated from `results/final/`.

## Ideas parked
- Cross-house transfer: house-agnostic tokens (sensor type, cleaned state, canonical room ~10 types, ToD/DoW/deltas;
  drop sensor id) or TDOST-style text; leave-one-house-out with 0 / 1 / 3 / 7 days of target adaptation vs
  histograms. The one remaining argument for the learned encoder.
- Active / uncertainty sampling for the probe at 100/250; supervised baseline trained on the windows clustering
  picked (separates selection from representation).
- ToD input for DeepCASAS (fairness); day-block bootstrap CIs on headline tables; more BERT seeds; K=100/200 with m=1.

## 2026-11-17 — Section 6 main table: significance + weighted-F1 consistency fix
- Question: is DISCOVER (SCAN TF+ToD) vs best supervised baseline significant per house / budget?
- Command: `python -m tist_revision.pipeline.report_final --pdf` (new `significance`, `main_table`, `significance_table`).
- Method: paired day-level bootstrap (2,000 resamples of test days; both methods on the same days, each averaged over
  5 seeds); average row = houses resampled independently, best baseline by three-house mean.
- Output: `results/final/data/significance.csv`, `tables/main_comparison.tex`, `tables/significance.tex`.
- Result: 100 / 250 labels significant (p<0.001) in Milan, Cairo and the average; Aruba W n.s. (−0.00 / −0.01),
  Aruba M +0.14*** at 100, +0.03 n.s. at 250. All labels: DISCOVER significantly below (avg −0.03** W / −0.05*** M).
- Bug found: clustering "masked" weighted F1 was averaged over masked classes only, baselines/probes over all classes
  (clustering ≈0.003 high, max 0.005). Fixed in `score_clustering`; runs.csv patched with the "all"-variant weighted
  F1 (identical values to a rescore). Single-seed temp explorations (e.g. SCAN on histograms) still use the old definition.
- Verdict: promoted to final (paper Table clf_comparison + appendix significance table).

## 2026-11-17 — Table 4 layout preview: split by metric, houses as columns
- Question: does Table 4 read better as two subtables (weighted F1, macro F1) with Milan / Aruba / Cairo / Average
  as column groups (100 / 250 / all labels each)?
- Command: `report_final.main_table_split(runs, sig, TEMP_DIR / 'table4_split' / <variant>, with_sd=...)`, preview
  `results/temp/table4_split/preview.tex` (acmsmall class, same macros).
- Result: inline seed std overflows the text width (~26pt even at footnotesize); stacked std (A) and no std (B) fit
  once the delta row drops the "+" sign and uses footnotesize with inline stars.
- Verdict: awaiting Alex's choice before replacing tables/clf_comparison.tex.
- 2026-11-17 follow-up: Alex chose variant B (no std). Promoted: `report_final` now writes
  `tables/main_comparison_{weighted,macro}.tex`; the old single `main_comparison.tex` generator is removed.

## 2026-11-19 — Section 5 annotation study re-analysis (v1 rater data)
- Question: do the Section 5 κ values survive the reviewer's critique (R2-2.12: rotating raters, cluster-level
  Fleiss), and what do the rated windows say about PDL vs CASAS (R2-3.2 Table 3, R2-2.14, R2-3.8)?
- Command: `python -m tist_revision.pipeline.rater_stats` (reads `discover-v1/evals/cluster_labels_from_raters.csv`
  + `discover-v1/rater_samples/<city>/encoded_samples/*.csv`).
- Output: `results/temp/rater_stats/{agreement,clusters,pdl_vs_casas}.csv`.
- Result: 100 windows × 2 ratings per house (599 ratings; 2 sample files missing → CASAS "unknown"). Window-level
  Fleiss κ = Krippendorff α = reported Cohen κ (M/A/C 0.85 / 0.89 / 0.87; level-up 0.88 / 0.92 / 0.92); 95% CIs
  (cluster bootstrap) ≈ ±0.08. Cluster-level Fleiss (v1 statistic) 0.60 / 0.63 / 0.63, CIs ≈ [0.45, 0.75].
  Agreement of two labels: same window 0.86–0.89, other window in same cluster 0.60–0.66, other cluster 0.04–0.11
  → the cluster-level κ gap is sample variation within clusters, not rater noise. "Other" never chosen.
  Categories used 26 / 22 / 14 (level-up 14 / 13 / 11); Cairo level-up top category 49% (prevalence).
  Majority share per cluster mean 0.74–0.78 (level-up 0.86–0.92); 8 / 60 clusters < 0.5.
  CASAS No_Activity among rated windows: 19% / 33% / 66%. PDL→CASAS homogeneity 0.60 / 0.62 / 0.33.
  Human majority share vs CASAS purity of the same 5 windows: Spearman 0.17 (n.s.).
- Verdict: κ numbers hold under the correct statistics (report Fleiss/α + CIs + decomposition). Table 3 cannot be
  regenerated (v1 cluster assignments lost); options in the 2026-11-19 chat (redo study on new clusters vs keep v1).
- 2026-11-19 follow-up: Alex chose to keep the v1 study. Added figures (`annotation_agreement`, `pdl_vs_casas`) and
  generated `kappa_scores.tex` to `rater_stats.py`; copied into the paper (Section 5, Table 2, Figs. 4–5); Table 3
  replaced by `pdl_vs_casas`. Rated-window check: Aruba office windows are CASAS Respirate (not Work) → claim dropped.

## 2026-11-19 — Annotation study redo prepared (not launched)
- Question: can the analyst rating tool (`src/analyst/rater.py` + `rater_template.html`) be reused to redo the
  Section 5 annotation on the revised clusters, as self-contained packets for independent raters?
- Command: `python -m tist_revision.annotation.packets build` (scoring: `... packets score`).
- Output: `results/temp/annotation_study/v2_k20/` (6 packets × 140 items, key, assignment, manifest).
- Checks: packet opened in a browser (replay, label picker, autosave, CSV export work); no cluster / stratum /
  CASAS strings in the HTML; scorer run end to end on simulated returns (840 annotations, 0 missing), then removed.
- Found while inspecting the v1 study and the taxonomy:
  - v1 `label_level_up` is not a function of `label` (e.g., Master bedroom all over → Master bathroom entering;
    Guest bathroom leaving → Guest bathroom entering; Bedroom→office ↔ Office→bedroom). The paper's level-up κ
    uses this column. The new tool derives the parent from the taxonomy group.
  - v1 dropdown ≠ appendix taxonomy figure (v1 had guest-bedroom entering/leaving, hallway "back and forth";
    figure is titled "Hierarchy of Atomic Activities"). New tool follows the figure.
  - Aruba has bathroom sensors (M004, M031, D003), contradicting the old Table 3 caption; Cairo has stairs,
    basement and laundry sensors with no taxonomy option.
- Open before launch: per-house option lists; random stratum (+40%); 10-min separation (top-5 reach rank 26,
  differs from the Section 6 simulated annotator); context 60 s; rater count and pilot timing; practice and
  repeated items; IRB/consent wording.
- Verdict: ready to send after Alex's decisions; Section 5 unchanged until results exist.
- 2026-11-19 follow-up (Alex): separation between windows of one cluster 10 → 2 min (top-5 now within
  confidence rank ≤ 11). IRB-exempt wording added to the packet instructions and to paper §5.2 (protocol number
  and contact email are placeholders `[XXXX]`).
- 2026-11-19 follow-up (Alex): packets record only rater ID, item ID, chosen label and note (timing and
  timestamps removed from the app, the export and the consent text).

## 2026-11-20 — Annotation study redo: final ratings (all houses)
- Rounds: Milan pilots `returned/v0.1_milan` (first instructions, leaf κ 0.37), `v0.2_milan` (new raters,
  reinstructed, 0.63), `v0.3_milan` (+ sensor summary, 0.76). Final: `returned/final`, all houses, packets with
  sensor summary + floor-plan overrides (Milan M003/M006, Aruba M004/M006/M029/M031/D003, Cairo M024/M011/M014).
- Command: `python -m tist_revision.annotation.packets score --returned final`
- Output: `results/temp/annotation_study/v2_k20/scores/final/` (agreement.csv, propagation.csv, ratings.csv).
- Result (top-5 windows/cluster, 100 windows × 2 raters per house; 840 ratings, 0 missing):
  window Fleiss κ leaf/parent Milan 0.79/0.83, Aruba 0.76/0.86, Cairo 0.78/0.80 (CIs ±~0.11);
  cluster κ leaf/parent 0.62/0.68, 0.61/0.74, 0.52/0.63; clusters with <50% majority 4/2/3;
  "None" 0.5%/0%/7.5%; propagation to random windows leaf/parent 0.29/0.56, 0.38/0.43, 0.21/0.35.
- Verdict: substantial (leaf) to almost perfect (parent) inter-rater agreement in all houses, ~0.07 below v1;
  propagation to random cluster members is weak. Paper §5 not yet updated.

## 2026-11-20 — Kyoto7 (twor.2009) trial: can a second multi-resident house go into Section 6?
- Question: do the Section 6 models (TF encoder + ToD histogram + SCAN balanced vs. DeepCASAS / TDOST) hold on
  Kyoto7 (2 residents, 54% of events inside ≥ 2 activities), and does the way the two residents are represented matter?
- Command: `python -m tist_revision.pipeline.kyoto_trial {prepare,report}`; output `results/temp/kyoto7_trial/`
  (`REPORT.md`, `data/`, figures). Round 1 = `kyotomerged`, `kyotoflat`, `kyotoresa`, `kyotoresb`; 5 seeds, Skynet.
- Result, round 1 (weighted F1, L1, ≈100 / ≈250 / all labels):
  merged: DISCOVER 0.64 / 0.65 / 0.75, DeepCASAS 0.71 / 0.78 / 0.86, TDOST 0.75 / 0.77 / 0.83;
  resident streams: DISCOVER 0.69 / 0.74 / 0.77, DeepCASAS 0.76 / 0.78 / 0.85, TDOST 0.75 / 0.77 / 0.84;
  R1 labels: DISCOVER 0.66 / 0.73 / 0.79, DeepCASAS 0.74 / 0.75 / 0.90, TDOST 0.70 / 0.75 / 0.90.
  A linear probe on the same DISCOVER features reaches 0.76–0.82 at ≈100–250 labels, so the features are fine and
  the loss is in the clustering step; the ToD block is the cause (see the ToD tables in `REPORT.md`).
- Verdict, round 1: DISCOVER as defined in Section 6 loses to both baselines at every budget in every variant.
  Not paper material as is.
- Round 2 (label representation, model unchanged): `kyotoaff` (concurrent event → running activity with the
  highest P(sensor | activity)), `kyotojoint` (household state, e.g. `Sleep+Work`), `kyotoattr` (resident
  streams, each event in one stream only). Data built; Skynet launch failed (host unreachable from 02:18, VPN route
  gone), nothing submitted. Local fallback running for seed 1 only (`num_workers: 0`, MPS).
- Label screen (no training, no test days): `python -m tist_revision.pipeline.kyoto_trial screen` →
  `results/temp/kyoto7_trial/LABEL_SCREEN.md`. 7 label schemes on the `kyotomerged` training windows and clusters,
  4 folds of training days × 5 seeds, weighted F1 at ≈100 / ≈250 labels; the "supervised" row is a logistic
  regression on the sensor histogram, a stand-in, not DeepCASAS / TDOST:
  most recently begun: DISCOVER 0.62 / 0.63, stand-in 0.71 / 0.72; sensor-attributed: 0.69 / 0.73 vs 0.77 / 0.79;
  household state (24 classes): 0.42 / 0.49 vs 0.43 / 0.46; R1 labels: 0.73 / 0.77 vs 0.79 / 0.81;
  R2 labels: 0.59 / 0.66 vs 0.71 / 0.73. k-means on TF features without ToD: 0.79 / 0.79 (sensor-attributed).
- Verdict, screen: sensor attribution lifts every method by about the same amount (+0.06 to +0.10), so it does not
  close the gap; only household-state labels put DISCOVER level with the stand-in, at F1 below 0.5. No label
  scheme screened makes the ToD-weighted model win. Provisional until the round-2 test-day runs exist.

## 2026-11-20 — Orange4Home trial: implemented, first run queued (preliminary)
- Question: can the unchanged pipeline run on a non-CASAS house?
- Data: open download, `data/raw/orange4home/` (1 occupant, 20 work days, 236 sources, 20 classes).
- Command: `python -m tist_revision.pipeline.orange_trial {prepare,report}`; output `results/temp/orange4home_trial/`.
- Choices (all mine, not yet reviewed): keep the 82 on/off and open/closed sources (presence, doors, switches,
  lights, shutters, TV status), drop the 154 meters and analog sources; state changes only (65,435 rows → 43,442
  events); `Cleaning` is one class; TDOST sentences built from the source names (TDOST has no layout for this house).
- Result so far (data only): 3,004 pool and 862 test windows, 4 test days; `Office_Computing` is 56% of the pool
  (one office presence detector produces 24,770 of the events); only 7 of 20 classes pass the macro-F1 rule
  (≥ 50 pool and ≥ 20 test windows).
- Verdict: runs end to end through sampling; no model results yet. The class imbalance and the 4 test days make
  this a weak benchmark as built; thinning the office presence detector is the first thing to try.
- 2026-11-20 05:00 follow-up, Kyoto7 round 2, seed 1 only, local encoder (MPS, `num_workers: 0`), quick scorer
  (top-5 mapping, weighted F1 on test windows; not the report code), ≈100 / ≈250 labels:
  DISCOVER `kyotomerged` 0.61 / 0.66 (Skynet seed 1), `kyotoaff` 0.68 / 0.82, `kyotojoint` 0.47 / 0.47,
  `kyotoattr` 0.79 / 0.86; k-means on TF without ToD: 0.82 / 0.81, 0.86 / 0.91, 0.55 / 0.62, 0.93 / 0.91.
  Baselines on the round-2 variants not run yet (seed 1 running locally), so no comparison. Caveat for
  `kyotoattr` (and `kyotoflat`): the streams are built from the activity annotations, test days included.
- 2026-11-20 05:20 follow-up, Kyoto7 round 2 with baselines, seed 1 only (local), report code:
  `python -m tist_revision.pipeline.kyoto_trial report --variants kyotoaff kyotojoint kyotoattr --tag round2`
  → `results/temp/kyoto7_trial/round2/`. Weighted F1, ≈100 / ≈250 / all labels:
  `kyotoaff`: DISCOVER 0.68 / 0.82 / 0.83, DeepCASAS 0.66 / 0.82 / 0.94, TDOST 0.74 / 0.72 / 0.81;
  `kyotojoint`: DISCOVER 0.47 / 0.47 / 0.56, DeepCASAS 0.48 / 0.58 / 0.74, TDOST 0.53 / 0.57 / 0.74;
  `kyotoattr`: DISCOVER 0.79 / 0.86 / 0.88, DeepCASAS 0.89 / 0.86 / 0.95, TDOST 0.80 / 0.88 / 0.93.
- Verdict, round 2 (one seed; round-1 seed std was 0.03–0.07, so gaps under ~0.07 are not evidence): with
  sensor-attributed labels DISCOVER is level with the baselines at ≈100–250 labels instead of 0.07–0.13 behind
  (merged stream); it still trails with all labels. Household-state labels do not help. Needs the 5 seeds on
  Skynet before any decision.
- 2026-11-20 06:05 follow-up, Orange4Home first model results, seed 1 only (local, MPS, `num_workers: 0`):
  `python -m tist_revision.pipeline.orange_trial report --rescore` → `results/temp/orange4home_trial/REPORT.md`.
  Weighted F1, ≈100 / ≈250 / all labels: DISCOVER 0.76 / 0.82 / 0.81, DeepCASAS 0.74 / 0.77 / 0.81,
  TDOST 0.72 / 0.74 / 0.82; macro F1 (7 of 20 classes): 0.47 / 0.61 / 0.58, 0.43 / 0.54 / 0.59, 0.38 / 0.47 / 0.65.
  ToD block: SCAN on TF without ToD 0.73 / 0.81 / 0.81 (ToD neither helps much nor hurts); k-means on TF 0.82 / 0.84 / 0.83;
  k-means on the sensor histogram 0.77 / 0.81 / 0.82.
- Verdict: DISCOVER is ahead of both baselines at ≈100–250 labels by 0.02–0.05 and level with all labels, the same
  pattern as Milan. One seed, 4 test days, 862 test windows, 59% of them `Office_Computing`: the margins are within
  the seed spread seen elsewhere (0.03–0.07). The significance stars in the generated report come from a bootstrap
  over 4 days and 1 seed and should not be read as evidence. A plain sensor-histogram k-means does as well, so the
  house may simply be easy at this window size. Needs 5 seeds and a decision on the office presence detector.
- 2026-11-20 15:35 follow-up, more local seeds (MPS, `num_workers: 0`; Skynet was unreachable until the afternoon,
  and SSH needs a fresh login, so nothing is submitted there). The lid was closed 09:45–15:18, which stalled the
  last run (Kyoto seed 3).
  - Orange4Home, 3 seeds (weighted F1, ≈100 / ≈250 / all labels, mean ± std): DISCOVER 0.77±0.01 / 0.81±0.01 /
    0.80±0.01, DeepCASAS 0.72±0.02 / 0.76±0.01 / 0.81±0.01, TDOST 0.71±0.02 / 0.74±0.01 / 0.80±0.04. Macro F1
    (7 of 20 classes): 0.53 / 0.60 / 0.59 vs DeepCASAS 0.42 / 0.50 / 0.59, TDOST 0.39 / 0.47 / 0.60. Day-block
    bootstrap vs. the better baseline: +0.05 [0.04, 0.07] and +0.05 [0.04, 0.05] weighted at ≈100 / ≈250 labels,
    −0.01 with all labels (4 test days, so the intervals are narrow for the wrong reason). ToD block: SCAN on TF
    alone 0.75 / 0.81 / 0.81; k-means on TF 0.81 / 0.84 / 0.84; k-means on the sensor histogram 0.76 / 0.80 / 0.82.
  - Kyoto7 `kyotoaff`, 2 seeds (seed 3 still training): DISCOVER 0.68±0.00 / 0.80±0.03 / 0.85±0.03, DeepCASAS
    0.68±0.03 / 0.81±0.01 / 0.93±0.01, TDOST 0.72±0.03 / 0.75±0.04 / 0.85±0.05. Without ToD: SCAN on TF
    0.87 / 0.88 / 0.91, k-means on TF 0.90 / 0.92 / 0.93.
- Verdict: Orange4Home reproduces the paper pattern (ahead at low budgets, level with all labels) on 3 seeds,
  with the caveats above (4 test days, one dominant class, a sensor-histogram k-means is as good). Kyoto7 with
  sensor-attributed labels: a tie at low budgets, behind DeepCASAS with all labels; the ToD block still costs
  0.08–0.19. `kyotojoint` and `kyotoattr` remain at one seed.
- 2026-11-20 16:00 follow-up, Kyoto7 `kyotoaff` with 3 local seeds (weighted F1, ≈100 / ≈250 / all labels):
  DISCOVER 0.71±0.04 / 0.81±0.03 / 0.87±0.03, DeepCASAS 0.68±0.02 / 0.79±0.04 / 0.94±0.01, TDOST 0.71±0.03 /
  0.74±0.04 / 0.86±0.04. Bootstrap vs. the better baseline: 0.00 [−0.05, +0.05] at ≈100, +0.02 [−0.01, +0.06] at
  ≈250, −0.07 [−0.11, −0.03] with all labels. Without ToD: SCAN on TF 0.87 / 0.87 / 0.92, k-means on TF
  0.91 / 0.93 / 0.93. Verdict unchanged: a tie at low budgets, a loss with all labels.

## 2026-11-20 — Orange4Home trial: core sweep, 5 seeds on Skynet
- Question: does the seed-1–3 local result hold with 5 seeds in the paper's environment? Core models only
  (DISCOVER = TF encoder + ToD + SCAN balanced, K 20 / 50; DeepCASAS; TDOST; budgets 100 / 250 / all), no ablations.
- Command: `skynet/tasks/orange_core_tf.txt` (5 tasks) and `orange_baselines.txt` (10 tasks), jobs 4817073 /
  4817074 on the `rail-lab` partition (A40, `--qos=short`; overcap was stuck on priority), then
  `python -m tist_revision.pipeline.orange_trial report --rescore` → `results/temp/orange4home_trial/`.
  The earlier local (MPS) Orange artifacts were moved to `artifacts/local_mps_orange/`; the report uses Skynet runs only.
- Result (weighted F1, ≈100 / ≈250 / all labels, mean ± std over 5 seeds):
  DISCOVER 0.75±0.02 / 0.80±0.02 / 0.81±0.01, DeepCASAS 0.70±0.03 / 0.75±0.03 / 0.82±0.01,
  TDOST 0.69±0.01 / 0.71±0.05 / 0.81±0.01. Macro F1 (7 of 20 classes): 0.46±0.06 / 0.61±0.05 / 0.62±0.02 vs
  DeepCASAS 0.39±0.05 / 0.46±0.09 / 0.59±0.00, TDOST 0.35±0.02 / 0.41±0.07 / 0.63±0.02.
  Day-block bootstrap vs. the better baseline: weighted +0.05 [0.05, 0.07] and +0.06 [0.05, 0.07] at ≈100 / ≈250,
  −0.01 [−0.02, 0.00] with all labels; macro +0.07 / +0.14 / −0.02 (n.s.). Per seed, DISCOVER ≥ both baselines at
  ≈100 labels in 5 of 5 seeds (one tie) and ahead at ≈250 in 5 of 5.
- Verdict: the paper pattern (ahead at low budgets, level with all labels) holds on 5 seeds. Limits unchanged:
  4 test days / 862 test windows, 59% `Office_Computing` (F1 0.97 for every method, so the gap comes from the
  small classes), 7 of 20 classes in the macro score, unreviewed preprocessing choices. Not promoted to `results/final`.

## 2026-11-20 — Section 5 figures from the final annotation study (preview)
- Question: what should Section 5 show with the new ratings (same clusters as Section 6)?
- Command: `python -m tist_revision.pipeline.section5_figures [--insets 18:l 13:l 17:r 14:r]`
- Output: `results/temp/section5/` (kappa_scores.tex, annotation_agreement, casas_to_pdl, tsne_milan as
  svg/pdf/png; clusters.csv; stats.json; cached t-SNE coordinates tsne_milan.npz).
- Figures are SVG → PDF via headless Chrome (style of the Section 6 figure). t-SNE: Milan, the main run's
  clustering features (TF s1 + ToD, 30,170 training windows), colored by the CASAS L1 label; clusters are split
  into several islands (kNN cluster purity 0.91 in features and in t-SNE), so each island is outlined.
- Grouping in casas_to_pdl: cluster → most frequent CASAS label of its windows, or "No CASAS label" if ≥ 75%
  of its windows are unlabeled.
- New evidence: windows without a CASAS label (33/49/74% of rated windows) are labeled consistently
  (κ 0.68/0.66/0.78, None 2/0/10%).
- Verdict: figures awaiting Alex's review; Fig. 5 (pdl_vs_casas), random windows and propagation go to the appendix.

## 2026-11-20 — Fig 6 t-SNE: three houses, clusters where PDLs beat CASAS
- Question: which clusters best show PDLs being more specific than CASAS (esp. kitchen); do Aruba/Cairo help?
- Command: `python -m tist_revision.pipeline.section5_figures [--rows milan:18,13 aruba:4,12 cairo:3,15]`
- Output: `results/temp/section5/tsne_clusters.{svg,pdf,png}` (t-SNE caches `tsne_<city>.npz`; Aruba/Cairo subsampled to 30k).
- Evidence: Milan medicine-cabinet clusters (5,15,18; 4,777 windows) fire M014 in 91-97% of windows, open cabinet D003 4x base
  rate, peak 8-10h and 17-19h; CASAS calls them Kitchen_Activity (77-82%) and has only 39 medication windows in total.
  Aruba c4 bedroom 97% unlabeled (8/10); Aruba c12 kitchen 54% unlabeled (10/10); Cairo c3 bedroom->office 82% unlabeled
  (10/10, office sensor 66%); Cairo c15 kitchen+dining split by CASAS into Breakfast 35% / wake 23% / none 38% (10/10).
  Rejected: Aruba "leaving via garage door" (c8/c18) - garage door opens in only 11-12% of windows.
- Verdict: draft v6 replaces the Milan-only figure; points s=1.1, alpha 0.9.
- Update (same day): user wants one house -> Milan only (`--tsne milan:18,13,16,14`): medicine cabinet, stove, guest bathroom
  (CASAS: none 30% / Guest_Bathroom 30% / Kitchen 19%), office armchair (none 44% / Watch_TV 24%). Points s=2.2, alpha 0.35
  so overlaps read as density.

## 2026-11-20 — Orange4Home: Section 5 rater packets (prepared, not launched)
- Question: do raters agree on DISCOVER's clusters in a non-CASAS home (same protocol as the CASAS study)?
- Command: `python -m tist_revision.annotation.orange_floorplan` (floor plan + sensor coordinates), then
  `python -m tist_revision.annotation.packets build --study orange`; score with
  `... packets score --study orange --returned <subfolder>`.
- Output: `results/temp/annotation_study/orange_k20/` (`packets/DISCOVER_annotation_homeD_R0x.html`, `key.csv`,
  `assignment.csv`, `manifest.json`); plan in `metadata/floor_plans_augmented/orange.{png,svg}`.
- Setup: run `orange__tf_s1+tod__scan_k20_todwbal` (seed 1), 20 clusters x (5 top + 2 random) = 140 items, 6 raters,
  2 per item (46-47 items each, all 15 pairs 9-10 times). "Home D", item ids D001-D140.
- Floor plan: walls, doors and furniture footprints are the vector geometry of the two Amiqual4Home PDFs, redrawn
  in the CASAS re-render style, both floors in one image. Orange4Home gives only the room of each sensor, so
  positions inside a room are schematic (presence over the named area, contacts on their door / cupboard / drawer,
  shutters on windows, lights on the ceiling, switch panels where the event stream puts them). Assumed: front door =
  the door at the bottom of the entrance (the plan has a second door top-left); office TV opposite the couch (not
  on the source plan); living-room shutters 1-5 on the five glazed openings in order.
- Replay app: typed sensors (`home.kinds`): presence as before, contacts as door squares, switch press / shutter as
  short flashes, light / TV marked while on, a text tag at each such event; the pause skip waits 4 s after an event.
  Sensor summary counts presence ON, contact OPEN, switch press, light / TV on and off, shutter open and close.
  CASAS packets rebuild to the same key, assignment, items and build ids.
- Taxonomy: 30 options for Home D. Reused: office, kitchen, living, front door, kitchen + living, bedroom <-> office,
  walk3, none. New: bedroom, bathroom (+ shower), toilet, upstairs walkway, cupboards, living-room table,
  bedroom <-> bathroom, office <-> bathroom, going up / down stairs. Left out: stove, desk (no binary sensor), simul.
- Caveat: 10 of the 20 clusters contain only `office_presence` events (53% of windows; 74 of the 140 items are
  Office_Computing). They differ by time of day (five at 9-11.5 h, five at 13-16.5 h), which the taxonomy does not
  ask about, so raters can only give them the same label. Window-level agreement will look high for a trivial
  reason, and the cluster-level contrast (same cluster vs. other cluster) will look low.
- Verdict: packets ready to send; not launched. Scoring checked end to end on synthetic answers (deleted).

## 2026-11-20 — Orange4Home: Section 5 rater statistics
- Question: same agreement statistics as for Milan / Aruba / Cairo, on the returned Home D packets.
- Command: `python -m tist_revision.annotation.packets score --study orange` (CSVs sit directly in
  `orange_k20/returned/`), then `python -m tist_revision.pipeline.section5_figures --study orange`.
- Output: `results/temp/annotation_study/orange_k20/scores/` (agreement.csv, propagation.csv, ratings.csv) and
  `results/temp/section5_orange/` (kappa_scores.tex, stats.json, clusters.csv; no figures, they are laid out for
  three houses).
- Result (100 top windows × 2 raters, 40 random windows × 2; 280 ratings, 0 missing, no "None"):
  window Fleiss κ leaf/parent 0.92 [0.83, 0.98] / 0.94 [0.88, 1.00] (α 0.92 / 0.94), exact agreement 0.94;
  cluster κ 0.79 [0.66, 0.91] / 0.83 [0.70, 0.95]; majority share 0.91 / 0.93, 1 cluster below 50% (c2, 0.4);
  P(agree) same window 0.94, same cluster other window 0.84, other cluster 0.25; propagation 0.71 / 0.78.
  CASAS for reference: window κ 0.76-0.78, cluster κ 0.52-0.62, other cluster 0.04-0.08, propagation 0.21-0.38.
- Office clusters: 10 of 20 clusters (53% of windows) are all "Movement in the office", 10/10 labels each. Without
  them (50 windows) window κ is 0.86, still above CASAS. The 20 clusters carry 7 distinct majority patterns
  (office ×10, kitchen + living ×2, couch ×2, bedroom ×2, walking through the house ×2, shower, living-room table),
  so P(agree | other cluster) is 0.25 against 0.04-0.08 in CASAS. Propagation is 1.00 in the office clusters and
  0.43 in the other ten.
- Reading: raters agree, but the labels are coarser than the dataset's activities. Bedroom reading vs. napping
  (c14), cooking vs. eating at the table (c19), TV vs. computing on the couch (c5, c6) get one pattern each; the
  binary sensors show the place, not the activity. Six disagreements in 100 windows, all between neighboring
  options (room vs. fixture, stairs vs. walking through the house).
- Added to `stats.json` for all studies: `distinct_patterns`, `top_pattern*`, `kappa_other_clusters`.
- Verdict: the protocol carries over to a non-CASAS home; headline κ is inflated by the office clusters, report
  0.86 next to 0.92. Not in the paper.


## 2026-11-20 — What changes if Orange4Home joins Sections 5 and 6 (preview)
- Question: effect of a fourth house on every Section 5 / 6 table, figure and headline number.
- Commands: `python -m tist_revision.pipeline.section5_figures --study with_orange`;
  `python -m tist_revision.pipeline.report_final --cities milan aruba cairo orange --out tist_revision/results/temp/with_orange/section6`
  (CASAS scores reused from `results/final/data/runs.csv`, Orange scored fresh incl. probes);
  `python -m tist_revision.pipeline.confusion --cities orange --out .../with_orange/section6` (+ merged per-class figure).
- Output: `results/temp/with_orange/{section5,section6}/`. Fig 6 parts for hand assembly: `results/temp/section5/fig6_parts/`.
- Section 6, four-house average vs three (L1 masked, W / M): DISCOVER 0.73/0.46, 0.80/0.57, 0.82/0.57 (M at 250 +0.02);
  Δ vs best baseline W +0.12/+0.15/−0.02 (was +0.15/+0.18/−0.03), M +0.15/+0.17/−0.05 (was +0.18/+0.17/−0.05), all
  p<0.01. Orange alone: Δ W +0.05/+0.06/−0.01 (n.s. at all), M +0.07/+0.14/−0.02 (n.s.). Probe vs DISCOVER: probe ahead
  by 0.02/0.03 M at 100/250 with Orange (was 0.02).
- Missing for Orange (cells are nan or lines drop): k-means TF+ToD (per-house appendix rows, design-choices table),
  K=100 (K×m figure loses its K=100 line), L2 baselines (per_house_l2), BERT. Local MPS ablations (3 seeds) are in
  `artifacts/local_mps_orange/`, not used.
- Section 5: Orange window κ 0.92/0.94, cluster κ 0.79/0.83, exact 0.94, other-cluster agreement 0.25 (CASAS 0.04–0.08);
  mean over houses window κ 0.78→0.81, cluster κ 0.58→0.64. Unlabeled-time result does not apply (1% of Orange
  windows unlabeled). casas_to_pdl Orange column: Office_Computing → office ×10 + shower + walking; Kitchen_Cooking →
  living-room table; patterns name places, Orange labels name activities.
- Verdict: pending Alex's decision; not promoted.

## 2026-11-20 — Orange4Home: runs still missing for full inclusion in the paper (count only, nothing launched)
- Question: if Orange4Home becomes a fourth house, which runs are needed to fill every table and figure of
  Section 6 and the appendix ablations?
- Method: compared `artifacts/assignments`, `artifacts/embeddings` and `artifacts/baselines/final` for orange with
  the grid of a complete house (Milan), and mapped each run group to the paper item that reads it.
- Done (45 runs): 5 transformer encoders, 10 SCAN balanced fits (K 20 / 50 × 5 seeds), 30 L1 baseline trainings
  (DeepCASAS, TDOST × 100 / 250 / all × 5 seeds), TDOST text embeddings. Encoders and embeddings are on Skynet.
- Missing (138 runs, 16 Skynet jobs):
  - 95 five-seed clustering fits on cached features: tf without ToD k-means / SCAN / DEC (30); tf+tod k-means,
    DEC, SCAN unweighted (30); hist_sensor k-means (10); hist_sensor_tod k-means (10) and SCAN (10); tf+tod SCAN
    balanced K=100 (5). 5 jobs, one per seed.
  - BERT, seed 1: 1 encoder training + 12 fits (bert, bert+tod × k-means / SCAN / DEC × K 20 / 50). 1 job.
  - L2 baselines: 30 trainings (DeepCASAS, TDOST × 3 budgets × 5 seeds). 10 jobs. Only for `per_house_l2`.
- No runs (local scoring): linear probes, m in {1, 5, all}, L2 re-scoring of the clusters, significance bootstrap,
  confusion matrices, per-class F1, dataset tables, Section 5 statistics.
- Not counted: the single-seed CASAS sweeps quoted in `app:clustering` (fine-tuned SCAN, entropy weight, neighbors)
  and the Cairo-only fixed ToD weight sweep (40 fits per house if wanted).
- Correction (2026-11-20): Orange4Home does have a published TDOST label set (Thukral et al., IMWUT 9(1), Art. 19,
  2025, doi:10.1145/3712278, Table 1 and Appendix Table 3; `TDOST/data_preprocess/merge_classes.py`). Our L2 mapping
  in `labels.py` matches it on all 24 room|activity labels (0 differences): 10 classes, Other, Relax, Cook,
  Leave_Home, Enter_Home, Sleep, Eat, Work, Bathing, Personal_Hygiene. No Bed_to_Toilet or Take_Medicine.
  Table 3 in the paper omits the Cooking → Cook and Eating → Eat rows; the code has them. Staircase going up/down and
  Bedroom dressing → Other. TDOST evaluates on pre-segmented activity windows (first 100 triggers), not our
  sliding windows, so their Orange numbers (DeepCASAS F1 79.1, TDOST-Basic 79.6) are not directly comparable.
  `artifacts/local_mps_orange/` (3 seeds, locally trained encoders) is not usable for the paper.


## 2026-11-20 — Orange4Home: launched the missing ablation runs on Skynet
- Question: run the 16 jobs counted above so Orange4Home can fill every Section 6 / appendix table.
- Command: `bash tist_revision/skynet/launch_final.sh orange_ablation_clustering orange_ablation_bert
  orange_ablation_baselines_l2` (task files in `skynet/tasks/`, same commands as the `kyoto_ablation_*` files).
  Jobs 4818078 (clustering, 5 seeds × 19 fits), 4818079 (BERT s1 + 12 fits), 4818080 (L2 baselines, 10 × 3 budgets).
- Output: `artifacts/assignments/orange__*`, `artifacts/embeddings/orange_bert_s1_embeddings.npz`,
  `artifacts/baselines/final/{deepcasas,tdost}/orange_l2_*`; pull with `skynet/sync.sh pull-results`.
- Expected time: about 15 minutes (comparable jobs took 2–9 min per clustering seed, 6–11 min for BERT, under 1 min per
  baseline). Actual: all 16 tasks finished in about 6 minutes, no failures.
- Result: 95 clustering fits + 12 BERT fits + BERT embeddings + 30 L2 baseline npz, all pulled locally (117 Orange
  assignment files in `artifacts/assignments/`). Verdict: Orange now has the full grid of a CASAS house; scoring
  (`report_final` / `confusion` with orange) still to run.

## 2026-11-20 — Orange4Home with meter events (`orangem`): Section 6 core models, 5 seeds
- Question: does adding on/off events derived from the meters the occupant drives (appliance and plug power,
  water flow, bed pressure) change the Section 6 comparison on Orange4Home?
- Repeated-state decision (checked on the raw rows, all 82 binary sources): keep state changes only, as in `orange`.
  TV status is polled at second :02 of every minute (20,844 of 20,910 rows are repeats, 97% "off"); a repeated light
  row comes 1 s after the first (median), after the same switch press (switch within 5 s for 97%); a repeated shutter
  row comes ~22 s after the command (motor travel), switch nearby for 23%. Presence and doors never repeat (a person
  moving keeps producing ON/OFF changes, which are kept). The repeats are logging artifacts, not actions.
- Meters (`orange_trial.METERS`, thresholds from the value distributions only, never from labels): cooktop > 100 W,
  hood > 20 W, five plugs > 5 W (standby 0–1 W), sink / shower / toilet water and bed pressure = any reading. A
  reading is active until the next one (≤ 60 s); gaps < 120 s are bridged (cooktop cycling). Left out: ambient
  (temperature, humidity, CO2, luminosity, noise), automatic loads (fridge, water heater, heaters), house totals;
  oven / dishwasher / washing machine never ran. 333 on/off pairs = 666 events (1.5% of 44,108), e.g. cooktop 19
  (= 19 Cooking), shower 19 (= 19 Showering), TV plugs 18 / 14 (= Watching_TV), toilet 12, bed 72, desk 40.
- Data: 3,053 pool / 878 test windows (orange: 3,004 / 862), same 4 test days.
- Command: `python -m tist_revision.pipeline.orange_trial prepare --variants orangem`; TDOST text local
  (`tdost_text --cities orangem`, 12 meter phrases in `tdost_text.ORANGE_METERS`); Skynet jobs 4818127
  (`orangem_core_tf`, 5 encoders + SCAN) and 4818128 (`orangem_baselines`), rail-lab A40 qos short; then
  `python -m tist_revision.pipeline.orange_trial report --variants orange orangem --tag meters --rescore`
  → `results/temp/orange4home_trial/meters/`.

## 2026-11-20 — Orange4Home ablations folded into results/final
- Command: scored Orange (clustering incl. the 107 new fits, L2 baselines, BERT probes; tf / hist probes reused,
  identical to the earlier scoring) via `report_final.score_*`, replaced the Orange rows of
  `results/final/data/runs.csv` (510 → 2,235 rows), then `report_final` (no rescore) and `paper_sync`.
- Result: every appendix table / figure now has Orange; figure axes say 4-house. Hand-written `design_choices.tex`
  and the clustering-methods table updated to four-house means. Orange: SCAN −0.12 macro vs k-means at K=20 (like
  Cairo); balanced ToD costs Orange 0.03 macro; no ToD dependence; probe +0.05 W / +0.13 M over DISCOVER at 100.
- Verdict: promoted. Orange confusion matrices still missing (`confusion/orange_250.png`).
