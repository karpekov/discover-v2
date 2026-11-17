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
