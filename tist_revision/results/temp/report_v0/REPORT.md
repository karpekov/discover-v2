# DISCOVER revision: clustering ablations (2026-11-12, single seed 42)

> **Superseded protocol.** Everything below used the pre-segmented FL_20_p annotation pool and coordinate encoders.
> The tables/PDF now hold only the histogram baselines under the continuous-pool protocol (no coordinates).
> Continuous pool with No_Activity always predicted (L1 masked macro F1, m=5, Milan / Aruba / Cairo): Sensor+ToD
> hist. K=20 0.31 / 0.23 / 0.20, K=50 0.32 / 0.37 / 0.27; Cairo had 16/20 clusters mapped to No_Activity.
> Fix (Alex, as in the paper): No_Activity is never predicted on test. Annotators skip No_Activity windows;
> classifiers take the best non-No_Activity class. Then: K=20 0.35 / 0.37 / 0.52, K=50 0.42 / 0.57 / 0.62
> (pre-segmented pool: 0.37 / 0.34 / 0.50 and 0.50 / 0.61 / 0.62).

**All tables:** `report_tables.pdf` (13 pages) or `tables/*.md|tex`. Regenerate:
`python -m tist_revision.pipeline.label_levels && python -m tist_revision.pipeline.probes && python -m tist_revision.pipeline.report --pdf`

| File | Content |
|---|---|
| `tables/clustering_m5.*` | main clustering tables, m = 5 labelled windows per cluster, K = 20 / 50 / 100 blocks |
| `tables/clustering_m1.*`, `clustering_mall.*` | same with m = 1 and m = all (paper protocol: every labelled window votes) |
| `tables/probes.*` | linear probes on each representation with 100 / 500 / all random labels |
| `tables/probe_vs_clustering.*` | three-house averages, clustering vs probe at the same number of labels |

Each file has three evaluations: **L1 masked (primary)**, L1 all classes, L2 all classes (TDOST-comparable).
Cells are weighted / macro F1. Bold = best in its column within a block; shaded blue = within 0.03 of it.
Why 0.03: per-house 95% day-block CIs are about ±0.05, but every method is scored on the same test days, so
differences between methods are much less noisy than any one CI. 0.03 is the "indistinguishable" band; differences
of 0.05 or more are real for one seed. Averages over three houses are less noisy still.

## Protocol
- Clusters are fit on continuous FL_20 train windows (no labels). Each cluster gets the majority label of its m
  most confident labelled windows from FL_20_p train. Scored on FL_20_p test (unseen days).
- **L1** = each house's own CASAS activities. The resident id is removed: Cairo "R1 sleep" / "R2 sleep" become
  "sleep", because ambient sensors cannot tell which resident acts. This gives Milan 15, Aruba 10 and Cairo 11
  classes. **L2** = TDOST merged classes, kept only for comparability.
- **masked**: every window stays in, but F1 is averaged only over classes with ≥ 50 pool and ≥ 20 test windows
  (L1 keeps 11 / 7 / 10 classes).
- Representations:
  - *Sensor hist.* = sensor-id counts in the window. *Sensor+ToD hist.* adds time-of-day bucket counts.
  - *MLM* = our transformer (sensor, state, room and coordinates; no time).
  - *MLM-time* = MLM plus time-of-day / day-of-week tokens and time gaps between events.
  - *+ ToD* = the embedding with the window's time-of-day histogram appended, at equal weight.
- SCAN and DEC run on the frozen MLM encoder.

## Key numbers (L1 masked, three-house average, weighted / macro F1)

| | 100 labels | 250 labels | 500 labels | all labels |
|---|---|---|---|---|
| k-means, MLM + ToD (K = 20 / 50 / 100, m = 5; m = all at K = 100) | 0.77/0.40 | 0.82/0.56 | **0.84/0.61** | 0.85/0.65 |
| k-means, MLM-time + ToD | 0.77/0.44 | **0.83/0.59** | 0.83/0.60 | 0.84/0.64 |
| k-means, Sensor+ToD hist. | 0.75/0.41 | 0.81/0.58 | 0.83/0.60 | 0.83/0.64 |
| k-means, MLM (paper encoder, no time) | 0.65/0.48 | 0.68/0.54 | 0.69/0.53 | 0.69/0.55 |
| SCAN, MLM | 0.66/0.41 | 0.67/0.48 | 0.68/0.52 | 0.69/0.54 |
| DEC, MLM | 0.63/0.45 | 0.67/0.49 | 0.66/0.48 | 0.67/0.50 |
| Linear probe, MLM + ToD (random labels) | 0.77/0.51 | – | 0.83/0.62 | 0.86/0.69 |
| Linear probe, Sensor+ToD hist. | 0.80/0.54 | – | 0.83/0.63 | 0.86/0.68 |

## Assessment: where we are
1. **The representation is the bottleneck, not the clustering algorithm.**
   - At 500 labels, k-means plus cluster labelling matches a supervised linear classifier trained on 500 random
     labels from the same features (0.84/0.61 vs 0.83/0.62).
   - With all labels it comes within 0.01–0.04 of the full linear probe.
   - SCAN and DEC never beat k-means on the same features, and SCAN's earlier Aruba gain came from one rare
     class. Deep clustering should become an appendix ablation, not part of the method.
2. **Time of day is required, and has to be explicit.**
   - Without it, Cairo L1 (meals, sleep / wake) falls to about 0.3 weighted F1 for every clustering method.
   - The time-aware encoder (MLM-time) does contain time: its full probe reaches 0.86/0.68, against 0.72/0.59
     for MLM. But k-means only uses it once the time-of-day histogram is appended.
3. **The learned encoder is not better than hand-crafted Sensor+ToD histograms.**
   - The two are equal in clustering (0.84/0.61 vs 0.83/0.60 at K = 100) and in probes (0.86 both with all labels).
   - This is the biggest risk for the paper (Reviewer 2, comment 2.9(a)). The encoder needs a justification that
     these metrics don't capture: transfer across houses, the text / LLM interface, or robustness to sensor
     layout. Otherwise the paper should present the encoder as one option rather than the source of the gains.
4. **Number of clusters: 20 is too few, and 50 gets almost all of 100's benefit.**
   - At K = 20, time features *hurt* Aruba (0.80/0.35 vs 0.89/0.66 without them): the clusters split Relax / Cook
     by hour instead of by activity.
   - K = 50 gives 0.83/0.59, against 0.84/0.61 at K = 100. **Recommend K = 50**: 150 labels at m = 3, or 250 at
     m = 5.
   - Taking every labelled window (m = all) adds only about +0.01 W / +0.04 M over m = 5.
5. **Hierarchical merging was tested** (single seed, L1 masked, offline script):
   - Cutting a Ward tree over the 100 k-means centroids into 20–50 groups ≈ k-means directly at 20–50. No gain.
   - Adaptive splitting labels 3 windows per group and splits a group when they disagree. It ends up labelling
     46–126 groups per house and is not better than K = 50.
   - Clusters that share an activity are merged anyway once labelled, so the annotation cost is about K × m
     window glances. K = 50 with m = 3 is a reasonable real-world budget.
6. **Remaining hard cases:**
   - Cairo Breakfast / Lunch / Dinner: separable only by hour; the merged-resident L1 helps.
   - Rare classes with under 20 test windows (medication, Bed_to_Toilet, Enter / Leave_Home) are never recovered.
     They are reported, but masked from the headline macro F1.

## Next steps
1. **Baselines on the same splits.** Run TDOST, DeepCASAS and the third baseline on FL_20_p train → test, at L1
   and L2, with all labels and with 100 / 250 / 500 random labels. They are supervised, so the paper's argument
   is label efficiency: at 150–250 labels, our pipeline should beat them.
2. **Fix the method:** k-means, K = 50 (K = 20 as the paper-comparable reference), m = 3 and 5, on MLM + ToD and
   on Sensor+ToD hist.
3. **Retrain the encoders without coordinates** (configs ready: `configs/alignment/*_nocoord_*`). Then run
   5 seeds: k-means seeds, and ideally encoder pre-training seeds.
4. **Decide the encoder story** (see point 3 of the assessment) before rewriting the method section.
5. Move SCAN and DEC to an appendix ablation. Answer the reviewer's deep-clustering request with them.
