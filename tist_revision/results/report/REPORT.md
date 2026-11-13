# DISCOVER revision: clustering comparison report (2026-11-12, single seed 42)

Regenerate tables: `python -m tist_revision.pipeline.label_levels && python -m tist_revision.pipeline.report --m 5`
(LaTeX versions: `tables_m5.tex`, `tables_m1.tex`; raw numbers incl. 95% day-block CIs: `../label_levels/results.csv`).

## Protocol
- Clusters fit on continuous FL_20 train windows (no labels). Labelled pool = pre-segmented FL_20_p train;
  test = pre-segmented FL_20_p test (unseen days). Each cluster gets the majority label of its m most
  confident pool windows (m = 5 below; budget = m × K labelled windows).
- **L2** = TDOST/DeepCASAS merged classes (Milan 10, Aruba 9, Cairo 7). **L1** = raw CASAS activities (15 / 11 / 13).
- **masked** = all windows stay in pool and test, but F1 is averaged only over classes with ≥ 50 pool and
  ≥ 20 test windows (drops e.g. Aruba Enter/Leave_Home, Milan Take_Medicine). L1 keeps 11 / 7 / 12 classes.
  Class counts: `../label_levels/class_counts.csv`.
- W = weighted F1, M = macro F1. Median 95% CI width ≈ ±0.05 (W) / ±0.06 (M): differences < 0.05 are noise.
- Representations: *Sensor hist.* / *Sensor+ToD hist.* = normalised counts of sensor ids (and time-of-day
  buckets) in the window; *MLM* = our transformer pre-trained with masked event modelling (6 layers, d=512,
  random init); *MLM-time* = same with time-of-day / day-of-week tokens and time deltas in the input;
  *+ ToD* = L2-normalised embedding concatenated with the window's normalised time-of-day histogram.
- SCAN: frozen encoder, entropy weight 2, 20 mined neighbours, 5 heads. DEC: frozen encoder + linear adapter.

## Headline: three-house average, masked, by annotation budget (m labelled windows per cluster)

| Setup | L2 m=1 | L2 m=3 | L2 m=5 | L2 all | L1 m=1 | L1 m=3 | L1 m=5 | L1 all |
|---|---|---|---|---|---|---|---|---|
| k-means, MLM + ToD, K=100 | **0.83/0.58** | **0.86/0.61** | **0.86/0.62** | **0.87/0.66** | **0.78/0.54** | **0.81/0.59** | **0.82/0.59** | **0.83/0.62** |
| k-means, Sensor+ToD hist., K=100 | 0.81/0.57 | 0.84/0.59 | 0.85/0.60 | 0.86/0.63 | 0.77/0.54 | 0.81/0.57 | 0.81/0.57 | 0.82/0.61 |
| k-means, MLM, K=100 | 0.80/0.58 | 0.83/0.61 | 0.83/0.60 | 0.84/0.62 | 0.66/0.51 | 0.68/0.53 | 0.68/0.52 | 0.68/0.53 |
| SCAN, MLM, K=100 | 0.78/0.53 | 0.83/0.58 | 0.83/0.59 | 0.84/0.61 | 0.63/0.45 | 0.66/0.50 | 0.67/0.49 | 0.68/0.52 |
| DEC, MLM-time, K=100 | 0.80/0.53 | 0.82/0.55 | 0.83/0.56 | 0.83/0.58 | 0.64/0.46 | 0.65/0.47 | 0.66/0.48 | 0.67/0.50 |
| k-means, MLM, K=20 (≈ paper setup) | 0.79/0.51 | 0.81/0.54 | 0.80/0.52 | 0.82/0.59 | 0.61/0.41 | 0.63/0.46 | 0.65/0.47 | 0.66/0.50 |

## Full tables (m = 5)

### L2 (TDOST classes), all classes — comparable to TDOST / DeepCASAS

| Method | Representation | K | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M |
|---|---|---|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | 20 | 0.70 | 0.38 | 0.88 | 0.50 | 0.73 | 0.29 |
| k-means | Sensor hist. | 100 | 0.78 | 0.38 | **0.89** | 0.50 | 0.77 | 0.38 |
| k-means | Sensor+ToD hist. | 20 | 0.71 | 0.28 | 0.78 | 0.32 | 0.84 | 0.39 |
| k-means | Sensor+ToD hist. | 100 | 0.79 | 0.43 | 0.87 | 0.47 | **0.87** | **0.50** |
| k-means | MLM | 20 | 0.77 | 0.33 | **0.89** | 0.51 | 0.74 | 0.29 |
| k-means | MLM | 100 | 0.80 | 0.46 | **0.89** | 0.49 | 0.80 | 0.44 |
| k-means | MLM-time | 20 | 0.73 | 0.32 | 0.89 | 0.50 | 0.77 | 0.32 |
| k-means | MLM-time | 100 | 0.79 | 0.46 | 0.89 | 0.49 | 0.79 | 0.35 |
| k-means | MLM + ToD | 20 | 0.72 | 0.29 | 0.81 | 0.32 | 0.86 | 0.44 |
| k-means | MLM + ToD | 100 | **0.81** | 0.43 | 0.89 | 0.48 | **0.88** | 0.48 |
| k-means | MLM-time + ToD | 20 | 0.74 | 0.32 | 0.76 | 0.31 | 0.86 | 0.41 |
| k-means | MLM-time + ToD | 100 | 0.80 | 0.43 | 0.88 | 0.51 | **0.87** | 0.48 |
| SCAN | MLM | 20 | 0.76 | 0.33 | 0.85 | 0.36 | 0.77 | 0.33 |
| SCAN | MLM | 100 | 0.80 | **0.47** | 0.88 | **0.58** | 0.81 | 0.41 |
| SCAN | MLM-time | 20 | 0.75 | 0.39 | 0.88 | 0.49 | 0.77 | 0.34 |
| SCAN | MLM-time | 100 | 0.79 | 0.41 | 0.85 | 0.55 | 0.81 | 0.46 |
| DEC | MLM | 20 | 0.78 | 0.38 | 0.89 | 0.47 | 0.77 | 0.32 |
| DEC | MLM | 100 | 0.78 | 0.40 | 0.88 | 0.46 | 0.79 | 0.35 |
| DEC | MLM-time | 20 | 0.78 | 0.40 | **0.89** | 0.47 | 0.78 | 0.35 |
| DEC | MLM-time | 100 | 0.79 | 0.41 | **0.89** | 0.51 | 0.79 | 0.35 |

### L2, rare classes masked from the average

| Method | Representation | K | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M |
|---|---|---|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | 20 | 0.71 | 0.45 | 0.88 | 0.79 | 0.73 | 0.33 |
| k-means | Sensor hist. | 100 | 0.78 | 0.48 | **0.89** | 0.80 | 0.77 | 0.38 |
| k-means | Sensor+ToD hist. | 20 | 0.71 | 0.35 | 0.78 | 0.50 | 0.84 | 0.46 |
| k-means | Sensor+ToD hist. | 100 | 0.79 | 0.52 | 0.88 | 0.76 | **0.88** | 0.52 |
| k-means | MLM | 20 | 0.77 | 0.42 | **0.90** | **0.81** | 0.75 | 0.34 |
| k-means | MLM | 100 | 0.80 | 0.57 | **0.89** | 0.79 | 0.80 | 0.43 |
| k-means | MLM-time | 20 | 0.73 | 0.39 | 0.89 | 0.80 | 0.77 | 0.37 |
| k-means | MLM-time | 100 | 0.80 | 0.57 | 0.89 | 0.79 | 0.79 | 0.41 |
| k-means | MLM + ToD | 20 | 0.73 | 0.36 | 0.81 | 0.52 | 0.86 | 0.52 |
| k-means | MLM + ToD | 100 | **0.82** | 0.53 | 0.89 | 0.77 | **0.88** | 0.55 |
| k-means | MLM-time + ToD | 20 | 0.74 | 0.40 | 0.76 | 0.49 | 0.86 | 0.48 |
| k-means | MLM-time + ToD | 100 | 0.81 | 0.53 | 0.88 | 0.78 | **0.88** | **0.56** |
| SCAN | MLM | 20 | 0.77 | 0.41 | 0.85 | 0.58 | 0.77 | 0.39 |
| SCAN | MLM | 100 | 0.81 | **0.59** | 0.88 | 0.70 | 0.82 | 0.48 |
| SCAN | MLM-time | 20 | 0.76 | 0.49 | 0.88 | 0.78 | 0.77 | 0.40 |
| SCAN | MLM-time | 100 | 0.79 | 0.51 | 0.85 | 0.71 | 0.81 | 0.46 |
| DEC | MLM | 20 | 0.79 | 0.47 | 0.89 | 0.75 | 0.77 | 0.37 |
| DEC | MLM | 100 | 0.78 | 0.50 | 0.88 | 0.74 | 0.80 | 0.41 |
| DEC | MLM-time | 20 | 0.79 | 0.50 | **0.89** | 0.75 | 0.78 | 0.41 |
| DEC | MLM-time | 100 | 0.79 | 0.51 | **0.89** | 0.76 | 0.80 | 0.41 |

### L1 (raw CASAS activities), rare classes masked from the average

| Method | Representation | K | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M |
|---|---|---|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | 20 | 0.70 | 0.46 | 0.88 | 0.65 | 0.21 | 0.21 |
| k-means | Sensor hist. | 100 | 0.78 | 0.52 | 0.89 | **0.66** | 0.30 | 0.29 |
| k-means | Sensor+ToD hist. | 20 | 0.70 | 0.37 | 0.77 | 0.34 | 0.70 | 0.38 |
| k-means | Sensor+ToD hist. | 100 | **0.80** | 0.57 | 0.87 | 0.63 | 0.75 | 0.53 |
| k-means | MLM | 20 | 0.77 | 0.50 | **0.89** | **0.66** | 0.28 | 0.26 |
| k-means | MLM | 100 | 0.79 | 0.57 | 0.89 | 0.65 | 0.36 | 0.33 |
| k-means | MLM-time | 20 | 0.72 | 0.41 | 0.88 | 0.66 | 0.27 | 0.27 |
| k-means | MLM-time | 100 | 0.79 | **0.60** | 0.88 | 0.65 | 0.30 | 0.29 |
| k-means | MLM + ToD | 20 | 0.71 | 0.32 | 0.80 | 0.35 | 0.73 | 0.46 |
| k-means | MLM + ToD | 100 | **0.80** | 0.55 | 0.88 | 0.63 | **0.77** | **0.57** |
| k-means | MLM-time + ToD | 20 | 0.75 | 0.40 | 0.76 | 0.39 | 0.75 | 0.45 |
| k-means | MLM-time + ToD | 100 | 0.80 | 0.54 | 0.87 | 0.62 | 0.74 | 0.55 |
| SCAN | MLM | 20 | 0.76 | 0.40 | 0.84 | 0.48 | 0.34 | 0.28 |
| SCAN | MLM | 100 | **0.80** | 0.58 | 0.87 | 0.58 | 0.32 | 0.32 |
| SCAN | MLM-time | 20 | 0.75 | 0.49 | 0.88 | 0.62 | 0.27 | 0.23 |
| SCAN | MLM-time | 100 | 0.79 | 0.53 | 0.86 | 0.56 | 0.32 | 0.33 |
| DEC | MLM | 20 | 0.77 | 0.46 | 0.88 | 0.61 | 0.19 | 0.22 |
| DEC | MLM | 100 | 0.76 | 0.51 | 0.88 | 0.61 | 0.29 | 0.27 |
| DEC | MLM-time | 20 | 0.78 | 0.52 | **0.89** | 0.62 | 0.27 | 0.26 |
| DEC | MLM-time | 100 | 0.78 | 0.53 | 0.89 | 0.63 | 0.30 | 0.29 |

Linear probe (all pool labels, L1 masked) for reference: Cairo Sensor hist. 0.35/0.35, Sensor+ToD 0.79/0.61,
MLM 0.39/0.41, MLM-time 0.79/0.63; Milan 0.82–0.84 / 0.63–0.67 for all four.

## What the tables say
1. **K=100 beats K=20 on Milan and Cairo for k-means and SCAN** (+0.03–0.10 W, +0.05–0.20 M). Aruba is flat
   (two classes = 88% of windows) except for ToD features, which at K=20 split Relax/Cook by hour instead of
   activity (0.81/0.52 → 0.89/0.77 at K=100). DEC is flat in K.
2. **Time of day is necessary for Cairo** (meals Breakfast/Lunch/Dinner = 66% of L1 test, distinguishable only
   by hour). Without explicit ToD every method falls to 0.2–0.36 W on Cairo L1. MLM-time *contains* the time
   (probe 0.79/0.63) but clustering ignores it — it is a small part of the embedding variance.
3. **k-means on MLM + ToD (K=100) is the most consistent setup**: best or within noise of best in every
   house × label level; best three-house average at every budget.
4. **The learned encoder adds little over hand-crafted Sensor+ToD histograms** (+0.01–0.02 on the averages,
   within CI). It helps on Milan L2 (0.82 vs 0.79 W) and Cairo L1 macro (0.57 vs 0.53).
5. **SCAN's apparent macro gain on Aruba (L2 all: 0.58 vs 0.49) comes from one rare class** (Bed_to_Toilet,
   19 test windows). With rare classes masked SCAN is *worse* than k-means on the same encoder on Aruba
   (0.70 vs 0.79 M), equal on Milan (0.59 vs 0.57) and slightly better on Cairo L2 (0.82/0.48 vs 0.80/0.43),
   but still below k-means with ToD. Unfreezing the encoder (earlier grid) gave no gain at 5–10× cost.
6. **DEC ≈ k-means**, never better.
7. **Budget**: m=3 recovers ~95% of m=all; m=1 costs ~0.03 W.

## Pros and cons

| Method | Pros | Cons |
|---|---|---|
| k-means + Sensor(+ToD) histograms | No training, seconds to run, transparent; strong (R2-2.9(a) baseline) | Needs hand-chosen features; no transfer across houses |
| k-means + MLM | Learned features, no labels, best on Milan; probes show richer content | Ignores time unless added; ≈ histograms on clustering |
| k-means + MLM + ToD | Most consistent everywhere; cheap once encoder trained | Margin over histograms within CI; ToD weighting is a (fixed, untuned) choice |
| SCAN (frozen) | Uses neighbourhood structure; best Milan macro; recovers some rare classes | Extra training + hyper-parameters; no time info; gains vanish when rare classes masked; unstable without embedding standardisation |
| DEC | Simple deep-clustering baseline | ≈ k-means; no benefit |
| MLM-time encoder | Best linear probes, time encoded | Clustering does not exploit it; ≈ MLM |

## Proposed next steps
1. **Pick the headline evaluation** (needs your decision): L2 weighted (TDOST-comparable) + L2 masked macro,
   with L1 masked as a harder secondary table and budget curves (m = 1, 3, 5, all) in the appendix.
2. **SCAN on MLM + ToD** (neighbours mined and head trained on the concatenated vector): the one untested
   combination that could make SCAN competitive on Cairo. ~1 GPU-hour on skynet.
3. **5 seeds** (clustering seed; ideally also MLM pre-training seed) for the shortlist: k-means MLM + ToD K=100,
   k-means Sensor+ToD K=100, k-means MLM K=100, SCAN MLM K=100 (+ SCAN MLM + ToD if step 2 helps). Report
   mean ± std and keep the day-block CIs.
4. **Paper framing**: the honest claim is "annotation-efficient labelling of clusters with ~3 labels per
   cluster; learned sequence features + time of day ≈/≥ strong hand-crafted features", not "SCAN fine-tuning
   is needed". Update Table 4 / k-means table; fix "closest to centroid" vs highest-probability wording.
5. **BERT backbone (S2)**: given that the encoder barely beats histograms, decide whether it is still worth
   building, or reported as future work.
