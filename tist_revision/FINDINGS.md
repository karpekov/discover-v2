# Latest findings (TIST revision) — updated 2026-11-20

Curated current state; rewrite (don't append) when results change. Numbers: `results/final/` (L1 masked,
four-house mean over 5 seeds, weighted / macro F1; houses Milan, Aruba, Cairo, Orange4Home). History and raw
explorations: `EXPLORATIONS.md`.

## Headline
Main method (Alex, 2026-11-13): TF + ToD + SCAN (frozen encoder, balanced ToD weighting, `--tod_weight balanced`).
| | ≈100 labels | ≈250 labels | all labels |
|---|---|---|---|
| **SCAN TF+ToD** (K=20 m=5 / K=50 m=5 / K=50 m=all) | 0.73 / 0.46 | 0.80 / 0.57 | 0.82 / 0.57 |
| k-means TF+ToD (same budgets) | 0.77 / 0.47 | 0.81 / 0.58 | 0.82 / 0.58 |
| Linear probe TF+ToD (random labels) | 0.76 / 0.51 | 0.80 / 0.60 | 0.85 / 0.66 |
| DeepCASAS | 0.61 / 0.31 | 0.66 / 0.40 | 0.73 / 0.57 |
| TDOST | 0.59 / 0.27 | 0.63 / 0.33 | 0.84 / 0.62 |

Weighted F1 is over all true classes for every method (fixed 2026-11-17).

Per house (`tables/main_comparison_{weighted,macro}.tex`, Δ = SCAN minus best supervised, paired day-bootstrap
`data/significance.csv`): average Δ +0.12 / +0.15 at 100, +0.15 / +0.17 at 250 (p<0.001), −0.02 / −0.05 with all
labels (TDOST, p<0.01). Cairo carries most of the gain (supervised ≈0.3 W with few labels); Milan +0.05–0.08 W /
+0.15–0.17 M; Aruba W on par with DeepCASAS at 100/250 (n.s.), M +0.14 at 100 (p<0.001), +0.03 at 250 (n.s.);
Orange4Home +0.05 W / +0.07–0.14 M at 100/250 (p<0.001), on par with all labels (n.s.).

## What matters
- **Time of day.** Without it every representation collapses on Cairo (~0.33 W). DeepCASAS has no time input and
  stays at 0.41 / 0.40 on Cairo even with all labels. Orange4Home does not need it (k-means K=50 m=5 TF 0.83 / 0.63
  without vs 0.82 / 0.62 with ToD): its rooms and typed sensors already separate the activities.
- **Frozen, ToD-aware features + few parameters** win at 100–250 labels and are stable (seed std 0.01–0.02 vs
  0.05–0.09 for supervised).
- **K over m.** K=20 → 50 adds +0.11 macro; m=5 → all adds +0.01. At 100 labels K=100 m=1 0.80 / 0.58 vs K=20 m=5
  0.73 / 0.46; K=20 m=1 (20 labels) 0.70 / 0.42 > DeepCASAS @250 W.
- **With all labels**, TDOST and probes pass clustering; clustering's ceiling is cluster purity.

## What doesn't
- **Encoder choice** on a single house: TF ≈ BERT (1 seed) ≈ Sensor+ToD histograms (≤0.02; Orange4Home BERT+ToD
  +0.02 M, histograms −0.05 M).
- **Event order is not an argument for the encoder**: histogram probes beat TF probes on Leave/Enter_Home and
  Bed_to_Toilet (door-sensor counts).
- **DEC**; **m=all** over m=5.
- **Cluster-then-label vs a random-label probe**: probe ≥ clustering at equal budget (four-house +0.03 W / +0.05 M at
  100; Orange4Home +0.05 / +0.13) → argue clustering by workflow (inspect, discover, label 5 per cluster), not accuracy.

## SCAN vs k-means (5 seeds)
Four-house SCAN ≈ k-means: −0.04 W / −0.01 M at ≈100, −0.01 / −0.01 at ≈250, equal / −0.01 with all labels. At K=20
SCAN +0.10 macro on Milan (0.46 vs 0.36) and Aruba (0.50 vs 0.40), −0.11 on Cairo (0.41 vs 0.52) and −0.12 on
Orange4Home (0.46 vs 0.58). Milan Leave_Home: SCAN 0.33 (K=20) / 0.47 (K=50, m=5) vs k-means ≤0.02; lost again at
m=all (majority vote). Same gain with SCAN on histograms → from SCAN's objective, not the encoder. Balanced ToD
weighting (label-free) is required for Cairo (K=50 m=5 0.65 / 0.53 unweighted → 0.79 / 0.54) and costs Milan /
Aruba / Orange4Home some macro (K=20: 0.50→0.46, 0.56→0.50; Orange4Home K=50 m=5 0.64→0.61).
v1's claim "k-means substantially worse than SCAN" does not hold — corrected in Section 6.4.

## Label-level patterns (`results/final/confusion/`)
- **Same room + same time → inseparable** for every method/budget: Aruba Wash_Dishes → Meal_Preparation (86–97%),
  Milan Eve/Morning_Meds → Kitchen (100%), Cairo take medicine → Breakfast (80–100%), Milan Chores, Aruba
  Housekeeping. Label-definition limits; consider merging in L1.
- **Time-only classes** (Cairo Breakfast / Lunch / Dinner, wake / sleep): ToD clustering 0.68–0.95 at ≈250; TDOST
  @250 sends 71% of Lunch to Dinner, but is best with all labels (0.96 / 0.96 / 0.88).
- **Short transitions** (Bed_to_Toilet) split between sleep and bathroom / Night wandering; clustering ≈0.
- **Large clusters swallow small classes** (Milan Leave_Home → Kitchen 76–92% under k-means, 86% under SCAN m=all);
  majority vote makes m=all no help. Probes with all labels recover them.
- Orange4Home confusion matrices not generated yet.

## Decisions
- TreeCNN dropped (2026-11-13; see README). DeepCASAS + TDOST are the supervised baselines.
- Keep the small transformer over BERT (gate not met; 1 BERT seed).
- Encoder framed as one interchangeable representation unless cross-house transfer shows a gain.
- Orange4Home joined the paper as a fourth house (2026-11-20); its L2 labels follow TDOST's published mapping.

## Open / next
1. Orange4Home confusion matrices and per-class F1 figure.
2. Appendix prose still describes three houses in places (SCAN vs k-means, ToD weighting).
3. Cross-house transfer design (house-agnostic tokens); active-learning probe; ToD for DeepCASAS; CIs; BERT seeds.
