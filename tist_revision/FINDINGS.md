# Latest findings (TIST revision) — updated 2026-11-17

Curated current state; rewrite (don't append) when results change. Numbers: `results/final/` (L1 masked,
three-house mean over 5 seeds, weighted / macro F1). History and raw explorations: `EXPLORATIONS.md`.

## Headline
Main method (Alex, 2026-11-13): TF + ToD + SCAN (frozen encoder, balanced ToD weighting, `--tod_weight balanced`).
| | ≈100 labels | ≈250 labels | all labels |
|---|---|---|---|
| **SCAN TF+ToD** (K=20 m=5 / K=50 m=5 / K=50 m=all) | 0.73 / 0.46 | 0.80 / 0.55 | 0.82 / 0.56 |
| k-means TF+ToD (same budgets) | 0.76 / 0.43 | 0.81 / 0.56 | 0.82 / 0.57 |
| Linear probe TF+ToD (random labels) | 0.74 / 0.48 | 0.80 / 0.57 | 0.85 / 0.65 |
| DeepCASAS | 0.58 / 0.28 | 0.63 / 0.38 | 0.70 / 0.56 |
| TDOST | 0.55 / 0.24 | 0.60 / 0.31 | 0.85 / 0.61 |

Weighted F1 is over all true classes for every method (fixed 2026-11-17; clustering was ≈0.003 high before).

Per house (`tables/main_comparison_{weighted,macro}.tex`, Δ = SCAN minus best supervised, paired day-bootstrap
`data/significance.csv`): average Δ +0.15 / +0.18 at 100, +0.17 / +0.17 at 250 (p<0.001), −0.03 / −0.05 with all
labels (TDOST, p<0.01). Cairo carries most of the gain (supervised ≈0.3 W with few labels); Milan +0.05–0.08 W /
+0.15–0.17 M; Aruba W on par with DeepCASAS at 100/250 (n.s.), M +0.14 at 100 (p<0.001), +0.03 at 250 (n.s.).

## What matters
- **Time of day.** Without it every representation collapses on Cairo (~0.33 W). DeepCASAS has no time input and
  stays at 0.41 / 0.40 on Cairo even with all labels.
- **Frozen, ToD-aware features + few parameters** win at 100–250 labels (+0.2 W / +0.2 M over supervised) and are
  stable (seed std 0.01–0.02 vs 0.05–0.09).
- **K over m.** K=20 → 50 adds ~+0.13 macro; m=1 → 5 → all adds ~+0.03. K=20, m=1 (20 labels) beats DeepCASAS @250.
- **With all labels**, TDOST and probes pass clustering; clustering's ceiling is cluster purity.

## What doesn't
- **Encoder choice** on a single house: TF ≈ BERT (1 seed) ≈ Sensor+ToD histograms (≤0.02).
- **Event order is not an argument for the encoder**: histogram probes beat TF probes on Leave/Enter_Home and
  Bed_to_Toilet (door-sensor counts).
- **DEC**; **m=all** over m=5.
- **Cluster-then-label vs a random-label probe**: equal F1 at equal budget → argue clustering by workflow
  (inspect, discover, label 5 per cluster), not accuracy.

## SCAN vs k-means (5 seeds)
SCAN ≈ k-means overall: SCAN −0.03 W / +0.03 M at ≈100, equal at ≈250 and all. At K=20 SCAN +0.10 macro on Milan
(0.46 vs 0.36) and Aruba (0.50 vs 0.40), −0.11 on Cairo (0.41 vs 0.52); seed std up to 0.07. Milan Leave_Home:
SCAN 0.33 (K=20) / 0.47 (K=50, m=5) vs k-means ≤0.02; lost again at m=all (majority vote). Same gain with SCAN on
histograms → from SCAN's objective, not the encoder. Balanced ToD weighting (label-free) is required for Cairo
(K=50 m=5 0.65/0.53 unweighted → 0.79/0.54) and costs Milan/Aruba some K=20 macro (0.50→0.46, 0.56→0.50).
K over m: at 100 labels K=100 m=1 0.80/0.56 vs K=20 m=5 0.73/0.46; K=20 m=1 (20 labels) 0.70/0.41 > DeepCASAS@250.
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

## Decisions
- TreeCNN dropped (2026-11-13; see README). DeepCASAS + TDOST are the supervised baselines.
- Keep the small transformer over BERT (gate not met; 1 BERT seed).
- Encoder framed as one interchangeable representation unless cross-house transfer shows a gain.

## Open / next
1. Cross-house transfer design (house-agnostic tokens); active-learning probe; ToD for DeepCASAS; CIs; BERT seeds.
