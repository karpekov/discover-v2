# Behavioral Analyst (`src/analyst/`)

Answers complex questions ("how has sedentary activity changed over the past months?")
with a short grounded report + figure, and exports blinded samples for human raters.
Implements g(Q_user, M) → {q_1..q_m} and f_HART from the paper. Supersedes the
prototype in `src/query/llm_reasoning.py` (first-vs-last-week only, Flash, SQLite caches).

## Design: LLM only at two points, everything else deterministic

```
question + home layout M
  │
  ▼  [LLM 1: planner, gemini-3.1-pro-preview, schema = AnalysisPlan]
plan: concepts (definition, detectability, 1-6 grounded queries,
      hard filters tod/dow/rooms-visited/required_sensors) + scope + analyses (timing | trend | compare_periods |
      time_of_day_profile | day_of_week_profile | overall_summary)
  │
  ▼  retrieval: cosine(query, window) ≥ γ_home, max over a concept's queries, hard filters
  ▼  measurement: union of [start, start+60s] intervals → matched minutes/day
  │               ÷ observed minutes/day under the same time filters (share_pct)
  ▼  stats: Kendall τ + OLS slope (trend), Mann-Whitney (periods), ToD/DoW profiles
  ▼  figure (PNG + PDF), matches/daily CSVs, rater item sampling
  │
  ▼  [LLM 2: reporter, same model, schema = Report] numbers only from stats
  ▼  ungrounded-number check (regex vs stats payload) → warnings
```

## Key decisions

- **Data: FD_60** (label-free 60 s sliding windows, 50% overlap, train+val+test merged).
  FD_60_p is presegmented with ground-truth boundaries and is not used here.
- **Sequence length 256** (matches paper evals). `SmartHomeRetrieval` (used by
  `SmartQuery` / dashboards) uses 20, which truncates windows.
- **Thresholds γ** (`analyst/config.py`): macro F0.5-optimal on FD_60 test
  (`results/evals/imwut_paper/retrieval_threshold_sensitivity_FD_60.md`):
  milan 0.075, aruba 0.05, cairo 0.14. F1 is degenerate for Aruba.
- **Minutes, not window counts**: overlapping windows would double count.
- **Coverage normalization**: days with < 30 observed minutes are excluded; Milan data
  before 2009-11-13 is excluded (effective start 2009-11-18, data gap).
- **Primary amount = matched minutes/day** (count of ~1-min matched windows; not the behavior's
  duration — reporter must phrase it as "N min/day of activity near X", never "spent N min doing X").
  Trend `significant`, summary arrows, figure (y = matched min/day, legend with episodes/day) and
  compare tests (`minutes_*` keys; `summary._compare_test` falls back to share for old results) use
  minutes; share_pct is secondary (reporter mentions it only if its verdict differs).
- **Amount vs timing**: trend/compare/profiles measure how much (minutes/day, share). `timing`
  measures when: episodes (gap ≤ 120 s) assigned to activity days starting at
  `day_boundary_hour` (default 4); `timing_event` first (wake-up) / last (bedtime) → per-day
  clock time, median/IQR, Kendall trend (min/week), first-vs-last-third (or plan periods)
  Mann-Whitney; `all` (meals, medication) → typical times, hourly % of days, episodes/day.
  Figure shows clock time per day instead of share. Summary arrows: ↑ later / ↓ earlier.
  Sessions: `session_gap_minutes` / `min_session_minutes` (meals 30/5). Scheduled behaviors are
  split into one concept per occurrence (breakfast/lunch/dinner, tod words in queries) with
  `Concept.timing_hours` = start window; sessions come from matches WITHOUT the tod filter and
  are kept if they start in the window (no clipping at 12:00/17:00 filter edges). Figure adds an
  overview row (one colour per occurrence). Known limitation: tod words in queries make the
  embedding favour windows after its bucket edges; `pct_days_within_10min_after_time_bucket_edge`
  > 30 → reporter flags the start time as edge-pinned (e.g. Milan dinner at 17:0x).
- **Structural filters**: `rooms` = window visits any listed room (not dominant room;
  bathroom trips are usually bedroom-dominated); `required_sensors` = a listed sensor
  fires ON/OPEN in the window (doors for leaving home). `retrieval.n_matched_without_structural_filters`
  gives the embedding-only count for ablation.
- **GT labels** never reach the LLM; `diagnostics.gt_label_mix_matched` in result.json
  is for internal sanity checks only.

## Caching (all default-on)

| What | Where | Key |
|------|-------|-----|
| LLM responses (planner, reporter) | `metadata/analyst_llm_cache/*.json` (git-tracked) | sha256(model, prompts, schema, settings) |
| Window embeddings + metadata + event stream | `data/analyst_cache/<home>/windows_*.{npz,meta.csv.gz,events.csv.gz}` | checkpoint + split file size/mtime |
| Query sentence embeddings | `data/analyst_cache/<home>/query_embeddings_*.npz` | sentence text |

`--refresh-llm` re-calls the API; `--offline` errors on any LLM cache miss. With warm
caches a full run needs no API and no checkpoint load.

## Usage

```bash
conda activate discover-v2-env
python src/analyst/run.py run                              # all questions × homes
python src/analyst/run.py run --homes aruba --only sedentary_trend
python src/analyst/run.py summarize                        # summary tables + dashboard
python src/analyst/run.py packet --name v1                 # rater packets
python src/analyst/run.py score --name v1 --ratings r1.csv r2.csv   # + rebuilds dashboard
python src/analyst/run.py dashboard                        # rebuild dashboard.html only
python src/analyst/run.py replot                           # redraw figures from saved artifacts (no LLM)
python src/analyst/run.py run --reuse-plan                 # keep saved plans, re-run retrieval/stats/report
python src/analyst/run.py review-packet --name review_v1   # 4-step pipeline review app (see below)
python src/analyst/run.py review-score --name review_v1 --ratings review_*.csv
python src/analyst/imwut_tables.py                          # paper table from review_v2 → results/analyst/imwut_tables/
```

Questions: `metadata/analyst_questions.json` (default; 29 caregiver questions, fields
name, homes, questions[num, id, category occurrence|change, optional group, question, optional homes]).
Numbered questions use the key `<num:02d>_<id>` (e.g. `07_sleep_well`) for run folders and
question_id in all outputs; `--only` accepts num, id, or key.
`configs/analyst/questions.yaml` holds the original 10 dev questions (`--questions` to use).

All outputs of a question file live in `results/analyst/runs/<name>/` (`name` field of the
file: `analyst_questions`, `dev_questions`):
- `<home>/<question_id>/`: `plan.json, result.json, report.md, figure.{png,pdf}` (no figure
  if all concepts undetectable; one row per concept: time series / clock-time scatter + compact
  time-of-day table (% of matches, % of active time per bucket) | floor-plan sensor bubbles), `daily_<concept>.csv, matches_<concept>.csv, rater_items.json`.
  Undetectable concepts skip retrieval/stats/rater items; the reporter explains why.
- `summary.{md,csv}`: one row per question, per home status (✓ detectable / ◐ partial /
  ✗ undetectable) + 1-3 word result (one cached LLM call compressing the report headlines,
  tag `shortanswers_<name>`).
- `summary_concepts.{csv,md,tex}`: one row per concept × home with the primary statistic.
- `rater_packets/<packet name>/`.
- `dashboard.html` (`dashboard.py` + `dashboard_template.html`, data embedded, figures linked
  relatively → open in place or serve the folder). Tabs: Summary (KPIs, per-home
  detectability/arrows, question table, skip reasons), Questions (one expandable row per
  question, 3 home columns: report, figure, behaviors/queries/stats, plan), Review.
  Review data is `review/review_v2_results/review_scores.json` when that packet exists, otherwise
  the latest `review/<name>_results/review_scores.json`, plus that packet's key and rating CSVs.
  Summary chips and the Review tab show step 1 (agreement with the analyst's
  answerable / not verdict, one item per question × home) and step 2 (share of the planner's
  retrieval queries rated relevant). `review_v2` adds 30 far-caption decoys per home (one per
  question, a second on the question with the most planner queries): the first training caption of
  a window far from that question's queries, unlabeled in the review app, `stratum=decoy` in the
  key, listed on the dashboard, and kept out of the step 2 relevance rate. Step 2 κ includes them. Steps 3–4,
  per-reviewer rates with a per-home breakdown, pairwise Cohen's κ, and Fleiss' κ per step are on the Review tab.
  Rebuilt by `summarize`, `score`, `review-score`, `dashboard`.

## Human rater evaluation

Per concept (seeded, ≥10 min apart): `random` 8 matched (unbiased precision of the
analyzed set — report this), `top` 4 highest-score matched, `negative` 4 eligible but
unmatched under the same time filters (false-positive control). Counts in
`AnalysisSettings`.

`packet` writes `results/analyst/runs/<set>/rater_packets/<name>/<home>.html` — a self-contained
rating app (animated floor-plan replay of the window ± 120 s context, target segment
highlighted, Yes/No/Unsure, ratings in localStorage, CSV export with columns
rater, packet, home, question_id, concept_id, item_id, rating, note, rated_at) — and `<home>_key.csv`
(hidden: stratum, score, sample_id). `score` reports yes-rate per concept/stratum with
Wilson 95% CI and pairwise Cohen's κ (unsure excluded).

## Pipeline review (4 steps → Sankey)

Report schema has `findings` [{concept_id, kind level|change|timing|comparison|pattern, statement}]:
atomic claims (one amount OR one change), rated individually in step 4.

`review.py` + `review_app/` → `results/analyst/runs/<set>/review/<name>/` (zip and send; static,
works from file://: index.html, app.js, style.css, data/<home>.js, img/). Answer key in
`<name>_key/<home>_key.csv` (outside the zip). Reviewer picks a house, rates one item at a time
(y/1 yes, n/2 no, u/3 unsure → auto-advance; ←/→ navigate). Questions appear in one fixed shuffled
order (seed independent of packet/home), not the YAML order. Each step has an intro card
(`tpl-intro-N` in index.html; examples must not overlap real questions), open until the first
rating in that step, then collapsed (click to reopen).
1. per question, blind (house map + sensor inventory, no analyst verdict): answerable at all?
   Scored as agreement with the analyst's status (undetectable vs not).
2. per retrieval query of each analyzed behavior: relevant? `review_v2` also shows 30 far-caption
   decoys per home in this step (unlabeled; `stratum=decoy` in the key, excluded from the relevance rate).
3. per behavior 4 top + 2 random matched windows + 2 far negatives (bottom 10% max-query similarity
   anywhere in scope, no filters); all samples of a home shuffled together with opaque ids, question
   not shown; replay as in rater packets.
4. per finding, with the question figure, an hour-of-day bar chart (`plots.plot_hourly`, matched
   min/day per hour on valid days, axis from noon; review-only, `img/<home>/<qkey>_hourly.png`)
   and key stats (`review._key_stats`: plain-language verdict first, p in parentheses; trends in min/day
   then share; first vs last third of valid days — or the plan's periods — with means, delta and
   Mann-Whitney p, computed from `daily_*.csv`).
All steps are reviewed regardless of earlier answers; export CSV columns
reviewer, packet, home, step, question_id, concept_id, item_id, rating, note, rated_at.

`review-score`: majority vote per item across reviewers (tie → unsure); per question × home a step
passes by majority of its decided items (step 3: top + random only). Funnel: all → answerable
(agreed) / not answerable (agreed) / wrongly answerable / missed / unsure → correct queries →
good retrieval → correct analysis. Writes `<name>_results/`: sankey_{consensus,home_*,reviewer_*}.{png,pdf},
review_scores.json, funnel_per_question.csv, review_report.html (also ungated per-step yes-rates
with Wilson CI; step-3 negatives yes-rate should be ~0; pairwise κ per step).

## Modules

| File | Role |
|------|------|
| `config.py` | homes, γ, paths, model id, `AnalysisSettings` |
| `llm.py` | `GeminiClient` (google-genai, structured output, file cache) |
| `schemas.py` | `AnalysisPlan`, `Concept`, `AnalysisSpec`, `Report` |
| `home_context.py` | layout M for the planner (rooms, sensors, date range, style captions, `layout_notes`); unlisted doors/rooms = no sensor. Milan door sensors: D001 front door (only exterior), D002 coat cabinet, D003 medicine cabinet (training captions still call D003 "garage door in kitchen"). Aruba has no guest bathroom: M023 and M024 are both `guest bedroom` |
| `index.py` | `SensorIndex`: cached embeddings, metadata, event stream, query encoding |
| `planner.py` / `reporter.py` | the two LLM steps (+ ungrounded-number check) |
| `retrieval.py` | concept → matched windows, interval union, episodes |
| `analysis.py` | daily table, trend/compare/profile/overall stats |
| `plots.py` | per-run figure |
| `rater.py` + `rater_template.html` | item sampling, HTML packet, scoring |
| `summary.py` | status / arrows / primary-stat helpers shared by summary + dashboard |
| `dashboard.py` + `dashboard_template.html` | static results dashboard |
| `review.py` + `review_app/` | 4-step reviewer package, scoring, Sankey funnel |
| `pipeline.py` / `run.py` | orchestration / CLI |
| `imwut_tables.py` | paper tables in `results/analyst/imwut_tables/`. `analyst_review` (review-tab rates; Fleiss' κ on Milan), `analyst_funnel` (`tab:analyst_funnel`), `analyst_questions` (`tab:analyst_questions`, 9 questions with ellipsis rows; check if the analyst answered; answered count over all 29; caption points at `tab:analyst_questions_all`), `analyst_questions_all` (appendix, all 29, same answered count) |
