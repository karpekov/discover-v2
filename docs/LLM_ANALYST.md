# LLM analyst: implementation and human-review results

Source note for the paper. Describes the system in `src/analyst/` and the human review of the
October 2026 run (`results/analyst/runs/analyst_questions/`). Counts and statistics below are
read from the saved `result.json` files. Review numbers are the majority vote in
`review/review_v2_results/review_scores.json`.

The planner and reporter were not called again. Two inventory errors were corrected in the
stored plans, and retrieval, stats, reports, and figures were recomputed locally for those
concepts. The review packet is `review_v2`: the original ratings, plus 30 far-caption decoys
per home, with the bad queries removed from the scored set. See
[What was corrected, and what the review still scores](#what-was-corrected-and-what-the-review-still-scores).

## 1. What the system is

A caregiver asks a free-text question about one instrumented home. The system answers with a
short report, a list of atomic findings, and a figure. Two calls to Gemini 3.1 Pro
(`gemini-3.1-pro-preview`, structured JSON output, thinking level `high`, temperature 1.0)
do the only language generation:

1. **Planner.** Question plus a description of that home's sensors → an `AnalysisPlan`.
2. **Reporter.** The plan plus statistics computed by code → a `Report`.

Retrieval, filters, time series, tests, and plots are deterministic. Ground-truth activity
labels never enter either prompt. Each LLM response is cached by
`sha256(model, system prompt, user prompt, schema, thinking level, temperature)` under
`metadata/analyst_llm_cache/`, so a paper re-run with `--offline` replays these exact outputs.

This replaces the earlier prototype in `src/query/llm_reasoning.py` (one comparison of the
first week against the last week, a smaller model, SQLite caches).

## 2. Data

Three CASAS homes, label-free FD_60 windows: 60-second windows, 50% overlap, train + val +
test concatenated. The pre-segmented FD_60_p split is not used. Embeddings use sequence
length 256 (the paper's retrieval model; a length of 20 truncates these windows). Checkpoint
suffix `fd60_seq_rb1_textclip_projmlp_clipmlm_v3`.

| Home | Residents | Windows in the index | Calendar span used | Days | Threshold γ |
|---|---|---|---|---|---|
| Milan | 1 | 42,829 | 2009-11-18 → 2010-01-06 | 50 | 0.075 |
| Aruba | 1 | 143,571 | 2010-11-04 → 2011-06-11 | 220 | 0.05 |
| Cairo | 2 | 45,955 | 2009-06-10 → 2009-08-05 | 57 | 0.14 |

γ is the macro-F0.5 cosine cutoff for label-prototype retrieval on the FD_60 test split
(`results/evals/imwut_paper/retrieval_threshold_sensitivity_FD_60.md`). F1 is degenerate for
Aruba (it keeps rising toward the lowest threshold, at about 24% precision), which is why
F0.5 is the operating point. Milan windows before 2009-11-13 are dropped in configuration
(the effective series starts 2009-11-18).

Sensors are ambient motion and door/contact sensors, plus a few temperature sensors that are
not used for retrieval. There are no cameras, wearables, appliance meters, or lock sensors.
The planner is shown an inventory built from `metadata/casas_metadata.json`: each sensor's
room id, type, and a short fixture description, the date span, the time-of-day buckets, and
a few example captions from the embedding model's training data so that queries match the
vocabulary the embedding was aligned to. Only listed rooms and fixtures exist. `layout_notes`
records home-specific facts the inventory alone does not (Aruba has no guest bathroom; Milan's
D003 is a medicine cabinet, not a door). The original plans were written before that field.
The stored plans for those two errors were edited afterwards; the other plans were not.

Time-of-day buckets, also available as hard filters:

| Bucket | Hours |
|---|---|
| early_morning | 05:00–08:00 |
| late_morning | 08:00–12:00 |
| afternoon | 12:00–17:00 |
| evening | 17:00–21:00 |
| night_before_midnight | 21:00–24:00 |
| night_after_midnight | 00:00–05:00 |

## 3. Questions

`metadata/analyst_questions.json`, 29 caregiver questions, run in every home (87 runs).
Questions 1–23 are occurrence questions; 24–29 are change questions. Wording is passed to
the planner verbatim. Run folders are `<num>_<id>`, e.g. `07_sleep_well`.

| # | Category | Question |
|---|---|---|
| 1 | occurrence | Does she ever leave the front door unlocked? |
| 2 | occurrence | Does she ever leave the garage door unlocked? |
| 3 | occurrence | Does she ever leave the side door unlocked? |
| 4 | occurrence | Does she ever leave the front door open? |
| 5 | occurrence | Does she ever leave the garage door open? |
| 6 | occurrence | Does she ever leave the side door open? |
| 7 | occurrence | Is she sleeping well? |
| 8 | occurrence | Is she sleeping for long stretches? |
| 9 | occurrence | Is she getting a good amount of sleep each night? |
| 10 | occurrence | Is she up and about in the middle of the night? |
| 11 | occurrence | Is she struggling to go up and down stairs? |
| 12 | occurrence | How much time is she spending out of the house? |
| 13 | occurrence | Is she eating regular meals? |
| 14 | occurrence | Is she going to the bathroom too frequently? |
| 15 | occurrence | Is she staying hydrated? |
| 16 | occurrence | Is she drinking enough? |
| 17 | occurrence | Is she not using the bathroom too frequently? |
| 18 | occurrence | Is she taking her medications on time? |
| 19 | occurrence | Is the door ever open at night? |
| 20 | occurrence | Is she up and about at night other than bathroom visits? |
| 21 | occurrence | Has she had any recent falls? |
| 22 | occurrence | Is she leaving appliances running when she's not home? |
| 23 | occurrence | Does she have people come to visit? |
| 24 | change | Are there sudden changes in her activity, not getting up to go about her day, not going out as much? |
| 25 | change | Have there been any major changes since she started her new medication? |
| 26 | change | Has there been a noticeable change in her appetite? |
| 27 | change | Has her routine changed in a major way? |
| 28 | change | Has her morning wake-up time shifted? |
| 29 | change | Is she forgetting things? |

## 4. Planner

The planner reads the inventory and the question and returns an `AnalysisPlan`:

- **interpretation** of the question and **caveats** (proxies, multiple residents, a short span).
- **concepts**, one per distinct behavior. A scheduled behavior that recurs at separate times
  of day (breakfast / lunch / dinner, morning / evening medication, wake-up / bedtime) is
  split into one concept per occurrence, each with its own time-of-day filter and with the
  time of day written into every query. The embedding encodes time of day, so the phrase
  already biases retrieval before the hard filter.
- **scope** dates. Phrases like "recently" are resolved against the end of the recorded data,
  not the date of the call. Null means the start or end of the available series.
- **analyses**, primary first, drawn from `trend`, `compare_periods`, `timing`,
  `time_of_day_profile`, `day_of_week_profile`, `overall_summary`.

Each concept has:

- `physical_definition`: one sentence a rater can check on a floor-plan replay.
- `detectable` and `detectability_note`. Motion and door sensors observe presence, dwell,
  movement between rooms, and door use. They do not observe posture, objects, appliances,
  locks, or hand actions. If a behavior can only be approximated, the note says so. If it
  cannot be observed, `detectable` is false and the query list is empty. The concept is then
  skipped: no retrieval, no statistics, no samples. The reporter must say the question is
  not feasible in this home and why.
- `retrieval_queries`: 1–6 caption-style sentences. One place or pattern per sentence
  (a disjunction retrieves poorly). No sensor ids. The sentence describes observable
  behavior, not an activity-label name.
- Hard filters, applied after retrieval, only when they are part of the definition:
  - `time_of_day`, `days_of_week`
  - `rooms`: the window must visit at least one listed room. This is a visit, not the
    dominant room, because a bathroom trip is usually dominated by the bedroom.
  - `required_sensors`: at least one listed sensor fires ON or OPEN. Used for doors.
    If the home has no such sensor, the planner is told not to substitute another sensor.
- For timing concepts, `timing_hours` is the window in which the event may start
  (breakfast [5, 11], lunch [11, 15], dinner [16, 21], wake-up [4, 12], bedtime [19, 3]).
  Windows of different occurrences do not overlap.

The instruction also says to ground every concept in this home's inventory. A door, room,
or fixture with no listed sensor is not feasible; another door or a cabinet contact must
not be substituted. Cabinet contacts are not exterior doors. A bedroom is not a bathroom,
and a room id that is not in the inventory must not be invented. Those sentences were
tightened after the original call, because of the Milan garage-door and Aruba guest-bathroom
errors. The stored plans for the affected questions were then edited by hand to match the
corrected inventory, and retrieval was rerun. Every other plan is the original model output.

## 5. Retrieval and measurement

For a detectable concept with queries q₁…qₘ:

1. Restrict to the date scope (and the home's configured start).
2. Apply time-of-day and day-of-week filters.
3. Apply room-visit and required-sensor filters. The count of matches before these two
   filters is stored as `n_matched_without_structural_filters` so embedding-only retrieval
   can be compared with filtered retrieval.
4. Encode each query with the same text tower as the window embeddings. The window score
   is the max cosine over the concept's queries. A window matches if that score is ≥ γ.

Overlapping windows are not counted as separate minutes. Matched minutes on a day are the
measure of the union of intervals `[window_start, window_start + 60s]`. The same union,
over every in-scope window that passes the time filters, is the day's observed active time.
The share is matched minutes divided by that observed time. Days with fewer than 30 observed
minutes are invalid and dropped from means, trends, and plots.

Episodes merge matched windows across gaps of at most 120 seconds. The headline quantities
are matched minutes per day, episodes per day, median episode length, and share of active
time. A matched minute is one minute of sensor activity whose window looks like the behavior.
It is not the duration of the behavior. "22 min/day near the medicine cabinet" means about
22 one-minute windows, grouped into fewer episodes, not 22 minutes of pill-taking. The
reporter prompt now forbids "spent N minutes doing X". The reports in this run predate that
wording rule; several findings still say "spent".

`share_pct` is a secondary measure. It moves when the resident's other activity changes even
if the behavior's minutes do not. Current code sets the trend flag, the figure, the dashboard
arrows, and the period test from minutes per day, and mentions share only when its verdict
differs. The concepts that were recomputed follow that rule. On the other concepts the stored
`significant` flag is still share's Kendall p < 0.05, which is what the original reporter saw.
Of the 53 stored trend tests, minutes and share disagree at p < 0.05 in 8, and in all 8 the
stored flag follows share.

### Timing

Amount statistics cannot answer "when". A timing analysis assigns episodes to activity days
that start at `day_boundary_hour` (default 4), so 01:00 belongs to the previous evening.

- `first` / `last`: one clock time per day (wake-up, bedtime). Reports median, IQR,
  percent of days with the event, a Kendall trend of the clock time in minutes per week,
  and a Mann–Whitney comparison of the last third of days against the first (or the plan's
  own periods).
- `all`: every session (meals, medication). Reports typical hours and how consistently they
  recur.
- Sessions for timing can use a wider gap and a minimum length (`session_gap_minutes`,
  `min_session_minutes`; meals were instructed to use 30 and 5). Momentary events stay at
  the 2-minute episode gap.
- Sessions are built from matches at any time of day, then kept if they start inside
  `timing_hours`, so a bucket edge at 12:00 or 17:00 does not clip the start.
- Known embedding artifact: time-of-day words in the query pull mass to just after a bucket
  edge (05:00, 08:00, 12:00, 17:00, 21:00, 00:00). If more than 30% of days fall within 10
  minutes after such an edge, the reporter is told to say the true start may be earlier and
  not to treat that clock time or its tight spread as the resident's schedule.

### Tests

- **Trend.** On valid days, ordinary least squares of the daily value on days since the
  first day, and Kendall's τ of the same pair. Reported for share and for minutes per day:
  slope per week, Kendall τ and p, fitted value on the first day and the last day, relative
  change of the fitted line. Current code sets `significant` from the minutes series.
  Unrecomputed concepts still store the original share flag.
- **Periods.** Default is the first third of valid days against the last third. Mann–Whitney
  U, two-sided. The saved results store the share comparison. The review app, for Aruba and
  Cairo, recomputed both minutes and share from `daily_*.csv` and showed those to the rater
  (see the review protocol).
- **Profiles.** Per time-of-day bucket and per weekday: matched minutes, observed minutes,
  share.

## 6. Reporter, findings, figure

The reporter receives the question, the plan, and the statistics with private keys stripped.
Rules in force for this run:

- Use only numbers that appear in the statistics. A regex check flags report numbers that
  match nothing in the payload, up to rounding. Integers from 0 to 10 are ignored. One
  warning fired in the whole run: a report contained `1.2` with no counterpart in the stats.
- Answer in a one-sentence headline and a 3–6 sentence body.
- Trend wording: p < 0.05 → "increased" or "decreased"; otherwise "no clear trend" or a
  slight, not statistically reliable, change. Same rule for a clock-time shift
  ("shifted" versus "no clear shift").
- **Findings:** 2–8 atomic claims. Each finding is one concept, one kind
  (`level`, `change`, `timing`, `comparison`, `pattern`), and one statement of at most about
  20 words. One finding is one amount or one change, never both. No findings for concepts
  that were not analyzed, and no findings that only restate a limitation. A rater scores
  each finding on its own.

The figure has one row per analyzed concept: a daily scatter, a 7-day mean, a trend line,
a small time-of-day table, and a floor plan with bubbles sized by sensor activations in
matched windows. Timing concepts use a clock-time scatter. Current code draws the amount
axis in matched minutes per day and puts the mean and the episode rate in the legend.
That redraw was applied only to the questions whose retrieval was recomputed (Milan 12, 19,
24 and Aruba 14, 17, 23, 25; Milan 5 no longer has a figure). The other figures are the
original share-of-active-time charts. The review packet for Aruba and Cairo additionally
showed an hour-of-day bar chart (matched minutes per day by hour, axis starting at noon)
built from the saved matches.

## 7. What the run produced

87 runs, all completed. No concept was partially detectable: each question was either fully
analyzed or entirely skipped.

| | Milan | Aruba | Cairo | All |
|---|---|---|---|---|
| Questions analyzed (at least one concept) | 19 | 19 | 14 | 52 |
| Questions judged not feasible | 10 | 10 | 15 | 35 |
| Concepts proposed | 40 | 42 | 41 | 123 |
| Concepts skipped | 10 | 10 | 15 | 35 |
| Concepts analyzed | 30 | 32 | 26 | 88 |
| Retrieval queries | 88 | 94 | 80 | 262 |
| Findings | 75 | 85 | 64 | 224 |

About 3.0 queries per analyzed concept. Query counts are lower than the original model output
because the Milan garage-door concept is now skipped and because garage-door and guest-bathroom
queries were removed from concepts that were kept. Finding counts are lower because question 5
has none and because the rewritten reports are shorter. Findings by kind, from the stored reports:

| Kind | Milan | Aruba | Cairo | All |
|---|---|---|---|---|
| level (amount or frequency) | 28 | 35 | 27 | 90 |
| change | 26 | 32 | 26 | 84 |
| timing | 16 | 13 | 10 | 39 |
| pattern (time of day or weekday) | 5 | 5 | 1 | 11 |
| comparison | 0 | 0 | 0 | 0 |

Analysis blocks actually computed on analyzed concepts (a concept can have several):

| | Milan | Aruba | Cairo |
|---|---|---|---|
| trend | 16 | 20 | 17 |
| timing | 18 | 12 | 17 |
| time-of-day profile | 10 | 12 | 6 |
| day-of-week profile | 6 | 5 | 3 |
| period comparison | 5 | 11 | 3 |

### Why questions were refused

The planner refused a question when the inventory could not support it. Recurring reasons,
paraphrased from `detectability_note`:

- **Locks.** Contact sensors report open and close, not locked. All three homes refused
  "left unlocked" for the front, garage, and side door, except where the reviewer later
  disagreed (section 9).
- **No such door.** Milan has no garage door and no side door. D001 is the only exterior door;
  D002 is a coat cabinet and D003 is the medicine-cabinet door. Aruba's doors are front, back,
  and garage (D004 is a real garage door). Cairo has no door contacts at all (entry is motion
  near the door).
- **Open versus a brief open/close event.** A 60-second window shows a door event, not that
  the door stayed open. This was used to refuse some "left open" questions and, in other
  homes, not used: Milan analyzes the front door, and Aruba analyzes the front door and the
  garage door, as door activity. Milan's garage-door question is refused.
- **Hydration and drinking.** No faucet, cup, or water sensor. Refused in all three homes.
- **Falls.** Presence and motion do not show posture or a drop to the floor. Refused in all
  three homes.
- **Appliances left on.** No power or appliance sensing. Refused in all three homes.
- **Stairs.** Refused in Milan and Aruba (no stair sensor). Analyzed in Cairo, which has
  sensors labeled as stairs.
- **Bathroom.** Analyzed in Milan (master and guest bathrooms; both exist) and in Aruba
  (bathroom and master bathroom only; there is no guest bathroom). Refused in Cairo:
  the inventory has no bathroom sensor. The reviewer disagreed for Cairo (section 9).
- **Medication on time.** Refused in Aruba (no pillbox or cabinet sensor) and in Cairo
  (M021 is a kitchen motion sensor described as the pantry; there is no medicine cabinet).
  Analyzed in Milan, where M014 is the kitchen medicine cabinet and D003 is its door.
- **Time out of the house.** Analyzed in Milan and Aruba. Refused in Cairo because the home
  has two residents and no door contacts, so absence of one person is not identifiable.
  The reviewer disagreed.
- **Forgetting, visitors, "sleeping well", appetite, a new medication.** These were usually
  analyzed via proxies (pacing, guest-room motion, in-bed presence, kitchen and dining
  motion). The reviewer often rejected that proxy as an answer to the question.

## 8. Human review protocol

The scored packet is `review_v2` at `results/analyst/runs/analyst_questions/review/review_v2/`.
It is a copy of `review_v1` with the additions below. Static files, one house at a time, one
item at a time. Ratings stay in the browser until export. Columns: reviewer, packet, home,
step, question_id, concept_id, item_id, rating, note, rated_at. The packet column is
`review_v2`.

Questions are shown in one fixed shuffled order (seeded, same order in every home), not
the order in the question file. Keys: y/1 yes, n/2 no, u/3 unsure, then the next item.
The answer key (`review_v2_key/`) was not in the packet.

Two reviewers. `alexkarpekov_oct241` rated all three homes on 24–25 October 2026.
`Archith Iyer` rated Milan only, including the Milan decoys. After the inventory edits,
1,374 items are in the key and 1,373 have a rating (1,797 rating rows). The one unrated
item is an Aruba retrieval sample that was already missing in v1. Majority vote: a yes/no
tie, or no decided rating, is unsure. An unsure rating is ignored when the other rater
decided. Aruba and Cairo have one rater, so the vote is his. Cohen's κ is Milan only.

### Step 1 — is the question answerable here?

The rater sees the floor plan (motion sensors and door sensors drawn and labeled) and the
sensor inventory, then the question, and answers whether the question can be answered at
least partly with this home's sensors. The analyst's verdict is not shown.

Scoring compares that answer with the analyst's status. Not feasible means every concept
was skipped. Anything else counts as the analyst saying answerable.

### Step 2 — is each query relevant?

For every analyzed concept, each retrieval query is shown with the question, the behavior
name, the physical definition, and the hard filters. The rater says whether that sentence
is relevant for finding the behavior in order to answer the question. Skipped concepts
contribute no queries.

`review_v2` adds 30 far-caption decoys per home, shown in the same step and not labeled as
controls. Twenty-nine questions get one decoy; the extra goes to the question with the most
planner queries (`27_routine_change` in every home). A decoy is the first training caption of
a window whose embedding has the lowest max cosine to that question's queries (or to all of
the home's queries, if the question has none). Windows are at least 30 minutes apart, and a
sample, a caption, or a caption that exactly matches a real query is not reused. In the key
they are `stratum=decoy`. They are excluded from the relevance rate and from the funnel's
step-2 pass, and included in step-2 κ, because that is the comparison that was stuck near
the ceiling when almost every planner query was relevant. Both raters marked every decoy
they saw as not relevant (Alex 90/90, Archith 30/30 Milan).

### Step 3 — does the highlighted minute show the behavior?

Per analyzed concept, eight windows:

- 4 highest-scoring matched windows (top)
- 2 random matched windows, separated in time (random)
- 2 far negatives: windows in the date scope whose best query similarity is at or below the
  10th percentile of that score, with no time, room, or sensor filter

All of a home's samples are shuffled together and given opaque ids (`mi-3-001`, …).
The question is not shown. The rater sees the behavior name, including any time-of-day
restriction, the physical definition, and a floor-plan replay of the window plus 120 seconds
of context on each side. The target segment is highlighted. "No" is expected for the
negatives, so a yes-rate near 0 on that stratum is the false-positive control.
The funnel uses top and random only.

### Step 4 — is each finding supported?

Findings of one question are listed together. The current finding is highlighted. The rater
has the question, the figure, and a statistics card (overall amount, trend, and, where
computed, clock times). Aruba and Cairo also had the hour-of-day chart and a plain-language
restatement of the tests: first third versus last third, the difference in minutes per day
and in share, and "statistically significant" or "no statistically significant change" with
the p-value in parentheses. Milan's findings were rated on the preceding packet, before that
chart and that restatement were added. The finding sentences themselves are the same objects
in all three homes. Unsure is allowed and is dropped from the decided-item rate.

### How a question moves through the funnel

Every question is reviewed at every step that has items. The funnel is conditional:

1. Reviewer unsure → stop as step-1 unsure. None did.
2. Analyst said not feasible, reviewer said no → **not answerable (agreed)**. Stop.
3. Analyst said not feasible, reviewer said yes → **missed** (the reviewer thinks it was
   answerable). Stop. Later steps have no items.
4. Analyst said feasible, reviewer said no → **wrongly judged answerable**. Stop. Queries,
   samples, and findings of that question were still rated and still count in the ungated
   item rates below, but the question does not continue in the funnel.
5. Both say answerable. Step 2 passes if a majority of its queries are yes. Step 3 passes
   if a majority of its top and random samples are yes. Step 4 passes if a majority of its
   decided findings are yes. The first failure stops the question.

Ungated item rates are the complementary summary: they do not drop questions that failed
an earlier step. For step 1 the "yes" rate is agreement with the analyst, not the fraction
of questions the reviewer called answerable. For steps 2–4 it is the fraction of decided
items rated yes. For negatives it is the fraction rated yes, which should be low.
Intervals are Wilson 95%.

## 9. Review results

Step 1 agreement, the query relevance rate, and the funnel use the corrected key: Milan
question 5 is not answerable, and queries that named a Milan garage door or an Aruba guest
bathroom are not in the scored set. Step 3 windows and step 4 finding sentences are the
original packet, except Milan question 5, whose samples and findings were removed with the
analysis. The recomputed match sets were not resampled.

### Funnel (question × home, majority vote)

| Outcome | Milan | Aruba | Cairo | All |
|---|---|---|---|---|
| Questions | 29 | 29 | 29 | 87 |
| Not answerable, agreed | 8 | 8 | 7 | 23 |
| Missed (vote: answerable; analyst: not) | 0 | 2 | 8 | 10 |
| Wrongly judged answerable | 2 | 4 | 3 | 9 |
| Step 1 unsure (raters split) | 4 | 0 | 0 | 4 |
| Agreed answerable | 15 | 15 | 11 | 41 |
| … and queries passed | 15 | 15 | 11 | 41 |
| … and retrieval passed | 15 | 14 | 10 | 39 |
| … and findings passed | 14 | 14 | 10 | 38 |

Of the 41 questions both sides called answerable, 41 passed the query check, 39 passed
retrieval, and 38 passed findings (38/41 = 92.7%). End to end, 38/87 questions were answered
with a report the vote accepted, and another 23/87 were correctly left unanswered.
The other 26/87 are the 10 misses, the 9 over-claims, the 4 Milan splits, the 2 retrieval
failures, and the 1 findings failure. The four splits stop at step 1 and do not continue.

Conditional failures:

| Home | Question | Where it stopped |
|---|---|---|
| Aruba | Is she going to the bathroom too frequently? | Retrieval. The stored queries and stats no longer include a guest bathroom. The rated windows are still the original sample, which did. |
| Cairo | Is she sleeping for long stretches? | Retrieval. |
| Milan | Are there sudden changes in her activity, not getting up to go about her day, not going out as much? | Findings. Queries and retrieval had passed. The rated sentences are the original findings, not the recomputed front-door report. |

### Item accuracy (ungated, majority vote)

The fraction in parentheses is yeses over decided items. `u=` is votes left unsure, which are
not in that fraction. Intervals are Wilson 95% on the decided items. Step 2 is planner queries
only. For steps 2–4, "yes" means the item was rated relevant, present, or supported. For
step 1 it means the vote agrees with the analyst.

| | Milan | Aruba | Cairo | All |
|---|---|---|---|---|
| Step 1 agreement | 92.0% (23/25, u=4) [75–98] | 79.3% (23/29) [62–90] | 62.1% (18/29) [44–77] | 77.1% (64/83, u=4) [67–85] |
| Step 1, analyst said answerable | 88.2% (15/17, u=2) [66–97] | 78.9% (15/19) [57–91] | 78.6% (11/14) [52–92] | 82.0% (41/50, u=2) [69–90] |
| Step 1, analyst said not | 100% (8/8, u=2) [68–100] | 80.0% (8/10) [49–94] | 46.7% (7/15) [25–70] | 69.7% (23/33, u=2) [53–83] |
| Step 2 query relevant | 97.4% (75/77, u=11) [91–99] | 91.5% (86/94) [84–96] | 91.2% (73/80) [83–96] | 93.2% (234/251, u=11) [89–96] |
| Step 2 decoy rated relevant | 0% (0/30) [0–11] | 0% (0/30) [0–11] | 0% (0/30) [0–11] | 0% (0/90) [0–4] |
| Step 3 top matches | 97.2% (103/106, u=14) [92–99] | 93.0% (119/128) [87–96] | 88.5% (92/104) [81–93] | 92.9% (314/338, u=14) [90–95] |
| Step 3 random matches | 92.7% (38/41, u=19) [81–98] | 88.9% (56/63) [79–94] | 94.2% (49/52) [84–98] | 91.7% (143/156, u=19) [86–95] |
| Step 3 far negatives rated yes | 1.9% (1/52, u=8) [0–10] | 15.6% (10/64) [9–26] | 25.0% (13/52) [15–38] | 14.3% (24/168, u=8) [10–20] |
| Step 4 finding correct | 90.5% (57/63, u=15) [81–96] | 100% (88/88, u=1) [96–100] | 100% (64/64) [94–100] | 97.2% (209/215, u=16) [94–99] |

Milan's step-1 and step-3 rates are higher than a single rater's because a split becomes
unsure and leaves the decided set. Alex alone, on the same corrected key, agrees with the
analyst on 74.7% of 87 questions, rates 90.8% of 262 planner queries relevant, rates 17.6%
of 176 far negatives as showing the behavior, and accepts 97.2% of 215 decided findings.
Aruba and Cairo are unchanged from his ratings. Cairo's step-1 agreement is lower because he
called 8 of 15 refused questions answerable anyway.

Top-match precision stays high (93% pooled). Random matched windows are close behind (92%).
Far negatives are not at 0. On decided votes, 14% were rated as showing the described
behavior; Alex alone is at 18%. That is the false-positive rate of the bottom decile of
similarity, with no structural filter. It is highest in Cairo (25%). Milan's 1.9% is both
raters saying yes. Archith's own negative rate on Milan is 3.3% (2/60); Alex's is higher,
and most of those conflicts are the 8 unsures.

Findings were rarely rejected. Archith did not rate step 4, so that row is Alex's vote in
every home. The six Milan "no" findings and the Milan unsures are his. The rate is not
conditional on step 1: it includes findings of questions he had called unanswerable.

### Agreement between the two Milan raters

Cohen's κ, yes/no only, on items both rated. Step 2 includes the 30 decoys.

| Step | κ | n | What is being compared |
|---|---|---|---|
| 1 | 0.684 | 27 | Is the question answerable? |
| 2 | 0.780 | 116 | Is the query relevant? Planner queries plus decoys. |
| 3 | 0.621 | 240 | Does the window show the behavior? |
| 4 | — | 0 | Archith did not rate findings. |

Before the decoys, step 2 was mostly "yes" from both raters (77 both-yes, 2 both-no, 13
splits on 92 planner queries), so chance agreement was already 0.83 and κ was 0.17 even
though they agreed on 86% of items. The decoys are both-no controls. They pull chance
agreement down, and κ on the enlarged set is 0.78. The relevance rate above does not include
them. The 13 planner-query splits were concentrated on visitors, forgetting, the medication
change, the garage door, and one vague front-door query. The garage-door queries are no
longer in this κ: they were removed with the inventory fix.

### Where the reviewer and the analyst disagreed at step 1

**Missed** — reviewer said answerable, analyst refused:

| Question | Milan | Aruba | Cairo |
|---|---|---|---|
| Front door left unlocked | raters split | missed | missed |
| Garage door left unlocked | agreed not answerable | missed | missed |
| Front door left open | (analyzed; passed) | (analyzed; passed) | missed |
| Garage door left open | agreed not answerable | (analyzed; passed) | missed |
| Time out of the house | (analyzed; passed) | (analyzed; passed) | missed |
| Bathroom too frequent | (analyzed; passed) | (analyzed; retrieval failed) | missed |
| Bathroom not too frequent | (analyzed; passed) | (analyzed; passed) | missed |
| Door open at night | (analyzed; passed) | (analyzed; passed) | missed |

Cairo accounts for 8 of 10 misses. Cairo has motion sensors only. The analyst treated door
state, absence from the house, and bathroom use as unobservable. Alex still marked those
questions answerable. Milan's front-door "unlocked" question is no longer a miss: Alex said
answerable and Archith said not, so the vote is unsure.

**Wrongly judged answerable** — vote said not answerable, analyst produced an analysis:

| Question | Homes |
|---|---|
| Is she sleeping well? | Milan, Aruba |
| Does she have people come to visit? | Aruba, Cairo (Milan: raters split) |
| Have there been any major changes since she started her new medication? | Aruba, Cairo (Milan: raters split) |
| Is she forgetting things? | Milan, Aruba, Cairo |

Visitors, a new medication, and forgetting were rejected by Alex in every home: the proxies
(guest-room motion, kitchen or sleep changes, pacing) were not accepted as answers to those
questions. Archith accepted the visitor and medication proxies in Milan, which is why those
two are splits rather than over-claims. "Sleeping well" was rejected in Milan and Aruba but
accepted in Cairo, where the same style of nighttime bed and wandering concepts then passed
queries, retrieval, and findings. Milan's garage-door question is now an agreed refusal.

**Milan splits** — one rater yes, the other no:

| Question | Alex | Archith | Analyst |
|---|---|---|---|
| Front door left unlocked | answerable | not | not answerable |
| Struggling on stairs | not | answerable | not answerable |
| Visitors | not | answerable | analyzed |
| Change since a new medication | not | answerable | analyzed |

Archith was unsure, and Alex said not answerable, on both hydration questions. That is an
agreed refusal, because an unsure rating does not outvote a no.

### Per-question outcome

`pass` = agreed answerable and findings passed. `refused` = agreed not answerable.
`unsure` = the two Milan raters split, so the question stops at step 1.
`miss` / `overclaim` / `retrieval` / `findings` as in the funnel. Milan cells are the
majority vote. Aruba and Cairo are Alex's ratings.

| # | Question | Milan | Aruba | Cairo |
|---|---|---|---|---|
| 1 | Front door unlocked | unsure | miss | miss |
| 2 | Garage door unlocked | refused | miss | miss |
| 3 | Side door unlocked | refused | refused | refused |
| 4 | Front door left open | pass | pass | miss |
| 5 | Garage door left open | refused | pass | miss |
| 6 | Side door left open | refused | refused | refused |
| 7 | Sleeping well | overclaim | overclaim | pass |
| 8 | Sleeping for long stretches | pass | pass | retrieval |
| 9 | Good amount of sleep | pass | pass | pass |
| 10 | Up in the middle of the night | pass | pass | pass |
| 11 | Struggling on stairs | unsure | refused | pass |
| 12 | Time out of the house | pass | pass | miss |
| 13 | Regular meals | pass | pass | pass |
| 14 | Bathroom too frequent | pass | retrieval | miss |
| 15 | Staying hydrated | refused | refused | refused |
| 16 | Drinking enough | refused | refused | refused |
| 17 | Bathroom not too frequent | pass | pass | miss |
| 18 | Medication on time | pass | refused | refused |
| 19 | Door open at night | pass | pass | miss |
| 20 | Up at night, not bathroom | pass | pass | pass |
| 21 | Recent falls | refused | refused | refused |
| 22 | Appliances left on | refused | refused | refused |
| 23 | Visitors | unsure | overclaim | overclaim |
| 24 | Sudden activity change | findings | pass | pass |
| 25 | Change since a new medication | unsure | overclaim | overclaim |
| 26 | Appetite change | pass | pass | pass |
| 27 | Routine change | pass | pass | pass |
| 28 | Wake-up time shifted | pass | pass | pass |
| 29 | Forgetting things | overclaim | overclaim | overclaim |

Questions that passed in every home where the analyst attempted them, and were not missed:
regular meals, up at night other than bathroom visits, appetite, routine change, and
wake-up time. Questions refused in every home: side door unlocked, side door left open,
hydration, drinking, falls, appliances left on.

## 10. What was corrected, and what the review still scores

The planner was not called again. Metadata, `layout_notes`, and the planner prompt now
describe the inventory below. The stored plans, retrieval, stats, reports, and figures were
edited locally for the two errors. `dashboard.html` reads those files and `review_v2`.

**Milan has no garage door.** D001 is the front entrance and the only exterior door. D002 is
a coat cabinet. D003 is the medicine-cabinet door in the kitchen; M014 is that cabinet.
Training captions still say "garage door in kitchen" for D003. Question 5 is undetectable.
Questions 2 and 6 say so in the note. Questions 12, 19, and 24 keep only D001:

| Question | Concept | After the correction |
|---|---|---|
| Time out of the house | leaving or arriving | 3.5 min/day, 3.0 episodes/day, 0.8% of active time (was about 10 min/day and 7.5 episodes, counting D003) |
| Door open at night | nighttime door | 0.2 min/day, 0.12 episodes/day (was 0.4 min/day) |
| Sudden activity change | going out | 4.6 min/day, 3.8 episodes/day, no trend (p=0.34). Wake-up is unchanged at 06:11 |

**Aruba has no guest bathroom.** M023 and M024 are both in the guest bedroom. Guest-bathroom
queries and that room id were removed from questions 14, 17, 23, and 25, and those concepts
were recomputed. Aruba does have a garage door (D004); those queries were left as they were.
Milan's guest bathroom is real and was left as it was.

| Question | After the correction |
|---|---|
| Bathroom too frequent | 5.1 min/day, 4.1 episodes/day, no trend (p=0.43). Was about 27 min/day and 8.3 episodes |
| Bathroom not too frequent | 7.0 min/day, 5.2 episodes/day, no trend (p=0.54) |
| Visitors | Guest bedroom only, 4.9 min/day, 3.3 episodes/day, no trend (p=0.92). Was about 20.5 min/day |
| Bathroom use since a new medication | 2.4 min/day, 2.1 episodes/day, no clear change (last third 18% lower, p=0.10). Sleep and sedentary findings were kept |

**Ratings adjusted to match.** Queries that named a Milan garage door or an Aruba guest
bathroom were set to no and dropped from the key, so they do not enter the relevance rate.
Milan question 5 lost its step 3 windows and step 4 findings. Archith's step-1 yes on the
Milan side door was set to no; Alex had already said no, and the analyst already refused it.
Decoy captions were not edited.

**What the review did not re-rate.** Step 3 windows for the recomputed concepts are the
original sample. That is why Aruba "bathroom too frequent" still fails retrieval: the raters
saw windows from the old guest-bathroom matches. Step 4 sentences are the original findings,
except the removed garage-door question. Milan's sudden-change failure is a judgment of those
original sentences. The dashboard shows the recomputed reports.

**Cairo has no medicine cabinet.** M021 is a kitchen motion sensor. Our inventory says
pantry. The upstream CASAS translate file says `Kitchen MedCabinet`; the analyst does not
read that file. Question 18 was refused. No correction needed for the analyst inventory.

**Figure axis.** Raters saw daily share of active time, except on the seven recomputed
questions, whose figures were redrawn in matched minutes per day. The other figures are
still the share charts from 23 October 2026.

**Review UI differences across homes.** Milan step 4 was rated before the packet was
rebuilt, so that pass did not have the hour-of-day chart or the plain-language first-third /
last-third tests. Aruba and Cairo did. The statistics card was later moved under the charts;
that is layout only.

## 11. Files

| Path | Contents |
|---|---|
| `src/analyst/` | Implementation. Modules listed in `docs/ANALYST.md`. |
| `metadata/analyst_questions.json` | The 29 questions. |
| `metadata/casas_metadata.json` | Sensor rooms, types, descriptions, coordinates. |
| `metadata/analyst_llm_cache/` | Cached planner and reporter responses for this run. |
| `results/analyst/runs/analyst_questions/<home>/<question>/` | `plan.json`, `result.json`, `report.md`, `figure.png`, `daily_*.csv`, `matches_*.csv`. |
| `results/analyst/runs/analyst_questions/summary.{md,csv}` | One row per question × home. |
| `results/analyst/runs/analyst_questions/dashboard.html` | Summary, per-question reports, and the review tab. Reads `review_v2`. |
| `results/analyst/imwut_tables/analyst_review.{tex,pdf,png}` | Paper table of the review-tab step rates. |
| `results/analyst/imwut_tables/analyst_funnel.{tex,pdf,png}` | Main-text funnel table, `tab:analyst_funnel`. Same counts as §9. |
| `results/analyst/imwut_tables/analyst_questions.{tex,pdf,png}` | `tab:analyst_questions`. Nine questions, with an ellipsis row in each category. Check if the analyst answered. Last row is the answered count over all 29. Caption points at the full list. |
| `results/analyst/imwut_tables/analyst_questions_all.{tex,pdf,png}` | `tab:analyst_questions_all`. Appendix table of all 29 questions, same marks, with the answered count at the bottom. |
| `results/analyst/runs/analyst_questions/review/review_v1/` | The packet that was first rated. Kept as the source copy. |
| `results/analyst/runs/analyst_questions/review/review_v2/` | Scored packet: v1 plus decoys, with the inventory edits applied to the key and the app data. |
| `results/analyst/runs/analyst_questions/review/review_v2_key/` | Answer key, including `stratum=decoy`. Not shown to the reviewer. |
| `results/analyst/runs/analyst_questions/review/review_v2_results/` | `review_scores.json`, `review_report.html`, Sankey figures, `funnel_per_question.csv`. |
| `review/review_v2/review_review_v2_{milan,aruba,cairo}_alexkarpekov_oct241.csv` | Alex's ratings, decoys included. |
| `review/review_v2/review_review_v2_milan_Archith_Iyer.csv` | Archith's Milan ratings, decoys included. No step 4. |

Recompute the review tables and rebuild the dashboard with:

```bash
python src/analyst/run.py review-score --name review_v2 --ratings \
  results/analyst/runs/analyst_questions/review/review_v2/review_review_v2_milan_alexkarpekov_oct241.csv \
  results/analyst/runs/analyst_questions/review/review_v2/review_review_v2_aruba_alexkarpekov_oct241.csv \
  results/analyst/runs/analyst_questions/review/review_v2/review_review_v2_cairo_alexkarpekov_oct241.csv \
  results/analyst/runs/analyst_questions/review/review_v2/review_review_v2_milan_Archith_Iyer.csv
```
