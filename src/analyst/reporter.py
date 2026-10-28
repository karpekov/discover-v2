"""Reporter: turn computed statistics into a short grounded report."""

from __future__ import annotations

import json
import re
from typing import Iterable, List

from .llm import GeminiClient
from .schemas import Report

REPORTER_SYSTEM = """\
You write short analytical reports for researchers studying in-home behavior from ambient
sensors. You receive a user's question, the analysis plan, and statistics computed by
deterministic code. Write the answer.

Rules:
- Use ONLY numbers that appear in the provided statistics (you may round them). Never
  compute new numbers, never invent values.
- What the numbers measure: the data are ~1-minute windows of sensor events. matched_min_per_day
  = minutes per day of windows whose sensor activity matches the behavior's description (and
  filters); episodes = runs of matched windows merged across gaps of up to 2 minutes. It is the
  amount of sensor activity that looks like the behavior, NOT the duration of the behavior itself:
  "22 min/day" for medication access means 22 one-minute windows of kitchen activity near the
  medicine cabinet, not 22 minutes spent taking pills. Phrase amounts accordingly, naming where the
  activity was ("about 22 minutes per day of kitchen activity near the medicine cabinet, in about
  9 separate episodes"), never "spent N minutes doing X" or "accessed X for N minutes".
  share_pct is the same amount as a % of all sensor-active minutes in the time window; it is a
  secondary measure. Only when its trend verdict differs from the minutes verdict, add at most one
  sentence / finding about it, explicitly labeled "share of all sensor-active time", with the
  share's own Kendall p-value and the same wording rule.
- Answer the question directly in the headline. Amounts and trends use minutes per day (the
  trend's "minutes" block; for period comparisons minutes_change_pct and
  minutes_mann_whitney_p_value). For trends, state direction and size (fitted start → end in
  min/day, or slope per week). Use the minutes Kendall p-value to decide wording: p < 0.05 →
  "increased/decreased"; otherwise → "no clear trend" or "a slight, not statistically reliable,
  increase/decrease".
- "timing" statistics are clock times (HH:MM) of the behavior's first / last / all episodes per
  day. For timing questions, answer with times, not amounts: typical (median) time, spread
  (IQR in minutes), how many days the event was seen, and any shift (fitted shift or median shift
  in minutes, earlier/later). Use the Kendall (trend) or Mann-Whitney (comparison) p-value for
  "shifted" vs "no clear shift", with the same p < 0.05 rule. Timing is computed on sessions
  (matched activity merged across short gaps; see session_gap_minutes / min_session_minutes).
  When a behavior is split into occurrences (e.g. breakfast / lunch / dinner), report each one's
  typical time, consistency and days_with_event_pct (days without it = possibly skipped).
  If pct_days_within_10min_after_time_bucket_edge is above 30, the start times are pinned to a
  time-of-day bucket boundary of the embedding (05:00, 08:00, 12:00, 17:00, 21:00, 00:00): say the
  true start may be earlier, do not present that time or its tight spread as the resident's
  schedule, and list it as a limitation.
- Briefly mention how the behavior was operationalized with these sensors (rooms/fixtures,
  no sensor IDs).
- Plain prose, no markdown, no bullet points. Body: 3-6 sentences.
- Findings: list every factual claim made in the headline and body about analyzed concepts as
  separate atomic findings, each tied to one concept_id (copied exactly). One finding = one
  checkable claim: either one amount / frequency / time, or one change (direction or "no clear
  change"), never both. E.g. "About 27 minutes per day of sensor activity in bed" and, separately,
  "In-bed presence did not change over the observation period". A human will judge each finding
  against the figure and statistics, so keep them concrete, short (<= 20 words), and use the same
  numbers and p < 0.05 wording rule as the body. Usually 1-3 findings per analyzed concept,
  2-8 in total. No findings for concepts that were not analyzed; no findings about limitations.
- Limitations: 0-3 short items grounded in the provided data (e.g. excluded days, short
  span, proxy definitions, multiple residents). No generic disclaimers.
- Concepts marked "not analyzed" have no statistics because the home's sensors cannot
  observe them. Say plainly that the question (or that part of it) cannot be answered with
  this home's sensors and why, based on the detectability note. If the home has no sensor for
  the door, room or fixture asked about (e.g. no garage door), say it is not feasible in this
  home because there is no such sensor. If useful, name what kind of sensor would be needed.
  Do not speculate about the answer.
"""


def _strip_private(obj):
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


def build_reporter_payload(question: str, home: str, data_range: tuple, plan, concept_results: List[dict]) -> str:
    payload = {
        "question": question,
        "home": home,
        "data_range": [str(d)[:10] for d in data_range],
        "interpretation": plan.interpretation,
        "planner_caveats": plan.caveats,
        "concepts": [
            {
                "id": r["concept"]["id"],
                "name": r["concept"]["name"],
                "physical_definition": r["concept"]["physical_definition"],
                "detectable": r["concept"]["detectable"],
                "detectability_note": r["concept"]["detectability_note"],
                **({"status": "not analyzed: no sensor evidence available"} if "skipped" in r else {
                    "filters": {k: r["concept"][k] for k in ("time_of_day", "days_of_week", "rooms", "required_sensors")},
                    "retrieval": r["retrieval"],
                    "statistics": _strip_private(r["stats"]),
                }),
            }
            for r in concept_results
        ],
    }
    return json.dumps(payload, indent=1, default=str)


def write_report(llm: GeminiClient, home: str, payload: str) -> tuple[Report, dict]:
    return llm.generate_json(tag=f"report_{home}", system=REPORTER_SYSTEM, user=payload, schema=Report)


_NUM = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?")


def _numbers_in(obj) -> Iterable[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, str):
        for m in _NUM.findall(obj):
            yield float(m.replace(",", ""))
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _numbers_in(v)


def ungrounded_numbers(report: Report, payload: str) -> List[str]:
    """Numbers in the report that do not match (up to rounding) any number in the payload."""
    allowed = set()
    for v in _numbers_in(json.loads(payload)):
        for x in (v, abs(v)):
            allowed.update({round(x, 2), round(x, 1), float(round(x))})
    flagged = []
    text = " ".join([report.headline, report.body, *report.limitations, *(f.statement for f in report.findings)])
    for raw in _NUM.findall(text):
        x = abs(float(raw.replace(",", "")))
        if x <= 10 and x == int(x):
            continue
        tol = 0.501 if x == int(x) else 0.051
        if not any(abs(x - a) <= tol for a in allowed):
            flagged.append(raw)
    return flagged
