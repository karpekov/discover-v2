"""Helpers that condense a run's result.json for summary tables and the dashboard."""

from __future__ import annotations


def primary_stat(concept: dict) -> str:
    for a in concept["stats"]["analyses"]:
        if a["type"] == "timing" and not a.get("insufficient_data"):
            if a["event"] == "all":
                return "peaks: " + ", ".join(p["hour"] for p in a["peak_hours"])
            s = f"{a['event']} at {a['daily_time']['median']} (IQR {a['daily_time']['iqr_minutes']} min)"
            if "trend" in a:
                s += f", {a['trend']['slope_minutes_per_week']:+.1f} min/wk (p={a['trend']['kendall_p_value']})"
            return s
        if a["type"] == "trend" and not a.get("insufficient_data"):
            s = a["minutes"]
            return f"{s['slope_per_week']:+.1f} min/day per wk (τ={s['kendall_tau']}, p={s['kendall_p_value']})"
        if a["type"] == "compare_periods" and a["tests"]:
            p, change = _compare_test(a["tests"][0])
            return f"{change:+}% min/day vs {a['tests'][0]['baseline']} (p={p})" if change is not None else "—"
        if a["type"] in ("time_of_day_profile", "day_of_week_profile"):
            peak = max(a["buckets"], key=lambda b: b["share_pct"] or 0)
            return f"peak: {peak['bucket']} ({peak['share_pct']}%)"
    ov = concept["stats"]["overall"]
    return f"{ov['matched_min_per_day_mean']} min/day"


def _compare_test(t: dict) -> tuple:
    """(p, % change) of a period comparison in minutes/day; results saved before that use the share."""
    if "minutes_mann_whitney_p_value" in t or "minutes_change_pct" in t:
        return t.get("minutes_mann_whitney_p_value"), t.get("minutes_change_pct")
    return t.get("mann_whitney_p_value"), t.get("share_change_pct")


def direction(concept: dict, analysis_type: str) -> str:
    """↑ / ↓ if the trend or period comparison is significant (p < 0.05), else →; '' if not computed.
    Amount analyses use matched minutes per day; timing uses clock time (↑ later / ↓ earlier)."""
    for a in concept["stats"]["analyses"]:
        if a["type"] != analysis_type or a.get("insufficient_data"):
            continue
        if analysis_type == "timing":
            tests = [(a["trend"]["kendall_p_value"], a["trend"]["kendall_tau"])] if "trend" in a else []
            if "compare" in a:
                tests.append((a["compare"]["mann_whitney_p_value"], a["compare"]["median_shift_minutes"]))
            sig = [(p, e) for p, e in tests if p is not None and e is not None and p < 0.05]
            if not tests:
                continue
            p, effect = sig[0] if sig else tests[0]
        elif analysis_type == "trend":
            p, effect = a["minutes"]["kendall_p_value"], a["minutes"]["kendall_tau"]
        elif a.get("tests"):
            p, effect = _compare_test(a["tests"][0])
        else:
            continue
        if p is None or effect is None or p >= 0.05:
            return "→"
        return "↑" if effect > 0 else "↓"
    return ""


def trend_arrows(result: dict) -> list:
    """[(concept name, arrow)] for analyzed concepts, from the plan's first trend / comparison / timing."""
    kinds = [a["type"] for a in result["plan"]["analyses"]
             if a["type"] in ("trend", "compare_periods") or (a["type"] == "timing" and a.get("timing_event") != "all")]
    if not kinds:
        return []
    out = []
    for c in result["concepts"]:
        if "skipped" in c:
            continue
        arrow = next((x for k in kinds if (x := direction(c, k))), "")
        out.append((c["concept"]["name"], arrow))
    return [(name, arrow) for name, arrow in out if arrow]


def status(result: dict) -> str:
    n_skipped = sum("skipped" in c for c in result["concepts"])
    if n_skipped == 0:
        return "detectable"
    return "undetectable" if n_skipped == len(result["concepts"]) else "partial"
