"""Deterministic statistics over matched windows.

Unit of measurement: minutes of sensor activity per day. Windows overlap (50%), so
minutes are computed from the union of [start, start + 60 s] intervals, never by
counting windows. Each concept's minutes are normalized by the observed minutes
(any sensor activity) that pass the same time-of-day / day-of-week filters, which
corrects for days with partial data.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps

from .config import DOW_BUCKETS, TOD_BUCKETS, AnalysisSettings
from .index import SensorIndex
from .retrieval import ConceptMatches, episodes, time_filter_mask, union_intervals
from .schemas import AnalysisSpec, Period


def _minutes_by_day(starts: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    iv = union_intervals(starts)
    if not iv:
        return pd.Series(0.0, index=days)
    arr = np.array(iv)
    day = pd.to_datetime(arr[:, 0], unit="s").normalize()
    mins = pd.Series((arr[:, 1] - arr[:, 0]) / 60.0, index=day).groupby(level=0).sum()
    return mins.reindex(days, fill_value=0.0)


def _union_minutes(starts: pd.Series) -> float:
    return float(sum(e - s for s, e in union_intervals(starts)) / 60.0)


def _r(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def build_daily(
    index: SensorIndex, scope: np.ndarray, cm: ConceptMatches, settings: AnalysisSettings
) -> pd.DataFrame:
    meta = index.meta
    in_scope = meta.loc[scope]
    days = pd.date_range(in_scope["start"].min().normalize(), in_scope["start"].max().normalize(), freq="D")
    observed = _minutes_by_day(in_scope["start"], days)
    filt = meta.loc[scope & time_filter_mask(index, cm.concept)]
    observed_f = _minutes_by_day(filt["start"], days)
    matched = _minutes_by_day(cm.matched["start"], days)
    ep = episodes(cm.matched, settings.episode_gap_seconds)
    ep_counts = ep["start"].dt.normalize().value_counts().reindex(days, fill_value=0) if len(ep) else pd.Series(0, index=days)

    df = pd.DataFrame({
        "observed_min": observed,
        "observed_filtered_min": observed_f,
        "matched_min": matched,
        "episodes": ep_counts.astype(int),
    })
    df.index.name = "date"
    df["valid"] = df["observed_min"] >= settings.min_observed_minutes_per_day
    with np.errstate(divide="ignore", invalid="ignore"):
        df["share_pct"] = np.where(df["observed_filtered_min"] > 0, 100 * df["matched_min"] / df["observed_filtered_min"], np.nan)
    return df


def _period_key(idx: pd.DatetimeIndex, granularity: str) -> pd.DatetimeIndex:
    if granularity == "week":
        return (idx - pd.to_timedelta(idx.dayofweek, unit="D")).normalize()
    if granularity == "month":
        return idx.to_period("M").to_timestamp()
    return idx.normalize()


def bucket_series(daily: pd.DataFrame, granularity: str) -> List[dict]:
    d = daily[daily["valid"]]
    if d.empty:
        return []
    g = d.groupby(_period_key(d.index, granularity))
    out = []
    min_days = {"day": 1, "week": 3, "month": 10}[granularity]
    for key, grp in g:
        out.append({
            "period_start": key.strftime("%Y-%m-%d"),
            "n_days": int(len(grp)),
            "partial": bool(len(grp) < min_days),
            "matched_min_per_day": _r(grp["matched_min"].mean()),
            "share_pct": _r(100 * grp["matched_min"].sum() / max(grp["observed_filtered_min"].sum(), 1e-9)),
            "episodes_per_day": _r(grp["episodes"].mean(), 2),
        })
    return out


def trend_stats(daily: pd.DataFrame, granularity: str) -> dict:
    d = daily[daily["valid"] & daily["share_pct"].notna()]
    if len(d) < 7:
        return {"type": "trend", "insufficient_data": True, "n_days": int(len(d))}
    x = (d.index - d.index[0]).days.to_numpy(dtype=float)
    out = {"type": "trend", "granularity": granularity, "n_days": int(len(d)),
           "first_day": d.index[0].strftime("%Y-%m-%d"), "last_day": d.index[-1].strftime("%Y-%m-%d")}
    for col, name in (("share_pct", "share"), ("matched_min", "minutes")):
        y = d[col].to_numpy(dtype=float)
        lr = sps.linregress(x, y)
        tau, p_tau = sps.kendalltau(x, y)
        fit0, fit1 = lr.intercept, lr.intercept + lr.slope * x[-1]
        out[name] = {
            "slope_per_week": _r(lr.slope * 7, 2),
            "slope_p_value": _r(lr.pvalue, 4),
            "kendall_tau": _r(tau, 3),
            "kendall_p_value": _r(p_tau, 4),
            "fitted_start": _r(fit0),
            "fitted_end": _r(fit1),
            "relative_change_pct": _r(100 * (fit1 - fit0) / fit0) if fit0 > 0 else None,
        }
    out["significant"] = bool(out["minutes"]["kendall_p_value"] is not None and out["minutes"]["kendall_p_value"] < 0.05)
    out["series"] = bucket_series(daily, granularity)
    return out


def _default_periods(daily: pd.DataFrame) -> List[Period]:
    d = daily[daily["valid"]].index
    if len(d) < 6:
        return []
    thirds = np.array_split(d, 3)
    return [
        Period(label="first third", start_date=thirds[0][0].strftime("%Y-%m-%d"), end_date=thirds[0][-1].strftime("%Y-%m-%d")),
        Period(label="last third", start_date=thirds[-1][0].strftime("%Y-%m-%d"), end_date=thirds[-1][-1].strftime("%Y-%m-%d")),
    ]


def compare_stats(daily: pd.DataFrame, periods: List[Period]) -> dict:
    periods = periods or _default_periods(daily)
    rows = []
    for p in periods:
        sel = daily.loc[p.start_date:p.end_date]
        sel = sel[sel["valid"] & sel["share_pct"].notna()]
        rows.append({
            "label": p.label, "start_date": p.start_date, "end_date": p.end_date, "n_days": int(len(sel)),
            "share_pct_mean": _r(sel["share_pct"].mean()), "share_pct_sd": _r(sel["share_pct"].std()),
            "matched_min_per_day": _r(sel["matched_min"].mean()),
            "_share": sel["share_pct"].to_numpy(), "_min": sel["matched_min"].to_numpy(),
        })
    tests = []
    if len(rows) >= 2:
        base = rows[0]
        for other in rows[1:]:
            test = {"baseline": base["label"], "comparison": other["label"]}
            if len(base["_min"]) >= 3 and len(other["_min"]) >= 3:
                test["minutes_mann_whitney_p_value"] = _r(sps.mannwhitneyu(base["_min"], other["_min"], alternative="two-sided")[1], 4)
                test["share_mann_whitney_p_value"] = _r(sps.mannwhitneyu(base["_share"], other["_share"], alternative="two-sided")[1], 4)
            if base["matched_min_per_day"]:
                test["minutes_change_pct"] = _r(100 * (other["matched_min_per_day"] - base["matched_min_per_day"]) / base["matched_min_per_day"])
            if base["share_pct_mean"]:
                test["share_change_pct"] = _r(100 * (other["share_pct_mean"] - base["share_pct_mean"]) / base["share_pct_mean"])
            tests.append(test)
    for r in rows:
        r.pop("_share"), r.pop("_min")
    return {"type": "compare_periods", "periods": rows, "tests": tests}


def clock(minutes) -> str | None:
    """Minutes after the activity day's midnight (may exceed 1440) → 'HH:MM'."""
    if minutes is None or not np.isfinite(minutes):
        return None
    m = int(round(minutes)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def episode_times(matched: pd.DataFrame, day_boundary_hour: int, gap_minutes: float = 2,
                  min_minutes: float = 0) -> pd.DataFrame:
    """Sessions (matched activity merged across gaps ≤ gap_minutes, shorter than min_minutes dropped) with
    their activity day (starting at day_boundary_hour) and start time in minutes after that day's midnight
    (e.g. 01:30 with boundary 4 → 1530)."""
    ep = episodes(matched, gap_minutes * 60)
    ep = ep[ep["minutes"] >= min_minutes]
    if ep.empty:
        return pd.DataFrame(columns=["day", "t_min", "minutes"])
    day = (ep["start"] - pd.Timedelta(hours=day_boundary_hour)).dt.normalize()
    return pd.DataFrame({"day": day, "t_min": (ep["start"] - day).dt.total_seconds() / 60.0,
                         "minutes": ep["minutes"]})


def _time_summary(t: pd.Series) -> dict:
    return {"median": clock(t.median()), "q25": clock(t.quantile(0.25)), "q75": clock(t.quantile(0.75)),
            "iqr_minutes": _r(t.quantile(0.75) - t.quantile(0.25)), "sd_minutes": _r(t.std())}


def start_window_hours(concept) -> List[int]:
    """Clock hours in which a timed occurrence may start: concept.timing_hours, else its time_of_day buckets."""
    if len(concept.timing_hours) == 2:
        s, e = concept.timing_hours
        return [h % 24 for h in range(s, e if e > s else e + 24)]
    hours = []
    for b in concept.time_of_day:
        s, e = (int(x.split(":")[0]) for x in TOD_BUCKETS[b].split("-"))
        hours += list(range(s, e))
    return sorted(set(hours)) or list(range(24))


def timed_sessions(matched: pd.DataFrame, spec: AnalysisSpec, hours: List[int]) -> pd.DataFrame:
    et = episode_times(matched, spec.day_boundary_hour, spec.session_gap_minutes, spec.min_session_minutes)
    return et[((et["t_min"] // 60) % 24).astype(int).isin(hours)] if len(et) else et


def timing_stats(daily: pd.DataFrame, cm: ConceptMatches, spec: AnalysisSpec, settings: AnalysisSettings,
                 cm_any_time: ConceptMatches | None = None) -> dict:
    """When the behavior happens: per-day first/last occurrence time (trend + period comparison), or for
    'all' the typical hours and how consistently they recur across days. Sessions are built from matches
    without the time-of-day filter (cm_any_time) and kept if they start inside the concept's window, so a
    session that begins before a bucket edge is not clipped to the edge."""
    valid_days = daily.index[daily["valid"]]
    hours = start_window_hours(cm.concept)
    et = timed_sessions((cm_any_time or cm).matched, spec, hours)
    et = et[et["day"].isin(valid_days)]
    out = {"type": "timing", "event": spec.timing_event, "day_boundary_hour": spec.day_boundary_hour,
           "session_gap_minutes": spec.session_gap_minutes, "min_session_minutes": spec.min_session_minutes,
           "start_window": f"{hours[0]:02d}:00-{(hours[-1] + 1) % 24:02d}:00" if hours != list(range(24)) else "any",
           "_start_hours": hours,
           "n_valid_days": int(len(valid_days)), "n_days_with_event": int(et["day"].nunique()),
           "days_with_event_pct": _r(100 * et["day"].nunique() / max(len(valid_days), 1))}
    if et.empty:
        out["insufficient_data"] = True
        return out

    if spec.timing_event == "all":
        hours = (et["t_min"] // 60 % 24).astype(int)
        days_per_hour = et.assign(hour=hours).groupby("hour")["day"].nunique()
        prof = [{"hour": f"{h:02d}:00", "pct_of_days": _r(100 * days_per_hour.get(h, 0) / len(valid_days))}
                for h in range(24)]
        per_day = et.groupby("day").size().reindex(valid_days, fill_value=0)
        out.update({
            "session_start_times": _time_summary(et["t_min"]),
            "session_minutes_median": _r(et["minutes"].median()),
            "sessions_per_day_mean": _r(per_day.mean(), 2), "sessions_per_day_sd": _r(per_day.std(), 2),
            "peak_hours": sorted(prof, key=lambda p: -(p["pct_of_days"] or 0))[:3],
            "hourly_pct_of_days": prof,
        })
        return out

    agg = "min" if spec.timing_event == "first" else "max"
    t = et.groupby("day")["t_min"].agg(agg).sort_index()
    out["daily_time"] = _time_summary(t)
    # Captions in the training data name time-of-day buckets, so retrieval with "in the evening" etc.
    # favours windows after a bucket edge; start times piling up just after an edge are an artifact.
    edges = sorted({int(v.split("-")[0].split(":")[0]) * 60 for v in TOD_BUCKETS.values()})
    tod_min = t % 1440
    at_edge = np.zeros(len(t), dtype=bool)
    for e in edges:
        at_edge |= ((tod_min - e) % 1440 < 10).to_numpy()
    out["pct_days_within_10min_after_time_bucket_edge"] = _r(100 * at_edge.mean())
    out["session_minutes_median"] = _r(et.loc[et.groupby("day")["t_min"].idxmin() if agg == "min"
                                               else et.groupby("day")["t_min"].idxmax(), "minutes"].median())
    if len(t) >= 7:
        x = (t.index - t.index[0]).days.to_numpy(dtype=float)
        lr = sps.linregress(x, t.to_numpy())
        tau, p_tau = sps.kendalltau(x, t.to_numpy())
        fit0, fit1 = lr.intercept, lr.intercept + lr.slope * x[-1]
        out["trend"] = {
            "first_day": t.index[0].strftime("%Y-%m-%d"), "last_day": t.index[-1].strftime("%Y-%m-%d"),
            "slope_minutes_per_week": _r(lr.slope * 7, 1), "kendall_tau": _r(tau, 3), "kendall_p_value": _r(p_tau, 4),
            "fitted_start": clock(fit0), "fitted_end": clock(fit1), "fitted_shift_minutes": _r(fit1 - fit0),
        }
    periods = spec.periods or _default_periods(daily)
    rows = []
    for p in periods:
        v = t.loc[p.start_date:p.end_date]
        rows.append({"label": p.label, "start_date": p.start_date, "end_date": p.end_date,
                     "n_days": int(len(v)), **_time_summary(v), "_v": v.to_numpy()})
    if len(rows) >= 2 and len(rows[0]["_v"]) >= 3 and len(rows[-1]["_v"]) >= 3:
        _, p = sps.mannwhitneyu(rows[0]["_v"], rows[-1]["_v"], alternative="two-sided")
        out["compare"] = {"baseline": rows[0]["label"], "comparison": rows[-1]["label"],
                          "median_shift_minutes": _r(np.median(rows[-1]["_v"]) - np.median(rows[0]["_v"])),
                          "mann_whitney_p_value": _r(p, 4)}
    for r in rows:
        r.pop("_v")
    out["periods"] = rows
    weeks = _period_key(t.index, "week")
    out["weekly_median"] = [{"week_start": k.strftime("%Y-%m-%d"), "n_days": int(len(g)), "median": clock(g.median())}
                            for k, g in t.groupby(weeks)]
    out["_daily_minutes"] = t
    return out


def _profile(index: SensorIndex, scope: np.ndarray, cm: ConceptMatches, column: str, buckets: List[str]) -> List[dict]:
    meta = index.meta
    out = []
    for b in buckets:
        obs = _union_minutes(meta.loc[scope & (meta[column] == b).to_numpy(), "start"])
        mat = _union_minutes(cm.matched.loc[cm.matched[column] == b, "start"])
        out.append({"bucket": b, "matched_min": _r(mat), "observed_min": _r(obs),
                    "share_pct": _r(100 * mat / obs) if obs > 0 else None})
    return out


def overall_stats(daily: pd.DataFrame, cm: ConceptMatches, settings: AnalysisSettings) -> dict:
    d = daily[daily["valid"]]
    ep = episodes(cm.matched, settings.episode_gap_seconds)
    return {
        "type": "overall_summary",
        "n_valid_days": int(len(d)),
        "n_excluded_days": int((~daily["valid"]).sum()),
        "matched_hours_total": _r(d["matched_min"].sum() / 60),
        "matched_min_per_day_mean": _r(d["matched_min"].mean()),
        "share_pct_mean": _r(100 * d["matched_min"].sum() / max(d["observed_filtered_min"].sum(), 1e-9)),
        "n_episodes": int(len(ep)),
        "episode_minutes_median": _r(ep["minutes"].median()) if len(ep) else None,
        "episodes_per_day_mean": _r(d["episodes"].mean(), 2),
    }


def run_analyses(
    index: SensorIndex, scope: np.ndarray, cm: ConceptMatches, specs: List[AnalysisSpec], settings: AnalysisSettings,
    cm_any_time: ConceptMatches | None = None,
) -> Tuple[Dict, pd.DataFrame]:
    daily = build_daily(index, scope, cm, settings)
    results = {"overall": overall_stats(daily, cm, settings), "analyses": []}
    for spec in specs:
        if spec.type == "trend":
            results["analyses"].append(trend_stats(daily, spec.granularity))
        elif spec.type == "compare_periods":
            results["analyses"].append(compare_stats(daily, spec.periods))
        elif spec.type == "timing":
            ts = timing_stats(daily, cm, spec, settings, cm_any_time)
            if "_daily_minutes" in ts:
                col = f"{spec.timing_event}_time_min"
                daily[col] = ts.pop("_daily_minutes").reindex(daily.index)
            results["analyses"].append(ts)
        elif spec.type == "time_of_day_profile":
            results["analyses"].append({"type": "time_of_day_profile",
                                        "buckets": _profile(index, scope, cm, "tod", list(TOD_BUCKETS))})
        elif spec.type == "day_of_week_profile":
            results["analyses"].append({"type": "day_of_week_profile",
                                        "buckets": _profile(index, scope, cm, "dow", DOW_BUCKETS)})
    # Always computed for plots, even if not requested.
    results["_tod_profile"] = _profile(index, scope, cm, "tod", list(TOD_BUCKETS))
    return results, daily


def retrieval_summary(cm: ConceptMatches) -> dict:
    m = cm.matched
    return {
        "threshold": cm.threshold,
        "n_eligible_windows": int(len(cm.eligible)),
        "n_matched_windows": int(len(m)),
        "n_matched_without_structural_filters": cm.n_matched_without_structural_filters,
        "per_query_matched_windows": cm.per_query_counts,
        "matched_score_median": _r(m["score"].median(), 3) if len(m) else None,
        "top_primary_rooms": m["primary_room"].value_counts().head(4).to_dict() if len(m) else {},
    }
