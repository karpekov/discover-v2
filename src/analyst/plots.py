"""Figures for an analyst run (one row per concept)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from .home_context import sensor_coordinates
from .index import SensorIndex
from .analysis import episode_times

_TOD_SHORT = {
    "early_morning": "early morning", "late_morning": "late morning", "afternoon": "afternoon",
    "evening": "evening", "night_before_midnight": "night <24h", "night_after_midnight": "night >0h",
}
_C_MAIN = "#3A6BC4"
_C_TREND = "#C4503A"


def _load_floor_plan(path: Path):
    if path.exists():
        img = plt.imread(str(path))
        return img, img.shape[1], img.shape[0]
    return None, 2600, 2800


def sensor_on_counts(index: SensorIndex, windows: pd.DataFrame, max_windows: int = 400, seed: int = 0) -> pd.Series:
    if len(windows) > max_windows:
        windows = windows.sample(max_windows, random_state=seed)
    counts: dict = {}
    for s, e in zip(windows["start"], windows["end"]):
        ev = index.events_between(s, e)
        on = ev[ev["state"].str.upper().isin(["ON", "OPEN"])]
        for sid, c in on["sensor_id"].value_counts().items():
            counts[sid] = counts.get(sid, 0) + c
    return pd.Series(counts, dtype=float)


def _draw_floor_heatmap(ax, index: SensorIndex, windows: pd.DataFrame):
    img, w, h = _load_floor_plan(index.cfg.floor_plan)
    coords = sensor_coordinates(index.cfg.home)
    if img is not None:
        ax.imshow(img, extent=[0, w, h, 0], alpha=0.5, zorder=0)
    counts = sensor_on_counts(index, windows)
    if len(counts):
        pts = [(coords[s][0], h - coords[s][1], c) for s, c in counts.items() if s in coords]
        if pts:
            xs, ys, cs = map(np.array, zip(*pts))
            sizes = 40 + 900 * cs / cs.max()
            ax.scatter(xs, ys, s=sizes, c=cs, cmap="YlOrRd", alpha=0.75, edgecolors="darkred", linewidths=0.5, zorder=2)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Sensor activations in matched windows", fontsize=9)


def _draw_timeseries(ax, daily: pd.DataFrame, stats: dict, rolling_days: int):
    d = daily[daily["valid"]]
    ov = stats["overall"]
    ax.scatter(d.index, d["matched_min"], s=8, color=_C_MAIN, alpha=0.35,
               label=f"daily (mean {ov['matched_min_per_day_mean']} min/day in {ov['episodes_per_day_mean']} episodes/day)")
    roll = d["matched_min"].rolling(f"{rolling_days}D", min_periods=3).mean()
    ax.plot(roll.index, roll.values, color=_C_MAIN, lw=2, label=f"{rolling_days}-day mean")
    trend = next((a for a in stats["analyses"] if a["type"] == "trend" and not a.get("insufficient_data")), None)
    if trend:
        mn = trend["minutes"]
        x0, x1 = pd.Timestamp(trend["first_day"]), pd.Timestamp(trend["last_day"])
        ax.plot([x0, x1], [mn["fitted_start"], mn["fitted_end"]], "--", color=_C_TREND, lw=1.8,
                label=f"trend {mn['slope_per_week']:+.1f} min/day per week (τ={mn['kendall_tau']}, p={mn['kendall_p_value']})")
    for a in stats["analyses"]:
        if a["type"] == "compare_periods":
            for i, p in enumerate(a["periods"]):
                ax.axvspan(pd.Timestamp(p["start_date"]), pd.Timestamp(p["end_date"]) + pd.Timedelta(days=1),
                           color=["#88a", "#a88", "#8a8"][i % 3], alpha=0.12, lw=0)
    excluded = daily[~daily["valid"]].index
    for x in excluded:
        ax.axvspan(x, x + pd.Timedelta(days=1), color="#999", alpha=0.08, lw=0)
    ax.set_ylabel("matched minutes per day", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")


def _clock_axis(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{int(v) % 24:02d}:{int(round(v % 1 * 60)) % 60:02d}" + (" +1d" if v >= 24 else "")))
    ax.set_ylabel("clock time", fontsize=8)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")


def _draw_timing(ax, daily: pd.DataFrame, matched: pd.DataFrame, timing: dict, rolling_days: int,
                 color: str = _C_MAIN):
    et = episode_times(matched, timing["day_boundary_hour"], timing.get("session_gap_minutes", 2),
                       timing.get("min_session_minutes", 0))
    if len(et) and "_start_hours" in timing:
        et = et[((et["t_min"] // 60) % 24).astype(int).isin(timing["_start_hours"])]
    if len(et):
        ax.scatter(et["day"], et["t_min"] / 60, s=6, color="#999", alpha=0.35, label="session starts")
    col = f"{timing['event']}_time_min"
    if col in daily:
        d = daily[col].dropna() / 60
        ax.scatter(d.index, d.values, s=12, color=color, alpha=0.7, label=f"{timing['event']} per day")
        roll = d.rolling(f"{rolling_days}D", min_periods=3).median()
        ax.plot(roll.index, roll.values, color=color, lw=2, label=f"{rolling_days}-day median")
        tr = timing.get("trend")
        if tr:
            h0, m0 = map(int, tr["fitted_start"].split(":"))
            fit0 = h0 + m0 / 60 + (24 if h0 < timing["day_boundary_hour"] else 0)
            fit1 = fit0 + tr["fitted_shift_minutes"] / 60
            ax.plot([pd.Timestamp(tr["first_day"]), pd.Timestamp(tr["last_day"])], [fit0, fit1], "--", color=_C_TREND,
                    lw=1.8, label=f"trend {tr['slope_minutes_per_week']:+.1f} min/wk "
                                  f"(τ={tr['kendall_tau']}, p={tr['kendall_p_value']})")
    for i, p in enumerate(timing.get("periods", [])):
        ax.axvspan(pd.Timestamp(p["start_date"]), pd.Timestamp(p["end_date"]) + pd.Timedelta(days=1),
                   color=["#88a", "#a88", "#8a8"][i % 3], alpha=0.12, lw=0)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    _clock_axis(ax)


_PALETTE = ["#3A6BC4", "#D98E04", "#2E9E5B", "#C4503A", "#8A5CC4", "#4BA3B5"]


def _draw_timing_overview(ax, rows: List[dict], dailies: dict, rolling_days: int):
    """All timed sub-behaviors (e.g. breakfast / lunch / dinner) on one clock-time axis, one color each."""
    for i, (r, timing) in enumerate(rows):
        col = f"{timing['event']}_time_min"
        d = dailies[r["concept"]["id"]].get(col)
        if d is None:
            continue
        d = d.dropna() / 60
        c = _PALETTE[i % len(_PALETTE)]
        dt = timing["daily_time"]
        ax.scatter(d.index, d.values, s=10, color=c, alpha=0.55)
        ax.plot(d.index, d.rolling(f"{rolling_days}D", min_periods=3).median().values, color=c, lw=2,
                label=f"{r['concept']['name']}: median {dt['median']}, IQR {dt['iqr_minutes']} min, "
                      f"{timing['days_with_event_pct']}% of days")
    ax.legend(fontsize=7, loc="upper left", frameon=False, ncol=1)
    ax.set_title("All timed behaviors", fontsize=10, loc="left", fontweight="bold")
    _clock_axis(ax)


def _draw_tod_table(ax, profile: List[dict]):
    ax.axis("off")
    total = sum(b["matched_min"] or 0 for b in profile) or 1
    fmt = lambda v: "–" if v is None else f"{v:.1f}"
    cells = [[fmt(100 * (b["matched_min"] or 0) / total) for b in profile], [fmt(b["share_pct"]) for b in profile]]
    tab = ax.table(cellText=cells, colLabels=[_TOD_SHORT[b["bucket"]] for b in profile],
                   rowLabels=["% of matches", "% of active time"], cellLoc="center", loc="center",
                   bbox=[0.09, 0, 0.91, 1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    peaks = [int(np.argmax([float(v) if v != "–" else -1 for v in row])) for row in cells]
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#bbb")
        if r == 0:
            cell.set_facecolor("#f1f3f7")
        elif c >= 0 and c == peaks[r - 1]:
            cell.set_text_props(fontweight="bold")


def plot_run(index: SensorIndex, question: str, concept_results: List[dict], dailies: dict,
             matched: dict, out_path: Path, rolling_days: int = 7, timing_matched: dict | None = None) -> None:
    timing_matched = timing_matched or matched
    rows = [r for r in concept_results if r["concept"]["detectable"] and r["concept"]["retrieval_queries"]]
    if not rows:
        return
    def _timing(r):
        return next((a for a in r["stats"]["analyses"]
                     if a["type"] == "timing" and not a.get("insufficient_data")), None)

    timed = [(r, t) for r in rows if (t := _timing(r)) and t["event"] != "all"]
    overview = len(timed) >= 2
    n = len(rows) + overview
    height = 3.6 * n + 0.6
    fig = plt.figure(figsize=(13, height))
    gs = fig.add_gridspec(n, 2, width_ratios=[2.6, 1.1], hspace=0.32, wspace=0.12, top=1 - 0.6 / height)
    if overview:
        _draw_timing_overview(fig.add_subplot(gs[0, :]), timed, dailies, rolling_days)
    for i, r in enumerate(rows, start=int(overview)):
        cid = r["concept"]["id"]
        sub = gs[i, 0].subgridspec(2, 1, height_ratios=[1, 0.17], hspace=0.62)
        ax_ts = fig.add_subplot(sub[0])
        timing = _timing(r)
        if timing:
            color = _PALETTE[[t[0] for t in timed].index(r) % len(_PALETTE)] if overview and timing["event"] != "all" else _C_MAIN
            _draw_timing(ax_ts, dailies[cid], timing_matched[cid], timing, rolling_days, color)
        else:
            _draw_timeseries(ax_ts, dailies[cid], r["stats"], rolling_days)
        cc = r["concept"]
        filt = ", ".join(cc["time_of_day"] + cc["days_of_week"] + cc["rooms"] + cc["required_sensors"])
        ax_ts.set_title(f"{r['concept']['name']}" + (f"  [{filt}]" if filt else ""), fontsize=10, loc="left", fontweight="bold")
        _draw_tod_table(fig.add_subplot(sub[1]), r["stats"]["_tod_profile"])
        _draw_floor_heatmap(fig.add_subplot(gs[i, 1]), index, matched[cid])
    fig.suptitle(f"{index.cfg.home.capitalize()} — " + "\n".join(textwrap.wrap(question, 110)), fontsize=12, y=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


_BUCKET_EDGES = [(5, "early morning"), (8, "late morning"), (12, "afternoon"), (17, "evening"),
                 (21, "night before\nmidnight"), (0, "night after\nmidnight")]


def plot_hourly(rows: List[dict], out_path: Path):
    """Matched minutes per day by hour of day, one panel per concept.

    rows: [{name, matches (DataFrame with start/end), n_days}]. Axis starts at noon so the night is contiguous.
    """
    order = [(12 + k) % 24 for k in range(24)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 2.3 * len(rows) + 0.3), squeeze=False)
    for ax, r in zip(axes[:, 0], rows):
        m = r["matches"]
        minutes = ((m["end"] - m["start"]).dt.total_seconds() / 60).groupby(m["start"].dt.hour).sum()
        vals = [minutes.get(h, 0.0) / max(r["n_days"], 1) for h in order]
        ax.bar(range(24), vals, width=0.85, color=_C_MAIN)
        ax.set_xticks(range(24), [f"{h:02d}" for h in order], fontsize=8)
        ax.set_xlim(-0.6, 23.6)
        ax.set_ylim(0, max(max(vals), 0.1) * 1.3)
        ax.set_ylabel("min / day", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for h, label in _BUCKET_EDGES:
            x = order.index(h) - 0.5
            ax.axvline(x, color="#9ca3af", ls=":", lw=1)
            ax.text(x + 0.1, 1.0, label, transform=ax.get_xaxis_transform(), fontsize=7, color="#6b7280", va="top")
        ax.set_title(f"{r['name']} — matched minutes per day by hour of day", fontsize=10, loc="left",
                     fontweight="bold", pad=14)
    axes[-1, 0].set_xlabel("hour of day (starts at noon)", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
