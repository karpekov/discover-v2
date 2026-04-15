#!/usr/bin/env python3
"""
Longitudinal analysis of smart-home sensor activity patterns.

Runs a natural-language query through the SmartQuery pipeline (LLM rewriting +
FAISS retrieval), then analyses **when** the matched events occurred over time.

Outputs (results/long/{dataset_name}/{dataset_split}/{safe_query}/):
  longitudinal.png  – daily / weekly counts with moving-average trend +
                      variability band
  heatmap.png       – sensor activation density on the house floor plan (KDE)
  summary.md        – structured markdown summary
  summary.txt       – plain-text summary
  results.json      – full retrieval results (for later interactive use)

Usage (programmatic):
    from evals.longitudinal import LongitudinalAnalyzer
    from query import SmartQuery

    sq = SmartQuery.from_data_dir(
        checkpoint_path="trained_models/milan/.../best_model.pt",
        data_dir="data/processed/casas/milan/FD_60",
        home="milan",
    )
    analyzer = LongitudinalAnalyzer(sq, dataset_name="milan", dataset_split="FD_60")
    analyzer.analyze("morning kitchen routines", threshold=0.10)

Usage (CLI):
    python src/evals/longitudinal/longitudinal_analysis.py \\
        --checkpoint trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt \\
        --data_dir   data/processed/casas/milan/FD_60 \\
        --home       milan \\
        --query      "morning kitchen routines" \\
        --threshold  0.10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Project root on sys.path so relative imports work when run as __main__
# ---------------------------------------------------------------------------
_FILE      = Path(__file__).resolve()
_SRC       = _FILE.parents[2]          # src/
_ROOT      = _FILE.parents[3]          # project root
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ============================================================================
# Helpers
# ============================================================================

def _safe_query_name(query: str, max_len: int = 60) -> str:
    """Convert a query string to a filesystem-safe directory name."""
    safe = re.sub(r"[^\w\s-]", "", query.lower())
    safe = re.sub(r"[\s_-]+", "_", safe).strip("_")
    return safe[:max_len]


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    """Parse a timestamp string into a datetime object."""
    if not ts:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def _load_casas_metadata(dataset_name: str) -> Dict:
    """Load sensor coordinates and floor-plan metadata for a dataset."""
    metadata_path = _ROOT / "metadata" / "casas_metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path) as f:
        all_meta = json.load(f)
    return all_meta.get(dataset_name, {})


def _load_floor_plan(dataset_name: str) -> Optional[np.ndarray]:
    """Load floor-plan image; returns None if not found."""
    path = _ROOT / "metadata" / "floor_plans_augmented" / f"{dataset_name}.png"
    if not path.exists():
        return None
    try:
        return plt.imread(str(path))
    except Exception:
        return None


# ============================================================================
# Core analyser
# ============================================================================

class LongitudinalAnalyzer:
    """
    Analyse when retrieved sensor sequences occur over calendar time.

    Parameters
    ----------
    smart_query:
        A ready ``SmartQuery`` instance (already pointing at a dataset).
    dataset_name:
        Short identifier used in output paths and plot titles (e.g. ``"milan"``).
    dataset_split:
        Split label used in the output path (e.g. ``"FD_60"``).
    output_base:
        Root directory for all outputs (default: ``results/long``).
    """

    def __init__(
        self,
        smart_query,
        dataset_name: str = "milan",
        dataset_split: str = "FD_60",
        output_base: str = "results/long",
    ):
        self.sq            = smart_query
        self.dataset_name  = dataset_name
        self.dataset_split = dataset_split
        self.output_base   = Path(output_base)

        # Load sensor metadata for heatmap
        meta = _load_casas_metadata(dataset_name)
        self.sensor_coords    = meta.get("sensor_coordinates", {})
        self.sensor_locations = meta.get("sensor_location", {})

        # Floor plan
        self.floor_plan_img = _load_floor_plan(dataset_name)
        if self.floor_plan_img is not None:
            self.img_h, self.img_w = self.floor_plan_img.shape[:2]
        else:
            self.img_h, self.img_w = 600, 800

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(
        self,
        query: str,
        mode: str = "multi_location",
        threshold: Optional[float] = 0.10,
        top_k: int = 200,
        time_window: str = "day",
        ma_window: int = 7,
        force_rewrite: bool = False,
        force_retrieve: bool = False,
        max_subqueries: Optional[int] = None,
    ) -> Path:
        """
        Run the full longitudinal pipeline for one query.

        Parameters
        ----------
        query:         Natural-language query.
        mode:          Rewrite mode (single / multi_location / multi_wording).
        threshold:     Cosine similarity floor (recommended 0.05–0.25).
                       When set, ``top_k`` is ignored.
        top_k:         Max results when threshold is None.
        time_window:   Aggregation bucket – ``"day"`` or ``"week"``.
        ma_window:     Moving-average window width (in ``time_window`` units).
        force_rewrite: Bypass LLM rewrite cache.
        force_retrieve:Bypass FAISS result cache.
        max_subqueries: Max LLM sentences for multi_* modes (defaults 8 / 6).

        Returns
        -------
        Path to the output directory.
        """
        print(f"\n{'='*70}")
        print(f"LONGITUDINAL ANALYSIS")
        print(f"  Query  : {query!r}")
        print(f"  Home   : {self.dataset_name} / {self.dataset_split}")
        _ms = "" if max_subqueries is None else f"  |  max_subqueries={max_subqueries}"
        print(f"  Mode   : {mode}  |  threshold={threshold}  |  window={time_window}{_ms}")
        print(f"{'='*70}\n")

        # ── 1. Retrieve ──────────────────────────────────────────────
        out = self.sq.query(
            query,
            mode=mode,
            top_k=top_k,
            threshold=threshold,
            force_rewrite=force_rewrite,
            force_retrieve=force_retrieve,
            max_subqueries=max_subqueries,
        )
        results = out.get("results", [])
        sentences = out.get("sentences", [])
        reasoning = out.get("reasoning", "")

        if not results:
            print("[LongitudinalAnalyzer] No results returned — nothing to plot.")
            return self._empty_output(query, out)

        print(f"\n[LongitudinalAnalyzer] {len(results)} results retrieved.")

        # ── 2. Parse timestamps ──────────────────────────────────────
        dated, undated = self._extract_dated_results(results)
        print(f"[LongitudinalAnalyzer] {len(dated)} with timestamps, "
              f"{undated} without.")

        # ── 3. Build output directory ────────────────────────────────
        out_dir = (
            self.output_base
            / self.dataset_name
            / self.dataset_split
            / _safe_query_name(query)
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── 4. Aggregate into time buckets ───────────────────────────
        time_series = self._aggregate(dated, time_window)

        # ── 5. Plots ─────────────────────────────────────────────────
        if time_series:
            long_path = self._plot_longitudinal(
                time_series, query, out_dir, time_window, ma_window, out
            )
        else:
            long_path = None
            print("[LongitudinalAnalyzer] No dated results — skipping longitudinal plot.")

        heat_path = self._plot_heatmap(results, query, out_dir, out)

        # ── 6. Score distribution plot ───────────────────────────────
        score_path = self._plot_score_distribution(results, query, out_dir)

        # ── 7. Summaries ─────────────────────────────────────────────
        self._write_summaries(
            query=query,
            out=out,
            results=results,
            dated=dated,
            time_series=time_series,
            time_window=time_window,
            ma_window=ma_window,
            out_dir=out_dir,
            long_path=long_path,
            heat_path=heat_path,
            score_path=score_path,
        )

        # ── 8. Save raw results ──────────────────────────────────────
        self._save_raw_results(out, out_dir)

        print(f"\n[LongitudinalAnalyzer] Done. Outputs in: {out_dir}")
        return out_dir

    # ------------------------------------------------------------------
    # Timestamp extraction
    # ------------------------------------------------------------------

    def _extract_dated_results(
        self, results: List[Dict]
    ) -> Tuple[List[Tuple[datetime, Dict]], int]:
        """
        Extract (datetime, result) pairs from retrieval results.

        Looks in ``result['labels']['start_time']`` and falls back to
        ``result['events'][0]['timestamp']`` if present.
        """
        dated: List[Tuple[datetime, Dict]] = []
        undated = 0

        for r in results:
            ts_str = None
            labels = r.get("labels", {})

            # Primary: sample metadata.start_time (always present for our datasets)
            ts_str = labels.get("start_time")

            # Fallback: first event timestamp
            if not ts_str:
                events = r.get("events", [])
                if events:
                    ts_str = events[0].get("timestamp") or events[0].get("time")

            dt = _parse_timestamp(ts_str)
            if dt is not None:
                dated.append((dt, r))
            else:
                undated += 1

        return dated, undated

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self, dated: List[Tuple[datetime, Dict]], window: str
    ) -> Dict[datetime, List[Dict]]:
        """Group results into daily or weekly buckets."""
        buckets: Dict[datetime, List[Dict]] = defaultdict(list)
        for dt, r in dated:
            if window == "week":
                # Monday of the ISO week
                key = dt - timedelta(days=dt.weekday())
                key = key.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                key = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            buckets[key].append(r)
        return dict(buckets)

    # ------------------------------------------------------------------
    # Total-dataset daily counts (for normalisation)
    # ------------------------------------------------------------------

    def _get_total_daily_counts(self, time_window: str) -> Dict[datetime, int]:
        """
        Count ALL samples in the loaded dataset per day/week bucket.
        Used as the denominator for the normalised activity chart.
        Returns an empty dict if the dataset is not accessible.
        """
        totals: Dict[datetime, int] = defaultdict(int)
        try:
            dataset = self.sq.retrieval.test_dataset
            for sample in dataset.data:
                ts_str = (
                    sample.get("metadata", {}).get("start_time")
                    or sample.get("start_time")
                )
                dt = _parse_timestamp(ts_str)
                if dt is None:
                    continue
                if time_window == "week":
                    key = dt - timedelta(days=dt.weekday())
                    key = key.replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    key = dt.replace(hour=0, minute=0, second=0, microsecond=0)
                totals[key] += 1
        except Exception:
            pass
        return dict(totals)

    # ------------------------------------------------------------------
    # Longitudinal plot
    # ------------------------------------------------------------------

    def _plot_longitudinal(
        self,
        time_series: Dict[datetime, List[Dict]],
        query: str,
        out_dir: Path,
        time_window: str,
        ma_window: int,
        out: Dict,
    ) -> Path:
        """
        Three-panel longitudinal chart (light theme):
          Panel 1 – raw matched sample counts per day/week
          Panel 2 – counts normalised by total dataset activity that day (%)
          Panel 3 – total time spent on matched activities per day (minutes)
        All panels share the x-axis and show a moving-average trend + ±1σ band.
        """
        # ── build continuous date axis ──────────────────────────────────
        dates = sorted(time_series.keys())
        if len(dates) > 1:
            dates = _fill_date_range(dates, time_window)

        bar_w = 0.8 if time_window == "day" else 5.0
        dates_arr = np.array(dates)

        # ── Panel 1: raw counts ──────────────────────────────────────────
        counts = np.array([len(time_series.get(d, [])) for d in dates], dtype=float)

        # ── Panel 2: normalised counts ───────────────────────────────────
        total_counts = self._get_total_daily_counts(time_window)
        has_totals   = bool(total_counts)
        norm = np.array([
            (counts[i] / total_counts[d] * 100.0) if (has_totals and total_counts.get(d, 0) > 0)
            else np.nan
            for i, d in enumerate(dates)
        ])

        # ── Panel 3: time spent (minutes) ───────────────────────────────
        duration = np.array([
            sum(
                (r.get("labels", {}).get("duration_seconds") or 0.0)
                for r in time_series.get(d, [])
            ) / 60.0
            for d in dates
        ])

        # ── moving averages ──────────────────────────────────────────────
        ma_counts   = _moving_average(counts,   ma_window)
        std_counts  = _rolling_std(counts,      ma_window)
        ma_norm     = _moving_average(norm,     ma_window)
        std_norm    = _rolling_std(norm,        ma_window)
        ma_dur      = _moving_average(duration, ma_window)
        std_dur     = _rolling_std(duration,    ma_window)

        # ── figure ───────────────────────────────────────────────────────
        fig, axes = plt.subplots(
            3, 1,
            figsize=(15, 12),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.08},
            facecolor="white",
        )
        for ax in axes:
            ax.set_facecolor("#f7f7f7")

        # colour palette — grey faded bars, each panel owns its MA colour
        C_BAR   = "#999999"   # grey bars (all three panels)
        C_BAR2  = "#999999"
        C_BAR3  = "#999999"
        C_MA1   = "#4C8BE8"   # blue  MA (raw counts)
        C_MA2   = "#F07C2A"   # orange MA (normalised)
        C_MA3   = "#4BAE8A"   # green  MA (duration)
        GRID_KW = dict(color="#dddddd", linestyle="--", linewidth=0.7, zorder=0)

        short_q = (query[:90] + "…") if len(query) > 90 else query
        mode    = out.get("rewrite_mode", "")
        n_sents = len(out.get("sentences", []))

        def _add_trend(ax, ma, std, color):
            valid = ~np.isnan(ma)
            if valid.any():
                ax.plot(dates_arr[valid], ma[valid], color=color, linewidth=2.8,
                        label=f"{ma_window}-{time_window} MA", zorder=4)
                ax.fill_between(
                    dates_arr[valid],
                    np.maximum((ma - std)[valid], 0),
                    (ma + std)[valid],
                    color=color, alpha=0.18, label="±1σ", zorder=3,
                )

        def _style_ax(ax, ylabel):
            ax.set_ylabel(ylabel, fontsize=10, color="#333")
            ax.tick_params(axis="both", colors="#444", labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#bbb")
            ax.spines["bottom"].set_color("#bbb")
            ax.yaxis.grid(True, **GRID_KW)
            ax.set_axisbelow(True)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85,
                      edgecolor="#ccc", facecolor="white")

        # ── ax0: raw counts ──────────────────────────────────────────────
        ax0 = axes[0]
        ax0.bar(dates_arr, counts, width=bar_w, color=C_BAR, alpha=0.18,
                label="Matched samples", zorder=2)
        _add_trend(ax0, ma_counts, std_counts, C_MA1)
        _style_ax(ax0, "Matched samples / period")
        ax0.set_title(
            f'Longitudinal analysis  –  "{short_q}"\n'
            f'{self.dataset_name.capitalize()} / {self.dataset_split}   '
            f'mode={mode}   {n_sents} sub-quer{"y" if n_sents==1 else "ies"}   '
            f'{len(dates)} {time_window}s',
            fontsize=11, color="#222", pad=10,
        )

        # ── ax1: normalised counts ───────────────────────────────────────
        ax1 = axes[1]
        if has_totals and not np.all(np.isnan(norm)):
            ax1.bar(dates_arr, np.where(np.isnan(norm), 0, norm),
                    width=bar_w, color=C_BAR2, alpha=0.18,
                    label="% of all samples that day", zorder=2)
            _add_trend(ax1, ma_norm, std_norm, C_MA2)
            _style_ax(ax1, "% of daily activity")
        else:
            ax1.text(0.5, 0.5, "Dataset totals not available\n(normalisation requires full dataset)",
                     ha="center", va="center", transform=ax1.transAxes,
                     fontsize=10, color="#888")
            _style_ax(ax1, "% of daily activity")

        # ── ax2: duration ────────────────────────────────────────────────
        ax2 = axes[2]
        ax2.bar(dates_arr, duration, width=bar_w, color=C_BAR3, alpha=0.18,
                label="Total time (min)", zorder=2)
        _add_trend(ax2, ma_dur, std_dur, C_MA3)
        _style_ax(ax2, "Total matched time (min)")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.get_xticklabels(), rotation=35, ha="right", fontsize=8)

        fig.align_ylabels(axes)
        plt.tight_layout()

        out_path = out_dir / "longitudinal.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[plot] Saved longitudinal chart → {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Score distribution plot
    # ------------------------------------------------------------------

    def _plot_score_distribution(
        self, results: List[Dict], query: str, out_dir: Path
    ) -> Path:
        """Histogram of retrieval scores."""
        scores = [r["score"] for r in results if "score" in r]
        if not scores:
            return out_dir / "scores.png"

        fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
        ax.set_facecolor("#f7f7f7")

        n_bins = min(40, max(10, len(scores) // 5))
        ax.hist(scores, bins=n_bins, color="#4C8BE8", alpha=0.80, edgecolor="#2c6fbd")
        ax.axvline(np.mean(scores), color="#C0392B", linewidth=2,
                   label=f"mean={np.mean(scores):.3f}")
        ax.axvline(np.median(scores), color="#F07C2A", linewidth=2, linestyle="--",
                   label=f"median={np.median(scores):.3f}")

        ax.set_xlabel("Cosine similarity score", color="#333")
        ax.set_ylabel("Count", color="#333")
        ax.tick_params(colors="#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#bbb")
        ax.spines["bottom"].set_color("#bbb")
        ax.legend(framealpha=0.9, edgecolor="#ccc")
        ax.set_title(f"Score distribution  ({len(scores)} results)", fontsize=11)
        ax.yaxis.grid(True, color="#ddd", linestyle="--", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        plt.tight_layout()
        out_path = out_dir / "scores.png"
        plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[plot] Saved score distribution → {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Sensor heatmap
    # ------------------------------------------------------------------

    def _collect_sensor_activations(
        self, results: List[Dict]
    ) -> Tuple[List[Tuple[float, float]], Counter]:
        """
        Collect (x, y_flipped) pixel coordinates for all ON/OPEN sensor events
        in the result set, plus raw activation counts per sensor ID.
        """
        coords: List[Tuple[float, float]] = []
        sensor_counts: Counter = Counter()

        for r in results:
            events = r.get("events", [])
            for evt in events:
                sensor_id = (
                    evt.get("sensor_id")
                    or evt.get("sensor")
                    or evt.get("id", "")
                )
                state = str(
                    evt.get("event_type", evt.get("state", ""))
                ).upper()

                if state not in ("ON", "OPEN", "1", "TRUE"):
                    continue

                if sensor_id and sensor_id in self.sensor_coords:
                    x, y = self.sensor_coords[sensor_id]
                    y_flip = self.img_h - y
                    coords.append((x, y_flip))
                    sensor_counts[sensor_id] += 1

        return coords, sensor_counts

    def _plot_heatmap(
        self, results: List[Dict], query: str, out_dir: Path, out: Dict
    ) -> Path:
        """
        KDE density heatmap on the house floor plan showing which sensors
        were most activated in the matched samples.
        """
        coords, sensor_counts = self._collect_sensor_activations(results)

        fig, ax = plt.subplots(figsize=(12, 9), facecolor="white")

        # Floor plan background
        if self.floor_plan_img is not None:
            ax.imshow(
                self.floor_plan_img,
                extent=[0, self.img_w, self.img_h, 0],
                alpha=0.65, aspect="auto", zorder=0,
            )
        else:
            ax.set_xlim(0, self.img_w)
            ax.set_ylim(self.img_h, 0)

        if len(coords) >= 3:
            xs = np.array([c[0] for c in coords])
            ys = np.array([c[1] for c in coords])

            x_grid = np.linspace(0, self.img_w, 150)
            y_grid = np.linspace(0, self.img_h, 150)
            X, Y   = np.meshgrid(x_grid, y_grid)
            pos    = np.vstack([X.ravel(), Y.ravel()])

            try:
                kernel = stats.gaussian_kde(np.vstack([xs, ys]), bw_method=0.15)
                Z = np.reshape(kernel(pos).T, X.shape)
                levels = np.linspace(Z.max() * 0.05, Z.max(), 14)
                cf = ax.contourf(X, Y, Z, levels=levels,
                                 cmap="YlOrRd", alpha=0.72, zorder=1)
                plt.colorbar(cf, ax=ax, label="Activation density", shrink=0.75)
                ax.scatter(xs, ys, c="darkred", s=14, alpha=0.7,
                           edgecolors="black", linewidths=0.3, zorder=2)
            except Exception as e:
                print(f"[heatmap] KDE failed ({e}), falling back to scatter.")
                ax.scatter([c[0] for c in coords], [c[1] for c in coords],
                           c="red", s=18, alpha=0.7, zorder=2)
        elif coords:
            # Too few points for KDE
            ax.scatter([c[0] for c in coords], [c[1] for c in coords],
                       c="red", s=30, alpha=0.8, zorder=2)

        # Annotate all sensors with counts
        for sid, (sx, sy) in self.sensor_coords.items():
            sy_f = self.img_h - sy
            count = sensor_counts.get(sid, 0)
            color = "darkred" if count > 0 else "navy"
            weight = "bold" if count > 0 else "normal"
            label = f"{sid}\n({count})" if count > 0 else sid
            ax.annotate(label, (sx, sy_f), fontsize=5.5, alpha=0.85,
                        ha="center", va="bottom", color=color,
                        fontweight=weight, zorder=3)

        ax.set_xlim(0, self.img_w)
        ax.set_ylim(self.img_h, 0)
        ax.set_xticks([])
        ax.set_yticks([])

        short_q = (query[:70] + "…") if len(query) > 70 else query
        total_events = sum(sensor_counts.values())
        top5 = sensor_counts.most_common(5)
        top5_str = ", ".join(f"{s}({c})" for s, c in top5) if top5 else "—"
        ax.set_title(
            f'Sensor activation heatmap: "{short_q}"\n'
            f'{self.dataset_name.capitalize()} / {self.dataset_split}  |  '
            f'{len(results)} samples  |  {total_events} ON events\n'
            f'Top sensors: {top5_str}',
            fontsize=10, fontweight="bold", pad=8,
        )

        plt.tight_layout()
        out_path = out_dir / "heatmap.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[plot] Saved sensor heatmap → {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Summary writers
    # ------------------------------------------------------------------

    def _write_summaries(
        self,
        query: str,
        out: Dict,
        results: List[Dict],
        dated: List[Tuple[datetime, Dict]],
        time_series: Dict[datetime, List[Dict]],
        time_window: str,
        ma_window: int,
        out_dir: Path,
        long_path: Optional[Path],
        heat_path: Optional[Path],
        score_path: Optional[Path],
    ) -> None:
        scores  = [r["score"] for r in results if "score" in r]
        dates   = sorted(time_series.keys()) if time_series else []
        counts  = [len(time_series[d]) for d in dates]
        ma      = _moving_average(np.array(counts), ma_window) if counts else []

        # Activity label distribution
        label_counter: Counter = Counter()
        for _, r in dated:
            l1 = r.get("labels", {}).get("activity_l1") or "Unknown"
            label_counter[l1] += 1

        # Room distribution
        room_counter: Counter = Counter()
        for _, r in dated:
            room = r.get("labels", {}).get("primary_room") or "Unknown"
            room_counter[room] += 1

        # Time-of-day distribution
        tod_counter: Counter = Counter()
        for dt, _ in dated:
            hour = dt.hour
            if   5 <= hour < 12: tod_counter["morning"]   += 1
            elif 12 <= hour < 17: tod_counter["afternoon"] += 1
            elif 17 <= hour < 21: tod_counter["evening"]   += 1
            else:                 tod_counter["night"]     += 1

        # Day-of-week distribution
        dow_counter: Counter = Counter()
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for dt, _ in dated:
            dow_counter[day_names[dt.weekday()]] += 1

        span_days = (max(dates) - min(dates)).days + 1 if len(dates) > 1 else 1
        date_range = (
            f"{min(dates).strftime('%Y-%m-%d')} → {max(dates).strftime('%Y-%m-%d')}"
            if dates else "—"
        )
        avg_per_window = np.mean(counts) if counts else 0.0
        peak_date = dates[int(np.argmax(counts))].strftime("%Y-%m-%d") if counts else "—"
        peak_count = max(counts) if counts else 0

        # ---- Markdown --------------------------------------------------
        md_lines = [
            f"# Longitudinal Analysis Report",
            f"",
            f"**Query:** {query}",
            f"**Dataset:** {self.dataset_name} / {self.dataset_split}",
            f"**Rewrite mode:** {out.get('rewrite_mode', '—')}",
            f"**Model:** {out.get('model_used', '—') or 'cache hit'}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
            f"---",
            f"",
            f"## Rewritten Sub-queries",
            f"",
        ]
        for i, s in enumerate(out.get("sentences", [query]), 1):
            md_lines.append(f"{i}. {s}")
        if out.get("reasoning"):
            md_lines += ["", "### Reasoning", "", f"> {out['reasoning'].strip()[:600]}"]

        md_lines += [
            "",
            "---",
            "",
            "## Retrieval Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total results | {len(results)} |",
            f"| Results with timestamps | {len(dated)} |",
            f"| Date range | {date_range} |",
            f"| Span (days) | {span_days} |",
            f"| Active {time_window}s | {len(dates)} |",
            f"| Avg matches / {time_window} | {avg_per_window:.1f} |",
            f"| Peak {time_window} | {peak_date} ({peak_count} matches) |",
        ]
        if scores:
            md_lines += [
                f"| Score mean | {np.mean(scores):.4f} |",
                f"| Score median | {np.median(scores):.4f} |",
                f"| Score min / max | {min(scores):.4f} / {max(scores):.4f} |",
            ]

        if label_counter:
            md_lines += ["", "## Ground-truth Activity Labels", "", "| Label | Count |", "|-------|-------|"]
            for lbl, cnt in label_counter.most_common():
                md_lines.append(f"| {lbl} | {cnt} |")

        if room_counter:
            md_lines += ["", "## Primary Room Distribution", "", "| Room | Count |", "|------|-------|"]
            for room, cnt in room_counter.most_common():
                md_lines.append(f"| {room} | {cnt} |")

        if tod_counter:
            md_lines += ["", "## Time-of-Day Distribution", "", "| Period | Count |", "|--------|-------|"]
            for tod in ["morning", "afternoon", "evening", "night"]:
                md_lines.append(f"| {tod.capitalize()} | {tod_counter.get(tod, 0)} |")

        if dow_counter:
            md_lines += ["", "## Day-of-Week Distribution", "", "| Day | Count |", "|-----|-------|"]
            for day in day_names:
                md_lines.append(f"| {day} | {dow_counter.get(day, 0)} |")

        md_lines += ["", "---", "", "## Output Files", ""]
        for fname, label in [
            ("longitudinal.png", "Time-series plot"),
            ("heatmap.png",      "Sensor activation heatmap"),
            ("scores.png",       "Score distribution"),
            ("results.json",     "Full raw results (JSON)"),
        ]:
            md_lines.append(f"- `{fname}` — {label}")

        md_path = out_dir / "summary.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"[summary] Saved → {md_path}")

        # ---- Plain text ------------------------------------------------
        txt_lines = [
            "=" * 70,
            "LONGITUDINAL ANALYSIS SUMMARY",
            "=" * 70,
            f"Query      : {query}",
            f"Dataset    : {self.dataset_name} / {self.dataset_split}",
            f"Mode       : {out.get('rewrite_mode', '—')}",
            f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "--- RESULTS ---",
            f"Total results        : {len(results)}",
            f"With timestamps      : {len(dated)}",
            f"Date range           : {date_range}",
            f"Span (days)          : {span_days}",
            f"Active {time_window}s{'':<11}: {len(dates)}",
            f"Avg / {time_window}{'':<12}: {avg_per_window:.1f}",
            f"Peak {time_window}{'':<13}: {peak_date} ({peak_count} matches)",
        ]
        if scores:
            txt_lines += [
                f"Score mean           : {np.mean(scores):.4f}",
                f"Score median         : {np.median(scores):.4f}",
                f"Score range          : {min(scores):.4f} – {max(scores):.4f}",
            ]
        if label_counter:
            txt_lines += ["", "--- ACTIVITY LABELS ---"]
            for lbl, cnt in label_counter.most_common():
                txt_lines.append(f"  {lbl:<30} {cnt}")
        if room_counter:
            txt_lines += ["", "--- ROOMS ---"]
            for room, cnt in room_counter.most_common():
                txt_lines.append(f"  {room:<30} {cnt}")
        if tod_counter:
            txt_lines += ["", "--- TIME OF DAY ---"]
            for tod in ["morning", "afternoon", "evening", "night"]:
                txt_lines.append(f"  {tod.capitalize():<30} {tod_counter.get(tod, 0)}")
        if dow_counter:
            txt_lines += ["", "--- DAY OF WEEK ---"]
            for day in day_names:
                txt_lines.append(f"  {day:<30} {dow_counter.get(day, 0)}")

        txt_lines += ["", "--- REWRITTEN SENTENCES ---"]
        for i, s in enumerate(out.get("sentences", [query]), 1):
            txt_lines.append(f"  [{i}] {s}")

        txt_path = out_dir / "summary.txt"
        txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
        print(f"[summary] Saved → {txt_path}")

    # ------------------------------------------------------------------
    # Save raw results
    # ------------------------------------------------------------------

    def _save_raw_results(self, out: Dict, out_dir: Path) -> None:
        """Serialise the full retrieval output (minus non-JSON-serialisable bits)."""
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()
                        if k not in ("batch_idx", "sample_idx")}
            if isinstance(obj, list):
                return [_clean(x) for x in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        path = out_dir / "results.json"
        with open(path, "w") as f:
            json.dump(_clean(out), f, indent=2, default=str)
        print(f"[data] Saved raw results → {path}")

    # ------------------------------------------------------------------
    # Empty-output fallback
    # ------------------------------------------------------------------

    def _empty_output(self, query: str, out: Dict) -> Path:
        out_dir = (
            self.output_base
            / self.dataset_name
            / self.dataset_split
            / _safe_query_name(query)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save_raw_results(out, out_dir)
        txt = out_dir / "summary.txt"
        txt.write_text(
            f"Query: {query}\n"
            f"No results returned (check threshold / query / model).\n"
            f"Sentences: {out.get('sentences', [])}\n"
            f"Reasoning: {out.get('reasoning', '')}\n",
            encoding="utf-8",
        )
        return out_dir


# ============================================================================
# Utility functions
# ============================================================================

def _fill_date_range(
    dates: List[datetime], window: str
) -> List[datetime]:
    """Return a complete list of dates/weeks between min and max."""
    step = timedelta(weeks=1) if window == "week" else timedelta(days=1)
    result = []
    cur = min(dates)
    end = max(dates)
    while cur <= end:
        result.append(cur)
        cur += step
    return result


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average; edges filled with NaN."""
    if len(arr) < window:
        return np.full_like(arr, np.nan, dtype=float)
    ma = np.full_like(arr, np.nan, dtype=float)
    half = window // 2
    for i in range(half, len(arr) - half):
        ma[i] = arr[i - half : i + half + 1].mean()
    return ma


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling standard deviation; edges filled with 0."""
    if len(arr) < window:
        return np.zeros_like(arr, dtype=float)
    rs = np.zeros_like(arr, dtype=float)
    half = window // 2
    for i in range(half, len(arr) - half):
        rs[i] = arr[i - half : i + half + 1].std()
    return rs


# ============================================================================
# CLI
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Longitudinal sensor-activity analysis via LLM query rewriting + FAISS retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Full dataset (train+val+test merged), threshold mode
          python src/evals/longitudinal/longitudinal_analysis.py \\
              --checkpoint trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v1/best_model.pt \\
              --data_dir   data/processed/casas/milan/FD_60 \\
              --home       milan \\
              --split      FD_60 \\
              --query      "morning kitchen routines" \\
              --threshold  0.10

          # Weekly aggregation with 4-week moving average
          python src/evals/longitudinal/longitudinal_analysis.py \\
              --checkpoint ... \\
              --data_dir   ... \\
              --home milan --split FD_60 \\
              --query "sedentary activities" \\
              --threshold 0.08 \\
              --time_window week --ma_window 4
        """),
    )
    p.add_argument("--checkpoint",  required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--data_dir",    default=None,
                   help="Dataset directory with train/val/test JSON files. "
                        "Loads and merges all splits automatically.")
    p.add_argument("--test_data",   default=None,
                   help="Single split JSON (alternative to --data_dir)")
    p.add_argument("--vocab",       default=None,
                   help="vocab.json path (required when using --test_data)")
    p.add_argument("--home",        default="milan",
                   help="Dataset name for LLM rewriter and cache (default: milan)")
    p.add_argument("--split",       default="FD_60",
                   help="Split label used in output path (default: FD_60)")
    p.add_argument("--query",       required=True,
                   help="Natural-language query")
    p.add_argument("--mode",        default="multi_location",
                   choices=["single", "multi_location", "multi_wording"])
    p.add_argument("--threshold",   type=float, default=0.10,
                   help="Cosine similarity floor (default: 0.10)")
    p.add_argument("--top_k",       type=int, default=200,
                   help="Max results when threshold is None (default: 200)")
    p.add_argument("--max_samples", type=int, default=50_000,
                   help="Max samples to load into FAISS index (default: 50 000)")
    p.add_argument("--time_window", default="day", choices=["day", "week"],
                   help="Time aggregation bucket (default: day)")
    p.add_argument("--ma_window",   type=int, default=7,
                   help="Moving-average window width in time_window units (default: 7)")
    p.add_argument("--output_base", default="results/long",
                   help="Root output directory (default: results/long)")
    p.add_argument("--llm_backend", default="gemini", choices=["gemini", "openai"])
    p.add_argument("--llm_model",   default=None)
    p.add_argument("--gemini_api_key", default=None)
    p.add_argument("--llm_captions_path", default=None)
    p.add_argument("--llm_n_examples", type=int, default=6)
    p.add_argument(
        "--max_subqueries",
        type=int,
        default=None,
        help="Max LLM retrieval sentences for multi_location (default 8) / "
             "multi_wording (default 6). Clamped 1–20.",
    )
    p.add_argument("--force_rewrite",   action="store_true")
    p.add_argument("--force_retrieve",  action="store_true")
    p.add_argument("--caption_style",   default="baseline")
    p.add_argument("--splits",          default="train,val,test")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    from query import SmartQuery

    if args.data_dir:
        sq = SmartQuery.from_data_dir(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            home=args.home,
            splits=[s.strip() for s in args.splits.split(",")],
            caption_style=args.caption_style,
            max_samples=args.max_samples,
            gemini_api_key=args.gemini_api_key,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            llm_captions_path=args.llm_captions_path,
            llm_n_examples=args.llm_n_examples,
        )
    else:
        if not args.test_data or not args.vocab:
            _build_parser().error(
                "--test_data and --vocab are required when not using --data_dir"
            )
        sq = SmartQuery.from_checkpoint(
            checkpoint_path=args.checkpoint,
            test_data_path=args.test_data,
            vocab_path=args.vocab,
            home=args.home,
            max_samples=args.max_samples,
            gemini_api_key=args.gemini_api_key,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            llm_captions_path=args.llm_captions_path,
            llm_n_examples=args.llm_n_examples,
        )

    analyzer = LongitudinalAnalyzer(
        smart_query=sq,
        dataset_name=args.home,
        dataset_split=args.split,
        output_base=args.output_base,
    )

    analyzer.analyze(
        query=args.query,
        mode=args.mode,
        threshold=args.threshold,
        top_k=args.top_k,
        time_window=args.time_window,
        ma_window=args.ma_window,
        force_rewrite=args.force_rewrite,
        force_retrieve=args.force_retrieve,
        max_subqueries=args.max_subqueries,
    )


if __name__ == "__main__":
    main()
