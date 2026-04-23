"""
Visualise longitudinal analysis results from llm_reasoning.py output JSON.

Layout
------
  1. Header (question + metadata)
  2. Model Response chat bubble
  3. One row per activity: [box plot | query sentence table]

Usage
-----
python src/query/plot_longitudinal.py results/longitudinal/milan_<ts>.json
python src/query/plot_longitudinal.py results/longitudinal/milan_<ts>.json --output report.png --show
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — avoids display/fork issues
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines
import numpy as np
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
_FONT_FAMILY   = "DejaVu Sans"
_COL_A         = "#3A6BC4"   # deep blue  — window A (baseline)
_COL_B         = "#C4503A"   # brick red  — window B (recent)
_COL_A_SOFT    = "#C8D9F5"
_COL_B_SOFT    = "#F5C8C0"
_BG_CARD       = "#F7F8FC"   # activity row background
_BG_BUBBLE     = "#EEF3FF"   # chat bubble fill
_BORDER_BUBBLE = "#BDD0F5"   # chat bubble border
_TABLE_HDR     = "#E8EDF8"
_TABLE_ALT     = "#F4F6FB"
_GRID          = "#EBEBEB"
_TEXT_MAIN     = "#1A1A2E"
_TEXT_SUB      = "#6B7280"

matplotlib.rcParams.update({
    "font.family":         _FONT_FAMILY,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.spines.left":    True,
    "axes.spines.bottom":  True,
    "axes.edgecolor":      "#CCCCCC",
    "axes.linewidth":      0.8,
    "xtick.color":         _TEXT_SUB,
    "ytick.color":         _TEXT_SUB,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
})


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fmt_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%-d %b '%y")
    except Exception:
        return iso[:10]


def _pct_str(pct: float | None) -> str:
    if pct is None:
        return "—"
    return f"{'+'if pct>=0 else ''}{pct:.1f}%"


def _sig_str(stats: dict) -> str:
    p = stats.get("p_value")
    if p is None:
        return ""
    sig = "significant" if stats.get("is_significant") else "not significant"
    return f"p = {p:.3f}  ({sig})"


def _wrap(text: str, width: int = 60) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def _rounded_box(ax: plt.Axes, x: float, y: float, w: float, h: float,
                 facecolor: str, edgecolor: str = "none",
                 lw: float = 1.2, pad: float = 0.02, zorder: int = 1):
    """Draw a rounded-rectangle patch in axes-fraction coordinates."""
    patch = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, transform=ax.transAxes,
        clip_on=False, zorder=zorder,
    )
    ax.add_patch(patch)


# Wrap width used consistently for both height estimation and rendering
_BUBBLE_WRAP_WIDTH = 80

# ---------------------------------------------------------------------------
# Chat bubble (Model Response)
# ---------------------------------------------------------------------------

def _bubble_line_count(report: str) -> int:
    """Number of wrapped lines the report will occupy (label line + text lines)."""
    if not report:
        return 2
    return 1 + len(textwrap.wrap(report, _BUBBLE_WRAP_WIDTH))


def _draw_bubble(ax: plt.Axes, report: str):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Layout constants (all in axes-fraction units)
    PAD_TOP    = 0.06   # space above "Model Response:" label
    PAD_SIDE   = 0.018
    PAD_BOT    = 0.06
    LABEL_H    = 0.18   # height of the bold label row
    LINE_H     = 0.145  # height per wrapped text line

    lines = textwrap.wrap(report, _BUBBLE_WRAP_WIDTH) if report else ["(No report generated.)"]
    n_lines = len(lines)

    content_h = LABEL_H + n_lines * LINE_H
    total_h   = PAD_TOP + content_h + PAD_BOT

    # If content is taller than axes (shouldn't happen after sizing), scale down
    if total_h > 0.98:
        scale  = 0.98 / total_h
        LABEL_H *= scale
        LINE_H  *= scale
        PAD_TOP *= scale
        PAD_BOT *= scale
        total_h  = 0.98

    bubble_y = 1.0 - total_h

    # Rounded bubble background — sized to content
    _rounded_box(ax, 0.0, bubble_y, 1.0, total_h - 0.01,
                 facecolor=_BG_BUBBLE, edgecolor=_BORDER_BUBBLE,
                 lw=1.4, pad=0.015, zorder=1)

    # "Model Response:" bold label
    label_y = 1.0 - PAD_TOP - LABEL_H * 0.5
    ax.text(PAD_SIDE, label_y, "Model Response:",
            transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", color=_COL_A,
            va="center", zorder=2)

    # Report text lines, top-to-bottom
    text_start_y = 1.0 - PAD_TOP - LABEL_H
    for i, line in enumerate(lines):
        ly = text_start_y - (i + 0.5) * LINE_H
        ax.text(PAD_SIDE, ly, line,
                transform=ax.transAxes,
                fontsize=8.2, color=_TEXT_MAIN,
                va="center", zorder=2)


# ---------------------------------------------------------------------------
# Box plot
# ---------------------------------------------------------------------------

def _draw_boxplot(ax: plt.Axes, result: dict, wa_short: str, wb_short: str):
    wa_counts = np.array(result["window_a"]["daily_counts"], dtype=float)
    wb_counts = np.array(result["window_b"]["daily_counts"], dtype=float)
    stats     = result["stats"]

    ax.set_facecolor(_BG_CARD)
    ax.yaxis.grid(True, color=_GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    bp = ax.boxplot(
        [wa_counts, wb_counts],
        positions=[1, 2], widths=0.46,
        patch_artist=True, notch=False, showfliers=False,
        medianprops=dict(color="white", linewidth=2.2),
        whiskerprops=dict(linewidth=1.2, linestyle="--"),
        capprops=dict(linewidth=1.4),
        boxprops=dict(linewidth=0),
    )
    for patch, col in zip(bp["boxes"], [_COL_A, _COL_B]):
        patch.set_facecolor(col); patch.set_alpha(0.72)
    for item, cols in [("whiskers", [_COL_A]*2 + [_COL_B]*2),
                       ("caps",     [_COL_A]*2 + [_COL_B]*2)]:
        for line, col in zip(bp[item], cols):
            line.set_color(col); line.set_alpha(0.8)

    # Jittered individual points
    rng = np.random.default_rng(42)
    for pos, arr, col in [(1, wa_counts, _COL_A), (2, wb_counts, _COL_B)]:
        jitter = rng.uniform(-0.11, 0.11, size=len(arr))
        ax.scatter(np.full(len(arr), pos) + jitter, arr,
                   s=24, color=col, zorder=5, alpha=0.9,
                   edgecolors="white", linewidths=0.6)
        ax.scatter(pos, arr.mean(), marker="D", s=44, color="white",
                   zorder=6, edgecolors=col, linewidths=1.8)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([wa_short, wb_short], fontsize=8.2)
    ax.set_ylabel("Matches / day", fontsize=8, color=_TEXT_SUB)
    ax.set_xlim(0.35, 2.65)

    # Clamp y-axis: 90th percentile of combined daily counts.
    # IQR-based fences collapse to 0 on sparse data (many zero days), so we
    # use a percentile ceiling instead. Values above are shown as ▲ + label.
    combined = np.concatenate([wa_counts, wb_counts])
    if len(combined) > 0 and combined.max() > 0:
        fence = float(np.percentile(combined, 90))
        # Guarantee the fence is at least 1 and shows at least the second-largest value
        sorted_vals = np.sort(combined)
        second_largest = sorted_vals[-2] if len(sorted_vals) >= 2 else sorted_vals[-1]
        fence = max(fence, second_largest, 1.0)
        y_max = fence * 1.35
        ax.set_ylim(0, y_max)
        # Mark clipped points with ▲ at the ceiling + their actual value
        for pos, arr, col in [(1, wa_counts, _COL_A), (2, wb_counts, _COL_B)]:
            clipped = arr[arr > y_max * 0.97]
            if len(clipped) > 0:
                jitter = rng.uniform(-0.08, 0.08, size=len(clipped))
                ax.scatter(np.full(len(clipped), pos) + jitter,
                           np.full(len(clipped), y_max * 0.95),
                           marker="^", s=38, color=col, zorder=7,
                           edgecolors="white", linewidths=0.6, alpha=0.95)
                ax.text(pos, y_max * 0.98,
                        f"({int(clipped.max())})",
                        ha="center", va="bottom", fontsize=6.5,
                        color=col, zorder=8)

    # Totals + % change as a text annotation inside the plot (top-centre)
    wa_tot  = result["window_a"]["total_matches"]
    wb_tot  = result["window_b"]["total_matches"]
    pct     = stats.get("pct_change")
    col_pct = _COL_B if (pct or 0) >= 0 else _COL_A
    ax.text(0.5, 0.97,
            f"total  {wa_tot} → {wb_tot}   {_pct_str(pct)}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7.8,
            color=col_pct if pct and abs(pct) >= 5 else _TEXT_SUB)

    sig = _sig_str(stats)
    if sig:
        ax.text(0.5, -0.20, sig, transform=ax.transAxes,
                ha="center", fontsize=7.2, color=_TEXT_SUB)

    # Store proxy artists for the combined legend drawn in plot_report
    ax._mean_proxy   = ax.scatter([], [], marker="D", s=28, color="white",
                                  edgecolors=_TEXT_SUB, linewidths=1.4)
    ax._median_proxy = ax.plot([], [], color=_TEXT_SUB, lw=2)[0]


# ---------------------------------------------------------------------------
# Sentence table
# ---------------------------------------------------------------------------

def _draw_table(ax: plt.Axes, result: dict, wa_col_hdr: str, wb_col_hdr: str):
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    wa_qc    = result["window_a"]["per_query_counts"]
    wb_qc    = result["window_b"]["per_query_counts"]
    wa_dqc   = result["window_a"].get("per_query_daily_counts", {})
    wb_dqc   = result["window_b"].get("per_query_daily_counts", {})
    sentences = list(wa_qc.keys())

    WRAP_W   = 34
    FONT_SZ  = 8.2
    LINE_H   = 0.080
    PAD_V    = 0.012

    # Column layout [sentence | wa | wb | delta]
    cx = [0.00, 0.40, 0.54, 0.68]
    cw = [0.39, 0.13, 0.13, 0.31]

    def _short_hdr(h: str) -> str:
        parts = h.split("(")
        if len(parts) == 2:
            dates = parts[1].rstrip(")").replace(" '09", "").replace(" '10", "")
            label = parts[0].strip().replace("first", "A").replace("last", "B")
            return f"{label}\n{dates}"
        return h

    hdrs = ["Retrieval sentence", _short_hdr(wa_col_hdr), _short_hdr(wb_col_hdr), "Δ raw (%) / p-val"]

    wrapped_rows = [textwrap.wrap(s, WRAP_W) for s in sentences]
    row_heights  = [len(r) * LINE_H + PAD_V * 2 for r in wrapped_rows]
    hdr_h        = LINE_H * 1.8 + PAD_V * 2   # taller header for 2-line date labels

    total_h = hdr_h + sum(row_heights)
    scale   = min(1.0, 0.96 / total_h) if total_h > 0.96 else 1.0
    hdr_h       *= scale
    row_heights  = [h * scale for h in row_heights]
    fs           = FONT_SZ * max(0.72, scale)

    def _bg(x, y, w, h, color):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="square,pad=0",
            facecolor=color, edgecolor="none",
            transform=ax.transAxes, clip_on=True, zorder=1,
        ))

    def _txt(x, y, text, ha="left", bold=False, color=_TEXT_MAIN):
        ax.text(x, y, text,
                transform=ax.transAxes, ha=ha, va="center",
                fontsize=fs * (1.05 if bold else 1.0),
                fontweight="bold" if bold else "normal",
                color=color, zorder=2)

    # Header
    cur_y = 1.0 - hdr_h
    for hdr, x, w in zip(hdrs, cx, cw):
        _bg(x, cur_y, w, hdr_h, _TABLE_HDR)
        ha = "center" if hdr != hdrs[0] else "left"
        tx = x + (w / 2 if ha == "center" else 0.010)
        _txt(tx, cur_y + hdr_h / 2, hdr, ha=ha, bold=True, color=_TEXT_MAIN)

    ax.plot([0, 1], [cur_y, cur_y], color="#BBCCE8", lw=0.9,
            transform=ax.transAxes, clip_on=False, zorder=3)

    for i, (lines, rh) in enumerate(zip(wrapped_rows, row_heights)):
        row_y  = cur_y - rh
        row_bg = _TABLE_ALT if i % 2 else "white"

        for x, w in zip(cx, cw):
            _bg(x, row_y, w, rh, row_bg)

        # Sentence text
        lh = rh / max(len(lines), 1)
        for li, line in enumerate(lines):
            ly = row_y + rh - (li + 0.5) * lh
            ax.text(cx[0] + 0.010, ly, line,
                    transform=ax.transAxes, ha="left", va="center",
                    fontsize=fs, color=_TEXT_MAIN, zorder=2)

        # Count cells (A and B)
        wa_val = wa_qc.get(sentences[i], 0)
        wb_val = wb_qc.get(sentences[i], 0)
        for col_i, (val, col) in enumerate([(wa_val, _COL_A), (wb_val, _COL_B)], 1):
            tx = cx[col_i] + cw[col_i] / 2
            _txt(tx, row_y + rh / 2, str(val), ha="center",
                 bold=(val > 0), color=col if val > 0 else "#AAAAAA")

        # Delta cell: raw Δ, % Δ, and per-sentence p-value
        delta = wb_val - wa_val
        if delta == 0:
            delta_col  = "#AAAAAA"
            delta_sign = ""
            pct_str    = "—"
        elif delta > 0:
            delta_col  = _TEXT_MAIN
            delta_sign = "+"
            pct_str    = f"+{round(delta / wa_val * 100):d}%" if wa_val > 0 else "new"
        else:
            delta_col  = _TEXT_MAIN
            delta_sign = ""
            pct_str    = f"{round(delta / wa_val * 100):d}%" if wa_val > 0 else "–100%"

        delta_str = f"{delta_sign}{delta} ({pct_str})" if delta != 0 else "—"

        # Per-sentence Mann-Whitney U → significance stars
        wa_days = wa_dqc.get(sentences[i], [])
        wb_days = wb_dqc.get(sentences[i], [])
        sig_str = ""
        if wa_days and wb_days and (any(v > 0 for v in wa_days) or any(v > 0 for v in wb_days)):
            try:
                _, pv = scipy_stats.mannwhitneyu(wa_days, wb_days, alternative="two-sided")
                if pv < 0.01:
                    sig_str = "***"
                elif pv < 0.05:
                    sig_str = "**"
                elif pv < 0.1:
                    sig_str = "*"
            except Exception:
                pass

        tx = cx[3] + cw[3] / 2
        cell_mid = row_y + rh / 2
        if sig_str:
            offset = rh * 0.22
            _txt(tx, cell_mid + offset, delta_str, ha="center",
                 bold=(delta != 0), color=delta_col)
            _txt(tx, cell_mid - offset, sig_str, ha="center",
                 bold=True, color="#444444")
        else:
            _txt(tx, cell_mid, delta_str, ha="center",
                 bold=(delta != 0), color=delta_col)

        ax.plot([0, 1], [row_y, row_y], color="#E4EAF5", lw=0.5,
                transform=ax.transAxes, clip_on=False, zorder=3)
        cur_y = row_y


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def plot_report(data: dict, output_path: str | None = None, show: bool = False):
    results  = data["results"]
    n_act    = len(results)
    wa_meta  = data["windows"]["window_a"]
    wb_meta  = data["windows"]["window_b"]

    wa_short = wa_meta["label"]
    wb_short = wb_meta["label"]
    wa_hdr   = f"{wa_meta['label']}  ({_fmt_date(wa_meta['start'])} – {_fmt_date(wa_meta['end'])})"
    wb_hdr   = f"{wb_meta['label']}  ({_fmt_date(wb_meta['start'])} – {_fmt_date(wb_meta['end'])})"

    report = (data.get("report") or "").strip()

    # Estimate row heights from actual content
    max_sents = max(len(r["window_a"]["per_query_counts"]) for r in results)
    act_row_h = max(4.0, 1.0 + max_sents * 0.52)
    # Bubble: label line (0.22 in) + text lines (0.17 in each) + top/bot padding (0.12 in each)
    n_report_lines = _bubble_line_count(report)
    bubble_h = max(0.65, 0.22 + n_report_lines * 0.17 + 0.24)

    # Header is drawn via fig.text (no gridspec row) — eliminates hspace gap
    hdr_h   = 0.40   # reserved inches at the top of the figure for header text
    total_h = hdr_h + bubble_h + n_act * act_row_h
    fig = plt.figure(figsize=(8.8, total_h))
    fig.patch.set_facecolor("white")

    # hspace only applies between bubble and activity rows now
    top_frac = 1.0 - hdr_h / total_h   # where gridspec starts (just below header)
    h_ratios = [bubble_h] + [act_row_h] * n_act
    gs = fig.add_gridspec(
        1 + n_act, 2,
        height_ratios=h_ratios,
        width_ratios=[1.1, 1.8],
        hspace=0.40, wspace=0.06,
        left=0.06, right=0.99, top=top_frac, bottom=0.02,
    )

    # ── Header drawn directly on the figure canvas ───────────────────────────
    question = data.get("question", "")
    gen      = data.get("generated_at", "")[:16].replace("T", " ")
    thr      = data.get("similarity_threshold", "?")
    period   = f"{_fmt_date(wa_meta['start'])} → {_fmt_date(wb_meta['end'])}"

    fig.text(0.06, 1.0 - 0.04 / total_h,
             f'"{question}"',
             fontsize=12, fontweight="bold", color=_TEXT_MAIN,
             va="top", transform=fig.transFigure)
    fig.text(0.06, 1.0 - 0.26 / total_h,
             f"Generated {gen}   ·   similarity threshold = {thr}   ·   {period}",
             fontsize=8, color=_TEXT_SUB, va="top", transform=fig.transFigure)
    # Thin divider at the top_frac boundary
    fig.add_artist(matplotlib.lines.Line2D(
        [0.06, 0.99], [top_frac, top_frac],
        transform=fig.transFigure, color="#DDDDDD", lw=0.8, clip_on=False,
    ))

    # ── Chat bubble ─────────────────────────────────────────────────────────
    ax_bub = fig.add_subplot(gs[0, :])
    _draw_bubble(ax_bub, report)

    # ── Activity rows ────────────────────────────────────────────────────────
    for row_i, result in enumerate(results):
        ax_box = fig.add_subplot(gs[1 + row_i, 0])
        ax_tbl = fig.add_subplot(gs[1 + row_i, 1])

        activity = result["activity"].title()
        _draw_boxplot(ax_box, result, wa_short, wb_short)
        # Activity label as a clean text above the axes (avoids title collision)
        ax_box.text(-0.02, 1.13, activity,
                    transform=ax_box.transAxes,
                    fontsize=11, fontweight="bold", color=_TEXT_MAIN, va="bottom")
        _draw_table(ax_tbl, result, wa_hdr, wb_hdr)

        patch_a = mpatches.Patch(color=_COL_A, alpha=0.72, label=wa_short)
        patch_b = mpatches.Patch(color=_COL_B, alpha=0.72, label=wb_short)
        handles = [patch_a, patch_b]
        labels  = [wa_short, wb_short]
        # Append mean / median markers stored by _draw_boxplot
        if hasattr(ax_box, "_mean_proxy"):
            handles += [ax_box._mean_proxy, ax_box._median_proxy]
            labels  += ["mean", "median"]
        ax_box.legend(handles=handles, labels=labels, fontsize=6.8,
                      loc="upper right", framealpha=0.9,
                      edgecolor="#DDDDDD", fancybox=True,
                      handlelength=1.4, handleheight=0.9)

        # Subtle card background behind the whole row
        for ax in (ax_box, ax_tbl):
            ax.set_facecolor(_BG_CARD)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved → {output_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Plot longitudinal analysis results.")
    p.add_argument("input", help="Path to llm_reasoning.py output JSON.")
    p.add_argument("--output", default=None,
                   help="Output image (.png/.pdf). Defaults to input path with .png extension.")
    p.add_argument("--show", action="store_true",
                   help="Open an interactive matplotlib window after saving.")
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    out = args.output or str(Path(args.input).with_suffix(".png"))
    plot_report(data, output_path=out, show=args.show)


if __name__ == "__main__":
    main()
