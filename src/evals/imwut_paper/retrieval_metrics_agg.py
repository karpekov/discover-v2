"""Aggregate prototype retrieval metrics across multiple CASAS datasets.

Reads  results/evals/{dataset}/FD_60_p/{model}/comprehensive_results.json
for each dataset and collects per-label macro metrics (MRR, mAP, P@10, P@50,
P@100) for prototype2sensor (p2s) and prototype2text (p2t) directions at both
L1 and L2 label levels.

Label normalisation
-------------------
Raw label variants are mapped to canonical display names via LABEL_NORM.
When two raw labels from the *same* dataset and *same* level map to the same
canonical name their metric values are averaged.

Outputs
-------
  {stem}.json   -- machine-readable aggregation (summary + per-label tables)
  {stem}.md     -- Markdown report (summary tables then per-label detail)
  (terminal)    -- summary tables only (per-label detail is in the .md file)

Usage
-----
  python src/evals/imwut_paper/retrieval_metrics_agg.py \\
      --datasets milan aruba cairo \\
      --model_suffix fd60_seq_rb1_textclip_projmlp_clipmlm_v3 \\
      --base_dir results/evals \\
      --results_subdir FD_60_p \\
      --output_dir results/evals/imwut_paper
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── constants ─────────────────────────────────────────────────────────────────

# Directions to analyse: (json_key, short_tag, display_name)
DIRECTIONS: List[Tuple[str, str, str]] = [
    ("prototype2sensor", "p2s", "Prototype → Sensor"),
    ("prototype2text",   "p2t", "Prototype → Text"),
]

METRICS: List[Tuple[str, str]] = [
    ("mrr",  "MRR"),
    ("map",  "mAP"),
    ("10",   "P@10"),
    ("50",   "P@50"),
    ("100",  "P@100"),
]
MISSING = "--"

# Raw label name (lowercase) → canonical display name.
LABEL_NORM: Dict[str, str] = {
    "sleep":                "Sleep",
    "sleeping":             "Sleep",
    "bed_to_toilet":        "Bed to Toilet",
    "bed to toilet":        "Bed to Toilet",
    "leave home":           "Leave Home",
    "leave_home":           "Leave Home",
    "chores":               "Chores",
    "laundry":              "Chores",
    "eating":               "Eating: General",
    "eat":                  "Eating: General",
    "dining_rm_activity":   "Eating: General",
    "breakfast":            "Eating: Breakfast",
    "lunch":                "Eating: Lunch",
    "dinner":               "Eating: Dinner",
    "work":                 "Work",
    "desk_activity":        "Work",
    "r1 work in office":    "Work",
    "relax":                "Relax: General",
    "watch_tv":             "Relax: Watch TV",
    "read":                 "Relax: Read",
}

CANONICAL_ORDER: List[str] = [
    "Sleep", "Bed to Toilet", "Leave Home", "Chores",
    "Eating: General", "Eating: Breakfast", "Eating: Lunch", "Eating: Dinner",
    "Work",
    "Relax: General", "Relax: Watch TV", "Relax: Read",
]
_CANONICAL_RANK: Dict[str, int] = {lbl: i for i, lbl in enumerate(CANONICAL_ORDER)}


# ── label helpers ─────────────────────────────────────────────────────────────

def canonical_sort_key(label: str) -> Tuple[int, str]:
    return (_CANONICAL_RANK.get(label, len(CANONICAL_ORDER)), label)


def normalize_label(raw: str) -> str:
    return LABEL_NORM.get(raw.lower().strip(), raw)


# ── data loading ──────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_per_label(
    data: dict, label_level: str, direction: str
) -> Dict[str, Dict[str, float]]:
    """Return {canonical_label: {metric_key: value}}, macro only, for one direction."""
    try:
        pl = data["retrieval_metrics"][label_level]["prototype_based"]["per_label"][direction]
    except KeyError:
        return {}

    raw_labels: Dict[str, Dict[str, float]] = {}
    for metric_key, _ in METRICS:
        for label, value in pl.get(metric_key, {}).items():
            raw_labels.setdefault(label, {})[metric_key] = float(value)

    acc: Dict[str, Dict[str, List[float]]] = {}
    for raw, metrics in raw_labels.items():
        canon = normalize_label(raw)
        bucket = acc.setdefault(canon, {})
        for mk, v in metrics.items():
            bucket.setdefault(mk, []).append(v)

    return {
        canon: {mk: sum(vs) / len(vs) for mk, vs in metric_vals.items()}
        for canon, metric_vals in acc.items()
    }


def extract_overall(data: dict, label_level: str, direction: str) -> Dict[str, float]:
    """Return {metric_key: macro_value} for one direction."""
    try:
        ov = data["retrieval_metrics"][label_level]["prototype_based"]["overall"][direction]
    except KeyError:
        return {}
    result = {}
    for metric_key, _ in METRICS:
        entry = ov.get(metric_key, {})
        result[metric_key] = float(entry.get("macro", 0.0) if isinstance(entry, dict) else entry)
    return result


def collect_data(
    datasets: List[str],
    model_suffix: str,
    base_dir: Path,
    results_subdir: str,
    direction: str,
) -> Tuple[
    Dict[str, Dict[str, Dict[str, float]]],  # per_label[ds]["[L1] canon"][metric]
    Dict[str, Dict[str, float]],              # overall[ds]
    List[str],                                # ordered_labels
]:
    per_label: Dict[str, Dict[str, Dict[str, float]]] = {}
    overall:   Dict[str, Dict[str, float]]             = {}
    l1_per_ds: Dict[str, List[str]] = {}
    l2_per_ds: Dict[str, List[str]] = {}

    for ds in datasets:
        model_name = f"{ds}_{model_suffix}"
        json_path  = base_dir / ds / results_subdir / model_name / "comprehensive_results.json"
        if not json_path.exists():
            print(f"[WARN] Missing: {json_path}", file=sys.stderr)
            per_label[ds] = {}
            overall[ds]   = {}
            l1_per_ds[ds] = []
            l2_per_ds[ds] = []
            continue

        data = load_json(json_path)
        l1 = extract_per_label(data, "L1", direction)
        l2 = extract_per_label(data, "L2", direction)

        merged: Dict[str, Dict[str, float]] = {}
        for canon, metrics in l1.items():
            merged[f"[L1] {canon}"] = metrics
        for canon, metrics in l2.items():
            merged[f"[L2] {canon}"] = metrics

        per_label[ds] = merged
        l1_per_ds[ds] = [f"[L1] {c}" for c in l1]
        l2_per_ds[ds] = [f"[L2] {c}" for c in l2]
        overall[ds]   = {
            "L1": extract_overall(data, "L1", direction),
            "L2": extract_overall(data, "L2", direction),
        }

    all_l1 = sorted(
        {lbl for ds in datasets for lbl in l1_per_ds.get(ds, [])},
        key=lambda x: canonical_sort_key(x[4:]),
    )
    all_l2 = sorted(
        {lbl for ds in datasets for lbl in l2_per_ds.get(ds, [])},
        key=lambda x: canonical_sort_key(x[4:]),
    )
    return per_label, overall, all_l1 + all_l2


# ── table builder ─────────────────────────────────────────────────────────────

def build_metric_table(
    datasets: List[str],
    per_label: Dict[str, Dict[str, Dict[str, float]]],
    overall:   Dict[str, Dict[str, float]],
    ordered_labels: List[str],
    metric_key: str,
    metric_name: str,
    direction: str,
    direction_display: str,
) -> Dict:
    rows = []
    for lbl in ordered_labels:
        level = "L1" if lbl.startswith("[L1]") else "L2"
        row   = {"label": lbl[4:], "level": level}
        for ds in datasets:
            val = per_label.get(ds, {}).get(lbl, {}).get(metric_key)
            row[ds] = round(val, 4) if val is not None else None
        rows.append(row)

    averages = {}
    for ds in datasets:
        for level in ("L1", "L2"):
            vals = [r[ds] for r in rows if r["level"] == level and r[ds] is not None]
            averages[f"{ds}_{level}"] = round(sum(vals) / len(vals), 4) if vals else None
        all_vals = [r[ds] for r in rows if r[ds] is not None]
        averages[f"{ds}_all"] = round(sum(all_vals) / len(all_vals), 4) if all_vals else None

    return {
        "metric_key":        metric_key,
        "metric_name":       metric_name,
        "direction":         direction,
        "direction_display": direction_display,
        "datasets":          datasets,
        "rows":              rows,
        "averages":          averages,
        "overall":           {ds: overall.get(ds, {}) for ds in datasets},
    }


# ── terminal: summary only ────────────────────────────────────────────────────

def _ds_avg(vals: List[Optional[float]]) -> Optional[float]:
    present = [v for v in vals if v is not None]
    return round(sum(present) / len(present), 4) if present else None


def print_summary(all_tables: List[dict], datasets: List[str], direction_display: str) -> None:
    """Print three compact summary tables (L1 / L2 / ALL) for one direction."""
    col_w   = max(10, max(len(ds) for ds in datasets))
    met_w   = max(8,  max(len(t["metric_name"]) for t in all_tables))
    avg_w   = max(10, len("Average"))
    total_w = met_w + 2 + (col_w + 3) * len(datasets) + avg_w + 3

    border = "=" * total_w
    print(f"\n{border}")
    print(f"  SUMMARY — {direction_display}  (macro)")
    print(border)

    for level_label, level_key in [
        ("L1 Labels",           "L1"),
        ("L2 Labels",           "L2"),
        ("All Labels (L1+L2)",  "all"),
    ]:
        print(f"\n  {level_label}")
        hdr = f"  {'Metric':<{met_w}}"
        for ds in datasets:
            hdr += f"  {ds.upper():>{col_w}}"
        hdr += f"  {'Average':>{avg_w}}"
        print(hdr)
        print("  " + "-" * (total_w - 2))

        for t in all_tables:
            ds_vals = [t["averages"].get(f"{ds}_{level_key}") for ds in datasets]
            cross   = _ds_avg(ds_vals)
            line    = f"  {t['metric_name']:<{met_w}}"
            for v in ds_vals:
                cell = f"{v:.2f}" if v is not None else MISSING
                line += f"  {cell:>{col_w}}"
            cross_cell = f"{cross:.2f}" if cross is not None else MISSING
            line += f"  {cross_cell:>{avg_w}}"
            print(line)

    print(f"\n{border}")


# ── markdown builders ─────────────────────────────────────────────────────────

def build_summary_md(
    all_tables: List[dict], datasets: List[str], direction_display: str
) -> str:
    def fmt(v) -> str:
        return f"{v:.2f}" if v is not None else MISSING

    col_headers = [ds.capitalize() for ds in datasets] + ["**Average**"]
    header_row  = "| Metric | " + " | ".join(col_headers) + " |"
    sep_row     = "| --- | " + " | ".join(["---"] * len(col_headers)) + " |"

    lines: List[str] = [f"## Summary — {direction_display} (macro)", ""]
    for level_label, level_key in [
        ("L1 Labels", "L1"), ("L2 Labels", "L2"), ("All Labels (L1 + L2)", "all")
    ]:
        lines += [f"### {level_label}", "", header_row, sep_row]
        for t in all_tables:
            ds_vals   = [t["averages"].get(f"{ds}_{level_key}") for ds in datasets]
            cross_avg = _ds_avg(ds_vals)
            cells     = " | ".join(fmt(v) for v in ds_vals) + " | " + fmt(cross_avg)
            lines.append(f"| {t['metric_name']} | {cells} |")
        lines.append("")

    lines.append("---\n")
    return "\n".join(lines)


def table_to_markdown(table: dict, datasets: List[str]) -> str:
    rows        = table["rows"]
    averages    = table["averages"]
    metric_name = table["metric_name"]
    dir_display = table["direction_display"]

    def fmt(v) -> str:
        return f"{v:.2f}" if v is not None else MISSING

    lines: List[str] = [
        f"### {metric_name} — {dir_display} (macro)",
        "",
        f"| Level | Label | {' | '.join(ds.capitalize() for ds in datasets)} |",
        f"| --- | --- | {' | '.join(['---'] * len(datasets))} |",
    ]

    prev_level = None
    for r in rows:
        if r["level"] != prev_level and prev_level is not None:
            lines.append(f"| | | {' | '.join([''] * len(datasets))} |")
        prev_level = r["level"]
        cells = " | ".join(fmt(r[ds]) for ds in datasets)
        lines.append(f"| {r['level']} | {r['label']} | {cells} |")

    lines.append(f"| | | {' | '.join([''] * len(datasets))} |")
    for level in ("L1", "L2", "all"):
        cells = " | ".join(fmt(averages.get(f"{ds}_{level}")) for ds in datasets)
        lines.append(f"| | **AVG ({level})** | {cells} |")

    lines.append("")
    return "\n".join(lines)


# ── LaTeX comparison table (auto-saved as retrieval_results_quant.tex) ───────

# Number of L1 activity classes per dataset — used for random baseline.
RANDOM_BASELINE_N: Dict[str, int] = {"milan": 11, "aruba": 9, "cairo": 8}

# Metric display order for the comparison table (left → right).
LATEX_METRIC_ORDER: List[str] = ["P@10", "P@50", "P@100", "mAP", "MRR"]

# Maps direction json_key → row label in the LaTeX table.
DIRECTION_ROW_LABELS: Dict[str, str] = {
    "prototype2text":   "Text-Only Baseline",
    "prototype2sensor": "\\ToolName",
}


def compute_random_baselines(datasets: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """Return {metric_name: {dataset: value, 'average': value}} for a random ranker.

    P@k / mAP baseline = 1/N.
    MRR baseline       = mean(1/i for i in 1..N).
    """
    rows: Dict[str, Dict[str, Optional[float]]] = {}
    for metric_name in LATEX_METRIC_ORDER:
        row: Dict[str, Optional[float]] = {}
        for ds in datasets:
            N = RANDOM_BASELINE_N.get(ds)
            if N is None:
                row[ds] = None
            elif metric_name == "MRR":
                row[ds] = round(sum(1.0 / i for i in range(1, N + 1)) / N, 4)
            else:
                row[ds] = round(1.0 / N, 4)
        present = [v for v in row.values() if v is not None]
        row["average"] = round(sum(present) / len(present), 4) if present else None
        rows[metric_name] = row
    return rows


def _fmt_cell(value: Optional[float], is_best: bool) -> str:
    if value is None:
        return MISSING
    s = f"{value:.2f}"
    return f"\\textbf{{{s}}}" if is_best else s


def _build_comparison_tabular(
    datasets: List[str],
    random_baselines: Dict[str, Dict[str, Optional[float]]],
    summaries_by_dir: Dict[str, Dict[str, Dict[str, Optional[float]]]],
) -> List[str]:
    """Return LaTeX lines from \\begin{tabular} to \\end{tabular} (inclusive)."""
    sub_keys = datasets + ["average"]
    ds_abbr  = {ds: ds.capitalize()[:2] + "." for ds in datasets}
    ds_abbr["average"] = "Avg."

    our_method_label = DIRECTION_ROW_LABELS.get("prototype2sensor")
    non_our: List[Tuple[str, Dict]] = [("Random Baseline", random_baselines)]
    our: List[Tuple[str, Dict]] = []
    for dir_key, _, _ in DIRECTIONS:
        label = DIRECTION_ROW_LABELS.get(dir_key, dir_key)
        data  = summaries_by_dir.get(dir_key, {})
        (our if label == our_method_label else non_our).append((label, data))
    row_defs = non_our + our  # Our Method always last

    best: Dict[Tuple[str, str], float] = {}
    for metric in LATEX_METRIC_ORDER:
        for sk in sub_keys:
            vals    = [d.get(metric, {}).get(sk) for _, d in row_defs]
            present = [v for v in vals if v is not None]
            if present:
                best[(metric, sk)] = max(present)

    # Metrics after which a double rule separates the next group
    _DOUBLE_AFTER = {"P@100", "mAP"}

    n = len(LATEX_METRIC_ORDER)

    # Column spec: l | rrr|r | rrr|r | rrr|r || rrr|r || rrr|r
    #   - rrr|r  : single rule between Ca. and Avg. within every group
    #   - |  / || : single / double rule between metric groups
    col_parts: List[str] = ["l|"]
    for i, metric in enumerate(LATEX_METRIC_ORDER):
        # !{\vrule} is drawn traditionally (interrupted by hlines); | is drawn by
        # nicematrix via TikZ (uninterrupted). Use !{\vrule} for Ca./Avg. separator.
        col_parts.append("rrr!{\\vrule}r")
        if i < n - 1:
            col_parts.append("||" if metric in _DOUBLE_AFTER else "|")
    col_spec = "".join(col_parts)

    # \multicolumn headers: double right border after P@100 and mAP
    mc_headers = []
    for i, m in enumerate(LATEX_METRIC_ORDER):
        if i == n - 1:
            border = "c"
        elif m in _DOUBLE_AFTER:
            border = "c||"
        else:
            border = "c|"
        mc_headers.append(f"\\multicolumn{{4}}{{{border}}}{{\\textbf{{{m}}}}}")

    # NiceTabular (nicematrix package) draws all vertical rules via TikZ *after*
    # horizontal rules, so || appears as a solid, uninterrupted double line.
    hdr_cmidrules = [
        f"\\cmidrule({'lr' if i < n - 1 else 'l'}){{{2 + i * 4}-{5 + i * 4}}}"
        for i in range(n)
    ]
    sub_abbrs = [ds_abbr[sk] for sk in sub_keys]

    lines: List[str] = [
        f"\\begin{{NiceTabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join([""] + mc_headers) + " \\\\",
        "".join(hdr_cmidrules),
        "\\textbf{Method} & " + " & ".join(sub_abbrs * n) + " \\\\",
        "\\midrule",
    ]
    last_idx = len(row_defs) - 1
    for i, (label, data) in enumerate(row_defs):
        cells = [label]
        for metric in LATEX_METRIC_ORDER:
            for sk in sub_keys:
                v       = data.get(metric, {}).get(sk)
                top     = best.get((metric, sk))
                is_best = v is not None and top is not None and abs(v - top) < 1e-9
                cells.append(_fmt_cell(v, is_best))
        lines.append(" & ".join(cells) + " \\\\")
        if i == 0:
            lines.append("\\midrule")
        elif i == last_idx - 1:
            # Heavier rule above Our Method row (same weight as \toprule/\bottomrule)
            lines.append("\\specialrule{0.10em}{\\aboverulesep}{\\belowrulesep}")
    lines += ["\\bottomrule", "\\end{NiceTabular}"]
    return lines


def generate_latex_comparison_table(
    datasets: List[str],
    random_baselines: Dict[str, Dict[str, Optional[float]]],
    summaries_by_dir: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    output_path: Path,
    label_level: str = "L1",
) -> None:
    """Write a 3-row LaTeX comparison table wrapped in table* for an ACM paper."""
    tabular = _build_comparison_tabular(datasets, random_baselines, summaries_by_dir)
    lines: List[str] = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        *tabular,
        "}",
        (
            f"\\caption{{Prototype-to-sensor retrieval performance on CASAS datasets using their annotations, macro averaged."
            f"Best result per column in \\textbf{{bold}}.}}"
        ),
        "\\label{tab:retrieval_quant}",
        "\\end{table*}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅  TEX  saved: {output_path}")


# ── shared LaTeX preamble + compile helper ────────────────────────────────────

_TEX_FALLBACKS = [
    "/Library/TeX/texbin/pdflatex",
    "/usr/local/texlive/2026/bin/universal-darwin/pdflatex",
    "/usr/local/texlive/2024/bin/universal-darwin/pdflatex",
    "/usr/local/texlive/2023/bin/universal-darwin/pdflatex",
]

_STANDALONE_PREAMBLE: List[str] = [
    r"\documentclass[border=8pt]{standalone}",
    r"\usepackage{booktabs}",
    r"\usepackage{multirow}",
    r"\usepackage{array}",
    r"\usepackage{xcolor}",
    r"\usepackage{nicematrix}",
    r"\providecommand{\ToolName}{\textbf{SensorRAG}}",
]


def _compile_to_png(standalone_tex: str, output_path: Path) -> None:
    """Compile a standalone LaTeX string and convert the resulting PDF to PNG."""
    import shutil
    import subprocess
    import tempfile

    pdflatex = shutil.which("pdflatex") or next(
        (p for p in _TEX_FALLBACKS if Path(p).exists()), None
    )
    if not pdflatex:
        print("[WARN] pdflatex not found — skipping PNG render. "
              "Install BasicTeX: brew install --cask basictex", file=sys.stderr)
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tex = tmp / "table.tex"
        tex.write_text(standalone_tex, encoding="utf-8")

        # nicematrix requires two passes
        for _pass in range(2):
            result = subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", str(tex)],
                cwd=tmp, capture_output=True, text=True,
            )
        pdf = tmp / "table.pdf"
        if not pdf.exists():
            tail = result.stdout[-800:] if result.stdout else result.stderr[-800:]
            print(f"[WARN] pdflatex failed:\n{tail}", file=sys.stderr)
            return

        pdftoppm = shutil.which("pdftoppm")
        if pdftoppm:
            base = str(tmp / "out")
            subprocess.run(
                [pdftoppm, "-r", "250", "-png", "-singlefile", str(pdf), base],
                check=True,
            )
            candidate = tmp / "out.png"
            if candidate.exists():
                import shutil as _sh
                _sh.copy(candidate, output_path)
                print(f"✅  PNG  saved: {output_path}")
                return

        try:
            from pdf2image import convert_from_path  # type: ignore
            imgs = convert_from_path(str(pdf), dpi=250)
            if imgs:
                imgs[0].save(str(output_path))
                print(f"✅  PNG  saved: {output_path}")
                return
        except ImportError:
            pass

        convert = shutil.which("convert")
        if convert:
            subprocess.run(
                [convert, "-density", "250", str(pdf), str(output_path)],
                check=True,
            )
            print(f"✅  PNG  saved: {output_path}")
            return

        print(
            "[WARN] No PDF→PNG converter found.\n"
            "       Install one of: pdftoppm (brew install poppler), "
            "pdf2image (pip install pdf2image), or ImageMagick.",
            file=sys.stderr,
        )


def generate_png_preview(
    datasets: List[str],
    random_baselines: Dict[str, Dict[str, Optional[float]]],
    summaries_by_dir: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    output_path: Path,
    label_level: str = "L1",
) -> None:
    """Compile the quant comparison table to PNG (exact Overleaf rendering)."""
    tabular = _build_comparison_tabular(datasets, random_baselines, summaries_by_dir)
    standalone = "\n".join([*_STANDALONE_PREAMBLE, r"\begin{document}", r"\small",
                             *tabular, r"\end{document}"])
    _compile_to_png(standalone, output_path)


# ── per-L1-label table (retrieval_results_quant_all_labels.tex / .png) ───────

# Reverse map: display name → json metric key (e.g. "P@10" → "10")
_METRIC_NAME_TO_KEY: Dict[str, str] = {name: key for key, name in METRICS}


def _build_all_labels_tabular(
    datasets: List[str],
    per_label: Dict[str, Dict[str, Dict[str, float]]],
    ordered_l1_labels: List[str],
) -> List[str]:
    """Return LaTeX lines from \\begin{NiceTabular} to \\end{NiceTabular}.

    Rows = L1 activity labels (one per row).
    Column groups = metrics (LATEX_METRIC_ORDER), sub-columns = datasets.
    No average column; ``--`` where a label is absent from a household.
    """
    _DOUBLE_AFTER = {"P@100", "mAP"}
    n_m  = len(LATEX_METRIC_ORDER)
    n_ds = len(datasets)
    ds_abbr = {ds: ds.capitalize()[:2] + "." for ds in datasets}

    # Column spec: l | (rrr separator)×5
    col_parts: List[str] = ["l|"]
    for i, metric in enumerate(LATEX_METRIC_ORDER):
        col_parts.append("r" * n_ds)
        if i < n_m - 1:
            col_parts.append("||" if metric in _DOUBLE_AFTER else "|")
    col_spec = "".join(col_parts)

    # \multicolumn headers
    mc_headers: List[str] = []
    for i, m in enumerate(LATEX_METRIC_ORDER):
        if i == n_m - 1:
            border = "c"
        elif m in _DOUBLE_AFTER:
            border = "c||"
        else:
            border = "c|"
        mc_headers.append(f"\\multicolumn{{{n_ds}}}{{{border}}}{{\\textbf{{{m}}}}}")

    # \cmidrule spans — columns start at 2 (label col = 1)
    cmidrules: List[str] = []
    for i in range(n_m):
        start = 2 + i * n_ds
        end   = start + n_ds - 1
        lr    = "lr" if i < n_m - 1 else "l"
        cmidrules.append(f"\\cmidrule({lr}){{{start}-{end}}}")

    sub_abbrs = [ds_abbr[ds] for ds in datasets] * n_m

    def tex_escape(s: str) -> str:
        return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    lines: List[str] = [
        f"\\begin{{NiceTabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join([""] + mc_headers) + " \\\\",
        "".join(cmidrules),
        "\\textbf{Activity} & " + " & ".join(sub_abbrs) + " \\\\",
        "\\midrule",
    ]

    for lbl_key in ordered_l1_labels:
        # lbl_key is "[L1] Label Name"
        display = tex_escape(lbl_key[4:].strip())
        cells = [display]
        for metric_name in LATEX_METRIC_ORDER:
            mk = _METRIC_NAME_TO_KEY[metric_name]
            for ds in datasets:
                val = per_label.get(ds, {}).get(lbl_key, {}).get(mk)
                cells.append(f"{val:.2f}" if val is not None else MISSING)
        lines.append(" & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{NiceTabular}"]
    return lines


def generate_all_labels_table(
    datasets: List[str],
    per_label: Dict[str, Dict[str, Dict[str, float]]],
    ordered_l1_labels: List[str],
    output_path: Path,
) -> None:
    """Write retrieval_results_quant_all_labels.tex — one row per L1 label, \\ToolName only."""
    tabular = _build_all_labels_tabular(datasets, per_label, ordered_l1_labels)
    lines: List[str] = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        *tabular,
        "}",
        (
            "\\caption{\\ToolName prototype-to-sensor retrieval per L1 activity label "
            "(macro-averaged). {\\bf --} = label not present in that household.}"
        ),
        "\\label{tab:retrieval_quant_all_labels}",
        "\\end{table*}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅  TEX  saved: {output_path}")


def generate_all_labels_png(
    datasets: List[str],
    per_label: Dict[str, Dict[str, Dict[str, float]]],
    ordered_l1_labels: List[str],
    output_path: Path,
) -> None:
    """Compile the per-L1-label table to PNG (exact Overleaf rendering)."""
    tabular   = _build_all_labels_tabular(datasets, per_label, ordered_l1_labels)
    standalone = "\n".join([
        *_STANDALONE_PREAMBLE,
        r"\begin{document}", r"\small",
        *tabular,
        r"\end{document}",
    ])
    _compile_to_png(standalone, output_path)


# ── qual table (retrieval_results_qual.tex / .png) ────────────────────────────

QUAL_METRICS: List[str] = ["P@10", "mAP@10", "MRR@10"]
_PLACEHOLDER = r"\textcolor{green!50!black}{\textit{x.xx}}"

QUAL_QUERIES: List[str] = [
    "Kitchen Activity",
    "Eating a Meal",
    "Working at a Desk",
    "Watching TV",
    "Using Bathroom at Night",
    "Using Kitchen at Night",
    "Wandering around the house during the day",
    "Wandering around the house during the night",
    "Sedentary Activity",
    "Sedentary Activity in the Morning",
]


def _build_qual_tabular(datasets: List[str]) -> List[str]:
    """Return LaTeX lines from \\begin{NiceTabular} to \\end{NiceTabular}."""
    sub_keys = datasets + ["average"]
    ds_abbr  = {ds: ds.capitalize()[:2] + "." for ds in datasets}
    ds_abbr["average"] = "Avg."

    n        = len(QUAL_METRICS)
    n_subcols = len(sub_keys)
    ph_row   = [_PLACEHOLDER] * (n * n_subcols)

    # Single | between metric groups; !{\vrule} for Ca./Avg. (interrupted)
    col_parts: List[str] = ["l|"]
    for i in range(n):
        col_parts.append("rrr!{\\vrule}r")
        if i < n - 1:
            col_parts.append("|")
    col_spec = "".join(col_parts)

    mc_headers = [
        f"\\multicolumn{{4}}{{{'c|' if i < n - 1 else 'c'}}}{{\\textbf{{{m}}}}}"
        for i, m in enumerate(QUAL_METRICS)
    ]
    cmidrules = [
        f"\\cmidrule({'lr' if i < n - 1 else 'l'}){{{2 + i * 4}-{5 + i * 4}}}"
        for i in range(n)
    ]
    sub_abbrs = [ds_abbr[sk] for sk in sub_keys]

    lines: List[str] = [
        f"\\begin{{NiceTabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join([""] + mc_headers) + " \\\\",
        "".join(cmidrules),
        "\\textbf{Query} & " + " & ".join(sub_abbrs * n) + " \\\\",
        "\\midrule",
    ]
    for query in QUAL_QUERIES:
        lines.append(" & ".join([query] + ph_row) + " \\\\")
    # Average row with heavier rule above it
    lines.append("\\specialrule{0.10em}{\\aboverulesep}{\\belowrulesep}")
    lines.append(" & ".join(["\\textbf{Average}"] + ph_row) + " \\\\")
    lines += ["\\bottomrule", "\\end{NiceTabular}"]
    return lines


def generate_qual_table(datasets: List[str], output_dir: Path) -> None:
    """Write retrieval_results_qual.tex and retrieval_results_qual.png."""
    tabular = _build_qual_tabular(datasets)

    tex_lines: List[str] = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        *tabular,
        "}",
        "\\caption{Qualitative retrieval results on CASAS datasets "
        "(\\textcolor{green!50!black}{\\textit{x.xx}} = placeholder).}",
        "\\label{tab:retrieval_qual}",
        "\\end{table*}",
    ]
    tex_path = output_dir / "retrieval_results_qual.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"✅  TEX  saved: {tex_path}")

    standalone = "\n".join([*_STANDALONE_PREAMBLE, r"\begin{document}", r"\small",
                             *tabular, r"\end{document}"])
    _compile_to_png(standalone, output_dir / "retrieval_results_qual.png")


# ── LaTeX per-metric table (not auto-saved; kept for on-demand use) ──────────

def table_to_latex(table: dict, datasets: List[str]) -> str:
    rows        = table["rows"]
    averages    = table["averages"]
    metric_name = table["metric_name"]
    dir_display = table["direction_display"]

    def fmt(v) -> str:
        return f"{v:.4f}" if v is not None else r"\textemdash"

    def tex_escape(s: str) -> str:
        return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    ds_cols  = " & ".join(r"\textbf{" + ds.capitalize() + "}" for ds in datasets)
    col_spec = "ll" + "r" * len(datasets)

    lines: List[str] = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{" + tex_escape(metric_name) + " — " + tex_escape(dir_display) + r" (macro)}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        r"\textbf{Level} & \textbf{Activity} & " + ds_cols + r" \\",
        r"\midrule",
    ]
    prev_level = None
    for r in rows:
        if r["level"] != prev_level and prev_level is not None:
            lines.append(r"\midrule")
        prev_level = r["level"]
        cells = " & ".join(fmt(r[ds]) for ds in datasets)
        lines.append(f"{r['level']} & {tex_escape(r['label'])} & {cells} \\\\")

    lines.append(r"\midrule")
    for level in ("L1", "L2", "all"):
        cells = " & ".join(fmt(averages.get(f"{ds}_{level}")) for ds in datasets)
        lines.append(r"\textbf{AVG (" + level + r")} & & " + cells + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate prototype retrieval metrics (p2s + p2t) across CASAS datasets."
    )
    parser.add_argument("--datasets",        nargs="+", default=["milan", "aruba", "cairo"])
    parser.add_argument("--model_suffix",    default="fd60_seq_rb1_textclip_projmlp_clipmlm_v3")
    parser.add_argument("--base_dir",        default="results/evals")
    parser.add_argument("--results_subdir",  default="FD_60_p")
    parser.add_argument("--output_dir",      default="results/evals/imwut_paper")
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_part = "all_casas" if set(args.datasets) >= {"milan", "aruba", "cairo"} else "_".join(args.datasets)
    vmatch  = re.search(r"_(v\d+)$", args.model_suffix)
    vtag    = vmatch.group(1) + "model" if vmatch else args.model_suffix

    print(f"\n{'='*70}")
    print(f"  CASAS Retrieval Aggregation")
    print(f"  Datasets  : {args.datasets}")
    print(f"  Directions: {[tag for _, tag, _ in DIRECTIONS]}")
    print(f"{'='*70}")

    # Accumulate L1 summaries for the LaTeX comparison table
    latex_summaries: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}

    # Stash p2s per-label data for the per-L1-label all_labels table
    p2s_per_label:       Dict[str, Dict[str, Dict[str, float]]] = {}
    p2s_ordered_l1:      List[str] = []

    # ── one output file per direction ─────────────────────────────────────────
    for dir_key, dir_tag, dir_display in DIRECTIONS:
        stem = f"{ds_part}_retrieval_{vtag}_{dir_tag}_all_labels"

        per_label, overall, ordered_labels = collect_data(
            datasets=args.datasets,
            model_suffix=args.model_suffix,
            base_dir=base_dir,
            results_subdir=args.results_subdir,
            direction=dir_key,
        )

        if dir_key == "prototype2sensor":
            p2s_per_label  = per_label
            p2s_ordered_l1 = [lbl for lbl in ordered_labels if lbl.startswith("[L1]")]

        all_tables: List[dict] = []
        for metric_key, metric_name in METRICS:
            table = build_metric_table(
                datasets=args.datasets,
                per_label=per_label,
                overall=overall,
                ordered_labels=ordered_labels,
                metric_key=metric_key,
                metric_name=metric_name,
                direction=dir_key,
                direction_display=dir_display,
            )
            all_tables.append(table)

        # terminal: summary only
        print_summary(all_tables, args.datasets, dir_display)

        # markdown: summary then per-label detail
        md_sections: List[str] = [
            f"# CASAS Retrieval Metrics — {dir_display} (macro)\n",
            f"**Datasets**: {', '.join(args.datasets)}  ",
            f"**Model suffix**: `{args.model_suffix}`  ",
            f"**Direction**: `{dir_key}`\n",
            "---\n",
            build_summary_md(all_tables, args.datasets, dir_display),
            "## Per-label Detail\n",
        ]
        for table in all_tables:
            md_sections.append(table_to_markdown(table, args.datasets))
            md_sections.append("---\n")

        # summary dict for JSON
        summary: Dict[str, Dict[str, Dict]] = {}
        for level_key in ("L1", "L2", "all"):
            level_rows: Dict[str, Dict] = {}
            for t in all_tables:
                ds_vals = {ds: t["averages"].get(f"{ds}_{level_key}") for ds in args.datasets}
                level_rows[t["metric_name"]] = {
                    **ds_vals,
                    "average": _ds_avg(list(ds_vals.values())),
                }
            summary[level_key] = level_rows

        json_path = output_dir / f"{stem}.json"
        with open(json_path, "w") as f:
            json.dump({
                "stem":       stem,
                "direction":  dir_key,
                "label_norm": LABEL_NORM,
                "summary":    summary,
                "tables":     all_tables,
            }, f, indent=2)
        print(f"\n✅  JSON  saved: {json_path}")

        md_path = output_dir / f"{stem}.md"
        md_path.write_text("\n".join(md_sections), encoding="utf-8")
        print(f"✅  MD   saved: {md_path}")

        # Stash L1 summary for the LaTeX comparison table
        latex_summaries[dir_key] = summary["L1"]

    # ── LaTeX + HTML comparison table (all directions collected) ─────────────
    random_baselines = compute_random_baselines(args.datasets)

    generate_latex_comparison_table(
        datasets=args.datasets,
        random_baselines=random_baselines,
        summaries_by_dir=latex_summaries,
        output_path=output_dir / "retrieval_results_quant.tex",
        label_level="L1",
    )
    generate_png_preview(
        datasets=args.datasets,
        random_baselines=random_baselines,
        summaries_by_dir=latex_summaries,
        output_path=output_dir / "retrieval_results_quant.png",
        label_level="L1",
    )
    generate_qual_table(args.datasets, output_dir)

    # ── per-L1-label all_labels table ─────────────────────────────────────────
    generate_all_labels_table(
        datasets=args.datasets,
        per_label=p2s_per_label,
        ordered_l1_labels=p2s_ordered_l1,
        output_path=output_dir / "retrieval_results_quant_all_labels.tex",
    )
    generate_all_labels_png(
        datasets=args.datasets,
        per_label=p2s_per_label,
        ordered_l1_labels=p2s_ordered_l1,
        output_path=output_dir / "retrieval_results_quant_all_labels.png",
    )


if __name__ == "__main__":
    main()
