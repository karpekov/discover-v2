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
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── constants ─────────────────────────────────────────────────────────────────

# Directions to analyse: (json_key, short_tag, display_name)
DIRECTIONS: List[Tuple[str, str, str]] = [
    ("prototype2sensor",      "p2s",        "Prototype → Sensor"),
    ("prototype2text_noproj", "p2t_noproj", "Prototype → Text (pure CLIP)"),
    ("prototype2text",        "p2t",        "Prototype → Text (CLIP + learned projection)"),
]

# Pre-specified significance test: \ToolName vs the pure-CLIP baseline. mAP and
# P@100 were fixed in advance; the rest are only shown as stars in the table.
SIG_TOOL_DIR     = "prototype2sensor"
SIG_BASELINE_DIR = "prototype2text_noproj"
SIG_METRICS: List[Tuple[str, str]] = [
    ("map", "primary"), ("100", "secondary"),
    ("10", "exploratory"), ("50", "exploratory"), ("mrr", "exploratory"),
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

# Metric display order for the comparison table (left → right).
LATEX_METRIC_ORDER: List[str] = ["mAP", "P@10", "P@50", "P@100", "MRR"]

# Maps direction json_key → row label in the LaTeX table.
DIRECTION_ROW_LABELS: Dict[str, str] = {
    "prototype2text_noproj": "Text Only: Pure CLIP",
    "prototype2text":        "Text Only: CLIP + Learned Projection",
    "prototype2sensor":      "\\ToolName",
}


def load_l1_label_counts(
    ds: str, model_suffix: str, base_dir: Path, results_subdir: str
) -> Tuple[Dict[str, int], List[str]]:
    """Return (window count per L1 label over the whole retrieval index, query labels)."""
    run_dir = base_dir / ds / results_subdir / f"{ds}_{model_suffix}"
    index_counts = load_json(run_dir / "retrieval" / "retrieval_results_l1.json")["label_counts"]
    query_labels = list(
        load_json(run_dir / "comprehensive_results.json")["retrieval_metrics"]["L1"]
        ["prototype_based"]["per_label"]["prototype2sensor"]["count"]
    )
    return {k: int(v) for k, v in index_counts.items()}, query_labels


def expected_random_metrics(n: int, M: int) -> Dict[str, float]:
    """Exact expected P@k, AP and RR for one query under a uniformly random ranking
    of M windows of which n are relevant.

    P@k (incl. k capped at n) = n/M.
    RR: P(first relevant at rank r) = C(M-r, n-1) / C(M, n).
    AP: H_M/M + (n-1)(M-H_M) / (M(M-1)).
    """
    log_c_mn = math.lgamma(M + 1) - math.lgamma(n + 1) - math.lgamma(M - n + 1)
    rr = sum(
        math.exp(math.lgamma(M - r + 1) - math.lgamma(n) - math.lgamma(M - r - n + 2) - log_c_mn) / r
        for r in range(1, M - n + 2)
    )
    h_m = sum(1.0 / i for i in range(1, M + 1))
    ap = h_m / M + ((n - 1) * (M - h_m) / (M * (M - 1)) if M > 1 else 0.0)
    return {"P@k": n / M, "mAP": ap, "MRR": rr}


def compute_random_baselines(
    datasets: List[str],
    label_counts: Dict[str, Tuple[Dict[str, int], List[str]]],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Return {metric_name: {dataset: value, 'average': value}} for a uniformly random
    ranking of each dataset's retrieval index, macro-averaged over query labels."""
    per_ds: Dict[str, Dict[str, float]] = {}
    for ds in datasets:
        if ds not in label_counts:
            continue
        index_counts, query_labels = label_counts[ds]
        M = sum(index_counts.values())
        per_label = [expected_random_metrics(index_counts[l], M) for l in query_labels]
        per_ds[ds] = {m: sum(p[m] for p in per_label) / len(per_label) for m in ("P@k", "mAP", "MRR")}

    rows: Dict[str, Dict[str, Optional[float]]] = {}
    for metric_name in LATEX_METRIC_ORDER:
        key = "P@k" if metric_name.startswith("P@") else metric_name
        row: Dict[str, Optional[float]] = {
            ds: round(per_ds[ds][key], 4) if ds in per_ds else None for ds in datasets
        }
        present = [v for v in row.values() if v is not None]
        row["average"] = round(sum(present) / len(present), 4) if present else None
        rows[metric_name] = row
    return rows


def wilcoxon_signed_rank_exact(x: List[float], y: List[float]) -> Dict[str, Optional[float]]:
    """Two-sided exact Wilcoxon signed-rank test of x vs y (paired).

    Zero differences are dropped (Wilcoxon's method); tied |d| get average ranks and
    the null distribution of W+ is enumerated exactly over all 2^n sign assignments
    of the observed ranks.
    """
    d = [round(a - b, 10) for a, b in zip(x, y)]
    nz = [v for v in d if v != 0.0]
    n = len(nz)
    res: Dict[str, Optional[float]] = {
        "n_labels": len(d),
        "n_nonzero": n,
        "wins": sum(v > 0 for v in d),
        "losses": sum(v < 0 for v in d),
        "ties": sum(v == 0.0 for v in d),
        "mean_diff": sum(d) / len(d) if d else None,
        "median_diff": sorted(d)[len(d) // 2] if len(d) % 2 else
                       (sorted(d)[len(d) // 2 - 1] + sorted(d)[len(d) // 2]) / 2 if d else None,
        "w_plus": None, "w_minus": None, "p_value": None,
    }
    if n == 0:
        res["p_value"] = 1.0
        return res

    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks2 = [0] * n  # doubled average ranks → integers
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks2[order[k]] = i + j + 2  # 2 * mean(i+1 .. j+1)
        i = j + 1

    total2 = sum(ranks2)
    w2_obs = sum(r for r, v in zip(ranks2, nz) if v > 0)
    counts = [0] * (total2 + 1)
    counts[0] = 1
    for r in ranks2:
        for s in range(total2, r - 1, -1):
            counts[s] += counts[s - r]
    dev_obs = abs(2 * w2_obs - total2)
    extreme = sum(c for s, c in enumerate(counts) if abs(2 * s - total2) >= dev_obs)
    res["w_plus"] = w2_obs / 2
    res["w_minus"] = (total2 - w2_obs) / 2
    res["p_value"] = min(1.0, extreme / 2 ** n)
    return res


def compute_significance(
    datasets: List[str],
    per_label_by_dir: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
) -> Dict:
    """Pre-specified test: SIG_TOOL_DIR vs SIG_BASELINE_DIR over L1 labels, per home and pooled."""
    tool, base = per_label_by_dir.get(SIG_TOOL_DIR, {}), per_label_by_dir.get(SIG_BASELINE_DIR, {})
    out: Dict = {
        "comparison": f"{DIRECTION_ROW_LABELS[SIG_TOOL_DIR]} vs {DIRECTION_ROW_LABELS[SIG_BASELINE_DIR]}",
        "directions": [SIG_TOOL_DIR, SIG_BASELINE_DIR],
        "test": "two-sided exact Wilcoxon signed-rank over L1 labels (zero differences dropped, "
                "ties given average ranks, exact null enumeration)",
        "label_level": "L1",
        "metrics": {},
    }
    for mk, role in SIG_METRICS:
        mname = dict(METRICS)[mk]
        entry: Dict = {"role": role, "per_home": {}, "pooled": None}
        pooled_x: List[float] = []
        pooled_y: List[float] = []
        for ds in datasets:
            labels = sorted(
                l for l in tool.get(ds, {})
                if l.startswith("[L1]") and mk in tool[ds][l] and mk in base.get(ds, {}).get(l, {})
            )
            x = [tool[ds][l][mk] for l in labels]
            y = [base[ds][l][mk] for l in labels]
            if not labels:
                continue
            r = wilcoxon_signed_rank_exact(x, y)
            r["per_label"] = {l[4:].strip(): {"tool": a, "baseline": b} for l, a, b in zip(labels, x, y)}
            entry["per_home"][ds] = r
            pooled_x += x
            pooled_y += y
        entry["pooled"] = wilcoxon_signed_rank_exact(pooled_x, pooled_y) if pooled_x else None
        out["metrics"][mname] = entry
    return out


def significance_to_markdown(sig: Dict, datasets: List[str]) -> str:
    lines = [
        f"## Significance — {sig['comparison']}",
        "",
        f"> {sig['test']}. Pre-specified: mAP primary, P@100 secondary; P@10, P@50 and MRR exploratory. "
        "Per home and pooled across homes.",
        "",
        "| Metric | Scope | Labels | Wins / Losses / Ties | Mean Δ | Median Δ | W+ | W− | p |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mname, entry in sig["metrics"].items():
        scopes = [(ds.capitalize(), entry["per_home"].get(ds)) for ds in datasets] + [("Pooled", entry["pooled"])]
        for scope, r in scopes:
            if not r:
                continue
            lines.append(
                f"| {mname} ({entry['role']}) | {scope} | {r['n_labels']} | "
                f"{r['wins']} / {r['losses']} / {r['ties']} | {r['mean_diff']:+.3f} | {r['median_diff']:+.3f} | "
                f"{r['w_plus'] if r['w_plus'] is not None else MISSING} | "
                f"{r['w_minus'] if r['w_minus'] is not None else MISSING} | {r['p_value']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


# Stars that must come out of the saved Wilcoxon tests (SensorQ vs Pure CLIP).
# * p<0.05, ** p<0.01. No p<0.001 and no p<0.1 marks occur.
_EXPECTED_STARS: Dict[Tuple[str, str], str] = {
    ("mAP", "average"): "**",
    ("mAP", "milan"): "*",
    ("mAP", "cairo"): "**",
    ("P@50", "average"): "*",
    ("P@50", "cairo"): "*",
    ("P@100", "average"): "**",
    ("P@100", "cairo"): "**",
}
_WILCOXON_JSON = (
    Path(__file__).resolve().parents[3]
    / "results/evals/imwut_paper/retrieval_v3/wilcoxon_vs_pure_clip.json"
)


def _star_mark(p: Optional[float]) -> str:
    if p is None:
        return ""
    if p < 0.001:
        raise SystemExit(f"Wilcoxon p={p} is below 0.001; the table has no *** level")
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _sig_record(significance: Dict, metric: str, sk: str) -> Optional[Dict]:
    entry = significance.get("metrics", {}).get(metric, {})
    if sk == "average":
        return entry.get("pooled")
    return entry.get("per_home", {}).get(sk)


def _assert_significance(significance: Dict, sub_keys: List[str]) -> Dict[Tuple[str, str], str]:
    """Stars for SensorQ vs Pure CLIP. Stops if the saved tests disagree with the table."""
    if _WILCOXON_JSON.exists():
        saved = json.loads(_WILCOXON_JSON.read_text())
        for metric in ("mAP", "P@100"):
            for scope in ("milan", "aruba", "cairo", "pooled"):
                sk = "average" if scope == "pooled" else scope
                rec = _sig_record(significance, metric, sk)
                got = None if not rec else rec.get("p_value")
                exp = saved["results"][metric][scope]["p"]
                if got is None or abs(float(got) - float(exp)) > 1e-9:
                    raise SystemExit(
                        f"Wilcoxon JSON {metric} {scope} p={exp} but table source has p={got}"
                    )
    marks: Dict[Tuple[str, str], str] = {}
    for metric in LATEX_METRIC_ORDER:
        for sk in sub_keys:
            rec = _sig_record(significance, metric, sk)
            p = None if not rec else rec.get("p_value")
            mark = _star_mark(None if p is None else float(p))
            if mark and rec["w_plus"] < rec["w_minus"]:
                raise SystemExit(
                    f"{metric} {sk}: p={p} favors Pure CLIP (W+={rec['w_plus']}, W-={rec['w_minus']}). "
                    "Stopping; no star was written."
                )
            marks[(metric, sk)] = mark
    if marks != {**{k: "" for k in marks}, **_EXPECTED_STARS}:
        bad = {k: marks[k] for k in marks if marks[k] != _EXPECTED_STARS.get(k, "")}
        raise SystemExit(f"Significance marks disagree with the current table: {bad}")
    return marks


def _fmt_value(value: Optional[float], *, bold: bool, grey: bool, stars: str) -> str:
    if value is None:
        text = MISSING
    else:
        text = f"{value:.2f}"
        if bold:
            text = f"\\textbf{{{text}}}"
    if stars == "**":
        # Right-aligned cell: the double star is wider than the right padding and
        # crosses the rule. Pull the number left into the existing left padding and
        # give that width back on the right, so the zero-width star stays inside
        # the cell and the column does not grow. \hspace* so the leading skip is
        # not discarded at the start of the entry.
        text = f"\\hspace*{{-3pt}}{text}\\rlap{{$^{{{stars}}}$}}\\hspace{{3pt}}"
    elif stars:
        text += f"\\rlap{{$^{{{stars}}}$}}"
    if grey:
        text = f"{{\\color{{black!50}}{text}}}"
    return text


def _build_comparison_tabular(
    datasets: List[str],
    random_baselines: Dict[str, Dict[str, Optional[float]]],
    summaries_by_dir: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    significance: Optional[Dict] = None,
) -> List[str]:
    """Return LaTeX lines from \\begin{tabular} to \\end{tabular} (inclusive)."""
    sub_keys = ["average"] + datasets
    ds_abbr  = {ds: ds.capitalize()[:2] + "." for ds in datasets}
    ds_abbr["average"] = "Avg."

    # grey rows are reference points and are never bold. Bold is Pure CLIP vs \ToolName only.
    clip = summaries_by_dir.get("prototype2text_noproj", {})
    proj = summaries_by_dir.get("prototype2text", {})
    ours = summaries_by_dir.get("prototype2sensor", {})
    row_defs: List[Tuple[str, str, Dict]] = [
        ("grey", "Random (chance)", random_baselines),
        ("base", "Text only (Pure CLIP)", clip),
        ("grey", r"\quad$\hookrightarrow$ + learned projection", proj),
        ("ours", r"\textbf{\ToolName} (ours)", ours),
    ]
    stars = _assert_significance(significance, sub_keys) if significance else {}

    def _bold(metric: str, sk: str, role: str) -> bool:
        if role == "grey":
            return False
        a, b = clip.get(metric, {}).get(sk), ours.get(metric, {}).get(sk)
        if a is None or b is None:
            return False
        sa, sb = f"{a:.2f}", f"{b:.2f}"
        return sa >= sb if role == "base" else sb >= sa

    # Metrics after which a double rule separates the next group
    _DOUBLE_AFTER = {"P@100", "mAP"}

    n = len(LATEX_METRIC_ORDER)

    # Column spec: l | r|rrr | r|rrr | r|rrr || r|rrr || r|rrr
    #   - r|rrr  : shaded Avg. first, single rule before the homes in every group
    #   - |  / || : single / double rule between metric groups
    col_parts: List[str] = ["l|"]
    for i, metric in enumerate(LATEX_METRIC_ORDER):
        # !{\vrule} is drawn traditionally (interrupted by hlines); | is drawn by
        # nicematrix via TikZ (uninterrupted). Use !{\vrule} for the Avg./Mi. separator.
        col_parts.append("r!{\\vrule}rrr")
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

    # NiceTabular ignores \columncolor in the preamble, so shade in \CodeBefore,
    # from the sub-header row (row 2) down, leaving the metric-name row white.
    n_rows = 2 + len(row_defs)
    avg_shading = " ".join(
        f"\\rectanglecolor{{black!5}}{{2-{2 + i * 4}}}{{{n_rows}-{2 + i * 4}}}" for i in range(n)
    )
    lines: List[str] = [
        f"\\begin{{NiceTabular}}{{{col_spec}}}",
        f"\\CodeBefore {avg_shading} \\Body",
        "\\toprule",
        " & ".join([""] + mc_headers) + " \\\\",
        "".join(hdr_cmidrules),
        "\\textbf{Method} & " + " & ".join(sub_abbrs * n) + " \\\\",
        "\\midrule",
    ]
    last_idx = len(row_defs) - 1
    for i, (role, label, data) in enumerate(row_defs):
        shown = f"{{\\color{{black!50}}{label}}}" if role == "grey" else label
        cells = [shown]
        for metric in LATEX_METRIC_ORDER:
            for sk in sub_keys:
                mark = stars.get((metric, sk), "") if role == "ours" else ""
                cells.append(_fmt_value(
                    data.get(metric, {}).get(sk),
                    bold=_bold(metric, sk, role),
                    grey=role == "grey",
                    stars=mark,
                ))
        lines.append(" & ".join(cells) + " \\\\")
        if i == 0:
            lines.append("\\midrule")
        elif i == last_idx - 1:
            # Heavier rule above the \ToolName row (same weight as \toprule/\bottomrule)
            lines.append("\\specialrule{0.10em}{\\aboverulesep}{\\belowrulesep}")
    lines += ["\\bottomrule", "\\end{NiceTabular}"]
    return lines


def generate_latex_comparison_table(
    datasets: List[str],
    random_baselines: Dict[str, Dict[str, Optional[float]]],
    summaries_by_dir: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    output_path: Path,
    label_level: str = "L1",
    significance: Optional[Dict] = None,
) -> None:
    """Write the LaTeX comparison table wrapped in table* for an ACM paper."""
    tabular = _build_comparison_tabular(datasets, random_baselines, summaries_by_dir, significance)
    lines: List[str] = [
        "% requires nicematrix, booktabs, xcolor",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Text-to-sensor retrieval on CASAS test windows: mean Average Precision (mAP), Precision at $k$ (P@$k$), and Mean Reciprocal Rank (MRR), macro-averaged over",
        "activities, for the Milan (Mi.), Aruba (Ar.), and Cairo (Ca.) homes and their mean (Avg., shaded).",
        "Pure CLIP caption embeddings are the main baseline; Random gives a lower bound, and CLIP + learned projection row shows the effect of adding \\ToolName{}'s text projection to the baseline.",
        "On average over homes, \\ToolName{} outperforms Pure CLIP on every metric except MRR.",
        "Bold marks show the best result between the baseline and \\ToolName{} in each column.",
        "$^{*}$ ($p<0.05$) and $^{**}$ ($p<0.01$) mark significant improvements over the baseline",
        "(two-sided exact Wilcoxon signed-rank test over activities); no metric favors Pure CLIP significantly.}",
        "\\label{tab:retrieval_quant}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        *tabular,
        "}",
    ]
    lines.append("\\end{table*}")
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
    r"\providecommand{\ToolName}{\textbf{SensorQ}}",
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
    significance: Optional[Dict] = None,
) -> None:
    """Compile the quant comparison table to PNG (exact Overleaf rendering)."""
    tabular = _build_comparison_tabular(datasets, random_baselines, summaries_by_dir, significance)
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
        (
            "\\caption{\\ToolName{} text-to-sensor retrieval per annotated activity, with one natural-language "
            "query per activity. $n$ is the number of relevant test windows; {\\bf --} marks an activity not "
            "present in that home.}"
        ),
        "\\label{tab:retrieval_quant_all_labels}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        *tabular,
        "}",
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

# Path where the human-rated retrieval CSV lives
_QUAL_CSV_PATH = Path("results/evals/imwut_paper/custom_query_retrieval/retrieved_samples_rated.csv")


def load_qual_ratings(csv_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load retrieved_samples_rated.csv and compute P@10, mAP@10, MRR@10.

    For queries that have multiple long-query variants (q1, q2, …) the metrics
    are averaged across variants before being stored per (house, query).

    Returns
    -------
    metrics[house][query][metric_name] = float
    """
    import csv as _csv
    from collections import defaultdict

    rows: List[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            rows.append(row)

    # Group by (house, custom_query, long_query_idx) → sorted list of (rank, rel)
    groups: Dict[Tuple, List[Tuple[int, int]]] = defaultdict(list)
    for row in rows:
        r1 = row.get("R1", "").strip()
        if r1 not in ("0", "1"):          # skip unrated rows
            continue
        key = (row["house"], row["custom_query"], row["long_query_idx"])
        groups[key].append((int(row["rank"]), int(r1)))

    # Compute per-variant metrics
    variant_metrics: Dict[Tuple, Dict[str, float]] = {}
    for key, results in groups.items():
        results.sort()                    # sort by rank ascending
        rel = [r for _, r in results]    # 0/1 list in rank order

        # P@10
        p10 = sum(rel) / len(rel) if rel else 0.0

        # MRR@10 — reciprocal rank of first relevant result
        mrr = 0.0
        for k_idx, r in enumerate(rel):
            if r == 1:
                mrr = 1.0 / (k_idx + 1)
                break

        # mAP@10 — average precision capped at cutoff
        n_rel = sum(rel)
        if n_rel == 0:
            ap = 0.0
        else:
            running, ap_sum = 0, 0.0
            for k, r in enumerate(rel, start=1):
                if r == 1:
                    running += 1
                    ap_sum  += running / k
            ap = ap_sum / n_rel

        variant_metrics[key] = {"P@10": p10, "mAP@10": ap, "MRR@10": mrr}

    # Average variant metrics over long_query_idx for each (house, query)
    query_lists: Dict[Tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (house, query, _qidx), mdict in variant_metrics.items():
        for metric, val in mdict.items():
            query_lists[(house, query)][metric].append(val)

    result: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (house, query), mdict in query_lists.items():
        result[house][query] = {m: sum(vs) / len(vs) for m, vs in mdict.items()}

    return dict(result)


def _fmt_qual(val: Optional[float]) -> str:
    """Format a qual metric value for LaTeX (two decimal places, dash if absent)."""
    return f"{val:.2f}" if val is not None else r"\textemdash"


def _build_qual_tabular(
    datasets: List[str],
    qual_data: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> List[str]:
    """Return LaTeX lines from \\begin{NiceTabular} to \\end{NiceTabular}.

    If *qual_data* is provided (house → query → metric → float) the actual
    computed values are inserted; otherwise the green placeholder is used.
    """
    sub_keys  = datasets + ["average"]
    ds_abbr   = {ds: ds.capitalize()[:2] + "." for ds in datasets}
    ds_abbr["average"] = "Avg."
    n         = len(QUAL_METRICS)

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

    # ── per-query rows ────────────────────────────────────────────────────────
    for query in QUAL_QUERIES:
        cells: List[str] = []
        for metric in QUAL_METRICS:
            ds_vals: List[Optional[float]] = []
            for ds in datasets:
                v = qual_data.get(ds, {}).get(query, {}).get(metric) if qual_data else None
                ds_vals.append(v)
                cells.append(_fmt_qual(v))
            # per-metric row average across datasets
            valid = [v for v in ds_vals if v is not None]
            cells.append(_fmt_qual(sum(valid) / len(valid) if valid else None))
        lines.append(" & ".join([query] + cells) + " \\\\")

    # ── macro-average row ─────────────────────────────────────────────────────
    avg_cells: List[str] = []
    for metric in QUAL_METRICS:
        ds_avgs: List[Optional[float]] = []
        for ds in datasets:
            if qual_data and ds in qual_data:
                vals = [
                    qual_data[ds][q][metric]
                    for q in QUAL_QUERIES
                    if q in qual_data[ds] and metric in qual_data[ds][q]
                ]
                ds_avgs.append(sum(vals) / len(vals) if vals else None)
            else:
                ds_avgs.append(None)
            avg_cells.append(_fmt_qual(ds_avgs[-1]))
        valid = [v for v in ds_avgs if v is not None]
        avg_cells.append(_fmt_qual(sum(valid) / len(valid) if valid else None))

    lines.append("\\specialrule{0.10em}{\\aboverulesep}{\\belowrulesep}")
    lines.append(" & ".join(["\\textbf{Average}"] + avg_cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{NiceTabular}"]
    return lines


def generate_qual_table(datasets: List[str], output_dir: Path) -> None:
    """Write retrieval_results_qual.tex and retrieval_results_qual.png.

    If *retrieved_samples_rated.csv* exists in the custom_query_retrieval
    output folder, real P@10 / mAP@10 / MRR@10 values are computed from it
    and embedded; otherwise green placeholders are used.
    """
    qual_data: Optional[Dict] = None
    csv_path = _QUAL_CSV_PATH
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path

    if csv_path.exists():
        try:
            qual_data = load_qual_ratings(csv_path)
            print(f"✅  Qual ratings loaded from {csv_path}")
        except Exception as exc:
            print(f"[WARN] Could not load qual ratings: {exc}", file=sys.stderr)

    has_data  = qual_data is not None
    tabular   = _build_qual_tabular(datasets, qual_data)
    caption   = (
        "Qualitative retrieval results on CASAS datasets. "
        "P@10, mAP@10, and MRR@10 are computed from human-rated top-10 retrievals "
        "and averaged over long-query variants per short query. "
        "A dash (---) indicates that the activity was not present in that household."
        if has_data else
        "Qualitative retrieval results on CASAS datasets "
        "(\\textcolor{green!50!black}{\\textit{x.xx}} = placeholder)."
    )

    tex_lines: List[str] = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:retrieval_qual}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        *tabular,
        "}",
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
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{" + tex_escape(metric_name) + " — " + tex_escape(dir_display) + r" (macro)}",
        r"\label{tab:" + metric_name.lower().replace(" ", "_") + r"}",
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
    per_label_by_dir:    Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

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

        per_label_by_dir[dir_key] = per_label
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
    label_counts: Dict[str, Tuple[Dict[str, int], List[str]]] = {}
    for ds in args.datasets:
        try:
            label_counts[ds] = load_l1_label_counts(ds, args.model_suffix, base_dir, args.results_subdir)
        except (FileNotFoundError, KeyError) as exc:
            print(f"[WARN] No label counts for random baseline ({ds}): {exc}", file=sys.stderr)
    random_baselines = compute_random_baselines(args.datasets, label_counts)
    significance = compute_significance(args.datasets, per_label_by_dir)

    generate_latex_comparison_table(
        datasets=args.datasets,
        random_baselines=random_baselines,
        summaries_by_dir=latex_summaries,
        output_path=output_dir / "retrieval_results_quant.tex",
        label_level="L1",
        significance=significance,
    )
    generate_png_preview(
        datasets=args.datasets,
        random_baselines=random_baselines,
        summaries_by_dir=latex_summaries,
        output_path=output_dir / "retrieval_results_quant.png",
        label_level="L1",
        significance=significance,
    )
    generate_qual_table(args.datasets, output_dir)

    # ── combined quant results + significance (retrieval_results_quant.{json,md}) ──
    rows = {"Random": random_baselines}
    rows.update({DIRECTION_ROW_LABELS[d]: latex_summaries.get(d, {}) for d, _, _ in DIRECTIONS})
    l1_per_label = {
        DIRECTION_ROW_LABELS[d]: {
            ds: {lbl[4:].strip(): v for lbl, v in per_label_by_dir.get(d, {}).get(ds, {}).items()
                 if lbl.startswith("[L1]")}
            for ds in args.datasets
        }
        for d, _, _ in DIRECTIONS
    }
    quant_json = output_dir / "retrieval_results_quant.json"
    with open(quant_json, "w") as f:
        json.dump({
            "model_suffix":   args.model_suffix,
            "results_subdir": args.results_subdir,
            "label_level":    "L1",
            "aggregation":    "macro over L1 labels per home; 'average' = mean over homes",
            "index_size":     {ds: sum(c.values()) for ds, (c, _) in label_counts.items()},
            "rows":           rows,
            "per_label":      l1_per_label,
            "significance":   significance,
        }, f, indent=2)
    print(f"✅  JSON  saved: {quant_json}")

    md_lines = [
        "# Retrieval Results (L1, macro)",
        "",
        f"**Model suffix**: `{args.model_suffix}` · **Data**: `{args.results_subdir}` test split · "
        f"**Index size**: " + ", ".join(f"{ds} {sum(c.values())}" for ds, (c, _) in label_counts.items()),
        "",
        "| Method | " + " | ".join(f"{m} {ds[:2].capitalize()}." for m in LATEX_METRIC_ORDER
                                   for ds in args.datasets + ["avg"]) + " |",
        "| --- | " + " | ".join(["---"] * len(LATEX_METRIC_ORDER) * (len(args.datasets) + 1)) + " |",
    ]
    for name, data in rows.items():
        cells = [
            f"{v:.2f}" if (v := data.get(m, {}).get(sk)) is not None else MISSING
            for m in LATEX_METRIC_ORDER for sk in args.datasets + ["average"]
        ]
        md_lines.append(f"| {name} | " + " | ".join(cells) + " |")
    md_lines += ["", significance_to_markdown(significance, args.datasets)]
    quant_md = output_dir / "retrieval_results_quant.md"
    quant_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"✅  MD   saved: {quant_md}")

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
