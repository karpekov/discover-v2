"""Wilcoxon table (tab:retrieval_wilcoxon): \\ToolName vs Pure CLIP, per activity.

Reads the same per-home comprehensive_results.json files that retrieval_metrics_agg.py
turns into tab:retrieval_quant (no retrieval is recomputed) and reruns only the tests:
two-sided Wilcoxon signed-rank over L1 activities on mAP and P@100, per home and pooled.

Exact p-values come from scipy.stats.wilcoxon(method="exact", zero_method="wilcox").
scipy's exact null assumes untied ranks, so when nonzero |differences| tie the p-value
comes from exact enumeration over the tied ranks instead (both are saved to the JSON).
Differences are rounded to 1e-10 first so float noise does not split equal differences.

Fails if the per-activity values disagree with retrieval_results_quant.json or if the
pooled p-values differ from the ones behind the stars in tab:retrieval_quant.

Usage
-----
  python src/evals/imwut_paper/retrieval_wilcoxon_table.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from scipy.stats import wilcoxon

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.evals.imwut_paper.retrieval_metrics_agg import (  # noqa: E402
    DIRECTION_ROW_LABELS,
    SIG_BASELINE_DIR,
    SIG_TOOL_DIR,
    _STANDALONE_PREAMBLE,
    _compile_to_png,
    collect_data,
    wilcoxon_signed_rank_exact,
)

METRICS = [("map", "mAP"), ("100", "P@100")]
HOMES = ["milan", "aruba", "cairo"]
ROUND_DECIMALS = 10
P_RTOL = 1e-9


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_info() -> Dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=_PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def run_test(x: List[float], y: List[float]) -> Dict[str, object]:
    d = [round(a - b, ROUND_DECIMALS) for a, b in zip(x, y)]
    nz = [v for v in d if v != 0.0]
    n = len(nz)
    res: Dict[str, object] = {
        "n_activities": len(d),
        "n_nonzero": n,
        "wins": sum(v > 0 for v in d),
        "losses": sum(v < 0 for v in d),
        "ties": sum(v == 0.0 for v in d),
        "mean_diff": sum(d) / len(d),
        "W": None, "W_plus": None, "W_minus": None,
        "p": None, "p_scipy_exact": None, "p_method": None,
        "rank_ties": None, "min_attainable_p": None, "at_min_attainable_p": False,
    }
    if n == 0:
        res.update(p=1.0, p_method="no nonzero differences")
        return res

    sp = wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="exact")
    enum = wilcoxon_signed_rank_exact(x, y)
    rank_ties = len({abs(v) for v in nz}) < n
    w = min(enum["w_plus"], enum["w_minus"])
    if not math.isclose(float(sp.statistic), w, abs_tol=1e-9):
        raise AssertionError(f"W mismatch: scipy {sp.statistic} vs enumeration {w}")
    if not rank_ties and not math.isclose(float(sp.pvalue), enum["p_value"], rel_tol=P_RTOL):
        raise AssertionError(f"untied p mismatch: scipy {sp.pvalue} vs enumeration {enum['p_value']}")

    p = enum["p_value"] if rank_ties else float(sp.pvalue)
    # Only the all-positive and all-negative sign patterns reach the extreme W, with or without tied ranks.
    min_p = min(1.0, 2.0 / 2 ** n)
    res.update(
        W=w, W_plus=enum["w_plus"], W_minus=enum["w_minus"],
        p=p, p_scipy_exact=float(sp.pvalue),
        p_method="exact enumeration over tied ranks" if rank_ties else "scipy.stats.wilcoxon exact",
        rank_ties=rank_ties, min_attainable_p=min_p,
        at_min_attainable_p=math.isclose(p, min_p, rel_tol=P_RTOL),
    )
    return res


def fmt_p(r: Dict[str, object]) -> str:
    p = r["p"]
    s = "$<$0.001" if p < 0.001 else f"{p:#.3g}"
    return s + ("$^\\dagger$" if r["at_min_attainable_p"] else "")


def fmt_wlt(r: Dict[str, object]) -> str:
    return f"{r['wins']} / {r['losses']} / {r['ties']}"


def build_tabular(results: Dict[str, Dict[str, Dict]]) -> List[str]:
    def row(name: str, key: str) -> str:
        m, p100 = results["mAP"][key], results["P@100"][key]
        return (f"{name} & {m['n_activities']} & {fmt_wlt(m)} & {fmt_p(m)} & & "
                f"{fmt_wlt(p100)} & {fmt_p(p100)} \\\\")

    return [
        "\\begin{tabular}{lrrrcrr}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{mAP} & & \\multicolumn{2}{c}{P@100} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){6-7}",
        "Home & Activities & W / L / T & $p$ & & W / L / T & $p$ \\\\",
        "\\midrule",
        *[row(h.capitalize(), h) for h in HOMES],
        "\\midrule",
        row("Pooled", "pooled"),
        "\\bottomrule",
        "\\end{tabular}",
    ]


def build_table(results: Dict[str, Dict[str, Dict]]) -> str:
    any_dagger = any(r["at_min_attainable_p"] for m in results.values() for r in m.values())
    caption = (
        "\\ToolName{} vs.\\ Pure CLIP, two-sided exact Wilcoxon signed-rank\n"
        "test over activities. Wins, losses, and ties count activities on which\n"
        "\\ToolName{} scores higher, lower, or equal; ties are excluded from the test.\n"
        "The pooled test treats activities from different homes as independent."
    )
    if any_dagger:
        caption += "\n$^\\dagger$ smallest attainable $p$ for this number of activities."

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        "\\label{tab:retrieval_wilcoxon}",
        *build_tabular(results),
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_suffix",   default="fd60_seq_rb1_textclip_projmlp_clipmlm_v3")
    parser.add_argument("--base_dir",       default="results/evals")
    parser.add_argument("--results_subdir", default="FD_60_p")
    parser.add_argument("--output_dir",     default="results/evals/imwut_paper")
    parser.add_argument("--json_out",       default="results/evals/imwut_paper/retrieval_v3/wilcoxon_vs_pure_clip.json")
    args = parser.parse_args()

    base_dir, output_dir = Path(args.base_dir), Path(args.output_dir)
    quant_path = output_dir / "retrieval_results_quant.json"
    quant = json.loads(quant_path.read_text())
    if quant["model_suffix"] != args.model_suffix or quant["results_subdir"] != args.results_subdir:
        raise AssertionError(f"{quant_path} is for {quant['model_suffix']}/{quant['results_subdir']}, "
                             f"expected {args.model_suffix}/{args.results_subdir}")

    per_dir = {
        d: collect_data(HOMES, args.model_suffix, base_dir, args.results_subdir, d)[0]
        for d in (SIG_TOOL_DIR, SIG_BASELINE_DIR)
    }

    # Same per-activity values as tab:retrieval_quant, or the two tables would disagree.
    for d in (SIG_TOOL_DIR, SIG_BASELINE_DIR):
        saved = quant["per_label"][DIRECTION_ROW_LABELS[d]]
        for h in HOMES:
            fresh = {l[4:].strip(): v for l, v in per_dir[d][h].items() if l.startswith("[L1]")}
            if set(fresh) != set(saved[h]):
                raise AssertionError(f"{d}/{h}: activities differ from {quant_path}")
            for act, vals in fresh.items():
                for mk, _ in METRICS:
                    if not math.isclose(vals[mk], saved[h][act][mk], rel_tol=1e-12, abs_tol=1e-12):
                        raise AssertionError(f"{d}/{h}/{act}/{mk}: {vals[mk]} != {saved[h][act][mk]} in {quant_path}; "
                                             "rerun retrieval_metrics_agg.py")

    tool, base = per_dir[SIG_TOOL_DIR], per_dir[SIG_BASELINE_DIR]
    results: Dict[str, Dict[str, Dict]] = {}
    per_activity: Dict[str, Dict[str, Dict]] = {}
    for mk, mname in METRICS:
        results[mname], per_activity[mname] = {}, {}
        px: List[float] = []
        py: List[float] = []
        for h in HOMES:
            acts = sorted(l for l in tool[h] if l.startswith("[L1]"))
            x = [tool[h][a][mk] for a in acts]
            y = [base[h][a][mk] for a in acts]
            results[mname][h] = run_test(x, y)
            per_activity[mname][h] = {a[4:].strip(): {"tool": xi, "pure_clip": yi, "diff": xi - yi}
                                      for a, xi, yi in zip(acts, x, y)}
            px += x
            py += y
        results[mname]["pooled"] = run_test(px, py)

    # Pooled p must match the p behind the stars in tab:retrieval_quant.
    for mname in ("mAP", "P@100"):
        star_p = quant["significance"]["metrics"][mname]["pooled"]["p_value"]
        ours = results[mname]["pooled"]["p"]
        if not math.isclose(ours, star_p, rel_tol=P_RTOL):
            raise AssertionError(f"Pooled {mname} p = {ours} but tab:retrieval_quant stars use p = {star_p}")

    tex_path = output_dir / "retrieval_wilcoxon.tex"
    tex_path.write_text(build_table(results), encoding="utf-8")
    print(f"✅  TEX  saved: {tex_path}")
    png_path = output_dir / "retrieval_wilcoxon.png"
    _compile_to_png("\n".join([*_STANDALONE_PREAMBLE, r"\begin{document}", r"\small",
                               *build_tabular(results), r"\end{document}"]), png_path)

    comp_paths = [base_dir / h / args.results_subdir / f"{h}_{args.model_suffix}" / "comprehensive_results.json"
                  for h in HOMES]
    out = {
        "comparison": f"{DIRECTION_ROW_LABELS[SIG_TOOL_DIR]} minus {DIRECTION_ROW_LABELS[SIG_BASELINE_DIR]}",
        "directions": {"tool": SIG_TOOL_DIR, "baseline": SIG_BASELINE_DIR},
        "baseline_note": "frozen CLIP pooler_output (L2-normalised) on both query and window caption",
        "model_suffix": args.model_suffix,
        "results_subdir": args.results_subdir,
        "unit": "L1 activity",
        "test": {
            "name": "Wilcoxon signed-rank",
            "alternative": "two-sided",
            "zero_method": "wilcox",
            "scipy_call": "scipy.stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided', method='exact')",
            "tied_ranks": "exact enumeration over the observed tied ranks; scipy value kept as p_scipy_exact",
            "difference_rounding_decimals": ROUND_DECIMALS,
            "W": "min(W+, W-)",
        },
        "results": results,
        "per_activity": per_activity,
        "inputs": {
            "comprehensive_results": [{"path": str(p), "sha256": _sha256(p)} for p in comp_paths],
            "retrieval_results_quant": {"path": str(quant_path), "sha256": _sha256(quant_path)},
        },
        "outputs": {"tex": str(tex_path), "png": str(png_path)},
        "git": _git_info(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"✅  JSON saved: {json_path}")

    for mname in results:
        for g, r in results[mname].items():
            print(f"  {mname:6s} {g:7s} n={r['n_activities']:2d} W/L/T={fmt_wlt(r):10s} W={r['W']} "
                  f"p={r['p']:.4g} ({r['p_method']}; scipy exact {r['p_scipy_exact']:.4g})")


if __name__ == "__main__":
    main()
