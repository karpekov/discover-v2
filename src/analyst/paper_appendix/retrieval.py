"""Appendix section 6: retrieval benchmark details behind tab:retrieval_quant. Reads saved results only.

    conda activate discover-v2-env
    python src/analyst/paper_appendix/retrieval.py     # → results/paper_appendix/retrieval/
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from evals.evaluate_embeddings import EmbeddingEvaluator  # noqa: E402
from evals.imwut_paper.retrieval_metrics_agg import expected_random_metrics, normalize_label  # noqa: E402
from utils.label_utils import convert_labels_to_text  # noqa: E402

MODEL = "fd60_seq_rb1_textclip_projmlp_clipmlm_v3"
HOMES = ["milan", "aruba", "cairo"]
OUT = ROOT / "results/paper_appendix/retrieval"
METRICS = [("map", "mAP"), ("10", "P@10"), ("50", "P@50"), ("100", "P@100"), ("mrr", "MRR")]
DIRS = {"prototype2sensor": "sensorq", "prototype2text_noproj": "pure_clip"}

DISPLAY = {
    "Sleep": "Sleep", "Bed to Toilet": "Bed to toilet", "Leave Home": "Leave home", "Chores": "Chores",
    "Eating: General": "Eating", "Eating: Breakfast": "Breakfast", "Eating: Lunch": "Lunch",
    "Eating: Dinner": "Dinner", "Work": "Work", "Relax: General": "Relax", "Relax: Watch TV": "Watching TV",
    "Relax: Read": "Reading", "Enter_Home": "Enter home", "Guest_Bathroom": "Guest bathroom",
    "Housekeeping": "Housekeeping", "Kitchen_Activity": "Kitchen activity", "Master_Bathroom": "Master bathroom",
    "Meal_Preparation": "Meal preparation", "Night wandering": "Night wandering",
}

# Values quoted in the paper's Results text: (home, canonical label, SensorQ mAP, comparison, value).
PAPER_CLAIMS = [
    ("cairo", "Eating: Breakfast", 0.61, "pure_clip", 0.25), ("cairo", "Eating: Lunch", 0.64, "pure_clip", 0.36),
    ("cairo", "Work", 0.48, "pure_clip", 0.14), ("aruba", "Work", 0.56, "pure_clip", 0.27),
    ("milan", "Guest_Bathroom", 0.54, "pure_clip", 0.20), ("aruba", "Sleep", 0.72, "pure_clip", 0.89),
    ("aruba", "Meal_Preparation", 0.79, "pure_clip", 0.87), ("milan", "Master_Bathroom", 0.51, "pure_clip", 0.66),
    ("aruba", "Bed to Toilet", 0.63, "random", 0.004),
]


def run_dir(h: str) -> Path:
    return ROOT / "results/evals" / h / "FD_60_p" / f"{h}_{MODEL}"


def test_label_counts(h: str) -> collections.Counter:
    d = json.loads((ROOT / f"data/processed/casas/{h}/FD_60_p/test.json").read_text())
    return collections.Counter(s["metadata"]["ground_truth_labels"]["primary_l1"] for s in d["samples"])


def tex(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, index, raw_counts = [], {}, {}
    for h in HOMES:
        comp = json.loads((run_dir(h) / "comprehensive_results.json").read_text())
        pl = comp["retrieval_metrics"]["L1"]["prototype_based"]["per_label"]
        label_counts = json.loads((run_dir(h) / "retrieval/retrieval_results_l1.json").read_text())["label_counts"]
        M = sum(label_counts.values())
        index[h] = {"size": M, "label_counts": label_counts}
        raw = test_label_counts(h)
        merged = collections.Counter()
        for k, v in raw.items():
            merged[EmbeddingEvaluator._CAIRO_L1_MERGES.get(k.lower().strip(), k) if h == "cairo" else k] += v
        raw_counts[h] = merged
        labels = list(pl["prototype2sensor"]["count"])
        queries = dict(zip(labels, convert_labels_to_text(labels, single_description=True, house_name=h,
                                                          description_style="long_desc")))
        for raw in labels:
            canon = normalize_label(raw)
            n = int(pl["prototype2sensor"]["count"][raw])
            assert n == label_counts[raw] == pl["prototype2text_noproj"]["count"][raw], (h, raw)
            r = {"home": h, "raw_label": raw, "canonical": canon, "activity": DISPLAY.get(canon, canon), "n": n,
                 "query": queries[raw].strip()}
            for d, tag in DIRS.items():
                for mk, mn in METRICS:
                    r[f"{tag}_{mn}"] = float(pl[d][mk][raw])
            r["random_mAP"] = expected_random_metrics(n, M)["mAP"]
            rows.append(r)
    df = pd.DataFrame(rows)
    df["_h"] = df.home.map(HOMES.index)
    df = df.sort_values(["_h", "activity"]).drop(columns="_h").reset_index(drop=True)
    df.to_csv(OUT / "per_activity_metrics.csv", index=False)

    # macro averages must reproduce tab:retrieval_quant
    quant = json.loads((ROOT / "results/evals/imwut_paper/retrieval_results_quant.json").read_text())["rows"]
    errors = []
    for h in HOMES:
        sub = df[df.home == h]
        for tag, row in (("sensorq", "\\ToolName"), ("pure_clip", "Text Only: Pure CLIP")):
            for _, mn in METRICS:
                got, want = sub[f"{tag}_{mn}"].mean(), quant[row][mn][h]
                if abs(got - want) > 6e-5:
                    errors.append(f"{h} {tag} {mn}: macro {got:.4f} vs tab:retrieval_quant {want:.4f}")
    if len(df) != 28:
        errors.append(f"expected 28 activities, got {len(df)}")

    checks = []
    for h, canon, sq, kind, val in PAPER_CLAIMS:
        r = df[(df.home == h) & (df.canonical == canon)].iloc[0]
        got_sq = r["sensorq_mAP"]
        got_cmp = r["pure_clip_mAP"] if kind == "pure_clip" else r["random_mAP"]
        nd = 3 if kind == "random" else 2
        ok = round(got_sq, 2) == sq and round(got_cmp, nd) == val
        checks.append({"home": h, "activity": r["activity"], "paper_sensorq": sq, "saved_sensorq": round(got_sq, 4),
                       "comparison": kind, "paper_value": val, "saved_value": round(got_cmp, 4), "match": ok})
    pd.DataFrame(checks).to_csv(OUT / "paper_value_check.csv", index=False)

    excl = EmbeddingEvaluator._MINOR_LABELS_BY_DATASET
    merges = EmbeddingEvaluator._CAIRO_L1_MERGES
    noisy = ["other", "no_activity", "no_sensor_readings", "unknown", "none", "null", "nan", "other activity",
             "miscellaneous", "misc"]
    idx = {h: {"index_size": index[h]["size"], "kept_label_counts": index[h]["label_counts"],
               "raw_test_label_counts": dict(raw_counts[h].most_common()),
               "minor_labels_excluded_by_list": sorted(excl.get(h, [])),
               "excluded_test_windows": {k: v for k, v in raw_counts[h].items()
                                         if k not in index[h]["label_counts"]}}
           for h in HOMES}
    idx["cairo_l1_merges"] = merges
    idx["noisy_labels_always_excluded"] = noisy
    (OUT / "index_and_exclusions.json").write_text(json.dumps(idx, indent=1))

    q = [r"\begin{table*}[t]", r"\centering\small",
         r"\caption{Natural-language query used for each annotated activity in the retrieval benchmark "
         r"(one query per activity and home; the same sentence is used by \ToolName{} and the text-only baselines).}",
         r"\label{tab:retrieval_queries}", r"\begin{tabular}{l l p{0.66\textwidth}}", r"\toprule",
         r"Home & Activity & Query \\", r"\midrule"]
    for i, h in enumerate(HOMES):
        sub = df[df.home == h]
        for j, (_, r) in enumerate(sub.iterrows()):
            home = rf"\multirow{{{len(sub)}}}{{*}}{{{h.capitalize()}}}" if j == 0 else ""
            q.append(f"{home} & {tex(r['activity'])} & {tex(r['query'])} \\\\")
        if i < len(HOMES) - 1:
            q.append(r"\midrule")
    q += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (OUT / "retrieval_queries.tex").write_text("\n".join(q))

    t = [r"\begin{table*}[t]", r"\centering\small",
         r"\caption{\ToolName{} text-to-sensor retrieval per annotated activity, with one natural-language query per "
         r"activity (Table~\ref{tab:retrieval_queries}). $n$ is the number of relevant test windows; the index of "
         r"each home is its labeled test windows (Milan " + f"{index['milan']['size']:,}" + r", Aruba "
         + f"{index['aruba']['size']:,}" + r", Cairo " + f"{index['cairo']['size']:,}" + r"). "
         r"Pure CLIP is the text-only baseline (frozen CLIP on the query and on each window's caption).}",
         r"\label{tab:retrieval_quant_all_labels}",
         r"\begin{tabular}{l l r rr rrrr}", r"\toprule",
         r" & & & \multicolumn{2}{c}{mAP} & \multicolumn{4}{c}{\ToolName{}} \\",
         r"\cmidrule(lr){4-5}\cmidrule(l){6-9}",
         r"Home & Activity & $n$ & \ToolName{} & Pure CLIP & P@10 & P@50 & P@100 & MRR \\", r"\midrule"]
    for i, h in enumerate(HOMES):
        sub = df[df.home == h]
        for j, (_, r) in enumerate(sub.iterrows()):
            home = rf"\multirow{{{len(sub)}}}{{*}}{{{h.capitalize()}}}" if j == 0 else ""
            a, b = r["sensorq_mAP"], r["pure_clip_mAP"]
            fa = rf"\textbf{{{a:.2f}}}" if round(a, 2) > round(b, 2) else f"{a:.2f}"
            fb = rf"\textbf{{{b:.2f}}}" if round(b, 2) > round(a, 2) else f"{b:.2f}"
            t.append(f"{home} & {tex(r['activity'])} & {r['n']:,} & {fa} & {fb} & "
                     + " & ".join(f"{r[f'sensorq_{m}']:.2f}" for m in ("P@10", "P@50", "P@100", "MRR")) + r" \\")
        t.append(r"\cmidrule(l){2-9}")
        sm = sub[[f"sensorq_{m}" for _, m in METRICS] + ["pure_clip_mAP"]].mean()
        t.append(rf" & \textit{{Mean}} & & {sm['sensorq_mAP']:.2f} & {sm['pure_clip_mAP']:.2f} & "
                 + " & ".join(f"{sm[f'sensorq_{m}']:.2f}" for m in ("P@10", "P@50", "P@100", "MRR")) + r" \\")
        if i < len(HOMES) - 1:
            t.append(r"\midrule")
    t += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (OUT / "retrieval_quant_all_labels.tex").write_text("\n".join(t))

    print(df[["home", "activity", "n", "sensorq_mAP", "pure_clip_mAP", "random_mAP"]].round(3).to_string())
    print(pd.DataFrame(checks).to_string())
    if errors:
        raise SystemExit("FAIL: per-activity values do not reproduce tab:retrieval_quant:\n  " + "\n  ".join(errors))
    print("OK: per-activity values reproduce the macro averages in tab:retrieval_quant")
    bad = [c for c in checks if not c["match"]]
    print(f"paper text check: {len(checks) - len(bad)}/{len(checks)} match" + (f"; MISMATCH: {bad}" if bad else ""))


if __name__ == "__main__":
    main()
