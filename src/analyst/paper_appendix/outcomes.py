"""Appendix section 4: per-question outcomes (majority vote) and raw rating counts. Reads saved files only.

    conda activate discover-v2-env
    python src/analyst/paper_appendix/outcomes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from analyst.review import _funnel_path, _is_excluded, consensus_votes  # noqa: E402

REVIEW = ROOT / "results/analyst/runs/analyst_questions/review"
OUT = ROOT / "results/paper_appendix"
HOMES = ["milan", "aruba", "cairo"]

OUTCOME = {"s4_ok": "pass", "s1_no_ok": "refused", "s1_missed": "missed", "s1_false_ans": "wrongly answerable",
           "s3_bad": "poor retrieval", "s4_bad": "unsupported findings", "s1_unsure": "unsure"}
EXPECTED = {"pass": 42, "refused": 24, "missed": 11, "wrongly answerable": 4, "poor retrieval": 5,
            "unsupported findings": 1, "unsure": 0}
SANKEY_NODE = {v: k for k, v in OUTCOME.items()}


def load():
    key = pd.concat([pd.read_csv(p) for p in sorted((REVIEW / "review_v2_key").glob("*_key.csv"))], ignore_index=True)
    csvs = sorted((REVIEW / "review_v2").glob("*.csv"))
    ratings = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    ratings["rating"] = ratings["rating"].astype(str).str.lower().str.strip()
    ratings = ratings.drop_duplicates(["reviewer", "item_id"], keep="last")
    return key[key.home.isin(HOMES)], ratings, csvs


def sankey_data() -> dict:
    js = (ROOT / "figures/src/paper_charts/raters_funnel/data.js").read_text()
    return json.loads(js.split("=", 1)[1].strip().rstrip(";"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    key, ratings, csvs = load()
    votes = consensus_votes(ratings)
    qs = json.loads((ROOT / "metadata/analyst_questions.json").read_text())["questions"]
    by_num = {q["num"]: q for q in qs}
    num_of = {f"{q['num']:02d}_{q['id']}": q["num"] for q in qs}

    rows = []
    for (home, qid), r in key.groupby(["home", "question_id"], sort=False):
        path = _funnel_path(r, votes)
        if path[-1] not in OUTCOME:
            raise SystemExit(f"FAIL: {home}/{qid} ends at {path[-1]}, which has no outcome code")
        rows.append({"home": home, "question_id": qid, "num": num_of[qid], "path": ">".join(path),
                     "outcome": OUTCOME[path[-1]]})
    out = pd.DataFrame(rows)

    errors = []
    totals = out.outcome.value_counts().to_dict()
    errors += [f"{k}: got {totals.get(k, 0)}, expected {v}" for k, v in EXPECTED.items() if totals.get(k, 0) != v]
    sk = sankey_data()
    for oc, node in SANKEY_NODE.items():
        for h in HOMES:
            got = int(((out.outcome == oc) & (out.home == h)).sum())
            fig = sk["by_home"].get(node, {}).get(h, 0)
            if got != fig:
                errors.append(f"{oc} {h}: got {got}, Sankey data.js has {fig}")
    if len(out) != 87 or out.groupby("home").size().ne(29).any():
        errors.append(f"expected 29 questions x 3 homes, got {out.groupby('home').size().to_dict()}")
    if errors:
        raise SystemExit("FAIL: outcomes do not match the Sankey:\n  " + "\n  ".join(errors))
    print("OK: outcome counts match the expected totals and the Sankey per-home splits")
    print(pd.crosstab(out.outcome, out.home, margins=True).reindex(index=list(EXPECTED), columns=HOMES + ["All"]).fillna(0).astype(int))
    out.sort_values(["num", "home"]).to_csv(OUT / "per_question_outcomes.csv", index=False)

    wide = out.pivot(index="num", columns="home", values="outcome")
    tex = [r"\begin{table*}[t]", r"\centering\small",
           r"\caption{Outcome of each caregiver question in each home under the raters' majority vote. "
           r"\emph{refused}: the system and the raters agreed the question cannot be answered with the home's sensors; "
           r"\emph{missed}: the raters judged it answerable but the system declined; "
           r"\emph{wrongly answerable}: the system answered but the raters judged it not answerable; "
           r"\emph{poor retrieval}: most rated windows did not show the behavior; "
           r"\emph{unsupported findings}: most findings were not supported; "
           r"\textbf{pass}: every step passed.}",
           r"\label{tab:per_question_outcomes}",
           r"\begin{tabular}{r p{0.5\textwidth} c c c}", r"\toprule",
           r"\# & Question & Milan & Aruba & Cairo \\", r"\midrule"]
    for num, row in wide.sort_index().iterrows():
        text = by_num[num]["question"].replace("&", r"\&").replace("’", "'")
        cells = [(r"\textbf{pass}" if row[h] == "pass" else row[h]) for h in HOMES]
        tex.append(f"{num} & {text} & " + " & ".join(cells) + r" \\")
    tex.append(r"\midrule")
    for oc in EXPECTED:
        counts = [int(((out.outcome == oc) & (out.home == h)).sum()) for h in HOMES]
        tex.append(f" & \\textit{{{oc}}} (total {sum(counts)}) & " + " & ".join(map(str, counts)) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (OUT / "per_question_outcomes.tex").write_text("\n".join(tex))

    item = key.drop_duplicates("item_id").set_index("item_id")
    subset = pd.Series("all", index=item.index)
    subset[item.step == 2] = item.loc[item.step == 2, "stratum"].fillna("query")
    subset[item.step == 3] = item.loc[item.step == 3, "stratum"]
    item = item.assign(subset=subset)
    rated = ratings[ratings.item_id.isin(item.index)].copy()
    rated["step"] = rated.item_id.map(item["step"])
    rated["subset"] = rated.item_id.map(item["subset"])
    recs = []
    for (rv, h, s, ss), g in rated.groupby(["reviewer", "home", "step", "subset"]):
        n_key = int(((item.home == h) & (item.step == s) & (item.subset == ss)).sum())
        vc = g.rating.value_counts()
        recs.append({"rater": rv, "counted_in_vote": not _is_excluded(rv), "home": h, "step": s, "subset": ss,
                     "items_in_key": n_key, "rated": len(g), "yes": int(vc.get("yes", 0)),
                     "no": int(vc.get("no", 0)), "unsure": int(vc.get("unsure", 0))})
    for (h, s, ss), g in item.groupby(["home", "step", "subset"]):
        v = pd.Series([votes.get(i, "missing") for i in g.index]).value_counts()
        recs.append({"rater": "MAJORITY_VOTE", "counted_in_vote": "", "home": h, "step": s, "subset": ss,
                     "items_in_key": len(g), "rated": int(len(g) - v.get("missing", 0)), "yes": int(v.get("yes", 0)),
                     "no": int(v.get("no", 0)), "unsure": int(v.get("unsure", 0))})
    rc = pd.DataFrame(recs)
    rc["_h"] = rc.home.map(HOMES.index)
    rc = rc.sort_values(["step", "subset", "_h", "rater"]).drop(columns="_h")
    rc.to_csv(OUT / "rating_counts.csv", index=False)
    print(f"wrote per_question_outcomes.{{tex,csv}} and rating_counts.csv from {len(csvs)} rating files")


if __name__ == "__main__":
    main()
