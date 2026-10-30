"""Static evaluation dashboard for a question set: <runs dir>/dashboard.html.

Self-contained (data embedded, figures referenced relative to the runs dir). Reads result.json of every
run, summary.csv (short answers; run `summarize` first), the latest 4-step review
(review/<name>_results/review_scores.json plus that packet's key and rating CSVs), and the older
rater_packets/<name>/scores.json.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import get_home
from .rater import _wilson
from .review import _step_metrics, consensus_votes
from .summary import primary_stat, status, trend_arrows

_TEMPLATE = Path(__file__).with_name("dashboard_template.html")


def _concept_payload(c: dict, rater: Dict[str, dict]) -> dict:
    cc = c["concept"]
    out = {
        "id": cc["id"], "name": cc["name"], "definition": cc["physical_definition"],
        "detectable": cc["detectable"], "detectability_note": cc["detectability_note"],
        "skipped": "skipped" in c,
        "filters": {k: cc.get(k, []) for k in ("time_of_day", "days_of_week", "rooms", "required_sensors", "timing_hours")},
        "rater": rater.get(cc["id"], {}),
        "step2": None,
    }
    if out["skipped"]:
        return out
    ret, ov = c["retrieval"], c["stats"]["overall"]
    timing = next((a for a in c["stats"]["analyses"] if a["type"] == "timing" and not a.get("insufficient_data")), None)
    out.update({
        "primary": primary_stat(c),
        "n_matched": ret["n_matched_windows"], "n_eligible": ret["n_eligible_windows"],
        "n_matched_without_structural_filters": ret["n_matched_without_structural_filters"],
        "queries": [{"q": q, "n": n} for q, n in ret["per_query_matched_windows"].items()],
        "min_per_day": ov["matched_min_per_day_mean"], "share_pct": ov["share_pct_mean"],
        "episodes_per_day": ov["episodes_per_day_mean"],
        "timing": None if not timing else {
            k: timing.get(k) for k in ("event", "start_window", "daily_time", "days_with_event_pct", "trend",
                                       "compare", "peak_hours", "pct_days_within_10min_after_time_bucket_edge")},
        "gt_label_mix": c.get("diagnostics", {}).get("gt_label_mix_matched", {}),
    })
    return out


def _rater_data(rdir: Path) -> dict:
    """Packet composition (from *_key.csv) and scores (scores.json, if ratings were scored)."""
    packets = {}
    for pdir in sorted((rdir / "rater_packets").glob("*")):
        keys = sorted(pdir.glob("*_key.csv"))
        if not keys:
            continue
        k = pd.concat([pd.read_csv(p) for p in keys], ignore_index=True)
        scores_path = pdir / "scores.json"
        packets[pdir.name] = {
            "n_items": {h: int(n) for h, n in k["question_id"].str.split("/").str[0].value_counts().items()},
            "n_behaviors": int(k.groupby(["question_id", "concept_id"]).ngroups),
            "strata": {s: int(n) for s, n in k["stratum"].value_counts().items()},
            "scores": json.loads(scores_path.read_text()) if scores_path.exists() else None,
        }
    return packets


def _rate(votes: List[str]) -> dict:
    decided = [v for v in votes if v in ("yes", "no")]
    unsure = votes.count("unsure")
    if not decided:
        return {"yes_rate": None, "n_decided": 0, "n_items": len(votes), "unsure": unsure, "wilson_95ci": None}
    k = decided.count("yes")
    return {"yes_rate": round(k / len(decided), 3), "n_decided": len(decided), "n_items": len(votes),
            "unsure": unsure, "wilson_95ci": _wilson(k, len(decided))}


def _load_review(rdir: Path) -> dict:
    """review_v2 when that packet has been scored, otherwise the latest scored review.

    Step 2 rates are the planner's own queries. Far-caption controls are ``stratum=decoy`` and are
    returned separately so they do not move the relevance number.
    """
    preferred = rdir / "review" / "review_v2_results" / "review_scores.json"
    results = sorted((rdir / "review").glob("*_results/review_scores.json"), key=lambda p: p.stat().st_mtime)
    if preferred.exists():
        scores_path = preferred
    elif results:
        scores_path = results[-1]
    else:
        return {}
    name = scores_path.parent.name.removesuffix("_results")
    scores = json.loads(scores_path.read_text())
    key_paths = sorted((rdir / "review" / f"{name}_key").glob("*_key.csv"))
    rating_paths = sorted(p for p in (rdir / "review" / name).glob("*.csv") if p.is_file())
    cells: Dict[tuple, dict] = {}
    concepts: Dict[tuple, dict] = {}
    decoy_items: List[dict] = []
    if key_paths and rating_paths:
        key = pd.concat([pd.read_csv(p) for p in key_paths], ignore_index=True)
        ratings = pd.concat([pd.read_csv(p) for p in rating_paths], ignore_index=True)
        ratings["rating"] = ratings["rating"].astype(str).str.lower().str.strip()
        ratings = ratings.drop_duplicates(["reviewer", "item_id"], keep="last")
        consensus = consensus_votes(ratings)
        outcome = {}
        for q in scores["consensus"]["funnel"]["per_question"]:
            step1 = next((n for n in q["path"] if str(n).startswith("s1_")), q["outcome"])
            outcome[(q["home"], q["question_id"])] = step1
        if "stratum" not in key.columns:
            key["stratum"] = ""
        key["stratum"] = key["stratum"].fillna("")
        for (home, qid), rows in key.groupby(["home", "question_id"], sort=False):
            s2 = rows[rows.step == 2]
            real, decoys = s2[s2.stratum != "decoy"], s2[s2.stratum == "decoy"]
            cells[(home, qid)] = {
                "step1": outcome.get((home, qid)),
                "step2": _rate([consensus.get(i, "unsure") for i in real.item_id]),
                "decoys": [{"query": r.query, "score": None if pd.isna(r.score) else round(float(r.score), 3),
                            "concept_id": "" if pd.isna(r.concept_id) else r.concept_id,
                            "rating": consensus.get(r.item_id)} for r in decoys.itertuples(index=False)],
            }
            decoy_items.extend({"home": home, "question_id": qid, **d} for d in cells[(home, qid)]["decoys"])
            for cid, crows in real.groupby(real.concept_id.fillna("")):
                if not cid:
                    continue
                concepts[(home, qid, cid)] = _rate([consensus.get(i, "unsure") for i in crows.item_id])
        by_reviewer_home: Dict[str, dict] = {}
        for rv, sub in ratings.groupby("reviewer"):
            votes = pd.DataFrame({"item_id": sub.item_id.tolist(), "vote": sub.rating.tolist()})
            by_reviewer_home[rv] = {
                h: _step_metrics(key[key.home == h], votes) for h in sorted(sub.home.unique())
            }
    else:
        by_reviewer_home = {}
    return {
        "name": name,
        "reviewers": scores["reviewers"],
        "consensus_reviewers": scores.get("consensus_reviewers", scores["reviewers"]),
        "excluded_reviewers": scores.get("excluded_reviewers", []),
        "n_ratings": scores["n_ratings"],
        "n_items": scores["n_items"],
        "n_items_rated": scores["n_items_rated"],
        "steps": scores["consensus"]["steps"],
        "per_home": {h: {"steps": v["steps"]} for h, v in scores["per_home"].items()},
        "per_reviewer": {rv: {"steps": v["steps"]} for rv, v in scores["per_reviewer"].items()},
        "per_reviewer_home": by_reviewer_home,
        "kappa": scores["kappa"],
        "fleiss": scores.get("fleiss") or {},
        "decoys": decoy_items,
        "cells": cells,
        "concepts": concepts,
    }


def build_dashboard(rdir: Path, qfile: dict, question_key) -> Path:
    homes = qfile.get("homes", ["milan", "aruba", "cairo"])
    summary_csv = rdir / "summary.csv"
    short = pd.read_csv(summary_csv).set_index("question_id") if summary_csv.exists() else None
    packets = _rater_data(rdir)
    review = _load_review(rdir)
    latest = next((p for p in reversed(list(packets.values())) if p["scores"]), None)
    per_concept_scores = latest["scores"]["per_concept"] if latest else {}

    home_info, questions = {}, []
    for q in qfile["questions"]:
        key = question_key(q)
        row = {"num": q.get("num"), "id_sonia": q.get("id_sonia", ""), "key": key, "category": q.get("category", ""),
               "group": q.get("group", ""), "question": q["question"], "cells": {}}
        for home in q.get("homes", homes):
            f = rdir / home / key / "result.json"
            if not f.exists():
                row["cells"][home] = {"status": "not run"}
                continue
            r = json.loads(f.read_text())
            if home not in home_info:
                cfg = get_home(home)
                home_info[home] = {"threshold": r["threshold"], "data_range": r["data_range"],
                                   "data_start": cfg.data_start}
            rater = {k.split("/")[-1]: v for k, v in per_concept_scores.items() if k.startswith(f"{home}/{key}/")}
            cell_review = review.get("cells", {}).get((home, key))
            st = status(r)
            arrows = trend_arrows(r) if st != "undetectable" and q.get("category") == "change" else []
            run_dir = f"{home}/{key}"
            row["cells"][home] = {
                "status": st,
                "short": "" if short is None or st == "undetectable" else str(short.loc[key].get(f"{home}_result", "") or ""),
                "arrows": "".join(a for _, a in arrows),
                "arrow_names": [{"name": n, "arrow": a} for n, a in arrows],
                "headline": r["report"]["headline"], "body": r["report"]["body"],
                "limitations": r["report"]["limitations"],
                "findings": r["report"].get("findings", []),
                "interpretation": r["plan"]["interpretation"], "caveats": r["plan"]["caveats"],
                "analyses": [a["type"] + (f":{a['timing_event']}" if a["type"] == "timing" else "")
                             for a in r["plan"]["analyses"]],
                "scope": r["scope_range"], "warnings": r["warnings"], "ungrounded": r["ungrounded_numbers"],
                "figure": f"{run_dir}/figure.png" if (rdir / run_dir / "figure.png").exists() else None,
                "links": {n: f"{run_dir}/{n}" for n in ("report.md", "plan.json", "result.json")},
                "concepts": [_concept_payload(c, rater) for c in r["concepts"]],
                "review": cell_review,
            }
            if cell_review:
                for concept in row["cells"][home]["concepts"]:
                    concept["step2"] = review.get("concepts", {}).get((home, key, concept["id"]))
        questions.append(row)

    data = {
        "set_name": rdir.name, "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "homes": homes, "home_info": home_info, "questions": questions, "rater_packets": packets,
        "review": {k: v for k, v in review.items() if k not in ("cells", "concepts")} if review else None,
    }
    html = _TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, default=str).replace("</", "<\\/"))
    out = rdir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out
