"""Appendix section 1: facts about the final analyst run, read from saved result.json files and the LLM cache.

    conda activate discover-v2-env
    python src/analyst/paper_appendix/qa_facts.py      # → results/paper_appendix/qa_config_facts.json
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from analyst.planner import PLANNER_SYSTEM  # noqa: E402
from analyst.reporter import REPORTER_SYSTEM  # noqa: E402

RUN = ROOT / "results/analyst/runs/analyst_questions"
CACHE = ROOT / "metadata/analyst_llm_cache"
OUT = ROOT / "results/paper_appendix"
HOMES = ["milan", "aruba", "cairo"]


def results():
    for h in HOMES:
        for p in sorted((RUN / h).glob("*/result.json")):
            yield h, p.parent.name, json.loads(p.read_text())


def main():
    f = {"runs": 0, "threshold": defaultdict(set), "settings": Counter(), "data_range": {}, "scope": Counter(),
         "trend": [], "compare": [], "timing": [], "ungrounded": [], "warnings": [], "llm": {"planner": [], "reporter": []},
         "concepts_analyzed": 0}
    for h, q, r in results():
        f["runs"] += 1
        f["threshold"][h].add(r["threshold"])
        f["settings"][json.dumps(r["settings"], sort_keys=True)] += 1
        f["data_range"][h] = r["data_range"]
        f["scope"]["full" if not (r["plan"]["scope_start_date"] or r["plan"]["scope_end_date"]) else "restricted"] += 1
        if r["ungrounded_numbers"]:
            f["ungrounded"].append({"home": h, "question": q, "values": r["ungrounded_numbers"]})
        for w in r["warnings"]:
            f["warnings"].append({"home": h, "question": q, "warning": w})
        for role in ("planner", "reporter"):
            f["llm"][role].append({"home": h, "question": q, **r["llm"][role]})
        for c in r["concepts"]:
            if "skipped" in c:
                continue
            f["concepts_analyzed"] += 1
            for a in c["stats"]["analyses"]:
                if a.get("insufficient_data"):
                    continue
                if a["type"] == "trend":
                    m, s = a["minutes"]["kendall_p_value"], a["share"]["kendall_p_value"]
                    f["trend"].append({"home": h, "question": q, "concept": c["concept"]["id"],
                                       "significant": a["significant"], "minutes_p": m, "share_p": s})
                elif a["type"] == "compare_periods":
                    f["compare"].append({"home": h, "question": q, "concept": c["concept"]["id"],
                                         "keys": sorted(k for t in a["tests"] for k in t)})
                elif a["type"] == "timing":
                    f["timing"].append({"home": h, "question": q, "concept": c["concept"]["id"], "event": a["event"],
                                        "day_boundary_hour": a["day_boundary_hour"],
                                        "session_gap_minutes": a["session_gap_minutes"],
                                        "min_session_minutes": a["min_session_minutes"],
                                        "edge_pct": a.get("pct_days_within_10min_after_time_bucket_edge")})

    trend = f["trend"]
    flag_from = Counter()
    for t in trend:
        m_sig, s_sig = t["minutes_p"] is not None and t["minutes_p"] < 0.05, t["share_p"] is not None and t["share_p"] < 0.05
        t["follows"] = ("both agree" if m_sig == s_sig else "share" if t["significant"] == s_sig else "minutes")
        flag_from[t["follows"]] += 1
    timing = f["timing"]

    llm = {}
    for role in ("planner", "reporter"):
        entries, missing, systems, dates = [], [], Counter(), Counter()
        for e in f["llm"][role]:
            p = CACHE / f"{e['cache_key']}.json"
            if not p.exists():
                missing.append(e)
                continue
            c = json.loads(p.read_text())
            entries.append(c["latency_s"])
            systems[c["system"]] += 1
            dates[c["created_at"][:10]] += 1
        current = PLANNER_SYSTEM if role == "planner" else REPORTER_SYSTEM
        llm[role] = {"n_runs": len(f["llm"][role]), "n_unique_keys": len({e["cache_key"] for e in f["llm"][role]}),
                     "n_cache_files_found": len(entries), "missing": missing,
                     "latency_s_median": st.median(entries) if entries else None,
                     "latency_s_iqr": [round(x, 2) for x in st.quantiles(entries, n=4)[::2]] if len(entries) > 3 else None,
                     "latency_s_min_max": [min(entries), max(entries)] if entries else None,
                     "created_dates": dict(sorted(dates.items())),
                     "n_distinct_system_prompts": len(systems),
                     "system_prompt_equals_current_source": [s == current for s in systems]}
    all_cache = Counter(p.name.split("_")[0] for p in CACHE.glob("*.json"))

    out = {
        "n_runs": f["runs"], "n_concepts_analyzed": f["concepts_analyzed"],
        "threshold": {h: sorted(v) for h, v in f["threshold"].items()},
        "settings_variants": {k: v for k, v in f["settings"].items()},
        "data_range": f["data_range"], "plan_scope": dict(f["scope"]),
        "trend_tests": {"n": len(trend), "significant_flag_follows": dict(flag_from),
                        "disagreements": [t for t in trend if t["follows"] != "both agree"]},
        "compare_tests": {"n": len(f["compare"]), "key_sets": dict(Counter(json.dumps(c["keys"]) for c in f["compare"]))},
        "timing": {"n": len(timing),
                   "day_boundary_hour": dict(Counter(t["day_boundary_hour"] for t in timing)),
                   "gap_min_session": dict(Counter(f"{t['event']}: gap={t['session_gap_minutes']}, min={t['min_session_minutes']}"
                                                   for t in timing)),
                   "edge_flagged_gt30": [t for t in timing if (t["edge_pct"] or 0) > 30],
                   "all": timing},
        "ungrounded_numbers": f["ungrounded"], "warnings": f["warnings"],
        "llm": llm, "llm_cache_files_by_tag": dict(all_cache), "llm_cache_files_total": sum(all_cache.values()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qa_config_facts.json").write_text(json.dumps(out, indent=1, default=str))
    summary = {k: v for k, v in out.items() if k not in ("timing", "trend_tests", "warnings")}
    summary["trend_tests"] = {k: v for k, v in out["trend_tests"].items() if k != "disagreements"}
    summary["timing"] = {k: v for k, v in out["timing"].items() if k != "all"}
    print(json.dumps(summary, indent=1, default=str))
    print("trend disagreements:")
    for t in out["trend_tests"]["disagreements"]:
        print("  ", t)
    print("warnings:", len(out["warnings"]))
    for w in out["warnings"]:
        print("  ", w)


if __name__ == "__main__":
    main()
