"""CLI for the behavioral analyst.

  # Run every question in the config (LLM + embeddings cached by default)
  python src/analyst/run.py run --questions metadata/analyst_questions.json

  # Only some homes / questions; re-call the LLM; or forbid any API call
  python src/analyst/run.py run --homes aruba --only sedentary_trend --refresh-llm
  python src/analyst/run.py run --offline

Outputs go to results/analyst/runs/<question set name>/ (the `name` field of the questions file).

  # Question × home table (summary.{md,csv}) + per-concept table (summary_concepts.{csv,md,tex})
  python src/analyst/run.py summarize

  # Build blinded rater packets from all runs, then score returned ratings
  python src/analyst/run.py packet --name v1
  python src/analyst/run.py score --name v1 --ratings path/to/ratings_*.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
# Checkpoints were pickled with `src.`-prefixed module paths, so the project root is needed too.
for _p in (str(_SRC), str(_SRC.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

from analyst.config import PROJECT_ROOT, RESULTS_DIR, AnalysisSettings  # noqa: E402

DEFAULT_QUESTIONS = PROJECT_ROOT / "metadata" / "analyst_questions.json"


def load_question_file(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def runs_dir(path: Path) -> Path:
    """results/analyst/runs/<question set name>/ — one folder per question file."""
    return RESULTS_DIR / "runs" / (load_question_file(path).get("name") or Path(path).stem)


def question_key(q: dict) -> str:
    """Run folder / output id: '<num:02d>_<id>' when the question is numbered, else '<id>'."""
    return f"{q['num']:02d}_{q['id']}" if "num" in q else q["id"]


def selected(q: dict, only: list | None) -> bool:
    """--only accepts ids, numbers, or full keys."""
    return not only or bool({q["id"], question_key(q), str(q.get("num"))} & {str(o) for o in only})


def load_questions(path: Path, homes: list | None, only: list | None) -> list:
    cfg = load_question_file(path)
    default_homes = cfg.get("homes", ["milan", "aruba", "cairo"])
    jobs = []
    for q in cfg["questions"]:
        if not selected(q, only):
            continue
        for h in q.get("homes", default_homes):
            if homes and h not in homes:
                continue
            jobs.append((question_key(q), q["question"], h))
    return jobs


def cmd_run(args):
    from analyst.pipeline import Analyst

    cache_mode = "offline" if args.offline else ("refresh" if args.refresh_llm else "use")
    analyst = Analyst(llm_model=args.model, cache_mode=cache_mode, settings=AnalysisSettings(),
                      runs_dir=runs_dir(args.questions))
    jobs = load_questions(args.questions, args.homes, args.only)
    for qid, question, home in jobs:
        analyst.run(qid, question, home, threshold=args.threshold, reuse_plan=args.reuse_plan)
    print(f"\n[analyst] {len(jobs)} runs done → {analyst.runs_dir}  (table: run.py summarize)")


def cmd_replot(args):
    from analyst.pipeline import Analyst

    analyst = Analyst(cache_mode="offline", settings=AnalysisSettings(), runs_dir=runs_dir(args.questions))
    jobs = load_questions(args.questions, args.homes, args.only)
    n = sum(analyst.replot(qid, home) for qid, _, home in jobs)
    print(f"[analyst] redrew {n} figures → {analyst.runs_dir}")


def cmd_packet(args):
    from analyst.config import get_home
    from analyst.index import SensorIndex
    from analyst.rater import build_packet

    settings = AnalysisSettings()
    rdir = runs_dir(args.questions)
    out_dir = rdir / "rater_packets" / args.name
    jobs = load_questions(args.questions, args.homes, args.only)
    by_home: dict = {}
    for qid, _, home in jobs:
        f = rdir / home / qid / "rater_items.json"
        if not f.exists():
            print(f"  skip {home}/{qid}: no run found")
            continue
        by_home.setdefault(home, []).extend(json.loads(f.read_text()))
    for home, items in by_home.items():
        paths = build_packet(SensorIndex(get_home(home)), items, out_dir, settings, args.name)
        print(f"[packet] {home}: {len(items)} items → {paths['html']}")


SHORT_ANSWER_SYSTEM = """\
You compress analysis report headlines into 1-3 word answers for a summary table.
For each item you get the user's question and the report headline. Return an answer of
1-3 words that answers the question as the headline does, e.g. "Yes, daily", "No clear trend",
"Decreasing", "Peaks evenings", "Rarely", "~20 min/day". Be faithful to the headline: keep its
direction and hedging ("no clear trend" stays "No clear trend"), add nothing new.
Copy each item's key exactly.
"""


def _short_answers(items: list, set_name: str, model: str) -> dict:
    """{key: 1-3 word answer} for items [{key, question, headline}], one cached LLM call."""
    from analyst.llm import GeminiClient
    from analyst.schemas import ShortAnswers

    if not items:
        return {}
    llm = GeminiClient(model=model, thinking_level="low")
    res, _ = llm.generate_json(tag=f"shortanswers_{set_name}", system=SHORT_ANSWER_SYSTEM,
                               user=json.dumps(items, indent=1), schema=ShortAnswers)
    return {a.key: a.answer.strip() for a in res.answers}


def cmd_summarize(args):
    import pandas as pd

    from analyst.summary import primary_stat as _primary_stat, status as _status, trend_arrows as _trend_arrows

    qfile = load_question_file(args.questions)
    rdir = runs_dir(args.questions)
    homes = args.homes or qfile.get("homes", ["milan", "aruba", "cairo"])
    questions = [q for q in qfile["questions"] if selected(q, args.only)]

    results = {}
    for q in questions:
        for home in q.get("homes", homes):
            f = rdir / home / question_key(q) / "result.json"
            if home in homes and f.exists():
                results[(q["id"], home)] = json.loads(f.read_text())

    to_shorten = [{"key": f"{home}/{qid}", "question": r["question"], "headline": r["report"]["headline"]}
                  for (qid, home), r in results.items() if _status(r) != "undetectable"]
    short = _short_answers(to_shorten, rdir.name, args.model)

    # One row per question, one column per home.
    table = []
    for q in questions:
        row = {"#": q.get("num", ""), "id_sonia": q.get("id_sonia", ""), "question_id": question_key(q),
               "category": q.get("category", ""),
               "group": q.get("group", ""), "question": q["question"]}
        for home in homes:
            r = results.get((q["id"], home))
            status = _status(r) if r else "not run"
            analyzed = status in ("detectable", "partial")
            arrows = _trend_arrows(r) if analyzed and q.get("category") == "change" else []
            row[f"{home}_status"] = status
            row[f"{home}_arrows"] = "".join(a for _, a in arrows)
            row[f"{home}_trend"] = "; ".join(f"{n} {a}" for n, a in arrows)
            row[f"{home}_result"] = short.get(f"{home}/{q['id']}", "") if analyzed else ""
        table.append(row)
    qt = pd.DataFrame(table)
    qt.drop(columns=[f"{h}_arrows" for h in homes]).to_csv(rdir / "summary.csv", index=False)

    mark = {"detectable": "✓", "partial": "◐", "undetectable": "✗", "not run": "–"}
    md = qt[["category", "id_sonia", "question"]].rename(columns={"id_sonia": "id"})
    for home in homes:
        md[home] = [mark[s] if s in ("undetectable", "not run") else " ".join(x for x in (mark[s], arrows, res) if x)
                    for s, arrows, res in zip(qt[f"{home}_status"], qt[f"{home}_arrows"], qt[f"{home}_result"])]
    legend = ("✓ detectable · ◐ partly detectable (some sub-behaviors skipped) · ✗ undetectable · – not run\n\n"
              "Arrows (behavior-change questions only; one per analyzed behavior, in plan order — see "
              "summary.csv `<home>_trend` for names): ↑ / ↓ significant increase / decrease of the daily "
              "share of active time (Kendall trend test, or Mann-Whitney for period comparisons, p < 0.05); "
              "for timing questions (e.g. wake-up time) ↑ / ↓ = significantly later / earlier; "
              "→ no significant change.")
    (rdir / "summary.md").write_text(f"# {rdir.name}\n\n{legend}\n\n" + md.to_markdown(index=False) + "\n")
    print(md.to_string(index=False))

    rows = []
    for (_, home), r in results.items():
        for c in r["concepts"]:
            skipped = "skipped" in c
            ov = {} if skipped else c["stats"]["overall"]
            rows.append({
                "home": home, "question_id": r["question_id"], "concept": c["concept"]["name"],
                "detectable": c["concept"]["detectable"], "n_queries": len(c["concept"]["retrieval_queries"]),
                "matched_windows": None if skipped else c["retrieval"]["n_matched_windows"],
                "min_per_day": ov.get("matched_min_per_day_mean"), "share_pct": ov.get("share_pct_mean"),
                "primary": "not detectable" if skipped else _primary_stat(c),
                "headline": r["report"]["headline"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(rdir / "summary_concepts.csv", index=False)
    (rdir / "summary_concepts.md").write_text(df.drop(columns=["headline"]).to_markdown(index=False))
    cols = ["home", "question_id", "concept", "min_per_day", "share_pct", "primary"]
    (rdir / "summary_concepts.tex").write_text(df[cols].to_latex(index=False, escape=True))
    print(f"→ {rdir}/summary.{{md,csv}} (per question × home), summary_concepts.{{csv,md,tex}} (per concept)")
    cmd_dashboard(args)


def cmd_dashboard(args):
    from analyst.dashboard import build_dashboard

    out = build_dashboard(runs_dir(args.questions), load_question_file(args.questions), question_key)
    print(f"→ {out}")


def cmd_review_packet(args):
    from analyst.review import build_review_packet

    homes = args.homes or [h for h in ("milan", "aruba", "cairo")]
    out = build_review_packet(runs_dir(args.questions), load_question_file(args.questions), question_key, args.name,
                              homes, AnalysisSettings())
    print(f"→ {out}  (zip this folder; the answer key is in {out.name}_key/)")


def cmd_review_score(args):
    from analyst.review import score_review, write_review_report

    base = runs_dir(args.questions) / "review"
    res = score_review(base / f"{args.name}_key", [Path(p) for p in args.ratings])
    out = write_review_report(res, base / f"{args.name}_results")
    print(json.dumps(res["consensus"]["funnel"]["nodes"], indent=1))
    print(f"→ {out}")
    cmd_dashboard(args)


def cmd_score(args):
    from analyst.rater import score_ratings

    pdir = runs_dir(args.questions) / "rater_packets" / args.name
    res = score_ratings(sorted(pdir.glob("*_key.csv")), [Path(p) for p in args.ratings])
    (pdir / "scores.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("raters", "overall", "cohen_kappa")}, indent=1))
    print(f"→ {pdir / 'scores.json'}")
    cmd_dashboard(args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "replot", "packet", "summarize"):
        sp = sub.add_parser(name)
        sp.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
        sp.add_argument("--homes", nargs="+")
        sp.add_argument("--only", nargs="+", help="question ids")
        if name in ("run", "summarize"):
            sp.add_argument("--model", default="gemini-3.1-pro-preview")
        if name == "run":
            sp.add_argument("--threshold", type=float, default=None, help="override per-home γ")
            g = sp.add_mutually_exclusive_group()
            g.add_argument("--refresh-llm", action="store_true", help="ignore LLM cache and re-call the API")
            g.add_argument("--offline", action="store_true", help="fail on LLM cache miss instead of calling the API")
            sp.add_argument("--reuse-plan", action="store_true",
                            help="keep each run's saved plan (no planner call); re-run retrieval, stats, report")
        elif name == "packet":
            sp.add_argument("--name", default="v1")
    sp = sub.add_parser("score")
    sp.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    sp.add_argument("--name", default="v1")
    sp.add_argument("--ratings", nargs="+", required=True)
    sp = sub.add_parser("dashboard")
    sp.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    sp = sub.add_parser("review-packet", help="4-step review app (answerable / queries / retrieval / findings)")
    sp.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    sp.add_argument("--homes", nargs="+")
    sp.add_argument("--name", default="review_v1")
    sp = sub.add_parser("review-score", help="score review CSVs → Sankey funnel report")
    sp.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    sp.add_argument("--name", default="review_v1")
    sp.add_argument("--ratings", nargs="+", required=True)
    args = p.parse_args()
    {"run": cmd_run, "replot": cmd_replot, "packet": cmd_packet, "summarize": cmd_summarize, "score": cmd_score,
     "dashboard": cmd_dashboard, "review-packet": cmd_review_packet,
     "review-score": cmd_review_score}[args.cmd](args)


if __name__ == "__main__":
    main()
