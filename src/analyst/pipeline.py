"""End-to-end analyst run: question -> plan -> retrieval -> stats -> figure -> report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .analysis import retrieval_summary, run_analyses
from .config import DEFAULT_LLM_MODEL, RESULTS_DIR, AnalysisSettings, get_home
from .home_context import build_home_context, home_metadata, room_names
from .index import SensorIndex
from .llm import CacheMode, GeminiClient
from .planner import make_plan
from .plots import plot_run
from .rater import sample_items
from .reporter import build_reporter_payload, ungrounded_numbers, write_report
from .retrieval import retrieve_concept, scope_mask
from .schemas import AnalysisPlan, Concept


def _parse_date(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    try:
        return pd.Timestamp(s).normalize()
    except (ValueError, TypeError):
        return None


def _sanitize_plan(plan: AnalysisPlan, home: str) -> list:
    """Drop invalid room / sensor filters and excess queries; return a list of warnings."""
    warnings, rooms, sensors = [], set(room_names(home)), set(home_metadata(home).get("sensor_location", {}))
    for c in plan.concepts:
        bad = [r for r in c.rooms if r not in rooms]
        if bad:
            warnings.append(f"{c.id}: dropped unknown rooms {bad}")
            c.rooms = [r for r in c.rooms if r in rooms]
        bad = [s for s in c.required_sensors if s not in sensors]
        if bad:
            warnings.append(f"{c.id}: dropped unknown sensors {bad}")
            c.required_sensors = [s for s in c.required_sensors if s in sensors]
        if len(c.retrieval_queries) > 6:
            c.retrieval_queries = c.retrieval_queries[:6]
            warnings.append(f"{c.id}: truncated retrieval queries to 6")
    return warnings


class Analyst:
    def __init__(
        self,
        llm_model: str = DEFAULT_LLM_MODEL,
        cache_mode: CacheMode = "use",
        settings: Optional[AnalysisSettings] = None,
        runs_dir: Path = RESULTS_DIR / "runs",
    ):
        self.llm = GeminiClient(model=llm_model, cache_mode=cache_mode)
        self.settings = settings or AnalysisSettings()
        self.runs_dir = Path(runs_dir)
        self._indexes: Dict[str, SensorIndex] = {}

    def index(self, home: str) -> SensorIndex:
        if home not in self._indexes:
            self._indexes[home] = SensorIndex(get_home(home))
        return self._indexes[home]

    def run_dir(self, home: str, question_id: str) -> Path:
        return self.runs_dir / home / question_id

    def run(self, question_id: str, question: str, home: str, threshold: Optional[float] = None,
            reuse_plan: bool = False) -> dict:
        cfg = get_home(home)
        index = self.index(home)
        threshold = cfg.threshold if threshold is None else threshold
        out_dir = self.run_dir(home, question_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[analyst] {home} / {question_id}: {question}")

        d0, d1 = index.date_range
        prev = out_dir / "result.json"
        if reuse_plan and prev.exists():
            old = json.loads(prev.read_text())
            plan, plan_info = AnalysisPlan.model_validate(old["plan"]), old["llm"]["planner"]
        else:
            plan, plan_info = make_plan(self.llm, question, home, build_home_context(cfg, d0, d1))
        warnings = _sanitize_plan(plan, home)
        (out_dir / "plan.json").write_text(plan.model_dump_json(indent=2))
        for c in plan.concepts:
            filters = c.time_of_day + c.days_of_week + c.rooms + c.required_sensors
            print(f"  concept {c.id!r} detectable={c.detectable} filters={filters}")
            for q in c.retrieval_queries:
                print(f"    · {q}")

        scope = scope_mask(index, _parse_date(plan.scope_start_date), _parse_date(plan.scope_end_date))
        if scope.sum() == 0:
            warnings.append("plan scope contains no data; using full data range")
            scope = scope_mask(index, None, None)
        scope_range = (index.meta.loc[scope, "start"].min(), index.meta.loc[scope, "start"].max())

        concept_results, dailies, matched, timing_matched, rater_items = [], {}, {}, {}, []
        for c in plan.concepts:
            if not c.detectable:
                concept_results.append({"concept": c.model_dump(), "skipped": "not detectable"})
                print("  → skipped (planner: not detectable)")
                continue
            cm = retrieve_concept(index, c, scope, threshold)
            cm_any = None
            if any(a.type == "timing" for a in plan.analyses) and c.time_of_day:
                cm_any = retrieve_concept(index, c.model_copy(update={"time_of_day": []}), scope, threshold)
            stats, daily = run_analyses(index, scope, cm, plan.analyses, self.settings, cm_any)
            dailies[c.id], matched[c.id] = daily, cm.matched
            timing_matched[c.id] = (cm_any or cm).matched
            # Ground-truth label mix is a diagnostic only and must never reach the reporter payload.
            label_mix = (cm.matched["label_l1"].value_counts(normalize=True).head(6).round(3).to_dict()
                         if len(cm.matched) else {})
            concept_results.append({"concept": c.model_dump(), "retrieval": retrieval_summary(cm), "stats": stats,
                                    "diagnostics": {"gt_label_mix_matched": label_mix}})
            daily.to_csv(out_dir / f"daily_{c.id}.csv")
            cm.matched[["sample_id", "start", "end", "tod", "primary_room", "score", "best_query", "label_l1"]].to_csv(
                out_dir / f"matches_{c.id}.csv", index=False)
            rater_items.extend(sample_items(f"{home}/{question_id}", cm, self.settings))
            ov = stats["overall"]
            print(f"  → {cm.matched.shape[0]:,} matched windows, {ov['matched_min_per_day_mean']} min/day, "
                  f"{ov['share_pct_mean']}% of observed active time")

        payload = build_reporter_payload(question, home, scope_range, plan, concept_results)
        report, report_info = write_report(self.llm, home, payload)
        flagged = ungrounded_numbers(report, payload)
        if flagged:
            warnings.append(f"report contains numbers not found in statistics: {flagged}")
        analyzed_ids = {r["concept"]["id"] for r in concept_results if "skipped" not in r}
        bad = [f.concept_id for f in report.findings if f.concept_id not in analyzed_ids]
        if bad:
            warnings.append(f"findings reference non-analyzed concepts: {bad}")

        analyzed = [r for r in concept_results if "skipped" not in r]
        if analyzed:
            plot_run(index, question, analyzed, dailies, matched, out_dir / "figure",
                     rolling_days=self.settings.rolling_days, timing_matched=timing_matched)
        else:
            for ext in ("png", "pdf"):
                (out_dir / f"figure.{ext}").unlink(missing_ok=True)
        (out_dir / "rater_items.json").write_text(json.dumps(rater_items, indent=1))

        result = {
            "question_id": question_id,
            "question": question,
            "home": home,
            "data_range": [str(d0)[:10], str(d1)[:10]],
            "scope_range": [str(scope_range[0])[:10], str(scope_range[1])[:10]],
            "threshold": threshold,
            "settings": asdict(self.settings),
            "plan": plan.model_dump(),
            "concepts": concept_results,
            "report": report.model_dump(),
            "ungrounded_numbers": flagged,
            "warnings": warnings,
            "llm": {"planner": plan_info, "reporter": report_info},
        }
        (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=_json_default))
        (out_dir / "report.md").write_text(_report_md(result))
        print(f"  report: {report.headline}")
        for w in warnings:
            print(f"  ⚠ {w}")
        return result

    def replot(self, question_id: str, home: str) -> bool:
        """Redraw figure.{png,pdf} from a finished run's saved artifacts (no LLM calls)."""
        out_dir = self.run_dir(home, question_id)
        r = json.loads((out_dir / "result.json").read_text())
        analyzed = [c for c in r["concepts"] if "skipped" not in c]
        if not analyzed:
            return False
        index = self.index(home)
        plan = r["plan"]
        scope = scope_mask(index, _parse_date(plan["scope_start_date"]), _parse_date(plan["scope_end_date"]))
        if scope.sum() == 0:
            scope = scope_mask(index, None, None)
        has_timing = any(a["type"] == "timing" for a in plan["analyses"])
        dailies, matched, timing_matched = {}, {}, {}
        for cr in analyzed:
            c = Concept(**cr["concept"])
            dailies[c.id] = pd.read_csv(out_dir / f"daily_{c.id}.csv", index_col=0, parse_dates=True)
            m = pd.read_csv(out_dir / f"matches_{c.id}.csv", parse_dates=["start", "end"])
            matched[c.id] = timing_matched[c.id] = m
            if has_timing and c.time_of_day:
                timing_matched[c.id] = retrieve_concept(
                    index, c.model_copy(update={"time_of_day": []}), scope, r["threshold"]).matched
        plot_run(index, r["question"], analyzed, dailies, matched, out_dir / "figure",
                 rolling_days=self.settings.rolling_days, timing_matched=timing_matched)
        return True


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def _report_md(r: dict) -> str:
    lines = [
        f"# {r['question']}",
        f"*{r['home'].capitalize()} · data {r['scope_range'][0]} → {r['scope_range'][1]} · γ = {r['threshold']}*",
        "",
        f"**{r['report']['headline']}**",
        "",
        r["report"]["body"],
        "",
    ]
    names = {c["concept"]["id"]: c["concept"]["name"] for c in r["concepts"]}
    if r["report"].get("findings"):
        lines += ["**Findings:**", ""]
        lines += [f"{i}. {f['statement']} _({names.get(f['concept_id'], f['concept_id'])}; {f['kind']})_"
                  for i, f in enumerate(r["report"]["findings"], 1)]
        lines.append("")
    if r["report"]["limitations"]:
        lines += ["**Limitations:** " + " ".join(r["report"]["limitations"]), ""]
    if any("skipped" not in c for c in r["concepts"]):
        lines += ["![figure](figure.png)", ""]
    lines += ["## Plan", "", f"_{r['plan']['interpretation']}_", ""]
    for c in r["concepts"]:
        cc = c["concept"]
        lines += [
            f"### {cc['name']} (`{cc['id']}`)",
            f"- Definition: {cc['physical_definition']}",
            f"- Detectable: {cc['detectable']} — {cc['detectability_note']}",
        ]
        if "skipped" in c:
            lines += [f"- Skipped: {c['skipped']}", ""]
            continue
        ret = c["retrieval"]
        lines += [
            f"- Filters: tod={cc['time_of_day'] or 'any'}, dow={cc['days_of_week'] or 'any'}, "
            f"rooms={cc['rooms'] or 'any'}, sensors={cc['required_sensors'] or 'any'}",
            f"- Matched windows: {ret['n_matched_windows']:,} / {ret['n_eligible_windows']:,} eligible "
            f"({ret['n_matched_without_structural_filters']:,} without room/sensor filters)",
        ]
        lines += [f"  - `{q}` → {n:,}" for q, n in ret["per_query_matched_windows"].items()]
        lines.append("")
    if r["plan"]["caveats"]:
        lines += ["**Planner caveats:** " + " ".join(r["plan"]["caveats"]), ""]
    if r["warnings"]:
        lines += ["**Warnings:** " + "; ".join(r["warnings"]), ""]
    return "\n".join(lines)
