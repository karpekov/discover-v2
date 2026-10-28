"""Four-step human review of the analyst pipeline (per home, cascading).

  1  answerable   — per question (blind to the analyst's verdict): can it be answered with this home's
                    sensors? Agreement with the analyst's verdict is computed at scoring time.
  2  queries      — per data query of each analyzed behavior: is it relevant?
  3  retrieval    — per behavior: 4 top + 2 random matched windows, 2 far negatives
                    (bottom 10% similarity anywhere in the data); does the window show the behavior?
                    All samples of a home are shuffled together; the question is not shown.
  4  findings     — per atomic report finding: is it correct given the figure and statistics?

All steps are reviewed regardless of earlier answers; the funnel (Sankey) applies the gating.
The package is a static folder (index.html + data/*.js + img/) that works from file://.
The answer key (strata, scores, analyst verdicts) is written next to it, never inside it.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import scipy.stats as sps
from PIL import Image

from .config import TOD_BUCKETS, AnalysisSettings, get_home
from .home_context import build_home_context, sensor_coordinates
from .index import SensorIndex
from .plots import plot_hourly
from .rater import _fleiss, _item_payload, _kappa, _pick_separated, _seed, _wilson
from .retrieval import scope_mask
from .summary import primary_stat, status

APP_DIR = Path(__file__).with_name("review_app")
STEPS = {1: "answerable", 2: "queries", 3: "retrieval", 4: "findings"}


# --------------------------------------------------------------------------- packet

def _p(p) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def _verdict(delta: float, p, up: str = "increase", down: str = "decrease") -> str:
    """'statistically significant increase (p = 0.004)' / 'no statistically significant change (p = 0.19)'."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "not testable"
    if p >= 0.05 or delta == 0:
        return f"no statistically significant change ({_p(p)})"
    return f"statistically significant {up if delta > 0 else down} ({_p(p)})"


def _period_lines(daily: pd.DataFrame, periods: List[dict] | None = None) -> List[str]:
    """Plain-language before/after comparison of the daily amount (first vs last third unless periods given)."""
    d = daily[daily["valid"] & daily["share_pct"].notna()]
    if periods:
        parts = [d.loc[p["start_date"]:p["end_date"]] for p in (periods[0], periods[-1])]
        labels = [periods[0]["label"], periods[-1]["label"]]
    elif len(d) >= 6:
        idx = np.array_split(np.arange(len(d)), 3)
        parts, labels = [d.iloc[idx[0]], d.iloc[idx[-1]]], ["First third", "Last third"]
    else:
        return []
    a, b = parts
    if len(a) < 3 or len(b) < 3:
        return []
    fmt = lambda x: f"{x.index[0]:%b %d}–{x.index[-1]:%b %d}, {len(x)} days"
    out = [f"{labels[0].capitalize()} ({fmt(a)}): {a['matched_min'].mean():.1f} min/day, "
           f"{a['share_pct'].mean():.1f}% of active time, {a['episodes'].mean():.1f} episodes/day",
           f"{labels[1].capitalize()} ({fmt(b)}): {b['matched_min'].mean():.1f} min/day, "
           f"{b['share_pct'].mean():.1f}% of active time, {b['episodes'].mean():.1f} episodes/day"]
    for col, unit, name in (("matched_min", " min/day", "Minutes per day"), ("share_pct", " percentage points", "Share of active time")):
        ma, mb = a[col].mean(), b[col].mean()
        p = sps.mannwhitneyu(a[col], b[col], alternative="two-sided").pvalue
        rel = f", {100 * (mb - ma) / ma:+.0f}%" if ma > 0 else ""
        out.append(f"{name}, {labels[1].lower()} vs {labels[0].lower()}: {_verdict(mb - ma, p)}; "
                   f"difference {mb - ma:+.1f}{unit}{rel}")
    return out


def _daily(run_dir: Path, c: dict) -> pd.DataFrame | None:
    fp = run_dir / f"daily_{c['concept']['id']}.csv"
    return pd.read_csv(fp, parse_dates=["date"], index_col="date") if fp.exists() else None


def _key_stats(c: dict, daily: pd.DataFrame | None = None) -> List[str]:
    ov = c["stats"]["overall"]
    out = [f"Average {ov['matched_min_per_day_mean']} matched minutes per day (one-minute sensor windows matching this "
           f"behavior, not its exact duration), grouped into {ov['episodes_per_day_mean']} episodes per day "
           f"(matches less than 2 min apart count as one episode; median episode {ov['episode_minutes_median']} min) · "
           f"{ov['share_pct_mean']}% of all sensor-active time in its time window · {ov['n_valid_days']} valid days"]
    for a in c["stats"]["analyses"]:
        if a.get("insufficient_data"):
            continue
        if a["type"] == "trend":
            for key, name in (("minutes", "minutes per day"), ("share", "share of active time")):
                s = a[key]
                p = s["kendall_p_value"]
                out.append(f"Trend in {name} ({a['first_day']} → {a['last_day']}): "
                           f"{_verdict(s['fitted_end'] - s['fitted_start'], p)}; fitted line "
                           f"{s['fitted_start']} → {s['fitted_end']}" + ("%" if key == "share" else " min/day"))
            if daily is not None and not any(x["type"] == "compare_periods" for x in c["stats"]["analyses"]):
                out += _period_lines(daily)
        elif a["type"] == "compare_periods":
            if daily is not None:
                out += _period_lines(daily, a.get("periods"))
        elif a["type"] == "timing" and a["event"] != "all":
            dt = a["daily_time"]
            out.append(f"{a['event'].capitalize()} occurrence (start window {a['start_window']}): median {dt['median']}, "
                       f"IQR {dt['iqr_minutes']} min, seen on {a['days_with_event_pct']}% of days")
            if "trend" in a:
                t = a["trend"]
                s = t["slope_minutes_per_week"]
                out.append(f"Shift of the daily time: {_verdict(s, t['kendall_p_value'], 'shift later', 'shift earlier')}; "
                           f"{abs(s):.1f} min {'later' if s > 0 else 'earlier'} per week")
            if "compare" in a:
                cmp_ = a["compare"]
                d = cmp_["median_shift_minutes"]
                out.append(f"Median time, {cmp_['comparison']} vs {cmp_['baseline']}: "
                           f"{_verdict(d, cmp_['mann_whitney_p_value'], 'shift later', 'shift earlier')}; "
                           f"{abs(d):.0f} min {'later' if d > 0 else 'earlier'}")
        elif a["type"] == "timing":
            out.append("Session peaks: " + ", ".join(f"{p['hour']} ({p['pct_of_days']}% of days)" for p in a["peak_hours"]))
        elif a["type"] in ("time_of_day_profile", "day_of_week_profile"):
            b = sorted(a["buckets"], key=lambda x: -(x["share_pct"] or 0))[:3]
            out.append(("Time-of-day" if a["type"] == "time_of_day_profile" else "Day-of-week") + " peaks: "
                       + ", ".join(f"{x['bucket'].replace('_', ' ')} {x['share_pct']}%" for x in b))
    return out


def _filters(cc: dict) -> List[str]:
    out = []
    if cc["time_of_day"]:
        out.append("time: " + ", ".join(f"{b.replace('_', ' ')} ({TOD_BUCKETS[b]})" for b in cc["time_of_day"]))
    if cc["days_of_week"]:
        out.append("days: " + ", ".join(cc["days_of_week"]))
    if cc["rooms"]:
        out.append("must visit: " + ", ".join(cc["rooms"]))
    if cc["required_sensors"]:
        out.append("must trigger sensor: " + ", ".join(cc["required_sensors"]))
    return out


def _hourly_row(run_dir: Path, cc: dict) -> dict:
    m = pd.read_csv(run_dir / f"matches_{cc['id']}.csv", parse_dates=["start", "end"])
    daily = pd.read_csv(run_dir / f"daily_{cc['id']}.csv", parse_dates=["date"])
    valid = set(daily.loc[daily["valid"], "date"].dt.date)
    return {"name": cc["name"], "matches": m[m["start"].dt.date.isin(valid)], "n_days": len(valid)}


def _sample_windows(index: SensorIndex, run_dir: Path, result: dict, c: dict, n_top: int, n_random: int,
                    n_negative: int, neg_quantile: float, seed: int) -> List[dict]:
    cc = c["concept"]
    m = pd.read_csv(run_dir / f"matches_{cc['id']}.csv", parse_dates=["start", "end"])
    if m.empty:
        return []
    plan = result["plan"]
    scope = scope_mask(index, pd.Timestamp(plan["scope_start_date"]) if plan["scope_start_date"] else None,
                       pd.Timestamp(plan["scope_end_date"]) if plan["scope_end_date"] else None)
    best = index.similarity(cc["retrieval_queries"]).max(axis=0)
    cut = np.quantile(best[scope], neg_quantile)
    far = index.meta.loc[scope & (best <= cut), ["sample_id", "start", "end"]].copy()
    far["score"] = best[scope & (best <= cut)]
    taken: List[pd.Timestamp] = []
    strata = [("top", m.sort_values("score", ascending=False), n_top),
              ("random", m.sample(frac=1.0, random_state=seed % 2**32), n_random),
              ("negative", far.sample(frac=1.0, random_state=(seed + 1) % 2**32), n_negative)]
    out = []
    for stratum, pool, n in strata:
        for _, row in _pick_separated(pool, n, taken).iterrows():
            out.append({"stratum": stratum, "sample_id": row["sample_id"], "start": row["start"].isoformat(),
                        "end": row["end"].isoformat(), "score": round(float(row["score"]), 4)})
    random.Random(seed).shuffle(out)
    return out


def build_review_packet(rdir: Path, qfile: dict, question_key: Callable[[dict], str], name: str, homes: List[str],
                        settings: AnalysisSettings, n_top: int = 4, n_random: int = 2, n_negative: int = 2,
                        neg_quantile: float = 0.10) -> Path:
    out = rdir / "review" / name
    key_dir = rdir / "review" / f"{name}_key"
    if out.exists():
        shutil.rmtree(out)
    for sub in ("data", "img"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    for f in APP_DIR.iterdir():
        shutil.copy(f, out / f.name)

    manifest = {"packet": name, "created": datetime.now().strftime("%Y-%m-%d %H:%M"), "homes": []}
    keys = [question_key(q) for q in qfile["questions"]]
    random.Random(_seed(settings.seed, "question_order")).shuffle(keys)
    order = {k: i for i, k in enumerate(keys)}
    for home in homes:
        cfg, index = get_home(home), SensorIndex(get_home(home))
        d0, d1 = index.date_range
        (out / "img" / home).mkdir(exist_ok=True)
        fp = cfg.floor_plan
        if fp.exists():
            shutil.copy(fp, out / "img" / f"{home}_floor.png")
        w, h = Image.open(fp).size if fp.exists() else (2600, 2800)
        key_rows, questions, samples = [], [], []
        for q in qfile["questions"]:
            if q.get("homes") and home not in q["homes"]:
                continue
            qkey = question_key(q)
            run_dir = rdir / home / qkey
            if not (run_dir / "result.json").exists():
                continue
            r = json.loads((run_dir / "result.json").read_text())
            st = status(r)
            answerable = st != "undetectable"
            num = q.get("num", len(questions) + 1)
            base = {"home": home, "question_id": qkey}
            qd = {"key": qkey, "num": num, "id_sonia": q.get("id_sonia", ""), "category": q.get("category", ""),
                  "question": q["question"], "interpretation": r["plan"]["interpretation"]}

            sid = f"{home}-1-{num}"
            qd["step1"] = {"id": sid}
            key_rows.append({**base, "step": 1, "item_id": sid, "concept_id": "", "analyst_answerable": answerable,
                             "status": st})

            analyzed = [c for c in r["concepts"] if "skipped" not in c]
            qd["step2"] = []
            for c in analyzed:
                cc = c["concept"]
                beh = {"concept_id": cc["id"], "name": cc["name"], "definition": cc["physical_definition"],
                       "filters": _filters(cc)}
                items = []
                for i, text in enumerate(cc["retrieval_queries"], 1):
                    iid = f"{home}-2-{num}-{cc['id']}-{i}"
                    items.append({"id": iid, "query": text})
                    key_rows.append({**base, "step": 2, "item_id": iid, "concept_id": cc["id"], "query": text})
                qd["step2"].append({**beh, "items": items})

                seed = _seed(settings.seed, name, home, qkey, cc["id"])
                tod = (" (" + ", ".join(f"{b.replace('_', ' ')} {TOD_BUCKETS[b]}" for b in cc["time_of_day"]) + ")"
                       if cc["time_of_day"] else "")
                for wdw in _sample_windows(index, run_dir, r, c, n_top, n_random, n_negative, neg_quantile, seed):
                    samples.append(({"behavior": cc["name"] + tod, "definition": cc["physical_definition"],
                                     **_item_payload(index, wdw, settings.rater_context_seconds)},
                                    {**base, "step": 3, "concept_id": cc["id"], **wdw}))

            findings = r["report"].get("findings", [])
            names = {c["concept"]["id"]: c["concept"]["name"] for c in analyzed}
            fitems = []
            for i, f in enumerate(findings, 1):
                iid = f"{home}-4-{num}-{i}"
                fitems.append({"id": iid, "statement": f["statement"], "kind": f["kind"],
                               "behavior": names.get(f["concept_id"], f["concept_id"])})
                key_rows.append({**base, "step": 4, "item_id": iid, "concept_id": f["concept_id"],
                                 "statement": f["statement"], "kind": f["kind"]})
            fig = run_dir / "figure.png"
            if analyzed and fig.exists():
                shutil.copy(fig, out / "img" / home / f"{qkey}.png")
            hourly = None
            if analyzed:
                hourly = f"img/{home}/{qkey}_hourly.png"
                plot_hourly([_hourly_row(run_dir, c["concept"]) for c in analyzed], out / hourly)
            qd["step4"] = {"figure": f"img/{home}/{qkey}.png" if analyzed and fig.exists() else None, "hourly": hourly,
                           "stats": [{"name": c["concept"]["name"], "definition": c["concept"]["physical_definition"],
                                      "primary": primary_stat(c), "lines": _key_stats(c, _daily(run_dir, c))}
                                     for c in analyzed],
                           "items": fitems, "n_analyzed": len(analyzed)}
            questions.append(qd)

        questions.sort(key=lambda qd: order[qd["key"]])
        random.Random(_seed(settings.seed, name, home, "samples")).shuffle(samples)
        sample_items = []
        for k, (payload, key) in enumerate(samples, 1):
            iid = f"{home[:2]}-3-{k:03d}"
            sample_items.append({"id": iid, **payload})
            key_rows.append({**key, "item_id": iid})

        data = {
            "packet": name, "home": home, "data_range": [f"{d0:%Y-%m-%d}", f"{d1:%Y-%m-%d}"],
            "house_context": build_home_context(cfg, d0, d1, n_examples=0).split("## Example window captions")[0].strip(),
            "image": {"src": f"img/{home}_floor.png", "w": w, "h": h},
            "sensors": {k: [v[0], h - v[1]] for k, v in sensor_coordinates(home).items()},
            "questions": questions,
            "samples": sample_items,
        }
        (out / "data" / f"{home}.js").write_text(
            f"window.REVIEW_DATA = window.REVIEW_DATA || {{}};\nwindow.REVIEW_DATA[{json.dumps(home)}] = "
            + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
        pd.DataFrame(key_rows).to_csv(key_dir / f"{home}_key.csv", index=False)
        counts = pd.DataFrame(key_rows)["step"].value_counts().sort_index().to_dict()
        manifest["homes"].append({"home": home, "n_questions": len(questions),
                                  "n_items": {str(k): int(v) for k, v in counts.items()}})
        print(f"[review] {home}: {len(questions)} questions, items per step {counts}")
    (out / "data" / "manifest.js").write_text("window.REVIEW_MANIFEST = " + json.dumps(manifest) + ";\n")
    return out


N_DECOYS = 30
_DECOY_GAP = pd.Timedelta(minutes=30)


def _norm_text(text: str) -> str:
    return " ".join(text.lower().split())


def _load_captions(path: Path) -> Dict[str, str]:
    raw = json.loads(path.read_text())
    items = raw["captions"] if isinstance(raw, dict) else raw
    out = {}
    for it in items:
        caps = [c.strip() for c in (it.get("captions") or []) if c and str(c).strip()]
        if caps and it.get("sample_id"):
            out[it["sample_id"]] = caps[0]
    return out


def _read_home_js(path: Path) -> tuple:
    text = path.read_text(encoding="utf-8")
    key = "window.REVIEW_DATA["
    start = text.index(key)
    eq = text.index("=", start)
    payload = text[eq + 1:].strip()
    if payload.endswith(";"):
        payload = payload[:-1]
    return text[:eq + 1], json.loads(payload)


def _pick_far(meta: pd.DataFrame, score: np.ndarray, eligible: np.ndarray, captions: Dict[str, str],
              blocked: set, used_ids: set, taken: List[pd.Timestamp], k: int) -> List[dict]:
    chosen = []
    for i in np.argsort(score, kind="stable"):
        if not eligible[i]:
            continue
        sid = meta["sample_id"].iloc[i]
        if sid in used_ids or sid not in captions:
            continue
        text = captions[sid]
        norm = _norm_text(text)
        if norm in blocked:
            continue
        start = meta["start"].iloc[i]
        if any(abs(start - t) < _DECOY_GAP for t in taken):
            continue
        chosen.append({"sample_id": sid, "query": text, "score": round(float(score[i]), 4),
                       "start": start.isoformat(), "end": meta["end"].iloc[i].isoformat()})
        used_ids.add(sid)
        blocked.add(norm)
        taken.append(start)
        if len(chosen) >= k:
            break
    if len(chosen) < k:
        raise RuntimeError(f"only found {len(chosen)} of {k} far captions")
    return chosen


def add_decoy_queries(rdir: Path, qfile: dict, question_key: Callable[[dict], str],
                      src_name: str = "review_v1", dst_name: str = "review_v2", n_decoys: int = N_DECOYS) -> Path:
    """Copy a scored review packet and add far-caption decoys to step 2.

    One decoy per question, with the leftover slots (so the total is ``n_decoys``) going to the
    questions that already have the most planner queries. A decoy is the first training caption of
    a window whose sensor embedding is far from that question's queries (or from every query in
    the home, when the question has none). Existing item ids and ratings are kept. Decoys are
    unlabeled in the review app and tagged ``stratum=decoy`` in the key.
    """
    src = rdir / "review" / src_name
    dst = rdir / "review" / dst_name
    src_key = rdir / "review" / f"{src_name}_key"
    dst_key = rdir / "review" / f"{dst_name}_key"
    if not src.exists() or not src_key.exists():
        raise FileNotFoundError(f"missing {src} or {src_key}")
    if dst.exists():
        shutil.rmtree(dst)
    if dst_key.exists():
        shutil.rmtree(dst_key)
    shutil.copytree(src, dst)
    shutil.copytree(src_key, dst_key)

    for csv_path in sorted(dst.glob("*.csv")):
        ratings = pd.read_csv(csv_path)
        if "packet" in ratings.columns:
            ratings["packet"] = dst_name
            ratings.to_csv(csv_path, index=False)

    manifest = json.loads((dst / "data" / "manifest.js").read_text().split("=", 1)[1].strip().rstrip(";"))
    manifest["packet"] = dst_name
    for home in [h["home"] for h in manifest["homes"]]:
        cfg = get_home(home)
        index = SensorIndex(cfg)
        captions = _load_captions(cfg.captions_path)
        plans = []
        for q in qfile["questions"]:
            if q.get("homes") and home not in q["homes"]:
                continue
            qkey = question_key(q)
            result_path = rdir / home / qkey / "result.json"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text())
            concepts = []
            queries = []
            for c in result["concepts"]:
                cc = c["concept"]
                qs = [t for t in (cc.get("retrieval_queries") or []) if t and str(t).strip()]
                concepts.append({"id": cc["id"], "name": cc["name"], "definition": cc["physical_definition"],
                                 "filters": _filters(cc), "queries": qs, "analyzed": "skipped" not in c})
                if "skipped" not in c:
                    queries.extend(qs)
            plan = result["plan"]
            plans.append({"key": qkey, "num": q.get("num", 0), "queries": queries, "concepts": concepts,
                          "scope_start": plan.get("scope_start_date"), "scope_end": plan.get("scope_end_date")})
        if len(plans) > n_decoys:
            raise RuntimeError(f"{home}: {len(plans)} questions is more than {n_decoys} decoys")
        extra = n_decoys - len(plans)
        ranked = sorted(plans, key=lambda p: (-len(p["queries"]), p["num"]))
        n_for = {p["key"]: 1 for p in plans}
        for p in ranked[:extra]:
            n_for[p["key"]] += 1

        texts = list(dict.fromkeys(t for p in plans for t in p["queries"]))
        sims = index.similarity(texts) if texts else None
        text_at = {t: i for i, t in enumerate(texts)}
        blocked = {_norm_text(t) for t in texts}
        used_ids: set = set()
        taken: List[pd.Timestamp] = []
        meta = index.meta.reset_index(drop=True)
        cap_ok = meta["sample_id"].isin(captions).to_numpy()
        picked: Dict[str, List[dict]] = {}
        ordered = sorted(plans, key=lambda p: (not p["queries"], p["num"]))
        for p in ordered:
            if p["queries"]:
                rows = [text_at[t] for t in p["queries"]]
                score = sims[rows].max(axis=0)
            else:
                score = sims.max(axis=0)
            scope = scope_mask(index, pd.Timestamp(p["scope_start"]) if p["scope_start"] else None,
                               pd.Timestamp(p["scope_end"]) if p["scope_end"] else None)
            picked[p["key"]] = _pick_far(meta, score, scope & cap_ok, captions, blocked, used_ids, taken, n_for[p["key"]])
            print(f"[decoy] {home} {p['key']}: " +
                  ", ".join(f"{c['score']:.3f}" for c in picked[p["key"]]))

        prefix, data = _read_home_js(dst / "data" / f"{home}.js")
        data["packet"] = dst_name
        by_key = {q["key"]: q for q in data["questions"]}
        plan_by = {p["key"]: p for p in plans}
        for qkey, choices in picked.items():
            qd = by_key[qkey]
            plan = plan_by[qkey]
            analyzed = [b for b in qd["step2"]]
            hosts = analyzed or []
            if not hosts:
                src_c = next(c for c in plan["concepts"])
                hosts = [{"concept_id": src_c["id"], "name": src_c["name"], "definition": src_c["definition"],
                          "filters": src_c["filters"], "items": []}]
                qd["step2"] = hosts
            for j, choice in enumerate(choices):
                host = hosts[min(j, len(hosts) - 1)]
                iid = f"{home}-2-{plan['num']}-decoy-{j + 1}"
                host["items"].append({"id": iid, "query": choice["query"]})
                choice["item_id"] = iid
                choice["concept_id"] = host["concept_id"]
        (dst / "data" / f"{home}.js").write_text(
            prefix + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")

        key = pd.read_csv(dst_key / f"{home}_key.csv")
        key.loc[key.step.eq(2) & key.stratum.isna(), "stratum"] = "query"
        extra_rows = []
        for qkey, choices in picked.items():
            for choice in choices:
                extra_rows.append({"home": home, "question_id": qkey, "step": 2, "item_id": choice["item_id"],
                                   "concept_id": choice["concept_id"], "query": choice["query"], "stratum": "decoy",
                                   "sample_id": choice["sample_id"], "start": choice["start"], "end": choice["end"],
                                   "score": choice["score"]})
        key = pd.concat([key, pd.DataFrame(extra_rows)], ignore_index=True)
        key.to_csv(dst_key / f"{home}_key.csv", index=False)
        n_step2 = int((key.step == 2).sum())
        for entry in manifest["homes"]:
            if entry["home"] == home:
                entry["n_items"]["2"] = n_step2
    (dst / "data" / "manifest.js").write_text("window.REVIEW_MANIFEST = " + json.dumps(manifest) + ";\n")
    print(f"[decoy] → {dst}")
    return dst


# --------------------------------------------------------------------------- scoring

def _vote(vals: List[str]) -> str:
    y, n = vals.count("yes"), vals.count("no")
    return "yes" if y > n else "no" if n > y else "unsure"


def _majority_pass(vals: List[str]):
    """True / False by majority of decided votes, None if nothing decided."""
    d = [v for v in vals if v in ("yes", "no")]
    if not d:
        return None
    return sum(v == "yes" for v in d) > len(d) / 2


FUNNEL_NODES = [
    ("all", "All questions"),
    ("s1_ok", "Answerable (agreed)"), ("s1_no_ok", "Not answerable (agreed)"),
    ("s1_false_ans", "Wrongly judged answerable"), ("s1_missed", "Missed: was answerable"), ("s1_unsure", "Step 1 unsure"),
    ("s2_ok", "Correct queries"), ("s2_bad", "Irrelevant queries"), ("s2_unsure", "Queries unsure"),
    ("s3_ok", "Good retrieval"), ("s3_bad", "Poor retrieval"), ("s3_unsure", "Retrieval unsure"),
    ("s4_ok", "Correct analysis"), ("s4_bad", "Incorrect analysis"), ("s4_unsure", "Analysis unsure / no findings"),
]


def _funnel_path(qkey_rows: pd.DataFrame, vote: Dict[str, str]) -> List[str]:
    """Sequence of funnel nodes one question × home passes through."""
    s1 = qkey_rows[qkey_rows.step == 1].iloc[0]
    v1 = vote.get(s1.item_id, "unsure")          # reviewer: is the question answerable in this home?
    ans = bool(s1.analyst_answerable)
    if v1 == "unsure":
        return ["all", "s1_unsure"]
    if not ans:
        return ["all", "s1_no_ok" if v1 == "no" else "s1_missed"]
    if v1 == "no":
        return ["all", "s1_false_ans"]
    path = ["all", "s1_ok"]
    s2 = qkey_rows[(qkey_rows.step == 2) & (qkey_rows.stratum.fillna("") != "decoy")]
    s3 = qkey_rows[(qkey_rows.step == 3) & qkey_rows.stratum.isin(["top", "random"])]
    for step, rows in ((2, s2), (3, s3), (4, qkey_rows[qkey_rows.step == 4])):
        ok = _majority_pass([vote.get(i, "unsure") for i in rows.item_id])
        path.append(f"s{step}_{'unsure' if ok is None else 'ok' if ok else 'bad'}")
        if ok is not True:
            break
    return path


def _funnel(key: pd.DataFrame, vote: Dict[str, str]) -> dict:
    links, nodes = defaultdict(int), defaultdict(int)
    per_question = []
    for (home, qid), rows in key.groupby(["home", "question_id"], sort=False):
        path = _funnel_path(rows, vote)
        per_question.append({"home": home, "question_id": qid, "path": path, "outcome": path[-1]})
        for n in path:
            nodes[n] += 1
        for a, b in zip(path, path[1:]):
            links[(a, b)] += 1
    return {"nodes": dict(nodes), "links": [{"source": a, "target": b, "value": v} for (a, b), v in links.items()],
            "per_question": per_question}


def _step_metrics(key: pd.DataFrame, votes: pd.DataFrame) -> dict:
    """Ungated item-level yes-rates per step (step 3 split by stratum)."""
    df = key.merge(votes, on="item_id", how="inner")
    s1 = df.step == 1
    agree = np.where(df["vote"] == "unsure", "unsure",
                     np.where((df["vote"] == "yes") == df["analyst_answerable"].astype(bool), "yes", "no"))
    df.loc[s1, "vote"] = agree[s1.to_numpy()]
    out = {}
    step2_real = df[(df.step == 2) & (df.stratum.fillna("") != "decoy")]
    groups = [(f"step{s}", df[df.step == s] if s != 2 else step2_real) for s in (1, 2, 4)]
    groups.append(("step2_decoy", df[(df.step == 2) & (df.stratum.fillna("") == "decoy")]))
    groups += [(f"step3_{st}", df[(df.step == 3) & (df.stratum == st)]) for st in ("top", "random", "negative")]
    groups.insert(1, ("step1_answerable", df[(df.step == 1) & (df.analyst_answerable == True)]))  # noqa: E712
    groups.insert(2, ("step1_not_answerable", df[(df.step == 1) & (df.analyst_answerable == False)]))  # noqa: E712
    for name, sub in groups:
        v = sub["vote"].tolist()
        k, n = v.count("yes"), v.count("yes") + v.count("no")
        out[name] = {"n_items": len(sub), "n_decided": n, "yes_rate": round(k / n, 3) if n else None,
                     "wilson_95ci": _wilson(k, n), "unsure": v.count("unsure")}
    return out


def score_review(key_dir: Path, rating_csvs: List[Path]) -> dict:
    key = pd.concat([pd.read_csv(p) for p in sorted(key_dir.glob("*_key.csv"))], ignore_index=True)
    ratings = pd.concat([pd.read_csv(p) for p in rating_csvs], ignore_index=True)
    ratings["rating"] = ratings["rating"].astype(str).str.lower().str.strip()
    ratings = ratings.drop_duplicates(["reviewer", "item_id"], keep="last")
    reviewers = sorted(ratings["reviewer"].unique())
    rated_homes = sorted(ratings["home"].unique())
    key = key[key["home"].isin(rated_homes)]

    by_item = ratings.groupby("item_id")["rating"].apply(list).to_dict()
    consensus = {i: _vote(v) for i, v in by_item.items()}
    cons_df = pd.DataFrame({"item_id": list(consensus), "vote": list(consensus.values())})
    res = {"reviewers": reviewers, "homes": rated_homes,
           "n_ratings": int(len(ratings)), "n_items": int(len(key)),
           "n_items_rated": int(key["item_id"].isin(ratings["item_id"]).sum()),
           "consensus": {"funnel": _funnel(key, consensus), "steps": _step_metrics(key, cons_df)},
           "per_home": {}, "per_reviewer": {}, "kappa": {}}
    for h in rated_homes:
        kh = key[key.home == h]
        res["per_home"][h] = {"funnel": _funnel(kh, consensus), "steps": _step_metrics(kh, cons_df)}
    for rv in reviewers:
        sub = ratings[ratings.reviewer == rv]
        mine = dict(zip(sub.item_id, sub.rating))
        kr = key[key.home.isin(sub.home.unique())]
        res["per_reviewer"][rv] = {"funnel": _funnel(kr, mine),
                                   "steps": _step_metrics(kr, pd.DataFrame({"item_id": list(mine), "vote": list(mine.values())}))}
    wide = ratings.pivot(index="item_id", columns="reviewer", values="rating")
    keyed = key.drop_duplicates("item_id").set_index("item_id")
    step_of = keyed["step"]
    for i in range(len(reviewers)):
        for j in range(i + 1, len(reviewers)):
            a, b = wide[reviewers[i]], wide[reviewers[j]]
            ok = a.isin(["yes", "no"]) & b.isin(["yes", "no"])
            pair = {}
            for s in (1, 2, 3, 4):
                # Step 2 includes far-caption decoys. They are the items that keep the yes-rate off the ceiling.
                m = ok & wide.index.map(step_of).to_series(index=wide.index).eq(s)
                k = _kappa(a[m].tolist(), b[m].tolist())
                pair[f"step{s}"] = {"kappa": None if k is None else round(k, 3), "n": int(m.sum())}
            res["kappa"][f"{reviewers[i]}~{reviewers[j]}"] = pair
    # Fleiss: one coefficient per step over reviewers who rated that step.
    # An item counts only when every one of those reviewers marked it yes or no.
    # Step 2 includes far-caption decoys, same as Cohen's κ.
    res["fleiss"] = {}
    for s in (1, 2, 3, 4):
        sub = wide.loc[wide.index.intersection(step_of[step_of.eq(s)].index)]
        active = [rv for rv in reviewers if rv in sub.columns and sub[rv].isin(["yes", "no"]).any()]
        if len(active) < 2 or sub.empty:
            res["fleiss"][f"step{s}"] = {"kappa": None, "n": 0, "raters": len(active)}
            continue
        ok = sub[active].apply(lambda r: r.isin(["yes", "no"]).all(), axis=1)
        rows = sub.loc[ok, active].astype(str).values.tolist()
        k = _fleiss(rows)
        res["fleiss"][f"step{s}"] = {
            "kappa": None if k is None else round(k, 3), "n": len(rows), "raters": len(active)}
    return res


# --------------------------------------------------------------------------- sankey

_NODE_COL = {"all": 0, "s1": 1, "s2": 2, "s3": 3, "s4": 4}
_NODE_COLOR = {"ok": "#2E9E5B", "no_ok": "#8FBF9F", "false_ans": "#C4503A", "missed": "#D98E04", "bad": "#C4503A",
               "unsure": "#B8BEC8", "all": "#3A6BC4"}


def draw_sankey(funnel: dict, title: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MPath

    nodes = {k: v for k, v in funnel["nodes"].items() if v}
    labels = dict(FUNNEL_NODES)
    order = [k for k, _ in FUNNEL_NODES if k in nodes]
    total = nodes.get("all", 1)
    cols = defaultdict(list)
    for k in order:
        cols[_NODE_COL[k.split("_")[0]]].append(k)
    H, gap, w = 1.0, 0.04, 0.035
    pos = {}
    for c, ks in cols.items():
        y = 0.0
        for k in ks:
            h = nodes[k] / total * H
            pos[k] = [c, y, h]
            y += h + gap
    fig, ax = plt.subplots(figsize=(13, 5.8))
    out_off = {k: 0.0 for k in nodes}
    in_off = {k: 0.0 for k in nodes}
    for lk in sorted(funnel["links"], key=lambda l: order.index(l["target"])):
        s, t, v = lk["source"], lk["target"], lk["value"]
        hs = v / total * H
        x0, x1 = pos[s][0] + w, pos[t][0]
        y0, y1 = pos[s][1] + out_off[s], pos[t][1] + in_off[t]
        out_off[s] += hs
        in_off[t] += hs
        xm = (x0 + x1) / 2
        verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1), (x1, y1 + hs), (xm, y1 + hs), (xm, y0 + hs), (x0, y0 + hs), (x0, y0)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
                 MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        color = _NODE_COLOR.get(t.split("_", 1)[1] if "_" in t else t, "#999")
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, alpha=0.28, lw=0))
    for k, (c, y, h) in pos.items():
        kind = k.split("_", 1)[1] if "_" in k else k
        ax.add_patch(Rectangle((c, y), w, h, color=_NODE_COLOR.get(kind, "#999")))
        ax.text(c + w + 0.02, y + h / 2, f"{labels[k]}  {nodes[k]} ({100 * nodes[k] / total:.0f}%)",
                va="center", fontsize=9, bbox=dict(facecolor="white", alpha=0.75, lw=0, pad=1.5))
    for c, name in enumerate(["", "1 · answerable?", "2 · queries", "3 · retrieval", "4 · findings"]):
        ax.text(c + w / 2, H + gap * 4 + 0.03, name, ha="center", fontsize=10, fontweight="bold", color="#444")
    ax.set_xlim(-0.05, 5.0)
    ax.set_ylim(H + gap * 4 + 0.08, -0.02)
    ax.axis("off")
    ax.set_title(title, fontsize=12, loc="left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_review_report(res: dict, out_dir: Path) -> Path:
    """Sankey figures (consensus, per home, per reviewer) + step metrics as a small HTML page."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "review_scores.json").write_text(json.dumps(res, indent=1, default=str))
    figs = [("consensus", f"All rated homes ({', '.join(res['homes'])}) — majority vote of {len(res['reviewers'])} reviewer(s)",
             res["consensus"])]
    figs += [(f"home_{h}", f"{h.capitalize()} — majority vote", v) for h, v in res["per_home"].items()]
    if len(res["reviewers"]) > 1:
        figs += [(f"reviewer_{r}", f"Reviewer {r}", v) for r, v in res["per_reviewer"].items()]
    pd.DataFrame([{"scope": n, **q} for n, _, v in figs for q in v["funnel"]["per_question"]]).to_csv(
        out_dir / "funnel_per_question.csv", index=False)

    step_names = {"step1": "1 · analyst verdict = reviewer verdict (all)",
                  "step1_answerable": "1 · … analyst said answerable",
                  "step1_not_answerable": "1 · … analyst said not answerable", "step2": "2 · query relevant",
                  "step2_decoy": "2 · far-caption decoy rated relevant (should be ~0)",
                  "step3_top": "3 · top matches correct", "step3_random": "3 · random matches correct",
                  "step3_negative": "3 · far negatives rated yes (should be ~0)", "step4": "4 · finding correct"}

    def cell(m):
        if not m or m["yes_rate"] is None:
            return "–"
        ci = m["wilson_95ci"]
        return f"{100 * m['yes_rate']:.1f}% <span class=m>(n={m['n_decided']}, {100 * ci[0]:.0f}–{100 * ci[1]:.0f}%)</span>"

    scopes = [("all rated", res["consensus"])] + list(res["per_home"].items())
    rows = "".join(f"<tr><td>{lbl}</td>" + "".join(f"<td>{cell(v['steps'].get(k))}</td>" for _, v in scopes) + "</tr>"
                   for k, lbl in step_names.items())
    kappa = "".join(f"<tr><td>{p}</td>" + "".join(f"<td>{v[f'step{s}']['kappa'] if v[f'step{s}']['kappa'] is not None else '–'}"
                                                  f" <span class=m>(n={v[f'step{s}']['n']})</span></td>" for s in (1, 2, 3, 4)) + "</tr>"
                    for p, v in res["kappa"].items()) or "<tr><td colspan=5 class=m>needs ≥ 2 reviewers on the same items</td></tr>"
    fleiss = res.get("fleiss") or {}
    fleiss_cells = "".join(
        f"<td>{(fleiss.get(f'step{s}') or {}).get('kappa') if (fleiss.get(f'step{s}') or {}).get('kappa') is not None else '–'}"
        f" <span class=m>(n={(fleiss.get(f'step{s}') or {}).get('n', 0)}, "
        f"{(fleiss.get(f'step{s}') or {}).get('raters', 0)} raters)</span></td>"
        for s in (1, 2, 3, 4))
    imgs = ""
    for n, title, v in figs:
        draw_sankey(v["funnel"], title, out_dir / f"sankey_{n}")
        imgs += f"<h2>{title}</h2><img src='sankey_{n}.png'>"
    html = f"""<!DOCTYPE html><html><head><meta charset=utf-8><title>Analyst review results</title><style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#1a1a2e;max-width:1300px}}
table{{border-collapse:collapse;margin:8px 0 18px}} td,th{{border-bottom:1px solid #e5e7eb;padding:5px 10px;text-align:left;font-size:13px}}
th{{background:#f3f4f6}} .m{{color:#6b7280;font-size:11px}} img{{max-width:100%;border:1px solid #e5e7eb}}</style></head><body>
<h1>Analyst review results</h1>
<p class=m>Reviewers: {', '.join(res['reviewers'])} · {res['n_ratings']} ratings · {res['n_items_rated']}/{res['n_items']} items rated ·
majority vote per item; a step passes by majority of its decided items; unsure excluded.</p>
{imgs}
<h2>Per-step item accuracy (ungated)</h2>
<table><tr><th>step</th>{''.join(f'<th>{n}</th>' for n, _ in scopes)}</tr>{rows}</table>
<h2>Inter-reviewer agreement (Cohen's κ)</h2>
<p class=m>Step 2 includes far-caption decoys. Pairwise only.</p>
<table><tr><th>pair</th><th>step 1</th><th>step 2</th><th>step 3</th><th>step 4</th></tr>{kappa}</table>
<h2>Fleiss' κ</h2>
<p class=m>One coefficient per step across reviewers who rated it. An item counts only when each of them marked it yes or no. Step 2 includes decoys. Undefined when fewer than two reviewers rated the step.</p>
<table><tr><th>step 1</th><th>step 2</th><th>step 3</th><th>step 4</th></tr><tr>{fleiss_cells}</tr></table>
</body></html>"""
    path = out_dir / "review_report.html"
    path.write_text(html, encoding="utf-8")
    return path
