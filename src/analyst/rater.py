"""Human-rater evaluation: sample items, build blinded HTML packets, score ratings.

Per concept, three strata are sampled (seeded, time-separated):
  random    — uniform from all matched windows: unbiased precision of the set the
              analysis is computed on (the number to report).
  top       — highest-scoring matched windows (precision of the ranking head).
  negative  — eligible but unmatched windows under the same time filters
              (control: raters should mostly answer "no").
Raters never see the stratum or score; items are shuffled per home.
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image

from .config import TOD_BUCKETS, AnalysisSettings
from .home_context import sensor_coordinates
from .index import SensorIndex
from .retrieval import ConceptMatches

_TEMPLATE = Path(__file__).with_name("rater_template.html")
_MIN_SEPARATION = pd.Timedelta(minutes=10)


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _pick_separated(df: pd.DataFrame, n: int, taken: List[pd.Timestamp]) -> pd.DataFrame:
    chosen = []
    for idx, row in df.iterrows():
        if len(chosen) >= n:
            break
        if all(abs(row["start"] - t) >= _MIN_SEPARATION for t in taken):
            chosen.append(idx)
            taken.append(row["start"])
    return df.loc[chosen]


def sample_items(question_id: str, cm: ConceptMatches, settings: AnalysisSettings) -> List[dict]:
    c = cm.concept
    if not c.detectable or not c.retrieval_queries:
        return []
    rng_seed = _seed(settings.seed, question_id, c.id)
    matched, unmatched = cm.matched, cm.eligible[~cm.eligible["matched"]]
    taken: List[pd.Timestamp] = []
    strata = [
        ("top", matched.sort_values("score", ascending=False), settings.rater_n_top),
        ("random", matched.sample(frac=1.0, random_state=rng_seed % 2**32), settings.rater_n_random),
        ("negative", unmatched.sample(frac=1.0, random_state=(rng_seed + 1) % 2**32), settings.rater_n_negative),
    ]
    items = []
    for stratum, pool, n in strata:
        for _, row in _pick_separated(pool, n, taken).iterrows():
            items.append({
                "question_id": question_id,
                "concept_id": c.id,
                "concept_name": c.name,
                "definition": c.physical_definition,
                "time_of_day": list(c.time_of_day),
                "stratum": stratum,
                "sample_id": row["sample_id"],
                "start": row["start"].isoformat(),
                "end": row["end"].isoformat(),
                "score": None if pd.isna(row["score"]) else round(float(row["score"]), 4),
                "label_l1": row["label_l1"],
            })
    return items


def _item_payload(index: SensorIndex, item: dict, context_s: int) -> dict:
    t0, t1 = pd.Timestamp(item["start"]), pd.Timestamp(item["end"])
    t1 = max(t1, t0 + pd.Timedelta(seconds=60))
    c0, c1 = t0 - pd.Timedelta(seconds=context_s), t1 + pd.Timedelta(seconds=context_s)
    ev = index.events_between(c0, c1)
    rel = (ev["timestamp"] - c0).dt.total_seconds().round(2).tolist()
    return {
        "clock_start": c0.strftime("%Y-%m-%dT%H:%M:%S"),
        "weekday": c0.day_name(),
        "duration": round((c1 - c0).total_seconds(), 1),
        "target": [round((t0 - c0).total_seconds(), 1), round((t1 - c0).total_seconds(), 1)],
        "events": [[r, s, st] for r, s, st in zip(rel, ev["sensor_id"], ev["state"].str.upper())],
    }


def build_packet(index: SensorIndex, items: List[dict], out_dir: Path, settings: AnalysisSettings,
                 packet_name: str) -> Dict[str, Path]:
    """Write <home>.html (rating app) and <home>_key.csv (hidden answer key)."""
    home = index.cfg.home
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(_seed(settings.seed, packet_name, home))
    items = list(items)
    rng.shuffle(items)

    key_rows, payload_items = [], []
    for i, it in enumerate(items, 1):
        item_id = f"{home[:2].upper()}{i:03d}"
        key_rows.append({"item_id": item_id, **it})
        tod_note = ""
        if it["time_of_day"]:
            tod_note = " (" + ", ".join(f"{b.replace('_', ' ')} {TOD_BUCKETS[b]}" for b in it["time_of_day"]) + ")"
        payload_items.append({
            "id": item_id,
            "question_id": it["question_id"].split("/", 1)[-1],
            "concept_id": it["concept_id"],
            "behavior": it["concept_name"] + tod_note,
            "definition": it["definition"],
            **_item_payload(index, it, settings.rater_context_seconds),
        })

    img_path = index.cfg.floor_plan
    img_b64 = base64.b64encode(img_path.read_bytes()).decode() if img_path.exists() else ""
    w, h = Image.open(img_path).size if img_path.exists() else (2600, 2800)
    data = {
        "packet": packet_name,
        "home": home,
        "image": {"src": f"data:image/png;base64,{img_b64}", "w": w, "h": h},
        "sensors": {k: [v[0], h - v[1]] for k, v in sensor_coordinates(home).items()},
        "items": payload_items,
    }
    html = _TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    html_path = out_dir / f"{home}.html"
    html_path.write_text(html, encoding="utf-8")
    key_path = out_dir / f"{home}_key.csv"
    pd.DataFrame(key_rows).to_csv(key_path, index=False)
    return {"html": html_path, "key": key_path}


# --------------------------------------------------------------------------- scoring

def _wilson(k: int, n: int, z: float = 1.96) -> list | None:
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [round(centre - half, 3), round(centre + half, 3)]


def _kappa(a: List[str], b: List[str]) -> float | None:
    if not a:
        return None
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _fleiss(rows: List[List[str]]) -> float | None:
    """Fleiss' κ for nominal yes/no ratings. Every item must have the same number of raters, at least 2."""
    if not rows:
        return None
    n = len(rows[0])
    if n < 2 or any(len(r) != n for r in rows):
        return None
    cats = ("no", "yes")
    n_items = len(rows)
    totals = {c: 0 for c in cats}
    agree = 0
    for row in rows:
        for c in cats:
            nk = row.count(c)
            totals[c] += nk
            agree += nk * (nk - 1)
    p_bar = agree / (n_items * n * (n - 1))
    pe = sum((totals[c] / (n_items * n)) ** 2 for c in cats)
    return None if pe == 1 else (p_bar - pe) / (1 - pe)


def score_ratings(key_csvs: List[Path], rating_csvs: List[Path]) -> dict:
    """Precision per concept/stratum and pairwise Cohen's kappa (yes/no; 'unsure' excluded)."""
    key = pd.concat([pd.read_csv(p) for p in key_csvs], ignore_index=True)
    ratings = pd.concat([pd.read_csv(p) for p in rating_csvs], ignore_index=True)
    ratings["rating"] = ratings["rating"].str.lower().str.strip()
    ratings = ratings.drop_duplicates(["rater", "item_id"], keep="last")
    wide = ratings.pivot(index="item_id", columns="rater", values="rating")
    raters = list(wide.columns)
    df = key.set_index("item_id").join(wide, how="inner")

    def _yes_rate(sub: pd.DataFrame) -> dict:
        vals = sub[raters].to_numpy().ravel()
        decided = [v for v in vals if v in ("yes", "no")]
        n, k = len(decided), sum(v == "yes" for v in decided)
        return {
            "n_items": int(len(sub)),
            "n_judgements": int(n),
            "yes_rate": round(k / n, 3) if n else None,
            "wilson_95ci": _wilson(k, n),
            "unsure": int(sum(v == "unsure" for v in vals)),
        }

    per_concept = defaultdict(dict)
    for (qid, cid, stratum), sub in df.groupby(["question_id", "concept_id", "stratum"]):
        per_concept[f"{qid}/{cid}"][stratum] = _yes_rate(sub)
    overall = {s: _yes_rate(sub) for s, sub in df.groupby("stratum")}

    kappas = {}
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            a, b = df[raters[i]], df[raters[j]]
            ok = a.isin(["yes", "no"]) & b.isin(["yes", "no"])
            k = _kappa(a[ok].tolist(), b[ok].tolist())
            kappas[f"{raters[i]}~{raters[j]}"] = {"kappa": None if k is None else round(k, 3), "n": int(ok.sum())}
    return {"raters": raters, "overall": overall, "per_concept": dict(per_concept), "cohen_kappa": kappas,
            "note": "yes_rate on 'random' = precision of the analyzed set; on 'negative' = rater false-positive control."}
