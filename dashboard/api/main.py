#!/usr/bin/env python3
"""
HAR Discover Dashboard – FastAPI backend.

Start with:
    conda activate discover-v2-env
    python dashboard/api/main.py

    or (with auto-reload for development):
    uvicorn dashboard.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_FILE = Path(__file__).resolve()
_DASHBOARD = _FILE.parents[1]   # dashboard/
_ROOT = _FILE.parents[2]        # project root
_SRC = _ROOT / "src"

for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="HAR Discover Dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory SmartQuery cache  { cache_key -> SmartQuery }
_sq_cache: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def _moving_average(arr: np.ndarray, w: int) -> List[Optional[float]]:
    if len(arr) < w or w < 2:
        return [None] * len(arr)
    result: List[Optional[float]] = [None] * len(arr)
    half = w // 2
    for i in range(half, len(arr) - half):
        result[i] = float(arr[i - half: i + half + 1].mean())
    return result


def _rolling_std(arr: np.ndarray, w: int) -> List[float]:
    if len(arr) < w or w < 2:
        return [0.0] * len(arr)
    result = [0.0] * len(arr)
    half = w // 2
    for i in range(half, len(arr) - half):
        result[i] = float(arr[i - half: i + half + 1].std())
    return result


def _load_home_meta(home: str) -> Dict:
    path = _ROOT / "metadata" / "casas_metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f).get(home, {})


def _get_floor_plan_dims(home: str) -> Tuple[int, int]:
    """Return (img_h, img_w) for the floor plan image."""
    fp_path = _ROOT / "metadata" / "floor_plans_augmented" / f"{home}.png"
    if fp_path.exists():
        try:
            import matplotlib.pyplot as plt
            img = plt.imread(str(fp_path))
            return int(img.shape[0]), int(img.shape[1])
        except Exception:
            pass
    return 600, 800


# ---------------------------------------------------------------------------
# Config / discovery endpoints
# ---------------------------------------------------------------------------

@app.get("/api/homes")
def get_homes():
    """List homes that have at least one trained checkpoint."""
    models_dir = _ROOT / "trained_models"
    if not models_dir.exists():
        return []
    return sorted(
        d.name for d in models_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


@app.get("/api/checkpoints/{home}")
def get_checkpoints(home: str):
    """List model checkpoints for a given home."""
    models_dir = _ROOT / "trained_models" / home
    if not models_dir.exists():
        return []
    result = []
    for d in sorted(models_dir.iterdir()):
        if d.is_dir() and (d / "best_model.pt").exists():
            result.append({
                "name": d.name,
                "path": str((d / "best_model.pt").relative_to(_ROOT)),
            })
    return result


@app.get("/api/splits/{home}")
def get_splits(home: str):
    """List available data splits for a home."""
    data_dir = _ROOT / "data" / "processed" / "casas" / home
    if not data_dir.exists():
        return []
    _ignore = {"layout_embeddings", "seq20", "seq50"}
    return sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and d.name not in _ignore and not d.name.endswith("_p")
    )


@app.get("/api/floor-plan/{home}")
def get_floor_plan(home: str):
    """Serve the floor plan PNG for a home."""
    path = _ROOT / "metadata" / "floor_plans_augmented" / f"{home}.png"
    if not path.exists():
        raise HTTPException(404, f"No floor plan for '{home}'")
    return FileResponse(str(path), media_type="image/png")


@app.get("/api/sensor-info/{home}")
def get_sensor_info(home: str):
    """Return sensor coordinates and locations for a home."""
    meta = _load_home_meta(home)
    img_h, img_w = _get_floor_plan_dims(home)
    coords: Dict[str, List[int]] = meta.get("sensor_coordinates", {})
    locs: Dict[str, str] = meta.get("sensor_location", {})
    # Pre-compute y_flip for each sensor
    sensors = {
        sid: {
            "x": c[0],
            "y": c[1],
            "y_flip": img_h - c[1],
            "location": locs.get(sid, ""),
        }
        for sid, c in coords.items()
    }
    return {
        "sensors": sensors,
        "floor_plan_width": img_w,
        "floor_plan_height": img_h,
    }


# ---------------------------------------------------------------------------
# Analyze request model
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    home: str = "milan"
    checkpoint: str             # relative to ROOT, e.g. "trained_models/milan/.../best_model.pt"
    data_dir: Optional[str] = None   # relative to ROOT
    test_data: Optional[str] = None  # alternative to data_dir
    vocab: Optional[str] = None
    split: str = "FD_60"
    query: str
    mode: str = "multi_location"
    threshold: Optional[float] = 0.10
    top_k: int = 200
    time_window: str = "day"
    ma_window: int = 7
    force_rewrite: bool = False
    force_retrieve: bool = False
    llm_backend: str = "gemini"
    llm_n_examples: int = 6
    caption_style: str = "baseline"
    splits_to_load: str = "train,val,test"


# ---------------------------------------------------------------------------
# Model loader (blocking, runs in thread pool)
# ---------------------------------------------------------------------------

def _build_smart_query(req: AnalyzeRequest):
    from query import SmartQuery  # type: ignore

    ckpt = str(_ROOT / req.checkpoint)
    if req.data_dir:
        return SmartQuery.from_data_dir(
            checkpoint_path=ckpt,
            data_dir=str(_ROOT / req.data_dir),
            home=req.home,
            splits=[s.strip() for s in req.splits_to_load.split(",")],
            caption_style=req.caption_style,
            llm_backend=req.llm_backend,
            llm_n_examples=req.llm_n_examples,
        )
    if req.test_data and req.vocab:
        return SmartQuery.from_checkpoint(
            checkpoint_path=ckpt,
            test_data_path=str(_ROOT / req.test_data),
            vocab_path=str(_ROOT / req.vocab),
            home=req.home,
            llm_backend=req.llm_backend,
            llm_n_examples=req.llm_n_examples,
        )
    raise ValueError("Provide data_dir or (test_data + vocab)")


# ---------------------------------------------------------------------------
# Analytics computation
# ---------------------------------------------------------------------------

def _compute_analytics(out: Dict, req: AnalyzeRequest) -> Dict:
    """Build the full analytics payload from a sq.query() result dict."""
    results: List[Dict] = out.get("results", [])
    home = req.home
    tw = req.time_window
    w = req.ma_window

    # Sensor metadata
    meta = _load_home_meta(home)
    sensor_coords: Dict[str, List[int]] = meta.get("sensor_coordinates", {})
    sensor_location: Dict[str, str] = meta.get("sensor_location", {})
    img_h, img_w = _get_floor_plan_dims(home)

    # --- Parse timestamps ---
    dated: List[Tuple[datetime, Dict]] = []
    for r in results:
        ts = r.get("labels", {}).get("start_time")
        if not ts:
            evts = r.get("events", [])
            if evts:
                ts = evts[0].get("timestamp") or evts[0].get("time")
        dt = _parse_ts(ts)
        if dt:
            dated.append((dt, r))

    # --- Time-series aggregation ---
    buckets: Dict[datetime, List] = defaultdict(list)
    for dt, r in dated:
        if tw == "week":
            key = (dt - timedelta(days=dt.weekday())).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        else:
            key = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        buckets[key].append(r)

    dates_sorted: List[datetime] = sorted(buckets)
    if len(dates_sorted) > 1:
        step = timedelta(weeks=1) if tw == "week" else timedelta(days=1)
        full: List[datetime] = []
        cur, end = min(dates_sorted), max(dates_sorted)
        while cur <= end:
            full.append(cur)
            cur += step
        dates_sorted = full

    counts = np.array([len(buckets.get(d, [])) for d in dates_sorted], dtype=float)
    durations = np.array([
        sum(
            (r.get("labels", {}).get("duration_seconds") or 0.0)
            for r in buckets.get(d, [])
        ) / 60.0
        for d in dates_sorted
    ])

    # --- Distributions ---
    label_ctr: Counter = Counter()
    room_ctr: Counter = Counter()
    tod_ctr: Counter = Counter()
    dow_ctr: Counter = Counter()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for dt, r in dated:
        labels = r.get("labels", {})
        label_ctr[labels.get("activity_l1") or "Unknown"] += 1
        room_ctr[labels.get("primary_room") or "Unknown"] += 1
        h = dt.hour
        if 5 <= h < 12:
            tod_ctr["Morning"] += 1
        elif 12 <= h < 17:
            tod_ctr["Afternoon"] += 1
        elif 17 <= h < 21:
            tod_ctr["Evening"] += 1
        else:
            tod_ctr["Night"] += 1
        dow_ctr[day_names[dt.weekday()]] += 1

    # --- Scores ---
    scores = [float(r["score"]) for r in results if "score" in r]

    # --- Sensor activations ---
    act: Counter = Counter()
    for r in results:
        for evt in r.get("events", []):
            sid = evt.get("sensor_id") or evt.get("sensor") or evt.get("id", "")
            state = str(evt.get("event_type", evt.get("state", ""))).upper()
            if state in ("ON", "OPEN", "1", "TRUE") and sid:
                act[sid] += 1

    sensor_data = {
        sid: {
            "count": act.get(sid, 0),
            "x": c[0],
            "y": c[1],
            "y_flip": img_h - c[1],
            "location": sensor_location.get(sid, ""),
        }
        for sid, c in sensor_coords.items()
    }

    # --- Summary stats ---
    peak_idx = int(np.argmax(counts)) if len(counts) > 0 else 0

    return {
        "metadata": {
            "query": out.get("original_query", req.query),
            "reasoning": out.get("reasoning", ""),
            "sentences": out.get("sentences", []),
            "model_used": out.get("model_used") or "cache hit",
            "rewrite_mode": out.get("rewrite_mode", req.mode),
            "rewrite_cache_hit": bool(out.get("rewrite_cache_hit")),
            "result_cache_hit": bool(out.get("result_cache_hit")),
            "home": home,
            "split": req.split,
        },
        "stats": {
            "total_results": len(results),
            "dated_results": len(dated),
            "date_range": (
                f"{min(dates_sorted).strftime('%Y-%m-%d')} → "
                f"{max(dates_sorted).strftime('%Y-%m-%d')}"
            ) if dates_sorted else "—",
            "span_days": (
                (max(dates_sorted) - min(dates_sorted)).days + 1
            ) if len(dates_sorted) > 1 else 1,
            "active_periods": int(np.count_nonzero(counts)) if len(counts) > 0 else 0,
            "time_window": tw,
            "avg_per_period": float(np.mean(counts)) if len(counts) > 0 else 0.0,
            "peak_date": dates_sorted[peak_idx].strftime("%Y-%m-%d") if dates_sorted else "—",
            "peak_count": int(counts[peak_idx]) if len(counts) > 0 else 0,
            "score_mean": float(np.mean(scores)) if scores else None,
            "score_median": float(np.median(scores)) if scores else None,
            "score_min": float(min(scores)) if scores else None,
            "score_max": float(max(scores)) if scores else None,
            "labels": dict(label_ctr.most_common()),
            "rooms": dict(room_ctr.most_common()),
            "time_of_day": {
                k: tod_ctr.get(k, 0)
                for k in ("Morning", "Afternoon", "Evening", "Night")
            },
            "day_of_week": {d: dow_ctr.get(d, 0) for d in day_names},
        },
        "time_series": {
            "dates": [d.strftime("%Y-%m-%d") for d in dates_sorted],
            "counts": counts.tolist(),
            "durations": durations.tolist(),
            "ma_counts": _moving_average(counts, w),
            "ma_durations": _moving_average(durations, w),
            "std_counts": _rolling_std(counts, w),
        },
        "sensor_data": sensor_data,
        "scores": scores,
        "floor_plan": {
            "url": f"/api/floor-plan/{home}",
            "width": img_w,
            "height": img_h,
        },
    }


# ---------------------------------------------------------------------------
# Main analysis endpoint
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.data_dir and not (req.test_data and req.vocab):
        raise HTTPException(400, "Provide data_dir or (test_data + vocab)")

    cache_key = f"{req.checkpoint}::{req.data_dir or req.test_data}::{req.home}"
    loop = asyncio.get_event_loop()

    if cache_key not in _sq_cache:
        try:
            sq = await loop.run_in_executor(None, _build_smart_query, req)
            _sq_cache[cache_key] = sq
        except Exception as e:
            raise HTTPException(500, f"Model loading failed: {e}")

    sq = _sq_cache[cache_key]

    try:
        out = await loop.run_in_executor(
            None,
            lambda: sq.query(
                req.query,
                mode=req.mode,
                top_k=req.top_k,
                threshold=req.threshold,
                force_rewrite=req.force_rewrite,
                force_retrieve=req.force_retrieve,
            ),
        )
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")

    return JSONResponse(_compute_analytics(out, req))


@app.get("/api/cache-status")
def cache_status():
    """Return which model configurations are currently loaded."""
    return {"loaded": list(_sq_cache.keys())}


# ---------------------------------------------------------------------------
# Static frontend (must be last — catches all unmatched routes)
# ---------------------------------------------------------------------------
_static_dir = _DASHBOARD / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  HAR Discover Dashboard")
    print(f"  http://localhost:{args.port}")
    print(f"  Project root: {_ROOT}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
