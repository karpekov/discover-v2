#!/usr/bin/env python3
"""
HAR Discover Dashboard v2 – Longitudinal Analysis.

Start with:
    conda activate discover-v2-env
    python dashboard-v2/api/main.py

    or (with auto-reload for development):
    uvicorn dashboard-v2.api.main:app --host 0.0.0.0 --port 8001 --reload
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_FILE      = Path(__file__).resolve()
_DASHBOARD = _FILE.parents[1]        # dashboard-v2/
_ROOT      = _FILE.parents[2]        # project root
_SRC       = _ROOT / "src"

for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Hardcoded configuration
# ---------------------------------------------------------------------------

# Per-home retrieval similarity thresholds (calibrated via eval_retrieval_threshold.py)
_RETRIEVAL_THRESHOLDS: Dict[str, float] = {
    "milan": 0.075,
    "aruba": 0.065,
    "cairo": 0.13,
}
_DEFAULT_THRESHOLD = 0.10  # fallback for homes not in the dict


def _threshold_for(home: str) -> float:
    return _RETRIEVAL_THRESHOLDS.get(home.lower(), _DEFAULT_THRESHOLD)

# Per-home configs — always FD_60_p data + _v3 model
_HOMES: Dict[str, Dict[str, str]] = {}

def _discover_homes() -> None:
    """Auto-discover homes that have a _v3 checkpoint and FD_60_p data."""
    models_dir = _ROOT / "trained_models"
    data_base  = _ROOT / "data" / "processed" / "casas"
    if not models_dir.exists():
        return
    for home_dir in sorted(models_dir.iterdir()):
        home = home_dir.name
        v3_ckpt = home_dir / f"{home}_fd60_seq_rb1_textclip_projmlp_clipmlm_v3" / "best_model.pt"
        data_dir = data_base / home / "FD_60_p"
        if v3_ckpt.exists() and data_dir.exists():
            _HOMES[home] = {
                "checkpoint": str(v3_ckpt.relative_to(_ROOT)),
                "data_dir":   str(data_dir.relative_to(_ROOT)),
                "label":      home.capitalize(),
            }

_discover_homes()


# ---------------------------------------------------------------------------
# Longitudinal result cache (SQLite)
# ---------------------------------------------------------------------------

_CACHE_DB = _ROOT / "data" / "query_cache" / "longitudinal_v2.db"


class LongitudinalCache:
    """SQLite-backed cache for full longitudinal analysis results."""

    def __init__(self, db_path: Path = _CACHE_DB):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS longitudinal_cache (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash   TEXT    NOT NULL,
                home         TEXT    NOT NULL,
                question     TEXT    NOT NULL,
                result_json  TEXT    NOT NULL,
                created_at   TEXT    NOT NULL,
                hit_count    INTEGER NOT NULL DEFAULT 0,
                UNIQUE(query_hash, home)
            );
            CREATE INDEX IF NOT EXISTS idx_longitudinal
                ON longitudinal_cache (query_hash, home);
        """)
        self._conn.commit()

    @staticmethod
    def _hash(question: str) -> str:
        return hashlib.sha256(question.strip().lower().encode()).hexdigest()[:24]

    def get(self, question: str, home: str) -> Optional[dict]:
        qh = self._hash(question)
        row = self._conn.execute(
            "SELECT id, result_json FROM longitudinal_cache "
            "WHERE query_hash = ? AND home = ?",
            (qh, home.lower()),
        ).fetchone()
        if row is None:
            return None
        self._conn.execute(
            "UPDATE longitudinal_cache SET hit_count = hit_count + 1 WHERE id = ?",
            (row["id"],),
        )
        self._conn.commit()
        return json.loads(row["result_json"])

    def store(self, question: str, home: str, result: dict) -> None:
        qh  = self._hash(question)
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO longitudinal_cache
                (query_hash, home, question, result_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(query_hash, home) DO UPDATE SET
                result_json = excluded.result_json,
                created_at  = excluded.created_at,
                hit_count   = 0
            """,
            (qh, home.lower(), question.strip(), json.dumps(result), now),
        )
        self._conn.commit()

    def list_entries(self, home: str = "") -> list[dict]:
        if home:
            rows = self._conn.execute(
                "SELECT id, home, question, created_at, hit_count "
                "FROM longitudinal_cache WHERE home = ? ORDER BY created_at DESC",
                (home.lower(),),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, home, question, created_at, hit_count "
                "FROM longitudinal_cache ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, question: str, home: str) -> bool:
        qh = self._hash(question)
        c = self._conn.execute(
            "DELETE FROM longitudinal_cache WHERE query_hash = ? AND home = ?",
            (qh, home.lower()),
        )
        self._conn.commit()
        return c.rowcount > 0


_cache = LongitudinalCache()

# ---------------------------------------------------------------------------
# In-memory SmartQuery + LongitudinalAnalyzer caches
# ---------------------------------------------------------------------------

_sq_cache:       Dict[str, Any] = {}   # home -> SmartQuery
_analyzer_cache: Dict[str, Any] = {}   # home -> LongitudinalAnalyzer


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError(
            "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY."
        )
    return key


def _load_smart_query(home: str):
    """Load (or return cached) SmartQuery for the given home."""
    if home in _sq_cache:
        return _sq_cache[home]

    cfg = _HOMES.get(home)
    if cfg is None:
        raise ValueError(f"Unknown home: {home!r}. Available: {list(_HOMES)}")

    from query import SmartQuery  # type: ignore  # noqa: E402

    sq = SmartQuery.from_data_dir(
        checkpoint_path=str(_ROOT / cfg["checkpoint"]),
        data_dir=str(_ROOT / cfg["data_dir"]),
        home=home,
        splits=["train", "val", "test"],
        max_samples=100_000,
        llm_backend="gemini",
        verbose=True,
    )
    _sq_cache[home] = sq
    return sq


def _load_analyzer(home: str):
    """Load (or return cached) LongitudinalAnalyzer for the given home."""
    if home in _analyzer_cache:
        return _analyzer_cache[home]

    sq = _load_smart_query(home)

    from query.llm_reasoning import LongitudinalAnalyzer  # noqa: E402

    cfg = _HOMES[home]
    captions_path = str(_ROOT / cfg["data_dir"] / "train_captions_baseline.json")
    if not Path(captions_path).exists():
        captions_path = None  # type: ignore[assignment]

    analyzer = LongitudinalAnalyzer(
        retrieval=sq.retrieval,
        home=home,
        backend="gemini",
        api_key=_get_api_key(),
        similarity_threshold=_threshold_for(home),
        captions_path=captions_path,
    )
    _analyzer_cache[home] = analyzer
    return analyzer


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _generate_plot_b64(result: dict) -> str:
    """Render the longitudinal report figure and return a data-URI base64 PNG."""
    from query.plot_longitudinal import plot_report  # noqa: E402

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        plot_report(result, output_path=tmp_path, show=False)
        with open(tmp_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{b64}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="HAR Discover Dashboard v2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    home: str = "milan"
    question: str
    force_refresh: bool = False
    include_plot: bool = False  # set True to get the matplotlib PNG (slower)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/homes")
def get_homes():
    """Return available homes with their display labels."""
    return [
        {"id": home, "label": cfg["label"]}
        for home, cfg in _HOMES.items()
    ]


@app.get("/api/cache")
def get_cache(home: str = ""):
    """List cached queries."""
    return _cache.list_entries(home=home)


@app.delete("/api/cache")
def delete_cache_entry(question: str, home: str = "milan"):
    """Delete a specific cached result."""
    deleted = _cache.delete(question, home)
    return {"deleted": deleted}


@app.get("/api/loader-status")
def loader_status():
    """Which homes are already loaded in memory."""
    return {
        "smart_query": list(_sq_cache.keys()),
        "analyzer":    list(_analyzer_cache.keys()),
        "homes_available": list(_HOMES.keys()),
    }


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Run longitudinal analysis for a natural-language question.

    Returns the analysis result, the report text, and a base64-encoded plot.
    """
    home = req.home.strip().lower()
    if home not in _HOMES:
        raise HTTPException(400, f"Unknown home: {home!r}. Available: {list(_HOMES)}")

    question = req.question.strip()
    if not question:
        raise HTTPException(400, "question must not be empty")

    loop = asyncio.get_event_loop()

    # ── Cache check ──────────────────────────────────────────────────────────
    if not req.force_refresh:
        cached = _cache.get(question, home)
        if cached is not None:
            plot_b64 = ""
            if req.include_plot:
                try:
                    plot_b64 = await loop.run_in_executor(None, _generate_plot_b64, cached)
                except Exception as plot_err:
                    print(f"[DashboardV2] Plot from cache failed (non-fatal): {plot_err}")
            return JSONResponse({
                "from_cache": True,
                "home":       home,
                "question":   question,
                "report":     cached.get("report", ""),
                "plot_image": plot_b64,
                "result":     cached,
            })

    # ── Load model (blocking, cached after first load) ────────────────────────
    try:
        analyzer = await loop.run_in_executor(None, _load_analyzer, home)
    except Exception as e:
        raise HTTPException(500, f"Model loading failed: {e}")

    # ── Run analysis ──────────────────────────────────────────────────────────
    def _run_analysis():
        return analyzer.analyze(question)

    try:
        result = await loop.run_in_executor(None, _run_analysis)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")

    # ── Cache result ──────────────────────────────────────────────────────────
    try:
        _cache.store(question, home, result)
    except Exception as e:
        print(f"[DashboardV2] Cache store failed (non-fatal): {e}")

    # ── Generate plot (optional) ──────────────────────────────────────────────
    plot_b64 = ""
    if req.include_plot:
        try:
            plot_b64 = await loop.run_in_executor(None, _generate_plot_b64, result)
        except Exception as e:
            print(f"[DashboardV2] Plot generation failed (non-fatal): {e}")

    return JSONResponse({
        "from_cache": False,
        "home":       home,
        "question":   question,
        "report":     result.get("report", ""),
        "plot_image": plot_b64,
        "result":     result,
    })


# ---------------------------------------------------------------------------
# Static frontend (must be last)
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
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  HAR Discover Dashboard v2 (Longitudinal)")
    print(f"  http://localhost:{args.port}")
    print(f"  Project root: {_ROOT}")
    print(f"  Homes available: {list(_HOMES.keys())}")
    print(f"  Thresholds: { {h: _threshold_for(h) for h in _HOMES} }")
    print(f"{'='*60}\n")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
