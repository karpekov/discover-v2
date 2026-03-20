"""
SQLite-backed cache for LLM-rewritten queries.

Schema:
  - original_query: the raw user input
  - home:           dataset identifier (e.g. "milan", "aruba")
  - rewrite_mode:   "single" | "multi_location" | "multi_wording"
  - sentences:      JSON-encoded list[str] of rewritten sentences
  - model_used:     which LLM produced this rewrite
  - created_at:     ISO timestamp
  - hit_count:      how many times this cached entry was reused

The unique key is (original_query, home, rewrite_mode).
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "query_cache" / "queries.db"


def _json_serializer(obj):
    """Fallback JSON serializer for numpy/torch scalar types in result dicts."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class QueryCache:
    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create base tables without any columns that may need migration."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS query_cache (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                original_query TEXT    NOT NULL,
                home           TEXT    NOT NULL DEFAULT '',
                sentences      TEXT    NOT NULL DEFAULT '[]',
                model_used     TEXT    NOT NULL DEFAULT '',
                created_at     TEXT    NOT NULL DEFAULT '',
                hit_count      INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS result_cache (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                original_query TEXT    NOT NULL,
                home           TEXT    NOT NULL DEFAULT '',
                rewrite_mode   TEXT    NOT NULL DEFAULT 'single',
                checkpoint     TEXT    NOT NULL DEFAULT '',
                top_k          INTEGER NOT NULL DEFAULT 5,
                results_json   TEXT    NOT NULL,
                created_at     TEXT    NOT NULL DEFAULT '',
                hit_count      INTEGER NOT NULL DEFAULT 0,
                UNIQUE(original_query, home, rewrite_mode, checkpoint, top_k)
            );

            CREATE INDEX IF NOT EXISTS idx_result_cache
                ON result_cache (original_query, home, rewrite_mode, checkpoint, top_k);
        """)
        self._conn.commit()
        # Migrations run after the base tables exist
        self._migrate()

    def _migrate(self) -> None:
        """Idempotent migrations — safe to run on every startup."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(query_cache)")}

        # 1. Rename legacy column rewritten_query → sentences
        if "rewritten_query" in cols and "sentences" not in cols:
            self._conn.execute(
                "ALTER TABLE query_cache RENAME COLUMN rewritten_query TO sentences"
            )
            self._conn.commit()
            cols.discard("rewritten_query")
            cols.add("sentences")

        # 2. Add rewrite_mode column if missing
        if "rewrite_mode" not in cols:
            self._conn.execute(
                "ALTER TABLE query_cache ADD COLUMN rewrite_mode TEXT NOT NULL DEFAULT 'single'"
            )
            self._conn.commit()

        # 3. Create index on (original_query, home, rewrite_mode)
        #    — only safe after column exists
        self._conn.executescript("""
            DROP INDEX IF EXISTS idx_query_home;
            CREATE INDEX IF NOT EXISTS idx_query_home_mode
                ON query_cache (original_query, home, rewrite_mode);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        original_query: str,
        home: str = "",
        rewrite_mode: str = "single",
    ) -> Optional[list[str]]:
        """Return cached sentences or None on miss. Increments hit_count."""
        row = self._conn.execute(
            "SELECT id, sentences FROM query_cache "
            "WHERE original_query = ? AND home = ? AND rewrite_mode = ?",
            (original_query.strip(), home, rewrite_mode),
        ).fetchone()

        if row is None:
            return None

        self._conn.execute(
            "UPDATE query_cache SET hit_count = hit_count + 1 WHERE id = ?",
            (row["id"],),
        )
        self._conn.commit()
        return json.loads(row["sentences"])

    def store(
        self,
        original_query: str,
        sentences: list[str],
        home: str = "",
        rewrite_mode: str = "single",
        model_used: str = "",
    ) -> None:
        """Persist (original_query, home, rewrite_mode) → sentences. Upserts on conflict."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO query_cache
                (original_query, home, rewrite_mode, sentences, model_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(original_query, home, rewrite_mode) DO UPDATE SET
                sentences  = excluded.sentences,
                model_used = excluded.model_used,
                created_at = excluded.created_at
            """,
            (
                original_query.strip(),
                home,
                rewrite_mode,
                json.dumps(sentences),
                model_used,
                now,
            ),
        )
        self._conn.commit()

    def list_entries(self, home: str = "") -> list[dict]:
        if home:
            rows = self._conn.execute(
                "SELECT * FROM query_cache WHERE home = ? ORDER BY created_at DESC", (home,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM query_cache ORDER BY created_at DESC"
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["sentences"] = json.loads(d["sentences"])
            result.append(d)
        return result

    def delete(self, original_query: str, home: str = "", rewrite_mode: str = "") -> bool:
        if rewrite_mode:
            cursor = self._conn.execute(
                "DELETE FROM query_cache "
                "WHERE original_query = ? AND home = ? AND rewrite_mode = ?",
                (original_query.strip(), home, rewrite_mode),
            )
        else:
            cursor = self._conn.execute(
                "DELETE FROM query_cache WHERE original_query = ? AND home = ?",
                (original_query.strip(), home),
            )
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self, home: str = "") -> int:
        if home:
            cursor = self._conn.execute("DELETE FROM query_cache WHERE home = ?", (home,))
        else:
            cursor = self._conn.execute("DELETE FROM query_cache")
        self._conn.commit()
        return cursor.rowcount

    def stats(self) -> dict:
        qrows = self._conn.execute(
            "SELECT home, rewrite_mode, COUNT(*) AS entries, SUM(hit_count) AS total_hits "
            "FROM query_cache GROUP BY home, rewrite_mode"
        ).fetchall()
        rrows = self._conn.execute(
            "SELECT home, rewrite_mode, COUNT(*) AS entries, SUM(hit_count) AS total_hits "
            "FROM result_cache GROUP BY home, rewrite_mode"
        ).fetchall()
        return {
            "rewrites": {"by_home_mode": [dict(r) for r in qrows], "total": sum(r["entries"] for r in qrows)},
            "results":  {"by_home_mode": [dict(r) for r in rrows], "total": sum(r["entries"] for r in rrows)},
        }

    # ------------------------------------------------------------------
    # Result cache
    # ------------------------------------------------------------------

    def get_results(
        self,
        original_query: str,
        home: str = "",
        rewrite_mode: str = "single",
        checkpoint: str = "",
        top_k: int = 5,
    ) -> Optional[list[dict]]:
        """Return cached retrieval results or None on miss."""
        row = self._conn.execute(
            "SELECT id, results_json FROM result_cache "
            "WHERE original_query=? AND home=? AND rewrite_mode=? "
            "AND checkpoint=? AND top_k=?",
            (original_query.strip(), home, rewrite_mode, checkpoint, top_k),
        ).fetchone()
        if row is None:
            return None
        self._conn.execute(
            "UPDATE result_cache SET hit_count = hit_count + 1 WHERE id = ?",
            (row["id"],),
        )
        self._conn.commit()
        return json.loads(row["results_json"])

    def store_results(
        self,
        original_query: str,
        results: list[dict],
        home: str = "",
        rewrite_mode: str = "single",
        checkpoint: str = "",
        top_k: int = 5,
    ) -> None:
        """Persist retrieval results. Upserts on conflict."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO result_cache
                (original_query, home, rewrite_mode, checkpoint, top_k, results_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(original_query, home, rewrite_mode, checkpoint, top_k) DO UPDATE SET
                results_json = excluded.results_json,
                created_at   = excluded.created_at
            """,
            (
                original_query.strip(), home, rewrite_mode, checkpoint, top_k,
                json.dumps(results, default=_json_serializer), now,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
