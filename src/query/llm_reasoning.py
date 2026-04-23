"""
LLM-powered longitudinal activity analyzer.

Answers natural-language questions about how activity patterns have changed
over time in smart-home sensor data.

Flow
----
1. LLM parses the user's question → activity concepts + time period + window size
2. LLMRewriter expands each activity concept → retrieval sentences (auto mode)
3. Dataset is filtered per time window → per-window FAISS sub-index
4. Each sentence is retrieved against both sub-indices → match counts (above threshold)
5. Matches are deduplicated across sentences, aggregated by day
6. Stats computed: % change, daily variability, Mann-Whitney significance test
7. LLM synthesizes a natural-language report covering all activities
8. Full intermediate data saved to JSON (for dashboard use)

Usage
-----
python src/query/llm_reasoning.py \\
    --question "how has sedentary activity changed over the past month?" \\
    --home milan \\
    --checkpoint trained_models/milan/best_model.pt \\
    --test_data data/processed/casas/milan/FD_60/test.json \\
    --vocab data/processed/casas/milan/FD_60/vocab.json \\
    --output results/longitudinal/milan_sedentary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running as a script from any cwd
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from query.llm_rewriter import LLMRewriter, _GeminiBackend, _OpenAICompatibleBackend  # noqa: E402


# ---------------------------------------------------------------------------
# Per-home data quality cutoffs
# Samples with start_time before the cutoff are excluded from all retrieval.
# ---------------------------------------------------------------------------
_HOME_DATA_CUTOFFS: dict[str, datetime] = {
    # Milan: first ~3 weeks show anomalous sensor patterns; usable data starts Nov 13
    "milan": datetime(2009, 11, 13),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TimeWindow:
    def __init__(self, start: datetime, end: datetime, label: str):
        self.start = start
        self.end = end
        self.label = label

    def contains(self, ts: datetime) -> bool:
        return self.start <= ts < self.end

    def to_dict(self) -> dict:
        return {"label": self.label, "start": self.start.isoformat(), "end": self.end.isoformat()}


class WindowResult:
    def __init__(
        self,
        total_matches: int,
        daily_counts: list[int],
        per_query_counts: dict[str, int],
        per_query_daily_counts: dict[str, list[int]] | None = None,
    ):
        self.total_matches = total_matches
        self.daily_counts = daily_counts
        self.per_query_counts = per_query_counts
        self.per_query_daily_counts: dict[str, list[int]] = per_query_daily_counts or {}
        arr = np.array(daily_counts, dtype=float)
        self.mean_per_day = float(arr.mean()) if len(arr) > 0 else 0.0
        self.std_per_day = float(arr.std()) if len(arr) > 0 else 0.0

    def to_dict(self, window: TimeWindow) -> dict:
        return {
            "label": window.label,
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "total_matches": self.total_matches,
            "daily_counts": self.daily_counts,
            "mean_per_day": round(self.mean_per_day, 2),
            "std_per_day": round(self.std_per_day, 2),
            "per_query_counts": self.per_query_counts,
            "per_query_daily_counts": self.per_query_daily_counts,
        }


# ---------------------------------------------------------------------------
# LLM question parser
# ---------------------------------------------------------------------------

_PARSE_SYSTEM_PROMPT = """\
You are a structured query parser for a smart-home activity analysis system.

Given a user's natural-language question about activity patterns over time, extract:
1. The activity concepts being asked about (there may be multiple).
2. The time period of interest.
3. The comparison window size (how large each comparison chunk should be).

Output ONLY valid JSON in exactly this format — no extra text, no markdown fences:
{
  "activities": ["activity concept 1", "activity concept 2"],
  "period": {
    "type": "auto" | "relative" | "absolute",
    "description": "past month",
    "start_date": "YYYY-MM-DD or null",
    "end_date": "YYYY-MM-DD or null"
  },
  "window_size": "day" | "week" | "month"
}

Rules:
- activities: extract each distinct activity concept as a short phrase. CRITICAL: preserve
  any temporal, location, or contextual qualifiers that the user attached to the activity
  (e.g. "at night", "in the morning", "on weekends", "after dinner"). Do NOT strip these
  qualifiers — they are essential retrieval filters. If a qualifier applies to all activities
  in the question, include it in every activity phrase.
  Examples:
    "snacking or watching TV during nighttime hours" → ["snacking at night", "watching TV at night"]
    "morning exercise and evening relaxation" → ["morning exercise", "evening relaxation"]
    "sedentary activity" → ["sedentary activity"]
- period.type "auto"     — no specific period mentioned; use the full dataset range.
- period.type "relative" — relative period ("past month", "last 3 months"); leave dates null.
- period.type "absolute" — specific calendar dates mentioned; fill start_date and end_date.
- window_size: the granularity of each comparison chunk. Default "week" if not mentioned.
"""


def _parse_user_question(question: str, backend) -> dict:
    """Use the LLM to parse a natural-language question into structured form."""
    raw = backend.call(question, _PARSE_SYSTEM_PROMPT, max_subqueries=1).strip()
    # Strip markdown fences if the LLM wrapped its output
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "activities": [question],
            "period": {"type": "auto", "description": "full dataset"},
            "window_size": "week",
        }


# ---------------------------------------------------------------------------
# Timestamp pre-computation
# ---------------------------------------------------------------------------

_TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d")
_BATCH_SIZE = 64  # must match SmartHomeRetrieval's internal batch size


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    s = str(raw)
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _precompute_timestamps(retrieval) -> list[datetime | None]:
    """
    Build a flat list: FAISS index position → start_time datetime (or None).
    Reads directly from test_dataset.data to avoid decoding every batch.
    """
    timestamps: list[datetime | None] = []
    for batch_idx, sample_idx in retrieval.sample_indices:
        dataset_idx = batch_idx * _BATCH_SIZE + sample_idx
        ts = None
        try:
            if dataset_idx < len(retrieval.test_dataset.data):
                meta = retrieval.test_dataset.data[dataset_idx].get("metadata", {})
                ts = _parse_ts(meta.get("start_time"))
        except Exception:
            pass
        timestamps.append(ts)
    return timestamps


# ---------------------------------------------------------------------------
# Per-window FAISS sub-index
# ---------------------------------------------------------------------------

def _build_window_subindex(
    retrieval,
    timestamps: list[datetime | None],
    window: TimeWindow,
):
    """
    Build a FAISS sub-index containing only samples whose start_time falls
    within [window.start, window.end).

    Returns (sub_index, original_flat_positions).
    """
    import faiss  # imported here to keep the module importable without faiss

    valid_positions = [
        pos for pos, ts in enumerate(timestamps)
        if ts is not None and window.contains(ts)
    ]
    if not valid_positions:
        d = retrieval.sensor_embeddings.shape[1]
        return faiss.IndexFlatIP(d), []

    subset = retrieval.sensor_embeddings[valid_positions].astype("float32")
    sub_index = faiss.IndexFlatIP(subset.shape[1])
    sub_index.add(subset)
    return sub_index, valid_positions


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _encode_query(retrieval, text: str) -> "np.ndarray":
    import torch
    with torch.no_grad():
        emb = retrieval.text_encoder.encode_texts_clip([text], retrieval.device)
        if retrieval.text_projection is not None:
            emb = retrieval.text_projection(emb)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.cpu().numpy().astype("float32")


def _retrieve_in_window(
    retrieval,
    sentences: list[str],
    sub_index,
    valid_positions: list[int],
    timestamps: list[datetime | None],
    window: TimeWindow,
    threshold: float,
) -> WindowResult:
    """
    Retrieve all matches above `threshold` for every rewritten sentence.
    Matches are deduplicated across sentences by flat FAISS position.
    """
    if sub_index.ntotal == 0:
        return WindowResult(0, [], {s: 0 for s in sentences})

    matched_positions: set[int] = set()
    per_query_counts: dict[str, int] = {}
    per_query_hits: dict[str, list[int]] = {}

    n_days = max(1, (window.end - window.start).days)
    k = sub_index.ntotal  # search everything in the window
    for sentence in sentences:
        query_emb = _encode_query(retrieval, sentence)
        scores, sub_indices = sub_index.search(query_emb, k=k)
        hits = [
            valid_positions[si]
            for si, sc in zip(sub_indices[0], scores[0])
            if 0 <= si < len(valid_positions) and sc >= threshold
        ]
        per_query_counts[sentence] = len(hits)
        per_query_hits[sentence] = hits
        matched_positions.update(hits)

    # Aggregate matched positions into daily counts (deduplicated total)
    day_counts = [0] * n_days
    for flat_pos in matched_positions:
        ts = timestamps[flat_pos]
        if ts is not None:
            offset = (ts.date() - window.start.date()).days
            if 0 <= offset < n_days:
                day_counts[offset] += 1

    # Per-sentence daily counts (not deduplicated — for per-query stats)
    per_query_daily_counts: dict[str, list[int]] = {}
    for sentence, hits in per_query_hits.items():
        q_day = [0] * n_days
        for flat_pos in hits:
            ts = timestamps[flat_pos]
            if ts is not None:
                offset = (ts.date() - window.start.date()).days
                if 0 <= offset < n_days:
                    q_day[offset] += 1
        per_query_daily_counts[sentence] = q_day

    return WindowResult(
        total_matches=len(matched_positions),
        daily_counts=day_counts,
        per_query_counts=per_query_counts,
        per_query_daily_counts=per_query_daily_counts,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _compute_stats(wa: WindowResult, wb: WindowResult) -> dict[str, Any]:
    """Compare two windows: % change, daily variability, significance test."""
    pct_change = None
    if wa.total_matches > 0:
        pct_change = round((wb.total_matches - wa.total_matches) / wa.total_matches * 100, 1)

    direction = None
    if pct_change is not None:
        direction = "stable" if abs(pct_change) < 5 else ("increase" if pct_change > 0 else "decrease")

    p_value = None
    is_significant = None
    if len(wa.daily_counts) >= 3 and len(wb.daily_counts) >= 3:
        try:
            from scipy import stats as scipy_stats
            _, pv = scipy_stats.mannwhitneyu(
                wa.daily_counts, wb.daily_counts, alternative="two-sided"
            )
            p_value = round(float(pv), 4)
            is_significant = p_value < 0.05
        except Exception:
            pass

    return {
        "pct_change": pct_change,
        "direction": direction,
        "is_significant": is_significant,
        "p_value": p_value,
        "window_a_mean_per_day": round(wa.mean_per_day, 2),
        "window_a_std_per_day": round(wa.std_per_day, 2),
        "window_b_mean_per_day": round(wb.mean_per_day, 2),
        "window_b_std_per_day": round(wb.std_per_day, 2),
    }


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM_PROMPT = """\
You are summarising smart-home sensor retrieval counts for a research dashboard.
Your only job is to describe the raw numbers you are given — nothing more.

STRICT RULES — violating any of these is a failure:
1. Report ONLY what is in the data: occurrence counts, daily averages, and percent changes.
2. NEVER mention statistical significance, p-values, confidence, or hypothesis testing.
3. NEVER use hedging phrases such as: "not statistically significant", "may fall within
   normal variability", "typical fluctuations", "within expected range", "should be
   interpreted with caution", or any similar qualification.
4. NEVER speculate about whether a change is "meaningful" or "clinically relevant".
5. State changes as plain observed facts: "X increased by Y%", "Z dropped from A to B per day".
6. Note daily variability (mean ± std) only if it changed substantially between windows.
7. If multiple activities are given, briefly compare their magnitudes.
8. 3–5 sentences of plain prose. No bullet points. No caveats. No hedging.
"""


def _synthesize_report(
    user_question: str,
    all_results: list[dict],
    backend,
    window_a_label: str,
    window_b_label: str,
) -> str:
    lines = [
        f"User question: {user_question}",
        f"Comparison: {window_a_label} (baseline) vs {window_b_label} (recent)",
        "",
    ]
    for res in all_results:
        wa = res["window_a"]
        wb = res["window_b"]
        st = res["stats"]
        lines += [
            f"Activity: {res['activity']}",
            f"  {window_a_label}: {wa['total_matches']} occurrences, "
            f"{wa['mean_per_day']:.1f}/day ± {wa['std_per_day']:.1f}",
            f"  {window_b_label}: {wb['total_matches']} occurrences, "
            f"{wb['mean_per_day']:.1f}/day ± {wb['std_per_day']:.1f}",
        ]
        if st["pct_change"] is not None:
            lines.append(f"  Change: {st['pct_change']:+.1f}% ({st['direction']})")
        lines.append("")

    # max_subqueries=8 gives ~1700 token budget — enough for a paragraph
    return backend.call("\n".join(lines), _SYNTHESIS_SYSTEM_PROMPT, max_subqueries=8)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LongitudinalAnalyzer:
    """
    Answers natural-language questions about longitudinal activity patterns.

    Combines LLMRewriter (query expansion) with SmartHomeRetrieval (FAISS search)
    to compare activity match counts across two time windows.

    Args:
        retrieval:            A loaded SmartHomeRetrieval instance.
        home:                 Dataset identifier ("milan", "aruba", …).
        backend:              "gemini" (default) or "openai".
        api_key:              LLM API key (falls back to env vars).
        model:                Override LLM model name.
        base_url:             For OpenAI-compatible endpoints.
        similarity_threshold: Cosine similarity cutoff for a "match" (default 0.25).
        window_size_days:     Number of days in each comparison window (default 7).
        metadata_path:        Override path to casas_metadata.json.
        captions_path:        Style-example captions for LLMRewriter (False to disable).
    """

    def __init__(
        self,
        retrieval,
        home: str = "milan",
        backend: str = "gemini",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        similarity_threshold: float = 0.25,
        window_size_days: int = 7,
        metadata_path: str | Path | None = None,
        captions_path: str | Path | bool | None = None,
    ):
        self.retrieval = retrieval
        self.threshold = similarity_threshold
        self.window_size_days = window_size_days

        # Shared LLM backend
        if backend == "gemini":
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
            if not key:
                raise ValueError("Gemini API key required. Set GEMINI_API_KEY or pass api_key=.")
            self._llm = _GeminiBackend(api_key=key, model=model)
        elif backend in ("openai", "openai_compatible"):
            key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key=.")
            if not model:
                raise ValueError("model= is required for the openai backend.")
            self._llm = _OpenAICompatibleBackend(api_key=key, model=model, base_url=base_url)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        self.rewriter = LLMRewriter(
            home=home,
            backend=backend,
            api_key=api_key,
            model=model,
            base_url=base_url,
            metadata_path=metadata_path,
            captions_path=captions_path,
        )

        print("[LongitudinalAnalyzer] Pre-computing sample timestamps…")
        self._timestamps = _precompute_timestamps(retrieval)

        # Apply per-home data-quality cutoff: mask out samples before the cutoff
        cutoff = _HOME_DATA_CUTOFFS.get(home)
        if cutoff is not None:
            before = sum(1 for t in self._timestamps if t is not None and t < cutoff)
            if before:
                self._timestamps = [
                    t if (t is None or t >= cutoff) else None
                    for t in self._timestamps
                ]
                print(
                    f"[LongitudinalAnalyzer] Applied data cutoff for '{home}': "
                    f"excluded {before:,} samples before {cutoff.date()}"
                )

        valid_ts = [t for t in self._timestamps if t is not None]
        if valid_ts:
            self._data_start = min(valid_ts)
            self._data_end = max(valid_ts)
            print(
                f"[LongitudinalAnalyzer] Dataset spans "
                f"{self._data_start.date()} → {self._data_end.date()} "
                f"({len(valid_ts):,} timestamped samples)"
            )
        else:
            self._data_start = self._data_end = None
            print("[LongitudinalAnalyzer] WARNING: No valid timestamps found in dataset.")

    # ------------------------------------------------------------------

    def _resolve_windows(self, parsed: dict) -> tuple[TimeWindow, TimeWindow]:
        period = parsed.get("period", {})
        ws_days = {"day": 1, "week": 7, "month": 30}.get(
            parsed.get("window_size", "week"), self.window_size_days
        )

        if period.get("type") == "absolute" and period.get("start_date") and period.get("end_date"):
            p_start = datetime.strptime(period["start_date"], "%Y-%m-%d")
            p_end = datetime.strptime(period["end_date"], "%Y-%m-%d")
        elif self._data_start and self._data_end:
            p_start = self._data_start.replace(hour=0, minute=0, second=0, microsecond=0)
            p_end = self._data_end.replace(hour=23, minute=59, second=59, microsecond=0)
        else:
            raise ValueError(
                "Cannot determine time period: dataset has no timestamps and "
                "no absolute period was specified in the question."
            )

        wa = TimeWindow(p_start, p_start + timedelta(days=ws_days), f"first {ws_days} days")
        wb = TimeWindow(p_end - timedelta(days=ws_days), p_end, f"last {ws_days} days")
        return wa, wb

    # ------------------------------------------------------------------

    def analyze(self, user_question: str) -> dict:
        """
        Run longitudinal analysis for a natural-language question.

        Returns a dict ready for JSON serialisation and dashboard consumption.
        """
        print(f"\n[LongitudinalAnalyzer] Question: {user_question!r}")

        # 1. Parse question
        print("[LongitudinalAnalyzer] Parsing question with LLM…")
        parsed = _parse_user_question(user_question, self._llm)
        activities = parsed.get("activities") or [user_question]
        print(f"[LongitudinalAnalyzer] Activities identified: {activities}")
        print(f"[LongitudinalAnalyzer] Period: {parsed.get('period', {}).get('description', 'auto')}")

        # 2. Resolve time windows
        window_a, window_b = self._resolve_windows(parsed)
        print(f"[LongitudinalAnalyzer] Window A ({window_a.label}): {window_a.start.date()} → {window_a.end.date()}")
        print(f"[LongitudinalAnalyzer] Window B ({window_b.label}): {window_b.start.date()} → {window_b.end.date()}")

        # 3. Build per-window FAISS sub-indices (temporal filtering happens here)
        print("[LongitudinalAnalyzer] Building temporal sub-indices…")
        sub_idx_a, valid_a = _build_window_subindex(self.retrieval, self._timestamps, window_a)
        sub_idx_b, valid_b = _build_window_subindex(self.retrieval, self._timestamps, window_b)
        print(f"[LongitudinalAnalyzer] Samples in Window A: {len(valid_a):,} | Window B: {len(valid_b):,}")

        # 4. Process each activity
        all_results: list[dict] = []
        for activity in activities:
            print(f"\n[LongitudinalAnalyzer] ── Activity: {activity!r}")

            query_reasoning, sentences = self.rewriter.rewrite(activity)
            print(f"[LongitudinalAnalyzer]    {len(sentences)} retrieval sentence(s) generated")

            wa_result = _retrieve_in_window(
                self.retrieval, sentences, sub_idx_a, valid_a,
                self._timestamps, window_a, self.threshold,
            )
            wb_result = _retrieve_in_window(
                self.retrieval, sentences, sub_idx_b, valid_b,
                self._timestamps, window_b, self.threshold,
            )
            stats = _compute_stats(wa_result, wb_result)

            change_str = f"{stats['pct_change']:+.1f}%" if stats["pct_change"] is not None else "N/A"
            print(
                f"[LongitudinalAnalyzer]    {window_a.label}: {wa_result.total_matches} matches | "
                f"{window_b.label}: {wb_result.total_matches} matches | Δ {change_str}"
            )

            all_results.append({
                "activity": activity,
                "query_reasoning": query_reasoning,
                "rewritten_queries": sentences,
                "window_a": wa_result.to_dict(window_a),
                "window_b": wb_result.to_dict(window_b),
                "stats": stats,
            })

        # 5. LLM synthesis
        print("\n[LongitudinalAnalyzer] Synthesizing report…")
        report = _synthesize_report(
            user_question, all_results, self._llm,
            window_a.label, window_b.label,
        )

        return {
            "question": user_question,
            "generated_at": datetime.now().isoformat(),
            "parsed": parsed,
            "windows": {
                "window_a": window_a.to_dict(),
                "window_b": window_b.to_dict(),
            },
            "similarity_threshold": self.threshold,
            "results": all_results,
            "report": report,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Longitudinal activity analysis via LLM + FAISS retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--question", required=True,
                   help="Natural-language question about activity over time.")
    p.add_argument("--home", default="milan",
                   help="Dataset home identifier (milan, aruba, …).")
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained model checkpoint (.pt).")
    p.add_argument("--test_data", required=True,
                   help="Path to test.json data file.")
    p.add_argument("--vocab", required=True,
                   help="Path to vocab.json.")
    p.add_argument("--captions", default=None,
                   help="Path to captions JSON for style examples (optional).")
    p.add_argument("--metadata", default=None,
                   help="Path to casas_metadata.json (optional override).")
    p.add_argument("--output", default=None,
                   help="Output JSON path. Defaults to results/longitudinal/<home>_<timestamp>.json.")
    p.add_argument("--backend", default="gemini", choices=["gemini", "openai"],
                   help="LLM backend.")
    p.add_argument("--model", default=None,
                   help="Override LLM model name.")
    p.add_argument("--api_key", default=None,
                   help="LLM API key (falls back to GEMINI_API_KEY / OPENAI_API_KEY env vars).")
    p.add_argument("--threshold", type=float, default=0.25,
                   help="Cosine similarity threshold for counting a retrieval match.")
    p.add_argument("--window_days", type=int, default=7,
                   help="Number of days in each comparison window.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Max sensor sequences to load into the FAISS index. Defaults to all samples.")
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Lazy import of the heavy retrieval stack
    from evals.query_retrieval import SmartHomeRetrieval

    print(f"[llm_reasoning] Loading retrieval system for {args.home}…")
    max_samples = args.max_samples if args.max_samples is not None else 10_000_000
    retrieval = SmartHomeRetrieval(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        test_data_path=args.test_data,
        captions_path=args.captions,
        max_samples=max_samples,
        metadata_path=args.metadata,
    )

    analyzer = LongitudinalAnalyzer(
        retrieval=retrieval,
        home=args.home,
        backend=args.backend,
        api_key=args.api_key,
        model=args.model,
        similarity_threshold=args.threshold,
        window_size_days=args.window_days,
        metadata_path=args.metadata,
        captions_path=args.captions or False,
    )

    output = analyzer.analyze(args.question)

    # Determine output path
    out_path = args.output
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results/longitudinal")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{args.home}_{ts}.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[llm_reasoning] Results saved to {out_path}")
    print(f"\n{'='*60}\nREPORT\n{'='*60}")
    print(output["report"])

    # Auto-generate chart alongside the JSON
    try:
        from query.plot_longitudinal import plot_report
        plot_path = str(Path(out_path).with_suffix(".png"))
        plot_report(output, output_path=plot_path, show=False)
        print(f"[llm_reasoning] Chart saved to  {plot_path}")
    except Exception as e:
        print(f"[llm_reasoning] Chart generation skipped: {e}")


if __name__ == "__main__":
    main()
