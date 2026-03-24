"""
SmartQuery: LLM-powered interface for the smart-home retrieval system.

Rewrite modes
-------------
single          One rich sentence → one FAISS search.
multi_location  One sentence per relevant location → one FAISS search per
                sentence, results merged by best score across queries.
multi_wording   Several paraphrases → one FAISS search per paraphrase,
                results merged by best score.

For multi modes the final ranked list deduplicates by sample index and
keeps the best (highest) similarity score seen across all sub-queries.
Each result also records which sub-query produced the best match.

Usage (programmatic)
--------------------
    from query import SmartQuery

    sq = SmartQuery.from_checkpoint(
        checkpoint_path="trained_models/milan/best_model.pt",
        test_data_path="data/processed/casas/milan/FD_60/test.json",
        vocab_path="data/processed/casas/milan/FD_60/vocab.json",
        home="milan",
    )
    out = sq.query("give me all sedentary activities",
                   mode="multi_location", top_k=10)

Usage (CLI)
-----------
    python src/query/smart_query.py \\
        --checkpoint trained_models/milan/best_model.pt \\
        --test_data  data/processed/casas/milan/FD_60/test.json \\
        --vocab      data/processed/casas/milan/FD_60/vocab.json \\
        --home       milan \\
        --query      "give me all sedentary activities" \\
        --mode       multi_location \\
        --top_k      10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from query.query_cache import QueryCache
from query.llm_rewriter import LLMRewriter, RewriteMode

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MERGED_DIR   = _PROJECT_ROOT / "data" / "query_cache" / "merged"

# ---------------------------------------------------------------------------
# Split-merge helpers
# ---------------------------------------------------------------------------

def _load_json_samples(path: Path) -> list[dict]:
    """Load samples from a JSON file (handles both list and {samples: [...]} formats)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognised format in {path}")


def _merge_splits(
    data_paths: list[Path],
    caption_style: str = "baseline",
) -> tuple[Path, Path | None]:
    """
    Merge split JSON files into single cached files.

    Returns:
        (merged_data_path, merged_captions_path | None)

    The merged files are stored in data/query_cache/merged/ and reused as
    long as the source files haven't changed (checked via mtime).
    """
    import hashlib

    _MERGED_DIR.mkdir(parents=True, exist_ok=True)

    existing = [p for p in data_paths if p.exists()]
    if not existing:
        raise FileNotFoundError(f"None of the data paths exist: {data_paths}")

    # Cache key: sorted paths + their modification times
    key_str = "|".join(f"{p}:{p.stat().st_mtime:.0f}" for p in sorted(existing))
    key_hash = hashlib.md5(key_str.encode()).hexdigest()[:10]

    merged_data_path = _MERGED_DIR / f"data_{key_hash}.json"
    merged_cap_path  = _MERGED_DIR / f"captions_{caption_style}_{key_hash}.json"

    # ---- data ---------------------------------------------------------------
    if not merged_data_path.exists():
        all_samples: list[dict] = []
        for p in existing:
            all_samples.extend(_load_json_samples(p))
        with open(merged_data_path, "w") as f:
            json.dump(all_samples, f)
        print(f"[merge] Written {len(all_samples)} samples → {merged_data_path.name}")
    else:
        print(f"[merge] Using cached merged data: {merged_data_path.name}")

    # ---- captions -----------------------------------------------------------
    # Corresponding caption files live next to each split file
    cap_files = [
        p.parent / p.name.replace(".json", f"_captions_{caption_style}.json")
        for p in existing
    ]
    existing_cap = [p for p in cap_files if p.exists()]

    if existing_cap and not merged_cap_path.exists():
        all_items: list[dict] = []
        for cp in existing_cap:
            with open(cp) as f:
                d = json.load(f)
            items = d.get("captions", d) if isinstance(d, dict) else d
            if isinstance(items, list):
                all_items.extend(items)
        merged_cap_data = {"captions": all_items}
        with open(merged_cap_path, "w") as f:
            json.dump(merged_cap_data, f)
        print(f"[merge] Written {len(all_items)} captions → {merged_cap_path.name}")
    elif not existing_cap:
        merged_cap_path = None  # type: ignore[assignment]
    else:
        print(f"[merge] Using cached merged captions: {merged_cap_path.name}")

    return merged_data_path, merged_cap_path


# ---------------------------------------------------------------------------
# Multi-query FAISS retrieval helper
# ---------------------------------------------------------------------------

def _encode_sentence(retrieval, sentence: str):
    """Encode one sentence through text encoder + projection → (1, d) numpy float32."""
    import numpy as np
    import torch

    with torch.no_grad():
        emb = retrieval.text_encoder.encode_texts_clip([sentence], retrieval.device)
        if retrieval.text_projection is not None:
            emb = retrieval.text_projection(emb)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.cpu().numpy().astype(np.float32)   # (1, d)


def _search_topk(retrieval, emb_np, fetch_k: int):
    """Top-k FAISS search. Returns (scores_1d, indices_1d)."""
    scores, indices = retrieval.sensor_index.search(emb_np, k=fetch_k)
    return scores[0], indices[0]


def _search_threshold(retrieval, emb_np, threshold: float):
    """
    Range search — returns ALL samples with cosine similarity >= threshold.
    Uses IndexFlatIP.range_search which is exact and allocation-free on the caller side.
    Returns (scores_1d, indices_1d).
    """
    lims, distances, indices = retrieval.sensor_index.range_search(emb_np, thresh=threshold)
    return distances, indices   # already 1-D arrays for a single query vector


def _multi_query_retrieve(
    retrieval,
    sentences: list[str],
    top_k: int,
    threshold: float | None = None,
) -> list[dict]:
    """
    Encode + FAISS-search one sentence at a time, then merge by best score.

    Args:
        top_k:      Max results when threshold=None (classic top-k mode).
        threshold:  Cosine similarity floor (0–1). When set, returns ALL samples
                    above the threshold with no top_k cap.

    Each result is augmented with:
      matched_query  – sentence that produced the best score
      all_scores     – {sentence: score} across all sub-queries
    """
    import numpy as np

    fetch_k = min(top_k * max(len(sentences), 2), retrieval.sensor_index.ntotal)

    # Merge: best score per FAISS sample index across all sentences
    best: dict[int, dict] = {}
    for sent in sentences:
        emb_np = _encode_sentence(retrieval, sent)

        if threshold is not None:
            scores_row, indices_row = _search_threshold(retrieval, emb_np, threshold)
        else:
            scores_row, indices_row = _search_topk(retrieval, emb_np, fetch_k)

        for score, idx in zip(scores_row, indices_row):
            if idx < 0:
                continue
            score = float(score)
            prev = best.get(idx)
            if prev is None or score > prev["score"]:
                all_scores = (prev["all_scores"] if prev else {})
                all_scores[sent] = score
                best[idx] = {"score": score, "matched_query": sent, "all_scores": all_scores}
            else:
                best[idx]["all_scores"][sent] = score

    # Sort by best score descending; cap at top_k only when no threshold is set
    ranked = sorted(best.items(), key=lambda x: x[1]["score"], reverse=True)
    if threshold is None:
        ranked = ranked[:top_k]

    # Decode each result using the retrieval system's existing methods
    results = []
    for rank, (sample_idx, meta) in enumerate(ranked):
        batch_idx, s_idx = retrieval.sample_indices[sample_idx]
        caption_data, labels_info = retrieval._get_caption_and_labels(batch_idx, s_idx)
        events, seq_meta = retrieval._decode_sequence(batch_idx, s_idx)

        result = {
            "rank": rank + 1,
            "score": meta["score"],
            "matched_query": meta["matched_query"],
            "all_scores": meta.get("all_scores", {}),
            "caption_data": caption_data,
            "events": events,
            "metadata": seq_meta,
            "labels": labels_info,
            "batch_idx": batch_idx,
            "sample_idx": s_idx,
        }
        results.append(result)

    return results


_MAX_DISPLAY = 10


def _print_results(results: list[dict], sentences: list[str]) -> None:
    """Pretty-print merged retrieval results (capped at _MAX_DISPLAY)."""
    if len(sentences) > 1:
        print("\nSub-queries used:")
        for i, s in enumerate(sentences, 1):
            print(f"  [{i}] {s}")

    displayed = results[:_MAX_DISPLAY]
    for r in displayed:
        labels = r.get("labels", {})
        sample_id = labels.get("sample_id", f"sample_{r['rank']}")
        print(f"\n{'='*70}")
        print(f"RANK {r['rank']:>2}  |  score={r['score']:.4f}  |  {sample_id}")

        if len(sentences) > 1:
            # Show which sub-query matched best and all individual scores
            mq = r.get("matched_query", "")
            short = mq[:70] + "…" if len(mq) > 70 else mq
            print(f"  Best match: \"{short}\"")
            score_parts = [
                f"[{i+1}]={v:.3f}"
                for i, s in enumerate(sentences)
                for k, v in r.get("all_scores", {}).items()
                if k == s
            ]
            if score_parts:
                print(f"  Per-query : {' '.join(score_parts)}")

        print("-" * 70)
        l1 = labels.get("activity_l1", "")
        l2 = labels.get("activity_l2", "")
        if l1 or l2:
            label_str = " | ".join(x for x in [f"L1: {l1}", f"L2: {l2}"] if x.split(": ")[1])
            print(f"  Labels   : {label_str}")

        parts = []
        if labels.get("num_events"):
            parts.append(f"{labels['num_events']} events")
        if labels.get("duration_seconds"):
            parts.append(f"{labels['duration_seconds']:.1f}s")
        events = r.get("events", [])
        if events:
            tod = events[0].get("tod_bucket", "")
            if tod and tod != "UNK":
                parts.append(f"ToD: {tod}")
        if parts:
            print(f"  Time     : {' | '.join(parts)}")

        if labels.get("primary_room"):
            print(f"  Location : {labels['primary_room']}")

        caps = r.get("caption_data", {}).get("captions", [])
        if caps and caps[0]:
            cap = caps[0]
            print(f"  Caption  : {cap[:120]}{'…' if len(cap)>120 else ''}")

    if len(results) > _MAX_DISPLAY:
        print(f"\n  ... {len(results) - _MAX_DISPLAY} more result(s) not shown "
              f"(full set in returned dict / cache)")


# ---------------------------------------------------------------------------
# SmartQuery
# ---------------------------------------------------------------------------

class SmartQuery:
    """
    Combines LLM query rewriting with FAISS-based sensor-sequence retrieval.
    """

    def __init__(
        self,
        retrieval_system=None,
        rewriter: Optional[LLMRewriter] = None,
        cache: Optional[QueryCache] = None,
        home: str = "milan",
        verbose: bool = True,
    ):
        self.retrieval = retrieval_system
        self.rewriter = rewriter
        self.cache = cache or QueryCache()
        self.home = home
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_data_dir(
        cls,
        checkpoint_path: str,
        data_dir: str,
        home: str = "milan",
        splits: list[str] | None = None,
        caption_style: str = "baseline",
        max_samples: int = 50_000,
        gemini_api_key: Optional[str] = None,
        llm_backend: str = "gemini",
        llm_model: Optional[str] = None,
        llm_captions_path: Optional[str] = None,
        llm_n_examples: int = 6,
        cache_db_path: Optional[str] = None,
        verbose: bool = True,
    ) -> "SmartQuery":
        """
        Load ALL splits (train + val + test by default) from a dataset directory,
        merge them into a single index, and return a ready SmartQuery.

        Args:
            checkpoint_path: Path to model checkpoint.
            data_dir:        Directory containing train.json, val.json, test.json
                             and vocab.json (e.g. data/processed/casas/milan/FD_60).
            home:            Dataset name for the LLM rewriter and cache.
            splits:          Which splits to include. Defaults to ["train","val","test"].
            caption_style:   Caption file suffix (e.g. "baseline" →
                             train_captions_baseline.json). Defaults to "baseline".
            max_samples:     Upper cap on samples passed to the FAISS index.
                             Defaults to 50 000 (effectively all samples for most datasets).
        """
        data_dir  = Path(data_dir)
        splits    = splits or ["train", "val", "test"]
        vocab_path = str(data_dir / "vocab.json")

        data_paths = [data_dir / f"{s}.json" for s in splits]
        merged_data, merged_caps = _merge_splits(data_paths, caption_style)

        if verbose:
            n = len(_load_json_samples(merged_data))
            print(f"[SmartQuery] Full dataset: {n} samples across splits {splits}")

        return cls.from_checkpoint(
            checkpoint_path=checkpoint_path,
            test_data_path=str(merged_data),
            vocab_path=vocab_path,
            home=home,
            captions_path=str(merged_caps) if merged_caps else None,
            max_samples=max_samples,
            gemini_api_key=gemini_api_key,
            llm_backend=llm_backend,
            llm_model=llm_model,
            llm_captions_path=llm_captions_path,
            llm_n_examples=llm_n_examples,
            cache_db_path=cache_db_path,
            verbose=verbose,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        test_data_path: str,
        vocab_path: str,
        home: str = "milan",
        captions_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        max_samples: int = 5000,
        gemini_api_key: Optional[str] = None,
        llm_backend: str = "gemini",
        llm_model: Optional[str] = None,
        llm_captions_path: Optional[str] = None,
        llm_n_examples: int = 6,
        cache_db_path: Optional[str] = None,
        verbose: bool = True,
    ) -> "SmartQuery":
        from evals.query_retrieval import SmartHomeRetrieval

        if verbose:
            print(f"[SmartQuery] Loading retrieval system for home='{home}' …")

        retrieval = SmartHomeRetrieval(
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            test_data_path=test_data_path,
            captions_path=captions_path,
            max_samples=max_samples,
            metadata_path=metadata_path,
        )

        rewriter = None
        api_key = (
            gemini_api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        if api_key:
            rewriter = LLMRewriter(
                home=home,
                backend=llm_backend,
                api_key=api_key,
                model=llm_model,
                captions_path=llm_captions_path,
                n_examples=llm_n_examples,
            )
            if verbose:
                print(f"[SmartQuery] LLM rewriter ready ({rewriter.model_id})")
        else:
            if verbose:
                print(
                    "[SmartQuery] No API key found — queries will not be rewritten. "
                    "Set GEMINI_API_KEY or GOOGLE_API_KEY."
                )

        cache = QueryCache(cache_db_path) if cache_db_path else QueryCache()
        return cls(
            retrieval_system=retrieval,
            rewriter=rewriter,
            cache=cache,
            home=home,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Core query method
    # ------------------------------------------------------------------

    def query(
        self,
        user_query: str,
        mode: RewriteMode = "single",
        top_k: int = 5,
        threshold: float | None = None,
        force_rewrite: bool = False,
        force_retrieve: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a natural-language query against the sensor embedding space.

        Args:
            user_query:     Raw user question.
            mode:           "single" | "multi_location" | "multi_wording"
            top_k:          Max results when threshold=None (default: 5).
            threshold:      Cosine similarity floor (e.g. 0.10). When set,
                            returns ALL samples above the threshold with no
                            top_k cap. Supersedes top_k.
            force_rewrite:  Bypass rewrite cache; always call the LLM.
            force_retrieve: Bypass result cache; always run FAISS retrieval.

        Returns:
            dict with keys:
              original_query, rewrite_mode, sentences (list[str]),
              rewrite_cache_hit, result_cache_hit, model_used, results (list[dict])
        """
        if self.retrieval is None:
            raise RuntimeError(
                "No retrieval system attached. Use SmartQuery.from_checkpoint()."
            )

        original = user_query.strip()
        reasoning, sentences, rewrite_cache_hit, model_used = self._rewrite(
            original, mode, force_rewrite
        )

        if self.verbose:
            rewrite_status = "cache hit" if rewrite_cache_hit else f"rewritten via {model_used or 'passthrough'}"
            print(f"\n[SmartQuery] Query  : {original!r}  ({rewrite_status})")
            if reasoning:
                print(f"\n[SmartQuery] Reasoning:\n{reasoning}\n")
            print(f"[SmartQuery] Mode   : {mode}  →  {len(sentences)} sentence(s)")
            for i, s in enumerate(sentences, 1):
                print(f"  [{i}] {s}")
            if threshold is not None:
                print(f"[SmartQuery] Threshold: >={threshold:.3f}  (no top_k cap)")
            else:
                print(f"[SmartQuery] Top-k    : {top_k}")

        if not sentences:
            print("\n" + "=" * 60)
            print("  NOT DETECTABLE")
            print("  The LLM determined this activity cannot be inferred")
            print("  from the sensors available in this home.")
            print("=" * 60)
            return {
                "original_query": original,
                "rewrite_mode": mode,
                "reasoning": reasoning,
                "sentences": [],
                "rewrite_cache_hit": rewrite_cache_hit,
                "result_cache_hit": False,
                "model_used": model_used,
                "results": [],
            }

        checkpoint_key = self._checkpoint_key()
        cache_top_k = top_k if threshold is None else -1

        # Check result cache
        result_cache_hit = False
        if not force_retrieve:
            cached_results = self.cache.get_results(
                original, self.home, mode, checkpoint_key, cache_top_k
            )
            if cached_results is not None:
                result_cache_hit = True
                if self.verbose:
                    print(f"[SmartQuery] Results : cache hit ({len(cached_results)} results)")
                _print_results(cached_results, sentences)
                return {
                    "original_query": original,
                    "rewrite_mode": mode,
                    "reasoning": reasoning,
                    "sentences": sentences,
                    "rewrite_cache_hit": rewrite_cache_hit,
                    "result_cache_hit": True,
                    "model_used": model_used,
                    "results": cached_results,
                }

        results = _multi_query_retrieve(self.retrieval, sentences, top_k, threshold=threshold)
        _print_results(results, sentences)

        if self.verbose and threshold is not None:
            print(f"\n[SmartQuery] {len(results)} sample(s) above threshold {threshold:.3f}")

        # Store results — strip non-serialisable fields (raw batch refs)
        # Use negative top_k as the cache key when threshold mode is active
        cache_top_k = top_k if threshold is None else -1
        self.cache.store_results(
            original_query=original,
            results=[{k: v for k, v in r.items() if k not in ("batch_idx", "sample_idx")}
                     for r in results],
            home=self.home,
            rewrite_mode=mode,
            checkpoint=checkpoint_key,
            top_k=cache_top_k,
        )

        return {
            "original_query": original,
            "rewrite_mode": mode,
            "reasoning": reasoning,
            "sentences": sentences,
            "rewrite_cache_hit": rewrite_cache_hit,
            "result_cache_hit": result_cache_hit,
            "model_used": model_used,
            "results": results,
        }

    def rewrite_only(
        self,
        user_query: str,
        mode: RewriteMode = "single",
        force_rewrite: bool = False,
    ) -> Dict[str, Any]:
        """Return the rewritten sentences without running retrieval."""
        original = user_query.strip()
        reasoning, sentences, cache_hit, model_used = self._rewrite(original, mode, force_rewrite)
        return {
            "original_query": original,
            "rewrite_mode": mode,
            "reasoning": reasoning,
            "sentences": sentences,
            "cache_hit": cache_hit,
            "model_used": model_used,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _checkpoint_key(self) -> str:
        """Stable, path-independent key for the loaded checkpoint."""
        if self.retrieval is None:
            return ""
        path = getattr(self.retrieval, "checkpoint_path", "")
        return Path(path).name if path else ""

    def _rewrite(
        self, original: str, mode: RewriteMode, force: bool
    ) -> tuple[str, list[str], bool, str]:
        """Returns (reasoning, sentences, cache_hit, model_used)."""
        if not force:
            cached = self.cache.get(original, self.home, rewrite_mode=mode)
            if cached is not None:
                return "", cached, True, ""

        if self.rewriter is None:
            return "", [original], False, ""

        reasoning, sentences = self.rewriter.rewrite(original, mode=mode)
        self.cache.store(
            original_query=original,
            sentences=sentences,
            home=self.home,
            rewrite_mode=mode,
            model_used=self.rewriter.model_id,
        )
        return reasoning, sentences, False, self.rewriter.model_id

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def cache_stats(self) -> dict:
        return self.cache.stats()

    def list_cached_queries(self) -> list[dict]:
        return self.cache.list_entries(home=self.home)

    def clear_cache(self) -> int:
        return self.cache.clear(home=self.home)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM-powered smart-home query interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pt) [required for retrieval]")
    p.add_argument("--test_data",  default=None, help="Path to a single split JSON [use --data_dir for all splits]")
    p.add_argument("--data_dir",   default=None,
                   help="Dataset directory (e.g. data/processed/casas/milan/FD_60). "
                        "Loads train+val+test and merges them automatically. "
                        "Supersedes --test_data / --vocab / --captions.")
    p.add_argument("--splits",     default="train,val,test",
                   help="Comma-separated splits to load when --data_dir is used (default: train,val,test)")
    p.add_argument("--caption_style", default="baseline",
                   help="Caption file suffix when --data_dir is used (default: baseline)")
    p.add_argument("--vocab",      default=None, help="Path to vocab.json [required without --data_dir]")
    p.add_argument("--captions",   default=None)
    p.add_argument("--metadata",   default=None)
    p.add_argument("--home",       default="milan")
    p.add_argument("--query",      default=None)
    p.add_argument("--mode",       default="single",
                   choices=["single", "multi_location", "multi_wording"],
                   help="Rewrite mode (default: single)")
    p.add_argument("--top_k",        type=int,   default=5,
                   help="Max results in top-k mode (default: 5). Ignored when --threshold is set.")
    p.add_argument("--threshold",    type=float, default=None,
                   help="Cosine similarity floor (e.g. 0.10). Returns ALL samples above this "
                        "score with no top_k cap. Recommended range: 0.05–0.25.")
    p.add_argument("--max_samples",  type=int, default=5000)
    p.add_argument("--rewrite_only", action="store_true",
                   help="Show rewritten sentences without loading the model")
    p.add_argument("--cache_stats",  action="store_true")
    p.add_argument("--list_cache",   action="store_true")
    p.add_argument("--llm_backend",  default="gemini", choices=["gemini", "openai"])
    p.add_argument("--llm_model",    default=None)
    p.add_argument("--gemini_api_key", default=None)
    p.add_argument("--llm_captions_path", default=None,
                   help="Path to train_captions_*.json for style examples "
                        "(default: data/processed/casas/{home}/FD_60/train_captions_baseline.json). "
                        "Pass 'none' to disable.")
    p.add_argument("--llm_n_examples", type=int, default=6,
                   help="Number of random captions to use as style examples (default: 6)")
    p.add_argument("--cache_db",     default=None)
    p.add_argument("--force_rewrite",   action="store_true",
                   help="Bypass rewrite cache; always call the LLM")
    p.add_argument("--force_retrieve",  action="store_true",
                   help="Bypass result cache; always run FAISS retrieval")
    return p


def main():
    args = _build_parser().parse_args()

    # Cache-only operations
    if args.cache_stats or args.list_cache:
        cache = QueryCache(args.cache_db) if args.cache_db else QueryCache()
        if args.cache_stats:
            print(json.dumps(cache.stats(), indent=2))
        if args.list_cache:
            for e in cache.list_entries(home=args.home):
                print(f"[{e['home']}] [{e['rewrite_mode']}] {e['original_query']!r}  "
                      f"hits={e['hit_count']}  model={e['model_used']}")
                for i, s in enumerate(e["sentences"], 1):
                    print(f"  [{i}] {s}")
                print()
        return

    # Rewrite-only (no model loading)
    if args.rewrite_only:
        caps_path = None if (not args.llm_captions_path or args.llm_captions_path.lower() == "none") \
                    else args.llm_captions_path
        rewriter = LLMRewriter(
            home=args.home,
            backend=args.llm_backend,
            api_key=args.gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            model=args.llm_model,
            captions_path=False if args.llm_captions_path and args.llm_captions_path.lower() == "none" else caps_path,
            n_examples=args.llm_n_examples,
        )
        cache = QueryCache(args.cache_db) if args.cache_db else QueryCache()
        sq = SmartQuery(rewriter=rewriter, cache=cache, home=args.home)

        queries = [args.query] if args.query else []
        if not queries:
            print("Enter queries (empty line to quit):")
            while True:
                q = input("Query: ").strip()
                if not q:
                    break
                queries.append(q)

        for q in queries:
            out = sq.rewrite_only(q, mode=args.mode, force_rewrite=args.force_rewrite)
            print(f"\nOriginal  : {out['original_query']}")
            print(f"Mode      : {out['rewrite_mode']}")
            print(f"Cache hit : {out['cache_hit']}  model: {out['model_used']}")
            if out.get("reasoning"):
                print(f"\nReasoning :\n{out['reasoning']}\n")
            if out["sentences"]:
                for i, s in enumerate(out["sentences"], 1):
                    print(f"  [{i}] {s}")
            else:
                print("\n" + "=" * 60)
                print("  NOT DETECTABLE")
                print("  The LLM determined this activity cannot be inferred")
                print("  from the sensors available in this home.")
                print("=" * 60)
        return

    # Full mode: rewrite + retrieve
    if not args.checkpoint:
        _build_parser().error("--checkpoint is required for retrieval")

    if args.data_dir:
        sq = SmartQuery.from_data_dir(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            home=args.home,
            splits=[s.strip() for s in args.splits.split(",")],
            caption_style=args.caption_style,
            max_samples=args.max_samples,
            gemini_api_key=args.gemini_api_key,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            llm_captions_path=args.llm_captions_path,
            llm_n_examples=args.llm_n_examples,
            cache_db_path=args.cache_db,
        )
    else:
        missing = [f"--{f}" for f, v in [("test_data", args.test_data), ("vocab", args.vocab)] if v is None]
        if missing:
            _build_parser().error(
                f"the following arguments are required when not using --data_dir: {', '.join(missing)}"
            )
        sq = SmartQuery.from_checkpoint(
            checkpoint_path=args.checkpoint,
            test_data_path=args.test_data,
            vocab_path=args.vocab,
            home=args.home,
            captions_path=args.captions,
            metadata_path=args.metadata,
            max_samples=args.max_samples,
            gemini_api_key=args.gemini_api_key,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            llm_captions_path=args.llm_captions_path,
            llm_n_examples=args.llm_n_examples,
            cache_db_path=args.cache_db,
        )

    if args.query:
        sq.query(args.query, mode=args.mode, top_k=args.top_k,
                 threshold=args.threshold,
                 force_rewrite=args.force_rewrite, force_retrieve=args.force_retrieve)
    else:
        thresh_str = f"  threshold={args.threshold}" if args.threshold else f"  top_k={args.top_k}"
        print(f"\n[SmartQuery] Interactive mode — mode={args.mode!r}{thresh_str}")
        print("Commands: 'mode single|multi_location|multi_wording', 'cache', 'quit'\n")
        current_mode: RewriteMode = args.mode
        while True:
            try:
                raw = input("Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not raw:
                continue
            if raw.lower() in ("quit", "exit", "q"):
                break
            if raw.lower() == "cache":
                print(json.dumps(sq.cache_stats(), indent=2))
                continue
            if raw.lower().startswith("mode "):
                m = raw.split(None, 1)[1].strip()
                if m in ("single", "multi_location", "multi_wording"):
                    current_mode = m  # type: ignore[assignment]
                    print(f"Mode set to: {current_mode}")
                else:
                    print("Unknown mode. Choose: single | multi_location | multi_wording")
                continue
            sq.query(raw, mode=current_mode, top_k=args.top_k,
                     threshold=args.threshold,
                     force_rewrite=args.force_rewrite, force_retrieve=args.force_retrieve)


if __name__ == "__main__":
    main()
