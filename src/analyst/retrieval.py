"""Deterministic retrieval: concept queries -> matched windows -> episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import WINDOW_SECONDS
from .index import SensorIndex
from .schemas import Concept


@dataclass
class ConceptMatches:
    concept: Concept
    threshold: float
    eligible: pd.DataFrame          # windows passing scope + hard filters, with score columns
    per_query_counts: dict
    n_matched_without_structural_filters: int

    @property
    def matched(self) -> pd.DataFrame:
        return self.eligible[self.eligible["matched"]]


def scope_mask(index: SensorIndex, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> np.ndarray:
    s = index.meta["start"]
    mask = np.ones(len(s), dtype=bool)
    lo = max([t for t in (start, index.cfg.data_start) if t is not None], default=None)
    if lo is not None:
        mask &= (s >= pd.Timestamp(lo)).to_numpy()
    if end is not None:
        mask &= (s < pd.Timestamp(end) + pd.Timedelta(days=1)).to_numpy()
    return mask


def time_filter_mask(index: SensorIndex, concept: Concept) -> np.ndarray:
    m = np.ones(len(index.meta), dtype=bool)
    if concept.time_of_day:
        m &= index.meta["tod"].isin(concept.time_of_day).to_numpy()
    if concept.days_of_week:
        m &= index.meta["dow"].isin(concept.days_of_week).to_numpy()
    return m


def structural_filter_mask(index: SensorIndex, concept: Concept) -> np.ndarray:
    """Room-visit and required-sensor filters (instrumentation-level constraints)."""
    m = np.ones(len(index.meta), dtype=bool)
    if concept.rooms:
        rooms = set(concept.rooms)
        m &= index.meta["rooms_visited"].map(lambda s: bool(rooms.intersection(s.split("|")))).to_numpy()
    if concept.required_sensors:
        m &= index.windows_with_activation(concept.required_sensors)
    return m


def retrieve_concept(index: SensorIndex, concept: Concept, scope: np.ndarray, threshold: float) -> ConceptMatches:
    queries = concept.retrieval_queries
    time_ok = scope & time_filter_mask(index, concept)
    eligible_mask = time_ok & structural_filter_mask(index, concept)
    df = index.meta.loc[eligible_mask].copy()
    if not queries or not concept.detectable:
        df["score"] = np.nan
        df["best_query"] = -1
        df["matched"] = False
        return ConceptMatches(concept, threshold, df, {}, 0)

    sims_all = index.similarity(queries)
    best_all = sims_all.max(axis=0)
    sims = sims_all[:, eligible_mask]          # (Q, n_eligible)
    df["score"] = sims.max(axis=0)
    df["best_query"] = sims.argmax(axis=0)
    df["matched"] = df["score"] >= threshold
    per_query = {q: int((sims[i] >= threshold).sum()) for i, q in enumerate(queries)}
    embedding_only = int((time_ok & (best_all >= threshold)).sum())
    return ConceptMatches(concept, threshold, df, per_query, embedding_only)


def union_intervals(starts: pd.Series, window_s: float = WINDOW_SECONDS, gap_s: float = 0.0) -> List[tuple]:
    """Merge [start, start+window] intervals; intervals closer than gap_s are joined."""
    if len(starts) == 0:
        return []
    t = np.sort(starts.to_numpy().astype("datetime64[ns]").astype(np.int64)) / 1e9
    out, cur_s, cur_e = [], t[0], t[0] + window_s
    for x in t[1:]:
        if x <= cur_e + gap_s:
            cur_e = max(cur_e, x + window_s)
        else:
            out.append((cur_s, cur_e))
            cur_s, cur_e = x, x + window_s
    out.append((cur_s, cur_e))
    return out


def episodes(matched: pd.DataFrame, gap_s: float) -> pd.DataFrame:
    iv = union_intervals(matched["start"], gap_s=gap_s)
    if not iv:
        return pd.DataFrame(columns=["start", "end", "minutes"])
    arr = np.array(iv)
    return pd.DataFrame({
        "start": pd.to_datetime(arr[:, 0], unit="s"),
        "end": pd.to_datetime(arr[:, 1], unit="s"),
        "minutes": (arr[:, 1] - arr[:, 0]) / 60.0,
    })
