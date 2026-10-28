"""Static configuration for the behavioral analyst pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "metadata" / "casas_metadata.json"
FLOOR_PLAN_DIR = PROJECT_ROOT / "metadata" / "floor_plans_augmented"

# LLM responses are small JSON files; kept under metadata/ (tracked by git) so the
# exact planner/reporter outputs used in the paper are reproducible without API access.
LLM_CACHE_DIR = PROJECT_ROOT / "metadata" / "analyst_llm_cache"
EMBED_CACHE_DIR = PROJECT_ROOT / "data" / "analyst_cache"
RESULTS_DIR = PROJECT_ROOT / "results" / "analyst"

DEFAULT_LLM_MODEL = "gemini-3.1-pro-preview"

MODEL_SUFFIX = "fd60_seq_rb1_textclip_projmlp_clipmlm_v3"
DATA_VARIANT = "FD_60"
SEQUENCE_LENGTH = 256
WINDOW_SECONDS = 60

TOD_BUCKETS: Dict[str, str] = {
    "early_morning": "05:00-08:00",
    "late_morning": "08:00-12:00",
    "afternoon": "12:00-17:00",
    "evening": "17:00-21:00",
    "night_before_midnight": "21:00-24:00",
    "night_after_midnight": "00:00-05:00",
}
DOW_BUCKETS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


@dataclass(frozen=True)
class HomeConfig:
    home: str
    threshold: float
    data_start: Optional[datetime] = None
    splits: tuple = ("train", "val", "test")
    data_variant: str = DATA_VARIANT
    model_suffix: str = MODEL_SUFFIX

    @property
    def checkpoint(self) -> Path:
        return PROJECT_ROOT / "trained_models" / self.home / f"{self.home}_{self.model_suffix}" / "best_model.pt"

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed" / "casas" / self.home / self.data_variant

    @property
    def vocab(self) -> Path:
        return self.data_dir / "vocab.json"

    def split_path(self, split: str) -> Path:
        return self.data_dir / f"{split}.json"

    @property
    def captions_path(self) -> Path:
        return self.data_dir / "train_captions_baseline.json"

    @property
    def floor_plan(self) -> Path:
        return FLOOR_PLAN_DIR / f"{self.home}.png"


# Thresholds: macro F0.5-optimal (precision-weighted) cosine cutoffs for label-prototype ->
# sensor retrieval on the FD_60 test split
# (results/evals/imwut_paper/retrieval_threshold_sensitivity_FD_60.md). F1 is degenerate for
# Aruba (monotonically increasing towards the lowest threshold, ~24% precision).
HOMES: Dict[str, HomeConfig] = {
    "milan": HomeConfig("milan", threshold=0.075, data_start=datetime(2009, 11, 13)),
    "aruba": HomeConfig("aruba", threshold=0.05),
    "cairo": HomeConfig("cairo", threshold=0.14),
}


@dataclass
class AnalysisSettings:
    min_observed_minutes_per_day: float = 30.0
    episode_gap_seconds: float = 120.0
    rolling_days: int = 7
    rater_n_random: int = 8
    rater_n_top: int = 4
    rater_n_negative: int = 4
    rater_context_seconds: int = 120
    seed: int = 0
    extra: dict = field(default_factory=dict)


def get_home(home: str) -> HomeConfig:
    key = home.strip().lower()
    if key not in HOMES:
        raise KeyError(f"Unknown home {home!r}; configured: {sorted(HOMES)}")
    return HOMES[key]
