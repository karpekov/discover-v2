"""Instrumentation description M of a home, as shown to the planner LLM."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List

from query.llm_rewriter import _SENSOR_TYPE_DESCRIPTIONS, _load_example_captions

from .config import METADATA_PATH, TOD_BUCKETS, HomeConfig


@lru_cache(maxsize=None)
def _metadata() -> dict:
    with open(METADATA_PATH) as f:
        return json.load(f)


def home_metadata(home: str) -> dict:
    return _metadata().get(home, {})


def room_names(home: str) -> List[str]:
    locs = home_metadata(home).get("sensor_location", {})
    return sorted({r for r in locs.values() if r and r.lower() not in ("unk", "unknown")})


def sensor_coordinates(home: str) -> Dict[str, tuple]:
    return {k: tuple(v) for k, v in home_metadata(home).get("sensor_coordinates", {}).items()}


def sensor_inventory(home: str) -> str:
    meta = home_metadata(home)
    locs, details = meta.get("sensor_location", {}), meta.get("sensor_details", {})
    by_room: Dict[str, List[str]] = {}
    for sid, room in sorted(locs.items()):
        by_room.setdefault(room, []).append(sid)
    lines = []
    for room in sorted(by_room):
        if room.lower() in ("unk", "unknown"):
            continue
        lines.append(f"  room `{room}`:")
        for sid in sorted(by_room[room]):
            prefix = "".join(c for c in sid if c.isalpha())
            kind = _SENSOR_TYPE_DESCRIPTIONS.get(prefix, "sensor")
            detail = details.get(sid)
            lines.append(f"    {sid} [{kind}]" + (f" — {detail}" if detail else ""))
    return "\n".join(lines)


def build_home_context(cfg: HomeConfig, date_start, date_end, n_examples: int = 6) -> str:
    meta = home_metadata(cfg.home)
    examples = _load_example_captions(cfg.captions_path, n=n_examples)
    tod = "\n".join(f"  {k}: {v}" for k, v in TOD_BUCKETS.items())
    ex = "\n".join(f'  - "{c}"' for c in examples) if examples else "  (none available)"
    return f"""## Household
Residents: {meta.get("num_residents", 1)}
Sensor data available: {date_start:%Y-%m-%d} to {date_end:%Y-%m-%d} ({(date_end - date_start).days + 1} days)
Data unit: ~60-second windows of ambient sensor events (motion / door). No cameras, wearables, or appliance-level power sensing.

## Sensor inventory (exact room ids in backticks)
{sensor_inventory(cfg.home)}
Only the rooms, doors and fixtures listed above are monitored; anything else (e.g. a garage,
side door, or guest bathroom that is not listed) has no sensor. Door/contact sensors report
open/close of whatever they are attached to — some are cabinets, not exterior doors — and none
of them sense locks.
{meta.get("layout_notes") or ""}

## Time-of-day buckets (hard filters available)
{tod}

## Example window captions from the embedding model's training data
These show the vocabulary and style the retrieval embedding space was aligned with.
{ex}"""
