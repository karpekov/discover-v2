#!/usr/bin/env python3
"""
Combine milan, aruba, and cairo CASAS datasets into a unified all_casas dataset
for multi-household training, keeping FD_60 and FD_60_p as separate subfolders.

Output structure:
  data/processed/casas/all_casas/
    FD_60/
      vocab.json
      train.json / val.json / test.json
      {split}_captions_baseline.json
      {split}_embeddings_baseline_clip.npz
      val_{house}.json / test_{house}.json   ← per-household eval subsets
    FD_60_p/
      (same layout)

Key transformations applied to every sample:
  - sensor_id prefixed with house name  (M001 → milan_M001)  so each
    physically distinct sensor gets its own vocab token.
  - 'household' added at top level and inside 'metadata'.
  - Room names, activities, and temporal buckets are merged as-is
    (shared semantics across houses).

The combined vocab is identical for both variants (same houses → same sensors).

Usage:
  conda activate discover-v2-env
  python scripts/combine_casas_datasets.py
  python scripts/combine_casas_datasets.py --dry-run   # print stats only
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOUSES = ["milan", "aruba", "cairo"]
VARIANTS = ["FD_60", "FD_60_p"]
SPLITS = ["train", "val", "test"]

BASE = Path("data/processed/casas")
OUTPUT_BASE = BASE / "all_casas"

# Fields where each house's tokens are semantically distinct → prefix required
HOUSE_SPECIFIC_FIELDS = {"sensor"}

# Fields shared across houses (same token = same concept) → union-merge
SHARED_FIELDS = {
    "state", "sensor_type",
    "activity", "activity_l2", "activity_full",
    "tod_bucket", "dow_bucket", "time_delta_bucket",
    "room_id",        # rooms with same name share similar semantics
    "sensor_detail",  # descriptive labels (not used in v3 training configs)
}


# ---------------------------------------------------------------------------
# Helper: sensor ID prefixing
# ---------------------------------------------------------------------------

def prefix_sensor_id(sensor_id: str, house: str) -> str:
    """Add house prefix to a sensor ID; leave UNK tokens unchanged."""
    if sensor_id == "UNK" or sensor_id.startswith("UNK_"):
        return sensor_id
    return f"{house}_{sensor_id}"


# ---------------------------------------------------------------------------
# Core: load + transform one (house, variant, split)
# ---------------------------------------------------------------------------

def load_and_transform(house: str, variant: str, split: str):
    """Load a JSON split, apply all transformations, and return (samples, id_map).

    id_map maps new_sample_id → old_sample_id for embedding re-indexing.
    Sample IDs are kept as-is (no renaming needed since variants are separate).
    """
    path = BASE / house / variant / f"{split}.json"
    with open(path) as f:
        raw = json.load(f)

    samples = raw["samples"] if "samples" in raw else raw
    id_map: dict[str, str] = {}

    for sample in samples:
        sid = sample.get("sample_id", "")
        id_map[sid] = sid  # identity mapping — no renaming needed

        # Add household at top level and inside metadata
        sample["household"] = house
        if "metadata" not in sample:
            sample["metadata"] = {}
        sample["metadata"]["household"] = house

        # Prefix all sensor IDs in the event sequence
        for event in sample.get("sensor_sequence", []):
            raw_sid = event.get("sensor_id")
            if raw_sid:
                event["sensor_id"] = prefix_sensor_id(raw_sid, house)

    return samples, id_map


# ---------------------------------------------------------------------------
# Core: build merged vocabulary (same result for FD_60 and FD_60_p)
# ---------------------------------------------------------------------------

def build_combined_vocab(house_vocabs: dict[str, dict]) -> dict:
    """Merge per-house vocabs into a single combined vocab.

    Sensor tokens are house-prefixed; all other fields are union-merged with
    contiguous integer indices (0 reserved for UNK/padding).
    """
    all_fields: set[str] = set()
    for vocab in house_vocabs.values():
        all_fields |= set(vocab.keys())

    combined: dict[str, dict[str, int]] = {}
    for field in sorted(all_fields):
        if field in HOUSE_SPECIFIC_FIELDS:
            all_tokens: set[str] = {"UNK"}
            for house, vocab in house_vocabs.items():
                for token in vocab.get(field, {}):
                    if token == "UNK" or token.startswith("UNK_"):
                        all_tokens.add(token)
                    else:
                        all_tokens.add(f"{house}_{token}")
        else:
            all_tokens = set()
            for vocab in house_vocabs.values():
                all_tokens |= set(vocab.get(field, {}).keys())

        tokens_sorted = sorted(t for t in all_tokens if t != "UNK")
        mapping: dict[str, int] = {"UNK": 0}
        for i, tok in enumerate(tokens_sorted, start=1):
            mapping[tok] = i
        combined[field] = mapping

    return combined


# ---------------------------------------------------------------------------
# Core: combine caption files
# ---------------------------------------------------------------------------

def combine_captions(house: str, variant: str, split: str) -> list:
    """Load captions for one (house, variant, split), annotate with household."""
    path = BASE / house / variant / f"{split}_captions_baseline.json"
    if not path.exists():
        print(f"  [WARN] captions not found: {path}")
        return []

    with open(path) as f:
        raw = json.load(f)

    items = raw["captions"] if "captions" in raw else raw
    for item in items:
        item["household"] = house
    return items


# ---------------------------------------------------------------------------
# Core: combine NPZ embeddings
# ---------------------------------------------------------------------------

def combine_embeddings(
    sources: list[tuple[str, str]],  # [(house, variant), ...]
    split: str,
) -> dict:
    """Stack embeddings from all houses for one (variant, split), preserving sample_ids."""
    all_embeddings: list[np.ndarray] = []
    all_sample_ids: list[str] = []
    all_caption_indices: list[np.ndarray] = []
    meta: dict = {}

    for house, variant in sources:
        path = BASE / house / variant / f"{split}_embeddings_baseline_clip.npz"
        if not path.exists():
            print(f"  [WARN] embeddings not found: {path}")
            continue

        data = np.load(path)
        embeddings = data["embeddings"]

        if "sample_ids" in data and "caption_indices" in data:
            sample_ids = data["sample_ids"].tolist()
            cap_indices = data["caption_indices"]
        else:
            n = len(embeddings)
            sample_ids = [f"{house}_{split}_{i:06d}" for i in range(n)]
            cap_indices = np.zeros(n, dtype=np.int64)

        all_embeddings.append(embeddings)
        all_sample_ids.extend(sample_ids)
        all_caption_indices.append(cap_indices)

        if not meta:
            for key in ("encoder_type", "model_name", "embedding_dim",
                        "normalize", "use_projection", "projection_dim"):
                if key in data:
                    meta[key] = data[key]

    stacked = np.concatenate(all_embeddings, axis=0)
    stacked_ids = np.array(all_sample_ids, dtype=str)
    stacked_cap = np.concatenate(all_caption_indices, axis=0)
    unique = len(set(all_sample_ids))

    result = {
        "embeddings": stacked,
        "sample_ids": stacked_ids,
        "caption_indices": stacked_cap,
        "num_unique_samples": np.array([unique], dtype=np.int64),
        "avg_captions_per_sample": np.array([len(stacked) / unique], dtype=np.float64),
    }
    result.update(meta)
    return result


# ---------------------------------------------------------------------------
# Core: process one variant (FD_60 or FD_60_p)
# ---------------------------------------------------------------------------

def process_variant(variant: str, combined_vocab: dict, dry_run: bool):
    out_dir = OUTPUT_BASE / variant
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  Variant: {variant}  →  {out_dir}")
    print(f"{'=' * 70}")

    split_samples: dict[str, list] = {s: [] for s in SPLITS}
    sources: list[tuple[str, str]] = []  # (house, variant) pairs that exist

    # --- Load all houses for this variant ---
    for house in HOUSES:
        for split in SPLITS:
            path = BASE / house / variant / f"{split}.json"
            if not path.exists():
                print(f"  [SKIP] {path}")
                continue
            samples, _ = load_and_transform(house, variant, split)
            split_samples[split].extend(samples)
            print(f"  {house}/{split}: {len(samples):>7,} samples")
        # Track which (house, variant) pairs exist (check train as proxy)
        if (BASE / house / variant / "train.json").exists():
            sources.append((house, variant))

    print()
    for split in SPLITS:
        print(f"  Combined {split}: {len(split_samples[split]):>7,} samples")

    if dry_run:
        return

    # --- Save vocab ---
    vocab_out = out_dir / "vocab.json"
    with open(vocab_out, "w") as f:
        json.dump(combined_vocab, f, indent=2)

    # --- Save sensor JSON splits ---
    for split in SPLITS:
        out_path = out_dir / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump({"samples": split_samples[split]}, f)

    # --- Per-household evaluation subsets ---
    for split in ("val", "test"):
        for house in HOUSES:
            subset = [s for s in split_samples[split] if s.get("household") == house]
            out_path = out_dir / f"{split}_{house}.json"
            with open(out_path, "w") as f:
                json.dump({"samples": subset}, f)
            print(f"  {split}_{house}.json: {len(subset):,} samples")

    # --- Combine captions ---
    print()
    for split in SPLITS:
        all_items: list = []
        for house, _ in sources:
            items = combine_captions(house, variant, split)
            all_items.extend(items)
        out_path = out_dir / f"{split}_captions_baseline.json"
        with open(out_path, "w") as f:
            json.dump({"captions": all_items}, f)
        print(f"  {split}_captions_baseline.json: {len(all_items):,} entries")

    # --- Combine embeddings ---
    print()
    for split in SPLITS:
        result = combine_embeddings(sources, split)
        out_path = out_dir / f"{split}_embeddings_baseline_clip.npz"
        np.savez(out_path, **result)
        n_emb = result["embeddings"].shape[0]
        n_samp = int(result["num_unique_samples"][0])
        print(f"  {split}_embeddings_baseline_clip.npz: {n_emb:,} embeddings / {n_samp:,} samples")

    print(f"\n  Files in {out_dir}:")
    for p in sorted(out_dir.iterdir()):
        print(f"    {p.name:<52s}  {p.stat().st_size / 1e6:>7.1f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Combine CASAS multi-house datasets")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print dataset sizes without writing files")
    args = parser.parse_args()

    # Build one shared combined vocab (same for both variants)
    print("Building combined vocabulary from all houses …")
    house_vocabs: dict[str, dict] = {}
    for house in HOUSES:
        vocab_path = BASE / house / "FD_60" / "vocab.json"
        with open(vocab_path) as f:
            house_vocabs[house] = json.load(f)

    combined_vocab = build_combined_vocab(house_vocabs)
    print(f"  sensor: {len(combined_vocab['sensor'])} tokens  "
          f"(was {sum(len(v['sensor']) for v in house_vocabs.values())} across houses before prefixing)")
    for field, mapping in combined_vocab.items():
        if field != "sensor":
            print(f"  {field}: {len(mapping)} tokens")

    if not args.dry_run:
        OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    for variant in VARIANTS:
        process_variant(variant, combined_vocab, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\n{'=' * 70}")
        print(f"Done!  Output: {OUTPUT_BASE.resolve()}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
