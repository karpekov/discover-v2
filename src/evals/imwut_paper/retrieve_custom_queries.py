"""Retrieve top-10 sensor windows for custom queries and generate house heatmaps.

Implements prototype→sensor retrieval exactly as evaluate_embeddings.py does:
  - Prototypes  = custom long_queries encoded with CLIP + trained text projection MLP
  - Targets     = sensor embeddings from the per-household test split, produced by
                  running the household's trained sensor encoder on test.json
  - Retrieval   = cosine similarity, top-10 unique samples per prototype
  - Heatmaps    = sensor activation density on floor plan (same rendering as
                  create_prototype_sensor_heatmaps in evaluate_embeddings.py)

One model per household (not all_casas):
  trained_models/milan/milan_fd60_seq_rb1_textclip_projmlp_clipmlm_v3/best_model.pt
  trained_models/aruba/aruba_fd60_seq_rb1_textclip_projmlp_clipmlm_v3/best_model.pt
  trained_models/cairo/cairo_fd60_seq_rb1_textclip_projmlp_clipmlm_v3/best_model.pt

Data used (per household):
  data/processed/casas/{home}/FD_60/test.json
  data/processed/casas/{home}/FD_60/vocab.json

Outputs
-------
  results/evals/imwut_paper/custom_query_retrieval/
    heatmaps/  milan_kitchen_activity_q1.png  ...
    report.md
    retrieved_samples.json

Usage
-----
  conda activate discover-v2-env
  python src/evals/imwut_paper/retrieve_custom_queries.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evals.eval_utils import create_text_encoder_from_checkpoint, get_global_categorical_features
from alignment.model import AlignmentModel
from dataio.dataset import SmartHomeDataset
from dataio.collate import create_data_loader

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_SUFFIX = "fd60_seq_rb1_textclip_projmlp_clipmlm_v3"
METADATA     = _ROOT / "metadata/casas_metadata.json"
QUERIES_JSON = _ROOT / "metadata/custom_retrieval_queries.json"
OUT_DIR      = _ROOT / "results/evals/imwut_paper/custom_query_retrieval"
HEATMAP_DIR  = OUT_DIR / "heatmaps"

HOUSEHOLDS = ["milan", "aruba", "cairo"]
TOP_K = 10

# Preferred captions file per household (fall back to baseline if absent)
_CAPTIONS_CANDIDATES = [
    "test_captions_rb_and_llm.json",
    "test_captions_baseline.json",
]

_TOD_SHORT = {
    "early_morning":         "early morning",
    "late_morning":          "late morning",
    "afternoon":             "afternoon",
    "evening":               "evening",
    "night_before_midnight": "night",
    "night_after_midnight":  "late night",
}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# ── Per-household paths ───────────────────────────────────────────────────────

def _checkpoint(home: str) -> Path:
    return _ROOT / f"trained_models/{home}/{home}_{MODEL_SUFFIX}/best_model.pt"

def _vocab(home: str) -> Path:
    return _ROOT / f"data/processed/casas/{home}/FD_60/vocab.json"

def _test_json(home: str) -> Path:
    return _ROOT / f"data/processed/casas/{home}/FD_60/test.json"

def _captions_json(home: str) -> Optional[Path]:
    base = _ROOT / f"data/processed/casas/{home}/FD_60"
    for name in _CAPTIONS_CANDIDATES:
        p = base / name
        if p.exists():
            return p
    return None

def load_captions_lookup(home: str) -> Dict[str, str]:
    """Return {sample_id: first_caption} for a household. Empty dict if no file."""
    path = _captions_json(home)
    if path is None:
        print(f"  [{home}] No captions file found — captions will be empty")
        return {}
    with open(path) as f:
        data = json.load(f)
    cap_list = data.get("captions", data if isinstance(data, list) else [])
    lookup: Dict[str, str] = {}
    for entry in cap_list:
        sid = entry.get("sample_id", "")
        caps = entry.get("captions", [])
        if sid and caps:
            lookup[sid] = caps[0]
    print(f"  [{home}] Loaded captions for {len(lookup)} samples from {path.name}")
    return lookup


def build_sample_summary(
    sample: dict,
    caption: str,
) -> dict:
    """Extract a compact metadata summary from one test.json sample."""
    meta = sample.get("metadata", {})
    seq  = sample.get("sensor_sequence", [])

    # Rooms visited (preserve order, deduplicate)
    rooms_visited = meta.get("rooms_visited", [])

    # Special sensors: unique sensor_ids that fired ON/OPEN/1/TRUE, grouped by room
    special: Dict[str, List[str]] = {}
    tod_counts: Counter = Counter()
    for ev in seq:
        state = str(ev.get("event_type", ev.get("state", ""))).upper()
        if state in ("ON", "OPEN", "1", "TRUE"):
            room = ev.get("room", "unknown")
            sid  = ev.get("sensor_id", ev.get("sensor", ""))
            if sid:
                special.setdefault(room, [])
                if sid not in special[room]:
                    special[room].append(sid)
        tod = ev.get("tod_bucket", "")
        if tod:
            tod_counts[tod] += 1

    gt = meta.get("ground_truth_labels", {})
    return {
        "sample_id":       sample.get("sample_id", ""),
        "caption":         caption,
        "start_time":      meta.get("start_time", ""),
        "end_time":        meta.get("end_time", ""),
        "primary_room":    meta.get("primary_room", ""),
        "rooms_visited":   rooms_visited,
        "sensors_activated_by_room": special,
        "tod":             tod_counts.most_common(1)[0][0] if tod_counts else "",
        "activity_labels": gt.get("all_labels_l2", gt.get("all_labels_l1", [])),
    }


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(home: str, device: torch.device):
    """Load text encoder + sensor encoder for one household.

    Returns (text_encoder, sensor_encoder, vocab_sizes)
    """
    ckpt_path = _checkpoint(home)
    print(f"  Loading {ckpt_path.name} …")

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Text encoder with trained projection MLP
    text_encoder = create_text_encoder_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
        data_path=str(_ROOT / f"data/processed/casas/{home}/FD_60/train.json"),
    )
    if text_encoder is None:
        raise RuntimeError(f"Could not load text encoder for {home}")
    text_encoder.to(device).eval()

    # Sensor encoder via AlignmentModel
    model = AlignmentModel.load(str(ckpt_path), device=device, vocab_path=str(_vocab(home)))
    model.eval()
    sensor_encoder = model.sensor_encoder
    vocab_sizes    = model.vocab_sizes

    return text_encoder, sensor_encoder, vocab_sizes


# ── Sensor embedding extraction ───────────────────────────────────────────────

def extract_sensor_embeddings(
    home: str,
    sensor_encoder,
    text_encoder,
    vocab_sizes: Dict[str, int],
    device: torch.device,
) -> Tuple[np.ndarray, List[str]]:
    """Run sensor encoder over test.json → L2-normalised sensor embeddings.

    Returns (sensor_embs, sample_ids)  — same as extract_embeddings_and_labels.
    """
    dataset = SmartHomeDataset(
        data_path=str(_test_json(home)),
        vocab_path=str(_vocab(home)),
        sequence_length=256,
        max_captions=1,
    )
    print(f"  [{home}] test dataset: {len(dataset)} samples")

    loader = create_data_loader(
        dataset=dataset,
        text_encoder=text_encoder,
        span_masker=None,
        vocab_sizes=vocab_sizes,
        device=device,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        apply_mlm=False,
    )

    all_embs: List[np.ndarray] = []
    all_ids:  List[str]        = []

    sensor_encoder.eval()
    with torch.no_grad():
        for batch in loader:
            input_data = {
                "categorical_features":        batch["categorical_features"],
                "coordinates":                 batch["coordinates"],
                "time_deltas":                 batch["time_deltas"],
                "global_categorical_features": get_global_categorical_features(
                    sensor_encoder, batch, dataset=dataset, device=device
                ),
            }
            emb = sensor_encoder.forward_clip(
                input_data=input_data,
                attention_mask=batch["mask"],
            )
            all_embs.append(emb.cpu().numpy())
            all_ids.extend(str(s) for s in batch.get("sample_ids", []))

    sensor_embs = np.vstack(all_embs).astype(np.float32)
    print(f"  [{home}] sensor embeddings: {sensor_embs.shape}")
    return sensor_embs, all_ids


# ── Prototype encoding ────────────────────────────────────────────────────────

def encode_query(text: str, text_encoder, device: torch.device) -> np.ndarray:
    """Encode one query string → (1, D) L2-normalised in shared embedding space.

    Mirrors encode_text_prototypes: CLIP base → normalize → text projection MLP → normalize.
    """
    with torch.no_grad():
        emb = text_encoder.encode_texts_clip([text], device)
    return emb.cpu().numpy().astype(np.float32)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_top_k(
    query_emb: np.ndarray,       # (1, D)
    sensor_embs: np.ndarray,     # (N, D)
    sample_ids: List[str],
    k: int = TOP_K,
) -> Tuple[List[str], List[float]]:
    """Cosine similarity → top-k unique (sample_ids, cosine_scores) pairs."""
    sims  = (query_emb @ sensor_embs.T).squeeze(0)   # (N,)
    order = np.argsort(sims)[::-1]
    seen: OrderedDict = OrderedDict()
    for idx in order:
        sid = sample_ids[idx]
        if sid not in seen:
            seen[sid] = float(sims[idx])
        if len(seen) >= k:
            break
    return list(seen.keys()), list(seen.values())


# ── Heatmap rendering ─────────────────────────────────────────────────────────

def render_heatmap(
    top_sample_ids: List[str],
    id_to_sample: Dict[str, dict],
    sensor_coords: Dict[str, Tuple[float, float]],
    floor_plan_img: Optional[np.ndarray],
    img_w: int,
    img_h: int,
    title: str,
    save_path: Path,
) -> dict:
    """Identical rendering logic to create_prototype_sensor_heatmaps."""
    coords:  List[Tuple[float, float]] = []
    rooms:   Counter = Counter()
    sensors: Counter = Counter()
    tods:    Counter = Counter()

    for sid in top_sample_ids:
        sample = id_to_sample.get(sid)
        if sample is None:
            continue
        for event in sample.get("sensor_sequence", []):
            sensor_id = event.get("sensor_id", event.get("sensor", ""))
            state     = str(event.get("event_type", event.get("state", ""))).upper()
            room      = event.get("room", "")
            tod       = event.get("tod_bucket", "")

            if room:
                rooms[room.replace("_", " ")] += 1
            if tod:
                tods[tod] += 1
            if state not in ("ON", "OPEN", "1", "TRUE"):
                continue

            sensors[sensor_id] += 1
            if sensor_id in sensor_coords:
                ex, ey = sensor_coords[sensor_id]
                coords.append((ex, img_h - ey))

    fig, ax = plt.subplots(figsize=(7, 6.5))
    if floor_plan_img is not None:
        ax.imshow(floor_plan_img, extent=[0, img_w, img_h, 0],
                  alpha=0.55, aspect="auto", zorder=0)
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)

    unique_coords = list(dict.fromkeys(coords))   # deduplicate, preserve order
    radius = min(img_w, img_h) * 0.04            # ~4 % of the smaller dimension

    if 1 <= len(unique_coords) <= 3:
        # Few distinct sensor locations — one semi-transparent circle per location
        for cx, cy in unique_coords:
            circle = plt.Circle((cx, cy), radius,
                                 color="crimson", alpha=0.45, zorder=2)
            ax.add_patch(circle)
        xs_u = [c[0] for c in unique_coords]
        ys_u = [c[1] for c in unique_coords]
        ax.scatter(xs_u, ys_u, c="crimson", s=40, alpha=0.9,
                   edgecolors="darkred", linewidths=0.8, zorder=3)
    elif coords:
        xs, ys = [c[0] for c in coords], [c[1] for c in coords]
        try:
            from scipy import stats as _stats
            xg, yg = np.linspace(0, img_w, 120), np.linspace(0, img_h, 120)
            X, Y   = np.meshgrid(xg, yg)
            kde    = _stats.gaussian_kde(np.vstack([xs, ys]), bw_method=0.15)
            Z      = np.reshape(kde(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)
            lvl    = np.linspace(Z.max() * 0.05, Z.max(), 12)
            ax.contourf(X, Y, Z, levels=lvl, cmap="YlOrRd", alpha=0.72, zorder=1)
        except Exception:
            pass
        ax.scatter(xs, ys, c="darkred", s=8, alpha=0.65, edgecolors="none", zorder=2)

    for sid, (sx, sy) in sensor_coords.items():
        ax.annotate(sid, (sx, img_h - sy), fontsize=3.5, alpha=0.6,
                    ha="center", va="bottom", color="navy")

    tod_line = "—"
    if tods:
        total = sum(tods.values())
        cumul, parts = 0.0, []
        for tname, cnt in tods.most_common(3):
            share = cnt / total; cumul += share
            parts.append(f"{_TOD_SHORT.get(tname, tname)} ({share:.0%})")
            if cumul >= 0.70:
                break
        tod_line = ", ".join(parts)

    ax.set_title(
        f"{title}\n{tod_line}\n(top-{len(top_sample_ids)}, {len(coords)} ON events)",
        fontsize=8, fontweight="bold", pad=4,
    )
    ax.set_xticks([]); ax.set_yticks([])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"      → {save_path.name}")

    return {
        "top_rooms":   rooms.most_common(3),
        "top_sensors": sensors.most_common(5),
        "top_tods":    tods.most_common(3),
        "n_events":    len(coords),
    }


# ── Markdown report ───────────────────────────────────────────────────────────

def _fmt(items: list) -> str:
    return "; ".join(f"{n} ({c})" for n, c in items) if items else "—"

def _fmt_tods(items: list) -> str:
    return "; ".join(f"{_TOD_SHORT.get(n, n)} ({c})" for n, c in items) if items else "—"


def build_report(records: List[dict]) -> str:
    lines = [
        "# Custom Query Retrieval — Summary",
        "",
        "**Retrieval:** prototype→sensor (CLIP + trained text projection MLP → sensor encoder embeddings), top-10  ",
        "**Models:** per-household `*_fd60_seq_rb1_textclip_projmlp_clipmlm_v3`  ",
        "",
        "## Results table",
        "",
        "| Household | Short Query | # | Long Query | Heatmap | Top Rooms | Top Sensors | Time-of-Day |",
        "|-----------|-------------|---|------------|---------|-----------|-------------|-------------|",
    ]
    for r in records:
        s   = r["stats"]
        img = f"![{r['file']}](heatmaps/{r['file']})"
        lq  = r["long_query"].replace("|", "\\|")
        lines.append(
            f"| {r['home'].capitalize()} | {r['short_query']} | {r['q_idx']} "
            f"| {lq} | {img} "
            f"| {_fmt(s['top_rooms'])} | {_fmt(s['top_sensors'])} | {_fmt_tods(s['top_tods'])} |"
        )

    lines += ["", "## Per-household findings", ""]
    for home in list(dict.fromkeys(r["home"] for r in records)):
        lines.append(f"### {home.capitalize()}")
        lines.append("")
        for sq in list(dict.fromkeys(r["short_query"] for r in records if r["home"] == home)):
            sq_recs = [r for r in records if r["home"] == home and r["short_query"] == sq]
            lines.append(f"**{sq}**")
            for r in sq_recs:
                s = r["stats"]
                rooms   = ", ".join(n for n, _ in s["top_rooms"])   or "—"
                sensors = ", ".join(n for n, _ in s["top_sensors"]) or "—"
                tods    = ", ".join(_TOD_SHORT.get(n, n) for n, _ in s["top_tods"]) or "—"
                lq_prev = r["long_query"][:90] + ("…" if len(r["long_query"]) > 90 else "")
                lines.append(
                    f"- **{r['q_idx']}** `{lq_prev}`  \n"
                    f"  Rooms: {rooms} | Sensors: {sensors} | TOD: {tods}"
                )
            lines.append("")

    return "\n".join(lines) + "\n"


# ── CSV export ────────────────────────────────────────────────────────────────

_CSV_COLUMNS = [
    "house",
    "custom_query",
    "long_query_idx",
    "long_query",
    "rank",
    "cosine_similarity",
    "sample_id",
    "tod",
    "activity_labels",
    "primary_room",
    "caption",
]


def write_retrieved_csv(retrieved_json: dict, out_path: Path) -> None:
    """Flatten retrieved_json into a tidy CSV with one row per retrieved sample."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for house, queries in retrieved_json.items():
            for short_query, variants in queries.items():
                for q_idx, payload in variants.items():
                    long_query = payload.get("long_query", "")
                    for s in payload.get("retrieved_samples", []):
                        writer.writerow({
                            "house":            house,
                            "custom_query":     short_query,
                            "long_query_idx":   q_idx,
                            "long_query":       long_query,
                            "rank":             s.get("rank", ""),
                            "cosine_similarity": s.get("cosine_similarity", ""),
                            "sample_id":        s.get("sample_id", ""),
                            "tod":              _TOD_SHORT.get(s.get("tod", ""), s.get("tod", "")),
                            "activity_labels":  "; ".join(s.get("activity_labels", [])),
                            "primary_room":     s.get("primary_room", ""),
                            "caption":          s.get("caption", ""),
                        })


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    with open(METADATA)     as f: all_meta       = json.load(f)
    with open(QUERIES_JSON) as f: custom_queries = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    retrieved_json: dict = {}   # home → short_query → q_idx → list of sample summaries

    for home in HOUSEHOLDS:
        if home not in custom_queries:
            print(f"Skipping {home}: not in custom_retrieval_queries.json\n")
            continue

        print(f"\n{'='*60}")
        print(f"  {home.upper()}")
        print(f"{'='*60}")

        # ── Load per-household model ──────────────────────────────────────────
        text_encoder, sensor_encoder, vocab_sizes = load_models(home, device)

        # ── Sensor embeddings from test.json via sensor encoder ───────────────
        sensor_embs, sample_ids = extract_sensor_embeddings(
            home, sensor_encoder, text_encoder, vocab_sizes, device
        )

        # ── Raw sample lookup for heatmap event data ──────────────────────────
        with open(_test_json(home)) as f:
            raw = json.load(f)
        samples = raw.get("samples", raw if isinstance(raw, list) else [])
        id_to_sample = {s.get("sample_id", ""): s for s in samples if s.get("sample_id")}

        # ── Captions lookup ───────────────────────────────────────────────────
        captions_lookup = load_captions_lookup(home)

        # ── Floor plan + sensor coordinates (unprefixed IDs match test.json) ──
        dataset_meta  = all_meta.get(home, {})
        sensor_coords = dataset_meta.get("sensor_coordinates", {})

        floor_plan_path = _ROOT / "metadata/floor_plans_augmented" / f"{home}.png"
        floor_plan_img  = None
        img_w, img_h    = 2600, 2800
        if floor_plan_path.exists():
            try:
                floor_plan_img = plt.imread(str(floor_plan_path))
                img_h, img_w   = floor_plan_img.shape[:2]
            except Exception as e:
                print(f"  ⚠ floor plan load failed: {e}")

        retrieved_json[home] = {}

        # ── Per-query retrieval + heatmap ─────────────────────────────────────
        queries = custom_queries[home].get("queries", [])
        for entry in queries:
            short_query = entry.get("short_query", "")
            long_list   = entry.get("long_queries", [])

            if not long_list:
                print(f"\n  [{short_query}] no long queries — skipped")
                continue

            slug = slugify(short_query)
            print(f"\n  [{short_query}]  ({len(long_list)} variants)")
            retrieved_json[home][short_query] = {}

            for q_idx, long_query in enumerate(long_list, start=1):
                preview = long_query[:70] + ("…" if len(long_query) > 70 else "")
                print(f"    q{q_idx}: {preview}")

                query_emb  = encode_query(long_query, text_encoder, device)   # (1, D)
                top_ids, top_sims = retrieve_top_k(query_emb, sensor_embs, sample_ids, TOP_K)

                fname     = f"{home}_{slug}_q{q_idx}.png"
                save_path = HEATMAP_DIR / fname

                stats = render_heatmap(
                    top_sample_ids = top_ids,
                    id_to_sample   = id_to_sample,
                    sensor_coords  = sensor_coords,
                    floor_plan_img = floor_plan_img,
                    img_w=img_w, img_h=img_h,
                    title=f"{home.capitalize()} — {short_query} (q{q_idx})",
                    save_path=save_path,
                )

                # Build per-sample summaries for the JSON output
                sample_summaries = []
                for rank, (sid, sim) in enumerate(zip(top_ids, top_sims), start=1):
                    sample = id_to_sample.get(sid)
                    if sample is None:
                        continue
                    caption = captions_lookup.get(sid, "")
                    summary = build_sample_summary(sample, caption)
                    summary["rank"] = rank
                    summary["cosine_similarity"] = round(sim, 4)
                    sample_summaries.append(summary)

                key = f"q{q_idx}"
                retrieved_json[home][short_query][key] = {
                    "long_query":        long_query,
                    "retrieved_samples": sample_summaries,
                }

                records.append({
                    "home":        home,
                    "short_query": short_query,
                    "q_idx":       f"q{q_idx}",
                    "long_query":  long_query,
                    "file":        fname,
                    "stats":       stats,
                })

    report_path = OUT_DIR / "report.md"
    report_path.write_text(build_report(records), encoding="utf-8")
    print(f"\nReport: {report_path}")

    samples_path = OUT_DIR / "retrieved_samples.json"
    samples_path.write_text(json.dumps(retrieved_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Retrieved samples: {samples_path}")

    csv_path = OUT_DIR / "retrieved_samples.csv"
    write_retrieved_csv(retrieved_json, csv_path)
    print(f"Retrieved samples CSV: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
