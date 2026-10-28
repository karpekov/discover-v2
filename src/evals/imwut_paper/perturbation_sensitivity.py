"""Perturbation Sensitivity Analysis for CASAS household activity retrieval.

For each activity label in milan, aruba, and cairo (at both L1 and L2 levels):
  1. Takes the label's `long_desc` (first sentence) as the original query.
  2. Calls Gemini to produce 10 paraphrases of that query.
  3. Runs retrieval against the per-household test split (FD_60_p) for the
     original and each paraphrase using TWO modes:
       a. Top-{TOP_K}: fixed top-100 cosine-similarity neighbours.
       b. Threshold: all samples with cosine sim ≥ per-household threshold
          (milan: 0.075, aruba: 0.065, cairo: 0.13).
  4. Computes four stability metrics per (household, label, level, mode):
       - Jaccard Similarity   : |orig ∩ para| / |orig ∪ para| over retrieved sets
       - Kendall Tau          : rank correlation of ordered ID lists
       - Query Similarity     : cosine similarity between text embeddings
       - Performance Stability: std-dev of Precision / mAP across 11 queries
  5. Reports results per class (L1 and L2 separately) and per household,
     with clearly separated sections for each retrieval mode.

Label normalisation (mirrors retrieval_metrics_agg.py):
  Raw label variants are mapped to canonical display names via LABEL_NORM.
  When two raw labels map to the same canonical name their long_descs are
  concatenated and a single merged paraphrase set is generated.
  Noisy / transitional labels are dropped via _SKIP_LABELS.
  Cairo resident-specific labels (R1 sleep, R2 sleep, R1 work …) are renamed.

Models used (per household, NOT a single all-casas model):
  trained_models/{home}/{home}_fd60_seq_rb1_textclip_projmlp_clipmlm_v3/best_model.pt

Data used (FD_60_p, not FD_60):
  data/processed/casas/{home}/FD_60_p/test.json
  data/processed/casas/{home}/FD_60_p/vocab.json

Outputs:
  results/evals/imwut_paper/perturbation_sensitivity/
    paraphrases.json   -- cached LLM paraphrases; reused on re-runs
    results.json       -- full per-class (L1+L2) metric results
    summary.md         -- human-readable report (per household, per level)

Usage:
  conda activate discover-v2-env
  python src/evals/imwut_paper/perturbation_sensitivity.py
  python src/evals/imwut_paper/perturbation_sensitivity.py --force_recompute
  python src/evals/imwut_paper/perturbation_sensitivity.py --households milan aruba
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import kendalltau

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evals.eval_utils import create_text_encoder_from_checkpoint, get_global_categorical_features
from alignment.model import AlignmentModel
from dataio.dataset import SmartHomeDataset
from dataio.collate import create_data_loader

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_SUFFIX  = "fd60_seq_rb1_textclip_projmlp_clipmlm_v3"
METADATA      = _ROOT / "metadata/casas_metadata.json"
OUT_DIR       = _ROOT / "results/evals/imwut_paper/perturbation_sensitivity"

HOUSEHOLDS    = ["milan", "aruba", "cairo"]
TOP_K         = 100
N_PARAPHRASES = 10

# Cosine similarity thresholds for threshold-based retrieval
# (mirrors llm_reasoning.py RETRIEVAL_THRESHOLDS)
RETRIEVAL_THRESHOLDS: Dict[str, float] = {
    "milan": 0.075,
    "aruba": 0.065,
    "cairo": 0.13,
}

# Label levels to analyse
LEVELS: List[str] = ["L1", "L2"]


# ── Per-household minor label filtering (mirrors evaluate_embeddings.py) ──────
# Canonical labels to exclude per household — these are noisy, rare, or
# transitional activity classes that inflate/deflate metrics.
# Values must match canonical names produced by LABEL_NORM (after normalisation).
# Source: evaluate_embeddings.py _MINOR_LABELS_BY_DATASET.
_MINOR_LABELS_BY_HOUSEHOLD: Dict[str, set] = {
    "milan": {"Meditate", "Take Medicine", "Bedroom Activity"},
    "aruba": set(),   # respirate not in our labels; wash_dishes merged to Eating: General
    "cairo": {"Bed to Toilet", "Take Medicine"},
}


# ── Label normalisation (mirrors retrieval_metrics_agg.py, extended) ──────────

# Raw label name (lowercase, stripped) → canonical display name.
# When two raw labels from the same household/level map to the same canonical
# name, only the first long_desc found is used for paraphrasing (they describe
# the same concept).
LABEL_NORM: Dict[str, str] = {
    # Sleep / rest
    "sleep":                 "Sleep",
    "sleeping":              "Sleep",
    "r1 sleep":              "Sleep",
    "r2 sleep":              "Sleep",
    # Bed-to-toilet
    "bed_to_toilet":         "Bed to Toilet",
    "bed to toilet":         "Bed to Toilet",
    # Leave home
    "leave home":            "Leave Home",
    "leave_home":            "Leave Home",
    # Chores / housekeeping / laundry
    "chores":                "Chores",
    "laundry":               "Chores",
    "housekeeping":          "Chores",
    # Eating — general / meal-specific
    "eating":                "Eating: General",
    "eat":                   "Eating: General",
    "dining_rm_activity":    "Eating: General",
    "meal_preparation":      "Eating: General",
    "cook":                  "Eating: General",
    "wash_dishes":           "Eating: General",
    "breakfast":             "Eating: Breakfast",
    "lunch":                 "Eating: Lunch",
    "dinner":                "Eating: Dinner",
    # Work / desk
    "work":                  "Work",
    "desk_activity":         "Work",
    "r1 work in office":     "Work",
    # Relax variants
    "relax":                 "Relax: General",
    "watch_tv":              "Relax: Watch TV",
    "read":                  "Relax: Read",
    # Bathroom (milan)
    "master_bathroom":       "Bathroom",
    "guest_bathroom":        "Bathroom",
    "bathing":               "Bathroom",
    # Milan fine-grained medication
    "morning_meds":          "Take Medicine",
    "eve_meds":              "Take Medicine",
    "take_medicine":         "Take Medicine",
    # Cairo take medicine (resident-specific)
    "r2 take medicine":      "Take Medicine",
    # Milan bedroom activity
    "master_bedroom_activity": "Bedroom Activity",
    # Meditate (milan)
    "meditate":              "Meditate",
    # Cairo night wandering
    "night wandering":       "Night Wandering",
    # Kitchen activity (milan L1)
    "kitchen_activity":      "Kitchen Activity",
}

# Canonical sort order — unrecognised labels sort after these.
CANONICAL_ORDER: List[str] = [
    "Sleep", "Bed to Toilet", "Leave Home", "Chores",
    "Kitchen Activity",
    "Eating: General", "Eating: Breakfast", "Eating: Lunch", "Eating: Dinner",
    "Work",
    "Relax: General", "Relax: Watch TV", "Relax: Read",
    "Bathroom", "Bedroom Activity",
    "Take Medicine", "Meditate", "Night Wandering",
]
_CANONICAL_RANK: Dict[str, int] = {lbl: i for i, lbl in enumerate(CANONICAL_ORDER)}

# Labels to drop — noise, transitional, or meta labels.
_SKIP_LABELS: set = {
    # No-activity / meta
    "no_activity", "No_Activity", "no_sensor_readings",
    # Too vague
    "Other", "other",
    # Enter home (too symmetric with leave home; not in LABEL_NORM)
    "Enter_Home", "enter_home",
    # Wake-up transitions — short, boundary events, not full activities
    "R1 wake", "R2 wake", "R1_wake", "R2_wake",
    "r1 wake", "r2 wake",
}


def normalize_label(raw: str) -> str:
    return LABEL_NORM.get(raw.lower().strip(), raw)


def canonical_sort_key(label: str) -> Tuple[int, str]:
    return (_CANONICAL_RANK.get(label, len(CANONICAL_ORDER)), label)


# ── Per-household paths (FD_60_p) ─────────────────────────────────────────────

def _checkpoint(home: str) -> Path:
    return _ROOT / f"trained_models/{home}/{home}_{MODEL_SUFFIX}/best_model.pt"

def _vocab(home: str) -> Path:
    return _ROOT / f"data/processed/casas/{home}/FD_60_p/vocab.json"

def _test_json(home: str) -> Path:
    return _ROOT / f"data/processed/casas/{home}/FD_60_p/test.json"

def _train_json(home: str) -> Path:
    return _ROOT / f"data/processed/casas/{home}/FD_60_p/train.json"


# ── Label discovery ───────────────────────────────────────────────────────────

def get_test_label_sets(home: str) -> Dict[str, set]:
    """Return {'L1': set_of_raw_labels, 'L2': set_of_raw_labels} from test.json."""
    with open(_test_json(home)) as f:
        raw = json.load(f)
    samples = raw.get("samples", raw if isinstance(raw, list) else [])
    l1, l2 = set(), set()
    for s in samples:
        gt = s.get("metadata", {}).get("ground_truth_labels", {})
        l1.update(gt.get("all_labels_l1", []))
        l2.update(gt.get("all_labels_l2", []))
    return {"L1": l1, "L2": l2}


def get_canonical_labels_for_level(
    home: str,
    level: str,
    all_meta: dict,
) -> Dict[str, str]:
    """Return {canonical_label: long_desc} for one household at one label level.

    Steps:
    1. Get the set of raw labels that appear in test data at `level`.
    2. Filter out _SKIP_LABELS (global) and _MINOR_LABELS_BY_HOUSEHOLD (per-household).
    3. Map to canonical names via LABEL_NORM.
    4. Look up long_desc from casas_metadata label_to_text.
    5. When multiple raw labels share a canonical name, use the first long_desc
       found (they describe the same concept).
    """
    label_sets    = get_test_label_sets(home)
    raw_labels    = label_sets[level]
    label_to_text = all_meta.get(home, {}).get("label_to_text", {})
    minor_skip    = _MINOR_LABELS_BY_HOUSEHOLD.get(home, set())

    canon_to_desc: Dict[str, str] = {}

    for raw in sorted(raw_labels):
        if raw in _SKIP_LABELS:
            continue

        canon = normalize_label(raw)

        # Drop minor/noisy labels (checked after normalisation)
        if canon in minor_skip:
            continue

        # Already filled from another raw label that maps here — skip
        if canon in canon_to_desc:
            continue

        # Look up long_desc in metadata (try exact, then case-insensitive)
        info = label_to_text.get(raw) or label_to_text.get(raw.lower())
        if info is None:
            info = next(
                (v for k, v in label_to_text.items() if k.lower() == raw.lower()),
                None,
            )
        if info is None:
            continue

        long_desc = info.get("long_desc", [])
        if isinstance(long_desc, list):
            long_desc = long_desc[0] if long_desc else ""
        if long_desc:
            canon_to_desc[canon] = long_desc

    # Sort by canonical order
    return dict(
        sorted(canon_to_desc.items(), key=lambda kv: canonical_sort_key(kv[0]))
    )


# ── Gemini paraphrase generation ──────────────────────────────────────────────

def _build_paraphrase_prompt(original: str, label: str, n: int) -> str:
    return (
        f"You are a smart home activity recognition expert.\n"
        f"The following sentence describes sensor-based evidence for the activity '{label}':\n\n"
        f'  "{original}"\n\n'
        f"Generate exactly {n} paraphrases of this sentence. "
        f"Each paraphrase should convey the same sensor evidence and activity meaning "
        f"but use different wording, sentence structure, or vocabulary. "
        f"Do NOT change the factual content (rooms, sensors, time-of-day). "
        f"Output ONLY a JSON array of {n} strings, no extra commentary.\n"
        f"Example format: [\"paraphrase 1\", \"paraphrase 2\", ...]"
    )


def generate_paraphrases_gemini(
    original: str,
    label: str,
    n: int = N_PARAPHRASES,
    max_retries: int = 3,
) -> List[str]:
    """Call Gemini to generate n paraphrases of `original`."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai not installed. Run: pip install google-generativeai"
        )

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = _build_paraphrase_prompt(original, label, n)

    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            paraphrases = json.loads(text.strip())
            if isinstance(paraphrases, list) and len(paraphrases) >= 1:
                return paraphrases[:n]
        except Exception as exc:
            print(f"      [Gemini] attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    print(f"      [Gemini] all retries failed for '{label}' — returning empty list")
    return []


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(home: str, device: torch.device):
    """Load text encoder + sensor encoder for one household."""
    ckpt_path = _checkpoint(home)
    print(f"  Loading {ckpt_path.name} …")

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    text_encoder = create_text_encoder_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
        data_path=str(_train_json(home)),
    )
    if text_encoder is None:
        raise RuntimeError(f"Could not load text encoder for {home}")
    text_encoder.to(device).eval()

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
) -> Tuple[np.ndarray, List[str], Dict[str, dict]]:
    """Run sensor encoder over FD_60_p test.json.

    Returns (sensor_embs, sample_ids, id_to_sample).
    """
    dataset = SmartHomeDataset(
        data_path=str(_test_json(home)),
        vocab_path=str(_vocab(home)),
        sequence_length=256,
        max_captions=1,
    )
    print(f"  [{home}] FD_60_p test dataset: {len(dataset)} samples")

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

    with open(_test_json(home)) as f:
        raw = json.load(f)
    samples = raw.get("samples", raw if isinstance(raw, list) else [])
    id_to_sample = {s.get("sample_id", ""): s for s in samples if s.get("sample_id")}

    return sensor_embs, all_ids, id_to_sample


# ── Text encoding ─────────────────────────────────────────────────────────────

def encode_query(text: str, text_encoder, device: torch.device) -> np.ndarray:
    """Encode one query string → (1, D) L2-normalised embedding."""
    with torch.no_grad():
        emb = text_encoder.encode_texts_clip([text], device)
    return emb.cpu().numpy().astype(np.float32)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_ranked(
    query_emb: np.ndarray,
    sensor_embs: np.ndarray,
    sample_ids: List[str],
) -> Tuple[List[str], List[float]]:
    """Full-corpus cosine similarity ranking — returns ALL samples, deduplicated.

    Used for mAP computation (must rank the full corpus, not just top-k).
    """
    sims  = (query_emb @ sensor_embs.T).squeeze(0)
    order = np.argsort(sims)[::-1]
    seen: OrderedDict = OrderedDict()
    for idx in order:
        sid = sample_ids[idx]
        if sid not in seen:
            seen[sid] = float(sims[idx])
    return list(seen.keys()), list(seen.values())


def retrieve_top_k(
    query_emb: np.ndarray,
    sensor_embs: np.ndarray,
    sample_ids: List[str],
    k: int = TOP_K,
) -> Tuple[List[str], List[float]]:
    """Cosine similarity → ordered top-k unique (sample_ids, scores).

    Used for Jaccard / Kendall Tau stability metrics in top-k mode.
    """
    all_ids, all_sims = retrieve_ranked(query_emb, sensor_embs, sample_ids)
    return all_ids[:k], all_sims[:k]


def retrieve_within_threshold(
    query_emb: np.ndarray,
    sensor_embs: np.ndarray,
    sample_ids: List[str],
    threshold: float,
) -> Tuple[List[str], List[float]]:
    """Return all samples with cosine similarity ≥ threshold, ordered desc.

    Used for threshold-based stability metrics.
    """
    all_ids, all_sims = retrieve_ranked(query_emb, sensor_embs, sample_ids)
    # all_sims is already sorted descending; find cutoff
    filtered_ids  = [sid for sid, s in zip(all_ids, all_sims) if s >= threshold]
    filtered_sims = [s   for s          in all_sims              if s >= threshold]
    return filtered_ids, filtered_sims


# ── Ground-truth relevance ────────────────────────────────────────────────────

def build_relevance_map(
    sample_ids: List[str],
    id_to_sample: Dict[str, dict],
    canonical_label: str,
    level: str,
) -> Dict[str, bool]:
    """Return {sample_id: is_relevant} for the given canonical label and level.

    Relevance: the canonical name of at least one ground-truth label (at the
    given level) matches `canonical_label`.
    """
    label_key = f"all_labels_{level.lower()}"
    rel: Dict[str, bool] = {}
    for sid in sample_ids:
        sample = id_to_sample.get(sid)
        if sample is None:
            rel[sid] = False
            continue
        gt     = sample.get("metadata", {}).get("ground_truth_labels", {})
        labels = gt.get(label_key, [])
        rel[sid] = any(normalize_label(lbl) == canonical_label for lbl in labels)
    return rel


# ── Metric computation ────────────────────────────────────────────────────────

def jaccard(set_a: set, set_b: set) -> float:
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 1.0


def kendall_tau_score(ids_a: List[str], ids_b: List[str]) -> float:
    """Kendall Tau between two ordered ID lists (union as universe)."""
    universe = list(dict.fromkeys(ids_a + ids_b))
    rank_a = {sid: i for i, sid in enumerate(ids_a)}
    rank_b = {sid: i for i, sid in enumerate(ids_b)}
    vec_a = [rank_a.get(sid, len(ids_a)) for sid in universe]
    vec_b = [rank_b.get(sid, len(ids_b)) for sid in universe]
    tau, _ = kendalltau(vec_a, vec_b)
    return float(tau) if not np.isnan(tau) else 0.0


def query_cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Cosine similarity between two (1, D) L2-normalised embeddings."""
    return float((emb_a @ emb_b.T).squeeze())


def precision_at_k(
    top_k_ids: List[str],
    relevance_map: Dict[str, bool],
    k: int,
) -> float:
    """P@k — fraction of the top-k retrieved items that are relevant."""
    top  = top_k_ids[:k]
    hits = sum(1 for sid in top if relevance_map.get(sid, False))
    return hits / k if k > 0 else 0.0


def precision_of_set(
    retrieved_ids: List[str],
    relevance_map: Dict[str, bool],
) -> float:
    """Precision of an arbitrary retrieved set (used for threshold mode)."""
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for sid in retrieved_ids if relevance_map.get(sid, False))
    return hits / len(retrieved_ids)


def average_precision_full_corpus(
    all_ranked_ids: List[str],
    relevance_map: Dict[str, bool],
) -> float:
    """Full-corpus Average Precision matching evaluate_embeddings.py exactly.

    AP = mean( precision at rank r_j ) for each of the n_rel relevant items,
    where precision at r_j = j / r_j  (j = how many relevant found so far).

    Requires ALL test samples to be ranked (no top-k cutoff).
    """
    hits, prec_at_rel = 0, []
    for rank, sid in enumerate(all_ranked_ids, start=1):
        if relevance_map.get(sid, False):
            hits += 1
            prec_at_rel.append(hits / rank)
    if not prec_at_rel:
        return 0.0
    return float(np.mean(prec_at_rel))


def compute_metrics(
    original_all_ids: List[str],        # full corpus ranking (for mAP)
    original_sel_ids: List[str],        # selection set: top-k OR threshold-retrieved
    original_emb: np.ndarray,
    paraphrase_all_ids_list: List[List[str]],
    paraphrase_sel_ids_list: List[List[str]],
    paraphrase_embs: List[np.ndarray],
    relevance_map: Dict[str, bool],
    mode: str = "topk",                 # "topk" or "threshold"
    k: int = TOP_K,
) -> dict:
    """Return stability metrics comparing original vs. paraphrases.

    Stability metrics (Jaccard, Kendall Tau) operate on the selection set
    (top-k or threshold-retrieved). Performance metrics (mAP) always use the
    full corpus ranking so AP values match evaluate_embeddings.py.

    For top-k mode   : precision metric = P@k (fixed denominator k).
    For threshold mode: precision metric = precision of the retrieved set
                        (variable denominator = |retrieved|).
    """
    orig_set  = set(original_sel_ids)
    orig_map  = average_precision_full_corpus(original_all_ids, relevance_map)

    if mode == "topk":
        orig_prec = precision_at_k(original_sel_ids, relevance_map, k)
    else:
        orig_prec = precision_of_set(original_sel_ids, relevance_map)

    all_maps  = [orig_map]
    all_precs = [orig_prec]
    all_n_ret = [len(original_sel_ids)]
    per_para  = []
    jaccards, taus, q_sims = [], [], []

    for para_all_ids, para_sel_ids, para_emb in zip(
        paraphrase_all_ids_list, paraphrase_sel_ids_list, paraphrase_embs
    ):
        para_set  = set(para_sel_ids)
        jac       = jaccard(orig_set, para_set)
        tau       = kendall_tau_score(original_sel_ids, para_sel_ids)
        q_sim     = query_cosine_similarity(original_emb, para_emb)
        para_map  = average_precision_full_corpus(para_all_ids, relevance_map)

        if mode == "topk":
            para_prec = precision_at_k(para_sel_ids, relevance_map, k)
        else:
            para_prec = precision_of_set(para_sel_ids, relevance_map)

        jaccards.append(jac)
        taus.append(tau)
        q_sims.append(q_sim)
        all_maps.append(para_map)
        all_precs.append(para_prec)
        all_n_ret.append(len(para_sel_ids))

        per_para.append({
            "jaccard":     round(jac, 4),
            "kendall_tau": round(tau, 4),
            "query_sim":   round(q_sim, 4),
            "map":         round(para_map, 4),
            "precision":   round(para_prec, 4),
            "n_retrieved": len(para_sel_ids),
        })

    def _m(lst): return round(float(np.mean(lst)), 4) if lst else None
    def _s(lst): return round(float(np.std(lst)),  4) if lst else None

    # para-only lists (index 0 is the original; indices 1+ are paraphrases)
    para_maps  = all_maps[1:]
    para_precs = all_precs[1:]

    return {
        "mode":                     mode,
        "per_paraphrase":           per_para,
        "mean_jaccard":             _m(jaccards),
        "std_jaccard":              _s(jaccards),
        "mean_kendall_tau":         _m(taus),
        "std_kendall_tau":          _s(taus),
        "mean_query_sim":           _m(q_sims),
        "std_query_sim":            _s(q_sims),
        "original_map":             round(orig_map,  4),
        "mean_map_para":            _m(para_maps),   # mean mAP of paraphrases
        "perf_stability_map_std":   round(float(np.std(all_maps)),  4),  # std over all 11
        "original_precision":       round(orig_prec, 4),
        "mean_prec_para":           _m(para_precs),  # mean precision of paraphrases
        "perf_stability_prec_std":  round(float(np.std(all_precs)), 4),  # std over all 11
        "orig_n_retrieved":         len(original_sel_ids),
        "mean_n_retrieved":         _m(all_n_ret[1:]),  # paraphrases only
    }


# ── Paraphrase caching ────────────────────────────────────────────────────────

def load_cached_paraphrases(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_paraphrases(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Paraphrases saved → {path}")


# ── Markdown report ───────────────────────────────────────────────────────────

MISSING = "--"

def _fmt(v) -> str:
    return f"{v:.4f}" if v is not None else MISSING


def _mean_field(all_metrics: List[dict], field: str) -> Optional[float]:
    vals = [m[field] for m in all_metrics if m.get(field) is not None]
    return float(np.mean(vals)) if vals else None


def build_report(results: dict) -> str:
    """Build a full Markdown report from the results dict.

    Structure of results:
        results[home][level][canon_label] = {
            original, paraphrases, n_relevant,
            metrics_topk, metrics_threshold
        }
    """
    MODES = [
        ("metrics_topk",      f"Top-{TOP_K} Retrieval"),
        ("metrics_threshold", "Threshold Retrieval (sim ≥ threshold)"),
    ]

    # ── Cross-household summary (metrics × households) ────────────────────────
    def _cross_summary_table(level: str, mkey: str, mode_label: str) -> List[str]:
        """Return Markdown lines for one level × mode cross-household summary.

        Rows = metrics, Columns = households.
        """
        homes_present = [h for h in HOUSEHOLDS if results.get(h, {}).get(level)]
        if not homes_present:
            return []

        col_headers = " | ".join(h.capitalize() for h in homes_present)
        sep         = " | ".join(["---"] * len(homes_present))

        if mkey == "metrics_topk":
            prec_label = f"P@{TOP_K}"
            metric_rows = [
                ("Orig mAP",              "original_map"),
                ("Mean mAP (para)",       "mean_map_para"),
                ("mAP std",               "perf_stability_map_std"),
                (f"Orig {prec_label}",    "original_precision"),
                (f"Mean {prec_label} (para)", "mean_prec_para"),
                (f"{prec_label} std",     "perf_stability_prec_std"),
                ("Avg Jaccard",           "mean_jaccard"),
                ("Avg Kendall τ",         "mean_kendall_tau"),
                ("Avg Query Sim",         "mean_query_sim"),
            ]
        else:
            metric_rows = [
                ("Orig mAP",              "original_map"),
                ("Mean mAP (para)",       "mean_map_para"),
                ("mAP std",               "perf_stability_map_std"),
                ("Orig Precision",        "original_precision"),
                ("Mean Precision (para)", "mean_prec_para"),
                ("Precision std",         "perf_stability_prec_std"),
                ("Orig n retrieved",      "orig_n_retrieved"),
                ("Mean n retrieved",      "mean_n_retrieved"),
                ("Avg Jaccard",           "mean_jaccard"),
                ("Avg Kendall τ",         "mean_kendall_tau"),
                ("Avg Query Sim",         "mean_query_sim"),
            ]

        tbl = [
            f"| Metric | {col_headers} |",
            f"| --- | {sep} |",
        ]
        for label, field in metric_rows:
            cells = []
            for home in homes_present:
                all_m = [
                    e[mkey] for e in results[home][level].values()
                    if mkey in e
                ]
                cells.append(_fmt(_mean_field(all_m, field)))
            tbl.append(f"| {label} | {' | '.join(cells)} |")
        return tbl

    # ── Per-level detail table for one household × mode ───────────────────────
    def _detail_table(home: str, level: str, mkey: str) -> List[str]:
        level_data = results[home].get(level, {})
        if not level_data:
            return []

        is_thresh = (mkey == "metrics_threshold")
        prec_hdr  = "Prec" if is_thresh else f"P@{TOP_K}"

        if is_thresh:
            header = (
                f"| Label | n_rel | n_retr | "
                f"Orig mAP | Mean mAP | mAP std | "
                f"Orig {prec_hdr} | Mean {prec_hdr} | {prec_hdr} std | "
                f"Jaccard μ±σ | Kendall τ μ±σ | Query Sim μ±σ |"
            )
            sep_row = (
                "|-------|-------|--------|"
                "---------|----------|---------|"
                f"----------|------------|-----------|"
                "------------|---------------|---------------|"
            )
        else:
            header = (
                f"| Label | n_rel | "
                f"Orig mAP | Mean mAP | mAP std | "
                f"Orig {prec_hdr} | Mean {prec_hdr} | {prec_hdr} std | "
                f"Jaccard μ±σ | Kendall τ μ±σ | Query Sim μ±σ |"
            )
            sep_row = (
                "|-------|-------|"
                "---------|----------|---------|"
                f"------------|------------|-----------|"
                "------------|---------------|---------------|"
            )

        rows = [header, sep_row]
        all_m = []
        for canon, entry in level_data.items():
            if mkey not in entry:
                continue
            m    = entry[mkey]
            jac  = f"{_fmt(m['mean_jaccard'])} ± {_fmt(m['std_jaccard'])}"
            tau  = f"{_fmt(m['mean_kendall_tau'])} ± {_fmt(m['std_kendall_tau'])}"
            qsim = f"{_fmt(m['mean_query_sim'])} ± {_fmt(m['std_query_sim'])}"
            if is_thresh:
                rows.append(
                    f"| {canon} "
                    f"| {entry['n_relevant']} "
                    f"| {m.get('orig_n_retrieved', '--')} "
                    f"| {_fmt(m['original_map'])} "
                    f"| {_fmt(m.get('mean_map_para'))} "
                    f"| {_fmt(m['perf_stability_map_std'])} "
                    f"| {_fmt(m['original_precision'])} "
                    f"| {_fmt(m.get('mean_prec_para'))} "
                    f"| {_fmt(m['perf_stability_prec_std'])} "
                    f"| {jac} | {tau} | {qsim} |"
                )
            else:
                rows.append(
                    f"| {canon} "
                    f"| {entry['n_relevant']} "
                    f"| {_fmt(m['original_map'])} "
                    f"| {_fmt(m.get('mean_map_para'))} "
                    f"| {_fmt(m['perf_stability_map_std'])} "
                    f"| {_fmt(m['original_precision'])} "
                    f"| {_fmt(m.get('mean_prec_para'))} "
                    f"| {_fmt(m['perf_stability_prec_std'])} "
                    f"| {jac} | {tau} | {qsim} |"
                )
            all_m.append(m)

        if all_m:
            rows += [
                "",
                f"**{home.capitalize()} {level} averages:**  ",
                f"mAP: {_fmt(_mean_field(all_m, 'original_map'))} → "
                f"{_fmt(_mean_field(all_m, 'mean_map_para'))} "
                f"(std {_fmt(_mean_field(all_m, 'perf_stability_map_std'))}) | "
                f"{prec_hdr}: {_fmt(_mean_field(all_m, 'original_precision'))} → "
                f"{_fmt(_mean_field(all_m, 'mean_prec_para'))} "
                f"(std {_fmt(_mean_field(all_m, 'perf_stability_prec_std'))}) | "
                f"Jaccard: {_fmt(_mean_field(all_m, 'mean_jaccard'))} | "
                f"Kendall τ: {_fmt(_mean_field(all_m, 'mean_kendall_tau'))} | "
                f"Query Sim: {_fmt(_mean_field(all_m, 'mean_query_sim'))}",
                "",
            ]
        return rows

    # ── Assemble report ───────────────────────────────────────────────────────
    thresh_info = ", ".join(
        f"{h}: {RETRIEVAL_THRESHOLDS.get(h, '?')}"
        for h in HOUSEHOLDS
    )
    lines = [
        "# Perturbation Sensitivity Analysis",
        "",
        f"**Setup:** Each activity label's `long_desc` is paraphrased {N_PARAPHRASES} "
        "times by Gemini 2.5 Flash.  ",
        f"**Models:** per-household `*_{MODEL_SUFFIX}`, data: FD_60_p.  ",
        "**Label normalisation:** raw labels → canonical names via `LABEL_NORM`; "
        "noisy/transitional labels excluded.  ",
        "",
        "## Retrieval Modes",
        f"- **Top-{TOP_K}**: fixed top-{TOP_K} cosine-similarity neighbours per query.",
        f"- **Threshold**: all samples with cosine sim ≥ threshold  "
        f"({thresh_info}).",
        "",
        "## Metrics",
        "- **Jaccard**: |orig ∩ para| / |orig ∪ para| over retrieved sets  "
        "(mean ± std across 10 paraphrases)",
        "- **Kendall τ**: rank correlation between original and paraphrase ordered ID lists",
        "- **Query Sim**: cosine similarity between original and paraphrase text embeddings",
        "- **mAP**: full-corpus AP (matches evaluate_embeddings.py); "
        "reported as orig baseline and std across 11 queries",
        "- **Prec**: P@100 for top-k mode; precision of retrieved set for threshold mode; "
        "reported as orig baseline and std across 11 queries",
        "",
        "---",
        "",
        "## Cross-Household Summary",
        "",
    ]

    for mkey, mode_label in MODES:
        lines += [f"### {mode_label}", ""]
        if mkey == "metrics_threshold":
            lines += [
                f"> Thresholds: {thresh_info}  ",
                "> n_retr = avg number of samples retrieved per query.",
                "",
            ]
        for level in LEVELS:
            tbl = _cross_summary_table(level, mkey, mode_label)
            if tbl:
                lines += [f"#### {level} Labels", ""] + tbl + [""]

    lines += ["---", ""]

    # ── Per-household per-level detail ────────────────────────────────────────
    for home in HOUSEHOLDS:
        if home not in results:
            continue
        lines += [f"## {home.capitalize()}", ""]
        thresh = RETRIEVAL_THRESHOLDS.get(home, "?")

        for level in LEVELS:
            level_data = results[home].get(level, {})
            if not level_data:
                continue

            lines += [f"### {home.capitalize()} — {level} Labels", ""]

            # Top-k sub-section
            lines += [f"#### Top-{TOP_K} Retrieval", ""]
            lines += _detail_table(home, level, "metrics_topk")

            # Threshold sub-section
            lines += [
                f"#### Threshold Retrieval (sim ≥ {thresh})",
                "",
            ]
            lines += _detail_table(home, level, "metrics_threshold")

    return "\n".join(lines) + "\n"


# ── Terminal summary printer ──────────────────────────────────────────────────

def _f3(v) -> str:
    """Format float to 3 decimal places or '--' if None."""
    return f"{v:.3f}" if v is not None else " -- "


def print_home_summary(
    home: str,
    home_results: dict,
    threshold: float,
    k: int,
) -> None:
    """Print a compact per-label summary table to stdout after a household finishes.

    Prints two sub-tables per label level:
      1. Top-k retrieval metrics
      2. Threshold retrieval metrics
    """
    SEP = "─" * 100
    print(f"\n  {SEP}")
    print(f"  {home.upper()} — Perturbation Sensitivity Summary"
          f"   (top-k={k}  |  threshold ≥ {threshold})")
    print(f"  {SEP}")

    for level in LEVELS:
        level_data = home_results.get(level, {})
        if not level_data:
            continue

        labels = list(level_data.items())
        LW = max(len(c) for c, _ in labels)   # label column width
        LW = max(LW, 5)

        # ── Top-k table ───────────────────────────────────────────────────────
        print(f"\n  {level} Labels  ·  Top-{k} Retrieval")
        hdr = (
            f"  {'Label':<{LW}}  {'n_rel':>5}  "
            f"{'OmAP':>6} {'MmAP':>6} {'σmAP':>6}  "
            f"{'OP@k':>6} {'MP@k':>6} {'σP@k':>6}  "
            f"{'Jacc':>6} {'Kend':>6} {'QSim':>6}"
        )
        print(hdr)
        print(f"  {'─'*LW}  {'─'*5}  "
              f"{'─'*6} {'─'*6} {'─'*6}  "
              f"{'─'*6} {'─'*6} {'─'*6}  "
              f"{'─'*6} {'─'*6} {'─'*6}")

        topk_cols = [
            "original_map", "mean_map_para", "perf_stability_map_std",
            "original_precision", "mean_prec_para", "perf_stability_prec_std",
            "mean_jaccard", "mean_kendall_tau", "mean_query_sim",
        ]
        agg: Dict[str, List[float]] = {c: [] for c in topk_cols}
        for canon, entry in labels:
            m = entry.get("metrics_topk", {})
            vals = [m.get(c) for c in topk_cols]
            for c, v in zip(topk_cols, vals):
                if v is not None:
                    agg[c].append(v)
            print(
                f"  {canon:<{LW}}  {entry['n_relevant']:>5}  "
                + "  ".join(_f3(v) for v in [vals[0], vals[1], vals[2]])
                + "  "
                + "  ".join(_f3(v) for v in [vals[3], vals[4], vals[5]])
                + "  "
                + "  ".join(_f3(v) for v in [vals[6], vals[7], vals[8]])
            )

        # averages row
        avgs = [float(np.mean(agg[c])) if agg[c] else None for c in topk_cols]
        print(f"  {'─'*LW}  {'─'*5}  "
              f"{'─'*6} {'─'*6} {'─'*6}  "
              f"{'─'*6} {'─'*6} {'─'*6}  "
              f"{'─'*6} {'─'*6} {'─'*6}")
        print(
            f"  {'AVERAGE':<{LW}}  {'':>5}  "
            + "  ".join(_f3(v) for v in avgs[:3])
            + "  "
            + "  ".join(_f3(v) for v in avgs[3:6])
            + "  "
            + "  ".join(_f3(v) for v in avgs[6:])
        )

        # ── Threshold table ───────────────────────────────────────────────────
        print(f"\n  {level} Labels  ·  Threshold Retrieval (sim ≥ {threshold})")
        hdr2 = (
            f"  {'Label':<{LW}}  {'n_rel':>5}  {'n_ret':>6}  "
            f"{'OmAP':>6} {'MmAP':>6} {'σmAP':>6}  "
            f"{'OPrec':>6} {'MPrec':>6} {'σPrec':>6}  "
            f"{'Jacc':>6} {'Kend':>6} {'QSim':>6}"
        )
        print(hdr2)
        print(f"  {'─'*LW}  {'─'*5}  {'─'*6}  "
              f"{'─'*6} {'─'*6} {'─'*6}  "
              f"{'─'*6} {'─'*6} {'─'*6}  "
              f"{'─'*6} {'─'*6} {'─'*6}")

        thresh_cols = [
            "orig_n_retrieved",
            "original_map", "mean_map_para", "perf_stability_map_std",
            "original_precision", "mean_prec_para", "perf_stability_prec_std",
            "mean_jaccard", "mean_kendall_tau", "mean_query_sim",
        ]
        agg2: Dict[str, List[float]] = {c: [] for c in thresh_cols}
        for canon, entry in labels:
            m = entry.get("metrics_threshold", {})
            vals = [m.get(c) for c in thresh_cols]
            for c, v in zip(thresh_cols, vals):
                if v is not None:
                    agg2[c].append(v)
            n_ret = vals[0]
            n_ret_s = f"{int(n_ret):>6}" if n_ret is not None else "  -- "
            print(
                f"  {canon:<{LW}}  {entry['n_relevant']:>5}  {n_ret_s}  "
                + "  ".join(_f3(v) for v in [vals[1], vals[2], vals[3]])
                + "  "
                + "  ".join(_f3(v) for v in [vals[4], vals[5], vals[6]])
                + "  "
                + "  ".join(_f3(v) for v in [vals[7], vals[8], vals[9]])
            )

        avgs2 = [float(np.mean(agg2[c])) if agg2[c] else None for c in thresh_cols]
        print(f"  {'─'*LW}  {'─'*5}  {'─'*6}  "
              f"{'─'*6} {'─'*6} {'─'*6}  "
              f"{'─'*6} {'─'*6} {'─'*6}  "
              f"{'─'*6} {'─'*6} {'─'*6}")
        n_ret_avg = f"{avgs2[0]:.1f}" if avgs2[0] is not None else "  -- "
        print(
            f"  {'AVERAGE':<{LW}}  {'':>5}  {n_ret_avg:>6}  "
            + "  ".join(_f3(v) for v in avgs2[1:4])
            + "  "
            + "  ".join(_f3(v) for v in avgs2[4:7])
            + "  "
            + "  ".join(_f3(v) for v in avgs2[7:])
        )

    print(f"\n  {SEP}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--households", nargs="+", default=HOUSEHOLDS,
        help="Households to process (default: milan aruba cairo)",
    )
    parser.add_argument(
        "--force_recompute", action="store_true",
        help="Force LLM paraphrase regeneration even if cached paraphrases exist",
    )
    parser.add_argument(
        "--top_k", type=int, default=TOP_K,
        help=f"Number of neighbors to retrieve (default: {TOP_K})",
    )
    args = parser.parse_args()

    k      = args.top_k
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    with open(METADATA) as f:
        all_meta = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paraphrases_path = OUT_DIR / "paraphrases.json"
    paraphrase_cache = load_cached_paraphrases(paraphrases_path)

    # ── Collect labels per household per level ────────────────────────────────
    # home → level → {canon_label: long_desc}
    home_level_labels: Dict[str, Dict[str, Dict[str, str]]] = {}
    for home in args.households:
        home_level_labels[home] = {}
        for level in LEVELS:
            labels = get_canonical_labels_for_level(home, level, all_meta)
            home_level_labels[home][level] = labels
            print(f"[{home}] {level}: {len(labels)} canonical labels  "
                  f"({', '.join(labels.keys())})")

    # ── Generate paraphrases (LLM step) ──────────────────────────────────────
    # Key in cache: home → level → canon_label → {original, paraphrases}
    paraphrase_dirty = False
    for home in args.households:
        paraphrase_cache.setdefault(home, {})
        for level in LEVELS:
            paraphrase_cache[home].setdefault(level, {})
            for canon, long_desc in home_level_labels[home][level].items():
                cached = paraphrase_cache[home][level].get(canon, {}).get("paraphrases")
                if cached is not None and not args.force_recompute:
                    continue

                print(f"\n  [{home}/{level}] Paraphrasing '{canon}' …")
                print(f"    Original: {long_desc[:100]}")
                paraphrases = generate_paraphrases_gemini(long_desc, canon, n=N_PARAPHRASES)
                paraphrase_cache[home][level][canon] = {
                    "original":    long_desc,
                    "paraphrases": paraphrases,
                }
                paraphrase_dirty = True
                save_paraphrases(paraphrase_cache, paraphrases_path)

    if paraphrase_dirty:
        save_paraphrases(paraphrase_cache, paraphrases_path)

    # ── Retrieval + metrics per household ─────────────────────────────────────
    # Results structure: home → level → canon_label → {original, paraphrases, metrics, n_relevant}
    full_results: dict = {}

    for home in args.households:
        has_labels = any(
            bool(home_level_labels[home].get(lv))
            for lv in LEVELS
        )
        if not has_labels:
            print(f"\nSkipping {home}: no labels found")
            continue

        print(f"\n{'='*60}")
        print(f"  {home.upper()}")
        print(f"{'='*60}")

        text_encoder, sensor_encoder, vocab_sizes = load_models(home, device)
        sensor_embs, sample_ids, id_to_sample = extract_sensor_embeddings(
            home, sensor_encoder, text_encoder, vocab_sizes, device
        )

        full_results[home] = {}

        for level in LEVELS:
            labels = home_level_labels[home].get(level, {})
            if not labels:
                print(f"\n  [{home}/{level}] No labels — skipping")
                continue

            print(f"\n  ── {level} labels ──")
            full_results[home][level] = {}

            for canon, long_desc in labels.items():
                print(f"\n  [{level}] {canon}")

                cached_entry = paraphrase_cache.get(home, {}).get(level, {}).get(canon, {})
                paraphrases  = cached_entry.get("paraphrases", [])
                if not paraphrases:
                    print(f"    WARNING: no paraphrases for '{canon}' — skipping")
                    continue

                relevance_map = build_relevance_map(
                    sample_ids, id_to_sample, canon, level
                )
                n_relevant = sum(relevance_map.values())
                print(f"    relevant samples in test: {n_relevant}")

                orig_emb = encode_query(long_desc, text_encoder, device)
                # Full-corpus ranking shared by both retrieval modes
                orig_all_ids, _ = retrieve_ranked(orig_emb, sensor_embs, sample_ids)
                orig_top_ids    = orig_all_ids[:k]
                threshold       = RETRIEVAL_THRESHOLDS[home]
                orig_thresh_ids, _ = retrieve_within_threshold(
                    orig_emb, sensor_embs, sample_ids, threshold
                )

                para_all_ids_list:    List[List[str]] = []
                para_top_ids_list:    List[List[str]] = []
                para_thresh_ids_list: List[List[str]] = []
                para_embs:            List[np.ndarray] = []
                for i, para_text in enumerate(paraphrases, start=1):
                    para_emb = encode_query(para_text, text_encoder, device)
                    para_all_ids, _  = retrieve_ranked(para_emb, sensor_embs, sample_ids)
                    para_top_ids     = para_all_ids[:k]
                    para_thresh_ids, _ = retrieve_within_threshold(
                        para_emb, sensor_embs, sample_ids, threshold
                    )
                    para_all_ids_list.append(para_all_ids)
                    para_top_ids_list.append(para_top_ids)
                    para_thresh_ids_list.append(para_thresh_ids)
                    para_embs.append(para_emb)
                    print(
                        f"    {i:2d}: "
                        f"jac_k={jaccard(set(orig_top_ids), set(para_top_ids)):.3f}  "
                        f"jac_t={jaccard(set(orig_thresh_ids), set(para_thresh_ids)):.3f}  "
                        f"qsim={query_cosine_similarity(orig_emb, para_emb):.3f}  "
                        f"n_t={len(para_thresh_ids)}"
                    )

                metrics_topk = compute_metrics(
                    original_all_ids        = orig_all_ids,
                    original_sel_ids        = orig_top_ids,
                    original_emb            = orig_emb,
                    paraphrase_all_ids_list = para_all_ids_list,
                    paraphrase_sel_ids_list = para_top_ids_list,
                    paraphrase_embs         = para_embs,
                    relevance_map           = relevance_map,
                    mode                    = "topk",
                    k                       = k,
                )

                metrics_threshold = compute_metrics(
                    original_all_ids        = orig_all_ids,
                    original_sel_ids        = orig_thresh_ids,
                    original_emb            = orig_emb,
                    paraphrase_all_ids_list = para_all_ids_list,
                    paraphrase_sel_ids_list = para_thresh_ids_list,
                    paraphrase_embs         = para_embs,
                    relevance_map           = relevance_map,
                    mode                    = "threshold",
                    k                       = k,
                )

                full_results[home][level][canon] = {
                    "original":          long_desc,
                    "paraphrases":       paraphrases,
                    "n_relevant":        n_relevant,
                    "metrics_topk":      metrics_topk,
                    "metrics_threshold": metrics_threshold,
                }

        # Print per-household summary table after all levels are done
        print_home_summary(
            home,
            full_results[home],
            threshold=RETRIEVAL_THRESHOLDS[home],
            k=k,
        )

    # ── Export ────────────────────────────────────────────────────────────────
    results_path = OUT_DIR / "results.json"
    results_path.write_text(
        json.dumps(full_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResults JSON → {results_path}")

    report_path = OUT_DIR / "summary.md"
    report_path.write_text(build_report(full_results), encoding="utf-8")
    print(f"Markdown report → {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
