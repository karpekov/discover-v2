#!/usr/bin/env python3
"""
Retrieval Threshold Sensitivity Evaluator for CASAS HAR datasets.

Measures precision-at-threshold (P@θ) for each L1 activity label:
for each label's projected text prototype, retrieve test sensor samples
whose cosine similarity to that prototype is ≥ θ, and report what fraction
actually belong to that label.

Thresholds evaluated: 0.05, 0.10, 0.15, 0.20, 0.30
  - 0.05/0.10 : current operational range used in the system
  - 0.15/0.20 : moderate selectivity — expected to improve precision
  - 0.30      : high-confidence tier — very selective

Usage:
  python src/evals/eval_retrieval_threshold.py \\
      --datasets milan aruba cairo \\
      --model_suffix fd60_seq_rb1_textclip_projmlp_clipmlm_v3 \\
      --thresholds 0.05 0.10 0.15 0.20 0.30 \\
      --max_samples 10000 \\
      --filter_noisy_labels \\
      --filter_minor_labels \\
      --output_dir results/evals/imwut_paper
"""

import sys
import os
_HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(_HERE, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '../..')))

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evals.evaluate_embeddings import EmbeddingEvaluator
from evals.compute_retrieval_metrics import normalize_embeddings


# ── label normalisation (mirrors retrieval_metrics_agg.py) ────────────────────

LABEL_NORM: Dict[str, str] = {
    "sleep":                "Sleep",
    "sleeping":             "Sleep",
    "bed_to_toilet":        "Bed to Toilet",
    "bed to toilet":        "Bed to Toilet",
    "leave home":           "Leave Home",
    "leave_home":           "Leave Home",
    "chores":               "Chores",
    "laundry":              "Chores",
    "eating":               "Eating: General",
    "eat":                  "Eating: General",
    "dining_rm_activity":   "Eating: General",
    "breakfast":            "Eating: Breakfast",
    "lunch":                "Eating: Lunch",
    "dinner":               "Eating: Dinner",
    "work":                 "Work",
    "desk_activity":        "Work",
    "r1 work in office":    "Work",
    "relax":                "Relax: General",
    "watch_tv":             "Relax: Watch TV",
    "read":                 "Relax: Read",
}

CANONICAL_ORDER: List[str] = [
    "Sleep", "Bed to Toilet", "Leave Home", "Chores",
    "Eating: General", "Eating: Breakfast", "Eating: Lunch", "Eating: Dinner",
    "Work",
    "Relax: General", "Relax: Watch TV", "Relax: Read",
]
_CANONICAL_RANK = {lbl: i for i, lbl in enumerate(CANONICAL_ORDER)}


def normalize_label(raw: str) -> str:
    return LABEL_NORM.get(raw.lower().strip(), raw.strip())


def canonical_sort_key(label: str) -> Tuple[int, str]:
    return (_CANONICAL_RANK.get(label, len(CANONICAL_ORDER)), label)


# ── core threshold evaluation ─────────────────────────────────────────────────

def compute_precision_at_thresholds(
    prototypes:  Dict[str, np.ndarray],
    sensor_emb:  np.ndarray,
    labels_l1:   List[str],
    thresholds:  List[float],
) -> Dict[str, Dict[float, Dict]]:
    """Return per-label precision/recall/counts at each threshold.

    Direction: text-prototype → test-sensor-samples.
    For each label L and threshold θ:
      retrieved = {i : sim(proto_L, sensor_i) ≥ θ}
      precision = |{i ∈ retrieved : label_i == L}| / |retrieved|
      recall    = |{i ∈ retrieved : label_i == L}| / |{i : label_i == L}|
      f1        = 2 * precision * recall / (precision + recall)

    Args:
        prototypes:  {label: (D,) embedding}
        sensor_emb:  (N, D) test sensor embeddings (will be L2-normalised)
        labels_l1:   ground-truth L1 labels, length N
        thresholds:  sorted list of cosine-similarity cut-offs

    Returns:
        {label: {threshold: {precision, recall, f1, retrieved_count, total_count}}}
    """
    sensor_norm = normalize_embeddings(sensor_emb)           # (N, D)
    label_list  = list(prototypes.keys())
    proto_mat   = np.stack([
        normalize_embeddings(prototypes[l].reshape(1, -1))[0]
        for l in label_list
    ])                                                        # (L, D)

    # similarity matrix: (L, N)
    sim_matrix = np.dot(proto_mat, sensor_norm.T)

    results: Dict[str, Dict[float, Dict]] = {}
    labels_arr = np.array([l.lower().strip() for l in labels_l1])

    for i, label in enumerate(label_list):
        sims        = sim_matrix[i]                           # (N,)
        label_lower = label.lower().strip()
        pos_mask    = labels_arr == label_lower
        total_pos   = int(pos_mask.sum())

        results[label] = {}
        for thr in thresholds:
            ret_mask   = sims >= thr
            ret_count  = int(ret_mask.sum())

            if ret_count == 0:
                precision = None
                recall    = None
                f1        = None
            else:
                tp        = int((ret_mask & pos_mask).sum())
                precision = tp / ret_count
                recall    = tp / total_pos if total_pos > 0 else None
                if precision is not None and recall is not None and (precision + recall) > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = None

            results[label][thr] = {
                "precision":       precision,
                "recall":          recall,
                "f1":              f1,
                "retrieved_count": ret_count,
                "total_count":     total_pos,
            }

    return results


# ── per-dataset evaluation ─────────────────────────────────────────────────────

def evaluate_dataset(
    dataset:             str,
    model_suffix:        str,
    thresholds:          List[float],
    max_samples:         int,
    filter_noisy:        bool,
    filter_minor:        bool,
    description_style:   str,
    base_dir:            Path,
) -> Dict:
    """Run threshold evaluation for one dataset. Returns structured result dict."""

    model_name   = f"{dataset}_{model_suffix}"
    checkpoint   = base_dir / "trained_models" / dataset / model_name / "best_model.pt"
    data_root    = base_dir / "data" / "processed" / "casas" / dataset / "FD_60_p"

    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset.upper()}")
    print(f"{'='*70}")

    config = {
        "checkpoint_path": str(checkpoint),
        "train_data_path": str(data_root / "train.json"),
        "test_data_path":  str(data_root / "test.json"),
        "vocab_path":      str(data_root / "vocab.json"),
        "output_dir":      str(base_dir / "results" / "evals" / dataset / "FD_60_p" / model_name),
        "description_style": description_style,
    }

    evaluator = EmbeddingEvaluator(config)

    # ── 1. Build L1 prototype set (from metadata, with filtering) ─────────────
    train_labels_l1, _ = evaluator.get_labels_from_metadata(dataset)

    _extra_excl: Optional[set] = None
    if filter_minor:
        filter_noisy = True
        _extra_excl = evaluator._MINOR_LABELS_BY_DATASET.get(dataset.lower(), set())
        if _extra_excl:
            print(f"  ⚠️  filter_minor_labels: also dropping {sorted(_extra_excl)}")

    if filter_noisy:
        exclude = {
            "other", "no_activity", "no activity",
            "no_sensor_readings", "no sensor readings", "no sensor reading",
            "unknown", "none", "null", "nan",
            "other activity", "miscellaneous", "misc",
        }
        if _extra_excl:
            exclude |= {e.lower().strip() for e in _extra_excl}
        train_labels_l1 = [l for l in train_labels_l1 if l.lower().strip() not in exclude]

    # ── 2. Create projected text prototypes ───────────────────────────────────
    print(f"\n  Creating projected text prototypes for {len(train_labels_l1)} L1 labels…")
    prototypes, _ = evaluator.create_text_prototypes(
        train_labels_l1, apply_projection=True
    )
    print(f"  ✅ {len(prototypes)} prototypes created")

    # ── 3. Extract test sensor embeddings ─────────────────────────────────────
    print(f"\n  Extracting test sensor embeddings (max_samples={max_samples})…")
    sensor_emb, labels_l1, labels_l2, _ = evaluator.extract_embeddings_and_labels(
        "test", max_samples
    )

    # ── 4. Cairo label merges ─────────────────────────────────────────────────
    if dataset.lower() == "cairo":
        labels_l1 = evaluator._apply_cairo_label_merges(labels_l1)

    # ── 5. Apply sample-level label filtering ─────────────────────────────────
    if filter_noisy:
        sensor_emb, labels_l1, labels_l2, _ = evaluator.filter_noisy_labels(
            sensor_emb, labels_l1, labels_l2,
            extra_exclude=_extra_excl,
        )

    print(f"  📊 {len(labels_l1)} test samples after filtering")

    # ── 6. Threshold evaluation ────────────────────────────────────────────────
    print(f"\n  Computing precision at thresholds: {thresholds}")
    raw_results = compute_precision_at_thresholds(
        prototypes, sensor_emb, labels_l1, thresholds
    )

    # ── 7. Normalise label names ───────────────────────────────────────────────
    normalised: Dict[str, Dict[float, Dict]] = {}
    for raw_label, thr_stats in raw_results.items():
        canon = normalize_label(raw_label)
        if canon not in normalised:
            normalised[canon] = {}
        for thr, stats in thr_stats.items():
            if thr not in normalised[canon]:
                normalised[canon][thr] = {
                    "precision": [],
                    "recall":    [],
                    "f1":        [],
                    "retrieved_count": 0,
                    "total_count":     0,
                }
            bucket = normalised[canon][thr]
            if stats["precision"] is not None:
                bucket["precision"].append(stats["precision"])
            if stats["recall"] is not None:
                bucket["recall"].append(stats["recall"])
            if stats["f1"] is not None:
                bucket["f1"].append(stats["f1"])
            bucket["retrieved_count"] += stats["retrieved_count"]
            bucket["total_count"]     += stats["total_count"]

    # Average any merged labels
    merged: Dict[str, Dict[float, Dict]] = {}
    for canon, thr_map in normalised.items():
        merged[canon] = {}
        for thr, bucket in thr_map.items():
            prec_list = bucket["precision"]
            rec_list  = bucket["recall"]
            f1_list   = bucket["f1"]
            p = float(np.mean(prec_list)) if prec_list else None
            r = float(np.mean(rec_list))  if rec_list  else None
            merged[canon][thr] = {
                "precision":       p,
                "recall":          r,
                "f1":              float(np.mean(f1_list)) if f1_list else None,
                "retrieved_count": bucket["retrieved_count"],
                "total_count":     bucket["total_count"],
            }

    return {
        "dataset":    dataset,
        "n_test":     len(labels_l1),
        "thresholds": thresholds,
        "per_label":  {
            lbl: {str(thr): stats for thr, stats in thr_stats.items()}
            for lbl, thr_stats in merged.items()
        },
    }


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate_results(
    per_dataset: List[Dict],
    thresholds:  List[float],
) -> Dict:
    """Compute macro-average precision, recall and F1 per dataset and cross-dataset."""
    datasets = [d["dataset"] for d in per_dataset]
    thr_strs = [str(t) for t in thresholds]

    def _macro(res: Dict, metric: str) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        for thr_str in thr_strs:
            vals = [
                lbl_stats[thr_str][metric]
                for lbl_stats in res["per_label"].values()
                if lbl_stats.get(thr_str, {}).get(metric) is not None
            ]
            out[thr_str] = float(np.mean(vals)) if vals else None
        return out

    per_dataset_macro: Dict[str, Dict] = {}
    for res in per_dataset:
        ds = res["dataset"]
        per_dataset_macro[ds] = {
            "precision": _macro(res, "precision"),
            "recall":    _macro(res, "recall"),
            "f1":        _macro(res, "f1"),
        }

    def _cross(metric: str) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        for thr_str in thr_strs:
            vals = [
                per_dataset_macro[ds][metric].get(thr_str)
                for ds in datasets
                if per_dataset_macro[ds][metric].get(thr_str) is not None
            ]
            out[thr_str] = float(np.mean(vals)) if vals else None
        return out

    return {
        "datasets":          datasets,
        "thresholds":        thresholds,
        "per_dataset_macro": per_dataset_macro,
        "overall_macro": {
            "precision": _cross("precision"),
            "recall":    _cross("recall"),
            "f1":        _cross("f1"),
        },
    }


# ── markdown report ───────────────────────────────────────────────────────────

_MISS = "—"

def _fmt(v: Optional[float], pct: bool = True) -> str:
    if v is None:
        return _MISS
    return f"{v*100:.1f}" if pct else f"{v:.4f}"


def _per_label_table(
    ds_res:   Dict,
    thr_strs: List[str],
    thr_cols: List[str],
    metric:   str,
) -> List[str]:
    """Return markdown rows for one dataset × one metric."""
    ds_labels  = sorted(ds_res["per_label"].keys(), key=canonical_sort_key)
    macro_vals = {t: [] for t in thr_strs}
    rows: List[str] = []

    for lbl in ds_labels:
        stats = ds_res["per_label"].get(lbl, {})
        total = stats.get(thr_strs[0], {}).get("total_count", 0)
        cells = []
        for thr_str in thr_strs:
            v = stats.get(thr_str, {}).get(metric)
            cells.append(_fmt(v))
            if v is not None:
                macro_vals[thr_str].append(v)
        rows.append(f"| {lbl} | {' | '.join(cells)} | {total} |")

    macro_cells = [
        _fmt(float(np.mean(macro_vals[t])) if macro_vals[t] else None)
        for t in thr_strs
    ]
    rows.append(f"| **Macro avg** | {' | '.join(macro_cells)} | |")
    return rows


def build_markdown(
    per_dataset:  List[Dict],
    agg:          Dict,
    thresholds:   List[float],
    model_suffix: str,
) -> str:
    datasets  = [d["dataset"] for d in per_dataset]
    thr_strs  = [str(t) for t in thresholds]
    thr_cols  = [f"θ={t}" for t in thresholds]

    col_sep = f"| --- | {' | '.join(['---'] * len(thresholds))} | --- |"

    lines: List[str] = [
        "# Retrieval Threshold Sensitivity Analysis",
        "",
        f"> **Model:** `{model_suffix}`  ",
        f"> **Direction:** projected text prototype → test sensor samples  ",
        f"> **Metrics:** Precision@θ, Recall@θ, F1@θ  ",
        f"> **Filtering:** `filter_noisy_labels=True`, `filter_minor_labels=True`",
        "",
    ]

    # ── macro summary tables (one per metric) ─────────────────────────────────
    for metric, title, note in [
        ("precision", "1. Macro Precision per Dataset",
         "Macro-averaged precision (%) — what fraction of retrieved samples belong to the queried label."),
        ("recall",    "2. Macro Recall per Dataset",
         "Macro-averaged recall (%) — what fraction of true positives are retrieved."),
        ("f1",        "3. Macro F1 per Dataset",
         "Macro-averaged F1 (%) — harmonic mean of precision and recall. Use this to find the best threshold."),
    ]:
        lines += [
            f"## {title}",
            "",
            f"| Dataset | {' | '.join(thr_cols)} |",
            f"| --- | {' | '.join(['---'] * len(thresholds))} |",
        ]
        for ds in datasets:
            row_vals = " | ".join(
                _fmt(agg["per_dataset_macro"][ds][metric].get(t)) for t in thr_strs
            )
            lines.append(f"| **{ds.capitalize()}** | {row_vals} |")
        ov_vals = " | ".join(
            _fmt(agg["overall_macro"][metric].get(t)) for t in thr_strs
        )
        lines += [
            f"| **Average** | {ov_vals} |",
            "",
            f"> {note}",
            "",
            "---",
            "",
        ]

    # ── per-label detail tables ────────────────────────────────────────────────
    section_num = 4
    for metric, title, note in [
        ("precision", "Per-Label Precision (%)",
         "Fraction of retrieved samples that truly belong to the queried label."),
        ("recall",    "Per-Label Recall (%)",
         "Fraction of all true positives that are retrieved at each threshold."),
        ("f1",        "Per-Label F1 (%)",
         "Harmonic mean of precision and recall. Higher = better threshold for that label."),
    ]:
        lines += [
            f"## {section_num}. {title}",
            "",
            f"> {note}",
            "",
        ]
        for ds_res in per_dataset:
            ds = ds_res["dataset"]
            lines += [
                f"### {ds.capitalize()}",
                "",
                f"| Label | {' | '.join(thr_cols)} | Total samples |",
                col_sep,
            ]
            lines += _per_label_table(ds_res, thr_strs, thr_cols, metric)
            lines.append("")
        lines += ["---", ""]
        section_num += 1

    # ── retrieved counts ───────────────────────────────────────────────────────
    lines += [
        f"## {section_num}. Retrieved Sample Counts per Label",
        "",
        "> Absolute number of test samples retrieved at each threshold.",
        "",
    ]
    for ds_res in per_dataset:
        ds        = ds_res["dataset"]
        ds_labels = sorted(ds_res["per_label"].keys(), key=canonical_sort_key)
        lines += [
            f"### {ds.capitalize()}",
            "",
            f"| Label | {' | '.join(thr_cols)} | Total samples |",
            col_sep,
        ]
        for lbl in ds_labels:
            stats = ds_res["per_label"].get(lbl, {})
            total = stats.get(thr_strs[0], {}).get("total_count", 0)
            cells = [
                str(stats.get(t, {}).get("retrieved_count", "—"))
                for t in thr_strs
            ]
            lines.append(f"| {lbl} | {' | '.join(cells)} | {total} |")
        lines.append("")

    return "\n".join(lines)


# ── JSON serialisation ────────────────────────────────────────────────────────

def build_json(
    per_dataset:  List[Dict],
    agg:          Dict,
    model_suffix: str,
    thresholds:   List[float],
) -> dict:
    return {
        "meta": {
            "model_suffix":    model_suffix,
            "thresholds":      thresholds,
            "direction":       "prototype2sensor",
            "filter_noisy":    True,
            "filter_minor":    True,
        },
        "per_dataset":       {d["dataset"]: d for d in per_dataset},
        "aggregation":       agg,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval threshold sensitivity analysis for CASAS HAR"
    )
    parser.add_argument("--datasets",      nargs="+", default=["milan", "aruba", "cairo"])
    parser.add_argument("--model_suffix",  default="fd60_seq_rb1_textclip_projmlp_clipmlm_v3")
    parser.add_argument("--thresholds",    nargs="+", type=float,
                        default=[0.05, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--max_samples",   type=int, default=10000)
    parser.add_argument("--filter_noisy_labels",  action="store_true")
    parser.add_argument("--filter_minor_labels",  action="store_true")
    parser.add_argument("--description_style",    default="long_desc")
    parser.add_argument("--output_dir",    default="results/evals/imwut_paper")
    parser.add_argument("--base_dir",      default=".")
    args = parser.parse_args()

    base_dir    = Path(args.base_dir).resolve()
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = sorted(args.thresholds)

    # Run per-dataset evaluations
    per_dataset_results: List[Dict] = []
    for ds in args.datasets:
        result = evaluate_dataset(
            dataset=ds,
            model_suffix=args.model_suffix,
            thresholds=thresholds,
            max_samples=args.max_samples,
            filter_noisy=args.filter_noisy_labels or args.filter_minor_labels,
            filter_minor=args.filter_minor_labels,
            description_style=args.description_style,
            base_dir=base_dir,
        )
        per_dataset_results.append(result)

    # Aggregate
    agg = aggregate_results(per_dataset_results, thresholds)

    # Build outputs
    md_text  = build_markdown(per_dataset_results, agg, thresholds, args.model_suffix)
    json_obj = build_json(per_dataset_results, agg, args.model_suffix, thresholds)

    stem     = "retrieval_threshold_sensitivity"
    md_path  = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"

    md_path.write_text(md_text, encoding="utf-8")
    with open(json_path, "w") as f:
        json.dump(json_obj, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Results saved to:")
    print(f"    {md_path}")
    print(f"    {json_path}")
    print(f"{'='*70}")

    # Print macro summary to terminal
    thr_strs = [str(t) for t in thresholds]
    header   = f"{'Dataset':<12}" + "".join(f"  θ={t:<5}" for t in thresholds)
    divider  = "─" * len(header)

    for metric_key, metric_label in [
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("f1",        "F1"),
    ]:
        print(f"\n── Macro {metric_label}@θ (%) {'─' * (50 - len(metric_label))}")
        print(header)
        print(divider)
        for ds in args.datasets:
            row = f"{ds.capitalize():<12}"
            for thr_str in thr_strs:
                v = agg["per_dataset_macro"][ds][metric_key].get(thr_str)
                row += f"  {_fmt(v):<7}" if v is not None else f"  {'—':<7}"
            print(row)
        print(divider)
        ov_row = f"{'Average':<12}"
        for thr_str in thr_strs:
            v = agg["overall_macro"][metric_key].get(thr_str)
            ov_row += f"  {_fmt(v):<7}" if v is not None else f"  {'—':<7}"
        print(ov_row)


if __name__ == "__main__":
    main()
