"""
Cluster → label mapping evaluator with a limited annotation budget.

Protocol (shared by all clustering methods in the DISCOVER revision):
  0. Clusters are fit on continuous, unlabelled train windows (FL_20 train).
  1. Labelled windows always come from pre-segmented data (FL_20_p), so every window has
     exactly one true label. FL_20_p uses the same day split as FL_20. Pre-segmented train
     windows (pool) are assigned to clusters with a confidence score.
  2. For each cluster, the top-m pool windows by confidence are "annotated" with their
     CASAS label and the cluster is mapped to the majority label (m = 'all' uses every
     pool window of the cluster).
  3. Pre-segmented test windows (unseen days) get their cluster's mapped label and are
     scored against their own label. CIs come from a block bootstrap over test days.

Scoring variants:
  - 'all':        every labelled window (pre-segmented data has no No_Activity).
  - 'excl_other': windows labelled Other (or No_Activity) removed from both pool and test.
"""
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

EXCLUDED_LABELS = ('No_Activity', 'no_activity', 'Other')
UNMAPPED = '__unmapped__'
DEFAULT_M_VALUES = (5, 10, 50, 'all')
VARIANTS = ('all', 'excl_other')
DATA_ROOT = Path('data/processed/casas')
CACHE_DIR = Path('results/evals/dv1_review/cache')


def load_windows(data_dir: Union[str, Path], split: str, cache_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load one split of a processed CASAS dataset as a compact per-window DataFrame.

    Columns: sample_id, day, label (l2 of the last event), label_primary_l2, sensors, rooms, tods.
    """
    data_dir = Path(data_dir)
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{data_dir.parent.name}_{data_dir.name}_{split}_v2.pkl"
        if cache_path.exists():
            return pd.read_pickle(cache_path)

    with open(data_dir / f'{split}.json') as f:
        samples = json.load(f)['samples']

    rows = []
    for s in samples:
        seq = s['sensor_sequence']
        rows.append({
            'sample_id': s['sample_id'],
            'day': s['metadata']['start_time'][:10],
            'label': seq[-1]['activity_l2'],
            'label_primary_l2': s['metadata']['ground_truth_labels']['primary_l2'],
            'sensors': [e['sensor_id'] for e in seq],
            # 2 Aruba test events (malformed sensor id 'c') have no room.
            'rooms': [e.get('room', 'unknown') for e in seq],
            'tods': [e.get('tod_bucket', 'unknown') for e in seq],
        })
    del samples
    df = pd.DataFrame(rows)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
    return df


def load_protocol(city: str, fit_window: str = 'FL_20', label_window: str = 'FL_20_p'):
    """Return (fit, pool, test): unlabelled fit windows, labelled pool windows, labelled test windows."""
    fit = load_windows(DATA_ROOT / city / fit_window, 'train', CACHE_DIR)
    pool = load_windows(DATA_ROOT / city / label_window, 'train', CACHE_DIR)
    test = load_windows(DATA_ROOT / city / label_window, 'test', CACHE_DIR)
    return fit, pool, test


def map_clusters(pool_clusters: np.ndarray, pool_conf: np.ndarray, pool_labels: np.ndarray,
                 m: Union[int, str]) -> Dict[int, str]:
    """Map each cluster to the majority label of its top-m most confident pool windows."""
    mapping = {}
    for c in np.unique(pool_clusters):
        idx = np.flatnonzero(pool_clusters == c)
        if m != 'all':
            idx = idx[np.argsort(-pool_conf[idx], kind='stable')[:int(m)]]
        counts = Counter(pool_labels[idx])
        # Ties broken by label name for determinism.
        mapping[c] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return mapping


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, labels=np.unique(y_true), average='macro', zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }


def _scores_from_confusion(cm: np.ndarray, n_true_classes: int) -> Dict[str, float]:
    """Same metrics as `_scores`, from a confusion matrix whose first `n_true_classes`
    rows/cols are the true-label classes (remaining cols are predicted-only labels)."""
    tp = np.diag(cm)[:n_true_classes].astype(float)
    support = cm[:n_true_classes].sum(1).astype(float)
    pred_count = cm[:, :n_true_classes].sum(0).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.where(support + pred_count > 0, 2 * tp / (support + pred_count), 0.0)
    total = support.sum()
    present = support > 0
    return {
        'f1_weighted': float((f1 * support).sum() / total) if total else 0.0,
        'f1_macro': float(f1[present].mean()) if present.any() else 0.0,
        'accuracy': float(tp.sum() / total) if total else 0.0,
    }


def day_block_bootstrap(y_true: np.ndarray, y_pred: np.ndarray, days: np.ndarray,
                        n_boot: int = 1000, seed: int = 0) -> Dict[str, tuple]:
    """95% CIs from resampling whole test days with replacement."""
    rng = np.random.default_rng(seed)
    true_labels = list(np.unique(y_true))
    labels = true_labels + sorted(set(np.unique(y_pred)) - set(true_labels))
    code = {l: i for i, l in enumerate(labels)}
    t = np.array([code[l] for l in y_true])
    p = np.array([code[l] for l in y_pred])
    uniq, day_codes = np.unique(days, return_inverse=True)
    L = len(labels)
    day_cm = np.zeros((len(uniq), L, L), dtype=np.int64)
    np.add.at(day_cm, (day_codes, t, p), 1)
    boot = {k: [] for k in ('f1_weighted', 'f1_macro', 'accuracy')}
    for _ in range(n_boot):
        cm = day_cm[rng.integers(0, len(uniq), size=len(uniq))].sum(0)
        for k, v in _scores_from_confusion(cm, len(true_labels)).items():
            boot[k].append(v)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boot.items()}


def evaluate_clustering(pool: pd.DataFrame, test: pd.DataFrame,
                        m_values: Sequence[Union[int, str]] = DEFAULT_M_VALUES,
                        variants: Iterable[str] = VARIANTS,
                        n_boot: int = 1000, seed: int = 0,
                        return_mappings: bool = False):
    """Evaluate one clustering under every (variant, m).

    `pool` and `test` need columns: cluster, conf, label, day. Points with cluster == -1
    (e.g. HDBSCAN noise) are kept in test as unmapped (counted as errors) and never vote.
    Returns a DataFrame with one row per (variant, m).
    """
    rows, mappings = [], {}
    for variant in variants:
        p, t = pool, test
        if variant == 'excl_other':
            p = p[~p['label'].isin(EXCLUDED_LABELS)]
            t = t[~t['label'].isin(EXCLUDED_LABELS)]
        p = p[p['cluster'] != -1]
        y_true = t['label'].to_numpy()
        days = t['day'].to_numpy()
        for m in m_values:
            mapping = map_clusters(p['cluster'].to_numpy(), p['conf'].to_numpy(), p['label'].to_numpy(), m)
            y_pred = np.array([mapping.get(c, UNMAPPED) for c in t['cluster'].to_numpy()])
            ci = day_block_bootstrap(y_true, y_pred, days, n_boot=n_boot, seed=seed)
            row = {'variant': variant, 'm': str(m), 'n_test': len(t), 'n_test_days': len(np.unique(days)),
                   'n_clusters': int(p['cluster'].nunique()),
                   'n_unmapped_test': int((y_pred == UNMAPPED).sum()),
                   'n_labels_mapped': len(set(mapping.values()))}
            for k, v in _scores(y_true, y_pred).items():
                row[k] = v
                row[f'{k}_lo'], row[f'{k}_hi'] = ci[k]
            rows.append(row)
            mappings[(variant, str(m))] = mapping
    out = pd.DataFrame(rows)
    return (out, mappings) if return_mappings else out
