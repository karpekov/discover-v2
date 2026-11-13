"""
Cluster → label mapping evaluator with a limited annotation budget.

Protocol (shared by all clustering methods in the DISCOVER revision):
  0. Clusters are fit on continuous train windows (FL_20 train). No pre-segmented data is used for
     training or annotation.
  1. The annotation pool is the same continuous windows, with the confidence of their cluster assignment.
     A window's label is its last event's activity; No_Activity is a label like any other.
  2. For each cluster, the top-m pool windows by confidence are "annotated" and the cluster is mapped to
     their majority label (m = 'all' uses every pool window of the cluster). No_Activity windows are skipped
     (the annotator moves to the next window), so No_Activity is never predicted unless a cluster holds
     nothing else. Supervised methods likewise never predict No_Activity on test (as in the paper, whose
     test set has no No_Activity).
  3. Pre-segmented test windows (FL_20_p test: unseen days, one activity per window, no No_Activity) get
     their cluster's mapped label and are scored against their own label. CIs come from a block bootstrap
     over test days.

Labels: TDOST / DeepCASAS merged classes (see labels.py) of the last event's raw activity.

Scoring variants:
  - 'all':        every labelled window (pre-segmented data has no No_Activity). Comparable to
                  TDOST / DeepCASAS, which keep Other as a class.
  - 'excl_other': windows labelled Other (or No_Activity) removed from both pool and test.
  - 'masked':     every window stays in pool and test (rare-class windows can still be mispredicted),
                  but F1 is averaged only over classes with >= MIN_POOL pool and >= MIN_TEST test windows.
Label level: 'l2' (TDOST classes, default) or 'l1' (raw CASAS activity with the resident id removed, which
only affects Cairo) via `load_protocol(level=...)`.
"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .labels import tdost_label

NO_ACTIVITY = 'No_Activity'
EXCLUDED_LABELS = ('No_Activity', 'no_activity', 'Other')
UNMAPPED = '__unmapped__'
DEFAULT_M_VALUES = (1, 3, 5, 10, 50, 'all')
VARIANTS = ('all', 'excl_other')
MIN_POOL, MIN_TEST = 50, 20
DATA_ROOT = Path('data/processed/casas')
PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / 'results'
ARTIFACTS_DIR = PROJECT_DIR / 'artifacts'
CACHE_DIR = ARTIFACTS_DIR / 'cache'


def load_windows(data_dir: Union[str, Path], split: str, cache_dir: Optional[Union[str, Path]] = None,
                 labelled: bool = True) -> pd.DataFrame:
    """Load one split of a processed CASAS dataset as a compact per-window DataFrame.

    Columns: sample_id, day, label (TDOST class of the last event; raw l1 if not `labelled`),
    label_raw, sensors, rooms, tods.
    """
    data_dir = Path(data_dir)
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{data_dir.parent.name}_{data_dir.name}_{split}{'' if labelled else '_raw'}_v4.pkl"
        if cache_path.exists():
            return pd.read_pickle(cache_path)

    with open(data_dir / f'{split}.json') as f:
        samples = json.load(f)['samples']

    rows = []
    for s in samples:
        seq = s['sensor_sequence']
        raw = seq[-1]['activity_l1']
        rows.append({
            'sample_id': s['sample_id'],
            'day': s['metadata']['start_time'][:10],
            'label': tdost_label(raw) if labelled else raw,
            'label_raw': raw,
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


def load_protocol(city: str, fit_window: str = 'FL_20', test_window: str = 'FL_20_p', level: str = 'l2'):
    """Return (fit, pool, test). No pre-segmented data is used for training or annotation: the annotation pool
    is the continuous fit windows themselves (same rows and order as `fit`), labelled by their last event
    (No_Activity kept as a label an annotator would give). Only the test windows are pre-segmented."""
    pool = load_windows(DATA_ROOT / city / fit_window, 'train', CACHE_DIR)
    fit = pool
    test = load_windows(DATA_ROOT / city / test_window, 'test', CACHE_DIR)
    if level == 'l1':
        pool, test = pool.assign(label=pool['label_raw'].map(l1_label)), test.assign(label=test['label_raw'].map(l1_label))
    elif level != 'l2':
        raise ValueError(level)
    return fit, pool, test


def l1_label(raw: str) -> str:
    """Raw CASAS activity without the resident id (Cairo 'R1 sleep' / 'R2 sleep' -> 'sleep'): which of two
    residents acts is not observable from ambient sensors."""
    return 'No_Activity' if raw.lower() == 'no_activity' else re.sub(r'^R[12][ _]', '', raw)


def scored_classes(pool_labels, test_labels, min_pool: int = MIN_POOL, min_test: int = MIN_TEST) -> list:
    """Classes kept in the 'masked' variant."""
    pc, tc = Counter(pool_labels), Counter(test_labels)
    return sorted(l for l in tc if pc[l] >= min_pool and tc[l] >= min_test)


def save_assignments(name: str, pool_cluster, pool_conf, test_cluster, test_conf=None):
    """Store cluster assignments of the labelled pool/test windows for later plotting."""
    path = ARTIFACTS_DIR / 'assignments' / f'{name}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pool_cluster=np.asarray(pool_cluster), pool_conf=np.asarray(pool_conf),
                        test_cluster=np.asarray(test_cluster),
                        test_conf=np.zeros(len(test_cluster)) if test_conf is None else np.asarray(test_conf))


def map_clusters(pool_clusters: np.ndarray, pool_conf: np.ndarray, pool_labels: np.ndarray,
                 m: Union[int, str], skip: Sequence[str] = (NO_ACTIVITY,)) -> Dict[int, str]:
    """Map each cluster to the majority label of its top-m most confident pool windows.

    Windows labelled in `skip` are passed over (the annotator moves on to the next window), so a cluster is
    only mapped to No_Activity if it contains nothing else."""
    mapping = {}
    for c in np.unique(pool_clusters):
        idx = np.flatnonzero(pool_clusters == c)
        real = idx[~np.isin(pool_labels[idx], skip)]
        if len(real):
            idx = real
        if m != 'all':
            idx = idx[np.argsort(-pool_conf[idx], kind='stable')[:int(m)]]
        counts = Counter(pool_labels[idx])
        # Ties broken by label name for determinism.
        mapping[c] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return mapping


def predict_activity(proba: np.ndarray, classes: Sequence[str], skip: Sequence[str] = (NO_ACTIVITY,)) -> np.ndarray:
    """Argmax over class probabilities, never predicting a class in `skip` (unless it is the only class)."""
    classes = np.asarray(classes)
    allowed = ~np.isin(classes, skip)
    if not allowed.any():
        return classes[proba.argmax(1)]
    return classes[allowed][proba[:, allowed].argmax(1)]


def _scores(y_true: np.ndarray, y_pred: np.ndarray, classes: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """F1 averaged over `classes` (default: every true class); accuracy over all windows."""
    classes = np.unique(y_true) if classes is None else list(classes)
    return {
        'f1_weighted': f1_score(y_true, y_pred, labels=classes, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, labels=classes, average='macro', zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
    }


def _scores_from_confusion(cm: np.ndarray, keep: np.ndarray) -> Dict[str, float]:
    """Same metrics as `_scores`, from a full confusion matrix; F1 averaged over class indices `keep`."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    pred_count = cm.sum(0).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.where(support + pred_count > 0, 2 * tp / (support + pred_count), 0.0)
    f1, s = f1[keep], support[keep]
    present = s > 0
    total = cm.sum()
    return {
        'f1_weighted': float((f1 * s).sum() / s.sum()) if s.sum() else 0.0,
        'f1_macro': float(f1[present].mean()) if present.any() else 0.0,
        'accuracy': float(tp.sum() / total) if total else 0.0,
    }


def day_block_bootstrap(y_true: np.ndarray, y_pred: np.ndarray, days: np.ndarray,
                        n_boot: int = 1000, seed: int = 0,
                        classes: Optional[Sequence[str]] = None) -> Dict[str, tuple]:
    """95% CIs from resampling whole test days with replacement."""
    rng = np.random.default_rng(seed)
    true_labels = list(np.unique(y_true))
    labels = true_labels + sorted(set(np.unique(y_pred)) - set(true_labels))
    code = {l: i for i, l in enumerate(labels)}
    keep = np.array([code[l] for l in (true_labels if classes is None else classes)])
    t = np.array([code[l] for l in y_true])
    p = np.array([code[l] for l in y_pred])
    uniq, day_codes = np.unique(days, return_inverse=True)
    L = len(labels)
    day_cm = np.zeros((len(uniq), L, L), dtype=np.int64)
    np.add.at(day_cm, (day_codes, t, p), 1)
    boot = {k: [] for k in ('f1_weighted', 'f1_macro', 'accuracy')}
    for _ in range(n_boot):
        cm = day_cm[rng.integers(0, len(uniq), size=len(uniq))].sum(0)
        for k, v in _scores_from_confusion(cm, keep).items():
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
        classes = None
        if variant == 'excl_other':
            p = p[~p['label'].isin(EXCLUDED_LABELS)]
            t = t[~t['label'].isin(EXCLUDED_LABELS)]
        elif variant == 'masked':
            classes = scored_classes(p['label'], t['label'])
        p = p[p['cluster'] != -1]
        y_true = t['label'].to_numpy()
        days = t['day'].to_numpy()
        for m in m_values:
            mapping = map_clusters(p['cluster'].to_numpy(), p['conf'].to_numpy(), p['label'].to_numpy(), m)
            y_pred = np.array([mapping.get(c, UNMAPPED) for c in t['cluster'].to_numpy()])
            ci = day_block_bootstrap(y_true, y_pred, days, n_boot=n_boot, seed=seed, classes=classes)
            row = {'variant': variant, 'm': str(m), 'n_test': len(t), 'n_test_days': len(np.unique(days)),
                   'n_classes_scored': len(np.unique(y_true)) if classes is None else len(classes),
                   'n_clusters': int(p['cluster'].nunique()),
                   'n_unmapped_test': int((y_pred == UNMAPPED).sum()),
                   'n_labels_mapped': len(set(mapping.values()))}
            for k, v in _scores(y_true, y_pred, classes).items():
                row[k] = v
                row[f'{k}_lo'], row[f'{k}_hi'] = ci[k]
            rows.append(row)
            mappings[(variant, str(m))] = mapping
    out = pd.DataFrame(rows)
    return (out, mappings) if return_mappings else out
