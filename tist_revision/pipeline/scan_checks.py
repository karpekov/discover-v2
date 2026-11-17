"""
Score the SCAN follow-up checks (L1, m=5 and m=all, 5 seeds; cluster labels from `map_clusters`):
  1. Time-of-day weight for SCAN on TF+ToD (`{city}__tf_s{s}+tod__scan_k{K}_todw{w}`), from
     `frozen_deep --tod_weight w --suffix _todw{w}`; w=`bal` is `--tod_weight balanced --suffix _todwbal`,
     w=`none` the unweighted final runs (no suffix).
  2. SCAN vs k-means on Sensor+ToD histograms vs TF+ToD embeddings, per class (order-sensitive classes).

Writes `results/temp/scan_checks/{todw_summary,todw_per_class,scan_hist_per_class}.csv`.

Usage:
    python -m tist_revision.pipeline.scan_checks
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .mapping_eval import ARTIFACTS_DIR, TEMP_DIR, load_protocol, map_clusters, scored_classes

CITIES = ['milan', 'aruba', 'cairo']
SEEDS = [1, 2, 3, 4, 5]
OUT = TEMP_DIR / 'scan_checks'


def score_runs(city, names, m_values=(5, 'all')):
    """Per (name, m): weighted F1, masked macro F1 and per-class F1 (all test classes)."""
    _, pool, test = load_protocol(city, level='l1')
    yp, yt = pool['label'].to_numpy(), test['label'].to_numpy()
    masked, classes = scored_classes(yp, yt), sorted(set(yt))
    summary, per_class = [], []
    for key, path in names:
        if not Path(path).exists():
            continue
        z = np.load(path)
        for m in m_values:
            mapping = map_clusters(z['pool_cluster'], z['pool_conf'], yp, m)
            pred = np.array([mapping.get(c, '(unmapped)') for c in z['test_cluster']])
            summary.append({**key, 'city': city, 'm': str(m),
                            'f1_weighted': f1_score(yt, pred, average='weighted', zero_division=0),
                            'f1_macro': f1_score(yt, pred, labels=masked, average='macro', zero_division=0)})
            for c, v in zip(classes, f1_score(yt, pred, labels=classes, average=None, zero_division=0)):
                per_class.append({**key, 'city': city, 'm': str(m), 'label': c, 'f1': v})
    return summary, per_class


def assignment(city, rep, method, K, suffix=''):
    return ARTIFACTS_DIR / 'assignments' / f'{city}__{rep}__{method}_k{K}{suffix}.npz'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 250)

    rows, pcs = [], []
    for city in CITIES:
        names = [({'K': K, 'w': w, 'seed': s}, assignment(city, f'tf_s{s}+tod', 'scan', K, f'_todw{w}'))
                 for K in (20, 50, 100) for w in ('1', '2', '4', '8', 'bal') for s in SEEDS]
        names += [({'K': K, 'w': 'kmeans', 'seed': s}, assignment(city, f'tf_s{s}+tod', 'kmeans', K))
                  for K in (20, 50, 100) for s in SEEDS]
        names += [({'K': K, 'w': 'none', 'seed': s}, assignment(city, f'tf_s{s}+tod', 'scan', K))
                  for K in (20, 50) for s in SEEDS]
        r, p = score_runs(city, names)
        rows += r
        pcs += p
    todw = pd.DataFrame(rows)
    todw.to_csv(OUT / 'todw_summary.csv', index=False)
    pd.DataFrame(pcs).to_csv(OUT / 'todw_per_class.csv', index=False)
    agg = todw.groupby(['city', 'K', 'm', 'w'])[['f1_weighted', 'f1_macro']].agg(['mean', 'std']).round(3)
    agg['n'] = todw.groupby(['city', 'K', 'm', 'w']).seed.nunique()
    print('=== SCAN TF+ToD, ToD weight w (vs k-means), L1 masked macro / weighted\n', agg.to_string())

    rows, pcs = [], []
    for city in CITIES:
        names = [({'K': K, 'setup': f'{meth} {tag}', 'seed': s}, assignment(city, rep.format(s=s), meth, K))
                 for K in (20, 50) for s in SEEDS
                 for meth in ('kmeans', 'scan')
                 for tag, rep in (('hist', 'hist_sensor_tod_s{s}'), ('tf', 'tf_s{s}+tod'))]
        r, p = score_runs(city, names)
        rows += r
        pcs += p
    hist = pd.DataFrame(rows)
    hist.to_csv(OUT / 'scan_hist_summary.csv', index=False)
    pc = pd.DataFrame(pcs)
    pc.to_csv(OUT / 'scan_hist_per_class.csv', index=False)
    print('\n=== k-means vs SCAN on hist / TF (+ToD), L1\n',
          hist.groupby(['city', 'K', 'm', 'setup'])[['f1_weighted', 'f1_macro']].mean().unstack('setup').round(2)
          .to_string())
    print('\n=== per-class F1, m=5\n',
          pc[pc.m == '5'].pivot_table(index=['city', 'label'], columns=['K', 'setup'], values='f1').round(2).to_string())


if __name__ == '__main__':
    main()
