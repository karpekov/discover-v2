"""
Re-score clusterings at both label levels (l2 = TDOST classes, l1 = raw CASAS activities).

Sources (no retraining):
  - k-means (K = 20 / 50) refit on frozen representations (histograms, MLM embeddings, embeddings +
    time-of-day histogram), confidence = -distance to centroid; saved as `{city}__{rep}__kmeans_k{K}`.
  - frozen SCAN / DEC assignments from `frozen_deep.py`.
Pool = continuous fit windows (see mapping_eval.load_protocol); assignments made for another pool are skipped.
Variants: 'all', 'masked' (F1 only over classes with >= MIN_POOL pool / >= MIN_TEST test windows),
and for l2 also 'excl_other'.

Usage:
    python -m tist_revision.pipeline.label_levels
"""
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .mapping_eval import (ARTIFACTS_DIR, MIN_POOL, MIN_TEST, RESULTS_DIR, evaluate_clustering, load_protocol,
                           save_assignments, scored_classes)
from .rare_classes import representations

OUT = RESULTS_DIR / 'label_levels'
FROZEN = re.compile(r'^(?P<city>[a-z]+)__(?P<rep>[\w+]+)__(?P<method>scan|dec)_k(?P<K>\d+)$')


def kmeans_assignments(city, fit, pool, test, reps, Ks, seed, max_fit=30000):
    for rep, (X_fit, X_pool, X_test) in representations(city, fit, pool, test, reps):
        sub = np.random.default_rng(seed).choice(len(X_fit), size=min(max_fit, len(X_fit)), replace=False)
        for K in Ks:
            name = f'{city}__{rep}__kmeans_k{K}'
            km = KMeans(n_clusters=K, n_init=3, random_state=seed).fit(X_fit[sub])
            pc, tc = km.predict(X_pool), km.predict(X_test)
            save_assignments(name, pc, -np.linalg.norm(X_pool - km.cluster_centers_[pc], axis=1), tc)
            yield {'method': 'kmeans', 'encoder': rep, 'run': f'k{K}', 'K': K}, np.load(ARTIFACTS_DIR / 'assignments' / f'{name}.npz')


def saved_assignments(city, n_pool, reps, suffix=''):
    """Frozen SCAN / DEC assignments from `frozen_deep.py` (`{city}__{rep}__{scan|dec}_k{K}{suffix}`)."""
    for path in sorted((ARTIFACTS_DIR / 'assignments').glob(f'{city}__*__*_k*{suffix}.npz')):
        mt = FROZEN.match(path.stem[:len(path.stem) - len(suffix)] if suffix else path.stem)
        if not mt or mt['rep'] not in reps:
            continue
        z = np.load(path)
        if len(z['pool_cluster']) != n_pool:
            continue
        yield {'method': mt['method'], 'encoder': mt['rep'], 'run': f'k{mt["K"]}', 'K': int(mt['K'])}, z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cities', nargs='+', default=['milan', 'aruba', 'cairo'])
    ap.add_argument('--representations', nargs='+',
                    default=['hist_sensor', 'hist_sensor_tod', 'mlm', 'mlm_time', 'mlm+tod', 'mlm_time+tod'])
    ap.add_argument('--K', type=int, nargs='+', default=[20, 50])
    ap.add_argument('--m', nargs='+', default=['1', '3', '5', '10', 'all'])
    ap.add_argument('--n_boot', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    m_values = [m if m == 'all' else int(m) for m in args.m]
    OUT.mkdir(parents=True, exist_ok=True)
    rows, classes = [], []
    for city in args.cities:
        fit, pool2, test2 = load_protocol(city, level='l2')
        _, pool1, test1 = load_protocol(city, level='l1')
        for level, pool, test in (('l2', pool2, test2), ('l1', pool1, test1)):
            kept = set(scored_classes(pool['label'], test['label']))
            pc, tc = pool['label'].value_counts(), test['label'].value_counts()
            classes += [{'city': city, 'level': level, 'label': l, 'n_pool': int(pc.get(l, 0)), 'n_test': int(n),
                         'scored_in_masked': l in kept} for l, n in tc.items()]
        sources = list(kmeans_assignments(city, fit, pool2, test2, args.representations, args.K, args.seed))
        sources += list(saved_assignments(city, len(pool2), args.representations))
        for meta, z in sources:
            for level, pool, test in (('l2', pool2, test2), ('l1', pool1, test1)):
                variants = ('all', 'masked', 'excl_other') if level == 'l2' else ('all', 'masked')
                res = evaluate_clustering(pool[['label', 'day']].assign(cluster=z['pool_cluster'], conf=z['pool_conf']),
                                          test[['label', 'day']].assign(cluster=z['test_cluster'], conf=0.0),
                                          m_values=m_values, variants=variants, n_boot=args.n_boot, seed=args.seed)
                rows += [r | meta | {'city': city, 'level': level} for r in res.to_dict('records')]
            print(f'[{city}] {meta["method"]} {meta["encoder"]} {meta["run"]}', flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'results.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT / 'class_counts.csv', index=False)
    print(f'Saved {OUT} (masked: >= {MIN_POOL} pool and >= {MIN_TEST} test windows)')


if __name__ == '__main__':
    main()
