"""
Rare-class experiments on frozen representations (k-means clusters fit on FL_20 train; labels from FL_20_p).

All strategies use the same annotation procedure (label the top-m windows of each cluster), so cost is
reported as `n_annotated` (number of labelled windows):
  - cluster_vote:      cluster -> majority label of its top-m windows (the paper's procedure).
  - agreement>=t:      as cluster_vote, but clusters whose top-m vote share < t stay unlabelled
                       (their test windows abstain).
  - window_conf q:     as cluster_vote, but test windows farther from their centroid than the
                       q-quantile of pool distances abstain.
  - propagate_1nn:     each test window takes the label of its nearest annotated window (cosine).
Scores: weighted / macro F1 counting abstentions as errors, coverage, F1 on covered windows, and
the number of classes with any correct prediction.

Usage:
    python -m tist_revision.pipeline.rare_classes --cities milan aruba cairo
"""
import argparse
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

from .histogram_baselines import FEATURE_COLUMNS, build_features
from .mapping_eval import ARTIFACTS_DIR, EXCLUDED_LABELS, RESULTS_DIR, UNMAPPED, load_protocol

OUT = RESULTS_DIR / 'rare_classes'


def representations(city, fit, pool, test, names):
    """`hist_*` histograms, or L2-normalised embeddings; `{emb}+tod` appends the window's
    L2-normalised time-of-day histogram (equal weight to the unit-norm embedding)."""
    for name in names:
        if name.startswith('hist_'):
            yield name, build_features(fit, [fit, pool, test], FEATURE_COLUMNS[name[len('hist_'):]])
        else:
            emb, _, extra = name.partition('+')
            path = ARTIFACTS_DIR / 'embeddings' / f'{city}_{emb}_embeddings.npz'
            if not path.exists():
                print(f'  [skip] {path}')
                continue
            z = np.load(path)
            X = [normalize(z[k].astype(np.float32)) for k in ('fit', 'pool', 'test')]
            if extra == 'tod':
                tod = build_features(fit, [fit, pool, test], ['tods'])
                X = [np.hstack([x, normalize(t)]) for x, t in zip(X, tod)]
            elif extra:
                raise ValueError(name)
            yield name, X


def score(y_true, y_pred):
    covered = y_pred != UNMAPPED
    labels = np.unique(y_true)
    out = {'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
           'f1_macro': f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
           'coverage': covered.mean(),
           'classes_hit': int(sum(((y_true == l) & (y_pred == l)).any() for l in labels)),
           'n_classes': len(labels)}
    if covered.any():
        out['f1_weighted_covered'] = f1_score(y_true[covered], y_pred[covered], average='weighted', zero_division=0)
        out['f1_macro_covered'] = f1_score(y_true[covered], y_pred[covered], labels=np.unique(y_true[covered]),
                                           average='macro', zero_division=0)
    return out


def annotate(pool_c, pool_d, m):
    """Indices of the top-m pool windows (closest to centroid) of every cluster."""
    idx = []
    for c in np.unique(pool_c):
        members = np.flatnonzero(pool_c == c)
        idx.append(members[np.argsort(pool_d[members], kind='stable')[:m]])
    return np.concatenate(idx)


def run(X_fit, X_pool, X_test, y_pool, y_test, K, m_values, seed, max_fit=30000):
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X_fit), size=min(max_fit, len(X_fit)), replace=False)
    km = KMeans(n_clusters=K, n_init=3, random_state=seed).fit(X_fit[sub])
    pc, tc = km.predict(X_pool), km.predict(X_test)
    pd_ = np.linalg.norm(X_pool - km.cluster_centers_[pc], axis=1)
    td = np.linalg.norm(X_test - km.cluster_centers_[tc], axis=1)
    rows = []
    for m in m_values:
        ann = annotate(pc, pd_, m)
        votes = {}
        for c in np.unique(pc[ann]):
            labs = y_pool[ann][pc[ann] == c]
            (lab, n), = Counter(labs).most_common(1)
            votes[c] = (lab, n / len(labs))
        base = {'K': K, 'm': m, 'n_annotated': len(ann),
                'annotated_classes': len(np.unique(y_pool[ann]))}

        pred = np.array([votes[c][0] if c in votes else UNMAPPED for c in tc])
        rows.append(base | {'strategy': 'cluster_vote'} | score(y_test, pred))
        for t in (0.6, 0.8, 1.0):
            pred_t = np.array([votes[c][0] if c in votes and votes[c][1] >= t else UNMAPPED for c in tc])
            rows.append(base | {'strategy': f'agreement>={t}'} | score(y_test, pred_t))
        for q in (0.8, 0.6):
            pred_q = np.where(td <= np.quantile(pd_, q), pred, UNMAPPED)
            rows.append(base | {'strategy': f'window_conf q={q}'} | score(y_test, pred_q))
        A = normalize(X_pool[ann])
        nn = (normalize(X_test) @ A.T).argmax(1)
        rows.append(base | {'strategy': 'propagate_1nn'} | score(y_test, y_pool[ann][nn]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cities', nargs='+', default=['milan', 'aruba', 'cairo'])
    ap.add_argument('--representations', nargs='+',
                    default=['hist_sensor', 'hist_sensor_tod', 'mlm', 'mlm_time'])
    ap.add_argument('--K', type=int, nargs='+', default=[20, 50, 100])
    ap.add_argument('--m', type=int, nargs='+', default=[1, 3, 5])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for city in args.cities:
        fit, pool, test = load_protocol(city)
        for name, (X_fit, X_pool, X_test) in representations(city, fit, pool, test, args.representations):
            for variant in ('all', 'excl_other'):
                pm = np.ones(len(pool), bool) if variant == 'all' else ~pool['label'].isin(EXCLUDED_LABELS).to_numpy()
                tm = np.ones(len(test), bool) if variant == 'all' else ~test['label'].isin(EXCLUDED_LABELS).to_numpy()
                for K in args.K:
                    t0 = time.time()
                    r = run(X_fit, X_pool[pm], X_test[tm], pool['label'].to_numpy()[pm], test['label'].to_numpy()[tm],
                            K, args.m, args.seed)
                    rows += [x | {'city': city, 'representation': name, 'variant': variant} for x in r]
                    print(f'[{city}] {name} {variant} K={K} ({time.time() - t0:.0f}s)')
            pd.DataFrame(rows).to_csv(OUT / 'results.csv', index=False)
    print(f'Saved {OUT / "results.csv"}')


if __name__ == '__main__':
    main()
