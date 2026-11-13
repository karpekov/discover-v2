"""
Score the final seeded runs: clustering (k-means / SCAN / DEC on encoder and histogram representations), linear
probes on the same representations, and the supervised baselines. One row per (house, level, variant, setup, seed);
`summary.csv` holds mean / std over seeds.

Inputs:
  - assignments `{city}__{rep}_s{seed}[+tod]__{method}_k{K}` from `frozen_deep.py` (rep: tf, bert, hist_sensor,
    hist_sensor_tod),
  - embeddings `{city}_{enc}_s{seed}_embeddings.npz` (probes),
  - baseline predictions `results/baselines/final/{method}/{city}_{level}_{budget}_s{seed}.npz`.
Label budgets count activity labels only (`baselines/data.budget_indices`): clustering at m = 5 uses K x 5 activity
labels, probes and baselines draw random windows until 100 / 250 activity labels are collected.

Usage:
    python -m tist_revision.pipeline.final_scores
"""
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..baselines.data import budget_indices
from .mapping_eval import (ARTIFACTS_DIR, EXCLUDED_LABELS, RESULTS_DIR, _scores, evaluate_clustering, load_protocol,
                           predict_activity, scored_classes)
from .rare_classes import representations

OUT = RESULTS_DIR / 'final'
ASSIGN = re.compile(r'^(?P<city>[a-z]+)__(?P<enc>[a-z_]+?)_s(?P<seed>\d+)(?P<tod>\+tod)?__(?P<method>kmeans|scan|dec)_k(?P<K>\d+)$')
EMB = re.compile(r'^(?P<city>[a-z]+)_(?P<enc>tf|bert)_s(?P<seed>\d+)_embeddings$')


def variants(level, pool_labels, test_labels):
    v = {'all': None, 'masked': scored_classes(pool_labels, test_labels)}
    if level == 'l2':
        v['excl_other'] = sorted(set(test_labels) - set(EXCLUDED_LABELS))
    return v


def score_preds(y_test, pred, level, pool_labels):
    rows = []
    for name, classes in variants(level, pool_labels, y_test).items():
        keep = ~np.isin(y_test, EXCLUDED_LABELS) if name == 'excl_other' else np.ones(len(y_test), bool)
        rows.append({'variant': name} | _scores(y_test[keep], pred[keep], classes))
    return rows


def clustering_rows(city, protocols, m_values, n_boot):
    rows = []
    for path in sorted((ARTIFACTS_DIR / 'assignments').glob(f'{city}__*.npz')):
        mt = ASSIGN.match(path.stem)
        if not mt:
            continue
        z = np.load(path)
        meta = {'family': 'clustering', 'method': mt['method'], 'rep': mt['enc'] + (mt['tod'] or ''),
                'K': int(mt['K']), 'seed': int(mt['seed'])}
        for level, (pool, test) in protocols.items():
            if len(z['pool_cluster']) != len(pool):
                continue
            vs = ('all', 'masked', 'excl_other') if level == 'l2' else ('all', 'masked')
            res = evaluate_clustering(pool[['label', 'day']].assign(cluster=z['pool_cluster'], conf=z['pool_conf']),
                                      test[['label', 'day']].assign(cluster=z['test_cluster'], conf=0.0),
                                      m_values=m_values, variants=vs, n_boot=n_boot, seed=meta['seed'])
            for r in res.to_dict('records'):
                rows.append(meta | {'city': city, 'level': level, 'budget': f'm{r["m"]}',
                                    'n_labels_mapped': r['n_labels_mapped']} | {k: r[k] for k in
                                    ('variant', 'f1_weighted', 'f1_macro', 'accuracy', 'n_classes_scored')})
        print(f'[{city}] {path.stem}', flush=True)
    return rows


def probe_rows(city, fit, protocols, budgets, max_full=20000):
    rows = []
    reps = []
    for path in sorted((ARTIFACTS_DIR / 'embeddings').glob(f'{city}_*_s*_embeddings.npz')):
        mt = EMB.match(path.stem)
        if mt:
            reps += [(f'{mt["enc"]}_s{mt["seed"]}', int(mt['seed'])), (f'{mt["enc"]}_s{mt["seed"]}+tod', int(mt['seed']))]
    seeds = sorted({s for _, s in reps}) or [1]
    reps += [(f'hist_{h}_s{s}', s) for h in ('sensor', 'sensor_tod') for s in seeds]
    any_pool, any_test = next(iter(protocols.values()))
    for (rep, seed), (_, (X_pool, _, X_test)) in zip(reps, representations(city, fit, any_pool, any_test, [r for r, _ in reps])):
        base = re.sub(r'_s\d+', '', rep)
        for level, (pool, test) in protocols.items():
            y_pool, y_test = pool['label'].to_numpy(), test['label'].to_numpy()
            for budget in budgets:
                if budget == 'all':
                    idx = np.random.default_rng(seed).choice(len(y_pool), size=min(max_full, len(y_pool)), replace=False)
                else:
                    tr, va = budget_indices(y_pool, budget, seed)
                    idx = np.concatenate([tr, va])
                scaler = StandardScaler().fit(X_pool[idx])
                clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X_pool[idx]), y_pool[idx])
                pred = predict_activity(clf.predict_proba(scaler.transform(X_test)), clf.classes_)
                for r in score_preds(y_test, pred, level, y_pool):
                    rows.append({'family': 'probe', 'method': 'logreg', 'rep': base, 'K': np.nan, 'seed': seed,
                                 'city': city, 'level': level, 'budget': budget, 'n_labels': len(idx)} | r)
        print(f'[{city}] probe {rep}', flush=True)
    return rows


def baseline_rows(city, protocols):
    rows = []
    for path in sorted((RESULTS_DIR / 'baselines' / 'final').glob(f'*/{city}_*_s*.npz')):
        level, budget, seed = re.match(rf'^{city}_(l[12])_(\w+)_s(\d+)$', path.stem).groups()
        z = np.load(path, allow_pickle=True)
        pool, test = protocols[level]
        assert (z['y_test'] == test['label'].to_numpy()).all(), path
        for r in score_preds(z['y_test'], z['pred'], level, pool['label'].to_numpy()):
            rows.append({'family': 'baseline', 'method': path.parent.name, 'rep': '', 'K': np.nan, 'seed': int(seed),
                         'city': city, 'level': level, 'budget': budget, 'n_labels': int(z['n_labels'])} | r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cities', nargs='+', default=['milan', 'aruba', 'cairo'])
    ap.add_argument('--parts', nargs='+', default=['clustering', 'probes', 'baselines'])
    ap.add_argument('--m', nargs='+', default=['1', '5', 'all'])
    ap.add_argument('--budgets', nargs='+', default=['100', '250', 'all'])
    ap.add_argument('--n_boot', type=int, default=50)
    args = ap.parse_args()
    m_values = [m if m == 'all' else int(m) for m in args.m]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for city in args.cities:
        fit, pool2, test2 = load_protocol(city, level='l2')
        _, pool1, test1 = load_protocol(city, level='l1')
        protocols = {'l1': (pool1, test1), 'l2': (pool2, test2)}
        if 'clustering' in args.parts:
            rows += clustering_rows(city, protocols, m_values, args.n_boot)
        if 'probes' in args.parts:
            rows += probe_rows(city, fit, protocols, args.budgets)
        if 'baselines' in args.parts:
            rows += baseline_rows(city, protocols)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'runs.csv', index=False)
    keys = ['family', 'method', 'rep', 'K', 'city', 'level', 'variant', 'budget']
    summary = df.groupby(keys, dropna=False).agg(
        n_seeds=('seed', 'nunique'), f1_macro=('f1_macro', 'mean'), f1_macro_std=('f1_macro', 'std'),
        f1_weighted=('f1_weighted', 'mean'), f1_weighted_std=('f1_weighted', 'std')).reset_index()
    summary.to_csv(OUT / 'summary.csv', index=False)
    print(f'Saved {OUT} ({len(df)} runs, {len(summary)} summary rows)')


if __name__ == '__main__':
    main()
