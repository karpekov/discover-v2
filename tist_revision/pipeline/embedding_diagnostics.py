"""
Stage 1.5 for the DISCOVER revision: is an embedding space worth running SCAN on?

Protocol (see mapping_eval): clusters are fit on continuous FL_20 train windows; every
labelled quantity uses pre-segmented FL_20_p windows (one true label per window).

For every representation (histogram features and frozen encoder embeddings) computes:
  (i)   kNN label purity at k=20 among pre-segmented train windows,
  (ii)  k-means K=20 fit on FL_20 train, scored with the top-m mapping evaluator,
  (iii) a linear probe trained on n labelled windows per class (separability ceiling).
Existing SCAN checkpoints are also re-scored (cluster head argmax, confidence = max prob).

Encoders receive inputs exactly as in MLM pre-training (see encoding.py).

Usage:
    python -m tist_revision.pipeline.embedding_diagnostics --cities milan aruba cairo
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .encoding import encode, encoder_fields, get_device, load_model, load_vocab, tensorize
from .histogram_baselines import FEATURE_COLUMNS, build_features
from .mapping_eval import (ARTIFACTS_DIR, DATA_ROOT, EXCLUDED_LABELS, RESULTS_DIR, VARIANTS, evaluate_clustering,
                           load_protocol, save_assignments)

MODEL_ROOT = Path('trained_models')
OUT_ROOT = RESULTS_DIR / 'stage1_5_embeddings'
EMB_DIR = ARTIFACTS_DIR / 'embeddings'
M_VALUES = (5, 'all')


def split_paths(city):
    return {'fit': DATA_ROOT / city / 'FL_20' / 'train.json',
            'pool': DATA_ROOT / city / 'FL_20_p' / 'train.json',
            'test': DATA_ROOT / city / 'FL_20_p' / 'test.json'}


def encode_splits(model, city, device):
    """{'fit'|'pool'|'test': (embeddings, probabilities)} for one city."""
    vocab = load_vocab(DATA_ROOT / city / 'FL_20' / 'vocab.json')
    cat_fields, global_fields = encoder_fields(model, vocab)
    return {k: encode(model, tensorize(p, vocab, cat_fields, global_fields), device)
            for k, p in split_paths(city).items()}


def load_encoders(city: str, device, extra_dirs=(), only_extra=False):
    """Yield (name, model) for extra encoder dirs, the frozen MLM encoder and the SCAN-fine-tuned model."""
    mlm_dir = MODEL_ROOT / city / f'{city}_fl20_seq_discover_v1_mlm_only'
    scan_ckpt = MODEL_ROOT / city / 'scan_fl20_20cl_discover_v1' / 'best_model.pt'
    vocab = DATA_ROOT / city / 'FL_20' / 'vocab.json'
    for d in extra_dirs:
        d = Path(d)
        if d.exists():
            yield d.name, load_model(d, vocab, device)
        else:
            print(f'  [skip] no encoder at {d}')
    if only_extra:
        return
    if not mlm_dir.exists():
        print(f'  [skip] no MLM encoder at {mlm_dir}')
        return
    # Clustering head is random for 'mlm'; only its embeddings are used.
    yield 'mlm', load_model(mlm_dir, vocab, device)
    if scan_ckpt.exists():
        yield 'scan_enc', load_model(mlm_dir, vocab, device, scan_ckpt=scan_ckpt)
    else:
        print(f'  [skip] no SCAN checkpoint at {scan_ckpt}')


def knn_purity(X, labels, k=20, max_n=20000, seed=0, metric='cosine'):
    """Mean share of each anchor's k nearest neighbours with the same label
    (over all anchors, and over anchors whose label is not Other)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(max_n, len(X)), replace=False)
    Xs, ys = X[idx], labels[idx]
    nbr = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(Xs).kneighbors(Xs, return_distance=False)[:, 1:]
    same = (ys[nbr] == ys[:, None]).mean(1)
    keep = ~np.isin(ys, EXCLUDED_LABELS)
    return float(same.mean()), float(same[keep].mean())


def linear_probe(X_pool, y_pool, X_test, y_test, n_per_class, seeds=(0, 1, 2), max_full=20000):
    """Logistic regression on n labelled pool windows per class ('full' = up to max_full); scored on test."""
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if n_per_class == 'full':
            idx = rng.choice(len(X_pool), size=min(max_full, len(X_pool)), replace=False)
        else:
            idx = np.concatenate([rng.choice(np.flatnonzero(y_pool == c), size=min(n_per_class, (y_pool == c).sum()),
                                             replace=False) for c in np.unique(y_pool)])
        scaler = StandardScaler().fit(X_pool[idx])
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X_pool[idx]), y_pool[idx])
        pred = clf.predict(scaler.transform(X_test))
        out.append((f1_score(y_test, pred, average='weighted', zero_division=0),
                    f1_score(y_test, pred, labels=np.unique(y_test), average='macro', zero_division=0)))
    return np.mean(out, axis=0)


def diagnose(name, X_fit, X_pool, X_test, pool, test, rows, city, metric, num_clusters=20, k_neighbors=20,
             probe_n=('5', '20', 'full'), n_boot=500, seed=42, m_values=M_VALUES):
    """kNN purity, k-means through the evaluator, and linear probes for one representation (appends to rows)."""
    t0 = time.time()
    y_pool, y_test = pool['label'].to_numpy(), test['label'].to_numpy()

    p_all, p_excl = knn_purity(X_pool, y_pool, k=k_neighbors, seed=seed, metric=metric)
    rows['knn'].append({'city': city, 'representation': name, 'purity_all': p_all, 'purity_excl_other': p_excl})

    km = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed).fit(X_fit)
    pc, tc = km.predict(X_pool), km.predict(X_test)
    pconf = -np.linalg.norm(X_pool - km.cluster_centers_[pc], axis=1)
    save_assignments(f'{city}__{name}__kmeans_k{num_clusters}', pc, pconf, tc)
    res = evaluate_clustering(pool.assign(cluster=pc, conf=pconf), test.assign(cluster=tc, conf=0.0),
                              m_values=m_values, n_boot=n_boot, seed=seed)
    rows['clustering'].append(res.assign(city=city, representation=name, method='kmeans'))

    for variant in VARIANTS:
        pm = np.ones(len(y_pool), bool) if variant == 'all' else ~np.isin(y_pool, EXCLUDED_LABELS)
        tm = np.ones(len(y_test), bool) if variant == 'all' else ~np.isin(y_test, EXCLUDED_LABELS)
        for n in probe_n:
            w, m = linear_probe(X_pool[pm], y_pool[pm], X_test[tm], y_test[tm], 'full' if n == 'full' else int(n))
            rows['probe'].append({'city': city, 'representation': name, 'variant': variant, 'n_per_class': str(n),
                                  'f1_weighted': w, 'f1_macro': m})
    r = res[(res.variant == 'excl_other') & (res.m == '5')].iloc[0]
    print(f'  {name:28s} purity all={p_all:.3f} excl={p_excl:.3f} | kmeans excl_other m=5 '
          f'w/m={r.f1_weighted:.3f}/{r.f1_macro:.3f} ({time.time() - t0:.0f}s)')


def save_rows(rows, out_dir: Path, append: bool = False):
    """Write knn_purity / clustering / linear_probe CSVs; with `append`, replace only matching (city, representation)."""
    frames = {'knn_purity': pd.DataFrame(rows['knn']),
              'clustering': pd.concat(rows['clustering']) if rows['clustering'] else pd.DataFrame(),
              'linear_probe': pd.DataFrame(rows['probe'])}
    for name, df in frames.items():
        path = out_dir / f'{name}.csv'
        if append and path.exists() and len(df):
            old = pd.read_csv(path)
            keys = set(zip(df['city'], df['representation']))
            old = old[[k not in keys for k in zip(old['city'], old['representation'])]]
            df = pd.concat([old, df])
        if len(df):
            df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cities', nargs='+', default=['milan', 'aruba', 'cairo'])
    ap.add_argument('--num_clusters', type=int, default=20)
    ap.add_argument('--k_neighbors', type=int, default=20)
    ap.add_argument('--probe_n', nargs='+', default=['5', '20', 'full'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_boot', type=int, default=500)
    ap.add_argument('--no_encoders', action='store_true')
    ap.add_argument('--no_histograms', action='store_true')
    ap.add_argument('--extra_encoders', nargs='*', default=[],
                    help='Extra encoder dirs (same FL_20 vocab); use {city} as placeholder')
    ap.add_argument('--only_extra', action='store_true', help='Skip the default MLM/SCAN encoders')
    ap.add_argument('--append', action='store_true', help='Merge into existing CSVs instead of overwriting')
    ap.add_argument('--output_dir', default=str(OUT_ROOT))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    rows = {'knn': [], 'clustering': [], 'probe': []}
    kw = dict(num_clusters=args.num_clusters, k_neighbors=args.k_neighbors, probe_n=args.probe_n,
              n_boot=args.n_boot, seed=args.seed)

    for city in args.cities:
        fit, pool, test = load_protocol(city)
        print(f'[{city}] fit={len(fit):,} labelled pool={len(pool):,} labelled test={len(test):,} device={device}')
        cols = ['label', 'day']

        if not args.no_histograms:
            for feat, columns in FEATURE_COLUMNS.items():
                X_fit, X_pool, X_test = build_features(fit, [fit, pool, test], columns)
                diagnose(f'hist_{feat}', X_fit, X_pool, X_test, pool[cols], test[cols], rows, city,
                         metric='euclidean', **kw)
            save_rows(rows, out_dir, args.append)

        if args.no_encoders:
            continue
        extra = [e.format(city=city) for e in args.extra_encoders]
        for enc_name, model in load_encoders(city, device, extra, args.only_extra):
            t0 = time.time()
            enc = encode_splits(model, city, device)
            assert len(enc['fit'][0]) == len(fit) and len(enc['pool'][0]) == len(pool) and len(enc['test'][0]) == len(test)
            print(f'  encoded {enc_name} ({time.time() - t0:.0f}s)')
            diagnose(enc_name, enc['fit'][0], enc['pool'][0], enc['test'][0], pool[cols], test[cols], rows, city,
                     metric='cosine', **kw)
            if enc_name == 'scan_enc':
                (_, P_pool), (_, P_test) = enc['pool'], enc['test']
                save_assignments(f'{city}__scan_enc__scan_head', P_pool.argmax(1), P_pool.max(1),
                                 P_test.argmax(1), P_test.max(1))
                res = evaluate_clustering(pool[cols].assign(cluster=P_pool.argmax(1), conf=P_pool.max(1)),
                                          test[cols].assign(cluster=P_test.argmax(1), conf=P_test.max(1)),
                                          m_values=M_VALUES, n_boot=args.n_boot, seed=args.seed)
                rows['clustering'].append(res.assign(city=city, representation='scan_enc', method='scan_head'))
            EMB_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(EMB_DIR / f'{city}_{enc_name}_embeddings.npz',
                                **{k: v[0].astype(np.float16) for k, v in enc.items()})
            del model, enc
            save_rows(rows, out_dir, args.append)

    save_rows(rows, out_dir, args.append)
    print(f'Saved results to {out_dir}')


if __name__ == '__main__':
    main()
