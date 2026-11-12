"""
Stage 1 baselines for the DISCOVER revision (R2-2.9a): cluster per-window sensor / room
histograms with k-means and HDBSCAN, then score with the shared top-m mapping evaluator
(clusters fit on continuous FL_20 train windows; mapping and scoring on pre-segmented FL_20_p).

Usage:
    python -m src.evals.dv1_review.histogram_baselines --cities milan aruba cairo
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.neighbors import KNeighborsClassifier

from .mapping_eval import DEFAULT_M_VALUES, evaluate_clustering, load_protocol

OUT_ROOT = Path('results/evals/dv1_review/stage1_histograms')
FEATURE_COLUMNS = {'sensor': ['sensors'], 'room': ['rooms'], 'sensor_tod': ['sensors', 'tods']}


def histogram_features(df: pd.DataFrame, column: str, vocab: list) -> np.ndarray:
    """Row-normalised counts of each vocabulary item in a window."""
    index = {v: i for i, v in enumerate(vocab)}
    X = np.zeros((len(df), len(vocab)), dtype=np.float32)
    for r, items in enumerate(df[column]):
        for it in items:
            j = index.get(it)
            if j is not None:
                X[r, j] += 1
    X /= np.maximum(X.sum(1, keepdims=True), 1)
    return X


def build_features(fit, frames, columns):
    """Concatenated histograms for each frame, with vocabularies taken from `fit`."""
    out = [[] for _ in frames]
    for col in columns:
        vocab = sorted({v for items in fit[col] for v in items})
        for i, df in enumerate(frames):
            out[i].append(histogram_features(df, col, vocab))
    return [np.hstack(parts) for parts in out]


def run_kmeans(X_fit, X_pool, X_test, k, seed):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X_fit)
    pool_c, test_c = km.predict(X_pool), km.predict(X_test)
    # Confidence = closeness to the assigned centroid.
    pool_conf = -np.linalg.norm(X_pool - km.cluster_centers_[pool_c], axis=1)
    return pool_c, pool_conf, test_c


def run_hdbscan(X_fit, X_pool, X_test, min_cluster_size, seed, max_fit=20000):
    """Fit HDBSCAN on a subsample of the fit windows; assign labelled windows to clusters by a
    kNN vote over clustered fit points (confidence = vote share)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_fit), size=min(max_fit, len(X_fit)), replace=False)
    labels = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit(X_fit[idx]).labels_
    clustered = labels != -1
    if clustered.sum() == 0:
        return np.full(len(X_pool), -1), np.zeros(len(X_pool)), np.full(len(X_test), -1)
    knn = KNeighborsClassifier(n_neighbors=10).fit(X_fit[idx][clustered], labels[clustered])
    proba = knn.predict_proba(X_pool)
    return knn.classes_[proba.argmax(1)], proba.max(1), knn.predict(X_test)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cities', nargs='+', default=['milan', 'aruba', 'cairo'])
    ap.add_argument('--features', nargs='+', default=['sensor', 'room', 'sensor_tod'])
    ap.add_argument('--num_clusters', type=int, default=20)
    ap.add_argument('--hdbscan_min_cluster_sizes', type=int, nargs='+', default=[10, 20, 50, 100, 200])
    ap.add_argument('--no_hdbscan', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--output_dir', default=str(OUT_ROOT))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for city in args.cities:
        fit, pool, test = load_protocol(city)
        print(f'[{city}] fit={len(fit):,} labelled pool={len(pool):,} labelled test={len(test):,}')

        for feat in args.features:
            X_fit, X_pool, X_test = build_features(fit, [fit, pool, test], FEATURE_COLUMNS[feat])
            configs = [('kmeans', {'k': args.num_clusters})]
            if not args.no_hdbscan:
                configs += [('hdbscan', {'min_cluster_size': s}) for s in args.hdbscan_min_cluster_sizes]

            for method, params in configs:
                t0 = time.time()
                if method == 'kmeans':
                    pc, pconf, tc = run_kmeans(X_fit, X_pool, X_test, params['k'], args.seed)
                else:
                    pc, pconf, tc = run_hdbscan(X_fit, X_pool, X_test, params['min_cluster_size'], args.seed)
                res = evaluate_clustering(
                    pool[['label', 'day']].assign(cluster=pc, conf=pconf),
                    test[['label', 'day']].assign(cluster=tc, conf=0.0),
                    m_values=DEFAULT_M_VALUES, n_boot=args.n_boot, seed=args.seed,
                )
                param_str = ','.join(f'{k}={v}' for k, v in params.items())
                res.insert(0, 'params', param_str)
                res.insert(0, 'method', method)
                res.insert(0, 'features', f'{feat}_hist')
                res.insert(0, 'city', city)
                all_rows.append(res)
                r = res[(res.variant == 'excl_other') & (res.m == '5')].iloc[0]
                print(f'  {feat:10s} {method:7s} {param_str:22s} clusters={r.n_clusters:3d} '
                      f'excl_other m=5 F1 w/m={r.f1_weighted:.3f}/{r.f1_macro:.3f} ({time.time() - t0:.0f}s)')

        pd.concat(all_rows).to_csv(out_dir / 'results.csv', index=False)

    print(f'\nSaved {out_dir / "results.csv"}')


if __name__ == '__main__':
    main()
