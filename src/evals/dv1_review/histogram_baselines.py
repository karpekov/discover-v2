"""
Stage 1 baselines for the DISCOVER revision (R2-2.9a): cluster per-window sensor / room
histograms with k-means and HDBSCAN, then score with the shared top-m mapping evaluator.

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

from .mapping_eval import DEFAULT_M_VALUES, evaluate_clustering, load_windows

DATA_ROOT = Path('data/processed/casas')
OUT_ROOT = Path('results/evals/dv1_review/stage1_histograms')


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


def run_kmeans(X_pool, X_test, k, seed):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X_pool)
    pool_c = km.labels_
    test_c = km.predict(X_test)
    # Confidence = closeness to the assigned centroid.
    pool_conf = -np.linalg.norm(X_pool - km.cluster_centers_[pool_c], axis=1)
    test_conf = -np.linalg.norm(X_test - km.cluster_centers_[test_c], axis=1)
    return pool_c, pool_conf, test_c, test_conf


def run_hdbscan(X_pool, X_test, min_cluster_size, seed, max_fit=20000):
    """Fit HDBSCAN on a pool subsample; assign the remaining pool and test windows to the
    nearest clustered fit points (kNN vote). Propagated points get confidence -1 so that
    top-m selection prefers windows HDBSCAN itself placed in the cluster."""
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(X_pool), size=min(max_fit, len(X_pool)), replace=False)
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit(X_pool[fit_idx])
    fit_c, fit_p = hdb.labels_, hdb.probabilities_

    pool_c = np.full(len(X_pool), -1)
    pool_conf = np.full(len(X_pool), -1.0)
    pool_c[fit_idx], pool_conf[fit_idx] = fit_c, fit_p

    clustered = fit_c != -1
    if clustered.sum() == 0:
        return pool_c, pool_conf, np.full(len(X_test), -1), np.full(len(X_test), -1.0)
    knn = KNeighborsClassifier(n_neighbors=10).fit(X_pool[fit_idx][clustered], fit_c[clustered])
    rest = np.setdiff1d(np.arange(len(X_pool)), fit_idx)
    if len(rest):
        pool_c[rest] = knn.predict(X_pool[rest])
    test_c = knn.predict(X_test)
    return pool_c, pool_conf, test_c, np.full(len(X_test), -1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cities', nargs='+', default=['milan', 'aruba', 'cairo'])
    ap.add_argument('--window', default='FL_20')
    ap.add_argument('--features', nargs='+', default=['sensor', 'room'])
    ap.add_argument('--num_clusters', type=int, default=20)
    ap.add_argument('--hdbscan_min_cluster_sizes', type=int, nargs='+', default=[50, 100, 200, 500])
    ap.add_argument('--no_hdbscan', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--output_dir', default=str(OUT_ROOT))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / 'cache'
    all_rows = []

    for city in args.cities:
        data_dir = DATA_ROOT / city / args.window
        t0 = time.time()
        pool = load_windows(data_dir, 'train', cache_dir)
        test = load_windows(data_dir, 'test', cache_dir)
        print(f'[{city}] pool={len(pool):,} test={len(test):,} ({time.time() - t0:.0f}s load)')

        for feat in args.features:
            col = {'sensor': 'sensors', 'room': 'rooms'}[feat]
            vocab = sorted({v for items in pool[col] for v in items})
            X_pool = histogram_features(pool, col, vocab)
            X_test = histogram_features(test, col, vocab)

            configs = [('kmeans', {'k': args.num_clusters})]
            if not args.no_hdbscan:
                configs += [('hdbscan', {'min_cluster_size': s}) for s in args.hdbscan_min_cluster_sizes]

            for method, params in configs:
                t0 = time.time()
                if method == 'kmeans':
                    pc, pconf, tc, tconf = run_kmeans(X_pool, X_test, params['k'], args.seed)
                else:
                    pc, pconf, tc, tconf = run_hdbscan(X_pool, X_test, params['min_cluster_size'], args.seed)
                res = evaluate_clustering(
                    pool.assign(cluster=pc, conf=pconf)[['cluster', 'conf', 'label', 'day']],
                    test.assign(cluster=tc, conf=tconf)[['cluster', 'conf', 'label', 'day']],
                    m_values=DEFAULT_M_VALUES, n_boot=args.n_boot, seed=args.seed,
                )
                param_str = ','.join(f'{k}={v}' for k, v in params.items())
                res.insert(0, 'params', param_str)
                res.insert(0, 'method', method)
                res.insert(0, 'features', f'{feat}_hist')
                res.insert(0, 'city', city)
                res['n_noise_test'] = int((tc == -1).sum())
                res['n_noise_pool_fit'] = int(((pc == -1)).sum())
                all_rows.append(res)
                best = res[(res.variant == 'all') & (res.m == 'all')].iloc[0]
                print(f'  {feat:6s} {method:7s} {param_str:22s} clusters={best.n_clusters:3d} '
                      f'F1w(all,m=all)={best.f1_weighted:.3f} ({time.time() - t0:.0f}s)')

        pd.concat(all_rows).to_csv(out_dir / 'results.csv', index=False)

    df = pd.concat(all_rows)
    df.to_csv(out_dir / 'results.csv', index=False)
    print(f'\nSaved {out_dir / "results.csv"}')


if __name__ == '__main__':
    main()
