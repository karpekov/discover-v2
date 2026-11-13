"""
SCAN and DEC on frozen, cached representations (embeddings, optionally + time-of-day histogram).

All methods keep the encoder frozen (the decision for the revision), so they only need the cached features:
  - k-means: fit on (up to) 30k fit windows, n_init 3; confidence = minus the distance to the centroid.
  - SCAN: `num_heads` heads (Standardize -> MLP) trained with the SCAN loss on mined cosine neighbours;
    the head with the lowest final loss is kept.
  - DEC: identity-initialised linear adapter + k-means-initialised centroids, Student-t soft assignment,
    KL to the sharpened target distribution (refreshed every `update_interval` steps).
Assignments of the pool (= fit windows) and test windows are saved as `{city}__{rep}__{method}_k{K}[suffix]`
(final runs put the seed in the representation name instead, e.g. `tf_s3+tod`, `hist_sensor_tod_s3`)
and scored by `label_levels.py`. Runs on GPU, MPS or CPU.

Usage:
    python -m tist_revision.pipeline.frozen_deep --cities milan --representations mlm_time mlm_time+tod --K 20 50
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from .dec import DECHead, soft_assign, target_distribution
from .encoding import get_device
from .mapping_eval import ARTIFACTS_DIR, load_protocol, save_assignments
from .rare_classes import representations
from .scan import make_head, mine_neighbors, scan_loss, set_seed


def run_scan(X, K, device, num_heads=5, num_neighbors=20, entropy_weight=2.0, epochs=50, lr=1e-3,
             batch_size=256, dropout=0.1):
    nbrs = mine_neighbors(X, num_neighbors)
    heads = nn.ModuleList([make_head(X.shape[1], K, dropout, X.mean(0), X.std(0)) for _ in range(num_heads)]).to(device)
    opt = torch.optim.Adam(heads.parameters(), lr=lr, weight_decay=1e-4)
    N = len(X)
    for _ in range(epochs):
        heads.train()
        perm, sums, nb_count = torch.randperm(N), torch.zeros(num_heads), 0
        for s in range(0, N, batch_size):
            a = perm[s:s + batch_size]
            nb = nbrs[a, torch.randint(0, num_neighbors, (len(a),))]
            fa, fn = X[a.to(device)], X[nb.to(device)]
            losses = [scan_loss(h(fa), h(fn), entropy_weight)[0] for h in heads]
            opt.zero_grad()
            sum(losses).backward()
            opt.step()
            sums += torch.tensor([l.item() for l in losses])
            nb_count += 1
    head = heads[int((sums / nb_count).argmin())].eval()
    return lambda Z: torch.cat([head(Z[s:s + 8192]).softmax(1) for s in range(0, len(Z), 8192)]).detach()


def run_kmeans(X, K, seed, max_fit=30000):
    """k-means as in `label_levels.kmeans_assignments`; confidence = closeness to the centroid."""
    Xn = X.cpu().numpy()
    sub = np.random.default_rng(seed).choice(len(Xn), size=min(max_fit, len(Xn)), replace=False)
    km = KMeans(n_clusters=K, n_init=3, random_state=seed).fit(Xn[sub])
    C = torch.as_tensor(km.cluster_centers_, dtype=X.dtype, device=X.device)

    def predict(Z):
        d = torch.cat([torch.cdist(Z[s:s + 8192], C) for s in range(0, len(Z), 8192)])
        dmin, c = d.min(1)
        return c.cpu().numpy(), -dmin.cpu().numpy()
    return predict


def _argmax(predict_proba):
    def predict(Z):
        P = predict_proba(Z).cpu().numpy()
        return P.argmax(1), P.max(1)
    return predict


def run_dec(X, K, device, seed, epochs=30, lr=1e-3, batch_size=256, update_interval=200):
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(X.cpu().numpy())
    head = DECHead(X.shape[1], km.cluster_centers_).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    N, step = len(X), 0
    for _ in range(epochs):
        perm = torch.randperm(N, device=device)
        for s in range(0, N, batch_size):
            if step % update_interval == 0:
                P = target_distribution(soft_assign(head, X))
            idx = perm[s:s + batch_size]
            loss = nn.functional.kl_div(head(X[idx]).clamp_min(1e-10).log(), P[idx], reduction='batchmean')
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
    return lambda Z: soft_assign(head, Z)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cities', nargs='+', default=['milan', 'aruba', 'cairo'])
    ap.add_argument('--representations', nargs='+', default=['mlm_time', 'mlm_time+tod'])
    ap.add_argument('--methods', nargs='+', default=['scan', 'dec'], choices=['kmeans', 'scan', 'dec'])
    ap.add_argument('--K', type=int, nargs='+', default=[20, 50])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--suffix', default='', help='appended to assignment names, e.g. _s1 for seed runs')
    args = ap.parse_args()
    device = get_device()
    for city in args.cities:
        fit, pool, test = load_protocol(city)
        for rep, (X_fit, _, X_test) in representations(city, fit, pool, test, args.representations):
            Xf = torch.as_tensor(np.asarray(X_fit, dtype=np.float32), device=device)
            Xt = torch.as_tensor(np.asarray(X_test, dtype=np.float32), device=device)
            for K in args.K:
                for method in args.methods:
                    t0 = time.time()
                    set_seed(args.seed)
                    if method == 'kmeans':
                        predict = run_kmeans(Xf, K, args.seed)
                    elif method == 'scan':
                        predict = _argmax(run_scan(Xf, K, device))
                    else:
                        predict = _argmax(run_dec(Xf, K, device, args.seed))
                    (c_pool, conf_pool), (c_test, conf_test) = predict(Xf), predict(Xt)
                    save_assignments(f'{city}__{rep}__{method}_k{K}{args.suffix}', c_pool, conf_pool, c_test, conf_test)
                    sizes = np.bincount(c_pool, minlength=K)
                    print(f'[{city}] {rep} {method} K={K}: {(sizes > 0).sum()}/{K} clusters used, '
                          f'largest {sizes.max() / sizes.sum():.1%} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
