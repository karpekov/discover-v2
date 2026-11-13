"""
DEC (Xie et al., 2016) on a frozen pre-trained encoder, as a second deep-clustering baseline next to SCAN.

The encoder stays frozen; a residual linear adapter (initialised to identity) and the cluster centroids
are trained. Centroids are initialised with k-means on the fit embeddings; soft assignments use a
Student-t kernel (alpha=1) and are sharpened towards the DEC target distribution (KL loss), with the
target refreshed every `update_interval` steps. Scored with the shared evaluator like scan.py.

Usage:
    python -m tist_revision.pipeline.dec --city milan --num_clusters 20
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from .embedding_diagnostics import split_paths
from .encoding import encode, encoder_fields, get_device, load_model, load_vocab, tensorize
from .mapping_eval import DATA_ROOT, DEFAULT_M_VALUES, evaluate_clustering, load_protocol, save_assignments
from .scan import OUT_ROOT, set_seed


class DECHead(nn.Module):
    def __init__(self, d, centroids):
        super().__init__()
        self.adapter = nn.Linear(d, d)
        nn.init.eye_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)
        self.centroids = nn.Parameter(torch.as_tensor(centroids, dtype=torch.float32))

    def forward(self, x):
        z = self.adapter(x)
        q = 1.0 / (1.0 + torch.cdist(z, self.centroids) ** 2)
        return q / q.sum(1, keepdim=True)


def target_distribution(q):
    w = q ** 2 / q.sum(0)
    return w / w.sum(1, keepdim=True)


@torch.no_grad()
def soft_assign(head, X, bs=8192):
    return torch.cat([head(X[s:s + bs]) for s in range(0, len(X), bs)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--city', required=True)
    ap.add_argument('--encoder_dir', default='trained_models/{city}/{city}_fl20_seq_discover_v1_mlm_only')
    ap.add_argument('--num_clusters', type=int, default=20)
    ap.add_argument('--run_name', default=None)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--update_interval', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_boot', type=int, default=500)
    args = ap.parse_args()
    city = args.city
    encoder_dir = Path(args.encoder_dir.format(city=city))
    run_name = args.run_name or f'dec_k{args.num_clusters}'
    run = f'{encoder_dir.name}__{run_name}'
    set_seed(args.seed)
    device = get_device()
    t0 = time.time()

    vocab_path = DATA_ROOT / city / 'FL_20' / 'vocab.json'
    vocab = load_vocab(vocab_path)
    model = load_model(encoder_dir, vocab_path, device)
    cat_fields, global_fields = encoder_fields(model, vocab)
    E = {k: torch.from_numpy(encode(model, tensorize(p, vocab, cat_fields, global_fields), device)[0]).to(device)
         for k, p in split_paths(city).items()}
    del model

    km = KMeans(n_clusters=args.num_clusters, n_init=10, random_state=args.seed).fit(E['fit'].cpu().numpy())
    head = DECHead(E['fit'].shape[1], km.cluster_centers_).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    N, step = len(E['fit']), 0
    for epoch in range(args.epochs):
        perm = torch.randperm(N, device=device)
        total = 0.0
        for s in range(0, N, args.batch_size):
            if step % args.update_interval == 0:
                P = target_distribution(soft_assign(head, E['fit']))
            idx = perm[s:s + args.batch_size]
            q = head(E['fit'][idx])
            loss = torch.nn.functional.kl_div(q.clamp_min(1e-10).log(), P[idx], reduction='batchmean')
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            step += 1
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f'  epoch {epoch:3d} kl={total / ((N + args.batch_size - 1) // args.batch_size):.4f} {time.time() - t0:.0f}s')

    Q = {k: soft_assign(head, E[k]).cpu().numpy() for k in ('fit', 'pool', 'test')}
    sizes = np.bincount(Q['fit'].argmax(1), minlength=args.num_clusters)
    _, pool, test = load_protocol(city)
    cols = ['label', 'day']
    save_assignments(f'{city}__{run}__dec', Q['pool'].argmax(1), Q['pool'].max(1), Q['test'].argmax(1), Q['test'].max(1))
    res = evaluate_clustering(pool[cols].assign(cluster=Q['pool'].argmax(1), conf=Q['pool'].max(1)),
                              test[cols].assign(cluster=Q['test'].argmax(1), conf=Q['test'].max(1)),
                              m_values=DEFAULT_M_VALUES, n_boot=args.n_boot, seed=args.seed)
    meta = {'city': city, 'encoder': encoder_dir.name, 'run': run_name, 'method': 'dec',
            'num_clusters': args.num_clusters, 'epochs': args.epochs, 'seed': args.seed,
            'active_clusters': int((sizes > 0).sum()), 'largest_cluster_share': float(sizes.max() / sizes.sum()),
            'minutes': (time.time() - t0) / 60}
    res = res.assign(**meta)
    run_dir = OUT_ROOT / 'runs' / f'{city}__{run}'
    run_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(run_dir / 'scan_head.csv', index=False)
    with open(run_dir / 'run.json', 'w') as f:
        json.dump(meta | {'args': vars(args)}, f, indent=2)
    for variant in ('all', 'excl_other'):
        s = res[res.variant == variant].set_index('m')
        print(f'  {variant:10s} ' + '  '.join(f'm={m}: {s.loc[m, "f1_weighted"]:.3f}/{s.loc[m, "f1_macro"]:.3f}'
                                              for m in s.index))
    print(f'  active clusters {meta["active_clusters"]}/{args.num_clusters}, largest {meta["largest_cluster_share"]:.1%}, '
          f'{meta["minutes"]:.1f} min')


if __name__ == '__main__':
    main()
