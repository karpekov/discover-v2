"""
Stage 1.5 for the DISCOVER revision: is an embedding space worth running SCAN on?

For every representation (histogram features and frozen encoder embeddings) computes:
  (i)   kNN label purity at k=20 on train windows (what SCAN mines as positives),
  (ii)  k-means K=20 scored with the shared top-m mapping evaluator,
  (iii) a linear probe trained on n labelled windows per class (separability ceiling).
Existing SCAN checkpoints are also re-scored with the shared evaluator (cluster head argmax,
confidence = max softmax probability).

Representations:
  hist_sensor, hist_room, hist_sensor_tod  - row-normalised histograms
  mlm      - frozen pretrained encoder (trained_models/{city}/{city}_fl20_seq_discover_v1_mlm_only)
  scan_enc - encoder after SCAN fine-tuning (trained_models/{city}/scan_fl20_20cl_discover_v1)

Usage:
    python -m src.evals.dv1_review.embedding_diagnostics --cities milan aruba cairo
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .histogram_baselines import histogram_features
from .mapping_eval import EXCLUDED_LABELS, evaluate_clustering, load_windows

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataio.dataset import SmartHomeDataset  # noqa: E402
from models.scan_model import SCANClusteringModel  # noqa: E402

DATA_ROOT = Path('data/processed/casas')
MODEL_ROOT = Path('trained_models')
OUT_ROOT = Path('results/evals/dv1_review/stage1_5_embeddings')
M_VALUES = (5, 'all')


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def collate(batch, device):
    """Same input format as train_scan.extract_embeddings."""
    cat = {f: torch.stack([s['categorical_features'][f] for s in batch]).to(device)
           for f in batch[0]['categorical_features']}
    return {
        'input_data': {
            'categorical_features': cat,
            'coordinates': torch.stack([s['coordinates'] for s in batch]).to(device),
            'time_deltas': torch.stack([s['time_deltas'] for s in batch]).to(device),
        },
        'attention_mask': torch.stack([s['mask'] for s in batch]).to(device),
    }


@torch.no_grad()
def encode(model: SCANClusteringModel, data_path: Path, vocab_path: Path, device, batch_size=512):
    """Return (embeddings, cluster probabilities) for every window in `data_path`, in file order."""
    ds = SmartHomeDataset(data_path=str(data_path), vocab_path=str(vocab_path), sequence_length=50,
                          max_captions=1, caption_types='long')
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                    collate_fn=lambda b: collate(b, device))
    embs, probs = [], []
    model.eval()
    for batch in dl:
        logits, emb = model(batch['input_data'], batch['attention_mask'], return_embeddings=True)
        embs.append(emb.float().cpu().numpy())
        probs.append(torch.softmax(logits, dim=1).float().cpu().numpy())
    return np.vstack(embs), np.vstack(probs)


def load_encoders(city: str, device, extra_dirs=(), only_extra=False):
    """Yield (name, model) for extra encoder dirs, the frozen MLM encoder and the SCAN-fine-tuned model."""
    mlm_dir = MODEL_ROOT / city / f'{city}_fl20_seq_discover_v1_mlm_only'
    scan_ckpt = MODEL_ROOT / city / 'scan_fl20_20cl_discover_v1' / 'best_model.pt'
    vocab = DATA_ROOT / city / 'FL_20' / 'vocab.json'
    for d in extra_dirs:
        d = Path(d)
        if d.exists():
            yield d.name, SCANClusteringModel(str(d), num_clusters=20, vocab_path=str(vocab), device=device).to(device)
        else:
            print(f'  [skip] no encoder at {d}')
    if only_extra:
        return
    if not mlm_dir.exists():
        print(f'  [skip] no MLM encoder at {mlm_dir}')
        return
    # Clustering head is random for 'mlm'; only its embeddings are used.
    yield 'mlm', SCANClusteringModel(str(mlm_dir), num_clusters=20, vocab_path=str(vocab), device=device).to(device)
    if scan_ckpt.exists():
        model = SCANClusteringModel(str(mlm_dir), num_clusters=20, vocab_path=str(vocab), device=device)
        state = torch.load(scan_ckpt, map_location='cpu', weights_only=False)['model_state_dict']
        model.load_state_dict(state)
        yield 'scan_enc', model.to(device)
    else:
        print(f'  [skip] no SCAN checkpoint at {scan_ckpt}')


def knn_purity(X, labels, k=20, max_n=20000, seed=0, metric='cosine'):
    """Mean share of each anchor's k nearest neighbours with the same label.
    Returns purity over all anchors and over anchors whose label is not No_Activity/Other."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(max_n, len(X)), replace=False)
    Xs, ys = X[idx], labels[idx]
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(Xs)
    nbr = nn.kneighbors(Xs, return_distance=False)[:, 1:]
    same = (ys[nbr] == ys[:, None]).mean(1)
    keep = ~np.isin(ys, EXCLUDED_LABELS)
    return float(same.mean()), float(same[keep].mean())


def linear_probe(X_pool, y_pool, X_test, y_test, n_per_class, seeds=(0, 1, 2), max_full=20000):
    """Logistic regression on n labelled pool windows per class; scored on test."""
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if n_per_class == 'full':
            idx = rng.choice(len(X_pool), size=min(max_full, len(X_pool)), replace=False)
        else:
            idx = np.concatenate([rng.choice(np.flatnonzero(y_pool == c), size=min(n_per_class, (y_pool == c).sum()),
                                             replace=False) for c in np.unique(y_pool)])
        scaler = StandardScaler().fit(X_pool[idx])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight=None)
        clf.fit(scaler.transform(X_pool[idx]), y_pool[idx])
        pred = clf.predict(scaler.transform(X_test))
        labels = np.unique(y_test)
        out.append((f1_score(y_test, pred, average='weighted', zero_division=0),
                    f1_score(y_test, pred, labels=labels, average='macro', zero_division=0)))
    return np.mean(out, axis=0)


def diagnose(name, X_pool, X_test, pool, test, args, rows_knn, rows_clu, rows_probe, city, metric):
    t0 = time.time()
    y_pool, y_test = pool['label'].to_numpy(), test['label'].to_numpy()

    p_all, p_excl = knn_purity(X_pool, y_pool, k=args.k_neighbors, seed=args.seed, metric=metric)
    rows_knn.append({'city': city, 'representation': name, 'purity_all': p_all, 'purity_excl_na': p_excl})

    km = KMeans(n_clusters=args.num_clusters, n_init=10, random_state=args.seed).fit(X_pool)
    tc = km.predict(X_test)
    res = evaluate_clustering(
        pool.assign(cluster=km.labels_, conf=-np.linalg.norm(X_pool - km.cluster_centers_[km.labels_], axis=1)),
        test.assign(cluster=tc, conf=0.0), m_values=M_VALUES, n_boot=args.n_boot, seed=args.seed)
    rows_clu.append(res.assign(city=city, representation=name, method='kmeans'))

    for variant in ('all', 'excl_na'):
        pm = np.ones(len(y_pool), bool) if variant == 'all' else ~np.isin(y_pool, EXCLUDED_LABELS)
        tm = np.ones(len(y_test), bool) if variant == 'all' else ~np.isin(y_test, EXCLUDED_LABELS)
        for n in args.probe_n:
            n_ = 'full' if n == 'full' else int(n)
            w, m = linear_probe(X_pool[pm], y_pool[pm], X_test[tm], y_test[tm], n_)
            rows_probe.append({'city': city, 'representation': name, 'variant': variant, 'n_per_class': str(n),
                               'f1_weighted': w, 'f1_macro': m})
    print(f'  {name:16s} purity all={p_all:.3f} excl={p_excl:.3f} ({time.time() - t0:.0f}s)')


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
                    help='Extra encoder dirs (same FL_20 data/vocab); use {city} as placeholder')
    ap.add_argument('--only_extra', action='store_true', help='Skip the default MLM/SCAN encoders')
    ap.add_argument('--output_dir', default=str(OUT_ROOT))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path('results/evals/dv1_review/cache')
    device = get_device()
    rows_knn, rows_clu, rows_probe = [], [], []

    def save():
        pd.DataFrame(rows_knn).to_csv(out_dir / 'knn_purity.csv', index=False)
        if rows_clu:
            pd.concat(rows_clu).to_csv(out_dir / 'clustering.csv', index=False)
        pd.DataFrame(rows_probe).to_csv(out_dir / 'linear_probe.csv', index=False)

    for city in args.cities:
        data_dir = DATA_ROOT / city / 'FL_20'
        pool = load_windows(data_dir, 'train', cache_dir)
        test = load_windows(data_dir, 'test', cache_dir)
        print(f'[{city}] pool={len(pool):,} test={len(test):,} device={device}')
        base_cols = ['label', 'day']

        # Histogram references.
        hist_sets = [] if args.no_histograms else [('hist_sensor', ['sensors']), ('hist_room', ['rooms']),
                                                    ('hist_sensor_tod', ['sensors', 'tods'])]
        for name, cols in hist_sets:
            parts_p, parts_t = [], []
            for col in cols:
                vocab = sorted({v for items in pool[col] for v in items})
                parts_p.append(histogram_features(pool, col, vocab))
                parts_t.append(histogram_features(test, col, vocab))
            diagnose(name, np.hstack(parts_p), np.hstack(parts_t), pool[base_cols], test[base_cols],
                     args, rows_knn, rows_clu, rows_probe, city, metric='euclidean')
        save()

        if args.no_encoders:
            continue
        vocab_path = data_dir / 'vocab.json'
        extra = [e.format(city=city) for e in args.extra_encoders]
        for enc_name, model in load_encoders(city, device, extra, args.only_extra):
            t0 = time.time()
            E_pool, P_pool = encode(model, data_dir / 'train.json', vocab_path, device)
            E_test, P_test = encode(model, data_dir / 'test.json', vocab_path, device)
            assert len(E_pool) == len(pool) and len(E_test) == len(test)
            print(f'  encoded {enc_name} {E_pool.shape} ({time.time() - t0:.0f}s)')
            diagnose(enc_name, E_pool, E_test, pool[base_cols], test[base_cols],
                     args, rows_knn, rows_clu, rows_probe, city, metric='cosine')
            if enc_name == 'scan_enc':
                res = evaluate_clustering(
                    pool[base_cols].assign(cluster=P_pool.argmax(1), conf=P_pool.max(1)),
                    test[base_cols].assign(cluster=P_test.argmax(1), conf=P_test.max(1)),
                    m_values=M_VALUES, n_boot=args.n_boot, seed=args.seed)
                rows_clu.append(res.assign(city=city, representation='scan_enc', method='scan_head'))
            np.savez_compressed(out_dir / f'{city}_{enc_name}_embeddings.npz',
                                pool=E_pool.astype(np.float16), test=E_test.astype(np.float16))
            del model
            save()

    save()
    print(f'Saved results to {out_dir}')


if __name__ == '__main__':
    main()
