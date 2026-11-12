"""
Stage 1.5 for the DISCOVER revision: is an embedding space worth running SCAN on?

Protocol (see mapping_eval): clusters are fit on continuous FL_20 train windows; every
labelled quantity uses pre-segmented FL_20_p windows (one true label per window).

For every representation (histogram features and frozen encoder embeddings) computes:
  (i)   kNN label purity at k=20 among pre-segmented train windows,
  (ii)  k-means K=20 fit on FL_20 train, scored with the top-m mapping evaluator,
  (iii) a linear probe trained on n labelled windows per class (separability ceiling).
Existing SCAN checkpoints are also re-scored (cluster head argmax, confidence = max prob).

Encoders receive inputs exactly as in MLM pre-training (alignment/dataset.py): raw events,
global context tokens from the first event, raw time deltas, dynamic padding.
`--legacy_dataio` instead uses the dataio/dataset.py path that train_scan.py uses
(adds two UNK pseudo-events for tod/dow and pads to 50), to quantify that mismatch.

Usage:
    python -m tist_revision.pipeline.embedding_diagnostics --cities milan aruba cairo
"""
import argparse
import json
import sys
import time
from datetime import datetime
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

from .histogram_baselines import FEATURE_COLUMNS, build_features
from .mapping_eval import (ARTIFACTS_DIR, DATA_ROOT, EXCLUDED_LABELS, RESULTS_DIR, VARIANTS, evaluate_clustering,
                           load_protocol, save_assignments)

SRC_DIR = Path(__file__).resolve().parents[2] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataio.dataset import SmartHomeDataset  # noqa: E402
from models.scan_model import SCANClusteringModel  # noqa: E402

MODEL_ROOT = Path('trained_models')
OUT_ROOT = RESULTS_DIR / 'stage1_5_embeddings'
EMB_DIR = ARTIFACTS_DIR / 'embeddings'
M_VALUES = (5, 'all')
# Vocab field → event key, as in alignment/dataset.py.
FIELD_TO_EVENT_KEY = {'sensor': 'sensor_id', 'state': 'event_type', 'room_id': 'room'}


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _lookup(vocab_field, value):
    return vocab_field[value] if value in vocab_field else vocab_field.get('UNK', 0)


def alignment_batches(samples, vocab, cat_fields, global_fields, batch_size):
    """Yield (input_data, attention_mask) batches encoded like alignment/dataset.py."""
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        L = max(len(s['sensor_sequence']) for s in chunk)
        B = len(chunk)
        cat = {f: torch.zeros((B, L), dtype=torch.long) for f in cat_fields}
        coords = torch.zeros((B, L, 2))
        deltas = torch.zeros((B, L))
        mask = torch.zeros((B, L), dtype=torch.bool)
        glob = {f: torch.zeros(B, dtype=torch.long) for f in global_fields}
        for i, s in enumerate(chunk):
            seq = s['sensor_sequence']
            for f in global_fields:
                glob[f][i] = _lookup(vocab[f], seq[0].get(FIELD_TO_EVENT_KEY.get(f, f), 'UNK'))
            prev = None
            for j, e in enumerate(seq):
                for f in cat_fields:
                    cat[f][i, j] = _lookup(vocab[f], e.get(FIELD_TO_EVENT_KEY.get(f, f), 'UNK'))
                coords[i, j, 0], coords[i, j, 1] = e.get('x', 0.0), e.get('y', 0.0)
                t = datetime.fromisoformat(e['timestamp'])
                deltas[i, j] = 0.0 if prev is None else (t - prev).total_seconds()
                prev = t
                mask[i, j] = True
        yield {'categorical_features': cat, 'global_categorical_features': glob,
               'coordinates': coords, 'time_deltas': deltas}, mask


def _to(x, device):
    if isinstance(x, dict):
        return {k: _to(v, device) for k, v in x.items()}
    return x.to(device)


@torch.no_grad()
def encode(model: SCANClusteringModel, data_path: Path, vocab_path: Path, device, legacy_dataio=False,
           batch_size=512):
    """Return (embeddings, cluster probabilities) for every window in `data_path`, in file order."""
    model.eval()
    embs, probs = [], []
    if legacy_dataio:
        ds = SmartHomeDataset(data_path=str(data_path), vocab_path=str(vocab_path), sequence_length=50,
                              max_captions=1, caption_types='long')
        batches = ((
            {'categorical_features': {f: torch.stack([s['categorical_features'][f] for s in b])
                                      for f in b[0]['categorical_features']},
             'coordinates': torch.stack([s['coordinates'] for s in b]),
             'time_deltas': torch.stack([s['time_deltas'] for s in b])},
            torch.stack([s['mask'] for s in b]))
            for b in DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x))
    else:
        with open(data_path) as f:
            samples = json.load(f)['samples']
        with open(vocab_path) as f:
            vocab = json.load(f)
        meta = model.alignment_config.encoder.get('metadata', {})
        cat_fields = [f for f in meta.get('categorical_fields', ['sensor', 'state', 'room_id']) if f in vocab]
        global_fields = [f for f in meta.get('global_categorical_fields', []) if f in vocab]
        batches = alignment_batches(samples, vocab, cat_fields, global_fields, batch_size)
    for input_data, mask in batches:
        logits, emb = model(_to(input_data, device), mask.to(device), return_embeddings=True)
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


def diagnose(name, X_fit, X_pool, X_test, pool, test, args, rows, city, metric):
    t0 = time.time()
    y_pool, y_test = pool['label'].to_numpy(), test['label'].to_numpy()

    p_all, p_excl = knn_purity(X_pool, y_pool, k=args.k_neighbors, seed=args.seed, metric=metric)
    rows['knn'].append({'city': city, 'representation': name, 'purity_all': p_all, 'purity_excl_other': p_excl})

    km = KMeans(n_clusters=args.num_clusters, n_init=10, random_state=args.seed).fit(X_fit)
    pc, tc = km.predict(X_pool), km.predict(X_test)
    pconf = -np.linalg.norm(X_pool - km.cluster_centers_[pc], axis=1)
    save_assignments(f'{city}__{name}__kmeans_k{args.num_clusters}', pc, pconf, tc)
    res = evaluate_clustering(pool.assign(cluster=pc, conf=pconf), test.assign(cluster=tc, conf=0.0),
                              m_values=M_VALUES, n_boot=args.n_boot, seed=args.seed)
    rows['clustering'].append(res.assign(city=city, representation=name, method='kmeans'))

    for variant in VARIANTS:
        pm = np.ones(len(y_pool), bool) if variant == 'all' else ~np.isin(y_pool, EXCLUDED_LABELS)
        tm = np.ones(len(y_test), bool) if variant == 'all' else ~np.isin(y_test, EXCLUDED_LABELS)
        for n in args.probe_n:
            w, m = linear_probe(X_pool[pm], y_pool[pm], X_test[tm], y_test[tm], 'full' if n == 'full' else int(n))
            rows['probe'].append({'city': city, 'representation': name, 'variant': variant, 'n_per_class': str(n),
                                  'f1_weighted': w, 'f1_macro': m})
    r = res[(res.variant == 'excl_other') & (res.m == '5')].iloc[0]
    print(f'  {name:28s} purity all={p_all:.3f} excl={p_excl:.3f} | kmeans excl_other m=5 '
          f'w/m={r.f1_weighted:.3f}/{r.f1_macro:.3f} ({time.time() - t0:.0f}s)')


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
    ap.add_argument('--legacy_dataio', action='store_true', help='Encode with the train_scan.py input path')
    ap.add_argument('--output_dir', default=str(OUT_ROOT))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    rows = {'knn': [], 'clustering': [], 'probe': []}

    def save():
        pd.DataFrame(rows['knn']).to_csv(out_dir / 'knn_purity.csv', index=False)
        if rows['clustering']:
            pd.concat(rows['clustering']).to_csv(out_dir / 'clustering.csv', index=False)
        pd.DataFrame(rows['probe']).to_csv(out_dir / 'linear_probe.csv', index=False)

    for city in args.cities:
        fit, pool, test = load_protocol(city)
        print(f'[{city}] fit={len(fit):,} labelled pool={len(pool):,} labelled test={len(test):,} device={device}')
        cols = ['label', 'day']

        if not args.no_histograms:
            for feat, columns in FEATURE_COLUMNS.items():
                X_fit, X_pool, X_test = build_features(fit, [fit, pool, test], columns)
                diagnose(f'hist_{feat}', X_fit, X_pool, X_test, pool[cols], test[cols], args, rows, city,
                         metric='euclidean')
            save()

        if args.no_encoders:
            continue
        vocab_path = DATA_ROOT / city / 'FL_20' / 'vocab.json'
        paths = {'fit': DATA_ROOT / city / 'FL_20' / 'train.json',
                 'pool': DATA_ROOT / city / 'FL_20_p' / 'train.json',
                 'test': DATA_ROOT / city / 'FL_20_p' / 'test.json'}
        extra = [e.format(city=city) for e in args.extra_encoders]
        for enc_name, model in load_encoders(city, device, extra, args.only_extra):
            t0 = time.time()
            enc = {k: encode(model, p, vocab_path, device, args.legacy_dataio) for k, p in paths.items()}
            assert len(enc['fit'][0]) == len(fit) and len(enc['pool'][0]) == len(pool) and len(enc['test'][0]) == len(test)
            print(f'  encoded {enc_name} ({time.time() - t0:.0f}s)')
            diagnose(enc_name, enc['fit'][0], enc['pool'][0], enc['test'][0], pool[cols], test[cols],
                     args, rows, city, metric='cosine')
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
            save()

    save()
    print(f'Saved results to {out_dir}')


if __name__ == '__main__':
    main()
