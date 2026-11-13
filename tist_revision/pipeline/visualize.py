"""
Figures for the clustering runs (reads artifacts written by histogram_baselines / embedding_diagnostics):
  - confusion_{city}_{variant}.png: row-normalised confusion matrices on pre-segmented test windows,
    one row per method, one column per annotation budget m.
  - clusters_{city}.png: label composition of every cluster (pre-segmented train pool), with the
    label each cluster gets at m=5 and m=all.
  - tsne_{city}.png: t-SNE of pre-segmented test windows per representation, coloured by true
    label and by assigned cluster.

Usage:
    python -m tist_revision.pipeline.visualize --cities milan aruba cairo
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from .histogram_baselines import FEATURE_COLUMNS, build_features  # noqa: E402
from .mapping_eval import (ARTIFACTS_DIR, EXCLUDED_LABELS, RESULTS_DIR, UNMAPPED, load_protocol,  # noqa: E402
                           map_clusters)

FIG_DIR = RESULTS_DIR / 'figures'
DEFAULT_METHODS = ('hist_sensor__kmeans_k20', 'hist_sensor_tod__kmeans_k20', 'mlm__kmeans_k20', 'scan_enc__scan_head')
TSNE_REPRESENTATIONS = ('hist_sensor', 'hist_sensor_tod', 'mlm', 'scan_enc')


def label_colors(labels):
    meta = json.load(open('metadata/casas_metadata.json'))['milan']['label_deepcasas_color']
    by_lower = {k.lower(): v for k, v in meta.items()}
    by_lower['enter_home'] = '#a6d96a'
    fallback = plt.get_cmap('tab20')
    return {l: by_lower.get(l.lower(), fallback(i % 20)) for i, l in enumerate(labels)}


def short(method):
    """Display name: drop the '{city}_fl20_seq_discover_v1_' encoder prefix."""
    return method.replace('{city}_fl20_seq_discover_v1_', '')


def load_assignments(city, method):
    """`method` may contain a {city} placeholder (e.g. '{city}_fl20_seq_discover_v1_mlm_only__frozen_k50_ent2__scan_head')."""
    path = ARTIFACTS_DIR / 'assignments' / f'{city}__{method.format(city=city)}.npz'
    return dict(np.load(path)) if path.exists() else None


def _subset(pool, test, a, variant):
    pm = np.ones(len(pool), bool) if variant == 'all' else ~pool['label'].isin(EXCLUDED_LABELS).to_numpy()
    tm = np.ones(len(test), bool) if variant == 'all' else ~test['label'].isin(EXCLUDED_LABELS).to_numpy()
    return pm, tm


def predict(pool, test, a, variant, m):
    pm, tm = _subset(pool, test, a, variant)
    pc, pconf, plab = a['pool_cluster'][pm], a['pool_conf'][pm], pool['label'].to_numpy()[pm]
    keep = pc != -1
    mapping = map_clusters(pc[keep], pconf[keep], plab[keep], m)
    y_pred = np.array([mapping.get(c, UNMAPPED) for c in a['test_cluster'][tm]])
    return test['label'].to_numpy()[tm], y_pred, mapping


def plot_confusions(city, pool, test, methods, m_values, variant, out_dir):
    methods = [(mt, a) for mt in methods if (a := load_assignments(city, mt)) is not None]
    if not methods:
        return
    labels = test['label'].value_counts().index.tolist()
    if variant != 'all':
        labels = [l for l in labels if l not in EXCLUDED_LABELS]
    fig, axes = plt.subplots(len(methods), len(m_values), figsize=(4.6 * len(m_values), 4.2 * len(methods)),
                             squeeze=False)
    for r, (method, a) in enumerate(methods):
        for c, m in enumerate(m_values):
            y_true, y_pred, _ = predict(pool, test, a, variant, m)
            cols = labels + ([UNMAPPED] if (y_pred == UNMAPPED).any() else [])
            cm = pd.crosstab(pd.Categorical(y_true, labels), pd.Categorical(y_pred, cols), dropna=False).to_numpy()
            norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax = axes[r, c]
            ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
            for i in range(len(labels)):
                for j in range(len(cols)):
                    if cm[i, j]:
                        ax.text(j, i, f'{norm[i, j]:.2f}', ha='center', va='center', fontsize=6,
                                color='white' if norm[i, j] > 0.5 else 'black')
            ax.set_xticks(range(len(cols)), [x if x != UNMAPPED else 'unmapped' for x in cols], rotation=60,
                          ha='right', fontsize=7)
            ax.set_yticks(range(len(labels)), [f'{l} ({n})' for l, n in zip(labels, cm.sum(1))], fontsize=7)
            w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            mac = f1_score(y_true, y_pred, labels=np.unique(y_true), average='macro', zero_division=0)
            ax.set_title(f'{short(method)}\nm={m}  weighted F1={w:.2f}  macro F1={mac:.2f}', fontsize=8)
            if c == 0:
                ax.set_ylabel('true label (n test windows)', fontsize=7)
            ax.set_xlabel('predicted (cluster majority label)', fontsize=7)
    fig.suptitle(f'{city}: confusion on pre-segmented test windows (variant={variant}, row-normalised)')
    fig.tight_layout()
    fig.savefig(out_dir / f'confusion_{city}_{variant}.png', dpi=130)
    plt.close(fig)


def plot_cluster_composition(city, pool, test, methods, out_dir):
    methods = [(mt, a) for mt in methods if (a := load_assignments(city, mt)) is not None]
    if not methods:
        return
    labels = pool['label'].value_counts().index.tolist()
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 7), squeeze=False)
    for ax, (method, a) in zip(axes[0], methods):
        _, _, map5 = predict(pool, test, a, 'all', 5)
        _, _, map_all = predict(pool, test, a, 'all', 'all')
        pc = a['pool_cluster']
        ct = pd.crosstab(pd.Series(pc, name='cluster'), pd.Categorical(pool['label'], labels), dropna=False)
        ct = ct.loc[ct.index != -1]
        order = sorted(ct.index, key=lambda k: (labels.index(map_all[k]) if k in map_all else 99, -ct.loc[k].sum()))
        ct = ct.loc[order]
        frac = ct.to_numpy() / np.maximum(ct.to_numpy().sum(1, keepdims=True), 1)
        ax.imshow(frac, cmap='Purples', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha='right', fontsize=7)
        ax.set_yticks(range(len(ct)), [f'c{k} n={ct.loc[k].sum()} → {map5.get(k, "-")} / {map_all.get(k, "-")}'
                                       for k in ct.index], fontsize=6)
        ax.set_title(f'{short(method)}\nrow: cluster, size → label at m=5 / m=all', fontsize=8)
    fig.suptitle(f'{city}: label composition of each cluster (pre-segmented train pool)')
    fig.tight_layout()
    fig.savefig(out_dir / f'clusters_{city}.png', dpi=130)
    plt.close(fig)


def test_features(city, rep, fit, pool, test):
    if rep.startswith('hist_'):
        return build_features(fit, [test], FEATURE_COLUMNS[rep[len('hist_'):]])[0]
    path = ARTIFACTS_DIR / 'embeddings' / f'{city}_{rep}_embeddings.npz'
    if not path.exists():
        return None
    X = np.load(path)['test'].astype(np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)


def plot_tsne(city, fit, pool, test, reps, max_points, seed, out_dir):
    rng = np.random.default_rng(seed)
    y = test['label'].to_numpy()
    # Keep every window of rare classes so they stay visible.
    idx = []
    for l in np.unique(y):
        members = np.flatnonzero(y == l)
        n = min(len(members), max(max_points // 20, int(max_points * len(members) / len(y))))
        idx.append(rng.choice(members, size=n, replace=False))
    idx = np.sort(np.concatenate(idx))
    labels = test['label'].value_counts().index.tolist()
    colors = label_colors(labels)
    reps = [(r, X) for r in reps if (X := test_features(city, r, fit, pool, test)) is not None]
    fig, axes = plt.subplots(len(reps), 2, figsize=(11, 4.8 * len(reps)), squeeze=False)
    for row, (rep, X) in enumerate(reps):
        Xs = X[idx]
        if Xs.shape[1] > 50:
            Xs = PCA(n_components=50, random_state=seed).fit_transform(Xs)
        Z = TSNE(n_components=2, init='pca', perplexity=30, random_state=seed).fit_transform(Xs)
        ax = axes[row, 0]
        for l in labels:
            sel = y[idx] == l
            ax.scatter(Z[sel, 0], Z[sel, 1], s=3, color=colors[l], label=f'{l} ({sel.sum()})', alpha=0.7)
        ax.legend(fontsize=6, markerscale=3, loc='best', frameon=False)
        ax.set_title(f'{rep}: true label', fontsize=9)
        method = 'scan_head' if rep == 'scan_enc' else 'kmeans_k20'
        a = load_assignments(city, f'{rep}__{method}')
        ax = axes[row, 1]
        if a is not None:
            cl = a['test_cluster'][idx]
            ax.scatter(Z[:, 0], Z[:, 1], s=3, c=cl, cmap='tab20', alpha=0.7)
            for k in np.unique(cl):
                cx, cy = np.median(Z[cl == k], axis=0)
                ax.text(cx, cy, str(k), fontsize=7, weight='bold')
        ax.set_title(f'{rep}: {method} cluster', fontsize=9)
        for a_ in axes[row]:
            a_.set_xticks([]), a_.set_yticks([])
    fig.suptitle(f'{city}: t-SNE of pre-segmented test windows (n={len(idx)}, rare classes kept in full)')
    fig.tight_layout()
    fig.savefig(out_dir / f'tsne_{city}.png', dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cities', nargs='+', default=['milan', 'aruba', 'cairo'])
    ap.add_argument('--methods', nargs='+', default=list(DEFAULT_METHODS))
    ap.add_argument('--m_values', nargs='+', default=['5', 'all'])
    ap.add_argument('--variants', nargs='+', default=['all', 'excl_other'])
    ap.add_argument('--tsne_reps', nargs='+', default=list(TSNE_REPRESENTATIONS))
    ap.add_argument('--tsne_points', type=int, default=4000)
    ap.add_argument('--no_tsne', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default=str(FIG_DIR))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_values = [m if m == 'all' else int(m) for m in args.m_values]
    for city in args.cities:
        fit, pool, test = load_protocol(city)
        for variant in args.variants:
            plot_confusions(city, pool, test, args.methods, m_values, variant, out_dir)
        plot_cluster_composition(city, pool, test, args.methods, out_dir)
        if not args.no_tsne:
            plot_tsne(city, fit, pool, test, args.tsne_reps, args.tsne_points, args.seed, out_dir)
        print(f'[{city}] figures saved to {out_dir}')


if __name__ == '__main__':
    main()
