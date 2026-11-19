"""
Confusion matrices and per-class F1 for the main methods (L1, continuous FL_20 protocol, all test classes).

Methods: DISCOVER = SCAN on TF+ToD (balanced ToD weighting), k-means on TF+ToD (K=50), linear probe on TF+ToD, TDOST.
Budgets: ≈250 labels (K=50 m=5 / 250 random activity labels) and all labels (K=50 m=all / all windows).
Each matrix pools the predictions of all 5 seeds; rows are normalized (recall). Classes marked * are excluded
from the masked macro F1 (too rare in pool or test).

Writes `results/final/confusion/{confusion_matrices.pdf, <city>_<budget>.png, <city>_per_class_f1.png,
per_class_f1.csv}` and the paper figure `results/final/figures/per_class_f1.pdf`.

Usage:
    python -m tist_revision.pipeline.confusion
"""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from ..baselines.data import budget_indices
from .mapping_eval import (ARTIFACTS_DIR, BASELINES_DIR, FINAL_DIR, load_protocol, map_clusters, predict_activity,
                           scored_classes)
from .rare_classes import representations

CITIES = ['milan', 'aruba', 'cairo']
SEEDS = [1, 2, 3, 4, 5]
OUT = FINAL_DIR / 'confusion'
METHODS = ['DISCOVER (SCAN)', 'k-means TF+ToD', 'Linear probe TF+ToD', 'TDOST']
CLUSTERING = {'DISCOVER (SCAN)': 'scan_k50_todwbal', 'k-means TF+ToD': 'kmeans_k50'}
PAPER_METHODS = ['DISCOVER (SCAN)', 'Linear probe TF+ToD', 'TDOST']
BUDGETS = {'250': '≈250 labels (K=50, m=5 / 250 random labels)', 'all': 'all labels (K=50, m=all / all windows)'}
MAX_FULL = 20000
UNMAPPED = '(unmapped)'


def clustering_preds(city, run, seed, budget, pool_labels, n_test):
    z = np.load(ARTIFACTS_DIR / 'assignments' / f'{city}__tf_s{seed}+tod__{run}.npz')
    assert len(z['pool_cluster']) == len(pool_labels) and len(z['test_cluster']) == n_test
    mapping = map_clusters(z['pool_cluster'], z['pool_conf'], pool_labels, 5 if budget == '250' else 'all')
    return np.array([mapping.get(c, UNMAPPED) for c in z['test_cluster']])


def probe_preds(X_pool, X_test, y_pool, seed, budget):
    if budget == 'all':
        idx = np.random.default_rng(seed).choice(len(y_pool), size=min(MAX_FULL, len(y_pool)), replace=False)
    else:
        idx = np.concatenate(budget_indices(y_pool, budget, seed))
    scaler = StandardScaler().fit(X_pool[idx])
    clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X_pool[idx]), y_pool[idx])
    return predict_activity(clf.predict_proba(scaler.transform(X_test)), clf.classes_)


def collect(city):
    """{(method, budget): [(y_true, y_pred) per seed]}"""
    fit, pool, test = load_protocol(city, level='l1')
    y_pool, y_test = pool['label'].to_numpy(), test['label'].to_numpy()
    out = {}
    for budget in BUDGETS:
        out[('TDOST', budget)] = []
        for seed in SEEDS:
            z = np.load(BASELINES_DIR / 'final' / 'tdost' / f'{city}_l1_{budget}_s{seed}.npz', allow_pickle=True)
            assert (z['y_test'] == y_test).all()
            out[('TDOST', budget)].append((y_test, z['pred'].astype(str)))
        for name, run in CLUSTERING.items():
            out[(name, budget)] = [(y_test, clustering_preds(city, run, s, budget, y_pool, len(y_test))) for s in SEEDS]
        out[('Linear probe TF+ToD', budget)] = []
    for seed in SEEDS:
        (_, (X_pool, _, X_test)), = representations(city, fit, pool, test, [f'tf_s{seed}+tod'])
        for budget in BUDGETS:
            out[('Linear probe TF+ToD', budget)].append((y_test, probe_preds(X_pool, X_test, y_pool, seed, budget)))
        print(f'  {city} probe seed {seed}', flush=True)
    return out, scored_classes(y_pool, y_test)


def confusion(runs, rows, cols):
    r, c = {l: i for i, l in enumerate(rows)}, {l: i for i, l in enumerate(cols)}
    cm = np.zeros((len(rows), len(cols)))
    for y, p in runs:
        np.add.at(cm, ([r[l] for l in y], [c[l] for l in p]), 1)
    return cm


def draw(ax, cm, rows, cols, masked, title):
    support = cm.sum(1, keepdims=True)
    rec = np.divide(cm, support, out=np.zeros_like(cm), where=support > 0)
    ax.imshow(rec, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = rec[i, j]
            if v >= 0.05 or (i == j and v > 0):
                ax.text(j, i, f'{100 * v:.0f}', ha='center', va='center', fontsize=7,
                        color='white' if v > 0.6 else 'black', fontweight='bold' if i == j else 'normal')
    n = len(SEEDS)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{l}{"" if l in masked else "*"} ({support[i, 0] / n:.0f})' for i, l in enumerate(rows)],
                       fontsize=7.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=55, ha='right', fontsize=7.5)
    ax.set_xlabel('predicted', fontsize=8)
    ax.set_title(title, fontsize=9.5)
    for k in range(min(len(rows), len(cols))):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, ec='0.35', lw=0.5))


def per_class_f1(runs, classes):
    scores = np.array([f1_score(y, p, labels=classes, average=None, zero_division=0) for y, p in runs])
    return scores.mean(0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f1_rows = []
    with PdfPages(OUT / 'confusion_matrices.pdf') as pdf:
        for city in CITIES:
            print(f'{city}...', flush=True)
            preds, masked = collect(city)
            y_test = next(iter(preds.values()))[0][0]
            rows = [l for l, _ in Counter(y_test).most_common()]
            extra = sorted({l for runs in preds.values() for _, p in runs for l in np.unique(p)} - set(rows))
            cols = rows + extra
            for budget, btitle in BUDGETS.items():
                fig, axes = plt.subplots(2, 2, figsize=(15, 13.5))
                for ax, method in zip(axes.flat, METHODS):
                    runs = preds[(method, budget)]
                    w = np.mean([f1_score(y, p, average='weighted', zero_division=0) for y, p in runs])
                    m = np.mean([f1_score(y, p, labels=masked, average='macro', zero_division=0) for y, p in runs])
                    draw(ax, confusion(runs, rows, cols), rows, cols, masked,
                         f'{method}\nweighted F1 {w:.2f} · masked macro F1 {m:.2f}')
                fig.suptitle(f'{city.capitalize()} — L1, {btitle}\nrow-normalized recall (%), 5 seeds pooled; '
                             f'(n) = test windows per seed; * = rare class, not in masked macro F1', fontsize=11)
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                pdf.savefig(fig)
                fig.savefig(OUT / f'{city}_{budget}.png', dpi=110)
                plt.close(fig)
            settings = [(m, b) for b in BUDGETS for m in METHODS]
            f1 = np.array([per_class_f1(preds[s], rows) for s in settings]).T
            for i, l in enumerate(rows):
                for (m, b), v in zip(settings, f1[i]):
                    f1_rows.append({'city': city, 'label': l, 'masked_in': l in masked,
                                    'n_test': int((y_test == l).sum()), 'method': m, 'budget': b, 'f1': v})
            fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 2.2))
            ax.imshow(f1, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            for i in range(len(rows)):
                for j in range(len(settings)):
                    ax.text(j, i, f'{f1[i, j]:.2f}', ha='center', va='center', fontsize=8)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([f'{l}{"" if l in masked else "*"} ({(y_test == l).sum()})' for l in rows], fontsize=8)
            ax.set_xticks(range(len(settings)))
            ax.set_xticklabels([f'{m}\n{"≈250" if b == "250" else "all"}' for m, b in settings], fontsize=7.5,
                               rotation=30, ha='right')
            ax.axvline(len(METHODS) - 0.5, color='k', lw=1.2)
            ax.set_title(f'{city.capitalize()} — per-class F1 (L1, mean over 5 seeds); left: ≈250 labels, right: all',
                         fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(OUT / f'{city}_per_class_f1.png', dpi=110)
            plt.close(fig)
    pd.DataFrame(f1_rows).to_csv(OUT / 'per_class_f1.csv', index=False)
    paper_figure(pd.DataFrame(f1_rows))
    print(f'Saved {OUT}')


def paper_figure(f1):
    """One heatmap per house: L1 classes x (DISCOVER, probe, TDOST) at ≈250 and all labels."""
    settings = [(m, b) for b in BUDGETS for m in PAPER_METHODS]
    short = {'DISCOVER (SCAN)': 'DISCOVER', 'Linear probe TF+ToD': 'Probe', 'TDOST': 'TDOST'}
    heights = [f1[f1.city == c].label.nunique() for c in CITIES]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.32 * max(heights) + 1.8),
                             gridspec_kw={'width_ratios': [1, 1, 1]})
    for ax, city in zip(axes, CITIES):
        d = f1[f1.city == city]
        labels = list(dict.fromkeys(d.sort_values('n_test', ascending=False).label))
        M = np.array([[d[(d.label == l) & (d.method == m) & (d.budget == b)].f1.iloc[0] for m, b in settings]
                      for l in labels])
        ax.imshow(M, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.5)
        masked = set(d[d.masked_in].label)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([f'{l.replace("_", " ")}{"" if l in masked else "*"}' for l in labels], fontsize=7)
        ax.set_xticks(range(len(settings)))
        ax.set_xticklabels([f'{short[m]}\n{"≈250" if b == "250" else "all"}' for m, b in settings], fontsize=6.5)
        ax.axvline(len(PAPER_METHODS) - 0.5, color='k', lw=1)
        ax.set_title(city.capitalize(), fontsize=10)
    fig.tight_layout()
    figdir = FINAL_DIR / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / 'per_class_f1.pdf', bbox_inches='tight')
    fig.savefig(figdir / 'per_class_f1.png', bbox_inches='tight', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
