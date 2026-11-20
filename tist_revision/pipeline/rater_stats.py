"""Re-analysis of the v1 human annotation study (Section 5) from the raw rater file.

Inputs (v1 repo): evals/cluster_labels_from_raters.csv (city, task_code, rater slot, cluster, label, label_level_up)
and rater_samples/<city>/encoded_samples/<task_code>.csv (event-level CASAS labels of each replayed window).

    python -m tist_revision.pipeline.rater_stats [--recompute]   # bootstrap CIs take ~3 min; cached otherwise
"""
from pathlib import Path
from collections import Counter
import itertools
import re

import numpy as np
import pandas as pd

from .mapping_eval import TEMP_DIR

V1_DIR = Path.home() / 'code' / 'har' / 'discover-v1'
OUT = TEMP_DIR / 'rater_stats'
CITIES = ['milan', 'aruba', 'cairo']
N_BOOT = 2000


def fleiss_items(df, col):
    """Fleiss' kappa with each replayed window as an item (2 ratings, rotating raters)."""
    items = [g[col].tolist() for _, g in df.groupby('task_code') if len(g) == 2]
    cats = sorted({c for it in items for c in it})
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(items), len(cats)))
    for i, it in enumerate(items):
        for c in it:
            m[i, idx[c]] += 1
    n = 2
    p_i = ((m ** 2).sum(1) - n) / (n * (n - 1))
    p_j = m.sum(0) / m.sum()
    pe = (p_j ** 2).sum()
    return (p_i.mean() - pe) / (1 - pe)


def kripp_alpha(df, col):
    """Krippendorff's alpha (nominal), pairable values only."""
    units = [g[col].tolist() for _, g in df.groupby('task_code') if len(g) >= 2]
    o = Counter()
    for u in units:
        for a, b in itertools.permutations(range(len(u)), 2):
            o[(u[a], u[b])] += 1 / (len(u) - 1)
    n_c = Counter()
    for (a, _), v in o.items():
        n_c[a] += v
    n = sum(n_c.values())
    d_o = sum(v for (a, b), v in o.items() if a != b) / n
    d_e = (n * n - sum(v * v for v in n_c.values())) / (n * (n - 1))
    return 1 - d_o / d_e


def fleiss_clusters(df, col):
    """v1 cluster-agreement statistic: each cluster is one item with its ~10 labels."""
    t = pd.crosstab(df['cluster'], df[col]).to_numpy().astype(float)
    n_i = t.sum(1)
    p_i = ((t ** 2).sum(1) - n_i) / (n_i * (n_i - 1))
    p_j = t.sum(0) / t.sum()
    pe = (p_j ** 2).sum()
    return (p_i.mean() - pe) / (1 - pe)


def pair_agreement(df, col):
    """Agreement of two labels drawn from: same window / other window same cluster / other cluster."""
    lab = df[[col, 'task_code', 'cluster']].to_numpy()
    same_w, same_c, diff_c = [], [], []
    for i, j in itertools.combinations(range(len(lab)), 2):
        eq = lab[i][0] == lab[j][0]
        if lab[i][1] == lab[j][1]:
            same_w.append(eq)
        elif lab[i][2] == lab[j][2]:
            same_c.append(eq)
        else:
            diff_c.append(eq)
    return np.mean(same_w), np.mean(same_c), np.mean(diff_c)


def boot_ci(df, fn, rng):
    """95% CI resampling clusters (windows of a cluster are not independent)."""
    clusters = df['cluster'].unique()
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(clusters, len(clusters))
        parts = [df[df['cluster'] == c].assign(cluster=k, task_code=lambda d, k=k: d['task_code'] + f'_{k}')
                 for k, c in enumerate(pick)]
        try:
            vals.append(fn(pd.concat(parts)))
        except ZeroDivisionError:
            continue
    return np.nanpercentile(vals, [2.5, 97.5])


def window_casas_label(city, task):
    path = V1_DIR / 'rater_samples' / city / 'encoded_samples' / f'{task}.csv'
    if not path.exists():
        return 'unknown'
    ev = pd.read_csv(path, usecols=['first_activity'])
    acts = ev['first_activity'].fillna('no_activity')
    acts = acts[acts.str.lower() != 'no_activity']
    return acts.value_counts().idxmax() if len(acts) else 'No_Activity'


def casas_l1(city, raw):
    """Cairo resident labels merged as in the evaluation protocol ('R1 wake' -> 'wake')."""
    if raw == 'No_Activity':
        return 'No CASAS label'
    if city == 'cairo':
        raw = re.sub(r'^R[12] ', '', raw)
    return raw.replace('_', ' ')


_SUB = [
    (r'^Movement through Living areas: Kitchen \+ Living \+ Dining - All three rooms$', 'Kitchen + living + dining'),
    (r'^Movement through Living areas: Kitchen \+ Living \+ Dining - (\w+) \+ (\w+)$',
     lambda m: f'{m.group(1)} + {m.group(2).lower()}'),
    (r'^Simultaneous activity in two or more rooms.*', 'Simultaneous activity, 2+ rooms'),
    (r'^Walking through the House.*', 'Walking through house, 3+ rooms'),
    (r'^Walking between:.* - Walking from (.+) to (.+)$', lambda m: f'{m.group(1).capitalize()} → {m.group(2)}'),
    (r'^(.+?) - \("all over"\)$', r'\1 (all over)'),
    (r'^(.+?) - (?:Entering|Leaving) the house through (\w+) door$',
     lambda m: f'{m.group(1)}: {"entering" if "Entering" in m.group(0) else "leaving"} house'),
    (r'^(.+?) - Walking back and forth through the hallway$', r'\1: back and forth'),
    (r'^(.+?) - (Entering|Leaving|Movement inside|Movement in) (?:the )?.*$',
     lambda m: f'{m.group(1)}: {m.group(2).replace("Movement inside", "inside").replace("Movement in", "inside").lower()}'),
    (r'^(.+?) - Movement near (.+)$', lambda m: f'{m.group(1)}: near {m.group(2).strip().lower()}'),
    (r'^(.+?) - Sitting (.+)$', lambda m: f'{m.group(1)}: sitting {m.group(2)}'),
]
_ROOM = {'Kitchen area': 'Kitchen', 'Office area': 'Office', 'Front Door Hallway': 'Front-door hallway',
         'Back Door Hallway': 'Back-door hallway', 'Garage Hallway': 'Garage hallway', 'Living room': 'Living room',
         'Dining Room': 'Dining room', 'Guest Bathroom': 'Guest bathroom', 'Master Bathroom': 'Master bathroom',
         'Guest Bedroom': 'Guest bedroom', 'Master Bedroom': 'Master bedroom', 'computer/desk': 'desk'}


def short(label):
    s = re.sub(r'\s+', ' ', re.sub(r'^\d\+? room\(s\):\s*', '', label.strip())).replace('Motion in ', '')
    for pat, rep in _SUB:
        if re.match(pat, s):
            s = re.sub(pat, rep, s)
            break
    for k, v in _ROOM.items():
        s = s.replace(k, v)
    return s


HOUSE = {'milan': 'Milan', 'aruba': 'Aruba', 'cairo': 'Cairo'}


def plot_agreement(stats, clus, path):
    """(a) agreement of two labels by source; (b) per-cluster majority share."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={'width_ratios': [1, 1.25]})
    kinds = [('agree_same_window', 'Same window\n(two raters)'),
             ('agree_same_cluster_other_window', 'Same cluster,\nother window'),
             ('agree_other_cluster', 'Different\nclusters')]
    colors = {'milan': '#1f77b4', 'aruba': '#ff7f0e', 'cairo': '#2ca02c'}
    w = 0.26
    for i, city in enumerate(CITIES):
        s = stats[stats['city'] == city].set_index('level')
        x = np.arange(len(kinds)) + (i - 1) * w
        a.bar(x, [s.loc['original', k] for k, _ in kinds], w, color=colors[city], label=HOUSE[city])
        a.bar(x, [s.loc['level_up', k] - s.loc['original', k] for k, _ in kinds], w,
              bottom=[s.loc['original', k] for k, _ in kinds], color=colors[city], alpha=0.35)
    a.set_xticks(range(len(kinds)), [t for _, t in kinds])
    a.set_ylim(0, 1)
    a.set_ylabel('P(two labels agree)')
    a.set_title('(a) Label agreement by source', loc='left', fontsize=8.5)
    a.legend(frameon=False, fontsize=7, loc='upper right')
    a.text(0.99, 0.62, 'light: gain at\nparent level', transform=a.transAxes, ha='right', fontsize=6.5, color='0.35')
    for i, city in enumerate(CITIES):
        c = clus[clus['city'] == city].sort_values(['majority_share', 'majority_share_level_up'], ascending=False)
        x = np.arange(len(c)) + i * (len(c) + 3)
        b.vlines(x, c['majority_share'], c['majority_share_level_up'], color=colors[city], alpha=0.35, lw=1)
        b.scatter(x, c['majority_share_level_up'], s=9, facecolors='none', edgecolors=colors[city], lw=0.7)
        b.scatter(x, c['majority_share'], s=9, color=colors[city])
        b.text(x.mean(), 0.03, HOUSE[city], ha='center', fontsize=7.5)
    b.axhline(0.5, color='0.6', lw=0.6, ls='--')
    b.set_xticks([])
    b.set_ylim(0, 1.04)
    b.set_ylabel('Share of the 10 labels')
    b.set_title('(b) Majority label per cluster (filled: PDL, open: parent)', loc='left', fontsize=8.5)
    for ax in (a, b):
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


def plot_pdl_vs_casas(df, path):
    """Rated windows: PDL label (rows) x CASAS label (columns), one panel per house."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    d = df[df['casas'] != 'unknown'].assign(pdl=lambda x: x['label'].map(short),
                                            cas=lambda x: [casas_l1(c, r) for c, r in zip(x['city'], x['casas'])])
    tabs = {}
    for city in CITIES:
        t = pd.crosstab(d.loc[d['city'] == city, 'pdl'], d.loc[d['city'] == city, 'cas']) / 2
        cols = [c for c in t.sum().sort_values(ascending=False).index if c != 'No CASAS label']
        cols += ['No CASAS label'] if 'No CASAS label' in t else []
        t = t[cols]
        order = pd.DataFrame({'col': t.idxmax(axis=1).map(cols.index), 'n': -t.max(axis=1)})
        t = t.loc[order.sort_values(['col', 'n']).index]
        tabs[city] = t
    heights = [len(t) for t in tabs.values()]
    fig, axes = plt.subplots(len(CITIES), 1, figsize=(5.6, 0.13 * sum(heights) + 2.4),
                             gridspec_kw={'height_ratios': heights})
    for ax, (city, t) in zip(axes, tabs.items()):
        ax.imshow(t.to_numpy(), cmap='Blues', aspect='auto', vmin=0, vmax=max(6, t.to_numpy().max()))
        for (i, j), v in np.ndenumerate(t.to_numpy()):
            if v:
                ax.text(j, i, f'{v:g}', ha='center', va='center', fontsize=5.5,
                        color='white' if v > 0.6 * t.to_numpy().max() else 'black')
        ax.set_yticks(range(len(t)), t.index)
        ax.set_xticks(range(t.shape[1]), t.columns, rotation=35, ha='right')
        if 'No CASAS label' in t:
            ax.axvline(t.shape[1] - 1.5, color='0.3', lw=0.8)
        ax.set_title(f'{HOUSE[city]} ({int(t.to_numpy().sum())} rated windows)', loc='left', fontsize=8)
        ax.tick_params(length=0)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


def kappa_table(stats, path):
    def cell(r, k):
        lo, hi = r[f'{k}_ci']
        return f'{r[k]:.2f} {{\\scriptsize [{lo:.2f}, {hi:.2f}]}}'
    lines = [
        '% Generated by tist_revision/pipeline/rater_stats.py; do not edit by hand.',
        r'\begin{table}[ht]', r'\centering', r'\small',
        r'\begin{tabular}{l cc cc}', r'\toprule',
        r' & \multicolumn{2}{c}{Inter-rater $\kappa$ (per window)} & \multicolumn{2}{c}{Cluster agreement $\kappa$} \\',
        r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}',
        r'House & PDL label & Parent label & PDL label & Parent label \\', r'\midrule']
    for city in CITIES:
        o = stats[(stats['city'] == city) & (stats['level'] == 'original')].iloc[0]
        u = stats[(stats['city'] == city) & (stats['level'] == 'level_up')].iloc[0]
        lines.append(f'{HOUSE[city]} & {cell(o, "fleiss_window")} & {cell(u, "fleiss_window")} & '
                     f'{cell(o, "fleiss_cluster")} & {cell(u, "fleiss_cluster")} \\\\')
    caption = (
        r'\tist[R2-2.12]{Agreement of the human annotations (100 windows per house, two annotations each, '
        r"95\% cluster-bootstrap intervals in brackets). \emph{Inter-rater}: Fleiss' $\kappa$ with each window as "
        r"one item, which allows a different pair of raters per window (Krippendorff's $\alpha$ gives the same values). "
        r"\emph{Cluster agreement}: Fleiss' $\kappa$ "
        r'with each cluster as one item and its 10 labels (5 windows $\times$ 2 annotations) as ratings; it therefore also counts differences between windows of the '
        r'same cluster. \emph{Parent label}: labels merged one level up the taxonomy (e.g., \emph{movement near '
        r'fridge} $\rightarrow$ \emph{movement in kitchen}).}')
    lines += [r'\bottomrule', r'\end{tabular}', r'\vspace{0.5em}', rf'\caption{{{caption}}}',
              r'\vspace*{-1.5em}', r'\label{tab:kappa_scores}', r'\end{table}', '']
    path.write_text('\n'.join(lines))


def main(recompute=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if not recompute and (OUT / 'agreement.csv').exists() and (OUT / 'ratings.csv').exists():
        stats = pd.read_csv(OUT / 'agreement.csv', converters={k: eval for k in ('fleiss_window_ci',
                                                                                  'fleiss_cluster_ci')})
        report(stats, pd.read_csv(OUT / 'clusters.csv'), pd.read_csv(OUT / 'ratings.csv'))
        return
    rng = np.random.default_rng(0)
    df = pd.read_csv(V1_DIR / 'evals' / 'cluster_labels_from_raters.csv')
    df['casas'] = [window_casas_label(c, t) for c, t in zip(df['city'], df['task_code'])]
    df.to_csv(OUT / 'ratings.csv', index=False)

    rows, clus_rows = [], []
    for city in CITIES:
        d = df[df['city'] == city]
        w = d.pivot_table(index='task_code', columns='rater', values='label', aggfunc='first').dropna()
        wu = d.pivot_table(index='task_code', columns='rater', values='label_level_up', aggfunc='first').dropna()
        for col, tag in [('label', 'original'), ('label_level_up', 'level_up')]:
            ww = w if tag == 'original' else wu
            counts = d[col].value_counts(normalize=True)
            sw, sc, dc = pair_agreement(d, col)
            rows.append(dict(
                city=city, level=tag, windows=d['task_code'].nunique(), ratings=len(d),
                paired_windows=len(ww), categories_used=d[col].nunique(),
                other_rate=d[col].str.contains('Other', case=False).mean(),
                top_category_share=counts.iloc[0], top_category=short(counts.index[0]),
                exact_agreement=(ww['r1'] == ww['r2']).mean(),
                fleiss_window=fleiss_items(d, col),
                fleiss_window_ci=boot_ci(d, lambda x: fleiss_items(x, col), rng).round(3).tolist(),
                kripp_alpha=kripp_alpha(d, col),
                fleiss_cluster=fleiss_clusters(d, col),
                fleiss_cluster_ci=boot_ci(d, lambda x: fleiss_clusters(x, col), rng).round(3).tolist(),
                agree_same_window=sw, agree_same_cluster_other_window=sc, agree_other_cluster=dc,
            ))
        for c, g in d.groupby('cluster'):
            maj = g['label'].value_counts()
            maj_up = g['label_level_up'].value_counts()
            cas = g.drop_duplicates('task_code')['casas'].value_counts()
            clus_rows.append(dict(
                city=city, cluster=c, n_labels=len(g), majority=short(maj.index[0]),
                majority_share=maj.iloc[0] / len(g), majority_share_level_up=maj_up.iloc[0] / len(g),
                casas_majority=cas.index[0], casas_purity=cas.iloc[0] / cas.sum(),
                casas_labels=', '.join(f'{k}:{v}' for k, v in cas.items()),
            ))

    stats = pd.DataFrame(rows)
    clus = pd.DataFrame(clus_rows)
    stats.to_csv(OUT / 'agreement.csv', index=False)
    clus.to_csv(OUT / 'clusters.csv', index=False)

    report(stats, clus, df)


def report(stats, clus, df):
    pdl_casas = df.assign(label=df['label'].map(short), casas=[casas_l1(c, r) for c, r in zip(df['city'], df['casas'])])
    pdl_casas = pdl_casas.groupby(['city', 'label', 'casas']).size().div(2).rename('windows').reset_index()
    pdl_casas.to_csv(OUT / 'pdl_vs_casas.csv', index=False)
    plot_agreement(stats, clus, OUT / 'annotation_agreement')
    plot_pdl_vs_casas(df, OUT / 'pdl_vs_casas')
    kappa_table(stats, OUT / 'kappa_scores.tex')

    pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.max_colwidth', 40)
    print(stats.round(3).to_string(index=False))
    print(clus.groupby('city')[['majority_share', 'majority_share_level_up', 'casas_purity']]
          .agg(['mean', 'min']).round(2))
    print('clusters with majority share < 0.5:\n',
          clus[clus['majority_share'] < 0.5][['city', 'cluster', 'majority', 'majority_share',
                                               'majority_share_level_up', 'casas_labels']].to_string(index=False))
    print(f'outputs: {OUT}')


if __name__ == '__main__':
    import sys
    main(recompute='--recompute' in sys.argv)
