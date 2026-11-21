"""
Section 5 figures and tables from the final annotation study (`annotation/packets.py score --returned final`):
  - kappa_scores.tex: inter-rater and cluster agreement (top-confidence windows), 95% cluster-bootstrap CIs.
  - annotation_agreement.{svg,pdf,png}: (a) agreement of two labels by source, (b) majority share per cluster.
  - casas_to_pdl.{svg,pdf,png}: for every CASAS activity, the patterns raters gave to the clusters it dominates.
  - tsne_milan.{svg,pdf,png}: t-SNE of the Milan clustering features with four annotated clusters on the floor plan.
  - stats.json: numbers quoted in the text.
Figures are SVG printed to PDF with headless Chrome (same style as the Section 6 figure).

  - random_windows.tex (appendix): agreement on the two random windows per cluster and label propagation.
  - cluster_summary.tex (appendix): majority label, votes and dataset-activity composition of every cluster.
  - pdl_vs_casas.{pdf,png} (appendix): rated windows, pattern label x dataset activity, one panel per house.
The paper study (all four houses) writes to results/final/section5; the others are previews under results/temp.

Usage:
    python -m tist_revision.pipeline.section5_figures                  # paper: Milan, Aruba, Cairo, Orange4Home
    python -m tist_revision.pipeline.section5_figures --study casas    # preview, CASAS houses only
"""
import base64
import io
import json

import numpy as np
import pandas as pd

from ..annotation.packets import FLOOR_PLANS, METADATA, OUT_ROOT, load_city
from . import rater_stats as rs
from .mapping_eval import ARTIFACTS_DIR, DATA_ROOT, FINAL_DIR, TEMP_DIR, load_protocol
from .report_final import FIG_FONT, svg_to_pdf, text_width

OUT = FINAL_DIR / 'section5'
STUDY = OUT_ROOT / 'v2_k20'
RUN = 'tf_s1+tod__scan_k20_todwbal'
RETURNED = 'final'
CITIES = ['milan', 'aruba', 'cairo', 'orange']
TSNE_CACHE = ARTIFACTS_DIR / 'tsne'
STUDIES = {  # study -> (annotation study folder(s), returned/ subfolder, houses, output folder)
    'paper': ([('v2_k20', 'final'), ('orange_k20', '')], None, ['milan', 'aruba', 'cairo', 'orange'],
              FINAL_DIR / 'section5'),
    'casas': ('v2_k20', 'final', ['milan', 'aruba', 'cairo'], TEMP_DIR / 'section5'),
    'orange': ('orange_k20', '', ['orange'], TEMP_DIR / 'section5_orange'),
}
HOUSE = {'milan': 'Milan', 'aruba': 'Aruba', 'cairo': 'Cairo', 'orange': 'Orange4Home'}
HOUSE_COLOR = {'milan': '#2563eb', 'aruba': '#d97706', 'cairo': '#059669', 'orange': '#7c3aed'}
MIN_WINDOWS = 5  # fewer windows than this: no agreement statistic for the subset
INK, SUB, GRID = '#0f172a', '#334155', '#e2e8f0'
UNLABELED = 'No_Activity'
UNLABELED_GROUP = 0.75  # clusters with at least this share of unlabeled windows go under 'No CASAS label'

PDL = {
    'mbr_any': 'Master bedroom', 'mbr_bed': 'Bed', 'gbr_any': 'Guest bedroom',
    'mba_any': 'Master bathroom', 'mba_enter': 'Entering master bathroom', 'mba_leave': 'Leaving master bathroom',
    'mba_inside': 'Inside master bathroom', 'gba_any': 'Guest bathroom', 'gba_enter': 'Entering guest bathroom',
    'gba_leave': 'Leaving guest bathroom', 'gba_inside': 'Inside guest bathroom',
    'off_any': 'Office', 'off_desk': 'Office desk', 'off_chair': 'Office armchair',
    'kit_any': 'Kitchen', 'kit_stove': 'Stove', 'kit_fridge': 'Fridge', 'kit_med': 'Medicine cabinet',
    'liv_any': 'Living room', 'liv_couch': 'Couch', 'liv_chair': 'Living-room armchair', 'din_any': 'Dining area',
    'fd_hall': 'Front-door hallway', 'fd_enter': 'Entering via front door', 'fd_leave': 'Leaving via front door',
    'bd_hall': 'Back-door hallway', 'bd_enter': 'Entering via back door', 'bd_leave': 'Leaving via back door',
    'gd_hall': 'Garage-door hallway', 'gd_enter': 'Entering via garage door', 'gd_leave': 'Leaving via garage door',
    'la_all': 'Kitchen + living + dining', 'la_kd': 'Kitchen + dining', 'la_kl': 'Kitchen + living',
    'la_ld': 'Living + dining', 'bo_b2o': 'Bedroom → office', 'bo_o2b': 'Office → bedroom',
    'bl_b2l': 'Bedroom → living spaces', 'bl_l2b': 'Living spaces → bedroom', 'ol_o2l': 'Office → living spaces',
    'ol_l2o': 'Living spaces → office', 'walk3': 'Walking through the house', 'simul': 'Simultaneous movement',
    'none': 'None of the above',
    'br_any': 'Bedroom', 'ba_any': 'Bathroom', 'ba_shower': 'Shower', 'toilet': 'Toilet',
    'liv_table': 'Living-room table',
}


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def casas_name(label):
    return 'No activity label' if label == UNLABELED else label.replace('_', ' ')


def write_figure(svg, name):
    from subprocess import run
    path = OUT / f'{name}.svg'
    path.write_text(svg)
    svg_to_pdf(path)
    run(['rsvg-convert', '-z', '2', '-o', str(OUT / f'{name}.png'), str(path)], check=True)


# ---------------------------------------------------------------- data

def load():
    sources = STUDY if isinstance(STUDY, list) else [(STUDY, RETURNED)]
    r = pd.concat([pd.read_csv(f / 'scores' / ret / 'ratings.csv') for f, ret in sources], ignore_index=True)
    stats = pd.concat([pd.read_csv(f / 'scores' / ret / 'agreement.csv',
                                   converters={k: eval for k in ('fleiss_window_ci', 'fleiss_cluster_ci')})
                       for f, ret in sources], ignore_index=True)
    windows = {}
    for city in CITIES:
        w, _ = load_city(city)
        z = np.load(ARTIFACTS_DIR / 'assignments' / f'{city}__{RUN}.npz')
        w['cluster'], w['conf'] = z['pool_cluster'], z['pool_conf']
        windows[city] = w
    return r, stats, windows


def cluster_table(r, windows):
    """One row per cluster: majority pattern of its top windows and the CASAS composition of all its windows."""
    rows = []
    top = r[r['stratum'] == 'top']
    for (city, c), g in top.groupby(['city', 'cluster']):
        vc, vu = g['label'].value_counts(), g['label_level_up'].value_counts()
        w = windows[city]
        cw = w[w['cluster'] == c]
        lab = cw[cw['casas_last'] != UNLABELED]['casas_last'].value_counts()
        unl = (cw['casas_last'] == UNLABELED).mean()
        rows.append(dict(city=city, cluster=c, pattern=vc.index[0], share=vc.iloc[0] / len(g),
                         share_parent=vu.iloc[0] / len(g), size=len(cw), size_frac=len(cw) / len(w),
                         unlabeled=unl, casas=lab.index[0] if len(lab) else UNLABELED,
                         casas_purity=lab.iloc[0] / lab.sum() if len(lab) else np.nan,
                         group=UNLABELED if unl >= UNLABELED_GROUP or not len(lab) else lab.index[0]))
    return pd.DataFrame(rows)


def text_stats(r, stats, clus):
    out = {}
    top = r[r['stratum'] == 'top']
    for city, d in top.groupby('city'):
        w = d.pivot_table(index='task_code', columns='rater', values='label', aggfunc='first')
        u = d[d['casas'] == UNLABELED]
        s = stats[(stats.city == city)].set_index('level')
        c = clus[clus.city == city]
        common = c['pattern'].value_counts()
        rest = d[~d['cluster'].isin(c[c['pattern'] == common.index[0]]['cluster'])]

        def kappa(x):
            return float(rs.fleiss_items(x, 'label')) if x['task_code'].nunique() >= MIN_WINDOWS else None
        out[city] = dict(
            kappa=s.loc['original', 'fleiss_window'], kappa_parent=s.loc['level_up', 'fleiss_window'],
            alpha=s.loc['original', 'kripp_alpha'], alpha_parent=s.loc['level_up', 'kripp_alpha'],
            exact=float((w['r1'] == w['r2']).mean()), none_rate=float((d['label'] == 'none').mean()),
            categories=int(d['label'].nunique()), categories_parent=int(d['label_level_up'].nunique()),
            unlabeled_windows=float((d.drop_duplicates('task_code')['casas'] == UNLABELED).mean()),
            kappa_unlabeled=kappa(u), none_unlabeled=float((u['label'] == 'none').mean()) if len(u) else None,
            majority_share=float(c['share'].mean()), majority_share_parent=float(c['share_parent'].mean()),
            weak_clusters=int((c['share'] < 0.5).sum()), weak_parent=int((c['share_parent'] < 0.5).sum()),
            agree_same_window=s.loc['original', 'agree_same_window'],
            agree_same_cluster=s.loc['original', 'agree_same_cluster_other_window'],
            agree_other_cluster=s.loc['original', 'agree_other_cluster'],
            # clusters that share one majority pattern cannot be told apart by their label
            distinct_patterns=int(len(common)), top_pattern=common.index[0], top_pattern_clusters=int(common.iloc[0]),
            top_pattern_size=float(c[c['pattern'] == common.index[0]]['size_frac'].sum()),
            kappa_other_clusters=kappa(rest),
        )
    return out


# ---------------------------------------------------------------- table

def kappa_table(stats, path, avg=None):
    def cell(row, k):
        lo, hi = row[f'{k}_ci']
        return f'{row[k]:.2f} {{\\scriptsize [{lo:.2f}, {hi:.2f}]}}'
    lines = [
        '% Generated by tist_revision/pipeline/section5_figures.py; do not edit by hand.',
        r'\begin{table}[ht]', r'\centering', r'\small', r'\begin{tabular}{l cc cc}', r'\toprule',
        r' & \multicolumn{2}{c}{Inter-rater $\kappa$ (per window)} & \multicolumn{2}{c}{Cluster agreement $\kappa$} \\',
        r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}', r'House & PDL label & Parent label & PDL label & Parent label \\',
        r'\midrule']
    for city in CITIES:
        o = stats[(stats.city == city) & (stats.level == 'original')].iloc[0]
        u = stats[(stats.city == city) & (stats.level == 'level_up')].iloc[0]
        lines.append(f'{HOUSE[city]} & {cell(o, "fleiss_window")} & {cell(u, "fleiss_window")} & '
                     f'{cell(o, "fleiss_cluster")} & {cell(u, "fleiss_cluster")} \\\\')
    if avg is not None:
        lines += [r'\midrule', r'\textit{Average} & ' + ' & '.join(
            cell(avg[level], k) for k in ('fleiss_window', 'fleiss_cluster') for level in ('original', 'level_up'))
            + r' \\']
    caption = (
        r'\tist[R2-2.12]{Agreement of the human annotations (100 windows per house, two annotations each, '
        r"95\% cluster-bootstrap intervals in brackets; for the average, the houses are resampled independently). "
        r"\emph{Inter-rater}: Fleiss' $\kappa$ with each window as "
        r"one item, which allows a different pair of raters per window (Krippendorff's $\alpha$ differs by at most 0.01). "
        r"\emph{Cluster agreement}: Fleiss' $\kappa$ with each cluster as one item and its 10 labels (5 windows "
        r'$\times$ 2 annotations) as ratings; it therefore also counts differences between windows of the same '
        r'cluster. \emph{Parent label}: labels merged one level up the taxonomy (e.g., \emph{movement near '
        r'fridge} $\rightarrow$ \emph{movement in kitchen}).}')
    lines += [r'\bottomrule', r'\end{tabular}', r'\vspace{0.5em}', rf'\caption{{{caption}}}', r'\vspace*{-1.5em}',
              r'\label{tab:kappa_scores}', r'\end{table}', '']
    path.write_text('\n'.join(lines))


def _fleiss(m):
    """Fleiss' kappa from an items x categories count matrix (items may have different numbers of ratings)."""
    n = m.sum(1)
    p_i = ((m ** 2).sum(1) - n) / (n * (n - 1))
    p_j = m.sum(0) / m.sum()
    pe = (p_j ** 2).sum()
    return (p_i.mean() - pe) / (1 - pe)


class HouseBoot:
    """Per-cluster count matrices of one house's ratings, so that a cluster-bootstrap resample is a concatenation
    of precomputed rows. Statistics per label level: window kappa on top windows (as rater_stats.fleiss_items),
    cluster kappa (as rater_stats.fleiss_clusters), window kappa on random windows, and the share of random-window
    annotations matching the cluster's majority label from its top windows."""

    def __init__(self, d):
        self.clusters = sorted(d['cluster'].unique())
        self.parts = {}
        for col in ('label', 'label_level_up'):
            cats = {c: i for i, c in enumerate(sorted(d[col].unique()))}

            def items(x):
                g = [v for _, v in x.groupby('task_code')[col] if len(v) == 2]
                m = np.zeros((len(g), len(cats)))
                for i, v in enumerate(g):
                    for c in v:
                        m[i, cats[c]] += 1
                return m
            per = []
            for c in self.clusters:
                x = d[d['cluster'] == c]
                top, rnd = x[x['stratum'] == 'top'], x[x['stratum'] == 'random']
                row = np.zeros(len(cats))
                for v in top[col]:
                    row[cats[v]] += 1
                maj = top[col].value_counts().index[0]
                per.append((items(top), row, items(rnd), ((rnd[col] == maj).sum(), len(rnd))))
            self.parts[col] = per

    def stats(self, pick=None):
        pick = range(len(self.clusters)) if pick is None else pick
        out = {}
        for col, tag in (('label', ''), ('label_level_up', '_parent')):
            per = [self.parts[col][k] for k in pick]
            hits, n = np.sum([p[3] for p in per], axis=0)
            out |= {f'kappa{tag}': _fleiss(np.vstack([p[0] for p in per])),
                    f'cluster{tag}': _fleiss(np.vstack([p[1] for p in per])),
                    f'kappa_random{tag}': _fleiss(np.vstack([p[2] for p in per])), f'match{tag}': hits / n}
        return out

    def boot(self, rng, n_boot=rs.N_BOOT):
        k = len(self.clusters)
        return pd.DataFrame([self.stats(rng.choice(k, k)) for _ in range(n_boot)])


def bootstraps(r, seed=0):
    rng = np.random.default_rng(seed)
    hb = {c: HouseBoot(r[r['city'] == c]) for c in CITIES}
    return {c: (h.stats(), h.boot(rng)) for c, h in hb.items()}


def average_agreement(stats, boots):
    """Mean over houses of the window and cluster kappas, with a 95% CI from resampling the clusters of every
    house independently and averaging the per-house statistics."""
    mean_boot = sum(b for _, b in boots.values()) / len(boots)
    out = {}
    for level, tag in (('original', ''), ('level_up', '_parent')):
        s = stats[stats.level == level].set_index('city').loc[CITIES]
        out[level] = {}
        for k, bk in (('fleiss_window', f'kappa{tag}'), ('fleiss_cluster', f'cluster{tag}')):
            out[level][k] = float(s[k].mean())
            out[level][f'{k}_ci'] = np.nanpercentile(mean_boot[bk], [2.5, 97.5]).tolist()
    return out


def random_windows(r, boots):
    """Appendix (R2-2.13): agreement on the random windows and how often their labels match the cluster label."""
    rows = []
    for city in CITIES:
        d = r[r['city'] == city]
        point, b = boots[city]
        row = dict(city=city, windows=int(d.loc[d['stratum'] == 'random', 'task_code'].nunique()))
        for col, tag in (('label', ''), ('label_level_up', '_parent')):
            row[f'kappa{tag}'] = float(point[f'kappa_random{tag}'])
            row[f'kappa{tag}_ci'] = np.nanpercentile(b[f'kappa_random{tag}'], [2.5, 97.5]).tolist()
            row[f'match{tag}'] = float(point[f'match{tag}'])
            row[f'match{tag}_ci'] = np.nanpercentile(b[f'match{tag}'], [2.5, 97.5]).tolist()
            row[f'match_top{tag}'] = float(rs.pair_agreement(d[d['stratum'] == 'top'], col)[1])
        rows.append(row)
    return pd.DataFrame(rows)


def random_table(rw, path):
    def cell(row, k):
        lo, hi = row[f'{k}_ci']
        return f'{row[k]:.2f} {{\\scriptsize [{lo:.2f}, {hi:.2f}]}}'
    lines = ['% Generated by tist_revision/pipeline/section5_figures.py; do not edit by hand.',
             r'\begin{tabular}{l c cc cc}', r'\toprule',
             r' & & \multicolumn{2}{c}{Inter-rater $\kappa$} & \multicolumn{2}{c}{Matches cluster label} \\',
             r'\cmidrule(lr){3-4} \cmidrule(lr){5-6}',
             r'House & Windows & PDL label & Parent label & PDL label & Parent label \\', r'\midrule']
    for _, row in rw.iterrows():
        lines.append(f'{HOUSE[row.city]} & {row.windows} & ' + ' & '.join(
            cell(row, k) for k in ('kappa', 'kappa_parent', 'match', 'match_parent')) + r' \\')
    m = rw[['kappa', 'kappa_parent', 'match', 'match_parent']].mean()
    lines += [r'\midrule', rf'\textit{{Average}} & {int(rw.windows.sum() / len(rw))} & '
              + ' & '.join(f'{m[k]:.2f}' for k in m.index) + r' \\', r'\bottomrule', r'\end{tabular}', '']
    path.write_text('\n'.join(lines))


def taxonomy_table(path):
    """Appendix: the PDL vocabulary of the labeling tool, one row per parent label with its child labels and the homes
    where it was offered (annotation/pdl_taxonomy.json)."""
    tax = json.loads((OUT_ROOT.parents[2] / 'annotation' / 'pdl_taxonomy.json').read_text())
    homes = list(HOUSE)
    offered = {h: (set(tax['houses'][h]['groups']), set(tax['houses'][h].get('exclude', []))) for h in homes}
    out = ['% Generated by tist_revision/pipeline/section5_figures.py; do not edit by hand.',
           r'\begin{tabular}{@{}>{\raggedright}p{0.2\linewidth} >{\raggedright\arraybackslash}p{0.6\linewidth} cccc@{}}', r'\toprule',
           r'Parent label & Child labels & ' + ' & '.join(HOUSE[h][0] for h in homes) + r' \\']
    for cat, cat_name in tax['categories'].items():
        out += [r'\midrule', rf'\multicolumn{{6}}{{@{{}}l}}{{\emph{{{cat_name}}}}} \\']
        for gid, g in tax['groups'].items():
            if g['category'] != cat:
                continue
            in_group = [h for h in homes if gid in offered[h][0]]
            kids = []
            for lab in (l for l in tax['labels'] if l['group'] == gid):
                where = [h for h in in_group if lab['id'] not in offered[h][1] and h in lab.get('only', homes)]
                name = lab['label'].replace('→', r'$\rightarrow$')
                kids.append(name if where == in_group else name + rf'$^{{{",".join(HOUSE[h][0] for h in where)}}}$')
            marks = ' & '.join(r'\checkmark' if h in in_group else '' for h in homes)
            out.append(rf'{g["label"]} & {"; ".join(kids)} & {marks} \\')
    out += [r'\bottomrule', r'\end{tabular}', '']
    path.write_text('\n'.join(out))


def cluster_summary_table(r, clus, path):
    """Appendix: one table per house with every cluster's majority label and its dataset-activity composition."""
    import re
    tax = json.loads((OUT_ROOT.parents[2] / 'annotation' / 'pdl_taxonomy.json').read_text())
    group = {k: v['label'] for k, v in tax['groups'].items()}
    top = r[r['stratum'] == 'top']

    def tex(x):
        return re.sub(r'([&%_#])', r'\\\1', str(x)).replace('→', r'$\rightarrow$')
    out = ['% Generated by tist_revision/pipeline/section5_figures.py; do not edit by hand.']
    for i, city in enumerate(CITIES):
        c = clus[clus.city == city].sort_values('cluster')
        out += [r'\begin{table}[p]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{4pt}',
                r'\begin{tabular}{r l c l c l r}', r'\toprule',
                r'Cluster & PDL label & Votes & Parent label & Votes & Most frequent activity (share) & Unlabeled \\',
                r'\midrule']
        for _, row in c.iterrows():
            g = top[(top.city == city) & (top.cluster == row.cluster)]
            up = g['label_level_up'].value_counts()
            parent = group.get(up.index[0], PDL.get(up.index[0], up.index[0]))
            act = '--' if row.casas == UNLABELED else f'{tex(casas_name(rs.casas_l1(city, row.casas)))} ({row.casas_purity:.0%})'
            out.append(f'{row.cluster} & {tex(PDL[row.pattern])} & {int(round(row.share * len(g)))}/{len(g)} & '
                       f'{tex(parent)} & {up.iloc[0]}/{len(g)} & {act} & {row.unlabeled:.0%} \\\\'.replace('%', r'\%'))
        caption = (rf'{HOUSE[city]}: annotated pattern of each cluster ($K{{=}}20$). \emph{{Votes}}: annotations of the '
                   r"cluster's five top windows (two per window) that chose the label; \emph{Parent label}: the label "
                   r'one level up in the taxonomy (\cref{tab:pdl_taxonomy}). \emph{Most frequent activity}: the '
                   r"dataset activity of most of the cluster's labeled training windows and its share among them; "
                   r"\emph{Unlabeled}: share of the cluster's training windows without an activity label."
                   if i == 0 else rf'{HOUSE[city]}: as \cref{{tab:cluster_annotation}}.')
        out += [r'\bottomrule', r'\end{tabular}', rf'\caption{{\tist[R2-5.4]{{{caption}}}}}',
                rf'\label{{tab:cluster_annotation{"" if i == 0 else "_" + city}}}', r'\end{table}', '']
    path.write_text('\n'.join(out))


def pdl_vs_casas_plot(r, path):
    """Rated top windows: pattern label (rows) x dataset activity of the window (columns), one panel per house."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    nol = casas_name(UNLABELED)
    d = r[r['stratum'] == 'top'].assign(
        pdl=lambda x: x['label'].map(PDL),
        cas=lambda x: [nol if c == UNLABELED else casas_name(rs.casas_l1(h, c)) for h, c in zip(x['city'], x['casas'])])
    tabs = {}
    for city in CITIES:
        t = pd.crosstab(d.loc[d['city'] == city, 'pdl'], d.loc[d['city'] == city, 'cas']) / 2
        cols = [c for c in t.sum().sort_values(ascending=False).index if c != nol] + ([nol] if nol in t else [])
        t = t[cols]
        order = pd.DataFrame({'col': t.idxmax(axis=1).map(cols.index), 'n': -t.max(axis=1)})
        tabs[city] = t.loc[order.sort_values(['col', 'n']).index]
    heights = [len(t) for t in tabs.values()]
    fig, axes = plt.subplots(len(CITIES), 1, figsize=(6.2, 0.13 * sum(heights) + 1.2 * len(CITIES)),
                             gridspec_kw={'height_ratios': heights}, squeeze=False)
    for ax, (city, t) in zip(axes[:, 0], tabs.items()):
        v = t.to_numpy()
        ax.imshow(v, cmap='Blues', aspect='auto', vmin=0, vmax=max(6, v.max()))
        for (i, j), x in np.ndenumerate(v):
            if x:
                ax.text(j, i, f'{x:g}', ha='center', va='center', fontsize=5.5,
                        color='white' if x > 0.6 * v.max() else 'black')
        ax.set_yticks(range(len(t)), t.index)
        ax.set_xticks(range(t.shape[1]), t.columns, rotation=35, ha='right')
        if nol in t:
            ax.axvline(t.shape[1] - 1.5, color='0.3', lw=0.8)
        ax.set_title(f'{HOUSE[city]} ({int(v.sum())} rated windows)', loc='left', fontsize=8)
        ax.tick_params(length=0)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- figure: agreement

def agreement_svg(stats, clus):
    W, H = 760, 300
    mt, ph = 58, 170
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="{FIG_FONT}">', f'<rect width="{W}" height="{H}" fill="#fff"/>']

    def yaxis(x0, pw, Y, ticks, label):
        for t in ticks:
            out.append(f'<line x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + pw}" y2="{Y(t):.1f}" stroke="{GRID}" stroke-width="0.75"/>'
                       f'<text x="{x0 - 7}" y="{Y(t) + 4:.1f}" font-size="12" text-anchor="end" fill="{SUB}">{t:.1f}</text>')
        out.append(f'<line x1="{x0}" y1="{mt}" x2="{x0}" y2="{mt + ph}" stroke="{SUB}" stroke-width="1.1"/>'
                   f'<line x1="{x0}" y1="{mt + ph}" x2="{x0 + pw}" y2="{mt + ph}" stroke="{SUB}" stroke-width="1.1"/>'
                   f'<text transform="translate({x0 - 36},{mt + ph / 2}) rotate(-90)" font-size="12" '
                   f'text-anchor="middle" fill="#64748b">{label}</text>')

    def Y(v):
        return mt + ph - v * ph
    ticks = np.arange(0, 1.01, 0.2)

    # (a) agreement by source
    x0, pw = 62, 300
    out.append(f'<text x="{x0 - 50}" y="{mt - 30}" font-size="14" fill="{SUB}">(a) Agreement of two labels by source</text>')
    yaxis(x0, pw, Y, ticks, 'P(two labels agree)')
    kinds = [('agree_same_window', 'Same window', '(two raters)'),
             ('agree_same_cluster_other_window', 'Same cluster,', 'other window'),
             ('agree_other_cluster', 'Different', 'clusters')]
    gw, bw = pw / 3, min(22, (pw / 3 - 12) / len(CITIES))
    for k, (col, l1, l2) in enumerate(kinds):
        cx = x0 + gw * (k + 0.5)
        for i, city in enumerate(CITIES):
            s = stats[stats.city == city].set_index('level')
            lo, hi = s.loc['original', col], s.loc['level_up', col]
            bx = cx + (i - len(CITIES) / 2) * bw + 1
            out.append(f'<rect x="{bx:.1f}" y="{Y(hi):.1f}" width="{bw - 2}" height="{Y(lo) - Y(hi):.1f}" '
                       f'fill="{HOUSE_COLOR[city]}" fill-opacity="0.28"/>'
                       f'<rect x="{bx:.1f}" y="{Y(lo):.1f}" width="{bw - 2}" height="{Y(0) - Y(lo):.1f}" '
                       f'fill="{HOUSE_COLOR[city]}"/>')
            if k < 2:
                out.append(f'<text x="{bx + (bw - 2) / 2:.1f}" y="{Y(lo) + 13:.1f}" font-size="9.5" fill="#fff" '
                           f'text-anchor="middle">{lo:.2f}</text>')
        out.append(f'<text x="{cx:.1f}" y="{mt + ph + 17}" font-size="12" text-anchor="middle" fill="{SUB}">{l1}</text>'
                   f'<text x="{cx:.1f}" y="{mt + ph + 31}" font-size="12" text-anchor="middle" fill="{SUB}">{l2}</text>')

    # (b) majority share per cluster
    x0, pw = 450, 296
    out.append(f'<text x="{x0 - 50}" y="{mt - 30}" font-size="14" fill="{SUB}">(b) Share of a cluster\'s 10 labels '
               f'for its majority label</text>')
    yaxis(x0, pw, Y, ticks, 'Majority share')
    out.append(f'<line x1="{x0}" y1="{Y(0.5)}" x2="{x0 + pw}" y2="{Y(0.5)}" stroke="#94a3b8" stroke-width="1" '
               f'stroke-dasharray="4 3"/>')
    n, gap = 20, 4
    step = (pw - 12) / (len(CITIES) * n + (len(CITIES) - 1) * gap)
    for i, city in enumerate(CITIES):
        c = clus[clus.city == city].sort_values(['share', 'share_parent'], ascending=False)
        xs = x0 + 6 + step * (np.arange(len(c)) + i * (n + gap) + 0.5)
        col = HOUSE_COLOR[city]
        for x, a, b in zip(xs, c['share'], c['share_parent']):
            if b > a:
                out.append(f'<line x1="{x:.1f}" y1="{Y(a):.1f}" x2="{x:.1f}" y2="{Y(b):.1f}" stroke="{col}" '
                           f'stroke-opacity="0.45" stroke-width="1.2"/>'
                           f'<circle cx="{x:.1f}" cy="{Y(b):.1f}" r="2.6" fill="#fff" stroke="{col}" stroke-width="1.1"/>')
            out.append(f'<circle cx="{x:.1f}" cy="{Y(a):.1f}" r="2.9" fill="{col}"/>')
        out.append(f'<text x="{xs.mean():.1f}" y="{mt + ph + 17}" font-size="12" text-anchor="middle" '
                   f'fill="{col}">{HOUSE[city]}</text>')

    # legend
    ly = H - 34
    items = [(HOUSE[c], HOUSE_COLOR[c], 'house') for c in CITIES] + [('PDL label', SUB, 'dark'),
                                                                      ('gain at parent level', SUB, 'light')]
    widths = [text_width(t, 12, 'normal') * 1.06 + 34 for t, *_ in items]
    lx = (W - sum(widths) - 12) / 2
    out.append(f'<rect x="{lx:.1f}" y="{ly}" width="{sum(widths) + 12:.1f}" height="26" rx="4" fill="#fff" stroke="{GRID}"/>')
    cx = lx + 10
    for (t, col, kind), w in zip(items, widths):
        if kind == 'light':
            out.append(f'<rect x="{cx:.1f}" y="{ly + 7}" width="14" height="12" fill="{SUB}" fill-opacity="0.25"/>'
                       f'<circle cx="{cx + 7:.1f}" cy="{ly + 13}" r="3" fill="#fff" stroke="{SUB}" stroke-width="1.1"/>')
        elif kind == 'dark':
            out.append(f'<rect x="{cx:.1f}" y="{ly + 7}" width="14" height="12" fill="{SUB}"/>')
        else:
            out.append(f'<rect x="{cx:.1f}" y="{ly + 7}" width="14" height="12" fill="{col}"/>')
        out.append(f'<text x="{cx + 20:.1f}" y="{ly + 17}" font-size="12" fill="{SUB}">{t}</text>')
        cx += w
    out.append('</svg>')
    return '\n'.join(out) + '\n'


# ---------------------------------------------------------------- figure: CASAS -> patterns

def casas_to_pdl_svg(clus):
    """Three columns (houses); per CASAS activity, one chip per distinct pattern of the clusters it dominates."""
    colw, x_pad, chip_h, chip_gap = 268, 12, 20, 5
    W = len(CITIES) * colw + 2 * x_pad + 10 * (len(CITIES) - 1)
    blocks, heights = {}, {}
    for city in CITIES:
        c = clus[clus.city == city]
        groups = []
        for g, gg in c.groupby('group'):
            pats = gg.groupby('pattern').agg(n=('cluster', 'size'), frac=('size_frac', 'sum'))
            pats = pats.sort_values(['n', 'frac'], ascending=False)
            groups.append((g, gg['size_frac'].sum(), pats))
        groups.sort(key=lambda t: (t[0] == UNLABELED, -len(t[2]), -t[1]))
        layout, y = [], 0
        for g, frac, pats in groups:
            rows, x, row = [], 0, []
            for p, pr in pats.iterrows():
                label = PDL[p] + (f' ×{int(pr.n)}' if pr.n > 1 else '')
                w = text_width(label, 11.5, 'normal') * 1.05 + 16
                if row and x + w > colw - 16:
                    rows.append(row)
                    row, x = [], 0
                row.append((label, w))
                x += w + chip_gap
            rows.append(row)
            layout.append((g, frac, rows, y))
            y += 22 + len(rows) * (chip_h + chip_gap) + 10
        blocks[city], heights[city] = layout, y
    top = 44
    H = top + max(heights.values()) + 4
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="{FIG_FONT}">', f'<rect width="{W}" height="{H}" fill="#fff"/>']
    for i, city in enumerate(CITIES):
        x0 = x_pad + i * (colw + 10)
        col = HOUSE_COLOR[city]
        out.append(f'<text x="{x0}" y="24" font-size="15" font-weight="500" fill="{col}">{HOUSE[city]}</text>')
        for g, frac, rows, y in blocks[city]:
            yy = top + y
            unl = g == UNLABELED
            out.append(f'<text x="{x0}" y="{yy + 13}" font-size="12.5" font-weight="500" '
                       f'fill="{"#64748b" if unl else INK}">{esc(casas_name(g))} '
                       f'<tspan font-weight="400" fill="#64748b">({frac:.0%})</tspan></text>')
            for k, row in enumerate(rows):
                cx = x0
                cy = yy + 20 + k * (chip_h + chip_gap)
                for label, w in row:
                    fill, op = ('#f1f5f9', 1) if unl else (col, 0.16)
                    out.append(f'<rect x="{cx:.1f}" y="{cy}" width="{w:.1f}" height="{chip_h}" rx="10" fill="{fill}" '
                               f'fill-opacity="{op:.2f}" stroke="{"#cbd5e1" if unl else col}" stroke-opacity="0.6"/>'
                               f'<text x="{cx + 8:.1f}" y="{cy + 14}" font-size="11.5" fill="{INK}">{esc(label)}</text>')
                    cx += w + chip_gap
        if i:
            out.append(f'<line x1="{x0 - 6}" y1="10" x2="{x0 - 6}" y2="{H - 6}" stroke="{GRID}"/>')
    out.append('</svg>')
    return '\n'.join(out) + '\n'


# ---------------------------------------------------------------- figure: t-SNE

CASAS_COLORS = {  # L1 labels of the three houses; meals orange/yellow, rest blue, sitting red/pink
    'Kitchen_Activity': '#ea7f2c', 'Meal_Preparation': '#ea7f2c', 'Dinner': '#ea7f2c', 'Breakfast': '#f2b45c',
    'Lunch': '#c9861a', 'Dining_Rm_Activity': '#f2b45c', 'Eating': '#f2b45c', 'Wash_Dishes': '#8c6d31',
    'Read': '#c53a32', 'Relax': '#c53a32', 'Watch_TV': '#e8837a', 'Desk_Activity': '#de77ae', 'Work': '#de77ae',
    'work in office': '#de77ae', 'Master_Bedroom_Activity': '#6b8fd6', 'Sleep': '#2f6397', 'Sleeping': '#2f6397',
    'sleep': '#2f6397', 'wake': '#6b8fd6', 'Night wandering': '#9467bd', 'Master_Bathroom': '#0b8ca8',
    'Guest_Bathroom': '#5cc2c9', 'Bed_to_Toilet': '#4b68c9', 'Bed to toilet': '#4b68c9', 'Chores': '#8c6d31',
    'Housekeeping': '#8c6d31', 'Laundry': '#8c6d31', 'Leave_Home': '#519e3e', 'Enter_Home': '#7fc06a',
    'Leave home': '#519e3e', 'Meditate': '#9467bd', 'Respirate': '#9467bd', 'Morning_Meds': '#d4b106',
    'Eve_Meds': '#d4b106', 'take medicine': '#d4b106',
}
NO_LABEL_COLOR = '#c3c8d0'
MAX_TSNE = 30000


def tsne_coords(city, seed=0):
    """t-SNE of the clustering features (TF s1 + ToD) of the training windows; houses with more than MAX_TSNE
    windows are subsampled at random. Returns (row indices, coordinates)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from .rare_classes import representations
    TSNE_CACHE.mkdir(parents=True, exist_ok=True)
    cache = TSNE_CACHE / f'{city}.npz'
    if cache.exists():
        z = np.load(cache)
        return z['idx'], z['Z']
    fit, pool, test = load_protocol(city)
    (_, (X, _, _)), = representations(city, fit, pool, test, ['tf_s1+tod'])
    idx = np.arange(len(X))
    if len(X) > MAX_TSNE:
        idx = np.sort(np.random.default_rng(seed).choice(len(X), MAX_TSNE, replace=False))
    Xp = PCA(n_components=50, random_state=seed).fit_transform(X[idx])
    Z = TSNE(n_components=2, init='pca', perplexity=40, random_state=seed, n_jobs=-1).fit_transform(Xp)
    np.savez(cache, idx=idx, Z=Z)
    return idx, Z


def scatter_png(Z, labels, size_px):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(size_px[0] / 100, size_px[1] / 100), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    order = np.argsort(labels != UNLABELED, kind='stable')
    cols = np.array([CASAS_COLORS.get(l, NO_LABEL_COLOR) for l in labels])
    ax.scatter(Z[order, 0], Z[order, 1], s=2.2, c=cols[order], linewidths=0, alpha=0.35, rasterized=True)
    ax.set_xlim(Z[:, 0].min(), Z[:, 0].max())
    ax.set_ylim(Z[:, 1].min(), Z[:, 1].max())
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def sensor_heat(city, w, cluster, top_n=300):
    """Mean ON/OPEN activations per sensor over the cluster's top_n most confident windows."""
    with open(DATA_ROOT / city / 'FL_20' / 'train.json') as f:
        samples = json.load(f)['samples']
    idx = w[w['cluster'] == cluster].sort_values('conf', ascending=False).index[:top_n]
    counts = {}
    for i in idx:
        for e in samples[i]['sensor_sequence']:
            if e['event_type'].upper() in ('ON', 'OPEN'):
                counts[e['sensor_id']] = counts.get(e['sensor_id'], 0) + 1
    return {k: v / len(idx) for k, v in counts.items()}


def hull(points):
    pts = sorted(map(tuple, points))
    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(p)
        return h
    lo, up = half(pts), half(pts[::-1])
    return lo[:-1] + up[:-1]


def islands(Z, min_frac=0.05):
    """Connected parts of one cluster in t-SNE space (DBSCAN), largest first; parts below min_frac are dropped."""
    from sklearn.cluster import DBSCAN
    lab = DBSCAN(eps=0.022 * np.ptp(Z, axis=0).max(), min_samples=6).fit_predict(Z)
    parts = [Z[lab == k] for k in set(lab) if k != -1]
    return sorted([p for p in parts if len(p) >= min_frac * len(Z)], key=len, reverse=True)


def casas_mix(cw, n=2):
    """'Kitchen Activity 82% · no label 14%': the n most frequent CASAS labels of a cluster's windows."""
    vc = cw['casas_last'].value_counts(normalize=True).head(n)
    vc = vc[(vc >= 0.05) | (np.arange(len(vc)) == 0)]
    return ' · '.join(f'{"no label" if l == UNLABELED else casas_name(l)} {v:.0%}' for l, v in vc.items())


def tsne_svg(city, cl, windows, clus, parts_dir=None):
    """t-SNE of one house's windows colored by CASAS label (left) and the annotated clusters `cl` drawn on the floor
    plan in a two-column grid (right). parts_dir: also write the pieces separately (scatter PNG, outlines, legend,
    one SVG per floor-plan inset, texts.json) for assembling the figure by hand."""
    parts_out = {'outlines': [], 'legend': [], 'insets': {}, 'texts': {}}
    from PIL import Image
    meta_all = json.load(open(METADATA))
    W, ts, iw_px, gap, top = 910, 380, 228, 18, 40
    pw, ph = Image.open(FLOOR_PLANS / f'{city}.png').size
    ih_row = 48 + iw_px / pw * ph + 22
    H = int(top + max(ts + 90, ih_row * -(-len(cl) // 2)))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" '
           f'height="{H}" viewBox="0 0 {W} {H}" font-family="{FIG_FONT}">', f'<rect width="{W}" height="{H}" fill="#fff"/>',
           '<defs><radialGradient id="heat"><stop offset="0" stop-color="#dc2626" stop-opacity="0.9"/>'
           '<stop offset="1" stop-color="#dc2626" stop-opacity="0"/></radialGradient></defs>',
           f'<text x="{W / 2}" y="22" font-size="16" font-weight="500" text-anchor="middle" fill="{INK}">'
           't-SNE of DISCOVER window representations and annotated clusters</text>']
    tag = iter('abcdefghij')
    for r, (city, cl) in enumerate([(city, cl)]):
        y0 = top
        w = windows[city]
        meta = meta_all[city]
        idx, Z = tsne_coords(city)
        labels = w['casas_last'].to_numpy()[idx]
        plan = FLOOR_PLANS / f'{city}.png'
        pw, ph = Image.open(plan).size
        out.append(f'<defs><image id="plan_{city}" width="{pw}" height="{ph}" '
                   f'xlink:href="data:image/png;base64,{base64.b64encode(plan.read_bytes()).decode()}"/>'
                   f'<clipPath id="clip_{city}"><rect width="{pw}" height="{ph}"/></clipPath></defs>')
        sx0, sy0 = 14, y0 + 22
        scatter = scatter_png(Z, labels, (ts, ts))
        parts_out['scatter'] = scatter
        out.append(f'<text x="{sx0}" y="{y0 + 12}" font-size="15" font-weight="500" fill="{INK}">{HOUSE[city]}</text>'
                   f'<image x="{sx0}" y="{sy0}" width="{ts}" height="{ts}" '
                   f'xlink:href="data:image/png;base64,{scatter}"/>')
        zx0, zx1, zy0, zy1 = Z[:, 0].min(), Z[:, 0].max(), Z[:, 1].min(), Z[:, 1].max()

        def P(z):
            return sx0 + (z[0] - zx0) / (zx1 - zx0) * ts, sy0 + ts - (z[1] - zy0) / (zy1 - zy0) * ts
        # legend: labels with at least 2% of the windows, plus "no CASAS label"
        vc = pd.Series(labels).value_counts(normalize=True)
        leg = [l for l, v in vc.items() if v >= 0.02 and l != UNLABELED][:6] + [UNLABELED]
        for k, l in enumerate(leg):
            lx, ly = sx0 + (k % 2) * 136, sy0 + ts + 14 + (k // 2) * 14
            item = (f'<circle cx="{lx + 4}" cy="{ly - 4}" r="4" fill="{CASAS_COLORS.get(l, NO_LABEL_COLOR)}"/>'
                    f'<text x="{lx + 12}" y="{ly}" font-size="10.5" fill="{SUB}">{esc(casas_name(l))}</text>')
            out.append(item)
            parts_out['legend'].append(item)
        for j, c in enumerate(cl):
            t = next(tag)
            row = clus[(clus.city == city) & (clus.cluster == c)].iloc[0]
            m = (w['cluster'] == c).to_numpy()[idx]
            conf = w['conf'].to_numpy()[idx]
            parts = sorted(islands(Z[m & (conf >= np.quantile(conf[m], 0.25))]), key=len, reverse=True)
            for p_i, part in enumerate(parts):
                hp = [P(p) for p in hull(part)]
                out.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in hp)}" fill="none" '
                           f'stroke="{INK}" stroke-width="1.3" stroke-dasharray="3.5 2" stroke-linejoin="round"/>')
                parts_out['outlines'].append(out[-1])
                if p_i and len(part) < 0.25 * len(parts[0]):
                    continue
                bx, by = min(hp, key=lambda q: q[1])
                out.append(f'<text x="{bx:.1f}" y="{by - 3:.1f}" font-size="12" font-weight="600" fill="{INK}" '
                           f'text-anchor="middle" stroke="#fff" stroke-width="3" paint-order="stroke">{t}</text>')
                parts_out['outlines'].append(out[-1])
            x = sx0 + ts + gap + 10 + (j % 2) * (iw_px + gap)
            yi = y0 + (j // 2) * ih_row
            s = iw_px / pw
            ih_px = ph * s
            heat = sensor_heat(city, w, c)
            mx = max(heat.values())
            circles = []
            for sid, v in sorted(heat.items(), key=lambda q: q[1]):
                if sid not in meta['sensor_coordinates'] or v / mx < 0.1:
                    continue
                px, py = meta['sensor_coordinates'][sid]
                rad = (0.03 + 0.04 * np.sqrt(v / mx)) * max(pw, ph)
                circles.append(f'<circle cx="{px}" cy="{ph - py}" r="{rad:.0f}" fill="url(#heat)" '
                               f'opacity="{0.3 + 0.7 * v / mx:.2f}"/>')
            cw = w[w['cluster'] == c]
            parts_out['insets'][t] = ''.join(circles)
            parts_out['texts'][t] = dict(cluster=int(c), title=PDL[row.pattern], agreement=f'{row.share * 10:.0f}/10 raters agree',
                                         agreement_parent=f'{row.share_parent * 10:.0f}/10 at parent level',
                                         casas=casas_mix(cw, n=3), windows=int(len(cw)))
            out.append(f'<text x="{x}" y="{yi + 12}" font-size="13.5" font-weight="500" fill="{INK}">'
                       f'({t}) {esc(PDL[row.pattern])}</text>'
                       f'<text x="{x}" y="{yi + 27}" font-size="10.5" fill="#64748b">cluster {c} · '
                       f'{row.share * 10:.0f}/10 raters agree</text>'
                       f'<text x="{x}" y="{yi + 41}" font-size="10" fill="#64748b">CASAS: {esc(casas_mix(cw))}</text>'
                       f'<g transform="translate({x},{yi + 48})"><rect x="-1" y="-1" width="{iw_px + 2}" '
                       f'height="{ih_px + 2:.1f}" fill="#fff" stroke="#cbd5e1" rx="3"/>'
                       f'<g transform="scale({s:.5f})" clip-path="url(#clip_{city})"><use xlink:href="#plan_{city}" '
                       f'opacity="0.5"/>{"".join(circles)}</g></g>')
        if r:
            out.append(f'<line x1="10" y1="{y0 - 8}" x2="{W - 10}" y2="{y0 - 8}" stroke="{GRID}"/>')
    out.append('</svg>')
    if parts_dir:
        write_tsne_parts(parts_dir, parts_out, (sx0, sy0, ts), city, (pw, ph))
    return '\n'.join(out) + '\n'


def write_tsne_parts(d, parts, frame, city, plan_size):
    d.mkdir(parents=True, exist_ok=True)
    sx0, sy0, ts = frame
    pw, ph = plan_size
    head = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
    (d / 'tsne_scatter.png').write_bytes(base64.b64decode(parts['scatter']))
    (d / 'tsne_outlines.svg').write_text(
        f'{head}width="{ts}" height="{ts}" viewBox="0 0 {ts} {ts}" font-family="{FIG_FONT}">'
        f'<g transform="translate({-sx0},{-sy0})">{"".join(parts["outlines"])}</g></svg>\n')
    (d / 'tsne_with_outlines.svg').write_text(
        f'{head}width="{ts}" height="{ts}" viewBox="0 0 {ts} {ts}" font-family="{FIG_FONT}">'
        f'<image width="{ts}" height="{ts}" xlink:href="data:image/png;base64,{parts["scatter"]}"/>'
        f'<g transform="translate({-sx0},{-sy0})">{"".join(parts["outlines"])}</g></svg>\n')
    (d / 'tsne_legend.svg').write_text(
        f'{head}width="{ts}" height="70" viewBox="0 0 {ts} 70" font-family="{FIG_FONT}">'
        f'<g transform="translate({-sx0},{-(sy0 + ts)})">{"".join(parts["legend"])}</g></svg>\n')
    plan = base64.b64encode((FLOOR_PLANS / f'{city}.png').read_bytes()).decode()
    for t, circles in parts['insets'].items():
        (d / f'inset_{t}_cluster{parts["texts"][t]["cluster"]}.svg').write_text(
            f'{head}width="{pw}" height="{ph}" viewBox="0 0 {pw} {ph}"><defs><radialGradient id="heat">'
            '<stop offset="0" stop-color="#dc2626" stop-opacity="0.9"/><stop offset="1" stop-color="#dc2626" '
            f'stop-opacity="0"/></radialGradient></defs><rect width="{pw}" height="{ph}" fill="#fff"/>'
            f'<image width="{pw}" height="{ph}" opacity="0.5" xlink:href="data:image/png;base64,{plan}"/>'
            f'{circles}</svg>\n')
    (d / 'texts.json').write_text(json.dumps(parts['texts'], indent=1, ensure_ascii=False))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsne', default='milan:18,13,16,14', help='t-SNE figure: house:cluster,cluster,...')
    ap.add_argument('--study', choices=list(STUDIES), default='paper')
    args = ap.parse_args()
    global OUT, STUDY, RETURNED, CITIES
    name, RETURNED, CITIES, OUT = STUDIES[args.study]
    STUDY = [(OUT_ROOT / f, ret) for f, ret in name] if isinstance(name, list) else OUT_ROOT / name
    OUT.mkdir(parents=True, exist_ok=True)
    r, stats, windows = load()
    clus = cluster_table(r, windows)
    clus.to_csv(OUT / 'clusters.csv', index=False)
    st = text_stats(r, stats, clus)
    boots = bootstraps(r)
    avg = average_agreement(stats, boots) if len(CITIES) > 1 else None
    rw = random_windows(r, boots)
    rw.to_csv(OUT / 'random_windows.csv', index=False)
    st['average'] = {'agreement': avg, 'random_windows': rw.drop(columns='city').mean(numeric_only=True).to_dict()}
    for _, row in rw.iterrows():
        st[row.city]['random_windows'] = row.drop('city').to_dict()
    (OUT / 'stats.json').write_text(json.dumps(st, indent=1, default=float))
    kappa_table(stats, OUT / 'kappa_scores.tex', avg)
    random_table(rw, OUT / 'random_windows.tex')
    cluster_summary_table(r, clus, OUT / 'cluster_summary.tex')
    taxonomy_table(OUT / 'pdl_taxonomy.tex')
    pdl_vs_casas_plot(r, OUT / 'pdl_vs_casas')
    write_figure(agreement_svg(stats, clus), 'annotation_agreement')
    write_figure(casas_to_pdl_svg(clus), 'casas_to_pdl')
    city, ks = args.tsne.split(':')
    if city in CITIES:
        write_figure(tsne_svg(city, [int(k) for k in ks.split(',')], windows, clus, parts_dir=OUT / 'fig6_parts'),
                     'tsne_clusters')
    print(f'outputs: {OUT}')


if __name__ == '__main__':
    main()
