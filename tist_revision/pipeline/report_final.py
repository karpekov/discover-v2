"""
Tables, charts and CSVs for the seeded final runs (continuous FL_20 protocol, no coordinates).

Reads cluster assignments (`artifacts/assignments/`) + baseline npzs (`artifacts/baselines/final/`), scores them
into `results/final/data/{runs,summary}.csv`, and writes `results/final/{tables,figures,REPORT.md,report.pdf}`.
`results/final/` holds current paper-grade results only; explorations go to `results/temp/`.

Usage:
    python -m tist_revision.pipeline.report_final
    python -m tist_revision.pipeline.report_final --rescore --pdf
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from ..baselines.data import budget_indices
from .mapping_eval import (ARTIFACTS_DIR, BASELINES_DIR, EXCLUDED_LABELS, FINAL_DIR, PROJECT_DIR, UNMAPPED, evaluate_clustering,
                           load_protocol, map_clusters, predict_activity, scored_classes)
from .rare_classes import representations

MARGIN = 0.03
HOUSE_NAMES = {'milan': 'Milan', 'aruba': 'Aruba', 'cairo': 'Cairo', 'orange': 'Orange4Home'}
CITIES = ['milan', 'aruba', 'cairo', 'orange']  # Orange4Home joined the paper 2026-11-20
OUT = FINAL_DIR
ASSIGN = re.compile(
    r'^(?P<city>[a-z]+)__(?P<enc>[a-z_]+?)_s(?P<seed>\d+)(?P<tod>\+tod)?__'
    r'(?P<method>kmeans|scan|dec)_k(?P<K>\d+)(?P<suffix>_todw\w+)?$')
REP_NAMES = {
    'tf': 'TF', 'tf+tod': 'TF+ToD', 'bert': 'BERT', 'bert+tod': 'BERT+ToD',
    'hist_sensor': 'Sensor hist.', 'hist_sensor_tod': 'Sensor+ToD hist.',
}
METHOD_NAMES = {'kmeans': 'k-means', 'scan': 'SCAN', 'scan_unw': 'SCAN (unweighted ToD)', 'dec': 'DEC',
                'deepcasas': 'DeepCASAS', 'tdost': 'TDOST', 'treecnn': 'TreeCNN', 'logreg': 'Linear probe'}
BASELINES = ('deepcasas', 'tdost')  # TreeCNN dropped 2026-11-13, see tist_revision/README.md
CLU_ORDER = [
    ('kmeans', 'hist_sensor'), ('kmeans', 'hist_sensor_tod'),
    ('kmeans', 'tf'), ('kmeans', 'tf+tod'), ('kmeans', 'bert'), ('kmeans', 'bert+tod'),
    ('scan', 'tf'), ('scan', 'tf+tod'), ('scan_unw', 'tf+tod'), ('scan', 'bert'), ('scan_unw', 'bert+tod'),
    ('scan', 'hist_sensor_tod'),
    ('dec', 'tf'), ('dec', 'tf+tod'), ('dec', 'bert'), ('dec', 'bert+tod'),
]


def variants(level, pool_labels, test_labels):
    v = {'all': None, 'masked': scored_classes(pool_labels, test_labels)}
    if level == 'l2':
        v['excl_other'] = sorted(set(test_labels) - set(EXCLUDED_LABELS))
    return v


def score_preds(y_test, pred, level, pool_labels):
    rows = []
    for name, classes in variants(level, pool_labels, y_test).items():
        keep = ~np.isin(y_test, EXCLUDED_LABELS) if name == 'excl_other' else np.ones(len(y_test), bool)
        yt, yp = y_test[keep], pred[keep]
        rows.append({
            'variant': name,
            'f1_weighted': f1_score(yt, yp, average='weighted', zero_division=0),
            'f1_macro': f1_score(yt, yp, labels=classes, average='macro', zero_division=0)
            if classes is not None else f1_score(yt, yp, average='macro', zero_division=0),
        })
    return rows


def scan_variant(rep, suffix):
    """Main SCAN = balanced ToD weighting (`_todwbal`) on `+tod` embeddings; SCAN without a ToD block (or on
    histograms) is unweighted. Fixed-weight sweeps (`_todw1/2/4/8`, Cairo only) stay in results/temp."""
    if suffix:
        return 'scan' if suffix == '_todwbal' else None
    return 'scan_unw' if rep.endswith('+tod') else 'scan'


def score_clustering(m_values=('1', '5', 'all'), n_boot=1, cities=None):
    """`cities` defaults to the paper's houses; other datasets' assignments in artifacts/ (e.g. the Kyoto7
    trial) are scored only when asked for."""
    cities = CITIES if cities is None else cities
    rows = []
    paths = [p for p in sorted((ARTIFACTS_DIR / 'assignments').glob('*.npz')) if 'smoke' not in p.stem]
    m_vals = [m if m == 'all' else int(m) for m in m_values]
    for i, path in enumerate(paths):
        mt = ASSIGN.match(path.stem)
        if not mt:
            continue
        city, seed, K, method = mt['city'], int(mt['seed']), int(mt['K']), mt['method']
        if city not in cities:
            continue
        rep = mt['enc'] + (mt['tod'] or '')
        if method == 'scan':
            method = scan_variant(rep, mt['suffix'])
            if method is None:
                continue
        z = np.load(path)
        for level in ('l1', 'l2'):
            _, pool, test = load_protocol(city, level=level)
            if len(z['pool_cluster']) != len(pool):
                continue
            vs = ('all', 'masked') if level == 'l1' else ('all', 'masked', 'excl_other')
            res = evaluate_clustering(
                pool[['label', 'day']].assign(cluster=z['pool_cluster'], conf=z['pool_conf']),
                test[['label', 'day']].assign(cluster=z['test_cluster'], conf=0.0),
                m_values=m_vals, variants=vs, n_boot=n_boot, seed=seed)
            # Weighted F1 is over all true classes in every variant (as for probes and baselines); masking only
            # restricts the macro average.
            w_all = res[res.variant == 'all'].set_index('m').f1_weighted
            res.loc[res.variant == 'masked', 'f1_weighted'] = res.loc[res.variant == 'masked', 'm'].map(w_all).values
            for r in res.to_dict('records'):
                rows.append({
                    'family': 'clustering', 'method': method, 'rep': rep, 'K': K, 'seed': seed,
                    'city': city, 'level': level, 'budget': f"m{r['m']}", 'variant': r['variant'],
                    'f1_weighted': r['f1_weighted'], 'f1_macro': r['f1_macro'],
                    'n_labels': r['n_labels_mapped'],
                })
        if (i + 1) % 50 == 0:
            print(f'  clustering {i + 1}/{len(paths)}', flush=True)
    return pd.DataFrame(rows)


def score_baselines(cities=None):
    cities = CITIES if cities is None else cities
    rows = []
    for path in sorted((BASELINES_DIR / 'final').glob('*/*.npz')):
        mt = re.match(rf'^({"|".join(cities)})_(l[12])_(\w+)_s(\d+)$', path.stem)
        if not mt:
            continue
        city, level, budget, seed = mt.groups()
        z = np.load(path, allow_pickle=True)
        pool, test = load_protocol(city, level=level)[1:]
        assert (z['y_test'] == test['label'].to_numpy()).all(), path
        for r in score_preds(z['y_test'], z['pred'], level, pool['label'].to_numpy()):
            rows.append({
                'family': 'baseline', 'method': path.parent.name, 'rep': '', 'K': np.nan,
                'seed': int(seed), 'city': city, 'level': level, 'budget': budget,
                'variant': r['variant'], 'f1_weighted': r['f1_weighted'], 'f1_macro': r['f1_macro'],
                'n_labels': int(z['n_labels']),
            })
    return pd.DataFrame(rows)


def score_probes(budgets=('100', '250', 'all'), max_full=20000, cities=None):
    rows = []
    for city in (CITIES if cities is None else cities):
        fit, pool2, test2 = load_protocol(city, level='l2')
        _, pool1, test1 = load_protocol(city, level='l1')
        protocols = {'l1': (pool1, test1), 'l2': (pool2, test2)}
        reps = []
        for path in sorted((ARTIFACTS_DIR / 'embeddings').glob(f'{city}_*_s*_embeddings.npz')):
            if 'smoke' in path.stem:
                continue
            mt = re.match(rf'^{city}_(tf|bert)_s(\d+)_embeddings$', path.stem)
            if mt:
                enc, seed = mt.groups()
                reps += [(f'{enc}_s{seed}', int(seed)), (f'{enc}_s{seed}+tod', int(seed))]
        seeds = sorted({s for _, s in reps}) or [1]
        reps += [(f'hist_{h}_s{s}', s) for h in ('sensor', 'sensor_tod') for s in seeds]
        any_pool, any_test = next(iter(protocols.values()))
        for (rep, seed), (_, (X_pool, _, X_test)) in zip(
                reps, representations(city, fit, any_pool, any_test, [r for r, _ in reps])):
            base = re.sub(r'_s\d+', '', rep)
            for level, (pool, test) in protocols.items():
                y_pool, y_test = pool['label'].to_numpy(), test['label'].to_numpy()
                for budget in budgets:
                    if budget == 'all':
                        idx = np.random.default_rng(seed).choice(
                            len(y_pool), size=min(max_full, len(y_pool)), replace=False)
                    else:
                        tr, va = budget_indices(y_pool, budget, seed)
                        idx = np.concatenate([tr, va])
                    scaler = StandardScaler().fit(X_pool[idx])
                    clf = LogisticRegression(max_iter=2000).fit(
                        scaler.transform(X_pool[idx]), y_pool[idx])
                    pred = predict_activity(clf.predict_proba(scaler.transform(X_test)), clf.classes_)
                    for r in score_preds(y_test, pred, level, y_pool):
                        rows.append({
                            'family': 'probe', 'method': 'logreg', 'rep': base, 'K': np.nan,
                            'seed': seed, 'city': city, 'level': level, 'budget': budget,
                            'variant': r['variant'], 'f1_weighted': r['f1_weighted'],
                            'f1_macro': r['f1_macro'], 'n_labels': len(idx),
                        })
            print(f'  probe {city} {rep}', flush=True)
    return pd.DataFrame(rows)


def summarize(df):
    keys = ['family', 'method', 'rep', 'K', 'city', 'level', 'variant', 'budget']
    return df.groupby(keys, dropna=False).agg(
        n_seeds=('seed', 'nunique'),
        f1_macro=('f1_macro', 'mean'), f1_macro_std=('f1_macro', 'std'),
        f1_weighted=('f1_weighted', 'mean'), f1_weighted_std=('f1_weighted', 'std'),
        n_labels=('n_labels', 'mean')).reset_index()


def cell(v, best, style):
    """v = (mean over seeds, std over seeds); std shown small and grey under the mean when >1 seed."""
    v, sd = v
    if pd.isna(v):
        return '--' if style == 'tex' else '–'
    s = f'{v:.2f}'
    if style == 'tex':
        shade = r'\cellcolor{blue!12}' if best - v <= MARGIN + 1e-9 else ''
        if abs(v - best) < 5e-3:
            s = rf'\textbf{{{s}}}'
        sd_txt = f'[{sd:.2f}]' if pd.notna(sd) else r'\phantom{[0.00]}'
        return shade + (rf'\begin{{tabular}}[c]{{@{{}}c@{{}}}}{s}\\[-2.5pt]'
                        rf'{{\color{{gray}}\fontsize{{6.5}}{{7}}\selectfont {sd_txt}}}\end{{tabular}}')
    if abs(v - best) < 5e-3:
        s = f'**{s}**'
    elif best - v <= MARGIN + 1e-9:
        s = f'*{s}*'
    return s + (f'<br><sub>[{sd:.2f}]</sub>' if pd.notna(sd) else '')


def seed_stats(sub, metric):
    """Per city: (mean, std) over seeds. Average: mean over cities, std of the per-seed 3-house mean."""
    p = sub.pivot_table(index='seed', columns='city', values=metric, aggfunc='mean').reindex(columns=CITIES)
    per_city = {c: (p[c].mean(), p[c].std()) for c in CITIES}
    means = [m for m, _ in per_city.values() if pd.notna(m)]
    per_seed = p.dropna().mean(axis=1)
    avg = (np.mean(means) if means else np.nan, per_seed.std() if len(per_seed) > 1 else np.nan)
    return per_city, avg


def city_row(sub):
    row = {}
    stats = {k: seed_stats(sub, m) for k, m in (('W', 'f1_weighted'), ('M', 'f1_macro'))}
    for c in CITIES:
        for k in ('W', 'M'):
            row[(c, k)] = stats[k][0][c]
    for k in ('W', 'M'):
        row[('average', k)] = stats[k][1]
    return row


def clustering_blocks(summary, level, variant, m):
    x = summary[(summary.family == 'clustering') & (summary.level == level)
                & (summary.variant == variant) & (summary.budget == f'm{m}')]
    for K in (20, 50, 100):
        rows = []
        for method, rep in CLU_ORDER:
            sub = x[(x.method == method) & (x.rep == rep) & (x.K == K)]
            if sub.empty:
                continue
            rows.append((METHOD_NAMES[method], REP_NAMES.get(rep, rep), city_row(sub)))
        labels = K * (int(m) if m != 'all' else 0)
        header = (f'$K = {K}$ clusters ({labels} labels)' if m != 'all'
                  else f'$K = {K}$ clusters (all labeled windows vote)')
        yield header, rows


def baseline_blocks(summary, level, variant):
    x = summary[(summary.family == 'baseline') & (summary.level == level) & (summary.variant == variant)]
    for budget in ('100', '250', 'all'):
        rows = []
        for method in BASELINES:
            sub = x[(x.method == method) & (x.budget == budget)]
            if sub.empty:
                continue
            rows.append((METHOD_NAMES[method], '—', city_row(sub)))
        yield f'budget = {budget} activity labels' if budget != 'all' else 'all training windows', rows


def probe_blocks(summary, level, variant):
    x = summary[(summary.family == 'probe') & (summary.level == level) & (summary.variant == variant)]
    for budget in ('100', '250', 'all'):
        rows = []
        for rep in ('hist_sensor', 'hist_sensor_tod', 'tf', 'tf+tod', 'bert', 'bert+tod'):
            sub = x[(x.rep == rep) & (x.budget == budget)]
            if sub.empty:
                continue
            rows.append(('Linear probe', REP_NAMES.get(rep, rep), city_row(sub)))
        yield f'budget = {budget}' if budget != 'all' else 'all labels (max 20k)', rows


MATCHED = [
    # (column, clustering filter, budget for probes / baselines)
    ('≈100 labels', dict(budget='m5', K=20), '100'),
    ('≈250 labels', dict(budget='m5', K=50), '250'),
    ('all labels', dict(budget='mall', K=50), 'all'),
]


def comparison_blocks(summary, level, variant):
    """One column per label budget, every method filled in. Clustering: K=20 m=5 / K=50 m=5 / K=50 m=all."""
    x = summary[(summary.level == level) & (summary.variant == variant)]

    def three_house(sub):
        nan = (np.nan, np.nan)
        if sub.city.nunique() < len(CITIES):
            return nan, nan
        return seed_stats(sub, 'f1_weighted')[1], seed_stats(sub, 'f1_macro')[1]

    def row_for(select):
        row = {}
        for name, clu, budget in MATCHED:
            row[(name, 'W')], row[(name, 'M')] = three_house(select(clu, budget))
        return row

    clu = x[x.family == 'clustering']
    rows = [(METHOD_NAMES[m], REP_NAMES[r], row_for(
        lambda c, b, m=m, r=r: clu[(clu.method == m) & (clu.rep == r) & (clu.budget == c['budget']) & (clu.K == c['K'])]))
        for m, r in [('scan', 'tf+tod'), ('kmeans', 'tf+tod'), ('kmeans', 'hist_sensor_tod'), ('kmeans', 'bert+tod'),
                     ('dec', 'tf+tod')]]
    yield r'Clustering ($K{=}20, m{=}5$ / $K{=}50, m{=}5$ / $K{=}50, m{=}$all)', rows

    probe = x[x.family == 'probe']
    rows = [('Linear probe', REP_NAMES[r], row_for(lambda c, b, r=r: probe[(probe.rep == r) & (probe.budget == b)]))
            for r in ('tf+tod', 'hist_sensor_tod', 'bert+tod')]
    yield 'Linear probe on the same features (random activity labels)', rows

    base = x[x.family == 'baseline']
    rows = [(METHOD_NAMES[m], 'end-to-end', row_for(lambda c, b, m=m: base[(base.method == m) & (base.budget == b)]))
            for m in BASELINES]
    yield 'Supervised baselines (random activity labels)', rows


def best_of(rows):
    return pd.DataFrame([{k: v[0] for k, v in r.items()} for *_, r in rows]).max()


def to_markdown(title, blocks, table_best=False):
    lines = [f'### {title}', '']
    blocks = [(h, rows) for h, rows in blocks if rows]
    overall = best_of([r for _, rows in blocks for r in rows]) if blocks else None
    for header, rows in blocks:
        cols = list(rows[0][2].keys())
        best = overall if table_best else best_of(rows)
        lines += [f'**{header}**', '',
                  '| Method | Representation | ' + ' | '.join(
                      f'{c if isinstance(c, str) and " " in str(c) else (c[0].capitalize() if isinstance(c, tuple) else c)} {k}'
                      for c, k in cols) + ' |',
                  '|---|---|' + '---:|' * len(cols)]
        for method, rep, row in rows:
            lines.append('| {} | {} | {} |'.format(
                method, rep, ' | '.join(cell(row[c], best[c], 'md') for c in cols)))
        lines.append('')
    return '\n'.join(lines)


def tex_header(g):
    return g.replace('≈', r'$\approx$') if isinstance(g, str) and ' ' in g else str(g).capitalize()


def to_latex(title, blocks, key, table_best=False):
    blocks = [(h, rows) for h, rows in blocks if rows]
    if not blocks:
        return f'% empty {key}\n'
    cols = list(blocks[0][1][0][2].keys())
    groups = list(dict.fromkeys(c for c, _ in cols))
    ncol = 2 + len(cols)
    spec = 'll' + '|'.join(['cc'] * len(groups))
    out = [r'\begin{table}[H]', r'\centering\small',
           rf'\caption{{{title}. W/M = weighted/macro F1 (mean over seeds); bold = best in column '
           rf'{"over the whole table" if table_best else "within a block"}, shaded = within {MARGIN:.2f} of it; '
           r'grey [\,$\cdot$\,] = std over seeds (Average: std of the per-seed three-house mean; '
           r'none = single seed).}',
           rf'\label{{tab:{key}}}', r'\setlength{\tabcolsep}{3.2pt}',
           rf'\begin{{tabular}}{{{spec}}}', r'\toprule',
           r'Method & Repr. & ' + ' & '.join(
               rf'\multicolumn{{2}}{{c}}{{{tex_header(g)}}}'
               for g in groups) + r' \\',
           ''.join(rf'\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}' for i in range(len(groups))),
           r' & & ' + ' & '.join(['W & M'] * len(groups)) + r' \\']
    overall = best_of([r for _, rows in blocks for r in rows])
    for header, rows in blocks:
        best = overall if table_best else best_of(rows)
        out += [r'\midrule', rf'\multicolumn{{{ncol}}}{{l}}{{\textit{{{header}}}}} \\', r'\midrule']
        for method, rep, row in rows:
            out.append(f'{method} & {rep} & ' + ' & '.join(cell(row[c], best[c], 'tex') for c in cols) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(out)


PER_HOUSE = [
    (r'\ToolName{} (SCAN)', 'clustering', dict(method='scan', rep='tf+tod')),
    (r'\ToolName{} with k-means', 'clustering', dict(method='kmeans', rep='tf+tod')),
    ('Linear probe', 'probe', dict(rep='tf+tod')),
    ('DeepCASAS', 'baseline', dict(method='deepcasas')),
    ('TDOST', 'baseline', dict(method='tdost')),
]


def per_house_latex(runs, level, variant, title):
    """Paper appendix table: houses x main methods, matched budgets, mean with grey seed std (`\\sd`)."""
    x = runs[(runs.level == level) & (runs.variant == variant)]
    lines = [r'\begin{tabular}{ll cc cc cc}', r'\toprule',
             r' & & \multicolumn{2}{c}{$\approx$100 labels} & \multicolumn{2}{c}{$\approx$250 labels} '
             r'& \multicolumn{2}{c}{all labels} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
             r'House & Method & W & M & W & M & W & M \\']
    for city in CITIES:
        lines.append(r'\midrule')
        cells = []
        for name, fam, filt in PER_HOUSE:
            row = []
            for _, clu, budget in MATCHED:
                q = x[(x.family == fam) & (x.city == city)]
                for k, val in (filt | (clu if fam == 'clustering' else {'budget': budget})).items():
                    q = q[q[k] == val]
                row += [(q[mt].mean(), q[mt].std()) for mt in ('f1_weighted', 'f1_macro')]
            cells.append((name, row))
        best = [np.nanmax([r[i][0] for _, r in cells]) for i in range(6)]
        for j, (name, row) in enumerate(cells):
            vals = ['--' if pd.isna(m) else (rf'\textbf{{{m:.2f}}}' if abs(m - b) < 5e-3 else f'{m:.2f}')
                    + (rf'\sd{{{s:.2f}}}'.replace('0.', '.', 1) if pd.notna(s) else '')
                    for (m, s), b in zip(row, best)]
            lines.append((HOUSE_NAMES[city] if j == 0 else '') + f' & {name} & ' + ' & '.join(vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return f'% generated by report_final.py ({title}, {variant}); requires \\sd and \\ToolName\n' + '\n'.join(lines) + '\n'


def write_tables(summary):
    runs_or_summary = summary
    tables = OUT / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    sections = []
    evals = [
        ('l1', 'masked', 'L1 masked (primary)'),
        ('l1', 'all', 'L1 all classes'),
        ('l2', 'all', 'L2 all classes'),
    ]
    for m in ('5', '1', 'all'):
        parts = []
        for lv, v, t in evals:
            blocks = list(clustering_blocks(summary, lv, v, m))
            parts.append((to_markdown(f'{t}; $m={m}$', blocks),
                          to_latex(f'Clustering, {t}; $m={m}$', blocks, f'clu_{lv}_{v}_m{m}')))
        sections.append((f'clustering_m{m}', f'Clustering, $m={m}$', parts))

    parts = []
    for lv, v, t in evals:
        blocks = list(baseline_blocks(summary, lv, v if lv == 'l1' or v != 'masked' else 'all'))
        # L2 masked less interesting; keep all/excl for l2
        if lv == 'l2':
            blocks = list(baseline_blocks(summary, 'l2', 'all'))
        parts.append((to_markdown(f'Baselines, {t}', blocks),
                      to_latex(f'Supervised baselines, {t}', blocks, f'base_{lv}_{v}')))
    sections.append(('baselines', 'Supervised baselines (DeepCASAS, TDOST)', parts))

    if (summary.family == 'probe').any():
        parts = []
        for lv, v, t in evals:
            blocks = list(probe_blocks(summary, lv, v if not (lv == 'l2' and v == 'masked') else 'all'))
            parts.append((to_markdown(f'Probes, {t}', blocks),
                          to_latex(f'Linear probes, {t}', blocks, f'probe_{lv}_{v}')))
        sections.append(('probes', 'Linear probes on frozen representations', parts))

    parts = []
    for lv, v, t in [('l1', 'masked', 'L1 masked'), ('l2', 'all', 'L2 all')]:
        blocks = list(comparison_blocks(summary, lv, v))
        parts.append((to_markdown(f'Matched budgets, {t}', blocks, table_best=True),
                      to_latex(f'Clustering vs probes vs supervised at matched label budgets, {t}', blocks,
                               f'cmp_{lv}_{v}', table_best=True)))
    sections.append(('matched_budgets', 'Clustering vs supervised at matched label budgets', parts))

    for lv, v, t in [('l1', 'masked', 'L1'), ('l2', 'all', 'L2')]:
        (tables / f'per_house_{lv}.tex').write_text(per_house_latex(runs_or_summary, lv, v, t))

    for stem, _, parts in sections:
        (tables / f'{stem}.md').write_text('\n\n'.join(md for md, _ in parts) + '\n')
        (tables / f'{stem}.tex').write_text(
            '% requires booktabs, float, [table]xcolor\n' + '\n\n'.join(tex for _, tex in parts) + '\n')
    return sections


def make_figures(summary):
    figdir = OUT / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'figure.dpi': 140})

    # 1) Clustering K=50 m5 L1 masked: methods × reps, 3-house avg macro
    x = summary[(summary.family == 'clustering') & (summary.level == 'l1') & (summary.variant == 'masked')
                & (summary.budget == 'm5') & (summary.K == 50)]
    avg = x.groupby(['method', 'rep']).apply(
        lambda d: d.groupby('city')[['f1_weighted', 'f1_macro']].mean().mean()).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    reps = ['hist_sensor', 'hist_sensor_tod', 'tf', 'tf+tod', 'bert', 'bert+tod']
    methods = ['kmeans', 'scan', 'dec']
    width = 0.25
    xs = np.arange(len(reps))
    for i, method in enumerate(methods):
        vals = []
        for rep in reps:
            sub = avg[(avg.method == method) & (avg.rep == rep)]
            vals.append(sub.f1_macro.iloc[0] if len(sub) else np.nan)
        ax.bar(xs + (i - 1) * width, vals, width, label=METHOD_NAMES[method])
    ax.set_xticks(xs)
    ax.set_xticklabels([REP_NAMES[r] for r in reps], rotation=25, ha='right')
    ax.set_ylabel('Macro F1 (3-house mean)')
    ax.set_title('Clustering L1 masked, $K=50$, $m=5$ (~250 labels)')
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False)
    ax.axhline(0.55, color='0.7', ls='--', lw=0.8)
    fig.tight_layout()
    fig.savefig(figdir / 'clustering_k50_m5_macro.pdf')
    fig.savefig(figdir / 'clustering_k50_m5_macro.png')
    plt.close(fig)

    # 2b) m=5 vs m=all at K=50 (show saturation)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    setups = [
        ('kmeans', 'tf+tod'), ('kmeans', 'hist_sensor_tod'), ('scan', 'tf+tod'),
        ('dec', 'tf+tod'), ('kmeans', 'bert+tod'),
    ]
    xs = np.arange(len(setups))
    for i, (budget, label) in enumerate([('m5', '$m=5$ (~250)'), ('mall', '$m=$all')]):
        vals = []
        for method, rep in setups:
            sub = summary[(summary.family == 'clustering') & (summary.level == 'l1')
                          & (summary.variant == 'masked') & (summary.budget == budget)
                          & (summary.K == 50) & (summary.method == method) & (summary.rep == rep)]
            vals.append(sub.groupby('city').f1_macro.mean().mean() if len(sub) else np.nan)
        ax.bar(xs + (i - 0.5) * 0.35, vals, 0.35, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{METHOD_NAMES[m]}\n{REP_NAMES[r]}' for m, r in setups], fontsize=8)
    ax.set_ylabel('Macro F1 (3-house)')
    ax.set_title('Clustering $K=50$: $m=5$ vs $m=$all (L1 masked)')
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / 'clustering_m5_vs_mall.pdf')
    fig.savefig(figdir / 'clustering_m5_vs_mall.png')
    plt.close(fig)

    # 3) Per-house K=50 m5 kmeans tf+tod vs hist vs baselines@250
    fig, ax = plt.subplots(figsize=(7.5, 4))
    setups = [
        ('kmeans tf+tod', 'clustering', dict(method='kmeans', rep='tf+tod', budget='m5', K=50)),
        ('kmeans hist+ToD', 'clustering', dict(method='kmeans', rep='hist_sensor_tod', budget='m5', K=50)),
        ('DeepCASAS@250', 'baseline', dict(method='deepcasas', budget='250')),
        ('TDOST@250', 'baseline', dict(method='tdost', budget='250')),
        ('TDOST@all', 'baseline', dict(method='tdost', budget='all')),
    ]
    x = np.arange(len(CITIES))
    width = 0.15
    for i, (name, fam, filt) in enumerate(setups):
        vals = []
        for city in CITIES:
            q = summary[(summary.family == fam) & (summary.level == 'l1') & (summary.variant == 'masked')
                        & (summary.city == city)]
            for k, v in filt.items():
                q = q[q[k] == v]
            vals.append(q.f1_macro.mean() if len(q) else np.nan)
        ax.bar(x + (i - 2) * width, vals, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([HOUSE_NAMES[c] for c in CITIES])
    ax.set_ylabel('Macro F1')
    ax.set_title('Per-house L1 masked (clustering $K=50,m=5$ vs baselines)')
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, 0.75)
    fig.tight_layout()
    fig.savefig(figdir / 'per_house_macro.pdf')
    fig.savefig(figdir / 'per_house_macro.png')
    plt.close(fig)

    # 4) BERT vs TF gate
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    pairs = []
    for method in ('kmeans', 'scan', 'dec'):
        for tod in ('', '+tod'):
            tf = summary[(summary.family == 'clustering') & (summary.level == 'l1') & (summary.variant == 'masked')
                         & (summary.budget == 'm5') & (summary.K == 50) & (summary.method == method)
                         & (summary.rep == f'tf{tod}')]
            bert = summary[(summary.family == 'clustering') & (summary.level == 'l1') & (summary.variant == 'masked')
                           & (summary.budget == 'm5') & (summary.K == 50) & (summary.method == method)
                           & (summary.rep == f'bert{tod}')]
            if tf.empty or bert.empty:
                continue
            tf_m = tf.groupby('city').f1_macro.mean().mean()
            bert_m = bert.groupby('city').f1_macro.mean().mean()
            pairs.append((f'{METHOD_NAMES[method]}\n{tod or "emb"}', tf_m, bert_m))
    xs = np.arange(len(pairs))
    ax.bar(xs - 0.2, [p[1] for p in pairs], 0.4, label='TF (5 seeds)')
    ax.bar(xs + 0.2, [p[2] for p in pairs], 0.4, label='BERT (1 seed)')
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in pairs])
    ax.set_ylabel('Macro F1 (3-house)')
    ax.set_title('Encoder gate: BERT vs our transformer ($K=50$, $m=5$)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / 'bert_vs_tf.pdf')
    fig.savefig(figdir / 'bert_vs_tf.png')
    plt.close(fig)

    # 5) Seed std for top clustering setups
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = summary[(summary.family == 'clustering') & (summary.level == 'l1') & (summary.variant == 'masked')
                & (summary.budget == 'm5') & (summary.K == 50) & (summary.rep.isin(['tf+tod', 'hist_sensor_tod']))
                & (summary.method == 'kmeans')]
    # need runs not summary for std across seeds after city-average... use city-level means then std of house-avg? 
    # Better: per-seed 3-house macro
    runs = getattr(make_figures, '_runs', None)
    print(f'Saved figures in {figdir}')
    return figdir


def three_house_per_seed(runs, metric, cities=None, **filt):
    """Per-seed mean over the houses (L1 masked); seeds missing a house are dropped."""
    q = runs[(runs.level == 'l1') & (runs.variant == 'masked')]
    for k, v in filt.items():
        q = q[q[k] == v]
    p = q.pivot_table(index='seed', columns='city', values=metric, aggfunc='mean').reindex(columns=cities or CITIES)
    return p.dropna().mean(axis=1)


SEEDS = [1, 2, 3, 4, 5]
N_BOOT = 2000
MAIN_BASELINES = ('deepcasas', 'tdost')


def discover_preds(city, seed, K, m, pool_labels):
    z = np.load(ARTIFACTS_DIR / 'assignments' / f'{city}__tf_s{seed}+tod__scan_k{K}_todwbal.npz')
    mapping = map_clusters(z['pool_cluster'], z['pool_conf'], pool_labels, m)
    return np.array([mapping.get(c, UNMAPPED) for c in z['test_cluster']])


def boot_f1(y, preds, day_idx, W, macro_labels):
    """Mean over seeds of weighted F1 (all true classes) and macro F1 (`macro_labels`) for every bootstrap
    day-weighting in W (n_boot x n_days). Returns two arrays of shape (n_boot,)."""
    classes = sorted(set(y) | {l for p in preds for l in np.unique(p)})
    ci = {c: i for i, c in enumerate(classes)}
    D, C = W.shape[1], len(classes)
    yi = np.array([ci[l] for l in y])
    support = np.zeros((D, C))
    np.add.at(support, (day_idx, yi), 1)
    S = W @ support
    keep = np.array([ci[l] for l in macro_labels])
    w_out, m_out = [], []
    for p in preds:
        pi = np.array([ci[l] for l in p])
        tp, pc = np.zeros((D, C)), np.zeros((D, C))
        np.add.at(tp, (day_idx[yi == pi], yi[yi == pi]), 1)
        np.add.at(pc, (day_idx, pi), 1)
        TP, PC = W @ tp, W @ pc
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = np.where(S + PC > 0, 2 * TP / (S + PC), 0.0)
        w_out.append((f1 * S).sum(1) / S.sum(1))
        m_out.append(f1[:, keep].mean(1))
    return np.mean(w_out, 0), np.mean(m_out, 0)


def significance(runs, n_boot=N_BOOT, seed=0):
    """Paired day-level block bootstrap of DISCOVER minus the best supervised baseline (per house, budget, metric;
    best = higher mean over seeds; for the three-house average, higher three-house mean). Same resampled test days
    for both methods; the average resamples each house independently. Two-sided p from the bootstrap distribution of the difference."""
    rng = np.random.default_rng(seed)
    x = runs[(runs.level == 'l1') & (runs.variant == 'masked')]
    rows, diffs = [], {}
    for city in CITIES:
        _, pool, test = load_protocol(city, level='l1')
        yp, yt = pool['label'].to_numpy(), test['label'].to_numpy()
        days, day_idx = np.unique(test['day'].to_numpy(), return_inverse=True)
        W = np.stack([np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days)) for _ in range(n_boot)])
        masked = scored_classes(yp, yt)
        for col, clu, budget in MATCHED:
            m = 'all' if clu['budget'] == 'mall' else int(clu['budget'][1:])
            disc = boot_f1(yt, [discover_preds(city, s, clu['K'], m, yp) for s in SEEDS], day_idx, W, masked)
            base = {}
            for meth in MAIN_BASELINES:
                preds = [np.load(BASELINES_DIR / 'final' / meth / f'{city}_l1_{budget}_s{s}.npz', allow_pickle=True)
                         ['pred'].astype(str) for s in SEEDS]
                base[meth] = boot_f1(yt, preds, day_idx, W, masked)
            for k, metric in enumerate(('f1_weighted', 'f1_macro')):
                means = {meth: x[(x.family == 'baseline') & (x.method == meth) & (x.city == city)
                                 & (x.budget == budget)][metric].mean() for meth in MAIN_BASELINES}
                best = max(means, key=means.get)
                for meth in MAIN_BASELINES:
                    diffs[(city, col, metric, meth)] = disc[k] - base[meth][k]
                d = diffs[(city, col, metric, best)]
                rows.append({'city': city, 'budget': col, 'metric': metric, 'best_baseline': best,
                             'diff_boot_mean': d.mean(), 'ci_low': np.quantile(d, 0.025), 'ci_high': np.quantile(d, 0.975),
                             'p': min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))})
        print(f'  significance {city}', flush=True)
    for col, _, budget in MATCHED:
        for metric in ('f1_weighted', 'f1_macro'):
            means = {meth: x[(x.family == 'baseline') & (x.method == meth) & (x.budget == budget)]
                     .groupby('city')[metric].mean().mean() for meth in MAIN_BASELINES}
            best = max(means, key=means.get)
            d = np.mean([diffs[(c, col, metric, best)] for c in CITIES], 0)
            rows.append({'city': 'average', 'budget': col, 'metric': metric, 'best_baseline': best,
                         'diff_boot_mean': d.mean(), 'ci_low': np.quantile(d, 0.025), 'ci_high': np.quantile(d, 0.975),
                         'p': min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))})
    sig = pd.DataFrame(rows)
    sig.to_csv(OUT / 'data' / 'significance.csv', index=False)
    return sig


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def main_table_split(runs, sig, out_dir, with_sd=True):
    """Section 6 table, one tabular per metric: rows DeepCASAS / TDOST / DISCOVER / delta, columns house x budget.
    Writes main_comparison_{weighted,macro}.tex to out_dir. with_sd: True (inline \\sd), 'stack' (std on a small
    line below, needs \\sdrow), False (none)."""
    x = runs[(runs.level == 'l1') & (runs.variant == 'masked')]
    setups = {'DeepCASAS': ('baseline', dict(method='deepcasas')), 'TDOST': ('baseline', dict(method='tdost')),
              r'\textbf{\ToolName{}}': ('clustering', dict(method='scan', rep='tf+tod'))}
    houses = CITIES + ['average']

    def value(name, city, metric, clu, budget):
        fam, filt = setups[name]
        q = x[x.family == fam]
        for k, v in (filt | (clu if fam == 'clustering' else {'budget': budget})).items():
            q = q[q[k] == v]
        if city == 'average':
            return seed_stats(q, metric)[1]
        qq = q[q.city == city][metric]
        return qq.mean(), qq.std()

    def fmt(v, bold):
        s = rf'\textbf{{{v[0]:.2f}}}' if bold else f'{v[0]:.2f}'
        return s + (rf'\sd{{{v[1]:.2f}}}'.replace('0.', '.', 1) if with_sd is True else '')

    def sd_row(vs):
        return ' & ' + ' & '.join(rf'\sdrow{{{v[1]:.2f}}}'.replace('0.', '.', 1) for v in vs) + r' \\[1pt]'

    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, tag in (('f1_weighted', 'weighted'), ('f1_macro', 'macro')):
        cols = [(city, col, clu, budget) for city in houses for col, clu, budget in MATCHED]
        vals = {n: [value(n, c, metric, clu, b) for c, _, clu, b in cols] for n in setups}
        names = list(setups)
        best = [max(vals[n][i][0] for n in names) for i in range(len(cols))]
        dbl = r'!{\hspace{2pt}\vrule width 0.4pt\hspace{1.2pt}\vrule width 0.4pt\hspace{2pt}}'
        heads = [r'\multicolumn{3}{c' + (dbl if c == CITIES[-1] else '') + '}{'
                 + (HOUSE_NAMES[c] if c != 'average' else r'\textit{Average}') + '}' for c in houses]
        # Double vertical rule before the average group; booktabs rule padding is zeroed (and replaced by row
        # height) so the vertical lines meet the horizontal rules without gaps. The rule's 6pt belong to the Cairo
        # column, so its cmidrule is trimmed by that much more on the right.
        trims = ['lr'] * (len(CITIES) - 1) + [r'l{.5em}r{\dimexpr.5em+6pt}', 'lr']
        lines = [r'\begingroup\setlength{\aboverulesep}{0pt}\setlength{\belowrulesep}{0pt}'
                 r'\setlength{\extrarowheight}{2.5pt}',
                 r'\begin{tabular}{l ' + 'ccc ' * len(CITIES) + dbl + ' ccc}',
                 r'\toprule',
                 ' & ' + ' & '.join(heads) + r' \\',
                 ''.join(rf'\cmidrule({trims[i]}){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(len(houses))),
                 'Method & ' + ' & '.join(['100', '250', 'all'] * len(houses)) + r' \\', r'\midrule']
        for n in names:
            if n == names[-1]:
                lines.append(r'\midrule')
            lines.append(f'{n} & ' + ' & '.join(fmt(v, abs(v[0] - b) < 5e-3) for v, b in zip(vals[n], best))
                         + (r' \\[-2pt]' if with_sd == 'stack' else r' \\'))
            if with_sd == 'stack':
                lines.append(sd_row(vals[n]))
        deltas = []
        for i, (city, col, _, _) in enumerate(cols):
            d = round(vals[names[-1]][i][0], 2) - round(max(vals[n][i][0] for n in names[:2]), 2)
            r = sig[(sig.city == city) & (sig.budget == col) & (sig.metric == metric)].iloc[0]
            kind = 'pos' if d > 5e-3 else 'neg' if d < -5e-3 else 'zero'
            deltas.append(rf'\dlt{kind}{{{abs(d):.2f}}}{{{stars(r.p)}}}')
        lines.append(r'$\Delta$ vs.\ best & ' + ' & '.join(deltas) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup']
        (out_dir / f'main_comparison_{tag}.tex').write_text(
            '% generated by report_final.py (main_table_split); requires \\sd, \\dltpos/neg/zero\n'
            + '\n'.join(lines) + '\n')


def significance_table(sig):
    """Appendix table: bootstrap mean difference (DISCOVER minus best baseline), 95% CI and p, per house."""
    lines = [r'\begin{tabular}{l l ccc ccc}', r'\toprule',
             r' & & \multicolumn{3}{c}{$\Delta F1_W$} & \multicolumn{3}{c}{$\Delta F1_M$} \\',
             r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
             r'House & Labels & $\Delta$ & 95\% CI & $p$ & $\Delta$ & 95\% CI & $p$ \\']
    for city in CITIES + ['average']:
        lines.append(r'\midrule')
        for j, (col, _, budget) in enumerate(MATCHED):
            cells = []
            for metric in ('f1_weighted', 'f1_macro'):
                r = sig[(sig.city == city) & (sig.budget == col) & (sig.metric == metric)].iloc[0]
                p = '$<$0.001' if r.p < 0.001 else f'{r.p:.3f}'
                cells += [f'{r.diff_boot_mean:+.2f}', f'[{r.ci_low:+.2f}, {r.ci_high:+.2f}]', p]
            house = (HOUSE_NAMES[city] if city != 'average' else r'\textit{Average}') if j == 0 else ''
            lines.append(f'{house} & {budget} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'tables' / 'significance.tex').write_text(
        '% generated by report_final.py (significance_table)\n' + '\n'.join(lines) + '\n')


RESIDENTS = {'milan': 1, 'aruba': 1, 'cairo': 2, 'orange': 1}


def dataset_tables():
    """Section 4 summary (dataset_summary.tex) and appendix L2/L1 label distribution (label_distribution.tex),
    over continuous training windows and pre-segmented test windows of the evaluation protocol."""
    from .labels import tdost_label
    from .mapping_eval import CACHE_DIR, DATA_ROOT, NO_ACTIVITY, load_windows

    def name(label):
        return label.replace('_', ' ').capitalize().replace('tv', 'TV').replace('rm ', 'room ')

    def pct(n, total):
        if n == 0:
            return '0'
        p = 100 * n / total
        return f'{n:,}'.replace(',', '{,}') + (r' ($<$0.1)' if 0 < p < 0.05 else f' ({p:.1f})')

    meta = json.loads((PROJECT_DIR.parent / 'metadata' / 'casas_metadata.json').read_text())

    def kind(city, sensor):
        """CASAS: M = motion, D = door; Orange4Home: presence = motion, door/cupboard/drawer contacts = door,
        switches, lights, shutters and TV status = other."""
        if re.match(r'^[MD]\d', sensor):
            return {'M': 'motion', 'D': 'door'}[sensor[0]]
        t = meta[city].get('sensor_type', {}).get(sensor)
        return None if t is None else {'presence': 'motion', 'door': 'door'}.get(t, 'other')  # None: malformed id

    summary, dist = {}, []
    for city in CITIES:
        _, pool, test = load_protocol(city, level='l1')
        cont = [load_windows(DATA_ROOT / city / 'FL_20', s, CACHE_DIR) for s in ('train', 'val', 'test')]
        sensors = {s for df in cont for seq in df['sensors'] for s in seq}
        pool, test = (df.assign(l2=df['label_raw'].map(tdost_label)) for df in (pool, test))
        counts = pd.concat([pool.groupby(['l2', 'label']).size().rename('train'),
                            test.groupby(['l2', 'label']).size().rename('test')], axis=1).fillna(0).astype(int)
        counts = counts.reset_index('l2')
        acts = counts.drop(NO_ACTIVITY, errors='ignore')
        summary[city] = [
            RESIDENTS[city], len({d for df in cont for d in df['day']}),
            ' / '.join(str(sum(kind(city, s) == k for s in sensors)) for k in ('motion', 'door', 'other')),
            f'{len(pool):,}'.replace(',', '{,}'), f"{100 * (pool['label'] == NO_ACTIVITY).mean():.0f}\\%",
            f'{len(test):,}'.replace(',', '{,}'),
            len({r.lower() for r in pd.concat([pool, test])['label_raw']} - {NO_ACTIVITY.lower()}),
            len(acts), len(scored_classes(pool['label'], test['label'])), acts['l2'].nunique()]
        dist.append((city, counts))

    rows = ['Residents', 'Days', 'Motion / door / other sensors', 'Training windows (continuous)',
            r'\quad without activity label', 'Test windows (pre-segmented)',
            'Annotated activities', 'L1 activities', r'\quad frequent (scored in $F1_M$)',
            'L2 classes (DeepCASAS, TDOST)']
    lines = [r'\begin{tabular}{l ' + 'r' * len(CITIES) + '}', r'\toprule',
             ' & ' + ' & '.join(HOUSE_NAMES[c] for c in CITIES) + r' \\', r'\midrule']
    for i, r in enumerate(rows):
        if r == 'Annotated activities':
            lines.append(r'\midrule')
        lines.append(f'{r} & ' + ' & '.join(str(summary[c][i]) for c in CITIES) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'tables' / 'dataset_summary.tex').write_text(
        '% generated by report_final.py (dataset_tables)\n' + '\n'.join(lines) + '\n')

    lines = [r'\begin{tabular}{l rr l rr}', r'\toprule',
             r'\multicolumn{3}{c}{L2 class (DeepCASAS / TDOST)} & \multicolumn{3}{c}{L1 activity} \\',
             r'\cmidrule(lr){1-3}\cmidrule(lr){4-6}',
             r'Class & Train & Test & Activity & Train & Test \\']
    for city, counts in dist:
        n_tr, n_te = counts['train'].sum(), counts['test'].sum()
        lines += [r'\midrule', rf'\multicolumn{{6}}{{l}}{{\textbf{{{HOUSE_NAMES[city]}}}}} \\']
        l2 = counts.groupby('l2')[['train', 'test']].sum().sort_values('train', ascending=False)
        l2 = l2.loc[[c for c in l2.index if c != NO_ACTIVITY] + [c for c in l2.index if c == NO_ACTIVITY]]
        for cls, tot in l2.iterrows():
            sub = counts[counts['l2'] == cls].sort_values('train', ascending=False)
            for j, (act, r) in enumerate(sub.iterrows()):
                head = [name(cls), pct(tot['train'], n_tr), pct(tot['test'], n_te)] if j == 0 else ['', '', '']
                if cls == NO_ACTIVITY:
                    head, act = [r'\textit{No activity}', pct(tot['train'], n_tr), '--'], ''
                lines.append(' & '.join(head + [name(act) if act else '', pct(r['train'], n_tr) if act else '',
                                                pct(r['test'], n_te) if act else '']) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'tables' / 'label_distribution.tex').write_text(
        '% generated by report_final.py (dataset_tables)\n' + '\n'.join(lines) + '\n')


FIG_FONT = 'Helvetica Neue, Helvetica, Arial, sans-serif'
FIG_SERIES = {  # stroke, fill opacity of the seed band, line width, dash
    'DISCOVER': ('#6d28d9', 0.14, 2.6, None),
    'DISCOVER linear probe': ('#94a3b8', 0.10, 1.6, '4 3'),
    'DeepCASAS': ('#ea580c', 0.12, 2.0, None),
    'TDOST': ('#be185d', 0.12, 2.0, None),
}


CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'


def svg_to_pdf(svg_path):
    """Print an SVG to a one-page PDF with headless Chrome, which embeds TrueType fonts (rsvg-convert emits Type 3
    fonts, which ACM's PDF checks reject)."""
    svg = svg_path.read_text()
    w, h = (re.search(rf'{k}="([\d.]+)"', svg).group(1) for k in ('width', 'height'))
    html = svg_path.with_suffix('.html')
    html.write_text(f'<html><head><style>@page{{size:{w}px {h}px;margin:0}}html,body{{margin:0}}</style></head>'
                    f'<body>{svg}</body></html>')
    # Chrome writes the PDF within seconds but can hang for a minute on teardown and then exit non-zero, so wait
    # for the file to appear and stop growing, then end the process.
    import time
    pdf = svg_path.with_suffix('.pdf')
    pdf.unlink(missing_ok=True)
    proc = subprocess.Popen([CHROME, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                             f'--print-to-pdf={pdf}', html.resolve().as_uri()],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    size, deadline = -1, time.time() + 120
    while time.time() < deadline:
        time.sleep(0.5)
        if pdf.exists() and pdf.stat().st_size > 0 and pdf.stat().st_size == size:
            break
        size = pdf.stat().st_size if pdf.exists() else -1
        if proc.poll() is not None and not pdf.exists():
            break
    if proc.poll() is None:
        proc.kill()
    proc.wait()
    if not pdf.exists() or pdf.stat().st_size == 0:
        raise RuntimeError(f'headless Chrome did not write {pdf}')
    html.unlink()


def text_width(text, size, weight):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    return TextPath((0, 0), text, size=size, prop=FontProperties(family=['Helvetica Neue', 'Helvetica', 'Arial'], weight=weight)).get_extents().width


def dash_attr(dash):
    return f' stroke-dasharray="{dash}"' if dash else ''


def label_efficiency_svg(curves):
    """Section 6 figure as a self-contained SVG: one panel per metric, mean over houses and seeds with a +-1 std band.
    `curves[(metric, series)] = (mean, std)` over the budgets 100 / 250 / all."""
    panels = [('F1 Weighted', 'f1_weighted'), ('F1 Macro', 'f1_macro')]
    x_frac, break_frac = (0.0, 0.3, 1.0), 0.65
    ml, pw, gap, mt, ph = 50, 236, 64, 74, 172
    width, height = ml + 2 * pw + gap + 14, mt + ph + 92
    xs_lab = ['100', '250', 'All']
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
           f'font-family="{FIG_FONT}">', f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{width / 2}" y="24" font-size="16" font-weight="500" text-anchor="middle" fill="#0f172a">'
           'DISCOVER vs. Supervised Baselines at Varying Labeling Budgets</text>']
    for i, (title, metric) in enumerate(panels):
        x0 = ml + i * (pw + gap)
        lo = min((m - s).min() for (mt_, _), (m, s) in curves.items() if mt_ == metric)
        hi = max((m + s).max() for (mt_, _), (m, s) in curves.items() if mt_ == metric)
        y_lo, y_hi = np.floor(lo * 10) / 10, np.ceil(hi * 10) / 10
        def X(j):
            return x0 + 22 + x_frac[j] * (pw - 44)
        def Y(v):
            return mt + ph - (v - y_lo) / (y_hi - y_lo) * ph
        out.append(f'<text x="{x0}" y="{mt - 16}" font-size="14" font-weight="400" fill="#334155">{title}</text>')
        for t in np.arange(y_lo, y_hi + 1e-9, 0.1):
            out.append(f'<line x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + pw}" y2="{Y(t):.1f}" stroke="#e2e8f0" stroke-width="0.75"/>'
                       f'<text x="{x0 - 7}" y="{Y(t) + 4:.1f}" font-size="13" text-anchor="end" fill="#334155">{t:.1f}</text>')
        out.append(f'<line x1="{x0}" y1="{mt}" x2="{x0}" y2="{mt + ph}" stroke="#334155" stroke-width="1.1"/>'
                   f'<line x1="{x0}" y1="{mt + ph}" x2="{x0 + pw}" y2="{mt + ph}" stroke="#334155" stroke-width="1.1"/>')
        for j, lab in enumerate(xs_lab):
            out.append(f'<text x="{X(j):.1f}" y="{mt + ph + 19}" font-size="13" text-anchor="middle" fill="#334155">{lab}</text>')
        bx, by = x0 + 22 + break_frac * (pw - 44), mt + ph
        out.append(f'<rect x="{bx - 5}" y="{by - 3}" width="10" height="6" fill="#ffffff"/>'
                   f'<line x1="{bx - 7}" y1="{by + 5}" x2="{bx - 1}" y2="{by - 5}" stroke="#334155" stroke-width="1.1"/>'
                   f'<line x1="{bx + 1}" y1="{by + 5}" x2="{bx + 7}" y2="{by - 5}" stroke="#334155" stroke-width="1.1"/>')
        out.append(f'<text x="{x0 + pw / 2}" y="{mt + ph + 38}" font-size="12.5" text-anchor="middle" fill="#64748b">'
                   'Labeled windows per house</text>')
        for name in list(FIG_SERIES)[::-1]:
            color, alpha, lw, dash = FIG_SERIES[name]
            m, s = curves[metric, name]
            band = [f'{X(j):.1f},{Y(m[j] + s[j]):.1f}' for j in range(3)] + [f'{X(j):.1f},{Y(m[j] - s[j]):.1f}' for j in (2, 1, 0)]
            out.append(f'<polygon points="{" ".join(band)}" fill="{color}" fill-opacity="{alpha}"/>')
        for name in list(FIG_SERIES)[::-1]:
            color, alpha, lw, dash = FIG_SERIES[name]
            m, _ = curves[metric, name]
            pts = ' '.join(f'{X(j):.1f},{Y(v):.1f}' for j, v in enumerate(m))
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{lw}" stroke-linejoin="round" '
                       f'stroke-linecap="round"{dash_attr(dash)}/>')
            for j, v in enumerate(m):
                if name == 'DISCOVER':
                    out.append(f'<circle cx="{X(j):.1f}" cy="{Y(v):.1f}" r="6" fill="{color}" opacity="0.16"/>')
                r, fill = (4.2, color) if name == 'DISCOVER' else (3.4, '#ffffff' if dash else color)
                out.append(f'<circle cx="{X(j):.1f}" cy="{Y(v):.1f}" r="{r}" fill="{fill}" '
                           f'stroke="{color if dash else "#ffffff"}" stroke-width="1.2"/>')
        m, _ = curves[metric, 'DISCOVER']
        for j, v in enumerate(m):
            others = [curves[metric, n][0][j] for n in FIG_SERIES if n != 'DISCOVER']
            span = 19 / ph * (y_hi - y_lo)
            above = not any(0 <= o - v < span for o in others)
            out.append(f'<text x="{X(j):.1f}" y="{Y(v) + (-11 if above else 20):.1f}" font-size="12.5" font-weight="500" '
                       f'text-anchor="middle" fill="{FIG_SERIES["DISCOVER"][0]}">{v:.2f}</text>')
    lx, ly, items = ml, mt + ph + 50, []
    for name, (color, _, lw, dash) in FIG_SERIES.items():
        label = name.replace('DISCOVER linear probe', 'Linear probe (DISCOVER features)')
        items.append((label, color, dash, 1.06 * text_width(label, 12.5, 'normal') + 44))
    total = sum(w for *_, w in items) + 12
    lx = (width - total) / 2
    out.append(f'<rect x="{lx:.1f}" y="{ly}" width="{total:.1f}" height="28" rx="4" fill="#fff" stroke="#e2e8f0"/>')
    cx = lx + 10
    for label, color, dash, w in items:
        y = ly + 14
        out.append(f'<line x1="{cx:.1f}" y1="{y}" x2="{cx + 20:.1f}" y2="{y}" stroke="{color}" stroke-width="2" '
                   f'stroke-linecap="round"{dash_attr(dash)}/>'
                   f'<circle cx="{cx + 10:.1f}" cy="{y}" r="3.2" fill="{"#fff" if dash else color}" stroke="{color if dash else "#fff"}" stroke-width="1.1"/>'
                   f'<text x="{cx + 26:.1f}" y="{y + 4}" font-size="12.5" font-weight="400" fill="{color if not dash else "#64748b"}">{label}</text>')
        cx += w
    out.append('</svg>')
    return '\n'.join(out) + '\n'


def paper_figures(runs):
    """Figures used in the paper: label efficiency (Section 6) and the K x m budget split (appendix)."""
    figdir = OUT / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    clu, probe, base = (runs[runs.family == f] for f in ('clustering', 'probe', 'baseline'))
    budgets = [('100', dict(budget='m5', K=20)), ('250', dict(budget='m5', K=50)), ('all', dict(budget='mall', K=50))]
    series = [  # name, (frame, filter)
        ('DISCOVER', lambda b, c: (clu, dict(method='scan', rep='tf+tod', **c))),
        ('DISCOVER linear probe', lambda b, c: (probe, dict(rep='tf+tod', budget=b))),
        ('DeepCASAS', lambda b, c: (base, dict(method='deepcasas', budget=b))),
        ('TDOST', lambda b, c: (base, dict(method='tdost', budget=b))),
    ]
    curves = {}
    for metric in ('f1_weighted', 'f1_macro'):
        for name, pick in series:
            vals = []
            for b, c in budgets:
                df, filt = pick(b, c)
                vals.append(three_house_per_seed(df, metric, **filt))
            curves[metric, name] = (np.array([v.mean() for v in vals]), np.array([v.std() for v in vals]))
    svg = label_efficiency_svg(curves)
    (figdir / 'label_efficiency.svg').write_text(svg)
    svg_to_pdf(figdir / 'label_efficiency.svg')
    subprocess.run(['rsvg-convert', '-z', '3', '-o', str(figdir / 'label_efficiency.png'),
                    str(figdir / 'label_efficiency.svg')], check=True)

    # K x m: macro F1 against the number of labels, one line per K (SCAN, main features), over the houses that
    # have every K (houses whose K sweep is still running are left out, not averaged in partially)
    full = [c for c in CITIES if set(clu[(clu.city == c) & (clu.method == 'scan') & (clu.rep == 'tf+tod')].K) >= {20, 50, 100}]
    clu_km = clu[clu.city.isin(full)]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for K, color in zip((20, 50, 100), ('#1b9e77', '#d95f02', '#7570b3')):
        xs, mu, sd = [], [], []
        for m in ('1', '5', 'all'):
            v = three_house_per_seed(clu_km, 'f1_macro', method='scan', rep='tf+tod', K=K, budget=f'm{m}', cities=full)
            if len(v):
                xs.append(K * int(m) if m != 'all' else 5000), mu.append(v.mean()), sd.append(v.std())
        if xs:
            ax.errorbar(xs, mu, yerr=sd, marker='o', color=color, capsize=3, label=f'$K={K}$')
    ax.set_xscale('log')
    ax.set_xticks([20, 50, 100, 250, 500, 5000])
    ax.set_xticklabels(['20', '50', '100', '250', '500', 'all'])
    ax.minorticks_off()
    ax.set_xlabel('Activity labels ($K \\times m$; $m \\in \\{1, 5, \\mathrm{all}\\}$)')
    ax.set_ylabel(f'Macro F1 (mean over {len(full)} houses)')
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / 'k_m_budget.pdf', bbox_inches='tight')
    fig.savefig(figdir / 'k_m_budget.png', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved paper figures in {figdir}')


def write_report_md(summary, sections):
    def avg3(fam, **filt):
        q = summary[summary.family == fam]
        for k, v in filt.items():
            q = q[q[k] == v]
        if q.empty:
            return float('nan'), float('nan')
        g = q.groupby('city')[['f1_weighted', 'f1_macro']].mean().mean()
        return float(g.f1_weighted), float(g.f1_macro)

    tf50 = avg3('clustering', level='l1', variant='masked', budget='m5', K=50, method='kmeans', rep='tf+tod')
    hist50 = avg3('clustering', level='l1', variant='masked', budget='m5', K=50, method='kmeans', rep='hist_sensor_tod')
    bert50 = avg3('clustering', level='l1', variant='masked', budget='m5', K=50, method='kmeans', rep='bert+tod')
    tfall = avg3('clustering', level='l1', variant='masked', budget='mall', K=50, method='kmeans', rep='tf+tod')
    histall = avg3('clustering', level='l1', variant='masked', budget='mall', K=50, method='kmeans', rep='hist_sensor_tod')
    dc250 = avg3('baseline', level='l1', variant='masked', budget='250', method='deepcasas')
    td250 = avg3('baseline', level='l1', variant='masked', budget='250', method='tdost')
    tdall = avg3('baseline', level='l1', variant='masked', budget='all', method='tdost')
    dcall = avg3('baseline', level='l1', variant='masked', budget='all', method='deepcasas')
    pr250 = avg3('probe', level='l1', variant='masked', budget='250', rep='hist_sensor_tod')
    prall = avg3('probe', level='l1', variant='masked', budget='all', rep='hist_sensor_tod')
    tf20m1 = avg3('clustering', level='l1', variant='masked', budget='m1', K=20, method='kmeans', rep='tf+tod')
    dc_cairo_all = avg3('baseline', level='l1', variant='masked', budget='all', method='deepcasas', city='cairo')

    n_clu = summary[summary.family == 'clustering'].groupby(['method', 'rep', 'K']).ngroups
    n_base = summary[summary.family == 'baseline'].groupby(['method', 'budget']).ngroups
    seeds_tf = int(summary[(summary.family == 'clustering') & (summary.rep == 'tf+tod')].n_seeds.max()
                   if 'n_seeds' in summary else 5)

    md = f"""# DISCOVER revision — final seeded runs (2026-11-13)

Continuous FL_20 train/val; FL_20_p test only. No coordinates. No_Activity never predicted on test.
Activity-only label budgets. Cells in tables: **mean over seeds** of weighted / macro F1.

**Artifacts:** `tables/*.md|tex`, `figures/*.pdf|png`, `confusion/`, `report.pdf`, scored CSVs in `data/`.

| Coverage | |
|---|---|
| Clustering setups scored | {n_clu} (method × rep × K) |
| Baseline setups | {n_base} (method × budget) |
| TF encoder seeds | 5 houses×seeds complete |
| BERT encoder seeds | 1 (gate) |
| DeepCASAS / TDOST npz | 5 seeds × L1/L2 × 100/250/all |

## Headline numbers (L1 masked, 3-house avg, W/M)

| Setup | Labels | W / M |
|---|---|---|
| k-means **TF+ToD**, $K=50$, $m=5$ | ~250 | **{tf50[0]:.2f} / {tf50[1]:.2f}** |
| k-means Sensor+ToD hist., $K=50$, $m=5$ | ~250 | {hist50[0]:.2f} / {hist50[1]:.2f} |
| k-means BERT+ToD, $K=50$, $m=5$ (1 seed) | ~250 | {bert50[0]:.2f} / {bert50[1]:.2f} |
| DeepCASAS | 250 | {dc250[0]:.2f} / {dc250[1]:.2f} |
| TDOST | 250 | {td250[0]:.2f} / {td250[1]:.2f} |
| k-means TF+ToD, $K=50$, **$m=$all** | all pool labels | {tfall[0]:.2f} / {tfall[1]:.2f} |
| k-means Sensor+ToD hist., $K=50$, $m=$all | all pool labels | {histall[0]:.2f} / {histall[1]:.2f} |
| TDOST | all | **{tdall[0]:.2f} / {tdall[1]:.2f}** |
| DeepCASAS | all | {dcall[0]:.2f} / {dcall[1]:.2f} |

![Label efficiency](figures/label_efficiency.png)

![m=5 vs m=all](figures/clustering_m5_vs_mall.png)

![Clustering K=50](figures/clustering_k50_m5_macro.png)

Interpretation, decisions and next steps: `tist_revision/FINDINGS.md` (curated) and
`tist_revision/EXPLORATIONS.md` (log). This file only holds generated numbers.

![Per-house](figures/per_house_macro.png)

![BERT vs TF](figures/bert_vs_tf.png)

## Tables
"""
    for stem, title, _ in sections:
        md += f'- [`tables/{stem}.md`](tables/{stem}.md) / `.tex` — {title}\n'
    md += """- [`confusion/confusion_matrices.pdf`](confusion/confusion_matrices.pdf) — per-house confusion matrices and
  per-class F1 for TDOST, k-means hist / TF+ToD and the TF+ToD probe (`python -m tist_revision.pipeline.confusion`)

## Caveats
- Means over seeds; day-block bootstrap CIs not in these tables (run `final_scores --n_boot 50` for CIs).
- BERT is a single seed; TF and most baselines are 5 seeds.
- Preliminary if any baseline npz is still landing on skynet.

## Regenerate
```bash
python -m tist_revision.pipeline.report_final --rescore --pdf --probes
```
"""
    (OUT / 'REPORT.md').write_text(md)
    print(f'Saved {OUT / "REPORT.md"}')


def compile_pdf(sections):
    build = OUT / '_build'
    build.mkdir(exist_ok=True)
    body = [r'\section*{Figures}']
    for fig in ['label_efficiency', 'clustering_m5_vs_mall', 'clustering_k50_m5_macro',
                'per_house_macro', 'bert_vs_tf']:
        p = OUT / 'figures' / f'{fig}.pdf'
        if p.exists():
            body += [rf'\begin{{figure}}[H]\centering\includegraphics[width=\textwidth]{{../figures/{fig}.pdf}}',
                     rf'\caption{{{fig.replace("_", " ")}}}', r'\end{figure}']
    for stem, title, _ in sections:
        body += [rf'\section*{{{title}}}', rf'\input{{../tables/{stem}.tex}}', r'\clearpage']
    (build / 'report_final.tex').write_text('\n'.join([
        r'\documentclass[10pt]{article}',
        r'\usepackage[margin=1.6cm]{geometry}',
        r'\usepackage{booktabs,float,graphicx}',
        r'\usepackage[table]{xcolor}',
        r'\usepackage[T1]{fontenc}', r'\usepackage{lmodern}',
        r'\pagestyle{plain}',
        r'\begin{document}',
        r'\section*{DISCOVER revision — final seeded runs}',
        r'Continuous FL\_20 protocol; mean over seeds; W/M = weighted/macro F1.',
        *body,
        r'\end{document}',
    ]))
    subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-quiet', 'report_final.tex'],
                   cwd=build, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.copy(build / 'report_final.pdf', OUT / 'report.pdf')
    shutil.rmtree(build)
    print(f'Saved {OUT / "report.pdf"}')


def main():
    global CITIES, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rescore', action='store_true', help='recompute clustering+baselines (+probes if --probes)')
    ap.add_argument('--probes', action='store_true', help='also score linear probes (slow)')
    ap.add_argument('--n_boot', type=int, default=1)
    ap.add_argument('--pdf', action='store_true')
    ap.add_argument('--cities', nargs='+', default=CITIES,
                    help='houses; others than the paper houses need --out under results/temp')
    ap.add_argument('--out', type=Path, default=FINAL_DIR)
    args = ap.parse_args()
    if args.out.resolve() == FINAL_DIR.resolve() and args.cities != CITIES:
        ap.error('results/final holds the paper houses only; write other house sets to --out results/temp/<name>')
    CITIES, OUT = args.cities, args.out
    OUT.mkdir(parents=True, exist_ok=True)
    final_dir = OUT / 'data'
    final_dir.mkdir(parents=True, exist_ok=True)
    runs_path = final_dir / 'runs.csv'

    source = runs_path if runs_path.exists() else FINAL_DIR / 'data' / 'runs.csv'
    if not args.rescore and source.exists():
        # reuse existing scores; houses not in them yet are scored and appended
        old = pd.read_csv(source)
        missing = [c for c in CITIES if c not in set(old.city)]
        parts = [old[old.city.isin(CITIES)]]
        if missing:
            print(f'Scoring houses missing from {source}: {missing}', flush=True)
            parts += [score_clustering(cities=missing), score_baselines(cities=missing), score_probes(cities=missing)]
        runs = pd.concat(parts, ignore_index=True)
        if missing or source != runs_path:
            runs.to_csv(runs_path, index=False)
    elif args.rescore or not runs_path.exists():
        print('Scoring clustering...', flush=True)
        parts = [score_clustering(n_boot=args.n_boot)]
        print('Scoring baselines...', flush=True)
        parts.append(score_baselines())
        if args.probes:
            print('Scoring probes...', flush=True)
            parts.append(score_probes())
        elif runs_path.exists():
            old = pd.read_csv(runs_path)
            parts.append(old[old.family == 'probe'])
            print('Reusing probe scores from the previous runs.csv', flush=True)
        runs = pd.concat(parts, ignore_index=True)
        runs.to_csv(runs_path, index=False)

    summary = summarize(runs)
    summary.to_csv(final_dir / 'summary.csv', index=False)
    print(f'runs={len(runs)} summary={len(summary)}', flush=True)

    sections = write_tables(runs)
    make_figures._runs = runs
    make_figures(summary)
    paper_figures(runs)
    sig = significance(runs)
    main_table_split(runs, sig, OUT / 'tables', with_sd=False)
    significance_table(sig)
    dataset_tables()
    write_report_md(summary, sections)
    if args.pdf:
        compile_pdf(sections)
    print(f'Done → {OUT}')


if __name__ == '__main__':
    main()
