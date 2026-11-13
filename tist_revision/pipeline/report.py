"""
Report tables (markdown + LaTeX + one combined PDF) from `results/label_levels/results.csv` (clustering)
and `results/probes/results.csv` (linear probes).

Layout of `results/report/`:
    REPORT.md                 hand-written summary
    report_tables.pdf         all tables below, rendered (with --pdf)
    tables/clustering_m{m}.*  clustering: K = 20 / 50 / 100 blocks; m = labelled windows per cluster
    tables/probes.*           linear probes: label budgets 100 / 500 / full
    tables/probe_vs_clustering.*  three-house averages at matched label budgets

Setups are fixed a priori (k-means on every representation, SCAN frozen with entropy weight 2, DEC).
Cells: weighted / macro F1. Bold = best in the column within a block; shaded (blue) = within MARGIN of it.

Usage:
    python -m tist_revision.pipeline.report --pdf
"""
import argparse
import shutil
import subprocess

import pandas as pd

from .mapping_eval import RESULTS_DIR

MARGIN = 0.03
CITIES = ['milan', 'aruba', 'cairo']
EVALS = [('l1', 'masked', 'L1 (per-house activities, resident id removed), rare classes masked'),
         ('l1', 'all', 'L1, all classes'),
         ('l2', 'all', 'L2 (TDOST merged classes), all classes --- comparable to TDOST / DeepCASAS')]
REPS = ['hist_sensor', 'hist_sensor_tod', 'mlm', 'mlm_time', 'mlm+tod', 'mlm_time+tod']
REP_NAMES = {'hist_sensor': 'Sensor hist.', 'hist_sensor_tod': 'Sensor+ToD hist.', 'mlm': 'MLM',
             'mlm_only': 'MLM', 'mlm_time': 'MLM-time', 'mlm+tod': 'MLM + ToD', 'mlm_time+tod': 'MLM-time + ToD'}
METHOD_NAMES = {'kmeans': 'k-means', 'scan': 'SCAN', 'dec': 'DEC', 'probe': 'Linear probe'}
KS = (20, 50, 100)
BUDGETS = ('100', '500', 'full')


def clustering_setups(K):
    return ([('kmeans', r, f'k{K}') for r in REPS]
            + [('scan', e, f'frozen_k{K}_ent2') for e in ['mlm_only', 'mlm_time']]
            + [('dec', e, f'dec_k{K}') for e in ['mlm_only', 'mlm_time']])


def city_row(s, wcol='f1_weighted', mcol='f1_macro'):
    """s: frame indexed by city -> {(city, W/M)} + three-house average."""
    row = {}
    for c in CITIES:
        row[(c, 'W')], row[(c, 'M')] = s[wcol].get(c), s[mcol].get(c)
    row[('average', 'W')], row[('average', 'M')] = s[wcol].reindex(CITIES).mean(), s[mcol].reindex(CITIES).mean()
    return row


def group_of(method, rep):
    return 'hist' if rep.endswith('hist.') else method


def clustering_blocks(r, level, variant, m):
    x = r[(r.level == level) & (r.variant == variant) & (r.m == m)]
    for K in KS:
        rows = []
        for method, enc, run in clustering_setups(K):
            s = x[(x.method == method) & (x.encoder == enc) & (x.run == run)].set_index('city')
            if not s.empty:
                rows.append((METHOD_NAMES[method], REP_NAMES[enc], city_row(s)))
        yield f'$K = {K}$ clusters ({K * int(m)} labels)' if m != 'all' else f'$K = {K}$ clusters', rows


def probe_blocks(p, level, variant):
    x = p[(p.level == level) & (p.variant == variant)]
    for b in BUDGETS:
        rows = []
        for rep in REPS:
            s = x[(x.representation == rep) & (x.budget == b)].set_index('city')
            if not s.empty:
                rows.append(('Linear probe', REP_NAMES[rep], city_row(s)))
        yield (f'{b} random labelled windows' if b != 'full' else 'all labelled windows (max 20k)'), rows


def comparison_blocks(r, p, level, variant):
    """Three-house averages: clustering (m = 5) vs probe at the same number of labels, and both upper bounds."""
    x = r[(r.level == level) & (r.variant == variant)]
    q = p[(p.level == level) & (p.variant == variant)]
    cols = [('K20, m5', 'clu', 20, '5'), ('100 labels', 'probe', '100', None),
            ('K100, m5', 'clu', 100, '5'), ('500 labels', 'probe', '500', None),
            ('K100, m all', 'clu', 100, 'all'), ('all labels', 'probe', 'full', None)]
    rows = []
    for method, enc in [('kmeans', r_) for r_ in REPS] + [('scan', 'mlm_only'), ('dec', 'mlm_only')]:
        row = {}
        for name, kind, a, m in cols:
            if kind == 'clu':
                run = {'kmeans': f'k{a}', 'scan': f'frozen_k{a}_ent2', 'dec': f'dec_k{a}'}[method]
                s = x[(x.method == method) & (x.encoder == enc) & (x.run == run) & (x.m == m)]
            else:
                s = q[(q.representation == enc) & (q.budget == a)] if method == 'kmeans' else q.iloc[:0]
            ok = s.city.nunique() == len(CITIES)
            row[(name, 'W')] = s.f1_weighted.mean() if ok else float('nan')
            row[(name, 'M')] = s.f1_macro.mean() if ok else float('nan')
        rows.append((METHOD_NAMES[method] + ' / probe' if method == 'kmeans' else METHOD_NAMES[method],
                     REP_NAMES[enc], row))
    yield 'three-house average; clustering columns = k-means / SCAN / DEC, probe columns = same representation', rows


def best_of(rows):
    df = pd.DataFrame([r for *_, r in rows])
    return df.max()


def cell(v, best, style):
    if pd.isna(v):
        return '--' if style == 'tex' else '–'
    s = f'{v:.2f}'
    if abs(v - best) < 5e-3:
        return rf'\cellcolor{{blue!12}}\textbf{{{s}}}' if style == 'tex' else f'**{s}**'
    if best - v <= MARGIN + 1e-9:
        return rf'\cellcolor{{blue!12}}{s}' if style == 'tex' else f'*{s}*'
    return s


def to_markdown(title, blocks):
    lines = [f'### {title}', '']
    for header, rows in blocks:
        if not rows:
            continue
        cols = list(rows[0][2].keys())
        best = best_of(rows)
        lines += [f'**{header}**', '', '| Method | Representation | '
                  + ' | '.join(f'{c if " " in c else c.capitalize()} {k}' for c, k in cols) + ' |',
                  '|---|---|' + '---:|' * len(cols)]
        prev = None
        for method, rep, row in rows:
            g = group_of(method, rep)
            if prev is not None and g != prev:
                lines.append('| | |' + ' |' * len(cols))
            prev = g
            lines.append(f'| {method} | {rep} | ' + ' | '.join(cell(row[c], best[c], 'md') for c in cols) + ' |')
        lines.append('')
    return '\n'.join(lines)


def to_latex(title, blocks, key):
    blocks = [(h, rows) for h, rows in blocks if rows]
    cols = list(blocks[0][1][0][2].keys())
    groups = list(dict.fromkeys(c for c, _ in cols))
    ncol = 2 + len(cols)
    spec = 'll' + '|'.join(['cc'] * len(groups))
    out = [r'\begin{table}[H]', r'\centering\small', rf'\caption{{{title}. W/M = weighted/macro F1; bold = best in '
           rf'column within a block, shaded = within {MARGIN:.2f} of it.}}', rf'\label{{tab:{key}}}',
           r'\setlength{\tabcolsep}{3.5pt}', rf'\begin{{tabular}}{{{spec}}}', r'\toprule',
           r'Method & Repr. & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{g if " " in g else g.capitalize()}}}'
                                              for g in groups) + r' \\',
           ''.join(rf'\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}' for i in range(len(groups))),
           r' & & ' + ' & '.join(['W & M'] * len(groups)) + r' \\']
    for header, rows in blocks:
        best = best_of(rows)
        out += [r'\midrule', rf'\multicolumn{{{ncol}}}{{l}}{{\textit{{{header}}}}} \\', r'\midrule']
        prev = None
        for method, rep, row in rows:
            g = group_of(method, rep)
            if prev is not None and g != prev:
                out.append(rf'\cmidrule{{1-{ncol}}}')
            prev = g
            out.append(f'{method} & {rep} & ' + ' & '.join(cell(row[c], best[c], 'tex') for c in cols) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--m', nargs='+', default=['5', '1', 'all'], help='labelled windows per cluster')
    ap.add_argument('--pdf', action='store_true', help='also compile report_tables.pdf (needs latexmk)')
    args = ap.parse_args()
    r = pd.read_csv(RESULTS_DIR / 'label_levels' / 'results.csv')
    r['m'] = r['m'].astype(str)
    probes_path = RESULTS_DIR / 'probes' / 'results.csv'
    p = pd.read_csv(probes_path) if probes_path.exists() else None
    if p is not None:
        p['budget'] = p['budget'].astype(str)
    out = RESULTS_DIR / 'report'
    tables = out / 'tables'
    tables.mkdir(parents=True, exist_ok=True)

    sections = []  # (file stem, section title, [(md, tex)])
    for m in args.m:
        parts = [(to_markdown(f'{t} (m = {m})', list(clustering_blocks(r, lv, v, m))),
                  to_latex(f'Clustering, {t}; $m={m}$', list(clustering_blocks(r, lv, v, m)), f'clu_{lv}_{v}_m{m}'))
                 for lv, v, t in EVALS]
        sections.append((f'clustering_m{m}', f'Clustering, $m = {m}$ labelled windows per cluster', parts))
    if p is not None:
        parts = [(to_markdown(t, list(probe_blocks(p, lv, v))),
                  to_latex(f'Linear probes, {t}', list(probe_blocks(p, lv, v)), f'probe_{lv}_{v}')) for lv, v, t in EVALS]
        sections.append(('probes', 'Linear probes (logistic regression on frozen representations)', parts))
        parts = [(to_markdown(t, list(comparison_blocks(r, p, lv, v))),
                  to_latex(f'Probe vs clustering at matched label budgets, {t}', list(comparison_blocks(r, p, lv, v)),
                           f'cmp_{lv}_{v}')) for lv, v, t in EVALS]
        sections.append(('probe_vs_clustering', 'Linear probe vs clustering at matched label budgets', parts))

    for stem, _, parts in sections:
        (tables / f'{stem}.md').write_text('\n\n'.join(md for md, _ in parts) + '\n')
        (tables / f'{stem}.tex').write_text('% requires booktabs, float, [table]xcolor\n'
                                           + '\n\n'.join(tex for _, tex in parts) + '\n')
    if args.pdf:
        build = out / '_build'
        build.mkdir(exist_ok=True)
        body = []
        for stem, title, _ in sections:
            body += [rf'\section*{{{title}}}', rf'\input{{../tables/{stem}.tex}}', r'\clearpage']
        (build / 'report_tables.tex').write_text('\n'.join([
            r'\documentclass[10pt]{article}', r'\usepackage[margin=1.8cm]{geometry}', r'\usepackage{booktabs}',
            r'\usepackage{float}', r'\usepackage[table]{xcolor}', r'\pagestyle{plain}', r'\begin{document}',
            *body, r'\end{document}', '']))
        subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-quiet', 'report_tables.tex'], cwd=build,
                       check=True, stdout=subprocess.DEVNULL)
        shutil.copy(build / 'report_tables.pdf', out / 'report_tables.pdf')
        shutil.rmtree(build)
        print(f'Saved {out / "report_tables.pdf"}')
    print(f'Saved tables in {tables}')


if __name__ == '__main__':
    main()
