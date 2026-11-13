"""
Comparison tables (markdown + LaTeX) from `results/label_levels/results.csv`.

Fixed setups (chosen a priori, not by test score): k-means on every representation, SCAN frozen with
entropy weight 2, DEC; each at K = 20 and K = 100. Cells: weighted / macro F1, best per column in bold.

Usage:
    python -m tist_revision.pipeline.report --m 5
"""
import argparse
import subprocess

import pandas as pd

from .mapping_eval import RESULTS_DIR

CITIES = ['milan', 'aruba', 'cairo']
EVALS = [('l2', 'all', 'L2 (TDOST classes), all classes - comparable to TDOST / DeepCASAS'),
         ('l2', 'masked', 'L2, rare classes masked from the average'),
         ('l1', 'masked', 'L1 (raw CASAS activities), rare classes masked from the average')]
REP_NAMES = {'hist_sensor': 'Sensor hist.', 'hist_sensor_tod': 'Sensor+ToD hist.', 'mlm': 'MLM',
             'mlm_only': 'MLM', 'mlm_time': 'MLM-time', 'mlm+tod': 'MLM + ToD', 'mlm_time+tod': 'MLM-time + ToD'}
SETUPS = ([('kmeans', r, f'k{K}', K) for r in ['hist_sensor', 'hist_sensor_tod', 'mlm', 'mlm_time', 'mlm+tod',
                                               'mlm_time+tod'] for K in (20, 100)]
          + [('scan', e, f'frozen_k{K}_ent2', K) for e in ['mlm_only', 'mlm_time'] for K in (20, 100)]
          + [('dec', e, f'dec_k{K}', K) for e in ['mlm_only', 'mlm_time'] for K in (20, 100)])


def label(method, enc, K):
    return f'{ {"kmeans": "k-means", "scan": "SCAN", "dec": "DEC"}[method]} | {REP_NAMES[enc]} | {K}'


def build(r, level, variant, m):
    x = r[(r.level == level) & (r.variant == variant) & (r.m == m)]
    rows = []
    for method, enc, run, K in SETUPS:
        s = x[(x.method == method) & (x.encoder == enc) & (x.run == run)].set_index('city')
        if s.empty:
            continue
        row = {'setup': label(method, enc, K)}
        for c in CITIES:
            row[(c, 'W')] = s.f1_weighted.get(c)
            row[(c, 'M')] = s.f1_macro.get(c)
        row[('average', 'W')] = s.f1_weighted.reindex(CITIES).mean()
        row[('average', 'M')] = s.f1_macro.reindex(CITIES).mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index('setup')


def blocks(t):
    """Yield (K, rows) per cluster count; rows = [(group, method, repr, values)], groups separate
    histogram k-means, embedding k-means, SCAN and DEC. Bold = best per column within a K block."""
    parts = [s.split(' | ') for s in t.index]
    for K in sorted({p[2] for p in parts}, key=int):
        sub = t[[p[2] == K for p in parts]]
        rows = []
        for setup, row in sub.iterrows():
            method, rep, _ = setup.split(' | ')
            group = 'hist' if rep.endswith('hist.') else method
            rows.append((group, method, rep, row, sub.max()))
        yield K, rows


def fmt(v, best, bold):
    return bold.format(f'{v:.2f}') if abs(v - best) < 5e-3 else f'{v:.2f}'


def to_markdown(t):
    lines = []
    for K, rows in blocks(t):
        lines += [f'**K = {K}**', '', '| Method | Representation | '
                  + ' | '.join(f'{c.capitalize()} {k}' for c, k in t.columns) + ' |',
                  '|---|---|' + '---:|' * len(t.columns)]
        prev = None
        for group, method, rep, row, best in rows:
            if prev is not None and group != prev:
                lines.append('| | |' + ' |' * len(t.columns))
            prev = group
            lines.append(f'| {method} | {rep} | ' + ' | '.join(fmt(v, best[c], '**{}**') for c, v in row.items()) + ' |')
        lines.append('')
    return '\n'.join(lines)


def to_latex(t, caption, key):
    groups = list(dict.fromkeys(c for c, _ in t.columns))
    ncol = 2 + 2 * len(groups)
    out = [r'\begin{table}[t]', r'\centering\small', rf'\caption{{{caption}}}', rf'\label{{tab:{key}}}',
           r'\setlength{\tabcolsep}{4pt}',
           r'\begin{tabular}{ll' + 'cc' * (len(groups) - 1) + '|cc}', r'\toprule',
           r'Method & Repr. & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{c.capitalize()}}}' for c in groups)
           + r' \\', ''.join(rf'\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}' for i in range(len(groups))),
           r' & & ' + ' & '.join(['W-F1 & M-F1'] * len(groups)) + r' \\']
    for K, rows in blocks(t):
        out += [r'\midrule', rf'\multicolumn{{{ncol}}}{{l}}{{\textit{{$K = {K}$ clusters}}}} \\', r'\midrule']
        prev = None
        for group, method, rep, row, best in rows:
            if prev is not None and group != prev:
                out.append(rf'\cmidrule{{1-{ncol}}}')
            prev = group
            out.append(f'{method} & {rep} & ' + ' & '.join(fmt(v, best[c], r'\textbf{{{}}}') for c, v in row.items())
                       + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--m', default='5', help='labelled windows per cluster')
    ap.add_argument('--pdf', action='store_true', help='also compile a standalone PDF (needs latexmk)')
    args = ap.parse_args()
    r = pd.read_csv(RESULTS_DIR / 'label_levels' / 'results.csv')
    r['m'] = r['m'].astype(str)
    out = RESULTS_DIR / 'report'
    out.mkdir(parents=True, exist_ok=True)
    md, tex = [], []
    for level, variant, title in EVALS:
        t = build(r, level, variant, args.m)
        md.append(f'### {title} (m = {args.m})\n\n' + to_markdown(t))
        tex.append(to_latex(t, f'{title.replace(" - ", " --- ")}; $m={args.m}$ labelled windows per cluster. '
                               'W/M = weighted/macro F1.',
                            f'{level}_{variant}_m{args.m}'))
    (out / f'tables_m{args.m}.md').write_text('\n\n'.join(md) + '\n')
    (out / f'tables_m{args.m}.tex').write_text('% requires \\usepackage{booktabs}\n' + '\n\n'.join(tex) + '\n')
    if args.pdf:
        doc = out / f'tables_m{args.m}_pdf.tex'
        doc.write_text('\n'.join([
            r'\documentclass[10pt]{article}', r'\usepackage[margin=2cm]{geometry}', r'\usepackage{booktabs}',
            r'\usepackage{float}', r'\floatplacement{table}{H}', r'\pagestyle{empty}', r'\begin{document}',
            rf'\input{{tables_m{args.m}.tex}}', r'\end{document}', '']))
        subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-quiet', doc.name], cwd=out, check=True,
                       stdout=subprocess.DEVNULL)
        subprocess.run(['latexmk', '-c', doc.name], cwd=out, check=True, stdout=subprocess.DEVNULL)
        print(f'Saved {doc.with_suffix(".pdf")}')
    print('\n\n'.join(md))


if __name__ == '__main__':
    main()
