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
        rows.append(row)
    return pd.DataFrame(rows).set_index('setup')


def to_markdown(t):
    best = t.max()
    head = '| Method | Representation | K | ' + ' | '.join(f'{c.capitalize()} {k}' for c, k in t.columns) + ' |'
    lines = [head, '|' + '---|' * 3 + '---:|' * len(t.columns)]
    for setup, row in t.iterrows():
        cells = [f'**{v:.2f}**' if abs(v - best[col]) < 5e-3 else f'{v:.2f}' for col, v in row.items()]
        lines.append('| ' + setup + ' | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def to_latex(t, caption, key):
    best = t.max()
    out = [r'\begin{table}[t]', r'\centering\small', rf'\caption{{{caption}}}', rf'\label{{tab:{key}}}',
           r'\begin{tabular}{lll' + 'cc' * len(CITIES) + '}', r'\toprule',
           r'Method & Repr. & $K$ & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{c.capitalize()}}}' for c in CITIES)
           + r' \\', ''.join(rf'\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}' for i in range(len(CITIES))),
           r' & & & ' + ' & '.join(['W-F1 & M-F1'] * len(CITIES)) + r' \\', r'\midrule']
    prev = None
    for setup, row in t.iterrows():
        method = setup.split(' | ')[0]
        if prev is not None and method != prev:
            out.append(r'\midrule')
        prev = method
        cells = [rf'\textbf{{{v:.2f}}}' if abs(v - best[col]) < 5e-3 else f'{v:.2f}' for col, v in row.items()]
        out.append(' & '.join(setup.split(' | ')).replace('_', r'\_') + ' & ' + ' & '.join(cells) + r' \\')
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
