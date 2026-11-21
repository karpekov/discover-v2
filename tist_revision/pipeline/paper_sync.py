"""
Copy generated tables and figures from results/final into the paper (discover-v1/) and check that the paper is
coherent with them.

    python -m tist_revision.pipeline.paper_sync            # sync + check
    python -m tist_revision.pipeline.paper_sync --check    # check only

Checks: LaTeX compiles without undefined references, citations or missing files; every synced per-house table
covers all paper houses; every number quoted in the text (CLAIMS) matches the generated data; remaining \\tbd{}
placeholders are listed (allowed in the appendix only).
"""
import argparse
import json
import re
import shutil
import subprocess

import pandas as pd

from .mapping_eval import FINAL_DIR, PROJECT_DIR
from .report_final import CITIES, HOUSE_NAMES

PAPER = PROJECT_DIR.parent / 'discover-v1'
S5 = FINAL_DIR / 'section5'
SYNC = {  # source under results/final -> destination under discover-v1/
    **{f'tables/{t}.tex': f'tables/{t}.tex' for t in (
        'dataset_summary', 'label_distribution', 'main_comparison_weighted', 'main_comparison_macro',
        'per_house_l1', 'per_house_l2', 'significance')},
    'figures/label_efficiency.pdf': 'figures/label_efficiency.pdf',
    'figures/k_m_budget.pdf': 'figures/k_m_budget.pdf',
    'section5/kappa_scores.tex': 'tables/kappa_scores.tex',
    'section5/random_windows.tex': 'tables/random_windows.tex',
    'section5/cluster_summary.tex': 'tables/cluster_annotation_summary.tex',
    'section5/pdl_taxonomy.tex': 'tables/pdl_taxonomy.tex',
    'section5/annotation_agreement.pdf': 'figures/annotation_agreement.pdf',
    'section5/casas_to_pdl.pdf': 'figures/casas_to_pdl.pdf',
    'section5/pdl_vs_casas.pdf': 'figures/pdl_vs_casas.pdf',
    'section5/tsne_clusters.pdf': 'figures/tsne_clusters.pdf',
}
PER_HOUSE = ['dataset_summary', 'label_distribution', 'main_comparison_weighted', 'main_comparison_macro',
             'per_house_l1', 'per_house_l2', 'significance', 'kappa_scores', 'random_windows',
             'cluster_annotation_summary']


def sync():
    missing = []
    for src, dst in SYNC.items():
        s = FINAL_DIR / src
        if not s.exists():
            missing.append(src)
            continue
        shutil.copy(s, PAPER / dst)
    print(f'synced {len(SYNC) - len(missing)} files' + (f'; not generated yet: {missing}' if missing else ''))
    return missing


# ---------------------------------------------------------------- claims

def pct(x):
    return f'{round(100 * x + 1e-9):.0f}'


def f2(x):
    return f'{x + 1e-9:.2f}'


def span(vals, fmt=f2):
    vals = [fmt(v) for v in vals]
    return f'{min(vals, key=float)}--{max(vals, key=float)}'


def claims():
    """(file, expected text) pairs; the text must appear verbatim in the file."""
    st = json.loads((S5 / 'stats.json').read_text())
    h = {c: st[c] for c in CITIES}
    avg = st['average']['agreement']
    rw = st['average']['random_windows']
    clus = pd.read_csv(S5 / 'clusters.csv')
    s5 = '5_labeling_quality.tex'
    k = [h[c]['kappa'] for c in CITIES]
    out = [
        (s5, f'is at least {f2(min(k))}}}'),
        (s5, r'Milan \(\kappa=' + f2(h['milan']['kappa']) + r'\), Aruba \(\kappa=' + f2(h['aruba']['kappa'])
         + r'\), Cairo \(\kappa=' + f2(h['cairo']['kappa']) + r'\)'),
        (s5, r'Orange4Home \(\kappa=' + f2(h['orange']['kappa']) + r'\) (average '
         + f2(avg['original']['fleiss_window']) + ', 95\\% CI [' + ', '.join(map(f2, avg['original']['fleiss_window_ci'])) + '])'),
        (s5, f'same label for {span([h[c]["exact"] for c in CITIES], pct)}\\% of the samples'),
        (s5, f'at most {max(pct(h[c]["none_rate"]) for c in CITIES)}\\% of the labels'),
        (s5, f'to {span([h[c]["kappa_parent"] for c in CITIES])} (average {f2(avg["level_up"]["fleiss_window"])})'),
    ]
    agr = pd.DataFrame({c: h[c] for c in CITIES}).T
    sc = pd.read_csv(FINAL_DIR.parent / 'temp' / 'annotation_study' / 'v2_k20' / 'scores' / 'final' / 'agreement.csv')
    so = pd.read_csv(FINAL_DIR.parent / 'temp' / 'annotation_study' / 'orange_k20' / 'scores' / 'agreement.csv')
    ag = pd.concat([sc, so]).set_index(['city', 'level'])
    clu = [ag.loc[(c, 'original'), 'fleiss_cluster'] for c in CITIES]
    clu_up = [ag.loc[(c, 'level_up'), 'fleiss_cluster'] for c in CITIES]
    out += [
        (s5, f'{span(clu)} (average {f2(avg["original"]["fleiss_cluster"])})'),
        (s5, f'{span(clu_up)} (average {f2(avg["level_up"]["fleiss_cluster"])})'),
        (s5, f'agree {span(agr.agree_same_window, pct)}\\% of the time, two labels from different samples of the same '
             f'cluster agree {span(agr.agree_same_cluster, pct)}\\% of the time, and labels from different clusters '
             f'agree only {span(agr.agree_other_cluster, pct)}\\%'),
        (s5, f'receives {span(agr.majority_share, pct)}\\% of a cluster\'s labels ({span(agr.majority_share_parent, pct)}\\% '
             f'at the parent level), but {int((clus.share < 0.5).sum())} of the {len(clus)} clusters'),
        (s5, f'all but {["zero", "one", "two", "three", "four"][int((clus[clus.share < 0.5].share_parent < 0.5).sum())]} of them'),
        (s5, f'(average $\\kappa={f2(rw["kappa"])}$), but their labels match the cluster\'s majority label in '
             f'{pct(rw["match"])}\\% of the cases ({pct(rw["match_parent"])}\\% at the parent level), against '
             f'{pct(rw["match_top"])}\\% ({pct(rw["match_top_parent"])}\\%)'),
        (s5, f'({pct(h["milan"]["unlabeled_windows"])}\\% in Milan, {pct(h["aruba"]["unlabeled_windows"])}\\% in Aruba and '
             f'{pct(h["cairo"]["unlabeled_windows"])}\\% in Cairo, against {pct(h["orange"]["unlabeled_windows"])}\\% in '
             f'Orange4Home)'),
        (s5, f'($\\kappa$ {f2(h["milan"]["kappa_unlabeled"])}, {f2(h["aruba"]["kappa_unlabeled"])} and '
             f'{f2(h["cairo"]["kappa_unlabeled"])}; ``None of the above\'\' for {pct(h["milan"]["none_unlabeled"])}\\%, '
             f'{pct(h["aruba"]["none_unlabeled"])}\\% and {pct(h["cairo"]["none_unlabeled"])}\\%'),
    ]
    return out + section4_claims() + section6_claims()


def table_row(table, row):
    """Cells of the row starting with `row` in a generated table, one per house."""
    for line in (FINAL_DIR / 'tables' / f'{table}.tex').read_text().splitlines():
        if line.startswith(row + ' &'):
            return [c.strip().rstrip('\\').strip() for c in line.split('&')[1:]]
    raise KeyError(row)


def section4_claims():
    s4 = '4_data.tex'
    unl = dict(zip(CITIES, (int(c.replace('\\%', '')) for c in table_row('dataset_summary', r'\quad without activity label'))))
    days = dict(zip(CITIES, table_row('dataset_summary', 'Days')))
    casas = [unl[c] for c in CITIES if c != 'orange']
    dist = (FINAL_DIR / 'tables' / 'label_distribution.tex').read_text()
    office = re.search(r'Office computing & [\d{},]+ \([\d.]+\) & [\d{},]+ \(([\d.]+)\)', dist).group(1)
    return [(s4, f'{min(casas)}\\% to {max(casas)}\\% of the continuous training windows'),
            (s4, f'Orange4Home is labeled almost throughout ({unl["orange"]}\\%)'),
            (s4, f'\\textit{{Office computing}} {round(float(office))}\\% of Orange4Home'),
            (s4, f'over {days["orange"]} work days'), (s4, f"of Orange4Home's {days['orange']} days")]


def section6_claims():
    s6 = '6_supervised_benchmark.tex'
    sig = pd.read_csv(FINAL_DIR / 'data' / 'significance.csv')
    sm = pd.read_csv(FINAL_DIR / 'data' / 'summary.csv')
    sm = sm[(sm.level == 'l1') & (sm.variant == 'masked')]
    disc = sm[(sm.family == 'clustering') & (sm.method == 'scan') & (sm.rep == 'tf+tod')]
    probe = sm[(sm.family == 'probe') & (sm.rep == 'tf+tod')]
    budgets = {'100': (20, 'm5'), '250': (50, 'm5'), 'all': (50, 'mall')}

    def d(budget):
        K, m = budgets[budget]
        return disc[(disc.K == K) & (disc.budget == m)].set_index('city').loc[CITIES]

    def gap(city, budget, metric):
        """DISCOVER minus best baseline, as the Delta row of the main table (difference of rounded means)."""
        b = sm[(sm.family == 'baseline') & (sm.budget == budget) & (sm.city == city)].groupby('method')[metric].mean()
        return round(d(budget).loc[city, metric], 2) - round(b.max(), 2)

    def avg(budget, metric):
        return d(budget)[metric].mean()

    lo_sig = sig[(sig.city == 'average') & sig.budget.str.contains('100|250')]
    avg_gaps = []
    for budget in ('100', '250'):
        for metric in ('f1_weighted', 'f1_macro'):
            b = sm[(sm.family == 'baseline') & (sm.budget == budget)].groupby(['method', 'city'])[metric].mean()
            best = b.groupby('method').mean().max()
            avg_gaps.append(round(avg(budget, metric), 2) - round(best, 2))
    all_gaps = []
    for metric in ('f1_weighted', 'f1_macro'):
        b = sm[(sm.family == 'baseline') & (sm.budget == 'all')].groupby(['method', 'city'])[metric].mean()
        all_gaps.append(round(b.groupby('method').mean().max(), 2) - round(avg('all', metric), 2))
    p_max = lo_sig.p.max()
    orange_w = [gap('orange', b, 'f1_weighted') for b in ('100', '250')]
    orange_m = [gap('orange', b, 'f1_macro') for b in ('100', '250')]
    pr = {b: probe[probe.budget == b].set_index('city').loc[CITIES] for b in ('100', '250')}
    pw = [(pr[b]['f1_weighted'] - d(b)['f1_weighted']).mean() for b in ('100', '250')]
    pm = [(pr[b]['f1_macro'] - d(b)['f1_macro']).mean() for b in ('100', '250')]
    return [
        (s6, f'{f2(avg("100", "f1_weighted"))} / {f2(avg("100", "f1_macro"))} with 100 labels'),
        (s6, f'{f2(avg("250", "f1_weighted"))} / {f2(avg("250", "f1_macro"))} with 250 labels, '
             f'{span(avg_gaps)} above the best baseline ($p<{0.001 if p_max < 0.001 else 0.01}$)'),
        (s6, f'by {span(orange_w)} $F1_W$ and {span(orange_m)} $F1_M$'.replace(f'{f2(min(orange_w))}--{f2(min(orange_w))}', f2(min(orange_w)))),
        (s6, f'(by {f2(all_gaps[0])} $F1_W$ and {f2(all_gaps[1])} $F1_M$ on average'),
        (s6, f'it is {span(pw)} $F1_W$ and {span(pm)} $F1_M$ above'),
        (s6, f'leads the best baseline by {span(avg_gaps)} F1'),
        (s6, f'mean over {["zero", "one", "two", "three", "four", "five"][len(CITIES)]} houses and five seeds'),
    ]


# ---------------------------------------------------------------- checks

def check(compile_=True):
    problems = []
    if compile_:
        r = subprocess.run(['latexmk', '-g', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], cwd=PAPER,
                           capture_output=True, text=True)
        if r.returncode:
            problems.append('LaTeX build failed (see discover-v1/main.log)')
        log = (PAPER / 'main.log').read_text(errors='ignore')
        for pat in (r"LaTeX Warning: Reference `[^']+' on page \d+ undefined",
                    r"LaTeX Warning: Citation `[^']+' on page \d+ undefined", r'File `[^\']+\' not found',
                    r"LaTeX Warning: Label `[^']+' multiply defined"):
            problems += sorted(set(re.findall(pat, log)))
    for t in PER_HOUSE:
        text = (PAPER / 'tables' / f'{t}.tex').read_text()
        miss = [HOUSE_NAMES[c] for c in CITIES if HOUSE_NAMES[c] not in text]
        if miss:
            problems.append(f'tables/{t}.tex lacks {miss}')
    for f, expected in claims():
        if expected not in (PAPER / 'sections' / f).read_text():
            problems.append(f'{f}: expected text not found: {expected}')
    tbd = []
    for f in sorted((PAPER / 'sections').glob('*.tex')):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for m in re.finditer(r'\\tbd\{([^}]*)\}', line):
                tbd.append(f'{f.name}:{i}: {m.group(1)}')
                if f.name != '9_appendix.tex' and 'rater background' not in m.group(1):
                    problems.append(f'placeholder outside the appendix: {f.name}:{i}: {m.group(1)}')
    print(f'{len(tbd)} placeholders:' + ''.join(f'\n  {t}' for t in tbd))
    print('OK' if not problems else f'{len(problems)} problems:' + ''.join(f'\n  {p}' for p in problems))
    return not problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='check only, no sync')
    ap.add_argument('--no-compile', action='store_true')
    args = ap.parse_args()
    if not args.check:
        sync()
    raise SystemExit(0 if check(not args.no_compile) else 1)


if __name__ == '__main__':
    main()
