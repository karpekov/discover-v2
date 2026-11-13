"""
One readable table per variant: every SCAN run (results/stage3_scan/runs/*) next to the k-means
baselines on histograms and frozen encoders (results/stage1_5_*), weighted / macro F1 at m=5 and m=all.
Writes results/summary_{variant}.csv and prints markdown.

Usage:
    python -m tist_revision.pipeline.summarize
"""
import argparse
from pathlib import Path

import pandas as pd

from .mapping_eval import RESULTS_DIR


def collect() -> pd.DataFrame:
    rows = []
    for d in sorted(RESULTS_DIR.glob('stage1_5_*')):
        path = d / 'clustering.csv'
        if path.exists():
            df = pd.read_csv(path)
            df['m'] = df['m'].astype(str)
            df['method'] = df['representation'] + ' · ' + df['method']
            rows.append(df[['city', 'method', 'variant', 'm', 'f1_weighted', 'f1_macro']])
    for path in sorted((RESULTS_DIR / 'stage3_scan' / 'runs').glob('*/scan_head.csv')):
        df = pd.read_csv(path)
        df['m'] = df['m'].astype(str)
        kind = df['method'].str.upper() if 'method' in df else 'SCAN'
        df['method'] = kind + ' ' + df['encoder'].str.replace(r'^\w+?_fl20_seq_discover_v1_', '', regex=True) + ' · ' + df['run']
        rows.append(df[['city', 'method', 'variant', 'm', 'f1_weighted', 'f1_macro']])
    return pd.concat(rows).drop_duplicates(['city', 'method', 'variant', 'm'], keep='last')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--m_values', nargs='+', default=['5', 'all'])
    args = ap.parse_args()
    df = collect()
    df = df[df['m'].isin(args.m_values)]
    for variant in ('all', 'excl_other'):
        d = df[df.variant == variant].copy()
        d['cell'] = d['f1_weighted'].map('{:.2f}'.format) + ' / ' + d['f1_macro'].map('{:.2f}'.format)
        t = d.pivot_table(index='method', columns=['city', 'm'], values='cell', aggfunc='first')
        t = t.reindex(columns=[(c, m) for c in ('milan', 'aruba', 'cairo') for m in args.m_values if (c, m) in t.columns])
        t.columns = [f'{c} m={m}' for c, m in t.columns]
        t.to_csv(RESULTS_DIR / f'summary_{variant}.csv')
        print(f'\n### variant={variant} (weighted / macro F1)\n')
        print(t.fillna('').to_markdown())


if __name__ == '__main__':
    main()
