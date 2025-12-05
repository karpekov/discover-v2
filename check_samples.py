#!/usr/bin/env python3
"""Quick script to compare sample counts across datasets."""

import json
import os

datasets = ['aruba', 'milan', 'cairo']
configs = ['FD_60', 'FD_60_p']

print('Sample Count Comparison:')
print('=' * 80)

for dataset in datasets:
    print(f'\n{dataset.upper()}:')
    totals = {}
    for config in configs:
        path = f'data/processed/casas/{dataset}/{config}'
        if os.path.exists(path):
            train_path = f'{path}/train.json'
            val_path = f'{path}/val.json'
            test_path = f'{path}/test.json'

            train_count = val_count = test_count = 0

            if os.path.exists(train_path):
                with open(train_path) as f:
                    train_count = len(json.load(f))
            if os.path.exists(val_path):
                with open(val_path) as f:
                    val_count = len(json.load(f))
            if os.path.exists(test_path):
                with open(test_path) as f:
                    test_count = len(json.load(f))

            total = train_count + val_count + test_count
            totals[config] = total
            print(f'  {config:12s}: Train={train_count:6,d}, Val={val_count:5,d}, Test={test_count:6,d}, Total={total:7,d}')
        else:
            print(f'  {config:12s}: Not found')

    # Calculate ratio
    if 'FD_60' in totals and 'FD_60_p' in totals and totals['FD_60_p'] > 0:
        ratio = totals['FD_60'] / totals['FD_60_p']
        reduction = (1 - totals['FD_60_p']/totals['FD_60']) * 100
        print(f'  → FD_60 is {ratio:.2f}x larger than FD_60_p ({reduction:.1f}% reduction)')

print('\n' + '=' * 80)

