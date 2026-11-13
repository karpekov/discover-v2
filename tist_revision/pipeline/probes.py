"""
Linear probes on frozen representations, at the same label budgets as the clustering protocol.

Logistic regression (standardised features) trained on n randomly drawn labelled pool windows (continuous
FL_20 train, last-event labels incl. No_Activity) and scored on FL_20_p test, at both label levels and with the
same variants as `label_levels.py` ('all', 'masked'; l2 also 'excl_other'). Budgets: 100 (= K 20 × m 5),
250 (= K 50 × m 5) and 'full' (up to 20k pool windows; the probe upper bound). Random draws, not per-class, so the probe gets no label
information the clustering protocol does not have. Small budgets average 3 draws; 'full' uses one.
No_Activity is a training class but is never predicted on test (`predict_activity`). Budgets count activity labels:
windows are drawn until 100 / 250 activity windows are collected (`baselines.data.budget_indices`).

Usage:
    python -m tist_revision.pipeline.probes
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .mapping_eval import EXCLUDED_LABELS, RESULTS_DIR, _scores, load_protocol, predict_activity, scored_classes
from ..baselines.data import budget_indices
from .rare_classes import representations

OUT = RESULTS_DIR / 'probes'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cities', nargs='+', default=['milan', 'aruba', 'cairo'])
    ap.add_argument('--representations', nargs='+',
                    default=['hist_sensor', 'hist_sensor_tod', 'mlm', 'mlm_time', 'mlm+tod', 'mlm_time+tod'])
    ap.add_argument('--budgets', nargs='+', default=['100', '250', 'full'])
    ap.add_argument('--max_full', type=int, default=20000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for city in args.cities:
        fit, pool2, test2 = load_protocol(city, level='l2')
        _, pool1, test1 = load_protocol(city, level='l1')
        for rep, (_, X_pool, X_test) in representations(city, fit, pool2, test2, args.representations):
            for level, pool, test in (('l2', pool2, test2), ('l1', pool1, test1)):
                y_pool, y_test = pool['label'].to_numpy(), test['label'].to_numpy()
                variants = {'all': None, 'masked': scored_classes(y_pool, y_test)}
                if level == 'l2':
                    variants['excl_other'] = sorted(set(y_test) - set(EXCLUDED_LABELS))
                for budget in args.budgets:
                    n = min(args.max_full, len(y_pool)) if budget == 'full' else int(budget)
                    scores = {v: [] for v in variants}
                    for seed in ([0] if budget == 'full' else [0, 1, 2]):
                        if budget == 'full':
                            idx = np.random.default_rng(seed).choice(len(y_pool), size=n, replace=False)
                        else:
                            idx = np.concatenate(budget_indices(y_pool, budget, seed))
                        scaler = StandardScaler().fit(X_pool[idx])
                        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X_pool[idx]), y_pool[idx])
                        pred = predict_activity(clf.predict_proba(scaler.transform(X_test)), clf.classes_)
                        for v, classes in variants.items():
                            keep = ~np.isin(y_test, EXCLUDED_LABELS) if v == 'excl_other' else np.ones(len(y_test), bool)
                            scores[v].append(_scores(y_test[keep], pred[keep], classes))
                    for v, s in scores.items():
                        rows.append({'city': city, 'representation': rep, 'level': level, 'variant': v,
                                     'budget': budget, 'n_labels': n, 'n_draws': len(s)}
                                    | pd.DataFrame(s).mean().to_dict())
                print(f'[{city}] {rep} {level}', flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'results.csv', index=False)
    print(f'Saved {OUT / "results.csv"}')


if __name__ == '__main__':
    main()
