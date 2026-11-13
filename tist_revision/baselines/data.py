"""
Window data for the supervised baselines (TDOST, DeepCASAS, TreeCNN).

Protocols:
  - 'final':  train / val = continuous FL_20 train / val (last-event label, No_Activity kept),
              test = pre-segmented FL_20_p test. Same windows as the clustering protocol.
  - 'preseg': train / val / test = pre-segmented FL_20_p train / val / test. Only used to check the ports
              against published TDOST / DeepCASAS numbers (which use pre-segmented data).
Label budgets ('final' only): random train windows are drawn until `n` windows with an activity label (not
No_Activity) are collected; the No_Activity windows drawn on the way are kept. The drawn set is split 80/20 into
train / val, so the labels used never exceed the budget's draw. 'all' uses every train window and FL_20 val.
"""
import json
import pickle
from pathlib import Path

import numpy as np

from ..pipeline.labels import tdost_label
from ..pipeline.mapping_eval import CACHE_DIR, DATA_ROOT, NO_ACTIVITY, l1_label

SPLITS = {'final': {'train': ('FL_20', 'train'), 'val': ('FL_20', 'val'), 'test': ('FL_20_p', 'test')},
          'preseg': {'train': ('FL_20_p', 'train'), 'val': ('FL_20_p', 'val'), 'test': ('FL_20_p', 'test')}}


def load_split(city, window, split):
    """Dict of per-window lists: sensors, states, times (ISO strings), label_raw (last event's activity), day."""
    cache = CACHE_DIR / f'baseline_{city}_{window}_{split}.pkl'
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    with open(DATA_ROOT / city / window / f'{split}.json') as f:
        samples = json.load(f)['samples']
    out = {'sensors': [], 'states': [], 'times': [], 'label_raw': [], 'day': []}
    for s in samples:
        seq = s['sensor_sequence']
        out['sensors'].append([e['sensor_id'] for e in seq])
        out['states'].append([e['event_type'] for e in seq])
        out['times'].append([e['timestamp'] for e in seq])
        out['label_raw'].append(seq[-1]['activity_l1'])
        out['day'].append(s['metadata']['start_time'][:10])
    out['label_raw'] = np.array(out['label_raw'])
    out['day'] = np.array(out['day'])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


def labels(raw, level):
    if level == 'l1':
        return np.array([l1_label(r) for r in raw])
    if level == 'l2':
        return np.array([tdost_label(r) for r in raw])
    raise ValueError(level)


def load(city, protocol='final'):
    return {k: load_split(city, w, s) for k, (w, s) in SPLITS[protocol].items()}


def budget_indices(y_train, budget, seed):
    """(train_idx, val_idx) into the train split for a label budget ('all' -> all train windows, val split = None)."""
    if budget == 'all':
        return np.arange(len(y_train)), None
    n = int(budget)
    order = np.random.default_rng(seed).permutation(len(y_train))
    activity = np.cumsum(y_train[order] != NO_ACTIVITY)
    drawn = order[:int(np.searchsorted(activity, n)) + 1]
    is_val = np.arange(len(drawn)) % 5 == 4
    return drawn[~is_val], drawn[is_val]
