"""
Kyoto7 (CASAS twor.2009, two residents) trial: the Section 6 models under several ways of handling the residents.

54% of Kyoto7's motion / door events fall inside two or more annotated activities, so "which activity does a window
show" depends on how the two residents are handled. Every variant is a dataset of its own (`data/processed/casas/
<name>/{FL_20,FL_20_p}`), built from `data/raw/casas/twor.2009/data_processed.csv` with the same 70/10/20 day
split (seed 42) and run through the unchanged Section 6 pipeline:

  kyotomerged  One chronological stream, each event once (the Cairo treatment). An event's label is the most
               recently begun activity that is still running. No resident information is used.
  kyotoflat    Streams separated with the annotations (`src/data/multiresident_flatten.py`): an event is copied
               into the stream of every activity running at that time; streams R1, R2 and `both` (activities
               without a resident id) are windowed separately and pooled.
  kyotoresa    One chronological stream, each event once; labels follow resident R1 only: R1's own activity,
  kyotoresb    else a running activity without a resident id, else No_Activity. `kyotoresb`: same for R2.

Second round (2026-11-20), same model, other ways of labeling the concurrent events:

  kyotoaff     Merged stream; an event inside several activities is labeled with the one that most often fires
               its sensor: argmax over the running activities of P(sensor | activity), estimated from the events
               that fall inside exactly one activity. Same events and windows as `kyotomerged`, whose encoders
               are reused.
  kyotojoint   Merged stream; the label is the household state: all activities running at the event, resident
               ids removed and sorted (`Sleep+Work`). Encoders of `kyotomerged` reused.
  kyotoattr    Resident streams without copies: each event goes once, into the stream (R1 / R2 / both) of the
               activity chosen as in `kyotoaff`.

Steps:
    python -m tist_revision.pipeline.kyoto_trial prepare    # streams, sampling, encoder configs, skynet tasks
    bash tist_revision/skynet/launch_final.sh kyoto_tdost_text && ... kyoto_encoder_tf kyoto_baselines  (skynet)
    python -m tist_revision.pipeline.kyoto_trial report     # scores, tables, figures -> results/temp/kyoto7_trial
"""
import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import tdost_label
from .mapping_eval import DATA_ROOT, PROJECT_DIR, TEMP_DIR

OUT = TEMP_DIR / 'kyoto7_trial'
REPO = PROJECT_DIR.parent
RAW_DIR = Path('data/raw/casas/twor.2009')
STREAM_DIR = 'split_variants'  # under RAW_DIR
VARIANTS = {  # dataset name -> (short title, windows are built per resident stream)
    'kyotomerged': ('Merged stream', False),
    'kyotoflat': ('Resident streams', True),
    'kyotoresa': ('R1 labels', False),
    'kyotoresb': ('R2 labels', False),
    'kyotoaff': ('Merged, sensor-attributed labels', False),
    'kyotojoint': ('Merged, household-state labels', False),
    'kyotoattr': ('Resident streams, no copies', True),
}
ROUNDS = {'': ['kyotomerged', 'kyotoflat', 'kyotoresa', 'kyotoresb'], '2': ['kyotoaff', 'kyotojoint', 'kyotoattr']}
STREAMS = {}  # event CSV of a dataset, if not RAW_DIR / STREAM_DIR / <name>.csv (set by orange_trial)
HOUSE = 'Kyoto7'  # name in figure titles
HEADER = None  # report title and intro lines, if not the Kyoto7 ones (set by orange_trial)
ENCODER_OF = {'kyotoaff': 'kyotomerged', 'kyotojoint': 'kyotomerged'}  # same events and FL_20 windows: reuse encoders
SEEDS = [1, 2, 3, 4, 5]
NO_ACTIVITY = 'no_activity'  # raw spelling in the processed CASAS data


# --------------------------------------------------------------------------------------------------------------
# prepare: event streams
# --------------------------------------------------------------------------------------------------------------

def running_activities(activity, marker):
    """Per event, the activities running at that time in the order they began (an event carrying a `begin` or
    `end` marker belongs to that activity). Same sets as `activity_list` in `data_processed.csv`, which is sorted
    by name."""
    running, out = [], []
    for a, m in zip(activity, marker):
        if m == 'begin' and a not in running:
            running.append(a)
        out.append(tuple(running))
        if m == 'end' and a in running:
            running.remove(a)
    return out


def own(acts, resident):
    return [a for a in acts if a.startswith(f'R{resident}_')]


def shared(acts):
    return [a for a in acts if not re.match(r'R\d+_', a)]


def merged_label(acts):
    return acts[-1] if acts else NO_ACTIVITY


def resident_label(acts, resident):
    mine = own(acts, resident) or shared(acts)
    return mine[-1] if mine else NO_ACTIVITY


def sensor_affinity(df, acts):
    """P(sensor | activity) from the motion / door events that fall inside exactly one activity (all days; this
    defines labels, it is not a model). Add-one smoothing over the sensors."""
    ev = df[~df['is_numeric'] & (acts.map(len) == 1)]
    counts = pd.crosstab(ev['sensor'], acts[ev.index].str[0])
    counts = counts.reindex(sorted(df.loc[~df['is_numeric'], 'sensor'].unique()), fill_value=0) + 1
    return (counts / counts.sum()).to_dict()  # activity -> sensor -> probability


def attributed_label(acts, sensor, affinity):
    """Of the running activities (in the order they began), the one most likely to fire `sensor`; ties and
    activities never seen alone go to the most recently begun one."""
    if not acts:
        return NO_ACTIVITY
    return max(reversed(acts), key=lambda a: affinity.get(a, {}).get(sensor, 0.0))


def joint_label(acts):
    from .mapping_eval import l1_label
    return '+'.join(sorted(l1_label(a) for a in acts)) if acts else NO_ACTIVITY


def build_streams():
    """Write one event CSV per variant (`_load_flattened_data` format: label in `activity_flattened`)."""
    sys.path.insert(0, str(REPO / 'src'))  # as in sample_data.py
    from data.multiresident_flatten import flatten_processed

    df = pd.read_csv(RAW_DIR / 'data_processed.csv', low_memory=False)
    acts = running_activities(df['activity'], df['marker'])
    listed = df['activity_list'].map(ast.literal_eval)
    assert all(sorted(a) == l for a, l in zip(acts, listed)), 'running activities differ from activity_list'
    rooms = json.loads((REPO / 'metadata' / 'casas_metadata.json').read_text())['kyoto']['sensor_location']
    df['sensor_location'] = df['sensor'].map(rooms)
    df['running'] = acts
    # `activity_full` travels into every event of the processed windows: all activities running at that event.
    df['activity_full'] = ['|'.join(a) if a else np.nan for a in acts]
    assert df.loc[~df['is_numeric'], 'sensor_location'].ne('UNK').all()

    out_dir = RAW_DIR / STREAM_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    streams = {
        'kyotomerged': df.assign(activity_flattened=[merged_label(a) for a in acts]),
        'kyotoresa': df.assign(activity_flattened=[resident_label(a, 1) for a in acts]),
        'kyotoresb': df.assign(activity_flattened=[resident_label(a, 2) for a in acts]),
        'kyotoflat': flatten_processed(df.drop(columns='running')),
    }
    affinity = sensor_affinity(df, pd.Series(acts, index=df.index))
    attributed = [attributed_label(a, s, affinity) for a, s in zip(acts, df['sensor'])]
    streams['kyotoaff'] = df.assign(activity_flattened=attributed)
    streams['kyotojoint'] = df.assign(activity_flattened=[joint_label(a) for a in acts])
    streams['kyotoattr'] = df.assign(  # as `flatten_processed`: label without resident id, stream in resident_info
        activity_flattened=[re.sub(r'^R\d+_', '', a) for a in attributed],
        resident_info=[m.group(0)[:-1] if (m := re.match(r'R\d+_', a)) else 'both' for a in attributed])
    multi = pd.Series(acts, index=df.index).map(len).ge(2) & ~df['is_numeric']
    changed = multi & (pd.Series(attributed, index=df.index) != streams['kyotomerged']['activity_flattened'])
    print(f'sensor-attributed labels: {changed.sum()} of {multi.sum()} concurrent events ({changed.sum() / multi.sum():.1%}) '
          f'differ from the most-recently-begun label', flush=True)
    for name, s in streams.items():
        s = s.drop(columns='running', errors='ignore').copy()
        s['first_activity_l2'] = s['activity_flattened'].map(tdost_label)
        if (out_dir / f'{name}.csv').exists() and name in ROUNDS['']:
            continue  # first-round streams are in use; never rewrite them
        s.to_csv(out_dir / f'{name}.csv', index=False)
        ev = s[~s['is_numeric']]
        print(f'[{name}] {len(ev)} events; labels: {ev["activity_flattened"].value_counts().to_dict()}', flush=True)


# --------------------------------------------------------------------------------------------------------------
# prepare: windows, encoder configs, skynet tasks
# --------------------------------------------------------------------------------------------------------------

SAMPLING_YAML = """# Kyoto7 (twor.2009) trial, variant `{name}` ({title}): fixed-length windows of 20 events{preseg_note}.
# Generated by tist_revision/pipeline/kyoto_trial.py (prepare); see its docstring for the variants.

dataset_name: "kyoto"
raw_data_path: "{raw}"
output_dir: "data/processed/casas/{name}/{window}"

strategy: "fixed_length"

window_sizes: [20]
min_events_per_window: 8

# Train/val/test split by days (70/10/20), drawn once for the whole house
split_strategy: "random"
train_ratio: 0.8
random_seed: 42

overlap_factor: 0.5

use_presegmentation: {preseg}
presegment_label_level: "l1"
min_segment_events: 8
exclude_no_activity: {preseg}

filter_numeric_sensors: true
max_gap_minutes: 30

preserve_full_metadata: true
include_spatial_info: true
include_sensor_types: true

max_lines: null
max_windows: null

# Event stream of this variant (label in `activity_flattened`)
multiresident_flatten: true
multiresident_stream_file: "{stream}"
# true: windows are built inside each resident stream (R1 / R2 / both) and pooled
multiresident_split_by_resident: {by_resident}
"""


def sampling_config(name, window):
    title, by_resident = VARIANTS[name]
    preseg = window.endswith('_p')
    path = REPO / 'configs' / 'sampling' / f'{name}_{window}.yaml'
    path.write_text(SAMPLING_YAML.format(
        name=name, title=title, window=window, raw=RAW_DIR, stream=f'{STREAM_DIR}/{name}.csv',
        preseg=str(preseg).lower(), preseg_note=', pre-segmented by activity' if preseg else '',
        by_resident=str(by_resident).lower()))
    return path


def unused_text_embeddings(name):
    """`train.py` requires caption embeddings even when the contrastive loss has weight 0 (MLM only). The other
    houses reuse their old CLIP caption files; the trial writes unit-norm noise instead of generating captions.
    With `clip_weight: 0.0` these vectors receive no gradient and do not enter the loss."""
    rng = np.random.default_rng(0)
    for split in ('train', 'val'):
        n = len(json.loads((DATA_ROOT / name / 'FL_20' / f'{split}.json').read_text())['samples'])
        emb = rng.standard_normal((n, 512)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        np.savez_compressed(DATA_ROOT / name / 'FL_20' / f'{split}_embeddings_unused.npz', embeddings=emb,
                            encoder_type=np.array(['none']), model_name=np.array(['unused (MLM only)']))


def encoder_config(name):
    """Section 6 encoder config (no coordinates, MLM + time features), copied from Cairo's.

    Training stops at min(max_epochs, 10,000 steps). Milan / Aruba / Cairo all hit the step limit (85 / 23 / 56
    epochs); Kyoto7 has 4-12x fewer windows and would stop after 100 epochs = 3,700-5,800 steps, a third into
    the learning-rate schedule. `max_epochs` is raised so that Kyoto7 gets the same 10,000 steps."""
    src = (PROJECT_DIR / 'configs' / 'alignment' / 'cairo_fl20_seq_nocoord_mlm_time.yaml').read_text()
    cfg = src.replace('cairo', name).replace('# Cairo FL_20', f'# Kyoto7 trial ({VARIANTS[name][0]}) FL_20')
    cfg = cfg.replace('_embeddings_baseline_clip.npz', '_embeddings_unused.npz')
    cfg = cfg.replace('  max_epochs: 100\n', '  max_epochs: 400  # the 10,000-step limit ends training, as in the '
                                             'other houses (see kyoto_trial.encoder_config)\n')
    assert 'Cairo' not in cfg and cfg.count(name) == src.count('cairo') and cfg.count('_embeddings_unused.npz') == 2
    assert 'max_epochs: 400' in cfg
    path = PROJECT_DIR / 'configs' / 'alignment' / f'{name}_fl20_seq_nocoord_mlm_time.yaml'
    path.write_text(cfg)
    return path


def task_files():
    """Skynet task arrays, same commands as `encoder_tf.txt` / `baselines.txt` restricted to the main models; one
    set of files per round (`kyoto_*`, `kyoto2_*`). Variants in ENCODER_OF skip training and encode with the
    encoder of the variant that has the same events; they also get the two time-of-day-free reference clusterings."""
    tasks = PROJECT_DIR / 'skynet' / 'tasks'
    fd = 'python -m tist_revision.pipeline.frozen_deep'
    for suffix, names in ROUNDS.items():
        enc, base = [], []
        for name in names:
            for s in SEEDS:
                model = f'trained_models/{ENCODER_OF.get(name, name)}/tf_nocoord_mlm_time_s{s}'
                steps = [] if name in ENCODER_OF else [
                    f'python train.py --config tist_revision/configs/alignment/{name}_fl20_seq_nocoord_mlm_time.yaml '
                    f'--seed {s} --output-dir {model}']
                steps += [f'python -m tist_revision.pipeline.encode_embeddings --city {name} --kind tf --model_dir {model} '
                          f'--name tf_s{s}',
                          f'{fd} --cities {name} --representations tf_s{s}+tod '
                          f'--methods scan --K 20 50 --seed {s} --tod_weight balanced --suffix _todwbal']
                if suffix:
                    steps.append(f'{fd} --cities {name} --representations tf_s{s} --methods kmeans scan --K 20 50 --seed {s}')
                enc.append(' && '.join(steps))
        for method in ('deepcasas', 'tdost'):
            for name in names:
                for s in SEEDS:
                    base.append(f'python -m tist_revision.baselines.run --method {method} --city {name} --seed {s} '
                                f'--protocol final --levels l1')
        (tasks / f'kyoto{suffix}_encoder_tf.txt').write_text('\n'.join(enc) + '\n')
        (tasks / f'kyoto{suffix}_baselines.txt').write_text('\n'.join(base) + '\n')
        (tasks / f'kyoto{suffix}_tdost_text.txt').write_text(
            f'python -m tist_revision.baselines.tdost_text --cities {" ".join(names)}\n')
        print(f'Wrote {tasks}/kyoto{suffix}_*.txt ({len(enc)} encoder, {len(base)} baseline tasks)')


def ablation_task_files(variants):
    """Task arrays for the appendix ablations of the paper houses (`encoder_tf.txt`, `encoder_bert.txt`,
    `baselines.txt`), for `variants`. Reuses the encoders and embeddings of `kyoto_encoder_tf.txt` on skynet."""
    fd = 'python -m tist_revision.pipeline.frozen_deep'
    clu, bert, base = [], [], []
    for name in variants:
        for s in SEEDS:
            clu.append(' && '.join([
                f'{fd} --cities {name} --representations tf_s{s} tf_s{s}+tod --methods kmeans scan dec --K 20 50 --seed {s}',
                f'{fd} --cities {name} --representations hist_sensor_s{s} hist_sensor_tod_s{s} --methods kmeans '
                f'--K 20 50 --seed {s}',
                f'{fd} --cities {name} --representations hist_sensor_tod_s{s} --methods scan --K 20 50 --seed {s}',
                f'{fd} --cities {name} --representations tf_s{s}+tod --methods scan --K 100 --seed {s} '
                f'--tod_weight balanced --suffix _todwbal']))
        model = f'trained_models/{name}/bert_mlm_s1'
        bert.append(
            f'python -m tist_revision.pipeline.bert_mlm train --city {name} --seed 1 --out {model} && '
            f'python -m tist_revision.pipeline.encode_embeddings --city {name} --kind bert --model_dir {model} '
            f'--name bert_s1 && '
            f'{fd} --cities {name} --representations bert_s1 bert_s1+tod --methods kmeans scan dec --K 20 50 --seed 1')
        for method in ('deepcasas', 'tdost'):
            for s in SEEDS:
                base.append(f'python -m tist_revision.baselines.run --method {method} --city {name} --seed {s} '
                            f'--protocol final --levels l2')
    tasks = PROJECT_DIR / 'skynet' / 'tasks'
    for stem, lines in (('kyoto_ablation_clustering', clu), ('kyoto_ablation_bert', bert), ('kyoto_ablation_baselines_l2', base)):
        (tasks / f'{stem}.txt').write_text('\n'.join(lines) + '\n')
        print(f'Wrote {tasks}/{stem}.txt ({len(lines)} tasks)')


def prepare(args):
    if 'streams' in args.steps:
        build_streams()
    for name in args.variants:
        for window in ('FL_20', 'FL_20_p'):
            cfg = sampling_config(name, window)
            if 'windows' in args.steps:
                subprocess.run([sys.executable, 'sample_data.py', '--config', str(cfg)], check=True, cwd=REPO)
        if 'windows' in args.steps:
            unused_text_embeddings(name)
        encoder_config(name)
    task_files()


# --------------------------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------------------------

METHODS = [  # name, run family, filter, color
    ('DeepCASAS', 'baseline', dict(method='deepcasas'), '#ea580c'),
    ('TDOST', 'baseline', dict(method='tdost'), '#be185d'),
    ('DISCOVER', 'clustering', dict(method='scan', rep='tf+tod'), '#6d28d9'),
    ('Linear probe (DISCOVER features)', 'probe', dict(rep='tf+tod'), '#94a3b8'),
]
MAIN = ('DeepCASAS', 'TDOST', 'DISCOVER')  # rows of the Section 6 table; the probe is the label-efficiency reference
METRICS = (('f1_weighted', 'F1 weighted'), ('f1_macro', 'F1 macro'))
PAPER_HOUSES = ['milan', 'aruba', 'cairo']
TOD_SETUPS = [  # name, clustering method, representation, color
    ('DISCOVER (SCAN, TF+ToD)', 'scan', 'tf+tod', '#6d28d9'), ('SCAN, TF', 'scan', 'tf', '#a78bfa'),
    ('k-means, TF', 'kmeans', 'tf', '#0d9488'), ('DEC, TF', 'dec', 'tf', '#5eead4'),
    ('k-means, sensor hist.', 'kmeans', 'hist_sensor', '#94a3b8'),
]


def pick(runs, city, name, clu, budget, level='l1', variant='masked'):
    """Runs (one per seed) of a method in one budget column of the matched-budget table."""
    _, family, filt, _ = next(m for m in METHODS if m[0] == name)
    q = runs[(runs.family == family) & (runs.city == city) & (runs.level == level) & (runs.variant == variant)]
    for k, v in (filt | (clu if family == 'clustering' else {'budget': budget})).items():
        q = q[q[k] == v]
    return q


def score_runs(rescore):
    from .report_final import score_baselines, score_clustering, score_probes
    path = OUT / 'data' / 'runs.csv'
    if path.exists() and not rescore:
        return pd.read_csv(path, dtype={'budget': str, 'rep': str}, keep_default_na=False, na_values=[''])
    cities = list(VARIANTS)
    runs = pd.concat([score_clustering(cities=cities), score_baselines(cities=cities), score_probes(cities=cities)],
                     ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(path, index=False)
    return pd.read_csv(path, dtype={'budget': str, 'rep': str}, keep_default_na=False, na_values=[''])


def window_extras(city, window, split):
    """Per window: L1 names of all activities running at the last event, resident stream, and the event content."""
    from .mapping_eval import l1_label
    samples = json.loads((DATA_ROOT / city / window / f'{split}.json').read_text())['samples']
    rows = []
    for s in samples:
        seq = s['sensor_sequence']
        running = sorted({l1_label(a) for a in str(seq[-1].get('activity_full', '')).split('|') if a})
        rows.append({'running': running, 'resident': s['metadata'].get('resident_info') or 'all',
                     'events': tuple((e['timestamp'], e['sensor_id'], e['event_type']) for e in seq)})
    return pd.DataFrame(rows)


def predictions(city, name, clu, budget, seed, pool_labels):
    """Test predictions (L1) of one main method for one seed, or None if the run is missing."""
    from .mapping_eval import ARTIFACTS_DIR, BASELINES_DIR
    from .report_final import discover_preds
    if name == 'DISCOVER':
        if not (ARTIFACTS_DIR / 'assignments' / f'{city}__tf_s{seed}+tod__scan_k{clu["K"]}_todwbal.npz').exists():
            return None
        return discover_preds(city, seed, clu['K'], 'all' if clu['budget'] == 'mall' else int(clu['budget'][1:]),
                              pool_labels)
    path = BASELINES_DIR / 'final' / name.lower() / f'{city}_l1_{budget}_s{seed}.npz'
    return np.load(path, allow_pickle=True)['pred'].astype(str) if path.exists() else None


def dataset_table():
    from .mapping_eval import NO_ACTIVITY as NA, load_protocol, scored_classes
    rows = []
    for city, (title, _) in VARIANTS.items():
        _, pool, test = load_protocol(city, level='l1')
        ex_pool, ex_test = window_extras(city, 'FL_20', 'train'), window_extras(city, 'FL_20_p', 'test')
        events = pd.read_csv(STREAMS.get(city, RAW_DIR / STREAM_DIR / f'{city}.csv'), usecols=['is_numeric'])
        # Windows whose exact event sequence also occurs under another label (resident streams hold copies).
        labels_of = test.assign(events=ex_test['events']).groupby('events')['label'].agg(list)
        ceiling = sum(max(l.count(x) for x in set(l)) for l in labels_of) / len(test)
        rows.append({
            'variant': title, 'dataset': city, 'events': int((~events['is_numeric']).sum()),
            'pool windows (FL_20 train)': len(pool), 'test windows (FL_20_p test)': len(test),
            'pool No_Activity': (pool['label'] == NA).mean(),
            'pool windows, >=2 activities at last event': (ex_pool['running'].map(len) >= 2).mean(),
            'test windows, >=2 activities at last event': (ex_test['running'].map(len) >= 2).mean(),
            'test windows with a twin under another label': float(np.mean([len(set(labels_of[e])) > 1 for e in ex_test['events']])),
            'accuracy ceiling from twins': ceiling,
            'classes in test': test['label'].nunique(),
            'classes scored (masked)': len(scored_classes(pool['label'], test['label'])),
        })
    return pd.DataFrame(rows)


def significance(runs, n_boot=2000, seed=0):
    """Paired day-block bootstrap of DISCOVER minus each supervised baseline, as `report_final.significance`
    (same resampled test days for both methods, mean over seeds)."""
    from .mapping_eval import load_protocol, scored_classes
    from .report_final import MATCHED, boot_f1
    rng = np.random.default_rng(seed)
    rows = []
    for city in VARIANTS:
        _, pool, test = load_protocol(city, level='l1')
        yp, yt = pool['label'].to_numpy(), test['label'].to_numpy()
        days, day_idx = np.unique(test['day'].to_numpy(), return_inverse=True)
        W = np.stack([np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days)) for _ in range(n_boot)])
        masked = scored_classes(yp, yt)
        for col, clu, budget in MATCHED:
            preds = {n: [predictions(city, n, clu, budget, s, yp) for s in SEEDS] for n in MAIN}
            seeds = [i for i in range(len(SEEDS)) if all(preds[n][i] is not None for n in MAIN)]
            if not seeds:
                continue
            boot = {n: boot_f1(yt, [preds[n][i] for i in seeds], day_idx, W, masked) for n in MAIN}
            for k, (metric, _) in enumerate(METRICS):
                means = {n: pick(runs, city, n, clu, budget)[metric].mean() for n in MAIN[:2]}
                best = max(means, key=means.get)
                d = boot['DISCOVER'][k] - boot[best][k]
                rows.append({'city': city, 'budget': col, 'metric': metric, 'best_baseline': best, 'n_seeds': len(seeds),
                             'diff_boot_mean': d.mean(), 'ci_low': np.quantile(d, 0.025), 'ci_high': np.quantile(d, 0.975),
                             'p': min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))})
    return pd.DataFrame(rows)


def detail_tables():
    """Per-class F1, lenient scores (merged stream) and per-resident-stream scores (resident streams) of the main
    methods, mean over seeds; pooled confusion matrices of DISCOVER at ~250 labels."""
    from sklearn.metrics import confusion_matrix, f1_score
    from .mapping_eval import load_protocol, scored_classes
    from .report_final import MATCHED
    per_class, lenient, groups, confusions = [], [], [], {}
    for city in VARIANTS:
        _, pool, test = load_protocol(city, level='l1')
        yp, yt = pool['label'].to_numpy(), test['label'].to_numpy()
        classes = sorted(set(yt))
        masked = scored_classes(yp, yt)
        ex = window_extras(city, 'FL_20_p', 'test')
        for col, clu, budget in MATCHED:
            for name in MAIN:
                preds = [p for p in (predictions(city, name, clu, budget, s, yp) for s in SEEDS) if p is not None]
                if not preds:
                    continue
                f1 = np.mean([f1_score(yt, p, labels=classes, average=None, zero_division=0) for p in preds], 0)
                per_class += [{'city': city, 'budget': col, 'method': name, 'class': c, 'f1': v,
                               'test_windows': int((yt == c).sum()), 'scored': c in masked}
                              for c, v in zip(classes, f1)]
                if name == 'DISCOVER' and budget == '250':
                    confusions[city] = (classes, sum(confusion_matrix(yt, p, labels=classes) for p in preds))
                # Lenient: a prediction naming any activity running at the window's last event counts as correct.
                sc = {'strict': [], 'lenient': []}
                for p in preds:
                    hit = np.array([q in r for q, r in zip(p, ex['running'])])
                    for kind, y in (('strict', yt), ('lenient', np.where(hit, p, yt))):
                        sc[kind].append((f1_score(y, p, average='weighted', zero_division=0),
                                         f1_score(y, p, labels=masked, average='macro', zero_division=0),
                                         (y == p).mean()))
                for kind, v in sc.items():
                    w, m, a = np.mean(v, 0)
                    lenient.append({'city': city, 'budget': col, 'method': name, 'scoring': kind,
                                    'f1_weighted': w, 'f1_macro': m, 'accuracy': a})
                for g in sorted(ex['resident'].unique()):
                    keep = (ex['resident'] == g).to_numpy()
                    groups.append({'city': city, 'budget': col, 'method': name, 'stream': g, 'test_windows': int(keep.sum()),
                                   'accuracy': np.mean([(p[keep] == yt[keep]).mean() for p in preds]),
                                   'f1_weighted': np.mean([f1_score(yt[keep], p[keep], average='weighted', zero_division=0)
                                                           for p in preds])})
    return pd.DataFrame(per_class), pd.DataFrame(lenient), pd.DataFrame(groups), confusions


def figures(runs, per_class, confusions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .report_final import MATCHED
    figdir = OUT / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

    # 1. Label efficiency per variant: F1 against the labeling budget.
    fig, axes = plt.subplots(2, len(VARIANTS), figsize=(max(3.1 * len(VARIANTS), 6), 5.4), sharex=True, sharey='row',
                             squeeze=False)
    for j, (city, (title, _)) in enumerate(VARIANTS.items()):
        for i, (metric, mname) in enumerate(METRICS):
            ax = axes[i, j]
            for name, _, _, color in METHODS:
                q = [pick(runs, city, name, clu, budget)[metric] for _, clu, budget in MATCHED]
                if any(len(v) == 0 for v in q):
                    continue
                mu, sd = np.array([v.mean() for v in q]), np.array([v.std() if len(v) > 1 else 0.0 for v in q])
                probe = name.startswith('Linear')
                ax.fill_between(range(3), mu - sd, mu + sd, color=color, alpha=0.13, lw=0)
                ax.plot(range(3), mu, marker='o', ms=5 if name == 'DISCOVER' else 4, color=color, ls='--' if probe else '-',
                        lw=2.4 if name == 'DISCOVER' else 1.6, label=name, zorder=3 if name == 'DISCOVER' else 2)
            ax.set_xticks(range(3), ['≈100', '≈250', 'all'])
            ax.grid(axis='y', alpha=0.3)
            if i == 0:
                ax.set_title(title, fontsize=10)
            else:
                ax.set_xlabel('Labeled windows')
            if j == 0:
                ax.set_ylabel(mname)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'{HOUSE}: DISCOVER vs. supervised baselines (L1, mean ± std over seeds)', fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(figdir / 'label_efficiency.png', dpi=170)
    plt.close(fig)

    # 2. Per-class F1 at ~250 labels.
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(max(3.3 * len(VARIANTS), 5), 3.4 if HOUSE == 'Kyoto7' else 6),
                             sharey=False, squeeze=False)
    axes = axes[0]
    for ax, (city, (title, _)) in zip(axes, VARIANTS.items()):
        q = per_class[(per_class.city == city) & (per_class.budget == '≈250 labels')]
        if q.empty:
            ax.axis('off')
            continue
        t = q.pivot(index='class', columns='method', values='f1').reindex(columns=[m for m in MAIN if m in set(q.method)])
        n = q.drop_duplicates('class').set_index('class')['test_windows']
        t = t.loc[n.sort_values(ascending=False).index]
        im = ax.imshow(t.to_numpy(), vmin=0, vmax=1, cmap='Purples', aspect='auto')
        ax.set_xticks(range(t.shape[1]), t.columns, rotation=20, ha='right')
        ax.set_yticks(range(len(t)), [f'{c} ({n[c]})' for c in t.index])
        for (r, c), v in np.ndenumerate(t.to_numpy()):
            ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=8, color='white' if v > 0.6 else '#1e1b4b')
        ax.set_title(title, fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f'{HOUSE}: per-class F1 at ≈250 labels (test windows in parentheses)', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figdir / 'per_class_f1.png', dpi=170)
    plt.close(fig)

    # 3. DISCOVER confusion matrices at ~250 labels (K = 50, m = 5), pooled over seeds, row-normalized.
    if confusions:
        n = max(len(c) for c, _ in confusions.values())
        size = max(3.6, 0.42 * n)
        fig, axes = plt.subplots(1, len(VARIANTS), figsize=(size * len(VARIANTS) + 1, size + 0.6), squeeze=False)
        axes = axes[0]
        for ax, (city, (title, _)) in zip(axes, VARIANTS.items()):
            if city not in confusions:
                ax.axis('off')
                continue
            classes, cm = confusions[city]
            cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(cmn, vmin=0, vmax=1, cmap='Purples')
            short = [c.replace('Meal_Preparation', 'Meal_Prep').replace('Personal_Hygiene', 'Hygiene') for c in classes]
            ax.set_xticks(range(len(classes)), short, rotation=45, ha='right')
            ax.set_yticks(range(len(classes)), short)
            for (r, c), v in np.ndenumerate(cmn):
                if v >= 0.05:
                    ax.text(c, r, f'{v:.2f}'.lstrip('0'), ha='center', va='center', fontsize=7,
                            color='white' if v > 0.6 else '#1e1b4b')
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Predicted')
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[0].set_ylabel('True')
        fig.suptitle(f'{HOUSE}: DISCOVER confusion at ≈250 labels (rows sum to 1)', fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(figdir / 'confusion_discover.png', dpi=170)
        plt.close(fig)


def md_table(df, fmt='{:.2f}'):
    cols = list(df.columns)
    def f(v):
        return fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ('–' if pd.isna(v) else str(v))
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    lines += ['| ' + ' | '.join(f(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines)


def main_tables(runs, sig):
    """Section 6 layout: one table per metric, rows method within variant, columns label budget."""
    from .report_final import MATCHED, stars
    final = pd.read_csv(PROJECT_DIR / 'results' / 'final' / 'data' / 'summary.csv', dtype={'budget': str})
    final = final[(final.level == 'l1') & (final.variant == 'masked')]
    out, flat = {}, []
    for metric, mname in METRICS:
        rows = []
        for city, (title, _) in VARIANTS.items():
            cells = {n: [pick(runs, city, n, clu, budget)[metric] for _, clu, budget in MATCHED] for n, *_ in METHODS}
            best = [max((cells[n][i].mean() for n in MAIN if len(cells[n][i])), default=np.nan) for i in range(3)]
            for name, *_ in METHODS:
                if all(len(v) == 0 for v in cells[name]):
                    continue
                row = {'Variant': title, 'Method': name if name in MAIN else f'*{name}*',
                       'Seeds': '/'.join(str(len(v)) for v in cells[name])}
                for i, (col, _, _) in enumerate(MATCHED):
                    v = cells[name][i]
                    if len(v) == 0:
                        row[col] = '–'
                        continue
                    txt = f'{v.mean():.2f}'
                    if name in MAIN and abs(v.mean() - best[i]) < 5e-3:
                        txt = f'**{txt}**'
                    row[col] = txt + (f' ± {v.std():.2f}' if len(v) > 1 else '')
                    flat.append({'city': city, 'method': name, 'budget': col, 'metric': metric, 'mean': v.mean(),
                                 'std': v.std() if len(v) > 1 else np.nan, 'n_seeds': len(v)})
                rows.append(row)
            row = {'Variant': title, 'Method': 'Δ DISCOVER vs. best baseline', 'Seeds': ''}
            for col, _, _ in MATCHED:
                r = sig[(sig.city == city) & (sig.budget == col) & (sig.metric == metric)] if len(sig) else sig
                row[col] = (f'{r.iloc[0].diff_boot_mean:+.2f}{stars(r.iloc[0].p)} [{r.iloc[0].ci_low:+.2f}, '
                            f'{r.iloc[0].ci_high:+.2f}] vs. {r.iloc[0].best_baseline}') if len(r) else '–'
            rows.append(row)
        # Paper houses for reference (results/final, 5 seeds).
        for house in PAPER_HOUSES:
            for name in MAIN:
                _, family, filt, _ = next(m for m in METHODS if m[0] == name)
                row = {'Variant': f'{house.capitalize()} (paper)', 'Method': name, 'Seeds': '5'}
                for col, clu, budget in MATCHED:
                    q = final[(final.family == family) & (final.city == house)]
                    for k, v in (filt | (clu if family == 'clustering' else {'budget': budget})).items():
                        q = q[q[k] == v]
                    row[col] = f'{q[metric].iloc[0]:.2f}' if len(q) else '–'
                rows.append(row)
        out[mname] = pd.DataFrame(rows)
    return out, pd.DataFrame(flat)


def clustering_table(runs):
    """Diagnostic: every clustering configuration of the appendix ablations that has been run, weighted / macro F1
    (L1, mean over seeds). K = 100 x 5 labels is about 500 labels and has no matched supervised budget."""
    from .report_final import CLU_ORDER, METHOD_NAMES, REP_NAMES
    x = runs[(runs.family == 'clustering') & (runs.level == 'l1') & (runs.variant == 'masked')]
    cols = [('K = 20, m = 5', 20, 'm5'), ('K = 50, m = 5', 50, 'm5'), ('K = 50, m = all', 50, 'mall'), ('K = 100, m = 5', 100, 'm5')]
    rows = []
    for city, (title, _) in VARIANTS.items():
        for method, r in CLU_ORDER:
            q = x[(x.city == city) & (x.method == method) & (x.rep == r)]
            if q.empty:
                continue
            row = {'Variant': title, 'Clustering': METHOD_NAMES[method], 'Representation': REP_NAMES[r],
                   'Seeds': q.seed.nunique()}
            for col, K, budget in cols:
                v = q[(q.K == K) & (q.budget == budget)]
                row[col] = f'{v.f1_weighted.mean():.2f} / {v.f1_macro.mean():.2f}' if len(v) else '–'
            rows.append(row)
    return pd.DataFrame(rows)


def tod_table(runs):
    """Clustering with and without the time-of-day histogram block next to the better supervised baseline, at the
    matched budgets (L1, mean over seeds), long format. Paper houses from results/final for reference."""
    from .report_final import MATCHED
    final = pd.read_csv(PROJECT_DIR / 'results' / 'final' / 'data' / 'runs.csv', dtype={'budget': str, 'rep': str})
    rows = []
    for src, cities in ((runs, {c: t for c, (t, _) in VARIANTS.items()}), (final, {h: f'{h.capitalize()} (paper)' for h in PAPER_HOUSES})):
        x = src[(src.level == 'l1') & (src.variant == 'masked')]
        for city, title in cities.items():
            for col, clu, budget in MATCHED:
                q = x[(x.family == 'baseline') & (x.city == city) & (x.budget == budget) & x.method.isin(['deepcasas', 'tdost'])]
                g = q.groupby('method')[['f1_weighted', 'f1_macro']].mean()
                if len(g):
                    rows.append({'Variant': title, 'Method': 'Better supervised baseline', 'budget': col,
                                 'f1_weighted': g.f1_weighted.max(), 'f1_macro': g.f1_macro.max()})
                for name, method, r, _ in TOD_SETUPS:
                    v = x[(x.family == 'clustering') & (x.city == city) & (x.method == method) & (x.rep == r)
                          & (x.K == clu['K']) & (x.budget == clu['budget'])]
                    if len(v):
                        rows.append({'Variant': title, 'Method': name, 'budget': col,
                                     'f1_weighted': v.f1_weighted.mean(), 'f1_macro': v.f1_macro.mean()})
    return pd.DataFrame(rows)


def tod_figure(tod):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .report_final import MATCHED
    methods = [('Better supervised baseline', '#ea580c')] + [(n, c) for n, _, _, c in TOD_SETUPS]
    variants = list(dict.fromkeys(tod.Variant))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.6), sharey=True)
    for ax, (col, _, _) in zip(axes, MATCHED[:2]):
        t = tod[tod.budget == col].pivot(index='Variant', columns='Method', values='f1_weighted').reindex(variants)
        width = 0.8 / len(methods)
        for i, (name, color) in enumerate(methods):
            if name in t.columns:
                ax.bar(np.arange(len(variants)) + (i - (len(methods) - 1) / 2) * width, t[name], width, color=color, label=name)
        ax.axvline(len(VARIANTS) - 0.5, color='#94a3b8', lw=0.8, ls=':')
        ax.set_xticks(range(len(variants)), [v.replace(' (paper)', '\n(paper)').replace(' ', '\n', 1) if '(paper)' not in v
                                             else v.replace(' (paper)', '\n(paper)') for v in variants])
        ax.set_title(col, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('F1 weighted')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(methods), frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Clustering with and without the time-of-day block vs. the better supervised baseline', fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(OUT / 'figures' / 'tod_block.png', dpi=170)
    plt.close(fig)


def l2_table(runs):
    """TDOST label level (L2, variant `all` as in TDOST: every class incl. Other), weighted / macro F1."""
    from .report_final import MATCHED
    rows = []
    for city, (title, _) in VARIANTS.items():
        for name in MAIN:
            cells = [pick(runs, city, name, clu, budget, level='l2', variant='all') for _, clu, budget in MATCHED]
            if all(len(v) == 0 for v in cells):
                continue
            rows.append({'Variant': title, 'Method': name, 'Seeds': '/'.join(str(len(v)) for v in cells)}
                        | {col: f'{v.f1_weighted.mean():.2f} / {v.f1_macro.mean():.2f}' if len(v) else '–'
                           for (col, _, _), v in zip(MATCHED, cells)})
    return pd.DataFrame(rows)


def report(args):
    from .report_final import MATCHED
    global VARIANTS, OUT
    if getattr(args, 'variants', None):
        VARIANTS = {k: VARIANTS[k] for k in args.variants}
    if getattr(args, 'tag', None):
        OUT = OUT / args.tag
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'data').mkdir(exist_ok=True)
    runs = score_runs(args.rescore)
    sig = significance(runs, n_boot=args.n_boot)
    sig.to_csv(OUT / 'data' / 'significance.csv', index=False)
    data = dataset_table()
    data.to_csv(OUT / 'data' / 'dataset.csv', index=False)
    per_class, lenient, groups, confusions = detail_tables()
    per_class.to_csv(OUT / 'data' / 'per_class.csv', index=False)
    lenient.to_csv(OUT / 'data' / 'lenient.csv', index=False)
    groups.to_csv(OUT / 'data' / 'resident_streams.csv', index=False)
    tables, flat = main_tables(runs, sig)
    flat.to_csv(OUT / 'data' / 'main_comparison.csv', index=False)
    figures(runs, per_class, confusions)

    title = {c: t for c, (t, _) in VARIANTS.items()}
    rows = {
        'kyotomerged': 'all events, once, in time order | most recently begun activity still running',
        'kyotoflat': 'one stream per resident (R1, R2) and one for activities without a resident id; an event is '
                     'copied into the stream of every activity running at that time; windows never cross streams | '
                     'the activity that put the event into the stream',
        'kyotoresa': 'all events, once, in time order | R1\'s activity; else a running activity without resident id; '
                     'else No_Activity',
        'kyotoresb': 'all events, once, in time order | same for R2',
        'kyotoaff': 'all events, once, in time order (windows of `kyotomerged`) | of the running activities, the one '
                    'that most often fires the event\'s sensor when it runs alone',
        'kyotojoint': 'all events, once, in time order (windows of `kyotomerged`) | household state: all running '
                      'activities, e.g. `Sleep+Work`',
        'kyotoattr': 'one stream per resident and one for activities without a resident id; each event goes into one '
                     'stream only, that of its sensor-attributed activity | the sensor-attributed activity',
    }
    md = HEADER or [
        '# Kyoto7 (twor.2009) trial: Section 6 models under several ways of handling the two residents', '',
        'Generated by `python -m tist_revision.pipeline.kyoto_trial report`; do not edit. Findings are logged in '
        '`tist_revision/EXPLORATIONS.md`. Not part of the paper.', '']
    md = md + [
          'Protocol as in Section 6: clusters and supervised models are fit on continuous `FL_20` train windows '
          '(20 events, 50% overlap), tested on pre-segmented `FL_20_p` windows of held-out days (20% of the days); '
          'L1 labels (resident id removed), macro F1 over classes with ≥ 50 pool and ≥ 20 test windows. '
          'DISCOVER = MLM encoder + time-of-day histogram + SCAN (balanced); ≈100 = K 20 × 5 labels, ≈250 = K 50 × 5, '
          'all = K 50 with every pool window labeled. Mean ± std over seeds.', '']
    if any(c in rows for c in VARIANTS):
        md += ['## Variants', '', '| Variant | Dataset | Sensor stream | Label of an event |', '|---|---|---|---|']
        md += [f'| {t} | `{c}` | {rows[c]} |' for c, (t, _) in VARIANTS.items() if c in rows] + ['']
    md += ['## Data', '', md_table(data.assign(**{c: data[c].map('{:.1%}'.format) for c in data.columns
                                                 if data[c].dtype == float and data[c].max() <= 1})), '',
          '"Twin": another test window with exactly the same events and a different label (copies of one event in '
          'two resident streams). No method can label both twins correctly; the ceiling is the best achievable accuracy.', '']
    for mname, t in tables.items():
        md += [f'## {mname} (L1)', '', md_table(t), '',
               'Δ: paired day-block bootstrap of DISCOVER minus the better supervised baseline (mean [95% CI]; '
               '* p < 0.05, ** p < 0.01, *** p < 0.001). Paper houses: `results/final`.', '']
    md += ['![Label efficiency](figures/label_efficiency.png)', '']
    if len(lenient):
        t = lenient[lenient.city == 'kyotomerged'].pivot_table(index=['budget', 'method'], columns='scoring',
                                                                values=['f1_weighted', 'f1_macro', 'accuracy'])
        t.columns = [f'{a} ({b})' for a, b in t.columns]
        t = t.reindex([(c, n) for c, _, _ in MATCHED for n in MAIN if (c, n) in t.index]).reset_index()
        md += ['## Merged stream: strict vs. lenient scoring', '',
               'Lenient: a prediction naming any activity that is running at the window\'s last event counts as correct '
               '(the strict label is only the most recently begun one).', '', md_table(t), '']
    if len(groups):
        t = groups[groups.city == 'kyotoflat'].pivot_table(index=['budget', 'method'], columns='stream', values='accuracy')
        n = groups[groups.city == 'kyotoflat'].drop_duplicates('stream').set_index('stream')['test_windows']
        t.columns = [f'{c} (n = {n[c]})' for c in t.columns]
        t = t.reindex([(c, m) for c, _, _ in MATCHED for m in MAIN if (c, m) in t.index]).reset_index()
        md += ['## Resident streams: accuracy by stream', '', md_table(t), '']
    if len(per_class):
        md += ['## Per-class F1 at ≈250 labels', '']
        for city in VARIANTS:
            q = per_class[(per_class.city == city) & (per_class.budget == '≈250 labels')]
            if q.empty:
                continue
            t = q.pivot(index=['class', 'test_windows', 'scored'], columns='method', values='f1')
            t = t.reindex(columns=[m for m in MAIN if m in t.columns]).reset_index().sort_values('test_windows', ascending=False)
            md += [f'**{title[city]}**', '', md_table(t), '']
        md += ['![Per-class F1](figures/per_class_f1.png)', '', '![DISCOVER confusion](figures/confusion_discover.png)', '']
    l2 = l2_table(runs)
    if len(l2) and (l2.Method != 'DISCOVER').any():
        l2.to_csv(OUT / 'data' / 'l2_comparison.csv', index=False)
        md += ['## TDOST label level (L2), weighted / macro F1', '',
               'All L2 classes scored (incl. Other), as in the per-house appendix table of the paper.', '', md_table(l2), '']
    clu = clustering_table(runs)
    if clu.groupby('Variant').size().max() > 1:
        tod = tod_table(runs)
        tod.to_csv(OUT / 'data' / 'tod_block.csv', index=False)
        tod_figure(tod)
        wide = tod.assign(cell=[f'{w:.2f} / {m:.2f}' for w, m in zip(tod.f1_weighted, tod.f1_macro)]).pivot(
            index=['Variant', 'Method'], columns='budget', values='cell')
        order = list(dict.fromkeys(zip(tod.Variant, tod.Method)))
        wide = wide.reindex(order)[[c for c, _, _ in MATCHED]].reset_index()
        md += ['## With and without the time-of-day block (weighted / macro F1)', '',
               'Baseline cells take the better of DeepCASAS and TDOST per metric. Clustering rows other than DISCOVER '
               'were not chosen in advance; they are diagnostics, not results.', '', md_table(wide), '',
               '![Time-of-day block](figures/tod_block.png)', '']
        clu.to_csv(OUT / 'data' / 'clustering_variants.csv', index=False)
        md += ['## Clustering variants (weighted / macro F1)', '',
               'Same grid as the appendix ablations of the paper houses. DISCOVER is SCAN on TF+ToD.', '', md_table(clu), '']
    (OUT / 'REPORT.md').write_text('\n'.join(md))
    print(f'Saved {OUT}/REPORT.md ({len(runs)} runs)')


# --------------------------------------------------------------------------------------------------------------
# screen: label schemes for the merged stream, scored on training days only
# --------------------------------------------------------------------------------------------------------------

def screen(args):
    """Which label of a concurrent event suits the fixed model? Scored without the test set and without training.

    All schemes below label the same windows (continuous FL_20 training windows of `kyotomerged`), so the encoder
    and the clusters of `kyotomerged` apply to every scheme. The 32 training days are split into 4 folds: labels
    come from 3 folds (5 most confident windows per cluster for the clusterings; 100 / 250 random windows for a
    logistic regression on the sensor histogram, the stand-in for the supervised baselines), scores from the 4th.
    Windows are continuous, not pre-segmented as in the test protocol."""
    import ast
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from .mapping_eval import ARTIFACTS_DIR, UNMAPPED, l1_label, load_protocol, map_clusters, predict_activity
    from .mapping_eval import NO_ACTIVITY as NA
    from .rare_classes import representations

    df = pd.read_csv(RAW_DIR / 'data_processed.csv', low_memory=False)
    acts = pd.Series(running_activities(df['activity'], df['marker']), index=df.index)
    affinity = sensor_affinity(df, acts)
    fit, pool, _ = load_protocol('kyotomerged', level='l1')
    samples = json.loads((DATA_ROOT / 'kyotomerged' / 'FL_20' / 'train.json').read_text())['samples']
    last = [s['sensor_sequence'][-1] for s in samples]
    running = [tuple(a for a in str(e.get('activity_full', '')).split('|') if a and a != 'nan') for e in last]
    sensor = [e['sensor_id'] for e in last]
    def l1(a):
        return l1_label(a)
    schemes = {
        'Most recently begun (kyotomerged)': [l1(merged_label(r)) for r in running],
        'Sensor-attributed (kyotoaff)': [l1(attributed_label(r, s, affinity)) for r, s in zip(running, sensor)],
        'Household state (kyotojoint)': [l1(joint_label(r)) for r in running],
        'R1 labels (kyotoresa)': [l1(resident_label(r, 1)) for r in running],
        'R2 labels (kyotoresb)': [l1(resident_label(r, 2)) for r in running],
        'Most recently begun, resident id kept': [merged_label(r) if r else 'No_Activity' for r in running],
        'Sensor-attributed, resident id kept': [attributed_label(r, s, affinity) if r else 'No_Activity'
                                                 for r, s in zip(running, sensor)],
    }
    assert schemes['Most recently begun (kyotomerged)'] == pool['label'].tolist()
    single = np.array([len(r) <= 1 for r in running])
    subsets = {'all windows': np.ones(len(pool), bool), 'windows with one activity': single}

    days = pool['day'].to_numpy()
    uniq = np.random.default_rng(0).permutation(np.unique(days))
    folds = [set(uniq[i::4]) for i in range(4)]
    (_, (X_hist, _, _)), = representations('kyotomerged', fit, pool, pool.iloc[:1], ['hist_sensor_s1'])
    X_hist = np.asarray(X_hist, dtype=np.float32)
    clusterings = {'DISCOVER (SCAN, TF+ToD)': 'tf_s{s}+tod__scan_k{K}_todwbal', 'SCAN, TF': 'tf_s{s}__scan_k{K}',
                   'k-means, TF': 'tf_s{s}__kmeans_k{K}', 'k-means, sensor hist.': 'hist_sensor_s{s}__kmeans_k{K}'}

    def f1s(y, pred, y_lab):
        from collections import Counter
        pc, tc = Counter(y_lab), Counter(y)
        masked = sorted(c for c in tc if pc[c] >= 50 * 3 // 4 and tc[c] >= 20 and c != 'No_Activity')
        keep = y != 'No_Activity'  # the test protocol has no No_Activity windows
        return (f1_score(y[keep], pred[keep], average='weighted', zero_division=0),
                f1_score(y[keep], pred[keep], labels=masked, average='macro', zero_division=0) if masked else np.nan)

    rows = []
    for scheme, labels in schemes.items():
        y_all = np.array(labels)
        for subset, keep in subsets.items():
            for fi, fold in enumerate(folds):
                held = np.isin(days, list(fold)) & keep
                lab = ~np.isin(days, list(fold)) & keep
                li, hi = np.flatnonzero(lab), np.flatnonzero(held)
                for seed in SEEDS:
                    for name, pattern in clusterings.items():
                        for K, col in ((20, '≈100 labels'), (50, '≈250 labels')):
                            path = ARTIFACTS_DIR / 'assignments' / f'kyotomerged__{pattern.format(s=seed, K=K)}.npz'
                            if not path.exists():
                                continue
                            z = np.load(path)
                            mapping = map_clusters(z['pool_cluster'][li], z['pool_conf'][li], y_all[li], 5, skip=('No_Activity',))
                            pred = np.array([mapping.get(c, UNMAPPED) for c in z['pool_cluster'][hi]])
                            w, m = f1s(y_all[hi], pred, y_all[li])
                            rows.append({'scheme': scheme, 'subset': subset, 'fold': fi, 'seed': seed, 'method': name,
                                         'budget': col, 'f1_weighted': w, 'f1_macro': m, 'classes': len(set(y_all[hi]))})
                    for budget, col in (('100', '≈100 labels'), ('250', '≈250 labels')):
                        tr, va = budget_idx(y_all[li], budget, seed)
                        idx = li[np.concatenate([tr, va])]
                        scaler = StandardScaler().fit(X_hist[idx])
                        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X_hist[idx]), y_all[idx])
                        pred = predict_activity(clf.predict_proba(scaler.transform(X_hist[hi])), clf.classes_, skip=('No_Activity',))
                        w, m = f1s(y_all[hi], pred, y_all[li])
                        rows.append({'scheme': scheme, 'subset': subset, 'fold': fi, 'seed': seed,
                                     'method': 'Supervised stand-in (logistic regression, sensor hist.)',
                                     'budget': col, 'f1_weighted': w, 'f1_macro': m, 'classes': len(set(y_all[hi]))})
        print(f'  screened {scheme}', flush=True)
    out = pd.DataFrame(rows)
    (OUT / 'data').mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / 'data' / 'label_screen_runs.csv', index=False)
    g = out.groupby(['scheme', 'subset', 'method', 'budget'], sort=False)[['f1_weighted', 'f1_macro', 'classes']].mean().reset_index()
    g.to_csv(OUT / 'data' / 'label_screen.csv', index=False)
    wide = g.assign(cell=[f'{w:.2f} / {m:.2f}' for w, m in zip(g.f1_weighted, g.f1_macro)]).pivot(
        index=['scheme', 'subset', 'method'], columns='budget', values='cell')
    wide = wide.reindex(list(dict.fromkeys(zip(g.scheme, g.subset, g.method)))).reset_index()
    ncls = g.groupby('scheme', sort=False)['classes'].max().round().astype(int)
    wide.insert(1, 'classes', wide['scheme'].map(ncls))
    md = ['# Kyoto7: label schemes for concurrent events, screened on training days', '',
          'Generated by `python -m tist_revision.pipeline.kyoto_trial screen`; do not edit.', '',
          screen.__doc__.split('\n\n', 1)[1].replace('\n    ', ' ').strip(), '',
          'Weighted / macro F1, mean over 4 folds x 5 seeds. Macro F1 over classes with at least 38 labeled-fold and 20 '
          'held-out windows. No_Activity windows are not scored.', '', md_table(wide), '']
    (OUT / 'LABEL_SCREEN.md').write_text('\n'.join(md))
    print(f'Saved {OUT}/LABEL_SCREEN.md')


def budget_idx(y, budget, seed):
    """`baselines.data.budget_indices` with the L1 spelling of No_Activity."""
    order = np.random.default_rng(seed).permutation(len(y))
    drawn = order[:int(np.searchsorted(np.cumsum(y[order] != 'No_Activity'), int(budget))) + 1]
    is_val = np.arange(len(drawn)) % 5 == 4
    return drawn[~is_val], drawn[is_val]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--steps', nargs='+', default=['streams', 'windows'], choices=['streams', 'windows'],
                   help='configs and task files are always rewritten')
    p.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    p.set_defaults(fn=prepare)
    p = sub.add_parser('ablation', help='write the skynet task files of the appendix ablations')
    p.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    p.set_defaults(fn=lambda a: ablation_task_files(a.variants))
    p = sub.add_parser('screen', help='label schemes of the merged stream, scored on training days')
    p.set_defaults(fn=screen)
    p = sub.add_parser('report')
    p.add_argument('--rescore', action='store_true', help='re-score all runs instead of reading data/runs.csv')
    p.add_argument('--n_boot', type=int, default=2000)
    p.add_argument('--variants', nargs='+', default=ROUNDS[''], choices=list(VARIANTS))
    p.add_argument('--tag', help='write to a subfolder of the trial folder (e.g. round2)')
    p.set_defaults(fn=report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
