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
}
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
    for name, s in streams.items():
        s = s.drop(columns='running', errors='ignore').copy()
        s['first_activity_l2'] = s['activity_flattened'].map(tdost_label)
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
    """Skynet task arrays, same commands as `encoder_tf.txt` / `baselines.txt` restricted to the main models."""
    enc, base = [], []
    for name in VARIANTS:
        for s in SEEDS:
            model = f'trained_models/{name}/tf_nocoord_mlm_time_s{s}'
            enc.append(
                f'python train.py --config tist_revision/configs/alignment/{name}_fl20_seq_nocoord_mlm_time.yaml '
                f'--seed {s} --output-dir {model} && '
                f'python -m tist_revision.pipeline.encode_embeddings --city {name} --kind tf --model_dir {model} '
                f'--name tf_s{s} && '
                f'python -m tist_revision.pipeline.frozen_deep --cities {name} --representations tf_s{s}+tod '
                f'--methods scan --K 20 50 --seed {s} --tod_weight balanced --suffix _todwbal')
    for method in ('deepcasas', 'tdost'):
        for name in VARIANTS:
            for s in SEEDS:
                base.append(f'python -m tist_revision.baselines.run --method {method} --city {name} --seed {s} '
                            f'--protocol final --levels l1')
    tasks = PROJECT_DIR / 'skynet' / 'tasks'
    (tasks / 'kyoto_encoder_tf.txt').write_text('\n'.join(enc) + '\n')
    (tasks / 'kyoto_baselines.txt').write_text('\n'.join(base) + '\n')
    (tasks / 'kyoto_tdost_text.txt').write_text(
        f'python -m tist_revision.baselines.tdost_text --cities {" ".join(VARIANTS)}\n')
    print(f'Wrote {tasks}/kyoto_*.txt ({len(enc)} encoder, {len(base)} baseline tasks)')


def prepare(args):
    if 'streams' in args.steps:
        build_streams()
    for name in VARIANTS:
        for window in ('FL_20', 'FL_20_p'):
            cfg = sampling_config(name, window)
            if 'windows' in args.steps:
                subprocess.run([sys.executable, 'sample_data.py', '--config', str(cfg)], check=True, cwd=REPO)
        if 'windows' in args.steps:
            unused_text_embeddings(name)
        encoder_config(name)
    task_files()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--steps', nargs='+', default=['streams', 'windows'], choices=['streams', 'windows'],
                   help='configs and task files are always rewritten')
    p.set_defaults(fn=prepare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
