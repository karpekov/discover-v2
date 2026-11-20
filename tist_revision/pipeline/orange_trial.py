"""
Orange4Home (Cumin et al., UCAmI 2017) trial: the Section 6 models on a non-CASAS house. Preliminary.

One occupant, Amiqual4Home apartment, 20 work days (2017-01-30 to 2017-02-24, about 8:00 to 17:00), 236 data
sources, activities labeled in situ by the occupant while following a planned routine. Download (open):
https://github.com/JCumin/jcumin.github.io/raw/refs/heads/main/orange4home.zip -> `data/raw/orange4home/`.

What the trial keeps, to stay as close as possible to the CASAS houses (motion / door / item events):
  - Sources with an on/off or open/closed state: presence detectors, doors / cupboards / drawers / windows, wall
    switches, lights and shutters (0 / 100), TV status. Meters and analog sources (power, energy, water, noise,
    luminosity, temperature, humidity, CO2, heater set points, weather) are dropped, as the temperature sensors
    are in the CASAS houses.
  - State changes only: several sources re-send their state periodically (`livingroom_tv_status`: 10,455 rows,
    37 changes).
  - Labels: the 20 classes of the dataset paper, `<Place>_<Activity>`; `Cleaning` is one class for all places.
  - Room of a sensor = the place prefix of its name. No coordinates (not used by the Section 6 encoder).

The stream is written in the format of the Kyoto7 trial streams and goes through the unchanged pipeline:
    python -m tist_revision.pipeline.orange_trial prepare   # stream, metadata, windows, configs, task file
    (run `skynet/tasks/orange_*.txt`, or the same commands locally)
    python -m tist_revision.pipeline.orange_trial report    # -> results/temp/orange4home_trial
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import kyoto_trial as kt
from .labels import tdost_label
from .mapping_eval import ARTIFACTS_DIR, DATA_ROOT, PROJECT_DIR, TEMP_DIR

NAME = 'orange'
TITLE = 'Orange4Home'
OUT = TEMP_DIR / 'orange4home_trial'
REPO = PROJECT_DIR.parent
RAW_DIR = Path('data/raw/orange4home')
STREAM = 'data_processed.csv'
SEEDS = [1, 2, 3, 4, 5]
ROOMS = {'entrance': 'entrance', 'kitchen': 'kitchen', 'livingroom': 'living_room', 'toilet': 'toilet',
         'staircase': 'staircase', 'walkway': 'walkway', 'bathroom': 'bathroom', 'office': 'office',
         'bedroom': 'bedroom'}
# Kept sources: (name pattern, sensor type, raw value -> state)
ON_OFF = {'ON': 'ON', 'OFF': 'OFF'}
OPEN_CLOSED = {'OPEN': 'OPEN', 'CLOSED': 'CLOSE'}
KINDS = [
    (r'_presence', 'presence', ON_OFF),
    (r'_(door|cupboard\d|drawer\d|closet_door|shower_door|fridge_door|window\d?)$', 'door', OPEN_CLOSED),
    (r'_switch', 'switch', ON_OFF),
    (r'_light\d?$', 'light', {'100': 'ON', '0': 'OFF'}),
    # openHAB roller shutters: 0 = up (open), 100 = down (closed)
    (r'_shutter\d?$', 'shutter', {'0': 'OPEN', '100': 'CLOSE'}),
    (r'_tv_status$', 'tv', ON_OFF),
]


def kind_of(item):
    for pattern, kind, states in KINDS:
        if re.search(pattern, item) and item.split('_')[0] in ROOMS:
            return kind, states
    return None, None


def label_name(raw):
    """`Living_room|Watching_TV` -> `Living_room_Watching_TV`; cleaning is one class for all places (dataset paper)."""
    place, activity = raw.split('|')
    return 'Cleaning' if activity == 'Cleaning' else f'{place}_{activity}'


def detail(item):
    """`kitchen_cupboard3` -> 'cupboard 3', `livingroom_presence_table` -> 'presence table'."""
    rest = item.split('_', 1)[1]
    return re.sub(r'(?<=[a-z])(\d)', r' \1', rest.replace('_', ' '))


def build_stream():
    df = pd.read_csv(RAW_DIR / 'o4h_all_events.csv')
    df['datetime'] = pd.to_datetime(df['Time'])
    df = df.sort_values('datetime', kind='stable').reset_index(drop=True)

    # Activity intervals from the START / STOP rows.
    lab = df[df['ItemName'] == 'label']
    intervals, open_ = [], {}
    for t, v in zip(lab['datetime'], lab['Value']):
        marker, raw = v.split(':', 1)
        if marker == 'START':
            open_[raw] = t
        elif raw in open_:
            intervals.append((open_.pop(raw), t, label_name(raw)))
    assert not open_, f'unclosed activities: {open_}'
    intervals = pd.DataFrame(intervals, columns=['start', 'end', 'activity']).sort_values('start')
    assert (intervals['start'].iloc[1:].to_numpy() >= intervals['end'].iloc[:-1].to_numpy()).all(), 'activities overlap'

    ev = df[df['ItemName'] != 'label'].copy()
    kinds = {item: kind_of(item) for item in ev['ItemName'].unique()}
    ev = ev[ev['ItemName'].map(lambda i: kinds[i][0] is not None)].copy()
    ev['state'] = [kinds[i][1].get(str(v).strip()) for i, v in zip(ev['ItemName'], ev['Value'])]
    assert ev['state'].notna().all(), ev.loc[ev['state'].isna(), ['ItemName', 'Value']].drop_duplicates()
    n_rows = len(ev)
    ev = ev[ev['state'] != ev.groupby('ItemName')['state'].shift()]  # state changes only
    print(f'{ev["ItemName"].nunique()} sources kept of {df["ItemName"].nunique() - 1}; {n_rows} rows -> {len(ev)} state changes')

    # Activity of an event: the interval that contains it (intervals do not overlap), else no_activity.
    idx = np.searchsorted(intervals['start'].to_numpy(), ev['datetime'].to_numpy(), side='right') - 1
    inside = (idx >= 0) & (ev['datetime'].to_numpy() <= intervals['end'].to_numpy()[np.clip(idx, 0, None)])
    activity = np.where(inside, intervals['activity'].to_numpy()[np.clip(idx, 0, None)], kt.NO_ACTIVITY)

    out = pd.DataFrame({
        'date': ev['datetime'].dt.strftime('%Y-%m-%d'), 'sensor': ev['ItemName'], 'state': ev['state'],
        'datetime': ev['datetime'], 'activity_flattened': activity, 'first_activity': activity,
        'activity_full': np.where(inside, activity, None), 'num_activities': inside.astype(int),
        'sensor_data_type': 'str', 'is_numeric': False,
        'first_activity_l2': [tdost_label(a) for a in activity],
        'sensor_location': ev['ItemName'].map(lambda i: ROOMS[i.split('_')[0]]),
    })
    out.to_csv(RAW_DIR / STREAM, index=False)
    print(f'Wrote {RAW_DIR / STREAM}: {len(out)} events, {out["date"].nunique()} days, '
          f'{(activity == kt.NO_ACTIVITY).mean():.1%} outside activities')
    print(out['sensor'].map(lambda i: kinds[i][0]).value_counts().to_dict())
    print(out['activity_flattened'].value_counts().to_dict())

    # House entry in the shared metadata file (rooms, sensor details and types; no coordinates).
    path = REPO / 'metadata' / 'casas_metadata.json'
    sensors = sorted(out['sensor'].unique())
    entry = {
        'name': TITLE, 'num_residents': 1,
        'casas_url': 'https://github.com/JCumin/jcumin.github.io/raw/refs/heads/main/orange4home.zip',
        'data_raw_filename': 'orange4home/o4h_all_events.csv', 'data_processed_filename': f'orange4home/{STREAM}',
        'sensor_location': {s: ROOMS[s.split('_')[0]] for s in sensors},
        'sensor_details': {s: detail(s) for s in sensors},
        'sensor_type': {s: kinds[s][0] for s in sensors},
    }
    # Appended as text: the file is hand-formatted and the other houses must stay byte-identical.
    text = path.read_text()
    if json.loads(text).get(NAME) != entry:
        marker = f',\n  "{NAME}": '
        if marker in text:  # an earlier build of this entry (always the last one)
            text = text[:text.index(marker)] + '\n}\n'
        body = text.rstrip()
        assert body.endswith('}') and NAME not in json.loads(text), 'unexpected metadata layout'
        block = json.dumps(entry, indent=2).replace('\n', '\n  ')
        path.write_text(body[:-1].rstrip() + marker + block + '\n}\n')
        assert json.loads(path.read_text())[NAME] == entry
    return out


def configs():
    for window in ('FL_20', 'FL_20_p'):
        preseg = window.endswith('_p')
        cfg = kt.SAMPLING_YAML.format(
            name=NAME, title=TITLE, window=window, raw=RAW_DIR, stream=STREAM, preseg=str(preseg).lower(),
            preseg_note=', pre-segmented by activity' if preseg else '', by_resident='false')
        cfg = cfg.replace('# Kyoto7 (twor.2009) trial, variant `orange` (Orange4Home)', '# Orange4Home trial')
        cfg = cfg.replace('kyoto_trial.py (prepare); see its docstring for the variants', 'orange_trial.py (prepare)')
        cfg = cfg.replace('dataset_name: "kyoto"', f'dataset_name: "{NAME}"')
        cfg = cfg.replace('# Event stream of this variant', '# Event stream (single resident; read through the stream-file path)')
        assert f'dataset_name: "{NAME}"' in cfg
        (REPO / 'configs' / 'sampling' / f'{NAME}_{window}.yaml').write_text(cfg)
    src = (PROJECT_DIR / 'configs' / 'alignment' / 'cairo_fl20_seq_nocoord_mlm_time.yaml').read_text()
    cfg = src.replace('cairo', NAME).replace('# Cairo FL_20', '# Orange4Home trial FL_20')
    cfg = cfg.replace('_embeddings_baseline_clip.npz', '_embeddings_unused.npz')
    cfg = cfg.replace('  max_epochs: 100\n', '  max_epochs: 2000  # the 10,000-step limit ends training, as in the '
                                             'other houses (see kyoto_trial.encoder_config)\n')
    assert 'Cairo' not in cfg and cfg.count(NAME) == src.count('cairo') and 'max_epochs: 2000' in cfg
    (PROJECT_DIR / 'configs' / 'alignment' / f'{NAME}_fl20_seq_nocoord_mlm_time.yaml').write_text(cfg)

    enc, base = [], []
    fd = 'python -m tist_revision.pipeline.frozen_deep'
    for s in SEEDS:
        model = f'trained_models/{NAME}/tf_nocoord_mlm_time_s{s}'
        enc.append(' && '.join([
            f'python train.py --config tist_revision/configs/alignment/{NAME}_fl20_seq_nocoord_mlm_time.yaml '
            f'--seed {s} --output-dir {model}',
            f'python -m tist_revision.pipeline.encode_embeddings --city {NAME} --kind tf --model_dir {model} --name tf_s{s}',
            f'{fd} --cities {NAME} --representations tf_s{s}+tod --methods scan --K 20 50 --seed {s} '
            f'--tod_weight balanced --suffix _todwbal',
            f'{fd} --cities {NAME} --representations tf_s{s} tf_s{s}+tod --methods kmeans scan dec --K 20 50 --seed {s}',
            f'{fd} --cities {NAME} --representations hist_sensor_s{s} hist_sensor_tod_s{s} --methods kmeans --K 20 50 --seed {s}']))
    for method in ('deepcasas', 'tdost'):
        base += [f'python -m tist_revision.baselines.run --method {method} --city {NAME} --seed {s} --protocol final '
                 f'--levels l1' for s in SEEDS]
    tasks = PROJECT_DIR / 'skynet' / 'tasks'
    (tasks / f'{NAME}_encoder_tf.txt').write_text('\n'.join(enc) + '\n')
    (tasks / f'{NAME}_baselines.txt').write_text('\n'.join(base) + '\n')
    (tasks / f'{NAME}_tdost_text.txt').write_text(f'python -m tist_revision.baselines.tdost_text --cities {NAME}\n')
    print(f'Wrote configs and {tasks}/{NAME}_*.txt ({len(enc)} encoder, {len(base)} baseline tasks)')


def prepare(args):
    if 'stream' in args.steps:
        build_stream()
    configs()
    if 'windows' in args.steps:
        for window in ('FL_20', 'FL_20_p'):
            subprocess.run([sys.executable, 'sample_data.py', '--config', f'configs/sampling/{NAME}_{window}.yaml'],
                           check=True, cwd=REPO)
        kt.unused_text_embeddings(NAME)
        for cache in (ARTIFACTS_DIR / 'cache').glob(f'*{NAME}_FL_20*'):  # window caches of an earlier build
            cache.unlink()


def report(args):
    """The Kyoto7 trial report, pointed at this house."""
    kt.VARIANTS = {NAME: (TITLE, False)}
    kt.OUT = OUT
    kt.HOUSE = TITLE
    kt.STREAMS = {NAME: RAW_DIR / STREAM}
    kt.HEADER = [
        '# Orange4Home trial: Section 6 models on a non-CASAS house (preliminary)', '',
        'Generated by `python -m tist_revision.pipeline.orange_trial report`; do not edit. Findings are logged in '
        '`tist_revision/EXPLORATIONS.md`. Not part of the paper.', '',
        'One occupant, 20 work days, on/off and open/closed sources only (presence, doors, switches, lights, '
        'shutters, TV status), state changes only; labels are the 20 classes of the dataset paper.', '']
    kt.report(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--steps', nargs='+', default=['stream', 'windows'], choices=['stream', 'windows'])
    p.set_defaults(fn=prepare)
    p = sub.add_parser('report')
    p.add_argument('--rescore', action='store_true')
    p.add_argument('--n_boot', type=int, default=2000)
    p.set_defaults(fn=report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
