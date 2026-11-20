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
  - Repeated states are dropped for every kept source (2026-11-20 check on the raw rows): TV status is polled at
    second :02 of every minute (20,844 of 20,910 rows repeat, 97% of them "off"); a repeated light row follows
    the first one by 1 s (median) after the same switch press; a repeated shutter row comes ~22 s after the
    command (motor travel), with a switch press nearby in 23% of cases. Presence, doors and switches never repeat.

Variant `orangem` (2026-11-20): the same stream plus on/off events derived from the meters the occupant drives
(appliance power, plug power, water flow, bed pressure); see METERS. Thresholds are set from the value
distributions (standby vs. running levels), not from the activity labels. Ambient meters (temperature,
humidity, CO2, luminosity, noise), automatic loads (fridge, water heater, heaters) and whole-house totals stay out.

The stream is written in the format of the Kyoto7 trial streams and goes through the unchanged pipeline:
    python -m tist_revision.pipeline.orange_trial prepare [--variants orange orangem]  # stream, windows, configs, tasks
    (run `skynet/tasks/<variant>_*.txt`, or the same commands locally)
    python -m tist_revision.pipeline.orange_trial report [--variants ...] [--tag ...]  # -> results/temp/orange4home_trial
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
VARIANTS = {  # dataset name -> (report title, event CSV in RAW_DIR)
    NAME: ('Orange4Home', 'data_processed.csv'),
    'orangem': ('Orange4Home + meter events', 'data_processed_meters.csv'),
}
SEEDS = [1, 2, 3, 4, 5]
# Derived on/off sources of `orangem`: name -> (raw sources, a reading is active when its value is above this).
# Plugs idle at 0-1 W and run at >= 8 W; the cooktop idles at 10 W and heats at >= ~1 kW; the hood runs at ~100 W.
# Water meters and the bed pressure mat report only while water flows / someone lies on the bed.
METERS = {
    'kitchen_cooktop': (['kitchen_cooktop_power'], 100),
    'kitchen_hood': (['kitchen_hood_power'], 20),
    'kitchen_sink_water': (['kitchen_sink_hotwater_instantaneous', 'kitchen_sink_coldwater_instantaneous'], 0),
    'livingroom_tv_plug': (['livingroom_tv_plug_consumption'], 5),
    'livingroom_couch_plug': (['livingroom_couch_plug_consumption'], 5),
    'livingroom_table_plug': (['livingroom_table_plug_consumption'], 5),
    'office_tv_plug': (['office_tv_plug_consumption'], 5),
    'office_desk_plug': (['office_desk_plug_consumption'], 5),
    'bathroom_shower_water': (['bathroom_shower_hotwater_instantaneous', 'bathroom_shower_coldwater_instantaneous'], 0),
    'bathroom_sink_water': (['bathroom_sink_hotwater_instantaneous', 'bathroom_sink_coldwater_instantaneous'], 0),
    'toilet_flush': (['toilet_coldwater_instantaneous'], 0),
    'bedroom_bed': (['bedroom_bed_pressure'], 0),
}
SILENT_S = 60  # an active reading holds until the next reading of its source, at most this long
MERGE_S = 120  # off gaps shorter than this are bridged (burner cycling: 99% of gaps < 36 s; sink bursts)
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


def meter_events(df):
    """On/off events of the METERS sources: a reading above the threshold makes its source active until the next
    reading (at most SILENT_S); the raw sources of one meter are pooled, gaps under MERGE_S bridged."""
    rows = []
    for name, (sources, threshold) in METERS.items():
        spans = []
        for src in sources:
            r = df[df['ItemName'] == src]
            t = r['datetime'].to_numpy()
            end = np.minimum(np.append(t[1:], t[-1:]), t + np.timedelta64(SILENT_S, 's'))
            on = pd.to_numeric(r['Value']).to_numpy() > threshold
            spans += list(zip(t[on], end[on]))
        merged = []
        for s, e in sorted(spans):
            if merged and s - merged[-1][1] < np.timedelta64(MERGE_S, 's'):
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        rows += [(s, name, 'ON') for s, _ in merged] + [(e, name, 'OFF') for _, e in merged]
    out = pd.DataFrame(rows, columns=['datetime', 'ItemName', 'state'])
    print('meter events (on/off pairs):', (out[out.state == 'ON'].ItemName.value_counts()).to_dict())
    return out


def build_stream(name=NAME):
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
    if name != NAME:
        meters = meter_events(df)
        kinds |= {m: ('meter', None) for m in METERS}
        ev = pd.concat([ev[['datetime', 'ItemName', 'state']], meters]).sort_values('datetime', kind='stable')

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
    stream = VARIANTS[name][1]
    out.to_csv(RAW_DIR / stream, index=False)
    print(f'Wrote {RAW_DIR / stream}: {len(out)} events, {out["date"].nunique()} days, '
          f'{(activity == kt.NO_ACTIVITY).mean():.1%} outside activities')
    print(out['sensor'].map(lambda i: kinds[i][0]).value_counts().to_dict())
    print(out['activity_flattened'].value_counts().to_dict())
    if name != NAME:
        return out

    # House entry in the shared metadata file (rooms, sensor details and types; no coordinates).
    path = REPO / 'metadata' / 'casas_metadata.json'
    sensors = sorted(out['sensor'].unique())
    entry = {
        'name': TITLE, 'num_residents': 1,
        'casas_url': 'https://github.com/JCumin/jcumin.github.io/raw/refs/heads/main/orange4home.zip',
        'data_raw_filename': 'orange4home/o4h_all_events.csv', 'data_processed_filename': f'orange4home/{stream}',
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


def configs(name):
    title, stream = VARIANTS[name]
    for window in ('FL_20', 'FL_20_p'):
        preseg = window.endswith('_p')
        cfg = kt.SAMPLING_YAML.format(
            name=name, title=TITLE, window=window, raw=RAW_DIR, stream=stream, preseg=str(preseg).lower(),
            preseg_note=', pre-segmented by activity' if preseg else '', by_resident='false')
        head = '# Orange4Home trial' + ('' if name == NAME else f', variant `{name}` ({title})')
        cfg = cfg.replace(f'# Kyoto7 (twor.2009) trial, variant `{name}` (Orange4Home)', head)
        cfg = cfg.replace('kyoto_trial.py (prepare); see its docstring for the variants', 'orange_trial.py (prepare)')
        cfg = cfg.replace('dataset_name: "kyoto"', f'dataset_name: "{name}"')
        cfg = cfg.replace('# Event stream of this variant', '# Event stream (single resident; read through the stream-file path)')
        assert f'dataset_name: "{name}"' in cfg and cfg.startswith(head)
        (REPO / 'configs' / 'sampling' / f'{name}_{window}.yaml').write_text(cfg)
    src = (PROJECT_DIR / 'configs' / 'alignment' / 'cairo_fl20_seq_nocoord_mlm_time.yaml').read_text()
    cfg = src.replace('cairo', name).replace('# Cairo FL_20', '# Orange4Home trial' + ('' if name == NAME else f' ({title})') + ' FL_20')
    cfg = cfg.replace('_embeddings_baseline_clip.npz', '_embeddings_unused.npz')
    cfg = cfg.replace('  max_epochs: 100\n', '  max_epochs: 2000  # the 10,000-step limit ends training, as in the '
                                             'other houses (see kyoto_trial.encoder_config)\n')
    assert 'Cairo' not in cfg and cfg.count(name) == src.count('cairo') and 'max_epochs: 2000' in cfg
    (PROJECT_DIR / 'configs' / 'alignment' / f'{name}_fl20_seq_nocoord_mlm_time.yaml').write_text(cfg)

    enc, core, base = [], [], []
    fd = 'python -m tist_revision.pipeline.frozen_deep'
    for s in SEEDS:
        model = f'trained_models/{name}/tf_nocoord_mlm_time_s{s}'
        discover = [
            f'python train.py --config tist_revision/configs/alignment/{name}_fl20_seq_nocoord_mlm_time.yaml '
            f'--seed {s} --output-dir {model}',
            f'python -m tist_revision.pipeline.encode_embeddings --city {name} --kind tf --model_dir {model} --name tf_s{s}',
            f'{fd} --cities {name} --representations tf_s{s}+tod --methods scan --K 20 50 --seed {s} '
            f'--tod_weight balanced --suffix _todwbal']
        core.append(' && '.join(discover))  # DISCOVER only (Section 6 model)
        enc.append(' && '.join(discover + [  # plus the clustering variants of the ToD table
            f'{fd} --cities {name} --representations tf_s{s} tf_s{s}+tod --methods kmeans scan dec --K 20 50 --seed {s}',
            f'{fd} --cities {name} --representations hist_sensor_s{s} hist_sensor_tod_s{s} --methods kmeans --K 20 50 --seed {s}']))
    for method in ('deepcasas', 'tdost'):
        base += [f'python -m tist_revision.baselines.run --method {method} --city {name} --seed {s} --protocol final '
                 f'--levels l1' for s in SEEDS]
    tasks = PROJECT_DIR / 'skynet' / 'tasks'
    (tasks / f'{name}_encoder_tf.txt').write_text('\n'.join(enc) + '\n')
    (tasks / f'{name}_core_tf.txt').write_text('\n'.join(core) + '\n')
    (tasks / f'{name}_baselines.txt').write_text('\n'.join(base) + '\n')
    (tasks / f'{name}_tdost_text.txt').write_text(f'python -m tist_revision.baselines.tdost_text --cities {name}\n')
    print(f'Wrote configs and {tasks}/{name}_*.txt ({len(enc)} encoder, {len(base)} baseline tasks)')


def prepare(args):
    for name in args.variants:
        if 'stream' in args.steps:
            build_stream(name)
        configs(name)
        if 'windows' in args.steps:
            for window in ('FL_20', 'FL_20_p'):
                subprocess.run([sys.executable, 'sample_data.py', '--config', f'configs/sampling/{name}_{window}.yaml'],
                               check=True, cwd=REPO)
            kt.unused_text_embeddings(name)
            for cache in (ARTIFACTS_DIR / 'cache').glob(f'*{name}_FL_20*'):  # window caches of an earlier build
                cache.unlink()


def report(args):
    """The Kyoto7 trial report, pointed at this house."""
    kt.VARIANTS = {n: (VARIANTS[n][0], False) for n in args.variants}
    kt.OUT = OUT
    kt.HOUSE = TITLE
    kt.STREAMS = {n: RAW_DIR / VARIANTS[n][1] for n in args.variants}
    kt.HEADER = [
        '# Orange4Home trial: Section 6 models on a non-CASAS house (preliminary)', '',
        'Generated by `python -m tist_revision.pipeline.orange_trial report`; do not edit. Findings are logged in '
        '`tist_revision/EXPLORATIONS.md`. Not part of the paper.', '',
        'One occupant, 20 work days, on/off and open/closed sources only (presence, doors, switches, lights, '
        'shutters, TV status), state changes only; labels are the 20 classes of the dataset paper.'
        + (' "+ meter events": the same stream plus on/off events derived from appliance power, plug power, water '
           'flow and bed pressure (thresholds in `orange_trial.METERS`).' if len(args.variants) > 1 else ''), '']
    args.variants = list(kt.VARIANTS)
    kt.report(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--steps', nargs='+', default=['stream', 'windows'], choices=['stream', 'windows'])
    p.add_argument('--variants', nargs='+', default=[NAME], choices=list(VARIANTS))
    p.set_defaults(fn=prepare)
    p = sub.add_parser('report')
    p.add_argument('--rescore', action='store_true')
    p.add_argument('--n_boot', type=int, default=2000)
    p.add_argument('--variants', nargs='+', default=[NAME], choices=list(VARIANTS))
    p.add_argument('--tag', help='write to a subfolder of the trial folder')
    p.set_defaults(fn=report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
