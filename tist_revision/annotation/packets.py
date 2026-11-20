"""Human annotation study on the revised clusters (Section 5 redo): build blinded rater packets, score returns.

Items: per house and cluster (SCAN K=20, frozen TF+ToD encoder, balanced ToD weighting), the m most confident
training windows ('top', the windows that name the cluster, as in v1) plus n_random windows drawn from the rest
of the cluster in equal confidence strata ('random', label-propagation check, R2-2.13). Windows of one cluster
are at least `min_sep` minutes apart, so overlapping 50%-stride windows are not shown twice.

Each item is annotated by two raters (rotating pairs, balanced load). Every rater gets one self-contained HTML
file (floor plans, sensor events, taxonomy and app inlined; works offline from file://). Cluster ids, confidence,
stratum and CASAS labels stay in key.csv, which is never sent.

    python -m tist_revision.annotation.packets build [--name v2_k20] [--raters 6] [--n_random 2]
    python -m tist_revision.annotation.packets score [--name v2_k20]   # reads <out>/returned/*.csv

`--study orange` builds the same protocol for Orange4Home (one home, `orange_k20`): its sensors are typed
(presence, door / cupboard / drawer contacts, wall switches, lights, shutters, TV), so the replay marks each kind
differently and the taxonomy offers the rooms and fixtures of that apartment. The floor plan and the sensor
coordinates come from `orange_floorplan.py`.
"""
import argparse
import base64
import hashlib
import itertools
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..pipeline.mapping_eval import ARTIFACTS_DIR, DATA_ROOT, TEMP_DIR, l1_label

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
TAXONOMY = HERE / 'pdl_taxonomy.json'
TEMPLATE = HERE / 'template.html'
METADATA = PROJECT_ROOT / 'metadata' / 'casas_metadata.json'
FLOOR_PLANS = PROJECT_ROOT / 'metadata' / 'floor_plans_augmented'
OUT_ROOT = TEMP_DIR / 'annotation_study'
CITIES = ['milan', 'aruba', 'cairo']
HOME_NAMES = {'milan': 'Home A', 'aruba': 'Home B', 'cairo': 'Home C', 'orange': 'Home D'}
HOME_PREFIX = {'milan': 'A', 'aruba': 'B', 'cairo': 'C', 'orange': 'D'}
TYPED_HOMES = {'orange'}  # homes whose metadata `sensor_type` separates presence, contact, switch, light, shutter, TV

INSTRUCTIONS = """
<h2>Annotating sensor replays</h2>
<p>Each item replays the ambient sensors of a real home on its floor plan. Motion sensors light up when someone
moves near them; door sensors (blue squares) report a door opening or closing. There are no cameras: the replay
is all the information available.</p>
<p>The <b style="color:#b45309">highlighted segment</b> (yellow band on the timeline, yellow frame on the map) is
the part to annotate. A short context before and after is shown in gray only to help you interpret it.
Use <i>Play</i> or <i>Replay segment</i> (key <b>r</b>); long pauses are skipped by default.</p>
<p>The <i>Sensor summary</i> box lists the sensors activated in the highlighted segment: rooms in the order they
were first activated, how often, and what each sensor is placed at (e.g., <i>bed</i>, <i>stove</i>,
<i>armchair</i>), or <i>other</i> for general room sensors. Use it together with the replay.</p>
<p>Choose the pattern that best describes the movement in the highlighted segment. Patterns are grouped by room
or by the rooms involved; open a group to see its options.</p>
<ul>
<li>Pick the <b>most specific</b> option you are confident about (e.g., <i>Near the stove</i>). If you can only
tell the room, choose the general option of that room (e.g., <i>Movement in the kitchen</i>).</li>
<li>If the person moves between two rooms, use <i>Moving between two rooms</i>; for three or more rooms,
<i>Walking through the house</i>. If several far-apart sensors fire at the same time, use <i>Simultaneous
movement in several rooms</i>.</li>
<li>Use <i>None of the above</i> only if no option fits.</li>
<li>Annotate each item on its own; items are in random order and unrelated to each other.</li>
</ul>
<p>Press <b>Enter</b> (or <i>Save and next</i>) to save. You can go back and change any answer. Your work is
saved in this browser automatically, so you can close the file and continue later <i>in the same browser</i>.
When you are done, click <b>Export annotations (CSV)</b> and send the downloaded file back.</p>
<p class="hint">This study has been determined exempt by the Georgia Institute of Technology Institutional Review
Board (protocol [XXXX]). Participation is voluntary and you may stop at any time. No personal information is
recorded: only your assigned rater ID, your answers, and your optional notes. The sensor data come from
public datasets of anonymous homes. By continuing, you agree to take part. Questions: [contact email].</p>
<p class="hint">Keys: Space play/pause · r replay segment · ←/→ previous/next · Enter save and next.</p>
"""

INSTRUCTIONS_ORANGE = """
<h2>Annotating sensor replays</h2>
<p>Each item replays the ambient sensors of a real two-floor apartment on its floor plan (first floor on top,
ground floor below, connected by the staircase). There are no cameras: the replay is all the information
available. The apartment has several kinds of sensors:</p>
<ul>
<li><b>Presence sensors</b> (red circles) light up when someone moves in the area they cover.</li>
<li><b>Contact sensors</b> (blue squares) report a door, cupboard, drawer, the fridge or a window opening or
closing.</li>
<li><b>Wall switches</b> (purple diamonds) flash when a button is pressed. <b>Lights</b> (yellow) and <b>TVs</b>
(green) stay marked while they are on. <b>Shutters</b> (teal) flash when they open or close. A short tag
(e.g., <i>light off</i>) appears next to each of these events.</li>
</ul>
<p>Lights or TVs that were switched on before the replay starts are not shown.</p>
<p>The <b style="color:#b45309">highlighted segment</b> (yellow band on the timeline, yellow frame on the map) is
the part to annotate. A short context before and after is shown in gray only to help you interpret it.
Use <i>Play</i> or <i>Replay segment</i> (key <b>r</b>); long pauses are skipped by default.</p>
<p>The <i>Sensor summary</i> box lists what the sensors reported in the highlighted segment: rooms in the order
they were first active, and for each room what happened and how often (e.g., <i>motion ×12</i>,
<i>fridge opened ×1</i>, <i>light switched off ×2</i>). Use it together with the replay.</p>
<p>Choose the pattern that best describes the movement in the highlighted segment. Patterns are grouped by room
or by the rooms involved; open a group to see its options.</p>
<ul>
<li>Pick the <b>most specific</b> option you are confident about (e.g., <i>Near the fridge</i>). If you can only
tell the room, choose the general option of that room (e.g., <i>Movement in the kitchen</i>).</li>
<li>If the person moves between two rooms, use <i>Moving between two rooms</i>; if they change floors, use
<i>Going upstairs</i> or <i>Going downstairs</i>; for three or more rooms, <i>Walking through the house</i>.</li>
<li>Use <i>None of the above</i> only if no option fits.</li>
<li>Annotate each item on its own; items are in random order and unrelated to each other.</li>
</ul>
""" + INSTRUCTIONS[INSTRUCTIONS.index('<p>Press <b>Enter</b>'):]

STUDIES = {
    'casas': {'cities': CITIES, 'name': 'v2_k20', 'instructions': INSTRUCTIONS,
              'packet': 'DISCOVER_annotation_{rater}.html'},
    'orange': {'cities': ['orange'], 'name': 'orange_k20', 'instructions': INSTRUCTIONS_ORANGE,
               'packet': 'DISCOVER_annotation_homeD_{rater}.html'},
}

# Typed homes: what an event says in the sensor summary, by sensor kind and state. Events not listed (a presence
# sensor going off, a switch being released one second after the press, a door closing) are left out, as for CASAS.
EVENT_TEXT = {
    'presence': {'ON': 'motion{what}'},
    'door': {'OPEN': '{what} opened'},
    'switch': {'ON': 'wall switch pressed'},
    'light': {'ON': 'light switched on', 'OFF': 'light switched off'},
    'shutter': {'OPEN': 'shutter opened', 'CLOSE': 'shutter closed'},
    'tv': {'ON': 'TV switched on', 'OFF': 'TV switched off'},
}
REPLAY_KIND = {'presence': 'motion'}  # kind names of the replay app; the others keep the metadata name


def _seed(*parts) -> int:
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def load_city(city: str):
    """Training windows (same rows as the cluster assignments) and the deduplicated event stream."""
    with open(DATA_ROOT / city / 'FL_20' / 'train.json') as f:
        samples = json.load(f)['samples']
    windows, events = [], set()
    for s in samples:
        seq = s['sensor_sequence']
        gt = s['metadata'].get('ground_truth_labels', {})
        windows.append({
            'sample_id': s['sample_id'], 'start': seq[0]['timestamp'], 'end': seq[-1]['timestamp'],
            'casas_last': l1_label(seq[-1]['activity_l1']), 'casas_primary': l1_label(gt.get('primary_l1', 'no_activity')),
        })
        events.update((e['timestamp'], e['sensor_id'], e['event_type'].upper()) for e in seq)
    w = pd.DataFrame(windows)
    w['start'], w['end'] = pd.to_datetime(w['start'], format='ISO8601'), pd.to_datetime(w['end'], format='ISO8601')
    ev = pd.DataFrame(sorted(events), columns=['timestamp', 'sensor_id', 'state'])
    ev['timestamp'] = pd.to_datetime(ev['timestamp'], format='ISO8601')
    return w, ev


def select_items(w: pd.DataFrame, m: int, n_random: int, min_sep: pd.Timedelta, rng: np.random.Generator):
    rows = []
    for c in sorted(w['cluster'].unique()):
        cw = w[w['cluster'] == c].sort_values('conf', ascending=False)
        rank = {i: r for r, i in enumerate(cw.index, 1)}
        taken = []

        def free(i):
            return all(abs(w.at[i, 'start'] - t) >= min_sep for t in taken)

        top = []
        for i in cw.index:
            if len(top) == m:
                break
            if free(i):
                top.append(i)
                taken.append(w.at[i, 'start'])
        rest = [i for i in cw.index if i not in top and free(i)]
        rand = []
        for stratum in np.array_split(np.array(rest), n_random) if n_random and rest else []:
            for i in rng.permutation(stratum):
                if free(i):
                    rand.append(int(i))
                    taken.append(w.at[i, 'start'])
                    break
        for kind, idx in (('top', top), ('random', rand)):
            for i in idx:
                rows.append({'idx': int(i), 'cluster': int(c), 'stratum': kind, 'conf': float(w.at[i, 'conf']),
                             'conf_rank': rank[i], 'cluster_size': len(cw)})
    return pd.DataFrame(rows)


def sensor_info(meta_city: dict, overrides: dict | None = None) -> dict:
    """sensor id -> (room, intended use or None), in plain words for raters. `overrides` (taxonomy
    houses.<city>.sensor_overrides) replace metadata entries that contradict the floor plan."""
    def room(s):
        s = (s or '').strip().replace('_', ' ')
        return 'unmarked location' if s.lower() in ('', 'unk', 'unknown') else s.capitalize()
    def use(s):
        s = re.sub(r'\s*\(.*?\)', '', s or '')
        return re.sub(r'\s+in (the )?(kitchen|master bedroom|guest bedroom)$', '', s).strip() or None
    info = {k: (room(meta_city['sensor_location'].get(k)), use(meta_city['sensor_details'].get(k)))
            for k in meta_city['sensor_coordinates']}
    info.update({k: tuple(v) for k, v in (overrides or {}).items()})
    return info


def segment_summary(ev: pd.DataFrame, info: dict) -> str:
    """One-line text of the activations in the highlighted segment: rooms in order of first activation,
    with the named fixtures of each room."""
    on = ev[ev['state'].isin(['ON', 'OPEN'])]
    if on.empty:
        return 'No sensor activations in the highlighted segment.'
    rooms = {}
    for sid, st in zip(on['sensor_id'], on['state']):
        room, use = info.get(sid, ('unmarked location', None))
        what = (f'{use} opened' if st == 'OPEN' else use) if use else ('door opened' if st == 'OPEN' else 'other')
        rooms.setdefault(room, {}).setdefault(what, 0)
        rooms[room][what] += 1
    mins = (ev['timestamp'].iloc[-1] - ev['timestamp'].iloc[0]).total_seconds() / 60
    span = f'{mins:.0f} min' if mins >= 1 else 'less than 1 min'
    parts = []
    for room, c in rooms.items():
        detail = ', '.join(f'{w} ×{n}' for w, n in sorted(c.items(), key=lambda x: (x[0] == 'other', -x[1])))
        parts.append(f'{room} ×{sum(c.values())} ({detail})')
    return f'{len(on)} activations in {span}. ' + ' → '.join(parts) + '.'


def typed_sensor_info(meta_city: dict) -> dict:
    """sensor id -> (room, kind, what) for a typed home; `what` completes the EVENT_TEXT phrase of the kind."""
    info = {}
    for sid, kind in meta_city['sensor_type'].items():
        room = meta_city['sensor_location'][sid].replace('_', ' ').capitalize()
        what = re.sub(r'\s*\d+$', '', meta_city['sensor_details'][sid])
        if kind == 'presence':
            what = re.sub(r'^presence\s*', '', what)
            what = f' at the {what}' if what else ''
        elif kind == 'door':
            what = {'fridge door': 'fridge'}.get(what, 'front door' if (room, what) == ('Entrance', 'door') else what)
        info[sid] = (room, kind, what)
    return info


def typed_summary(ev: pd.DataFrame, info: dict) -> str:
    """segment_summary for a typed home: rooms in order of first event, with what happened in each room."""
    rooms, n = {}, 0
    for sid, st in zip(ev['sensor_id'], ev['state']):
        room, kind, what = info[sid]
        phrase = EVENT_TEXT[kind].get(st)
        if phrase is None:
            continue
        phrase = phrase.format(what=what)
        rooms.setdefault(room, {}).setdefault(phrase, 0)
        rooms[room][phrase] += 1
        n += 1
    if not n:
        return 'No sensor events in the highlighted segment.'
    mins = (ev['timestamp'].iloc[-1] - ev['timestamp'].iloc[0]).total_seconds() / 60
    span = f'{mins:.0f} min' if mins >= 1 else 'less than 1 min'
    parts = []
    for room, c in rooms.items():
        detail = ', '.join(f'{w} ×{k}' for w, k in sorted(c.items(), key=lambda x: -x[1]))
        parts.append(f'{room} ×{sum(c.values())} ({detail})')
    return f'{n} sensor events in {span}. ' + ' → '.join(parts) + '.'


def item_payload(ev: pd.DataFrame, start, end, context_s: int, info: dict, summarize=segment_summary) -> dict:
    c0, c1 = start - pd.Timedelta(seconds=context_s), end + pd.Timedelta(seconds=context_s)
    e = ev[(ev['timestamp'] >= c0) & (ev['timestamp'] <= c1)]
    rel = (e['timestamp'] - c0).dt.total_seconds().round(2).tolist()
    return {
        'clock_start': c0.strftime('%Y-%m-%dT%H:%M:%S'),
        'duration': round((c1 - c0).total_seconds(), 1),
        'target': [round((start - c0).total_seconds(), 1), round((end - c0).total_seconds(), 1)],
        'events': [[r, s, st] for r, s, st in zip(rel, e['sensor_id'], e['state'])],
        'summary': summarize(e[(e['timestamp'] >= start) & (e['timestamp'] <= end)], info),
    }


def assign_raters(item_ids, n_raters: int, rng: random.Random) -> dict:
    """Two raters per item; greedy balance of per-rater load and of pair counts (rotating pairs)."""
    load = {r: 0 for r in range(n_raters)}
    pair_n = {p: 0 for p in itertools.combinations(range(n_raters), 2)}
    out = {}
    for it in item_ids:
        best = min(pair_n, key=lambda p: (load[p[0]] + load[p[1]], pair_n[p], rng.random()))
        out[it] = best
        pair_n[best] += 1
        load[best[0]] += 1
        load[best[1]] += 1
    return out


def home_block(city: str, meta: dict, tax: dict) -> dict:
    img = FLOOR_PLANS / f'{city}.png'
    w, h = Image.open(img).size
    house = tax['houses'][city]
    labels = [l['id'] for l in tax['labels'] if l['group'] in house['groups'] and l['id'] not in house['exclude']
              and city in l.get('only', [city])]
    block = {
        'name': HOME_NAMES[city],
        'image': {'src': 'data:image/png;base64,' + base64.b64encode(img.read_bytes()).decode(), 'w': w, 'h': h},
        'sensors': {k: [v[0], h - v[1]] for k, v in meta[city]['sensor_coordinates'].items()},
        'labels': labels,
    }
    if city in TYPED_HOMES:
        block['kinds'] = {k: REPLAY_KIND.get(v, v) for k, v in meta[city]['sensor_type'].items()}
    return block


def build(args):
    study = STUDIES[args.study]
    cities = study['cities']
    out = OUT_ROOT / args.name
    (out / 'packets').mkdir(parents=True, exist_ok=True)
    (out / 'returned').mkdir(exist_ok=True)
    tax = json.loads(TAXONOMY.read_text())
    meta = json.loads(METADATA.read_text())
    rng = np.random.default_rng(args.seed)
    items, payload = [], {}
    for city in cities:
        w, ev = load_city(city)
        typed = city in TYPED_HOMES
        info = (typed_sensor_info(meta[city]) if typed
                else sensor_info(meta[city], tax['houses'][city].get('sensor_overrides')))
        summarize = typed_summary if typed else segment_summary
        z = np.load(ARTIFACTS_DIR / 'assignments' / f'{city}__{args.run}.npz')
        assert len(z['pool_cluster']) == len(w), (city, len(z['pool_cluster']), len(w))
        w['cluster'], w['conf'] = z['pool_cluster'], z['pool_conf']
        sel = select_items(w, args.m, args.n_random, pd.Timedelta(minutes=args.min_sep), rng)
        order = np.random.default_rng(_seed(args.seed, city)).permutation(len(sel))
        for k, (_, r) in zip(order, sel.iterrows()):
            win = w.loc[r['idx']]
            item_id = f'{HOME_PREFIX[city]}{k + 1:03d}'
            items.append({'item_id': item_id, 'home': city, 'cluster': r['cluster'], 'stratum': r['stratum'],
                          'conf': round(r['conf'], 4), 'conf_rank': r['conf_rank'], 'cluster_size': r['cluster_size'],
                          'sample_id': win['sample_id'], 'start': win['start'], 'end': win['end'],
                          'casas_last': win['casas_last'], 'casas_primary': win['casas_primary']})
            payload[item_id] = {'id': item_id, 'home': city,
                                **item_payload(ev, win['start'], win['end'], args.context, info, summarize)}
        print(f'{city}: {len(sel)} items ({(sel.stratum == "top").sum()} top, {(sel.stratum == "random").sum()} random)')
    key = pd.DataFrame(items).sort_values('item_id').reset_index(drop=True)

    pyrng = random.Random(args.seed)
    ids = key['item_id'].tolist()
    pyrng.shuffle(ids)
    pairs = assign_raters(ids, args.raters, pyrng)
    rater_ids = [f'R{r + 1:02d}' for r in range(args.raters)]
    key['raters'] = key['item_id'].map(lambda i: ';'.join(rater_ids[r] for r in pairs[i]))
    key.to_csv(out / 'key.csv', index=False)

    homes = {c: home_block(c, meta, tax) for c in cities}
    template = TEMPLATE.read_text()
    assignment = []
    for r, rid in enumerate(rater_ids):
        mine = [i for i in key['item_id'] if r in pairs[i]]
        rr = random.Random(_seed(args.seed, rid))
        city_order = cities[:]
        rr.shuffle(city_order)
        ordered = []
        for c in city_order:
            block = [i for i in mine if key.set_index('item_id').at[i, 'home'] == c]
            rr.shuffle(block)
            ordered += block
        build_id = hashlib.sha256(json.dumps([study['instructions'], [payload[i] for i in ordered]], default=str)
                                  .encode()).hexdigest()[:8]
        data = {
            'packet': f'{args.name}_{rid}', 'rater': rid, 'build': build_id,
            'homes': {c: homes[c] for c in city_order if any(payload[i]['home'] == c for i in ordered)},
            'taxonomy': {k: tax[k] for k in ('categories', 'groups', 'labels')},
            'items': [payload[i] for i in ordered], 'instructions': study['instructions'],
        }
        html = template.replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))
        path = out / 'packets' / study['packet'].format(rater=rid)
        path.write_text(html, encoding='utf-8')
        assignment += [{'rater': rid, 'item_id': i, 'position': n} for n, i in enumerate(ordered, 1)]
        print(f'{rid}: {len(ordered)} items, {path.stat().st_size / 1e6:.1f} MB')
    pd.DataFrame(assignment).to_csv(out / 'assignment.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps({**vars(args), 'n_items': len(key), 'raters': rater_ids,
                                                   'items_per_rater': pd.DataFrame(assignment)['rater'].value_counts().to_dict()},
                                                  indent=1, default=str))
    print(f'outputs: {out}  (send only packets/*.html; key.csv stays private)')


def score(args):
    from ..pipeline import rater_stats as rs
    out = OUT_ROOT / args.name
    tax = json.loads(TAXONOMY.read_text())
    lab = {l['id']: l for l in tax['labels']}
    parent = {i: (i if l['group'] in tax['self_parent_groups'] else l['group']) for i, l in lab.items()}
    key = pd.read_csv(out / 'key.csv')
    src = out / 'returned' / args.returned
    files = sorted(src.glob('*.csv'))
    if not files:
        raise SystemExit(f'no returned CSVs in {src}')
    r = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(['rater', 'item_id'], keep='last')
    expected = key.assign(rater=key['raters'].str.split(';')).explode('rater')[['item_id', 'rater']]
    missing = expected.merge(r[['item_id', 'rater']], how='left', indicator=True).query('_merge == "left_only"')
    unexpected = r.merge(expected, how='left', indicator=True).query('_merge == "left_only"')
    print(f'{len(r)} annotations from {r.rater.nunique()} raters; missing {len(missing)}, unexpected {len(unexpected)}')

    df = r.merge(key[['item_id', 'home', 'cluster', 'stratum', 'casas_last']], on='item_id')
    df['slot'] = df.groupby('item_id')['rater'].rank(method='first').map(lambda x: f'r{int(x)}')
    v1 = df.rename(columns={'home': 'city', 'item_id': 'task_code', 'rater': 'rater_id', 'slot': 'rater'})
    v1['label_level_up'] = v1['label_id'].map(parent)
    v1['label'] = v1['label_id']
    v1['casas'] = v1['casas_last']
    scores = out / 'scores' / args.returned
    scores.mkdir(parents=True, exist_ok=True)
    v1.to_csv(scores / 'ratings.csv', index=False)

    top = v1[v1['stratum'] == 'top']
    rng = np.random.default_rng(0)
    rows = []
    for city, d in top.groupby('city'):
        for col, level in (('label', 'original'), ('label_level_up', 'level_up')):
            sw, sc, dc = rs.pair_agreement(d, col)
            rows.append({'city': city, 'level': level, 'windows': d['task_code'].nunique(), 'ratings': len(d),
                         'fleiss_window': rs.fleiss_items(d, col), 'kripp_alpha': rs.kripp_alpha(d, col),
                         'fleiss_window_ci': rs.boot_ci(d, lambda x: rs.fleiss_items(x, col), rng).round(3).tolist(),
                         'fleiss_cluster': rs.fleiss_clusters(d, col),
                         'fleiss_cluster_ci': rs.boot_ci(d, lambda x: rs.fleiss_clusters(x, col), rng).round(3).tolist(),
                         'agree_same_window': sw, 'agree_same_cluster_other_window': sc, 'agree_other_cluster': dc,
                         'none_rate': (d['label_id'] == 'none').mean()})
    pd.DataFrame(rows).to_csv(scores / 'agreement.csv', index=False)

    maj = top.groupby(['city', 'cluster'])['label'].agg(lambda s: s.value_counts().index[0]).rename('cluster_label')
    maj_up = top.groupby(['city', 'cluster'])['label_level_up'].agg(lambda s: s.value_counts().index[0]).rename('cluster_parent')
    rnd = v1[v1['stratum'] == 'random'].join(maj, on=['city', 'cluster']).join(maj_up, on=['city', 'cluster'])
    prop = rnd.assign(match=rnd['label'] == rnd['cluster_label'], match_up=rnd['label_level_up'] == rnd['cluster_parent'])
    prop.groupby('city')[['match', 'match_up']].mean().to_csv(scores / 'propagation.csv')
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print('propagation accuracy (random windows vs cluster label from top windows):')
    print(prop.groupby('city')[['match', 'match_up']].mean().round(3))
    print(f'outputs: {scores}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['build', 'score'])
    p.add_argument('--study', choices=list(STUDIES), default='casas', help='casas: Milan, Aruba, Cairo; orange: Orange4Home')
    p.add_argument('--name', default=None, help='output folder under annotation_study/ (default: v2_k20 / orange_k20)')
    p.add_argument('--run', default='tf_s1+tod__scan_k20_todwbal', help='assignment file suffix (main method, seed 1)')
    p.add_argument('--m', type=int, default=5)
    p.add_argument('--n_random', type=int, default=2)
    p.add_argument('--raters', type=int, default=6)
    p.add_argument('--context', type=int, default=60, help='seconds of context before/after the window')
    p.add_argument('--min_sep', type=int, default=2, help='minutes between windows of one cluster')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--returned', default='', help='subfolder of returned/ to score (e.g. final)')
    args = p.parse_args()
    args.name = args.name or STUDIES[args.study]['name']
    build(args) if args.cmd == 'build' else score(args)


if __name__ == '__main__':
    main()
