"""
TDOST rule-based v1-1 sentences (time between readings) and their sentence embeddings, ported from
`~/code/har/TDOST/TDOST/generate_embeddings_v1-1.py` (cbalab/TDOST): same templates, house layout phrases,
value mapping and number-to-words. Encoder: `all-distilroberta-v1` (768-d, as in TDOST's best HAR setting).

Per window (20 events, chronological):
  event 0:  "{Motion sensor }{location} fired with value {ON} at {fourteen} hours {five} minutes {PM}"
  event j:  "After {twelve} seconds, {Motion sensor }{location} fired with value {OFF}"
Stored as `artifacts/tdost/{city}.npz`: `emb` (unique sentences x 768, float16) and `{protocol}_{split}` index
arrays (windows x 20). Usage:
    python -m tist_revision.baselines.tdost_text --cities milan aruba cairo
"""
import argparse
import math
from datetime import datetime

import inflect
import numpy as np

from ..pipeline.mapping_eval import ARTIFACTS_DIR
from .data import SPLITS, load_split

OUT = ARTIFACTS_DIR / 'tdost'
ENCODER = 'all-distilroberta-v1'
_P = inflect.engine()

SENSOR_TYPE = {"M": "Motion sensor ", "D": "Door sensor ", "T": "Temperature sensor ", "I": "Item sensor ",
               "L": "Light switch sensor "}

LAYOUT = {
    'aruba': {
        ("T003", "M015", "M019", "M017", "M016"): "in Kitchen",
        ("D002",): "between kitchen and back door",
        ("M018",): "between Kitchen and Dining area",
        ("M014",): "in Dining area",
        ("M013",): "between Dining area and Living room",
        ("T002", "M012", "M020"): "in Living room",
        ("M009", "M010"): "between Living room and home entrance aisle",
        ("M011", "M008", "D001"): "in home entrance aisle",
        ("M001", "M002", "M003", "M005", "M007", "T001"): "in the first bedroom",
        ("M006",): "between the first bedroom and home entrance aisle",
        ("M004",): "between the first bedroom and first bathroom",
        ("M021", "T004"): "in the aisle between second bathroom and dining area",
        ("M031", "D003"): "in second bathroom",
        ("M022",): "in the aisle between second bathroom and second bedroom",
        ("M024", "M023"): "in second bedroom",
        ("M029",): "between aisle and second bathroom",
        ("M030",): "in aisle between garage door and second bathroom",
        ("D004",): "on garage door",
        ("M028",): "between office and garage door aisle",
        ("M027", "M026", "M025", "T005"): "in office",
    },
    'milan': {
        ("M001",): "near home entrance",
        ("M002",): "near home entrance towards living room",
        ("M003",): "in dining room",
        ("M004",): "in living room",
        ("M005",): "in living room near slider door",
        ("M006",): "between living room and workspace / TV room",
        ("M007",): "in workspace / TV room near desk",
        ("M008",): "in workspace / TV room near corridor",
        ("M009",): "in corridor near washer and dryer",
        ("M010",): "in corridor between dining room and kitchen",
        ("M011",): "in corridor between kitchen and guest bathroom sink",
        ("M012",): "between dining room and kitchen",
        ("M013",): "in bathroom near sink",
        ("M014",): "in kitchen near door",
        ("M015",): "in kitchen near fridge",
        ("M016",): "in kitchen near corridor",
        ("M017",): "in guest bathroom sink",
        ("M018",): "in toilet / shower",
        ("M019",): "in corridor near workspace / TV room",
        ("M020", "M021", "M028"): "in bedroom",
        ("M022", "M023", "T001"): "in kitchen near stove",
        ("M024",): "in guest bedroom",
        ("M025",): "in walk-in closet",
        ("M026",): "in workspace / TV room",
        ("M027",): "in living room",
        ("T002",): "in corridor near guest bathroom sink",
        ("D001",): "on home entrance door",
        ("D002",): "on coat cabinet near home entrance door",
        ("D003",): "in kitchen",
    },
    'cairo': {
        ("M001",): "near work area in office",
        ("M002",): "in corridor near bedroom and guest bedroom",
        ("M003",): "in corridor near stairs",
        ("M004",): "in guest bedroom",
        ("M005", "M007", "M008", "M009", "T001"): "in bedroom",
        ("M006",): "in bedroom near its entrance",
        ("M010",): "near top of stairs",
        ("M011",): "in living room near bottom of stairs",
        ("M012", "M019", "M022", "M024"): "in kitchen",
        ("M013", "M017"): "near couch in living room",
        ("M014", "M016", "M018", "M023", "T003", "T005"): "in living room",
        ("M015",): "in living room near entrance door",
        ("M020",): "in dining area near kitchen",
        ("M021", "T004"): "near medicine cabinet in kitchen",
        ("M025",): "in other room near garage",
        ("M026",): "in laundry room near garage",
        ("M027",): "near garage door",
        ("T002",): "in office near work area",
    },
}
LOCATION = {city: {s: ctx for keys, ctx in m.items() for s in keys} for city, m in LAYOUT.items()}

VALUE = {"ON": "ON", "OFF": "OFF", "OPEN": "OPEN", "CLOSE": "CLOSE", "PRESENT": "PRESENT", "ABSENT": "ABSENT"}


def words(n):
    return _P.number_to_words(n)


def sensor_phrase(city, sensor, state):
    # Unknown sensors fall back to TDOST's default type ('Motion sensor') and an empty location.
    return (SENSOR_TYPE.get(sensor[0], "Motion sensor ") + LOCATION[city].get(sensor, '')
            + " fired with value " + VALUE.get(state, state))


def window_sentences(city, sensors, states, times):
    ts = [datetime.fromisoformat(t) for t in times]
    out = []
    for j, (s, v, t) in enumerate(zip(sensors, states, ts)):
        if j == 0:
            out.append(sensor_phrase(city, s, v)
                       + f" at {words(t.hour)} hours {words(t.minute)} minutes {'AM' if t.hour < 12 else 'PM'}")
        else:
            # As in TDOST: clock-time difference with seconds truncated (negative across midnight).
            prev = ts[j - 1]
            delta = (t.hour * 3600 + t.minute * 60 + t.second) - (prev.hour * 3600 + prev.minute * 60 + prev.second)
            out.append(f"After {words(math.floor(delta))} seconds, " + sensor_phrase(city, s, v))
    return out


def build(city, device=None, batch_size=512):
    from sentence_transformers import SentenceTransformer
    index, sentences, arrays = {}, [], {}
    for protocol, splits in SPLITS.items():
        for split, (window, name) in splits.items():
            key = f'{protocol}_{split}'
            d = load_split(city, window, name)
            L = max(len(x) for x in d['sensors'])
            idx = np.full((len(d['sensors']), L), -1, dtype=np.int32)
            for i, (s, v, t) in enumerate(zip(d['sensors'], d['states'], d['times'])):
                for j, sent in enumerate(window_sentences(city, s, v, t)):
                    if sent not in index:
                        index[sent] = len(sentences)
                        sentences.append(sent)
                    idx[i, j] = index[sent]
            arrays[key] = idx
    print(f'[{city}] {len(sentences)} unique sentences; e.g. {sentences[0]!r} / {sentences[1]!r}', flush=True)
    model = SentenceTransformer(ENCODER, device=device)
    emb = model.encode(sentences, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f'{city}.npz', emb=emb.astype(np.float16), sentences=np.array(sentences), **arrays)
    print(f'Saved {OUT / f"{city}.npz"} ({emb.shape})', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cities', nargs='+', default=['milan', 'aruba', 'cairo'])
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    for city in args.cities:
        build(city, args.device)


if __name__ == '__main__':
    main()
