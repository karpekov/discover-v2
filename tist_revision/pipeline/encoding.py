"""
Encoder loading and input encoding shared by the diagnostics and SCAN training.

Inputs are built exactly as in MLM pre-training (src/alignment/dataset.py + collate): raw events,
global context tokens (tod/dow) taken from the first event, raw time deltas between events,
coordinates (all zero for CASAS events), padding mask.
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[2] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.scan_model import SCANClusteringModel  # noqa: E402

# Vocab field → event key, as in alignment/dataset.py.
FIELD_TO_EVENT_KEY = {'sensor': 'sensor_id', 'state': 'event_type', 'room_id': 'room'}


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _lookup(vocab_field, value):
    return vocab_field[value] if value in vocab_field else vocab_field.get('UNK', 0)


def encoder_fields(model: SCANClusteringModel, vocab: dict):
    meta = model.alignment_config.encoder.get('metadata', {})
    cat_fields = [f for f in meta.get('categorical_fields', ['sensor', 'state', 'room_id']) if f in vocab]
    global_fields = [f for f in meta.get('global_categorical_fields', []) if f in vocab]
    return cat_fields, global_fields


def tensorize(data_path: Path, vocab: dict, cat_fields, global_fields) -> Dict[str, torch.Tensor]:
    """Encode every window of a split into padded CPU tensors (padded to the longest window)."""
    with open(data_path) as f:
        samples = json.load(f)['samples']
    N, L = len(samples), max(len(s['sensor_sequence']) for s in samples)
    out = {f'cat.{f}': torch.zeros((N, L), dtype=torch.long) for f in cat_fields}
    out.update({f'glob.{f}': torch.zeros(N, dtype=torch.long) for f in global_fields})
    deltas = np.zeros((N, L), dtype=np.float32)
    mask = np.zeros((N, L), dtype=bool)
    for i, s in enumerate(samples):
        seq = s['sensor_sequence']
        for f in global_fields:
            out[f'glob.{f}'][i] = _lookup(vocab[f], seq[0].get(FIELD_TO_EVENT_KEY.get(f, f), 'UNK'))
        for f in cat_fields:
            key = FIELD_TO_EVENT_KEY.get(f, f)
            out[f'cat.{f}'][i, :len(seq)] = torch.tensor([_lookup(vocab[f], e.get(key, 'UNK')) for e in seq])
        ts = [datetime.fromisoformat(e['timestamp']) for e in seq]
        deltas[i, 1:len(seq)] = [(b - a).total_seconds() for a, b in zip(ts[:-1], ts[1:])]
        mask[i, :len(seq)] = True
    out['time_deltas'] = torch.from_numpy(deltas)
    out['mask'] = torch.from_numpy(mask)
    out['coordinates'] = torch.zeros((N, L, 2))
    return out


def batch_inputs(T: Dict[str, torch.Tensor], idx, device):
    """(input_data, attention_mask) for the windows `idx` of a tensorized split."""
    input_data = {
        'categorical_features': {k[4:]: v[idx].to(device) for k, v in T.items() if k.startswith('cat.')},
        'global_categorical_features': {k[5:]: v[idx].to(device) for k, v in T.items() if k.startswith('glob.')},
        'coordinates': T['coordinates'][idx].to(device),
        'time_deltas': T['time_deltas'][idx].to(device),
    }
    return input_data, T['mask'][idx].to(device)


def load_model(encoder_dir: Path, vocab_path: Path, device, num_clusters: int = 20,
               scan_ckpt: Optional[Path] = None) -> SCANClusteringModel:
    """SCAN model around a pre-trained encoder; optionally load a SCAN checkpoint's weights."""
    model = SCANClusteringModel(str(encoder_dir), num_clusters=num_clusters, vocab_path=str(vocab_path), device=device)
    if scan_ckpt is not None:
        state = torch.load(scan_ckpt, map_location='cpu', weights_only=False)['model_state_dict']
        model.load_state_dict(state)
    return model.to(device)


@torch.no_grad()
def encode(model: SCANClusteringModel, T: Dict[str, torch.Tensor], device, batch_size: int = 512):
    """(embeddings, cluster-head probabilities) for every window of a tensorized split."""
    model.eval()
    embs, probs = [], []
    N = len(T['mask'])
    for start in range(0, N, batch_size):
        input_data, mask = batch_inputs(T, slice(start, min(start + batch_size, N)), device)
        logits, emb = model(input_data, mask, return_embeddings=True)
        embs.append(emb.float().cpu().numpy())
        probs.append(torch.softmax(logits, dim=1).float().cpu().numpy())
    return np.vstack(embs), np.vstack(probs)


def load_vocab(vocab_path: Path) -> dict:
    with open(vocab_path) as f:
        return json.load(f)
