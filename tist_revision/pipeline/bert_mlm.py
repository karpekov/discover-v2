"""
BERT MLM encoder (R2-2.9(b)): pretrained `bert-base-uncased` with sensor events as new single-word tokens (as in v1).

Window -> [CLS] tod dow e_1 dt_2 e_2 ... dt_n e_n [SEP], where
  - e_i  = `{sensor}_{state}` (room is a fixed function of the sensor, so it is implied; no coordinates),
  - dt_i = time-delta bucket between events i-1 and i, tod / dow = buckets of the first event
  (the same information as our transformer's `*_nocoord_mlm_time` configs). All are new tokens with fresh embeddings.
MLM: 15% of event tokens (80/10/10 mask/random/keep), loss on those only. Window embedding:
0.5 * CLS + 0.5 * mean over event tokens of the last layer, L2-normalised (768-d), as `cls_mean` in our transformer.

Usage:
    python -m tist_revision.pipeline.bert_mlm train --city milan --seed 1 --out trained_models/milan/bert_mlm_s1
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizerFast, get_linear_schedule_with_warmup

from .encoding import get_device
from .mapping_eval import DATA_ROOT

BASE = 'bert-base-uncased'


def event_word(e):
    return f"ev_{e['sensor_id']}_{e['event_type']}".lower()


def window_words(seq):
    words = [f"tod_{seq[0].get('tod_bucket', 'unk')}".lower(), f"dow_{seq[0].get('dow_bucket', 'unk')}".lower()]
    for i, e in enumerate(seq):
        if i:
            words.append(f"dt_{e.get('time_delta_bucket', 'unk')}".lower())
        words.append(event_word(e))
    return words


def load_words(path):
    with open(path) as f:
        return [window_words(s['sensor_sequence']) for s in json.load(f)['samples']]


class Codec:
    """Maps window words to BERT ids; new words are added to the tokenizer vocabulary."""

    def __init__(self, tokenizer):
        self.tok = tokenizer

    @classmethod
    def build(cls, windows):
        tok = BertTokenizerFast.from_pretrained(BASE)
        tok.add_tokens(sorted({w for ws in windows for w in ws}))
        return cls(tok)

    def encode(self, windows):
        """(ids, attention mask, event-token mask) as padded LongTensors."""
        unk = self.tok.unk_token_id
        vocab = self.tok.get_vocab()
        L = max(len(ws) for ws in windows) + 2
        ids = torch.full((len(windows), L), self.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(windows), L), dtype=torch.long)
        ev = torch.zeros((len(windows), L), dtype=torch.bool)
        for i, ws in enumerate(windows):
            row = [self.tok.cls_token_id] + [vocab.get(w, unk) for w in ws] + [self.tok.sep_token_id]
            ids[i, :len(row)] = torch.tensor(row)
            att[i, :len(row)] = 1
            ev[i, 1:len(ws) + 1] = torch.tensor([w.startswith('ev_') for w in ws])
        return ids, att, ev


def mask_events(ids, ev, codec, prob=0.15):
    labels = ids.clone()
    chosen = ev & (torch.rand(ids.shape) < prob)
    labels[~chosen] = -100
    ids = ids.clone()
    r = torch.rand(ids.shape)
    ids[chosen & (r < 0.8)] = codec.tok.mask_token_id
    rand = chosen & (r >= 0.8) & (r < 0.9)
    ids[rand] = torch.randint(len(codec.tok), (int(rand.sum()),))
    return ids, labels


def train(city, out, seed, epochs=20, batch_size=128, lr=5e-5, max_steps=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = get_device()
    tr_words = load_words(DATA_ROOT / city / 'FL_20' / 'train.json')
    va_words = load_words(DATA_ROOT / city / 'FL_20' / 'val.json')
    codec = Codec.build(tr_words)
    tr, va = codec.encode(tr_words), codec.encode(va_words)
    model = BertForMaskedLM.from_pretrained(BASE)
    model.resize_token_embeddings(len(codec.tok))
    model.to(device)
    steps = max_steps or epochs * math.ceil(len(tr[0]) / batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    best, step, t0 = float('inf'), 0, time.time()
    g = torch.Generator().manual_seed(seed)
    while step < steps:
        model.train()
        for b in torch.randperm(len(tr[0]), generator=g).split(batch_size):
            ids, labels = mask_events(tr[0][b], tr[2][b], codec)
            with torch.autocast('cuda', enabled=use_amp):
                loss = model(input_ids=ids.to(device), attention_mask=tr[1][b].to(device), labels=labels.to(device)).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step >= steps:
                break
        val = evaluate_loss(model, va, codec, device, seed)
        print(f'[{city} s{seed}] step {step}/{steps} train {loss.item():.3f} val {val:.3f} ({time.time() - t0:.0f}s)', flush=True)
        if val < best:
            best = val
            model.save_pretrained(out)
            codec.tok.save_pretrained(out)
    (out / 'meta.json').write_text(json.dumps({'city': city, 'seed': seed, 'steps': steps, 'best_val_loss': best,
                                               'base': BASE, 'batch_size': batch_size, 'lr': lr}))


@torch.no_grad()
def evaluate_loss(model, data, codec, device, seed, batch_size=512):
    model.eval()
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    tot, n = 0.0, 0
    for s in range(0, len(data[0]), batch_size):
        ids, labels = mask_events(data[0][s:s + batch_size], data[2][s:s + batch_size], codec)
        k = int((labels != -100).sum())
        if k:
            tot += model(input_ids=ids.to(device), attention_mask=data[1][s:s + batch_size].to(device),
                         labels=labels.to(device)).loss.item() * k
            n += k
    torch.random.set_rng_state(state)
    return tot / max(n, 1)


@torch.no_grad()
def encode(model_dir, data_path, device=None, batch_size=512):
    """L2-normalised window embeddings (0.5 CLS + 0.5 mean of event tokens) for one split."""
    device = device or get_device()
    codec = Codec(BertTokenizerFast.from_pretrained(model_dir))
    bert = BertForMaskedLM.from_pretrained(model_dir).bert.to(device).eval()
    ids, att, ev = codec.encode(load_words(data_path))
    out = []
    for s in range(0, len(ids), batch_size):
        h = bert(input_ids=ids[s:s + batch_size].to(device), attention_mask=att[s:s + batch_size].to(device)).last_hidden_state
        m = ev[s:s + batch_size].to(device).unsqueeze(-1).float()
        emb = 0.5 * h[:, 0] + 0.5 * (h * m).sum(1) / m.sum(1).clamp_min(1)
        out.append(torch.nn.functional.normalize(emb, dim=-1).float().cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['train'])
    ap.add_argument('--city', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--max_steps', type=int, default=None, help='smoke tests only')
    args = ap.parse_args()
    train(args.city, args.out, args.seed, args.epochs, args.batch_size, args.lr, args.max_steps)


if __name__ == '__main__':
    main()
