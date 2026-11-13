"""
Train and test one supervised baseline for one house and seed, over label levels and label budgets.

Training follows the TDOST code (`HAR/evaluate_with_classifier.py`, `DeepCASAS/arguments.py`): Adam + StepLR(10, 0.8),
cross-entropy, and the epoch with the best validation weighted F1 is kept. Validation (and test) predictions never
output No_Activity (`predict_activity`), and validation F1 is computed on activity windows only, as on test.
Every run trains for at least `MIN_STEPS` optimiser steps (small budgets would otherwise get a handful of steps).
Hyperparameters: DeepCASAS 75 epochs, lr 1e-3, batch 64; TDOST 50 epochs, lr 1e-3, wd 0, batch 64 (its tuned
v1-1 + all-distilroberta-v1 ConvLSTM setting for Milan, Aruba and Cairo); TreeCNN 25 epochs, lr 2e-4, L2 4e-4,
batch 128 (Cao et al. 2020), no scheduler.

Outputs `results/baselines/{protocol}/{method}/{city}_{level}_{budget}_s{seed}.npz` (test labels, predictions, days,
classes, validation curve). Scored by `baselines/score.py`.

Usage:
    python -m tist_revision.baselines.run --method deepcasas --city milan --seed 1
    python -m tist_revision.baselines.run --method tdost --city milan --protocol preseg --levels l2 --budgets all
"""
import argparse
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from ..pipeline.encoding import get_device
from ..pipeline.mapping_eval import ARTIFACTS_DIR, NO_ACTIVITY, RESULTS_DIR, predict_activity
from .data import budget_indices, labels, load
from .models import DeepCASAS, TDOSTConvLSTM, TreeCNN

OUT = RESULTS_DIR / 'baselines'
LENGTH = 20
MIN_STEPS = 500
HPARAMS = {'deepcasas': dict(epochs=75, lr=1e-3, wd=0.0, batch=64, step_lr=True),
           'tdost': dict(epochs=50, lr=1e-3, wd=0.0, batch=64, step_lr=True),
           'treecnn': dict(epochs=25, lr=2e-4, wd=4e-4, batch=128, step_lr=False)}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def token_inputs(splits):
    """sensor+value token ids, pre-padded to LENGTH (0 = padding); vocabulary from the train split."""
    vocab = {}
    for s, v in zip(splits['train']['sensors'], splits['train']['states']):
        for tok in zip(s, v):
            vocab.setdefault(tok, len(vocab) + 2)  # 1 = unknown
    out = {}
    for name, d in splits.items():
        X = np.zeros((len(d['sensors']), LENGTH), dtype=np.int64)
        for i, (s, v) in enumerate(zip(d['sensors'], d['states'])):
            ids = [vocab.get(t, 1) for t in zip(s, v)][-LENGTH:]
            X[i, LENGTH - len(ids):] = ids
        out[name] = torch.from_numpy(X)
    return out, len(vocab) + 2


def tdost_inputs(city, protocol):
    z = np.load(ARTIFACTS_DIR / 'tdost' / f'{city}.npz')
    emb = torch.from_numpy(np.vstack([z['emb'].astype(np.float32), np.zeros((1, z['emb'].shape[1]), np.float32)]))
    pad = len(emb) - 1
    out = {}
    for split in ('train', 'val', 'test'):
        idx = z[f'{protocol}_{split}'].astype(np.int64)
        idx[idx < 0] = pad
        out[split] = torch.from_numpy(idx)
    return out, emb


def predict(model, X, lookup, device, classes, batch=1024):
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = X[s:s + batch].to(device)
            probs.append(torch.softmax(model(lookup(xb)), 1).float().cpu().numpy())
    return predict_activity(np.vstack(probs), classes)


def weighted_f1_activity(y, pred):
    keep = y != NO_ACTIVITY
    return f1_score(y[keep], pred[keep], average='weighted', zero_division=0) if keep.any() else 0.0


def fit_eval(method, X, y, tr_idx, va_X, va_y, te_X, lookup, model_fn, device, seed):
    hp = HPARAMS[method]
    classes = np.unique(y[tr_idx])
    code = {c: i for i, c in enumerate(classes)}
    yt = torch.tensor([code[c] for c in y[tr_idx]])
    Xt = X[tr_idx]
    set_seed(seed)
    model = model_fn(len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.8) if hp['step_lr'] else None
    steps_per_epoch = math.ceil(len(tr_idx) / hp['batch'])
    epochs = max(hp['epochs'], math.ceil(MIN_STEPS / steps_per_epoch))
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    best, best_state, curve = -1.0, None, []
    for epoch in range(epochs):
        model.train()
        for b in torch.randperm(len(tr_idx), generator=g).split(hp['batch']):
            loss = loss_fn(model(lookup(Xt[b].to(device))), yt[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
        f1 = weighted_f1_activity(va_y, predict(model, va_X, lookup, device, classes))
        curve.append(f1)
        if f1 > best:
            best, best_state = f1, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(model, te_X, lookup, device, classes), classes, np.array(curve), epochs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', choices=list(HPARAMS), required=True)
    ap.add_argument('--city', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--protocol', choices=['final', 'preseg'], default='final')
    ap.add_argument('--levels', nargs='+', default=['l1', 'l2'])
    ap.add_argument('--budgets', nargs='+', default=['all', '100', '250'])
    args = ap.parse_args()
    device = get_device()
    splits = load(args.city, args.protocol)
    if args.method == 'tdost':
        X, emb = tdost_inputs(args.city, args.protocol)
        emb = emb.to(device)
        lookup = lambda xb: emb[xb]  # noqa: E731
        model_fn = lambda k: TDOSTConvLSTM(k)  # noqa: E731
    else:
        X, vocab_size = token_inputs(splits)
        lookup = lambda xb: xb  # noqa: E731
        model_fn = (lambda k: DeepCASAS(vocab_size, k)) if args.method == 'deepcasas' else (lambda k: TreeCNN(vocab_size, k, LENGTH))
    out_dir = OUT / args.protocol / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    for level in args.levels:
        y = {k: labels(d['label_raw'], level) for k, d in splits.items()}
        for budget in args.budgets:
            t0 = time.time()
            tr_idx, va_idx = budget_indices(y['train'], budget, args.seed)
            va_X, va_y = (X['val'], y['val']) if va_idx is None else (X['train'][va_idx], y['train'][va_idx])
            pred, classes, curve, epochs = fit_eval(args.method, X['train'], y['train'], tr_idx, va_X, va_y, X['test'],
                                                    lookup, model_fn, device, args.seed)
            n_labels = len(tr_idx) + (0 if va_idx is None else len(va_idx))
            np.savez_compressed(out_dir / f'{args.city}_{level}_{budget}_s{args.seed}.npz', y_test=y['test'], pred=pred,
                                day=splits['test']['day'], classes=classes, val_curve=curve, n_labels=n_labels)
            wf1 = f1_score(y['test'], pred, average='weighted', zero_division=0)
            print(f'[{args.method} {args.city} {args.protocol} {level} {budget} s{args.seed}] {n_labels} labels, '
                  f'{epochs} epochs, best val {curve.max():.3f} @ {curve.argmax()}, test weighted F1 {wf1:.3f} '
                  f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
