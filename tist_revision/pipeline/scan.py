"""
SCAN clustering on a pre-trained encoder (DISCOVER revision, replaces src/training/train_scan.py here).

  1. Encode continuous FL_20 train windows with the frozen encoder (inputs as in MLM pre-training)
     and mine each window's k nearest neighbours (cosine).
  2. Train `num_heads` clustering heads with the SCAN loss (consistency with a random neighbour
     - entropy_weight * entropy of the mean assignment). While frozen, heads train directly on
     cached embeddings (cheap, runs on CPU / laptop). After `freeze_epochs` the encoder is
     unfrozen and trained with `encoder_lr` (needs a GPU). Neighbours stay fixed.
  3. Keep the head with the lowest final loss and score it with the shared evaluator
     (pre-segmented FL_20_p pool/test, m-sweep, both variants). If the encoder was updated, its
     new embeddings are also scored (kNN purity, k-means, linear probes).

Usage:
    # frozen encoder (head only)
    python -m tist_revision.pipeline.scan --city milan --run_name frozen_k20
    # frozen for 10 epochs, then encoder fine-tuned at lr 1e-5 (GPU)
    python -m tist_revision.pipeline.scan --city milan --run_name unfreeze10_k20 --freeze_epochs 10 --encoder_lr 1e-5
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .embedding_diagnostics import diagnose, save_rows, split_paths
from .encoding import batch_inputs, encode, encoder_fields, get_device, load_model, load_vocab, tensorize
from .mapping_eval import DATA_ROOT, DEFAULT_M_VALUES, RESULTS_DIR, evaluate_clustering, load_protocol, save_assignments

OUT_ROOT = RESULTS_DIR / 'stage3_scan'
MODEL_OUT = Path('trained_models')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_head(d_model, k, dropout):
    """Same architecture as SCANClusteringModel.clustering_head."""
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, d_model // 2), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(d_model // 2, k))


def scan_loss(anchor_logits, neighbor_logits, entropy_weight):
    pa, pn = anchor_logits.softmax(1), neighbor_logits.softmax(1)
    consistency = -torch.log((pa * pn).sum(1).clamp_min(1e-8)).mean()
    mean_p = pa.mean(0).clamp_min(1e-8)
    entropy = -(mean_p * mean_p.log()).sum()
    return consistency - entropy_weight * entropy, consistency, entropy


@torch.no_grad()
def mine_neighbors(E: torch.Tensor, k: int, chunk: int = 4096) -> torch.Tensor:
    E = nn.functional.normalize(E, dim=1)
    out = []
    for s in range(0, len(E), chunk):
        sim = E[s:s + chunk] @ E.T
        sim[torch.arange(sim.shape[0]), torch.arange(s, s + sim.shape[0])] = -2
        out.append(sim.topk(k, dim=1).indices.cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--city', required=True)
    ap.add_argument('--encoder_dir', default='trained_models/{city}/{city}_fl20_seq_discover_v1_mlm_only')
    ap.add_argument('--run_name', required=True)
    ap.add_argument('--num_clusters', type=int, default=20)
    ap.add_argument('--num_heads', type=int, default=5)
    ap.add_argument('--num_neighbors', type=int, default=20)
    ap.add_argument('--entropy_weight', type=float, default=2.0)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--freeze_epochs', type=int, default=None, help='Epochs with frozen encoder (default: all)')
    ap.add_argument('--encoder_lr', type=float, default=1e-5)
    ap.add_argument('--lr', type=float, default=1e-3, help='Head learning rate (1e-3 beat 1e-4 on Cairo, frozen)')
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_boot', type=int, default=500)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--wandb_project', default='discover-v2-dv1-scan')
    args = ap.parse_args()
    freeze_epochs = args.epochs if args.freeze_epochs is None else args.freeze_epochs
    city = args.city
    encoder_dir = Path(args.encoder_dir.format(city=city))
    run = f'{encoder_dir.name}__{args.run_name}'
    set_seed(args.seed)
    device = get_device()
    t_start = time.time()

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=f'{city}_{run}', config=vars(args) | {'encoder': str(encoder_dir)},
                        dir='logs/wandb', tags=['tist_revision', city])

    vocab_path = DATA_ROOT / city / 'FL_20' / 'vocab.json'
    vocab = load_vocab(vocab_path)
    model = load_model(encoder_dir, vocab_path, device, num_clusters=args.num_clusters)
    cat_fields, global_fields = encoder_fields(model, vocab)
    T = {k: tensorize(p, vocab, cat_fields, global_fields) for k, p in split_paths(city).items()}
    E_fit = torch.from_numpy(encode(model, T['fit'], device)[0]).to(device)
    nbrs = mine_neighbors(E_fit, args.num_neighbors)
    print(f'[{city}] {run}: {len(E_fit):,} fit windows encoded, neighbours mined ({time.time() - t_start:.0f}s)')

    heads = nn.ModuleList([make_head(model.d_model, args.num_clusters, args.dropout)
                           for _ in range(args.num_heads)]).to(device)
    encoder = model.sensor_encoder
    for p in encoder.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(heads.parameters(), lr=args.lr, weight_decay=1e-4)

    N = len(E_fit)
    for epoch in range(args.epochs):
        frozen = epoch < freeze_epochs
        if not frozen and epoch == freeze_epochs:
            for p in encoder.parameters():
                p.requires_grad = True
            opt.add_param_group({'params': list(encoder.parameters()), 'lr': args.encoder_lr})
            print(f'  epoch {epoch}: encoder unfrozen (lr={args.encoder_lr})')
        heads.train()
        encoder.train(not frozen)
        perm = torch.randperm(N)
        sums = torch.zeros(args.num_heads, 3)
        n_batches = 0
        for s in range(0, N, args.batch_size):
            a = perm[s:s + args.batch_size]
            nb = nbrs[a, torch.randint(0, args.num_neighbors, (len(a),))]
            if frozen:
                fa, fn = E_fit[a.to(device)], E_fit[nb.to(device)]
            else:
                both = torch.cat([a, nb])
                inp, mask = batch_inputs(T['fit'], both, device)
                emb = encoder(input_data=inp, attention_mask=mask).embeddings
                fa, fn = emb[:len(a)], emb[len(a):]
            losses = [scan_loss(h(fa), h(fn), args.entropy_weight) for h in heads]
            total = sum(l[0] for l in losses)
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 1.0)
            opt.step()
            sums += torch.tensor([[x.item() for x in l] for l in losses])
            n_batches += 1
        avg = sums / n_batches
        best = int(avg[:, 0].argmin())
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f'  epoch {epoch:3d} {"frozen" if frozen else "unfrozen"} loss={avg[best, 0]:.4f} '
                  f'cons={avg[best, 1]:.4f} ent={avg[best, 2]:.4f} (head {best}) {time.time() - t_start:.0f}s')
        if wb:
            wb.log({'epoch': epoch, 'loss': avg[best, 0].item(), 'consistency': avg[best, 1].item(),
                    'entropy': avg[best, 2].item(), 'best_head': best})

    model.clustering_head.load_state_dict(heads[best].state_dict())
    encoder_updated = freeze_epochs < args.epochs
    enc = {k: encode(model, T[k], device) for k in ('pool', 'test')}
    if encoder_updated:
        enc['fit'] = encode(model, T['fit'], device)
    fit_clusters = (model.clustering_head.eval()(E_fit).argmax(1).cpu().numpy() if not encoder_updated
                    else enc['fit'][1].argmax(1))
    sizes = np.bincount(fit_clusters, minlength=args.num_clusters)

    out_dir = MODEL_OUT / city / 'tist_scan' / run
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'args': vars(args), 'encoder_dir': str(encoder_dir),
                'best_head': best}, out_dir / 'best_model.pt')

    _, pool, test = load_protocol(city)
    cols = ['label', 'day']
    (_, P_pool), (_, P_test) = enc['pool'], enc['test']
    save_assignments(f'{city}__{run}__scan_head', P_pool.argmax(1), P_pool.max(1), P_test.argmax(1), P_test.max(1))
    res = evaluate_clustering(pool[cols].assign(cluster=P_pool.argmax(1), conf=P_pool.max(1)),
                              test[cols].assign(cluster=P_test.argmax(1), conf=P_test.max(1)),
                              m_values=DEFAULT_M_VALUES, n_boot=args.n_boot, seed=args.seed)
    meta = {'city': city, 'encoder': encoder_dir.name, 'run': args.run_name, 'method': 'scan',
            'num_clusters': args.num_clusters, 'num_neighbors': args.num_neighbors,
            'freeze_epochs': freeze_epochs, 'epochs': args.epochs, 'encoder_lr': args.encoder_lr if encoder_updated else 0,
            'entropy_weight': args.entropy_weight, 'seed': args.seed, 'active_clusters': int((sizes > 0).sum()),
            'largest_cluster_share': float(sizes.max() / sizes.sum()), 'minutes': (time.time() - t_start) / 60}
    res = res.assign(**meta)

    # One file per run so concurrent SLURM tasks never write the same CSV (summarize.py merges them).
    run_dir = OUT_ROOT / 'runs' / f'{city}__{run}'
    run_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(run_dir / 'scan_head.csv', index=False)

    if encoder_updated:
        rows = {'knn': [], 'clustering': [], 'probe': []}
        diagnose(f'{run}__encoder', enc['fit'][0], enc['pool'][0], enc['test'][0], pool[cols], test[cols], rows,
                 city, metric='cosine', n_boot=args.n_boot, seed=args.seed)
        save_rows(rows, run_dir)

    r = res
    for variant in ('all', 'excl_other'):
        s = r[r.variant == variant].set_index('m')
        print(f'  {variant:10s} ' + '  '.join(f'm={m}: {s.loc[m, "f1_weighted"]:.3f}/{s.loc[m, "f1_macro"]:.3f}'
                                              for m in s.index))
    print(f'  active clusters {meta["active_clusters"]}/{args.num_clusters}, largest {meta["largest_cluster_share"]:.1%}, '
          f'{meta["minutes"]:.1f} min')
    if wb:
        wb.summary.update({f'{v}_m{m}_{k}': r[(r.variant == v) & (r.m == m)][k].iloc[0]
                           for v in ('all', 'excl_other') for m in ('5', 'all') for k in ('f1_weighted', 'f1_macro')})
        wb.finish()
    with open(out_dir / 'run.json', 'w') as f:
        json.dump(meta | {'args': vars(args)}, f, indent=2)


if __name__ == '__main__':
    main()
