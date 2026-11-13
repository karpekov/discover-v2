"""
Encode the protocol splits with a trained encoder into `artifacts/embeddings/{city}_{name}_embeddings.npz`
(keys `fit` = continuous FL_20 train, which is also the annotation pool, and `test` = pre-segmented FL_20_p test),
read by `rare_classes.representations` / `frozen_deep.py`.

Usage:
    python -m tist_revision.pipeline.encode_embeddings --city milan --kind tf \
        --model_dir trained_models/milan/tf_nocoord_mlm_time_s1 --name tf_s1
"""
import argparse

import numpy as np

from .encoding import encode, encoder_fields, get_device, load_model, load_vocab, tensorize
from .mapping_eval import ARTIFACTS_DIR, DATA_ROOT

SPLITS = {'fit': ('FL_20', 'train'), 'test': ('FL_20_p', 'test')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--city', required=True)
    ap.add_argument('--kind', choices=['tf', 'bert'], required=True)
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--name', required=True)
    args = ap.parse_args()
    device = get_device()
    paths = {k: DATA_ROOT / args.city / w / f'{s}.json' for k, (w, s) in SPLITS.items()}
    if args.kind == 'tf':
        vocab = load_vocab(DATA_ROOT / args.city / 'FL_20' / 'vocab.json')
        model = load_model(args.model_dir, DATA_ROOT / args.city / 'FL_20' / 'vocab.json', device)
        cat_fields, global_fields = encoder_fields(model, vocab)
        if model.alignment_config.encoder.get('metadata', {}).get('use_coordinates', True):
            raise ValueError(f'{args.model_dir} uses x-y coordinates; the revision only uses no-coordinate encoders')
        out = {k: encode(model, tensorize(p, vocab, cat_fields, global_fields), device)[0] for k, p in paths.items()}
    else:
        from .bert_mlm import encode as bert_encode
        out = {k: bert_encode(args.model_dir, p, device) for k, p in paths.items()}
    path = ARTIFACTS_DIR / 'embeddings' / f'{args.city}_{args.name}_embeddings.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v.astype(np.float16) for k, v in out.items()})
    print(f'Saved {path}: ' + ', '.join(f'{k} {v.shape}' for k, v in out.items()))


if __name__ == '__main__':
    main()
