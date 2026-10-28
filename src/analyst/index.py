"""Sensor-window embedding index for one home, cached on disk.

The full window history (train + val + test of the FD_60 variant) is encoded once
with the home's trained sensor encoder and stored as ``.npz`` together with a
per-window metadata table. Query sentences are encoded with the CLIP text encoder
+ trained projection and cached as well, so re-running cached analyses requires
neither the checkpoint nor a GPU.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import EMBED_CACHE_DIR, SEQUENCE_LENGTH, HomeConfig

_META_COLUMNS = [
    "sample_id", "split", "start", "end", "tod", "dow", "primary_room",
    "rooms_visited", "n_events", "label_l1", "label_l2",
]


def _file_fingerprint(paths: List[Path]) -> str:
    parts = [f"{p.resolve()}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in paths]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _window_meta(sample: dict, split: str) -> dict:
    meta = sample.get("metadata", {})
    seq = sample.get("sensor_sequence", [])
    tod = Counter(ev.get("tod_bucket", "") for ev in seq).most_common(1)
    gt = meta.get("ground_truth_labels", {})
    return {
        "sample_id": sample.get("sample_id", ""),
        "split": split,
        "start": meta.get("start_time"),
        "end": meta.get("end_time"),
        "tod": tod[0][0] if tod else "",
        "dow": seq[0].get("dow_bucket", "") if seq else "",
        "primary_room": meta.get("primary_room", ""),
        "rooms_visited": "|".join(meta.get("rooms_visited", [])),
        "n_events": meta.get("num_events", len(seq)),
        "label_l1": gt.get("primary_l1", ""),
        "label_l2": gt.get("primary_l2", ""),
    }


class SensorIndex:
    """Embeddings (N, D) + metadata DataFrame for every window of one home."""

    def __init__(self, cfg: HomeConfig, cache_dir: Path = EMBED_CACHE_DIR, device: Optional[str] = None):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir) / cfg.home
        self.device_name = device
        self._models = None
        self._query_cache_path = self.cache_dir / f"query_embeddings_{self._ckpt_tag()}.npz"
        self._query_cache: Dict[str, np.ndarray] = {}
        self._query_cache_dirty = False

        split_paths = [cfg.split_path(s) for s in cfg.splits]
        self._fingerprint = _file_fingerprint(split_paths + [cfg.checkpoint])
        self._emb_path = self.cache_dir / f"windows_{cfg.data_variant}_{self._ckpt_tag()}_{self._fingerprint}.npz"
        self._meta_path = self._emb_path.with_suffix(".meta.csv.gz")
        self._events_path = self._emb_path.with_suffix(".events.csv.gz")
        self._events: Optional[pd.DataFrame] = None

        if self._emb_path.exists() and self._meta_path.exists() and self._events_path.exists():
            self.embeddings = np.load(self._emb_path)["embeddings"].astype(np.float32)
            self.meta = pd.read_csv(self._meta_path, keep_default_na=False)
            print(f"[index] {cfg.home}: loaded {len(self.meta):,} cached window embeddings")
        else:
            self.embeddings, self.meta = self._build()
        self.meta["start"] = pd.to_datetime(self.meta["start"], format="mixed")
        self.meta["end"] = pd.to_datetime(self.meta["end"], format="mixed")
        self._load_query_cache()

    def _ckpt_tag(self) -> str:
        return self.cfg.checkpoint.parent.name

    # ------------------------------------------------------------------ models

    def _device(self):
        import torch
        if self.device_name:
            return torch.device(self.device_name)
        from utils.device_utils import get_optimal_device
        return get_optimal_device()

    def _load_models(self):
        if self._models is not None:
            return self._models
        import torch
        from alignment.model import AlignmentModel
        from evals.eval_utils import create_text_encoder_from_checkpoint

        device = self._device()
        ckpt = torch.load(str(self.cfg.checkpoint), map_location=device, weights_only=False)
        text_encoder = create_text_encoder_from_checkpoint(
            checkpoint=ckpt, device=device, data_path=str(self.cfg.split_path("train"))
        )
        if text_encoder is None:
            raise RuntimeError(f"Could not build text encoder for {self.cfg.checkpoint}")
        text_encoder.to(device).eval()
        model = AlignmentModel.load(str(self.cfg.checkpoint), device=device, vocab_path=str(self.cfg.vocab))
        model.eval()
        self._models = (device, text_encoder, model)
        return self._models

    # ------------------------------------------------------------------ build

    def _build(self):
        import torch
        from dataio.collate import create_data_loader
        from dataio.dataset import SmartHomeDataset
        from evals.eval_utils import get_global_categorical_features

        device, text_encoder, model = self._load_models()
        sensor_encoder = model.sensor_encoder
        all_embs, rows, events = [], [], set()
        for split in self.cfg.splits:
            path = self.cfg.split_path(split)
            print(f"[index] {self.cfg.home}: encoding {path.name} …")
            dataset = SmartHomeDataset(
                data_path=str(path), vocab_path=str(self.cfg.vocab),
                sequence_length=SEQUENCE_LENGTH, max_captions=1,
            )
            loader = create_data_loader(
                dataset=dataset, text_encoder=text_encoder, span_masker=None,
                vocab_sizes=model.vocab_sizes, device=device, batch_size=256,
                shuffle=False, num_workers=0, apply_mlm=False,
            )
            ids: List[str] = []
            for s in dataset.data:
                for ev in s.get("sensor_sequence", []):
                    events.add((ev["timestamp"], ev.get("sensor_id", ""), str(ev.get("event_type", "")), ev.get("room", "")))
            with torch.no_grad():
                for batch in loader:
                    input_data = {
                        "categorical_features": batch["categorical_features"],
                        "coordinates": batch["coordinates"],
                        "time_deltas": batch["time_deltas"],
                        "global_categorical_features": get_global_categorical_features(
                            sensor_encoder, batch, dataset=dataset, device=device
                        ),
                    }
                    emb = sensor_encoder.forward_clip(input_data=input_data, attention_mask=batch["mask"])
                    all_embs.append(emb.float().cpu().numpy())
                    ids.extend(str(s) for s in batch["sample_ids"])

            by_id = {s["sample_id"]: s for s in dataset.data}
            rows.extend(_window_meta(by_id[sid], split) for sid in ids)
            del dataset, loader, by_id

        emb = np.vstack(all_embs).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        meta = pd.DataFrame(rows, columns=_META_COLUMNS)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._emb_path, embeddings=emb.astype(np.float16))
        meta.to_csv(self._meta_path, index=False)
        ev = pd.DataFrame(sorted(events), columns=["timestamp", "sensor_id", "state", "room"])
        ev.to_csv(self._events_path, index=False)
        print(f"[index] {self.cfg.home}: cached {len(ev):,} unique sensor events")
        print(f"[index] {self.cfg.home}: cached {len(meta):,} windows → {self._emb_path.name}")
        return emb, meta

    # ------------------------------------------------------------------ queries

    def _load_query_cache(self):
        if self._query_cache_path.exists():
            data = np.load(self._query_cache_path, allow_pickle=False)
            texts = [str(t) for t in data["texts"]]
            self._query_cache = dict(zip(texts, data["embeddings"].astype(np.float32)))

    def _save_query_cache(self):
        if not self._query_cache_dirty:
            return
        texts = list(self._query_cache)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(self._query_cache_path, texts=np.array(texts), embeddings=np.stack([self._query_cache[t] for t in texts]))
        self._query_cache_dirty = False

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._query_cache]
        if missing:
            import torch
            device, text_encoder, _ = self._load_models()
            with torch.no_grad():
                emb = text_encoder.encode_texts_clip(missing, device).float().cpu().numpy()
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            for t, e in zip(missing, emb):
                self._query_cache[t] = e
            self._query_cache_dirty = True
            self._save_query_cache()
        return np.stack([self._query_cache[t] for t in texts])

    def similarity(self, texts: List[str]) -> np.ndarray:
        """Cosine similarity (Q, N) between query sentences and all windows."""
        return self.encode_queries(texts) @ self.embeddings.T

    @property
    def events(self) -> pd.DataFrame:
        """All unique sensor events of the home, sorted by time."""
        if self._events is None:
            ev = pd.read_csv(self._events_path, keep_default_na=False)
            ev["timestamp"] = pd.to_datetime(ev["timestamp"], format="mixed")
            self._events = ev.sort_values("timestamp", kind="stable").reset_index(drop=True)
        return self._events

    def events_between(self, start, end) -> pd.DataFrame:
        ts = self.events["timestamp"].to_numpy()
        lo = np.searchsorted(ts, np.datetime64(pd.Timestamp(start)), side="left")
        hi = np.searchsorted(ts, np.datetime64(pd.Timestamp(end)), side="right")
        return self.events.iloc[lo:hi]

    def windows_with_activation(self, sensor_ids: List[str]) -> np.ndarray:
        """Bool mask over windows: True if any ON/OPEN event of `sensor_ids` falls in [start, end]."""
        ev = self.events
        sel = ev["sensor_id"].isin(sensor_ids) & ev["state"].str.upper().isin(["ON", "OPEN"])
        ts = np.sort(ev.loc[sel, "timestamp"].to_numpy())
        lo = np.searchsorted(ts, self.meta["start"].to_numpy(), side="left")
        hi = np.searchsorted(ts, self.meta["end"].to_numpy(), side="right")
        return hi > lo

    @property
    def date_range(self) -> tuple:
        starts = self.meta["start"]
        if self.cfg.data_start is not None:
            starts = starts[starts >= self.cfg.data_start]
        return starts.min(), starts.max()
