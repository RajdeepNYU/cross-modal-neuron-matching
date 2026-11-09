"""Decode the dataset once into per-region .npy caches and build the region index (subject-level folds)."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import P
from .data import Dataset, centroids


def region_dir(sid: str) -> Path:
    return P.prep / sid


def _prep_one(args):
    root, sid, split = args
    ds = Dataset(root)
    reg = [r for r in (ds.train_regions() if split == "train" else ds.test_regions()) if r.sample_id == sid][0]
    d = region_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    iv, ex = ds.load_images(reg)
    np.save(d / "iv_raw.npy", iv)
    np.save(d / "ex_raw.npy", ex)
    meta = dict(sample_id=sid, split=split, subject=reg.subject, iv_shape=list(iv.shape), ex_shape=list(ex.shape))
    if split == "train":
        g = ds.load_gt(reg, iv.shape, ex.shape)
        np.save(d / "iv_lab.npy", g["iv_lab"].astype(np.uint16))
        np.save(d / "ex_lab.npy", g["ex_lab"].astype(np.uint16))
        np.save(d / "pairs.npy", g["pairs"])
        np.save(d / "iv_cent.npy", centroids(g["iv_lab"]))
        np.save(d / "ex_cent.npy", centroids(g["ex_lab"]))
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f)
    return meta


def prepare(cfg: dict, workers: int = 8, force: bool = False) -> pd.DataFrame:
    P.mk()
    ds = Dataset(cfg["data_root"])
    jobs = [(cfg["data_root"], r.sample_id, r.split) for r in ds.train_regions() + ds.test_regions()
            if force or not (region_dir(r.sample_id) / "meta.json").exists()]
    print(f"[prepare] {len(jobs)} regions to build")
    if jobs:
        with ProcessPoolExecutor(max(workers, 1)) as ex:
            list(ex.map(_prep_one, jobs))
    return build_index(cfg)


def build_index(cfg: dict) -> pd.DataFrame:
    ds = Dataset(cfg["data_root"])
    train_subjects = sorted({r.subject for r in ds.train_regions()})
    rows = []
    for r in ds.train_regions() + ds.test_regions():
        m = json.load(open(region_dir(r.sample_id) / "meta.json"))
        m["fold"] = train_subjects.index(r.subject) if r.split == "train" else -1
        rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(P.prep / "index.csv", index=False)
    return df


def load_index() -> pd.DataFrame:
    p = P.prep / "index.csv"
    if not p.exists():
        raise FileNotFoundError("Run `python run_pipeline.py --stage prepare` first.")
    return pd.read_csv(p)


class RegionData:
    """Memory-mapped access to the cached arrays of one region."""

    def __init__(self, sid: str):
        self.sid = sid
        self.d = region_dir(sid)
        self.meta = json.load(open(self.d / "meta.json"))
        self._c = {}

    def arr(self, name: str):
        if name not in self._c:
            self._c[name] = np.load(self.d / f"{name}.npy", mmap_mode="r")
        return self._c[name]
