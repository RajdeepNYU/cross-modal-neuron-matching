"""Greedy coordinate-descent tuning of the Cellpose decoding parameters on held-out maps (maximises mean region PQ)."""
from __future__ import annotations

import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .config import ROOT
from .metrics import seg_match
from .pipeline import load_instances
from .prep import RegionData, load_index

CP_SPACE = {
    "cellprob": [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    "flow": [0.2, 0.4, 0.6, 1.0, 2.0, 5.0],
    "min_size": [10, 15, 20, 30, 40, 60],
    "min_conf": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6],
    "max_area": [300, 450, 600, 900],
}


def _pq_job(args):
    sid, mod, params = args
    lab, _ = load_instances(sid, "oof", mod, params)
    return seg_match(np.asarray(RegionData(sid).arr(f"{mod}_lab")).astype(np.int32), lab)["pq"]


def tune_seg(cfg: dict, modalities=("iv", "ex"), workers: int = 12, passes: int = 2, out_cfg: str | None = None) -> dict:
    idx = load_index()
    sids = list(idx[idx.split == "train"].sample_id)
    new_cfg = copy.deepcopy(cfg)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        def pq(mod, params):
            return float(np.mean(list(pool.map(_pq_job, [(s, mod, params) for s in sids], chunksize=2))))

        for mod in modalities:
            cur = dict(new_cfg["seg"][mod])
            best = pq(mod, cur)
            print(f"[tune_seg] {mod}: start PQ {best:.4f}", flush=True)
            for _ in range(passes):
                improved = False
                for name, values in CP_SPACE.items():
                    for v in values:
                        if v == cur.get(name):
                            continue
                        trial = dict(cur, **{name: v})
                        r = pq(mod, trial)
                        if r > best + 1e-4:
                            best, cur, improved = r, trial, True
                            print(f"  [{mod}] {name}={v}  PQ {best:.4f}", flush=True)
                if not improved:
                    break
            new_cfg["seg"][mod] = cur
            print(f"[tune_seg] {mod}: final PQ {best:.4f} -> {cur}")
    if out_cfg:
        out = Path(out_cfg) if Path(out_cfg).is_absolute() else ROOT / out_cfg
        json.dump(new_cfg, open(out, "w"), indent=1)
        print(f"[tune_seg] wrote {out} ({time.time() - t0:.0f}s)")
    return new_cfg
