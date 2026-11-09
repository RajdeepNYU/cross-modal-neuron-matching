"""Cellpose-SAM inference: cached flow fields / cell probabilities and their decoding into instance masks."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

from .config import P
from .prep import RegionData, load_index


def cp_file(kind: str, sid: str, mod: str, tag: str) -> Path:
    d = P.probs / tag / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sid}__{mod}.cp.npz"


def load_cp_model(path: str):
    from cellpose import models

    return models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=str(path))


def cp_norm(raw: np.ndarray) -> np.ndarray:
    """Whole-region percentile normalisation, identical at training and inference."""
    x = raw.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    return (x - lo) / max(float(hi - lo), 1e-6)


def cp_forward(model, img: np.ndarray, diam: float):
    """(dP float16 (2,H,W), cellprob float16 (H,W)) at native resolution."""
    _, flows, _ = model.eval(cp_norm(img), diameter=diam, compute_masks=False, batch_size=8, bsize=256, tile_overlap=0.1,
                             augment=False, normalize=False)
    return flows[1].astype(np.float16), flows[2].astype(np.float16)


def _filter(lab, conf_map, min_area, max_area, min_conf):
    n = int(lab.max())
    if n == 0:
        return lab
    area = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64)
    conf = np.bincount(lab.ravel(), weights=conf_map.ravel(), minlength=n + 1) / np.maximum(area, 1)
    keep = (area >= min_area) & (area <= max_area) & (conf >= min_conf)
    keep[0] = False
    return np.where(keep[lab], lab, 0).astype(np.int32)


def _relabel(lab, conf_map):
    ids = np.unique(lab)
    ids = ids[ids > 0]
    remap = np.zeros(int(lab.max()) + 1, np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    lab = remap[lab]
    n = len(ids)
    if not n:
        return lab, dict(area=np.zeros(0), conf=np.zeros(0))
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64)
    conf = np.bincount(lab.ravel(), weights=conf_map.ravel().astype(np.float64), minlength=n + 1)[1:] / np.maximum(area, 1)
    return lab, dict(area=area, conf=conf)


def grow_area(lab: np.ndarray, prob: np.ndarray, scale: float, max_px: float = 3.0) -> np.ndarray:
    """Grow every instance to `scale` x its area by claiming the most probable background pixels within max_px
    (probability minus a small distance penalty); instances never overlap and ids are preserved."""
    n = int(lab.max())
    if n == 0:
        return lab
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    bg = lab == 0
    dist, (iy, ix) = ndi.distance_transform_edt(bg, return_indices=True)
    ys, xs = np.nonzero(bg & (dist <= max_px))
    if len(ys) == 0:
        return lab
    owner = lab[iy[ys, xs], ix[ys, xs]]
    order = np.lexsort((-(prob[ys, xs] - 0.1 * dist[ys, xs]), owner))
    ys, xs, owner = ys[order], xs[order], owner[order]
    need = np.maximum(np.round(areas * (scale - 1.0)).astype(np.int64), 0)
    rank = np.arange(len(owner)) - np.searchsorted(owner, np.arange(n + 1))[owner]
    keep = rank < need[owner]
    out = lab.copy()
    out[ys[keep], xs[keep]] = owner[keep]
    return out


def rescale_area(lab: np.ndarray, prob: np.ndarray, scale: float) -> np.ndarray:
    """Scale every instance to `scale` x its area: grow (scale > 1) or keep its most probable pixels (scale < 1)."""
    if scale > 1.001:
        return grow_area(lab, prob, scale)
    out = np.zeros_like(lab)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        a = int(m.sum())
        k = max(int(round(a * scale)), 6)
        if k >= a:
            out[sl][m] = i
            continue
        thr = np.partition(prob[sl][m], a - k)[a - k]
        keep = m & (prob[sl] >= thr)
        cl, n = ndi.label(keep)
        if n > 1:
            keep = cl == (1 + int(np.argmax(np.bincount(cl.ravel())[1:])))
        out[sl][keep] = i
    return out


def instances_from_cp(dP: np.ndarray, cellprob: np.ndarray, p: dict):
    """Cached Cellpose output -> (int32 label map 1..N, per-instance info).  p['area_scale'] != 1 rescales every mask."""
    from cellpose import dynamics

    niter = int(200 * float(p["diam"]) / 30.0)
    lab = dynamics.resize_and_compute_masks(dP.astype(np.float32), cellprob.astype(np.float32), niter=niter,
                                            cellprob_threshold=float(p["cellprob"]), flow_threshold=float(p["flow"]), resize=None,
                                            min_size=int(p["min_size"]), max_size_fraction=0.4, device=torch.device("cpu"))
    lab = np.asarray(lab).astype(np.int32)
    conf_map = 1.0 / (1.0 + np.exp(-cellprob.astype(np.float32)))
    if abs(float(p.get("area_scale", 1.0)) - 1.0) > 0.001:
        lab = rescale_area(lab, conf_map, float(p["area_scale"]))
    lab = _filter(lab, conf_map, p["min_area"], p["max_area"], p["min_conf"])
    return _relabel(lab, conf_map)


def cp_infer_stage(tag: str, mod: str, diam: float, fold_models: dict, kinds=("oof", "test"), force: bool = False):
    """Held-out maps: each training region is predicted by the model of its own fold (trained without that mouse).
    Test maps: weighted mean of all fold models' outputs; all-data models (fold >= 3) count double."""
    idx = load_index()
    t0 = time.time()
    oof = [(r.sample_id, int(r.fold)) for r in idx.itertuples() if r.split == "train" and "oof" in kinds]
    test = [r.sample_id for r in idx.itertuples() if r.split == "test" and "test" in kinds]
    oof = [(s, f) for s, f in oof if force or not cp_file("oof", s, mod, tag).exists()]
    test = [s for s in test if force or not cp_file("test", s, mod, tag).exists()]
    acc = {}
    for f in sorted(fold_models):
        model = load_cp_model(fold_models[f])
        for sid, fo in oof:
            if fo == f:
                dP, cp = cp_forward(model, np.asarray(RegionData(sid).arr(f"{mod}_raw")), diam)
                np.savez(cp_file("oof", sid, mod, tag), dP=dP, cp=cp)
        w = 2.0 if f >= 3 else 1.0
        for sid in test:
            dP, cp = cp_forward(model, np.asarray(RegionData(sid).arr(f"{mod}_raw")), diam)
            a = acc.setdefault(sid, [0.0, 0.0, 0.0])
            a[0] = a[0] + w * dP.astype(np.float32)
            a[1] = a[1] + w * cp.astype(np.float32)
            a[2] += w
        del model
        torch.cuda.empty_cache()
    for sid, (dPs, cps, n) in acc.items():
        np.savez(cp_file("test", sid, mod, tag), dP=(dPs / n).astype(np.float16), cp=(cps / n).astype(np.float16))
    print(f"[cp_infer] {tag}/{mod}: {len(oof)} held-out + {len(test)} test regions in {time.time() - t0:.0f}s")
