"""Fine-tuning of Cellpose-SAM per modality and subject-level fold.

Images are pre-scaled by 30/diam (the factor CellposeModel.eval applies at inference), labels with nearest neighbour.
IV: whole regions.  EX: random crops, most of them centred near annotated cells (EX frames are mostly empty tissue).
Fold k in {0,1,2} trains on the two other mice; fold 3 trains on all three; fold 4 is a second all-data model
(different image order and 1.5x learning rate).
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .cellpose_seg import cp_norm
from .config import P
from .prep import RegionData, load_index


def build_data(mod: str, fold: int, diam: float, ncrops: int, crop: int, cell_bias: float, seed: int, verified_bias: float = 0.0):
    idx = load_index()
    tr = idx[idx.split == "train"]
    sids = list(tr[tr.fold != fold].sample_id)
    f = 30.0 / diam
    rng = np.random.default_rng(seed)
    if fold >= 4:
        rng.shuffle(sids)
    data, labels = [], []
    if crop <= 0:
        for sid in sids:
            rd = RegionData(sid)
            img = cp_norm(np.asarray(rd.arr(f"{mod}_raw")))
            lab = np.asarray(rd.arr(f"{mod}_lab")).astype(np.int32)
            data.append(cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR))
            labels.append(cv2.resize(lab.astype(np.float32), None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST).astype(np.int32))
        return data, labels
    cache = {}
    cents = {s: np.load(RegionData(s).d / f"{mod}_cent.npy") for s in sids}
    vcents = {}
    for s in sids:
        pr = np.load(RegionData(s).d / "pairs.npy")
        vcents[s] = cents[s][pr[:, 0 if mod == "iv" else 1] - 1] if len(pr) else np.zeros((0, 2))
    for _ in range(ncrops):
        sid = sids[rng.integers(len(sids))]
        if sid not in cache:
            if len(cache) > 6:
                cache.pop(next(iter(cache)))
            rd = RegionData(sid)
            cache[sid] = (cp_norm(np.asarray(rd.arr(f"{mod}_raw"))), np.asarray(rd.arr(f"{mod}_lab")))
        raw, lab = cache[sid]
        H, W = raw.shape
        c, vc = cents[sid], vcents[sid]
        if len(vc) and rng.random() < verified_bias:
            cy, cx = vc[rng.integers(len(vc))] + rng.uniform(-crop / 3, crop / 3, 2)
        elif len(c) and rng.random() < cell_bias:
            cy, cx = c[rng.integers(len(c))] + rng.uniform(-crop / 3, crop / 3, 2)
        else:
            cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        y0 = int(np.clip(cy - crop / 2, 0, max(H - crop, 0)))
        x0 = int(np.clip(cx - crop / 2, 0, max(W - crop, 0)))
        im = raw[y0:y0 + crop, x0:x0 + crop].astype(np.float32)
        lb = lab[y0:y0 + crop, x0:x0 + crop].astype(np.float32)
        data.append(cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR))
        labels.append(cv2.resize(lb, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST).astype(np.int32))
    return data, labels


def model_path(cfg: dict, mod: str, fold: int) -> Path:
    return P.ckpt_dir(cfg) / f"cp_{cfg['cp_tag']}" / f"{mod}_f{fold}" / "models" / f"cp_{mod}_f{fold}"


def finetune(cfg: dict, mod: str, fold: int) -> Path:
    import torch
    from cellpose import models, train

    t = cfg["cp_train"][mod]
    diam = float(cfg["seg"][mod]["diam"])
    final = model_path(cfg, mod, fold)
    out_dir = final.parents[1]
    done = out_dir / "DONE"
    if final.exists() and done.exists():
        print(f"[cp_train] {final} already trained; skipping")
        return final
    out_dir.mkdir(parents=True, exist_ok=True)
    lr = float(t["lr"]) * (1.5 if fold == 4 else 1.0)
    data, labels = build_data(mod, fold, diam, int(t.get("ncrops", 600)), int(t.get("crop", 0)), float(t.get("cell_bias", 0.85)),
                              cfg["seed"] + fold, float(t.get("verified_bias", 0.0)))
    print(f"[cp_train] {cfg['cp_tag']} {mod} fold {fold}: {len(data)} images, scale {30 / diam:.2f}, {t['epochs']} epochs x {t['per_epoch']}, "
          f"lr {lr}, base model {cfg.get('cp_base', 'cpsam')}", flush=True)
    model = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=cfg.get("cp_base", "cpsam"))
    t0 = time.time()
    path, losses, _ = train.train_seg(model.net, train_data=data, train_labels=labels, batch_size=int(t["batch"]), learning_rate=lr,
                                      weight_decay=float(t["wd"]), n_epochs=int(t["epochs"]), nimg_per_epoch=int(t["per_epoch"]),
                                      save_path=str(out_dir), model_name=final.name, bsize=256, normalize=False,
                                      save_every=max(int(t["epochs"]) // 4, 1))
    done.write_text(f"finished {time.strftime('%F %T')}\n")
    print(f"[cp_train] done in {(time.time() - t0) / 60:.1f} min; final loss {losses[-1]:.4f}; {path}", flush=True)
    return Path(path)
