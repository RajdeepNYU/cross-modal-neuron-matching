"""Hidden-test prediction, Kaggle CSV writing and strict format validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from .config import cfg_hash
from .data import label_to_rle_dict, rle_decode, rle_encode
from .match import FEATURES, assign
from .pipeline import (MASK_FEATS, apply_mask_filter, build_instances, gbm, inst_hash, mask_features, mask_gbm, region_candidates,
                       register_regions, src_exists)
from .prep import load_index

HEADER = ["sample_id", "invivo_instances", "exvivo_instances", "match_pairs"]


def predict_test(cfg: dict, ev: dict, thr: float, workers: int = 8) -> dict:
    """Fit the mask filter and the pair classifier on all held-out training predictions, apply them to the test regions."""
    idx = load_index()
    sids = list(idx[idx.split == "test"].sample_id)
    if not all(src_exists("test", s, m, cfg["seg"][m]) for s in sids for m in ("iv", "ex")):
        raise FileNotFoundError("missing test Cellpose outputs; run cp_infer (and cp_blend) first")
    inst = build_instances(sids, "test", cfg["seg"], workers)
    p_by = {}
    for mod in ("iv", "ex"):
        pt = ev["mask_tabs"][mod]
        m = mask_gbm(cfg).fit(pt[MASK_FEATS].to_numpy(), pt["tp"].to_numpy())
        p_by[mod] = {s: m.predict_proba(mask_features(inst[s][mod]["tab"])[MASK_FEATS].to_numpy())[:, 1]
                     if len(inst[s][mod]["tab"]) else np.zeros(0) for s in sids}
    keep = {s: {m: p_by[m][s] >= ev["mask_thr"][m] for m in ("iv", "ex")} for s in sids}
    inst = {s: apply_mask_filter(inst[s], {m: p_by[m][s] for m in ("iv", "ex")}, ev["mask_thr"]) for s in sids}
    regs = register_regions(cfg, sids, inst, "test", cfg_hash(cfg["seg"]) + "_" + inst_hash(inst, sids))
    clf = gbm(cfg).fit(ev["cand"][FEATURES].to_numpy(), ev["cand"]["y"].to_numpy())
    out = {}
    for s in sids:
        c = region_candidates(cfg, s, inst[s], regs[s])
        pairs = assign(c, clf.predict_proba(c[FEATURES].to_numpy())[:, 1], thr) if len(c) else np.zeros((0, 2), int)
        out[s] = dict(iv_lab=inst[s]["iv"]["lab"], ex_lab=inst[s]["ex"]["lab"], pairs=pairs)
    print(f"[predict] {len(sids)} test regions, pair threshold {thr:.3f}, {sum(len(o['pairs']) for o in out.values())} pairs")
    for mod in ("iv", "ex"):
        scale = float(cfg.get("output", {}).get(f"{mod}_area_scale", 1.0))
        if abs(scale - 1.0) > 0.001:
            for s in sids:
                out[s][f"{mod}_lab"] = rescaled_masks(s, mod, cfg["seg"][mod], scale, keep[s][mod])
            print(f"[predict] {mod} masks rescaled x{scale} (ids and pairs unchanged)")
    return out


def rescaled_masks(sid: str, mod: str, p: dict, scale: float, keep: np.ndarray) -> np.ndarray:
    """Final output masks re-decoded with area_scale; the pair selection above used the unscaled masks."""
    from .pipeline import load_instances

    lab, _ = load_instances(sid, "test", mod, dict(p, area_scale=scale))
    if int(lab.max()) != len(keep):
        raise RuntimeError(f"{sid} {mod}: mask rescaling changed the number of instances")
    remap = np.zeros(len(keep) + 1, np.int32)
    remap[1:][keep] = np.arange(1, int(keep.sum()) + 1)
    return remap[lab]


def write_submission(pred: dict, path: Path, sample_csv: Path | str):
    rows = []
    for sid in pd.read_csv(sample_csv)["sample_id"]:
        o = pred.get(sid)
        if o is None:
            rows.append((sid, "{}", "{}", "[]"))
            continue
        iv = label_to_rle_dict(o["iv_lab"].astype(np.int32), "IV")
        ex = label_to_rle_dict(o["ex_lab"].astype(np.int32), "EX")
        pairs = [[f"IV_{int(a):06d}", f"EX_{int(b):06d}"] for a, b in o["pairs"]]
        pairs = [pr for pr in pairs if pr[0] in iv and pr[1] in ex]
        rows.append((sid, json.dumps(iv), json.dumps(ex), json.dumps(pairs)))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=HEADER).to_csv(path, index=False)


def validate_submission(csv_path: Path | str, data_root: str | Path) -> dict:
    """Strict check against the official format: header, sample ids, canonical 0-indexed RLE inside the image, one-to-one pairs."""
    csv_path, root = Path(csv_path), Path(data_root)
    errors, warns = [], []
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if list(df.columns) != HEADER:
        return dict(valid=False, errors=[f"header must be {HEADER}"], warnings=[])
    exp_ids = list(pd.read_csv(root / "sample_submission.csv", dtype=str)["sample_id"])
    if len(df) != len(exp_ids) or set(df["sample_id"]) != set(exp_ids) or df["sample_id"].duplicated().any():
        errors.append("sample_id rows do not match sample_submission.csv exactly")
    tot = dict(iv=0, ex=0, pairs=0)
    for _, r in df.iterrows():
        sid = r["sample_id"]
        subj, reg = sid.split("__", 1)
        d = root / "hidden_test" / subj / reg
        ids = {}
        for mod, fn, col in (("iv", "invivo.tif", "invivo_instances"), ("ex", "exvivo.tif", "exvivo_instances")):
            with tifffile.TiffFile(d / fn) as tf:
                H, W = (int(v) for v in tf.pages[0].shape[:2])
            obj = json.loads(r[col])
            occupied = np.zeros(H * W, dtype=np.uint8)
            for k, rle in obj.items():
                v = np.array(rle.split(), dtype=np.int64)
                st, ln = v[0::2], v[1::2]
                if len(v) == 0 or len(v) % 2 or (st < 0).any() or (ln < 1).any() or (st[1:] < (st[:-1] + ln[:-1])).any() \
                        or (st + ln > H * W).any():
                    errors.append(f"{sid}:{k}: malformed RLE")
                    continue
                m = rle_decode(rle, (H, W))
                if rle_encode(m) != rle:
                    errors.append(f"{sid}:{k}: RLE is not canonical")
                flat = m.reshape(-1)
                if (occupied[flat] > 0).any():
                    warns.append(f"{sid}:{k}: overlaps another mask")
                occupied[flat] = 1
            ids[mod] = set(obj.keys())
            tot[mod] += len(obj)
        pairs = json.loads(r["match_pairs"])
        a, b = [p[0] for p in pairs], [p[1] for p in pairs]
        if len(set(a)) != len(a) or len(set(b)) != len(b):
            errors.append(f"{sid}: a cell id is used in more than one pair")
        if any(x not in ids["iv"] for x in a) or any(x not in ids["ex"] for x in b):
            errors.append(f"{sid}: a pair references an undefined cell id")
        tot["pairs"] += len(pairs)
    rep = dict(valid=not errors, errors=errors[:20], warnings=warns[:20], sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(), **tot)
    print(f"[validate] {'VALID' if rep['valid'] else 'INVALID'}: rows {len(df)}, IV {tot['iv']}, EX {tot['ex']}, pairs {tot['pairs']}, "
          f"errors {len(errors)}, warnings {len(warns)}")
    for e in errors[:10]:
        print("   ERROR:", e)
    return rep
