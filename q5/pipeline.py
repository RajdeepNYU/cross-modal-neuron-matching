"""Held-out evaluation pipeline: cached Cellpose outputs -> instances -> mask filter -> registration -> candidate pairs ->
pair classifier -> one-to-one assignment.  Registration results are cached on disk keyed by the inputs that determine them."""
from __future__ import annotations

import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .cellpose_seg import cp_file, instances_from_cp
from .config import P, cfg_hash
from .match import FEATURES, assign, inst_table, label_candidates, make_candidates
from .metrics import aggregate, pair_correctness, seg_match
from .prep import RegionData, load_index
from .register import register


# ------------------------------------------------------------------------------ instances
def src_exists(kind, sid, mod, p) -> bool:
    return cp_file(kind, sid, mod, p["tag"]).exists()


def load_instances(sid, kind, mod, p):
    z = np.load(cp_file(kind, sid, mod, p["tag"]))
    return instances_from_cp(z["dP"], z["cp"], p)


def build_region(args):
    sid, kind, seg_cfg = args
    rd = RegionData(sid)
    out = {}
    for mod in ("iv", "ex"):
        lab, info = load_instances(sid, kind, mod, seg_cfg[mod])
        tab = inst_table(lab, np.asarray(rd.arr(f"{mod}_raw")), info["conf"])
        tab["aux_p"] = 0.0
        tab["aux_iou"] = 0.0
        out[mod] = dict(lab=lab.astype(np.uint16), tab=tab)
    return sid, out


def build_instances(sids, kind, seg_cfg, workers=8):
    jobs = [(s, kind, seg_cfg) for s in sids]
    if workers > 1 and len(jobs) > 2:
        with ProcessPoolExecutor(workers) as ex:
            return dict(ex.map(build_region, jobs, chunksize=1))
    return dict(build_region(j) for j in jobs)


# ------------------------------------------------------------------------------ mask filter (p_tp is also a pair feature)
MASK_FEATS = ["area", "conf", "solidity", "ecc", "int_mean", "int_max", "contrast", "n_near", "int_rank", "con_rank",
              "int_rel", "area_rel", "n_inst", "aux_p", "aux_iou"]


def mask_features(tab: pd.DataFrame) -> pd.DataFrame:
    from scipy.spatial import cKDTree

    t = tab.copy()
    if len(t) == 0:
        for c in MASK_FEATS:
            t[c] = []
        return t
    xy = t[["cx", "cy"]].to_numpy()
    t["n_near"] = [len(x) - 1 for x in cKDTree(xy).query_ball_point(xy, 40.0)]
    t["int_rank"] = t["int_mean"].rank(pct=True)
    t["con_rank"] = t["contrast"].rank(pct=True)
    t["int_rel"] = t["int_mean"] / max(float(t["int_mean"].median()), 1e-6)
    t["area_rel"] = t["area"] / max(float(t["area"].median()), 1e-6)
    t["n_inst"] = len(t)
    return t


def mask_table(sids, inst, segm):
    out = {}
    for mod in ("iv", "ex"):
        rows = []
        for s in sids:
            f = mask_features(inst[s][mod]["tab"])
            f["sample_id"] = s
            f["tp"] = (segm[s][mod]["pr2gt"][1:] > 0).astype(int)
            rows.append(f)
        out[mod] = pd.concat(rows, ignore_index=True)
    return out


def mask_gbm(cfg):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=4, min_samples_leaf=30,
                                          l2_regularization=1.0, random_state=cfg["seed"])


def pq_for_keep(m: dict, keep: np.ndarray) -> float:
    k = np.concatenate([[False], keep])
    tpm = (m["pr2gt"] > 0) & k
    tp = int(tpm.sum())
    den = tp + 0.5 * (int(k.sum()) - tp) + 0.5 * (m["n_gt"] - tp)
    return float(m["pr_iou"][tpm].sum() / den) if den > 0 else 1.0


def tune_mask_threshold(sids, segm, ptab, mod):
    grid = np.arange(0.0, 0.62, 0.02)
    by = {s: g["p_tp"].to_numpy() for s, g in ptab.groupby("sample_id")}
    scores = [np.mean([pq_for_keep(segm[s][mod], by.get(s, np.zeros(0)) >= t) for s in sids]) for t in grid]
    sc = pd.Series(scores).rolling(3, center=True, min_periods=1).mean().to_numpy()
    return float(grid[int(np.argmax(sc))])


def apply_mask_filter(inst_r: dict, p_tp: dict, thr: dict):
    out = {}
    for mod in ("iv", "ex"):
        lab, tab = inst_r[mod]["lab"], inst_r[mod]["tab"]
        keep = p_tp[mod] >= thr[mod]
        remap = np.zeros(len(keep) + 1, np.uint16)
        remap[1:][keep] = np.arange(1, keep.sum() + 1)
        t2 = tab[keep].reset_index(drop=True).copy()
        t2["p_tp"] = p_tp[mod][keep]
        out[mod] = dict(lab=remap[lab], tab=t2)
    return out


def fit_mask_cv(cfg, ptab, folds):
    """Leave-one-mouse-out probability that each predicted mask is a true positive."""
    ptab = ptab.copy()
    ptab["fold"] = ptab["sample_id"].map(folds)
    p = np.zeros(len(ptab))
    for k in sorted(ptab["fold"].unique()):
        tr, va = (ptab["fold"] != k).to_numpy(), (ptab["fold"] == k).to_numpy()
        m = mask_gbm(cfg).fit(ptab.loc[tr, MASK_FEATS].to_numpy(), ptab.loc[tr, "tp"].to_numpy())
        p[va] = m.predict_proba(ptab.loc[va, MASK_FEATS].to_numpy())[:, 1]
    return p


# ------------------------------------------------------------------------------ registration (cached)
def region_shapes(sid):
    rd = RegionData(sid)
    return tuple(rd.meta["iv_shape"]), tuple(rd.meta["ex_shape"])


def inst_hash(inst, sids) -> str:
    sig = []
    for s in sids:
        a, b = inst[s]["iv"]["tab"], inst[s]["ex"]["tab"]
        sig.append((s, len(a), len(b), round(float(a["cx"].sum() + a["cy"].sum()), 1), round(float(b["cx"].sum() + b["cy"].sum()), 1)))
    return cfg_hash(sig)


def _cached(f, fn):
    if f.exists():
        return pickle.load(open(f, "rb"))
    r = fn()
    pickle.dump(r, open(f, "wb"))
    return r


def register_regions(cfg, sids, inst, kind, seg_key, verbose=True):
    """Per-region pose search, a tight-prior pass, image evidence, mouse-level consensus, a constrained second pass
    around each mouse's dominant pose, and a final consensus vote."""
    from .consensus import apply_consensus, constrained_cfg, mouse_modes
    from .imgreg import add_image_z

    rc = cfg["reg"]
    d = P.reg / f"{cfg['name']}_{kind}_{seg_key}_{cfg_hash(rc)}"
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    xy = {s: (inst[s]["iv"]["tab"][["cx", "cy"]].to_numpy(float), inst[s]["ex"]["tab"][["cx", "cy"]].to_numpy(float)) for s in sids}
    shapes = {s: region_shapes(s) for s in sids}
    out = {s: _cached(d / f"{s}.pkl", lambda s=s: register(xy[s][0], xy[s][1], shapes[s][0], shapes[s][1], rc)) for s in sids}
    if verbose:
        print(f"  pass 1 registered ({time.time() - t0:.0f}s)", flush=True)

    prior = np.array(list(rc["prior_pose"]) + [0.0, 0.0])
    for s in sids:
        def pass_a(s=s):
            rcA = constrained_cfg(rc, prior, np.array([3.0, 0.02, 0.03, 0.02, 60.0, 60.0]), shapes[s][1],
                                  rot_half=rc["prior_rot_half"], t_half=0.53 * max(shapes[s][1]))
            rcA["n_cand"] = rc["prior_n_cand"]
            return register(xy[s][0], xy[s][1], shapes[s][0], shapes[s][1], rcA)
        rA = _cached(d / f"{s}.pA.pkl", pass_a)
        r1 = dict(out[s])
        r1["cands"] = list(r1.get("cands", [])) + list(rA.get("cands", []))
        if rA.get("M") is not None and rA.get("kc", 0) > (r1.get("kc", 0) if r1.get("M") is not None else -1):
            for k in ("M", "kc", "n_tight", "med_tight", "sx", "sy", "rot", "shear", "ok", "kc_second", "coarse_score"):
                if k in rA:
                    r1[k] = rA[k]
        out[s] = r1
    for s in sids:
        pool = _cached(d / f"{s}.z1.pkl", lambda s=s: add_image_z(s, out[s].get("cands", []), topk=rc["z_topk"]))
        out[s] = dict(out[s], cands=pool)
    if verbose:
        print(f"  prior pass + image evidence done ({time.time() - t0:.0f}s)", flush=True)

    groups = {s: s.split("__")[0] for s in sids}
    modes = mouse_modes(apply_consensus(out, shapes, groups, alpha=rc["cons_alpha"]), shapes, groups)
    reg2 = {}
    for s in sids:
        g = groups[s]
        if g not in modes:
            reg2[s] = out[s]
            continue
        m = modes[g]

        def pass_2(s=s, m=m):
            r2 = register(xy[s][0], xy[s][1], shapes[s][0], shapes[s][1], constrained_cfg(rc, m["p"], m["spread"], shapes[s][1]))
            return dict(r2, cands=add_image_z(s, r2.get("cands", []), topk=rc["z_topk"]))
        r2 = dict(_cached(d / f"{s}.p2_{cfg_hash(np.round(m['p'], 2).tolist())}.pkl", pass_2))
        r2["cands"] = list(r2.get("cands", [])) + list(out[s].get("cands", []))
        reg2[s] = r2
    final = apply_consensus(reg2, shapes, groups, alpha=rc["cons_alpha"])
    if verbose:
        print(f"  registration done: {len(modes)} mouse modes ({time.time() - t0:.0f}s)", flush=True)
    return final


# ------------------------------------------------------------------------------ candidates and pair classifier
def region_candidates(cfg, sid, inst_r, reg_r, gt=None):
    ivs, exs = region_shapes(sid)
    df = make_candidates(reg_r, inst_r["iv"]["tab"], inst_r["ex"]["tab"], ivs, exs, cfg["match"], cfg["reg"])
    if df.empty:
        return df.assign(sample_id=sid) if len(df.columns) else df
    df.insert(0, "sample_id", sid)
    if gt is not None:
        df = label_candidates(df, gt["iv_m"], gt["ex_m"], gt["pairs"])
    return df


def load_gt(sid):
    rd = RegionData(sid)
    return dict(iv_lab=np.asarray(rd.arr("iv_lab")).astype(np.int32), ex_lab=np.asarray(rd.arr("ex_lab")).astype(np.int32),
                pairs=np.load(rd.d / "pairs.npy"))


def gbm(cfg):
    g = cfg["match"]["gbm"]
    return HistGradientBoostingClassifier(max_iter=g["max_iter"], learning_rate=g["learning_rate"], max_depth=g["max_depth"],
                                          min_samples_leaf=g["min_samples_leaf"], l2_regularization=g["l2_regularization"],
                                          random_state=cfg["seed"])


def fit_predict_cv(cfg, cand: pd.DataFrame, folds: dict):
    """Leave-one-mouse-out pair probabilities for every training candidate."""
    cand = cand.copy()
    cand["fold"] = cand["sample_id"].map(folds)
    p = np.zeros(len(cand))
    for k in sorted(cand["fold"].unique()):
        tr, va = cand["fold"] != k, cand["fold"] == k
        if cand.loc[tr, "y"].sum() < 5:
            continue
        m = gbm(cfg).fit(cand.loc[tr, FEATURES].to_numpy(), cand.loc[tr, "y"].to_numpy())
        p[va.to_numpy()] = m.predict_proba(cand.loc[va, FEATURES].to_numpy())[:, 1]
    return p


def pick_threshold(cand, nver):
    """Pair-probability threshold maximising pooled F1 of the held-out predictions (3-point smoothed)."""
    grid = np.concatenate([np.arange(0.01, 0.05, 0.01), np.arange(0.05, 0.9, 0.025)])
    f1s = []
    for t in grid:
        tot_pred = tot_corr = 0
        for _, c in cand.groupby("sample_id"):
            pr = assign(c, c["p"].to_numpy(), t)
            if len(pr) == 0:
                continue
            ok = c[c["y"] == 1]
            tot_pred += len(pr)
            tot_corr += len(set(map(tuple, pr.tolist())) & set(zip(ok["iv_id"].tolist(), ok["ex_id"].tolist())))
        prec = tot_corr / tot_pred if tot_pred else 0.0
        rec = tot_corr / nver if nver else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    sm = pd.Series(f1s).rolling(3, center=True, min_periods=1).mean()
    return float(grid[int(sm.idxmax())])


# ------------------------------------------------------------------------------ evaluation on held-out maps
def evaluate(cfg: dict, workers: int = 8, verbose=True):
    t0 = time.time()
    idx = load_index()
    tr = idx[idx.split == "train"]
    sids = [s for s in tr.sample_id if all(src_exists("oof", s, m, cfg["seg"][m]) for m in ("iv", "ex"))]
    if len(sids) < len(tr):
        raise FileNotFoundError(f"held-out Cellpose outputs exist for only {len(sids)}/{len(tr)} training regions; run cp_infer first")
    folds = dict(zip(tr.sample_id, tr.fold))
    inst = build_instances(sids, "oof", cfg["seg"], workers)
    gts = {s: load_gt(s) for s in sids}

    def match_all(inst_):
        return {s: dict(iv=seg_match(gts[s]["iv_lab"], inst_[s]["iv"]["lab"].astype(np.int32)),
                        ex=seg_match(gts[s]["ex_lab"], inst_[s]["ex"]["lab"].astype(np.int32))) for s in sids}

    segm = match_all(inst)
    ptabs = mask_table(sids, inst, segm)
    thr_m, p_by = {}, {}
    for mod in ("iv", "ex"):
        ptabs[mod]["p_tp"] = fit_mask_cv(cfg, ptabs[mod], folds)
        thr_m[mod] = tune_mask_threshold(sids, segm, ptabs[mod], mod)
        p_by[mod] = {s: g["p_tp"].to_numpy() for s, g in ptabs[mod].groupby("sample_id")}
    inst = {s: apply_mask_filter(inst[s], {m: p_by[m].get(s, np.zeros(0)) for m in ("iv", "ex")}, thr_m) for s in sids}
    segm = match_all(inst)
    if verbose:
        print(f"[eval] instances + mask filter ({time.time() - t0:.0f}s)", flush=True)

    seg_key = cfg_hash(cfg["seg"]) + "_" + inst_hash(inst, sids)
    regs = register_regions(cfg, sids, inst, "oof", seg_key, verbose=verbose)
    cand = pd.concat([df for s in sids
                      if len(df := region_candidates(cfg, s, inst[s], regs[s],
                                                     dict(iv_m=segm[s]["iv"], ex_m=segm[s]["ex"], pairs=gts[s]["pairs"])))],
                     ignore_index=True)
    cand["p"] = fit_predict_cv(cfg, cand, folds)
    thr = pick_threshold(cand, sum(len(gts[s]["pairs"]) for s in sids))
    results = []
    for s in sids:
        c = cand[cand.sample_id == s]
        pr = assign(c, c["p"].to_numpy(), thr) if len(c) else np.zeros((0, 2), int)
        corr = pair_correctness(pr, gts[s]["pairs"], segm[s]["iv"], segm[s]["ex"])
        results.append(dict(iv=segm[s]["iv"], ex=segm[s]["ex"], n_pred_pairs=len(pr), n_correct_pairs=int(corr.sum()),
                            n_verified=len(gts[s]["pairs"])))
    summary = aggregate(results)
    summary["thr"] = thr
    summary["fold_scores"] = {int(k): aggregate([r for r, s in zip(results, sids) if folds[s] == k])["score"] for k in sorted(set(folds.values()))}
    if verbose:
        s = summary
        print(f"[eval] held-out: PQ_iv {s['PQ_iv']:.4f} PQ_ex {s['PQ_ex']:.4f} precision {s['precision']:.3f} recall {s['recall']:.3f} "
              f"F1 {s['F1']:.4f} score {s['score']:.4f} (pair threshold {thr:.3f}; per mouse {s['fold_scores']}) ({time.time() - t0:.0f}s)")
    return dict(cand=cand, summary=summary, mask_tabs=ptabs, mask_thr=thr_m)
