"""Mouse-level consensus over registration hypotheses: within one mouse the IV->EX pose is similar across regions, so a
region's hypothesis that agrees with the hypotheses of the other regions of the same mouse receives a bonus.
Only unlabelled predictions are used."""
from __future__ import annotations

import numpy as np

from .register import apply_affine, decompose

KSCALE = np.array([3.0, 0.03, 0.04, 0.03, 60.0, 60.0])  # kernel widths: rot (deg), sx, sy, shear, IV-centre offset x / y (px)
BETA = 1.0  # weight of the image-evidence z-score relative to the centroid kernel correlation


def pose_params(M, iv_shape, ex_shape):
    h, w = iv_shape
    He, We = ex_shape
    sx, sy, rot, sh = decompose(M)
    c = apply_affine(M, np.array([[w / 2.0, h / 2.0]]))[0] - np.array([We / 2.0, He / 2.0])
    return np.array([rot, sx, sy, sh, c[0], c[1]])


def cscore(c, beta=BETA):
    return c["kc"] + beta * max(c.get("z", 0.0), 0.0)


def _kern(p, q, scale):
    return float(np.exp(-0.5 * np.sum(((p - q) / scale) ** 2)))


def mouse_modes(regs: dict, shapes: dict, groups: dict, scale=KSCALE, min_group: int = 3):
    """Dominant pose of every mouse (density mode of all regions' hypotheses) -> {mouse: dict(p, support, n, spread)}."""
    pools = {}
    for s, r in regs.items():
        cs = [c for c in r.get("cands", []) if c.get("M") is not None and c["kc"] > 0]
        if cs:
            kmax = max(cscore(c) for c in cs)
            pools[s] = [dict(p=pose_params(c["M"], *shapes[s]), w=(cscore(c) / kmax) ** 2) for c in cs]
    modes = {}
    for g in set(groups.values()):
        mem = [s for s in regs if groups[s] == g and s in pools]
        if len(mem) < min_group:
            continue
        best, best_d = None, -1.0
        for s, p in [(s, o["p"]) for s in mem for o in pools[s]]:
            d = sum(max(o["w"] * _kern(p, o["p"], scale) for o in pools[s2]) for s2 in mem)
            if d > best_d:
                best, best_d = p, d
        sel = []
        for s2 in mem:
            k, p = max(((o["w"] * _kern(best, o["p"], scale), o["p"]) for o in pools[s2]), key=lambda t: t[0])
            if k > 0.3:
                sel.append(p)
        modes[g] = dict(p=np.mean(sel, 0) if sel else best, support=best_d, n=len(sel), spread=np.std(sel, 0) if len(sel) > 1 else scale)
    return modes


def constrained_cfg(rc: dict, mode_p: np.ndarray, spread: np.ndarray, ex_shape, rot_half=None, t_half=None):
    """Registration settings narrowed around a pose (rotation, scale, shape and translation window)."""
    rot, sx, sy, sh, cx, cy = mode_p
    He, We = ex_shape
    rh = rot_half if rot_half is not None else max(4.0, 2.5 * float(spread[0]))
    th = t_half if t_half is not None else max(110.0, 2.5 * float(max(spread[4], spread[5])))
    rc2 = dict(rc)
    rc2["rot_range"] = [rot - rh, rot + rh]
    rc2["rot_step"] = 0.75
    rc2["scales"] = [sx - 0.03, sx - 0.015, sx, sx + 0.015, sx + 0.03]
    rc2["shapes"] = [[sy / sx, sh]]
    rc2["t_center"] = (We / 2.0 + cx, He / 2.0 + cy)
    rc2["t_half"] = (th, th)
    rc2["n_cand"] = max(10, rc.get("n_cand", 24) // 2)
    return rc2


def apply_consensus(regs: dict, shapes: dict, groups: dict, alpha: float = 2.0, min_group: int = 3, scale=KSCALE):
    """For every region choose the hypothesis maximising (own evidence + alpha * support from the mouse's other regions)."""
    out = {s: dict(r) for s, r in regs.items()}
    by_group = {}
    for s in regs:
        by_group.setdefault(groups[s], []).append(s)
    pools = {}
    for s, r in regs.items():
        cs = [c for c in r.get("cands", []) if c.get("M") is not None and c["kc"] > 0]
        pool = []
        if cs:
            kmax = max(cscore(c) for c in cs)
            for c in sorted(cs, key=lambda c: -cscore(c)):
                p = pose_params(c["M"], *shapes[s])
                if any(_kern(p, q["p"], scale) > 0.6 for q in pool):
                    continue
                pool.append(dict(p=p, kc=c["kc"], s=cscore(c), w=(cscore(c) / kmax) ** 2, c=c))
        pools[s] = pool
    for g, members in by_group.items():
        if len(members) < min_group:
            continue
        for s in members:
            pool = pools[s]
            if not pool:
                continue
            best, best_sc = None, -1e9
            for q in pool:
                sup = sum(max(o["w"] * _kern(q["p"], o["p"], scale) for o in pools[s2]) for s2 in members if s2 != s and pools[s2])
                q["support"] = sup
                sc = q["s"] + alpha * sup
                if sc > best_sc:
                    best, best_sc = q, sc
            c = best["c"]
            r = out[s]
            r["M"], r["kc"] = c["M"], c["kc"]
            r["z"], r["ncc"] = float(c.get("z", 0.0)), float(c.get("ncc", 0.0))
            r["n_tight"], r["med_tight"] = c["n_tight"], c.get("med_tight", 99.0)
            r["sx"], r["sy"], r["rot"], r["shear"] = c["sx"], c["sy"], c["rot"], c["shear"]
            r["ok"] = c["n_tight"] >= 6
            r["cons_support"] = float(best["support"])
    for s in out:
        out[s].setdefault("cons_support", 0.0)
        out[s].setdefault("z", 0.0)
    return out
