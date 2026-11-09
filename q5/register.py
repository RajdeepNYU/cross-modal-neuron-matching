"""IV -> EX point-set registration on predicted cell centroids (all coordinates are (x, y)).

1. coarse: FFT cross-correlation of the rotated / scaled / sheared IV point cloud against a blurred EX point map,
   restricted to a translation window
2. refine: kernel-correlation (Gaussian mass) maximisation over a full affine with annealed sigma
3. local: leave-one-out Gaussian-weighted residual field from tight mutual matches
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from scipy.fft import next_fast_len
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ M[:, :2].T + M[:, 2]


def decompose(M: np.ndarray):
    """(scale_x, scale_y, rotation in degrees, shear) of M_lin = R(rot) @ [[sx, shear*sx], [0, sy]]."""
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    sx = np.hypot(a, c)
    rot = np.degrees(np.arctan2(c, a))
    sy = (a * d - b * c) / max(sx, 1e-9)
    shear = (a * b + c * d) / max(sx * sx, 1e-9)
    return sx, sy, rot, shear


def lin_from_params(rot_deg, sx, sy_ratio, shear):
    th = np.deg2rad(rot_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return R @ np.array([[sx, shear * sx], [0.0, sx * sy_ratio]])


def mat_from_lin(L, c0, t):
    """Affine with dst = L (a - c0) + t."""
    return np.concatenate([L, (t - L @ c0)[:, None]], 1)


def _splat(pts, shape, circular=True):
    """Bilinear splat of points onto a grid."""
    H, W = shape
    g = np.zeros((H, W), np.float32)
    x, y = pts[:, 0], pts[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    w = np.ones(len(x), np.float32)
    for dx, dy, ww in [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)]:
        xi, yi = x0 + dx, y0 + dy
        if circular:
            np.add.at(g, (yi % H, xi % W), ww * w)
        else:
            m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            np.add.at(g, (yi[m], xi[m]), (ww * w)[m])
    return g


def coarse_search(A, B, iv_shape, ex_shape, rc, dev):
    """Top `n_cand` distinct (rotation, scale, shape, translation) hypotheses by FFT correlation score."""
    ds = rc["coarse_ds"]
    He, We = ex_shape
    h, w = iv_shape
    c0 = np.array([w / 2.0, h / 2.0])
    smax = max(rc["scales"]) * 1.1
    L = 0.5 * np.hypot(h, w) * smax * 1.05
    Hg = next_fast_len(int(np.ceil((He + 2 * L) / ds)))
    Wg = next_fast_len(int(np.ceil((We + 2 * L) / ds)))
    E = cv2.GaussianBlur(_splat(B / ds, (Hg, Wg), circular=False), (0, 0), rc["coarse_sigma"] / ds)
    FE = torch.fft.rfft2(torch.from_numpy(E).to(dev))
    rots = np.arange(rc["rot_range"][0], rc["rot_range"][1] + 1e-6, rc["rot_step"])
    if rc.get("t_center") is not None:
        (cx_, cy_), (hx_, hy_) = rc["t_center"], rc["t_half"]
        ylo, yhi = int(max(0, (cy_ - hy_) / ds)), int(min(He / ds, (cy_ + hy_) / ds))
        xlo, xhi = int(max(0, (cx_ - hx_) / ds)), int(min(We / ds, (cx_ + hx_) / ds))
    else:
        hw = rc["t_window"]
        ylo, yhi = int(max(0, (0.5 - hw) * He / ds)), int(min(He / ds, (0.5 + hw) * He / ds))
        xlo, xhi = int(max(0, (0.5 - hw) * We / ds)), int(min(We / ds, (0.5 + hw) * We / ds))
    if yhi <= ylo + 1 or xhi <= xlo + 1:
        return []
    res = []
    for (kap, gam) in rc["shapes"]:
        for s in rc["scales"]:
            for th in rots:
                u = (A - c0) @ lin_from_params(th, s, kap, gam).T / ds
                FD = torch.fft.rfft2(torch.from_numpy(_splat(u, (Hg, Wg), circular=True)).to(dev))
                C = torch.fft.irfft2(FE * torch.conj(FD), s=(Hg, Wg))[ylo:yhi, xlo:xhi]
                v, k = torch.max(C.flatten(), 0)
                ty, tx = divmod(int(k), xhi - xlo)
                res.append(dict(theta=float(th), s=float(s), kap=kap, gam=gam,
                                t=np.array([(tx + xlo) * ds, (ty + ylo) * ds], np.float64), score=float(v)))
    res.sort(key=lambda r: -r["score"])
    out = []
    for r in res:
        if any(abs(r["theta"] - q["theta"]) <= 4 and abs(r["s"] - q["s"]) <= 0.05 and np.linalg.norm(r["t"] - q["t"]) < 30 for q in out):
            continue
        out.append(r)
        if len(out) >= rc["n_cand"]:
            break
    for r in out:
        r["M0"] = mat_from_lin(lin_from_params(r["theta"], r["s"], r["kap"], r["gam"]), c0, r["t"])
    return out


def kc_value(M, A, B, sigma):
    Ta = apply_affine(M, A)
    d2 = ((B[None, :, :] - Ta[:, None, :]) ** 2).sum(-1)
    return float(np.exp(-d2 / (2 * sigma ** 2)).sum())


def kc_refine(A, B, M0, sigmas, maxiter=40):
    """Maximise sum_ij exp(-|M a_i - b_j|^2 / 2s^2) over a 2x3 affine, annealing s (coordinates normalised by 100 px)."""
    S = 100.0
    c0 = A.mean(0)
    An = (A - c0) / S
    t0 = apply_affine(M0, c0[None])[0]
    Bn = (B - t0) / S
    p = np.concatenate([M0[:, :2].ravel(), [0.0, 0.0]])
    for sg in sigmas:
        sgn = sg / S

        def fun(q):
            Ln, tn = q[:4].reshape(2, 2), q[4:6]
            Ta = An @ Ln.T + tn[None, :]
            diff = Bn[None, :, :] - Ta[:, None, :]
            W = np.exp(-(diff ** 2).sum(-1) / (2 * sgn ** 2))
            gT = np.einsum("ij,ijr->ir", W, diff) / sgn ** 2
            return -W.sum(), -np.concatenate([(gT.T @ An).ravel(), gT.sum(0)])

        p = minimize(fun, p, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter)).x
    Ln, tn = p[:4].reshape(2, 2), p[4:6]
    return np.concatenate([Ln, ((t0 + tn * S) - Ln @ c0)[:, None]], 1)


def _mutual(Ta, B, r):
    """Mutual nearest neighbours closer than r: (idx in Ta, idx in B, distance)."""
    treeB, treeA = cKDTree(B), cKDTree(Ta)
    d1, j = treeB.query(Ta)
    _, i = treeA.query(B)
    idx = np.arange(len(Ta))
    mut = (i[j] == idx) & (d1 < r)
    return idx[mut], j[mut], d1[mut]


def register(A, B, iv_shape, ex_shape, rc, dev=None):
    """A: IV centroids (n,2), B: EX centroids (m,2).  Best affine by kernel correlation over the refined hypotheses."""
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(A) < 8 or len(B) < 8:
        return dict(M=None, ok=False, n_tight=0, kc=0.0, cands=[])
    cands = coarse_search(A, B, iv_shape, ex_shape, rc, dev)
    lim = rc["sane"]
    cinfo, best = [], None
    for c in cands:
        M = kc_refine(A, B, c["M0"], rc["kc_sigmas"])
        sx, sy, rot, shear = decompose(M)
        sane = lim["sx"][0] < sx < lim["sx"][1] and lim["sy"][0] < sy < lim["sy"][1] and lim["shear"][0] < shear < lim["shear"][1]
        kc = kc_value(M, A, B, rc["score_sigma"]) if sane else 0.0
        ia, jb, d = _mutual(apply_affine(M, A), B, rc["tight_r"])
        cinfo.append(dict(theta=c["theta"], s=c["s"], coarse=c["score"], kc=kc, n_tight=int(len(ia)), sx=sx, sy=sy,
                          rot=rot, shear=shear, M=M if sane else None, med_tight=float(np.median(d)) if len(d) else 99.0))
        if sane and (best is None or kc > best[0]):
            best = (kc, M, int(len(ia)), d, c)
    if best is None:
        return dict(M=None, ok=False, n_tight=0, kc=0.0, cands=cinfo)
    kc, M, nt, d, c = best
    sx, sy, rot, shear = decompose(M)
    other = [ci["kc"] for ci in cinfo if abs(ci["theta"] - c["theta"]) > 4 or abs(ci["s"] - c["s"]) > 0.05]
    return dict(M=M, ok=nt >= rc["min_tight"], n_tight=nt, kc=kc, kc_second=max(other) if other else 0.0, sx=sx, sy=sy, rot=rot,
                shear=shear, med_tight=float(np.median(d)) if len(d) else 99.0, coarse_score=c["score"], cands=cinfo)


def local_field(Ta, inl, sigma, lam=1.0, exclude_a=None, exclude_b=None):
    """Displacement (q,2) for the warped IV points Ta from the residuals of the inlier pairs inl = (pa, pb, ia, jb),
    excluding a point's own pair; also returns the supporting weight mass (q,)."""
    pa, pb, ia, jb = inl
    if len(pa) == 0:
        return np.zeros_like(Ta), np.zeros(len(Ta))
    res = pb - pa
    d2 = ((Ta[:, None, :] - pa[None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * sigma ** 2))
    if exclude_a is not None:
        w = w * (ia[None, :] != exclude_a[:, None])
    if exclude_b is not None:
        w = w * (jb[None, :] != exclude_b[:, None])
    sw = w.sum(1)
    return (w @ res) / (sw[:, None] + lam), sw
