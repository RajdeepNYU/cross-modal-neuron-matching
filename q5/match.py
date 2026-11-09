"""Instance features, candidate IV-EX pairs with features, and one-to-one assignment."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.measure import regionprops

from .register import _mutual, apply_affine, local_field


def inst_table(lab: np.ndarray, raw: np.ndarray, conf: np.ndarray | None = None) -> pd.DataFrame:
    """Per-instance geometry / appearance features (row i <-> label i+1)."""
    cols = ["cx", "cy", "area", "conf", "solidity", "ecc", "int_mean", "int_max", "contrast"]
    if int(lab.max()) == 0:
        return pd.DataFrame(columns=cols)
    raw = raw.astype(np.float32)
    p99 = float(np.percentile(raw, 99)) + 1e-6
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    H, W = lab.shape
    rows = []
    for pr in regionprops(lab, intensity_image=raw):
        y0, x0, y1, x1 = pr.bbox
        y0, x0, y1, x1 = max(y0 - 4, 0), max(x0 - 4, 0), min(y1 + 4, H), min(x1 + 4, W)
        l = lab[y0:y1, x0:x1]
        ring = cv2.dilate((l == pr.label).astype(np.uint8), k).astype(bool) & (l == 0)
        r = raw[y0:y1, x0:x1][ring]
        ring_mean = float(r.mean()) if r.size else float(pr.mean_intensity)
        cy, cx = pr.centroid
        rows.append((cx, cy, pr.area, 0.0, pr.solidity, pr.eccentricity, pr.mean_intensity / p99,
                     pr.max_intensity / p99, (pr.mean_intensity - ring_mean) / p99))
    df = pd.DataFrame(rows, columns=cols)
    if conf is not None and len(conf) == len(df):
        df["conf"] = conf
    return df


def make_candidates(reg: dict, iv: pd.DataFrame, ex: pd.DataFrame, iv_shape, ex_shape, mc: dict, rc: dict) -> pd.DataFrame:
    """For every EX cell, its k nearest warped IV cells within `radius` px, with pair features.  ids are 1-based labels."""
    if reg.get("M") is None or len(iv) == 0 or len(ex) == 0:
        return pd.DataFrame()
    M = reg["M"]
    A = iv[["cx", "cy"]].to_numpy(float)
    B = ex[["cx", "cy"]].to_numpy(float)
    Ta = apply_affine(M, A)
    ia, jb, _ = _mutual(Ta, B, rc["tight_r"])
    inl = (Ta[ia], B[jb], ia, jb)
    K, R = mc["knn"], mc["radius"]
    treeA, treeB = cKDTree(Ta), cKDTree(B)
    dd, ii = treeA.query(B, k=min(K + 1, len(Ta)), distance_upper_bound=R * 3)
    dd = np.atleast_2d(dd.reshape(len(B), -1))
    ii = np.atleast_2d(ii.reshape(len(B), -1))
    rowsj, rowsi = [], []
    for j in range(len(B)):
        for k in range(min(K, ii.shape[1])):
            if np.isfinite(dd[j, k]) and dd[j, k] <= R:
                rowsj.append(j)
                rowsi.append(int(ii[j, k]))
    if not rowsj:
        return pd.DataFrame()
    j = np.array(rowsj)
    i = np.array(rowsi)
    disp, sw = local_field(Ta[i], inl, rc["warp_sigma"], rc["warp_lam"], exclude_a=i, exclude_b=j)
    d_g = np.linalg.norm(Ta[i] - B[j], axis=1)
    d_l = np.linalg.norm(Ta[i] + disp - B[j], axis=1)
    dB2, _ = treeA.query(B, k=2) if len(Ta) >= 2 else (np.full((len(B), 2), 99.0), None)
    dA2, _ = treeB.query(Ta, k=2) if len(B) >= 2 else (np.full((len(Ta), 2), 99.0), None)
    h, w = iv_shape
    corners = apply_affine(M, np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
    inside = MPath(corners).contains_points(B)
    n_ex_in = int(inside.sum())
    edge = np.minimum.reduce([A[:, 0], A[:, 1], w - A[:, 0], h - A[:, 1]])
    dens_iv = np.array([len(x) for x in treeA.query_ball_point(Ta, 25.0)]) - 1
    dens_ex = np.array([len(x) for x in treeB.query_ball_point(B, 40.0)]) - 1
    ex_int_rank = ex["int_mean"].rank(pct=True).to_numpy()
    df = pd.DataFrame(dict(
        iv_id=i + 1, ex_id=j + 1,
        d_g=d_g, d_l=d_l, dx_l=(Ta[i] + disp - B[j])[:, 0], dy_l=(Ta[i] + disp - B[j])[:, 1],
        disp=np.linalg.norm(disp, axis=1), support=sw,
        j_d1=dB2[j, 0], j_d2=dB2[j, 1], i_d1=dA2[i, 0], i_d2=dA2[i, 1],
        in_foot=inside[j].astype(float), iv_edge=edge[i], dens_iv=dens_iv[i], dens_ex=dens_ex[j],
        iv_area=iv["area"].to_numpy()[i], ex_area=ex["area"].to_numpy()[j],
        area_ratio=iv["area"].to_numpy()[i] / np.maximum(ex["area"].to_numpy()[j], 1),
        iv_conf=iv["conf"].to_numpy()[i], ex_conf=ex["conf"].to_numpy()[j],
        iv_ptp=iv["p_tp"].to_numpy()[i], ex_ptp=ex["p_tp"].to_numpy()[j],
        iv_aux_p=iv["aux_p"].to_numpy()[i], ex_aux_p=ex["aux_p"].to_numpy()[j],
        iv_aux_iou=iv["aux_iou"].to_numpy()[i], ex_aux_iou=ex["aux_iou"].to_numpy()[j],
        iv_sol=iv["solidity"].to_numpy()[i], ex_sol=ex["solidity"].to_numpy()[j],
        iv_ecc=iv["ecc"].to_numpy()[i], ex_ecc=ex["ecc"].to_numpy()[j],
        iv_int=iv["int_mean"].to_numpy()[i], ex_int=ex["int_mean"].to_numpy()[j], ex_int_rank=ex_int_rank[j],
        iv_con=iv["contrast"].to_numpy()[i], ex_con=ex["contrast"].to_numpy()[j],
    ))
    df["rank_j"] = df.groupby("ex_id")["d_l"].rank(method="first")
    df["rank_i"] = df.groupby("iv_id")["d_l"].rank(method="first")
    df["n_cand_j"] = df.groupby("ex_id")["d_l"].transform("size")
    df["n_cand_i"] = df.groupby("iv_id")["d_l"].transform("size")
    gap = lambda s: (np.sort(s.to_numpy())[1] - np.sort(s.to_numpy())[0]) if len(s) > 1 else 10.0  # noqa: E731
    df["gap_j"] = df.groupby("ex_id")["d_l"].transform(gap)
    df["gap_i"] = df.groupby("iv_id")["d_l"].transform(gap)
    for k in ["kc", "kc_second", "n_tight", "med_tight", "sx", "sy", "rot", "shear", "cons_support", "z", "ncc"]:
        df[f"reg_{k}"] = float(reg.get(k, 0.0))
    df["reg_kc_ratio"] = df["reg_kc"] / max(reg.get("kc_second", 0.0), 1e-3)
    df["reg_n_ex_in"] = n_ex_in
    df["reg_tight_frac"] = reg.get("n_tight", 0) / max(n_ex_in, 1)
    df["reg_n_iv"] = len(iv)
    df["reg_n_ex"] = len(ex)
    return df


FEATURES = ["d_g", "d_l", "dx_l", "dy_l", "disp", "support", "j_d1", "j_d2", "i_d1", "i_d2", "in_foot", "iv_edge", "dens_iv",
            "dens_ex", "iv_area", "ex_area", "area_ratio", "iv_conf", "ex_conf", "iv_ptp", "ex_ptp", "iv_aux_p", "ex_aux_p",
            "iv_aux_iou", "ex_aux_iou", "iv_sol", "ex_sol", "iv_ecc", "ex_ecc", "iv_int", "ex_int", "ex_int_rank", "iv_con", "ex_con",
            "rank_j", "rank_i", "n_cand_j", "n_cand_i", "gap_j", "gap_i", "reg_kc", "reg_kc_second", "reg_n_tight", "reg_med_tight",
            "reg_sx", "reg_sy", "reg_rot", "reg_shear", "reg_kc_ratio", "reg_cons_support", "reg_z", "reg_ncc",
            "reg_n_ex_in", "reg_tight_frac", "reg_n_iv", "reg_n_ex"]


def label_candidates(df: pd.DataFrame, iv_m: dict, ex_m: dict, gt_pairs: np.ndarray) -> pd.DataFrame:
    """y = 1 iff both masks are true positives and their GT cells form a verified pair (the official rule)."""
    if df.empty:
        return df
    ver = {(int(a), int(b)) for a, b in gt_pairs}
    a = iv_m["pr2gt"][df["iv_id"].to_numpy()]
    b = ex_m["pr2gt"][df["ex_id"].to_numpy()]
    df = df.copy()
    verified = np.array([int((int(x), int(y)) in ver) if (x > 0 and y > 0) else 0 for x, y in zip(a, b)])
    df["y"] = ((a > 0) & (b > 0) & (verified > 0)).astype(int)
    return df


def assign(df: pd.DataFrame, p: np.ndarray, thr: float) -> np.ndarray:
    """One-to-one assignment maximising sum(p - thr) over the selected pairs.  Returns (k,2) [iv_id, ex_id]."""
    if df.empty:
        return np.zeros((0, 2), int)
    m = p > thr
    if not m.any():
        return np.zeros((0, 2), int)
    sub = df.loc[m, ["iv_id", "ex_id"]].to_numpy()
    ps = p[m]
    ui, inv_i = np.unique(sub[:, 0], return_inverse=True)
    uj, inv_j = np.unique(sub[:, 1], return_inverse=True)
    C = np.zeros((len(ui), len(uj)))
    C[inv_i, inv_j] = -(ps - thr)
    r, c = linear_sum_assignment(C)
    keep = C[r, c] < 0
    return np.stack([ui[r[keep]], uj[c[keep]]], 1)
