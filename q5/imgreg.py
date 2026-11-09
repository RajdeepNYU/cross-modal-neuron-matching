"""Image evidence for a registration hypothesis: z-score of the best IV-EX normalised cross-correlation near the pose
against poses shifted 40-120 px away (background-subtracted, lightly blurred images)."""
from __future__ import annotations

import cv2
import numpy as np

from .prep import RegionData

S_BG, S_SM = 25.0, 3.0


def prep_img(a: np.ndarray) -> np.ndarray:
    x = a.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99.5])
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    return cv2.GaussianBlur(x - cv2.GaussianBlur(x, (0, 0), S_BG), (0, 0), S_SM)


def _ncc(A, B, M, h, w, ones):
    Bw = cv2.warpAffine(B, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderValue=0)
    v = cv2.warpAffine(ones, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST, borderValue=0) > 0
    if v.sum() < 0.3 * h * w:
        return -1.0
    a, b = A[v], Bw[v]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def score_pose(A, B, M, rng, nnull=20):
    """Returns (best NCC over +-4 px translations, z vs far-shifted null)."""
    h, w = A.shape
    ones = np.ones_like(B)
    best = -1.0
    for dx in (-4, -2, 0, 2, 4):
        for dy in (-4, -2, 0, 2, 4):
            Mt = M.copy()
            Mt[0, 2] += dx
            Mt[1, 2] += dy
            best = max(best, _ncc(A, B, Mt, h, w, ones))
    nulls = []
    for _ in range(nnull):
        ang, r = rng.uniform(0, 2 * np.pi), rng.uniform(40, 120)
        Mt = M.copy()
        Mt[0, 2] += r * np.cos(ang)
        Mt[1, 2] += r * np.sin(ang)
        nulls.append(_ncc(A, B, Mt, h, w, ones))
    nulls = np.array(nulls)
    return best, float((best - nulls.mean()) / (nulls.std() + 1e-6))


def dedupe_pool(cands, topk):
    cs = sorted([c for c in cands if c.get("M") is not None and c.get("kc", 0) > 0], key=lambda c: -c["kc"])
    pool = []
    for c in cs:
        if any(np.abs(c["M"] - q["M"]).max() < 0.5 for q in pool):
            continue
        pool.append(c)
        if len(pool) >= topk:
            break
    return pool


def add_image_z(sid: str, cands: list, topk: int = 60, seed: int = 0) -> list:
    """Deduplicate the candidate pool (top `topk` by kc) and attach image evidence z / ncc to every member."""
    pool = dedupe_pool(cands, topk)
    if not pool:
        return []
    rd = RegionData(sid)
    A = prep_img(np.asarray(rd.arr("iv_raw")))
    B = prep_img(np.asarray(rd.arr("ex_raw")))
    rng = np.random.default_rng(seed)
    out = []
    for c in pool:
        c = dict(c)
        c["ncc"], c["z"] = score_pose(A, B, c["M"], rng)
        out.append(c)
    return out

