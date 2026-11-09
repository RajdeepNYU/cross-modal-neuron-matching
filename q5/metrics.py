"""Competition metric: per-region PQ at IoU > 0.75 (averaged over regions) and pooled pair F1."""
from __future__ import annotations

import numpy as np

IOU_THR = 0.75


def overlaps(gt: np.ndarray, pr: np.ndarray):
    """Sparse (gt_id, pr_id, intersection) triples of two label maps, plus per-label areas."""
    ng, npr = int(gt.max()), int(pr.max())
    ag = np.bincount(gt.ravel(), minlength=ng + 1)
    ap = np.bincount(pr.ravel(), minlength=npr + 1)
    m = (gt > 0) & (pr > 0)
    if not m.any():
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), ag, ap
    key = gt[m].astype(np.int64) * (npr + 1) + pr[m].astype(np.int64)
    uk, cnt = np.unique(key, return_counts=True)
    return uk // (npr + 1), uk % (npr + 1), cnt, ag, ap


def seg_match(gt: np.ndarray, pr: np.ndarray, thr: float = IOU_THR):
    """Greedy one-to-one matching at IoU > thr; returns counts, PQ and id maps gt2pr / pr2gt."""
    ng, npr = int(gt.max()), int(pr.max())
    gi, pi, inter, ag, ap = overlaps(gt, pr)
    iou = inter / np.maximum(ag[gi] + ap[pi] - inter, 1)
    gt2pr = np.zeros(ng + 1, np.int64)
    pr2gt = np.zeros(npr + 1, np.int64)
    pr_iou = np.zeros(npr + 1)
    sum_iou, tp = 0.0, 0
    for k in np.argsort(-iou):
        if iou[k] <= thr:
            break
        g, p = gi[k], pi[k]
        if gt2pr[g] == 0 and pr2gt[p] == 0:
            gt2pr[g], pr2gt[p], pr_iou[p] = p, g, iou[k]
            sum_iou += iou[k]
            tp += 1
    fp, fn = npr - tp, ng - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    return dict(n_gt=ng, n_pr=npr, tp=tp, fp=fp, fn=fn, pq=sum_iou / denom if denom > 0 else 1.0,
                gt2pr=gt2pr, pr2gt=pr2gt, pr_iou=pr_iou)


def pair_correctness(pred_pairs: np.ndarray, gt_pairs: np.ndarray, iv_m: dict, ex_m: dict) -> np.ndarray:
    if len(pred_pairs) == 0:
        return np.zeros(0, bool)
    verified = {(int(a), int(b)) for a, b in gt_pairs}
    out = np.zeros(len(pred_pairs), bool)
    for k, (i, j) in enumerate(pred_pairs):
        a, b = iv_m["pr2gt"][i], ex_m["pr2gt"][j]
        out[k] = a > 0 and b > 0 and (int(a), int(b)) in verified
    return out


def aggregate(results: list[dict]) -> dict:
    pq_iv = float(np.mean([r["iv"]["pq"] for r in results]))
    pq_ex = float(np.mean([r["ex"]["pq"] for r in results]))
    npred = sum(r["n_pred_pairs"] for r in results)
    ncorr = sum(r["n_correct_pairs"] for r in results)
    nver = sum(r["n_verified"] for r in results)
    prec = ncorr / npred if npred else 0.0
    rec = ncorr / nver if nver else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return dict(PQ_iv=pq_iv, PQ_ex=pq_ex, precision=prec, recall=rec, F1=f1, score=float(0.25 * pq_iv + 0.25 * pq_ex + 0.5 * f1),
                n_pred_pairs=int(npred), n_correct_pairs=int(ncorr), n_verified=int(nver))
