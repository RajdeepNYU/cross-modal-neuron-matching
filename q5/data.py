"""Dataset access and RLE codec (0-indexed starts, row-major flattening, "start length start length ...")."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi


def rle_decode(rle: str, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    flat = np.zeros(h * w, dtype=bool)
    toks = str(rle).split() if rle is not None else []
    if toks:
        v = np.asarray(toks, dtype=np.int64)
        for s, n in zip(v[0::2], v[1::2]):
            flat[s:s + n] = True
    return flat.reshape(h, w)


def rle_encode(mask: np.ndarray) -> str:
    x = np.asarray(mask, dtype=np.uint8).reshape(-1)
    pad = np.concatenate([[0], x, [0]])
    d = np.flatnonzero(pad[1:] != pad[:-1])
    starts, ends = d[0::2], d[1::2]
    out = np.empty(starts.size * 2, dtype=np.int64)
    out[0::2] = starts
    out[1::2] = ends - starts
    return " ".join(map(str, out.tolist()))


def rle_to_label(obj: dict, shape: tuple[int, int]):
    """{cell_id: rle} -> (int32 label map 1..N, list of ids)."""
    lab = np.zeros(shape, dtype=np.int32)
    ids = []
    for k, r in obj.items():
        m = rle_decode(r, shape)
        if not m.any():
            continue
        ids.append(str(k))
        lab[m] = len(ids)
    return lab, ids


def label_to_rle_dict(lab: np.ndarray, prefix: str) -> dict:
    """Label map (1..N) -> {'<prefix>_%06d': rle}."""
    out = {}
    if lab.max() == 0:
        return out
    h, w = lab.shape
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = np.zeros((h, w), dtype=bool)
        m[sl] = lab[sl] == i
        if m.any():
            out[f"{prefix}_{i:06d}"] = rle_encode(m)
    return out


@dataclass
class Region:
    sample_id: str
    split: str
    subject: str
    region: str
    iv_path: Path
    ex_path: Path


class Dataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.train_dir = self.root / "training"
        self.test_dir = self.root / "hidden_test"
        self.gt_csv = self.train_dir / "train_ground_truth.csv"
        self.sample_csv = self.root / "sample_submission.csv"
        self._gt_df = None

    def train_regions(self) -> list[Region]:
        return [self._mk(sid, "train", self.train_dir) for sid in self.gt_df()["sample_id"]]

    def test_regions(self) -> list[Region]:
        return [self._mk(sid, "test", self.test_dir) for sid in pd.read_csv(self.sample_csv)["sample_id"]]

    def _mk(self, sid, split, base):
        subj, reg = sid.split("__", 1)
        d = base / subj / reg
        return Region(sid, split, subj, reg, d / "invivo.tif", d / "exvivo.tif")

    def gt_df(self) -> pd.DataFrame:
        if self._gt_df is None:
            self._gt_df = pd.read_csv(self.gt_csv).set_index("sample_id", drop=False)
        return self._gt_df

    @staticmethod
    def read_tif(p: Path) -> np.ndarray:
        a = np.squeeze(tifffile.imread(str(p)))
        assert a.ndim == 2, (p, a.shape)
        return a

    def load_images(self, r: Region):
        return self.read_tif(r.iv_path), self.read_tif(r.ex_path)

    def load_gt(self, r: Region, iv_shape, ex_shape):
        row = self.gt_df().loc[r.sample_id]
        iv_lab, iv_ids = rle_to_label(json.loads(row["invivo_instances"]), iv_shape)
        ex_lab, ex_ids = rle_to_label(json.loads(row["exvivo_instances"]), ex_shape)
        iv_idx = {k: i + 1 for i, k in enumerate(iv_ids)}
        ex_idx = {k: i + 1 for i, k in enumerate(ex_ids)}
        pairs = np.array([(iv_idx[a], ex_idx[b]) for a, b in json.loads(row["match_pairs"]) if a in iv_idx and b in ex_idx],
                         dtype=np.int32).reshape(-1, 2)
        return dict(iv_lab=iv_lab, ex_lab=ex_lab, pairs=pairs)


def centroids(lab: np.ndarray) -> np.ndarray:
    """(N,2) (row, col) centroids for labels 1..N."""
    n = int(lab.max())
    if n == 0:
        return np.zeros((0, 2))
    return np.array(ndi.center_of_mass(lab > 0, lab, np.arange(1, n + 1)), dtype=np.float64).reshape(-1, 2)
