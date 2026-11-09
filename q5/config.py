"""Configuration loading and working-directory paths."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ["Q5_WORK_DIR"]) if os.environ.get("Q5_WORK_DIR") else ROOT


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, data_root: str | None = None) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if data_root:
        cfg["data_root"] = data_root
    return cfg


def cfg_hash(obj, n: int = 8) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


class P:
    cache = WORK / "cache"
    prep = WORK / "cache" / "prep"
    probs = WORK / "cache" / "probs"
    reg = WORK / "cache" / "reg"
    ckpt = WORK / "checkpoints"
    subs = WORK / "submissions"

    @classmethod
    def ckpt_dir(cls, cfg: dict) -> Path:
        return cls.ckpt / cfg.get("ckpt_tag", "cp")

    @classmethod
    def mk(cls):
        for p in [cls.cache, cls.prep, cls.probs, cls.reg, cls.ckpt, cls.subs]:
            p.mkdir(parents=True, exist_ok=True)
