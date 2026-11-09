"""CS-GY 6643 Project 2, Question 5: cell segmentation and in-vivo / ex-vivo cell matching.

  python run_pipeline.py --stage <stage> --config <config.json> [options]

stages
  prepare    decode the dataset into per-region caches and subject-level folds          (CPU, ~2 min)
  cp_train   fine-tune Cellpose-SAM for one model family (--modality, --folds)           (GPU, ~20 min per model)
  cp_infer   held-out + test Cellpose outputs of a model family (--modality, --folds)    (GPU)
  cp_blend   average the cached outputs of several families (--tags A B --out-tag C)     (CPU)
  tune_seg   tune the Cellpose decoding thresholds on held-out maps (--out <config>)     (CPU, optional)
  evaluate   held-out (leave-one-mouse-out) score of the full pipeline                   (CPU)
  submit     evaluate + predict the hidden test set + write and validate the CSV (--out) (CPU + light GPU)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["prepare", "cp_train", "cp_infer", "cp_blend", "tune_seg", "evaluate", "submit"])
    ap.add_argument("--config", default="configs/final.json")
    ap.add_argument("--data-root", default=None, help="dataset folder (overrides data_root in the config)")
    ap.add_argument("--modality", nargs="+", default=["iv", "ex"])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--tags", nargs="+", default=None)
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def stage_cp_infer(cfg, a):
    from q5.cellpose_seg import cp_infer_stage
    from q5.cellpose_train import model_path

    for mod in a.modality:
        models = {f: str(model_path(cfg, mod, f)) for f in a.folds if model_path(cfg, mod, f).exists()}
        if not models:
            sys.exit(f"no trained {cfg['cp_tag']} {mod} models found; run cp_train first")
        print(f"[cp_infer] {cfg['cp_tag']} {mod}: fold models {sorted(models)}")
        cp_infer_stage(cfg["cp_tag"], mod, float(cfg["seg"][mod]["diam"]), models, force=a.force)


def stage_cp_blend(a):
    import numpy as np

    from q5.config import P

    if not a.tags or not a.out_tag:
        sys.exit("usage: --stage cp_blend --tags ft1 ft6 --out-tag ftmix6 [--modality iv]")
    w = 1.0 / len(a.tags)
    n = 0
    for kind in ("oof", "test"):
        out = P.probs / a.out_tag / kind
        out.mkdir(parents=True, exist_ok=True)
        for mod in a.modality:
            for f in sorted((P.probs / a.tags[0] / kind).glob(f"*__{mod}.cp.npz")):
                parts = [P.probs / t / kind / f.name for t in a.tags]
                if not all(p.exists() for p in parts):
                    continue
                zs = [np.load(p) for p in parts]
                dP = sum(w * z["dP"].astype(np.float32) for z in zs)
                cp = sum(w * z["cp"].astype(np.float32) for z in zs)
                np.savez(out / f.name, dP=dP.astype(np.float16), cp=cp.astype(np.float16))
                n += 1
    print(f"[cp_blend] wrote {n} blended outputs to tag {a.out_tag}")


def stage_submit(cfg, a):
    from q5.config import P
    from q5.data import Dataset
    from q5.pipeline import evaluate
    from q5.submit import predict_test, validate_submission, write_submission

    ev = evaluate(cfg, workers=a.workers)
    scale = float(cfg.get("output", {}).get("pair_thr_scale", 1.0))
    thr = float(min(max(ev["summary"]["thr"] * scale, 0.02), 0.95))
    pred = predict_test(cfg, ev, thr, workers=a.workers)
    out = Path(a.out) if a.out else P.subs / "submission.csv"
    if not out.is_absolute():
        out = Path.cwd() / out
    write_submission(pred, out, Dataset(cfg["data_root"]).sample_csv)
    rep = validate_submission(out, cfg["data_root"])
    print(f"\nSUBMISSION  {out}  [{'VALID' if rep['valid'] else 'INVALID - DO NOT SUBMIT'}]  sha256 {rep['sha256'][:16]}")


def main():
    a = parse()
    from q5.config import P, load_config

    cfg = load_config(a.config, a.data_root)
    P.mk()
    if a.stage == "prepare":
        from q5.prep import prepare

        df = prepare(cfg, workers=a.workers, force=a.force)
        print(f"[prepare] {int((df.split == 'train').sum())} training / {int((df.split == 'test').sum())} test regions cached")
    elif a.stage == "cp_train":
        from q5.cellpose_train import finetune

        for mod in a.modality:
            for f in a.folds:
                finetune(cfg, mod, f)
    elif a.stage == "cp_infer":
        stage_cp_infer(cfg, a)
    elif a.stage == "cp_blend":
        stage_cp_blend(a)
    elif a.stage == "tune_seg":
        from q5.tune import tune_seg

        tune_seg(cfg, tuple(a.modality), workers=a.workers, out_cfg=a.out)
    elif a.stage == "evaluate":
        from q5.pipeline import evaluate

        evaluate(cfg, workers=a.workers)
    elif a.stage == "submit":
        stage_submit(cfg, a)


if __name__ == "__main__":
    main()
