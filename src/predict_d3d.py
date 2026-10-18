"""DIII-D PREDICTION FILE for any split, from a saved model (default: the Rung-3 ridge model).

    python src/predict_d3d.py --split private --name final_v1      # -> submissions/final_v1/diii_d_private_test.npz
    python src/predict_d3d.py --split public  --name check          # -> submissions/check/diii_d_public_test.npz

Shots are written in OFFICIAL STREAM ORDER (sorted filenames, stream_index from the cache manifest).
psirz float16 (never rounded), q95/betaN float32, every value finite. Nothing is pushed or uploaded.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMP = HERE.parent
sys.path.insert(0, str(HERE))
import build_submission as bs   # noqa: E402
import data                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rung3_k20")
    ap.add_argument("--split", default="private", choices=["public", "private"])
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    role, config = ("test", "diii_d_public_test") if a.split == "public" else ("final", "diii_d_private_test")
    model = bs.load_model(a.model)
    shots = data.load_role(role, config=config, sort_by_stream=True)
    idx = [s["stream_index"] for s in shots]
    assert idx == list(range(len(shots))), f"{config}: cache incomplete or out of order"
    t0 = time.time()
    out = {}
    for s in shots:
        i, T = s["stream_index"], len(s["efit_times"])
        p = model.predict_shot(s)
        psi = np.asarray(p["psirz"], np.float32)
        assert psi.shape == (T, 65, 65) and np.isfinite(psi).all(), (i, psi.shape)
        out[f"shot_{i:04d}_psirz"] = psi.astype(np.float16)
        for k in ("q95", "betaN"):
            v = np.asarray(p[k], np.float32)
            assert v.shape == (T,) and np.isfinite(v).all(), (i, k)
            out[f"shot_{i:04d}_{k}"] = v
    dest = COMP / "submissions" / a.name
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{config}.npz"
    np.savez_compressed(path, **out)
    print(f"{config}: {len(shots)} shots -> {path} ({path.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
