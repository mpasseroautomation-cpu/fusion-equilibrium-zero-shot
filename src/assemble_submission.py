"""ASSEMBLE A SUBMISSION from per-shot predictions: DIII-D file (an existing build) + MAST physics file.

    python src/assemble_submission.py --mast v1_sol0.5 --d3d rung3_k20__mast-zeros --name phys_v1

Writes submissions/<name>/mast_public_test.npz in OFFICIAL STREAM ORDER (sorted filenames), psirz
float16 (never rounded), q95/betaN float32, validated with the official checks; the DIII-D file is
hard-linked from the earlier build (no extra disk). Nothing is pushed or uploaded here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(COMP / "starter"))
import build_submission as bs   # noqa: E402
import data                     # noqa: E402


BV_MAX = 0.06   # tesla: a solve that needed a bigger correction field than this is treated as failed
JUMP_M = 0.04   # metres: axis jump (vs 5-frame running median) treated as a failed frame


def repair(z, times):
    """Frames the solver flagged as failed (or that needed an absurd correction field) are rebuilt by
    linear interpolation in time between the nearest good frames of the same shot. The plasma moves
    smoothly between EFIT times (~5 ms), so a neighbour is a far better guess than a broken solve."""
    psi = np.nan_to_num(z["psirz"].astype(np.float64), nan=0.0)
    q95 = np.asarray(z["q95"], np.float64)
    bN = np.asarray(z["betaN"], np.float64)
    d = z["diag"]
    t = np.asarray(times, float)
    good = (d[:, 0] > 0) & (np.abs(d[:, 3]) < BV_MAX) & np.isfinite(psi).all(axis=(1, 2))
    if len(d) >= 5:
        # a magnetic axis that jumps > JUMP_M from the 5-frame running median is numerical jitter, not physics
        from scipy.ndimage import median_filter
        jr = np.abs(d[:, 4] - median_filter(d[:, 4], 5, mode="mirror")) > JUMP_M
        jz = np.abs(d[:, 5] - median_filter(d[:, 5], 5, mode="mirror")) > JUMP_M
        good &= ~(jr | jz)
    if good.any() and not good.all():
        bad = np.flatnonzero(~good)
        tg = t[good]
        for k in bad:
            j = np.searchsorted(tg, t[k])
            if j == 0:
                w, a, b = 0.0, 0, 0
            elif j >= len(tg):
                w, a, b = 0.0, len(tg) - 1, len(tg) - 1
            else:
                a, b = j - 1, j
                w = (t[k] - tg[a]) / max(tg[b] - tg[a], 1e-9)
            ia, ib = np.flatnonzero(good)[a], np.flatnonzero(good)[b]
            psi[k] = (1 - w) * psi[ia] + w * psi[ib]
            q95[k] = (1 - w) * q95[ia] + w * q95[ib]
            bN[k] = (1 - w) * bN[ia] + w * bN[ib]
    q95 = np.nan_to_num(q95, nan=5.0)
    bN = np.nan_to_num(bN, nan=1.5)
    return psi, q95, bN


def mast_file(pred_dir: Path, role: str = "test", config: str = "mast_public_test") -> dict:
    out = {}
    recs = sorted(data.manifest(role, config), key=lambda r: r["stream_index"])
    n_expected = 1206 if config == "mast_public_test" else len(recs)
    assert [r["stream_index"] for r in recs] == list(range(n_expected)), "MAST cache incomplete"
    missing = []
    for r in recs:
        i = r["stream_index"]
        p = pred_dir / (Path(r["file"]).stem + ".npz")
        if not p.exists():
            missing.append(p.name)
            continue
        z = np.load(p)
        psi, q95, bN = repair(z, data.load(r)["efit_times"])
        out[f"shot_{i:04d}_psirz"] = psi.astype(np.float16)
        out[f"shot_{i:04d}_q95"] = q95.astype(np.float32)
        out[f"shot_{i:04d}_betaN"] = bN.astype(np.float32)
    if missing:
        raise SystemExit(f"{len(missing)} MAST shots not predicted yet, e.g. {missing[:3]}")
    return out


STEPS = """# Submission `{name}` — how to send it

Built {when}. DIII-D = `{d3d}` (the Rung-3 ridge model, 0.925 on 60 held-out DIII-D shots).
MAST = physics pipeline `{mast}` (free-boundary Grad-Shafranov solve, no MAST label used).
MAST config: `{cfg}`
Local validation: {valid}.

From `~/fusion-disruption/competition/starter`:
```
set -a; source ../.env; set +a
../.venv/bin/python push_predictions.py --repo $HF_USERNAME/fusion-eq-predictions --dir ../submissions/{name}
```
-> writes `submission_pointer.zip`. Upload THAT zip at https://www.codabench.org/competitions/17739/ (My Submissions).
Fact sheet: "DIII-D: PCA+ridge on engineered coil/Thomson inputs. MAST: zero-shot physics — free-boundary
Grad-Shafranov solve from coil currents + Ip + Thomson edge, no MAST labels. AI coding assistant (Claude) used, disclosed."
Tick compute-light (trains in seconds on a CPU; the MAST physics needs no training).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mast", required=True, help="cache/mast_pred/<name>")
    ap.add_argument("--d3d", default="rung3_k20__mast-zeros", help="submissions/<build> holding diii_d_public_test.npz")
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="public", choices=["public", "private"])
    a = ap.parse_args()
    dest = COMP / "submissions" / a.name
    dest.mkdir(parents=True, exist_ok=True)
    private = a.split == "private"
    pred_dir = COMP / "cache" / ("mast_pred_final" if private else "mast_pred") / a.mast
    mcfg = "mast_private_test" if private else "mast_public_test"
    dcfg = "diii_d_private_test" if private else "diii_d_public_test"
    t0 = time.time()
    sub = mast_file(pred_dir, "final" if private else "test", mcfg)
    path = dest / f"{mcfg}.npz"
    np.savez_compressed(path, **sub)
    errs = [] if private else bs.validate_local(sub, "mast_public_test")
    hard = [e for e in errs if not e.startswith("WARNING")]
    valid = f"{mcfg}: {len(sub) // 3} shots, {path.stat().st_size / 1e6:.0f} MB, {'OK' if not hard else 'PROBLEMS'}"
    print(valid)
    for e in errs[:10]:
        print("   ", e)
    if hard:
        return 1
    src = COMP / "submissions" / a.d3d / f"{dcfg}.npz"
    tgt = dest / f"{dcfg}.npz"
    if not tgt.exists():
        os.link(src, tgt)
    cfg = (pred_dir / "config.json").read_text() if (pred_dir / "config.json").exists() else "{}"
    (dest / "SUBMIT_STEPS.md").write_text(STEPS.format(
        name=a.name, when=time.strftime("%Y-%m-%d %H:%M"), d3d=a.d3d, mast=a.mast,
        cfg=json.dumps(json.loads(cfg)), valid=valid))
    print(f"assembled {dest} in {time.time() - t0:.0f} s — nothing pushed or uploaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
