"""PRE-FLIGHT CHECK of a MAST prediction run before it is assembled and sent.

Reads cache/mast_pred/<name>/*.npz and reports: how many shots/frames, how many frames the solver
judged sane / converged, the spread of q95 and beta_N, any NaN/inf, maps that are flat or absurd.
No MAST label is read.
"""
import glob
import sys
from pathlib import Path

import numpy as np

name = sys.argv[1]
fs = sorted(glob.glob(str(Path(__file__).resolve().parent.parent / "cache" / "mast_pred" / name / "mast_*.npz")))
T = 0
sane = conv = 0
q = []
b = []
bad = []
amp = []
bv = []
for f in fs:
    z = np.load(f)
    p, d = z["psirz"], z["diag"]
    T += len(d)
    sane += d[:, 0].sum()
    conv += d[:, 1].sum()
    q.append(z["q95"])
    b.append(z["betaN"])
    bv.append(d[:, 3])
    if not np.isfinite(p).all() or not np.isfinite(z["q95"]).all() or not np.isfinite(z["betaN"]).all():
        bad.append(Path(f).name)
    amp.append(p.max(axis=(1, 2)) - p.min(axis=(1, 2)))
q = np.concatenate(q)
b = np.concatenate(b)
amp = np.concatenate(amp)
bv = np.concatenate(bv)
pct = lambda a: np.percentile(a, [1, 5, 50, 95, 99]).round(3)
print(f"{name}: {len(fs)} shots, {T} frames | sane {sane / T:.3f} | converged {conv / T:.3f}")
print("q95   pct 1/5/50/95/99:", pct(q))
print("betaN pct 1/5/50/95/99:", pct(b))
print("map peak-to-peak (Wb/rad):", pct(amp))
print("correction field Bv (T):", pct(bv))
print("files with NaN/inf:", len(bad), bad[:5])
