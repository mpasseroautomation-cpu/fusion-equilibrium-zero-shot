"""LABEL-FREE SCAN of MAST pipeline settings: which settings make the map agree best with the laser?

For each setting, solve a fixed sample of MAST moments and report (no MAST label read):
  Te-mismatch   inboard vs outboard Te on the same flux surface (labelfree.py), lower = better
  edge psiN     where the laser's inboard edge lands on our map (1.000 = on our plasma skin)
  dR_in         model inner edge minus laser inner edge (m)
  Bv            median correction field the solver needed (T), ~0 = coils + profile already consistent
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMP = HERE.parent
sys.path.insert(0, str(HERE))


def _sub(s, st=3):
    d = dict(s)
    for k in ("currents", "ip", "efit_times", "ts_core_Te", "ts_core_ne", "ts_edge_Te", "ts_edge_ne",
              "current_slopes", "ip_slope"):
        d[k] = s[k][::st]
    return d


def job(args):
    f, cfgd = args
    os.environ["OMP_NUM_THREADS"] = "1"
    import data
    import kinetic as kin
    import labelfree as lf
    import mast_physics as mp
    s = _sub(data.load(Path(f)))
    cfg = mp.Config(**cfgd)
    p = mp.predict_shot(s, cfg)
    geo = mp.geometry(s)
    m, e = lf.shot_check(p["psirz"], s, geo, stride=1)
    d = p["diag"]
    E = kin.mast_edges_series(s, cfg.te_sep)
    return m, e, np.nanmedian(d[:, 6] - E[:, 0]), d[:, 0].mean(), np.median(d[:, 3])


if __name__ == "__main__":
    import glob
    grid = json.loads(sys.argv[1])          # e.g. '{"betap":[0.1,0.3],"alpha_n":[0.5,1,2]}'
    base = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    workers = int(os.environ.get("SCAN_WORKERS", "2"))
    fs = sorted(glob.glob(str(COMP / "cache" / "test" / "mast_*.npz")))[100:1200:90][:12]
    keys = list(grid)
    with ProcessPoolExecutor(workers) as pool:
        for vals in itertools.product(*[grid[k] for k in keys]):
            c = dict(base, **dict(zip(keys, vals)))
            t = time.time()
            r = np.array(list(pool.map(job, [(f, c) for f in fs])))
            print(json.dumps(c), "| Te-mism %.3f | edge psiN %.3f | dR_in %+.3f | sane %.2f | Bv %+.4f (%.0fs)" % (
                np.nanmedian(r[:, 0]), np.nanmedian(r[:, 1]), np.nanmedian(r[:, 2]), r[:, 3].mean(),
                np.median(r[:, 4]), time.time() - t), flush=True)
