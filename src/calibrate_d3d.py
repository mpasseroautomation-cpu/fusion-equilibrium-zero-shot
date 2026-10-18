"""THE TWO DIII-D CALIBRATIONS used by the MAST physics pipeline (DIII-D labels only; no MAST data).

  1. TE_SEP  — the electron temperature at the last closed flux surface. On DIII-D the vertical core
     Thomson chord (R = 1.94 m) crosses EFIT's LCFS near the top of the plasma; we interpolate the
     measured Te there. Median over frames -> kinetic.TE_SEP_EV (40 eV on 200 training shots).
  2. C_TOTAL — total plasma pressure / electron pressure, as seen by beta_N. Electron beta_N from the
     Thomson pressure mapped onto the TRUE DIII-D flux surfaces, volume-averaged; C = median of
     EFIT beta_N / electron beta_N -> scalars.C_TOTAL (2.42 on 150 training shots, R2 0.897).

    python src/calibrate_d3d.py --n-te 200 --n-beta 150
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402

MU0 = 4e-7 * np.pi
E = 1.602176634e-19


def core_chord(s):
    nm = s["thomson_chord_name"]
    m = np.array([str(x).startswith("TS_core") for x in nm])
    return s["thomson_chord_R"][m], s["thomson_chord_Z"][m]


def te_sep(shots, stride=7):
    tes = []
    for s in shots:
        Rc, Zc = core_chord(s)
        if len(Zc) != s["ts_core_Te"].shape[1]:
            continue
        o = np.argsort(Zc)
        Zc = Zc[o]
        for k in range(0, len(s["efit_times"]), stride):
            n = int(s["lcfs_n"][k])
            if n < 10:
                continue
            r = np.append(s["lcfs_r"][k, :n], s["lcfs_r"][k, 0])
            z = np.append(s["lcfs_z"][k, :n], s["lcfs_z"][k, 0])
            zc = []
            for i in range(n):
                if (r[i] - 1.94) * (r[i + 1] - 1.94) < 0:
                    t = (1.94 - r[i]) / (r[i + 1] - r[i])
                    zc.append(z[i] + t * (z[i + 1] - z[i]))
            zc = [q for q in zc if q > 0]
            if not zc:
                continue
            zt = max(zc)
            te = s["ts_core_Te"][k][o].astype(float)
            ok = np.isfinite(te) & (te > 0)
            if ok.sum() < 10 or not (Zc[ok].min() <= zt <= Zc[ok].max()):
                continue
            tes.append(np.interp(zt, Zc[ok], te[ok]))
    return np.array(tes)


def c_total(shots, stride=9):
    rows = []
    for s in shots:
        Rc, Zc = core_chord(s)
        if len(Zc) != s["ts_core_Te"].shape[1]:
            continue
        Rg, Zg = s["grid_R"], s["grid_Z"]
        RR, ZZ = np.meshgrid(Rg, Zg)
        dA = (Rg[1] - Rg[0]) * (Zg[1] - Zg[0])
        names = [str(c) for c in s["current_names"]]
        ib = names.index("magnetics_bcoil")
        for k in range(0, len(s["efit_times"]), stride):
            n = int(s["lcfs_n"][k])
            if n < 10:
                continue
            poly = np.c_[s["lcfs_r"][k, :n], s["lcfs_z"][k, :n]]
            inside = MPath(poly).contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)
            psi = -s["psi"][k].astype(float)
            iz, ir = np.unravel_index(np.argmax(np.where(inside, psi, -np.inf)), psi.shape)
            pa = psi[iz, ir]
            f = RegularGridInterpolator((Zg, Rg), psi)
            pb = float(np.mean(f(np.c_[poly[:, 1], poly[:, 0]])))
            psiN = (pa - psi) / (pa - pb)
            te = s["ts_core_Te"][k].astype(float)
            ne = s["ts_core_ne"][k].astype(float)
            ok = np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
            if ok.sum() < 10:
                continue
            oo = np.argsort(Zc[ok])
            tt = median_filter(te[ok][oo], 5, mode="nearest")
            nn = median_filter(ne[ok][oo], 5, mode="nearest")
            x = (pa - f(np.c_[Zc[ok][oo], Rc[ok][oo]])) / (pa - pb)
            pe = nn * tt * E
            sel = (x >= 0) & (x < 1.0)
            if sel.sum() < 8:
                continue
            so = np.argsort(x[sel])
            pe_grid = np.interp(np.clip(psiN, 0, 1), x[sel][so], pe[sel][so], left=pe[sel][so][0], right=0.0)
            dV = 2 * np.pi * RR * dA
            pav = (pe_grid * dV)[inside].sum() / dV[inside].sum()
            a = 0.5 * (poly[:, 0].max() - poly[:, 0].min())
            B0 = MU0 * 144 * abs(float(s["currents"][k, ib])) * 1e3 / (2 * np.pi * 1.6955)
            Ip = abs(float(s["ip"][k])) / 1e3
            bNe = 100 * 2 * MU0 * pav / B0 ** 2 * a * B0 / Ip
            rows.append((float(s["betaN"][k]), bNe))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-te", type=int, default=200)
    ap.add_argument("--n-beta", type=int, default=150)
    a = ap.parse_args()
    shots = data.load_role("train", n=max(a.n_te, a.n_beta))
    t = te_sep(shots[:a.n_te])
    print(f"TE_SEP: {len(t)} frames, median {np.median(t):.1f} eV (IQR {np.percentile(t, 25):.1f}-{np.percentile(t, 75):.1f})")
    r = c_total(shots[:a.n_beta])
    good = r[:, 1] < 3.0                               # drop broken-laser frames (electron beta_N > 3)
    c = float(np.median(r[good, 0] / r[good, 1]))
    pred = c * r[good, 1]
    r2 = 1 - ((r[good, 0] - pred) ** 2).sum() / ((r[good, 0] - r[good, 0].mean()) ** 2).sum()
    print(f"C_TOTAL: {good.sum()} frames ({(~good).sum()} dropped), C = {c:.3f}, R2 of C x electron beta_N vs EFIT = {r2:.3f}")


if __name__ == "__main__":
    main()
