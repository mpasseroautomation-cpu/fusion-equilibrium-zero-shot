"""LABEL-FREE CHECK OF A MAST MAP: does the laser agree with the map? (no MAST equilibrium label used)

Physics: heat runs along field lines ~a million times faster than across them, so the electron
temperature Te is (to a very good approximation) constant on each flux surface. On MAST the laser
crosses the midplane on BOTH sides of the magnetic axis. If our map is right, a point on the inboard
side and a point on the outboard side that sit on the same flux surface (same psiN) see the same Te.

    mismatch = median over psiN in [0.1, 0.9] of |Te_inboard(psiN) - Te_outboard(psiN)| / Te_peak

A wrong Shafranov shift, a wrong plasma size or a wrong current profile all show up as a mismatch.
We also report psiN at the laser's inboard edge (where Te crosses the skin temperature): ~1 if the
map's inboard boundary is where the laser says it is.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

import gs_solver as gs


def frame_check(phi, geo, R_ts, Te, te_sep=40.0):
    ok = np.isfinite(Te) & (Te > 0) & np.isfinite(R_ts)
    if ok.sum() < 20:
        return np.nan, np.nan
    o = np.argsort(R_ts[ok])
    R = R_ts[ok][o]
    T = median_filter(Te[ok][o], 5, mode="nearest")
    iz, ir = gs.find_axis(phi, geo)
    Ra, Za, pa = gs.subpixel_axis(phi, iz, ir, geo)
    lvl = gs.boundary_level(phi, geo, iz, ir)
    if pa - lvl <= 0:
        return np.nan, np.nan
    iz0 = int(np.argmin(np.abs(geo.Z)))           # the MAST laser is on Z = 0
    psiN = (pa - np.interp(R, geo.R, phi[iz0])) / (pa - lvl)
    tpk = T[(R > Ra - 0.15) & (R < Ra + 0.15)]
    if tpk.size == 0:
        return np.nan, np.nan
    tpk = float(np.max(tpk))
    inb = (R < Ra) & (psiN > 0.1) & (psiN < 0.9)
    outb = (R > Ra) & (psiN > 0.05) & (psiN < 0.95)
    if inb.sum() < 5 or outb.sum() < 5:
        mism = np.nan
    else:
        xo, to = psiN[outb], T[outb]
        s = np.argsort(xo)
        t_out = np.interp(psiN[inb], xo[s], to[s])
        mism = float(np.median(np.abs(T[inb] - t_out)) / tpk)
    # psiN where the inboard Te first drops below the skin temperature
    inside = np.flatnonzero((R < Ra) & (T >= te_sep))
    edge_psin = float(psiN[inside.min()]) if inside.size else np.nan
    return mism, edge_psin


def shot_check(pred_psi, shot, geo, stride=3):
    R_ts = np.concatenate([shot["ts_core_R"], shot["ts_edge_spatial"]])
    out = []
    for k in range(0, pred_psi.shape[0], stride):
        Te = np.concatenate([shot["ts_core_Te"][k], shot["ts_edge_Te"][k]]).astype(float)
        out.append(frame_check(pred_psi[k].astype(float), geo, R_ts, Te))
    a = np.array(out, float)
    return np.nanmedian(a[:, 0]), np.nanmedian(a[:, 1])
