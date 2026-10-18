"""KINETIC MEASUREMENTS -> PLASMA GEOMETRY: what the Thomson laser says about where the plasma is.

The laser measures electron temperature Te and density ne at points along a chord. Inside the plasma
Te is hundreds of eV; outside the last closed flux surface it collapses to a few eV within ~2 cm.
So the place where Te crosses the "skin temperature" marks the plasma's edge on that chord.

Skin temperature, calibrated on DIII-D (labels allowed there): Te at EFIT's LCFS where the vertical
core chord crosses it, 6,263 frames of 200 training shots -> median 40 eV (IQR 28-59 eV);
ne there is 0.22 of the profile maximum (IQR 0.15-0.29). No MAST equilibrium was used.

On MAST the core laser lies on the midplane (Z = 0), from R = 0.23 to 1.52 m, plus an edge system
(R 1.30-1.48 m), so it gives the plasma's inboard and outboard edges on the midplane directly.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

TE_SEP_EV = 40.0       # DIII-D-calibrated median Te at the LCFS


def _clean_profile(R, Te, ne):
    """Valid points sorted by R, Te despiked with a 5-point running median."""
    R = np.asarray(R, float)
    Te = np.asarray(Te, float)
    ne = np.asarray(ne, float)
    ok = np.isfinite(R) & np.isfinite(Te) & np.isfinite(ne) & (Te > 0) & (ne > 0)
    if ok.sum() < 8:
        return None
    o = np.argsort(R[ok])
    R, Te, ne = R[ok][o], Te[ok][o], ne[ok][o]
    Tm = median_filter(Te, size=5, mode="nearest")
    nm = median_filter(ne, size=5, mode="nearest")
    return R, Tm, nm


def _crossing(R, y, thr, start, step):
    """Walk from index `start` in direction `step` until y < thr; return the interpolated R (or None)."""
    i = start
    while 0 <= i + step < len(R):
        j = i + step
        if y[j] < thr <= y[i]:
            t = (y[i] - thr) / (y[i] - y[j])
            return R[i] + t * (R[j] - R[i])
        i = j
    return None


def mast_midplane_edges(shot: dict, k: int, te_sep: float = TE_SEP_EV):
    """(R_in, R_out, R_Te_peak) of the plasma on the MAST midplane at frame k, from core + edge Thomson.

    Returns NaN for an edge the laser does not reach (then the caller falls back to another estimate).
    """
    R = np.concatenate([shot["ts_core_R"], shot["ts_edge_spatial"]])
    Te = np.concatenate([shot["ts_core_Te"][k], shot["ts_edge_Te"][k]])
    ne = np.concatenate([shot["ts_core_ne"][k], shot["ts_edge_ne"][k]])
    cp = _clean_profile(R, Te, ne)
    if cp is None:
        return np.nan, np.nan, np.nan
    R, Tm, nm = cp
    # the hot core: the Te peak within the central part of the chord
    core = (R > 0.5) & (R < 1.3)
    if not core.any():
        return np.nan, np.nan, np.nan
    ipk = int(np.flatnonzero(core)[np.argmax(Tm[core])])
    thr = min(te_sep, 0.5 * Tm[ipk])
    r_out = _crossing(R, Tm, thr, ipk, +1)
    r_in = _crossing(R, Tm, thr, ipk, -1)
    if r_out is None:   # hot all the way to the last channel: the edge is at/after the last good point
        r_out = np.nan
    if r_in is None:
        r_in = np.nan
    return float(r_in), float(r_out), float(R[ipk])


def innermost_channel(shot: dict, k: int) -> float:
    """R (m) of the innermost valid laser point at frame k: where the laser's view of the inboard side ends."""
    R = np.concatenate([shot["ts_core_R"], shot["ts_edge_spatial"]])
    Te = np.concatenate([shot["ts_core_Te"][k], shot["ts_edge_Te"][k]])
    ne = np.concatenate([shot["ts_core_ne"][k], shot["ts_edge_ne"][k]])
    cp = _clean_profile(R, Te, ne)
    return float(cp[0][0]) if cp is not None else np.nan


def mast_edges_series(shot: dict, te_sep: float = TE_SEP_EV):
    """Per-frame midplane edges, gaps filled by interpolation in time (the plasma moves smoothly)."""
    T = len(shot["efit_times"])
    out = np.array([mast_midplane_edges(shot, k, te_sep) for k in range(T)], float)
    t = np.asarray(shot["efit_times"], float)
    for j in range(3):
        v = out[:, j]
        ok = np.isfinite(v)
        if ok.sum() >= 1:
            out[:, j] = np.interp(t, t[ok], v[ok])
        # light smoothing in time removes single-frame laser glitches
        if T >= 5:
            out[:, j] = median_filter(out[:, j], size=3, mode="nearest")
    return out   # (T, 3): r_in, r_out, r_te_peak
