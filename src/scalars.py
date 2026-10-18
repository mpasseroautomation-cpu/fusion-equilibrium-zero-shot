"""THE TWO SUBMITTED SCALARS, from physics: q95 by following field lines on the map, beta_N from the laser.

q95 (safety factor at 95% of the flux): follow a field line on the flux surface psiN = 0.95 and count
how many times it goes the long way round per poloidal turn:
    q = (F / 2 pi) x  loop-integral of  dl / (R^2 x Bp)          F = R x B_toroidal = mu0 N I_TF / (2 pi)
    Bp = |grad psi| / R  (psi in Wb/rad)
Only the map and the toroidal-field coil current go in. Nothing is learned.

beta_N (how hard the plasma pushes on its magnetic bottle, normalised):
    beta_N = beta_t[%] x a[m] x B0[T] / Ip[MA],    beta_t = 2 mu0 <p> / B0^2
<p> = volume average of the pressure. The laser gives the ELECTRON pressure ne x Te along its chord; we
put each laser point on its flux surface (psiN from OUR map), spread that profile over the plasma volume,
and multiply by C_TOTAL = 2.42 for the ions and fast particles the laser cannot see.
C_TOTAL is one number fitted on DIII-D (EFIT beta_N vs the same electron estimate from the true DIII-D
maps, 3,810 frames of 150 training shots: R2 = 0.897, correlation 0.947). No MAST label is used.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import median_filter
from skimage import measure

MU0 = 4e-7 * np.pi
E_CHARGE = 1.602176634e-19
C_TOTAL = 2.42
TF_TURNS = {"DIII-D": 144, "MAST": 24}
TF_COLUMN = {"DIII-D": "magnetics_bcoil", "MAST": "magnetics_tf_current"}
BETA_REF_R = {"DIII-D": 1.6955, "MAST": None}   # None -> geometric centre of the plasma


def f_vacuum(tf_kA: float, machine: str) -> float:
    """F = R x B_toroidal (T m) of the vacuum toroidal field."""
    return MU0 * TF_TURNS[machine] * abs(tf_kA) * 1e3 / (2 * np.pi)


def flux_surface(psiN, R, Z, level, axis_rz):
    """The closed contour of psiN at `level` that encloses the axis, as (N, 2) (R, Z), or None."""
    best = None
    for c in measure.find_contours(psiN, level):
        if len(c) < 8 or np.hypot(*(c[0] - c[-1])) > 1.5:
            continue
        rr = np.interp(c[:, 1], np.arange(len(R)), R)
        zz = np.interp(c[:, 0], np.arange(len(Z)), Z)
        poly = np.c_[rr, zz]
        from matplotlib.path import Path as MPath
        if MPath(poly).contains_point(axis_rz):
            if best is None or len(poly) > len(best):
                best = poly
    return best


def q_on_surface(psi, R, Z, poly, F):
    """q = F/(2 pi) loop-integral dl / (R^2 Bp) on the polygon `poly`."""
    gZ, gR = np.gradient(psi, Z, R)
    Bp_grid = np.hypot(gR, gZ) / np.maximum(R[None, :], 1e-3)
    f = RegularGridInterpolator((Z, R), Bp_grid, bounds_error=False, fill_value=None)
    P = np.vstack([poly, poly[:1]])
    mid = 0.5 * (P[1:] + P[:-1])
    dl = np.hypot(*np.diff(P, axis=0).T)
    Bp = np.maximum(f(np.c_[mid[:, 1], mid[:, 0]]), 1e-6)
    return float(F / (2 * np.pi) * np.sum(dl / (mid[:, 0] ** 2 * Bp)))


def q95_from_map(psi_axis_max, R, Z, axis_rz, psi_axis, psi_bnd, F, level=0.95):
    """q95 from a flux map whose axis is a MAXIMUM (phi convention)."""
    dpsi = psi_axis - psi_bnd
    if not np.isfinite(dpsi) or dpsi <= 0:
        return np.nan
    psiN = (psi_axis - psi_axis_max) / dpsi
    poly = flux_surface(psiN, R, Z, level, axis_rz)
    if poly is None:
        return np.nan
    return q_on_surface(psi_axis_max, R, Z, poly, F)


def electron_pressure_profile(R_pts, Z_pts, Te, ne, psiN_at):
    """(psiN, pe) of the laser points inside the plasma, despiked, sorted by psiN."""
    R_pts = np.asarray(R_pts, float)
    Z_pts = np.asarray(Z_pts, float)
    Te = np.asarray(Te, float)
    ne = np.asarray(ne, float)
    ok = np.isfinite(Te) & np.isfinite(ne) & (Te > 0) & (ne > 0) & np.isfinite(R_pts)
    if ok.sum() < 8:
        return None
    o = np.argsort(R_pts[ok] + Z_pts[ok])
    tt = median_filter(Te[ok][o], 5, mode="nearest")
    nn = median_filter(ne[ok][o], 5, mode="nearest")
    x = psiN_at(R_pts[ok][o], Z_pts[ok][o])
    pe = nn * tt * E_CHARGE
    sel = np.isfinite(x) & (x >= 0) & (x < 1)
    if sel.sum() < 6:
        return None
    oo = np.argsort(x[sel])
    return x[sel][oo], pe[sel][oo]


def volume_average(psiN_grid, mask, RR, dA, prof):
    """<p>_V of a profile p(psiN) over the plasma region, and the volume V."""
    x, y = prof
    # monotone envelope: average points in psiN bins so inboard/outboard scatter is smoothed
    bins = np.linspace(0, 1, 11)
    bx, by = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (x >= a) & (x < b)
        if m.any():
            bx.append(0.5 * (a + b))
            by.append(np.median(y[m]))
    if len(bx) < 3:
        return np.nan, np.nan
    bx = np.r_[bx, 1.0]
    by = np.r_[by, 0.0]
    p = np.interp(np.clip(psiN_grid, 0, 1), bx, by)
    dV = 2 * np.pi * RR * dA
    V = float(dV[mask].sum())
    return float((p * dV)[mask].sum() / V), V
