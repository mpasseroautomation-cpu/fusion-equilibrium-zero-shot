"""DIII-D ORACLE-FIELD BENCH: how good is the PLASMA part of the solver when the external field is perfect?

External field = true map minus the flux of the true plasma current (recovered with the Grad-Shafranov
operator inside EFIT's LCFS). So coil/solenoid/vessel modelling errors are removed, and what remains
is the solver's own physics: current profile, X-point/boundary, vertical and radial feedback.
Outer edge (and optionally inner edge) from the true LCFS, i.e. a perfect "laser". Scored with the
official scorer on every 4th frame of held-out DIII-D shots (DIII-D labels are allowed).
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gs_solver as gs  # noqa: E402

STRIDE = 4


def sub(shot):
    d = dict(shot)
    # laser profiles smoothed over 5 neighbouring EFIT frames (median, NaN-aware), computed at full time resolution
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for key in ("ts_core_Te", "ts_core_ne"):
            a = shot[key].astype(float)
            st = np.stack([np.roll(a, s_, axis=0) for s_ in (-2, -1, 0, 1, 2)])
            T = a.shape[0]
            idx = np.arange(T)
            for j, s_ in enumerate((-2, -1, 0, 1, 2)):
                bad = (idx - s_ < 0) | (idx - s_ >= T)
                st[j][bad] = np.nan
            d[key + "_s"] = np.nanmedian(st, axis=0)[::STRIDE]
    for k in ("psi", "q95", "betaN", "currents", "ip", "efit_times", "lcfs_n", "lcfs_r", "lcfs_z",
              "r_axis", "z_axis", "li", "ts_core_Te", "ts_core_ne"):
        d[k] = shot[k][::STRIDE]
    d["file"] = shot["file"] + f"_sub{STRIDE}"
    return d


def true_current(phi, geo, inside):
    R = geo.RR
    dpdR = np.gradient(phi, geo.dR, axis=1)
    lap = R * np.gradient(dpdR / R, geo.dR, axis=1) + np.gradient(np.gradient(phi, geo.dZ, axis=0), geo.dZ, axis=0)
    return np.where(inside, np.maximum(-lap / (gs.MU0 * R), 0.0), 0.0)


def predict(args):
    shot, kw = args
    geo1 = gs.Geometry(shot, "DIII-D")                       # official grid (truth lives here)
    ref = int(kw.get("refine", 1))
    geo = gs.Geometry(shot, "DIII-D", refine=ref) if ref > 1 else geo1
    T = len(shot["efit_times"])
    out = np.zeros((T, 65, 65), np.float32)
    J = None
    bv = 0.0
    an = kw.get("alpha_n", 1.0)
    for k in range(T):
        phi_true = -shot["psi"][k].astype(float)
        n = int(shot["lcfs_n"][k])
        poly = np.c_[shot["lcfs_r"][k, :n], shot["lcfs_z"][k, :n]]
        inside = MPath(poly).contains_points(np.c_[geo1.RR.ravel(), geo1.ZZ.ravel()]).reshape(geo1.shape)
        Jt = true_current(phi_true, geo1, inside)
        phi_ext = phi_true - geo1.plasma_flux_fft(np.where(geo1.dom, Jt, 0.0))
        if ref > 1:                                          # the external field is smooth: cubic interpolation
            from scipy.interpolate import RectBivariateSpline
            phi_ext = RectBivariateSpline(geo1.Z, geo1.R, phi_ext, kx=3, ky=3)(geo.Z, geo.R)
        ip = abs(float(shot["ip"][k])) * 1e3
        r_out = float(poly[:, 0].max())
        zrow = np.abs(poly[:, 1] - shot["z_axis"][k]) < 0.15
        r_in = float(poly[zrow, 0].min()) if zrow.any() else float(poly[:, 0].min())
        if J is None:
            J = gs.initial_current(geo, ip, 0.5 * (r_in + r_out), 0.0, 0.5 * (r_out - r_in) * 0.9, 1.7)
            bv, an = 0.0, kw.get("alpha_n", 1.0)
        kin = None
        if kw.get("kinetic"):
            nm = shot["thomson_chord_name"]
            m = np.array([str(x).startswith("TS_core") for x in nm])
            if m.sum() == shot["ts_core_Te"].shape[1]:
                sfx = "_s" if kw.get("ts_smooth") else ""
                kin = dict(R=shot["thomson_chord_R"][m], Z=shot["thomson_chord_Z"][m], Te=shot["ts_core_Te" + sfx][k],
                           ne=shot["ts_core_ne" + sfx][k], c_total=2.42, j_exponent=kw.get("jexp", 1.5))
        phi, info = gs.solve(phi_ext, ip, geo, J, r_out_target=r_out, bv0=bv, return_info=True, kinetic=kin,
                             max_iter=kw.get("max_iter", 100), relax=0.3, tol=3e-4, betap=kw.get("betap", 0.3), alpha_n=an,
                             alpha_m=kw.get("alpha_m", 1.0),
                             radial_gain=0.05, r_in_target=(r_in if kw.get("fit_in") else None),
                             radial_knob=kw.get("knob", "bv"), avg_last=kw.get("avg_last", 0))
        ax_r, ax_z = info["axis"]
        sane = (r_in - 0.05 <= ax_r <= r_out) and abs(ax_z) < 0.5 and info.get("iters", 0) > 0
        if sane:
            J, bv = info["J"], info["bv"]
            an = info.get("alpha_n", an)
        else:                                  # reset: next frame starts from a fresh guess
            J = None
        out[k] = -geo.to_official(phi)
    out = repair_jumps(out, geo1, np.asarray(shot["efit_times"], float))
    return {"psirz": out, "q95": np.asarray(shot["q95"], np.float32), "betaN": np.asarray(shot["betaN"], np.float32)}


def repair_jumps(psi_stored, geo, t, jump=0.04):
    """Same rule as assemble_submission.repair: axis jumps > 4 cm vs the 5-frame running median -> interpolate."""
    from scipy.ndimage import median_filter
    psi = psi_stored.astype(float)
    T = len(psi)
    if T < 5:
        return psi_stored
    ax = np.array([gs.subpixel_axis(-psi[k], *gs.find_axis(-psi[k], geo), geo)[:2] for k in range(T)])
    good = ~((np.abs(ax[:, 0] - median_filter(ax[:, 0], 5, mode="mirror")) > jump)
             | (np.abs(ax[:, 1] - median_filter(ax[:, 1], 5, mode="mirror")) > jump))
    gi = np.flatnonzero(good)
    if gi.size == 0:
        return psi_stored
    out = psi.copy()
    for k in np.flatnonzero(~good):
        j = np.searchsorted(t[gi], t[k])
        if j == 0:
            out[k] = psi[gi[0]]
        elif j >= len(gi):
            out[k] = psi[gi[-1]]
        else:
            a, b = gi[j - 1], gi[j]
            w = (t[k] - t[a]) / max(t[b] - t[a], 1e-9)
            out[k] = (1 - w) * psi[a] + w * psi[b]
    return out.astype(np.float32)


if __name__ == "__main__":
    import argparse
    import scoring
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--betap", type=float, default=0.3)
    ap.add_argument("--an", type=float, default=1.0)
    ap.add_argument("--fit-in", action="store_true")
    ap.add_argument("--knob", default="bv")
    ap.add_argument("--am", type=float, default=1.0)
    ap.add_argument("--kinetic", action="store_true")
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--avg-last", type=int, default=0)
    ap.add_argument("--ts-smooth", action="store_true")
    ap.add_argument("--jexp", type=float, default=1.5)
    a = ap.parse_args()
    shots = [sub(s) for s in scoring.load_eval(a.n)]
    t0 = time.time()
    kw = dict(betap=a.betap, alpha_n=a.an, alpha_m=a.am, fit_in=a.fit_in, knob=a.knob, kinetic=a.kinetic, jexp=a.jexp, refine=a.refine, max_iter=a.max_iter, avg_last=a.avg_last, ts_smooth=a.ts_smooth)
    with ProcessPoolExecutor(a.workers) as pool:
        preds = list(pool.map(predict, [(s, kw) for s in shots]))
    print(f"{kw}: solved {sum(len(s['efit_times']) for s in shots)} frames in {time.time() - t0:.0f} s")
    print(scoring.summary(scoring.score(preds, shots, workers=a.workers)))
