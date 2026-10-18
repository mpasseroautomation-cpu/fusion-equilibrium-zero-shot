"""MAST ZERO-SHOT PIPELINE: every MAST flux map from physics, with no MAST equilibrium label anywhere.

Per frame:
  1. coil flux, exact (Green's functions of the 812 shipped conductor elements x their currents)
  2. the plasma's outboard edge on the midplane, from the Thomson laser (Te crosses the 40 eV skin
     temperature calibrated on DIII-D)
  3. a free-boundary Grad-Shafranov solve (gs_solver.solve): the plasma current Ip is placed on the
     flux surfaces it creates, held vertically by force balance with the coils, and radially at the
     laser's edge by a small uniform vertical field (standing for unmeasured vessel currents)
  4. q95 by following field lines on that map (scalars.q95_from_map), beta_N from the laser's
     electron pressure mapped onto that map's flux surfaces x 2.42 (fitted on DIII-D)

Every knob is in Config and is recorded with each run. Inputs read: coil currents, Ip, TF current,
Thomson Te/ne, machine geometry. Nothing else.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gs_solver as gs        # noqa: E402
import kinetic as kin         # noqa: E402
import scalars as sc          # noqa: E402

MU0 = gs.MU0


@dataclasses.dataclass
class Config:
    sol_factor: float = 0.5        # multiplier on magnetics_sol_current (the "reads 2x" question)
    te_sep: float = kin.TE_SEP_EV  # skin temperature for the Thomson edge (eV)
    betap: float = 0.3             # poloidal beta of the current profile
    alpha_m: float = 1.0
    alpha_n: float = 1.0           # current peaking: J ~ (1 - psiN^alpha_m)^alpha_n
    relax: float = 0.3
    max_iter: int = 100
    avg_last: int = 0              # average the last iterates (removes the solver's oscillation)
    laser_row: bool = False        # read the model's edges on Z = 0, where MAST's laser looks (fix 2026-10-13)
    hold_z: bool = False           # vertical control: P6-pair correction current holds the axis at Z = 0
    radial_gain: float = 0.05
    tol: float = 3e-4
    c_total: float = sc.C_TOTAL
    tf_turns: int = 24
    r_out_default: float = 1.40    # only if the laser sees no edge in the whole shot
    r_in_default: float = 0.27
    fit_inner_edge: bool = False   # adjust the current peaking per frame so the inner edge matches the laser
    inner_one_sided: bool = False  # fit only where the laser SAW the inner edge; where it is hot down to its
                                   # innermost channel, only require the edge to lie inside it (fix 2026-10-13:
                                   # v12 filled 27% of frames with an inner edge interpolated from later frames)
    kinetic: bool = False          # current profile from the laser: p = C ne Te, rest ~ Te^1.5 (Spitzer)
    j_exponent: float = 1.5
    jexp_early: float = 0.5        # current diffusion (off unless jexp_t1 > jexp_t0): the Spitzer exponent is
    jexp_t0: float = 0.0           # jexp_early before jexp_t0 (ms after the MAST t = 0) and rises linearly to
    jexp_t1: float = 0.0           # j_exponent at jexp_t1 (label-free fit to the laser's inboard edge, 2026-10-16)
    peaking_gain: float = 1.5
    sane_width: float = 0.5        # a solve is "sane" only if the plasma is at least this fraction of the laser's width
    holdz_retry: bool = False      # if all free-height attempts fail, one more attempt with vertical control at Z = 0

    def tag(self) -> str:
        return (f"sol{self.sol_factor:g}_te{self.te_sep:g}_bp{self.betap:g}_an{self.alpha_n:g}"
                f"_am{self.alpha_m:g}_c{self.c_total:g}" + ("_fitin" if self.fit_inner_edge else "")
                + ("1s" if self.fit_inner_edge and self.inner_one_sided else "")
                + (f"_kin{self.j_exponent:g}" if self.kinetic else "")
                + (f"_ramp{self.jexp_early:g}-{self.jexp_t0:g}-{self.jexp_t1:g}"
                   if self.kinetic and self.jexp_t1 > self.jexp_t0 else "")
                + (f"_it{self.max_iter}avg{self.avg_last}" if self.avg_last else "")
                + ("_z0" if self.laser_row else "") + ("_holdz" if self.hold_z else ""))


_GEO = {}


def geometry(shot):
    if "MAST" not in _GEO:
        _GEO["MAST"] = gs.Geometry(shot, "MAST")
    return _GEO["MAST"]


def _region(phi, geo, near=None):
    iz, ir = gs.find_axis(phi, geo, near)
    Ra, Za, pa = gs.subpixel_axis(phi, iz, ir, geo)
    lvl = gs.boundary_level(phi, geo, iz, ir)
    for (rx, zx, vx, _, _) in gs.find_xpoints(phi, geo):
        if abs(vx - lvl) < 0.02 * (pa - lvl) and vx < pa:
            lvl = max(lvl, vx)
    mask, _ = gs.plasma_region(phi, geo, iz, ir, lvl)
    return dict(iz=iz, ir=ir, Ra=Ra, Za=Za, pa=pa, lvl=lvl, mask=mask)


def _sane(reg, r_in, r_out, geo, width_frac=0.5):
    m = reg["mask"]
    if m.sum() < 80:
        return False
    if not (r_in - 0.08 <= reg["Ra"] <= r_out + 0.02):
        return False
    if abs(reg["Za"]) > 0.45:
        return False
    rr = geo.RR[m]
    return (rr.max() - rr.min()) > width_frac * (r_out - r_in)


def predict_shot(shot: dict, cfg: Config) -> dict:
    geo = geometry(shot)
    names = [str(c) for c in shot["current_names"]]
    T = len(shot["efit_times"])
    E = kin.mast_edges_series(shot, cfg.te_sep)
    if cfg.fit_inner_edge and cfg.inner_one_sided:
        seen = np.array([np.isfinite(kin.mast_midplane_edges(shot, k, cfg.te_sep)[0]) for k in range(T)])
        r_in_max = np.array([kin.innermost_channel(shot, k) for k in range(T)])
    if not np.isfinite(E[:, 1]).any():
        E[:, 1] = cfg.r_out_default
    if not np.isfinite(E[:, 0]).any():
        E[:, 0] = cfg.r_in_default
    psirz = np.zeros((T,) + geo.shape, np.float32)
    q95 = np.full(T, np.nan)
    betaN = np.full(T, np.nan)
    diag = np.zeros((T, 9))
    J = None
    bv = 0.0
    an = cfg.alpha_n
    jx = cfg.j_exponent
    tf_col = names.index("magnetics_tf_current")
    R_ts = np.concatenate([shot["ts_core_R"], shot["ts_edge_spatial"]])
    t_ms = np.asarray(shot["efit_times"], float)
    if np.nanmax(t_ms) < 5.0:                      # times given in seconds -> ms
        t_ms = t_ms * 1e3
    ramp = cfg.kinetic and cfg.jexp_t1 > cfg.jexp_t0
    for k in range(T):
        if ramp:
            w = float(np.clip((t_ms[k] - cfg.jexp_t0) / (cfg.jexp_t1 - cfg.jexp_t0), 0.0, 1.0))
            jx = cfg.jexp_early + w * (cfg.j_exponent - cfg.jexp_early)
        cur = {n: float(shot["currents"][k, i]) for i, n in enumerate(names)}
        cur["magnetics_sol_current"] *= cfg.sol_factor
        ip = float(shot["ip"][k]) * 1e3
        s = np.sign(ip) or 1.0
        ip_abs = max(abs(ip), 1e4)
        r_in, r_out = float(E[k, 0]), float(E[k, 1])
        phi_ext = s * geo.coil_flux(cur)
        kin_ts = None
        if cfg.kinetic:
            kin_ts = dict(R=R_ts, Z=np.zeros_like(R_ts),
                       Te=np.concatenate([shot["ts_core_Te"][k], shot["ts_edge_Te"][k]]).astype(float),
                       ne=np.concatenate([shot["ts_core_ne"][k], shot["ts_edge_ne"][k]]).astype(float),
                       c_total=cfg.c_total, j_exponent=jx)
        best = None
        for attempt in range(4 if cfg.holdz_retry else 3):
            hz = cfg.hold_z or attempt == 3
            if J is None or attempt > 0:
                r0, a = 0.5 * (r_in + r_out), max(0.5 * (r_out - r_in), 0.3)
                J0 = gs.initial_current(geo, ip_abs, r0, 0.0, a * 0.9, 1.8)
                bv0, an0 = 0.0, cfg.alpha_n
            else:
                J0, bv0, an0 = J, bv, an
            phi, info = gs.solve(phi_ext, ip_abs, geo, J0, betap=cfg.betap, alpha_m=cfg.alpha_m,
                                 alpha_n=an0, relax=cfg.relax if attempt < 2 else 0.15,
                                 max_iter=cfg.max_iter, tol=cfg.tol, r_out_target=r_out, bv0=bv0,
                                 radial_gain=cfg.radial_gain, return_info=True,
                                 r_in_target=(r_in if cfg.fit_inner_edge and (not cfg.inner_one_sided or seen[k])
                                              else None),
                                 r_in_max=(r_in_max[k] if cfg.fit_inner_edge and cfg.inner_one_sided and not seen[k]
                                           else None),
                                 peaking_gain=cfg.peaking_gain, kinetic=kin_ts, avg_last=cfg.avg_last,
                                 edge_z=(0.0 if cfg.laser_row else None),
                                 z_target=(0.0 if hz else None),
                                 vctrl_map=(geo.coil_flux({"magnetics_p6u_current": 1.0, "magnetics_p6l_current": -1.0}) * s
                                            if hz else None))
            reg = _region(phi, geo)
            ok = _sane(reg, r_in, r_out, geo, cfg.sane_width)
            score = (ok, info.get("converged", False), -info.get("change", 1.0))
            if best is None or score > best[0]:
                best = (score, phi, info, reg)
            if ok:
                break
        (ok, conv, _), phi, info, reg = best
        if ok:
            J, bv = info["J"], info["bv"]
            an = info.get("alpha_n", an)
            jx = info.get("j_exponent", jx)
        else:
            J, bv, an = None, 0.0, cfg.alpha_n
            if k > 0 and diag[k - 1, 0] > 0:      # keep the last good map rather than a broken one
                phi = psirz[k - 1].astype(float)
                reg = _region(phi, geo)
        psirz[k] = phi.astype(np.float32)          # MAST convention: magnetic axis = maximum
        diag[k] = [float(ok), float(conv), info.get("iters", 0), info.get("bv", 0.0), reg["Ra"], reg["Za"],
                   info.get("r_in", np.nan), r_out, info.get("ivc", 0.0)]

        # ---- scalars
        F = MU0 * cfg.tf_turns * abs(float(shot["currents"][k, tf_col])) * 1e3 / (2 * np.pi)
        try:
            q95[k] = sc.q95_from_map(phi, geo.R, geo.Z, (reg["Ra"], reg["Za"]), reg["pa"], reg["lvl"], F)
        except Exception:
            q95[k] = np.nan
        try:
            m = reg["mask"]
            dphi = reg["pa"] - reg["lvl"]
            psiN = (reg["pa"] - phi) / dphi
            f = RegularGridInterpolator((geo.Z, geo.R), psiN, bounds_error=False, fill_value=np.nan)
            Te = np.concatenate([shot["ts_core_Te"][k], shot["ts_edge_Te"][k]])
            ne = np.concatenate([shot["ts_core_ne"][k], shot["ts_edge_ne"][k]])
            prof = sc.electron_pressure_profile(R_ts, np.zeros_like(R_ts), Te, ne,
                                                lambda r, z: f(np.c_[z, r]))
            if prof is not None:
                pav, V = sc.volume_average(psiN, m, geo.RR, geo.dA, prof)
                rr = geo.RR[m]
                a = 0.5 * (rr.max() - rr.min())
                Rgeo = 0.5 * (rr.max() + rr.min())
                B0 = F / Rgeo
                bt = 100 * 2 * MU0 * cfg.c_total * pav / B0 ** 2
                betaN[k] = bt * a * B0 / (ip_abs / 1e6)
        except Exception:
            betaN[k] = np.nan

    # time cleaning of the scalars: fill gaps, despike
    t = np.asarray(shot["efit_times"], float)
    for arr, lo, hi, default in ((q95, 1.0, 30.0, 6.0), (betaN, 0.0, 8.0, 2.0)):
        bad = ~np.isfinite(arr) | (arr < lo) | (arr > hi)
        if (~bad).sum() >= 1:
            arr[bad] = np.interp(t[bad], t[~bad], arr[~bad])
        else:
            arr[:] = default
        if T >= 5:
            arr[:] = median_filter(arr, size=3, mode="nearest")
    return {"psirz": psirz, "q95": q95.astype(np.float32), "betaN": betaN.astype(np.float32), "diag": diag}


# ----------------------------------------------------------------------------------- batch driver

def _worker(args):
    path, cfg_dict, out_dir = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import data
    cfg = Config(**cfg_dict)
    out = Path(out_dir) / (Path(path).stem + ".npz")
    if out.exists():
        return str(out), 0.0, None
    t0 = time.time()
    try:
        shot = data.load(Path(path))
        p = predict_shot(shot, cfg)
        np.savez(out, **p)
        return str(out), time.time() - t0, None
    except Exception as e:  # noqa: BLE001
        return str(out), time.time() - t0, repr(e)


def run(cfg: Config, paths, out_dir: Path, workers: int = 9):
    from concurrent.futures import ProcessPoolExecutor
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
    jobs = [(str(p), dataclasses.asdict(cfg), str(out_dir)) for p in paths]
    done = 0
    t0 = time.time()
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for out, dt, err in pool.map(_worker, jobs, chunksize=1):
            done += 1
            if err:
                errors.append((out, err))
            if done % 25 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} shots, {time.time() - t0:.0f} s, errors {len(errors)}", flush=True)
    return errors


if __name__ == "__main__":
    import argparse
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--sol", type=float, default=0.5)
    ap.add_argument("--te", type=float, default=kin.TE_SEP_EV)
    ap.add_argument("--betap", type=float, default=0.3)
    ap.add_argument("--an", type=float, default=1.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--fit-inner-edge", action="store_true")
    ap.add_argument("--inner-one-sided", action="store_true", help="fit the inner edge only where the laser saw it")
    ap.add_argument("--sane-width", type=float, default=0.5)
    ap.add_argument("--holdz-retry", action="store_true", help="last retry with vertical control at Z=0")
    ap.add_argument("--role", default="test", choices=["test", "final"], help="public test or Final-phase inputs")
    ap.add_argument("--kinetic", action="store_true", help="laser-built current profile (Spitzer Te^1.5)")
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--avg-last", type=int, default=0)
    ap.add_argument("--laser-row", action="store_true", help="read model edges on Z=0 (the laser's height)")
    ap.add_argument("--hold-z", action="store_true", help="vertical control: hold the axis at Z=0 with a P6-pair correction")
    ap.add_argument("--jexp", type=float, default=1.5, help="kinetic peaking: Spitzer exponent of the current profile")
    ap.add_argument("--c-total", type=float, default=sc.C_TOTAL, help="total / electron pressure (DIII-D calibration)")
    ap.add_argument("--jexp-early", type=float, default=0.5, help="current diffusion: exponent before --jexp-t0")
    ap.add_argument("--jexp-t0", type=float, default=0.0, help="ms; exponent rises linearly from --jexp-early ...")
    ap.add_argument("--jexp-t1", type=float, default=0.0, help="ms; ... to --jexp at this time (off if t1 <= t0)")
    a = ap.parse_args()
    cfg = Config(sol_factor=a.sol, te_sep=a.te, betap=a.betap, alpha_n=a.an, fit_inner_edge=a.fit_inner_edge,
                 kinetic=a.kinetic, max_iter=a.max_iter, avg_last=a.avg_last, laser_row=a.laser_row,
                 hold_z=a.hold_z, inner_one_sided=a.inner_one_sided,
                 sane_width=a.sane_width, holdz_retry=a.holdz_retry, j_exponent=a.jexp, c_total=a.c_total,
                 jexp_early=a.jexp_early, jexp_t0=a.jexp_t0, jexp_t1=a.jexp_t1)
    comp = HERE.parent
    paths = sorted(glob.glob(str(comp / "cache" / a.role / "mast_*.npz")))
    if a.limit:
        paths = paths[:a.limit]
    out_dir = comp / "cache" / ("mast_pred" if a.role == "test" else "mast_pred_final") / (a.name or cfg.tag())
    errs = run(cfg, paths, out_dir, a.workers)
    for e in errs[:20]:
        print("ERROR", e)
