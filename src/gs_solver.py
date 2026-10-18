"""FREE-BOUNDARY GRAD-SHAFRANOV SOLVER: the flux map from coil currents + plasma current, no learning.

The picture. The flux map psi(R, Z) is the sum of the "hills" made by every ring of current:
    psi = (coils' flux: exact, from geometry)  +  (plasma's own flux: from where its current flows)
And the plasma current is not free: it flows ON the flux surfaces of that same map
(Grad-Shafranov force balance: J = R p'(psi) + F F'(psi) / (mu0 R)). So we iterate:
    1. guess where the current is
    2. add its flux to the coils' flux  ->  a map
    3. find the axis (top of the hill) and the edge (the last closed contour, through the X-point)
    4. put the current back on those contours with a fixed profile shape, total = measured Ip
    5. repeat until the map stops changing (Picard iteration)
Only the coil currents, the measured Ip and a few profile numbers go in. No equilibrium label is
read anywhere in this file, so the same code runs on DIII-D and MAST.

Vertical instability. An elongated plasma is like a pencil balanced on its tip: the real machine
holds it with fast feedback, and a naive iteration lets it "fall" up or down. Each iteration we
therefore move the new current rigidly up/down to the height where the coils' radial field
exerts zero net vertical force on it (one Newton step on that force). At a true equilibrium the
force is zero anyway, so this changes where we converge only by removing the fall.

Conventions. Internally we work with phi = sign(Ip) * psi_physical, so the magnetic axis is always
a MAXIMUM of phi and the current density J >= 0 (amperes per m^2, in units of |Ip|).
psi_physical = R * A_phi in Wb/rad, the same absolute zero EFIT uses (flux -> 0 far away).
"""
from __future__ import annotations

import collections
import hashlib
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.special import ellipe, ellipk

MU0 = 4e-7 * np.pi
GREENS_DIR = Path(__file__).resolve().parent.parent / "cache" / "greens"

# Solenoid columns are in kA PER TURN (README "Current units"); everything else is total kA-turn.
PER_TURN_COLUMNS = {"magnetics_ECOILA", "magnetics_sol_current"}
ECOIL_EFFECTIVE_TURNS = 138.0      # DIII-D only, fitted on DIII-D training maps (physics_backbone.py)
COIL_SUBDIVISION = {"DIII-D": 6, "MAST": 3}

# Where plasma current may flow (a box minus a clearance around every coil conductor).
# DIII-D: from the envelope of training-set LCFS (DIII-D labels are allowed) + margin.
# MAST: from machine geometry only (centre column at R ~ 0.19 m, in-vessel P-coils), NO MAST EFIT.
DOMAIN = {
    "DIII-D": dict(r_min=1.00, r_max=2.38, z_max=1.36, coil_clearance=0.06),
    "MAST": dict(r_min=0.18, r_max=1.56, z_max=1.60, coil_clearance=0.07, exempt=("magnetics_sol_current",)),
}
# MAST fix 2026-10-13: the solenoid sits BEHIND the centre column, whose plasma-facing armour is at R ~ 0.19 m; a 7 cm
# clearance around the solenoid put an artificial wall at R ~ 0.24 m and made half of all MAST frames "wall-limited"
# instead of diverted (official detailed results: q95 R2 -0.47, kappa -0.98, D_LCFS 0.27). The solenoid is exempt
# from the clearance and the inner wall is the column surface.


# ----------------------------------------------------------------------------------- Green's functions

def ring_flux(R, Z, Rc, Zc):
    """Flux per radian (Wb/rad) at (R, Z) from a ring of ONE ampere at (Rc, Zc)."""
    k2 = 4.0 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)
    k2 = np.clip(k2, 1e-12, 1.0 - 1e-9)
    k = np.sqrt(k2)
    return MU0 / (np.pi * k) * np.sqrt(R * Rc) * ((1.0 - k2 / 2.0) * ellipk(k2) - ellipe(k2))


def _rect_points(rc, zc, w, h, n):
    off = (np.arange(n) + 0.5) / n - 0.5
    dr, dz = np.meshgrid(off * w, off * h)
    return rc + dr.ravel(), zc + dz.ravel()


class Geometry:
    """Everything about one machine that does not change between frames: grid, coils, domain, Green's tables."""

    def __init__(self, shot: dict, machine: str, refine: int = 1):
        self.machine = machine
        self.refine = int(refine)
        R0 = np.asarray(shot["grid_R"], float)
        Z0 = np.asarray(shot["grid_Z"], float)
        if self.refine > 1:      # internal grid `refine` times finer; every refine-th point is an official point
            self.R = np.linspace(R0[0], R0[-1], (len(R0) - 1) * self.refine + 1)
            self.Z = np.linspace(Z0[0], Z0[-1], (len(Z0) - 1) * self.refine + 1)
        else:
            self.R, self.Z = R0, Z0
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)          # rows = Z, cols = R
        self.dR = float(self.R[1] - self.R[0])
        self.dZ = float(self.Z[1] - self.Z[0])
        self.dA = self.dR * self.dZ
        self.shape = self.RR.shape

        cols = [str(c) for c in shot["coil_input_column"]]
        self.coil_cols = sorted(set(cols))
        self.coil_R = np.asarray(shot["coil_R"], float)
        self.coil_Z = np.asarray(shot["coil_Z"], float)
        self.coil_W = np.asarray(shot["coil_width"], float)
        self.coil_H = np.asarray(shot["coil_height"], float)
        self.coil_col_of = np.asarray(cols)

        key = hashlib.sha1(np.concatenate([self.R, self.Z, self.coil_R, self.coil_Z, self.coil_W,
                                           self.coil_H]).tobytes() + repr(DOMAIN[machine]).encode()).hexdigest()[:12]
        path = GREENS_DIR / f"{machine}_{key}.npz"
        if path.exists():
            z = np.load(path)
            self.Gc, self.dom = z["Gc"], z["dom"]
            self.Gp = z["Gp"] if "Gp" in z.files else None
        else:
            self.Gc = self._coil_matrix()
            self.dom = self._domain()
            # the dense (grid x domain) matrix is only kept on the official grid (too big when refined)
            self.Gp = self._plasma_matrix() if self.refine == 1 else None
            GREENS_DIR.mkdir(parents=True, exist_ok=True)
            if self.Gp is not None:
                np.savez(path, Gc=self.Gc, dom=self.dom, Gp=self.Gp)
            else:
                np.savez(path, Gc=self.Gc, dom=self.dom)
        self.dom_idx = np.flatnonzero(self.dom.ravel())
        er = ndimage.binary_erosion(self.dom, structure=np.ones((3, 3)))
        self.dom_edge = self.dom & ~er
        self.dom_inner = ndimage.binary_erosion(self.dom, structure=np.ones((3, 3)), iterations=2)
        self._build_fft_kernel()

    # -- fast plasma flux: a ring's flux depends on (R_target, R_source, Z_target - Z_source) only, so
    #    the (4225 x n_dom) matrix product is a sum of 1-D convolutions along Z, done with FFTs.
    def _build_fft_kernel(self):
        cols = np.flatnonzero(self.dom.any(axis=0))
        self.fft_cols = np.arange(cols.min(), cols.max() + 1)
        nZ = len(self.Z)
        m = np.arange(-(nZ - 1), nZ)                      # Z_target - Z_source, in cells
        self.nfft = 1 << int(np.ceil(np.log2(3 * nZ)))
        Rt = self.R[:, None, None]
        Rs = self.R[self.fft_cols][None, :, None]
        dz = (m * self.dZ)[None, None, :]
        g = np.zeros((len(self.R), len(self.fft_cols), len(m)))
        for fr, fz in ((-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)):
            # target at height dz above the source cell centre; sub-filament at fz*dZ above it
            g += ring_flux(Rt, dz - fz * self.dZ, Rs + fr * self.dR, 0.0)
        g /= 4.0
        # kernel index j -> offset m = j - (nZ - 1); place offset m at fft position m mod nfft
        ker = np.zeros((len(self.R), len(self.fft_cols), self.nfft))
        for j, mm in enumerate(m):
            ker[:, :, mm % self.nfft] = g[:, :, j]
        kh = np.fft.rfft(ker, axis=2)                     # (nR_t, nR_s, nfft//2+1)
        self.fft_kernel = np.ascontiguousarray(np.transpose(kh, (2, 0, 1)))   # (nf, nR_t, nR_s)

    def plasma_flux_fft(self, J: np.ndarray) -> np.ndarray:
        I = np.where(self.dom, J, 0.0)[:, self.fft_cols] * self.dA          # (nZ, nR_s) amperes
        Ih = np.fft.rfft(I, n=self.nfft, axis=0)                           # (nf, nR_s)
        out = np.matmul(self.fft_kernel, Ih[:, :, None])[:, :, 0]         # (nf, nR_t)
        psi = np.fft.irfft(out, n=self.nfft, axis=0)[: len(self.Z)]        # (nZ, nR_t)
        return psi

    # -- flux of every coil column per kA of the column's value, (4225, n_cols)
    def _coil_matrix(self):
        Rt, Zt = self.RR.ravel(), self.ZZ.ravel()
        n_sub = COIL_SUBDIVISION[self.machine]
        rows_per_col = collections.Counter(self.coil_col_of.tolist())
        G = np.zeros((Rt.size, len(self.coil_cols)))
        for j, col in enumerate(self.coil_cols):
            m = self.coil_col_of == col
            acc = np.zeros(Rt.size)
            for rc, zc, w, h in zip(self.coil_R[m], self.coil_Z[m], self.coil_W[m], self.coil_H[m]):
                pr, pz = _rect_points(rc, zc, w, h, n_sub)
                acc += ring_flux(Rt[:, None], Zt[:, None], pr[None, :], pz[None, :]).mean(axis=1)
            if col in PER_TURN_COLUMNS:          # every element is one turn carrying the column current
                turns = ECOIL_EFFECTIVE_TURNS if rows_per_col[col] == 1 else 1.0
                G[:, j] = acc * turns * 1e3
            else:                                # column = total kA-turn shared by its elements
                G[:, j] = acc / rows_per_col[col] * 1e3
        return G

    def _domain(self):
        d = DOMAIN[self.machine]
        m = (self.RR >= d["r_min"]) & (self.RR <= d["r_max"]) & (np.abs(self.ZZ) <= d["z_max"])
        # clearance from every in-vessel conductor element (rectangle distance)
        exempt = set(d.get("exempt", ()))
        for col, rc, zc, w, h in zip(self.coil_col_of, self.coil_R, self.coil_Z, self.coil_W, self.coil_H):
            if col in exempt:
                continue
            dr = np.maximum(np.abs(self.RR - rc) - w / 2, 0)
            dz = np.maximum(np.abs(self.ZZ - zc) - h / 2, 0)
            m &= np.hypot(dr, dz) >= d["coil_clearance"]
        # keep the largest connected piece
        lab, n = ndimage.label(m)
        if n > 1:
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            m = lab == (1 + int(np.argmax(sizes)))
        return m

    # -- flux at every grid point per ampere flowing in each domain cell, (4225, n_dom)
    def _plasma_matrix(self):
        Rt, Zt = self.RR.ravel(), self.ZZ.ravel()
        Rs, Zs = self.RR[self.dom], self.ZZ[self.dom]
        G = np.zeros((Rt.size, Rs.size))
        for fr, fz in ((-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)):
            for a in range(0, Rs.size, 400):
                b = slice(a, a + 400)
                G[:, b] += ring_flux(Rt[:, None], Zt[:, None], Rs[None, b] + fr * self.dR,
                                     Zs[None, b] + fz * self.dZ)
        return G / 4.0

    def coil_flux(self, currents_kA: dict) -> np.ndarray:
        """psi_physical from the coils (Wb/rad), (65, 65)."""
        v = np.array([currents_kA.get(c, 0.0) for c in self.coil_cols], float)
        return (self.Gc @ v).reshape(self.shape)

    def plasma_flux(self, J: np.ndarray) -> np.ndarray:
        """Flux (in phi units) of the current density J (A/m^2, zero outside the domain). Dense reference."""
        if self.Gp is None:
            return self.plasma_flux_fft(J)
        return (self.Gp @ (J.ravel()[self.dom_idx] * self.dA)).reshape(self.shape)

    def to_official(self, field: np.ndarray) -> np.ndarray:
        """Sample an internal-grid field at the official 65 x 65 points."""
        return field[:: self.refine, :: self.refine] if self.refine > 1 else field


# ----------------------------------------------------------------------------------- critical points

def find_axis(phi: np.ndarray, geo: Geometry, near=None):
    """Magnetic axis = interior local maximum of phi inside the domain (strongest, or nearest `near`)."""
    p = np.where(geo.dom_inner, phi, -np.inf)
    c = p[1:-1, 1:-1]
    ge = np.isfinite(c)
    for dz in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dz or dr:
                ge &= c >= p[1 + dz:p.shape[0] - 1 + dz, 1 + dr:p.shape[1] - 1 + dr]
    lm = np.zeros_like(p, bool)
    lm[1:-1, 1:-1] = ge
    idx = np.argwhere(lm)
    if len(idx) == 0:
        iz, ir = np.unravel_index(np.argmax(p), p.shape)
        return int(iz), int(ir)
    if near is not None and len(idx) > 1:
        d = np.hypot(geo.Z[idx[:, 0]] - near[1], geo.R[idx[:, 1]] - near[0])
        # nearest candidate, unless a much stronger maximum exists
        k = int(np.argmin(d))
        return int(idx[k, 0]), int(idx[k, 1])
    k = int(np.argmax(p[idx[:, 0], idx[:, 1]]))
    return int(idx[k, 0]), int(idx[k, 1])


def subpixel_axis(phi, iz, ir, geo):
    """Sub-pixel (R, Z, phi) of the maximum by a local quadratic fit."""
    if not (0 < iz < phi.shape[0] - 1 and 0 < ir < phi.shape[1] - 1):
        return geo.R[ir], geo.Z[iz], phi[iz, ir]
    fR = (phi[iz, ir + 1] - phi[iz, ir - 1]) / 2
    fZ = (phi[iz + 1, ir] - phi[iz - 1, ir]) / 2
    fRR = phi[iz, ir + 1] - 2 * phi[iz, ir] + phi[iz, ir - 1]
    fZZ = phi[iz + 1, ir] - 2 * phi[iz, ir] + phi[iz - 1, ir]
    fRZ = (phi[iz + 1, ir + 1] - phi[iz + 1, ir - 1] - phi[iz - 1, ir + 1] + phi[iz - 1, ir - 1]) / 4
    H = np.array([[fRR, fRZ], [fRZ, fZZ]])
    g = np.array([fR, fZ])
    try:
        d = -np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        d = np.zeros(2)
    d = np.clip(d, -1, 1)
    val = phi[iz, ir] + 0.5 * g @ d
    return geo.R[ir] + d[0] * geo.dR, geo.Z[iz] + d[1] * geo.dZ, val


def find_xpoints(phi: np.ndarray, geo: Geometry):
    """Saddle points of phi inside the domain: list of (R, Z, phi_X, iz, ir), sub-pixel (vectorised)."""
    gZ, gR = np.gradient(phi, geo.dZ, geo.dR)
    B2 = gR ** 2 + gZ ** 2
    p = np.where(geo.dom, B2, np.inf)
    c = p[1:-1, 1:-1]
    le = np.isfinite(c)
    for dz in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dz or dr:
                le &= c <= p[1 + dz:p.shape[0] - 1 + dz, 1 + dr:p.shape[1] - 1 + dr]
    iz, ir = np.nonzero(le)
    iz = iz + 1
    ir = ir + 1
    if iz.size == 0:
        return []
    f0 = phi[iz, ir]
    fRR = (phi[iz, ir + 1] - 2 * f0 + phi[iz, ir - 1]) / geo.dR ** 2
    fZZ = (phi[iz + 1, ir] - 2 * f0 + phi[iz - 1, ir]) / geo.dZ ** 2
    fRZ = (phi[iz + 1, ir + 1] - phi[iz + 1, ir - 1] - phi[iz - 1, ir + 1]
           + phi[iz - 1, ir - 1]) / (4 * geo.dR * geo.dZ)
    det = fRR * fZZ - fRZ ** 2
    keep = det < 0
    if not keep.any():
        return []
    iz, ir, f0, fRR, fZZ, fRZ, det = iz[keep], ir[keep], f0[keep], fRR[keep], fZZ[keep], fRZ[keep], det[keep]
    g_R, g_Z = gR[iz, ir], gZ[iz, ir]
    dRr = -(fZZ * g_R - fRZ * g_Z) / det
    dZz = -(-fRZ * g_R + fRR * g_Z) / det
    dRr = np.clip(dRr, -geo.dR, geo.dR)
    dZz = np.clip(dZz, -geo.dZ, geo.dZ)
    val = f0 + 0.5 * (g_R * dRr + g_Z * dZz)
    return [(float(geo.R[a] + x), float(geo.Z[b] + y), float(v), int(b), int(a))
            for a, b, x, y, v in zip(ir, iz, dRr, dZz, val)]


_CROSS = ndimage.generate_binary_structure(2, 1)


def plasma_region(phi, geo: Geometry, iz, ir, level):
    """Connected region {phi > level} containing the axis; returns (mask, touches_domain_edge)."""
    reg = (phi > level) & geo.dom
    lab, _ = ndimage.label(reg, structure=_CROSS)   # 4-connectivity: no leak through a diagonal X-point
    k = lab[iz, ir]
    if k == 0:
        return np.zeros_like(reg), True
    comp = lab == k
    return comp, bool((comp & geo.dom_edge).any())


def boundary_level(phi, geo: Geometry, iz, ir, n_iter=16):
    """Lowest phi level whose axis-enclosing region stays off the domain edge = the LCFS level.

    Below an X-point's level the region leaks through the X-point into the divertor legs, which run
    to the domain edge; for a limited plasma it reaches the edge directly. Either way the search
    lands on the separatrix (or limiter) level, the same definition the scorer uses.
    """
    hi = phi[iz, ir]
    lo = float(phi[geo.dom].min())
    _, t = plasma_region(phi, geo, iz, ir, lo)
    if not t:
        return lo
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        _, touch = plasma_region(phi, geo, iz, ir, mid)
        if touch:
            lo = mid
        else:
            hi = mid
    return hi


# ----------------------------------------------------------------------------------- profiles

class Profile:
    """J(R, psiN) = c_p (R / R0) f(psiN) + c_f (R0 / R) f(psiN),  f = (1 - psiN^alpha_m)^alpha_n.

    The R/R0 part is the pressure-driven current (R p'), the R0/R part the F F' current. Two
    constraints fix c_p and c_f: the total current = |Ip| (measured), and the poloidal beta.
    """

    def __init__(self, alpha_m=1.0, alpha_n=2.0, R0=1.0):
        self.am, self.an, self.R0 = alpha_m, alpha_n, R0
        x = np.linspace(0, 1, 2001)
        f = (1 - x ** self.am) ** self.an
        cum = np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(x))])
        self._x, self._tail = x, cum[-1] - cum      # tail(psiN) = integral_psiN^1 f

    def shape(self, psiN):
        return np.clip(1 - np.clip(psiN, 0, 1) ** self.am, 0, 1) ** self.an

    def tail(self, psiN):
        return np.interp(np.clip(psiN, 0, 1), self._x, self._tail)

    def current(self, psiN, mask, geo: Geometry, ip_abs, betap, dphi, lp):
        """J on the grid (A/m^2), and the pressure profile p (Pa), given the region and constraints."""
        f = np.where(mask, self.shape(psiN), 0.0)
        Rg = geo.RR
        A = float(np.sum(Rg / self.R0 * f)) * geo.dA
        B = float(np.sum(self.R0 / Rg * f)) * geo.dA
        dV = 2 * np.pi * Rg * geo.dA
        V = float(np.sum(dV[mask]))
        P1 = float(np.sum(np.where(mask, self.tail(psiN), 0) * dV))
        if A <= 0 or B <= 0 or V <= 0 or dphi <= 0 or P1 <= 0:
            return None, None
        Bpa = MU0 * ip_abs / max(lp, 1e-3)
        # <p>_V = c_p * dphi / R0 * P1 / V   and   betap = 2 mu0 <p>_V / Bpa^2
        c_p = betap * V * Bpa ** 2 * self.R0 / (2 * MU0 * dphi * P1)
        c_p = min(c_p, ip_abs / A)              # pressure current cannot exceed the total current
        c_f = (ip_abs - c_p * A) / B
        J = c_p * Rg / self.R0 * f + c_f * self.R0 / Rg * f
        p = c_p * dphi / self.R0 * np.where(mask, self.tail(psiN), 0.0)
        return J, p


class KineticProfile:
    """Current profile built from the Thomson laser instead of a fixed formula ("kinetic" equilibrium).

        J = R x p'(psi)            the pressure-driven part, p = C_TOTAL x ne x Te (measured, x ions/fast)
          + c_f x (R0/R) x g(psiN)  the rest, shaped like the plasma's conductivity: g = (Te / Te_axis)^1.5
                                    (Spitzer: a hot plasma conducts better, so the current crowds where Te is high)
    c_f is fixed by the measured total current Ip. Each update maps the laser points onto the current
    map's flux surfaces (psiN), so the profile and the map converge together.
    """

    def __init__(self, ts_R, ts_Z, ts_Te, ts_ne, R0, c_total=2.42, j_exponent=1.5, nbins=12):
        ok = np.isfinite(ts_R) & np.isfinite(ts_Z) & np.isfinite(ts_Te) & np.isfinite(ts_ne) & (ts_Te > 0) & (ts_ne > 0)
        self.R, self.Z = np.asarray(ts_R, float)[ok], np.asarray(ts_Z, float)[ok]
        self.Te, self.ne = np.asarray(ts_Te, float)[ok], np.asarray(ts_ne, float)[ok]
        self.R0, self.c, self.jexp, self.nbins = R0, c_total, j_exponent, nbins
        self.x = self.p = self.g = None
        self.am, self.an = 1.0, 1.0        # fallback shape until the first update
        self._fallback = Profile(1.0, 1.0, R0)

    def update(self, phi, phia, dphi, geo):
        if self.R.size < 8 or dphi <= 0:
            return
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((geo.Z, geo.R), phi, bounds_error=False, fill_value=np.nan)
        x = (phia - f(np.c_[self.Z, self.R])) / dphi
        keep = np.isfinite(x) & (x >= 0) & (x < 1.0)
        if keep.sum() < 6:
            return
        x, Te, ne = x[keep], self.Te[keep], self.ne[keep]
        edges = np.linspace(0, 1, self.nbins + 1)
        bx, bt, bn = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (x >= a) & (x < b)
            if m.sum() >= 1:
                bx.append(0.5 * (a + b))
                bt.append(np.median(Te[m]))
                bn.append(np.median(ne[m]))
        if len(bx) < 3:
            return
        bx, bt, bn = np.array(bx), np.array(bt), np.array(bn)
        bt = np.minimum.accumulate(np.maximum(bt, 1.0))      # monotone: hottest at the axis
        bn = np.minimum.accumulate(np.maximum(bn, 1e17))
        xx = np.r_[0.0, bx, 1.0]
        tt = np.r_[bt[0], bt, min(bt[-1], 40.0)]
        nn = np.r_[bn[0], bn, bn[-1] * 0.5]
        self.x = xx
        self.p = self.c * tt * nn * 1.602176634e-19
        self._tt = tt
        self.g = (tt / tt[0]) ** self.jexp

    def set_jexp(self, j):
        """Peaking knob of the kinetic profile (the Spitzer exponent): larger = current more crowded at the axis."""
        self.jexp = float(j)
        if self.x is not None:
            self.g = (self._tt / self._tt[0]) ** self.jexp

    def current(self, psiN, mask, geo, ip_abs, betap, dphi, lp):
        if self.x is None:
            return self._fallback.current(psiN, mask, geo, ip_abs, betap, dphi, lp)
        Rg = geo.RR
        xN = np.clip(psiN, 0, 1)
        dp_dx = np.gradient(self.p, self.x)
        Jp = np.where(mask, Rg * np.maximum(-np.interp(xN, self.x, dp_dx) / dphi, 0.0), 0.0)
        g = np.where(mask, np.interp(xN, self.x, self.g), 0.0)
        Ip_p = float(Jp.sum()) * geo.dA
        if Ip_p > 0.9 * ip_abs:                               # laser pressure too steep for this Ip
            Jp *= 0.9 * ip_abs / Ip_p
            Ip_p = 0.9 * ip_abs
        B = float(np.sum(self.R0 / Rg * g)) * geo.dA
        if B <= 0:
            return None, None
        J = Jp + (ip_abs - Ip_p) / B * self.R0 / Rg * g
        p = np.where(mask, np.interp(xN, self.x, self.p), 0.0)
        return J, p


# ----------------------------------------------------------------------------------- the solve

def contour_length(mask, geo):
    """Approximate perimeter of a region (m), from its boundary cells."""
    edge = mask & ~ndimage.binary_erosion(mask)
    n = edge.sum()
    return float(n) * 0.5 * (geo.dR + geo.dZ) * 1.12


def initial_current(geo: Geometry, ip_abs, r0, z0, a, kappa):
    x = ((geo.RR - r0) / a) ** 2 + ((geo.ZZ - z0) / (kappa * a)) ** 2
    J = np.where((x < 1) & geo.dom, 1 - x, 0.0)
    s = J.sum() * geo.dA
    return J * ip_abs / s if s > 0 else J


def midplane_edges(phi, mask, lvl, geo: Geometry, iz):
    """Inboard and outboard crossings of the boundary level on the row iz (sub-cell, linear in phi)."""
    row = np.flatnonzero(mask[iz])
    if row.size == 0:
        return np.nan, np.nan
    i0, i1 = row.min(), row.max()
    p = phi[iz]
    r_in = geo.R[i0]
    if i0 > 0 and p[i0] != p[i0 - 1]:
        r_in = geo.R[i0] - (p[i0] - lvl) / (p[i0] - p[i0 - 1]) * geo.dR
    r_out = geo.R[i1]
    if i1 < len(p) - 1 and p[i1] != p[i1 + 1]:
        r_out = geo.R[i1] + (p[i1] - lvl) / (p[i1] - p[i1 + 1]) * geo.dR
    return float(r_in), float(r_out)


def solve(phi_ext, ip_abs, geo: Geometry, J0, betap=0.5, alpha_m=1.0, alpha_n=2.0, R0=None,
          max_iter=80, tol=2e-5, relax=0.5, vertical_feedback=True, return_info=False,
          r_out_target=None, bv0=0.0, radial_gain=0.02, bv_max=0.06,
          r_in_target=None, peaking_gain=1.5, alpha_n_range=(0.3, 4.0),
          radial_knob="bv", kinetic=None, avg_last=0, edge_z=None,
          z_target=None, vctrl_map=None, vctrl_max=60.0, r_in_max=None):
    """Picard free-boundary solve. phi_ext = sign(Ip) * coil flux. Returns phi (65,65) and info.

    r_out_target: if given (m), a uniform vertical field B_v (phi units: phi += B_v R^2 / 2) is adjusted
    every iteration so the outboard midplane edge of the plasma sits at that radius. Physically it
    stands for what the coils alone do not provide (eddy currents in the vessel, profile errors).

    z_target / vctrl_map: VERTICAL CONTROL, like the machine's own radial-field (P6) coils. vctrl_map is the
    phi map of +1 kA in the upper and -1 kA in the lower control coil. Every iteration the current is moved so
    the magnetic axis sits at z_target and the control current (|.| <= vctrl_max kA) is set to cancel the net
    vertical force there. Stands for unmeasured asymmetric eddy currents; None = free height.

    edge_z: height (m) of the laser chord that measured the edges (MAST: 0.0). The model's edges are read on that
    row; None reads them on the magnetic axis's row (used when the target is the boundary's outermost point).

    r_in_target: if given (m), the current peaking alpha_n is adjusted every iteration so the inboard
    midplane edge sits at that radius (a flatter current pushes the inner separatrix outward). With the
    outer edge pinned by B_v, this reads the current profile's peaking off the laser's inner edge.
    r_in_max: one-sided version for frames where the laser finds no inner edge (still hot at its innermost
    channel): the edge must then lie INSIDE that channel, so the peaking only acts if the model's edge is outside it.
    With a kinetic profile the peaking knob is the Spitzer exponent instead of alpha_n.
    """
    R0 = R0 if R0 is not None else float(np.sum(geo.RR * J0) / max(np.sum(J0), 1e-30))
    prof = Profile(alpha_m, alpha_n, R0)
    if kinetic is not None:
        prof = KineticProfile(kinetic["R"], kinetic["Z"], kinetic["Te"], kinetic["ne"], R0,
                              kinetic.get("c_total", 2.42), kinetic.get("j_exponent", 1.5))
    J = J0.copy()
    half_R2 = 0.5 * geo.RR ** 2
    gZ = np.gradient(phi_ext, geo.dZ, axis=0)
    gZZ = np.gradient(gZ, geo.dZ, axis=0)
    hold = vctrl_map is not None and z_target is not None
    gZc = np.gradient(vctrl_map, geo.dZ, axis=0) if hold else None
    info = dict(converged=False, iters=0, dz_total=0.0)
    near = None
    phi_old = None
    bv = float(bv0)
    ivc = 0.0                                            # vertical-control correction current (kA)
    from collections import deque
    tail = deque(maxlen=max(int(avg_last), 1))          # last iterates, averaged at the end if avg_last > 0
    for it in range(max_iter):
        phi = phi_ext + bv * half_R2 + geo.plasma_flux_fft(J)
        if vctrl_map is not None:
            phi = phi + ivc * vctrl_map
        if avg_last:
            tail.append(phi)
        iz, ir = find_axis(phi, geo, near)
        Ra, Za, phia = subpixel_axis(phi, iz, ir, geo)
        near = (Ra, Za)
        lvl = boundary_level(phi, geo, iz, ir)
        # snap to a nearby X-point's exact (sub-pixel) level
        for (rx, zx, vx, _, _) in find_xpoints(phi, geo):
            if abs(vx - lvl) < 0.02 * (phia - lvl) and vx < phia:
                lvl = max(lvl, vx) if vx <= phia else lvl
        mask, _ = plasma_region(phi, geo, iz, ir, lvl)
        dphi = phia - lvl
        if mask.sum() < 4 or dphi <= 0:
            info["fail"] = "no plasma region"
            break
        psiN = (phia - phi) / dphi
        if kinetic is not None and it % 3 == 0:
            prof.update(phi, phia, dphi, geo)
        lp = contour_length(mask, geo)
        Jn, p = prof.current(psiN, mask, geo, ip_abs, betap, dphi, lp)
        if Jn is None:
            info["fail"] = "profile"
            break
        if vertical_feedback and hold:
            # VERTICAL CONTROL: move the current so the axis sits at z_target, then give the control coils the
            # current that cancels the net vertical force there (what the machine's radial-field coils do).
            # Fixed 2026-10-13: the first version adjusted the control current but left it out of the force
            # balance below, so the plasma never moved (v10 had the same height spread as v9).
            dz = float(np.clip(z_target - Za, -1.5 * geo.dZ, 1.5 * geo.dZ))
            if abs(dz) > 1e-6:
                Jn = ndimage.shift(Jn, (dz / geo.dZ, 0), order=1, mode="constant")
                Jn = np.where(geo.dom, Jn, 0.0)
                s = Jn.sum() * geo.dA
                if s > 0:
                    Jn *= ip_abs / s
                info["dz_total"] += dz
            fc = float(np.sum(Jn * gZc))
            if fc != 0:
                ivc = float(np.clip(-float(np.sum(Jn * gZ)) / fc, -vctrl_max, vctrl_max))
        elif vertical_feedback:
            F = float(np.sum(Jn * gZ))
            dF = float(np.sum(Jn * gZZ))
            if dF != 0:
                dz = float(np.clip(-F / dF, -1.5 * geo.dZ, 1.5 * geo.dZ))
                if abs(dz) > 1e-6:
                    Jn = ndimage.shift(Jn, (dz / geo.dZ, 0), order=1, mode="constant")
                    Jn = np.where(geo.dom, Jn, 0.0)
                    s = Jn.sum() * geo.dA
                    if s > 0:
                        Jn *= ip_abs / s
                    info["dz_total"] += dz
        if r_out_target is not None:
            row = iz if edge_z is None else int(np.argmin(np.abs(geo.Z - edge_z)))
            r_in, r_out = midplane_edges(phi, mask, lvl, geo, row)
            info["r_in"], info["r_out"] = r_in, r_out
            if np.isfinite(r_out) and radial_knob == "bv":
                # plasma too far out -> more negative B_v (pushes the current inward)
                bv -= radial_gain * (r_out - r_out_target)
                bv = float(np.clip(bv, -bv_max, bv_max))
            elif np.isfinite(r_out) and radial_knob == "peaking":
                # plasma too far out -> less hoop force -> broader current (lower li): smaller alpha_n
                an_new = float(np.clip(prof.an * np.exp(-2.0 * peaking_gain * (r_out - r_out_target)),
                                       *alpha_n_range))
                if abs(an_new - prof.an) > 1e-3:
                    prof = Profile(prof.am, an_new, prof.R0)
                info["alpha_n"] = prof.an
            target = None
            if r_in_target is not None and np.isfinite(r_in) and np.isfinite(r_in_target):
                target = r_in_target            # the laser saw the inner edge: match it
            elif r_in_max is not None and np.isfinite(r_in) and np.isfinite(r_in_max) and r_in > r_in_max:
                target = r_in_max               # the laser saw only hot plasma down to its innermost channel
            if target is not None:
                # plasma too wide on the inboard side (r_in < target) -> flatter current (smaller peaking)
                fac = np.exp(peaking_gain * (r_in - target))
                if isinstance(prof, KineticProfile):
                    prof.set_jexp(float(np.clip(prof.jexp * fac, *alpha_n_range)))
                    info["j_exponent"] = prof.jexp
                else:
                    an_new = float(np.clip(prof.an * fac, *alpha_n_range))
                    if abs(an_new - prof.an) > 1e-3:
                        prof = Profile(prof.am, an_new, prof.R0)
                    info["alpha_n"] = prof.an
        J = J + relax * (Jn - J)
        info["iters"] = it + 1
        if phi_old is not None:
            change = float(np.max(np.abs(phi - phi_old))) / max(dphi, 1e-9)
            info["change"] = change
            if change < tol:
                info["converged"] = True
                break
        phi_old = phi
    phi = phi_ext + bv * half_R2 + geo.plasma_flux_fft(J)
    if vctrl_map is not None:
        phi = phi + ivc * vctrl_map
    if avg_last and len(tail) >= 2:
        phi = np.mean(np.stack(list(tail) + [phi]), axis=0)
        info["averaged"] = len(tail) + 1
    info.update(axis=(near if near else (np.nan, np.nan)), J=J, bv=bv, ivc=ivc)
    return (phi, info) if return_info else phi
