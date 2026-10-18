"""RUNG 3 — better inputs for DIII-D (aim: pass the 0.85 gate), same PCA + ridge as the baseline.

The baseline sees 21 numbers at ONE instant. What that misses, and what is added here (inputs only, no labels):
  1. RATES (kA/ms) of every current and of Ip. A changing current induces eddy currents in the vessel wall;
     the data does not ship them, but their flux is in the map. The rate is the best proxy we have.
  2. HISTORY: solenoid change since the first EFIT time, and time since breakdown (t = 0 on DIII-D).
     The transformer's flux "already spent" sets part of each frame's overall flux level (the offset).
  3. RATIOS: each F-coil current / Ip — shaping relative to the plasma being shaped.
  4. THOMSON geometry + kinetics. DIII-D's core laser is a vertical line at R = 1.94 m: where the density
     falls off along it marks the plasma's top and bottom crossings — geometry the coils alone do not give.
     Plus peak Te, peak ne, mean ne, peak electron pressure, Te-weighted centroid height.
Missing Thomson values -> training median, plus a 0/1 'missing' flag.

    python src/features.py --k 20        # fit on the cached training shots, score, log, save results/models/rung3_k20.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data      # noqa: E402
import ledger    # noqa: E402
import scoring   # noqa: E402
from baseline import ip_sign   # noqa: E402  (reversed-Ip shots: EFIT stores the map with flipped sign)

ALPHAS = np.logspace(-3, 3, 13)
F_COILS = [f"magnetics_F{i}{s}" for i in range(1, 10) for s in "AB"]
TS_NAMES = ["ts_Te_peak", "ts_ne_peak", "ts_ne_mean", "ts_pe_peak", "ts_z_top", "ts_z_bottom", "ts_z_centroid"]


def core_chord_z(shot: dict) -> np.ndarray | None:
    """Z of each DIII-D core Thomson channel, from the chord table (names 'TS_core_*', in order)."""
    names = [str(n) for n in shot["thomson_chord_name"]]
    z = np.asarray(shot["thomson_chord_Z"], float)
    core = [i for i, n in enumerate(names) if n.startswith("TS_core")]
    n_ch = shot["ts_core_Te"].shape[1]
    if len(core) != n_ch:
        return None
    return z[core]


def thomson_summaries(shot: dict) -> np.ndarray:
    T = len(shot["efit_times"])
    out = np.full((T, len(TS_NAMES)), np.nan)
    Te, ne = shot["ts_core_Te"].astype(float), shot["ts_core_ne"].astype(float)
    if Te.size == 0:
        return out
    z = core_chord_z(shot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out[:, 0] = np.nanmax(Te, axis=1) / 1e3                      # keV
        out[:, 1] = np.nanmax(ne, axis=1) / 1e19                     # 1e19 m^-3
        out[:, 2] = np.nanmean(ne, axis=1) / 1e19
        out[:, 3] = np.nanmax(ne * Te, axis=1) * 1.602e-19 / 1e3     # kPa
        if z is not None:
            for k in range(T):
                n = ne[k]
                ok = np.isfinite(n) & np.isfinite(z)
                if ok.sum() < 5:
                    continue
                inside = ok & (n > 0.2 * np.nanmax(n[ok]))
                if inside.any():
                    out[k, 4] = z[inside].max()
                    out[k, 5] = z[inside].min()
                w = np.where(np.isfinite(Te[k]) & ok, Te[k], 0.0)
                if w.sum() > 0:
                    out[k, 6] = (w * np.where(ok, z, 0.0)).sum() / w.sum()
    return out


def raw_features(shot: dict) -> tuple[np.ndarray, list[str]]:
    names_cur = list(shot["current_names"])
    T = len(shot["efit_times"])
    base = data.baseline_features(shot)                                       # 20 currents + Ip
    cols, names = [base], [*data.D3D_BASELINE_CURRENTS, "ip"]
    sl = np.column_stack([shot["current_slopes"][:, names_cur.index(c)] if c in names_cur else np.full(T, np.nan)
                          for c in data.D3D_BASELINE_CURRENTS] + [shot["ip_slope"]])
    cols.append(sl); names += [f"rate_{c}" for c in data.D3D_BASELINE_CURRENTS] + ["rate_ip"]
    ecoil = base[:, 0]
    first = ecoil[np.isfinite(ecoil)][0] if np.isfinite(ecoil).any() else np.nan
    t = np.asarray(shot["efit_times"], float) / 1e3
    cols.append(np.column_stack([ecoil - first, t])); names += ["ecoil_since_start", "t_s"]
    ip = base[:, -1]
    ip_safe = np.where(np.abs(ip) < 50, np.sign(ip + 1e-9) * 50, ip)          # avoid /0 during ramp-up
    cols.append(np.column_stack([base[:, 1 + i] / ip_safe for i in range(len(F_COILS))]))
    names += [f"{c}_over_ip" for c in F_COILS]
    cols.append(thomson_summaries(shot)); names += TS_NAMES
    # direction-free magnitudes for q95 / betaN (q95 ~ B_T / Ip; both currents run either way on DIII-D)
    bt = np.abs(base[:, names_base_index("magnetics_bcoil")])
    cols.append(np.column_stack([np.abs(ip), bt, bt / np.maximum(np.abs(ip), 50.0)]))
    names += ["abs_ip", "abs_bcoil", "abs_bcoil_over_abs_ip"]
    return np.column_stack(cols), names


def names_base_index(col: str) -> int:
    return data.D3D_BASELINE_CURRENTS.index(col)


class Rung3:
    """Scaler + PCA(K) + RidgeCV on the Rung-3 inputs; NaN -> training median + missing flags."""

    def __init__(self, k: int):
        self.k = k

    def _prep(self, X):
        miss = ~np.isfinite(X[:, self.flag_cols])
        Xf = np.where(np.isfinite(X), X, self.median)
        return np.column_stack([Xf, miss.astype(float)])

    def fit(self, X, Y, q95, betaN, names):
        self.names = names
        self.median = np.nanmedian(X, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        self.flag_cols = [i for i, n in enumerate(names) if n.startswith("ts_")]
        Xp = self._prep(X)
        self.scaler = StandardScaler().fit(Xp)
        Xs = self.scaler.transform(Xp)
        self.pca = PCA(n_components=self.k, svd_solver="randomized", random_state=0).fit(Y)
        self.ridge = RidgeCV(alphas=ALPHAS).fit(Xs, self.pca.transform(Y))
        self.scalar = {}
        for name, t in (("q95", q95), ("betaN", betaN)):
            m = np.isfinite(t)
            self.scalar[name] = RidgeCV(alphas=ALPHAS).fit(Xs[m], t[m])
        return self

    def predict_shot(self, shot: dict) -> dict:
        X, _ = raw_features(shot)
        Xs = self.scaler.transform(self._prep(X))
        psi = self.pca.inverse_transform(self.ridge.predict(Xs)).reshape(len(X), 65, 65) * ip_sign(shot)
        return {"psirz": psi.astype(np.float32),
                "q95": self.scalar["q95"].predict(Xs).astype(np.float32),
                "betaN": self.scalar["betaN"].predict(Xs).astype(np.float32)}


def training_matrix(shots):
    X, Y, Q, B = [], [], [], []
    names = None
    for s in shots:
        x, names = raw_features(s)
        y = s["psi"].reshape(len(x), -1).astype(np.float32) * np.float32(ip_sign(s))   # EFIT sign convention
        ok = np.isfinite(x[:, :21]).all(1) & np.isfinite(y).all(1)          # currents + Ip must exist
        X.append(x[ok]); Y.append(y[ok])
        Q.append(np.asarray(s["q95"], float)[ok]); B.append(np.asarray(s["betaN"], float)[ok])
    return np.concatenate(X), np.concatenate(Y), np.concatenate(Q), np.concatenate(B), names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--train-shots", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    train = data.load_role("train", n=args.train_shots)
    X, Y, q, b, names = training_matrix(train)
    print(f"train: {len(train)} shots, {len(X)} frames, {X.shape[1]} inputs", flush=True)
    t_fit = time.time()
    model = Rung3(args.k).fit(X, Y, q, b, names)
    fit_s = time.time() - t_fit
    del Y
    shots = scoring.load_eval()
    res = scoring.score([model.predict_shot(s) for s in shots], shots)
    print(f"RUNG 3 PCA({args.k}) + ridge on {X.shape[1]} inputs:\n    " + scoring.summary(res), flush=True)
    print(f"    ridge alpha {model.ridge.alpha_:.3g} · fit {fit_s:.1f} s", flush=True)
    ledger.log(f"rung3_k{args.k}", f"PCA({args.k})+RidgeCV, {X.shape[1]} inputs (rates, history, F/Ip, Thomson)", res,
               train_shots=len(train), train_frames=len(X), fit_seconds=fit_s, total_seconds=time.time() - t0)
    out = data.COMP / "results" / "models"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"rung3_k{args.k}.pkl").write_bytes(pickle.dumps(model))
    return 0


if __name__ == "__main__":
    import features as _self             # pickles must reference features.Rung3, not __main__.Rung3
    raise SystemExit(_self.main())
