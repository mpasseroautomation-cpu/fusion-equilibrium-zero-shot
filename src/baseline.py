"""RUNG 2 — the official baseline (PCA + ridge), scored with the official scorer; and the PCA ceiling.

THE BASELINE (as in starter/experiments.py): the 21 DIII-D inputs (18 F-coils, E-coil solenoid,
B-coil, Ip) at the EFIT times -> standardise -> RidgeCV (alphas 10^-3 .. 10^3) -> the first K PCA
coefficients of the flux map. q95 and betaN each get their own RidgeCV on the same inputs.

THE CEILING: take the TRUE held-out maps and keep only their first K PCA coefficients. No model that
predicts K coefficients can beat that score, so it shows what K costs BEFORE we tune anything — and
which of the 7 shape numbers dies first.

    python src/baseline.py --k 20                 # fit, score, log to results/LEDGER.csv, save the model
    python src/baseline.py --ceiling 20 50 100    # the representation ceiling
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
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

MODELS = data.COMP / "results" / "models"
ALPHAS = np.logspace(-3, 3, 10)


def ip_sign(shot: dict) -> float:
    """+1 / -1: the direction of the plasma current in this shot (median over frames).

    Found 2026-10-05: DIII-D's EFIT stores the flux map with its sign tied to the current direction (the
    axis is always the minimum). On reversed-current shots (8% of the data) the stored map is the NEGATIVE
    of what the coils make, so a linear model on coil currents fits them backwards (5 of 60 held-out shots
    carried 82% of the baseline's flux error). signfix: flip those maps before fitting, flip predictions back.
    """
    ip = np.nanmedian(np.asarray(shot["ip"], float))
    return -1.0 if ip < 0 else 1.0


def training_matrix(shots: list[dict], signfix: bool = False):
    """Stack every frame of every training shot: X (frames, 21), Y (frames, 4225), q95, betaN."""
    X, Y, Q, B = [], [], [], []
    for s in shots:
        x = data.baseline_features(s)
        y = s["psi"].reshape(len(x), -1).astype(np.float32)
        if signfix:
            y = y * np.float32(ip_sign(s))
        ok = np.isfinite(x).all(1) & np.isfinite(y).all(1)
        X.append(x[ok]); Y.append(y[ok])
        Q.append(np.asarray(s["q95"], float)[ok]); B.append(np.asarray(s["betaN"], float)[ok])
    return np.concatenate(X), np.concatenate(Y), np.concatenate(Q), np.concatenate(B)


class Baseline:
    """PCA(K) on the flux maps + RidgeCV from the 21 standardised inputs to the K coefficients."""

    def __init__(self, k: int, signfix: bool = False):
        self.k = k
        self.signfix = signfix

    def fit(self, X, Y, q95, betaN):
        self.fill = np.nanmean(X, axis=0)                     # used only if a test input is NaN
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.pca = PCA(n_components=self.k, svd_solver="randomized", random_state=0).fit(Y)
        self.ridge = RidgeCV(alphas=ALPHAS).fit(Xs, self.pca.transform(Y))
        self.scalar = {}
        for name, t in (("q95", q95), ("betaN", betaN)):
            m = np.isfinite(t)
            self.scalar[name] = RidgeCV(alphas=ALPHAS).fit(Xs[m], t[m])
        return self

    def predict_shot(self, shot: dict) -> dict:
        x = data.baseline_features(shot)
        x = np.where(np.isfinite(x), x, self.fill)
        Xs = self.scaler.transform(x)
        psi = self.pca.inverse_transform(self.ridge.predict(Xs)).reshape(len(x), 65, 65)
        if getattr(self, "signfix", False):
            psi = psi * ip_sign(shot)
        return {"psirz": psi.astype(np.float32),
                "q95": self.scalar["q95"].predict(Xs).astype(np.float32),
                "betaN": self.scalar["betaN"].predict(Xs).astype(np.float32)}


def run_baseline(k: int, n_train: int | None, signfix: bool = False) -> dict:
    t0 = time.time()
    train = data.load_role("train", n=n_train)
    X, Y, q, b = training_matrix(train, signfix)
    print(f"train: {len(train)} shots, {len(X)} frames", flush=True)
    t_fit = time.time()
    model = Baseline(k, signfix).fit(X, Y, q, b)
    fit_s = time.time() - t_fit
    del Y
    shots = scoring.load_eval()
    preds = [model.predict_shot(s) for s in shots]
    res = scoring.score(preds, shots)
    tag = "_signfix" if signfix else ""
    print(f"BASELINE PCA({k}) + ridge{tag}:\n    " + scoring.summary(res), flush=True)
    print(f"    ridge alpha {model.ridge.alpha_:.3g} · q95 alpha {model.scalar['q95'].alpha_:.3g} · "
          f"betaN alpha {model.scalar['betaN'].alpha_:.3g} · PCA variance kept "
          f"{model.pca.explained_variance_ratio_.sum():.5f} · fit {fit_s:.1f} s", flush=True)
    ledger.log(f"baseline_pca{k}_ridge{tag}", f"PCA({k})+RidgeCV, 21 kit inputs{' + Ip-sign fix' if signfix else ''}",
               res, train_shots=len(train), train_frames=len(X), fit_seconds=fit_s, total_seconds=time.time() - t0,
               notes="official baseline recipe (starter/experiments.py), windowed-mean inputs, Ip time fix"
                     + ("; flux targets flipped on reversed-Ip shots (EFIT sign convention)" if signfix else ""))
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / f"baseline_pca{k}{tag}.pkl").write_bytes(pickle.dumps(model))
    return res


def run_ceiling(ks: list[int], n_train: int | None) -> None:
    t0 = time.time()
    train = data.load_role("train", n=n_train)
    _, Y, _, _ = training_matrix(train)
    shots = scoring.load_eval()
    for k in ks:
        pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(Y)
        preds = []
        for s in shots:
            T = s["psi"].shape[0]
            rec = pca.inverse_transform(pca.transform(s["psi"].reshape(T, -1))).reshape(T, 65, 65)
            preds.append({"psirz": rec.astype(np.float32),
                          "q95": np.asarray(s["q95"], np.float32),       # true scalars: isolate the map
                          "betaN": np.asarray(s["betaN"], np.float32)})
        res = scoring.score(preds, shots)
        print(f"CEILING PCA({k}) of the TRUE maps (scalars = truth):\n    " + scoring.summary(res), flush=True)
        ledger.log(f"ceiling_pca{k}", f"PCA({k}) reconstruction of TRUE maps (diagnostic, not a model)", res,
                   train_shots=len(train), train_frames=len(Y), total_seconds=time.time() - t0,
                   notes="representation ceiling; q95/betaN set to truth so R2_qb = 1 by construction")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--train-shots", type=int, default=None, help="cap the number of cached training shots used")
    ap.add_argument("--ceiling", type=int, nargs="*")
    ap.add_argument("--signfix", action="store_true", help="flip flux maps of reversed-Ip shots before fitting")
    args = ap.parse_args()
    if args.ceiling:
        run_ceiling(args.ceiling, args.train_shots)
    else:
        run_baseline(args.k, args.train_shots, args.signfix)
    return 0


if __name__ == "__main__":
    # Run through the importable module so pickled models reference `baseline.Baseline`
    # (not `__main__.Baseline`, which no other script can unpickle).
    import baseline as _self
    raise SystemExit(_self.main())
