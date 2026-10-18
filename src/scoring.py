"""Official-scorer evaluation on the cached held-out DIII-D shots (role 'eval').

Same functions Codabench runs (starter/fusion_scoring, byte-identical) via the kit's local_score.py,
composed exactly like src/eval_local.py — but reading truth from the cache, computing the expensive
per-frame references ONCE (cache/eval_refs.pkl), and scoring shots in parallel.

    from scoring import score
    res = score(preds)          # preds: list of {"psirz": (T,65,65), "q95": (T,), "betaN": (T,)}
                                # in the order of load_eval()
"""
from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMP = HERE.parent
sys.path.insert(0, str(COMP / "starter"))
sys.path.insert(0, str(COMP / "starter" / "fusion_scoring"))
sys.path.insert(0, str(HERE))

import local_score as ls                     # noqa: E402  (official harness functions)
from common import PSI_SIGNS                 # noqa: E402
from metrics import Accum, finalize_machine  # noqa: E402

import data                                  # noqa: E402

REFS = data.CACHE / "eval_refs.pkl"
_MASK = np.load(COMP / "starter" / "fusion_scoring" / "masks" / "d3d_envelope.npz")
R_GRID, Z_GRID = _MASK["grid_R"], _MASK["grid_Z"]
MC = _MASK["mask_coarse"].astype(bool)
MF = MC.astype(float)


def load_eval(n: int | None = None) -> list[dict]:
    """Held-out truth: {'file','psi','q95','betaN', + all inputs} per eval shot, sorted by file."""
    shots = data.load_role("eval", n=n)
    for s in shots:
        s["psi"] = s["psi"].astype(np.float32)
    return shots


def _ref_worker(psi):
    return ls.build_reference(psi, R_GRID, Z_GRID, MC, MF)


def references(shots: list[dict], workers: int = 6) -> list:
    """Per-shot (contours, cons, cmask, rgeo) — computed once per eval shot and cached on disk."""
    cached = pickle.loads(REFS.read_bytes()) if REFS.exists() else {}
    todo = [s for s in shots if s["file"] not in cached]
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for s, ref in zip(todo, pool.map(_ref_worker, [s["psi"] for s in todo])):
                cached[s["file"]] = ref
        REFS.write_bytes(pickle.dumps(cached))
    return [cached[s["file"]] for s in shots]


def _score_worker(args):
    gt, ref, pred, sign, means = args
    return ls.score_shot(gt, ref, pred, R_GRID, Z_GRID, MC, MF, sign, means)


def score(preds: list[dict], shots: list[dict] | None = None, workers: int = 6) -> dict:
    shots = shots if shots is not None else load_eval()
    assert len(preds) == len(shots), (len(preds), len(shots))
    refs = references(shots, workers)
    st = {"psi_sum": 0.0, "psi_sumsq": 0.0, "psi_n": 0.0,
          "scal_sum": np.zeros(ls.N_SCALARS), "scal_sumsq": np.zeros(ls.N_SCALARS), "scal_n": np.zeros(ls.N_SCALARS),
          "cons_sum": np.zeros(ls.N_CONS), "cons_sumsq": np.zeros(ls.N_CONS), "cons_n": np.zeros(ls.N_CONS)}
    for s, ref in zip(shots, refs):
        g = s["psi"].astype(np.float64)
        st["psi_sum"] += g.sum(); st["psi_sumsq"] += (g ** 2).sum(); st["psi_n"] += g.size
        for j, name in enumerate(ls.SCALARS):
            v = np.asarray(s[name], np.float64)
            v = v[np.isfinite(v)]
            st["scal_sum"][j] += v.sum(); st["scal_sumsq"][j] += (v ** 2).sum(); st["scal_n"][j] += v.size
        for j in range(ls.N_CONS):
            v = ref[1][ref[2][:, j], j]
            st["cons_sum"][j] += v.sum(); st["cons_sumsq"][j] += (v ** 2).sum(); st["cons_n"][j] += v.size
    mean_psi = st["psi_sum"] / st["psi_n"]
    means = (mean_psi, st["scal_sum"] / np.maximum(st["scal_n"], 1), st["cons_sum"] / np.maximum(st["cons_n"], 1))

    totals = {sg: 0.0 for sg in PSI_SIGNS}
    for s, p in zip(shots, preds):
        rr, _, _ = ls.psi_residuals(s["psi"], p["psirz"], mean_psi)
        for sg in PSI_SIGNS:
            totals[sg] += rr[sg]
    sign = min(PSI_SIGNS, key=lambda sg: totals[sg])

    gts = [{"psi": s["psi"], "q95": np.asarray(s["q95"], np.float64), "betaN": np.asarray(s["betaN"], np.float64)}
           for s in shots]
    acc = Accum()
    acc.psi_sign = sign
    jobs = [(g, r, {k: np.asarray(p[k], np.float64) for k in ("psirz", "q95", "betaN")}, sign, means)
            for g, r, p in zip(gts, refs, preds)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_score_worker, jobs):
            acc.add(part)
    res = finalize_machine(acc, st)
    res["n_shots"] = len(shots)
    res["n_frames"] = int(sum(s["psi"].shape[0] for s in shots))
    return res


def summary(res: dict) -> str:
    cons = ", ".join(f"{k} {('n/a' if v is None else f'{v:.3f}')}" for k, v in res["r2_cons_each"].items())
    return (f"S = {res['S']:.4f} | R2_psi {res['r2_psi']:.4f} | R2_qb {res['r2_qb']:.4f} "
            f"(q95 {res['r2_each']['q95']:.3f}, betaN {res['r2_each']['betaN']:.3f}) | "
            f"D_LCFS {res['dlcfs']:.4f} | Consistency {res['consistency']:.4f}\n"
            f"    per shape number: {cons}\n"
            f"    LCFS fail {res['lcfs_fail_frac']:.1%} · derivation fail {res['cons_fail_frac']:.1%} · "
            f"sign {res['psi_sign']:+d} · {res['n_shots']} shots / {res['n_frames']} frames")
