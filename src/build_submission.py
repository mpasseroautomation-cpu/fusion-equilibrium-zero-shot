"""BUILD A SUBMISSION — exact Codabench format, validated, ready for Matteo to push and upload.

    python src/build_submission.py --model baseline_pca20             # DIII-D from the model, MAST = zeros
    python src/build_submission.py --model baseline_pca20 --official-prefix 10   # + official validator on 10 shots

Writes submissions/<name>/diii_d_public_test.npz and mast_public_test.npz, plus SUBMIT_STEPS.md.
It NEVER pushes or uploads anything: sending is Matteo's step (the commands are in SUBMIT_STEPS.md).

Format (starter/submission_skeleton.py): per shot i in OFFICIAL STREAM ORDER (sorted filenames),
  shot_{i:04d}_psirz (T, 65, 65) float16 · shot_{i:04d}_q95 (T,) float32 · shot_{i:04d}_betaN (T,) float32
with T = len(efit_times). float16, never np.round (rounding destroys ~35% of MAST Consistency).

Why MAST = zeros by default: Rule 3 forbids using MAST equilibrium data to select or tune anything
that produces MAST predictions, and a MAST leaderboard score is computed from MAST labels. A zeros file
scores exactly 0 by construction, so it reveals nothing. Real MAST files are sent only when they are
frozen and pre-registered (proposal "MAST stays blind until frozen", 2026-10-05).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(COMP / "starter"))
import data                                                   # noqa: E402
from validate_submission import _quantization_warning         # noqa: E402  (official check)

EXPECTED = {"diii_d_public_test": 874, "mast_public_test": 1206}
MACHINE = {"diii_d_public_test": "DIII-D", "mast_public_test": "MAST"}
OUT = COMP / "submissions"


def load_model(name: str):
    """Unpickle results/models/<name>.pkl. Models pickled from a script run as __main__ (the first
    baseline, 2026-10-05) reference __main__.Baseline, so expose the class there before loading."""
    import __main__
    import baseline
    if not hasattr(__main__, "Baseline"):
        __main__.Baseline = baseline.Baseline
    return pickle.loads((COMP / "results" / "models" / f"{name}.pkl").read_bytes())


def test_shots(config: str) -> list[dict]:
    shots = data.load_role("test", config=config, sort_by_stream=True)
    idx = [s["stream_index"] for s in shots]
    if idx != list(range(EXPECTED[config])):
        missing = sorted(set(range(EXPECTED[config])) - set(idx))
        raise SystemExit(f"{config}: cache incomplete ({len(idx)}/{EXPECTED[config]}; first missing {missing[:5]})")
    return shots


def zeros_prediction(shot: dict) -> dict:
    T = len(shot["efit_times"])
    return {"psirz": np.zeros((T, 65, 65), np.float32), "q95": np.zeros(T, np.float32),
            "betaN": np.zeros(T, np.float32)}


def build(config: str, predict) -> dict:
    out = {}
    for s in test_shots(config):
        i, T = s["stream_index"], len(s["efit_times"])
        p = predict(s)
        assert p["psirz"].shape == (T, 65, 65), (config, i, p["psirz"].shape, T)
        out[f"shot_{i:04d}_psirz"] = np.asarray(p["psirz"]).astype(np.float16)
        for name in ("q95", "betaN"):
            a = np.asarray(p[name], np.float32)
            assert a.shape == (T,), (config, i, name, a.shape)
            out[f"shot_{i:04d}_{name}"] = a
    return out


def validate_local(sub: dict, config: str) -> list[str]:
    """The official validator's checks, run against the cached efit_times (no 10 GB re-stream)."""
    errors = []
    shots = test_shots(config)
    for s in shots:
        i, T = s["stream_index"], len(s["efit_times"])
        k = f"shot_{i:04d}_psirz"
        if k not in sub:
            errors.append(f"{k} missing")
            continue
        a = sub[k]
        if a.shape != (T, 65, 65) or not np.issubdtype(a.dtype, np.floating):
            errors.append(f"{k}: {a.shape} {a.dtype}")
        if MACHINE[config] == "DIII-D" and not np.isfinite(a).all():
            errors.append(f"{k}: NaN/Inf")
        w = _quantization_warning(a, k)
        if w:
            errors.append("WARNING " + w)
        for name in ("q95", "betaN"):
            k2 = f"shot_{i:04d}_{name}"
            if k2 not in sub or sub[k2].shape != (T,) or not np.issubdtype(sub[k2].dtype, np.floating):
                errors.append(f"{k2}: bad or missing")
            elif not np.isfinite(sub[k2]).all():
                errors.append(f"{k2}: NaN/Inf")
    expected = {f"shot_{i:04d}_{n}" for i in range(len(shots)) for n in ("psirz", "q95", "betaN")}
    extra = [k for k in sub if k not in expected]
    if extra:
        errors.append(f"unexpected keys {extra[:5]}")
    return errors


STEPS = """# How to send this submission (Matteo presses send — Claude never does)

Built {when} by `src/build_submission.py --model {model}`. DIII-D = {model}. MAST = {mast}.
Local validation: {valid}.

**One-time setup (≈15 min, Tuesday is fine):**
1. Hugging Face account → https://huggingface.co/settings/tokens → create a **WRITE** token.
2. In the VS Code terminal:
   ```
   cd ~/fusion-disruption/competition/starter
   ../.venv/bin/huggingface-cli login          # paste the WRITE token (it stays on your Mac)
   ../.venv/bin/python push_predictions.py --repo YOUR_HF_USERNAME/fusion-eq-predictions --dir ../submissions/{name} --dry-run
   ../.venv/bin/python push_predictions.py --repo YOUR_HF_USERNAME/fusion-eq-predictions --dir ../submissions/{name}
   ```
   The second command only CREATES the private repo (nothing uploaded yet).
3. Back at https://huggingface.co/settings/tokens → New token → **Fine-grained** → Repository
   permissions: pick `fusion-eq-predictions` → tick ONLY "Read access to contents of selected repos".

**Every submission (≈5 min):**
```
../.venv/bin/python push_predictions.py --repo YOUR_HF_USERNAME/fusion-eq-predictions --dir ../submissions/{name} --read-token hf_THE_FINE_GRAINED_READ_TOKEN
```
→ it writes `submission_pointer.zip`. Upload THAT zip at https://www.codabench.org/competitions/17739/
(My Submissions). Fact sheet: describe the model in one line; tick compute-light only if training took < 2 h (it did: {fit}).

⚠ Never upload in the last hours before a deadline (queue / 500 errors burn slots).
⚠ MAST = zeros on purpose: it scores 0 and reveals nothing about MAST (Rule-3 hygiene). Your Challenge-2
   G will show 0 until the pre-registered MAST file is sent.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="results/models/<model>.pkl (an object with predict_shot)")
    ap.add_argument("--mast", default="zeros", choices=["zeros"], help="MAST file: zeros only (Rule-3 hygiene)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--official-prefix", type=int, default=0,
                    help="also run the OFFICIAL validator on the first N shots (streams N shots from HF)")
    args = ap.parse_args()
    name = args.name or f"{args.model}__mast-{args.mast}"
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model)

    t0 = time.time()
    report = []
    for config, predict in (("diii_d_public_test", model.predict_shot), ("mast_public_test", zeros_prediction)):
        sub = build(config, predict)
        path = dest / f"{config}.npz"
        np.savez_compressed(path, **sub)
        errs = validate_local(sub, config)
        hard = [e for e in errs if not e.startswith("WARNING")]
        report.append(f"{config}: {len(sub) // 3} shots, {path.stat().st_size / 1e6:.0f} MB, "
                      f"{'OK' if not hard else f'{len(hard)} PROBLEMS'}" + (f", {len(errs) - len(hard)} warnings" if len(errs) > len(hard) else ""))
        print(report[-1], flush=True)
        for e in errs[:10]:
            print("   ", e)
        if hard:
            return 1
    if args.official_prefix:
        from validate_submission import validate
        for config in EXPECTED:
            if validate(dest / f"{config}.npz", config, args.official_prefix):
                print(f"OFFICIAL validator FAILED on the first {args.official_prefix} shots of {config}")
                return 1
        report.append(f"official validator OK on the first {args.official_prefix} shots of each fold")
    fit = "seconds on a CPU"
    (dest / "SUBMIT_STEPS.md").write_text(STEPS.format(
        when=time.strftime("%Y-%m-%d %H:%M"), model=args.model, mast=args.mast, name=name,
        valid="; ".join(report), fit=fit))
    print(f"built {dest} in {time.time() - t0:.0f} s — nothing was pushed or uploaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
