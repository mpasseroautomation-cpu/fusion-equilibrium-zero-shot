# Zero-shot MAST equilibria from coils + Thomson: a free-boundary Grad-Shafranov entry
NeurIPS 2026 Fusion Equilibrium Challenge: reproduction package for the Final phase (code frozen 2026-10-18).

**Entrant:** Matteo Passero. **Licence:** MIT. **AI disclosure:** built with an AI coding assistant (Anthropic Claude,
agentic, in Claude Code), directed and reviewed by the entrant, who takes responsibility for the submission (Rule 4).
Full write-up: [METHODS_REPORT.md](METHODS_REPORT.md).

## What it does
- **DIII-D** (`diii_d_*_test.npz`): PCA(20) of the flux map + ridge regression from 72 engineered inputs, trained on
  300 DIII-D training shots (~7 s, one CPU core). Reversed-current shots are sign-normalised before fitting.
  Weights: `results/models/rung3_k20.pkl`.
- **MAST** (`mast_*_test.npz`): no learned model and no MAST label. Per frame, a free-boundary Grad-Shafranov solve from
  the measured coil currents (812 conductor elements), the measured plasma current and the Thomson profiles:
  - current profile J = R p'(psi) + c (R0/R) (Te/Te_axis)^1.0, with p = 2.42 ne Te mapped onto the flux surfaces;
  - outboard midplane edge pinned at the 30 eV Thomson point by a uniform vertical correction field;
  - magnetic axis held at Z = 0 by a correction current in the P6 radial-field coil pair (as MAST's vertical control does);
  - the returned map is the average of the last 25 of 60 Picard iterates;
  - q95 by field-line integration on the computed map, beta_N from the Thomson pressure on the computed flux surfaces;
  - failed frames and axis jumps > 4 cm are rebuilt by interpolation in time.

  This is Development submission 992762 (S_MAST 0.7106), the best MAST pipeline we scored on the Development board.

## Environment
Python 3.11. `bash setup.sh` creates `.venv`, installs `requirements.lock` (every package pinned, 54 pins) and clones the
official starter kit (MIT) into `starter/` at commit 5b467988 for its `data_fixes.py`, push script and scorer. No GPU.

## Training entrypoint (rebuilds everything from scratch)
```
bash train.sh
#  1. python src/cache.py --train 300     stream + reduce DIII-D training shots (+ 60 held-out eval shots)   ~1 h, mostly download
#  2. python src/features.py --k 20       fit the DIII-D model -> results/models/rung3_k20.pkl               ~10 s
#  3. python src/calibrate_d3d.py         re-derive the DIII-D calibrations used by the MAST physics (2.42)   ~1 min
```
The MAST side has nothing to train: it is a physics solve (`src/gs_solver.py`, `src/kinetic.py`, `src/mast_physics.py`).

## Inference entrypoint (regenerates the submitted files)
```
bash predict.sh <revision of Sophelio/fusion-equilibrium-challenge-final>
#  1. python src/cache.py --final <revision>              Final-phase inputs (organizers' Thomson time rule applied)
#  2. python src/predict_d3d.py --split private            -> submissions/final/diii_d_private_test.npz     seconds
#  3. python src/mast_physics.py --role final --kinetic --max-iter 60 --avg-last 25 --laser-row --hold-z --te 30 --jexp 1.0
#  4. python src/assemble_submission.py --split private    -> submissions/final/mast_private_test.npz
```
Then `starter/push_predictions.py` uploads the two files to a private Hugging Face dataset repo and writes the pointer zip.

## Compute budget (declared)
CPU only (Apple M4, 10 cores). Training: ~7 s for the DIII-D model (plus ~1 h of data streaming). MAST: ~17 min for the
69,075 public-test frames on 9 cores. Within the compute-light limit (end-to-end training < 2 h on one CPU).

## Data provenance (per machine)
- **DIII-D:** `diii_d_train` of `Sophelio/fusion-equilibrium-challenge` @ 1e280905 (training, calibration, local
  validation on 60 held-out shots). Official Ip time fix (`data_fixes.fix_d3d_ip_times`) applied.
- **MAST:** inputs only (coil currents, Ip, TF current, Thomson Te/ne, coil geometry, grid) of `mast_public_test` and
  `mast_private_test`. **No MAST or MAST-U equilibrium data from any source** was used to train, calibrate, validate or
  select any component, and nothing was ever scored against MAST equilibria offline: not FAIR-MAST, not the three
  labelled demo shots, not `fusion_scoring/masks/mast_envelope.npz` (enforced in code by `src/rule3_guard.py`, tests in
  `tests/test_rule3_guard.py`). MAST machine geometry (domain from the conductor positions; TF turn count 24; the
  solenoid's two-in-hand winding) is engineering data.
- **Development-phase feedback (Rule 3, disclosed):** the aggregate scores and Detailed Results of our own Development
  submissions were used to choose between complete MAST pipelines: solenoid factor, inner-wall clearance, inboard-edge
  fitting (rejected), vertical hold (kept), Te_sep 40 vs 30 eV (30 kept), Spitzer exponent 2.5 / 1.5 / 1.0 (1.0 kept).
  The full list with scores is in METHODS_REPORT.md. No probe submissions (constant, shifted or rescaled predictions).
- No external datasets and no pre-trained models.
