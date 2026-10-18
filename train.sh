#!/usr/bin/env bash
# Rebuilds everything from scratch (CPU only, no GPU):
#   1. stream + reduce 360 DIII-D training shots (60 held out for local scoring)   ~1 h, mostly download
#   2. fit the DIII-D model (PCA 20 + ridge, 72 inputs)                              ~10 s (+ ~15 min scorer references)
#   3. re-derive the DIII-D calibrations used by the MAST physics: C_TOTAL 2.42 (and the DIII-D Te_sep, 40 eV)  ~1 min
# The MAST pipeline has no trained weights: it is a physics solve (src/gs_solver.py, src/mast_physics.py).
set -euo pipefail
cd "$(dirname "$0")"
.venv/bin/python src/cache.py --train 300 --workers 4
.venv/bin/python src/features.py --k 20
.venv/bin/python src/calibrate_d3d.py --n-te 200 --n-beta 150
