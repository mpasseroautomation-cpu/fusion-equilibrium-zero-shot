#!/usr/bin/env bash
# Inference on the Final-phase inputs (ganatma/fusion-equilibrium-challenge-final at REVISION):
#   usage: bash predict.sh <final-repo-revision> [name]
# Writes submissions/<name>/diii_d_private_test.npz and mast_private_test.npz.
set -euo pipefail
cd "$(dirname "$0")"
REV=$1; NAME=${2:-final}
.venv/bin/python src/cache.py --final "$REV" --workers 6
.venv/bin/python src/predict_d3d.py --split private --name "$NAME"
# MAST = Development submission 992762 ("v20", S_MAST 0.7106, the best MAST pipeline scored on the Development board):
# kinetic current profile with Spitzer exponent 1.0, Te_sep 30 eV, axis held at Z = 0, average of the last 25 of 60 iterates.
OMP_NUM_THREADS=1 .venv/bin/python src/mast_physics.py --role final --workers 9 \
    --kinetic --max-iter 60 --avg-last 25 --laser-row --hold-z --te 30 --jexp 1.0 --name "$NAME"
.venv/bin/python src/assemble_submission.py --mast "$NAME" --d3d "$NAME" --name "$NAME" --split private
echo "predictions in submissions/$NAME/  (diii_d_private_test.npz, mast_private_test.npz)"
