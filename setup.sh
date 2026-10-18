#!/usr/bin/env bash
# One-time setup: Python 3.11 venv + pinned packages + the official starter kit (MIT) at the commit we used.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3.11}
$PY -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.lock
if [ ! -d starter ]; then
  git clone https://github.com/Sophelio/fusion-equilibrium-challenge-starter starter
  git -C starter checkout 5b467988d2e5f28e20be490a926d0a71649de28d
fi
echo "setup OK"
