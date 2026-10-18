"""THE RUN LEDGER — one line per scored run, appended, never edited.

Why: the disruption paper was saved by a log of every result with its scope. Here the same habit
also feeds the methods report (what was tried, in what order) and proves the compute-light badge
(wall time of every training run). File: results/LEDGER.csv
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
LEDGER = COMP / "results" / "LEDGER.csv"
FIELDS = ["when", "run", "code_version", "model", "machine", "train_shots", "train_frames", "eval_shots",
          "eval_frames", "S", "R2_psi", "R2_qb", "R2_q95", "R2_betaN", "D_LCFS", "Consistency",
          "cons_each", "lcfs_fail", "cons_fail", "fit_seconds", "total_seconds", "notes"]


def code_version() -> str:
    """Short git SHA of the project, '+dirty' if src/ has uncommitted changes."""
    root = COMP.parent
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "competition/src"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def log(run: str, model: str, res: dict, machine: str = "DIII-D", train_shots: int | None = None,
        train_frames: int | None = None, fit_seconds: float | None = None,
        total_seconds: float | None = None, notes: str = "") -> dict:
    row = {
        "when": dt.datetime.now().isoformat(timespec="seconds"), "run": run, "code_version": code_version(),
        "model": model, "machine": machine, "train_shots": train_shots, "train_frames": train_frames,
        "eval_shots": res.get("n_shots"), "eval_frames": res.get("n_frames"),
        "S": round(res["S"], 5), "R2_psi": round(res["r2_psi"], 5), "R2_qb": round(res["r2_qb"], 5),
        "R2_q95": round(res["r2_each"]["q95"], 4), "R2_betaN": round(res["r2_each"]["betaN"], 4),
        "D_LCFS": round(res["dlcfs"], 5), "Consistency": round(res["consistency"], 5),
        "cons_each": json.dumps({k: (None if v is None else round(v, 4)) for k, v in res["r2_cons_each"].items()}),
        "lcfs_fail": round(res["lcfs_fail_frac"], 4), "cons_fail": round(res["cons_fail_frac"], 4),
        "fit_seconds": None if fit_seconds is None else round(fit_seconds, 1),
        "total_seconds": None if total_seconds is None else round(total_seconds, 1), "notes": notes,
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row
