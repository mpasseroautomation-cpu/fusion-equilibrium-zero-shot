"""RULE-3 GUARD — what the code is ALLOWED to read, enforced in one place.

Competition Rule 3 (effective 2026-09-23): no MAST equilibrium data, from any source, may be used to
train, calibrate, validate or SELECT anything that produces MAST predictions. MAST *inputs* (coil
currents, Ip, Thomson, geometry) without targets are fine.

Instead of trusting ourselves to remember, every loader asks this module first:
  - which columns of a shot are INPUTS (allowed for every machine and split);
  - which columns are LABELS (allowed ONLY for DIII-D training shots — never for MAST);
  - which local files are off-limits (the 3 labelled MAST demo shots; the MAST envelope mask,
    which is built from MAST equilibria — the scorer may use it, our model may not).
Any violation raises Rule3Violation. tests/test_rule3_guard.py proves each refusal works.
"""
from __future__ import annotations

from pathlib import Path


class Rule3Violation(RuntimeError):
    """Raised when code tries to touch data the competition rules forbid for that use."""


# Inputs: everything a model may read on ANY machine (train or test).
INPUT_EFIT_ALLOWED = {"efit_times", "efit_grid_R", "efit_grid_Z"}   # timestamps + grid only
FORBIDDEN_AS_INPUT = {"magnetics_dsep", "magnetics_dsep_times"}     # EFIT-derived despite the name

# Labels: supervision targets, DIII-D TRAINING shots only.
DIIID_LABELS = ["efit_psirz", "efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis",
                "efit_lcfs_n", "efit_lcfs_r", "efit_lcfs_z"]

# Files that must never be opened by data or model code.
_BANNED_NAME_PATTERNS = ("mast_shot_",)           # inside starter/parquet_data/ (labelled demo shots)
_BANNED_MASK_PATTERNS = ("mast",)                 # inside fusion_scoring/masks/


def is_input_column(name: str) -> bool:
    if name in FORBIDDEN_AS_INPUT:
        return False
    if name.startswith("efit_"):
        return name in INPUT_EFIT_ALLOWED
    return True


def input_columns(all_columns) -> list[str]:
    """The subset of a shot's columns that any model may read."""
    return [c for c in all_columns if is_input_column(c)]


def label_columns(machine: str, split: str) -> list[str]:
    """Label columns for supervision. Refuses MAST outright, and refuses every test split."""
    if machine != "DIII-D":
        raise Rule3Violation(f"labels requested for machine={machine!r}: MAST equilibria are banned (Rule 3)")
    if split != "train":
        raise Rule3Violation(f"labels requested for split={split!r}: only diii_d_train has labels")
    return list(DIIID_LABELS)


def check_columns(columns, machine: str, split: str) -> None:
    """Raise if a requested column list contains anything not allowed for (machine, split)."""
    allowed_labels = set(label_columns(machine, split)) if (machine == "DIII-D" and split == "train") else set()
    bad = [c for c in columns if not is_input_column(c) and c not in allowed_labels]
    if bad:
        raise Rule3Violation(f"columns not allowed for {machine}/{split}: {bad}")


def check_path(path) -> None:
    """Raise if a local file is one of the banned MAST equilibrium sources."""
    p = Path(path)
    parts = [s.lower() for s in p.parts]
    name = p.name.lower()
    if "parquet_data" in parts and any(name.startswith(b) for b in _BANNED_NAME_PATTERNS):
        raise Rule3Violation(f"{p}: the labelled MAST demo shots are for format inspection only (Rule 3)")
    if "masks" in parts and any(b in name for b in _BANNED_MASK_PATTERNS):
        raise Rule3Violation(f"{p}: the MAST envelope mask is built from MAST equilibria (Rule 3)")
