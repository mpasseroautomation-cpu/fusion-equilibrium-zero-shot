"""Proof that Rule 3 is enforced in code, not just promised.

Run:  .venv/bin/python -m pytest tests/ -q      (or: .venv/bin/python tests/test_rule3_guard.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMP / "src"))
sys.path.insert(0, str(COMP / "starter"))

import rule3_guard as guard  # noqa: E402

LABEL_KEYS = {"psi", "q95", "betaN", "li", "r_axis", "z_axis", "lcfs_n", "lcfs_r", "lcfs_z"}


def _raises(fn, *a):
    try:
        fn(*a)
    except guard.Rule3Violation:
        return True
    return False


def test_inputs_exclude_every_label():
    cols = ["efit_times", "efit_grid_R", "efit_grid_Z", "efit_psirz", "efit_q95", "efit_beta_n",
            "efit_li", "efit_lcfs_r", "magnetics_dsep", "magnetics_dsep_times", "magnetics_F1A",
            "magnetics_sol_current", "thomson_core_Te", "coil_R"]
    got = set(guard.input_columns(cols))
    assert got == {"efit_times", "efit_grid_R", "efit_grid_Z", "magnetics_F1A",
                   "magnetics_sol_current", "thomson_core_Te", "coil_R"}


def test_labels_refused_for_mast_and_for_test_splits():
    assert _raises(guard.label_columns, "MAST", "test")
    assert _raises(guard.label_columns, "MAST", "train")
    assert _raises(guard.label_columns, "DIII-D", "test")
    assert "efit_psirz" in guard.label_columns("DIII-D", "train")


def test_check_columns():
    assert _raises(guard.check_columns, ["efit_psirz"], "MAST", "test")
    assert _raises(guard.check_columns, ["magnetics_dsep"], "DIII-D", "train")
    guard.check_columns(["efit_psirz", "magnetics_F1A"], "DIII-D", "train")   # allowed: no raise


def test_banned_files():
    assert _raises(guard.check_path, COMP / "starter/parquet_data/mast_shot_28348.parquet")
    assert _raises(guard.check_path, COMP / "starter/fusion_scoring/masks/mast_envelope.npz")
    guard.check_path(COMP / "starter/parquet_data/d3d_shot_203702.parquet")  # DIII-D demo: fine
    guard.check_path(COMP / "starter/fusion_scoring/masks/d3d_envelope.npz")


def test_cache_refuses_mast_demo_shot():
    import cache
    demo = COMP / "starter/parquet_data/mast_shot_28348.parquet"
    assert _raises(cache.reduce_shot, demo, "mast_public_test", "test")


def test_no_cached_mast_file_holds_a_label():
    mast_files = sorted((COMP / "cache" / "test").glob("mast_shot_*.npz"))
    for f in mast_files[:200]:
        keys = set(np.load(f).files)
        assert not (keys & LABEL_KEYS), f"{f.name} holds label keys {keys & LABEL_KEYS}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
