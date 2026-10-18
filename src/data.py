"""Load the reduced shots written by cache.py.

Every shot is a dict of numpy arrays (see cache.py for the keys). Labels exist only for DIII-D
'train' and 'eval' shots — rule3_guard made sure of that when the cache was written.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
CACHE = COMP / "cache"

D3D_BASELINE_CURRENTS = ["magnetics_ECOILA"] + [f"magnetics_F{i}{s}" for i in range(1, 10) for s in "AB"] \
    + ["magnetics_bcoil"]                                     # 20 columns + Ip = the kit's 21 inputs


def manifest(role: str | None = None, config: str | None = None) -> list[dict]:
    """Manifest records with status ok (last record wins per file)."""
    recs = {}
    if (CACHE / "manifest.jsonl").exists():
        for line in (CACHE / "manifest.jsonl").read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[(r["role"], r["file"])] = r
    out = [r for r in recs.values() if r.get("status") == "ok"
           and (role is None or r["role"] == role) and (config is None or r["config"] == config)]
    return out


def cache_path(rec: dict) -> Path:
    return CACHE / rec["role"] / (Path(rec["file"]).stem + ".npz")


def load(rec_or_path) -> dict:
    p = rec_or_path if isinstance(rec_or_path, Path) else cache_path(rec_or_path)
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["file"] = p.stem
    return d


def load_role(role: str, n: int | None = None, config: str | None = None, sort_by_stream: bool = False) -> list[dict]:
    recs = manifest(role, config)
    if sort_by_stream:
        recs = sorted(recs, key=lambda r: (r["config"], r["stream_index"]))
    else:
        recs = sorted(recs, key=lambda r: r["file"])
    if n is not None:
        recs = recs[:n]
    shots = []
    for r in recs:
        d = load(r)
        d["stream_index"] = r.get("stream_index")
        shots.append(d)
    return shots


def baseline_features(shot: dict) -> np.ndarray:
    """The kit's 21 DIII-D inputs at the EFIT times: 20 coil/solenoid/TF currents + Ip (kA)."""
    names = list(shot["current_names"])
    cols = [shot["currents"][:, names.index(c)] if c in names else np.full(len(shot["efit_times"]), np.nan)
            for c in D3D_BASELINE_CURRENTS]
    return np.column_stack(cols + [shot["ip"]]).astype(np.float64)
