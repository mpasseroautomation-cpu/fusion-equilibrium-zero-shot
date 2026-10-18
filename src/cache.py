"""STREAM-AND-REDUCE CACHE — download one shot at a time, keep only what a model needs, delete the rest.

Why: the dataset is 104 GB and the disk has ~23 GB free. A DIII-D shot is ~13 MB on disk, but most of
those bytes are 20 kHz coil traces we only ever read at the ~240 EFIT times. So each shot is reduced
to arrays AT THE EFIT TIMES and the raw file is deleted:

  inputs (every machine, every split)            labels (DIII-D TRAINING shots only — rule3_guard)
  - every coil current, windowed mean (±1 ms)    - psi map (T, 65, 65): float16 for training shots,
    and slope (kA/ms, ±5 ms windows)                float32 for the held-out EVAL shots (exact truth)
  - Ip with the official DIII-D time fix         - q95, betaN, li, R_axis, Z_axis, LCFS outline
  - Thomson Te / ne per channel, median of the
    samples within ±10 ms of each EFIT time
  - geometry: coils, Thomson chords, grid

Roles and order (priority = what tomorrow needs first):
  eval   60 DIII-D train shots held out for scoring (same seed-11 list as eval_local.py, so the
         15 shots of the Oct 3 physics test are its first 15)
  train  DIII-D train shots for fitting (never overlaps eval)
  test   both public test folds, inputs only, with their official stream index (sorted filenames —
         verified 2026-10-05 against datasets' own data_files resolution)

Usage:
  python src/cache.py --train 300            # eval + 300 train + both test folds
  python src/cache.py --train 600            # later: top up to 600 (already-cached shots are skipped)
  python src/cache.py --only eval --limit 2  # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import socket
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
COMP = HERE.parent
sys.path.insert(0, str(COMP / "starter"))
sys.path.insert(0, str(HERE))

import rule3_guard as guard                                   # noqa: E402
from data_fixes import align_mast_thomson_core, fix_d3d_ip_times  # noqa: E402

REPO = "Sophelio/fusion-equilibrium-challenge"
REVISION = "1e280905b85f2a6fdde7e06fca8cf3a1edf447cb"   # pinned: the revision resolved on 2026-10-05
HF_FILE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
CACHE = COMP / "cache"
MANIFEST = CACHE / "manifest.jsonl"
MIN_FREE_GB = 6.0
N_EVAL = 60
CONFIG_MACHINE = {"diii_d_train": "DIII-D", "diii_d_public_test": "DIII-D", "mast_public_test": "MAST",
                  "diii_d_private_test": "DIII-D", "mast_private_test": "MAST"}
PUBLIC_CONFIGS = ("diii_d_train", "diii_d_public_test", "mast_public_test")
# Final phase (Oct 19-26): blind inputs published separately (starter issue #12 answer, 2026-10-07)
REPO_FINAL = "Sophelio/fusion-equilibrium-challenge-final"
FINAL_CONFIGS = ("diii_d_private_test", "mast_private_test")

MEAN_HALF_MS = 1.0     # coil currents: mean of samples within ±1 ms of the EFIT time
SLOPE_HALF_MS = 5.0    # slope: (mean over [t, t+5] - mean over [t-5, t]) / 5 ms
THOMSON_HALF_MS = 10.0
NEAREST_MAX_MS = 25.0  # Thomson fallback when the ±10 ms window is empty (MAST has gaps)


# ----------------------------------------------------------------------------- file lists

def list_files(repo: str = REPO, revision: str = REVISION, configs=PUBLIC_CONFIGS) -> dict[str, list[str]]:
    """Every parquet path per config, in OFFICIAL STREAM ORDER (sorted), at the pinned revision."""
    api = f"https://huggingface.co/api/datasets/{repo}/revision/{revision}"
    meta = json.load(urllib.request.urlopen(api, timeout=120))
    out = {c: [] for c in configs}
    for s in meta["siblings"]:
        f = s["rfilename"]
        for c in configs:
            if f.startswith(f"data/{c}/") and f.endswith(".parquet"):
                out[c].append(f)
    return {c: sorted(v) for c, v in out.items()}


def eval_files(train_files: list[str]) -> list[str]:
    """Same procedure as src/eval_local.py pick_files(seed=11): sorted list, Random(11).shuffle."""
    lst = sorted(train_files)
    random.Random(11).shuffle(lst)
    return lst[:N_EVAL]


def train_files(all_train: list[str], n: int, exclude: set[str]) -> list[str]:
    lst = [f for f in sorted(all_train) if f not in exclude]
    random.Random(2026).shuffle(lst)
    return lst[:n]


# ----------------------------------------------------------------------------- arrow -> numpy

def col_np(table: pa.Table, name: str):
    """One row's nested-list column -> rectangular numpy array (or None if missing/null)."""
    if name not in table.column_names:
        return None
    arr = table.column(name).combine_chunks()
    if arr.null_count == len(arr):
        return None
    shape = []
    a = arr
    while pa.types.is_list(a.type) or pa.types.is_large_list(a.type):
        offs = a.offsets.to_numpy()
        lens = np.diff(offs)
        if lens.size == 0:
            return np.zeros(shape + [0])
        if not (lens == lens[0]).all():                      # ragged: fall back to python lists
            return np.asarray(table.column(name).to_pylist()[0], dtype=object)
        shape.append(int(lens[0]))
        a = a.flatten()
    vals = a.to_numpy(zero_copy_only=False)
    if shape == []:                                          # scalar column (e.g. 'source')
        return vals[0]
    if vals.dtype == object:
        vals = vals.astype(str)
    return vals.reshape(shape)


# ----------------------------------------------------------------------------- time reduction

def _prep(t, x):
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = min(t.size, x.size)
    t, x = t[:n], x[:n]
    good = np.isfinite(t) & np.isfinite(x)
    t, x = t[good], x[good]
    if t.size > 1 and np.any(np.diff(t) < 0):
        o = np.argsort(t, kind="stable")
        t, x = t[o], x[o]
    return t, x


def window_mean(t, x, centers, half):
    """Mean of x(t) over [c-half, c+half] for each centre c; interpolation where the window is empty."""
    t, x = _prep(t, x)
    out = np.full(len(centers), np.nan)
    if t.size == 0:
        return out
    cs = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.searchsorted(t, centers - half, "left")
    hi = np.searchsorted(t, centers + half, "right")
    n = hi - lo
    ok = n > 0
    out[ok] = (cs[hi[ok]] - cs[lo[ok]]) / n[ok]
    if (~ok).any():
        out[~ok] = np.interp(centers[~ok], t, x, left=np.nan, right=np.nan)
    return out


def window_slope(t, x, centers, half):
    """(mean over [c, c+half] - mean over [c-half, c]) / half  -> units of x per ms."""
    left = window_mean(t, x, centers - half / 2, half / 2)
    right = window_mean(t, x, centers + half / 2, half / 2)
    return (right - left) / half


def thomson_at(times_ts, prof, centers, half=THOMSON_HALF_MS):
    """Per-channel median of the Thomson samples within ±half ms of each centre -> (T, n_ch)."""
    T = len(centers)
    if prof is None or np.size(prof) == 0 or times_ts is None or np.size(times_ts) == 0:
        n_ch = prof.shape[1] if (prof is not None and np.ndim(prof) == 2) else 0
        return np.full((T, n_ch), np.nan, dtype=np.float32)
    tt = np.asarray(times_ts, dtype=np.float64)
    P = np.asarray(prof, dtype=np.float64)
    # organizers' supported loader operation (starter issue #12, 2026-10-07): keep a sample only if its
    # time is later than every earlier sample's (drops padding and out-of-order samples)
    keep = tt > np.concatenate(([-np.inf], np.maximum.accumulate(tt)[:-1]))
    tt, P = tt[keep], P[keep]
    P = np.where(P > 0, P, np.nan)                           # zeros / negatives = no measurement
    o = np.argsort(tt, kind="stable")
    tt, P = tt[o], P[o]
    lo = np.searchsorted(tt, centers - half, "left")
    hi = np.searchsorted(tt, centers + half, "right")
    out = np.full((T, P.shape[1]), np.nan, dtype=np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN channel slices
        for k in range(T):
            if hi[k] > lo[k]:
                out[k] = np.nanmedian(P[lo[k]:hi[k]], axis=0)
            elif tt.size:                                     # no sample in ±half: nearest within 25 ms
                j = int(np.argmin(np.abs(tt - centers[k])))
                if abs(tt[j] - centers[k]) <= NEAREST_MAX_MS:
                    out[k] = P[j]
    return out


# ----------------------------------------------------------------------------- one shot

def reduce_shot(path: Path, config: str, role: str) -> dict:
    """Read one shot's parquet file and return the reduced arrays (no raw traces)."""
    machine = CONFIG_MACHINE[config]
    split = "train" if config == "diii_d_train" else "test"
    guard.check_path(path)
    all_cols = pq.ParquetFile(path).schema_arrow.names
    cols = guard.input_columns(all_cols)
    if role in ("train", "eval"):
        cols += [c for c in guard.label_columns(machine, split) if c in all_cols]
    guard.check_columns(cols, machine, split)
    tbl = pq.read_table(path, columns=cols)
    g = lambda name: col_np(tbl, name)                        # noqa: E731

    out: dict[str, np.ndarray] = {}
    times = np.asarray(g("efit_times"), dtype=np.float64)
    T = times.size
    out["efit_times"] = times
    out["grid_R"] = np.asarray(g("efit_grid_R"), dtype=np.float64)
    out["grid_Z"] = np.asarray(g("efit_grid_Z"), dtype=np.float64)

    # --- geometry (machine description; allowed on every machine)
    for k in ["coil_name", "coil_input_column", "thomson_chord_name"]:
        v = g(k)
        out[k] = np.asarray(v if v is not None else [], dtype=str)
    for k in ["coil_R", "coil_Z", "coil_width", "coil_height", "coil_angle1", "coil_angle2",
              "thomson_chord_R", "thomson_chord_Z"]:
        v = g(k)
        out[k] = np.asarray(v if v is not None else [], dtype=np.float64)

    # --- currents on the magnetics clock: every magnetics_* trace except the clock, dsep and Ip
    mt = np.asarray(g("magnetics_time"), dtype=np.float64)
    skip = {"magnetics_time", "magnetics_plasma_current", "magnetics_plasma_current_times"}
    cur_cols = sorted(c for c in cols if c.startswith("magnetics_") and c not in skip
                      and guard.is_input_column(c))
    means, slopes = [], []
    for c in cur_cols:
        x = g(c)
        x = np.asarray(x, dtype=np.float64) if x is not None else np.full(mt.size, np.nan)
        means.append(window_mean(mt, x, times, MEAN_HALF_MS))
        slopes.append(window_slope(mt, x, times, SLOPE_HALF_MS))
    out["current_names"] = np.asarray(cur_cols, dtype=str)
    out["currents"] = np.stack(means, axis=1).astype(np.float32) if means else np.zeros((T, 0), np.float32)
    out["current_slopes"] = np.stack(slopes, axis=1).astype(np.float32) if slopes else np.zeros((T, 0), np.float32)

    # --- plasma current, with the official DIII-D time-axis fix
    ip = np.asarray(g("magnetics_plasma_current"), dtype=np.float64)
    if machine == "DIII-D":
        row = {"magnetics_plasma_current_times": g("magnetics_plasma_current_times"),
               "magnetics_plasma_current": ip, "magnetics_time": mt}
        t_ip = fix_d3d_ip_times(row)
    else:
        t_ip = mt
    out["ip"] = window_mean(t_ip, ip, times, MEAN_HALF_MS).astype(np.float32)
    out["ip_slope"] = window_slope(t_ip, ip, times, SLOPE_HALF_MS).astype(np.float32)

    # --- Thomson (core + edge) at the EFIT times
    if machine == "MAST":
        row = {"thomson_core_R": g("thomson_core_R"),
               "thomson_core_Te": g("thomson_core_Te"), "thomson_core_ne": g("thomson_core_ne")}
        R_core, Te_core, ne_core = align_mast_thomson_core(row)
    else:
        R_core, Te_core, ne_core = g("thomson_core_R"), g("thomson_core_Te"), g("thomson_core_ne")
    ts_core = g("thomson_core_times")
    out["ts_core_R"] = np.asarray(R_core if R_core is not None else [], dtype=np.float64)
    out["ts_core_Te"] = thomson_at(ts_core, Te_core, times)
    out["ts_core_ne"] = thomson_at(ts_core, ne_core, times)
    ts_edge = g("thomson_edge_times")
    out["ts_edge_spatial"] = np.asarray(g("thomson_edge_spatial") if g("thomson_edge_spatial") is not None else [],
                                        dtype=np.float64)
    out["ts_edge_Te"] = thomson_at(ts_edge, g("thomson_edge_Te"), times)
    out["ts_edge_ne"] = thomson_at(ts_edge, g("thomson_edge_ne"), times)

    # --- labels: DIII-D training shots only (the guard already refused anything else)
    if role in ("train", "eval"):
        psi = np.asarray(g("efit_psirz"), dtype=np.float32)
        out["psi"] = psi if role == "eval" else psi.astype(np.float16)
        for k, name in [("efit_q95", "q95"), ("efit_beta_n", "betaN"), ("efit_li", "li"),
                        ("efit_r_axis", "r_axis"), ("efit_z_axis", "z_axis"), ("efit_lcfs_n", "lcfs_n")]:
            v = g(k)
            out[name] = np.asarray(v if v is not None else np.full(T, np.nan), dtype=np.float32)
        for k, name in [("efit_lcfs_r", "lcfs_r"), ("efit_lcfs_z", "lcfs_z")]:
            v = g(k)
            out[name] = np.asarray(v, dtype=np.float32) if v is not None and np.ndim(v) == 2 else np.zeros((T, 0), np.float32)

    out["meta_machine"] = np.asarray(machine)
    out["meta_config"] = np.asarray(config)
    out["meta_role"] = np.asarray(role)
    return out


def process_job(job: dict) -> dict:
    """Worker: download -> reduce -> save -> delete raw. Returns a manifest record."""
    t0 = time.time()
    rec = {k: job[k] for k in ("file", "config", "role", "stream_index")}
    dest = CACHE / job["role"] / (Path(job["file"]).stem + ".npz")
    tmpdir = Path(tempfile.mkdtemp(prefix="fec_", dir=CACHE / "_tmp"))
    raw = tmpdir / Path(job["file"]).name
    try:
        local = COMP / "data" / "eval" / Path(job["file"]).name      # reuse the 15 already on disk
        if job["role"] == "eval" and local.exists():
            src = local
        else:
            # A stalled connection must fail fast: without a socket timeout urlretrieve can hang for
            # 30+ minutes (seen 2026-10-05 00:31-01:05). 30 s without a byte -> retry.
            socket.setdefaulttimeout(30)
            for attempt in range(8):
                try:
                    urllib.request.urlretrieve(job.get("base", HF_FILE) + job["file"], raw)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 404 or attempt == 7:
                        raise
                    time.sleep(min(120, 5 * 2 ** attempt) if e.code == 429 else 5)
                except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                    if attempt == 7:
                        raise
                    time.sleep(3)
            src = raw
        rec["bytes_raw"] = src.stat().st_size
        arrays = reduce_shot(src, job["config"], job["role"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_dest = dest.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_dest, **arrays)
        os.replace(tmp_dest, dest)
        rec.update(status="ok", T=int(arrays["efit_times"].size), bytes_cache=dest.stat().st_size)
    except Exception as e:                                             # recorded, never silent
        rec.update(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-800:])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# ----------------------------------------------------------------------------- driver

def done_set() -> set[tuple[str, str]]:
    if not MANIFEST.exists():
        return set()
    out = set()
    for line in MANIFEST.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "ok":
            out.add((r["role"], r["file"]))
    return out


def build_jobs(n_train: int, only: str | None, limit: int | None) -> list[dict]:
    files = list_files()
    ev = eval_files(files["diii_d_train"])
    tr = train_files(files["diii_d_train"], n_train, set(ev))
    jobs = [dict(file=f, config="diii_d_train", role="eval", stream_index=None) for f in ev]
    first_train = tr[:min(300, n_train)]
    jobs += [dict(file=f, config="diii_d_train", role="train", stream_index=None) for f in first_train]
    for cfg in ("diii_d_public_test", "mast_public_test"):
        jobs += [dict(file=f, config=cfg, role="test", stream_index=i) for i, f in enumerate(files[cfg])]
    jobs += [dict(file=f, config="diii_d_train", role="train", stream_index=None) for f in tr[300:]]
    if only:
        jobs = [j for j in jobs if j["role"] == only]
    done = done_set()
    jobs = [j for j in jobs if (j["role"], j["file"]) not in done]
    return jobs[:limit] if limit else jobs


def build_final_jobs(revision: str, limit: int | None) -> list[dict]:
    """The blind Final-phase inputs (role 'final'), in official stream order, from the separate final repo."""
    files = list_files(REPO_FINAL, revision, FINAL_CONFIGS)
    base = f"https://huggingface.co/datasets/{REPO_FINAL}/resolve/{revision}/"
    for c in FINAL_CONFIGS:
        print(f"{c}: {len(files[c])} shots at {REPO_FINAL}@{revision}", flush=True)
    jobs = [dict(file=f, config=c, role="final", stream_index=i, base=base)
            for c in FINAL_CONFIGS for i, f in enumerate(files[c])]
    done = done_set()
    jobs = [j for j in jobs if (j["role"], j["file"]) not in done]
    return jobs[:limit] if limit else jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=300, help="number of DIII-D training shots to cache")
    ap.add_argument("--only", choices=["eval", "train", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--final", metavar="REVISION", help="cache the Final-phase inputs at this revision of the final repo")
    args = ap.parse_args()

    (CACHE / "_tmp").mkdir(parents=True, exist_ok=True)
    jobs = build_final_jobs(args.final, args.limit) if args.final else build_jobs(args.train, args.only, args.limit)
    print(f"{len(jobs)} shots to cache (already done are skipped); workers={args.workers}", flush=True)
    t0, n_ok, n_err, mb = time.time(), 0, 0, 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as pool, open(MANIFEST, "a") as man:
        pending = {}
        it = iter(jobs)

        def submit_next():
            free_gb = shutil.disk_usage(CACHE).free / 1e9
            if free_gb < MIN_FREE_GB:
                print(f"STOP: only {free_gb:.1f} GB free (< {MIN_FREE_GB})", flush=True)
                return False
            j = next(it, None)
            if j is None:
                return False
            pending[pool.submit(process_job, j)] = j
            return True

        for _ in range(args.workers * 2):
            if not submit_next():
                break
        while pending:
            fut = next(as_completed(pending))
            pending.pop(fut)
            rec = fut.result()
            man.write(json.dumps(rec) + "\n")
            man.flush()
            if rec["status"] == "ok":
                n_ok += 1
                mb += rec.get("bytes_raw", 0) / 1e6
            else:
                n_err += 1
                print(f"ERROR {rec['file']}: {rec['error']}", flush=True)
            done_n = n_ok + n_err
            if done_n % 20 == 0 or not pending:
                el = time.time() - t0
                print(f"{done_n}/{len(jobs)} ok={n_ok} err={n_err} | {mb:.0f} MB in {el/60:.1f} min "
                      f"({mb/max(el,1):.1f} MB/s) | last: {rec['role']} {Path(rec['file']).name}", flush=True)
            submit_next()
    print(f"DONE ok={n_ok} err={n_err} in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
