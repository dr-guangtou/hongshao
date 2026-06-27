"""exp27 step 1 — download the SubLink main-progenitor branch (MPB) for each of
the 3388 snap-72 (z=0.4) subhalos from the TNG API.

Each MPB hdf5 carries the official main-branch history (Group_M_Crit200,
SubhaloMass, SubhaloMassInRadType, SubhaloPos, ... over all snapshots) for one
galaxy. Row 0 is the snap-72 subhalo itself, so SubhaloPos[0] is the z=0.4
position used to cross-match against the local DiffMAH catalog (see crossmatch.py).

Resumable: a subhalo whose cache file already opens as valid hdf5 is skipped, so
re-running continues an interrupted pull. Concurrent (server-side tree extraction
is the bottleneck, ~5 s/gal serially).

API key lives OUTSIDE the repo at ~/.tng_api_key (line `TNG_API_KEY=...`).

Run: uv run python experiments/exp27_tng_api_crossmatch/fetch_mpb.py [--workers N] [--limit N]
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
from astropy.table import Table

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IDS_FITS = ROOT / "experiments/exp26_differential_profiles/outputs/subhalo_ids.fits"
CACHE = HERE / "outputs" / "mpb_cache"
API = "https://www.tng-project.org/api/TNG300-1/snapshots/72/subhalos/{sid}/sublink/mpb.hdf5"
KEY_FILE = Path.home() / ".tng_api_key"


def api_key() -> str:
    for line in KEY_FILE.read_text().splitlines():
        if line.startswith("TNG_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"no TNG_API_KEY= line in {KEY_FILE}")


def is_valid(path: Path) -> bool:
    """True if `path` is a non-empty, openable hdf5 with the fields we need."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with h5py.File(path, "r") as f:
            return "SubhaloPos" in f and "SnapNum" in f
    except Exception:
        return False


def fetch_one(sid: int, key: str, retries: int = 5) -> tuple[int, str]:
    """Download one MPB to CACHE/{sid}.hdf5. Returns (sid, status) where status is
    'ok' (downloaded), 'skip' (already cached), or 'fail:<reason>'."""
    dst = CACHE / f"{sid}.hdf5"
    if is_valid(dst):
        return sid, "skip"
    req = urllib.request.Request(API.format(sid=sid), headers={"api-key": key})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                content = resp.read()
            # validate the bytes before committing to disk
            with h5py.File(io.BytesIO(content), "r") as f:
                if "SubhaloPos" not in f:
                    return sid, "fail:no_SubhaloPos"
            tmp = dst.with_suffix(".tmp")
            tmp.write_bytes(content)
            tmp.rename(dst)
            return sid, "ok"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return sid, "fail:404_no_tree"
            time.sleep(2 ** attempt)  # 429 / 5xx -> back off and retry
        except Exception as exc:  # network hiccup -> retry
            if attempt == retries - 1:
                return sid, f"fail:{type(exc).__name__}"
            time.sleep(2 ** attempt)
    return sid, "fail:exhausted"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only first N (debug)")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    key = api_key()
    sids = [int(s) for s in Table.read(IDS_FITS)["subhalo_id_snap72"]]
    if args.limit:
        sids = sids[: args.limit]

    todo = [s for s in sids if not is_valid(CACHE / f"{s}.hdf5")]
    print(f"{len(sids)} galaxies, {len(sids) - len(todo)} already cached, "
          f"{len(todo)} to fetch with {args.workers} workers", flush=True)

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "fail": 0}
    fails: list[tuple[int, str]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, s, key): s for s in todo}
        for fut in as_completed(futs):
            sid, status = fut.result()
            done += 1
            if status.startswith("fail"):
                counts["fail"] += 1
                fails.append((sid, status))
            else:
                counts[status] += 1
            if done % 100 == 0 or done == len(todo):
                rate = done / (time.time() - t0)
                eta = (len(todo) - done) / rate if rate else 0
                print(f"  {done}/{len(todo)}  ok={counts['ok']} fail={counts['fail']}  "
                      f"{rate:.1f} gal/s  eta {eta/60:.1f} min", flush=True)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min: {counts}", flush=True)
    if fails:
        print(f"{len(fails)} failures (first 20): {fails[:20]}", flush=True)
        (HERE / "outputs" / "fetch_failures.txt").write_text(
            "\n".join(f"{s}\t{r}" for s, r in fails))


if __name__ == "__main__":
    sys.exit(main())
