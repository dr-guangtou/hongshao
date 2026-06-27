"""Chunked, resumable fetcher for TNG300-1 SubLink FULL merger trees.

The full-tree API generates each tree server-side on demand (~30-100 s each, and
SERIAL ONLY — concurrent requests get rejected / 503, measured in exp28), so
pulling all ~3154 matched galaxies is a multi-day campaign. This utility does it in
**chunks you run when convenient**: it keeps a fixed worklist (most massive first),
skips trees already on disk, retries transient 503s with backoff, and logs progress
so you can stop and resume anytime ("grab the next chunk").

Everything lives OUTSIDE the repo under $HONGSHAO_TNG_WORK (default
/Users/mac/work/tng)/full_trees — NOT Dropbox:
  worklist.csv         the plan: rank, index, subhalo_id, logmh (mass-descending)
  full_<sid>.hdf5      one downloaded tree per galaxy
  fetch_log.csv        append-only log of every fetch attempt
  PROGRESS.md          human-readable status + how to resume

Usage:
  uv run python scripts/fetch_full_trees.py --status        # progress, fetch nothing
  uv run python scripts/fetch_full_trees.py --next 10       # fetch next 10 undone
  uv run python scripts/fetch_full_trees.py --next 50 --gap 3
Needs the TNG API key at ~/.tng_api_key (line `TNG_API_KEY=...`, outside the repo).
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
WORKDIR = Path(__import__("os").environ.get("HONGSHAO_TNG_WORK", "/Users/mac/work/tng")) / "full_trees"
CROSSMATCH = ROOT / "experiments/exp27_tng_api_crossmatch/outputs/crossmatch.fits"
KEY_FILE = Path.home() / ".tng_api_key"
API = "https://www.tng-project.org/api/TNG300-1/snapshots/72/subhalos/{sid}/sublink/full.hdf5"
ROOT_SNAP = 72


def api_key() -> str:
    for line in KEY_FILE.read_text().splitlines():
        if line.startswith("TNG_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"no TNG_API_KEY= line in {KEY_FILE}")


def _opener(key: str):
    """urllib opener that re-attaches the api-key header across the API's 302
    redirect to the generated-file URL (otherwise the redirect 403s)."""
    class KeepAuth(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            new = super().redirect_request(req, fp, code, msg, headers, newurl)
            if new is not None:
                new.add_header("api-key", key)
            return new
    return urllib.request.build_opener(KeepAuth)


def build_worklist() -> Path:
    """Write worklist.csv (matched galaxies, mass-descending) if it doesn't exist."""
    wl = WORKDIR / "worklist.csv"
    if wl.exists():
        return wl
    from astropy.table import Table  # only needed to build the plan once
    if not CROSSMATCH.exists():
        raise SystemExit(f"need {CROSSMATCH} to build the worklist (run exp27 crossmatch)")
    t = Table.read(CROSSMATCH)
    t = t[t["matched"]]
    t.sort("logmh_z0p4")
    t.reverse()                                          # most massive first
    WORKDIR.mkdir(parents=True, exist_ok=True)
    with open(wl, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "index", "subhalo_id", "logmh_z0p4"])
        for rank, row in enumerate(t):
            w.writerow([rank, int(row["index"]), int(row["subhalo_id_snap72"]),
                        f"{float(row['logmh_z0p4']):.4f}"])
    print(f"built worklist: {wl} ({len(t)} galaxies)")
    return wl


def load_worklist() -> list[dict]:
    with open(build_worklist()) as f:
        return list(csv.DictReader(f))


def tree_path(sid: int) -> Path:
    return WORKDIR / f"full_{sid}.hdf5"


def is_done(sid: int) -> bool:
    p = tree_path(sid)
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        with h5py.File(p, "r") as f:
            return "SubhaloID" in f and "SnapNum" in f and int(f["SnapNum"][0]) == ROOT_SNAP
    except Exception:
        return False


def fetch_one(sid: int, opener, key: str, retries: int = 5):
    """Serial fetch with backoff. Returns (status, mb, nodes, seconds)."""
    if is_done(sid):
        return "skip", tree_path(sid).stat().st_size / 1e6, None, 0.0
    t0 = time.time()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API.format(sid=sid), headers={"api-key": key})
            with opener.open(req, timeout=900) as r:
                content = r.read()
            with h5py.File(io.BytesIO(content), "r") as f:
                if "SubhaloID" not in f or int(f["SnapNum"][0]) != ROOT_SNAP:
                    return "fail:bad_tree", len(content) / 1e6, None, time.time() - t0
                nodes = int(f["SnapNum"].shape[0])
            tmp = tree_path(sid).with_suffix(".tmp")
            tmp.write_bytes(content)
            tmp.rename(tree_path(sid))
            return "ok", len(content) / 1e6, nodes, time.time() - t0
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "fail:404", 0, None, time.time() - t0
            time.sleep(min(60, 2 ** attempt * 3))        # 503/5xx -> back off
        except Exception:
            if attempt == retries - 1:
                return "fail:error", 0, None, time.time() - t0
            time.sleep(min(60, 2 ** attempt * 3))
    return "fail:exhausted", 0, None, time.time() - t0


def append_log(sid, index, logmh, status, mb, nodes, secs):
    log = WORKDIR / "fetch_log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "sid", "index", "logmh", "status", "mb", "nodes", "seconds"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), sid, index, logmh,
                    status, f"{mb:.1f}", nodes if nodes else "", f"{secs:.0f}"])


def write_progress(worklist: list[dict]):
    done = [w for w in worklist if is_done(int(w["subhalo_id"]))]
    todo = [w for w in worklist if not is_done(int(w["subhalo_id"]))]
    total_mb = sum(tree_path(int(w["subhalo_id"])).stat().st_size for w in done) / 1e6
    nxt = todo[:10]
    lines = [
        "# TNG full-tree fetch — progress",
        "",
        f"_updated {datetime.now().isoformat(timespec='minutes')}_",
        "",
        f"- **done: {len(done)} / {len(worklist)}**  ({total_mb/1000:.2f} GB on disk)",
        f"- remaining: {len(todo)}",
        "",
        "## Resume — grab the next chunk",
        "```",
        "uv run python scripts/fetch_full_trees.py --next 10",
        "```",
        "",
        f"## Next up (most massive {len(nxt)} undone)",
        "| rank | logMh | subhalo_id |",
        "|---|---|---|",
        *[f"| {w['rank']} | {w['logmh_z0p4']} | {w['subhalo_id']} |" for w in nxt],
    ]
    (WORKDIR / "PROGRESS.md").write_text("\n".join(lines) + "\n")


def cmd_status(worklist):
    done = sum(is_done(int(w["subhalo_id"])) for w in worklist)
    todo = [w for w in worklist if not is_done(int(w["subhalo_id"]))]
    print(f"full_trees dir: {WORKDIR}")
    print(f"done {done}/{len(worklist)}  | remaining {len(todo)}")
    if todo:
        print("next up:", ", ".join(f"sid={w['subhalo_id']}(logMh {w['logmh_z0p4']})"
                                    for w in todo[:5]), "...")


def cmd_next(worklist, n, gap, key):
    opener = _opener(key)
    todo = [w for w in worklist if not is_done(int(w["subhalo_id"]))][:n]
    if not todo:
        print("nothing to fetch — all done.")
        return
    print(f"fetching {len(todo)} trees (most massive undone first), serial, gap {gap}s")
    ok = 0
    for i, w in enumerate(todo, 1):
        sid = int(w["subhalo_id"])
        status, mb, nodes, secs = fetch_one(sid, opener, key)
        append_log(sid, w["index"], w["logmh_z0p4"], status, mb, nodes, secs)
        flag = "OK" if status in ("ok", "skip") else "!!"
        print(f"  [{i}/{len(todo)}] {flag} sid={sid} logMh={w['logmh_z0p4']} "
              f"{status} {mb:.0f}MB {nodes or ''} {secs:.0f}s", flush=True)
        ok += status in ("ok", "skip")
        write_progress(worklist)
        if i < len(todo) and status != "skip":
            time.sleep(gap)                              # be polite to the server
    print(f"chunk done: {ok}/{len(todo)} ok. See {WORKDIR/'PROGRESS.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--next", type=int, metavar="N", help="fetch the next N undone trees")
    ap.add_argument("--status", action="store_true", help="show progress, fetch nothing")
    ap.add_argument("--gap", type=float, default=2.0, help="seconds between fetches (politeness)")
    args = ap.parse_args()

    worklist = load_worklist()
    write_progress(worklist)
    if args.status or args.next is None:
        cmd_status(worklist)
    if args.next:
        cmd_next(worklist, args.next, args.gap, api_key())


if __name__ == "__main__":
    sys.exit(main())
