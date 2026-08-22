"""exp54 data prep — the FULL TNG Halo Structure record for our 2397 galaxies,
at every snapshot the catalog provides.

Why this and not exp46's `conc_history.py`: that script kept **five** epochs and
**seven** of the catalog's fields, because it was answering a question about the
five profile epochs. exp54 conditions each DEPOSIT on the halo state at its own
formation time (plan §4.4), and 39% of z=0.4 stellar mass is deposited at z>2 —
so it needs the whole history, and the user asked for the whole record rather
than a hand-picked column subset.

Source: TNG supplementary catalog "Halo Structure" (`CatalogName =
HaloStructure`, Anbajagane et al., arXiv:2109.02713), one HDF5 per snapshot at
`/api/TNG300-1/files/halo_structure.<snap>.hdf5`. Twenty snapshots exist:
2, 3, 4, 6, 8, 11, 13, 17, 21, 25, 33, 40, 50, 59, 67, 72, 78, 84, 91, 99.

**We take the sixteen with snap <= 72 and no others.** The join key is the FoF
group index `SubhaloGrNr` read from exp27's cached SubLink MAIN PROGENITOR
branches, whose root IS snapshot 72 — so snapshots 78-99 are descendants of our
selection epoch and simply have no main-branch entry to join on. Taking them
would need descendant links, and they are after the epoch we observe anyway.

Raw files are ~2.7-3.0 GB each (~45 GB total) and live OUTSIDE the repo, at
``$HONGSHAO_HALO_STRUCTURE_DIR`` (default ~/Desktop/tng300_halo_structure).
The reduced product is a few MB and is written into the repo.

Fields are DISCOVERED from the file, not hard-coded, so the whole record is
preserved whatever the catalog carries.

Run:
    PYTHONPATH=. uv run python experiments/exp54_unpinned_amplitude/\
halo_structure.py {plan|fetch|extract|verify}

  plan     list snapshots, sizes and what is already cached (no download)
  fetch    download the missing snap<=72 files into the outside-repo cache
  extract  reduce every cached file to (2397, n_snap) arrays -> outputs/
  verify   re-run exp46's join validation against the project's own c200c
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

OUTDIR = HERE / "outputs"
CACHE = Path(os.environ.get("HONGSHAO_HALO_STRUCTURE_DIR",
                            Path.home() / "Desktop/tng300_halo_structure"))
KEY_FILE = Path.home() / ".tng_api_key"
BASE = "https://www.tng-project.org/api/TNG300-1/files/"
MPB = ROOT / "experiments/exp27_tng_api_crossmatch/outputs/mpb_cache"
XMATCH = ROOT / "experiments/exp27_tng_api_crossmatch/outputs/crossmatch.fits"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
OFFICIAL = ROOT / "experiments/exp27_tng_api_crossmatch/outputs/official_mah.npz"

#: every catalog snapshot at or before our selection epoch (72 = z 0.4).
#: 78/84/91/99 exist but are DESCENDANTS of the tree root and cannot be joined.
SNAPS = (2, 3, 4, 6, 8, 11, 13, 17, 21, 25, 33, 40, 50, 59, 67, 72)
PROFILE_SNAPS = (72, 59, 50, 40, 33)          # z = 0.4 0.7 1.0 1.5 2.0


def api_key():
    if not KEY_FILE.exists():
        raise SystemExit(f"No API key at {KEY_FILE} (line 'TNG_API_KEY=...').")
    for line in KEY_FILE.read_text().splitlines():
        if line.startswith("TNG_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"no TNG_API_KEY= line in {KEY_FILE}")


def snapshot_redshifts():
    """z at each snapshot index, from the cached official MAH time axis."""
    from hongshao.tng_data import _time_to_redshift
    tg = np.load(OFFICIAL)["cosmic_time_gyr"]
    ta, za = _time_to_redshift()
    return np.interp(tg, ta, za)


def cmd_plan():
    zs = snapshot_redshifts()
    print(f"cache (outside the repo): {CACHE}")
    have = tot = 0
    print(f"\n  {'snap':>5}{'z':>8}{'cached':>10}{'size':>10}")
    for s in SNAPS:
        p = CACHE / f"halo_structure.{s}.hdf5"
        ok = p.exists() and p.stat().st_size > 0
        sz = p.stat().st_size / 1e9 if ok else 0.0
        have += ok
        tot += sz
        print(f"  {s:>5}{zs[s]:>8.2f}{('yes' if ok else 'NO'):>10}"
              f"{(f'{sz:.2f} GB' if ok else '-'):>10}")
    print(f"\n  {have}/{len(SNAPS)} cached, {tot:.1f} GB on disk")
    print(f"  snapshots 78/84/91/99 are DELIBERATELY excluded: they postdate "
          f"the\n  tree root (snap 72) and have no main-branch entry to join on.")


#: per-read socket timeout. NOT the whole-file timeout: a stalled TNG
#: connection must fail fast enough to retry, and a 3 GB file at a few MB/s
#: takes many minutes, so a whole-file timeout is the wrong instrument.
READ_TIMEOUT = 120
RETRIES = 12


def _remote_size(url, key):
    """Content-Length via a Range probe (the API does not answer HEAD)."""
    req = urllib.request.Request(url, headers={"api-key": key,
                                               "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as r:
        cr = r.headers.get("Content-Range")            # "bytes 0-0/12345"
    return int(cr.split("/")[-1]) if cr else None


def _fetch_one(snap, key):
    """Download one snapshot, RESUMING a partial `.part` if one exists.

    These are ~3 GB over a link that stalls: the first version used a single
    urlopen with timeout=3600, so a dead socket hung for an hour and lost the
    whole file. This resumes with an HTTP Range request, so a stall costs at
    most READ_TIMEOUT and never re-downloads what is already on disk.
    """
    dst = CACHE / f"halo_structure.{snap}.hdf5"
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  cached  snap {snap:2d}  ({dst.stat().st_size / 1e9:.2f} GB)",
              flush=True)
        return
    url = BASE + f"halo_structure.{snap}.hdf5"
    tmp = dst.with_suffix(".part")
    total = _remote_size(url, key)
    for attempt in range(1, RETRIES + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if total is not None and have >= total:
            break
        headers = {"api-key": key}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as r:
                if have and r.status != 206:
                    # server ignored the Range -- start over rather than
                    # appending a second copy of the file to the first
                    have, mode = 0, "wb"
                else:
                    mode = "ab" if have else "wb"
                with open(tmp, mode) as fh:
                    while True:
                        chunk = r.read(1 << 24)
                        if not chunk:
                            break
                        fh.write(chunk)
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                OSError) as e:
            got = tmp.stat().st_size if tmp.exists() else 0
            print(f"    snap {snap:2d} attempt {attempt}/{RETRIES} stalled at "
                  f"{got / 1e9:.2f} GB ({type(e).__name__}); resuming",
                  flush=True)
            continue
        if total is None or tmp.stat().st_size >= total:
            break
    got = tmp.stat().st_size if tmp.exists() else 0
    if total is not None and got < total:
        raise SystemExit(f"snap {snap}: gave up at {got}/{total} bytes after "
                         f"{RETRIES} attempts; re-run `fetch` to resume")
    tmp.rename(dst)          # never leave a truncated file at the real name
    print(f"  fetched snap {snap:2d}  ({dst.stat().st_size / 1e9:.2f} GB)",
          flush=True)


def cmd_fetch():
    key = api_key()
    CACHE.mkdir(parents=True, exist_ok=True)
    for s in SNAPS:
        _fetch_one(s, key)

def group_ids(snaps=SNAPS):
    """(n, n_snap) FoF group index per galaxy per snapshot, -1 where absent."""
    import h5py
    from astropy.table import Table
    xm = Table.read(XMATCH)
    pop = np.load(POP_NPZ)
    i2s = {int(i): int(s)
           for i, s in zip(xm["index"], xm["subhalo_id_snap72"])}
    gid = np.full((len(pop["index"]), len(snaps)), -1, dtype=np.int64)
    for row, idx in enumerate(pop["index"]):
        sid = i2s.get(int(idx))
        if sid is None:
            continue
        with h5py.File(MPB / f"{sid}.hdf5", "r") as f:
            sn, gr = f["SnapNum"][:], f["SubhaloGrNr"][:]
        for j, snap in enumerate(snaps):
            m = sn == snap
            if m.any():
                gid[row, j] = int(gr[m][0])
    return gid, pop


def _schema(paths):
    """(fields, header) discovered from the LARGEST cached file.

    Not from the first one: **snapshot 2 contains zero haloes** (at z=12 no
    TNG300 halo has a valid structural fit), so its file has no datasets at
    all, and taking the schema from it yields an empty field list. Discovering
    from the largest file also guarantees the widest schema if the catalog ever
    varies between snapshots.
    """
    import h5py
    for path in sorted(paths, key=lambda q: q.stat().st_size, reverse=True):
        with h5py.File(path, "r") as f:
            fields = {k: f[k].shape[1:] for k in f.keys()
                      if isinstance(f[k], h5py.Dataset)}
            header = dict(f["Header"].attrs) if "Header" in f else {}
        if fields:
            return dict(sorted(fields.items())), header
    raise SystemExit("every cached file is empty — re-run `fetch`")


def cmd_extract():
    import h5py
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cached = [CACHE / f"halo_structure.{s}.hdf5" for s in SNAPS]
    cached = [p for p in cached if p.exists() and p.stat().st_size > 0]
    if not cached:
        raise SystemExit(f"nothing cached in {CACHE} — run `fetch` first")
    fields, header = _schema(cached)
    gid, pop = group_ids()
    n, ns = gid.shape
    zs = snapshot_redshifts()

    print(f"exp54 — full halo-structure record (n={n}, {ns} snapshots)")
    print(f"  {len(fields)} fields: " + ", ".join(
        f"{k}{'' if not v else str(v)}" for k, v in fields.items()) + "\n")
    # (n, ns) for scalars, (n, ns, 3) for the vector fields (Mean_vel, sigma_1D)
    out = {k: np.full((n, ns) + shp, np.nan) for k, shp in fields.items()}
    n_empty = 0

    for j, snap in enumerate(SNAPS):
        path = CACHE / f"halo_structure.{snap}.hdf5"
        if not path.exists() or path.stat().st_size == 0:
            print(f"  snap {snap:2d} (z={zs[snap]:5.2f}): NOT CACHED — skipped",
                  flush=True)
            continue
        with h5py.File(path, "r") as f:
            present = [k for k in fields if k in f]
            if not present:
                n_empty += 1
                print(f"  snap {snap:2d} (z={zs[snap]:5.2f}): catalog is EMPTY "
                      f"(no haloes with a valid fit) — left as NaN", flush=True)
                continue
            ncat = f[present[0]].shape[0]
            use = (gid[:, j] >= 0) & (gid[:, j] < ncat)
            # h5py fancy indexing needs strictly increasing, unique indices
            uniq, inv = np.unique(gid[use, j], return_inverse=True)
            for k in present:
                out[k][use, j] = np.asarray(f[k][uniq])[inv]
        flag = out["GroupFlag"][:, j] if "GroupFlag" in out else None
        nv = int(np.nansum(flag == 1)) if flag is not None else -1
        print(f"  snap {snap:2d} (z={zs[snap]:5.2f}): {int(use.sum()):4d} matched, "
              f"{nv:4d} valid fits, median c200c "
              f"{np.nanmedian(out['c200c'][:, j]):6.3f}", flush=True)

    dst = OUTDIR / "halo_structure_history.npz"
    np.savez_compressed(
        dst, index=pop["index"], gid=gid, snaps=np.array(SNAPS),
        z=zs[list(SNAPS)], fields=np.array(sorted(fields)),
        header=np.array(str(header)), **out)
    print(f"\n  wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    if n_empty:
        print(f"  {n_empty} snapshot(s) had an empty catalog and are all-NaN")
    print(f"  raw files stay OUTSIDE the repo at {CACHE}")

def cmd_verify():
    d = np.load(OUTDIR / "halo_structure_history.npz", allow_pickle=True)
    pop = np.load(POP_NPZ)
    snaps = list(d["snaps"])
    c = np.where(d["GroupFlag"] == 1, d["c200c"], np.nan)
    j72 = snaps.index(72)
    diff = c[:, j72] - pop["c200c"]
    print(f"JOIN VALIDATION at snap 72 against the project's own c200c:")
    print(f"  max |diff| = {np.nanmax(np.abs(diff)):.3e}   "
          f"({np.mean(np.abs(diff) < 1e-9):.1%} exact)")
    print(f"\n  {'snap':>5}{'z':>7}{'valid':>8}{'median c200c':>14}")
    for j, s in enumerate(snaps):
        print(f"  {s:>5}{d['z'][j]:>7.2f}{int(np.isfinite(c[:, j]).sum()):>8}"
              f"{np.nanmedian(c[:, j]):>14.3f}")
    # cross-check against exp46's five-epoch product where they overlap
    old = ROOT / "experiments/exp46_highz_ridge/outputs/conc_history.npz"
    if old.exists():
        o = np.load(old)
        worst = 0.0
        for k, s in enumerate(PROFILE_SNAPS):
            a = o["c200c"][:, k]
            b = d["c200c"][:, snaps.index(s)]
            m = np.isfinite(a) & np.isfinite(b)
            worst = max(worst, float(np.nanmax(np.abs(a[m] - b[m]))))
        print(f"\n  agreement with exp46's conc_history.npz on its 5 epochs: "
              f"max |diff| = {worst:.3e}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    {"plan": cmd_plan, "fetch": cmd_fetch, "extract": cmd_extract,
     "verify": cmd_verify}.get(cmd, lambda: (_ for _ in ()).throw(
         SystemExit(__doc__)))()
