"""exp27 step 2 — cross-match the 3388 snap-72 (z=0.4) galaxies to the local
DiffMAH/DiffStar catalog, and attach the official fits.

Bridge (validated): the DiffMAH catalog `diffmah_tng.h5` stores, for every z=0
halo, the position of its *main-progenitor branch* at each TNG snapshot, with the
array column index == snapshot number. So column 72 is the z=0.4 main-progenitor
position, in cMpc/h, taken from the same SUBFIND catalog as our galaxies. Our
galaxy's snap-72 SubhaloPos (from the MPB pull, ckpc/h -> /1000) therefore matches
its DiffMAH row *exactly* (to float precision) -- whenever our snap-72 subhalo is
the main progenitor of some z=0 halo. Massive z=0.4 clusters essentially always
are, so the match is unique; the few that are not (merged / off-main-branch) have
no DiffMAH counterpart and are flagged `matched=False`.

`diffmah_tng.h5` and `tng_diffstar_fits_default.h5` share the same contiguous
`halo_id` row order, so the matched row index attaches both catalogs at once.

Inputs : experiments/exp26.../outputs/subhalo_ids.fits  (index, subhalo_id, ...)
         outputs/mpb_cache/{sid}.hdf5                    (from fetch_mpb.py)
         $HONGSHAO_DATA_DIR/diffmah/{diffmah_tng,tng_diffstar_fits_default}.h5
Outputs: outputs/crossmatch.fits   per-galaxy match + DiffMAH/DiffStar params
         outputs/official_mah.npz  snapshot-aligned official main-branch MAH

Run: uv run python experiments/exp27_tng_api_crossmatch/crossmatch.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
IDS_FITS = ROOT / "experiments/exp26_differential_profiles/outputs/subhalo_ids.fits"
DATA_DIR = Path(os.environ.get("HONGSHAO_DATA_DIR", "/Users/mac/Desktop/tng300_mah_mprof"))
DIFFMAH = DATA_DIR / "diffmah" / "diffmah_tng.h5"
DIFFSTAR = DATA_DIR / "diffmah" / "tng_diffstar_fits_default.h5"
COSMIC_TIME = ROOT / "data" / "external" / "tng_cosmic_time.txt"

H = 0.6774                       # TNG little-h
MASS_UNIT = 1e10 / H             # raw (1e10 Msun/h) -> Msun
SNAP_Z04 = 72                    # the z=0.4 snapshot == diffmah column index
MATCH_TOL_CMPC = 1e-3            # 1 ckpc/h: "exact" position match
N_SNAP = 100                     # TNG full snapshot grid 0..99

# DiffMAH fit columns to attach (physically meaningful subset).
DIFFMAH_COLS = ("logmp_fit", "logmp_sim", "mah_logtc", "mah_k",
                "early_index", "late_index", "mah_slope", "t_infall", "upid")
# DiffStar fit columns (unbounded params; physical conversion is a diffstar concern).
DIFFSTAR_COLS = ("u_lgmcrit", "u_lgy_at_mcrit", "u_indx_lo", "u_indx_hi",
                 "u_tau_dep", "u_qt", "u_qs", "u_q_drop", "u_q_rejuv")


def log_msun(raw: np.ndarray) -> np.ndarray:
    """log10 mass [Msun] from raw TNG mass (1e10 Msun/h); 0 -> NaN."""
    out = np.full_like(raw, np.nan, dtype=np.float64)
    pos = raw > 0
    out[pos] = np.log10(raw[pos] * MASS_UNIT)
    return out


def main() -> None:
    gal = Table.read(IDS_FITS)
    n = len(gal)
    cache = OUT / "mpb_cache"

    # --- 1. read each galaxy's snap-72 position + official main-branch MAH ----
    pos72 = np.full((n, 3), np.nan)                 # cMpc/h
    mah_m200c = np.full((n, N_SNAP), np.nan)        # log10 M200c [Msun], snap-aligned
    mah_mstar = np.full((n, N_SNAP), np.nan)        # log10 M*(<2 r_half) [Msun]
    mah_msub = np.full((n, N_SNAP), np.nan)         # log10 total subhalo mass [Msun]
    have_mpb = np.zeros(n, bool)
    for i, sid in enumerate(gal["subhalo_id_snap72"]):
        p = cache / f"{int(sid)}.hdf5"
        if not p.exists():
            continue
        with h5py.File(p, "r") as f:
            snap = f["SnapNum"][:]
            assert snap[0] == SNAP_Z04, f"sid {sid}: MPB root snap {snap[0]} != 72"
            pos72[i] = f["SubhaloPos"][0] / 1000.0
            mah_m200c[i, snap] = log_msun(f["Group_M_Crit200"][:])
            mah_mstar[i, snap] = log_msun(f["SubhaloMassInRadType"][:, 4])
            mah_msub[i, snap] = log_msun(f["SubhaloMass"][:])
        have_mpb[i] = True
    print(f"loaded MPB for {have_mpb.sum()}/{n} galaxies", flush=True)

    # --- 2. build the DiffMAH snap-72 position tree and match -----------------
    with h5py.File(DIFFMAH, "r") as f:
        dm_pos = np.vstack([f["x"][:, SNAP_Z04], f["y"][:, SNAP_Z04],
                            f["z"][:, SNAP_Z04]]).T          # (N_halo, 3) cMpc/h
        dm = {c: f[c][:] for c in DIFFMAH_COLS}
        dm["diffmah_loss"] = f["loss"][:]
        dm_log_mah_sim = f["log_mah_sim"][:]                 # (N_halo, 100)
        dm_log_mah_fit = f["log_mah_fit"][:]
    tree = cKDTree(dm_pos)

    dist = np.full(n, np.nan)
    halo_id = np.full(n, -1, dtype=np.int64)
    ok = have_mpb & np.isfinite(pos72).all(axis=1)
    d, j = tree.query(pos72[ok])
    dist[ok] = d
    matched = np.zeros(n, bool)
    rows_ok = np.where(ok)[0]
    is_match = d < MATCH_TOL_CMPC
    matched[rows_ok[is_match]] = True
    halo_id[rows_ok[is_match]] = j[is_match]
    print(f"matched {matched.sum()}/{n}  "
          f"(median dist {np.nanmedian(dist[ok]):.2e}, "
          f"{(d >= MATCH_TOL_CMPC).sum()} beyond {MATCH_TOL_CMPC} cMpc/h)", flush=True)
    # main-branch crossings are unique -> matched halo_ids must be distinct
    mh = halo_id[matched]
    assert len(np.unique(mh)) == len(mh), "non-unique DiffMAH match!"

    # --- 3. assemble the output table ----------------------------------------
    t = Table()
    t["index"] = gal["index"]
    t["subhalo_id_snap72"] = gal["subhalo_id_snap72"]
    t["logmh_z0p4"] = gal["logmh_z0p4"]
    t["catgrp_id"] = gal["catgrp_id"]
    t["matched"] = matched
    t["halo_id"] = halo_id                 # DiffMAH/DiffStar row index, -1 if none
    t["match_dist_cmpc"] = dist.astype(np.float32)
    t["pos72_x"] = pos72[:, 0].astype(np.float32)
    t["pos72_y"] = pos72[:, 1].astype(np.float32)
    t["pos72_z"] = pos72[:, 2].astype(np.float32)

    def attach(src: dict, cols, prefix=""):
        for c in cols:
            arr = np.full(n, np.nan)
            arr[matched] = src[c][halo_id[matched]]
            t[prefix + c] = arr

    attach(dm, DIFFMAH_COLS)
    t["diffmah_loss"] = np.where(matched, dm["diffmah_loss"][halo_id], np.nan)

    with h5py.File(DIFFSTAR, "r") as f:
        ds = {c: f[c][:] for c in DIFFSTAR_COLS}
        ds["diffstar_success"] = f["success"][:]
        ds["diffstar_loss"] = f["loss"][:]
    attach(ds, DIFFSTAR_COLS)
    for c in ("diffstar_success", "diffstar_loss"):
        t[c] = np.where(matched, ds[c][halo_id], np.nan)

    OUT.mkdir(parents=True, exist_ok=True)
    t.write(OUT / "crossmatch.fits", overwrite=True)
    print(f"wrote {OUT/'crossmatch.fits'}  ({len(t)} rows, {len(t.colnames)} cols)", flush=True)

    # --- 4. snapshot-aligned MAH bundle (official main branch + DiffMAH curves) -
    cosmic_time = np.loadtxt(COSMIC_TIME) if COSMIC_TIME.exists() else np.array([])
    dm_sim = np.full((n, N_SNAP), np.nan, np.float32)
    dm_fit = np.full((n, N_SNAP), np.nan, np.float32)
    dm_sim[matched] = dm_log_mah_sim[halo_id[matched]]
    dm_fit[matched] = dm_log_mah_fit[halo_id[matched]]
    np.savez_compressed(
        OUT / "official_mah.npz",
        index=np.asarray(gal["index"]),
        halo_id=halo_id, matched=matched,
        cosmic_time_gyr=cosmic_time,                 # index == snapshot number
        log_m200c=mah_m200c.astype(np.float32),      # official SUBFIND main branch
        log_mstar_inrad=mah_mstar.astype(np.float32),
        log_msubhalo=mah_msub.astype(np.float32),
        diffmah_log_mah_sim=dm_sim,                  # what DiffMAH was fit to (Mpeak)
        diffmah_log_mah_fit=dm_fit,                  # the DiffMAH fit curve
    )
    print(f"wrote {OUT/'official_mah.npz'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
