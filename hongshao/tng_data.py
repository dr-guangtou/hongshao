"""Loaders for the TNG300-1 snapshot-72 (z=0.4) massive-galaxy dataset.

This module turns the raw data drop (MAH pickle + per-galaxy profile/aperture
``.npy`` files) into a single analysis-ready table for the first Ultimate-SHMR
experiment. See ``doc/tng300_data.md`` for the data description and caveats.

Run as a script to build the table:

    uv run python -m hongshao.tng_data --selftest          # quick checks on 60 galaxies
    uv run python -m hongshao.tng_data --full --out data/processed/tng300_072_z0p4.fits
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from astropy.table import Table

# --- raw data location (the external drop; not in the repo) ------------------
DEFAULT_DATA_DIR = Path("/Users/mac/Desktop/tng300_mah_mprof")

# --- physical constants / units ---------------------------------------------
H = 0.6774                       # TNG little-h (Planck 2015 cosmology)
N_GAL = 3388
# raw MAH masses are in 1e10 Msun/h; multiply by this to get Msun
PICKLE_MASS_UNIT = 1e10 / H

# --- radial grids (from save_tng300_072_file_structure.md) ------------------
# semi-major axes of the 7 fixed apertures in the `aper` array, kpc
SMA_KPC = np.asarray([10, 30, 50, 75, 100, 120, 150], dtype=float)
# 24 radii of the curve-of-growth `cog` array, kpc
COG_RAD_KPC = np.arange(2**0.25, 150**0.25, 0.1) ** 4

# --- snapshot / redshift anchors --------------------------------------------
# The profiles are stored at 5 redshifts. The integer TNG snapshot numbers are
# verified against the official TNG specifications page. MAH masses are indexed
# by these same snapshot numbers, so anchor halo masses need no z<->t table.
# redshift: (aper-key, prof-key, snapshot_number)
ANCHORS = {
    0.4: ("z0p4", "map_hist_z0p4", 72),
    0.7: ("z0p7", "map_hist_z0p7", 59),
    1.0: ("z1", "map_hist_z1", 50),
    1.5: ("z1p5", "map_hist_z1p5", 40),
    2.0: ("z2", "map_hist_z2", 33),
}

# Validate the documented grids at import time (cheap, catches format drift).
assert SMA_KPC.shape == (7,)
assert COG_RAD_KPC.shape == (24,), COG_RAD_KPC.shape
assert abs(COG_RAD_KPC[0] - 2.0) < 1e-9 and abs(COG_RAD_KPC[-1] - 148.22) < 0.01


# --- low-level loaders -------------------------------------------------------
def aper_path(index: int, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    return Path(data_dir) / "save_tng300_072_hist_aper_dir" / \
        f"galaxies_tng300_072_{index}_hist_aper.npy"


def prof_path(index: int, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    return Path(data_dir) / "save_tng300_072_hist_prof" / \
        f"galaxies_tng300_072_{index}_hist.npy"


def load_aper(index: int, data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Aperture/CoG dict for one galaxy: keys z0p4..z2, each {'aper','cog'}."""
    return np.load(aper_path(index, data_dir), allow_pickle=True).item()


def load_prof(index: int, data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """Isophote-profile dict for one galaxy: keys map_hist_z0p4..z2."""
    return np.load(prof_path(index, data_dir), allow_pickle=True).item()


def load_mah(data_dir: Path = DEFAULT_DATA_DIR) -> list:
    """Load the MAH pickle: list of 3388 ``(2, N)`` arrays (snap, mass)."""
    with open(Path(data_dir) / "galaxies_tng300_072_mah_hmc.txt", "rb") as f:
        return pickle.load(f)


# --- MAH peak history --------------------------------------------------------
def peak_history(entry) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Convert one raw pickle entry to a peak-mass history.

    Returns ``(snaps, log10_mpeak)`` sorted ascending in snapshot (early ->
    late), with a running maximum enforced so the history is monotonic. The
    raw histories are already monotonic; the running max only guards against
    numerical noise. Returns ``(None, None)`` for empty entries (7 galaxies).
    """
    arr = np.asarray(entry, dtype=float)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] == 0:
        return None, None
    order = np.argsort(arr[0])
    snaps = arr[0][order]
    mass = np.maximum.accumulate(arr[1][order] * PICKLE_MASS_UNIT)
    return snaps, np.log10(mass)


def logmpeak_at_snapshot(snaps, log_mpeak, target_snap: int) -> float:
    """Interpolate log10 Mpeak at an integer snapshot; nan if out of range.

    No extrapolation: snapshots beyond the recorded history (e.g. an early
    halo that never reaches z=2) return nan rather than a guessed value.
    """
    if snaps is None or target_snap < snaps[0] or target_snap > snaps[-1]:
        return np.nan
    return float(np.interp(target_snap, snaps, log_mpeak))


def _finite_array(values, n):
    """Coerce a possibly object/None/scalar value to a length-n float array.

    Handles the awkward cases found in the drop: ``None``, a 0-d/scalar value,
    a short array, or an object array containing ``None`` bins. Missing or
    unparseable entries become ``NaN``.
    """
    out = np.full(n, np.nan)
    arr = np.atleast_1d(np.asarray(values, dtype=object)).ravel()
    for i in range(min(n, arr.shape[0])):
        v = arr[i]
        if v is None:
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out


# --- dataset assembly --------------------------------------------------------
def build_dataset(data_dir: Path = DEFAULT_DATA_DIR, n_gal: int = N_GAL) -> Table:
    """Assemble the per-galaxy z=0.4 table (halo MAH summaries + stellar CoG).

    Stellar masses are stored as log10(Msun). MAH summaries are anchor-based
    (no z<->t conversion needed); formation-time summaries (z50/z75/z90) are
    deliberately omitted until a verified full snapshot->redshift table exists.
    """
    data_dir = Path(data_dir)
    mah = load_mah(data_dir)

    cols = {
        "index": np.arange(n_gal),
        "logm0_halo": np.full(n_gal, np.nan),      # log10 Mpeak at latest snap (~z=0.4)
        "latest_snap": np.full(n_gal, -1, dtype=int),
        "n_mah_pts": np.zeros(n_gal, dtype=int),
        "valid_mah": np.zeros(n_gal, dtype=bool),
        "flag": np.zeros(n_gal, dtype=bool),       # valid profile measurement
        "test": np.zeros(n_gal, dtype=bool),       # shape measurement hit default bound
        "logmstar_aper": np.full((n_gal, 7), np.nan),   # at SMA_KPC, z=0.4
        "logmstar_cog": np.full((n_gal, 24), np.nan),   # at COG_RAD_KPC, z=0.4
    }
    # anchor halo masses + growth fractions (relative to M0)
    anchor_z = sorted(ANCHORS)  # [0.4, 0.7, 1.0, 1.5, 2.0]
    for z in anchor_z:
        cols[f"logmpeak_z{z:g}".replace(".", "p")] = np.full(n_gal, np.nan)

    for i in range(n_gal):
        snaps, lmp = peak_history(mah[i])
        if lmp is not None:
            cols["valid_mah"][i] = True
            cols["n_mah_pts"][i] = len(snaps)
            cols["latest_snap"][i] = int(snaps[-1])
            cols["logm0_halo"][i] = lmp[-1]  # peak at the latest available snapshot
            for z in anchor_z:
                snap = ANCHORS[z][2]
                key = f"logmpeak_z{z:g}".replace(".", "p")
                if z == 0.4:
                    cols[key][i] = lmp[-1]   # snap 72 not stored; latest ~= z=0.4
                else:
                    cols[key][i] = logmpeak_at_snapshot(snaps, lmp, snap)

        aper = load_aper(i, data_dir)["z0p4"]
        with np.errstate(divide="ignore", invalid="ignore"):
            cols["logmstar_aper"][i] = np.log10(_finite_array(aper["aper"], 7))
            cols["logmstar_cog"][i] = np.log10(_finite_array(aper["cog"], 24))

        z04 = load_prof(i, data_dir)["map_hist_z0p4"]
        cols["flag"][i] = bool(z04["flag"])
        cols["test"][i] = bool(z04["test"])

    tbl = Table(cols)
    # growth summaries (directions doc: f_early, f_late, inter-epoch growth)
    m0 = tbl["logm0_halo"]
    tbl["f_early"] = 10 ** (tbl["logmpeak_z2"] - m0)          # Mpeak(z=2)/M0
    tbl["f_late"] = 1.0 - 10 ** (tbl["logmpeak_z1"] - m0)     # 1 - Mpeak(z=1)/M0
    tbl["dlogm_z2_z1"] = tbl["logmpeak_z1"] - tbl["logmpeak_z2"]
    tbl["dlogm_z1_z0p4"] = m0 - tbl["logmpeak_z1"]

    # convenience: a clean analysis cut (good profile, reliable M0, above 1e13)
    tbl["use"] = (
        tbl["flag"] & tbl["valid_mah"]
        & (tbl["latest_snap"] >= 70)
        & (tbl["logm0_halo"] >= 13.0)
        & np.isfinite(tbl["logmstar_cog"]).all(axis=1)
    )

    tbl.meta.update(
        sma_kpc=SMA_KPC.tolist(),
        cog_rad_kpc=COG_RAD_KPC.tolist(),
        mass_unit="log10(Msun)",
        h=H,
        note="logm0_halo = peak mass at latest available snapshot (~z=0.4); "
             "snap 72 mass not in pickle.",
    )
    return tbl


def qc_summary(tbl: Table) -> str:
    n = len(tbl)
    lines = [
        f"galaxies: {n}",
        f"valid MAH: {tbl['valid_mah'].sum()}  (empty: {(~tbl['valid_mah']).sum()})",
        f"profile flag True: {tbl['flag'].sum()}",
        f"latest_snap == 71: {(tbl['latest_snap'] == 71).sum()}  "
        f"< 70: {(tbl['latest_snap'] < 70).sum()}",
        f"logM0 in [{np.nanmin(tbl['logm0_halo']):.2f}, "
        f"{np.nanmax(tbl['logm0_halo']):.2f}]; below 1e13: "
        f"{(tbl['logm0_halo'] < 13.0).sum()}",
        f"CoG with any non-finite bin: "
        f"{(~np.isfinite(tbl['logmstar_cog']).all(axis=1)).sum()}",
        f"Mpeak(z=2) measured (history reaches snap 33): "
        f"{np.isfinite(tbl['logmpeak_z2']).sum()}",
        f"clean analysis cut 'use': {tbl['use'].sum()}",
    ]
    return "\n".join(lines)


def _selftest(data_dir: Path):
    """Cheap correctness checks on a small subsample (the runnable check)."""
    tbl = build_dataset(data_dir, n_gal=60)
    assert len(tbl) == 60
    assert tbl["logmstar_cog"].shape == (60, 24)
    assert tbl["logmstar_aper"].shape == (60, 7)
    # peak history must be monotonic non-decreasing where valid
    mah = load_mah(data_dir)
    for i in range(60):
        snaps, lmp = peak_history(mah[i])
        if lmp is not None:
            assert np.all(np.diff(lmp) >= -1e-9), f"non-monotonic MAH at {i}"
            assert np.all(np.diff(snaps) > 0), f"unsorted snaps at {i}"
    # anchor masses must not exceed M0 (mass grows with time)
    valid = tbl["valid_mah"] & np.isfinite(tbl["logmpeak_z1"])
    assert np.all(tbl["logmpeak_z1"][valid] <= tbl["logm0_halo"][valid] + 1e-6)
    print("selftest OK")
    print(qc_summary(tbl))


def main():
    ap = argparse.ArgumentParser(description="Build the TNG300 z=0.4 dataset.")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--full", action="store_true", help="use all 3388 galaxies")
    ap.add_argument("--n", type=int, default=200, help="subsample size if not --full")
    ap.add_argument("--out", default=None, help="output FITS path")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if args.selftest:
        _selftest(data_dir)
        return

    n = N_GAL if args.full else args.n
    tbl = build_dataset(data_dir, n_gal=n)
    print(qc_summary(tbl))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tbl.write(out, overwrite=True)
        print(f"wrote {out}  ({len(tbl)} rows)")


if __name__ == "__main__":
    main()
