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
import os
import pickle
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

from .profiles import fit_cog

# --- raw data location (the external drop; not in the repo) ------------------
# Override with the HONGSHAO_DATA_DIR env var so the repo isn't machine-bound.
DEFAULT_DATA_DIR = Path(
    os.environ.get("HONGSHAO_DATA_DIR", "/Users/mac/Desktop/tng300_mah_mprof")
)
# vendored 100-snapshot cosmic-time table (Gyr) from the diffmah NERSC portal;
# value index == TNG 0-based snapshot number (verified against astropy ages).
COSMIC_TIME_PATH = Path(__file__).resolve().parent.parent / \
    "data" / "external" / "tng_cosmic_time.txt"

# --- physical constants / units ---------------------------------------------
H = 0.6774                       # TNG little-h (Planck 2015 cosmology)
N_GAL = 3388
# raw MAH masses are in 1e10 Msun/h; multiply by this to get Msun
PICKLE_MASS_UNIT = 1e10 / H
# TNG cosmology (Planck 2015), for cosmic-time <-> redshift conversions
TNG_COSMO = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486)
# halos whose z=0.4 mass is more than this fraction below their historical peak
# have turned over (stripping / tracking loss) and are excluded from analysis.
# Per-step fluctuations of a few % are normal; this is a net-decline cut.
MAH_DECLINE_TOL = 0.05

# --- radial grids (from save_tng300_072_file_structure.md) ------------------
# semi-major axes of the 7 fixed apertures in the `aper` array, kpc
SMA_KPC = np.asarray([10, 30, 50, 75, 100, 120, 150], dtype=float)
# 24 radii of the curve-of-growth `cog` array, kpc
COG_RAD_KPC = np.arange(2**0.25, 150**0.25, 0.1) ** 4
# Inner radius (kpc) for the radial-DiffMAH CoG fit. The innermost ~2-5 kpc of
# the stellar profile is only marginally resolved in TNG300-1 (softening
# ~1-2 kpc) and is the sole source of the coherent single-sigmoid fit residual
# (exp07: an inner 5 kpc cut collapses the residual ~4.6x and flattens the
# S-shape). Fitting from 5 kpc outward makes the single-sigmoid fit essentially
# exact, so the cached rdm_* params use this inner cut.
COG_FIT_RMIN_KPC = 5.0

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


def cog_sigma_dex(index: int, data_dir: Path = DEFAULT_DATA_DIR) -> np.ndarray:
    """Approximate per-point curve-of-growth uncertainty (dex), on the 24-point
    ``COG_RAD_KPC`` grid. Needed for reduced chi-square of the profile fits.

    The CoG is a cumulative integral of the isophote surface density, so we
    propagate the isophote intensity error (``intens_err``) through an
    elliptical-annulus cumulative sum (assuming independent isophote errors),
    take the *fractional* error, and interpolate it onto the CoG grid. Because
    the CoG is cumulative, the fractional error is largest where the enclosed
    mass is small: ~0.013 dex at 2 kpc, falling to ~0.003 dex by 150 kpc. This
    is comparable to the radial-DiffMAH fit residuals, so reduced chi-square is
    order unity.

    Caveats (see exp07): (1) this is isophote-modeling / azimuthal scatter, not
    Poisson measurement noise; (2) a CoG reconstructed from the isophotes does
    not exactly reproduce the stored CoG (different pipeline/grid), so only the
    *fractional* error is trusted, not the absolute mass; (3) cumulative CoG
    points are strongly correlated, so a chi-square that treats them as
    independent overcounts the degrees of freedom. Returns NaNs if the isophote
    profile is unusable.
    """
    pr = load_prof(index, data_dir)["map_hist_z0p4"]["prof"]
    r = np.asarray(pr["r_kpc"], float)
    intens = np.asarray(pr["intensity"], float)
    intens_err = np.asarray(pr["intens_err"], float)
    # a few galaxies store only (r, intensity, intens_err); fall back to circular
    ellip = (np.asarray(pr["ellipticity"], float) if "ellipticity" in pr.colnames
             else np.zeros_like(r))
    m = (np.isfinite(r) & np.isfinite(intens) & np.isfinite(intens_err)
         & np.isfinite(ellip) & (r > 0) & (intens > 0))
    r, intens, intens_err, ellip = r[m], intens[m], intens_err[m], ellip[m]
    if len(r) < 4:
        return np.full(COG_RAD_KPC.shape, np.nan)
    area = np.pi * r**2 * (1.0 - ellip)            # enclosed elliptical area
    dA = np.diff(area)
    mass = np.concatenate([[intens[0] * area[0]],
                           intens[0] * area[0]
                           + np.cumsum(0.5 * (intens[1:] + intens[:-1]) * dA)])
    var_dm = dA**2 * 0.25 * (intens_err[1:]**2 + intens_err[:-1]**2)
    var_mass = np.concatenate([[(area[0] * intens_err[0])**2],
                               (area[0] * intens_err[0])**2 + np.cumsum(var_dm)])
    frac_err = np.sqrt(var_mass) / np.maximum(mass, 1.0)
    return np.interp(COG_RAD_KPC, r, frac_err) / np.log(10.0)


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


def mah_decline_fraction(entry) -> float:
    """Fractional shortfall of the latest-snapshot mass below the peak.

    ``1 - M_final/M_peak`` from the *raw* (non-running-max) history: 0 for a
    still-rising halo, >0 if the main-branch mass has declined by z=0.4
    (stripping, turnover, or merger-tree tracking loss). NaN for empty entries.
    """
    arr = np.asarray(entry, dtype=float)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] == 0:
        return np.nan
    m = arr[1][np.argsort(arr[0])]
    return float(1.0 - m[-1] / m.max())


def logmpeak_at_snapshot(snaps, log_mpeak, target_snap: int) -> float:
    """Interpolate log10 Mpeak at an integer snapshot; nan if out of range.

    No extrapolation: snapshots beyond the recorded history (e.g. an early
    halo that never reaches z=2) return nan rather than a guessed value.
    """
    if snaps is None or target_snap < snaps[0] or target_snap > snaps[-1]:
        return np.nan
    return float(np.interp(target_snap, snaps, log_mpeak))


def load_cosmic_time(path: Path = COSMIC_TIME_PATH) -> np.ndarray:
    """Cosmic time (Gyr) per TNG snapshot; ``t[snap]`` for 0-based snapshot."""
    t = np.loadtxt(path)
    assert t.shape == (100,), t.shape
    return t


def _time_to_redshift(cosmo=TNG_COSMO, zmax=20.5, n=4000):
    """Return (ages_asc, z_asc) tables for np.interp(t_gyr) -> redshift."""
    zg = np.linspace(0.0, zmax, n)
    ages = cosmo.age(zg).value          # decreases with z
    order = np.argsort(ages)            # ascending cosmic time
    return ages[order], zg[order]


def formation_times(snaps, log_mpeak, t_of_snap, fracs=(0.5, 0.75, 0.9)):
    """Cosmic time (Gyr) when the peak mass first reaches ``frac * M0``.

    M0 is the peak mass at the latest available snapshot. The peak history is
    monotonic in time, so we interpolate the (increasing) ``log_mpeak`` against
    cosmic time. A history that already exceeds ``frac*M0`` at its earliest
    record clamps to that earliest time.
    """
    times = t_of_snap[snaps.astype(int)]      # ascending with snaps
    return np.array([np.interp(log_mpeak[-1] + np.log10(f), log_mpeak, times)
                     for f in fracs])


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
def build_dataset(data_dir: Path = DEFAULT_DATA_DIR, n_gal: int = N_GAL,
                  fit_profiles: bool = True) -> Table:
    """Assemble the per-galaxy z=0.4 table (halo MAH summaries + stellar CoG).

    Stellar masses are stored as log10(Msun). MAH summaries include anchor
    halo masses, growth fractions, and formation times/redshifts (z50/z75/z90)
    derived from the vendored TNG cosmic-time table. If ``fit_profiles`` (the
    default), the radial-DiffMAH profile parameters (rdm_*) are fit to every
    finite curve of growth and cached as columns (~4 min for the full sample;
    pass False to skip). The fit uses radii ``R >= COG_FIT_RMIN_KPC`` (5 kpc;
    the inner core is marginally resolved, see exp07), so ``rdm_logMstar0`` is
    the normalization at the innermost fitted radius and reconstructions should
    use the same inner cut.
    """
    data_dir = Path(data_dir)
    mah = load_mah(data_dir)
    t_snap = load_cosmic_time()                  # Gyr per 0-based snapshot
    ages_asc, z_asc = _time_to_redshift()        # for t_form -> z_form
    fracs = (0.5, 0.75, 0.9)

    cols = {
        "index": np.arange(n_gal),
        "logm0_halo": np.full(n_gal, np.nan),      # log10 Mpeak at latest snap (~z=0.4)
        "latest_snap": np.full(n_gal, -1, dtype=int),
        "n_mah_pts": np.zeros(n_gal, dtype=int),
        "valid_mah": np.zeros(n_gal, dtype=bool),
        "mah_decline": np.full(n_gal, np.nan),     # 1 - M(z=0.4)/M_peak
        "mah_declined": np.zeros(n_gal, dtype=bool),
        "flag": np.zeros(n_gal, dtype=bool),       # valid profile measurement
        "test": np.zeros(n_gal, dtype=bool),       # shape measurement hit default bound
        "logmstar_aper": np.full((n_gal, 7), np.nan),   # at SMA_KPC, z=0.4
        "logmstar_cog": np.full((n_gal, 24), np.nan),   # at COG_RAD_KPC, z=0.4
        "t50": np.full(n_gal, np.nan), "t75": np.full(n_gal, np.nan),
        "t90": np.full(n_gal, np.nan),             # formation cosmic times (Gyr)
        "z50": np.full(n_gal, np.nan), "z75": np.full(n_gal, np.nan),
        "z90": np.full(n_gal, np.nan),             # formation redshifts
        # radial-DiffMAH profile parameters (fit to the CoG over R>=5 kpc; see
        # profiles.py and COG_FIT_RMIN_KPC)
        "rdm_logMstar0": np.full(n_gal, np.nan),
        "rdm_beta_in": np.full(n_gal, np.nan),
        "rdm_beta_out": np.full(n_gal, np.nan),
        "rdm_R_c": np.full(n_gal, np.nan),
        "rdm_Delta": np.full(n_gal, np.nan),
        "rdm_rms": np.full(n_gal, np.nan),
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
            dec = mah_decline_fraction(mah[i])
            cols["mah_decline"][i] = dec
            cols["mah_declined"][i] = dec > MAH_DECLINE_TOL
            for z in anchor_z:
                snap = ANCHORS[z][2]
                key = f"logmpeak_z{z:g}".replace(".", "p")
                if z == 0.4:
                    cols[key][i] = lmp[-1]   # snap 72 not stored; latest ~= z=0.4
                else:
                    cols[key][i] = logmpeak_at_snapshot(snaps, lmp, snap)
            tf = formation_times(snaps, lmp, t_snap, fracs)
            zf = np.interp(tf, ages_asc, z_asc)
            cols["t50"][i], cols["t75"][i], cols["t90"][i] = tf
            cols["z50"][i], cols["z75"][i], cols["z90"][i] = zf

        aper = load_aper(i, data_dir)["z0p4"]
        with np.errstate(divide="ignore", invalid="ignore"):
            cols["logmstar_aper"][i] = np.log10(_finite_array(aper["aper"], 7))
            cols["logmstar_cog"][i] = np.log10(_finite_array(aper["cog"], 24))

        cogvec = cols["logmstar_cog"][i]
        if fit_profiles and np.all(np.isfinite(cogvec)):
            f = fit_cog(COG_RAD_KPC, cogvec, r_min=COG_FIT_RMIN_KPC)
            for p in ("logMstar0", "beta_in", "beta_out", "R_c", "Delta", "rms"):
                cols[f"rdm_{p}"][i] = f[p]

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

    # convenience: a clean analysis cut (good profile, reliable & still-rising
    # M0 above 1e13, finite CoG). Excludes halos that have turned over by z=0.4.
    tbl["use"] = (
        tbl["flag"] & tbl["valid_mah"]
        & ~tbl["mah_declined"]
        & (tbl["latest_snap"] >= 70)
        & (tbl["logm0_halo"] >= 13.0)
        & np.isfinite(tbl["logmstar_cog"]).all(axis=1)
    )

    tbl.meta.update(
        sma_kpc=SMA_KPC.tolist(),
        cog_rad_kpc=COG_RAD_KPC.tolist(),
        cog_fit_rmin_kpc=COG_FIT_RMIN_KPC,
        mass_unit="log10(Msun)",
        h=H,
        note="logm0_halo = peak mass at latest available snapshot (~z=0.4); "
             "snap 72 mass not in pickle. tXX/zXX = time/redshift when peak "
             "mass first reached XX% of M0. rdm_* fit over R>=cog_fit_rmin_kpc.",
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
        f"z50 in [{np.nanmin(tbl['z50']):.2f}, {np.nanmax(tbl['z50']):.2f}] "
        f"median {np.nanmedian(tbl['z50']):.2f}",
        f"MAH declined (>{MAH_DECLINE_TOL:.0%} below peak, excluded): "
        f"{tbl['mah_declined'].sum()}",
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
    # formation order: t50 <= t75 <= t90 (later threshold reached later)
    v = tbl["valid_mah"]
    assert np.all(tbl["t50"][v] <= tbl["t75"][v] + 1e-6)
    assert np.all(tbl["t75"][v] <= tbl["t90"][v] + 1e-6)
    assert np.all((tbl["z50"][v] >= tbl["z90"][v] - 1e-6))  # z50 is earlier (higher z)
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
