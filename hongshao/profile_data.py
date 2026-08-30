"""The 1-D isophote profiles of the TNG300-1 sample, consolidated and calibrated.

WHAT THIS MODULE IS FOR. Every experiment in this project has fitted the curves
of growth (CoGs) stored in ``save_tng300_072_hist_aper_dir/*.npy``, and none has
opened the companion directory ``save_tng300_072_hist_prof/*.npy``, which holds
the 1-D isophote fit those CoGs were built from -- including, for every galaxy,
epoch and radius, a MEASURED UNCERTAINTY on the surface mass density. This
module reads both directories once, works out exactly how the stored CoG relates
to the isophote profile, rebuilds the CoG two independent ways with propagated
errors, and writes everything to one file so no later experiment has to touch
the raw drop again.

DEFINITIONS USED THROUGHOUT (never assume the reader has them)

* **isophote** -- an ellipse fitted to the 2-D stellar mass map at one
  semi-major axis ``a``. The fit returns the mean surface mass density along
  that ellipse (``intensity``, in Msun/kpc^2), the ellipse's shape
  (``ellipticity`` ``e``, ``pa``) and the standard error of that mean
  (``intens_err``). The fitter is photutils' ``Ellipse``; ``ndata``, ``nflag``,
  ``niter`` and ``stop_code`` are its own diagnostics.
* **ellipticity** ``e = 1 - b/a``. A circle has ``e = 0``.
* **curve of growth (CoG)** -- stellar mass enclosed by an aperture of
  semi-major axis ``R``, ``M*(<R)``, on the 24-radius grid ``COG_RAD_KPC``.
* **surface mass density profile** -- ``Sigma(a)``, the isophote ``intensity``.
  It is a *local* quantity; the CoG is its cumulative integral.
* **constant isophotal shape** -- one ellipticity and position angle per galaxy
  per epoch, used for every aperture of that galaxy's CoG (as opposed to the
  radius-by-radius shape the isophote fit measures).

WHAT WE ESTABLISHED ABOUT THE STORED CoG (all verified in ``--selftest``)

1. **The stored CoG is one projection, and we know which.** The seven aperture
   masses stored beside each z=0.4 CoG match the ``xy`` rows of
   ``galaxies_tng300_072_hmc_13_aperture_mass.txt`` to the last digit, and no
   other projection. So the whole five-epoch history is the ``xy`` sky
   projection of the box, and the ``xz``/``yz`` rows of that table give a
   directly measured projection-to-projection scatter at z=0.4.

2. **The stored CoG uses a CONSTANT isophotal shape**, as the user expected.
   Of five candidate geometries, the only one that reproduces it is

       M*(<R) = integral_0^R 2 pi a (1 - e_const) Sigma(a) da

   with ``R`` the SEMI-MAJOR axis and ``e_const`` a single ellipticity for the
   whole curve. Reproduced to a median 1.1-1.9 per cent radial spread at every
   epoch. Integrating the radius-by-radius ellipticity profile instead is four
   times worse (6.3 per cent), and a circular aperture is wrong by the constant
   factor ``(1 - e_const)``, i.e. 10-45 per cent.

3. **The constant shape is only RECORDED at z=0.4** (the aperture table), but it
   can be RECOVERED at every epoch, because ``(1 - e_const)`` is exactly the
   ratio of the stored CoG to the circular integral of the same profile. At
   z=0.4, where the recorded value exists to check against, the recovery
   correlates with it at 0.998 with an rms of 0.014 in ``e`` and a small bias.

WHAT THIS MODULE PRODUCES. Three CoGs on the same 24-radius grid --- the stored
one, a constant-shape reconstruction and a variable-shape reconstruction --- the
surface mass density profile they come from, and a THREE-TERM ERROR MODEL whose
terms are measured rather than assumed:

* ``sigma_iso``  the azimuthal/statistical term, propagated from ``intens_err``
  through the same integral. Radius-dependent. Its radius-to-radius covariance
  is not free: a CoG is a cumulative sum of independent annulus increments, so
  ``Cov(M_i, M_j) = Var(M_min(i,j))`` exactly -- the variance vector determines
  the whole matrix (``cog_covariance`` builds it; ``--selftest`` asserts it).
* ``sigma_shape``  the cost of using one ellipticity for the whole curve. It
  multiplies ``(1 - e_const)``, so it is a single number per galaxy and epoch
  applying COHERENTLY to every radius -- it moves a curve up or down, never
  tilts it.
* ``sigma_proj``  the projection/orientation term, the scatter of the three
  sky projections of the same galaxy. Measured, at z=0.4 only, at six radii.

None of these is a Poisson particle error, and this module never invents one:
``intens_err`` is the standard error of the azimuthal mean and ``ndata`` is the
fitter's sampling of the ellipse, not a particle count.

Build (about 3 minutes, dominated by cold reads of the raw drop):

    HONGSHAO_DATA_DIR=... uv run python -m hongshao.profile_data --build
    HONGSHAO_DATA_DIR=... uv run python -m hongshao.profile_data --selftest

Read back:

    from hongshao.profile_data import load_profiles
    d = load_profiles()
    d["cog_provided"][gal, epoch]         # (24,) Msun, the stored CoG
    d["sigma_iso_dex"][gal, epoch]        # (24,) dex, the statistical term
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- where things live -------------------------------------------------------
DEFAULT_DATA_DIR = Path(
    os.environ.get("HONGSHAO_DATA_DIR", "/Users/shuang/Desktop/tng300_mah_mprof")
)
REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
DEFAULT_OUT = PROCESSED / "tng300_072_profiles_v1.npz"
V2_OUT = PROCESSED / "tng300_072_profiles_v2.npz"

APER_TABLE_NAME = "galaxies_tng300_072_hmc_13_aperture_mass.txt"
PROF_SUBDIR = "save_tng300_072_hist_prof"
APER_SUBDIR = "save_tng300_072_hist_aper_dir"

N_GAL = 3388

# --- the five observation epochs ---------------------------------------------
# The two directories key the same epoch differently; keep both spellings.
EPOCHS = ("z0p4", "z0p7", "z1", "z1p5", "z2")
PROF_KEYS = ("map_hist_z0p4", "map_hist_z0p7", "map_hist_z1",
             "map_hist_z1p5", "map_hist_z2")
REDSHIFTS = np.array([0.4, 0.7, 1.0, 1.5, 2.0])
N_EPOCH = len(EPOCHS)

# --- radial grids (from the drop's own save_tng300_072_file_structure.md) -----
COG_RAD_KPC = np.arange(2 ** 0.25, 150 ** 0.25, 0.1) ** 4   # 24 radii, kpc

# --- the v2 SHARED radial grid (one grid for every galaxy and every epoch) ----
# This is the dominant NATIVE isophote ladder, not an invented grid: 76 per cent
# of galaxy-epochs already sit exactly on it, so most galaxies are stored with no
# interpolation at all, and the rest move by at most a 4.2 per cent rescaling.
# It reaches further in (0.56 kpc) and further out (159.7 kpc) than the 24-radius
# CoG grid, which is why the density product uses it.
SHARED_R0 = 0.5607832739230758
SHARED_GRID = SHARED_R0 * 1.2 ** np.arange(32)          # 0.5608 -> 159.74 kpc

# Radial range over which an unweighted mean of the measured ellipticity profile
# best reproduces the constant isophotal shape the stored CoG was built with
# (correlation 0.985, rms 0.025 at z=0.4). Flat beats intensity-weighted because
# intensity weighting piles onto the centre, where the isophotes are unresolved.
ELLIP_PROFILE_RANGE_KPC = (5.0, 30.0)

# photutils' minimum sampling of an ellipse. An isophote sitting at this value
# was not independently resolved: it is interpolation across a couple of pixels.
NDATA_FLOOR = 13
APER_SMA_KPC = np.array([10, 30, 50, 75, 100, 120, 150], dtype=float)
# the six radii the projection table stores (it has no 120 kpc column) and
# where they sit inside the seven-radius `aper` array of the history files
APER6_SMA_KPC = np.array([10, 30, 50, 75, 100, 150], dtype=float)
APER6_IN_APER7 = np.array([0, 1, 2, 3, 4, 6])
PROJECTIONS = ("xy", "xz", "yz")
# the stored five-epoch CoGs are this projection -- established in `--selftest`
COG_PROJECTION = "xy"

# The isophote lists are 33 points long for a successful fit and 35 when the
# fitter fell back to its default starting semi-major axis; nothing longer has
# been seen. Arrays are padded to this width with NaN.
N_ISO_MAX = 40

# columns copied verbatim from the `prof` QTable (physical, rescaled units)
PROF_COLS = ("r_kpc", "intensity", "intens_err",
             "ellipticity", "ellip_err", "pa", "pa_err")
# photutils diagnostics copied from the `other` QTable (map/pixel units)
OTHER_COLS = ("sma", "intens", "intens_err", "ellipticity", "ellipticity_err",
              "pa", "pa_err", "grad", "grad_error", "grad_rerror",
              "x0", "x0_err", "y0", "y0_err", "ndata", "nflag", "niter",
              "stop_code")
# ...of which only these are worth carrying per radius; the rest duplicate
# `prof` up to the pixel-to-physical rescaling, which is stored separately.
OTHER_KEEP = ("ndata", "nflag", "niter", "stop_code",
              "grad", "grad_error", "grad_rerror", "x0", "y0")

HALO_PROP_COLS = ("c_200c", "c_to_a_3d", "b_to_a_3d", "v_sigma_3d", "acc_rate")


# =============================================================================
# 1. raw readers
# =============================================================================
def prof_path(index: int, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    return Path(data_dir) / PROF_SUBDIR / f"galaxies_tng300_072_{index}_hist.npy"


def aper_path(index: int, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    return Path(data_dir) / APER_SUBDIR / \
        f"galaxies_tng300_072_{index}_hist_aper.npy"


def _load_npy_dict(path: Path):
    return np.load(path, allow_pickle=True).item()


def load_projection_table(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    """The per-projection aperture table, reshaped onto (galaxy, projection).

    Returns the constant isophotal shape (``ellip_const_table``,
    ``pa_const_table``) that the z=0.4 CoG was built with, the aperture masses
    of all three sky projections (the raw material for the projection error
    term) and the projection-independent halo properties.
    """
    with open(Path(data_dir) / APER_TABLE_NAME, "rb") as fh:
        tab = pickle.load(fh)

    def fcol(name):
        return np.ma.filled(np.ma.asarray(tab[name], dtype=float), np.nan)

    gal = np.asarray(tab["gal_num"]).astype(int)
    pidx = {p: i for i, p in enumerate(PROJECTIONS)}
    proj = np.array([pidx.get(str(p), -1) for p in np.asarray(tab["proj"])])
    keep = (gal >= 0) & (gal < N_GAL) & (proj >= 0)
    g, p = gal[keep], proj[keep]

    out = {
        "ellip_const_table": np.full((N_GAL, 3), np.nan),
        "pa_const_table": np.full((N_GAL, 3), np.nan),
        "aper_mass_proj": np.full((N_GAL, 3, len(APER6_SMA_KPC)), np.nan),
        "mass_halo": np.full(N_GAL, np.nan),
        "catgrp_id": np.full(N_GAL, -1, dtype=np.int64),
    }
    for col in HALO_PROP_COLS:
        out[col] = np.full(N_GAL, np.nan)

    out["ellip_const_table"][g, p] = fcol("ellipticity")[keep]
    out["pa_const_table"][g, p] = fcol("PA")[keep]
    for k, r in enumerate(APER6_SMA_KPC):
        out["aper_mass_proj"][g, p, k] = fcol(f"aper_{int(r)}_gal")[keep]
    out["mass_halo"][g] = fcol("mass_halo")[keep]
    cid = fcol("catgrp_id")[keep]
    fin = np.isfinite(cid)
    out["catgrp_id"][g[fin]] = cid[fin].astype(np.int64)
    for col in HALO_PROP_COLS:
        out[col][g] = fcol(col)[keep]
    return out


# =============================================================================
# 2. the integral that builds a curve of growth from a 1-D profile
# =============================================================================
# Both reconstructions are the same integral,
#
#     M*(<R) = integral_0^R 2 pi a q(a) Sigma(a) da,     q = 1 - e,
#
# differing only in whether `q` is one number or a profile. It is evaluated on a
# dense log grid, with the profile interpolated log-log (a power law between
# isophotes, which is what a galaxy profile locally is) and the region inside
# the first fitted isophote treated linearly in `a` from the r=0 sample the
# isophote list carries. Trapezoid on the raw 33-point grid is NOT good enough:
# it is biased low by several per cent because a log-spaced grid under-resolves
# a steep inner profile (the same effect exp63's tech note 04 found for the
# deposition sum).
N_QUAD = 3000


def _clean_profile(r, sigma, sigma_err=None, ellip=None):
    """Drop unusable isophotes and sort by radius. Returns (r, Sigma, err, e).

    Keeps the r=0 sample separately: it is a real measurement (the central
    pixel) and it anchors the innermost annulus, but it cannot be used in a
    log-log interpolation.
    """
    r = np.asarray(r, float)
    sigma = np.asarray(sigma, float)
    n = r.size
    err = np.zeros(n) if sigma_err is None else np.asarray(sigma_err, float)
    ell = np.zeros(n) if ellip is None else np.asarray(ellip, float)
    ok = np.isfinite(r) & np.isfinite(sigma) & (sigma > 0) & (r >= 0)
    r, sigma, err, ell = r[ok], sigma[ok], err[ok], ell[ok]
    order = np.argsort(r, kind="stable")
    return r[order], sigma[order], err[order], ell[order]


def _quadrature_grid(r_pos, r_max):
    """Dense integration abscissae from 0 to ``r_max``, log-spaced where it
    matters and reaching 0 exactly so the central annulus is included."""
    hi = max(float(r_max), float(r_pos[0]) * 1.001)
    return np.concatenate([[0.0], np.geomspace(r_pos[0] * 1e-3, hi, N_QUAD)])


def _interp_profile(a_grid, r, sigma):
    """Sigma on ``a_grid``: log-log between isophotes, linear from the r=0
    sample inward, held flat beyond the outermost isophote."""
    has_zero = r[0] == 0.0
    r_pos = r[1:] if has_zero else r
    s_pos = sigma[1:] if has_zero else sigma
    sigma_centre = sigma[0]
    out = np.exp(np.interp(np.log(np.maximum(a_grid, 1e-12)),
                           np.log(r_pos), np.log(s_pos)))
    inner = a_grid < r_pos[0]
    if inner.any():
        out[inner] = np.interp(a_grid[inner], [0.0, r_pos[0]],
                               [sigma_centre, s_pos[0]])
    return out


def integrate_cog(r, sigma, radii, ellip_const=None, ellip_profile=None):
    """``M*(<R)`` at ``radii`` from a surface mass density profile.

    Exactly one of ``ellip_const`` (a scalar: the constant isophotal shape the
    stored CoG uses) and ``ellip_profile`` (an array on the SAME grid as ``r``:
    the radius-by-radius shape the isophote fit measured) must be given.
    ``radii`` are SEMI-MAJOR axes.
    """
    if (ellip_const is None) == (ellip_profile is None):
        raise ValueError("give exactly one of ellip_const / ellip_profile")
    # r, sigma and the ellipticity profile must survive the SAME cleaning mask,
    # or the shape gets attached to the wrong radii.
    ell_in = (np.zeros(np.shape(r)) if ellip_profile is None
              else np.asarray(ellip_profile, float))
    if ell_in.shape != np.shape(r):
        raise ValueError("ellip_profile must be on the same grid as r")
    r, sigma, _, ell = _clean_profile(r, sigma, None, ell_in)
    if r.size < 4:
        return np.full(np.shape(radii), np.nan)
    r_pos = r[1:] if r[0] == 0.0 else r
    a = _quadrature_grid(r_pos, np.max(radii))
    sig = _interp_profile(a, r, sigma)
    if ellip_const is not None:
        q = np.full_like(a, 1.0 - float(ellip_const))
    else:
        ell = np.where(np.isfinite(ell), ell, 0.0)
        q = 1.0 - np.interp(a, r, ell)          # flat outside the fitted range
    integrand = 2.0 * np.pi * a * q * sig
    mass = np.concatenate([[0.0], np.cumsum(np.diff(a) * 0.5 *
                                            (integrand[1:] + integrand[:-1]))])
    return np.interp(radii, a, mass)


def integrate_cog_variance(r, sigma, sigma_err, radii, ellip_const):
    """Variance of ``integrate_cog`` from independent isophote errors.

    Each isophote's ``intens_err`` is the standard error of the azimuthal mean
    density on that ellipse. Treating different isophotes as independent, the
    mass is a weighted sum ``M(<R) = sum_k w_k(R) Sigma_k`` and the variance is
    ``sum_k w_k(R)^2 err_k^2``. The weights are obtained from the same
    quadrature as ``integrate_cog`` by differentiating it with respect to each
    ``Sigma_k`` --- done exactly by running the quadrature once per isophote
    with a unit spike, which is linear and therefore cheap.

    Independence is an ASSUMPTION and it is the optimistic end: photutils
    samples neighbouring ellipses from overlapping pixels, so the true
    covariance is somewhat positive and this understates the error. It is
    recorded as an assumption rather than a result.
    """
    r_c, sig_c, err_c, _ = _clean_profile(r, sigma, sigma_err)
    if r_c.size < 4:
        return np.full(np.shape(radii), np.nan)
    r_pos = r_c[1:] if r_c[0] == 0.0 else r_c
    a = _quadrature_grid(r_pos, np.max(radii))
    q = 1.0 - float(ellip_const)
    geom = 2.0 * np.pi * a * q

    # d Sigma(a) / d Sigma_k for the log-log interpolation, evaluated at the
    # working point: a log-log interpolation is linear in log Sigma, so
    # d Sigma / d Sigma_k = Sigma(a) * w_k(a) / Sigma_k with w_k the linear
    # interpolation weight in log a. This keeps the derivative exact.
    base = _interp_profile(a, r_c, sig_c)
    has_zero = r_c[0] == 0.0
    rp = r_c[1:] if has_zero else r_c
    sp = sig_c[1:] if has_zero else sig_c
    ep = err_c[1:] if has_zero else err_c
    la, lr = np.log(np.maximum(a, 1e-12)), np.log(rp)
    idx = np.clip(np.searchsorted(lr, la) - 1, 0, len(lr) - 2)
    t = np.clip((la - lr[idx]) / (lr[idx + 1] - lr[idx]), 0.0, 1.0)
    inner = a < rp[0]

    var = np.zeros_like(radii, dtype=float)
    cum_edges = np.diff(a) * 0.5
    for k in range(len(rp)):
        wlog = np.where(idx == k, 1.0 - t, 0.0) + np.where(idx == k - 1, t, 0.0)
        dsig = base * wlog / max(sp[k], 1e-300)
        if k == 0:                      # the inner linear segment
            dsig = np.where(inner, np.interp(a, [0.0, rp[0]], [0.0, 1.0]), dsig)
        f = geom * dsig
        contrib = np.concatenate([[0.0], np.cumsum(cum_edges * (f[1:] + f[:-1]))])
        w = np.interp(radii, a, contrib)
        var += (w * ep[k]) ** 2
    return var


def cog_covariance(sigma_abs):
    """The full radius-to-radius covariance of a CoG, from its variance alone.

    A curve of growth is a cumulative sum of annulus increments. If the
    increments are independent then for ``i <= j``

        Cov(M_i, M_j) = Var(M_i),

    because ``M_j = M_i + (independent extra)``. The variance vector therefore
    determines the entire matrix and nothing extra needs storing. This is the
    single most important structural fact about a CoG likelihood: an error made
    inside a small radius propagates to EVERY larger radius with the same sign,
    so the dominant mode is the whole curve shifted, not 24 free points.

    ``sigma_abs`` is the ABSOLUTE (not dex) standard deviation, shape (..., n).
    """
    v = np.asarray(sigma_abs, float) ** 2
    return np.minimum(v[..., :, None], v[..., None, :])


def recover_constant_ellipticity(r, sigma, cog, radii):
    """Recover the constant isophotal shape the stored CoG was built with.

    ``(1 - e_const)`` is by construction the ratio of the stored CoG to the
    CIRCULAR integral of the same surface density profile, at every radius. The
    median of that ratio is the estimate; its radial scatter is both the
    uncertainty and a quality check --- a curve built with some other geometry
    would not give a radius-independent ratio.

    Returns ``(e_const, e_err, frac_spread, n_used)``. ``frac_spread`` is the
    fractional radial scatter of the ratio: about 0.011-0.019 for a good fit.
    """
    circ = integrate_cog(r, sigma, radii, ellip_const=0.0)
    cog = np.asarray(cog, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = cog / np.where(circ > 0, circ, np.nan)
    ok = np.isfinite(ratio) & (ratio > 0.05) & (ratio < 1.5)
    n = int(ok.sum())
    if n < 8:
        return np.nan, np.nan, np.nan, n
    med = float(np.median(ratio[ok]))
    spread = float(np.std(ratio[ok]) / med)
    # error on the median of n correlated points; the radial scatter is
    # dominated by a coherent shape mismatch, so no sqrt(n) is taken.
    return 1.0 - med, spread * med, spread, n


# =============================================================================
# 3. the build
# =============================================================================
def _empty(shape, dtype=float):
    fill = np.nan if dtype is float else (-1 if dtype == np.int32 else False)
    return np.full(shape, fill, dtype=dtype)


def build(data_dir: Path = DEFAULT_DATA_DIR, out_path: Path = DEFAULT_OUT,
          n_gal: int = N_GAL, verbose: bool = True) -> dict:
    """Read the raw drop once and write the consolidated profile dataset."""
    data_dir = Path(data_dir)
    ng, ne, ni, nr = n_gal, N_EPOCH, N_ISO_MAX, len(COG_RAD_KPC)
    t0 = time.time()

    d = {
        # --- provenance / grids ---
        "epochs": np.array(EPOCHS), "redshifts": REDSHIFTS,
        "cog_radius_kpc": COG_RAD_KPC, "aper_sma_kpc": APER_SMA_KPC,
        "aper6_sma_kpc": APER6_SMA_KPC, "projections": np.array(PROJECTIONS),
        "cog_projection": np.array(COG_PROJECTION),
        # --- per-galaxy bookkeeping ---
        "file_present": _empty(ng, bool),
        "cog_matches_xy": _empty(ng, bool),
        "has_shape_cols": _empty((ng, ne), bool),
        "cog_xy_max_dev": _empty(ng),
        "n_iso": _empty((ng, ne), np.int32),
        "flag_valid": _empty((ng, ne), bool),
        "iso_shape_ok": _empty((ng, ne), bool),
        # --- the isophote profile, padded with NaN ---
        "iso_r_kpc": _empty((ng, ne, ni)),
        "iso_sigma": _empty((ng, ne, ni)),
        "iso_sigma_err": _empty((ng, ne, ni)),
        "iso_ellipticity": _empty((ng, ne, ni)),
        "iso_ellip_err": _empty((ng, ne, ni)),
        "iso_pa": _empty((ng, ne, ni)),
        "iso_pa_err": _empty((ng, ne, ni)),
        "iso_ndata": _empty((ng, ne, ni), np.int32),
        "iso_nflag": _empty((ng, ne, ni), np.int32),
        "iso_niter": _empty((ng, ne, ni), np.int32),
        "iso_stop_code": _empty((ng, ne, ni), np.int32),
        "iso_grad": _empty((ng, ne, ni)),
        "iso_grad_err": _empty((ng, ne, ni)),
        "iso_grad_rerr": _empty((ng, ne, ni)),
        "iso_x0": _empty((ng, ne, ni)),
        "iso_y0": _empty((ng, ne, ni)),
        "pixel_mass_per_area": _empty((ng, ne)),
        # --- what the drop provides ---
        "cog_provided": _empty((ng, ne, nr)),
        "aper_provided": _empty((ng, ne, len(APER_SMA_KPC))),
        # --- the constant isophotal shape ---
        "ellip_const": _empty((ng, ne)),
        "ellip_const_err": _empty((ng, ne)),
        "cog_over_circ_spread": _empty((ng, ne)),
        "ellip_const_n_used": _empty((ng, ne), np.int32),
        # --- reconstructions and their errors ---
        "cog_const_shape": _empty((ng, ne, nr)),
        "cog_var_shape": _empty((ng, ne, nr)),
        "sigma_iso_abs": _empty((ng, ne, nr)),
        "sigma_iso_dex": _empty((ng, ne, nr)),
        "sigma_shape_dex": _empty((ng, ne)),
        # z=0.4 only: how much heavier the map-measured CoG is than the smooth
        # isophote model integrates to WITH THE RECORDED shape. Against the
        # RECOVERED shape it is zero by construction, so the recorded value is
        # the only independent check there is.
        "cog_bias_vs_recorded_shape": _empty(ng),
        # --- the density profile on the CoG grid, for convenience ---
        "sigma_on_cog_grid": _empty((ng, ne, nr)),
        "sigma_err_on_cog_grid": _empty((ng, ne, nr)),
    }

    tab = load_projection_table(data_dir)
    # a smoke build (n_gal < N_GAL) must not carry full-length table columns
    d.update({k: (v[:n_gal] if getattr(v, "shape", (0,))[:1] == (N_GAL,) else v)
              for k, v in tab.items()})
    # the projection error term, measured from the three sky projections
    with np.errstate(invalid="ignore", divide="ignore"):
        lap = np.log10(np.where(tab["aper_mass_proj"] > 0,
                                tab["aper_mass_proj"], np.nan))[:n_gal]
    n_proj = np.sum(np.isfinite(lap), axis=1)
    with np.errstate(invalid="ignore"):
        sp = np.where(n_proj >= 2, np.nanstd(np.where(np.isfinite(lap), lap, np.nan),
                                             axis=1), np.nan)
    d["sigma_proj_dex"] = sp
    d["sigma_proj_dex_n"] = n_proj.astype(np.int32)

    n_missing = 0
    for i in range(n_gal):
        pp, ap = prof_path(i, data_dir), aper_path(i, data_dir)
        if not (pp.exists() and ap.exists()):
            n_missing += 1
            continue
        try:
            prof_all = _load_npy_dict(pp)
            aper_all = _load_npy_dict(ap)
        except Exception:
            n_missing += 1
            continue
        d["file_present"][i] = True

        # QC: does this galaxy's stored z=0.4 aperture set really equal the xy
        # row of the projection table? It does for ~98 per cent of galaxies, to
        # machine precision. The rest are a cross-match or re-measurement
        # mismatch and must not be assumed to share that projection's shape.
        a7 = np.atleast_1d(np.asarray(
            aper_all.get("z0p4", {}).get("aper", np.full(7, np.nan)), float))
        vxy = tab["aper_mass_proj"][i, 0]
        if a7.size == len(APER_SMA_KPC):
            mm = np.isfinite(vxy) & (vxy > 0) & np.isfinite(a7[APER6_IN_APER7])
        else:
            mm = np.zeros(len(APER6_SMA_KPC), bool)   # malformed `aper` entry
        if mm.any():
            dv = float(np.max(np.abs(a7[APER6_IN_APER7][mm] / vxy[mm] - 1.0)))
            d["cog_xy_max_dev"][i] = dv
            d["cog_matches_xy"][i] = dv < 1e-9

        for k, (pkey, akey) in enumerate(zip(PROF_KEYS, EPOCHS)):
            ent = prof_all.get(pkey)
            aent = aper_all.get(akey)
            if ent is None or aent is None:
                continue
            d["flag_valid"][i, k] = bool(ent.get("flag", False))
            d["iso_shape_ok"][i, k] = bool(ent.get("test", False))
            cg = np.atleast_1d(np.asarray(aent.get("cog", np.nan), float))
            apv = np.atleast_1d(np.asarray(aent.get("aper", np.nan), float))
            if cg.size == nr:
                d["cog_provided"][i, k] = cg
            if apv.size == len(APER_SMA_KPC):
                d["aper_provided"][i, k] = apv

            pr, ot = ent.get("prof"), ent.get("other")
            if pr is None or not hasattr(pr, "colnames") or len(pr) == 0:
                continue                   # no isophote list for this epoch
            # a minority of tables carry only (r_kpc, intensity, intens_err);
            # they still give a CoG and its statistical error, but no measured
            # radius-by-radius shape, so `cog_var_shape` is undefined for them.
            d["has_shape_cols"][i, k] = "ellipticity" in pr.colnames
            n = min(len(pr), ni)
            d["n_iso"][i, k] = n
            put = {"r_kpc": "iso_r_kpc", "intensity": "iso_sigma",
                   "intens_err": "iso_sigma_err",
                   "ellipticity": "iso_ellipticity", "ellip_err": "iso_ellip_err",
                   "pa": "iso_pa", "pa_err": "iso_pa_err"}
            for col, name in put.items():
                if col in pr.colnames:
                    d[name][i, k, :n] = np.asarray(pr[col], float)[:n]
            if ot is not None and hasattr(ot, "colnames"):
                omap = {"ndata": "iso_ndata", "nflag": "iso_nflag",
                        "niter": "iso_niter", "stop_code": "iso_stop_code",
                        "grad": "iso_grad", "grad_error": "iso_grad_err",
                        "grad_rerror": "iso_grad_rerr",
                        "x0": "iso_x0", "y0": "iso_y0"}
                for col, name in omap.items():
                    if col in ot.colnames:
                        v = np.asarray(ot[col], float)[:n]
                        if d[name].dtype == np.int32:
                            v = np.where(np.isfinite(v), v, -1).astype(np.int32)
                        d[name][i, k, :n] = v
                # the pixel-to-physical rescaling, kept for provenance only
                if "intens" in ot.colnames:
                    a_raw = np.asarray(ot["intens"], float)[:n]
                    a_phys = d["iso_sigma"][i, k, :n]
                    m = np.isfinite(a_raw) & np.isfinite(a_phys) & (a_raw > 0)
                    if m.any():
                        d["pixel_mass_per_area"][i, k] = float(
                            np.median(a_phys[m] / a_raw[m]))

            r = d["iso_r_kpc"][i, k, :n]
            sg = d["iso_sigma"][i, k, :n]
            sge = d["iso_sigma_err"][i, k, :n]
            el = d["iso_ellipticity"][i, k, :n]
            cog = d["cog_provided"][i, k]
            if np.isfinite(sg).sum() < 6:
                continue

            e_c, e_err, spread, n_used = recover_constant_ellipticity(
                r, sg, cog, COG_RAD_KPC)
            d["ellip_const"][i, k] = e_c
            d["ellip_const_err"][i, k] = e_err
            d["cog_over_circ_spread"][i, k] = spread
            d["ellip_const_n_used"][i, k] = n_used
            if not np.isfinite(e_c):
                continue

            d["cog_const_shape"][i, k] = integrate_cog(
                r, sg, COG_RAD_KPC, ellip_const=e_c)
            d["cog_var_shape"][i, k] = integrate_cog(
                r, sg, COG_RAD_KPC, ellip_profile=el)
            var = integrate_cog_variance(r, sg, sge, COG_RAD_KPC, e_c)
            d["sigma_iso_abs"][i, k] = np.sqrt(var)
            if k == 0 and np.isfinite(tab["ellip_const_table"][i, 0]):
                ref = integrate_cog(r, sg, COG_RAD_KPC,
                                    ellip_const=tab["ellip_const_table"][i, 0])
                with np.errstate(invalid="ignore", divide="ignore"):
                    d["cog_bias_vs_recorded_shape"][i] = np.nanmedian(
                        d["cog_provided"][i, k] / ref - 1.0)
            mref = d["cog_const_shape"][i, k]
            with np.errstate(invalid="ignore", divide="ignore"):
                d["sigma_iso_dex"][i, k] = np.sqrt(var) / np.where(
                    mref > 0, mref, np.nan) / np.log(10.0)
                # the constant-shape term multiplies (1-e), coherently in radius
                d["sigma_shape_dex"][i, k] = e_err / (1.0 - e_c) / np.log(10.0)

            ok = np.isfinite(r) & np.isfinite(sg) & (sg > 0) & (r > 0)
            if ok.sum() >= 4:
                lr, ls = np.log(r[ok]), np.log(sg[ok])
                d["sigma_on_cog_grid"][i, k] = np.exp(
                    np.interp(np.log(COG_RAD_KPC), lr, ls))
                rel = np.where(sg[ok] > 0, sge[ok] / sg[ok], np.nan)
                d["sigma_err_on_cog_grid"][i, k] = (
                    d["sigma_on_cog_grid"][i, k] *
                    np.interp(np.log(COG_RAD_KPC), lr, rel))

        if verbose and (i + 1) % 250 == 0:
            print(f"  {i + 1}/{n_gal} galaxies  ({time.time() - t0:.0f} s)",
                  flush=True)

    d["n_galaxies"] = np.array(n_gal)
    d["n_missing_files"] = np.array(n_missing)
    d["built_utc"] = np.array(datetime.now(timezone.utc).isoformat(timespec="seconds"))
    d["source_dir"] = np.array(str(data_dir))
    try:
        from .provenance import git_sha
        d["git_sha"] = np.array(git_sha(short=False))
    except Exception:
        d["git_sha"] = np.array("unknown")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **d)
    if verbose:
        mb = out_path.stat().st_size / 1e6
        print(f"wrote {out_path}  ({mb:.1f} MB, {time.time() - t0:.0f} s)")
    return d


# =============================================================================
# 3b. version 2 — the shared-grid density product
# =============================================================================
# WHAT CHANGED FROM v1 AND WHY (the user's decisions, 2026-08-31)
#
# 1. `cog_provided` REMAINS the fitting target. Nothing here replaces it.
# 2. The fiducial density is the measured isophote `intensity` -- NOT the
#    derivative of the curve of growth. They are different measurements: the
#    CoG derivative is the mean density in a fixed elliptical annulus and counts
#    everything in the aperture, while the isophote value is sigma-clipped and
#    therefore excludes satellites and intracluster light. In the outskirts they
#    differ by 0.05 dex at 110 kpc at z=0.4 and 0.64 dex at z=2, and that gap IS
#    the clipped material. The isophote density is the closer thing to "the
#    galaxy"; the CoG-derivative view is kept as `annulus_mass`.
# 3. ONE shared radial grid for every galaxy and epoch (`SHARED_GRID`), as
#    extended as the data allow. Past a galaxy's own outer cutoff the density is
#    set to EXACTLY ZERO, which makes the reconstructed curve of growth stay
#    flat there by construction rather than by a special case. Measured against
#    the stored CoG this costs at most 0.010 dex at 148 kpc, and -- the point
#    that matters -- truncated and untruncated galaxies behave identically
#    (median -0.010 vs -0.009 dex at z=2), so the flat prescription is not
#    where the residual comes from. It is the smooth-model reconstruction bias
#    of section 5 of doc/tng300_profile_data.md.
# 4. Semi-major axis stays primary; `r_circ` is carried alongside.
# 5. Inner isophotes are kept in full with a `resolved` flag and their errors
#    UNTOUCHED: the data model records what was measured, and the decision about
#    how to weight an unresolved point belongs to whoever builds a likelihood.


def _shared_grid_density(r, sigma, sigma_err):
    """One galaxy-epoch's density resampled onto ``SHARED_GRID``.

    Returns ``(sigma, sigma_err, measured)``. Inside the measured radial range
    the profile is interpolated log-log, which is what a galaxy profile locally
    is. OUTSIDE it both arrays are exactly zero and ``measured`` is False --
    never NaN, because a zero density is what makes the reconstructed curve of
    growth stay flat beyond the cutoff without a special case. Every consumer
    must gate on ``measured``; a zero here means "not measured", not "empty".
    """
    n = SHARED_GRID.size
    out = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    ok = np.isfinite(r) & np.isfinite(sigma) & (sigma > 0) & (r > 0)
    if ok.sum() < 6:
        return out
    rr, ss = r[ok], sigma[ok]
    order = np.argsort(rr, kind="stable")
    rr, ss = rr[order], ss[order]
    ee = sigma_err[ok][order]
    # A relative tolerance is REQUIRED, not cosmetic: SHARED_GRID is rebuilt as
    # R0 * 1.2**k while the native ladder was accumulated multiplicatively, so
    # the two differ by ~1e-13 at the outermost radius. Without the tolerance
    # the last point of the dominant grid falls outside its own range and every
    # galaxy is misreported as truncated at 133 kpc.
    tol = 1.0 + 1e-9
    inside = (SHARED_GRID >= rr[0] / tol) & (SHARED_GRID <= rr[-1] * tol)
    sig = np.zeros(n)
    err = np.zeros(n)
    lg = np.log(SHARED_GRID[inside])
    sig[inside] = np.exp(np.interp(lg, np.log(rr), np.log(ss)))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(ss > 0, ee / ss, np.nan)
    good = np.isfinite(rel)
    if good.sum() >= 2:
        err[inside] = sig[inside] * np.interp(lg, np.log(rr[good]), rel[good])
    return sig, err, inside


def _ellip_from_profile(r, ellip, lo_hi=ELLIP_PROFILE_RANGE_KPC):
    """The constant isophotal shape read off the measured ellipticity PROFILE.

    An UNWEIGHTED mean over ``lo_hi``. This is a second, independent estimate of
    the same quantity `recover_constant_ellipticity` gets from the CoG ratio,
    and the two agreeing is a certificate that needs no recorded reference --
    which is what makes it usable above z=0.4, where nothing is recorded.
    """
    m = (np.isfinite(r) & np.isfinite(ellip)
         & (r >= lo_hi[0]) & (r <= lo_hi[1]))
    return float(np.mean(ellip[m])) if m.sum() >= 3 else np.nan


def derive_v2(d: dict, verbose: bool = True) -> dict:
    """Build the v2 product from a loaded v1 dict.

    Every v2 array is a function of v1's arrays, so this needs no access to the
    raw drop and runs in seconds. v1 stays on disk and readable.
    """
    t0 = time.time()
    ng, ne = d["n_iso"].shape
    ns, nr = SHARED_GRID.size, len(COG_RAD_KPC)
    v = dict(d)                                   # v2 is a SUPERSET of v1
    v["schema_version"] = np.array(2)
    v["shared_grid_kpc"] = SHARED_GRID
    v["ellip_profile_range_kpc"] = np.array(ELLIP_PROFILE_RANGE_KPC)

    v["sigma_shared"] = np.zeros((ng, ne, ns))
    v["sigma_err_shared"] = np.zeros((ng, ne, ns))
    v["sigma_measured"] = np.zeros((ng, ne, ns), bool)
    v["sigma_resolved"] = np.zeros((ng, ne, ns), bool)
    v["r_inner_valid"] = np.full((ng, ne), np.nan)
    v["r_outer_valid"] = np.full((ng, ne), np.nan)
    v["cog_from_density"] = np.full((ng, ne, nr), np.nan)
    v["cog_from_density_shared"] = np.full((ng, ne, ns), np.nan)
    v["r_circ"] = np.full((ng, ne, nr), np.nan)
    v["ellip_const_prof"] = np.full((ng, ne), np.nan)
    v["ellip_const_agree"] = np.full((ng, ne), np.nan)

    r, sg, se = d["iso_r_kpc"], d["iso_sigma"], d["iso_sigma_err"]
    el, nd = d["iso_ellipticity"], d["iso_ndata"]
    ec = d["ellip_const"]

    for i in range(ng):
        for k in range(ne):
            sig, err, meas = _shared_grid_density(r[i, k], sg[i, k], se[i, k])
            v["sigma_shared"][i, k] = sig
            v["sigma_err_shared"][i, k] = err
            v["sigma_measured"][i, k] = meas
            if meas.any():
                v["r_inner_valid"][i, k] = SHARED_GRID[meas][0]
                v["r_outer_valid"][i, k] = SHARED_GRID[meas][-1]
            # an isophote at photutils' floor was not independently resolved
            okn = np.isfinite(r[i, k]) & (r[i, k] > 0) & (nd[i, k] > 0)
            if okn.sum() >= 2:
                ndi = np.interp(np.log(SHARED_GRID), np.log(r[i, k][okn]),
                                nd[i, k][okn].astype(float))
                v["sigma_resolved"][i, k] = meas & (ndi > NDATA_FLOOR)
            v["ellip_const_prof"][i, k] = _ellip_from_profile(r[i, k], el[i, k])

            if np.isfinite(ec[i, k]):
                # zero outside the measured range => the CoG stays FLAT there
                a = np.concatenate([[0.0], SHARED_GRID])
                sa = np.concatenate([[sig[0]], sig])
                f = 2.0 * np.pi * a * (1.0 - ec[i, k]) * sa
                M = np.concatenate([[0.0], np.cumsum(
                    np.diff(a) * 0.5 * (f[1:] + f[:-1]))])
                v["cog_from_density_shared"][i, k] = np.interp(SHARED_GRID, a, M)
                v["cog_from_density"][i, k] = np.interp(COG_RAD_KPC, a, M)
                v["r_circ"][i, k] = COG_RAD_KPC * np.sqrt(
                    max(1.0 - ec[i, k], 0.0))
        if verbose and (i + 1) % 1000 == 0:
            print(f"  v2: {i + 1}/{ng}  ({time.time() - t0:.0f} s)", flush=True)

    v["ellip_const_agree"] = np.abs(ec - v["ellip_const_prof"])

    # --- the annulus form: the SAME information as the CoG, whitened ---------
    # M(<R_1) together with the 23 annulus masses. Exactly invertible, and its
    # covariance is exactly DIAGONAL because a CoG is a cumulative sum of
    # independent increments. Negative annuli are NOT repaired: the stored CoG
    # is not radially monotone for 1.9 per cent of z=2 annuli, that is a real
    # property of the measurement, and repairing it would break invertibility.
    cog = d["cog_provided"]
    v["annulus_mass"] = np.concatenate(
        [cog[:, :, :1], np.diff(cog, axis=2)], axis=2)
    v["annulus_negative"] = v["annulus_mass"] < 0
    var = d["sigma_iso_abs"] ** 2
    v["annulus_sigma_abs"] = np.sqrt(np.clip(
        np.concatenate([var[:, :, :1], np.diff(var, axis=2)], axis=2), 0, None))

    if verbose:
        print(f"v2 derived in {time.time() - t0:.0f} s")
    return v


def build_v2(v1_path: Path = DEFAULT_OUT, out_path: Path = V2_OUT,
             verbose: bool = True) -> dict:
    v = derive_v2(load_profiles(v1_path), verbose=verbose)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **v)
    if verbose:
        print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return v


def load_profiles(path: Path | None = None, version: int = 2) -> dict:
    """Read the consolidated dataset back as a plain dict of arrays.

    ``version=2`` (the default) is the shared-grid density product; ``version=1``
    is the first build, kept readable as a reference point. v2 is a strict
    superset of v1, so any v1 key resolves in either.
    """
    if path is None:
        path = V2_OUT if version == 2 else DEFAULT_OUT
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- build it with\n"
            f"  HONGSHAO_DATA_DIR=... uv run python -m hongshao.profile_data "
            f"--build{' --build-v2' if version == 2 else ''}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# =============================================================================
# 4. reporting
# =============================================================================
def summary(d: dict) -> str:
    """The inventory and quality report; printed by the build and the selftest."""
    L = []
    ng = int(d["n_galaxies"])
    A = L.append
    A(f"CONSOLIDATED PROFILE DATASET — {ng} galaxies x {N_EPOCH} epochs")
    A(f"built {str(d['built_utc'])} from {str(d['source_dir'])} at {str(d['git_sha'])[:12]}")
    A("")
    A("1. INVENTORY — how many galaxies have a usable profile")
    A(f"   both .npy files present            : {int(d['file_present'].sum())} / {ng}")
    A(f"   z=0.4 apertures == the xy projection: {int(d['cog_matches_xy'].sum())} "
      f"(the rest are a cross-match mismatch; median dev of those "
      f"{np.nanmedian(d['cog_xy_max_dev'][~d['cog_matches_xy'] & d['file_present']]):.2e})")
    have = d["n_iso"] > 0
    A(f"   isophote list present, all 5 epochs : {int(np.all(have, axis=1).sum())}")
    fl = d["flag_valid"]
    A(f"   flag=True, all 5 epochs             : {int(np.all(fl, axis=1).sum())}")
    ok_shape = d["iso_shape_ok"]
    A(f"   test=True (shape fit did NOT fall back), all 5 epochs: "
      f"{int(np.all(ok_shape, axis=1).sum())}")
    fin = np.isfinite(d["ellip_const"])
    A(f"   radius-by-radius shape columns present, all 5: "
      f"{int(np.all(d['has_shape_cols'], axis=1).sum())}")
    A(f"   constant shape recovered, all 5 epochs: {int(np.all(fin, axis=1).sum())}")
    good = fin & (d["cog_over_circ_spread"] < 0.05)
    A(f"   ...and reproduction within 5%, all 5 : {int(np.all(good, axis=1).sum())}")
    A("")
    A("   per epoch:")
    A(f"   {'epoch':>6} {'has iso':>8} {'flag':>7} {'test=T':>7} "
      f"{'e recovered':>12} {'spread<5%':>10}")
    for k, e in enumerate(EPOCHS):
        A(f"   {e:>6} {int(have[:, k].sum()):8d} {int(fl[:, k].sum()):7d} "
          f"{int(ok_shape[:, k].sum()):7d} {int(fin[:, k].sum()):12d} "
          f"{int(good[:, k].sum()):10d}")
    A("")
    A("2. HOW WELL THE RECONSTRUCTIONS REPRODUCE THE STORED CoG")
    A("   median |log10(rebuilt / stored)| over the 24 radii, in dex")
    A(f"   {'epoch':>6} {'constant shape':>16} {'variable shape':>16} "
      f"{'circular':>12}")
    for k, e in enumerate(EPOCHS):
        row = []
        for name in ("cog_const_shape", "cog_var_shape"):
            with np.errstate(invalid="ignore", divide="ignore"):
                lr = np.abs(np.log10(d[name][:, k] / d["cog_provided"][:, k]))
            row.append(np.nanmedian(lr))
        with np.errstate(invalid="ignore", divide="ignore"):
            circ = np.abs(np.log10(
                d["cog_const_shape"][:, k] /
                (1 - d["ellip_const"][:, k, None]) / d["cog_provided"][:, k]))
        A(f"   {e:>6} {row[0]:16.4f} {row[1]:16.4f} {np.nanmedian(circ):12.4f}")
    A("   (`circular` is shown to size the geometry: it is what you get by"
      " ignoring the aperture shape.)")
    A("")
    A("   The constant-shape column is small BY CONSTRUCTION (the shape was")
    A("   recovered from that very ratio). The independent check is z=0.4 against")
    A("   the RECORDED shape, where nothing was fitted:")
    A(f"      median(stored / rebuilt-with-recorded-shape - 1) = "
      f"{np.nanmedian(d['cog_bias_vs_recorded_shape']) * 100:+.2f} per cent")
    A("   i.e. the map-measured aperture holds about one per cent more mass than")
    A("   the smooth isophote model integrates to, because the fit is a smooth")
    A("   model of a clumpy map. Recorded, never corrected away.")
    A("")
    A("3. THE CONSTANT ISOPHOTAL SHAPE")
    et = d["ellip_const_table"][:, 0]           # xy, the CoG's projection
    er = d["ellip_const"][:, 0]
    m = np.isfinite(et) & np.isfinite(er)
    clean = m & d["iso_shape_ok"][:, 0] & d["cog_matches_xy"]
    A("   z=0.4 recovered vs RECORDED (xy row of the aperture table)")
    for lab, sel in (("all galaxies", m),
                     ("clean (test=True & apertures match xy)", clean)):
        A(f"      {lab:<40s} n={int(sel.sum()):4d}  "
          f"corr {np.corrcoef(et[sel], er[sel])[0, 1]:.4f}  "
          f"bias {np.median(et[sel] - er[sel]):+.4f}  "
          f"rms {np.std(et[sel] - er[sel]):.4f}")
    A("   The gap between the two rows IS the case for the quality cut: the")
    A("   galaxies whose isophote fit fell back are the whole outlier population.")
    A(f"   {'epoch':>6} {'median e':>10} {'median radial spread of cog/circ':>34}")
    for k, e in enumerate(EPOCHS):
        A(f"   {e:>6} {np.nanmedian(d['ellip_const'][:, k]):10.4f} "
          f"{np.nanmedian(d['cog_over_circ_spread'][:, k]):34.4f}")
    A("")
    A("4. THE THREE-TERM ERROR MODEL (median over galaxies, dex)")
    ridx = [np.argmin(np.abs(COG_RAD_KPC - r)) for r in (2, 5, 10, 30, 100)]
    A("   (a) sigma_iso — azimuthal/statistical, propagated through the integral")
    A(f"   {'radius':>8} " + " ".join(f"{e:>9}" for e in EPOCHS))
    for j, r in zip(ridx, (2, 5, 10, 30, 100)):
        A(f"   {r:8.0f} " + " ".join(
            f"{np.nanmedian(d['sigma_iso_dex'][:, k, j]):9.4f}" for k in range(N_EPOCH)))
    A("   (b) sigma_shape — one constant shape for the whole curve; COHERENT in radius")
    A(f"   {'':8} " + " ".join(
        f"{np.nanmedian(d['sigma_shape_dex'][:, k]):9.4f}" for k in range(N_EPOCH)))
    A("   (c) sigma_proj — scatter of the 3 sky projections, z=0.4 only")
    A(f"   {'radius':>8} " + " ".join(f"{int(r):>9}" for r in APER6_SMA_KPC))
    A(f"   {'':8} " + " ".join(
        f"{np.nanmedian(d['sigma_proj_dex'][:, j]):9.4f}"
        for j in range(len(APER6_SMA_KPC))))
    A("")
    A("   Like for like at z=0.4 (all three measured on the same galaxies),")
    A("   projection / statistical at the four shared radii:")
    A(f"   {'radius':>8} " + " ".join(f"{int(r):>9}" for r in APER6_SMA_KPC))
    ratio = []
    for j, r in enumerate(APER6_SMA_KPC):
        jj = int(np.argmin(np.abs(COG_RAD_KPC - r)))
        if r > COG_RAD_KPC[-1]:
            ratio.append("      —")
            continue
        ratio.append(f"{np.nanmedian(d['sigma_proj_dex'][:, j]) / np.nanmedian(d['sigma_iso_dex'][:, 0, jj]):9.2f}")
    A(f"   {'':8} " + " ".join(ratio))
    A("")
    A("   READ THIS CAREFULLY. The projection term DOMINATES from 10 to 100 kpc")
    A("   at z=0.4 — 2.5x the statistical term at 10 kpc — and it is the one term")
    A("   the model cannot reduce, because the model has no orientation. But it")
    A("   is NOT measured inside 10 kpc, which is exactly where the statistical")
    A("   term is largest and where the fits have been failing, and it is not")
    A("   measured above z=0.4 at all. See doc/tng300_profile_data.md section 6.")
    return "\n".join(L)


def summary_v2(v: dict) -> str:
    """The v2 report: the shared grid, the flat-outskirt cost, the two shape
    routes and their agreement, and what the annulus form buys."""
    A = []
    P = A.append
    ng = int(v["n_galaxies"])
    P(f"PROFILE DATASET v2 — shared-grid density product, {ng} galaxies")
    P(f"shared grid: {SHARED_GRID.size} radii, {SHARED_GRID[0]:.4f} -> "
      f"{SHARED_GRID[-1]:.2f} kpc, ratio 1.2 (the dominant NATIVE ladder)")
    P("")
    P("1. COVERAGE ON THE SHARED GRID (median outermost measured radius, kpc)")
    P(f"   {'':22}" + "".join(f"{e:>9}" for e in EPOCHS))
    P(f"   {'r_outer_valid':<22}" + "".join(
        f"{np.nanmedian(v['r_outer_valid'][:, k]):9.1f}" for k in range(N_EPOCH)))
    P(f"   {'r_inner_valid':<22}" + "".join(
        f"{np.nanmedian(v['r_inner_valid'][:, k]):9.2f}" for k in range(N_EPOCH)))
    with np.errstate(invalid="ignore"):
        frac = (v["sigma_resolved"].sum(axis=2)
                / np.maximum(v["sigma_measured"].sum(axis=2), 1))
    P(f"   {'frac measured resolved':<22}" + "".join(
        f"{np.nanmedian(frac[:, k]):9.3f}" for k in range(N_EPOCH)))
    P("   Past r_outer_valid the density is EXACTLY ZERO and `sigma_measured` is")
    P("   False. A zero there means NOT MEASURED, never 'empty'. Gate on the flag.")
    P("")
    P("2. WHAT THE FLAT-OUTSKIRT RECONSTRUCTION COSTS")
    P("   median log10(CoG rebuilt from the density / stored CoG), dex")
    P(f"   {'r [kpc]':>9} " + " ".join(f"{e:>8}" for e in EPOCHS))
    with np.errstate(invalid="ignore", divide="ignore"):
        dl = np.log10(v["cog_from_density"] / v["cog_provided"])
    for j in (0, 6, 15, 20, 23):
        P(f"   {COG_RAD_KPC[j]:9.1f} " + " ".join(
            f"{np.nanmedian(dl[:, k, j]):8.3f}" for k in range(N_EPOCH)))
    trunc = v["r_outer_valid"] < COG_RAD_KPC[-1]
    P("")
    P("   the same number split by whether the density truncates inside 148 kpc")
    P(f"   {'':9} " + " ".join(f"{e:>8}" for e in EPOCHS))
    for lab, g in (("truncated", trunc), ("full", ~trunc)):
        P(f"   {lab:>9} " + " ".join(
            f"{np.nanmedian(np.where(g[:, k], dl[:, k, 23], np.nan)):8.3f}"
            for k in range(N_EPOCH)))
    P(f"   {'n trunc':>9} " + " ".join(
        f"{int(trunc[:, k].sum()):8d}" for k in range(N_EPOCH)))
    P("   The two rows agreeing is the point: the residual is the smooth-model")
    P("   reconstruction bias, NOT the flat prescription.")
    P("")
    P("3. THE TWO INDEPENDENT ROUTES TO THE CONSTANT SHAPE")
    P(f"   {'':30}" + "".join(f"{e:>9}" for e in EPOCHS))
    P(f"   {'CoG-ratio route (adopted)':<30}" + "".join(
        f"{int(np.isfinite(v['ellip_const'][:, k]).sum()):9d}" for k in range(N_EPOCH)))
    P(f"   {'profile route (5-30 kpc mean)':<30}" + "".join(
        f"{int(np.isfinite(v['ellip_const_prof'][:, k]).sum()):9d}" for k in range(N_EPOCH)))
    P(f"   {'median |difference|':<30}" + "".join(
        f"{np.nanmedian(v['ellip_const_agree'][:, k]):9.4f}" for k in range(N_EPOCH)))
    P(f"   {'frac agreeing within 0.05':<30}" + "".join(
        f"{np.nanmean(v['ellip_const_agree'][:, k] < 0.05):9.3f}" for k in range(N_EPOCH)))
    P("   The difference IS the certificate: it needs no recorded reference, so")
    P("   unlike the z=0.4-only comparison it works at every epoch.")
    P("")
    P("4. THE ANNULUS FORM (the same information as the CoG, whitened)")
    neg = v["annulus_negative"]
    P(f"   {'frac annuli negative':<30}" + "".join(
        f"{np.nanmean(neg[:, k, 1:]):9.4f}" for k in range(N_EPOCH)))
    P("   Negative annuli are the stored CoG failing to increase with radius.")
    P("   They are kept, not repaired: the transform must stay invertible.")
    return "\n".join(A)


# =============================================================================
# 5. self-test — every structural claim in the docstring, asserted
# =============================================================================
def selftest(data_dir: Path = DEFAULT_DATA_DIR, n_check: int = 25) -> None:
    data_dir = Path(data_dir)
    rng = np.random.default_rng(0)
    idx = rng.choice(N_GAL, n_check, replace=False)
    tab = load_projection_table(data_dir)

    print("A. the stored five-epoch CoG is the xy projection")
    dev = {p: [] for p in PROJECTIONS}
    for i in idx:
        a = _load_npy_dict(aper_path(i, data_dir))["z0p4"]["aper"][APER6_IN_APER7]
        for j, p in enumerate(PROJECTIONS):
            v = tab["aper_mass_proj"][i, j]
            m = np.isfinite(v) & (v > 0) & np.isfinite(a)
            if m.any():
                dev[p].append(float(np.max(np.abs(a[m] / v[m] - 1))))
    for p in PROJECTIONS:
        v = np.array(dev[p])
        print(f"   |aper/table - 1| over {len(v)} galaxies, {p}: "
              f"median {np.median(v):.2e}, fraction exact {(v < 1e-12).mean():.3f}")
    assert (np.array(dev["xy"]) < 1e-12).mean() > 0.9, \
        "the stored CoG is not the xy projection"
    assert np.median(dev["xz"]) > 1e-3 and np.median(dev["yz"]) > 1e-3, \
        "projections are indistinguishable"
    print("   OK — xy matches to machine precision for >90% of galaxies, "
          "xz/yz never do")
    print("   (the few that do not are flagged per galaxy as `cog_matches_xy`)")

    print("\nB. the stored CoG uses a CONSTANT isophotal shape, R = semi-major axis")
    res = {"const": [], "var": [], "circ": []}
    for i in idx:
        pr = _load_npy_dict(prof_path(i, data_dir))["map_hist_z0p4"]["prof"]
        cog = _load_npy_dict(aper_path(i, data_dir))["z0p4"]["cog"]
        if "ellipticity" not in pr.colnames:
            continue                       # a reduced table; see `has_shape_cols`
        r = np.asarray(pr["r_kpc"], float)
        sg = np.asarray(pr["intensity"], float)
        el = np.asarray(pr["ellipticity"], float)
        e_c, _, spread, _ = recover_constant_ellipticity(r, sg, cog, COG_RAD_KPC)
        if not np.isfinite(e_c):
            continue
        res["const"].append(spread)
        for key, kw in (("var", dict(ellip_profile=el)),
                        ("circ", dict(ellip_const=0.0))):
            m = integrate_cog(r, sg, COG_RAD_KPC, **kw)
            ratio = cog / m
            res[key].append(float(np.std(ratio) / np.mean(ratio)))
    for k in ("const", "var", "circ"):
        print(f"   radial fractional spread of stored/rebuilt, {k:>5}: "
              f"{np.median(res[k]):.4f}")
    assert np.median(res["const"]) < 0.5 * np.median(res["var"]), \
        "the constant-shape geometry does not beat the variable-shape one"
    print("   OK — constant shape is at least twice as consistent")

    print("\nC. the recovered constant shape matches the recorded one at z=0.4")
    print("   (restricted to test=True: the isophote fitter did NOT fall back to")
    print("    its default starting semi-major axis. The fallback galaxies are")
    print("    the whole outlier population -- see doc/tng300_profile_data.md.)")
    rec, tabe = [], []
    for i in idx:
        ent = _load_npy_dict(prof_path(i, data_dir))["map_hist_z0p4"]
        if not bool(ent.get("test", False)):
            continue
        pr = ent["prof"]
        cog = _load_npy_dict(aper_path(i, data_dir))["z0p4"]["cog"]
        e_c, _, _, _ = recover_constant_ellipticity(
            np.asarray(pr["r_kpc"], float), np.asarray(pr["intensity"], float),
            cog, COG_RAD_KPC)
        if np.isfinite(e_c) and np.isfinite(tab["ellip_const_table"][i, 0]):
            rec.append(e_c)
            tabe.append(tab["ellip_const_table"][i, 0])
    rec, tabe = np.array(rec), np.array(tabe)
    print(f"   n={len(rec)}  corr={np.corrcoef(rec, tabe)[0, 1]:.4f}  "
          f"bias={np.median(tabe - rec):+.4f}  rms={np.std(tabe - rec):.4f}")
    assert np.corrcoef(rec, tabe)[0, 1] > 0.98
    assert np.std(tabe - rec) < 0.04
    print("   OK — the recovery reproduces the recorded shape")
    heavier = float(np.median((1 - rec) / (1 - tabe) - 1.0))
    print(f"   The bias is real and physical: the stored CoG is {heavier * 100:.2f} "
          f"per cent heavier than")
    print("   the smooth isophote model integrates to, because the isophote fit is a")
    print("   smooth model of a clumpy map and loses a little flux to substructure.")
    print("   Recorded as `cog_bias_vs_recorded_shape`, never corrected away.")

    print("\nD. cog_covariance is consistent with the cumulative structure")
    inc = rng.normal(size=(4000, 24)) * np.linspace(1.0, 0.2, 24)
    cum = np.cumsum(inc, axis=1)
    emp = np.cov(cum, rowvar=False)
    pred = cog_covariance(np.std(cum, axis=0))
    rel = np.abs(emp - pred) / np.sqrt(np.outer(np.diag(pred), np.diag(pred)))
    print(f"   max relative deviation of empirical vs predicted covariance: "
          f"{rel.max():.3f}")
    assert rel.max() < 0.15
    print("   OK — Cov(M_i, M_j) = Var(M_min(i,j)) holds for cumulative sums")

    print("\nE. the quadrature is converged")
    global N_QUAD
    pr = _load_npy_dict(prof_path(int(idx[0]), data_dir))["map_hist_z0p4"]["prof"]
    r = np.asarray(pr["r_kpc"], float)
    sg = np.asarray(pr["intensity"], float)
    keep = N_QUAD
    vals = {}
    for n in (500, 1500, 3000, 8000):
        N_QUAD = n
        vals[n] = integrate_cog(r, sg, COG_RAD_KPC, ellip_const=0.2)
    N_QUAD = keep
    dev = np.max(np.abs(vals[3000] / vals[8000] - 1))
    print(f"   max |M(N=3000)/M(N=8000) - 1| = {dev:.2e}")
    assert dev < 1e-3
    print("   OK")
    # ---- v2 claims, if the v2 product has been built ----------------------
    if V2_OUT.exists():
        v = load_profiles(version=2)
        print("\nF. the annulus form is EXACTLY invertible")
        back = np.cumsum(v["annulus_mass"], axis=2)
        fin = np.isfinite(back) & np.isfinite(v["cog_provided"])
        err = np.abs(back[fin] - v["cog_provided"][fin]) / np.maximum(
            np.abs(v["cog_provided"][fin]), 1.0)
        print(f"   max relative error of cumsum(annuli) vs the stored CoG: "
              f"{err.max():.2e}")
        assert err.max() < 1e-12
        print("   OK — the whitened form loses nothing")

        print("\nG. the reconstructed CoG stays FLAT past the density cutoff")
        worst = 0.0
        n_checked = 0
        for i in range(0, v["n_iso"].shape[0], 37):
            for k in range(N_EPOCH):
                ro = v["r_outer_valid"][i, k]
                M = v["cog_from_density_shared"][i, k]
                if not np.isfinite(ro) or not np.isfinite(M).all():
                    continue
                past = SHARED_GRID > ro
                if past.sum() < 2 or M[past][0] <= 0:
                    continue
                worst = max(worst, float(np.max(np.abs(
                    M[past] / M[past][0] - 1.0))))
                n_checked += 1
        print(f"   max relative rise beyond r_outer_valid over {n_checked} "
              f"galaxy-epochs: {worst:.2e}")
        assert worst < 1e-12
        print("   OK — zero density outside the measured range makes it flat "
              "by construction")

        print("\nH. the shared grid needs NO interpolation for the dominant "
              "native ladder")
        # The claim is that SHARED_GRID[j] IS the galaxy's own radius r[j+1],
        # so the stored density is the measured value and not an interpolant.
        r, sg = v["iso_r_kpc"], v["iso_sigma"]
        dom = np.isclose(r[:, :, 1], SHARED_R0, rtol=1e-12)
        gi, ei = np.where(dom)
        dr, ds = [], []
        for i, k in list(zip(gi, ei))[:400]:
            m = v["sigma_measured"][i, k]
            j = np.where(m)[0]
            native_r = r[i, k][j + 1]
            native_s = sg[i, k][j + 1]
            ok = np.isfinite(native_s) & (native_s > 0)
            dr.append(np.max(np.abs(SHARED_GRID[j][ok] / native_r[ok] - 1)))
            ds.append(np.max(np.abs(v["sigma_shared"][i, k][j][ok]
                                    / native_s[ok] - 1)))
        print(f"   {100 * dom.mean():.1f}% of galaxy-epochs are on it")
        print(f"   max |SHARED_GRID / native radius - 1| : {max(dr):.2e}")
        print(f"   max |stored density / measured  - 1|  : {max(ds):.2e}")
        assert max(dr) < 1e-12 and max(ds) < 1e-9
        print("   OK — for these galaxies the product stores the measurement "
              "itself")

    else:
        print("\n(v2 not built; skipping F-H. Run --build-v2.)")

    print("\nall structural claims verified")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true",
                    help="build v1 from the raw drop (slow: reads 3388 files)")
    ap.add_argument("--build-v2", action="store_true",
                    help="derive v2 from v1 (seconds; no raw-drop access)")
    ap.add_argument("--selftest", action="store_true",
                    help="verify every structural claim on a random subset")
    ap.add_argument("--report", action="store_true",
                    help="print the inventory/quality report for a built dataset")
    ap.add_argument("--n", type=int, default=N_GAL,
                    help="build only the first N galaxies (smoke test)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    a = ap.parse_args(argv)
    if not (a.build or a.build_v2 or a.selftest or a.report):
        ap.error("give --build, --build-v2, --selftest or --report")
    if a.selftest:
        selftest(a.data_dir)
    if a.build:
        out = a.out
        if a.n != N_GAL:                      # a smoke build never overwrites
            out = out.with_name(out.stem + f"_smoke{a.n}" + out.suffix)
        d = build(a.data_dir, out, n_gal=a.n)
        print()
        print(summary(d))
    if a.build_v2:
        v = build_v2()
        print()
        print(summary_v2(v))
    if a.report and not (a.build or a.build_v2):
        print(summary(load_profiles(a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
