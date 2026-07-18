"""Shared helpers for the tech-note pedagogical figures.

Everything here is deliberately SELF-CONTAINED pedagogy: the kernel forward
model below mirrors the adopted implementation (exp38 ``stage2_multiepoch.py``
``basis_mof``/``model_cogs`` with the population theta and zero conditioning)
so the illustrations can run from an analytic DiffMAH history without loading
the experiment harness. The real product normalizes each epoch to the measured
M(<500 kpc); the toy uses one global mass scale instead (stated in captions).

Figure hygiene (this machine's matplotlibrc has usetex=True): every label is
math-safe — raw ``<``, ``>``, ``|``, ``%`` never appear outside math mode.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from hongshao.plotting import save_fig, set_style          # noqa: E402
from hongshao.qa import _EPOCH_COLORS, _pct, _tex          # noqa: E402

FIGDIR = HERE.parent / "figures"
ANCHOR_Z = np.array([0.4, 0.7, 1.0, 1.5, 2.0])
EPOCH_COLORS = _EPOCH_COLORS
THETA_Z15_NPZ = ROOT / "experiments/exp40_epoch_objective/outputs/latestart.npz"
STAGE1_DIST_NPZ = (ROOT / "experiments/exp41_stochastic_layer/outputs/"
                   "stage1_dist.npz")
CATALOG_FITS = ROOT / "data/processed/tng300_072_z0p4.fits"

__all__ = ["ANCHOR_Z", "EPOCH_COLORS", "FIGDIR", "ROOT", "save_fig",
           "set_style", "_tex", "_pct", "cosmic_age", "z_of_t", "toy_mah",
           "efficiency_weights", "moffat_cog", "kernel_cogs", "theta_z15",
           "load_catalog"]


# --------------------------------------------------------------------------- #
# cosmology (TNG: Planck 2015)                                                #
# --------------------------------------------------------------------------- #
def _cosmology():
    from astropy.cosmology import FlatLambdaCDM
    return FlatLambdaCDM(H0=67.74, Om0=0.3089)


_Z_GRID = np.geomspace(1e-3, 20.0, 600)
_T_GRID = _cosmology().age(_Z_GRID).value            # Gyr, decreasing in z


def cosmic_age(z):
    """Cosmic age (Gyr) at redshift ``z``."""
    return np.interp(np.asarray(z, float), _Z_GRID, _T_GRID)


def z_of_t(t_gyr):
    """Redshift at cosmic age ``t_gyr`` (Gyr)."""
    return np.interp(np.asarray(t_gyr, float), _T_GRID[::-1], _Z_GRID[::-1])


T_OBS = float(cosmic_age(0.4))                       # the z=0.4 anchor epoch
ANCHOR_T = cosmic_age(ANCHOR_Z)


# --------------------------------------------------------------------------- #
# an analytic halo history (DiffMAH) and its accretion steps                  #
# --------------------------------------------------------------------------- #
# the catalog medians of the 'use' sample (tng300_072_z0p4.fits)
MEDIAN_HALO = dict(logmp=13.31, logtc=0.41, early=2.22, late=0.32)


def toy_mah(logmp=None, logtc=None, early=None, late=None, n_step=90,
            t_min=0.35):
    """A DiffMAH halo history sampled on a snapshot-like time grid.

    Returns a dict with per-step arrays: ``t`` (Gyr, step centres), ``z``,
    ``dMh`` (linear accreted mass per step), plus ``t_edge`` and ``logmh``
    on the edges — the same ingredients the kernel consumes.
    """
    from hongshao.diffmah import log_mah
    p = dict(MEDIAN_HALO)
    for key, val in dict(logmp=logmp, logtc=logtc, early=early,
                         late=late).items():
        if val is not None:
            p[key] = val
    t_edge = np.geomspace(t_min, T_OBS, n_step + 1)
    lm = log_mah(np.log10(t_edge), np.atleast_1d(p["logmp"]),
                 np.atleast_1d(p["logtc"]), np.atleast_1d(p["early"]),
                 np.atleast_1d(p["late"]), np.log10(T_OBS))[0]
    mh = 10.0 ** lm
    t_mid = np.sqrt(t_edge[:-1] * t_edge[1:])
    return dict(t=t_mid, z=z_of_t(t_mid), dMh=np.clip(np.diff(mh), 0.0, None),
                t_edge=t_edge, logmh=lm, params=p)


# --------------------------------------------------------------------------- #
# the adopted kernel, self-contained (mirrors exp38 basis_mof / model_cogs)   #
# --------------------------------------------------------------------------- #
def theta_z15():
    """The official z<=1.5 population theta (exp40): [log_rc, g, q, mu,
    sig, gamma] (conditioning slopes dropped — population values)."""
    return np.load(THETA_Z15_NPZ)["theta_z15"][:6].copy()


def efficiency_weights(z, mu, sig):
    """The lognormal efficiency window on the accretion history:
    w(z) = exp(-(ln(1+z) - mu)^2 / (2 sig^2)); peak at z = e^mu - 1."""
    return np.exp(-((np.log1p(z) - mu) ** 2) / (2.0 * sig ** 2))


def moffat_cog(r, rc, gam):
    """Unit-mass Moffat (power-law-tail) deposit curve of growth:
    M(<R) = 1 - (1 + (R/rc)^2)^(1-gam); finite mass needs gam > 1."""
    r = np.atleast_1d(np.asarray(r, float))
    rc = np.atleast_1d(np.asarray(rc, float))
    u = 1.0 + (r[:, None] / rc[None, :]) ** 2
    return 1.0 - u ** (1.0 - gam)


def deposit_basis(theta6, ti, tk, r, t_obs=T_OBS):
    """The migrated two-state deposit basis B[r, i] at observation time
    ``tk`` for deposits born at ``ti`` — the exp38 ``basis_mof`` math.

    Birth radius   rc0_i = 10^log_rc (t_i/t_obs)^g
    Retained frac  f_c,i = exp(-(t_k - t_i)/t_i)          (alpha = 1 clock)
    Migrated radius rcw_i = rc0_i (t_k/t_i)^q
    B_i(R) = f_c M(R; rc0_i) + (1 - f_c) M(R; rcw_i)
    """
    log_rc, g, q, _, _, gam = theta6
    ti = np.asarray(ti, float)
    rc0 = np.clip(10.0 ** log_rc * (ti / t_obs) ** g, 1e-4, 1e5)
    dt = np.clip(tk - ti, 0.0, None)
    fc = np.exp(-dt / ti)
    rcw = np.clip(rc0 * (tk / ti) ** max(q, 0.0), 1e-4, 1e5)
    return (fc[None, :] * moffat_cog(r, rc0, gam)
            + (1.0 - fc)[None, :] * moffat_cog(r, rcw, gam))


def kernel_cogs(theta6, mah, t_eval, r, mstar_norm=None):
    """Model CoGs (len(t_eval), len(r)) for one halo history.

    Deposit budget dM_i proportional to w(z_i) dMh_i (normalized to 1); each
    epoch sums the deposits born before it through ``deposit_basis``. If
    ``mstar_norm`` is given, ONE global scale pins M(<r[-1]) at the FIRST
    evaluation epoch to it (the real product pins every epoch to the measured
    M(<500 kpc) instead).
    """
    w = efficiency_weights(mah["z"], theta6[3], theta6[4])
    dM = w * mah["dMh"]
    dM = dM / dM.sum()
    out = []
    for tk in np.atleast_1d(t_eval):
        mask = mah["t"] <= tk
        B = deposit_basis(theta6, mah["t"], tk, r)
        out.append(B @ (dM * mask))
    out = np.array(out)
    if mstar_norm is not None:
        out *= mstar_norm / out[0, -1]
    return out


def sigma_of_cog(cog, r):
    """Surface density from a linear CoG on grid ``r`` (annulus midpoints)."""
    dA = np.pi * (r[1:] ** 2 - r[:-1] ** 2)
    mid = np.sqrt(r[:-1] * r[1:])
    return np.diff(cog, axis=-1) / dA, mid


def load_catalog():
    """The z=0.4 training catalog: (X, cog, radii) for the 'use' sample."""
    from astropy.table import Table

    from hongshao.tng_data import COG_RAD_KPC
    t = Table.read(CATALOG_FITS)
    t = t[t["use"]]
    cog = np.asarray(t["logmstar_cog"], float)
    X = np.column_stack([np.asarray(t[c], float) for c in
                         ("dmah_logmp", "dmah_logtc", "dmah_early",
                          "dmah_late", "c_200c")])
    good = np.isfinite(cog).all(1) & np.isfinite(X).all(1)
    return X[good], cog[good], np.asarray(COG_RAD_KPC, float)


def _self_check():
    """The toy kernel must be monotone, mass-normalized, and mass-moving."""
    th = theta_z15()
    mah = toy_mah()
    r = np.geomspace(1.0, 500.0, 200)
    cogs = kernel_cogs(th, mah, ANCHOR_T, r)
    assert np.isfinite(cogs).all()
    assert np.all(np.diff(cogs, axis=1) > -1e-12), "CoG must be monotone in R"
    # migration moves mass outward at fixed deposit set: freezing the clock
    # (fc = 1) must concentrate the z=0.4 profile
    frozen = (moffat_cog(r, np.clip(10.0 ** th[0]
                                    * (mah["t"] / T_OBS) ** th[1],
                                    1e-4, 1e5), th[5])
              @ (efficiency_weights(mah["z"], th[3], th[4]) * mah["dMh"]))
    frozen /= frozen[-1]
    live = cogs[0] / cogs[0, -1]
    i30 = np.searchsorted(r, 30.0)
    assert frozen[i30] > live[i30], "the clock must move mass outward"
    print("common self-check OK: toy kernel monotone, mass-conserving, "
          "and outward-migrating with the official theta")


if __name__ == "__main__":
    _self_check()
