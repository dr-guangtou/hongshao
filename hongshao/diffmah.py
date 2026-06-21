"""DiffMAH rolling-power-law fit to a halo mass accretion history.

Our portable parameterization of the MAH (Hearin et al. 2021, arXiv:2105.05859):
the halo mass is a power law in cosmic time whose logarithmic index rolls
smoothly from an early value to a late value. In log10(time):

    alpha(logt) = early + (late - early) * sigmoid(k * (logt - logtc))
    log10 M(t)  = logmp + alpha(logt) * (logt - logt0)

so the curve passes through ``logmp`` at the anchor time ``logt0`` (we anchor at
the z=0.4 observation epoch, so ``logmp`` ~ the final peak mass M0). The
transition speed ``k`` is FIXED (DiffMAH uses ``MAH_K = 3.5``), leaving four
per-halo numbers: ``logmp`` (normalization), ``logtc`` (transition time),
``early`` and ``late`` (early/late accretion indices). These four are intrinsic
to each halo — unlike the TNG-sample-defined MAH-PCA basis — so the emulator
trained on them is portable to any simulation/observation with a DiffMAH fit.

Mirrors ``profiles.fit_cog``: a bounded, identifiable least-squares fit.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

MAH_K = 3.5                       # fixed transition speed (Hearin et al. 2021)


def log_mah(logt, logmp, logtc, early, late, logt0, k=MAH_K):
    """log10 M(t) at log10(cosmic time) ``logt``; anchored to ``logmp`` at logt0.

    Vectorized over the params (scalars or arrays) for fast reconstruction.
    """
    logt = np.asarray(logt, float)
    logmp, logtc, early, late = (np.asarray(v, float) for v in (logmp, logtc, early, late))
    alpha = early[..., None] + (late - early)[..., None] * expit(k * (logt - logtc[..., None]))
    return logmp[..., None] + alpha * (logt - logt0)


def fit_mah(t_gyr, log_mpeak, t0_gyr, t_min=0.0, k=MAH_K):
    """Fit the bounded DiffMAH model to one peak-mass history.

    t_gyr: cosmic time (Gyr) of each point. log_mpeak: log10 Mpeak at each time.
    t0_gyr: anchor time (the z=0.4 epoch). ``t_min`` (Gyr) drops the earliest,
    poorly-resolved history before fitting (the analog of the 5-kpc inner cut on
    the curve of growth). Returns the four params plus ``rms`` (dex) and
    ``success``.
    """
    t_gyr = np.asarray(t_gyr, float)
    y = np.asarray(log_mpeak, float)
    if t_min > 0.0:
        keep = t_gyr >= t_min
        t_gyr, y = t_gyr[keep], y[keep]
    logt = np.log10(t_gyr)
    logt0 = float(np.log10(t0_gyr))
    n = len(logt)
    if n < 5:                                   # too few resolved points to fit
        return {"logmp": np.nan, "logtc": np.nan, "early": np.nan,
                "late": np.nan, "rms": np.nan, "success": False}

    i_lo, i_hi = min(3, n - 1), max(0, n - 4)
    early0 = float(np.clip((y[i_lo] - y[0]) / (logt[i_lo] - logt[0]), 0.3, 12.0))
    late0 = float(np.clip((y[-1] - y[i_hi]) / (logt[-1] - logt[i_hi]), 0.05, 12.0))
    lo = [y[-1] - 2.0, logt[0], 0.1, 0.05]
    hi = [y[-1] + 2.0, logt[-1], 12.0, 12.0]
    x0 = list(np.clip([y[-1], float(np.median(logt)), early0, late0], lo, hi))

    def resid(p):
        return log_mah(logt, p[0], p[1], p[2], p[3], logt0, k) - y

    sol = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=2000)
    logmp, logtc, early, late = sol.x
    return {"logmp": float(logmp), "logtc": float(logtc), "early": float(early),
            "late": float(late), "rms": float(np.sqrt(np.mean(resid(sol.x) ** 2))),
            "success": bool(sol.success)}


if __name__ == "__main__":  # self-check: recover known params from a synthetic MAH
    t = np.geomspace(0.5, 9.5, 30)
    t0 = 9.66                                   # ~age at z=0.4 (Gyr)
    truth = dict(logmp=14.0, logtc=np.log10(2.5), early=3.0, late=0.7)
    y = log_mah(np.log10(t), logt0=np.log10(t0), **truth)
    assert np.all(np.diff(y) > 0), "synthetic MAH must be monotonic"
    fit = fit_mah(t, y, t0)
    assert fit["rms"] < 1e-3, fit
    for kk, vv in truth.items():
        assert abs(fit[kk] - vv) < 0.05 * abs(vv) + 0.05, (kk, fit[kk], vv)
    # vectorized reconstruction matches
    rec = log_mah(np.log10(t), np.full(4, truth["logmp"]),
                  *(np.full(4, truth[p]) for p in ("logtc", "early", "late")),
                  logt0=np.log10(t0))
    assert rec.shape == (4, 30)
    print("diffmah self-check OK:", {k: round(fit[k], 3) for k in truth})
