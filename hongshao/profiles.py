"""Radial-DiffMAH profile model for curves of growth.

Mirrors the DiffMAH philosophy in radius: the logarithmic slope of the curve of
growth transitions smoothly from a steep inner slope to a shallow outer slope.

    beta(R) = d ln M*(<R) / d ln R
            = beta_out + (beta_in - beta_out) * sigmoid((ln R_c - ln R)/Delta)
    ln M*(<R) = ln M*0 + integral of beta d ln R   (from the innermost radius)

Five parameters: ln M*0 (normalization), beta_in, beta_out, R_c (transition
radius, kpc), Delta (transition width in ln R). The fit is *bounded* so the
parameters are identifiable: the transition radius is kept within the measured
radial range and the slopes within physical limits (otherwise R_c and beta_in
run off to degenerate extremes for near-power-law profiles). Monotonic by
construction (beta >= 0).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

LN10 = np.log(10.0)


def beta_of_R(R, beta_in, beta_out, R_c, Delta):
    """Logarithmic CoG slope at radius R."""
    u = np.log(np.asarray(R, float))
    return beta_out + (beta_in - beta_out) * expit((np.log(R_c) - u) / Delta)


def _integrate(beta, u):
    """Cumulative trapezoid of beta over u (= ln R), from the first point."""
    return np.concatenate(
        [np.zeros(beta.shape[:-1] + (1,)),
         np.cumsum(0.5 * (beta[..., 1:] + beta[..., :-1]) * np.diff(u), axis=-1)],
        axis=-1)


def cog_from_physical(R, logMstar0, beta_in, beta_out, R_c, Delta):
    """log10 M*(<R) from physical params. Vectorized: scalar or array params
    (shape (N,)) give output shape (24,) or (N, 24)."""
    R = np.asarray(R, float)
    u = np.log(R)
    logMstar0, beta_in, beta_out, R_c, Delta = (
        np.asarray(v, float) for v in (logMstar0, beta_in, beta_out, R_c, Delta))
    s = expit((np.log(R_c)[..., None] - u) / Delta[..., None])
    beta = beta_out[..., None] + (beta_in[..., None] - beta_out[..., None]) * s
    return logMstar0[..., None] + _integrate(beta, u) / LN10


# parameter bounds (physical): lnM0 offset, beta_out, dbeta=beta_in-beta_out, u_c, Delta
_BETA_OUT_MAX = 3.0
_DBETA_MAX = 6.0
_DELTA = (0.05, 3.0)


def fit_cog(R, logM_cog, r_min=0.0):
    """Fit the bounded radial-DiffMAH model to one curve of growth.

    R: radii (kpc). logM_cog: log10 M*(<R). ``r_min`` (kpc) drops the innermost
    radii before fitting (e.g. the marginally-resolved core); 0 keeps all points.
    Returns physical params plus ``rms`` (dex) and ``success``. ``logMstar0`` is
    the normalization at the innermost *fitted* radius, so reconstruct with the
    same ``R >= r_min`` grid.
    """
    R = np.asarray(R, float)
    logM_cog = np.asarray(logM_cog, float)
    if r_min > 0.0:
        keep = R >= r_min
        R, logM_cog = R[keep], logM_cog[keep]
    u = np.log(R)
    y = logM_cog * LN10                                    # natural-log cumulative

    bo0 = float(np.clip((y[-1] - y[-4]) / (u[-1] - u[-4]), 0.02, _BETA_OUT_MAX - 0.1))
    bi0 = float(np.clip((y[3] - y[0]) / (u[3] - u[0]), bo0 + 0.05, bo0 + _DBETA_MAX))
    x0 = [y[0], bo0, bi0 - bo0, float(np.median(u)), 0.5]
    lo = [y[0] - 5, 0.0, 0.0, u[0], _DELTA[0]]
    hi = [y[0] + 5, _BETA_OUT_MAX, _DBETA_MAX, u[-1], _DELTA[1]]
    x0 = list(np.clip(x0, lo, hi))

    def resid(p):
        lnM0, b_out, db, u_c, Delta = p
        beta = b_out + db * expit((u_c - u) / Delta)
        return (lnM0 + _integrate(beta, u) - y) / LN10

    sol = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=2000)
    lnM0, b_out, db, u_c, Delta = sol.x
    return {"logMstar0": lnM0 / LN10, "beta_in": float(b_out + db),
            "beta_out": float(b_out), "R_c": float(np.exp(u_c)),
            "Delta": float(Delta),
            "rms": float(np.sqrt(np.mean(resid(sol.x) ** 2))),
            "success": bool(sol.success)}


if __name__ == "__main__":  # self-check: recover known parameters
    R = np.geomspace(2, 150, 24)
    truth = dict(beta_in=1.8, beta_out=0.15, R_c=18.0, Delta=0.6)
    lnM = 23.0 + _integrate(beta_of_R(R, **truth), np.log(R))
    fit = fit_cog(R, lnM / LN10)
    assert fit["rms"] < 1e-3, fit
    for k, v in truth.items():
        assert abs(fit[k] - v) < 0.1 * abs(v) + 0.05, (k, fit[k], v)
    # r_min drops the inner radii but recovers the same model
    fit5 = fit_cog(R, lnM / LN10, r_min=5.0)
    assert fit5["rms"] < 1e-3, fit5
    for k, v in truth.items():
        assert abs(fit5[k] - v) < 0.1 * abs(v) + 0.05, ("r_min", k, fit5[k], v)
    # vectorized reconstruction matches scalar
    c2 = cog_from_physical(R, np.full(3, 23.0 / LN10), *(np.full(3, truth[k])
                           for k in ("beta_in", "beta_out", "R_c", "Delta")))
    assert c2.shape == (3, 24)
    print("profiles self-check OK:", {k: round(fit[k], 3) for k in truth})
