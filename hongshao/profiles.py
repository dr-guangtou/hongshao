"""Radial-DiffMAH profile model for curves of growth.

Mirrors the DiffMAH philosophy in radius: the logarithmic slope of the curve of
growth transitions smoothly from a steep inner slope to a shallow outer slope.

    beta(R) = d ln M*(<R) / d ln R
            = beta_out + (beta_in - beta_out) * sigmoid((ln R_c - ln R)/Delta)
    ln M*(<R) = ln M*0 + integral of beta d ln R   (from the innermost radius)

Parameters (5): ln M*0 (normalization), beta_in, beta_out, R_c (transition
radius, kpc), Delta (transition width in ln R). Monotonic by construction
(beta >= 0), so the curve of growth never decreases.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

LN10 = np.log(10.0)


def _softplus(x):
    return np.logaddexp(0.0, x)        # ln(1 + e^x), overflow-safe


def _inv_softplus(y):
    y = np.maximum(y, 1e-6)
    return np.log(np.expm1(y))         # inverse of softplus for y > 0


def beta_of_R(R, beta_in, beta_out, R_c, Delta):
    """Logarithmic CoG slope at radius R."""
    u = np.log(np.asarray(R, float))
    return beta_out + (beta_in - beta_out) * expit((np.log(R_c) - u) / Delta)


def lncog_from_params(params, u):
    """Model ln M*(<R) on a log-radius grid u = ln R.

    params (unconstrained): (lnM0, b_out, db, u_c, d_R), mapped to physical
    (beta_out, beta_in, Delta) via softplus so beta_in > beta_out >= 0, Delta>0.
    """
    lnM0, b_out, db, u_c, d_R = params
    beta_out = _softplus(b_out)
    beta_in = beta_out + _softplus(db)
    Delta = _softplus(d_R)
    beta = beta_out + (beta_in - beta_out) * expit((u_c - u) / Delta)
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (beta[1:] + beta[:-1]) * np.diff(u))])
    return lnM0 + integ


def _unpack(params):
    lnM0, b_out, db, u_c, d_R = params
    beta_out = float(_softplus(b_out))
    beta_in = beta_out + float(_softplus(db))
    return {"logMstar0": lnM0 / LN10, "beta_in": beta_in, "beta_out": beta_out,
            "R_c": float(np.exp(u_c)), "Delta": float(_softplus(d_R))}


def fit_cog(R, logM_cog):
    """Fit the radial-DiffMAH model to one curve of growth.

    R: radii (kpc). logM_cog: log10 M*(<R). Returns a dict of physical params
    plus ``rms`` (dex) and ``success``.
    """
    u = np.log(np.asarray(R, float))
    y = np.asarray(logM_cog, float) * LN10           # natural-log cumulative mass

    # initial guess from finite differences
    bo = max((y[-1] - y[-4]) / (u[-1] - u[-4]), 0.02)
    bi = max((y[3] - y[0]) / (u[3] - u[0]), bo + 0.05)
    x0 = [y[0], _inv_softplus(bo), _inv_softplus(bi - bo),
          float(np.median(u)), _inv_softplus(0.5)]

    def resid(p):
        return (lncog_from_params(p, u) - y) / LN10   # residual in dex

    sol = least_squares(resid, x0, method="lm", max_nfev=2000)
    out = _unpack(sol.x)
    out["rms"] = float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    out["success"] = bool(sol.success)
    return out


if __name__ == "__main__":  # self-check: recover known parameters
    R = np.geomspace(2, 150, 24)
    truth = dict(beta_in=1.8, beta_out=0.15, R_c=18.0, Delta=0.6)
    beta = beta_of_R(R, **truth)
    u = np.log(R)
    lnM = 23.0 + np.concatenate(
        [[0.0], np.cumsum(0.5 * (beta[1:] + beta[:-1]) * np.diff(u))])
    fit = fit_cog(R, lnM / LN10)
    assert fit["rms"] < 1e-3, fit
    for k, v in truth.items():
        assert abs(fit[k] - v) < 0.1 * abs(v) + 0.05, (k, fit[k], v)
    print("profiles self-check OK:", {k: round(fit[k], 3) for k in truth})
