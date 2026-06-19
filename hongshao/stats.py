"""Reusable statistics helpers for HongShao experiments.

Graduated from exp01: partial (rank) correlation controlling for one variable,
and cross-validated linear-regression RMSE.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import t as tdist


def partial_spearman(x, y, z):
    """Rank partial correlation of x, y controlling for z. Returns (r, p, n).

    Computed on the rows where x, y, z are all finite. p is the two-sided
    t-test p-value with df = n - 3.
    """
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    n = len(x)
    if n < 20:
        return np.nan, np.nan, n
    rxy = spearmanr(x, y)[0]
    rxz = spearmanr(x, z)[0]
    ryz = spearmanr(y, z)[0]
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    if denom == 0:
        return np.nan, np.nan, n
    r = float(np.clip((rxy - rxz * ryz) / denom, -1, 1))
    df = n - 3
    if df <= 0 or abs(r) >= 1:
        return r, np.nan, n
    tstat = r * np.sqrt(df / (1 - r**2))
    return r, float(2 * tdist.sf(abs(tstat), df)), n


def cv_rmse(feature_cols, y, k=5, seed=0):
    """k-fold cross-validated RMSE of a linear fit. Inputs must be finite,
    equal length; feature_cols is a list of 1-D arrays (intercept added)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float)
                                             for c in feature_cols])
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    pred = np.full(n, np.nan)
    for fold in np.array_split(idx, k):
        tr = np.setdiff1d(np.arange(n), fold)
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[fold] = X[fold] @ beta
    return float(np.sqrt(np.mean((y - pred) ** 2)))


if __name__ == "__main__":  # tiny self-check
    rng = np.random.default_rng(0)
    z = rng.normal(size=2000)
    x = z + 0.5 * rng.normal(size=2000)
    y = z + 0.5 * rng.normal(size=2000)
    r_raw = spearmanr(x, y)[0]
    r_par, p, n = partial_spearman(x, y, z)
    assert r_par < r_raw - 0.2, (r_raw, r_par)   # controlling z removes shared signal
    assert p < 1e-3 or abs(r_par) < 0.1
    # cv_rmse: adding the true predictor lowers error
    e0 = cv_rmse([np.ones(2000)], y)
    e1 = cv_rmse([z], y)
    assert e1 < e0
    print("stats self-check OK", round(r_raw, 3), round(r_par, 3))
