"""Evaluation metrics for HongShao models (graduated from exp07).

The recommended suite for judging the Ultimate-SHMR models, in two families:

1. **Probabilistic / distributional scores** for *predictors* — models that
   output a predictive distribution P(target | halo). A good point prediction
   (low RMS) can still be badly calibrated, so we score the whole distribution:
   - ``crps_gaussian`` — Continuous Ranked Probability Score (proper, in dex).
   - ``gaussian_logscore`` — negative log predictive density (proper).
   - ``pit`` / ``interval_coverage`` — calibration (are the stated
     uncertainties honest?).

2. **Goodness-of-fit / model-selection** for *profile fits* — judging a
   parametric CoG model:
   - ``aic_bic`` — Akaike / Bayesian information criteria from the residual
     sum of squares (Gaussian likelihood, noise variance profiled out), so no
     external per-point error is needed to *rank* models.

All functions are vectorized and element-wise (caller aggregates with
``.mean()``); ``sigma`` is a predictive standard deviation in the same units as
the target. See exp07 for the demonstration and the rationale write-up.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri   # standard-normal CDF and its inverse

_SQRT_PI = np.sqrt(np.pi)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _phi(z):
    return _INV_SQRT_2PI * np.exp(-0.5 * z * z)


def crps_gaussian(y, mu, sigma):
    """CRPS of a Gaussian predictive N(mu, sigma) against observations y.

    Closed form (Gneiting & Raftery 2007). Lower is better; returns the same
    units as y. Element-wise.
    """
    sigma = np.asarray(sigma, float)
    z = (np.asarray(y, float) - np.asarray(mu, float)) / sigma
    return sigma * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * _phi(z) - 1.0 / _SQRT_PI)


def gaussian_logscore(y, mu, sigma):
    """Negative log predictive density of N(mu, sigma) at y (lower is better)."""
    sigma = np.asarray(sigma, float)
    z = (np.asarray(y, float) - np.asarray(mu, float)) / sigma
    return 0.5 * z * z + np.log(sigma) + 0.5 * np.log(2.0 * np.pi)


def pit(y, mu, sigma):
    """Probability integral transform F(y); ~Uniform(0,1) iff well-calibrated."""
    return ndtr((np.asarray(y, float) - np.asarray(mu, float)) / np.asarray(sigma, float))


def interval_coverage(y, mu, sigma, levels=(0.5, 0.68, 0.9, 0.95)):
    """Empirical coverage of central predictive intervals at nominal ``levels``.

    Returns ``(levels, coverage)``. Calibrated => coverage ~= levels.
    """
    y, mu, sigma = (np.asarray(v, float) for v in (y, mu, sigma))
    levels = np.asarray(levels, float)
    half = ndtri(0.5 + levels / 2.0)                      # z for each central level
    cov = np.array([np.mean(np.abs(y - mu) <= h * sigma) for h in half])
    return levels, cov


def aic_bic(rss, n, k):
    """AIC and BIC from a residual sum of squares (Gaussian, variance profiled).

    rss: sum of squared residuals. n: number of data points. k: number of free
    model parameters (the profiled noise variance counts as one, so pass
    ``k = n_model_params + 1``). Lower is better. Returns ``(aic, bic)``.
    """
    rss = np.asarray(rss, float)
    ll = -0.5 * n * (np.log(2.0 * np.pi * rss / n) + 1.0)   # maximized log-lik
    return 2.0 * k - 2.0 * ll, k * np.log(n) - 2.0 * ll


if __name__ == "__main__":  # self-checks: proper scores reward the truth
    rng = np.random.default_rng(0)
    n = 200_000
    y = rng.normal(0.0, 1.0, n)

    # CRPS & log-score are minimized at the true predictive distribution
    crps_true = crps_gaussian(y, 0.0, 1.0).mean()
    crps_wide = crps_gaussian(y, 0.0, 2.0).mean()
    crps_bias = crps_gaussian(y, 0.5, 1.0).mean()
    assert crps_true < crps_wide and crps_true < crps_bias, (crps_true, crps_wide, crps_bias)
    # mean CRPS of a calibrated N(0,1) against its own draws is 1/sqrt(pi)
    assert abs(crps_true - 1 / _SQRT_PI) < 5e-3, crps_true

    ls_true = gaussian_logscore(y, 0.0, 1.0).mean()
    ls_wrong = gaussian_logscore(y, 0.0, 2.0).mean()
    assert ls_true < ls_wrong
    # log-score of N(0,1) on its own samples ~ 0.5*ln(2*pi*e) = 1.4189
    assert abs(ls_true - 0.5 * np.log(2 * np.pi * np.e)) < 5e-3, ls_true

    # PIT of correct model is uniform; coverage matches nominal
    p = pit(y, 0.0, 1.0)
    assert abs(p.mean() - 0.5) < 5e-3 and abs(p.std() - 1 / np.sqrt(12)) < 5e-3
    lev, cov = interval_coverage(y, 0.0, 1.0)
    assert np.allclose(cov, lev, atol=5e-3), dict(zip(lev, cov))
    # an over-confident model under-covers
    _, cov_bad = interval_coverage(y, 0.0, 0.5)
    assert np.all(cov_bad < cov), (cov, cov_bad)

    # AIC/BIC: lower RSS lowers both; more params penalized
    a1, b1 = aic_bic(100.0, 100, 3)
    a2, b2 = aic_bic(50.0, 100, 3)
    assert a2 < a1 and b2 < b1
    a3, b3 = aic_bic(50.0, 100, 6)
    assert a3 > a2 and b3 > b2 and (b3 - b2) > (a3 - a2)   # BIC penalizes k harder
    print("metrics self-check OK  "
          f"crps*={crps_true:.4f} logscore*={ls_true:.4f} cov={np.round(cov,3).tolist()}")
