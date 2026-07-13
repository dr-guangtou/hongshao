"""The Ultimate-SHMR emulator: a portable, generative model of a massive
central galaxy's stellar-mass profile given its halo.

Graduated from exp14/exp15/exp17/exp19 (the frozen spec). For each of four
radial bins of the curve-of-growth stellar mass — the cumulative core
``M*[<10]`` and the annuli ``M*[10-30]``, ``[30-50]``, ``[50-100]`` kpc — the
emulator predicts a full Gaussian ``P(profile | halo)``:

  * **mean** ``mu_j(X)`` — linear (default) or a 7-term degree-2 polynomial
    (exp17) on the *portable* halo features ``X = [logmp, logtc, early, late,
    c200c]`` (DiffMAH's four MAH parameters + the NFW concentration ``c_200c``;
    all N-body-available, so the model transfers off TNG).
  * **scatter** ``Sigma(X) = D(X) R D(X)`` — heteroscedastic per-bin standard
    deviations ``sigma_j(X) = exp(0.5 * gamma_j . [1, X_std])`` (Gaussian MLE
    with a ridge on the slopes, exp14) times a *fixed* residual correlation
    ``R`` (exp19: ``late`` drives the scatter, ``c200c`` the mean).

The mean alone is under-dispersed (it is a conditional expectation), so the
model must be used **generatively**: ``sample()`` draws correlated,
heteroscedastic profiles that reproduce the population — including its tails
(exp15). The point prediction's apparent low-mass "bias" is regression to the
mean, not a defect.

Self-check (``python -m hongshao.emulator``) reproduces exp19: 5-fold-CV
CRPS ~0.083, joint NLL ~-3.43, conditional-coverage gap ~0.010.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

FEATURES = ["logmp", "logtc", "early", "late", "c200c"]   # portable halo features
TARGETS = ["<10", "10-30", "30-50", "50-100"]             # CoG bins [kpc]
# the 7 degree-2 terms the BIC-preferred poly-2 mean keeps (exp17), as index
# pairs into FEATURES: logmp^2, logtc*late, early^2, late*c200c, late^2,
# logmp*early, logmp*late.
_POLY2_TERMS = [(0, 0), (1, 3), (2, 2), (3, 4), (3, 3), (0, 2), (0, 3)]


@dataclass
class Emulator:
    """A fitted emulator over ``T`` targets. ``predict``/``sample`` are methods.

    Target-count agnostic: ``T`` is whatever ``fit`` was given (4 kpc bins, 6 Re
    bins, or a compressed profile vector ``[anchor, PCs]`` — see profile_emulator).
    """
    mean: str                 # "linear" or "poly2"
    mu_x: np.ndarray          # (F,) feature means (standardization)
    sd_x: np.ndarray          # (F,) feature stds
    beta: np.ndarray          # (T, P) mean coefficients per target
    gamma: np.ndarray         # (T, F+1) log-variance coefficients per target
    corr: np.ndarray          # (T, T) fixed residual correlation R

    def predict(self, X):
        """Predictive mean, marginal sigma, and full covariance for halos ``X``.

        ``X``: (N, F) ``[logmp, logtc, early, late, c200c]``. Returns
        ``(mu, sigma, cov)`` with shapes (N, T), (N, T), (N, T, T), where
        ``cov[i] = diag(sigma[i]) @ R @ diag(sigma[i])``.
        """
        Xs = (np.atleast_2d(np.asarray(X, float)) - self.mu_x) / self.sd_x
        mu = _mean_design(Xs, self.mean) @ self.beta.T
        var_design = np.column_stack([np.ones(len(Xs)), Xs])
        sigma = np.exp(0.5 * (var_design @ self.gamma.T))
        cov = sigma[:, :, None] * self.corr[None] * sigma[:, None, :]
        return mu, sigma, cov

    def sample(self, X, size=1, rng=None):
        """Draw ``size`` correlated, heteroscedastic profiles per halo.

        Returns (size, N, T). This is the generative path (exp15): the spread
        of these draws reproduces the population; the mean alone does not.
        """
        rng = np.random.default_rng(rng)
        mu, sigma, _ = self.predict(X)
        chol = _chol_psd(self.corr)                          # R = L L^T, shared
        z = rng.standard_normal((size, len(mu), mu.shape[1]))
        return mu[None] + sigma[None] * (z @ chol.T)


def fit(X, Y, mean="linear", ridge=2.0):
    """Fit the emulator: per-bin mean + heteroscedastic full covariance.

    ``X``: (N, 5) portable halo features (``FEATURES`` order). ``Y``: (N, 4)
    targets (``TARGETS`` order). ``mean``: "linear" or "poly2" (exp17 7-term
    degree-2). ``ridge``: L2 penalty on the log-variance slopes (exp14).
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if mean not in ("linear", "poly2"):
        raise ValueError(f"mean must be 'linear' or 'poly2', got {mean!r}")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError(
            f"fit requires finite X and Y (non-finite rows: X {int((~np.isfinite(X).all(1)).sum())}, "
            f"Y {int((~np.isfinite(Y).all(1)).sum())}): one NaN silently poisons that "
            "column's coefficients and the shared residual correlation — pre-filter rows")
    mu_x, sd_x = X.mean(0), X.std(0)
    Xs = (X - mu_x) / sd_x
    mean_design = _mean_design(Xs, mean)
    var_design = np.column_stack([np.ones(len(Xs)), Xs])

    nb = Y.shape[1]
    beta = np.empty((nb, mean_design.shape[1]))
    gamma = np.empty((nb, var_design.shape[1]))
    resid = np.empty_like(Y)
    sig = np.empty_like(Y)
    for j in range(nb):
        beta[j], *_ = np.linalg.lstsq(mean_design, Y[:, j], rcond=None)
        resid[:, j] = Y[:, j] - mean_design @ beta[j]
        gamma[j] = _fit_logvar(resid[:, j], Xs, ridge)
        sig[:, j] = np.exp(0.5 * (var_design @ gamma[j]))
    corr = np.corrcoef((resid / sig).T)                      # scatter removed
    return Emulator(mean, mu_x, sd_x, beta, gamma, corr)


def _chol_psd(corr):
    """Cholesky of a correlation matrix, robust to ``corrcoef`` roundoff.

    Near-duplicate targets can push an eigenvalue a hair below zero, where the
    bare cholesky raises. On failure, shrink toward the identity by the
    smallest amount that restores positive-definiteness (diagonal stays 1, the
    correlation structure is preserved to ~the eigenvalue defect).
    """
    try:
        return np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        w_min = float(np.linalg.eigvalsh(corr).min())
        lam = (1e-10 - w_min) / (1.0 - w_min)
        shrunk = (1.0 - lam) * corr + lam * np.eye(len(corr))
        return np.linalg.cholesky(shrunk)


def _mean_design(Xs, mean):
    """[1, features, (poly-2 terms)] from standardized features ``Xs`` (N, 5)."""
    cols = [np.ones(len(Xs))] + [Xs[:, i] for i in range(Xs.shape[1])]
    if mean == "poly2":
        cols += [Xs[:, i] * Xs[:, k] for i, k in _POLY2_TERMS]
    return np.column_stack(cols)


def _fit_logvar(r, Z, ridge):
    """MLE Gaussian log-variance regression: sigma^2(z) = exp(g . [1, z]).

    Minimizes 0.5 * sum[s + r^2 exp(-s)], s = A@g, + ridge on slopes (exp14).
    Returns ``g`` (intercept + one slope per column of ``Z``).
    """
    A = np.column_stack([np.ones(len(r)), Z])
    r2 = r ** 2

    def nll(g):
        s = np.clip(A @ g, -700.0, 700.0)     # exp overflow guard (near-zero r2)
        return 0.5 * np.sum(s + r2 * np.exp(-s)) + 0.5 * ridge * np.sum(g[1:] ** 2)

    def grad(g):
        s = np.clip(A @ g, -700.0, 700.0)
        out = A.T @ (0.5 * (1.0 - r2 * np.exp(-s)))
        out[1:] += ridge * g[1:]
        return out

    g0 = np.r_[np.log(max(r2.mean(), 1e-6)), np.zeros(Z.shape[1])]
    return minimize(nll, g0, jac=grad, method="L-BFGS-B").x


# --------------------------------------------------------------------------- #
# self-check                                                                   #
# --------------------------------------------------------------------------- #
def _cv_oof(X, Y, mean, k=5, seed=0):
    """Out-of-fold (mu, sigma, cov) from k-fold CV — the honest scores."""
    n, nb = Y.shape
    order = np.random.default_rng(seed).permutation(n)
    mu = np.empty((n, nb))
    sigma = np.empty((n, nb))
    cov = np.empty((n, nb, nb))
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(n), fold)
        emu = fit(X[tr], Y[tr], mean=mean)
        mu[fold], sigma[fold], cov[fold] = emu.predict(X[fold])
    return mu, sigma, cov


def _joint_nll(Y, mu, cov):
    resid = Y - mu
    out = np.empty(len(Y))
    for i in range(len(Y)):
        _, logdet = np.linalg.slogdet(cov[i])
        out[i] = 0.5 * (Y.shape[1] * np.log(2 * np.pi) + logdet
                        + resid[i] @ np.linalg.solve(cov[i], resid[i]))
    return float(out.mean())


def _cond_cov_gap(Y, mu, sigma):
    """Mean |68%-coverage(high sigma) - coverage(low sigma)| over the bins."""
    gaps = []
    for j in range(Y.shape[1]):
        s = sigma[:, j]
        edges = np.quantile(s, [1 / 3, 2 / 3])
        bins = np.digitize(s, edges)
        cov = [np.mean(np.abs(Y[bins == b, j] - mu[bins == b, j])
                       <= sigma[bins == b, j]) for b in (0, 2)]
        gaps.append(abs(cov[1] - cov[0]))
    return float(np.mean(gaps))


def _annulus(a_outer, a_inner):
    return np.log10(np.clip(10.0 ** a_outer - 10.0 ** a_inner, 1.0, None))


def _load_real():
    """exp19's data: X = [DiffMAH(4), c200c], Y = 4 CoG bins. None if missing."""
    from pathlib import Path
    table = Path(__file__).resolve().parents[1] / "data" / "processed" / "tng300_072_z0p4.fits"
    if not table.exists():
        return None
    from astropy.table import Table
    t = Table.read(table)
    t = t[t["use"]]
    aper = np.asarray(t["logmstar_aper"], float)
    X = np.column_stack([np.asarray(t[c], float) for c in
                         ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late", "c_200c")])
    Y = np.column_stack([aper[:, 0], _annulus(aper[:, 1], aper[:, 0]),
                         _annulus(aper[:, 2], aper[:, 1]), _annulus(aper[:, 4], aper[:, 2])])
    g = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    return X[g], Y[g]


if __name__ == "__main__":
    from hongshao.metrics import crps_gaussian

    # (1) synthetic: fit recovers a known mean, and sample() is calibrated.
    rng = np.random.default_rng(0)
    n = 4000
    Xs = rng.normal(size=(n, 5))
    beta_true = rng.normal(size=(4, 6))                      # [1, 5 features]
    design = np.column_stack([np.ones(n), Xs])
    sig_true = np.array([0.10, 0.15, 0.20, 0.25])
    Yt = design @ beta_true.T + sig_true * rng.standard_normal((n, 4))
    emu = fit(Xs, Yt)
    mu, sigma, cov = emu.predict(Xs)
    rms = np.sqrt(((mu - design @ beta_true.T) ** 2).mean(0))
    assert np.all(rms < 0.02), f"mean not recovered, rms={rms}"
    assert np.allclose(sigma.mean(0), sig_true, rtol=0.15), sigma.mean(0)
    draws = emu.sample(Xs, size=200, rng=1)
    assert draws.shape == (200, n, 4)
    # sample spread matches the truth; the mean alone is under-dispersed (exp15)
    assert np.allclose(draws.std(axis=(0, 1)), Yt.std(0), rtol=0.1), "sample mis-dispersed"
    assert np.all(mu.std(0) < Yt.std(0)), "mean should be under-dispersed vs truth"

    # fit must fail fast on non-finite input: a single NaN otherwise silently
    # NaNs that column's coefficients AND the shared corrcoef row/column.
    for bad_X, bad_Y in ((True, False), (False, True)):
        Xb, Yb = Xs.copy(), Yt.copy()
        (Xb if bad_X else Yb)[7, 1] = np.nan
        try:
            fit(Xb, Yb)
        except ValueError:
            pass
        else:
            raise AssertionError(f"fit accepted non-finite {'X' if bad_X else 'Y'}")

    # sample() must survive a numerically non-PSD correlation: corrcoef
    # roundoff on near-duplicate targets can push an eigenvalue just below 0,
    # where a bare cholesky raises.
    emu_psd = fit(Xs, np.column_stack([Yt[:, :2], Yt[:, 1]]))
    emu_psd.corr = np.array([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 1.0 + 1e-9],
                             [0.0, 1.0 + 1e-9, 1.0]])
    d_psd = emu_psd.sample(Xs[:50], size=100, rng=3)
    assert np.isfinite(d_psd).all(), "sample not robust to corrcoef roundoff"
    rr = np.corrcoef(((d_psd - d_psd.mean(0)) / d_psd.std(0)).reshape(-1, 3).T)
    assert rr[1, 2] > 0.99, "PSD repair must preserve the correlation structure"

    # (2) exp19 reproduction on the real catalog (skipped if data absent).
    data = _load_real()
    if data is None:
        print("emulator self-check OK (synthetic only; catalog FITS not found)")
    else:
        X, Y = data
        mu, sigma, cov = _cv_oof(X, Y, "linear")
        crps = crps_gaussian(Y, mu, sigma).mean()
        nll = _joint_nll(Y, mu, cov)
        gap = _cond_cov_gap(Y, mu, sigma)
        # poly-2 mean should be marginally sharper (exp17)
        mu2, sig2, _ = _cv_oof(X, Y, "poly2")
        crps2 = crps_gaussian(Y, mu2, sig2).mean()
        print(f"emulator self-check OK  n={len(Y)}  "
              f"CRPS={crps:.4f} (poly2 {crps2:.4f})  NLL={nll:+.3f}  condgap={gap:.3f}")
        assert abs(crps - 0.0832) < 0.004, f"CRPS {crps} != exp19 ~0.0832"
        assert abs(nll - (-3.43)) < 0.08, f"NLL {nll} != exp19 ~-3.43"
        assert gap < 0.025, f"conditional gap {gap} too large"
        assert crps2 <= crps + 1e-4, f"poly2 {crps2} should not be worse than linear {crps}"
