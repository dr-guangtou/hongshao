"""Deformation layer: a low-dimensional, physically-labeled forward model.

The frozen ``Emulator`` (hongshao.emulator) is the TNG-calibrated halo->galaxy
map — 54 numbers, fit once. For cosmological inference we do NOT vary all 54:
the summary statistics (stellar-mass function, stacked lensing in core/outskirt
bins, clustering) cannot constrain that many, and a 54-parameter fit would
absorb model error instead of *testing* the relation.

Instead we freeze the emulator and infer a small **deformation** of it: a few
scalar knobs (``Deform``) that nudge the emulator's predictions on the fly —
no re-fit, no training data — so the map applies to any N-body halo catalog at
every step of an MCMC / SBI loop. Each knob is one physical statement mapped to
one observable, and the baseline ``Deform()`` reproduces the frozen emulator
exactly, so a posterior sitting on the baseline means "consistent with TNG".

    mu'_j   = mu_j + d0 + d_slope*logmp_std + d_out*w_j + f_ab*mu_secondary_j
    sigma'  = s * sigma            (=> cov' = s^2 * cov; correlation R unchanged)

    d0       global M* normalization        <- SMF amplitude
    d_slope  halo-mass slope tilt           <- lensing: M_halo at fixed M*
    d_out    outskirt-vs-core differential  <- 2D core/outskirt lensing split
    f_ab     assembly-bias amplitude        <- clustering (secondary dependence
             on formation history & concentration at fixed mass; f_ab=-1 turns
             assembly bias off, f_ab=0 is TNG)
    s        global scatter scale           <- SMF shape (Eddington) + lensing spread

The deformation needs a **linear-mean** emulator (the inference baseline); the
"secondary" axis is the {logtc, early, late, c200c} block of the linear mean,
with logmp held out as the primary mass axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OUTSKIRT_WEIGHT = np.array([0.0, 0.0, 0.0, 1.0])   # which bins d_out moves


@dataclass(frozen=True)
class Deform:
    """The inference knobs; the defaults are the identity (frozen emulator)."""
    d0: float = 0.0          # global log-M* offset [dex]
    d_slope: float = 0.0     # extra dex of M* per +1 sigma of halo mass
    d_out: float = 0.0       # extra dex on the outskirt bin(s) [dex]
    f_ab: float = 0.0        # fractional change to the secondary (assembly-bias) dependence
    s: float = 1.0           # multiplicative scatter scale (>0)


def forward(emu, X, theta=Deform(), weight=OUTSKIRT_WEIGHT):
    """Deformed predictive ``(mu, sigma, cov)`` for halos ``X`` under ``theta``.

    Same shapes as ``Emulator.predict`` ((N,4), (N,4), (N,4,4)); ``Deform()``
    returns the emulator unchanged.
    """
    if emu.mean != "linear":
        raise ValueError("deformation baseline must be a linear-mean emulator")
    X = np.atleast_2d(np.asarray(X, float))
    Xs = (X - emu.mu_x) / emu.sd_x
    mu, sigma, cov = emu.predict(X)
    mu_secondary = Xs[:, 1:5] @ emu.beta[:, 2:6].T          # {logtc, early, late, c200c}
    mu = (mu + theta.d0
          + theta.d_slope * Xs[:, 0:1]                      # tilt about the mean halo mass
          + theta.d_out * np.asarray(weight, float)[None, :]
          + theta.f_ab * mu_secondary)
    return mu, theta.s * sigma, theta.s ** 2 * cov


def sample(emu, X, theta=Deform(), size=1, rng=None, weight=OUTSKIRT_WEIGHT):
    """Draw ``size`` correlated, heteroscedastic profiles from the deformed model.

    Returns (size, N, 4). The residual correlation ``R`` is invariant under the
    deformation, so we reuse the emulator's ``chol(R)``.
    """
    rng = np.random.default_rng(rng)
    mu, sigma, _ = forward(emu, X, theta, weight)
    chol = np.linalg.cholesky(emu.corr)
    z = rng.standard_normal((size, len(mu), 4))
    return mu[None] + sigma[None] * (z @ chol.T)


if __name__ == "__main__":  # self-check: baseline == emulator; each knob moves its target
    from hongshao.emulator import fit

    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 5))
    X[:, 0] = 13.5 + 0.6 * X[:, 0]                          # logmp-like scale
    Xs0 = (X - X.mean(0)) / X.std(0)
    beta_true = rng.normal(size=(4, 6))
    design = np.column_stack([np.ones(n), Xs0])
    sig_true = np.array([0.10, 0.15, 0.20, 0.25])
    Y = design @ beta_true.T + sig_true * rng.standard_normal((n, 4))
    emu = fit(X, Y)                                         # linear-mean baseline
    Xs = (X - emu.mu_x) / emu.sd_x
    p_mu, p_sig, p_cov = emu.predict(X)

    # (0) baseline is the identity
    b_mu, b_sig, b_cov = forward(emu, X)
    assert np.allclose(b_mu, p_mu) and np.allclose(b_sig, p_sig) and np.allclose(b_cov, p_cov)

    # (d0) rigid shift of every bin by exactly d0
    mu, *_ = forward(emu, X, Deform(d0=0.1))
    assert np.allclose(mu - p_mu, 0.1), "d0 not a rigid offset"

    # (d_slope) shift is linear in standardized logmp, slope d_slope, same all bins
    mu, *_ = forward(emu, X, Deform(d_slope=0.2))
    for j in range(4):
        sl, intc = np.polyfit(Xs[:, 0], (mu - p_mu)[:, j], 1)
        assert abs(sl - 0.2) < 1e-6 and abs(intc) < 1e-6, (j, sl, intc)

    # (d_out) moves only the outskirt bin; inner bins untouched
    mu, *_ = forward(emu, X, Deform(d_out=0.3))
    shift = mu - p_mu
    assert np.allclose(shift[:, :3], 0.0) and np.allclose(shift[:, 3], 0.3), "d_out leaked"

    # (f_ab=-1) removes the secondary dependence -> mean is the pure mass term
    mu, *_ = forward(emu, X, Deform(f_ab=-1.0))
    primary = emu.beta[:, 0] + np.outer(Xs[:, 0], emu.beta[:, 1])    # intercept + logmp
    assert np.allclose(mu, primary), "f_ab=-1 should leave only the mass term"
    assert mu.std(0).sum() < p_mu.std(0).sum(), "removing assembly bias must reduce spread"

    # (s) scales sigma by s and cov by s^2
    mu, sig, cov = forward(emu, X, Deform(s=2.0))
    assert np.allclose(sig, 2.0 * p_sig) and np.allclose(cov, 4.0 * p_cov), "scatter scale wrong"

    # sample(): standardized residuals are unit-variance and carry the correlation R
    res = (sample(emu, X, size=400, rng=2) - p_mu[None]) / p_sig[None]
    flat = res.reshape(-1, 4)
    assert np.allclose(flat.std(0), 1.0, rtol=0.05), "sample spread wrong"
    assert np.allclose(np.corrcoef(flat.T), emu.corr, atol=0.03), "sample correlation wrong"

    print("forward (deformer) self-check OK: baseline == emulator; "
          "d0/d_slope/d_out/f_ab/s each move only their intended target")
