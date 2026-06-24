"""Deformation layer: a low-dimensional, physically-labeled forward model.

The frozen emulators (``hongshao.emulator`` / ``hongshao.profile_emulator``) are
the TNG-calibrated halo->galaxy map. For cosmological inference we do NOT vary
all of their coefficients; instead we freeze them and infer a small
**deformation** — a few scalar knobs (``Deform``) that nudge the predictions on
the fly (no re-fit, no training data), so the map applies to any N-body halo
catalog at every step of an MCMC / SBI loop. Each knob is one physical statement
mapped to one observable, and the baseline ``Deform()`` reproduces the frozen
emulator exactly, so a posterior on the baseline means "consistent with TNG".

    mu'  = mu + (d0 + d_slope*logmp_std) * norm + d_out * outer + f_ab * mu_secondary
    sigma' = s * sigma            (=> cov' = s^2 * cov; correlation R unchanged)

    d0       global M* normalization        <- SMF amplitude
    d_slope  halo-mass slope tilt           <- lensing: M_halo at fixed M*
    d_out    outskirt-vs-core differential  <- 2D core/outskirt lensing split
    f_ab     assembly-bias amplitude        <- clustering (f_ab=-1 turns it off)
    s        global scatter scale           <- SMF shape (Eddington) + lensing spread

The deformation is **target-agnostic**: ``norm`` and ``outer`` are weight vectors
in the emulator's target space. For the aperture emulators (modes 1 & 2) the
defaults are ``norm = 1`` (every bin shifts together) and ``outer`` = the last
(outermost) bin. For the profile emulators (modes 3 & 4) use ``forward_profile``,
which sets ``norm`` on the anchor (a uniform profile shift) and builds ``outer``
as the compressed-space direction that lifts the large-radius profile — then
reconstructs the per-radius prediction. The baseline must be a **linear-mean**
emulator (the inference baseline; the secondary axis is its {logtc, early, late,
c200c, ...} block, with logmp held out as the primary mass axis).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Deform:
    """The inference knobs; the defaults are the identity (frozen emulator)."""
    d0: float = 0.0          # global log-M* offset [dex]
    d_slope: float = 0.0     # extra dex of M* per +1 sigma of halo mass
    d_out: float = 0.0       # extra dex along the outer-weight direction [dex]
    f_ab: float = 0.0        # fractional change to the secondary (assembly-bias) mean
    s: float = 1.0           # multiplicative scatter scale (>0)


def forward(emu, X, theta=Deform(), norm_weight=None, outer_weight=None):
    """Deformed predictive ``(mu, sigma, cov)`` for halos ``X`` under ``theta``.

    Same shapes as ``Emulator.predict`` ((N,T), (N,T), (N,T,T)); ``Deform()``
    returns the emulator unchanged. ``norm_weight`` (T,) is where the global
    normalization/tilt act (default: all targets); ``outer_weight`` (T,) is the
    outskirt-differential direction (default: the last target).
    """
    if emu.mean != "linear":
        raise ValueError("deformation baseline must be a linear-mean emulator")
    X = np.atleast_2d(np.asarray(X, float))
    Xs = (X - emu.mu_x) / emu.sd_x
    mu, sigma, cov = emu.predict(X)
    nt, nf = mu.shape[1], Xs.shape[1]
    norm = np.ones(nt) if norm_weight is None else np.asarray(norm_weight, float)
    outer = _last_unit(nt) if outer_weight is None else np.asarray(outer_weight, float)
    if norm.shape != (nt,) or outer.shape != (nt,):
        raise ValueError(f"norm/outer weights must have length T={nt}")
    mu_secondary = Xs[:, 1:] @ emu.beta[:, 2:1 + nf].T          # {logtc, early, late, c200c, ...}
    mu = (mu
          + (theta.d0 + theta.d_slope * Xs[:, 0:1]) * norm[None, :]
          + theta.d_out * outer[None, :]
          + theta.f_ab * mu_secondary)
    return mu, theta.s * sigma, theta.s ** 2 * cov


def sample(emu, X, theta=Deform(), size=1, rng=None, norm_weight=None, outer_weight=None):
    """Draw ``size`` correlated, heteroscedastic targets from the deformed model.

    Returns (size, N, T). The residual correlation ``R`` is invariant under the
    deformation, so we reuse the emulator's ``chol(R)``.
    """
    rng = np.random.default_rng(rng)
    mu, sigma, _ = forward(emu, X, theta, norm_weight, outer_weight)
    chol = np.linalg.cholesky(emu.corr)
    z = rng.standard_normal((size, len(mu), mu.shape[1]))
    return mu[None] + sigma[None] * (z @ chol.T)


# --------------------------------------------------------------------------- #
# profile deformer (modes 3 & 4): deform the core, reconstruct the profile     #
# --------------------------------------------------------------------------- #
def profile_deform_weights(pe, outer_radius_kpc=50.0):
    """Core-target-space ``(norm, outer)`` for deforming a ``ProfileEmulator``.

    ``norm`` acts on the anchor (logMtot) -> a uniform profile shift / mass tilt.
    ``outer`` is the [anchor, PC] direction whose reconstruction lifts radii
    beyond ``outer_radius_kpc`` as a *pure shape change* (zero-mean over radius,
    and the anchor component zeroed, so it does not move the total mass).
    """
    K = pe.modes.shape[0]
    norm = np.zeros(K + 1)
    norm[0] = 1.0
    A = np.column_stack([np.ones(len(pe.radii)), pe.modes.T])     # (R, K+1)
    w = (np.asarray(pe.radii, float) >= outer_radius_kpc).astype(float)
    w = w - w.mean()                                             # pure differential
    outer, *_ = np.linalg.lstsq(A, w, rcond=None)
    outer[0] = 0.0                                              # leave the total mass alone
    return norm, outer


def forward_profile(pe, X, theta=Deform(), outer_radius_kpc=50.0):
    """Deformed per-radius ``(mu, sigma)`` for a ``ProfileEmulator`` under ``theta``.

    ``Deform()`` reproduces ``pe.predict``. Each knob is profile-meaningful: d0 a
    uniform shift, d_slope a mass tilt, d_out an inner<->outer redistribution,
    f_ab the assembly-bias amplitude (on both the total mass and the shape), s the
    scatter scale. Returns (N, R), (N, R).
    """
    norm, outer = profile_deform_weights(pe, outer_radius_kpc)
    mu_c, _, cov_c = forward(pe.emu, X, theta, norm_weight=norm, outer_weight=outer)
    A = np.column_stack([np.ones(len(pe.radii)), pe.modes.T])
    mu_prof = mu_c @ A.T + pe.mean_shape
    var = np.einsum("rk,nkl,rl->nr", A, cov_c, A)
    return mu_prof, np.sqrt(var)


def _last_unit(n):
    w = np.zeros(n)
    w[-1] = 1.0
    return w


if __name__ == "__main__":  # self-check: baseline == emulator; each knob moves its target
    from hongshao.emulator import fit
    from hongshao.profile_emulator import fit_profile

    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 5))
    X[:, 0] = 13.5 + 0.6 * X[:, 0]                          # logmp-like scale
    Xs = (X - X.mean(0)) / X.std(0)
    beta_true = rng.normal(size=(4, 6))
    design = np.column_stack([np.ones(n), Xs])
    sig_true = np.array([0.10, 0.15, 0.20, 0.25])
    Y = design @ beta_true.T + sig_true * rng.standard_normal((n, 4))
    emu = fit(X, Y)                                         # 4-target linear baseline
    Xstd = (X - emu.mu_x) / emu.sd_x
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
        sl, intc = np.polyfit(Xstd[:, 0], (mu - p_mu)[:, j], 1)
        assert abs(sl - 0.2) < 1e-6 and abs(intc) < 1e-6, (j, sl, intc)

    # (d_out) default outer-weight moves only the last bin
    mu, *_ = forward(emu, X, Deform(d_out=0.3))
    shift = mu - p_mu
    assert np.allclose(shift[:, :3], 0.0) and np.allclose(shift[:, 3], 0.3), "d_out leaked"

    # (f_ab=-1) removes the secondary dependence -> mean is the pure mass term
    mu, *_ = forward(emu, X, Deform(f_ab=-1.0))
    primary = emu.beta[:, 0] + np.outer(Xstd[:, 0], emu.beta[:, 1])
    assert np.allclose(mu, primary), "f_ab=-1 should leave only the mass term"

    # (s) scales sigma by s and cov by s^2
    mu, sig, cov = forward(emu, X, Deform(s=2.0))
    assert np.allclose(sig, 2.0 * p_sig) and np.allclose(cov, 4.0 * p_cov), "scatter scale wrong"

    # sample(): standardized residuals are unit-variance and carry the correlation R
    res = (sample(emu, X, size=400, rng=2) - p_mu[None]) / p_sig[None]
    flat = res.reshape(-1, 4)
    assert np.allclose(flat.std(0), 1.0, rtol=0.05), "sample spread wrong"
    assert np.allclose(np.corrcoef(flat.T), emu.corr, atol=0.03), "sample correlation wrong"

    # (any T) a 6-target emulator works; default d_out hits the last of 6
    Y6 = np.column_stack([Y, Y[:, :2] + 0.05 * rng.standard_normal((n, 2))])
    emu6 = fit(X, Y6)
    m6 = forward(emu6, X, Deform(d_out=0.3))[0] - emu6.predict(X)[0]
    assert m6.shape[1] == 6 and np.allclose(m6[:, :5], 0.0) and np.allclose(m6[:, 5], 0.3)

    # (profile) forward_profile baseline == predict; knobs are profile-meaningful
    nr = 14
    rad = np.linspace(2.0, 120.0, nr)
    modes_t = np.linalg.svd(rng.normal(size=(nr, nr)))[0][:3]
    anchor_t = 11.5 + 0.4 * Xs[:, 0] + 0.05 * rng.normal(size=n)
    score_t = Xs[:, 1:4] * np.array([0.3, 0.2, 0.1]) + 0.1 * rng.normal(size=(n, 3))
    prof = anchor_t[:, None] + score_t @ modes_t
    pe = fit_profile(X, prof, anchor_t, rad, n_modes=3)
    mu0, sig0 = pe.predict(X)
    fmu0, fsig0 = forward_profile(pe, X)
    assert np.allclose(fmu0, mu0) and np.allclose(fsig0, sig0), "profile baseline != predict"
    # d0 shifts every radius uniformly
    fmu, _ = forward_profile(pe, X, Deform(d0=0.1))
    assert np.allclose(fmu - mu0, 0.1, atol=1e-6), "profile d0 not uniform"
    # d_out lifts the outskirts more than the core, with ~zero net (pure shape)
    fmu, _ = forward_profile(pe, X, Deform(d_out=0.2))
    dshift = (fmu - mu0).mean(0)
    inner, outerm = rad < 50.0, rad >= 50.0
    assert dshift[outerm].mean() > dshift[inner].mean() + 0.05, "profile d_out not outer-weighted"
    # s scales the per-radius sigma
    _, fsig = forward_profile(pe, X, Deform(s=1.5))
    assert np.allclose(fsig, 1.5 * sig0), "profile scatter scale wrong"

    print("forward (deformer) self-check OK: baseline == emulator for apertures (T=4,6) and "
          "profiles; d0/d_slope/d_out/f_ab/s each move only their intended target")
