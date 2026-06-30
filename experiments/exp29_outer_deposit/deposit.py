"""exp29 — the deposition kernel with an *outer-weighted* primitive.

exp25 deposited each epoch's new stellar mass as a centred 2-D Gaussian. exp26
showed the added mass is **inside-out and multiplicative** (peaks well outside the
centre, not at R=0). This module generalizes the centred Gaussian to a one-parameter
family that contains it:

    Sigma_i(R) = A_i * R^p * exp(-R^2 / 2 sigma_i^2)          (surface density)
    M_i(<R)    = dM*_i * P(p/2 + 1, R^2 / 2 sigma_i^2)        (curve of growth)

where ``P`` is the regularized lower incomplete gamma (``scipy.special.gammainc``)
and ``A_i`` is fixed by total mass ``dM*_i`` — so, exactly like the Gaussian, each
epoch still contributes a single shape number (its width), no free amplitude.

  p = 0  -> centred Gaussian   (P(1,x) = 1 - e^-x; recovers exp25 exactly)
  p > 0  -> off-centre deposit; Sigma peaks at R = sigma*sqrt(p) (a "shell"),
            leaving the core nearly fixed — the exp26 picture.

Pure math, no I/O, so the population step can reuse it. All masses h-free Msun.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammainc, gamma


def deposit_cog(dMstar: np.ndarray, sigma: np.ndarray, p: float, Rgrid: np.ndarray) -> np.ndarray:
    """Sum of outer-weighted deposits -> cumulative M*(<R). Closed-form (gammainc)."""
    x = Rgrid[:, None] ** 2 / (2.0 * sigma[None, :] ** 2)
    return (dMstar[None, :] * gammainc(p / 2.0 + 1.0, x)).sum(1)


def deposit_sigma(dMstar: np.ndarray, sigma: np.ndarray, p: float, Rgrid: np.ndarray) -> np.ndarray:
    """Sum of outer-weighted deposits -> surface density Sigma(R)."""
    amp = dMstar / (2.0 * np.pi * sigma ** (p + 2.0) * 2.0 ** (p / 2.0) * gamma(p / 2.0 + 1.0))
    return (amp[None, :] * Rgrid[:, None] ** p
            * np.exp(-Rgrid[:, None] ** 2 / (2.0 * sigma[None, :] ** 2))).sum(1)


def deposit_sigma_sersic(dMstar: np.ndarray, scale: np.ndarray, n: float, Rgrid: np.ndarray) -> np.ndarray:
    """Sum of mass-normalized **Sersic** deposits -> surface density Sigma(R).

    Sigma_i(R) = dM_i / (2 pi a_i^2 n Gamma(2n)) * exp(-(R/a_i)^{1/n}).
    The deposit's *wing* steepness is set by n: n=0.5 is a Gaussian (a=sigma*sqrt2),
    n=1 an exponential, n=4 de Vaucouleurs. A Gaussian (n=0.5) wing decays as
    exp(-R^2) — too steep to build the extended outskirts of a real Sigma profile;
    n>0.5 gives the shallower wing the multi-epoch profiles require (exp29)."""
    norm = 2.0 * np.pi * scale ** 2 * n * gamma(2.0 * n)
    return ((dMstar / norm)[None, :] * np.exp(-(Rgrid[:, None] / scale[None, :]) ** (1.0 / n))).sum(1)


def width_t(sigma0: float, g: float, t: np.ndarray, t_obs: float) -> np.ndarray:
    """Deposition width sigma(t) = sigma_0 (t/t_obs)^g (cosmic time, no R_200c)."""
    return sigma0 * (t / t_obs) ** g


def eff_two_epoch(z: np.ndarray, b_early: float, b_late: float, z_c: float) -> np.ndarray:
    """Two-epoch quenching efficiency weight, continuous at z_c (un-normalized)."""
    hi = (1.0 + z) ** b_early
    lo = (1.0 + z_c) ** (b_early - b_late) * (1.0 + z) ** b_late
    return np.where(z >= z_c, hi, lo)


def deposited(weight: np.ndarray, dMh: np.ndarray, Mstar_tot: float) -> np.ndarray:
    """Per-epoch stellar mass from efficiency weights, normalized to total mass."""
    return Mstar_tot * (weight * dMh) / (weight * dMh).sum()


def demo() -> None:
    """Self-check: p=0 recovers the centred Gaussian; Sigma peaks at sigma*sqrt(p)."""
    R = np.logspace(np.log10(2), np.log10(150), 80)
    dM, s = np.array([1.0]), np.array([20.0])
    gauss = 1.0 - np.exp(-R ** 2 / (2.0 * s[0] ** 2))
    assert np.allclose(deposit_cog(dM, s, 0.0, R), gauss), "p=0 != centred Gaussian"
    assert abs(deposit_cog(dM, s, 3.0, np.array([1e4]))[0] - 1.0) < 1e-6, "CoG must reach full mass"
    for p in (1.0, 4.0):
        peak = R[np.argmax(deposit_sigma(dM, s, p, R))]
        assert abs(peak - s[0] * np.sqrt(p)) < 0.1 * s[0], (p, peak)
    # Sersic n=0.5 == centred Gaussian with sigma = a/sqrt(2)
    a = np.array([s[0] * np.sqrt(2.0)])
    assert np.allclose(deposit_sigma_sersic(dM, a, 0.5, R), deposit_sigma(dM, s, 0.0, R), rtol=1e-6)
    print("deposit.demo OK: p=0 Gaussian, mass-conserving, Sigma peak at sigma*sqrt(p); "
          "Sersic n=0.5 == Gaussian")


if __name__ == "__main__":
    demo()
