"""exp38 — candidate deposit primitives (closed-form CoGs) + self-checks.

The exp36 verdict: the extended channel's Gaussian width scale rails at its
allowed bound (~1000 kpc) in every fit — a Gaussian has no outer wings, so
it can only supply outskirt light by inflating its scale. This module holds
the candidate replacement shapes. Every deposit family is expressed as a
unit-mass cumulative profile (CoG) on an arbitrary radius grid, summed over
deposits, so any candidate drops into the exp29/35/36 fitting machinery by
swapping one function.

Families (per-deposit surface density Sigma_i and closed-form M_i(<R)):

  gauss   Sigma ~ exp(-R^2/2s^2)            M ~ 1 - exp(-R^2/2s^2)
  sersic  Sigma ~ exp(-(R/a)^{1/n})         M ~ P(2n, (R/a)^{1/n})
          (n=0.5 == gauss with s = a/sqrt(2), EXACT; n=1 exponential)
  shell   Sigma ~ R^p exp(-R^2/2s^2)        M ~ P(p/2+1, R^2/2s^2)
          (p=0 == gauss; Sigma peaks at R = s*sqrt(p) — off-centre deposit)
  moffat  Sigma ~ (1+(R/rc)^2)^{-gam}       M ~ 1 - (1+(R/rc)^2)^{1-gam}
          (gam>1 for finite mass; a genuinely power-law, non-exponential tail)

P = regularized lower incomplete gamma (scipy.special.gammainc).
Pure math, no I/O. All CoGs return the SUM over deposits of dM_i * M_i(<R)
on Rgrid — shape (len(Rgrid),).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import gammainc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp29_outer_deposit"))
from deposit import deposit_cog, deposit_sigma_sersic                # noqa: E402


def gauss_cog(dM, s, Rgrid):
    """Centred-Gaussian deposits (the incumbent primitive)."""
    return deposit_cog(np.asarray(dM), np.asarray(s), 0.0, np.asarray(Rgrid))


def sersic_cog(dM, a, n, Rgrid):
    """Sersic-wing deposits: M_i(<R) = dM_i * P(2n, (R/a_i)^{1/n})."""
    dM, a, Rgrid = np.asarray(dM), np.asarray(a), np.asarray(Rgrid)
    x = (Rgrid[:, None] / a[None, :]) ** (1.0 / n)
    return (dM[None, :] * gammainc(2.0 * n, x)).sum(1)


def shell_cog(dM, s, p, Rgrid):
    """Off-centre 'shell' deposits (exp29 outer-weighted family, p free)."""
    return deposit_cog(np.asarray(dM), np.asarray(s), p, np.asarray(Rgrid))


def moffat_cog(dM, rc, gam, Rgrid):
    """Power-law-tail deposits: M_i(<R) = dM_i * [1 - (1+(R/rc_i)^2)^(1-gam)].

    Finite total mass requires gam > 1; the fit box must enforce it."""
    dM, rc, Rgrid = np.asarray(dM), np.asarray(rc), np.asarray(Rgrid)
    u = 1.0 + (Rgrid[:, None] / rc[None, :]) ** 2
    return (dM[None, :] * (1.0 - u ** (1.0 - gam))).sum(1)


def sigma_of_cog(cog_fn, Rgrid, *args):
    """Numerical surface density of a summed CoG (for checks/figures only)."""
    m = cog_fn(*args, Rgrid)
    dA = np.pi * (Rgrid[1:] ** 2 - Rgrid[:-1] ** 2)
    return np.diff(m) / dA, np.sqrt(Rgrid[:-1] * Rgrid[1:])


def _num_cog_from_sigma(sigma_fn, Rgrid, rmax=1e5, nquad=200_000):
    """Brute-force CoG by integrating 2*pi*R*Sigma(R) — validation only.

    Log-spaced radius grid: steep inner profiles concentrate their mass at
    small R, where a linear grid under-resolves the integrand."""
    r = np.logspace(-8, np.log10(rmax), nquad)
    integrand = 2.0 * np.pi * r * sigma_fn(r)
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))])
    return np.interp(Rgrid, r, cum)


def demo():
    R = np.logspace(np.log10(2.0), np.log10(148.0), 24)
    dM = np.array([0.7, 0.3])
    s = np.array([15.0, 60.0])

    # (1) EXACT nestings of the incumbent Gaussian
    a = s * np.sqrt(2.0)
    err = np.abs(sersic_cog(dM, a, 0.5, R) - gauss_cog(dM, s, R)).max()
    assert err < 1e-12, f"sersic n=0.5 != gauss: {err:.2e}"
    err = np.abs(shell_cog(dM, s, 0.0, R) - gauss_cog(dM, s, R)).max()
    assert err < 1e-12, f"shell p=0 != gauss: {err:.2e}"

    # (2) closed-form CoGs match brute-force integration of their Sigma
    for n in (1.0, 2.5, 4.0):
        ref = _num_cog_from_sigma(
            lambda r: deposit_sigma_sersic(dM, a, n, np.atleast_1d(r)), R)
        got = sersic_cog(dM, a, n, R)
        assert np.abs(got - ref).max() < 2e-4, (n, np.abs(got - ref).max())
    for gam in (1.5, 3.0):
        rc = np.array([20.0, 80.0])

        def sig_moffat(r):
            r = np.atleast_1d(r)
            amp = dM * (gam - 1.0) / (np.pi * rc ** 2)
            return (amp[None, :] * (1.0 + (r[:, None] / rc[None, :]) ** 2)
                    ** (-gam)).sum(1)
        ref = _num_cog_from_sigma(sig_moffat, R, rmax=1e6, nquad=400_000)
        got = moffat_cog(dM, rc, gam, R)
        assert np.abs(got - ref).max() < 2e-3, (gam, np.abs(got - ref).max())

    # (3) every family: monotone CoG, total mass reached far out (the n=4
    # Sersic tail converges slowly — needs a genuinely large radius)
    far = np.array([1e10])
    for label, m, tot in (
            ("gauss", gauss_cog(dM, s, R), gauss_cog(dM, s, far)),
            ("sersic4", sersic_cog(dM, a, 4.0, R), sersic_cog(dM, a, 4.0, far)),
            ("shell3", shell_cog(dM, s, 3.0, R), shell_cog(dM, s, 3.0, far)),
            ("moffat", moffat_cog(dM, s, 1.8, R), moffat_cog(dM, s, 1.8, far))):
        assert np.all(np.diff(m) > -1e-15), f"{label} CoG not monotone"
        assert abs(tot[0] - dM.sum()) < 1e-6, f"{label} misses total mass"

    # (4) wings: at equal HALF-MASS radius, heavier-tailed families put MORE
    # mass beyond 4 R50 than the Gaussian — the property the rail was faking
    def r50(f, *args):
        rr = np.logspace(-1, 5, 4000)
        return np.interp(0.5, f(*args, rr) / f(*args, np.array([1e7]))[0], rr)

    one = np.array([1.0])
    s1 = np.array([30.0])
    r50_g = r50(gauss_cog, one, s1)
    frac = {}
    for label, f, args in (("gauss", gauss_cog, (one, s1)),
                           ("sersic n=1", sersic_cog, (one, s1, 1.0)),
                           ("sersic n=4", sersic_cog, (one, s1, 4.0)),
                           ("moffat g=1.5", moffat_cog, (one, s1, 1.5))):
        scale = r50(f, *args)
        args_scaled = (one, args[1] * r50_g / scale, *args[2:])
        beyond = 1.0 - f(*args_scaled, np.array([4.0 * r50_g]))[0]
        frac[label] = beyond
    assert frac["sersic n=1"] > frac["gauss"]
    assert frac["sersic n=4"] > frac["sersic n=1"]
    assert frac["moffat g=1.5"] > frac["gauss"]

    print("shapes.demo OK: sersic n=0.5 and shell p=0 nest the Gaussian "
          "exactly (1e-12); closed forms match brute-force integration; all "
          "CoGs monotone + mass-conserving; at matched R50 the mass beyond "
          "4 R50 is " + ", ".join(f"{k} {100*v:.1f}%" for k, v in frac.items()))


if __name__ == "__main__":
    demo()
