"""exp53 — deposit profile families, every one parameterized by its HALF-MASS
RADIUS.

exp38 stage 1 rejected the Sersic deposit family with "LOWER rail 3/5 epochs
(n-scale degeneracy; needs an R50 reparameterization)". That is a numerical
defect, not a physical verdict, and it eliminated the family the MEASURED added
light actually points at (exp38 stage 0.2: Sersic n = 2.1-3.1, kernel R50
15-47 kpc). This module supplies the missing reparameterization, and supplies
it for every family so the comparison is fair.

**The defect.** A Sersic profile's scale `a` and index `n` both set its
physical extent: at fixed `a`, raising `n` moves the half-mass radius by orders
of magnitude (`R50 = a * b_n**n`, and `b_n**n` runs from 0.83 at n=0.5 to ~1200
at n=8). A fit that wants a particular R50 can buy it with either parameter, so
the two trade off along a near-perfect valley and whichever hits its box first
rails. exp38 stage 0.2 hit exactly this and noted it in its own kernel fits:
"(n, a) are degenerate over a wing-only band -- read the kernel's half-mass
radius, not the raw scale".

**The fix.** Make `R50` itself the fitted scale and derive the family's native
scale from it. Then the shape parameter changes ONLY the shape at fixed size,
which is the thing it is supposed to mean, and `log_R50` denotes the same
physical quantity in every family -- so a family shootout, a deposit-radius
ceiling and the halo-conditioning slopes are all on common ground.

Families (per-deposit surface density; `M(<R)` closed form; `x = R/scale`):

  moffat  Sigma ~ (1+x^2)^-gam        M ~ 1 - (1+x^2)^(1-gam)      scale rc
  sersic  Sigma ~ exp(-x^(1/n))       M ~ P(2n, x^(1/n))           scale a
  expo    sersic with n = 1 fixed
  gauss   sersic with n = 0.5 fixed  (EXACTLY the incumbent Gaussian)

`P` = regularized lower incomplete gamma. Pure math, no I/O.

Self-check: `python experiments/exp53_deposition_only/families.py`.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammainc, gammaincinv

#: family -> (shape parameter name, box, default) ; None = no shape parameter
SHAPE = {
    "moffat": ("gam", (1.05, 6.0), 1.3771),
    "sersic": ("n", (0.25, 8.0), 2.5),
    "expo": (None, None, 1.0),
    "gauss": (None, None, 0.5),
}
FAMILIES = tuple(SHAPE)


def _sersic_index(family, shape):
    """The Sersic index a family evaluates at (expo/gauss ignore `shape`)."""
    if family == "expo":
        return 1.0
    if family == "gauss":
        return 0.5
    return float(shape)


def scale_from_r50(family, r50, shape):
    """Convert half-mass radii to the family's native scale parameter.

    ``r50`` is an array (one entry per deposit); ``shape`` is the scalar shape
    parameter. Returns an array of the same shape as ``r50``.
    """
    r50 = np.asarray(r50, float)
    if family == "moffat":
        gam = float(shape)
        if gam <= 1.0:
            raise ValueError(f"moffat needs gam > 1 for finite mass, got {gam}")
        # 1 - (1+x50^2)^(1-gam) = 1/2  ->  x50 = sqrt(2^(1/(gam-1)) - 1)
        x50 = np.sqrt(2.0 ** (1.0 / (gam - 1.0)) - 1.0)
        return r50 / x50
    n = _sersic_index(family, shape)
    if n <= 0.0:
        raise ValueError(f"sersic needs n > 0, got {n}")
    # P(2n, b_n) = 1/2  ->  R50 = a * b_n^n
    return r50 / gammaincinv(2.0 * n, 0.5) ** n


def cog_unit(family, r50, shape, Rgrid):
    """Unit-mass cumulative profile of ONE deposit family on ``Rgrid``.

    Returns shape ``(len(Rgrid), len(r50))`` -- the per-deposit CoGs, NOT
    summed, because the kernel needs to weight them by the core fraction
    before summing.
    """
    r50 = np.atleast_1d(np.asarray(r50, float))
    Rgrid = np.asarray(Rgrid, float)
    s = scale_from_r50(family, r50, shape)
    x = Rgrid[:, None] / s[None, :]
    if family == "moffat":
        return 1.0 - (1.0 + x ** 2) ** (1.0 - float(shape))
    n = _sersic_index(family, shape)
    return gammainc(2.0 * n, x ** (1.0 / n))


def cog(family, dM, r50, shape, Rgrid):
    """Mass-weighted sum over deposits: ``sum_i dM_i * M_i(<R)``."""
    dM = np.asarray(dM, float)
    return (dM[None, :] * cog_unit(family, r50, shape, Rgrid)).sum(1)


def shape_bounds(family):
    """``(lo, hi)`` for the family's shape parameter, or ``None``."""
    return SHAPE[family][1]


def n_params(family, q_free=True):
    """Parameter count: base + shape? + q? + 6 conditioning slopes."""
    base = 5 if q_free else 4          # log_R50, g, [q], mu, sig
    return base + (1 if SHAPE[family][0] else 0) + 6


# --------------------------------------------------------------------------- #
def _num_cog_from_sigma(sigma_fn, Rgrid, rmax=1e6, nquad=400_000):
    """Brute-force CoG by integrating 2 pi R Sigma(R) -- validation only."""
    r = np.logspace(-8, np.log10(rmax), nquad)
    integrand = 2.0 * np.pi * r * sigma_fn(r)
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))])
    return np.interp(Rgrid, r, cum)


def demo():
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))
    from shapes import gauss_cog, moffat_cog, sersic_cog       # noqa: E402

    R = np.geomspace(2.0, 148.0, 24)
    far = np.array([1e10])
    one = np.array([1.0])
    rr = np.geomspace(1e-3, 1e6, 60_000)

    # (1) THE POINT OF THE MODULE: r50 really is the half-mass radius, for
    #     every family and every shape value, to 0.1%. `cog` is unit-mass by
    #     construction, so the half-mass level is 0.5 outright -- do NOT
    #     renormalize by the value at the largest grid radius, which for a
    #     near-gam=1 Moffat is still 9% short of the total 1 Mpc out (that
    #     unconverged tail is itself the exp52 reach problem, in miniature).
    worst = 0.0
    for fam, shapes in (("moffat", [1.06, 1.3771, 2.0, 4.0, 6.0]),
                        ("sersic", [0.3, 0.5, 1.0, 2.5, 4.0, 8.0]),
                        ("expo", [None]), ("gauss", [None])):
        for sh in shapes:
            for target in (0.5, 3.0, 30.0, 300.0):
                m = cog(fam, one, np.array([target]), sh, rr)
                got = np.interp(0.5, m, rr)
                worst = max(worst, abs(got / target - 1.0))
    assert worst < 1e-3, f"R50 reparameterization is off by {worst:.2e}"

    # (2) EXACT nestings: gauss == sersic n=0.5 == the incumbent Gaussian, and
    #     expo == sersic n=1. The incumbent takes a Gaussian sigma, so convert:
    #     R50 = sigma * sqrt(2 ln 2).
    s = np.array([15.0, 60.0])
    r50 = s * np.sqrt(2.0 * np.log(2.0))
    dM = np.array([0.7, 0.3])
    err = np.abs(cog("gauss", dM, r50, None, R) - gauss_cog(dM, s, R)).max()
    assert err < 1e-12, f"gauss != incumbent Gaussian: {err:.2e}"
    err = np.abs(cog("gauss", dM, r50, None, R)
                 - cog("sersic", dM, r50, 0.5, R)).max()
    assert err < 1e-12, f"gauss != sersic n=0.5: {err:.2e}"
    err = np.abs(cog("expo", dM, r50, None, R)
                 - cog("sersic", dM, r50, 1.0, R)).max()
    assert err < 1e-12, f"expo != sersic n=1: {err:.2e}"

    # (3) the reparameterization is a change of COORDINATES, not of the family:
    #     it must reproduce exp38's raw-scale closed forms exactly.
    for n in (0.5, 1.0, 2.5, 4.0):
        a = scale_from_r50("sersic", r50, n)
        err = np.abs(cog("sersic", dM, r50, n, R) - sersic_cog(dM, a, n, R)).max()
        assert err < 1e-12, f"sersic n={n} reparam mismatch: {err:.2e}"
    for gam in (1.3771, 1.5, 3.0):
        rc = scale_from_r50("moffat", r50, gam)
        err = np.abs(cog("moffat", dM, r50, gam, R)
                     - moffat_cog(dM, rc, gam, R)).max()
        assert err < 1e-12, f"moffat gam={gam} reparam mismatch: {err:.2e}"

    # (4) closed forms match brute-force integration of their own Sigma
    for gam in (1.2, 3.0):
        rc = scale_from_r50("moffat", r50, gam)

        def sig_moffat(r):
            r = np.atleast_1d(r)
            amp = dM * (gam - 1.0) / (np.pi * rc ** 2)
            return (amp[None, :]
                    * (1.0 + (r[:, None] / rc[None, :]) ** 2) ** (-gam)).sum(1)
        ref = _num_cog_from_sigma(sig_moffat, R)
        assert np.abs(cog("moffat", dM, r50, gam, R) - ref).max() < 2e-3

    # (5) monotone, mass-conserving, finite everywhere
    for fam, sh in (("moffat", 1.3771), ("sersic", 4.0), ("sersic", 0.4),
                    ("expo", None), ("gauss", None)):
        m = cog(fam, dM, r50, sh, R)
        assert np.isfinite(m).all() and np.all(np.diff(m) > -1e-15), (fam, sh)
        assert abs(cog(fam, dM, r50, sh, far)[0] - dM.sum()) < 1e-6, (fam, sh)

    # (6) AT MATCHED R50 the shape parameter now controls wings ALONE -- the
    #     property the raw-scale parameterization confounded. Mass beyond
    #     4 R50 must rise monotonically with n, and the moffat must beat them.
    beyond = {}
    for label, fam, sh in (("gauss", "gauss", None), ("n=1", "sersic", 1.0),
                           ("n=2.5", "sersic", 2.5), ("n=4", "sersic", 4.0),
                           ("moffat 1.38", "moffat", 1.3771)):
        beyond[label] = 1.0 - cog(fam, one, np.array([30.0]), sh,
                                  np.array([120.0]))[0]
    order = ["gauss", "n=1", "n=2.5", "n=4"]
    for lo, hi in zip(order, order[1:]):
        assert beyond[hi] > beyond[lo], (lo, hi, beyond)
    assert beyond["moffat 1.38"] > beyond["n=4"], beyond

    # (7) the degeneracy is GONE in the new coordinates: at fixed R50, varying
    #     n over its whole box moves the profile a lot; in the OLD coordinates
    #     it moves R50 by orders of magnitude, which is what the fit exploited.
    from shapes import sersic_cog as _raw
    r50_old = [np.interp(0.5, _raw(one, np.array([1.0]), n, rr), rr)
               for n in (0.5, 8.0)]
    span_old = r50_old[1] / r50_old[0]
    assert span_old > 100.0, span_old       # a=1 fixed: R50 spans >100x

    print("families.demo OK: R50 IS the half-mass radius for every family and "
          f"shape (worst error {worst:.1e}); gauss == sersic n=0.5 == the exp38 "
          "Gaussian and expo == sersic n=1 to 1e-12; both reparameterizations "
          "reproduce exp38's raw-scale closed forms exactly; closed forms match "
          "brute-force integration; at matched R50 the mass beyond 4 R50 is "
          + ", ".join(f"{k} {100*v:.1f}%" for k, v in beyond.items())
          + f"; the degeneracy the old coordinates carried: at fixed a=1, n "
          f"0.5->8 moves R50 by {span_old:.0f}x")


if __name__ == "__main__":
    demo()
