"""exp48 step C — deposit profiles with a CUSP, and a fair test for Sersic.

Moffat is not a physical choice, and its core is the suspect. Its surface
density

    Sigma ~ [1 + (R/rc)^2]^{-gamma}

is FLAT as R -> 0 (dSigma/dR = 0 at the centre) — Gaussian-like, with no
cusp. exp47 measured the model to under-fill compact galaxies' centres by
30-44%, which is exactly the failure a flat-cored primitive would produce
when the data are cuspy.

Two things about the incumbent's selection are worth stating, because they
set what this module has to do:

1. The exp38 stage-1 shootout that chose Moffat scored candidates with
   "mean per-galaxy relative RMS of the model growth curve" — the very
   objective exp48 step B is testing. Moffat's win (0.2001 at z=0.4)
   over gausswing (0.1997 — gausswing was actually AHEAD there) was
   inside the noise of a metric now under suspicion.
2. Sersic was NOT rejected on physics. The table records it as
   "LOWER rail 3/5 epochs (n-scale degeneracy; needs an R50
   reparameterization)". It was disqualified by a parameterization
   problem that nobody fixed.

Families added here (each a unit-mass CoG on a radius grid, the
`exp38/shapes.py` contract, so they drop into the same machinery):

  cuspy       Sigma ~ (R/rc)^{-alpha} [1+(R/rc)^2]^{-gamma}
              M(<R)/Mtot = I_u(1 - alpha/2, gamma - 1 + alpha/2),
              u = X/(1+X), X = (R/rc)^2
              alpha = 0 recovers Moffat EXACTLY. alpha is the cusp the
              incumbent cannot make. Needs 0 <= alpha < 2 and
              alpha/2 + gamma > 1 for finite mass.

  sersic_r50  Sersic reparameterized in the half-light radius:
              a = R50 / b_n^n with b_n ~ 2n - 1/3 + 4/(405n) + ...,
              which removes the a-n degeneracy that railed the original.

Run: PYTHONPATH=. uv run python experiments/exp48_objective_profile/\
profiles.py {demo}
"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import betainc, gammainc, gammaincinv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))


def cuspy_cog(dM, rc, alpha, gam, Rgrid):
    """Cored power law with a CUSP; nests Moffat exactly at alpha=0.

    Sigma(R) ~ (R/rc)^{-alpha} [1 + (R/rc)^2]^{-gamma}

    M(<R) = 2 pi int_0^R R' Sigma dR'. With X = (R/rc)^2 the integral is
    pi rc^2 int_0^X x^{-alpha/2} (1+x)^{-gamma} dx, which is the
    incomplete beta function under u = x/(1+x):

        M(<R)/Mtot = I_u(1 - alpha/2, gamma - 1 + alpha/2)

    Convergence at the centre needs alpha < 2; finite total mass needs
    alpha/2 + gamma > 1.
    """
    dM = np.asarray(dM, float)
    rc = np.asarray(rc, float)
    a = float(alpha)
    g = float(gam)
    if not (0.0 <= a < 2.0):
        raise ValueError(f"alpha must be in [0, 2): {a}")
    p = 1.0 - a / 2.0                       # > 0
    q = g - 1.0 + a / 2.0                   # > 0 for finite mass
    if q <= 0.0:
        raise ValueError(f"alpha/2 + gamma must exceed 1: {a}, {g}")
    X = (np.asarray(Rgrid, float)[:, None] / rc[None, :]) ** 2
    u = X / (1.0 + X)
    return (dM[None, :] * betainc(p, q, u)).sum(1)


def cuspy_sigma(dM, rc, alpha, gam, Rgrid):
    """The matching surface density, normalized so each deposit's CoG
    tends to dM. Used by the self-check and for figures."""
    from scipy.special import beta as B
    dM = np.asarray(dM, float)
    rc = np.asarray(rc, float)
    a, g = float(alpha), float(gam)
    p, q = 1.0 - a / 2.0, g - 1.0 + a / 2.0
    r = np.asarray(Rgrid, float)[:, None]
    x = (r / rc[None, :]) ** 2
    # Mtot = pi rc^2 B(p, q)  =>  divide the raw shape by that
    norm = np.pi * rc[None, :] ** 2 * B(p, q)
    return (dM[None, :] * x ** (-a / 2.0) * (1.0 + x) ** (-g) / norm).sum(1)


def b_n(n):
    """Sersic b_n: the constant making a the half-light radius. The
    standard asymptotic series, accurate to ~1e-6 for n > 0.4; refined by
    one inverse-gamma solve so it is exact to machine precision."""
    n = float(n)
    approx = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n**2)
    # gammainc(2n, b) = 0.5 defines b_n exactly
    return float(gammaincinv(2.0 * n, 0.5)) if n > 0.05 else approx


def sersic_r50_cog(dM, r50, n, Rgrid):
    """Sersic deposits parameterized by the HALF-MASS radius.

    The exp38 form takes (a, n) where a is the exponential scale; a and n
    trade off almost exactly, which is the degeneracy that sent the
    original fit to a bound. Substituting a = R50 / b_n^n makes the scale
    parameter an observable, and the degeneracy largely disappears.
    """
    dM = np.asarray(dM, float)
    r50 = np.asarray(r50, float)
    n = float(n)
    a = r50 / (b_n(n) ** n)
    x = (np.asarray(Rgrid, float)[:, None] / a[None, :]) ** (1.0 / n)
    return (dM[None, :] * gammainc(2.0 * n, x)).sum(1)


def demo():
    from shapes import moffat_cog, sersic_cog, _num_cog_from_sigma

    R = np.logspace(np.log10(2.0), np.log10(148.0), 24)
    dM = np.array([0.7, 0.3])
    rc = np.array([5.0, 20.0])

    # (1) THE NESTING: alpha = 0 must reproduce Moffat to machine precision
    for gam in (1.2, 1.38, 2.5):
        err = np.abs(cuspy_cog(dM, rc, 0.0, gam, R)
                     - moffat_cog(dM, rc, gam, R)).max()
        assert err < 1e-12, f"cuspy(alpha=0) != moffat at gam={gam}: {err:.2e}"

    # (2) the closed form matches brute-force integration of its Sigma
    for a, gam in ((0.0, 1.4), (0.5, 1.4), (1.0, 1.8), (1.5, 2.2)):
        ref = _num_cog_from_sigma(
            lambda r: cuspy_sigma(dM, rc, a, gam, np.atleast_1d(r)), R)
        got = cuspy_cog(dM, rc, a, gam, R)
        rel = np.abs(got - ref).max() / dM.sum()
        assert rel < 2e-3, f"cuspy closed form != numeric (a={a}): {rel:.2e}"

    # (3) alpha really does make a CUSP: the log-slope of Sigma at small R
    # must approach -alpha, where Moffat's approaches 0
    rr = np.array([1e-3, 2e-3])
    for a in (0.0, 0.8, 1.5):
        s = cuspy_sigma(np.array([1.0]), np.array([5.0]), a, 1.5, rr)
        slope = np.diff(np.log(s))[0] / np.diff(np.log(rr))[0]
        assert abs(slope + a) < 1e-3, f"inner log-slope {slope:+.4f} != {-a}"

    # (4) more cusp = more mass in the centre at fixed total
    tot = cuspy_cog(dM, rc, 0.0, 1.5, np.array([1e6]))[0]
    f_in = [cuspy_cog(dM, rc, a, 1.5, np.array([2.0]))[0] / tot
            for a in (0.0, 0.5, 1.0, 1.5)]
    assert all(np.diff(f_in) > 0), f"cusp must add central mass: {f_in}"

    # (5) b_n is right: a Sersic with a = R50/b_n^n encloses HALF its mass
    # inside R50 — the property the reparameterization exists to enforce
    for n in (0.5, 1.0, 2.0, 4.0):
        half = sersic_r50_cog(np.array([1.0]), np.array([10.0]), n,
                              np.array([10.0]))[0]
        assert abs(half - 0.5) < 1e-9, f"n={n}: encloses {half:.6f}, not 0.5"

    # (6) the reparameterization is a RELABELING, not a new family: it must
    # reproduce the exp38 Sersic at the equivalent scale
    for n in (1.0, 2.5):
        a_eq = np.array([10.0]) / (b_n(n) ** n)
        err = np.abs(sersic_r50_cog(np.array([1.0]), np.array([10.0]), n, R)
                     - sersic_cog(np.array([1.0]), a_eq, n, R)).max()
        assert err < 1e-12, f"sersic_r50 != sersic at n={n}: {err:.2e}"

    print("demo OK — cuspy nests Moffat exactly at alpha=0 (<1e-12), closed "
          "form matches numeric integration to <2e-3, the inner log-slope "
          "of Sigma equals -alpha, central mass fraction rises with alpha "
          f"({f_in[0]:.3f} -> {f_in[-1]:.3f}); sersic_r50 encloses exactly "
          "half its mass inside R50 and relabels the exp38 family exactly")


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "demo") == "demo":
        demo()
    else:
        raise SystemExit(__doc__)
