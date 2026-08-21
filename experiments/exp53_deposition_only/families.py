"""exp53 — deposit profile families, every one parameterized by its HALF-MASS
RADIUS.

exp38 stage 1 rejected the Sersic deposit family with "LOWER rail 3/5 epochs
(n-scale degeneracy; needs an R50 reparameterization)". That is a numerical
defect, not a physical verdict, and it eliminated the family the MEASURED added
light actually points at (exp38 stage 0.2: Sersic n = 2.1-3.1, kernel R50
15-47 kpc). **PRIOR ART, and exp53 does not supersede it**: exp48 step C already built
`sersic_r50` and already fitted it inside the FULL multi-epoch kernel
(conditioning, migration clock, M(<500) normalization, z <= 1.5 scope) --
see `exp48/profiles.py::sersic_r50_cog`. What this module adds is narrower:
the SAME reparameterization applied to EVERY family, so `log_R50` is one
common physical coordinate across the shootout rather than a relabelling of
one candidate. exp48 left the Moffat in raw `rc` coordinates, so its scale
parameter and Sersic's did not mean the same thing.

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

#: family -> tuple of (shape parameter name, box, default). Empty tuple = no
#: free shape parameter. `richards` carries TWO, which is why this is a list
#: rather than the single slot the first version used.
SHAPE = {
    "moffat": (("gam", (1.05, 6.0), 1.3771),),
    "sersic": (("n", (0.25, 8.0), 2.5),),
    "expo": (),
    "gauss": (),
    # --- the generalized-sigmoid family (exp48/gompertz.py), in R50 form --- #
    "gompertz_log": (("c", (0.2, 8.0), 1.2),),
    "loglogistic": (("k", (0.2, 8.0), 1.2),),
    "richards": (("k", (0.2, 8.0), 1.2), ("nu", (0.02, 12.0), 0.5)),
}
FAMILIES = tuple(SHAPE)
#: families whose CoG is defined directly, not via a native scale parameter
SIGMOIDS = ("gompertz_log", "loglogistic", "richards")


def _as_tuple(shape):
    """Normalize a shape argument to a tuple (scalar and None both accepted)."""
    if shape is None:
        return ()
    if np.isscalar(shape):
        return (float(shape),)
    return tuple(float(v) for v in shape)


def _sersic_index(family, shape):
    """The Sersic index a family evaluates at (expo/gauss ignore `shape`)."""
    if family == "expo":
        return 1.0
    if family == "gauss":
        return 0.5
    return float(_as_tuple(shape)[0])


def scale_from_r50(family, r50, shape):
    """Convert half-mass radii to the family's native scale parameter.

    ``r50`` is an array (one entry per deposit); ``shape`` is the scalar shape
    parameter. Returns an array of the same shape as ``r50``.
    """
    r50 = np.asarray(r50, float)
    if family in SIGMOIDS:
        raise ValueError(f"{family} has no native scale parameter; it is "
                         f"defined directly in R50 (see cog_unit)")
    if family == "moffat":
        gam = float(_as_tuple(shape)[0])
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
    sh = _as_tuple(shape)

    if family in SIGMOIDS:
        # Sigmoids in log R. Each is written so that R50 IS the half-mass
        # radius by construction, which is what makes `log_R50` the same
        # physical coordinate here as for moffat/sersic.
        #   gompertz_log  M = exp(-ln2 (R/R50)^-c)
        #   loglogistic   M = X/(1+X),  X = (R/R50)^k
        #   richards      M = (1 + nu Y)^(-1/nu),  Y = (R/lam)^-k,
        #                 lam = R50 ((2^nu - 1)/nu)^(1/k)
        #                 -> nu->0 gives gompertz_log, nu=1 gives loglogistic
        u = Rgrid[:, None] / r50[None, :]
        if family == "gompertz_log":
            c = sh[0]
            return np.exp(-np.log(2.0) * np.exp(
                np.clip(-c * np.log(u), -700.0, 700.0)))
        if family == "loglogistic":
            k = sh[0]
            X = np.exp(np.clip(k * np.log(u), -700.0, 700.0))
            return X / (1.0 + X)
        k, nu = sh[0], sh[1]
        lam_fac = ((2.0 ** nu - 1.0) / nu) ** (1.0 / k)
        Y = np.exp(np.clip(-k * np.log(u / lam_fac), -700.0, 700.0))
        return (1.0 + nu * Y) ** (-1.0 / nu)

    s = scale_from_r50(family, r50, shape)
    x = Rgrid[:, None] / s[None, :]
    if family == "moffat":
        return 1.0 - (1.0 + x ** 2) ** (1.0 - sh[0])
    n = _sersic_index(family, shape)
    return gammainc(2.0 * n, x ** (1.0 / n))


def cog(family, dM, r50, shape, Rgrid):
    """Mass-weighted sum over deposits: ``sum_i dM_i * M_i(<R)``."""
    dM = np.asarray(dM, float)
    return (dM[None, :] * cog_unit(family, r50, shape, Rgrid)).sum(1)


def shape_names(family):
    return tuple(e[0] for e in SHAPE[family])


def shape_bounds(family):
    """Tuple of ``(lo, hi)`` per shape parameter (possibly empty)."""
    return tuple(e[1] for e in SHAPE[family])


def shape_defaults(family):
    return tuple(e[2] for e in SHAPE[family])


def n_params(family, q_free=True):
    """Parameter count: base + shape(s) + q? + 6 conditioning slopes."""
    base = 5 if q_free else 4          # log_R50, g, [q], mu, sig
    return base + len(SHAPE[family]) + 6


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
                        ("expo", [None]), ("gauss", [None]),
                        ("gompertz_log", [0.4, 1.2, 3.0, 8.0]),
                        ("loglogistic", [0.4, 1.2, 3.0, 8.0]),
                        ("richards", [(0.4, 0.05), (1.2, 0.5), (1.2, 1.0),
                                      (3.0, 5.0), (8.0, 12.0)])):
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

    # (4b) THE RICHARDS NESTING, which is why it is the efficient entry point
    #      to the sigmoid family: nu -> 0 recovers gompertz_log and nu = 1
    #      recovers loglogistic EXACTLY, so the fitted nu measures which
    #      sigmoid the data want instead of assuming one.
    Rn = np.geomspace(0.5, 500.0, 400)
    one1 = np.array([1.0])
    r1 = np.array([20.0])
    err = np.abs(cog("richards", one1, r1, (1.3, 1.0), Rn)
                 - cog("loglogistic", one1, r1, 1.3, Rn)).max()
    assert err < 1e-12, f"richards(nu=1) != loglogistic: {err:.2e}"
    err = np.abs(cog("richards", one1, r1, (1.3, 1e-7), Rn)
                 - cog("gompertz_log", one1, r1, 1.3, Rn)).max()
    assert err < 1e-6, f"richards(nu->0) != gompertz_log: {err:.2e}"

    # (4c) the sigmoids' TAILS, which is what the outskirt test cares about:
    #      gompertz_log and loglogistic are power laws outside, gompertz_lin
    #      is not represented here because it has no power-law tail at all
    beyond_s = {}
    for lab, fam, sh in (("gompertz_log c=1.2", "gompertz_log", 1.2),
                         ("loglogistic k=1.2", "loglogistic", 1.2),
                         ("moffat 1.38", "moffat", 1.3771)):
        beyond_s[lab] = 1.0 - cog(fam, one1, np.array([30.0]), sh,
                                  np.array([120.0]))[0]
    assert all(v > 0.01 for v in beyond_s.values()), beyond_s

    # (5) monotone, mass-conserving, finite everywhere
    for fam, sh in (("moffat", 1.3771), ("sersic", 4.0), ("sersic", 0.4),
                    ("expo", None), ("gauss", None),
                    ("gompertz_log", 1.2), ("loglogistic", 1.2),
                    ("richards", (1.2, 0.5))):
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
          f"0.5->8 moves R50 by {span_old:.0f}x.\n  SIGMOID family in: "
          f"richards nests loglogistic EXACTLY at nu=1 and gompertz_log in the "
          f"nu->0 limit, so its fitted nu measures which sigmoid the data "
          f"want; mass beyond 4 R50 is "
          + ", ".join(f"{k} {100*v:.1f}%" for k, v in beyond_s.items()))


if __name__ == "__main__":
    demo()
