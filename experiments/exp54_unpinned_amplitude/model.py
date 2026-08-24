"""exp54 Stage 3 — the redesigned deposition model (Contract B).

    R_trunc,j   = C * R200c(t_j)                        the boundary
    dM*_j       = eps_surv(H_j) * dMh_j                 the budget
    M*(<R,t_k)  = SUM_{t_j <= t_k} dM*_j * F_j(<R)      the sum

`F_j` is a unit-mass projected cumulative profile TRUNCATED at `R_trunc,j` and
renormalized, so `F_j(R_trunc,j) = 1` and `dM*_j` is by construction the mass
inside a HALO-DEFINED radius. There is no total-at-infinity, no per-object
rescaling, and no stellar quantity anywhere: see `halo.Halo`, which carries none.

`eps_surv` is the **effective surviving-central mass fraction**, NOT a
star-formation efficiency. It absorbs baryon conversion, stellar evolutionary
mass loss, stripping, central-versus-satellite assignment and ex-situ delivery,
and those must not be interpreted separately.

**The truncation reparameterization, which is the delicate part.** Truncating a
profile moves its half-mass radius inward, so a family truncated at
`R_trunc` and still labelled by its UNtruncated `R50` is no longer in the same
coordinate system as an untruncated one -- and the family shootout would then
compare different quantities. This module therefore parameterizes by the TRUE
post-truncation half-mass radius. Writing `F0` for the untruncated unit profile
in units of its own half-mass radius (`F0(1) = 0.5`) and `u = R_trunc/R50_0`:

    F_trunc(x) = F0(x) / F0(u)
    x_h(u)     = F0^-1( 0.5 F0(u) )        the truncated half-mass radius
    h(u)       = x_h(u) / u                = R50_true / R_trunc

`h` decreases monotonically from `0.5**(1/a)` (as `u -> 0`, `a` the inner
log-slope of `F0`) to 0, so it inverts. Given the size law's `R50_true`, the
model solves `h(u) = R50_true/R_trunc` for `u` and hence for the family's own
scale. `demo()` asserts `F(R50) = 0.5` and `F(R_trunc) = 1` numerically for
every family.

**THE TIME DEPENDENCE OF THE EFFICIENCY (E6, E7), and why the basis matters
more than the order.** E1/E2 give `log10 eps_surv` a single power law in
`ln(1+z)`. E4 replaced that with a free piecewise-linear curve whose knot
VALUES are the parameters, and it was the worst of the 45 variants in Stage
3.2 with essentially undetermined parameters -- for a reason that is structural
rather than statistical: `a0 + interp(x, knots, k)` is EXACTLY degenerate,
because adding a constant to `a0` and subtracting it from every `k` gives the
identical model. E6 and E7 avoid that by construction. Both keep E2 whole and
ADD a correction that is orthogonal to a constant and to `x` over the deposits'
own range of `ln(1+z)`, so the new coefficients cannot trade against `a0` or
`a_z`:

    E6   + sum_{d=2..} c_d * P_d(s)     shifted Legendre, s = s(x) in [-1, 1]
    E7   + sum_i       s_i * g_i(x)     hinges at fixed knots, then Gram-Schmidt
                                        against {1, x} and rescaled

`P_d` for `d >= 2` is orthogonal to `P_0 = 1` and `P_1 = s` by construction;
`g_i` is made so. Each basis function is normalized to `max |.| = 1` on the
deposit range, so its coefficient reads directly as the largest swing in
`log10 eps_surv` that term is allowed to contribute. Setting every new
coefficient to zero reproduces E2 EXACTLY, which the self-check asserts.

The orthogonality is with respect to the UNIFORM measure on the range of
`ln(1+z)`, not to the deposits' own mass-weighted measure, which is heavily
skewed to low redshift (90% of the deposited halo mass sits below z = 2.9).
So the columns of the fitted Jacobian are decorrelated but not orthogonal, and
the identifiability report, not this construction, is what settles whether the
extra coefficients are determined.

Axes are switchable and NESTED, so "does this term earn its place" is a
question the identifiability report can answer:

    efficiency  E1 base | +E2 mass x redshift | +E3 concentration
                | E4 free spline in ln(1+z) | E5 double power law in mass
                | E6 E2 + orthogonal polynomial in ln(1+z)
                | E7 E2 + hinges in ln(1+z)
    profile     sersic | moffat | gompertz_log | richards | expo | gauss
    size        S2 R200c | S2a R200c/c200c    | +S3 conditioning slopes

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \\
     experiments/exp54_unpinned_amplitude/model.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp53_deposition_only"):
    sys.path.insert(0, str(p))

import families                                          # noqa: E402

M_PIV = 13.5
#: log-radius grid for the profile tables, in units of the untruncated R50.
#: Wide enough that the Moffat's slow tail is represented; 1024 points keeps
#: the interpolated F(R50)=0.5 assertion below 1e-6.
_XG = np.geomspace(1e-6, 1e8, 1024)
_TABLE_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# the truncated profile                                                        #
# --------------------------------------------------------------------------- #
def _table(family, shape):
    """(x, F0(x)) for the untruncated unit profile, cached on (family, shape)."""
    key = (family, None if shape is None else tuple(np.round(shape, 9)))
    hit = _TABLE_CACHE.get(key)
    if hit is None:
        f0 = families.cog_unit(family, np.array([1.0]), shape, _XG)[:, 0]
        f0 = np.maximum.accumulate(np.clip(f0, 0.0, 1.0))     # monotone guard
        hit = (_XG, f0)
        if len(_TABLE_CACHE) > 64:
            _TABLE_CACHE.clear()
        _TABLE_CACHE[key] = hit
    return hit


def r50_ratio(family, shape, u):
    """h(u) = R50_true/R_trunc for truncation at u = R_trunc/R50_untruncated."""
    x, f0 = _table(family, shape)
    fu = np.interp(u, x, f0)
    xh = np.interp(0.5 * fu, f0, x)                       # F0^-1(0.5 F0(u))
    return xh / u


_H_CACHE: dict = {}


def _h_grid(family, shape):
    """(u_grid, h(u_grid)) for the truncation inversion, cached on the family.

    `h` depends on nothing that varies from deposit to deposit, so rebuilding it
    inside every `solve_u` call was recomputing an identical 512-point curve
    once per galaxy. Caching it is a pure speed change: the returned arrays are
    the same values the previous code built, and `stage35_time_law.py` checks
    the whole predicted profile against the pre-change code path.
    """
    key = (family, None if shape is None else tuple(np.round(shape, 9)))
    hit = _H_CACHE.get(key)
    if hit is None:
        u_grid = np.geomspace(1e-4, 1e6, 512)
        h = np.minimum.accumulate(r50_ratio(family, shape, u_grid))
        u_grid.flags.writeable = False
        h.flags.writeable = False
        if len(_H_CACHE) > 64:
            _H_CACHE.clear()
        hit = (u_grid, h)
        _H_CACHE[key] = hit
    return hit


def solve_u(family, shape, rho):
    """Invert h: the u whose truncated half-mass radius is rho * R_trunc.

    `rho` must be below `h(0+) = 0.5**(1/a)`; a request above it is
    unrepresentable (the profile cannot be that extended relative to its own
    truncation) and is clipped, which the caller prices as a bad fit rather
    than a crash.
    """
    u_grid, h = _h_grid(family, shape)
    rho = np.clip(np.asarray(rho, float), h[-1] * 1.000001, h[0] * 0.999999)
    return np.interp(-rho, -h, u_grid)                    # h decreasing


def cog_truncated(family, shape, r50_true, r_trunc, R):
    """(len(R), len(r50_true)) unit-mass CoGs, truncated and renormalized.

    `F(r_trunc) = 1` and `F(r50_true) = 0.5`, both by construction.
    """
    r50_true = np.atleast_1d(np.asarray(r50_true, float))
    r_trunc = np.atleast_1d(np.asarray(r_trunc, float))
    u = solve_u(family, shape, r50_true / r_trunc)
    r50_0 = r_trunc / u                                   # untruncated scale
    x, f0 = _table(family, shape)
    xx = np.asarray(R)[:, None] / r50_0[None, :]
    F = np.interp(xx, x, f0)
    Fu = np.interp(u, x, f0)[None, :]
    return np.clip(F / Fu, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# the model                                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Spec:
    """Which model. Every option nests inside the richer one."""
    family: str = "sersic"
    eff: str = "E1"                # E1 | E2 | E3 | E4 | E5  (E2/E3 stack)
    size: str = "S2"               # S2 | S2a
    conditioning: bool = False     # S3: slopes on cond(t_j)
    e3: bool = False               # add the concentration term to any eff
    n_knots: int = 4               # E4 only
    n_time: int = 2                # E6 | E7: how many extra time coefficients
    trunc_C: float = 3.0

    @property
    def eff_names(self):
        if self.eff == "E1":
            n = ["a0", "a_M", "a_z"]
        elif self.eff == "E2":
            n = ["a0", "a_M", "a_z", "a_Mz"]
        elif self.eff == "E4":
            n = ["a0", "a_M"] + [f"k{i}" for i in range(self.n_knots)]
        elif self.eff == "E5":
            n = ["log_eps0", "logM1", "beta", "gamma", "a_z"]
        elif self.eff == "E6":
            n = ["a0", "a_M", "a_z", "a_Mz"] + [
                f"c{d}" for d in range(2, 2 + self.n_time)]
        elif self.eff == "E7":
            if self.n_time > len(E7_KNOT_Z):
                raise ValueError(f"E7 has {len(E7_KNOT_Z)} knots defined, "
                                 f"asked for {self.n_time}")
            n = ["a0", "a_M", "a_z", "a_Mz"] + [
                f"s{i}" for i in range(self.n_time)]
        elif self.eff == "E8":
            n = ["a0", "a_M", "a_z", "a_Mz"] + [
                f"v{i}" for i in range(self.n_time - 2)]
        else:
            raise ValueError(f"unknown eff {self.eff!r}")
        return n + (["a_c"] if self.e3 else [])

    @property
    def size_names(self):
        n = ["log_f0", "b"]
        return n + (["f:logMh", "f:dlogc", "f:fform"] if self.conditioning else [])

    @property
    def theta_names(self):
        return (self.eff_names + self.size_names
                + list(families.shape_names(self.family)))

    @property
    def n_theta(self):
        return len(self.theta_names)

    @property
    def label(self):
        e = self.eff
        if e == "E6":
            e = f"E6p{self.n_time}"                # p = polynomial degrees used
        elif e == "E7":
            e = f"E7k{self.n_time}"                # k = knots used
        elif e == "E8":
            e = f"E8f{self.n_time}"                # f = free curve, n_time knots
        s = f"{self.family}-{e}{'+E3' if self.e3 else ''}-{self.size}"
        return s + ("+S3" if self.conditioning else "")

    def bounds(self):
        b = {"a0": (-6, 2), "a_M": (-3, 3), "a_z": (-3, 3), "a_Mz": (-3, 3),
             "a_c": (-4, 4), "log_eps0": (-6, 2), "logM1": (10.0, 15.5),
             "beta": (0.0, 4.0), "gamma": (0.0, 4.0),
             "log_f0": (-4.0, 0.0), "b": (-6.0, 6.0),
             **{f"c{d}": (-3.0, 3.0) for d in range(2, 8)},
             **{f"s{i}": (-3.0, 3.0) for i in range(8)},
             **{f"v{i}": (-3.0, 3.0) for i in range(20)}}
        out = [b.get(n, (-8.0, 8.0)) for n in self.eff_names + self.size_names]
        return out + list(families.shape_bounds(self.family))

    def unpack(self, theta):
        theta = np.asarray(theta, float)
        if theta.shape != (self.n_theta,):
            raise ValueError(f"{self.label} wants {self.n_theta} params, "
                             f"got {theta.shape}")
        ne, ns = len(self.eff_names), len(self.size_names)
        return theta[:ne], theta[ne:ne + ns], (
            tuple(theta[ne + ns:]) if len(theta) > ne + ns else None)


#: knot positions for E4, in ln(1+z); spans z ~ 0.2 to ~ 12
E4_KNOTS = np.linspace(np.log(1.2), np.log(13.0), 16)

#: The range of `x = ln(1+z)` the DEPOSITS actually span. Every galaxy in the
#: sample carries the same 72 merger-tree steps, running from z = 14.989 to
#: z = 0.400, so this is a property of the snapshot grid and not of a fit.
#: E6 and E7 build their basis on exactly this interval.
E6_X_LO, E6_X_HI = np.log(1.4), np.log(16.0)
#: E7's knot redshifts, in the order they are switched on by `n_time`. z = 1
#: and z = 3 split the deposits into three spans that each carry a substantial
#: share of the deposited halo mass; z = 6 is a third knot for completeness and
#: is expected to be poorly determined, since only a few percent of the
#: deposited mass is laid down above it.
E7_KNOT_Z = (1.0, 3.0, 6.0)
_TIME_BASIS_CACHE: dict = {}


def _legendre_shifted(x, degree):
    """P_degree of the affine map of [E6_X_LO, E6_X_HI] onto [-1, 1].

    Orthogonal to 1 and to x over that interval for `degree >= 2`, and bounded
    by 1 in absolute value, so the coefficient reads as the largest swing in
    log10 eps_surv the term may contribute.
    """
    t = 2.0 * (np.asarray(x, float) - E6_X_LO) / (E6_X_HI - E6_X_LO) - 1.0
    if degree == 2:
        return 0.5 * (3.0 * t ** 2 - 1.0)
    if degree == 3:
        return 0.5 * (5.0 * t ** 3 - 3.0 * t)
    if degree == 4:
        return 0.125 * (35.0 * t ** 4 - 30.0 * t ** 2 + 3.0)
    if degree == 5:
        return 0.125 * (63.0 * t ** 5 - 70.0 * t ** 3 + 15.0 * t)
    raise ValueError(f"E6 degree {degree} is not low-order; 2-5 are defined")


def _hinge_basis(n_time):
    """(knots, A, B, scale) for E7: the hinge basis made orthogonal to {1, x}.

    `g_i(x) = max(0, x - x_i)` is strongly collinear with 1 and with x, which
    is what makes a raw hinge spline hard to identify alongside `a0` and `a_z`.
    Each hinge is therefore projected off the span of {1, x} under the uniform
    measure on [E6_X_LO, E6_X_HI] -- integrals done exactly, not sampled --
    and then rescaled so `max |g| = 1` on that interval:

        g_i(x) = [ max(0, x - x_i) - A_i - B_i * x ] / scale_i

    Cached: the coefficients depend on nothing that varies during a fit.
    """
    hit = _TIME_BASIS_CACHE.get(("hinge", n_time))
    if hit is not None:
        return hit
    lo, hi = E6_X_LO, E6_X_HI
    L = hi - lo
    knots = np.array([np.log(1.0 + z) for z in E7_KNOT_Z[:n_time]])
    # exact moments of max(0, x - k) over [lo, hi] against 1 and against x
    A, B, sc = [], [], []
    for k in knots:
        m0 = 0.5 * (hi - k) ** 2 / L                       # <g>
        m1 = ((hi ** 3 - k ** 3) / 3.0 - 0.5 * k * (hi ** 2 - k ** 2)) / L
        # project onto the orthonormal pair {1, sqrt(12)/L * (x - xbar)}
        xbar = 0.5 * (lo + hi)
        var = L ** 2 / 12.0
        b = (m1 - m0 * xbar) / var                         # slope to remove
        a = m0 - b * xbar
        xg = np.linspace(lo, hi, 4096)
        g = np.maximum(0.0, xg - k) - a - b * xg
        s_ = float(np.max(np.abs(g)))
        A.append(a); B.append(b); sc.append(s_)
    hit = (knots, np.array(A), np.array(B), np.array(sc))
    _TIME_BASIS_CACHE[("hinge", n_time)] = hit
    return hit


def hinge_terms(x, n_time):
    """(n_time, ...) orthogonalized, unit-amplitude hinge basis at `x`."""
    knots, A, B, sc = _hinge_basis(n_time)
    x = np.asarray(x, float)
    out = np.empty((n_time,) + x.shape)
    for i in range(n_time):
        out[i] = (np.maximum(0.0, x - knots[i]) - A[i] - B[i] * x) / sc[i]
    return out


def _pl_basis(n_knots):
    """(knots, V) for E8: a FREE piecewise-linear curve in `x`, minus {1, x}.

    E8 is a CEILING, not a candidate model. It asks how well the efficiency law
    could possibly do if its dependence on time were free rather than
    low-order, while still being one curve SHARED by every galaxy. The answer
    bounds E6, E7 and any other shared time law from above, because a
    piecewise-linear curve on `n_knots` knots contains all of them to within its
    resolution.

    `V` is (n_knots, n_knots - 2): each column holds the values of one basis
    function AT THE KNOTS, so evaluation is one `np.interp`. The two dimensions
    removed are exactly the constant and the linear trend, which `a0` and `a_z`
    already carry -- without that removal the fit has two exactly flat
    directions, which is the defect that made E4 useless. Columns are made
    mutually orthogonal in L2 over the deposit range and rescaled to
    `max |basis| = 1`.
    """
    hit = _TIME_BASIS_CACHE.get(("pl", n_knots))
    if hit is not None:
        return hit
    if n_knots < 4:
        raise ValueError("E8 needs at least 4 knots to add anything to E2")
    knots = np.linspace(E6_X_LO, E6_X_HI, n_knots)
    h = knots[1] - knots[0]
    # Every integral below is ANALYTIC. Quadrature on a grid is not good enough
    # here: a hat function has a kink at each knot, so unless the grid lands on
    # every knot the trapezoid rule leaks O(h) into exactly the overlaps this
    # construction exists to zero -- which it did, at the 5e-4 level.
    G = (np.diag(np.r_[h / 3, np.full(n_knots - 2, 2 * h / 3), h / 3])
         + np.diag(np.full(n_knots - 1, h / 6), 1)
         + np.diag(np.full(n_knots - 1, h / 6), -1))      # hat mass matrix
    m0 = np.r_[h / 2, np.full(n_knots - 2, h), h / 2]     # int hat_i dx
    m1 = np.r_[knots[0] * h / 2 + h ** 2 / 6,             # int hat_i x dx
               h * knots[1:-1],
               knots[-1] * h / 2 - h ** 2 / 6]
    C = np.stack([m0, m1])                                # (2, n_knots)
    _, _, Vt = np.linalg.svd(C)
    N = Vt[2:].T                                          # (n_knots, n_knots-2)
    L = np.linalg.cholesky(N.T @ G @ N)
    V = N @ np.linalg.inv(L).T                            # L2-orthonormal
    V = V / np.max(np.abs(V), axis=0, keepdims=True)      # max |basis| = 1
    hit = (knots, V)
    _TIME_BASIS_CACHE[("pl", n_knots)] = hit
    return hit


def pl_terms(x, n_knots):
    """(n_knots-2, ...) free-curve basis at `x`, orthogonal to 1 and to x."""
    knots, V = _pl_basis(n_knots)
    x = np.asarray(x, float)
    return np.stack([np.interp(x, knots, V[:, j]) for j in range(V.shape[1])])


def log_eps(spec, eff, halo):
    """log10 eps_surv per deposit."""
    m = halo.logmh - M_PIV
    x = np.log1p(halo.z)
    if spec.eff == "E1":
        le = eff[0] + eff[1] * m + eff[2] * x
        i = 3
    elif spec.eff == "E2":
        le = eff[0] + eff[1] * m + eff[2] * x + eff[3] * m * x
        i = 4
    elif spec.eff == "E4":
        k = np.linspace(E4_KNOTS[0], E4_KNOTS[-1], spec.n_knots)
        le = eff[0] + eff[1] * m + np.interp(x, k, eff[2:2 + spec.n_knots])
        i = 2 + spec.n_knots
    elif spec.eff == "E6":
        le = eff[0] + eff[1] * m + eff[2] * x + eff[3] * m * x
        for d in range(2, 2 + spec.n_time):
            le = le + eff[2 + d] * _legendre_shifted(x, d)
        i = 4 + spec.n_time
    elif spec.eff == "E7":
        le = eff[0] + eff[1] * m + eff[2] * x + eff[3] * m * x
        g = hinge_terms(x, spec.n_time)
        for j in range(spec.n_time):
            le = le + eff[4 + j] * g[j]
        i = 4 + spec.n_time
    elif spec.eff == "E8":
        le = eff[0] + eff[1] * m + eff[2] * x + eff[3] * m * x
        g = pl_terms(x, spec.n_time)
        for j in range(spec.n_time - 2):
            le = le + eff[4 + j] * g[j]
        i = 2 + spec.n_time
    else:                                                  # E5
        le0, lm1, beta, gam, a_z = eff[:5]
        r = 10.0 ** np.clip(halo.logmh - lm1, -30, 30)
        le = (le0 + np.log10(2.0)
              - np.log10(r ** -beta + r ** gam) + a_z * x)
        i = 5
    if spec.e3:
        le = le + eff[i] * halo.dlogc
    return le


def r50_of(spec, size, halo):
    """The TRUE post-truncation half-mass radius of each deposit [kpc]."""
    scale = halo.r200c if spec.size == "S2" else halo.r200c / halo.c_eff
    lf = size[0] + size[1] * np.log10(1.0 + halo.z)
    if spec.conditioning:
        lf = (lf + size[2] * (halo.logmh - M_PIV)
              + size[3] * halo.dlogc + size[4] * (halo.f_form - 0.5))
    return 10.0 ** np.clip(lf, -8, 2) * scale


def forward(spec, theta, halo, epochs, R):
    """M*(<R) in Msun at the given epoch indices. HALO INPUT ONLY.

    Returns (len(epochs), len(R)) or None if this theta is degenerate for this
    halo (the caller prices the failure; nothing here invents a value).
    """
    eff, size, shape = spec.unpack(theta)
    ok = np.isfinite(halo.r200c) & (halo.r200c > 0)
    if ok.sum() < 5:
        return None
    dm = 10.0 ** np.clip(log_eps(spec, eff, halo), -30, 10) * halo.dmh
    r50 = r50_of(spec, size, halo)
    r_tr = spec.trunc_C * halo.r200c
    good = ok & np.isfinite(dm) & (dm >= 0) & np.isfinite(r50) & (r50 > 0)
    if good.sum() < 5 or not np.isfinite(dm[good]).all():
        return None
    B = cog_truncated(spec.family, shape, r50[good], r_tr[good], R)
    out = np.empty((len(epochs), len(R)))
    for i, k in enumerate(epochs):
        out[i] = B @ (dm[good] * halo.epoch_mask[k][good])
    return out if np.isfinite(out).all() else None


def default_theta(spec):
    """A physically sane starting point (E1 values from the Stage 1 probe)."""
    v = {"a0": -2.50, "a_M": -0.19, "a_z": 0.36, "a_Mz": 0.0, "a_c": 0.0,
         "log_eps0": -2.2, "logM1": 12.0, "beta": 1.0, "gamma": 0.5,
         "log_f0": -1.3, "b": -0.5}
    out = [v.get(n, 0.0) for n in spec.eff_names + spec.size_names]
    if spec.eff == "E4":
        out = [v["a0"], v["a_M"]] + [0.36 * k for k in
                                     np.linspace(E4_KNOTS[0], E4_KNOTS[-1],
                                                 spec.n_knots)]
        out += [v["log_f0"], v["b"]]
        if spec.conditioning:
            out += [0.0, 0.0, 0.0]
        if spec.e3:
            out.insert(2 + spec.n_knots, 0.0)
    return np.array(out + list(families.shape_defaults(spec.family)), float)


if __name__ == "__main__":
    import halo as H
    recs = H.build_records(rows=np.arange(0, 2397, 120))
    R = np.geomspace(2.0, 148.22, 24)
    print()

    # (1) TRUNCATION CONTRACT: F(R50)=0.5 and F(R_trunc)=1 for every family
    worst_h, worst_t = 0.0, 0.0
    for fam in families.FAMILIES:
        sh = families.shape_defaults(fam) or None
        for rho in (0.02, 0.05, 0.1, 0.2, 0.35):
            rt = np.array([500.0]); r50 = np.array([rho * 500.0])
            g = cog_truncated(fam, sh, r50, rt, np.array([r50[0], rt[0]]))
            worst_h = max(worst_h, abs(g[0, 0] - 0.5))
            worst_t = max(worst_t, abs(g[1, 0] - 1.0))
    print(f"  (1) truncation contract, all {len(families.FAMILIES)} families x 5 "
          f"sizes:\n      max |F(R50)-0.5| = {worst_h:.2e}   "
          f"max |F(R_trunc)-1| = {worst_t:.2e}")
    assert worst_h < 5e-3 and worst_t < 1e-12

    # (2) the reparameterization MATTERS: untruncated R50 differs from the true one
    fam, sh = "moffat", families.shape_defaults("moffat")
    for rho in (0.05, 0.1, 0.2):
        u = float(np.atleast_1d(solve_u(fam, sh, np.array([rho])))[0])
        r50_untrunc = 500.0 / u
        print(f"  (2) moffat, R50_true/R_trunc={rho:.2f}: true R50 = "
              f"{rho * 500:5.1f} kpc, UNtruncated R50 = {r50_untrunc:6.1f} kpc "
              f"-> mislabelling by {100 * (r50_untrunc / (rho * 500) - 1):+.0f}%")

    # (3) forward runs for every family, is finite, monotone, in Msun
    h = recs[2]
    for fam in families.FAMILIES:
        sp = Spec(family=fam)
        m = forward(sp, default_theta(sp), h, [0, 4], R)
        assert m is not None and np.isfinite(m).all(), fam
        assert np.all(np.diff(m, axis=-1) >= -1e-9), f"{fam} not monotone in R"
        assert np.all(m[0] >= m[1] - 1e-9), f"{fam} not monotone in TIME"
    print(f"  (3) all families run; profiles monotone in radius AND in time")

    # (4) MASS CONSERVATION: the sum to R_trunc equals sum(eps dMh)
    sp = Spec(family="sersic")
    th = default_theta(sp)
    eff, size, shape = sp.unpack(th)
    ok = np.isfinite(h.r200c) & (h.r200c > 0)
    dm = 10.0 ** log_eps(sp, eff, h) * h.dmh
    want = float((dm * h.epoch_mask[0] * ok).sum())
    big = np.array([1e9])
    got = float(forward(sp, th, h, [0], big)[0, 0])
    print(f"  (4) mass conservation: |M(<1e9 kpc)/sum(eps dMh) - 1| = "
          f"{abs(got / want - 1):.2e}")
    assert abs(got / want - 1) < 1e-6

    # (5) POISONED DATA: the record has no stellar field to poison, and the
    #     forward call depends on nothing outside it
    assert not (set(vars(h)) & {"m500", "data", "logms"})
    print("  (5) poisoned-data: the halo record carries NO stellar field, so "
          "there is nothing to poison")

    # (6) every efficiency form runs and nests
    for eff in ("E1", "E2", "E4", "E5", "E6", "E7", "E8"):
        for e3 in (False, True):
            sp = Spec(family="sersic", eff=eff, e3=e3,
                      n_time=6 if eff == "E8" else 2)
            m = forward(sp, default_theta(sp), h, [0], R)
            assert m is not None and np.isfinite(m).all(), (eff, e3)
    sp1, sp2 = Spec(eff="E1"), Spec(eff="E2")
    t2 = default_theta(sp2); t2[3] = 0.0                  # a_Mz = 0 nests E1
    a = forward(sp1, default_theta(sp1), h, [0], R)
    b = forward(sp2, t2, h, [0], R)
    assert np.allclose(a, b, rtol=1e-12), "E2 does not nest E1"
    print("  (6) E1/E2/E4/E5/E6/E7/E8 all run, with and without E3; E2 nests "
          "E1 exactly")

    # (6b) THE NESTING THAT THE NEW TIME LAWS LIVE OR DIE BY. E6 and E7 are only
    #      interpretable as "E2 plus a correction" if zeroing every new
    #      coefficient gives back E2 to the last bit, at an ARBITRARY theta and
    #      not merely at the default one. Checked on all five epochs and on
    #      several galaxies, for every order of both forms.
    rng6 = np.random.default_rng(7)
    sp2 = Spec(family="gompertz_log", eff="E2")
    base2 = default_theta(sp2)
    worst_nest = 0.0
    for trial in range(4):
        t2 = base2 + rng6.normal(0, 0.25, base2.shape)
        for eff, nt in (("E6", 1), ("E6", 2), ("E6", 3), ("E7", 2), ("E7", 3),
                        ("E8", 6), ("E8", 8)):
            spx = Spec(family="gompertz_log", eff=eff, n_time=nt)
            n_new = spx.n_theta - sp2.n_theta
            tx = np.concatenate([t2[:4], np.zeros(n_new), t2[4:]])
            assert spx.theta_names[:4] == sp2.theta_names[:4]
            assert spx.n_theta == len(tx), (spx.label, spx.n_theta, len(tx))
            for hh in recs[:4]:
                a = forward(sp2, t2, hh, [0, 1, 2, 3, 4], R)
                b = forward(spx, tx, hh, [0, 1, 2, 3, 4], R)
                if a is None or b is None:
                    assert a is None and b is None, spx.label
                    continue
                worst_nest = max(worst_nest, float(np.max(np.abs(a - b))))
    assert worst_nest == 0.0, (f"E6/E7 do not nest E2 exactly: max |dM*| = "
                               f"{worst_nest:.3e} Msun")
    print(f"  (6b) E6 (1-3 extra terms), E7 (2-3 knots) and E8 (6, 8 knots) "
          f"reproduce E2 BIT FOR "
          f"BIT\n       when the new coefficients are zero: max |dM*(<R)| = "
          f"{worst_nest:.1e} Msun over 4 random\n       thetas x 5 epochs x 4 "
          f"galaxies x 24 radii")

    # (6c) the new basis functions really are orthogonal to a constant and to x
    #      over the deposit range, which is what stops them fighting a0 and a_z
    # the quadrature grid must CONTAIN every kink, or the test is measuring its
    # own trapezoid error rather than the basis
    kinks = np.r_[[np.log(1.0 + z) for z in E7_KNOT_Z],
                  np.linspace(E6_X_LO, E6_X_HI, 8)]
    xg = np.union1d(np.linspace(E6_X_LO, E6_X_HI, 20001), kinks)
    cols = ([_legendre_shifted(xg, d) for d in (2, 3, 4)]
            + list(hinge_terms(xg, 3)) + list(pl_terms(xg, 8)))
    bad = 0.0
    for c in cols:
        for ref in (np.ones_like(xg), xg - xg.mean()):
            ov = abs(np.trapezoid(c * ref, xg)) / (
                np.sqrt(np.trapezoid(ref ** 2, xg) * np.trapezoid(c ** 2, xg)))
            bad = max(bad, ov)
        # <= 1 exactly; a fine grid that misses the knots of a piecewise-
        # linear basis function samples just below its peak
        pk = float(np.max(np.abs(c)))
        assert 0.99 < pk <= 1.0 + 1e-9, pk
    print(f"  (6c) all {len(cols)} time-basis functions: max normalized "
          f"overlap with "
          f"1 and with x\n       over the deposit range = {bad:.2e}, and every "
          f"one peaks at |basis| = 1")
    assert bad < 1e-6

    # (7) both size laws run and differ
    ma = forward(Spec(size="S2"), default_theta(Spec(size="S2")), h, [0], R)
    mb = forward(Spec(size="S2a"), default_theta(Spec(size="S2a")), h, [0], R)
    print(f"  (7) S2 vs S2a differ as they must: median |dlog M(<R)| = "
          f"{np.median(np.abs(np.log10(ma / mb))):.3f} dex")

    print("\nmodel.py self-check OK: the truncated profile is parameterized by "
          "its TRUE post-truncation half-mass radius; mass is conserved to the "
          "truncation boundary; every family and efficiency form runs; E2 nests "
          "E1 exactly; profiles are monotone in radius and in time.")
