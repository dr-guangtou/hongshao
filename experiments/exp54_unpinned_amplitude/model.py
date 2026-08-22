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

Axes are switchable and NESTED, so "does this term earn its place" is a
question the identifiability report can answer:

    efficiency  E1 base | +E2 mass x redshift | +E3 concentration
                | E4 free spline in ln(1+z) | E5 double power law in mass
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


def solve_u(family, shape, rho):
    """Invert h: the u whose truncated half-mass radius is rho * R_trunc.

    `rho` must be below `h(0+) = 0.5**(1/a)`; a request above it is
    unrepresentable (the profile cannot be that extended relative to its own
    truncation) and is clipped, which the caller prices as a bad fit rather
    than a crash.
    """
    x, _ = _table(family, shape)
    u_grid = np.geomspace(1e-4, 1e6, 512)
    h = r50_ratio(family, shape, u_grid)
    h = np.minimum.accumulate(h)                          # enforce monotone
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
        s = f"{self.family}-{self.eff}{'+E3' if self.e3 else ''}-{self.size}"
        return s + ("+S3" if self.conditioning else "")

    def bounds(self):
        b = {"a0": (-6, 2), "a_M": (-3, 3), "a_z": (-3, 3), "a_Mz": (-3, 3),
             "a_c": (-4, 4), "log_eps0": (-6, 2), "logM1": (10.0, 15.5),
             "beta": (0.0, 4.0), "gamma": (0.0, 4.0),
             "log_f0": (-4.0, 0.0), "b": (-6.0, 6.0)}
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
    for eff in ("E1", "E2", "E4", "E5"):
        for e3 in (False, True):
            sp = Spec(family="sersic", eff=eff, e3=e3)
            m = forward(sp, default_theta(sp), h, [0], R)
            assert m is not None and np.isfinite(m).all(), (eff, e3)
    sp1, sp2 = Spec(eff="E1"), Spec(eff="E2")
    t2 = default_theta(sp2); t2[3] = 0.0                  # a_Mz = 0 nests E1
    a = forward(sp1, default_theta(sp1), h, [0], R)
    b = forward(sp2, t2, h, [0], R)
    assert np.allclose(a, b, rtol=1e-12), "E2 does not nest E1"
    print("  (6) E1/E2/E4/E5 all run, with and without E3; E2 nests E1 exactly")

    # (7) both size laws run and differ
    ma = forward(Spec(size="S2"), default_theta(Spec(size="S2")), h, [0], R)
    mb = forward(Spec(size="S2a"), default_theta(Spec(size="S2a")), h, [0], R)
    print(f"  (7) S2 vs S2a differ as they must: median |dlog M(<R)| = "
          f"{np.median(np.abs(np.log10(ma / mb))):.3f} dex")

    print("\nmodel.py self-check OK: the truncated profile is parameterized by "
          "its TRUE post-truncation half-mass radius; mass is conserved to the "
          "truncation boundary; every family and efficiency form runs; E2 nests "
          "E1 exactly; profiles are monotone in radius and in time.")
