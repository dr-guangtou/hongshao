"""exp57 — the expansion term: deposits grow after they are laid down.

WHY THIS EXISTS. exp54 established that its own defect is outside its model
class. With every deposit non-negative and its profile fixed at the moment of
deposition, `M*(<R,t)` is non-decreasing in `t` at every radius and every
parameter value — while **42% of these galaxies LOSE a median 14% of their
central stellar mass** between redshift 2 and 0.4. Four enrichments (Stages
3.5-3.8) failed against a target the model class forbids.

An expansion term is the smallest change that admits a declining centre: let
each deposit's radii grow after it is deposited. Mass is conserved, nothing is
removed, nothing goes negative, no stellar information enters — and old
deposits can spread outward faster than new ones arrive, so the mass inside a
fixed small radius can fall.

THE HISTORY THAT CONSTRAINS THE DESIGN, and it is the whole reason this module
is shaped the way it is. The program's FIRST model was deposition + transport
(`exp30_transport_kernel`, tech note 2). `exp52_growth_mechanism` diagnosed its
failure: **the right source with roughly ten times too much reach.** 93% of the
mass it moved came from inside 5 kpc — correct, that is where AGN-driven
adiabatic expansion operates — but 58.5% of it was delivered beyond 50 kpc,
29.4% beyond 148 kpc (outside the measured profile entirely) and 12.0% beyond
500 kpc, where the normalisation discarded it. By redshift 0.7-0.4, **96.5% of
the model's outskirt growth was redistribution of stars it already had.**

**So the reach must be bounded BY CONSTRUCTION, not by a fitted exponent whose
consequence depends on a ratio of cosmic times.** The deferred proposal was
`g = (t_k/t_j)^alpha` applied to both radii. That is exactly the shape that
failed: a deposit made at 0.5 Gyr and viewed at 9.4 Gyr is scaled by
`18.8^alpha` — a factor of 14.5 at the old kernel's fitted 0.91 — and because
this deposit family has a genuine power-law tail, homologous scaling moves the
TAIL furthest in absolute terms. A 5% tail at 50 kpc lands at 725 kpc.

THE LAWS. `g` multiplies BOTH the deposit's half-mass radius and its truncation
radius, so the deposit keeps its shape exactly and its mass exactly (see
`cog_scaled` for the proof that this is free).

    X1  g = 1 + A (1 - t_j/t_k)                       1 parameter
    X2  g = 1 + A (1 - t_j/t_k)**p                    2 parameters
    X0  g = (t_k/t_j)**alpha                          1 parameter, THE CONTROL

`X1` is the primary candidate and is bounded by construction: `(1 - t_j/t_k)`
lies in `[0, 1)` for every deposit at every epoch, so `g` lies between 1 and
`1+A` no matter how early the deposit was made. `A` is therefore directly
interpretable — *the oldest deposits end up `(1+A)` times larger* — and is
directly gateable. `X2` lets the growth be non-linear in fractional age. `X0`
is the old transport form, unbounded, included as a declared control so that
"bounded costs nothing" is measured rather than assumed.

**`A = 0` (and `alpha = 0`) nests the current model exactly, and the bounds are
deliberately two-sided** — `A` may go negative, which would mean deposits
CONTRACT. This is a lesson from this project's own record: Stage 3.8's compact
weight `w0` went to zero, which was its bound, and a null sitting ON a bound is
weaker evidence than a null sitting at an interior point. Here the null is
interior, so "the model does not want expansion" is a measurement.

Run the self-check:
  HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
  experiments/exp57_expansion_term/expand.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", EXP54, HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import model as M                                        # noqa: E402
import stage35_time_law as S35                           # noqa: E402

#: the laws, and how many parameters each adds
LAWS = {"": 0, "X1": 1, "X2": 2, "X0": 1, "X3": 2}
LAW_NAMES = {"": [], "X1": ["A"], "X2": ["A", "p"], "X0": ["alpha"],
             "X3": ["A", "log_Rc"]}
#: two-sided, so the nesting point is INTERIOR for every law
LAW_BOUNDS = {
    "A": (-0.9, 20.0),        # -0.9 = deposits shrink to a tenth; +20 = x21
    "p": (0.1, 6.0),          # shape of the growth in fractional age
    "alpha": (-1.0, 2.0),     # the old transport exponent fitted to 0.91
    #: X3's core radius, log10 kpc. 0.32 kpc to 316 kpc: small enough to do
    #: nothing, large enough to reproduce X1 exactly in the limit.
    "log_Rc": (-0.5, 2.5),
}
#: the laws whose expansion is HOMOLOGOUS -- every radius scaled by the same
#: factor. X3 is not one of them, and that is the entire point of X3.
HOMOLOGOUS = ("X1", "X2", "X0")


@dataclass(frozen=True)
class XSpec:
    """An exp54 `model.Spec` plus an expansion law.

    Presents the same interface the fitting harness uses — `theta_names`,
    `bounds()`, `n_theta`, `unpack` — so `fit.fit` and every diagnostic that
    takes a spec keeps working unchanged.
    """
    base: M.Spec
    law: str = ""

    def __post_init__(self):
        if self.law not in LAWS:
            raise ValueError(f"law must be one of {sorted(LAWS)}, got "
                             f"{self.law!r}")

    @property
    def label(self):
        return self.base.label + (f"+{self.law}" if self.law else "")

    @property
    def x_names(self):
        return list(LAW_NAMES[self.law])

    @property
    def theta_names(self):
        return list(self.base.theta_names) + self.x_names

    @property
    def n_theta(self):
        return self.base.n_theta + LAWS[self.law]

    @property
    def family(self):
        return self.base.family

    @property
    def trunc_C(self):
        return self.base.trunc_C

    def bounds(self):
        return list(self.base.bounds()) + [LAW_BOUNDS[n] for n in self.x_names]

    def unpack(self, theta):
        """(eff, size, shape, xtheta) — the base's three plus the expansion."""
        theta = np.asarray(theta, float)
        if theta.shape != (self.n_theta,):
            raise ValueError(f"{self.label} wants {self.n_theta} params, got "
                             f"{theta.shape}")
        nx = LAWS[self.law]
        base_t = theta[: self.n_theta - nx] if nx else theta
        eff, size, shape = self.base.unpack(base_t)
        return eff, size, shape, theta[self.n_theta - nx:] if nx else np.array([])

    def nest(self, theta_base):
        """The base model's theta lifted into this spec, expansion switched OFF.

        The identity that makes every comparison honest: at these values `g` is
        identically 1, so the enriched model reproduces the incumbent exactly
        and a fit launched from here can never end up worse.
        """
        off = {"A": 0.0, "p": 1.0, "alpha": 0.0, "log_Rc": 1.0}
        return np.concatenate([np.asarray(theta_base, float),
                               np.array([off[n] for n in self.x_names])])


def g_factor(law, xtheta, t_dep, t_view):
    """The homologous scale factor `g(t_dep, t_view)`, broadcast over both.

    `t_dep` is when a deposit was laid down, `t_view` when the profile is
    observed, both in Gyr. Returns 1.0 exactly when the law is off, so the
    caller never has to special-case the nested model.

    A deposit is only ever viewed at or after its own deposition, but padded
    slots and the epoch mask can present `t_view < t_dep`; the fraction is
    clipped at 0 so those give `g = 1` and contribute nothing anomalous.
    """
    if not law:
        return np.ones(np.broadcast(t_dep, t_view).shape)
    frac = np.clip(1.0 - t_dep / np.maximum(t_view, 1e-9), 0.0, 1.0)
    if law == "X1":
        return 1.0 + xtheta[0] * frac
    if law == "X2":
        return 1.0 + xtheta[0] * frac ** xtheta[1]
    if law == "X0":
        return np.maximum(t_view, 1e-9) ** xtheta[0] / \
            np.maximum(t_dep, 1e-9) ** xtheta[0]
    if law == "X3":            # the same bounded time law as X1; the DIFFERENCE
        return 1.0 + xtheta[0] * frac      # is radial, not temporal
    raise ValueError(f"unknown law {law!r}")


def remap_radius(R, G, Rc):
    """`r(R)`: the radius a shell came FROM, given that it is now at `R`.

    THE POINT OF THE WHOLE FUNCTION. Homologous expansion multiplies every
    radius by the same `g`, so a deposit's power-law TAIL moves furthest in
    absolute terms — a 5% tail at 50 kpc scaled by 2.5 lands at 125 kpc. exp57
    Stage 3 measured the consequence: with `X1` fitted, 85% of the mass
    expansion moved was delivered beyond 50 kpc and 52% beyond 148 kpc, against
    preregistered limits of 25% and 10%. Bounding the FACTOR does not bound the
    REACH in kpc.

    This map bounds the reach in kpc directly. It expands the core by `G` and
    leaves everything beyond `Rc` where it was:

        r(R) = R [ 1 - (1 - 1/G) / (1 + (R/Rc)^2) ]

    At `R -> 0` it gives `R/G`, i.e. the material now at radius `R` came from
    `R/G` — a relative core expansion of `G`, with no hole opened at the
    centre. At `R >> Rc` it gives `R`: the outskirts are untouched. That is what
    adiabatic expansion is understood to do physically — it flattens the inner
    stellar profile, it does not build a stellar halo at 50-150 kpc — and it is
    what exp52's diagnosis asked for.

    **It is mass-conserving by construction** because it is a monotone
    rearrangement of the same profile, not a reweighting: the mass inside `r`
    simply moves to inside `r(r)`. Monotone for every `G > 0`: the derivative at
    `R = 0` is `1/G > 0` and rises from there.

    **X3 DOES NOT NEST X1, and two drafts of this docstring claimed it did
    before the self-check settled it.** As `Rc -> infinity` the bracket becomes
    `1/G` at every radius, so `r(R) = R/G` and the deposit is scaled by `G`
    everywhere — but with the truncation radius held **fixed**. `X1` instead
    carries the truncation outward with the deposit and renormalises at
    `G * r_trunc`. They are two different conventions for what "the deposit's
    mass" means, and both conserve it exactly within their own boundary.

    The gap between them is the mass lying between `r_trunc / G` and `r_trunc`,
    and it does NOT become negligible at large truncation, because this deposit
    family has a genuine **power-law tail**. Measured in the self-check below at
    `G = 2.5` with a truncation 400 times the half-mass radius — a ratio typical
    of this model — the two differ by **6.2e-3** in the enclosed fraction. That
    is small but it is not rounding, and it is a property of the power-law tail
    rather than of either convention.

    Holding the truncation fixed is the more defensible choice here: the
    truncation is set by the HALO, at `3 R200c`, and a deposit's stars
    rearranging internally is no reason for the halo's boundary to move. `X1`
    and `X3` are therefore compared by FITTING both, not by nesting one in the
    other.
    """
    x2 = (np.asarray(R, float) / Rc) ** 2
    return np.asarray(R, float) * (1.0 - (1.0 - 1.0 / G) / (1.0 + x2))


def cog_core_expanded(family, shape, r50_true, r_trunc, R, G, Rc):
    """Truncated unit-mass CoGs after the core-only expansion of `remap_radius`.

    The normalisation is taken at the MAPPED truncation radius, so
    `F(r_trunc) = 1` holds exactly and the deposit's mass is exactly conserved
    however far the core is pushed. At `G = 1` this reduces to
    `model.cog_truncated` term by term.
    """
    r50_true = np.atleast_1d(np.asarray(r50_true, float))
    r_trunc = np.atleast_1d(np.asarray(r_trunc, float))
    G = np.asarray(G, float)
    u = M.solve_u(family, shape, r50_true / r_trunc)
    r50_0 = r_trunc / u
    x, f0 = M._table(family, shape)
    rr = remap_radius(np.asarray(R, float)[:, None], G[None, :], Rc)
    rt = remap_radius(r_trunc, G, Rc)
    num = np.interp(np.clip(rr, 0.0, None) / r50_0[None, :], x, f0)
    den = np.interp(rt / r50_0, x, f0)[None, :]
    return np.clip(num / np.maximum(den, 1e-300), 0.0, 1.0)


def cog_scaled(family, shape, r50_true, r_trunc, R, g):
    """Truncated unit-mass CoGs with both radii multiplied by `g`.

    WHY THIS IS FREE, and why it is written out rather than looped over
    `model.cog_truncated`. That function computes

        u     = solve_u(family, shape, r50_true / r_trunc)
        r50_0 = r_trunc / u
        F(R)  = interp(R / r50_0) / interp(u)

    and `u` depends on the two radii ONLY through their ratio. Scaling both by
    the same `g` leaves the ratio unchanged, so `u` and the normalisation
    `interp(u)` are unchanged and only `R / (g r50_0)` moves. Two consequences:

      * `solve_u` — the expensive part — is evaluated ONCE for all epochs
        rather than once per epoch;
      * `F(r_trunc * g) = 1` still holds exactly, which IS the statement that
        homologous expansion conserves the deposit's mass. Nothing is created
        and nothing is lost; the same mass is spread over a larger radius.

    `g` broadcasts against `r50_true`, so a scalar, a per-deposit vector or a
    per-(deposit, epoch) array all work.
    """
    r50_true = np.atleast_1d(np.asarray(r50_true, float))
    r_trunc = np.atleast_1d(np.asarray(r_trunc, float))
    u = M.solve_u(family, shape, r50_true / r_trunc)
    r50_0 = r_trunc / u
    x, f0 = M._table(family, shape)
    xx = np.asarray(R)[:, None] / (np.asarray(g, float) * r50_0)[None, :]
    return np.clip(np.interp(xx, x, f0) / np.interp(u, x, f0)[None, :], 0.0, 1.0)


class ExpandingProblem(S35.StackedProblem):
    """`StackedProblem` in which each deposit's radii grow after deposition.

    Structurally identical to its parent except that the deposit basis now
    depends on the VIEWING epoch as well as the deposition epoch, so it is
    built once per epoch instead of once. That is the whole cost of the model
    class change, and `stage0_cost.py` measures it rather than estimating it.

    With `law = ""` this class must reproduce `StackedProblem` bit-for-bit;
    `stage1_contract.py` asserts it.
    """

    def __init__(self, spec, halos, data, mask, **kw):
        base = spec.base if isinstance(spec, XSpec) else spec
        super().__init__(base, halos, data, mask, **kw)
        self.xspec = spec if isinstance(spec, XSpec) else XSpec(base, "")
        self.spec = self.xspec                    # what fit.fit reads bounds from
        n = len(halos)
        t = np.full((n, self.nd), np.nan)
        for i, h in enumerate(halos):
            t[i, :len(h.t)] = h.t
        self.t_dep = t
        # The anchor epochs' cosmic times. `epoch_mask[k]` is `snap <=
        # ANCHOR_SNAP[k]` and every anchor IS a snapshot in the tree, so the
        # last True entry sits exactly at that anchor. Two things are then
        # CHECKED rather than assumed: that the redshift there is the anchor's
        # redshift, and that every galaxy gives the same time — because the
        # whole tabulation rests on the snapshot grid being shared.
        import halo as H
        def _t_at(h):
            out = []
            for k in range(len(H.ANCHOR_SNAP)):
                w = np.flatnonzero(h.epoch_mask[k])
                if not len(w):
                    return None
                out.append((float(h.t[w[-1]]), float(h.z[w[-1]])))
            return out
        ref = None
        for h in halos:
            got = _t_at(h)
            if got is None:
                continue
            zz = np.array([g[1] for g in got])
            # ANCHOR_Z are ROUNDED LABELS; the snapshots themselves sit at
            # 0.3999, 0.7001, 0.9973, 1.4955, 2.0020. The tolerance below is
            # therefore the label's rounding, not a fudge — it is checking that
            # the right SNAPSHOT was found, and the cosmic times used for the
            # expansion factor are the snapshot's own, never the label's.
            if not np.allclose(zz, H.ANCHOR_Z, atol=1e-2):
                raise ValueError(
                    f"the last deposit admitted at each anchor is not AT that "
                    f"anchor: redshifts {zz} against {H.ANCHOR_Z}. The epoch "
                    f"mask and the anchor list disagree.")
            tt = np.array([g[0] for g in got])
            if ref is None:
                ref = tt
            elif not np.allclose(tt, ref, rtol=1e-9):
                raise ValueError(
                    "the anchor-epoch cosmic times are NOT shared across "
                    "galaxies, so the expansion factor cannot be tabulated on "
                    "the snapshot grid. Fix the assumption, not this check.")
        if ref is None:
            raise ValueError("no galaxy admits a deposit at every anchor epoch")
        self.t_view = ref[list(self.epochs)]

    def setup(self, theta):
        """Everything a prediction needs except the radial basis.

        Split out so that `predict` and the gate machinery in `gates.py` build
        the SAME arrays from the SAME code rather than from two copies that can
        drift. They did drift once — `gates.deposit_terms` kept the homologous
        basis when `X3` was added, and the decomposition's own self-check
        caught it at a relative 6.3e-1 (PROBLEMS.md P6). The fix is to remove
        the duplication, not to patch the copy.
        """
        sp = self.xspec
        eff, size, shape, xtheta = sp.unpack(theta)
        d = self.dep
        ok = np.isfinite(d.r200c) & (d.r200c > 0)
        dm = 10.0 ** np.clip(M.log_eps(sp.base, eff, d), -30, 10) * d.dmh
        r50 = M.r50_of(sp.base, size, d)
        r_tr = sp.base.trunc_C * d.r200c
        good = (ok & np.isfinite(dm) & (dm >= 0)
                & np.isfinite(r50) & (r50 > 0))
        return dict(shape=shape, xtheta=xtheta, ok=ok, good=good,
                    dm=np.where(good, dm, 0.0),
                    r50s=np.where(good, r50, 1.0).ravel(),
                    r_trs=np.where(good, r_tr, 100.0).ravel(),
                    td=np.where(np.isfinite(self.t_dep), self.t_dep,
                                1.0).ravel())

    def basis(self, s, k, R):
        """The unit-mass deposit CoGs as seen at epoch `k`, on radii `R`.

        The ONE place the expansion law chooses a radial form, so a new law
        cannot be added to `predict` and forgotten in the diagnostics.
        """
        sp = self.xspec
        g = g_factor(sp.law, s["xtheta"], s["td"], self.t_view[k])
        if sp.law == "X3":
            return cog_core_expanded(sp.base.family, s["shape"], s["r50s"],
                                     s["r_trs"], R, g, 10.0 ** s["xtheta"][1])
        return cog_scaled(sp.base.family, s["shape"], s["r50s"], s["r_trs"],
                          R, g)

    def predict(self, theta):
        s = self.setup(theta)
        n, nd = self.dep.z.shape
        nE, nR = len(self.epochs), len(self.r_all)
        out = np.empty((n, nE, nR))
        W = s["dm"][:, None, :] * self.em                      # (n, nE, nd)
        for k in range(nE):
            Bg = np.ascontiguousarray(
                self.basis(s, k, self.r_all).T.reshape(n, nd, nR))
            out[:, k, :] = np.matmul(W[:, k, :][:, None, :], Bg)[:, 0, :]
        dead = ((s["ok"].sum(1) < 5) | (s["good"].sum(1) < 5)
                | ~np.isfinite(out).all((1, 2)))
        out[dead] = np.nan
        return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("exp57 expand.py — self-check on synthetic inputs\n")
    rng = np.random.default_rng(0)
    fam, shp = "gompertz_log", (0.7965,)
    R = F.R_GRID
    r50 = rng.uniform(1.0, 40.0, 500)
    rtr = r50 * rng.uniform(3.0, 60.0, 500)

    # 1. g = 1 must reproduce model.cog_truncated exactly
    a = M.cog_truncated(fam, shp, r50, rtr, R)
    b = cog_scaled(fam, shp, r50, rtr, R, 1.0)
    print(f"  g=1 vs model.cog_truncated      max|diff| = "
          f"{np.max(np.abs(a - b)):.3e}   (must be 0)")
    assert np.array_equal(a, b)

    # 2. homogeneity: scaling both radii by g equals evaluating at R/g
    g = 3.7
    c = cog_scaled(fam, shp, r50, rtr, R, g)
    e = M.cog_truncated(fam, shp, g * r50, g * rtr, R)
    print(f"  cog_scaled(g) vs cog_truncated(g*r50, g*rtr)  max|diff| = "
          f"{np.max(np.abs(c - e)):.3e}   (must be ~0)")
    assert np.max(np.abs(c - e)) < 1e-12

    # 3. MASS CONSERVATION: the CoG still reaches 1 at the scaled truncation
    far = cog_scaled(fam, shp, r50, rtr, np.array([1e9]), g)
    print(f"  mass conservation, F(inf) - 1   max|diff| = "
          f"{np.max(np.abs(far - 1.0)):.3e}   (must be ~0)")
    assert np.max(np.abs(far - 1.0)) < 1e-12

    # 4. the laws switch off exactly at their nesting values
    td = np.array([0.5, 2.0, 5.0, 8.0])
    for law, xt in (("X1", [0.0]), ("X2", [0.0, 1.0]), ("X0", [0.0])):
        gg = g_factor(law, np.array(xt), td, 9.4)
        print(f"  {law:<3} at its nesting theta: g = {gg}   (must be all 1)")
        assert np.allclose(gg, 1.0, atol=0, rtol=0)

    # 5. X1's reach is bounded by 1+A for ANY deposit time, which is the
    #    property the whole design turns on
    A = 4.0
    worst = g_factor("X1", np.array([A]), np.array([1e-6]), 9.4).max()
    print(f"  X1 reach with A={A}: worst g = {worst:.6f}  <= 1+A = {1 + A}")
    assert worst <= 1.0 + A + 1e-12
    # 6. X3 nests both the current model (G=1) and homologous expansion (Rc->inf)
    Gv = np.full(500, 2.5)
    n1 = cog_core_expanded(fam, shp, r50, rtr, R, np.ones(500), 10.0)
    print(f"  X3 at G=1 vs model.cog_truncated  max|diff| = "
          f"{np.max(np.abs(n1 - a)):.3e}   (must be ~0)")
    assert np.max(np.abs(n1 - a)) < 1e-12
    # X3 at Rc -> infinity is homologous expansion with the truncation held
    # FIXED, so it matches X1 only where the truncation is far outside the
    # evaluated radii -- which is this model's regime (r_trunc = 3 R200c) but
    # not the synthetic case above, where rtr can fall inside the grid.
    far_tr = r50 * 400.0                       # r_trunc >> max(R), as in exp54
    n2 = cog_core_expanded(fam, shp, r50, far_tr, R, Gv, 1e9)
    n3 = cog_scaled(fam, shp, r50, far_tr, R, Gv)
    print(f"  X3 at Rc->inf vs X1 (both homologous, truncation 400x R50): "
          f"max|diff| = {np.max(np.abs(n2 - n3)):.3e}")
    print(f"      NOT zero, and not meant to be: X3 holds the truncation at "
          f"the halo's 3 R200c\n      while X1 carries it outward, so they "
          f"differ by the mass between r_trunc/G\n      and r_trunc — which a "
          f"POWER-LAW tail never makes negligible. Both conserve\n      mass "
          f"exactly within their own boundary; they are compared by fitting "
          f"both.")
    far3 = cog_core_expanded(fam, shp, r50, rtr, np.array([1e9]), Gv, 10.0)
    print(f"  X3 mass conservation, F(inf) - 1  max|diff| = "
          f"{np.max(np.abs(far3 - 1.0)):.3e}   (must be ~0)")
    assert np.max(np.abs(far3 - 1.0)) < 1e-12
    rm = remap_radius(np.array([0.01, 1.0, 10.0, 100.0, 1000.0]), 2.5, 10.0)
    print(f"  X3 remap at Rc=10 kpc, G=2.5: r(R) for R = 0.01, 1, 10, 100, "
          f"1000 kpc\n    -> " + "  ".join(f"{v:.4g}" for v in rm)
          + "   (R/G near 0, R far out)")
    d = np.diff(remap_radius(np.linspace(1e-4, 3000, 20000), 2.5, 10.0))
    print(f"  X3 remap is MONOTONE: min dr/dR = {d.min():.3e}  (must be > 0)")
    assert d.min() > 0

    old = g_factor("X0", np.array([0.91]), np.array([0.5]), 9.4)
    print(f"  X0 (the OLD transport form) at alpha=0.91, t_dep=0.5 Gyr: "
          f"g = {float(np.ravel(old)[0]):.2f}  <-- this is the failure mode")
    print("\nexpand.py self-check OK")
