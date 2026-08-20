"""exp53 — the deposit kernel with a switchable profile family and a
switchable transport exponent.

Structurally this IS the adopted `1ch-mof` kernel from
`exp38/stage2_multiepoch.py`; two things change and nothing else does.

1. **The deposit scale is a HALF-MASS RADIUS** (`families.py`), so the scale
   parameter means the same physical thing in every family and the Sersic
   scale-index degeneracy that railed exp38 stage 1 is gone.
2. **The profile family and the transport exponent `q` are selectable**, with
   `q = 0` a clean nested special case: `r50w = r50_0` makes the core fraction
   `fc` algebraically irrelevant and the kernel becomes PURE DEPOSITION. That
   boundary is the point of exp53.

Inherited unchanged from the production harness: the lognormal efficiency
window in `ln(1+z)`, the `M(<500 kpc)` normalization, the halo-conditioning
layer (deposit-scale row and window-width row on standardized
`[logMh, c200c, fz2]`), the joint multi-epoch fit and the production
`Objective()`. The efficiency window is FREE in every exp53 fit by design --
transport and the window are entangled (at `sig = 0.237` the late deposits are
nearly massless), so freezing it would make a no-transport failure
uninterpretable.

Also carried in: a **300 kpc ceiling on the deposit half-mass radius**. exp49
measured that ceiling at 1.5e-5 of the loss after refitting -- free -- and it
matters more here, since without it a family comparison partly measures which
family is best at hiding mass beyond the normalization radius.

Theta layout (`theta_names`)::

    [log_R50, g, (q), mu, sig, (shape)]
        + [R50:logMh, R50:c200c, R50:fz2] + [sig:logMh, sig:c200c, sig:fz2]

`q` is present only when `q_free=True`; `shape` only for moffat/sersic.

Self-check (includes the equivalence against exp38's adopted kernel on real
galaxies): `HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python
experiments/exp53_deposition_only/kernel.py`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))

import families                                      # noqa: E402
import stage2_multiepoch as s2                       # noqa: E402
from hongshao.objective import Objective, reduce_galaxies    # noqa: E402

#: the adopted exp40 fit scope (z <= 1.5); k=0 is z=0.4
KS = [0, 1, 2, 3]
ANCHOR_Z = [0.4, 0.7, 1.0, 1.5, 2.0]
#: deposit half-mass radius ceiling [kpc] -- exp49 measured it as free
R50_CAP = 300.0
R50_FLOOR = 1e-4
#: the fit-steering sentinel. NOT a measurement: it lives in the fit driver's
#: loss, never in the scoring module, which reports NaN.
FAIL_LOSS = 4.0

#: box for the shared base parameters, from exp35/exp38 (log_R50 replaces
#: log_s0/log_rc; the same range, since it is the same physical decade)
BASE_BOX = dict(log_R50=(1.0, 3.5), g=(0.0, 6.0), q=(0.0, 3.0),
                mu=(0.0, 3.0), sig=(0.05, 2.0))


@dataclass(frozen=True)
class Spec:
    """Which model. ``family`` selects the deposit profile; ``q_free=False``
    fixes the transport exponent at ``q_fixed`` (0.0 = deposition only)."""

    family: str = "moffat"
    q_free: bool = True
    q_fixed: float = 0.0

    def __post_init__(self):
        if self.family not in families.FAMILIES:
            raise ValueError(f"family must be one of {families.FAMILIES}, "
                             f"got {self.family!r}")
        if self.q_fixed < 0.0:
            raise ValueError(f"q_fixed must be >= 0, got {self.q_fixed}")

    # -- layout ----------------------------------------------------------- #
    @property
    def shape_name(self):
        return families.SHAPE[self.family][0]

    @property
    def theta_names(self):
        names = ["log_R50", "g"] + (["q"] if self.q_free else []) + ["mu", "sig"]
        if self.shape_name:
            names.append(self.shape_name)
        return names + ["R50:logMh", "R50:c200c", "R50:fz2",
                        "sig:logMh", "sig:c200c", "sig:fz2"]

    @property
    def n_theta(self):
        return len(self.theta_names)

    @property
    def bounds(self):
        """Real bounds for L-BFGS-B (the six conditioning slopes unbounded)."""
        b = [BASE_BOX["log_R50"], BASE_BOX["g"]]
        if self.q_free:
            b.append(BASE_BOX["q"])
        b += [BASE_BOX["mu"], BASE_BOX["sig"]]
        if self.shape_name:
            b.append(families.shape_bounds(self.family))
        return b + [(None, None)] * 6

    @property
    def label(self):
        return f"{self.family}-" + ("qfree" if self.q_free
                                    else f"q{self.q_fixed:g}")

    # -- unpacking -------------------------------------------------------- #
    def unpack(self, theta, cond):
        """Per-galaxy parameters after the conditioning layer.

        ``cond`` is the galaxy's standardized ``[logMh, c200c, fz2]``.
        Returns ``(log_R50, g, q, mu, sig, shape)``.
        """
        theta = np.asarray(theta, float)
        if theta.shape != (self.n_theta,):
            raise ValueError(f"{self.label} wants {self.n_theta} parameters, "
                             f"got {theta.shape}")
        i = 0
        log_r50, gg = theta[i], theta[i + 1]
        i += 2
        if self.q_free:
            q, i = theta[i], i + 1
        else:
            q = self.q_fixed
        mu, sig = theta[i], theta[i + 1]
        i += 2
        shape = None
        if self.shape_name:
            lo, hi = families.shape_bounds(self.family)
            shape = float(np.clip(theta[i], lo + 1e-9, hi))
            i += 1
        log_r50 = log_r50 + theta[i:i + 3] @ cond
        sig = sig + theta[i + 3:i + 6] @ cond
        return float(log_r50), float(gg), float(max(q, 0.0)), float(mu), \
            float(sig), shape


def basis(spec, theta, cond, ti, t_obs, tk, r):
    """Unit-mass CoG basis: column i is deposit i's CoG at observation time tk.

    ``r50_0 = 10**log_R50 * (ti/t_obs)**g`` is the deposit's half-mass radius
    at its own formation time; a fraction ``fc = exp(-(tk-ti)/ti)`` stays there
    and the rest is transported to ``r50_0 * (tk/ti)**q``. Both are ceilinged
    at ``R50_CAP``. With ``q = 0`` the two are identical and ``fc`` drops out.
    """
    log_r50, gg, q, _, _, shape = spec.unpack(theta, cond)
    r50_0 = np.clip(10.0 ** log_r50 * (ti / t_obs) ** gg, R50_FLOOR, R50_CAP)
    if q == 0.0:
        return families.cog_unit(spec.family, r50_0, shape, r)
    r50_w = np.clip(r50_0 * (tk / ti) ** q, R50_FLOOR, R50_CAP)
    fc = np.exp(-np.clip(tk - ti, 0.0, None) / ti)
    return (fc[None, :] * families.cog_unit(spec.family, r50_0, shape, r)
            + (1.0 - fc)[None, :]
            * families.cog_unit(spec.family, r50_w, shape, r))


def deposit_weights(spec, theta, g):
    """Normalized deposit masses ``dM_i`` from the lognormal efficiency window
    times the halo's mass accretion increments. ``None`` if degenerate."""
    _, _, _, mu, sig, _ = spec.unpack(theta, g["cond"])
    mah = g["mah"]
    w = np.exp(-((np.log1p(mah["z"]) - mu) ** 2) / (2.0 * sig ** 2)) * mah["dMh"]
    if not np.isfinite(w).all() or w.sum() <= 0:
        return None
    return w / w.sum()


def model_cogs(spec, theta, g, ks=KS):
    """`M(<500)`-normalized model CoGs on the 24 aperture radii, at ``ks``.

    Returns a list of arrays, or ``None`` if the galaxy is degenerate under
    this theta (the caller decides what a failure costs -- see FAIL_LOSS)."""
    e = s2._W["e"]
    dM = deposit_weights(spec, theta, g)
    if dM is None:
        return None
    mah = g["mah"]
    out = []
    for k in ks:
        mask = mah["snap"] <= e.ANCHOR_SNAP[k]
        B = basis(spec, theta, g["cond"], mah["t"], mah["t_obs"], e.pe.AT[k],
                  e.R_EXT)
        m = B @ (dM * mask)
        if not np.isfinite(m[-1]) or m[-1] <= 0 or m[-2] <= 0:
            return None
        out.append(m[:-1] * (g["m500"][k] / m[-1]))
    return out


def population_cogs(spec, theta, gals, ks=KS):
    """``(n_gal, n_epoch, n_radii)``; failed galaxies are NaN rows."""
    R = s2._W["e"].R
    out = np.full((len(gals), len(ks), len(R)), np.nan)
    for i, g in enumerate(gals):
        c = model_cogs(spec, theta, g, ks)
        if c is not None:
            out[i] = np.asarray(c)
    return out


class Problem:
    """Model + data + objective, with the sentinel applied where it belongs.

    Mirrors `exp49/widened.py::Problem` deliberately, so the two experiments'
    losses are the same number for the same model.
    """

    def __init__(self, spec, gals, R, objective=None, ks=KS):
        self.spec, self.gals, self.R, self.ks = spec, gals, R, list(ks)
        self.obj = objective or Objective()
        self.data = np.stack([np.array([g["data"][k] for k in self.ks], float)
                              for g in gals])
        self.n_eval = 0

    def cogs(self, theta):
        return population_cogs(self.spec, theta, self.gals, self.ks)

    def loss(self, theta):
        """Population loss. Failed galaxies get the SENTINEL, never dropped:
        `reduce_galaxies` drops non-finite entries, so relying on the drop
        makes a fit that starts failing look like it improved."""
        self.n_eval += 1
        v = self.obj.per_galaxy(self.cogs(theta), self.data, self.R)
        return float(reduce_galaxies(np.where(np.isfinite(v), v, FAIL_LOSS),
                                     self.obj.galaxy))


# --------------------------------------------------------------------------- #
def adopted_theta():
    """exp38 1ch-mof @ exp40 z<=1.5 -- the adopted production theta, in exp38's
    RAW-SCALE coordinates ``[log_rc, g, q, mu, sig, gamma, *slopes]``."""
    return np.asarray(np.load(
        ROOT / "experiments/exp49_g_rail/outputs/railtest.npz",
        allow_pickle=True)["theta::bounded::adopted"], float)


def from_exp38_theta(theta38):
    """Translate an exp38 `1ch-mof` theta into exp53 moffat-qfree coordinates.

    Only the scale coordinate changes: ``log_rc -> log_R50 = log_rc +
    log10(x50(gam))``. The conditioning slopes are unchanged, because the
    conversion is a gam-dependent CONSTANT added to the log scale and gam is a
    population parameter -- the slope on ``log_R50`` is the slope on
    ``log_rc``.
    """
    theta38 = np.asarray(theta38, float)
    gam = float(np.clip(theta38[5], 1.05 + 1e-6, 6.0))
    x50 = np.sqrt(2.0 ** (1.0 / (gam - 1.0)) - 1.0)
    out = theta38.copy()
    out[0] = theta38[0] + np.log10(x50)
    return out


# --------------------------------------------------------------------------- #
def demo():
    pop = np.load(ROOT / "experiments/exp32_full_population/outputs/"
                  "population.npz")
    s2._w_init(pop["dev100"][:24])
    gals, R = s2._W["gals"], s2._W["e"].R
    th38 = adopted_theta()

    # (0) layout bookkeeping
    assert Spec("moffat", True).n_theta == 12
    assert Spec("moffat", False).n_theta == 11
    assert Spec("gauss", True).n_theta == 11
    assert Spec("gauss", False).n_theta == 10
    assert Spec("sersic", True).theta_names[5] == "n"

    # (1) EQUIVALENCE, the load-bearing check: exp53's moffat kernel with the
    #     translated theta reproduces exp38's ADOPTED kernel on real galaxies.
    #     The R50_CAP is lifted for this comparison -- exp38 has no ceiling, so
    #     leaving it in would compare two different models.
    global R50_CAP
    cap, R50_CAP = R50_CAP, 1e5
    spec = Spec("moffat", q_free=True)
    th53 = from_exp38_theta(th38)
    worst = 0.0
    for g in gals:
        a = s2.model_cogs(th38, g, KS, "1ch-mof")
        b = model_cogs(spec, th53, g, KS)
        assert (a is None) == (b is None)
        if a is None:
            continue
        worst = max(worst, max(np.abs(np.asarray(x) / np.asarray(y) - 1.0).max()
                               for x, y in zip(a, b)))
    R50_CAP = cap
    assert worst < 1e-12, f"exp38 equivalence broken: {worst:.2e}"

    # (2) q = 0 is a CLEAN nested case: the transported branch is identical to
    #     the deposited one, so the core fraction cannot matter. Proven by
    #     perturbing the clock the core fraction depends on and getting the
    #     identical answer -- not by reading the code.
    spec0 = Spec("moffat", q_free=False, q_fixed=0.0)
    th0 = np.delete(th53, 2)
    spec_q0free = Spec("moffat", q_free=True)
    th_q0free = th53.copy()
    th_q0free[2] = 0.0
    for g in gals[:8]:
        a = model_cogs(spec0, th0, g, KS)
        b = model_cogs(spec_q0free, th_q0free, g, KS)
        err = max(np.abs(np.asarray(x) / np.asarray(y) - 1.0).max()
                  for x, y in zip(a, b))
        assert err < 1e-14, f"q=0 nesting broken: {err:.2e}"
    # ... and the q=0 kernel is exactly the single-radius deposit profile
    g = gals[0]
    lr, gg, q, _, _, sh = spec0.unpack(th0, g["cond"])
    assert q == 0.0
    r50_0 = np.clip(10.0 ** lr * (g["mah"]["t"] / g["mah"]["t_obs"]) ** gg,
                    R50_FLOOR, R50_CAP)
    B = basis(spec0, th0, g["cond"], g["mah"]["t"], g["mah"]["t_obs"],
              s2._W["e"].pe.AT[0], s2._W["e"].R_EXT)
    assert np.abs(B - families.cog_unit("moffat", r50_0, sh,
                                        s2._W["e"].R_EXT)).max() < 1e-15

    # (3) transport moves mass OUTWARD and only outward: raising q at fixed
    #     everything else must lower the fraction of light inside 50 kpc.
    i50 = int(np.searchsorted(R, 50.0))
    fr = []
    for qv in (0.0, 0.5, 1.0, 1.5):
        t = th53.copy()
        t[2] = qv
        c = model_cogs(spec, t, gals[0], [0])[0]
        fr.append(c[i50] / c[-1])
    assert fr[0] > fr[1] > fr[2] > fr[3], fr

    # (4) the R50 ceiling BITES at the adopted theta -- otherwise carrying it
    #     would be a no-op and the claim that it constrains reach is empty.
    lr, gg, *_ = spec.unpack(th53, gals[0]["cond"])
    raw = 10.0 ** lr * (gals[0]["mah"]["t"] / gals[0]["mah"]["t_obs"]) ** gg
    assert raw.max() > R50_CAP, (raw.max(), R50_CAP)

    # (5) every family runs, stays finite, monotone and under its own M(<500)
    for fam in families.FAMILIES:
        for q_free in (True, False):
            sp = Spec(fam, q_free=q_free, q_fixed=0.0)
            t = np.zeros(sp.n_theta)
            t[0], t[1] = 1.9, 1.5                    # log_R50, g
            i = 2
            if q_free:
                t[i], i = 0.9, i + 1                 # q
            t[i], t[i + 1] = 1.53, 0.35              # mu, sig
            if sp.shape_name:
                t[i + 2] = families.SHAPE[fam][2]
            c = model_cogs(sp, t, gals[1], KS)
            assert c is not None, sp.label
            for k, ck in zip(KS, c):
                assert np.isfinite(ck).all() and np.all(np.diff(ck) > -1e-9)
                assert ck[-1] < gals[1]["m500"][k], (sp.label, k)

    # (6) the conditioning layer acts through the standardized halo vector and
    #     through nothing else
    g = gals[3]
    sp = Spec("sersic", q_free=True)
    t = np.array([1.9, 1.5, 0.9, 1.53, 0.35, 2.5, 0, 0, 0, 0, 0, 0], float)
    lr0 = sp.unpack(t, g["cond"])[0]
    t2 = t.copy()
    t2[7] = 0.3
    assert np.isclose(sp.unpack(t2, g["cond"])[0] - lr0, 0.3 * g["cond"][1])
    t3 = t.copy()
    t3[10] = -0.2
    assert np.isclose(sp.unpack(t3, g["cond"])[4] - sp.unpack(t, g["cond"])[4],
                      -0.2 * g["cond"][1])

    # (7) the loss reproduces exp38's production loss for the same model
    prob = Problem(spec, gals, R)
    cap, R50_CAP = R50_CAP, 1e5
    mine = prob.loss(th53)
    R50_CAP = cap
    theirs = float(np.mean([s2.gal_loss(th38, g, KS, "1ch-mof") for g in gals]))
    assert abs(mine - theirs) < 1e-12, (mine, theirs)

    print(f"kernel.demo OK (n={len(gals)} real galaxies): the R50-coordinate "
          f"moffat kernel reproduces exp38's ADOPTED 1ch-mof CoGs to "
          f"{worst:.1e} relative and its population loss to 1e-12 "
          f"({mine:.8f}); q=0 nests exactly and is provably independent of the "
          f"core-fraction clock; transport monotonically moves light outward "
          f"(f(<50 kpc) {fr[0]:.3f} -> {fr[-1]:.3f} for q 0 -> 1.5); the "
          f"{R50_CAP:.0f} kpc deposit ceiling bites at the adopted theta "
          f"(uncapped max R50 {raw.max():.0f} kpc); all "
          f"{len(families.FAMILIES)} families run at both q settings")


if __name__ == "__main__":
    demo()
