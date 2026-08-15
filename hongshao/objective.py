"""The fitting objective: what a profile model is asked to minimize.

Graduated from exp48, which measured twenty objectives against the same fixed
model and found the objective to be the single most effective lever on the
kernel's known defect — it under-fills compact galaxies' centres at every
epoch. Comparing surface-DENSITY profiles with a LOG residual moved the mean
plane energy relative to the truth's own split-half floor from 2.72 to 1.95
(held-out 2.71 -> 1.96). That objective was NOT adopted (it steepens the
decline of stellar mass with redshift the wrong way), so the default here is
the objective the adopted model was fitted with, unchanged.

The point of this module is that varying the objective should no longer mean
editing a loss function by hand. Six axes:

    quantity   cog | density        compare M(<R) or Sigma(R)
    residual   frac | abs | log     (m-d)/d, (m-d)/norm, or log m - log d
    radial     rms | absmean | max  aggregate over radii (max is KS-like)
    weight     uniform | perdex | inner
    epoch      mean | weights       aggregate over epochs
    galaxy     mean | median | cvar20

``Objective()`` with no arguments is the production objective: fractional
residuals on the cumulative profile, root-mean-square over the 24 radii with
equal weight, arithmetic mean over epochs, arithmetic mean over galaxies.
``Objective(quantity="density", residual="log")`` is exp48's winner.

This module scores ARRAYS. It knows nothing about kernels, parameter vectors,
galaxy dictionaries or optimizers, so it can score any model — including ones
that do not exist yet. Model and truth come in as (n_galaxies, n_epochs,
n_radii) cumulative profiles in linear solar masses, on a shared radius grid in
kpc.

Two design points worth stating in full.

**A galaxy the model cannot represent scores NaN, not a magic number.** The
production code returns 4.0 (roughly ten times a typical good loss) when the
kernel fails to build a profile. That number is a "walk away" signal aimed at an
optimizer, not a measurement: a reported population mean silently containing a
few 4.0s cannot be told apart from a model that fits badly everywhere. Scoring
stays honest here and the sentinel belongs in the closure handed to the
minimizer. **The trap this creates:** ``reduce_galaxies`` DROPS non-finite
entries, so a fit that started failing on some galaxies would see its loss
silently improve. Anything driving a fit must substitute its sentinel BEFORE
reducing and must never rely on the drop.

**The density comparison differences both sides; it does not evaluate the model
analytically.** The deposit primitive really is a surface density (Moffat,
Sigma ~ (1 + (R/rc)^2)^-gamma, whose enclosed-mass form is what the kernel
evaluates), so the model could produce Sigma directly. It must not. TNG gives a
MEASURED curve of growth and no measured density profile, so the truth can only
be differenced — and differencing over a finite bin gives the area-AVERAGED
density across that annulus, not Sigma at the midpoint. Those differ wherever
the profile has curvature across the bin, most strongly where it is steepest:
the inner few kpc of compact galaxies, which is exactly the population under
study. Evaluating the model analytically while differencing the truth would
inject a bias shaped like the signal. (Differencing amplifies noise and a
non-monotonic measured curve gives a non-positive annulus — safe on TNG's
noiseless profiles, not safe on real observations.)

Self-check: ``python -m hongshao.objective``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .profile_emulator import density_from_cog

QUANTITIES = ("cog", "density")
RESIDUALS = ("frac", "abs", "log")
RADIALS = ("rms", "absmean", "max")
WEIGHTS = ("uniform", "perdex", "inner")
EPOCHS = ("mean", "weights")
GALAXIES = ("mean", "median", "cvar20")

_TINY = 1e-30
_MASS_FLOOR = 1.0            # Msun; matches density_from_cog's own clip

# Short axis tags for the canonical name. NOT k[0]: "residual" and "radial"
# would collide. These strings are also the keys exp48's result stores were
# written with, so they are fixed by backward compatibility, not taste.
_SHORT = dict(quantity="q", residual="res", radial="rad", weight="w",
              epoch="ep", galaxy="gal")
_AXES = ("quantity", "residual", "radial", "weight", "galaxy")

_WEIGHT_CACHE: dict = {}


def radial_weights(kind, radii):
    """Normalized weight per radius; sums to 1 so losses stay comparable.

    ``uniform`` weights every grid point equally — which, because the grid is
    logarithmic and the curve of growth is cumulative, means an outskirt error
    is effectively replicated across many nearly-identical points while the
    fast-changing centre gets few. ``perdex`` weights by the spacing in log
    radius, equalizing contribution per decade rather than per grid point.
    ``inner`` weights by 1/R.

    Cached on (kind, radii): the loss is called tens of thousands of times
    inside a Nelder-Mead fit and this does not vary between calls.
    """
    r = np.asarray(radii, float)
    key = (kind, r.tobytes(), r.shape)
    hit = _WEIGHT_CACHE.get(key)
    if hit is not None:
        return hit
    if kind == "uniform":
        w = np.ones(len(r))
    elif kind == "perdex":
        w = np.gradient(np.log10(r))
    elif kind == "inner":
        w = 1.0 / r
    else:
        raise ValueError(f"weight must be one of {WEIGHTS}, got {kind!r}")
    if not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError(f"{kind} weights are not positive and finite on this "
                         f"radius grid")
    w = w / w.sum()
    w.flags.writeable = False
    _WEIGHT_CACHE[key] = w
    return w


def reduce_galaxies(v, how):
    """Aggregate a per-galaxy loss vector into one population number.

    Non-finite entries are DROPPED, not propagated — see the module docstring
    for why anything driving a fit must substitute its own sentinel first.
    Returns NaN if nothing finite is left, rather than inventing a value.
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan")
    if how == "mean":
        return float(np.mean(v))
    if how == "median":
        return float(np.median(v))
    if how == "cvar20":
        k = max(int(np.ceil(0.20 * len(v))), 1)
        return float(np.mean(np.sort(v)[-k:]))          # the WORST fifth
    raise ValueError(f"galaxy must be one of {GALAXIES}, got {how!r}")


@dataclass(frozen=True)
class Objective:
    """One objective configuration. ``Objective()`` is the production default.

    Mirrors ``hongshao.forward.Deform``: a frozen set of knobs whose bare
    instance is the frozen baseline. Axis names are validated once here rather
    than on every call from inside a minimizer.

    ``epoch_weights`` is only meaningful with ``epoch="weights"``; give one
    number per epoch, in the order the epochs appear in the arrays. It is
    normalized to sum to 1.
    """

    quantity: str = "cog"
    residual: str = "frac"
    radial: str = "rms"
    weight: str = "uniform"
    epoch: str = "mean"
    galaxy: str = "mean"
    epoch_weights: tuple = None

    def __post_init__(self):
        for axis, allowed in (("quantity", QUANTITIES), ("residual", RESIDUALS),
                              ("radial", RADIALS), ("weight", WEIGHTS),
                              ("epoch", EPOCHS), ("galaxy", GALAXIES)):
            got = getattr(self, axis)
            if got not in allowed:
                raise ValueError(f"{axis} must be one of {allowed}, got {got!r}")
        if self.epoch == "weights":
            if self.epoch_weights is None:
                raise ValueError('epoch="weights" needs epoch_weights')
            w = np.asarray(self.epoch_weights, float)
            if w.ndim != 1 or len(w) == 0:
                raise ValueError("epoch_weights must be a 1-D sequence")
            if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
                raise ValueError("epoch_weights must be finite, non-negative "
                                 "and not all zero")
            object.__setattr__(self, "epoch_weights", tuple(float(x) for x in w))
        elif self.epoch_weights is not None:
            raise ValueError("epoch_weights is only meaningful with "
                             'epoch="weights"')

    # ---------------------------------------------------------------- naming #
    @property
    def name(self):
        """Canonical, order-stable string for use as a cache key.

        The epoch axis is omitted when it is the default, so names written by
        exp48 (five axes) still match what this produces.
        """
        parts = [f"{_SHORT[a]}={getattr(self, a)}" for a in _AXES]
        if self.epoch != "mean":
            w = ",".join(f"{x:g}" for x in self.epoch_weights)
            parts.insert(4, f"{_SHORT['epoch']}={self.epoch}[{w}]")
        return "|".join(parts)

    def to_dict(self):
        """Plain dict for JSON / npz stores. Emits the five exp48 axes always,
        and the epoch axis only when it is not the default, so configs written
        here stay readable by the exp48 scripts."""
        d = {a: getattr(self, a) for a in _AXES}
        if self.epoch != "mean":
            d["epoch"] = self.epoch
            d["epoch_weights"] = list(self.epoch_weights)
        return d

    @classmethod
    def from_dict(cls, d):
        """Build from a stored dict; tolerates the five-axis exp48 form."""
        d = dict(d)
        ew = d.pop("epoch_weights", None)
        return cls(**d, epoch_weights=None if ew is None else tuple(ew))

    def replace(self, **kw):
        """A copy with some axes changed (validated like a fresh instance)."""
        return replace(self, **kw)

    # --------------------------------------------------------------- scoring #
    def per_galaxy(self, model_cogs, data_cogs, radii):
        """Per-galaxy loss, shape (n_galaxies,).

        ``model_cogs`` and ``data_cogs`` are cumulative profiles in LINEAR
        solar masses, shape (n_galaxies, n_epochs, n_radii); a single galaxy
        may be passed as (n_epochs, n_radii). ``radii`` is the grid in kpc.

        A galaxy whose residual is not finite at any radius of any epoch —
        including one the model failed to produce, arriving as NaN — scores
        NaN. It is not given a stand-in value.
        """
        m, d, r = _as_arrays(model_cogs, data_cogs, radii)
        res, r_used = self._residual(m, d, r)
        w = radial_weights(self.weight, r_used)

        bad = ~np.isfinite(res).all(axis=(1, 2))            # (n_galaxies,)
        res = np.where(np.isfinite(res), res, 0.0)

        if self.radial == "rms":
            per_epoch = np.sqrt(np.einsum("gek,k->ge", res ** 2, w))
        elif self.radial == "absmean":
            per_epoch = np.einsum("gek,k->ge", np.abs(res), w)
        else:                                               # max: KS-like
            per_epoch = np.max(np.abs(res), axis=-1)        # weights unused

        if self.epoch == "mean":
            out = np.mean(per_epoch, axis=1)
        else:
            ew = np.asarray(self.epoch_weights, float)
            if len(ew) != per_epoch.shape[1]:
                raise ValueError(f"epoch_weights has {len(ew)} entries but the "
                                 f"data has {per_epoch.shape[1]} epochs")
            out = per_epoch @ (ew / ew.sum())
        return np.where(bad, np.nan, out)

    def __call__(self, model_cogs, data_cogs, radii):
        """The population number. NaN if no galaxy scored finitely."""
        return reduce_galaxies(self.per_galaxy(model_cogs, data_cogs, radii),
                               self.galaxy)

    def report(self, model_cogs, data_cogs, radii):
        """``{'value', 'n_failed', 'n_total'}`` — the number plus how many
        galaxies did not contribute to it."""
        v = self.per_galaxy(model_cogs, data_cogs, radii)
        return dict(value=reduce_galaxies(v, self.galaxy),
                    n_failed=int(np.isnan(v).sum()), n_total=int(len(v)))

    # ---------------------------------------------------------------------- #
    def _residual(self, m, d, r):
        """Per-radius residual (n_galaxies, n_epochs, n_used) and its grid."""
        if self.quantity == "density":
            m_v, r_used = _to_density(m, r)
            d_v, _ = _to_density(d, r)
            norm = np.max(d_v, axis=-1)                  # peak Sigma
        else:
            m_v, d_v, r_used = m, d, r
            norm = d_v[..., -1]                          # the galaxy's M(<Rmax)

        if self.residual == "frac":
            return (m_v - d_v) / np.clip(d_v, _TINY, None), r_used
        if self.residual == "abs":
            # Linear difference made dimensionless by the galaxy's OWN
            # normalization. Without that division this would also weight
            # massive galaxies above light ones, which is a different lever
            # than the intended radial re-weighting.
            return (m_v - d_v) / np.maximum(norm, _TINY)[..., None], r_used
        return (np.log10(np.clip(m_v, _TINY, None))
                - np.log10(np.clip(d_v, _TINY, None))), r_used


def _as_arrays(model_cogs, data_cogs, radii):
    m = np.asarray(model_cogs, float)
    d = np.asarray(data_cogs, float)
    r = np.asarray(radii, float)
    if m.ndim == 2:
        m = m[None]
    if d.ndim == 2:
        d = d[None]
    if m.ndim != 3 or d.ndim != 3:
        raise ValueError("profiles must be (n_galaxies, n_epochs, n_radii) or "
                         f"(n_epochs, n_radii), got {m.shape} and {d.shape}")
    if m.shape != d.shape:
        raise ValueError(f"model {m.shape} and data {d.shape} shapes differ")
    if r.ndim != 1 or len(r) != m.shape[-1]:
        raise ValueError(f"radii has {r.shape} entries but the profiles have "
                         f"{m.shape[-1]} radii")
    return m, d, r


def _to_density(cum, radii):
    """(g, e, R) linear cumulative masses -> (g, e, R-1) linear Sigma, and the
    midpoint radii. One flattened call; ``density_from_cog`` is the same
    operator applied on both sides, which is the point."""
    g, e, nr = cum.shape
    log_sigma, mid = density_from_cog(
        np.log10(np.clip(cum.reshape(-1, nr), _MASS_FLOOR, None)), radii)
    return (10.0 ** log_sigma).reshape(g, e, nr - 1), mid


if __name__ == "__main__":       # self-checks
    rng = np.random.default_rng(0)
    R = np.geomspace(1.0, 148.0, 24)
    # a plausible truth: Moffat-like cumulative profiles, 40 galaxies, 4 epochs
    rc = 10.0 ** rng.normal(0.7, 0.2, (40, 4, 1))
    truth = 1e11 * (1.0 - (1.0 + (R[None, None] / rc) ** 2) ** -1.2)

    # 1. a model equal to the truth scores exactly zero on every axis setting
    for q in QUANTITIES:
        for res in RESIDUALS:
            for rad in RADIALS:
                o = Objective(quantity=q, residual=res, radial=rad)
                assert o(truth, truth, R) == 0.0, (q, res, rad)

    # 2. radial aggregation ordering: max >= rms >= absmean, same residuals
    model = truth * (1.0 + rng.normal(0.0, 0.05, truth.shape))
    vals = {rad: Objective(radial=rad)(model, truth, R) for rad in RADIALS}
    assert vals["max"] >= vals["rms"] >= vals["absmean"], vals

    # 3. galaxy aggregation: the worst fifth is at least the mean
    assert (Objective(galaxy="cvar20")(model, truth, R)
            >= Objective(galaxy="mean")(model, truth, R))

    # 4. inner weighting must score a CENTRAL error above uniform weighting,
    #    and below it when the same error sits in the outskirts
    central = truth.copy()
    central[..., :4] *= 1.5
    outer = truth.copy()
    outer[..., -4:] *= 1.5
    assert (Objective(weight="inner")(central, truth, R)
            > Objective(weight="uniform")(central, truth, R))
    assert (Objective(weight="inner")(outer, truth, R)
            < Objective(weight="uniform")(outer, truth, R))

    # 5. every weight vector sums to 1 and is finite, on both radius grids
    mid_grid = np.sqrt(R[:-1] * R[1:])
    for grid in (R, mid_grid):
        for kind in WEIGHTS:
            w = radial_weights(kind, grid)
            assert abs(w.sum() - 1.0) < 1e-12 and np.isfinite(w).all(), kind
    # the density path lands on exactly that midpoint grid
    assert np.allclose(_to_density(truth, R)[1], mid_grid)

    # 6. a failed galaxy scores NaN, is counted, and does not poison the rest
    broken = model.copy()
    broken[3] = np.nan
    v = Objective().per_galaxy(broken, truth, R)
    assert np.isnan(v[3]) and np.isfinite(np.delete(v, 3)).all()
    rep = Objective().report(broken, truth, R)
    assert rep["n_failed"] == 1 and rep["n_total"] == 40
    assert abs(rep["value"] - Objective()(np.delete(broken, 3, axis=0),
                                          np.delete(truth, 3, axis=0), R)) < 1e-12
    # and reduce_galaxies returns NaN rather than inventing a number
    assert np.isnan(reduce_galaxies(np.full(5, np.nan), "mean"))

    # 7. epoch weighting: uniform weights reproduce the mean; a one-hot vector
    #    picks out that epoch alone
    assert abs(Objective(epoch="weights", epoch_weights=(1, 1, 1, 1))(model, truth, R)
               - Objective()(model, truth, R)) < 1e-12
    one_hot = Objective(epoch="weights", epoch_weights=(0, 0, 1, 0))
    only_k2 = Objective()(model[:, 2:3], truth[:, 2:3], R)
    assert abs(one_hot(model, truth, R) - only_k2) < 1e-12

    # 8. round-trips, naming, and validation
    o = Objective(quantity="density", residual="log")
    assert Objective.from_dict(o.to_dict()) == o
    assert o.name == "q=density|res=log|rad=rms|w=uniform|gal=mean"
    assert Objective.from_dict(dict(quantity="cog", residual="abs",
                                    radial="rms", weight="inner",
                                    galaxy="median")).residual == "abs"
    ew = Objective(epoch="weights", epoch_weights=(1, 2))
    assert Objective.from_dict(ew.to_dict()) == ew
    assert o.replace(residual="frac").residual == "frac"
    for bad_kw in (dict(quantity="sigma"), dict(residual="rel"),
                   dict(radial="mean"), dict(weight="outer"),
                   dict(galaxy="best"), dict(epoch="weights"),
                   dict(epoch_weights=(1, 1)),
                   dict(epoch="weights", epoch_weights=(0, 0))):
        try:
            Objective(**bad_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad_kw} should have raised")

    # 9. the default really is the production objective, written out longhand
    longhand = np.mean([np.mean([np.sqrt(np.mean(((model[i, k] - truth[i, k])
                                                  / truth[i, k]) ** 2))
                                 for k in range(4)]) for i in range(40)])
    assert abs(Objective()(model, truth, R) - longhand) < 1e-12, longhand

    print(f"objective self-check OK  default={Objective().name}\n"
          f"  default loss on a 5% perturbed model: "
          f"{Objective()(model, truth, R):.5f} (longhand {longhand:.5f})\n"
          f"  radial aggregation max {vals['max']:.5f} >= rms {vals['rms']:.5f}"
          f" >= absmean {vals['absmean']:.5f}")
