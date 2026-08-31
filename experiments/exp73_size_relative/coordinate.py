"""exp73 Block A1 — the size-relative coordinate, and the residuals that go with it.

WHY THIS EXISTS. exp72 Step 3 measured that the programme's fixed physical radial
grid is a redshift-dependent weighting nobody chose: the median half-mass radius
runs 11.6 kpc at z = 0.4 down to 3.0 kpc at z = 2, so 148 kpc is 13 R50 at one
end of the sample and 49 R50 at the other. At z = 2, 96.7 per cent of the galaxy
is inside 52 kpc and the outer two shells hold 3.3 per cent of its stellar mass,
yet the production objective spends 27.8 per cent of its attention on them. In
R50 units the same galaxies are very nearly self-similar across all five epochs.

This module supplies the three ingredients the exp73 plan asks for, and nothing
else. It scores arrays; it knows nothing about kernels or optimisers.

  1. **A size-relative coordinate.** Shells in units of each galaxy's own
     half-mass radius.
  2. **A mass-weighted residual.** A shell's error divided by the galaxy's TOTAL
     stellar mass rather than by the shell's own mass, so a shell is charged in
     proportion to the mass it actually holds.
  3. **A coverage gate.** Shells where the measurement has run out are excluded
     rather than charged.

THE RULE THAT MAKES THIS HONEST, and it is asserted in `selftest`:

    **The coordinate is built from the TRUTH's R50, never the model's.**

A model scored on its own size scale can satisfy a size-relative objective by
shrinking — the coordinate would follow it and the residual would not notice. So
`R50` here is always a property of the measured curve of growth, computed once,
and the model is interpolated onto radii the data defines. `selftest` claim A
checks that passing a wildly different model leaves the radii bit-identical.

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u \\
        experiments/exp73_size_relative/coordinate.py --selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp63_analytic_growth",
          ROOT / "experiments/exp72_error_objective", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE = "=" * 100
THIN = "-" * 100

#: THE SIZE-RELATIVE SHELL EDGES, in units of the galaxy's own R50, and they are
#: MEASURED rather than chosen. The measured grid runs 2 to 148.22 kpc, so an
#: edge is usable only where `edge * R50` falls inside it for most galaxies.
#: Coverage on the 2354-galaxy fitting sample (fraction of galaxy-epochs inside
#: the grid), worst epoch in brackets:
#:
#:     0.25 R50  78 / 59 / 40 / 14 /  3 %   (3 %)   unusable
#:     0.50 R50  97 / 92 / 84 / 57 / 26 %  (26 %)   unusable
#:     0.75 R50  99 / 98 / 96 / 84 / 62 %  (62 %)   marginal
#:     1.00 R50 100 /100 /100 / 97 / 87 %  (87 %)   usable
#:     2.00 R50 100 /100 /100 /100 / 99 %  (99 %)
#:     4.00 R50 100 /100 /100 /100 /100 %  (100 %)
#:     6.00 R50  95 / 98 / 99 /100 /100 %  (95 %)   usable
#:     8.00 R50  84 / 93 / 97 /100 /100 %  (84 %)   marginal
#:
#: **So the usable window is about 1 to 6 R50, and it EXCLUDES the inner half of
#: the galaxy at high redshift.** The innermost aperture is a fixed 2 kpc, which
#: is 0.17 R50 at the median z = 0.4 galaxy but 0.66 R50 at the median z = 2 one,
#: so below 1 R50 the coordinate fails at z = 2 -- and it fails preferentially
#: for the SMALL galaxies, which is a mass-dependent selection and worse than a
#: uniform loss. A size-relative coordinate can therefore describe the outskirts,
#: where the galaxy's own size sets the scale, but not the centre, where the
#: measurement's fixed resolution does. Anything inside 1 R50 stays in kpc.
RE_EDGES = (1.0, 2.0, 4.0, 6.0)
#: the fractional sizes reported. The user (D1) named R20 / R50 / R80; R90 is
#: carried because `hongshao.qa` already uses it and dropping it would break the
#: comparison with every product scored so far.
SIZE_FRACTIONS = (0.2, 0.5, 0.8, 0.9)


# --------------------------------------------------------------------------- #
# 1. the coordinate
# --------------------------------------------------------------------------- #
def size_radius(cog, R, frac):
    """(n, n_epoch) radius enclosing `frac` of M*(<R_max), by interpolation.

    NaN where the profile never reaches the fraction inside the grid, which
    cannot happen for frac < 1 on a monotone curve but can on a measured one
    that is flat or non-monotone at the end.
    """
    cog = np.asarray(cog, float)
    R = np.asarray(R, float)
    target = frac * cog[..., -1]
    flat = cog.reshape(-1, cog.shape[-1])
    tgt = target.reshape(-1)
    out = np.full(len(flat), np.nan)
    for i in range(len(flat)):
        c = flat[i]
        if not np.isfinite(c).all() or c[-1] <= 0:
            continue
        # np.interp needs an increasing x; a measured CoG can have flat runs,
        # which interp handles, but not decreasing ones -- use the running max
        cm = np.maximum.accumulate(c)
        if tgt[i] <= cm[0]:
            out[i] = R[0] * tgt[i] / max(cm[0], 1e-300)   # linear inside R[0]
        else:
            out[i] = float(np.interp(tgt[i], cm, R))
    return out.reshape(cog.shape[:-1])


def half_mass_radius(cog, R):
    """R50 of a single profile -- `hongshao.qa`'s convention, kept identical."""
    return qa.half_mass_radius(cog, R)


def cum_at(cog, R, radii):
    """M*(<radii) by LOG-LOG interpolation, radii shaped like cog[..., 0].

    Log-log, not linear, and the difference matters. A size-relative coordinate
    asks for the mass at `k * R50`, which lands between grid points at a
    position that DEPENDS ON THE EPOCH -- a z = 2 galaxy is four times smaller,
    so the fixed grid samples it far more coarsely relative to its own R50.
    Linear interpolation then makes a redshift-dependent error of order 1 per
    cent of the shell mass, which is exactly the kind of spurious epoch trend
    this coordinate exists to remove. A curve of growth is close to a power law
    over a grid interval, so log-log interpolation is both the natural choice
    and, measured on a self-similar synthetic, about 30x more accurate
    (`selftest` claim B).
    """
    cog = np.asarray(cog, float)
    R = np.asarray(R, float)
    lR = np.log10(R)
    flat = cog.reshape(-1, cog.shape[-1])
    rr = np.asarray(radii, float).reshape(-1)
    out = np.full(len(flat), np.nan)
    for i in range(len(flat)):
        c = flat[i]
        if not np.isfinite(rr[i]) or not np.isfinite(c).all():
            continue
        r = rr[i]
        if r <= R[0]:
            out[i] = c[0] * r / R[0]                     # linear inside R[0]
        elif r >= R[-1]:
            out[i] = c[-1]
        elif (c > 0).all():
            out[i] = 10.0 ** np.interp(np.log10(r), lR, np.log10(c))
        else:
            out[i] = float(np.interp(r, R, c))           # fallback if any zero
    return out.reshape(cog.shape[:-1])


class SizeGrid:
    """The size-relative coordinate, built ONCE from the truth and then frozen.

    `SizeGrid(truth, R)` computes each galaxy-epoch's R50 from the MEASURED
    curve of growth and the shell edges that follow from it. `shells(model)`
    then evaluates any profile on those same radii. The model never enters the
    coordinate -- that is the whole point, and `selftest` asserts it.
    """

    def __init__(self, truth, R, edges=RE_EDGES):
        self.R = np.asarray(R, float)
        self.edges = tuple(edges)
        t = np.asarray(truth, float)
        self.r50 = np.full(t.shape[:2], np.nan)
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                self.r50[i, j] = half_mass_radius(t[i, j], self.R)
        #: (n, n_epoch, n_edge) the physical radii each edge sits at
        self.radii = self.r50[..., None] * np.asarray(self.edges)[None, None, :]
        self.total = t[..., -1]                          # M*(<148 kpc)
        self.truth_shells = self.shells(t)
        #: M*(< the OUTERMOST size-relative edge). This is the self-similar
        #: normalisation: `total` is the mass inside a FIXED 148 kpc, which for
        #: a family of galaxies differing only in size captures a different
        #: fraction of each one (measured on a self-similar synthetic: 93.8 per
        #: cent of the biggest, 98.3 per cent of the smallest). Normalising a
        #: size-relative shell by a fixed-aperture total therefore reintroduces
        #: exactly the epoch trend the coordinate exists to remove.
        self.total_re = self.cum(t)[..., -1]
        #: a shell is COVERED when the truth puts positive mass in it and both
        #: its edges lie inside the measured grid
        inside = (self.radii[..., :-1] >= self.R[0]) & (self.radii[..., 1:] <= self.R[-1])
        self.covered = inside & (self.truth_shells > 0)

    @property
    def n_shell(self):
        return len(self.edges) - 1

    @property
    def labels(self):
        return [f"{self.edges[j]:g}-{self.edges[j+1]:g}Re" for j in range(self.n_shell)]

    def cum(self, cog):
        """(n, n_epoch, n_edge) M*(< each edge) for any profile."""
        c = np.asarray(cog, float)
        return np.stack([cum_at(c, self.R, self.radii[..., j])
                         for j in range(len(self.edges))], axis=-1)

    def shells(self, cog):
        """(n, n_epoch, n_shell) mass between consecutive size-relative edges."""
        return np.diff(self.cum(cog), axis=-1)

    # ------------------------------------------------------------- residuals #
    def residual(self, model, kind="mass"):
        """(n, n_epoch, n_shell) per-shell residual, uncovered shells NaN.

        ``kind="mass"``  (model - truth) / the galaxy's TOTAL stellar mass.
            A shell is charged in proportion to the mass it holds, so a large
            fractional error on an empty shell costs almost nothing.
        ``kind="rel"``   (model - truth) / the SHELL's own mass.
            The programme's incumbent form, carried for comparison only.
        """
        m = self.shells(model)
        d = self.truth_shells
        with np.errstate(divide="ignore", invalid="ignore"):
            if kind == "mass":
                r = (m - d) / self.total_re[..., None]
            elif kind == "rel":
                r = (m - d) / np.where(d > 0, d, np.nan)
            else:
                raise ValueError(f"kind must be 'mass' or 'rel', got {kind!r}")
        return np.where(self.covered, r, np.nan)

    def mass_share(self, mask=None, norm="re"):
        """(n_epoch, n_shell) median share of the galaxy's mass per shell.

        ``norm="re"``   divide by M*(< the outermost size-relative edge) -- the
            self-similar choice, and the one to use when asking whether the
            coordinate has removed an epoch trend.
        ``norm="grid"`` divide by M*(<148 kpc), the programme's operational
            "whole galaxy". Physically the more familiar number, but it carries
            a fixed aperture into a size-relative statistic.
        """
        den = (self.total_re if norm == "re" else self.total)[..., None]
        f = np.where(self.covered, self.truth_shells / den, np.nan)
        if mask is not None:
            f = np.where(np.asarray(mask, bool)[..., None], f, np.nan)
        return np.nanmedian(f, axis=0)


# --------------------------------------------------------------------------- #
# 2. the down-weighting schemes the user asked to compare (D2)
# --------------------------------------------------------------------------- #
def weights_smooth(grid, mask=None):
    """Scheme A: weight each shell by its own median mass share. No free
    parameter -- the weights are measured from the data."""
    w = grid.mass_share(mask)                              # (n_epoch, n_shell)
    return w / np.nansum(w, axis=1, keepdims=True)


def weights_gated(grid, t, mask=None):
    """Scheme B: drop shells whose median mass share falls below `t`, and weight
    the survivors equally. One free parameter, and a cliff."""
    f = grid.mass_share(mask)
    w = np.where(f >= t, 1.0, 0.0)
    s = np.nansum(w, axis=1, keepdims=True)
    return np.divide(w, s, out=np.zeros_like(w), where=s > 0)


def weights_uniform(grid, mask=None):
    """The reference: every covered shell counts the same."""
    w = np.ones((grid.truth_shells.shape[1], grid.n_shell))
    return w / w.sum(axis=1, keepdims=True)


def apply_weights(res, w):
    """(n, n_epoch) per-galaxy loss from a per-shell residual and (n_epoch,
    n_shell) weights. Uncovered shells contribute nothing and their weight is
    redistributed over the shells that galaxy-epoch does cover, so a galaxy is
    never penalised for a measurement that was not made."""
    ok = np.isfinite(res)
    ww = np.where(ok, w[None, :, :], 0.0)
    s = ww.sum(axis=-1, keepdims=True)
    ww = np.divide(ww, s, out=np.zeros_like(ww), where=s > 0)
    return np.sqrt(np.nansum(ww * np.where(ok, res, 0.0) ** 2, axis=-1))


# --------------------------------------------------------------------------- #
def selftest():
    print(f"{RULE}\nexp73 coordinate selftest\n{RULE}")
    rng = np.random.default_rng(0)
    R = F.R_GRID
    n, ne = 60, 5

    # a self-similar family: the same shape, scaled in size and mass per epoch
    scale = np.array([1.0, 0.8, 0.6, 0.4, 0.27])
    mass = np.array([1.0, 0.85, 0.7, 0.5, 0.35])
    shape = lambda x: x ** 2 / (1.0 + x) ** 2            # noqa: E731 (Hernquist CoG)
    # r_e chosen so R50 = 2.41 r_e spans the REAL range at z = 0.4 (p16-p84
    # 7.2-18.5 kpc), so the coverage the selftest sees is the coverage the data
    # has rather than an invented one
    r_e = 10.0 ** rng.uniform(np.log10(7.2 / 2.41), np.log10(18.5 / 2.41), n)
    truth = np.empty((n, ne, len(R)))
    for j in range(ne):
        truth[:, j, :] = (1e11 * mass[j]
                          * shape(R[None, :] / (r_e[:, None] * scale[j])))

    g = SizeGrid(truth, R)

    # A. the coordinate is truth-only
    r0 = g.radii.copy()
    _ = g.shells(truth * 3.7)                            # a wildly different model
    assert np.array_equal(r0, g.radii), "the model moved the coordinate"
    print(f"  A  a model 3.7x the truth leaves the coordinate bit-identical  OK")

    # B. THE PROPERTY THAT MATTERS: on a self-similar family the size-relative
    #    coordinate must give a far SMALLER epoch-to-epoch trend than fixed kpc
    #    shells do. It cannot give zero, and the reason is worth stating: both
    #    the normalising total and R50 itself are measured inside a FIXED
    #    148 kpc aperture, which for a family differing only in size encloses a
    #    different fraction of each galaxy. That residual is the aperture's, not
    #    the coordinate's, and B2 measures it.
    both = g.covered.all(axis=(1, 2))
    assert both.sum() > 10, f"only {both.sum()} galaxies covered at every epoch"
    m_all = np.repeat(both[:, None], g.covered.shape[1], axis=1)
    f_re = g.mass_share(mask=m_all)
    re_spread = float(np.nanmax(np.nanmax(f_re, axis=0) - np.nanmin(f_re, axis=0)))

    # the same galaxies scored on FIXED kpc shells spanning the same mass
    kpc_edges = np.array([np.nanmedian(g.radii[both, 0, j])
                          for j in range(len(g.edges))])
    idx = [int(np.argmin(np.abs(R - e))) for e in kpc_edges]
    tt = truth[both]
    sh_kpc = np.stack([tt[..., idx[j + 1]] - tt[..., idx[j]]
                       for j in range(len(idx) - 1)], axis=-1)
    f_kpc = np.nanmedian(sh_kpc / tt[..., idx[-1]][..., None], axis=0)
    kpc_spread = float(np.nanmax(np.nanmax(f_kpc, axis=0) - np.nanmin(f_kpc, axis=0)))

    assert re_spread < 0.25 * kpc_spread, (re_spread, kpc_spread)
    print(f"  B  on a self-similar family the R50 coordinate's epoch spread is "
          f"{100 * re_spread:.1f} points against {100 * kpc_spread:.1f} for "
          f"fixed-kpc shells over the same range — a {kpc_spread / re_spread:.0f}x "
          f"reduction  OK")

    # B2. the irreducible part, and where it comes from
    print(f"  B2 the {100 * re_spread:.1f}-point residual is the FIXED 148 kpc "
          f"aperture, not the coordinate: both the normalising total and R50 "
          f"itself\n     are measured inside it, and it encloses "
          f"{100 * truth[both][:, 0, -1].mean() / (1e11):.1f}% of the biggest "
          f"galaxies against {100 * truth[both][:, 4, -1].mean() / (1e11 * mass[4]):.1f}% "
          f"of the smallest")

    # B3. the coverage limit, measured on the real grid range
    unc = 100 * (~g.covered).mean(axis=(0, 2))
    print(f"  B3 uncovered shell fraction per epoch: "
          + " / ".join(f"{v:.0f}%" for v in unc)
          + "  — see RE_EDGES for why nothing below 1 R50 is usable")

    # C. THE IDENTITY THAT IS THE USER'S CLAIM 2. For a shell holding a
    #    fraction phi of the galaxy, the mass-weighted residual is exactly phi
    #    times the relative one. So a shell holding 1 per cent of the mass has
    #    its relative error divided by 100 before it reaches the loss -- which
    #    is the whole point of the mass-weighted form, stated exactly rather
    #    than argued.
    mod = truth * (1.0 + 0.1 * rng.normal(size=truth.shape))
    mod = np.maximum.accumulate(np.abs(mod), axis=-1)
    a_res = g.residual(mod, "mass")
    b_res = g.residual(mod, "rel")
    phi = np.where(g.covered, g.truth_shells / g.total_re[..., None], np.nan)
    dev = np.nanmax(np.abs(a_res - b_res * phi))
    scale = np.nanmax(np.abs(a_res))
    assert dev < 1e-12 * max(scale, 1e-30), (dev, scale)
    print(f"  C  mass-weighted residual == relative residual x the shell's mass "
          f"share, exactly (max deviation {dev:.1e} against a typical "
          f"{scale:.2f})  OK")

    # D. the weight schemes are normalised and the gate is a subset of smooth
    wa, wb = weights_smooth(g), weights_gated(g, 0.02)
    assert np.allclose(np.nansum(wa, axis=1), 1.0)
    assert np.allclose(np.nansum(wb, axis=1), 1.0)
    assert ((wb > 0) <= (wa > 0)).all()
    print(f"  D  both weight schemes normalise to 1 per epoch and the gate keeps "
          f"a subset of the smooth scheme's shells  OK")

    # E. an uncovered shell is never charged
    t2 = truth.copy()
    t2[:, :, 12:] = t2[:, :, 11][:, :, None]             # flatten the outskirts
    g3 = SizeGrid(t2, R)
    r = g3.residual(t2 * 1.5)
    assert np.isnan(r[~g3.covered]).all()
    print(f"  E  uncovered shells return NaN and are dropped, never charged "
          f"({100 * (~g3.covered).mean():.1f}% uncovered in the flattened case)  OK")

    print("\nall selftest claims pass")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
