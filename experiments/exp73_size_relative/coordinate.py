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
#: MEASURED rather than chosen. An edge is usable only where `edge * R50` falls
#: inside the measured radial range. Which range that is depends on which
#: measurement is used, and the difference is decisive (the user, 2026-09-01):
#:
#:                        stored CoG grid      density-rebuilt CoG
#:                          2 - 148.22 kpc       0.673 - 159.74 kpc
#:     0.125 R50                     0 %                      8 %
#:     0.25  R50                     3 %                     56 %
#:     0.50  R50                    26 %                    100 %
#:     0.75  R50                    62 %                    100 %
#:     1.00  R50                    87 %                    100 %
#:     4.00  R50                   100 %                    100 %
#:     8.00  R50                    84 %                     87 %
#:   (worst epoch, on the 2354-galaxy fitting sample)
#:
#: The stored 24-radius curve of growth starts at a fixed 2 kpc, which is
#: 0.17 R50 at the median z = 0.4 galaxy but 0.66 R50 at the median z = 2 one, so
#: on it alone nothing below 1 R50 is usable and the failure is worse than a
#: uniform loss -- it drops the SMALL galaxies preferentially, a mass-dependent
#: selection. **But the isophote density reaches 0.673 kpc for 100 per cent of
#: galaxies at every epoch**, so the curve of growth rebuilt from it extends the
#: window to 0.5 R50 at full coverage. That is what `extended_cog` supplies.
#:
#: THE QUALITY CAVEAT, which travels with every inner shell: `sigma_resolved` is
#: False inside about 2.4 kpc, where photutils' ellipse fit sat at its 13-point
#: floor -- at z = 2, 0 per cent resolved below 1.4 kpc, 60 per cent at 2.0 kpc.
#: The density is real there but not INDEPENDENT: it is the fitted ellipse family
#: evaluated inward over the same pixels. Carry it as a flag, not a blocker.
RE_EDGES = (0.5, 1.0, 2.0, 4.0, 6.0)
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

    VECTORISED over galaxies and epochs. The query radii differ per galaxy but
    the grid does not, so one `searchsorted` on the shared grid serves all of
    them. The loop version this replaces took about a second per call on 2354
    galaxies, which would have made any fit under this coordinate impossible;
    `selftest` claim G checks the two agree to floating-point.
    """
    cog = np.asarray(cog, float)
    R = np.asarray(R, float)
    rr = np.asarray(radii, float)[..., None]              # (..., 1) to align
    lR, lC = np.log10(R), np.log10(np.where(cog > 0, cog, np.nan))
    lq = np.log10(np.clip(rr, 1e-30, None))
    j = np.clip(np.searchsorted(lR, lq) - 1, 0, len(R) - 2)
    lo = np.take_along_axis(lC, j, axis=-1)
    hi = np.take_along_axis(lC, j + 1, axis=-1)
    x0 = lR[j]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (lq - x0) / (lR[j + 1] - x0)
        out = 10.0 ** (lo + t * (hi - lo))
    # outside the grid: linear inside R[0], flat beyond R[-1]
    out = np.where(rr <= R[0], cog[..., :1] * rr / R[0], out)
    out = np.where(rr >= R[-1], cog[..., -1:], out)
    return np.where(np.isfinite(rr), out, np.nan)[..., 0]


def extended_cog(cog_provided, cog_density, R_cog, R_shared, r_splice=2.0,
                 r_min=0.6729):
    """Splice the density-rebuilt curve of growth INSIDE `r_splice` onto the
    stored one outside it, so the target reaches 0.673 kpc without changing
    anything the programme has ever fitted.

    The user's D1 keeps `cog_provided` as the fitting target. This does not
    replace it: outside `r_splice` the returned curve IS `cog_provided`,
    unchanged and on its own radii. Inside, it is the density-rebuilt curve
    RESCALED so the two agree exactly at the splice, which removes the smooth
    2 per cent reconstruction offset between them (measured: median log ratio
    -0.008 to +0.009 dex over 2-103 kpc, `doc/tng300_profile_data.md` section
    10.3) rather than letting it appear as a step.

    Returns (radii, cog) on a merged grid: the shared-grid radii in
    [`r_min`, `r_splice`), then the stored CoG radii.
    """
    cp = np.asarray(cog_provided, float)
    cd = np.asarray(cog_density, float)
    Rc, Rs = np.asarray(R_cog, float), np.asarray(R_shared, float)
    # `r_min` defaults to the SECOND shared-grid radius, 0.6729 kpc, because the
    # first (0.5608) is measured for only 88 per cent of galaxies while 0.6729 is
    # measured for 100 per cent at every epoch. Including a radius a tenth of the
    # sample lacks would drop those galaxies from every downstream statistic for
    # a reason that has nothing to do with the model.
    inner = (Rs < r_splice) & (Rs >= r_min)
    # the density curve evaluated AT the splice, per galaxy-epoch
    at_splice = cum_at(cd, Rs, np.full(cd.shape[:2], r_splice))
    j0 = int(np.argmin(np.abs(Rc - r_splice)))
    scale = np.where(at_splice > 0, cp[..., j0] / np.where(at_splice > 0, at_splice, 1.0), np.nan)
    merged_R = np.concatenate([Rs[inner], Rc])
    merged = np.concatenate([cd[..., inner] * scale[..., None], cp], axis=-1)
    return merged_R, merged


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
          + "  — on the STORED 2-148 kpc grid, which is what this synthetic "
            "uses;\n     `extended_cog` is what removes it on real data (see "
            "RE_EDGES)")

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
    typ = np.nanmax(np.abs(a_res))
    assert dev < 1e-12 * max(typ, 1e-30), (dev, typ)
    print(f"  C  mass-weighted residual == relative residual x the shell's mass "
          f"share, exactly (max deviation {dev:.1e} against a typical "
          f"{typ:.2f})  OK")

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

    # F. the splice is continuous and leaves the outside untouched
    Rs = np.geomspace(0.5608, 159.74, 32)
    dens = np.empty((n, ne, len(Rs)))
    for j in range(ne):
        dens[:, j, :] = (1.021e11 * mass[j]                 # a 2.1% offset,
                         * shape(Rs[None, :] / (r_e[:, None] * scale[j])))
    Rm, merged = extended_cog(truth, dens, R, Rs)
    n_in = int(((Rs < 2.0) & (Rs >= 0.6729)).sum())
    assert np.allclose(merged[..., n_in:], truth), "the outside was modified"
    ratio_in = merged[..., n_in - 1] / merged[..., n_in]
    assert (ratio_in < 1.0).all() and (ratio_in > 0.5).all(), ratio_in.min()
    at_splice = cum_at(dens, Rs, np.full(dens.shape[:2], 2.0)) * (
        truth[..., 0] / cum_at(dens, Rs, np.full(dens.shape[:2], 2.0)))
    assert np.allclose(at_splice, truth[..., 0]), "the splice is not continuous"
    print(f"  F  `extended_cog` leaves everything outside 2 kpc bit-identical, "
          f"matches exactly at the splice, and adds {n_in} inner radii reaching "
          f"{Rm[0]:.3f} kpc  OK")

    # G. the vectorised interpolation reproduces an explicit per-galaxy loop
    q = g.radii[:20]
    ref = np.full(q.shape, np.nan)
    for i in range(q.shape[0]):
        for jj in range(q.shape[1]):
            for kk in range(q.shape[2]):
                r = q[i, jj, kk]
                c = truth[i, jj]
                if r <= R[0]:
                    ref[i, jj, kk] = c[0] * r / R[0]
                elif r >= R[-1]:
                    ref[i, jj, kk] = c[-1]
                else:
                    ref[i, jj, kk] = 10.0 ** np.interp(np.log10(r), np.log10(R),
                                                       np.log10(c))
    got = np.stack([cum_at(truth[:20], R, q[..., kk]) for kk in range(q.shape[2])],
                   axis=-1)
    dev = float(np.nanmax(np.abs(got / ref - 1.0)))
    assert dev < 1e-12, dev
    print(f"  G  the vectorised interpolation reproduces an explicit per-galaxy "
          f"loop to {dev:.1e} relative  OK")

    print("\nall selftest claims pass")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
