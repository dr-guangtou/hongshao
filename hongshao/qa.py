"""Standardized tiered QA for multi-epoch CoG predictions (graduated from exp29
``mass_qa`` + exp30 ``profile_qa``; exp31).

HongShao's forward models are scored on a ladder, and every approach must be
scored identically. One entry point, ``evaluate(model_cogs, data_cogs, R,
anchor_z)``, reports three tiers:

1. **Aperture masses** — M*(<R) at fixed kpc and R_half-relative radii: the
   basic-goal scoreboard (per-epoch median bias + dex scatter).
2. **Annulus + outskirt masses** — M* in [10,30], [30,50], [50,100], [100,150]
   kpc shells (+ Re analogs) and M*(>R) envelopes: the sensitive shape tier.
   Plus the **observational planes**: joint 2-D distributions (e.g. M*(<30 kpc)
   vs M*[50,100 kpc]) — the predicted population must reproduce the truth's
   relation (slope/scatter), because that plane is what observations use.
2c. **Cross-epoch growth planes** (``growth_planes``) — the SAME object at two
   epochs (R_half now, M_tot now). Every tier above is a single-epoch
   statistic, so a model with perfect per-epoch marginals can still evolve
   individual galaxies incoherently and score flawlessly; measured instance
   (exp44): a stochastic layer holding drawn galaxies at rank correlation
   +0.97 between their z=0.4 and z=2.0 sizes where the truth sits at +0.33.
   No per-epoch metric can ever see that, which is why this tier exists.
2d. **The mass-size plane** (``size_planes``) — log M* vs log R_half per
   epoch. ``PLANES`` pairs measured MASSES with each other, so the model's
   own SIZE distribution was never compared to the truth's; this closes
   that. Same self-consistency exception as 2c (model sizes from the MODEL).
3. **Profile** — worst-radius max|rel| quoted over ALL radii AND R>5 kpc (the
   inner 2-5 kpc is marginally resolved; exp07), with two visual products:
   median CoG by stellar-mass tercile with residual profiles, and a
   best/worst-case gallery with the worst radius marked.

Conventions: CoGs are linear masses, shape (n, nz, nr); R_half is measured on
the TRUTH CoG and shared by model and truth (isolates mass error from size
error); relative errors are (model-truth)/truth; dex scatter is the std of
log10(model/truth) over galaxies. ONE deliberate exception: tiers 2c and 2d
take the model's R_half from the MODEL CoG, because both ask about the
model's OWN sizes — importing truth sizes would erase the quantity under
test (the exp41 paired-vs-population lesson).

Use: ``from hongshao import qa; res = qa.evaluate(model, truth, R, ANCHOR_Z,
name="...", figdir=...)`` — returns a dict of all per-galaxy measurements and
summary tables for programmatic comparison (the exp31 scoreboard pattern).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from hongshao.profile_emulator import density_from_cog
from hongshao.plotting import set_style, save_fig

# --- the standard radial bins (edit here to change the standard) --------------
KPC_APER = [10.0, 30.0, 50.0, 100.0]          # M*(<R) [kpc]
KPC_ANN = [(10.0, 30.0), (30.0, 50.0), (50.0, 100.0), (100.0, 150.0)]
KPC_ENV = [50.0, 100.0]                        # M*(>R) [kpc]
RE_APER = [1.0, 2.0, 4.0]                      # M*(<k R_half)
RE_ANN = [(1.0, 2.0), (2.0, 4.0)]
RE_ENV = [2.0, 4.0]                            # M*(>k R_half)
PLANES = [("kpc:M(<30)", "kpc:M(30-50)"), ("kpc:M(<30)", "kpc:M(50-100)"),
          ("Re:M(<2Re)", "Re:M(2-4Re)")]
GROWTH_QUANTITIES = ("R_half", "Mtot")         # tier 2c, cross-epoch
SIZE_FRACTIONS = (0.2, 0.5, 0.8, 0.9)          # tier 2d: R20 / R50 / R80 / R90
# tier 2d GATE (the user's D1, 2026-09-01): the R20/R50/R80 fractional sizes'
# DISTRIBUTIONS at fixed stellar mass and at fixed halo mass are a first-class
# validation target, not a figure to be read. Two thresholds, both stated as
# choices with the measurement that motivated them:
#   SIZE_GATE_OFFSET  |median dlog R| in dex. 0.05 dex is 12% in size. exp63's
#                     joint mean holds R50 to 0.02 dex at every epoch and the
#                     exp72 chi-square refit missed by 0.10 at z=2, so 0.05 sits
#                     between a fit that has the mass-size relation and one that
#                     has lost it.
#   SIZE_GATE_WIDTH   the model's scatter at fixed mass over the truth's. exp61
#                     measured harder fitting NARROWING the predicted population
#                     to 0.46 of the truth's width where 0.54 was honest
#                     (memory `fitting-harder-narrows-the-population`); a 20%
#                     tolerance catches that regime and not less.
SIZE_GATE_OFFSET, SIZE_GATE_WIDTH = 0.05, 0.20
# tier 2e: the masses whose POPULATION DISTRIBUTION is reported. A subset,
# not every key -- one inner aperture, one mid and one outer annulus, and
# the two Re-relative quantities -- so the table and figure stay readable.
CDF_KEYS = ("kpc:M(<10)", "kpc:M(<30)", "kpc:M(30-50)", "kpc:M(50-100)",
            "Re:M(<2Re)", "Re:M(2-4Re)")
RMIN_KPC = 5.0                                 # inner 2-5 kpc marginally resolved


def enclosed_radius(cog, R, frac=0.5):
    """Radius enclosing ``frac`` of the CoG's total; log-log extrapolation
    below the grid for compact high-z galaxies whose radius < R[0].

    Sizes are APERTURE-relative: the total is the CoG's last point (148 kpc
    on the standard grid), so R80/R90 are 80/90% of the measured aperture
    mass, not of an extrapolated infinite total. They are always inside the
    grid by construction (the target is a fraction of the grid total), so
    unlike ``measure`` there is no clamping hazard here.

    Why more than one: R50 is set by where the INNER mass sits, R80/R90 by
    the outer profile, so their RELATIVE offsets separate an inner-mass
    defect (R50 biased more than R90) from a uniform size error (all three
    move together).

    Interpretation caveat, measured: the discrimination is real but WEAK in
    magnitude, because the CoG is nearly flat near R90 — a small fractional
    mass error there produces a large radius shift. A synthetic inner-only
    deficit that biases R50 by +0.048 dex still drags R90 by +0.042. So
    read the ORDERING (R50 > R80 > R90 implicates the inner region), not
    the size of the gap, and never conclude "the outskirts are wrong"
    from an R90 offset alone.
    """
    target = float(frac) * cog[-1]
    if target <= cog[0]:
        sl = (np.log(cog[1]) - np.log(cog[0])) / (np.log(R[1]) - np.log(R[0]))
        return float(max(R[0] * np.exp((np.log(target) - np.log(cog[0])) / max(sl, 1e-3)), 0.3))
    return float(np.interp(target, cog, R))


def half_mass_radius(cog, R):
    """Half-mass radius (``enclosed_radius`` at frac=0.5) — the shared size
    convention for the Re-relative bins."""
    return enclosed_radius(cog, R, 0.5)


def measure(cog, R, rhalf):
    """Standard aperture/annulus/envelope masses for one CoG, keyed '<set>:label'.

    A quantity whose radius leaves the measured grid (e.g. the 100-150 kpc
    annulus on the 148.2-kpc CoG grid, or 4 R_half for a large galaxy) is
    flagged NaN: ``np.interp`` would silently clamp to the last CoG point and
    report a wrong (usually ~zero) mass.
    """
    mtot = cog[-1]

    def cum(r):
        return float(np.interp(r, R, cog)) if r <= R[-1] else np.nan

    m = {}
    for a in KPC_APER:
        m[f"kpc:M(<{int(a)})"] = cum(a)
    for a, b in KPC_ANN:
        m[f"kpc:M({int(a)}-{int(b)})"] = cum(b) - cum(a)
    for a in KPC_ENV:
        m[f"kpc:M(>{int(a)})"] = mtot - cum(a)
    for k in RE_APER:
        m[f"Re:M(<{k:g}Re)"] = cum(k * rhalf)
    for a, b in RE_ANN:
        m[f"Re:M({a:g}-{b:g}Re)"] = cum(b * rhalf) - cum(a * rhalf)
    for k in RE_ENV:
        m[f"Re:M(>{k:g}Re)"] = mtot - cum(k * rhalf)
    return m


def measure_all(model_cogs, data_cogs, R):
    """Truth/model mass arrays (n, nz) per quantity; R_half from TRUTH (shared)."""
    n, nz = data_cogs.shape[:2]
    keys = list(measure(data_cogs[0, 0], R, 1.0).keys())
    truth = {k: np.zeros((n, nz)) for k in keys}
    model = {k: np.zeros((n, nz)) for k in keys}
    rhalf = np.zeros((n, nz))
    for i in range(n):
        for j in range(nz):
            rh = half_mass_radius(data_cogs[i, j], R)
            rhalf[i, j] = rh
            md = measure(data_cogs[i, j], R, rh)
            mm = measure(model_cogs[i, j], R, rh)
            for k in keys:
                truth[k][i, j] = md[k]
                model[k][i, j] = mm[k]
    return truth, model, rhalf, keys


def relerr(model, truth):
    return (model - truth) / np.clip(truth, 1.0, None)


def dex_scatter(model, truth, axis=0):
    """Std over galaxies of log10(model/truth); NaN-safe, floors masses at 1 Msun."""
    d = np.log10(np.clip(model, 1.0, None)) - np.log10(np.clip(truth, 1.0, None))
    return np.nanstd(d, axis=axis)


def profile_maxrel(model_cogs, data_cogs, R, rmin=None):
    """Worst-radius |rel| per galaxy/epoch (n, nz), optionally over R > rmin."""
    m = R > (rmin if rmin is not None else 0.0)
    return np.abs((model_cogs[..., m] - data_cogs[..., m])
                  / data_cogs[..., m]).max(axis=-1)


def plane_stats(logx, logy):
    """OLS slope + vertical scatter + Spearman rho of a log-log relation."""
    from scipy.stats import spearmanr
    good = np.isfinite(logx) & np.isfinite(logy)
    if good.sum() < 3:
        return dict(slope=np.nan, scatter=np.nan, rho=np.nan)
    b = np.polyfit(logx[good], logy[good], 1)
    res = logy[good] - np.polyval(b, logx[good])
    return dict(slope=float(b[0]), scatter=float(np.std(res)),
                rho=float(spearmanr(logx[good], logy[good])[0]))


def energy_distance_2d(A, B):
    """Energy distance between two 2-D samples (Szekely & Rizzo): a true metric
    on distributions (0 iff equal), sensitive to location/spread/shape/correlation
    at once — unlike the 2-D K-S, it needs no per-case calibration and no kernel
    bandwidth. A (n,2), B (m,2); rows with non-finite entries are dropped."""
    A = A[np.isfinite(A).all(axis=1)]
    B = B[np.isfinite(B).all(axis=1)]
    if len(A) < 2 or len(B) < 2:
        return np.nan

    def mean_pdist(X, Y):
        d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
        return np.sqrt(np.clip(d2, 0.0, None)).mean()

    e2 = 2.0 * mean_pdist(A, B) - mean_pdist(A, A) - mean_pdist(B, B)
    return float(np.sqrt(max(e2, 0.0)))


def plane_energy(truth_xy, model_xy, n_split=8, seed=0):
    """dict(energy, floor, energy_ratio, energy_ratio_centered).

    Both samples are standardized by the TRUTH's per-axis std (same transform),
    so values are dimensionless and comparable across planes/epochs. The floor
    is the median energy distance between two random halves of the truth (pure
    sampling noise): ratio ~ 1 means the predicted population is statistically
    indistinguishable from the real one. The CENTERED ratio shifts each sample
    to its own per-axis median first, isolating shape/spread mismatch (missing
    population diversity) from location mismatch (bias, fixable by
    recalibrating the amplitude model)."""
    T = truth_xy[np.isfinite(truth_xy).all(axis=1)]
    M = model_xy[np.isfinite(model_xy).all(axis=1)]
    sd = T.std(axis=0) + 1e-12
    T, M = T / sd, M / sd
    e = energy_distance_2d(T, M)
    e_c = energy_distance_2d(T - np.median(T, axis=0), M - np.median(M, axis=0))
    rng = np.random.default_rng(seed)
    floors = []
    for _ in range(n_split):
        perm = rng.permutation(len(T))
        h = len(T) // 2
        floors.append(energy_distance_2d(T[perm[:h]], T[perm[h:]]))
    floor = max(float(np.median(floors)), 1e-12)
    return dict(energy=e, floor=floor, energy_ratio=e / floor,
                energy_ratio_centered=e_c / floor)


def cdf_distances(truth_v, model_v, n_split=8, seed=0):
    """KS and 1-Wasserstein between two 1-D samples, each against the
    truth's own split-half floor (the ``plane_energy`` convention).

    Two statistics because they fail differently: KS is the largest
    VERTICAL gap between the CDFs, so it is blind to how far apart they
    are once separated; W1 is the mean HORIZONTAL gap, in dex, so it
    reports magnitude but is insensitive to a narrow, tall discrepancy.
    A model can look fine on one and bad on the other.

    Interpretation caveat: both are normalized by the TRUTH's own
    split-half floor, so a quantity whose truth distribution is very
    narrow has a near-zero floor and an inflated, unstable ratio. That is
    a property of the target, not of the model — check the raw ``ks`` /
    ``w1`` before believing a large ratio on a tight quantity. (Seen in
    the self-check's self-similar synthetic, where M(<2Re) is identical
    for every galaxy by construction.)
    """
    t = np.asarray(truth_v, float)
    m = np.asarray(model_v, float)
    t, m = t[np.isfinite(t)], m[np.isfinite(m)]
    if len(t) < 8 or len(m) < 8:
        return dict(ks=np.nan, w1=np.nan, ks_floor=np.nan, w1_floor=np.nan,
                    ks_ratio=np.nan, w1_ratio=np.nan, n=len(t))

    def _pair(a, b):
        grid = np.union1d(a, b)
        fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
        fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
        ks = float(np.max(np.abs(fa - fb)))
        # W1 on equal-count quantiles: robust to unequal sample sizes
        q = (np.arange(1, min(len(a), len(b)) + 1) - 0.5) / min(len(a), len(b))
        w1 = float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))
        return ks, w1

    ks, w1 = _pair(t, m)
    rng = np.random.default_rng(seed)
    fk, fw = [], []
    for _ in range(n_split):
        perm = rng.permutation(len(t))
        h = len(t) // 2
        a, b = _pair(t[perm[:h]], t[perm[h:]])
        fk.append(a)
        fw.append(b)
    ks_f = max(float(np.median(fk)), 1e-12)
    w1_f = max(float(np.median(fw)), 1e-12)
    return dict(ks=ks, w1=w1, ks_floor=ks_f, w1_floor=w1_f,
                ks_ratio=ks / ks_f, w1_ratio=w1 / w1_f, n=len(t))


def mass_cdf_distance(model_cogs, data_cogs, R, quantities=None,
                      truth=None, model=None):
    """Tier 2e — the POPULATION DISTRIBUTION of every aperture/annulus
    mass, per epoch, as a CDF comparison. Returns {(quantity, j): dict}.

    Tiers 1-2 report a median bias and a dex scatter per quantity, which
    are two moments of a distribution that may differ in shape, tails or
    skew while matching both. This tier compares the distributions
    themselves.

    Like ``plane_energy`` and for the same reason, this is PAIRING-BLIND:
    it compares populations, so a model that emitted the truth's own
    galaxies in a shuffled order would score perfectly. It measures
    whether the right population is produced, never whether the right
    galaxy got the right profile — read it alongside the per-object
    tiers, never instead of them.
    """
    if truth is None or model is None:
        truth, model, _, _ = measure_all(model_cogs, data_cogs, R)
    keys = list(quantities) if quantities is not None else list(truth)
    nz = data_cogs.shape[1]
    out = {}
    for k in keys:
        for j in range(nz):
            tv = np.log10(np.clip(truth[k][:, j], 1.0, None))
            mv = np.log10(np.clip(model[k][:, j], 1.0, None))
            # a quantity that left the measured grid is NaN by convention
            # in ``measure``; the truth-floor clip would turn it into a
            # spike at 0, so drop those rows instead.
            ok_t = np.isfinite(truth[k][:, j]) & (truth[k][:, j] > 1.0)
            ok_m = np.isfinite(model[k][:, j]) & (model[k][:, j] > 1.0)
            out[(k, j)] = cdf_distances(tv[ok_t], mv[ok_m], seed=j)
    return out


def _safe_rhalf(cogs, R, frac=0.5):
    """(n, nz) enclosed radii at ``frac``, NaN where the CoG is unusable."""
    n, nz = cogs.shape[:2]
    out = np.full((n, nz), np.nan)
    for i in range(n):
        for j in range(nz):
            c = cogs[i, j]
            if np.isfinite(c).all() and c[-1] > 0 and np.all(np.diff(c) >= 0):
                out[i, j] = enclosed_radius(c, R, frac)
    return out


def growth_planes(model_cogs, data_cogs, R, ref=0,
                  quantities=GROWTH_QUANTITIES):
    """Tier 2c — CROSS-EPOCH planes: a quantity at the reference epoch vs at
    each other epoch, for the SAME object. Returns {(quantity, ref, j): stats}.

    Every plane in ``PLANES`` is a SINGLE-EPOCH statistic, so a model whose
    per-epoch marginals are all correct can still evolve individual galaxies
    incoherently and score perfectly. Measured instance (exp44 stage 3): the
    adopted stochastic layer held drawn galaxies at rank correlation +0.97
    between their z=0.4 and z=2.0 sizes where the truth sits at +0.33 — a
    large defect that was invisible to every existing tier, and that no
    per-epoch metric can ever see. This tier closes that blind spot.

    SELF-CONSISTENT BY DESIGN, deliberately unlike ``measure_all``: the
    model's R_half is taken from the MODEL CoG, not the truth's. A
    cross-epoch coherence statistic asks whether the model's own galaxies
    evolve like real ones, so importing truth sizes would erase the very
    quantity under test (the exp41 paired-vs-population lesson: paired
    apertures are right for a mean prediction, structurally wrong for a
    population/draw comparison).

    ``coherence_*`` is the Spearman rank correlation between the two epochs
    — the interpretable number ("how loyal is a galaxy to its rank"). The
    energy ratio scores the full 2-D cloud against the truth's split-half
    floor, as elsewhere.

    Caveat on ``Mtot``: it is degenerate for models normalized to a measured
    total at each epoch (the transport-kernel family) — those totals ARE the
    data, so the plane trivially matches. It is informative only for models
    that predict totals from features (the statistical emulator).
    """
    model_cogs, data_cogs = np.asarray(model_cogs), np.asarray(data_cogs)
    nz = data_cogs.shape[1]
    out = {}
    if nz < 2:
        return out
    cache = {}
    for qname in quantities:
        if qname == "Mtot":
            t_q, m_q = data_cogs[..., -1], model_cogs[..., -1]
        elif qname == "R_half":
            if "rh" not in cache:
                cache["rh"] = (_safe_rhalf(data_cogs, R),
                               _safe_rhalf(model_cogs, R))
            t_q, m_q = cache["rh"]
        else:
            raise ValueError(f"unknown growth quantity {qname!r}")
        t = np.log10(np.clip(t_q, 1e-30, None))
        m = np.log10(np.clip(m_q, 1e-30, None))
        for j in range(nz):
            if j == ref:
                continue
            st = plane_energy(np.column_stack([t[:, ref], t[:, j]]),
                              np.column_stack([m[:, ref], m[:, j]]))
            st["coherence_truth"] = plane_stats(t[:, ref], t[:, j])["rho"]
            st["coherence_model"] = plane_stats(m[:, ref], m[:, j])["rho"]
            out[(qname, ref, j)] = st
    return out


def size_planes(model_cogs, data_cogs, R, fractions=SIZE_FRACTIONS, cond_x=None):
    """Tier 2d — the MASS-SIZE plane per epoch: log M*(<R_max) vs
    log R_half. Returns {(key, j): (truth_stats, model_stats)}.

    ``cond_x`` (n, nz) optionally replaces the x-axis for BOTH sides with an
    external variable — halo mass, in ``evaluate`` — so the size distribution
    can be scored at fixed halo mass as well as at fixed stellar mass (the
    user's D1: "at fixed stellar mass (or halo mass)"). With an external x the
    plane is no longer self-consistent on that axis, which is the point: at
    fixed HALO mass the model's size scatter is a prediction from an input the
    model does not control.

    The mass-size relation is one of the two planes any population claim
    about this model rests on, and it was NOT in ``PLANES``: that set pairs
    measured MASSES with each other, so the model's own SIZE distribution
    had never been compared to the truth's at any epoch.

    SELF-CONSISTENT, the same deliberate exception as ``growth_planes``:
    the model's R_half comes from the MODEL CoG. Using the truth's size
    for the model (the ``measure_all`` convention, right for paired mass
    comparisons) would make half the plane correct by construction and
    erase the quantity under test.

    Both axes are genuine predictions for the transport-kernel family:
    those models are normalized at 500 kpc, so M*(<R_max = 148 kpc) on
    the reported grid is a fitted prediction, not a pinned datum.
    """
    model_cogs, data_cogs = np.asarray(model_cogs), np.asarray(data_cogs)
    nz = data_cogs.shape[1]
    out = {}
    for frac in fractions:
        key = f"R{int(round(100 * frac))}"
        rh_t = _safe_rhalf(data_cogs, R, frac)
        rh_m = _safe_rhalf(model_cogs, R, frac)
        for j in range(nz):
            if cond_x is None:
                tx = np.log10(np.clip(data_cogs[:, j, -1], 1.0, None))
                mx = np.log10(np.clip(model_cogs[:, j, -1], 1.0, None))
            else:
                tx = mx = np.asarray(cond_x, float)[:, j]
            ty, my = np.log10(rh_t[:, j]), np.log10(rh_m[:, j])
            st = plane_stats(tx, ty)
            sm = plane_stats(mx, my)
            sm.update(plane_energy(np.column_stack([tx, ty]),
                                   np.column_stack([mx, my])))
            sm["median_logR_truth"] = float(np.nanmedian(ty))
            sm["median_logR_model"] = float(np.nanmedian(my))
            sm["median_dlogR"] = float(np.nanmedian(my - ty))
            # an R20 for a compact high-z galaxy can fall INSIDE the innermost
            # measured radius, where `enclosed_radius` extrapolates; report how
            # often, because a gate on an extrapolated size is a weaker gate
            sm["frac_below_grid_truth"] = float(np.nanmean(rh_t[:, j] < R[0]))
            sm["width_ratio"] = (float(sm["scatter"] / st["scatter"])
                                 if st["scatter"] > 0 else np.nan)
            out[(key, j)] = (st, sm)
    return out


def size_gate(sizes, anchor_z, fractions=(0.2, 0.5, 0.8),
              tol_offset=SIZE_GATE_OFFSET, tol_width=SIZE_GATE_WIDTH):
    """Tier 2d as a GATE: per fractional size and epoch, does the model's size
    DISTRIBUTION at fixed mass match the truth's?

    Two conditions, both must hold:
      offset   |median dlog R| <= tol_offset  -- the relation sits in the right place
      width    model scatter / truth scatter within 1 +/- tol_width -- and it
               is as WIDE as the truth's, which a conditional-mean model fitted
               harder tends to lose (`fitting-harder-narrows-the-population`)

    Returns {(key, j): dict(offset, width_ratio, pass_offset, pass_width, ok,
    frac_below_grid)} and a summary count. The width is the vertical scatter
    about an OLS line in the log-log plane, i.e. the spread of sizes at fixed
    mass, so "narrowed by 20%" means the model's galaxies of a given mass are
    20% too alike in size.
    """
    out = {}
    for frac in fractions:
        key = f"R{int(round(100 * frac))}"
        for j in range(len(anchor_z)):
            if (key, j) not in sizes:
                continue
            st, sm = sizes[(key, j)]
            off, wr = sm["median_dlogR"], sm["width_ratio"]
            po = bool(np.isfinite(off) and abs(off) <= tol_offset)
            pw = bool(np.isfinite(wr) and abs(wr - 1.0) <= tol_width)
            out[(key, j)] = dict(offset=off, width_ratio=wr, pass_offset=po,
                                 pass_width=pw, ok=po and pw,
                                 frac_below_grid=sm.get("frac_below_grid_truth", np.nan))
    n_ok = sum(v["ok"] for v in out.values())
    return out, n_ok, len(out)


def print_size_gate(gate, anchor_z, label):
    """One table per conditioning variable: offset | width ratio | verdict."""
    res, n_ok, n = gate
    print(f"\n  tier 2d GATE — fractional-size DISTRIBUTIONS at fixed {label}: "
          f"{n_ok} of {n} pass\n    (offset |median dlog R| <= {SIZE_GATE_OFFSET} dex "
          f"AND width within {int(100 * SIZE_GATE_WIDTH)}% of the truth's)")
    keys = sorted({k for k, _ in res}, key=lambda k: int(k[1:]))
    print(f"    {'size':<5}" + "".join(f"{f'z={z}':>22}" for z in anchor_z))
    for key in keys:
        row = []
        for j in range(len(anchor_z)):
            v = res.get((key, j))
            if v is None:
                row.append(f"{'—':>22}")
                continue
            tag = "ok" if v["ok"] else ("OFF" if not v["pass_offset"] else "WID")
            row.append(f"{v['offset']:>+7.3f} {v['width_ratio']:>5.2f} {tag:>4} ".rjust(22))
        print(f"    {key:<5}" + "".join(row))
    ext = [v["frac_below_grid"] for (k, _), v in res.items() if k == keys[0]]
    if ext and np.nanmax(ext) > 0.01:
        print(f"    note: the truth's {keys[0]} falls inside the innermost measured "
              f"radius for up to {100 * np.nanmax(ext):.0f}% of galaxies at some "
              f"epoch, where it is extrapolated — that row is a weaker gate")


# --- the standard entry point --------------------------------------------------
def evaluate(model_cogs, data_cogs, R, anchor_z, name="model", figdir=None,
             verbose=True, figures=True, bin_by=None, bin_label=None,
             draw_cogs=None, bin_by_ms=None, ms_label=None, halo_mass_epochs=None):
    """Tiered QA for one model. Prints the report, writes the standard figures
    (mass tables per bin set, observational planes, profile visual QA), and
    returns a dict with all measurements for programmatic comparison.
    ``draw_cogs`` (S, n, nz, nr): sampled CoG populations from the generative
    layer; the first is overlaid on the plane figure (the mean prediction is
    under-dispersed there by construction).
    ``halo_mass_epochs`` (n, nz): log halo mass at EACH epoch. When given, the
    tier 2d size gate is also scored at fixed HALO mass (the user's D1). It is
    separate from ``bin_by`` because that is a single z=0.4 column used for the
    binned figures, and the size distribution at z_k must be conditioned on the
    halo at z_k.

    **Two binned-profile figures are always written**, because the two views
    answer different questions and can disagree in SIGN (exp53):

    - ``qa_bins_*``     binned by ``bin_by``, normally a MODEL INPUT (halo
      mass). This is the view that matters for the model's purpose — predicting
      the stellar-mass distribution from halo properties — and it is free of
      the regression-to-the-mean artifact described in ``_bins_figure``.
    - ``qa_bins_ms_*``  binned by TOTAL STELLAR MASS (``bin_by_ms``, defaulting
      to the truth's outermost CoG point at the first epoch). This is the view
      an observed, mass-selected sample would give, so it is what a reader
      compares against published profiles — but for any conditional-mean model
      it converts unexplained amplitude scatter into apparent bin-wise bias, and
      the figure is annotated to say so. The artifact is RADIUS-DEPENDENT and
      was measured in exp53: the model-on-truth slope is 0.79 at ``M(<5)``
      (where the two binnings disagree in SIGN) but 1.03 at ``M(<148)``,
      because the ``M(<500)`` normalization pins the total per galaxy and
      leaves almost no scatter to dilute (r = 0.997). So an inner-radius trend
      in the stellar-mass view needs checking against the halo-mass view; an
      outer-radius one is largely safe.
    """
    set_style()
    model_cogs, data_cogs = np.asarray(model_cogs), np.asarray(data_cogs)
    n, nz = data_cogs.shape[:2]
    truth, model, rhalf, keys = measure_all(model_cogs, data_cogs, R)
    mr_all = profile_maxrel(model_cogs, data_cogs, R)
    mr_out = profile_maxrel(model_cogs, data_cogs, R, rmin=RMIN_KPC)
    planes = {}
    for kx, ky in PLANES:
        lt = dict(x=np.log10(np.clip(truth[kx], 1.0, None)),
                  y=np.log10(np.clip(truth[ky], 1.0, None)))
        lm = dict(x=np.log10(np.clip(model[kx], 1.0, None)),
                  y=np.log10(np.clip(model[ky], 1.0, None)))
        per_epoch = []
        for j in range(nz):
            st = plane_stats(lt["x"][:, j], lt["y"][:, j])
            sm = plane_stats(lm["x"][:, j], lm["y"][:, j])
            sm.update(plane_energy(
                np.column_stack([lt["x"][:, j], lt["y"][:, j]]),
                np.column_stack([lm["x"][:, j], lm["y"][:, j]])))
            per_epoch.append((st, sm))
        planes[(kx, ky)] = per_epoch
    growth = growth_planes(model_cogs, data_cogs, R)
    sizes = size_planes(model_cogs, data_cogs, R)
    # the same distributions at fixed HALO mass, when the caller supplies it
    sizes_mh = (size_planes(model_cogs, data_cogs, R, cond_x=np.asarray(halo_mass_epochs, float))
                if halo_mass_epochs is not None else None)
    gate_ms = size_gate(sizes, anchor_z)
    gate_mh = size_gate(sizes_mh, anchor_z) if sizes_mh is not None else None
    cdfs = mass_cdf_distance(model_cogs, data_cogs, R, quantities=CDF_KEYS,
                             truth=truth, model=model)

    if verbose:
        print(f"\n=== QA [{name}]  (n={n}) ===")
        print("  tier 1+2 — masses, per epoch: median rel bias % / dex scatter")
        print(f"    {'quantity':>16s} | " + " | ".join(f"z={z}".rjust(13) for z in anchor_z))
        for k in keys:
            cells = []
            for j in range(nz):
                t, m = truth[k][:, j], model[k][:, j]
                if not (np.isfinite(t) & np.isfinite(m)).any():
                    cells.append("  flagged NaN")     # e.g. beyond the CoG grid
                    continue
                bias = 100 * np.nanmedian(relerr(m, t))
                sc = dex_scatter(m, t)
                cells.append(f"{bias:+6.1f}%/{sc:.3f}")
            print(f"    {k:>16s} | " + " | ".join(cells))
        print("\n  tier 2b — observational planes (log-log): slope / scatter / rho, "
              "truth -> model | energy-distance ratio to sampling floor")
        for (kx, ky), st in planes.items():
            for j in range(nz):
                t, mo = st[j]
                print(f"    {kx} vs {ky}  z={anchor_z[j]}: "
                      f"{t['slope']:+.2f}/{t['scatter']:.3f}/{t['rho']:+.2f} -> "
                      f"{mo['slope']:+.2f}/{mo['scatter']:.3f}/{mo['rho']:+.2f} | "
                      f"E/floor {mo['energy_ratio']:.1f} "
                      f"(centered {mo['energy_ratio_centered']:.1f})")
        if growth:
            print("\n  tier 2c — cross-epoch growth planes (SAME object at two "
                  "epochs, model sizes from the MODEL): rank coherence truth "
                  "-> model | energy-distance ratio to floor")
            for (qn, ref, j), st in growth.items():
                note = ("  [degenerate if totals are pinned to data]"
                        if qn == "Mtot" else "")
                print(f"    {qn} z={anchor_z[ref]} vs z={anchor_z[j]}: "
                      f"{st['coherence_truth']:+.2f} -> "
                      f"{st['coherence_model']:+.2f} | E/floor "
                      f"{st['energy_ratio']:.1f} (centered "
                      f"{st['energy_ratio_centered']:.1f}){note}")
        print("\n  tier 2d — the MASS-SIZE planes (log M* vs log R50/R80/R90, "
              "model sizes from the MODEL): slope/scatter truth -> model | "
              "median dlog R (model-truth) | energy ratio")
        for key in [f"R{int(round(100 * f))}" for f in SIZE_FRACTIONS]:
            for j in range(nz):
                t, mo = sizes[(key, j)]
                print(f"    {key} z={anchor_z[j]}: "
                      f"{t['slope']:+.2f}/{t['scatter']:.3f}"
                      f" -> {mo['slope']:+.2f}/{mo['scatter']:.3f} | "
                      f"median dlogR {mo['median_dlogR']:+.3f} | E/floor "
                      f"{mo['energy_ratio']:.1f} "
                      f"(centered {mo['energy_ratio_centered']:.1f})")
        print_size_gate(gate_ms, anchor_z, "STELLAR mass")
        if gate_mh is not None:
            print_size_gate(gate_mh, anchor_z, "HALO mass")
        print("\n  tier 2e — mass DISTRIBUTIONS per epoch (log10 mass CDFs, "
              "population-level):\n    KS / W1[dex], each as a ratio to the "
              "truth's split-half floor (~1 = indistinguishable)")
        print(f"    {'quantity':>16s} | "
              + " | ".join(f"z={z}".rjust(13) for z in anchor_z))
        for k in CDF_KEYS:
            cells = []
            for j in range(nz):
                d = cdfs[(k, j)]
                cells.append("     n/a     " if not np.isfinite(d["ks_ratio"])
                             else f"{d['ks_ratio']:5.1f}/{d['w1_ratio']:5.1f}")
            print(f"    {_tex(k):>16s} | " + " | ".join(c.rjust(13)
                                                        for c in cells))

        print("\n  tier 3 — profile max|rel| median per epoch (all R | R>5 kpc):")
        row_a = " | ".join(f"{100*np.nanmedian(mr_all[:, j]):5.1f}%" for j in range(nz))
        row_o = " | ".join(f"{100*np.nanmedian(mr_out[:, j]):5.1f}%" for j in range(nz))
        print(f"    all R   : {row_a} | avg {100*np.nanmean(np.nanmedian(mr_all, 0)):.1f}%")
        print(f"    R>5 kpc : {row_o} | avg {100*np.nanmean(np.nanmedian(mr_out, 0)):.1f}%")

    if figures and figdir is not None:
        figdir = Path(figdir)
        figdir.mkdir(parents=True, exist_ok=True)
        _mass_figure("kpc", [k for k in keys if k.startswith("kpc:")],
                     truth, model, anchor_z, name, figdir)
        _mass_figure("Re", [k for k in keys if k.startswith("Re:")],
                     truth, model, anchor_z, name, figdir)
        draw_m = (None if draw_cogs is None
                  else measure_all(np.asarray(draw_cogs)[0], data_cogs, R)[1])
        _plane_figure(truth, model, anchor_z, name, figdir, draws=draw_m)
        _bins_figure(model_cogs, data_cogs, R, anchor_z, name, figdir,
                     bin_by=bin_by, bin_label=bin_label)
        # the stellar-mass view, always: standard since 2026-08-21 (exp53)
        ms = bin_by_ms
        if ms is None:
            ms = np.log10(np.clip(data_cogs[:, 0, -1], 1.0, None))
            ms_label = ms_label or f"logM$_*$($<${R[-1]:.0f} kpc, z={anchor_z[0]})"
        _bins_figure(model_cogs, data_cogs, R, anchor_z, name, figdir,
                     bin_by=ms, bin_label=ms_label or "logM$_*$",
                     tag="bins_ms", caveat=True)
        _density_figure(model_cogs, data_cogs, R, anchor_z, name, figdir,
                        bin_by=bin_by, bin_label=bin_label)
        _cases_figure(model_cogs, data_cogs, R, anchor_z, mr_all, name, figdir)
        if growth:
            _growth_figure(model_cogs, data_cogs, R, anchor_z, growth, name,
                           figdir)
        _size_figure(model_cogs, data_cogs, R, anchor_z, sizes, name, figdir)
        _cdf_figure(truth, model, anchor_z, name, figdir)

    return dict(truth=truth, model=model, rhalf=rhalf, keys=keys,
                mr_all=mr_all, mr_out=mr_out, planes=planes,
                growth=growth, sizes=sizes, sizes_mh=sizes_mh,
                size_gate_ms=gate_ms, size_gate_mh=gate_mh, cdfs=cdfs)


def _cdf_figure(truth, model, anchor_z, name, figdir):
    """Tier 2e visual: the population CDF of each mass, truth vs model, with
    a residual strip. The residual is what the KS statistic maximizes, so
    the strip shows exactly where the reported number comes from."""
    nz = len(anchor_z)
    cols = len(CDF_KEYS)
    fig, axes = plt.subplots(2, cols, figsize=(3.1 * cols, 5.6),
                             sharex="col",
                             gridspec_kw=dict(height_ratios=[3, 1.35]))
    axes = np.atleast_2d(axes)
    colors = _zcolors(nz)
    for c, k in enumerate(CDF_KEYS):
        ax, rax = axes[0, c], axes[1, c]
        for j in range(nz):
            tv = np.sort(np.log10(np.clip(truth[k][:, j], 1.0, None)))
            mv = np.sort(np.log10(np.clip(model[k][:, j], 1.0, None)))
            tv, mv = tv[tv > 0], mv[mv > 0]
            if len(tv) < 8 or len(mv) < 8:
                continue
            ft = np.arange(1, len(tv) + 1) / len(tv)
            fm = np.arange(1, len(mv) + 1) / len(mv)
            ax.plot(tv, ft, color=colors[j], lw=1.6, alpha=0.9,
                    label=f"z={anchor_z[j]}" if c == 0 else None)
            ax.plot(mv, fm, color=colors[j], lw=1.4, ls="--", alpha=0.9)
            grid = np.union1d(tv, mv)
            rax.plot(grid,
                     np.interp(grid, mv, fm) - np.interp(grid, tv, ft),
                     color=colors[j], lw=1.2)
        ax.set_title(_tex(k), fontsize=9)
        ax.set_ylim(0, 1)
        rax.axhline(0.0, color="0.5", lw=0.8)
        rax.set_ylim(-0.35, 0.35)
        rax.set_xlabel(_tex(f"log10 {k}"), fontsize=8)
        if c == 0:
            ax.set_ylabel("CDF (solid truth, dashed model)", fontsize=8)
            rax.set_ylabel("model - truth", fontsize=8)
            ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.suptitle(_tex(f"QA [{name}] — tier 2e mass distributions; the "
                      "residual strip is what KS maximizes"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, str(Path(figdir) / f"qa_cdf_{name}"))
    plt.close(fig)


def _size_figure(model_cogs, data_cogs, R, anchor_z, sizes, name, figdir):
    """Tier 2d visual. Row 1: the mass-size relation (R50), truth vs model,
    each side its OWN sizes. Row 2: the median size offset vs mass for
    R50/R80/R90 together — the discriminator between an INNER-mass defect
    (R50 biased, R90 not) and a genuine size error (all three move)."""
    nz = data_cogs.shape[1]
    cols = _zcolors(nz)
    keys = [f"R{int(round(100 * f))}" for f in SIZE_FRACTIONS]
    fr_styles = dict(zip(keys, ["-o", "--s", ":^"]))
    rt = {k: _safe_rhalf(data_cogs, R, f)
          for k, f in zip(keys, SIZE_FRACTIONS)}
    rm = {k: _safe_rhalf(model_cogs, R, f)
          for k, f in zip(keys, SIZE_FRACTIONS)}
    fig, axes = plt.subplots(2, nz, squeeze=False, figsize=(3.0 * nz, 6.4),
                             gridspec_kw=dict(height_ratios=[2.0, 1.2]))
    for j in range(nz):
        ax = axes[0][j]
        tx = np.log10(np.clip(data_cogs[:, j, -1], 1.0, None))
        mx = np.log10(np.clip(model_cogs[:, j, -1], 1.0, None))
        ax.scatter(tx, np.log10(rt["R50"][:, j]), s=4, c="0.65", alpha=0.5,
                   lw=0, label="truth")
        ax.scatter(mx, np.log10(rm["R50"][:, j]), s=4, c=cols[j], alpha=0.5,
                   lw=0, label="model")
        t, mo = sizes[("R50", j)]
        ax.set_title(_tex(f"z={anchor_z[j]}") + "\n"
                     + _tex(f"R50 slope {t['slope']:+.2f} -> "
                            f"{mo['slope']:+.2f}, E/floor "
                            f"{mo['energy_ratio']:.1f}"), fontsize=8.5)
        ax.set_xlabel(_tex(r"$\log_{10} M_\star(<R_{\max})$"))
        if j == 0:
            ax.set_ylabel(_tex(r"$\log_{10} R_{50}$ [kpc]"))
            ax.legend(fontsize=7, markerscale=2.5)
        bx = axes[1][j]
        edges = np.nanpercentile(tx, np.linspace(2, 98, 8))
        for k in keys:
            ctr, dd = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                st = (tx >= a) & (tx < b)
                sm = (mx >= a) & (mx < b)
                if st.sum() > 5 and sm.sum() > 5:
                    ctr.append(0.5 * (a + b))
                    dd.append(np.nanmedian(np.log10(rm[k][sm, j]))
                              - np.nanmedian(np.log10(rt[k][st, j])))
            if ctr:
                bx.plot(ctr, dd, fr_styles[k], color=cols[j], lw=1.5, ms=4,
                        alpha=1.0 if k == "R50" else 0.65, label=k)
        bx.axhline(0.0, color="0.8", lw=0.9, zorder=0)
        bx.set_xlabel(_tex(r"$\log_{10} M_\star(<R_{\max})$"))
        if j == 0:
            bx.set_ylabel(_tex(r"median $\Delta \log R$ (model$-$truth)"))
            bx.legend(fontsize=7, ncol=3)
    fig.suptitle(_tex(f"QA [{name}] — tier 2d mass-size planes; row 2 "
                      "separates an inner-mass defect from a size error"),
                 fontsize=11)
    fig.tight_layout()
    save_fig(fig, Path(figdir) / f"qa_size_{name}")
    plt.close(fig)


def _growth_figure(model_cogs, data_cogs, R, anchor_z, growth, name, figdir):
    """Tier 2c visual: the cross-epoch cloud, truth vs model, one column per
    non-reference epoch. Over-coherent models collapse onto the 1:1-like
    ridge; incoherent ones smear off it."""
    pairs = sorted({(qn, ref, j) for qn, ref, j in growth})
    quants = sorted({p[0] for p in pairs})
    cols = sorted({p[2] for p in pairs})
    if not cols:
        return
    nz = data_cogs.shape[1]
    rh_t, rh_m = _safe_rhalf(data_cogs, R), _safe_rhalf(model_cogs, R)
    fig, axes = plt.subplots(len(quants), len(cols), squeeze=False,
                             figsize=(3.2 * len(cols), 3.3 * len(quants)))
    cols_z = _zcolors(nz)
    for ri, qn in enumerate(quants):
        if qn == "Mtot":
            t_q, m_q = data_cogs[..., -1], model_cogs[..., -1]
            lab = r"$\log_{10} M_{\star,\rm tot}$"
        else:
            t_q, m_q = rh_t, rh_m
            lab = r"$\log_{10} R_{\rm half}$ [kpc]"
        t = np.log10(np.clip(t_q, 1e-30, None))
        m = np.log10(np.clip(m_q, 1e-30, None))
        for ci, j in enumerate(cols):
            ax = axes[ri][ci]
            ref = growth_ref = pairs[0][1]
            ax.scatter(t[:, ref], t[:, j], s=4, c="0.6", alpha=0.5, lw=0,
                       label="truth")
            ax.scatter(m[:, ref], m[:, j], s=4, alpha=0.5, lw=0,
                       c=cols_z[j % len(cols_z)], label="model")
            st = growth[(qn, growth_ref, j)]
            ax.set_title(_tex(f"z={anchor_z[ref]:g} vs z={anchor_z[j]:g}")
                         + "\n" + _tex(f"coh {st['coherence_truth']:+.2f}"
                                       f" -> {st['coherence_model']:+.2f}"),
                         fontsize=9)
            ax.set_xlabel(_tex(f"{lab} at z={anchor_z[ref]:g}"))
            if ci == 0:
                ax.set_ylabel(_tex(f"{lab} at the later epoch"))
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, markerscale=2)
    fig.suptitle(_tex(f"QA [{name}] — tier 2c cross-epoch growth planes "
                      "(same object, two epochs)"), fontsize=11)
    fig.tight_layout()
    save_fig(fig, Path(figdir) / f"qa_growth_{name}")
    plt.close(fig)


# --- figures ---------------------------------------------------------------------
# ordered cool->warm subset of the Okabe-Ito colorblind-safe palette: the
# cividis ramp made adjacent epochs (z=0.7..1.5) nearly indistinguishable
_EPOCH_COLORS = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]


def _zcolors(nz):
    if nz <= len(_EPOCH_COLORS):
        return _EPOCH_COLORS[:nz]
    return [matplotlib.colormaps["cividis"](v) for v in np.linspace(0.0, 0.92, nz)]


def _tex(s):
    """Make a label LaTeX-text-mode safe: <, >, | render as inverted
    punctuation / dashes under usetex unless wrapped in math mode."""
    return s.replace("<", "$<$").replace(">", "$>$").replace("|", "$|$")


def _pct():
    """A percent sign that survives usetex (% starts a LaTeX comment)."""
    return r"\%" if matplotlib.rcParams.get("text.usetex", False) else "%"


def _mass_figure(tag, keys, truth, model, anchor_z, name, figdir):
    """Two figures per bin set: cumulative apertures ('aper') and differential
    annuli+envelopes ('diff'). Each quantity is a COLUMN: truth-vs-model on top,
    relative error below, vertically stacked with a shared per-quantity x-axis
    whose range follows that quantity's own truth values."""
    groups = {"aper": [k for k in keys if "(<" in k],
              "diff": [k for k in keys if "(<" not in k]}
    for gtag, gkeys in groups.items():
        if gkeys:
            _mass_group_figure(tag, gtag, gkeys, truth, model, anchor_z, name, figdir)


def _mass_group_figure(tag, gtag, keys, truth, model, anchor_z, name, figdir):
    nz = len(anchor_z)
    cols = _zcolors(nz)
    ncol = len(keys)
    fig, axes = plt.subplots(2, ncol, figsize=(3.1 * ncol, 6.4), squeeze=False,
                             sharex="col", height_ratios=[2, 1])
    for ci, key in enumerate(keys):
        X, Y = truth[key], model[key]
        av, ar = axes[0, ci], axes[1, ci]
        for j in range(nz):
            x, y = X[:, j], Y[:, j]
            good = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
            av.scatter(x[good], y[good], s=8, c=[cols[j]], alpha=0.4, edgecolors="none",
                       label=f"z={anchor_z[j]}" if ci == 0 else None)
            re = relerr(y, x)
            ar.scatter(x[good], re[good], s=8, c=[cols[j]], alpha=0.3, edgecolors="none")
            if good.any():
                ar.plot(np.median(x[good]), np.median(re[good]), "o", c=cols[j], ms=8,
                        mec="k", mew=0.6)
        pos = (X > 0) & np.isfinite(X)
        if pos.any():
            # robust per-quantity range: near-empty bins (~1 Msun) must not
            # stretch the axes; the residual panel still shows every galaxy
            lo = np.percentile(X[pos], 0.5) / 2.0
            hi = np.percentile(X[pos], 99.5) * 2.0
            av.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=0)
        else:
            # all-NaN quantity (e.g. flagged beyond the CoG grid edge): give
            # the log axes finite limits — empty log axes crash tight_layout
            lo, hi = 1.0, 10.0
            av.text(0.5, 0.5, "flagged NaN\n(beyond CoG grid)", ha="center",
                    va="center", transform=av.transAxes, fontsize=8, color="0.4")
        av.set_xlim(lo, hi)
        av.set_ylim(lo, hi)
        av.set(xscale="log", yscale="log", title=_tex(key))
        ar.axhline(0, c="0.5", lw=0.8)
        for gpm in (0.1, -0.1):
            ar.axhline(gpm, c="0.7", ls=":", lw=0.8)
        ar.set(xscale="log", ylim=(-0.6, 0.6), xlabel=r"truth $M_*$ [$M_\odot$]")
        if ci == 0:
            av.set_ylabel(r"model $M_*$ [$M_\odot$]")
            ar.set_ylabel("(model$-$truth)/truth")
            av.legend(fontsize=7, loc="upper left", markerscale=1.6)
    unit = "physical kpc" if tag == "kpc" else r"$R_{\rm half}$ units"
    kind = "cumulative apertures" if gtag == "aper" else "annuli + envelopes"
    fig.suptitle(f"QA [{name}] — {unit}: {kind}", fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, figdir / f"qa_mass_{tag}_{gtag}_{name}")[0])


def _plane_figure(truth, model, anchor_z, name, figdir, draws=None):
    """Observational planes: truth (filled) vs the MEAN prediction (open) — the
    mean is under-dispersed by construction (regression to the mean), so when
    ``draws`` (a measured sampled population) is given, it is overlaid as the
    honest generative comparison and its scatter quoted in the title."""
    nz = len(anchor_z)
    cols = _zcolors(nz)
    fig, axes = plt.subplots(len(PLANES), nz, figsize=(3.1 * nz, 3.2 * len(PLANES)),
                             squeeze=False)
    for ri, (kx, ky) in enumerate(PLANES):
        for j in range(nz):
            ax = axes[ri, j]
            lx, ly = (np.log10(np.clip(truth[kx][:, j], 1.0, None)),
                      np.log10(np.clip(truth[ky][:, j], 1.0, None)))
            mx, my = (np.log10(np.clip(model[kx][:, j], 1.0, None)),
                      np.log10(np.clip(model[ky][:, j], 1.0, None)))
            ax.scatter(lx, ly, s=14, c=[cols[j]], alpha=0.75, edgecolors="none",
                       label="truth")
            ax.scatter(mx, my, s=18, facecolors="none", edgecolors="0.25", lw=0.7,
                       label="model mean")
            st, sm = plane_stats(lx, ly), plane_stats(mx, my)
            title = (f"z={anchor_z[j]}  slope {st['slope']:+.2f}$\\to$"
                     f"{sm['slope']:+.2f}\nscatter {st['scatter']:.3f}$\\to$"
                     f"{sm['scatter']:.3f}")
            if draws is not None:
                dx, dy = (np.log10(np.clip(draws[kx][:, j], 1.0, None)),
                          np.log10(np.clip(draws[ky][:, j], 1.0, None)))
                ax.scatter(dx, dy, s=8, marker="x", c="0.45", lw=0.5,
                           alpha=0.6, label="draw")
                sd = plane_stats(dx, dy)
                title += f" (draw {sd['scatter']:.3f})"
            ax.set_title(title, fontsize=8)
            # frame the TRUTH population: floored/absurd model points may clip
            good = np.isfinite(lx) & np.isfinite(ly)
            if good.sum() > 10:
                xr = np.percentile(lx[good], [0.5, 99.5])
                yr = np.percentile(ly[good], [0.5, 99.5])
                ax.set_xlim(xr[0] - 0.4, xr[1] + 0.4)
                ax.set_ylim(yr[0] - 0.4, yr[1] + 0.4)
            if j == 0:
                ax.set_ylabel(_tex(f"log {ky}"))
            ax.set_xlabel(_tex(f"log {kx}"))
            if ri == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"QA [{name}] — observational planes: truth (filled) vs "
                 "model mean (open)" + ("" if draws is None else " vs draw (x)"),
                 fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, figdir / f"qa_planes_{name}")[0])


def _tercile_curves(model_k, data_k):
    """Median log CoGs + median raw and amplitude-pinned residual curves for one
    tercile/epoch selection (n_sel, R) -> four (R,) arrays. NaN-safe: one bad
    galaxy must not blank the whole bin's medians."""
    med_d = np.nanmedian(np.log10(np.clip(data_k, 1.0, None)), axis=0)
    med_m = np.nanmedian(np.log10(np.clip(model_k, 1.0, None)), axis=0)
    rel = 100 * np.nanmedian((model_k - data_k) / data_k, axis=0)
    # amplitude-pinned residual: rescale each model CoG to the true total,
    # isolating SHAPE error from amplitude regression-to-the-mean (binning by
    # truth M* makes any conditional mean look biased by ~ the unexplained
    # amplitude scatter)
    pin = model_k * (data_k[:, -1:] / np.clip(model_k[:, -1:], 1.0, None))
    rel_pin = 100 * np.nanmedian((pin - data_k) / data_k, axis=0)
    return med_d, med_m, rel, rel_pin


def _case_picks(mr, n=2):
    """Best-``n`` + worst-``n`` galaxy indices by their WORST epoch (minimax).

    Ranking by the epoch average let a "best" galaxy hide one bad epoch;
    minimax makes the best gallery uniformly good. Galaxies with no finite
    epoch are excluded: NaN sorts last under argsort, which would silently
    promote broken galaxies into the "worst" gallery.
    """
    good = np.where(np.isfinite(mr).any(axis=1))[0]
    order = good[np.argsort(np.nanmax(mr[good], axis=1))]
    return list(order[:n]) + list(order[-n:])


def _bins_figure(model_cogs, data_cogs, R, anchor_z, name, figdir,
                 bin_by=None, bin_label=None, tag="bins", caveat=False):
    """Median data-vs-model CoG per tercile of ``bin_by`` + median residuals.

    Bin by a MODEL INPUT (halo mass) whenever available: binning by the truth
    stellar mass converts unexplained amplitude scatter into apparent bin-wise
    bias (regression to the mean) for any conditional-mean model. The truth-M*
    binning remains the fallback (and is the view an observed, M*-selected
    sample would give)."""
    nz = len(anchor_z)
    cols = _zcolors(nz)
    if bin_by is None:
        bin_by = np.log10(data_cogs[:, 0, -1])
        bin_label = bin_label or "logM* (z=0.4)"
    bin_label = bin_label or "bin quantity"
    edges = np.quantile(bin_by, [0, 1 / 3, 2 / 3, 1])
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.5), sharex=True,
                             height_ratios=[2, 1])
    rmax = 0.0
    for b in range(3):
        m = (bin_by >= edges[b]) & (bin_by <= edges[b + 1] + 1e-9)
        ax, rax = axes[0, b], axes[1, b]
        for k in range(nz):
            med_d, med_m, rel, rel_pin = _tercile_curves(model_cogs[m, k],
                                                         data_cogs[m, k])
            rmax = max(rmax, np.nanmax(np.abs(rel)), np.nanmax(np.abs(rel_pin)))
            ax.plot(R, med_d, "-", c=cols[k], lw=1.8,
                    label=f"z={anchor_z[k]}" if b == 0 else None)
            ax.plot(R, med_m, "--", c=cols[k], lw=1.4)
            rax.plot(R, rel, "-", c=cols[k], lw=1.4,
                     label="raw" if (b == 0 and k == 0) else None)
            rax.plot(R, rel_pin, ":", c=cols[k], lw=1.4,
                     label="amplitude-pinned (shape)" if (b == 0 and k == 0) else None)
        rax.axhline(0, c="0.6", lw=0.8)
        if b == 0:
            rax.legend(fontsize=7, loc="upper right")
        ax.set(xscale="log",
               title=f"{bin_label} {edges[b]:.2f}-{edges[b+1]:.2f}  (n={m.sum()})")
        rax.set(xscale="log", xlabel="R [kpc]")
    # residual range follows the curves (a fixed +-60 dwarfed a good model);
    # shared across panels for comparability
    lim = float(np.clip(1.25 * rmax, 10.0, 60.0))
    for b in range(3):
        for y in (-10, 10):
            axes[1, b].axhline(y, c="0.8", lw=0.7, ls=":")
        axes[1, b].set_ylim(-lim, lim)
    axes[0, 0].set(ylabel="median log$_{10}$ M$_*$($<$R) [M$_\\odot$]")
    axes[1, 0].set(ylabel=f"median (model$-$data)/data [{_pct()}]")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"QA [{name}] — CoGs by {bin_label} tercile "
                 "(data solid, model dashed)", fontsize=12)
    if caveat:
        fig.text(0.5, 0.005,
                 "Binned by TRUTH stellar mass: selecting on the noisy "
                 "quantity tilts the residual by (slope$-$1) dex/dex, where "
                 "slope $= r\\,\\sigma_{model}/\\sigma_{truth}$ "
                 "(regression to the mean). RADIUS-DEPENDENT: strongest at "
                 "small R where the model has predictive freedom (measured "
                 "slope 0.79 at $M(<5)$, where this view and the halo-mass "
                 "view disagree in SIGN), and negligible at large R where the "
                 "$M(<500)$ normalization pins the total to the truth "
                 "($r=0.997$). Check the halo-mass-binned figure before "
                 "reading an inner-radius trend as a model defect.",
                 ha="center", va="bottom", fontsize=7.0, style="italic",
                 color="0.35")
        fig.tight_layout(rect=(0, 0.035, 1, 1))
    else:
        fig.tight_layout()
    print("wrote", save_fig(fig, figdir / f"qa_{tag}_{name}")[0])


def _density_figure(model_cogs, data_cogs, R, anchor_z, name, figdir,
                    bin_by=None, bin_label=None):
    """Median SURFACE-DENSITY profile on an R^(1/4) axis, by tercile of
    ``bin_by`` (a model input, normally halo mass).

    WHY R^(1/4). For massive early-type galaxies it is the natural radial
    coordinate: it expands the inner region far more than a linear axis and far
    less violently than log R, and a de Vaucouleurs profile (Sersic n = 4) is a
    STRAIGHT LINE in (R^(1/4), log10 Sigma). Curvature away from the grey guide
    is therefore read directly as departure from an r^(1/4) law, with no fit.

    WHY IT EARNS A PLACE BESIDE THE CoG FIGURES. A curve of growth is
    cumulative and dominated by the inner regions, so it hides the outer
    LOGARITHMIC SLOPE almost completely. Measured on exp54's adopted model, the
    z = 2 curve of growth is only -0.024 dex low at 148 kpc while the outer
    density slope is 0.48 dex/dex too shallow -- a large shape error that the
    cumulative view does not show. Any model whose deposits are summed will have
    this blind spot.

    The residual panel is in DEX, unlike ``_bins_figure`` which uses per cent.
    Both are labelled; 0.043 dex = 10 per cent.
    """
    nz = len(anchor_z)
    cols = _zcolors(nz)
    if bin_by is None:
        bin_by = np.log10(data_cogs[:, 0, -1])
        bin_label = bin_label or "logM* (z=0.4)"
    bin_label = bin_label or "bin quantity"
    edges = np.quantile(bin_by, [0, 1 / 3, 2 / 3, 1])
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.5), sharex=True,
                             height_ratios=[2, 1])
    guide, rmax = None, 0.0
    for b in range(3):
        m = (bin_by >= edges[b]) & (bin_by <= edges[b + 1] + 1e-9)
        ax, rax = axes[0, b], axes[1, b]
        for k in range(nz):
            ld, mid = density_from_cog(
                np.log10(np.clip(data_cogs[m, k], 1.0, None)), R)
            lm, _ = density_from_cog(
                np.log10(np.clip(model_cogs[m, k], 1.0, None)), R)
            med_d, med_m = np.nanmedian(ld, axis=0), np.nanmedian(lm, axis=0)
            x = mid ** 0.25
            ax.plot(x, med_d, "-", c=cols[k], lw=1.8,
                    label=f"z={anchor_z[k]}" if b == 0 else None)
            ax.plot(x, med_m, "--", c=cols[k], lw=1.4)
            d = np.nanmedian(lm - ld, axis=0)
            rmax = max(rmax, np.nanmax(np.abs(d)))
            rax.plot(x, d, "-", c=cols[k], lw=1.4)
            if b == 0 and k == 0:
                ok = np.isfinite(med_d)
                guide = (np.polyfit(x[ok], med_d[ok], 1), x)
        if guide is not None:
            co, gx = guide
            ax.plot(gx, np.polyval(co, gx), ":", c="0.45", lw=1.2,
                    label=_tex("de Vaucouleurs (n=4), fit to first epoch")
                    if b == 0 else None)
        rax.axhline(0, c="0.6", lw=0.8)
        ax.set_title(f"{bin_label} {edges[b]:.2f}-{edges[b+1]:.2f}  "
                     f"(n={m.sum()})")
        rax.set_xticks([r ** 0.25 for r in (2, 5, 10, 20, 50, 100, 150)])
        rax.set_xticklabels(["2", "5", "10", "20", "50", "100", "150"])
        rax.set_xlabel(_tex("R [kpc], on an ") + r"$R^{1/4}$" + _tex(" scale"))
    lim = float(np.clip(1.25 * rmax, 0.15, 0.9))
    for b in range(3):
        axes[1, b].set_ylim(-lim, lim)
    axes[0, 0].set_ylabel(r"median $\log_{10}\Sigma$ [M$_\odot$ kpc$^{-2}$]")
    axes[1, 0].set_ylabel(r"median $\log_{10}$(model/truth) [dex]")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"QA [{name}] — surface density on an "
                 + r"$R^{1/4}$" + f" scale, by {bin_label} tercile "
                 "(data solid, model dashed)", fontsize=12)
    fig.text(0.5, 0.005, _tex(
        "A de Vaucouleurs (n=4) profile is a STRAIGHT LINE here, so curvature "
        "away from the grey guide is departure from it. Residual in DEX, "
        "unlike the qa_bins figures which use per cent (0.043 dex = 10 per "
        "cent). The cumulative view hides the outer slope: a model can be "
        "within 0.02 dex on the total and still be ~0.5 dex/dex too shallow "
        "in the outskirts."),
        ha="center", va="bottom", fontsize=7.0, style="italic", color="0.35")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    print("wrote", save_fig(fig, figdir / f"qa_dens_{name}")[0])



def _cases_figure(model_cogs, data_cogs, R, anchor_z, mr, name, figdir):
    """Best/worst gallery by epoch-avg max|rel|, worst radius marked."""
    nz = len(anchor_z)
    cols = _zcolors(nz)
    logms = np.log10(data_cogs[:, 0, -1])
    picks = _case_picks(mr)
    tags = ["best 1", "best 2", "worst 2", "worst 1"]
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 7.5), sharex=True,
                             height_ratios=[2, 1])
    for c, (i, tag) in enumerate(zip(picks, tags)):
        ax, rax = axes[0, c], axes[1, c]
        for k in range(nz):
            rel = 100 * (model_cogs[i, k] - data_cogs[i, k]) / data_cogs[i, k]
            ax.plot(R, np.log10(data_cogs[i, k]), "-", c=cols[k], lw=1.8,
                    label=f"z={anchor_z[k]}" if c == 0 else None)
            ax.plot(R, np.log10(np.clip(model_cogs[i, k], 1.0, None)), "--",
                    c=cols[k], lw=1.4)
            rax.plot(R, rel, "-", c=cols[k], lw=1.4)
            j = int(np.argmax(np.abs(rel)))
            rax.plot(R[j], rel[j], "o", c=cols[k], ms=5, mec="0.2", mew=0.5)
        rax.axhline(0, c="0.6", lw=0.8)
        ax.set(xscale="log", title=f"{tag}: idx {i}, logM$_*$={logms[i]:.2f}\n"
               f"worst-epoch max$|$rel$|$ {100*np.nanmax(mr[i]):.0f}{_pct()}")
        rax.set(xscale="log", xlabel="R [kpc]")
    axes[0, 0].set(ylabel="log$_{10}$ M$_*$($<$R) [M$_\\odot$]")
    axes[1, 0].set(ylabel=f"(model$-$data)/data [{_pct()}]")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"QA [{name}] — best/worst cases (data solid, model dashed; "
                 "dot = worst radius)", fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, figdir / f"qa_cases_{name}")[0])


def demo():
    """Self-check on synthetic CoGs with known errors: identity -> zero bias /
    scatter / max|rel| and identical plane stats; a global +10% amplitude scaling
    -> +10% bias on every mass, ~0 dex scatter, 10% max|rel|."""
    rng = np.random.default_rng(0)
    R = np.geomspace(2.0, 150.0, 24)
    n, nz = 12, 3
    sig = rng.uniform(5.0, 40.0, (n, nz, 1))
    amp = 10.0 ** rng.uniform(10.5, 11.5, (n, 1, 1)) * np.ones((1, nz, 1))
    truth = amp * (1.0 - np.exp(-R[None, None, :] ** 2 / (2.0 * sig ** 2)))
    res = evaluate(truth, truth, R, [0.4, 1.0, 2.0], name="identity",
                   verbose=False, figures=False)
    for k in res["keys"]:
        assert np.nanmax(np.abs(relerr(res["model"][k], res["truth"][k]))) < 1e-12
        assert dex_scatter(res["model"][k], res["truth"][k]).max() < 1e-12
    assert res["mr_all"].max() < 1e-12 and res["mr_out"].max() < 1e-12
    for st in res["planes"].values():
        for t, m in st:
            assert abs(t["slope"] - m["slope"]) < 1e-9

    # energy distance: identical samples -> 0; a large shift -> far above floor;
    # a same-distribution resample -> ratio ~ 1
    rng2 = np.random.default_rng(1)
    A = rng2.normal(size=(400, 2))
    assert energy_distance_2d(A, A.copy()) < 1e-12
    shift = plane_energy(A, A + 3.0)
    same = plane_energy(A, rng2.normal(size=(400, 2)))
    tight = plane_energy(A, 0.3 * rng2.normal(size=(400, 2)))
    assert shift["energy_ratio"] > 10, shift
    assert shift["energy_ratio_centered"] < 3, shift      # pure shift: centered ~ floor
    assert same["energy_ratio"] < 3, same
    assert tight["energy_ratio_centered"] > 3, tight      # too tight: centered catches it

    # NaN safety of the figure reductions: one bad galaxy must not blank a
    # tercile's median curves, and an all-NaN galaxy must not rank as "worst"
    model_nan = 1.1 * truth
    model_nan[3] = np.nan
    md, mm, rel, rel_pin = _tercile_curves(model_nan[:, 0], truth[:, 0])
    for arr in (md, mm, rel, rel_pin):
        assert np.isfinite(arr).all(), "one NaN galaxy blanked the tercile medians"
    mr_nan = profile_maxrel(model_nan, truth, R)
    picks = _case_picks(mr_nan)
    assert len(picks) == 4 and 3 not in picks, \
        f"all-NaN galaxy ranked in the best/worst gallery: {picks}"
    # the gallery ranks by the WORST epoch (minimax), not the epoch average:
    # "best" must be uniformly good, not good-on-average with one bad epoch
    mr_mix = np.array([[0.01, 0.01, 0.31],      # best AVERAGE, one bad epoch
                       [0.12, 0.12, 0.12],      # uniformly decent -> best
                       [0.30, 0.30, 0.30],
                       [0.60, 0.60, 0.60]])     # -> worst
    p_mix = _case_picks(mr_mix, n=1)
    assert p_mix == [1, 3], f"gallery must rank by worst epoch, got {p_mix}"

    # figure text must be LaTeX-text-mode safe: <, >, | render as inverted
    # punctuation / dashes under usetex unless math-wrapped; % starts a comment
    assert _tex("kpc:M(<30)") == "kpc:M($<$30)"
    assert _tex("kpc:M(>50)") == "kpc:M($>$50)"
    assert _tex("max|rel|") == "max$|$rel$|$"
    assert _pct() in ("%", r"\%")

    # grid-edge queries must FLAG (NaN), not silently clamp to the last CoG
    # point (the real 24-point grid ends at 148.2 kpc < the 150-kpc annulus edge)
    R_short = np.geomspace(2.0, 120.0, 16)
    cog_short = 1e11 * (1.0 - np.exp(-R_short / 10.0))
    m_short = measure(cog_short, R_short, 40.0)
    assert np.isfinite(m_short["kpc:M(<100)"]), "in-grid aperture must stay finite"
    assert np.isnan(m_short["kpc:M(100-150)"]), "annulus past the grid edge must flag NaN"
    assert np.isnan(m_short["Re:M(>4Re)"]), "Re envelope past the grid edge must flag NaN"
    # ... and the mass figures must survive a quantity that is ALL-NaN on such
    # a grid (an empty log-scaled panel crashes tight_layout otherwise)
    import tempfile
    truth_short = (amp * (1.0 - np.exp(-R_short[None, None, :] ** 2
                                       / (2.0 * sig ** 2))))
    evaluate(1.1 * truth_short, truth_short, R_short, [0.4, 1.0, 2.0],
             name="short_grid", verbose=False, figures=True,
             figdir=tempfile.mkdtemp(prefix="qa_demo_"),
             draw_cogs=np.stack([1.05 * truth_short, 0.95 * truth_short]))

    # --- tier 2c: the cross-epoch growth plane must catch what every
    # single-epoch tier is structurally blind to (exp44 stage 3). Build a
    # truth whose galaxies only weakly keep their size rank across epochs,
    # and an OVER-COHERENT model that keeps it perfectly while having the
    # SAME per-epoch size distribution — the per-epoch tiers cannot tell
    # them apart, this one must.
    rg = np.random.default_rng(11)
    ng, nzg = 400, 3
    fac = np.array([1.0, 0.6, 0.35])                     # sizes shrink with z
    V = np.sort(10.0 ** rg.normal(0.9, 0.22, ng))[:, None] * fac[None, :]

    def _assign(latent):
        """Rank-map latents onto the FIXED per-epoch marginals V, so every
        construction below has identical single-epoch size distributions and
        differs only in cross-epoch rank coherence."""
        out = np.empty_like(V)
        for j in range(nzg):
            out[np.argsort(latent[:, j]), j] = V[:, j]
        return out

    def _latent(rc):
        return (rc * rg.normal(size=(ng, 1))
                + np.sqrt(1 - rc ** 2) * rg.normal(size=(ng, nzg)))

    s_coh = _assign(np.repeat(rg.normal(size=(ng, 1)), nzg, axis=1))
    s_inc = _assign(_latent(0.55))                       # -> coherence ~0.3
    s_mat = _assign(_latent(0.55))                       # matched, indep. draw
    Rg = np.geomspace(1.0, 200.0, 30)
    amp_g = 10.0 ** rg.normal(11.0, 0.25, (ng, 1, 1)) * np.ones((1, nzg, 1))

    def _cog_from_size(s):
        return amp_g * (1.0 - np.exp(-Rg[None, None, :] ** 2
                                     / (2.0 * (s[:, :, None] / 1.1774) ** 2)))

    truth_g, model_g = _cog_from_size(s_inc), _cog_from_size(s_coh)
    # the per-epoch size distributions are identical by construction, so no
    # single-epoch tier can separate these two populations
    for j in range(nzg):
        assert np.allclose(np.sort(s_inc[:, j]), np.sort(s_coh[:, j])), j
    g_res = growth_planes(model_g, truth_g, Rg)
    g_mat = growth_planes(_cog_from_size(s_mat), truth_g, Rg)
    key = ("R_half", 0, nzg - 1)
    # the coherence number is the SHARP diagnostic (the energy distance is
    # dominated by the marginals, which are identical here by construction)
    assert g_res[key]["coherence_model"] > 0.99, "over-coherent model must show it"
    assert g_res[key]["coherence_truth"] < 0.6, g_res[key]["coherence_truth"]
    assert abs(g_mat[key]["coherence_model"]
               - g_mat[key]["coherence_truth"]) < 0.15, "matched must match"
    # and the tier must RANK them correctly: an over-coherent model scores
    # worse than one whose cross-epoch coherence is right, even though the
    # two have identical single-epoch marginals (which is exactly why no
    # per-epoch tier can separate them)
    assert g_res[key]["energy_ratio"] > 1.5, g_res[key]["energy_ratio"]
    assert g_res[key]["energy_ratio"] > 1.3 * g_mat[key]["energy_ratio"], \
        (g_res[key]["energy_ratio"], g_mat[key]["energy_ratio"])
    # identity is at the floor, and model sizes really do come from the MODEL
    g_id = growth_planes(truth_g, truth_g, Rg)
    assert g_id[key]["energy_ratio"] < 2.0, g_id[key]["energy_ratio"]
    assert abs(g_id[key]["coherence_model"]
               - g_id[key]["coherence_truth"]) < 1e-9
    # Mtot is degenerate when the model's totals ARE the data (the pinned
    # kernel family) — the documented caveat, pinned here
    m_pin = truth_g * 1.0
    m_pin = m_pin * (truth_g[..., -1:] / m_pin[..., -1:])
    assert growth_planes(m_pin, truth_g, Rg)[("Mtot", 0, nzg - 1)][
        "energy_ratio"] < 2.0, "pinned totals must make the Mtot plane trivial"
    # a single-epoch input must not raise, just return nothing to score
    assert growth_planes(truth_g[:, :1], truth_g[:, :1], Rg) == {}

    # --- tier 2d: the mass-size plane must use each side's OWN sizes, so a
    # model with correct masses but SYSTEMATICALLY WRONG sizes is flagged
    # (under the truth-shared-R_half convention it would score perfectly)
    s_off = _assign(np.repeat(rg.normal(size=(ng, 1)), nzg, axis=1)) * 1.6
    model_off = _cog_from_size(s_off)
    sp_id = size_planes(truth_g, truth_g, Rg)
    sp_off = size_planes(model_off, truth_g, Rg)
    assert sp_id[("R50", 0)][1]["energy_ratio"] < 2.0
    assert abs(sp_id[("R50", 0)][1]["median_dlogR"]) < 1e-9
    assert sp_off[("R50", 0)][1]["median_dlogR"] > 0.15, "1.6x offset must show"
    assert sp_off[("R50", 0)][1]["energy_ratio"] > \
        3.0 * sp_id[("R50", 0)][1]["energy_ratio"], \
        "tier 2d must flag a size-offset model that has the RIGHT masses"
    # a UNIFORM size offset moves R50, R80 and R90 together...
    for k in ("R50", "R80", "R90"):
        assert sp_off[(k, 0)][1]["median_dlogR"] > 0.15, k
    # ...and a UNIFORM dilation moves them by the SAME amount, so the
    # R50-vs-R90 GAP (not either alone) is what implicates the inner region
    g_off = (sp_off[("R50", 0)][1]["median_dlogR"]
             - sp_off[("R90", 0)][1]["median_dlogR"])
    assert abs(g_off) < 0.01, f"uniform dilation must not open a gap: {g_off}"
    # an inner-localized deficit at fixed total DOES open one. Suppress the
    # cumulative on a scale comparable to R50, taper to zero outside, then
    # restore the total. (Scaling the cumulative inside some radius instead
    # would be a no-op on R50 whenever R50 lies outside it — the curve
    # beyond is untouched; that mistake cost a debugging cycle.)
    sup = 1.0 - 0.25 * np.exp(-(Rg / 12.0) ** 2)
    inner_poor = truth_g * sup[None, None, :]
    inner_poor = inner_poor * (truth_g[..., -1:] / inner_poor[..., -1:])
    sp_in = size_planes(inner_poor, truth_g, Rg)
    d50 = sp_in[("R50", 0)][1]["median_dlogR"]
    d80 = sp_in[("R80", 0)][1]["median_dlogR"]
    d90 = sp_in[("R90", 0)][1]["median_dlogR"]
    assert d50 > d80 > d90 > 0.0, (d50, d80, d90)
    assert d50 - d90 > 0.004, f"inner deficit must open an R50-R90 gap: {d50-d90}"
    # enclosed_radius is monotone in the fraction, and R50 is half_mass_radius
    c0 = truth_g[0, 0]
    rr = [enclosed_radius(c0, Rg, f) for f in (0.2, 0.5, 0.8, 0.9)]
    assert rr[0] < rr[1] < rr[2] < rr[3], rr
    assert abs(rr[1] - half_mass_radius(c0, Rg)) < 1e-12
    # --- tier 2d GATE (the user's D1): R20 is scored, and the gate reads the
    # DISTRIBUTION at fixed mass, not just its median
    assert ("R20", 0) in sp_id, "R20 must be in the size planes"
    g_id, n_ok, n_all = size_gate(sp_id, [0.0] * nzg)
    assert n_ok == n_all, f"the identity must pass every size gate: {n_ok}/{n_all}"
    g_off, n_ok_off, _ = size_gate(sp_off, [0.0] * nzg)
    assert n_ok_off == 0 and all(not v["pass_offset"] for v in g_off.values()), \
        "a 1.6x size offset must fail every gate ON OFFSET"
    # a model with the RIGHT median size but sizes 40% too alike at fixed mass:
    # shrink each galaxy's log-size residual about the population's size-mass
    # line, at fixed total mass -- the failure `fitting-harder-narrows-the-
    # population` describes. It must fail on WIDTH and pass on offset.
    # NOT via `_assign`: that rank-maps onto FIXED marginals and so erases any
    # width change by construction (a first draft did, and measured a width
    # ratio of 1.0001 for a 40% narrowing).
    s_narrow = np.empty_like(s_inc)
    for j in range(nzg):
        lt_m = np.log10(truth_g[:, j, -1])
        ls = np.log10(s_inc[:, j])
        bline = np.polyfit(lt_m, ls, 1)
        s_narrow[:, j] = 10.0 ** (np.polyval(bline, lt_m) + 0.6 * (ls - np.polyval(bline, lt_m)))
    model_narrow = _cog_from_size(s_narrow)
    sp_nar = size_planes(model_narrow, truth_g, Rg)
    g_nar, _, _ = size_gate(sp_nar, [0.0] * nzg, fractions=(0.5,))
    v = g_nar[("R50", 0)]
    assert v["pass_offset"], f"narrowing must not move the median: {v['offset']}"
    assert not v["pass_width"] and v["width_ratio"] < 0.8, \
        f"a 40% narrowing at fixed mass must fail the WIDTH gate: {v['width_ratio']}"
    # conditioning on an external variable changes the x-axis and nothing else
    ext = np.log10(truth_g[:, :, -1])
    sp_ext = size_planes(truth_g, truth_g, Rg, cond_x=ext)
    assert abs(sp_ext[("R50", 0)][1]["median_dlogR"]) < 1e-9
    assert sp_ext[("R50", 0)][1]["width_ratio"] == 1.0

    # --- tier 2e: the mass DISTRIBUTIONS -----------------------------------
    rg2 = np.random.default_rng(5)
    base = rg2.normal(11.0, 0.25, 3000)
    # identity sits at the floor on both statistics
    d_id = cdf_distances(base, base.copy())
    assert d_id["ks_ratio"] < 1.5 and d_id["w1_ratio"] < 1.5, d_id
    # a pure SHIFT is caught by both
    d_sh = cdf_distances(base, base + 0.12)
    assert d_sh["ks_ratio"] > 5 and d_sh["w1_ratio"] > 5, d_sh
    # a too-TIGHT distribution with the right median is caught: this is the
    # kernel's actual failure mode, and tiers 1-2 (median bias + dex
    # scatter) report the median as perfect
    d_tt = cdf_distances(base, 11.0 + 0.4 * (base - 11.0))
    assert d_tt["ks_ratio"] > 5, d_tt
    # KS and W1 are NOT redundant, and the case that separates them is the
    # one this project cares about: a small fraction of galaxies placed FAR
    # from where they belong (a missing tail). KS only sees the fraction —
    # capped at ~0.10 here — while W1 sees the distance as well.
    tail = base.copy()
    tail[:300] += 3.0                       # 10% displaced by 3 dex
    d_tl = cdf_distances(base, tail)
    assert d_tl["w1_ratio"] > 3.0 * d_tl["ks_ratio"], \
        (d_tl["ks_ratio"], d_tl["w1_ratio"])
    # THE CAVEAT, pinned: tier 2e is PAIRING-BLIND, exactly like
    # plane_energy. A "model" that emits the truth's own values in a
    # different order scores perfectly, so this tier can never say whether
    # the right galaxy got the right profile.
    d_shuf = cdf_distances(base, rg2.permutation(base))
    assert d_shuf["ks_ratio"] < 1.5, d_shuf
    assert d_shuf["ks"] < 1e-12, "a permutation is the SAME sample"
    # and it survives NaNs (a quantity that left the measured grid)
    holed = base.copy()
    holed[:300] = np.nan
    assert np.isfinite(cdf_distances(base, holed)["ks_ratio"])

    res2 = evaluate(1.1 * truth, truth, R, [0.4, 1.0, 2.0], name="x1.1",
                    verbose=False, figures=False)
    for k in res2["keys"]:
        re = relerr(res2["model"][k], res2["truth"][k])
        nonzero = res2["truth"][k] > 1.0        # the truth-floor clip guards ~empty bins
        assert np.allclose(re[nonzero], 0.1, atol=1e-6), k
        d = np.log10(res2["model"][k][nonzero]) - np.log10(res2["truth"][k][nonzero])
        assert np.allclose(d, np.log10(1.1), atol=1e-9), k
    assert np.allclose(res2["mr_all"], 0.1, atol=1e-9)
    mr_rmin = profile_maxrel(1.1 * truth, truth, R, rmin=RMIN_KPC)
    assert np.allclose(mr_rmin, 0.1, atol=1e-9)
    # tier 2e must be present in the standard report
    assert "cdfs" in res2 and (CDF_KEYS[0], 0) in res2["cdfs"]

    print("qa.demo OK: identity exact, +10% scaling -> +10% bias everywhere, "
          "0 dex scatter, 10% max|rel| (all-R and R>5); energy distance: 0 on "
          f"identity; shift {shift['energy_ratio']:.0f} (centered "
          f"{shift['energy_ratio_centered']:.1f}); resample "
          f"{same['energy_ratio']:.1f}; over-tight centered "
          f"{tight['energy_ratio_centered']:.0f}; tier 2e KS/floor: "
          f"identity {d_id['ks_ratio']:.1f}, shift {d_sh['ks_ratio']:.0f}, "
          f"too-tight {d_tt['ks_ratio']:.0f}, shuffled {d_shuf['ks_ratio']:.1f} "
          "(pairing-blind, as documented)")


if __name__ == "__main__":
    demo()
