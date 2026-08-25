"""exp57 — the preregistered gates, ported from exp52's diagnosis.

**These decide the experiment, and the loss does not.** exp52 measured a model
that reproduced the summed profile at all five epochs to 10-15% while getting
its growth MECHANISM wrong: by redshift 0.7 to 0.4, 96.5% of its outskirt
growth was redistribution of stars it already had. The summed curve of growth
cannot see that. These functions can.

Every threshold was fixed in `doc/plans/2026-08-26-exp57-expansion-term.md`
BEFORE anything was fitted, and every one is anchored on a value exp52 actually
measured on the failing model rather than on a number chosen to be passable.

    G1  the mechanism split      expansion must not out-grow new deposition
                                 below z = 1.2      (exp52 failure: 76.9%, 96.5%)
    G2  the reach                <= 25% of displaced mass beyond 50 kpc,
                                 <= 10% beyond 148  (exp52 failure: 58.5%, 29.4%)
    G3  no invisible deletion    expansion must not buy score_A by pushing mass
                                 past 100 kpc, where score_F is normalised and
                                 therefore blind
    G1b the MEASURED companion   fraction of added mass landing beyond 50 kpc,
                                 model against data, same operator on both

G1b is the one with no threshold of mine in it at all: it computes the same
quantity from the model and from the measured curves of growth with the same
code path, so the model number is judged against the data number.

THE SPLIT, exactly. For an epoch pair (early -> late), separate the deposits
into those already present at the earlier epoch (`old`) and those that arrive
between the two (`new`). Then the change in the mass of any shell is exactly

    dM = sum_old dm_j [S_j(late) - S_j(early)]     EXPANSION  (same material)
       + sum_new dm_j  S_j(late)                   NEW DEPOSITION

where `S_j(k)` is deposit `j`'s mass fraction in that shell as seen at epoch
`k`. There is no third term: exp54 has no `M(<500 kpc)` pinning, so the
renormalisation term that manufactured 89% of the first model's growth does not
exist here. `dm_j` does not depend on the epoch, so the expansion term really is
the same material moving.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", EXP54, HERE):
    sys.path.insert(0, str(p))

import expand as X                                       # noqa: E402
import fit as F                                          # noqa: E402
import model as M                                        # noqa: E402

Z = (0.4, 0.7, 1.0, 1.5, 2.0)
#: (earlier, later) as EPOCH INDICES. Index 0 is z = 0.4, so time runs backward
#: through the index and "earlier" is the LARGER one.
ADJACENT = ((4, 3), (3, 2), (2, 1), (1, 0))
#: the long baseline, z = 2 -> 0.4
LONG = (4, 0)
#: shell edges for the reach measurement, in kpc. Reaches far past the measured
#: grid on purpose: exp52's failing model delivered 12% of the mass it moved
#: beyond 500 kpc, where the normalisation then discarded it, and a gate that
#: cannot see past 148 kpc could not have caught that.
REACH_EDGES = (0.0, 5.0, 10.0, 30.0, 50.0, 100.0, 148.0, 300.0, 500.0, 1e7)
#: where score_F stops being able to see, because it normalises there
R_BLIND = 100.0
#: exp52's measured failure values, quoted so a reader never has to look them up
EXP52_SHARE = {"z=1.0 -> 0.7": 0.769, "z=0.7 -> 0.4": 0.965}
EXP52_REACH_50, EXP52_REACH_148 = 0.585, 0.294
#: the thresholds, fixed in the plan before any fit
G2_MAX_BEYOND_50, G2_MAX_BEYOND_148 = 0.25, 0.10
#: the redshift below which expansion must not out-grow new deposition. exp52's
#: crossover sat at z ~ 1.2, exactly where minor-merger accretion should take
#: over, which is why that is the line.
G1_Z_LIMIT = 1.2


def deposit_terms(pr, theta, R):
    """Per-deposit weights and CoGs on an arbitrary radius grid.

    Returns `(dm, present, B)` with

        dm       (n, nd)          each deposit's stellar mass, epoch-independent
        present  (nE, n, nd) bool whether the deposit exists at that epoch
        B        (nE, n, nd, nR) its unit-mass CoG as seen at that epoch

    This repeats `ExpandingProblem.predict`'s setup rather than calling it,
    because the whole point is to keep the deposits SEPARATE instead of summing
    them. `check_terms` asserts that summing these reproduces `predict`, so the
    duplication cannot drift.
    """
    sp = pr.xspec
    eff, size, shape, xtheta = sp.unpack(theta)
    d = pr.dep
    n, nd = d.z.shape
    R = np.asarray(R, float)

    ok = np.isfinite(d.r200c) & (d.r200c > 0)
    dm = 10.0 ** np.clip(M.log_eps(sp.base, eff, d), -30, 10) * d.dmh
    r50 = M.r50_of(sp.base, size, d)
    r_tr = sp.base.trunc_C * d.r200c
    good = ok & np.isfinite(dm) & (dm >= 0) & np.isfinite(r50) & (r50 > 0)
    r50s = np.where(good, r50, 1.0).ravel()
    r_trs = np.where(good, r_tr, 100.0).ravel()
    td = np.where(np.isfinite(pr.t_dep), pr.t_dep, 1.0).ravel()

    nE = len(pr.epochs)
    B = np.empty((nE, n, nd, len(R)))
    for k in range(nE):
        g = X.g_factor(sp.law, xtheta, td, pr.t_view[k])
        B[k] = X.cog_scaled(sp.base.family, shape, r50s, r_trs, R,
                            g).T.reshape(n, nd, len(R))
    return (np.where(good, dm, 0.0), pr.em.transpose(1, 0, 2).astype(bool)
            if pr.em.ndim == 3 else pr.em, B)


def check_terms(pr, theta, tol=1e-9):
    """Summing the separated deposits MUST reproduce `predict`, or stop.

    Without this the decomposition could quietly describe a different model
    than the one that was fitted — which is precisely how exp52's own headline
    went wrong once (an aperture growth divided by an annulus growth).
    """
    dm, present, B = deposit_terms(pr, theta, F.R_GRID)
    got = np.einsum("nd,knd,kndr->nkr", dm, present.astype(float), B)
    want = pr.predict(theta)
    ok = np.isfinite(want)
    rel = float(np.max(np.abs(got[ok] - want[ok])
                       / np.maximum(np.abs(want[ok]), 1.0)))
    if rel > tol:
        raise SystemExit(
            f"REFUSING TO RUN: the deposit decomposition does not reproduce "
            f"the model it decomposes (max relative {rel:.3e}). Every gate "
            f"below would describe a different model than the one fitted.")
    return rel


def shell_mass(dm, present, B, k, i_in, i_out, which=None):
    """Total model mass in shell [i_in, i_out) at epoch `k`, per galaxy."""
    w = dm * present[k]
    if which is not None:
        w = w * which
    return np.einsum("nd,nd->n", w, B[k, :, :, i_out] - B[k, :, :, i_in])


def g1_mechanism(pr, theta, edges=(50.0, 100.0, 148.0)):
    """G1 — how much of each shell's growth is expansion, and how much is new?

    Returns one row per (epoch pair, shell) with the two terms and their share.
    """
    R = np.array([0.0] + list(edges))
    dm, present, B = deposit_terms(pr, theta, R)
    out = []
    for ke, kl in ADJACENT + (LONG,):
        old = present[ke]
        new = present[kl] & ~present[ke]
        for a in range(1, len(R) - 1):
            i_in, i_out = a, a + 1
            exp_t = float(np.sum(
                dm * old * (B[kl, :, :, i_out] - B[kl, :, :, i_in]
                            - B[ke, :, :, i_out] + B[ke, :, :, i_in])))
            new_t = float(np.sum(
                dm * new * (B[kl, :, :, i_out] - B[kl, :, :, i_in])))
            tot = exp_t + new_t
            out.append(dict(
                pair=f"z={Z[ke]} -> {Z[kl]}", z_late=Z[kl],
                shell=f"M[{R[i_in]:g},{R[i_out]:g}]",
                expansion=exp_t, new_deposition=new_t, total=tot,
                share=exp_t / tot if abs(tot) > 0 else np.nan))
    return out


def g2_reach(pr, theta, edges=REACH_EDGES):
    """G2 — where does expansion take mass FROM, and where does it deliver it?

    Measured on the material already assembled at the earlier epoch, which is
    the only material expansion can move. Negative entries are drained, positive
    delivered; because homologous expansion conserves each deposit's mass, the
    two balance to rounding, and `residual` reports that as a self-check.
    """
    R = np.array(edges)
    dm, present, B = deposit_terms(pr, theta, R)
    out = []
    for ke, kl in ADJACENT + (LONG,):
        old = present[ke]
        d_shell = np.array([
            float(np.sum(dm * old * (B[kl, :, :, a + 1] - B[kl, :, :, a]
                                     - B[ke, :, :, a + 1] + B[ke, :, :, a])))
            for a in range(len(R) - 1)])
        drained = float(-d_shell[d_shell < 0].sum())
        delivered = float(d_shell[d_shell > 0].sum())
        mid = np.array(R[1:])
        beyond50 = float(d_shell[(mid > 50.0) & (d_shell > 0)].sum())
        beyond148 = float(d_shell[(mid > 148.0) & (d_shell > 0)].sum())
        beyond500 = float(d_shell[(mid > 500.0) & (d_shell > 0)].sum())
        src_in10 = float(-d_shell[:2][d_shell[:2] < 0].sum())
        out.append(dict(
            pair=f"z={Z[ke]} -> {Z[kl]}", z_late=Z[kl], shells=d_shell,
            drained=drained, delivered=delivered,
            residual=(delivered - drained) / max(drained, 1e-30),
            f_beyond_50=beyond50 / max(delivered, 1e-30),
            f_beyond_148=beyond148 / max(delivered, 1e-30),
            f_beyond_500=beyond500 / max(delivered, 1e-30),
            f_source_inside_10=src_in10 / max(drained, 1e-30),
            moved_fraction=drained / max(float(np.sum(dm * old)), 1e-30)))
    return out


def g3_deletion(pr, theta, r_blind=R_BLIND):
    """G3 — how much deposited mass sits where `score_F` cannot see it?

    `score_F` normalises every profile at 100 kpc, so it is blind to everything
    beyond. exp54 over-supplies its own mass growth by 22%, and expansion moves
    mass outward: the concern is that expansion is fitted as an amplitude
    regulator disposing of that excess where nothing scores it. This measures
    the fraction of deposited stellar mass beyond 100 kpc at every epoch, which
    a fit doing that would have to raise.
    """
    R = np.array([0.0, r_blind, 1e7])
    dm, present, B = deposit_terms(pr, theta, R)
    out = []
    for k in range(len(pr.epochs)):
        w = dm * present[k]
        tot = float(np.sum(w))
        beyond = float(np.sum(w * (1.0 - B[k, :, :, 1])))
        out.append(dict(z=Z[k], f_beyond_blind=beyond / max(tot, 1e-30),
                        deposited=tot))
    return out


def g1b_measured(pr, theta, data, mask, r_split=50.0, r_out=148.0):
    """G1b — the MEASURED companion: where does added mass actually land?

    The fraction of a galaxy's mass growth inside `r_out` that lands beyond
    `r_split`, computed for the model and for the measured curves of growth
    with the **same operator**, so the model number is judged against the data
    number and no threshold of mine enters. exp53 called this the
    "GATE differential" and measured **0.37** on the data.

    Population-summed rather than per galaxy: a per-galaxy ratio is undefined
    wherever a galaxy's measured growth is near zero, and 42% of these galaxies
    have a falling centre.
    """
    i_s = int(np.argmin(np.abs(F.R_GRID - r_split)))
    i_o = int(np.argmin(np.abs(F.R_GRID - r_out)))
    mod = pr.predict(theta)
    out = []
    for ke, kl in ADJACENT + (LONG,):
        use = (mask[:, ke] & mask[:, kl]
               & np.isfinite(mod[:, ke, i_o]) & np.isfinite(mod[:, kl, i_o]))

        def frac(cube):
            d_out = cube[use, kl, i_o] - cube[use, ke, i_o]
            d_in = cube[use, kl, i_s] - cube[use, ke, i_s]
            tot = float(np.sum(d_out))
            return float(np.sum(d_out - d_in)) / tot if abs(tot) > 0 else np.nan

        out.append(dict(pair=f"z={Z[ke]} -> {Z[kl]}", z_late=Z[kl],
                        model=frac(mod), data=frac(data), n=int(use.sum())))
    return out


def verdict(g1, g2, g1b):
    """Apply the preregistered thresholds. Returns (passed, list of reasons)."""
    fails = []
    for r in g1:
        if r["z_late"] < G1_Z_LIMIT and r["share"] > 0.5 and r["total"] > 0:
            fails.append(
                f"G1: at {r['pair']} the {r['shell']} shell grows "
                f"{100 * r['share']:.1f}% by expansion, above new deposition, "
                f"below z = {G1_Z_LIMIT}")
    for r in g2:
        if r["moved_fraction"] < 1e-10:      # nothing moved: ratios undefined
            continue
        if r["f_beyond_50"] > G2_MAX_BEYOND_50:
            fails.append(f"G2: at {r['pair']}, "
                         f"{100 * r['f_beyond_50']:.1f}% of the displaced mass "
                         f"is delivered beyond 50 kpc (limit "
                         f"{100 * G2_MAX_BEYOND_50:.0f}%)")
        if r["f_beyond_148"] > G2_MAX_BEYOND_148:
            fails.append(f"G2: at {r['pair']}, "
                         f"{100 * r['f_beyond_148']:.1f}% beyond 148 kpc "
                         f"(limit {100 * G2_MAX_BEYOND_148:.0f}%)")
    return (not fails), fails


def g6_outskirt(pr, theta, data, mask, r_in=50.0, r_out=148.0):
    """G6 — the outskirt STRUCTURE: how much stellar halo is there?

    G1b asks where a galaxy's newly ADDED mass lands. This asks the
    complementary question about the standing structure: does the model put the
    right amount of mass in the 50-148 kpc annulus, at each epoch? It is
    reported, not gated, because the null defines the reference and no
    threshold was preregistered for it.

    It matters here for one specific reason. exp54 Stage 3.7 found that closing
    the central deficit with a shared size law cost **0.104 dex of outskirt
    mass at redshift 2** — one radial scale could not do both ends. Expansion
    is a different mechanism aimed at the same defect, so the natural question
    is whether it pays the same price. This is the measurement that answers it.
    """
    i_i = int(np.argmin(np.abs(F.R_GRID - r_in)))
    i_o = int(np.argmin(np.abs(F.R_GRID - r_out)))
    mod = pr.predict(theta)
    out = []
    for k in range(len(pr.epochs)):
        m = mod[:, k, i_o] - mod[:, k, i_i]
        d = data[:, k, i_o] - data[:, k, i_i]
        use = mask[:, k] & np.isfinite(m) & (m > 0) & (d > 0)
        err = np.log10(m[use]) - np.log10(d[use])
        out.append(dict(z=Z[k], median=float(np.median(err)),
                        n=int(use.sum()),
                        summed=float(np.log10(np.sum(m[use])
                                              / np.sum(d[use])))))
    return out
