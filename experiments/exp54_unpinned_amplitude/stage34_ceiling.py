"""exp54 Stage 3.4 — THE PROFILE-SHAPE REPRESENTATIONAL CEILING (v2, corrected).

Stage 3.0 computed a ceiling for the AMPLITUDE and correctly killed a survival
term before anyone built it. This asks the same question of the SHAPE.

THE EXPERIMENT. For one galaxy the model's deposits form a FIXED BASIS: one
truncated unit-mass cumulative profile `F_j(<R)` per MAH step, with `R50` from
the fitted size law and truncation at `3 R200c(t_j)`. The model chooses the
weights by a seven-parameter global law, `dM*_j = eps_surv(H_j) dMh_j`. Ask
instead: over ALL non-negative weights `w_j >= 0`, respecting causality (`w_j`
enters epoch `k` only if `t_j <= t_k`), what is the best fit to that galaxy's
measured curve of growth? That is non-negative least squares -- convex, no
optimiser tuning, milliseconds per galaxy.

--------------------------------------------------------------------------- #
WHAT CHANGED IN v2, AND WHY. An independent review found four defects in v1,
all of which are fixed here. v1's output is preserved as
`outputs/stage34_ceiling_v1_provisional.npz`.

  1. **The mask was applied only to the SCORE, never to the SOLVE.** v1 fitted
     every galaxy on all five epochs and then scored it on the per-epoch
     mh-complete mask, so a galaxy counted only at z=0.4 still had its excluded
     z=2 profile pulling on its weights. That is not the optimum of the problem
     it was compared against. v2 takes an explicit `use_epochs` mask INTO the
     solve and reports three sample definitions side by side (`SAMPLES`).

  2. **`increment_test` minimised one quantity and reported another.** The
     non-negative solve was weighted by the increment itself; the cost was
     reported as a fraction of the later epoch's `M*(<R)`. v2 solves in the
     units it reports, which is the only way the number is a minimum.

  3. **The five-interval rung is a TAUTOLOGY at z=2.** Every pre-z=2 deposit
     sits in one interval, so rescaling that interval's single amplitude cannot
     change a profile normalised at 100 kpc: its z=2 `score_F` must equal the
     fitted model's exactly, and v1 read that identity as "zero headroom".
     v2 replaces it with a TEMPORAL-RESOLUTION LADDER, `K` contiguous deposit
     groups for K = 1, 2, 3, 5, 8, 12, 20 and free, which puts several
     independent components before z=2 and measures how the bound falls as a
     global law is allowed finer time resolution.

  4. **The monotone reference is not a minimum of anything.** v1 used an
     unweighted isotonic projection in log10 mass. v2 adds `monotone_obj`, a
     WEIGHTED pool-adjacent-violators projection in linear mass with weights
     `1/d^2`, which is the exact minimiser of the same absolute-fractional
     surrogate the NNLS minimises, and therefore comparable to it. Both are
     reported; neither is called a floor in the production objective.
     `stage34_objective.py` computes the exact production-objective bounds.

--------------------------------------------------------------------------- #
THE THREE SAMPLE DEFINITIONS (`SAMPLES`), which answer different questions:

  `all5`    solve on all five epochs, score on the mask. v1's object, kept so
            the effect of the correction is visible rather than asserted.
  `masked`  solve only on the epochs the mask admits for that galaxy, score on
            the same. Matches `stage33_perepoch.MaskedProblem` exactly. **Read
            it with care**: 542 of 2397 galaxies are admitted at ONE epoch
            only, and 24 radii against ~71 free weights is not a bound on
            anything, so the run reports the ceiling split by how many epochs
            each galaxy contributed.
  `cohort`  the 840 galaxies above the z=2 threshold, solved and scored at all
            five epochs. The same objects at every epoch, so this is the one to
            quote for statements about EVOLUTION. It is a fixed massive-
            progenitor cohort and must not be read as a general high-redshift
            population.

THE LADDER, in decreasing per-galaxy freedom:

  `monotone_log` / `monotone_obj`  the truth projected onto non-decreasing-in-
        time enclosed mass. Basis-free, so it constrains EVERY deposition-only
        model. `_log` is v1's unweighted log-mass projection; `_obj` minimises
        the NNLS surrogate exactly.
  `epoch-alone`  NNLS per epoch, no requirement that one vector serve all five.
        NOT a deposition model: it answers "can this basis draw this profile at
        all", with the causal coupling switched off.
  `ceiling` / `ceiling_fb`  the joint causal NNLS, and the same with
        `eps_surv <= f_b = Omega_b/Omega_m` (no deposit may turn more than the
        universal baryon fraction of its accreted halo mass into stars).
  `bins<K>`  K contiguous deposit groups, the within-group distribution held at
        the fitted model's. K = 1 is one per-galaxy normalisation of the global
        law; K = free is the ceiling.
  `interval5`  the five epoch intervals, kept for continuity with v1 and
        labelled with its z=2 tautology.
  `model`  the fitted seven-parameter global law, no per-galaxy freedom.

HOW TO READ A CEILING, AND HOW NOT TO. NNLS gets ~71 non-negative numbers per
galaxy where the model has seven global ones. A LARGE ceiling proves the basis
inadequate; a SMALL one does NOT prove any global law can reach it. Part D
prices that freedom: the PERMUTED control fits one galaxy's deposits directly
to a DIFFERENT galaxy's measured stellar profile. It has no halo-specific basis
information but full access to the target's stellar data, so what it measures
is how overcomplete the dictionary is, not how much a predictor could learn.

CAVEAT ON DIRECTION. NNLS minimises the summed squared FRACTIONAL residual of
the ABSOLUTE curve of growth, not the production shape loss (which renormalises
at 100 kpc and aggregates differently). So every `score_F` here is the
production score of one feasible solution optimised under a surrogate: a low
value proves the basis can reach at least that, a high value does NOT prove
better is impossible. `stage34_objective.py` removes this caveat by optimising
the exact production objective.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp54_unpinned_amplitude/stage34_ceiling.py [--smoke] [--figures]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear, nnls

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import halo as H                                         # noqa: E402
import model as M                                        # noqa: E402
import stage30 as S30                                    # noqa: E402
import stage33 as S33                                    # noqa: E402
import stage33_perepoch as S33P                          # noqa: E402
from hongshao.objective import Objective                 # noqa: E402

POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
SEL = HERE / "outputs" / "selection.npz"
PEREPOCH = HERE / "outputs" / "stage33_perepoch.npz"
STAGE33 = HERE / "outputs" / "stage33.npz"
OUT = HERE / "outputs" / "stage34_ceiling.npz"
FIGDIR = HERE / "figures"

R = F.R_GRID
I100 = F.I100
Z = (0.4, 0.7, 1.0, 1.5, 2.0)
RULE = "=" * 100

#: Omega_b/Omega_m for the TNG300 cosmology. The cap on `eps_surv`: a deposit
#: cannot turn more baryons into surviving central stars than it accreted.
F_B = 0.0486 / 0.3089

#: which basis to bound. The basis depends on theta ONLY through the size law
#: (`log_f0`, `b`) and the profile shape parameter, so the efficiency
#: parameters of the source fit are irrelevant here -- NNLS replaces them.
BASES = (
    ("gompertz_log-E2-S2", "perepoch"),      # the adopted model, mh-complete fit
    ("moffat-E2-S2", "stage33"),             # the family control, full-sample fit
)
SAMPLES = ("all5", "masked", "cohort")
#: temporal-resolution ladder: K contiguous deposit groups; 0 means free weights
K_LADDER = (1, 2, 3, 5, 8, 12, 20, 0)
#: radius at which "the central mass declined" is preregistered, and its backup
DECLINE_R = 4.92
DECLINE_R2 = 2.76

LADDER = (("monotone_log", "monotone_obj", "epoch-alone", "ceiling",
           "ceiling_fb", "interval5") + tuple(f"bins{k}" for k in K_LADDER)
          + ("model",))
FREEDOM = {"monotone_log": "unbounded*", "monotone_obj": "unbounded*",
           "epoch-alone": "5 x ~71", "ceiling": "~71", "ceiling_fb": "~71 cap",
           "interval5": "5 (epoch)", "model": "0 (7 global)"}
for _k in K_LADDER:
    FREEDOM[f"bins{_k}"] = "~71 (free)" if _k == 0 else f"{_k}"


# --------------------------------------------------------------------------- #
# scoring                                                                      #
# --------------------------------------------------------------------------- #
def _score_masked(pred, data, mask, obj=None):
    """(score_A[k], score_F[k]) for an ARBITRARY prediction, under the mask.

    A transcription of `stage33_perepoch.MaskedProblem.per_epoch` that takes a
    prediction array instead of a theta, so a bound with no parameter vector is
    scored on exactly the footing the model is. `_gate_scoring` asserts they
    agree; without that the comparison is meaningless.
    """
    obj = Objective() if obj is None else obj
    finite = np.isfinite(pred).all(axis=2) & (pred[:, :, I100] > 0)
    use = np.asarray(mask, bool) & finite
    A_m = np.log10(np.clip(pred[:, :, I100], 1.0, None))
    A_d = np.log10(data[:, :, I100])
    dA = (A_m - A_d) / F.SIGMA_A[None, :]
    F_m = pred / np.clip(pred[:, :, I100], 1.0, None)[:, :, None]
    F_d = data / data[:, :, I100][:, :, None]
    sa, sf = [], []
    for k in range(pred.shape[1]):
        g = use[:, k]
        sa.append(np.sqrt(np.mean(dA[g, k] ** 2)) if g.sum() else np.nan)
        sf.append(obj(F_m[g][:, [k]], F_d[g][:, [k]], R) / F.L_F_REF
                  if g.sum() else np.nan)
    return np.array(sa), np.array(sf)


def per_galaxy_shape_rms(pred, data):
    """(n, 5) RMS fractional error of the ABSOLUTE curve of growth.

    Reported alongside the population score because the whole experiment is
    about fitting individual galaxies, and a population mean of a per-galaxy
    RMS hides its distribution.
    """
    return np.sqrt(np.mean(((pred - data) / np.clip(data, 1.0, None)) ** 2,
                           axis=2))


def _gate_scoring(spec, theta, recs, data, mask):
    """`_score_masked` MUST reproduce `MaskedProblem.per_epoch`, or stop."""
    pr = S33P.MaskedProblem(spec, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    sa0, sf0 = pr.per_epoch(theta)
    sa1, sf1 = _score_masked(pr.predict(theta), data, mask)
    da, df = np.nanmax(np.abs(sa1 - sa0)), np.nanmax(np.abs(sf1 - sf0))
    print(f"  SCORING GATE (`_score_masked` vs `MaskedProblem.per_epoch`)")
    print(f"    max |d score_A| = {da:.2e}     max |d score_F| = {df:.2e}")
    if not (da < 1e-12 and df < 1e-12):
        raise SystemExit(
            "REFUSING TO RUN: a ceiling scored by this module cannot be "
            "compared with a model scored by MaskedProblem. Fix the "
            "transcription, not the tolerance.")
    print(f"    OK -- bounds and model are scored by identical arithmetic\n")


# --------------------------------------------------------------------------- #
# the basis                                                                    #
# --------------------------------------------------------------------------- #
def build_basis(spec, theta, h):
    """(B, epoch_mask, dMh, good) for one galaxy: the model's own deposits."""
    _, size, shape = spec.unpack(theta)
    ok = np.isfinite(h.r200c) & (h.r200c > 0)
    r50 = M.r50_of(spec, size, h)
    good = ok & np.isfinite(r50) & (r50 > 0)
    B = M.cog_truncated(spec.family, shape, r50[good],
                        spec.trunc_C * h.r200c[good], R)
    return B, h.epoch_mask[:, good], h.dmh[good], good


def model_weights(spec, theta, h, good):
    """The weights the fitted global law itself chooses, `eps_surv * dMh`."""
    eff, _, _ = spec.unpack(theta)
    return 10.0 ** np.clip(M.log_eps(spec, eff, h), -30, 10)[good] * h.dmh[good]


def _design(B, em):
    """(5*nR, nd) causal design: deposit `j` enters epoch `k` iff `t_j <= t_k`."""
    return np.concatenate([B * em[k][None, :] for k in range(em.shape[0])],
                          axis=0)


def _rows_of(use, n_r=len(R)):
    """Row indices of the epoch blocks admitted by `use`."""
    return np.concatenate([np.arange(k * n_r, (k + 1) * n_r)
                           for k in range(len(use)) if use[k]])


def _solve(A, y, ub=None, row_w=None):
    """argmin ||row_w * (A w - y)||^2 over `0 <= w <= ub`.

    `row_w` defaults to `1/y`, i.e. fractional residuals. Columns are scaled by
    `ub` in the bounded solve because the raw problem spans 1e10 in the
    variables and 1e-11 in the matrix, and BVLS on that returns nonsense.
    """
    w_row = (1.0 / np.clip(y, 1.0, None)) if row_w is None else row_w
    Aw, yw = A * w_row[:, None], y * w_row
    if ub is None:
        w, _ = nnls(Aw, yw)
        return w
    s = np.maximum(ub, 1.0)
    r = lsq_linear(Aw * s[None, :], yw, bounds=(0.0, 1.0), method="bvls",
                   max_iter=2000)
    return r.x * s


# --------------------------------------------------------------------------- #
# the monotone references                                                      #
# --------------------------------------------------------------------------- #
def weighted_pava(y, w):
    """Weighted least-squares projection onto non-decreasing sequences.

    Pool-adjacent-violators with weights, which is the exact minimiser of
    `sum_k w_k (y_k - p_k)^2` subject to `p` non-decreasing. Taking
    `w_k = 1/y_k^2` on LINEAR masses makes it the exact minimiser of the same
    summed squared FRACTIONAL residual that `_solve` minimises -- so
    `monotone_obj` and `ceiling` are then bounds on the same functional, which
    `monotone_log` (Stage 3.0's unweighted log-mass projection) is not.
    """
    v, ww = list(map(float, y)), list(map(float, w))
    cnt = [1] * len(v)
    i = 0
    while i < len(v) - 1:
        if v[i] <= v[i + 1] + 1e-15:
            i += 1
            continue
        tot = ww[i] + ww[i + 1]
        v[i:i + 2] = [(v[i] * ww[i] + v[i + 1] * ww[i + 1]) / tot]
        ww[i:i + 2] = [tot]
        cnt[i:i + 2] = [cnt[i] + cnt[i + 1]]
        i = max(i - 1, 0)
    out = []
    for val, c in zip(v, cnt):
        out += [val] * c
    return np.array(out)


def monotone_reference(data, use_epochs, kind="obj"):
    """(n, 5, nR) the truth projected onto non-decreasing-in-time enclosed mass.

    Only the epochs a galaxy actually contributes are constrained; the rest are
    filled from the nearest constrained epoch, so an excluded epoch cannot
    change where the projection places its error.
    """
    n, ne, nr = data.shape
    out = np.array(data, float)
    for i in range(n):
        ks = [k for k in range(ne) if use_epochs[i, k]]
        if len(ks) < 2:
            continue
        order = ks[::-1]                       # earliest (highest z) first
        for c in range(nr):
            y = data[i, order, c]
            if kind == "log":
                p = weighted_pava(np.log10(np.clip(y, 1.0, None)),
                                  np.ones(len(y)))
                p = 10.0 ** p
            else:
                p = weighted_pava(y, 1.0 / np.clip(y, 1.0, None) ** 2)
            out[i, order, c] = p
    return out


# --------------------------------------------------------------------------- #
# the rungs                                                                    #
# --------------------------------------------------------------------------- #
def interval_groups(em):
    """Group index per deposit, one group per interval between epochs."""
    ne = em.shape[0]
    grp = np.full(em.shape[1], -1, int)
    grp[em[ne - 1]] = 0
    for k in range(ne - 2, -1, -1):
        grp[em[k] & (grp < 0)] = ne - 1 - k
    return grp


def count_groups(nd, k):
    """K contiguous groups of equal deposit count, earliest first.

    Equal COUNT rather than equal time, so the pre-z=2 half of the history --
    which is where the model's failure lives and where the epoch-interval
    grouping has exactly one component -- receives its proportional share of
    independent amplitudes. With 72 deposits and 33 before z=2, K = 8 puts
    about 3.7 free components before z=2.
    """
    return np.minimum((np.arange(nd) * k) // nd, k - 1)


def _grouped(A, grp, wmod, k):
    """Collapse the design to one column per group, weights from the model."""
    G = np.zeros((A.shape[0], k))
    for g in range(k):
        sel = grp == g
        if sel.any() and wmod[sel].sum() > 0:
            G[:, g] = A[:, sel] @ wmod[sel] / wmod[sel].sum()
    return G


def solve_galaxy(B, em, dmh, wmod, d, use):
    """Every rung for one galaxy. `use` (ne,) bool selects the FITTED epochs.

    Predictions are always returned at all epochs -- the weights determine them
    -- but only the admitted epochs enter the solve.
    """
    ne, nr = em.shape[0], B.shape[0]
    A = _design(B, em)
    rows = _rows_of(use, nr)
    y = d.ravel()
    out, extra = {}, {}

    w = _solve(A[rows], y[rows])
    out["ceiling"] = (A @ w).reshape(ne, nr)
    extra["w"] = w
    wfb = _solve(A[rows], y[rows], ub=F_B * dmh)
    out["ceiling_fb"] = (A @ wfb).reshape(ne, nr)
    extra["w_fb"] = wfb

    gi = interval_groups(em)
    Gi = _grouped(A, gi, wmod, ne)
    out["interval5"] = (Gi @ _solve(Gi[rows], y[rows])).reshape(ne, nr)

    for k in K_LADDER:
        if k == 0:
            out["bins0"] = out["ceiling"]
            continue
        gk = count_groups(len(wmod), k)
        Gk = _grouped(A, gk, wmod, k)
        out[f"bins{k}"] = (Gk @ _solve(Gk[rows], y[rows])).reshape(ne, nr)

    alone = np.array(d, float)
    for k in range(ne):
        Ak = B * em[k][None, :]
        alone[k] = Ak @ _solve(Ak, d[k])
    out["epoch-alone"] = alone
    return out, extra


# --------------------------------------------------------------------------- #
# diagnostics                                                                  #
# --------------------------------------------------------------------------- #
def increment_test(B, em, d, use):
    """Separate 'the truth goes DOWN' from 'the basis cannot draw the rise'.

    The causal design is block-triangular: the deposits of one epoch interval
    are the only ones that can build that interval's increase in `M*(<R)`. Two
    ways that fails, needing opposite fixes, and BOTH are now solved in the
    units they are reported in -- as an RMS error in the cumulative curve as a
    fraction of `M*(<R)` at the later epoch. (v1 solved weighted by the
    increment and reported weighted by the later total, so its basis cost was
    not the minimum of anything.)

    Returns (frac_negative, cost_negative, cost_basis) per interval, NaN where
    either endpoint epoch is not admitted for this galaxy.
    """
    ne = em.shape[0]
    grp = interval_groups(em)
    neg = np.full(ne - 1, np.nan)
    cost_neg, cost_bas = np.full(ne - 1, np.nan), np.full(ne - 1, np.nan)
    for g in range(1, ne):
        k_new, k_old = ne - 1 - g, ne - g       # epochs bracketing interval g
        if not (use[k_new] and use[k_old]):
            continue
        inc = d[k_new] - d[k_old]
        ref = np.clip(d[k_new], 1.0, None)
        neg[g - 1] = float(np.mean(inc < 0))
        cost_neg[g - 1] = float(np.sqrt(np.mean(
            (np.clip(inc, None, 0) / ref) ** 2)))
        sel = grp == g
        if not sel.any():
            continue
        tgt = np.clip(inc, 0.0, None)
        w = _solve(B[:, sel], tgt, row_w=1.0 / ref)      # SAME units as reported
        cost_bas[g - 1] = float(np.sqrt(np.mean(
            ((B[:, sel] @ w - tgt) / ref) ** 2)))
    return neg, cost_neg, cost_bas


def decline_flag(data, r=DECLINE_R):
    """Did the enclosed mass at `r` FALL between z=2 and z=0.4? Preregistered.

    A static non-negative deposition model cannot reproduce a decline at all,
    so this splits the sample into the galaxies whose central residual COULD be
    a basis problem and those whose could not.
    """
    c = int(np.argmin(np.abs(R - r)))
    return data[:, 0, c] < data[:, 4, c]


def _gal_slopes(cog, band=(50.0, 148.3)):
    """Per-galaxy outer density slope d log10 Sigma / d log10 R over the band."""
    from hongshao.profile_emulator import density_from_cog
    ls, mid = density_from_cog(np.log10(np.clip(cog, 1.0, None)), R)
    sel = (mid >= band[0]) & (mid <= band[1])
    x = np.log10(mid[sel])
    return np.array([np.polyfit(x, row[sel], 1)[0]
                     if np.isfinite(row[sel]).all() else np.nan for row in ls])


def outer_slope(cog):
    """d log10 Sigma / d log10 R over 50-148 kpc of the MEDIAN profile."""
    from hongshao.profile_emulator import density_from_cog
    ls, mid = density_from_cog(np.log10(np.clip(cog, 1.0, None)), R)
    med = np.median(ls, axis=0)
    sel = (mid >= 50.0) & (mid <= 148.3)
    return float(np.polyfit(np.log10(mid[sel]), med[sel], 1)[0])


def _boot_ci(x, n_boot=400, seed=0):
    """Median and its 16th-84th percentile bootstrap interval over galaxies."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    med = np.median(x)
    b = np.array([np.median(rng.choice(x, x.size)) for _ in range(n_boot)])
    return med, *np.percentile(b, [16, 84])


# --------------------------------------------------------------------------- #
# the run                                                                      #
# --------------------------------------------------------------------------- #
def run_sample(spec, theta, recs, data, mask, use_epochs, rows_keep,
               tag, quiet=False):
    """Every rung and diagnostic for one sample definition."""
    idx = np.asarray(rows_keep)
    recs_s = [recs[i] for i in idx]
    d_s, m_s, u_s = data[idx], mask[idx], use_epochs[idx]
    n, ne = len(recs_s), 5
    keys = ["ceiling", "ceiling_fb", "interval5", "epoch-alone", "model"] + \
           [f"bins{k}" for k in K_LADDER]
    preds = {k: np.empty((n, ne, len(R))) for k in keys}
    n_act = np.zeros(n, int)
    n_act_fb = np.zeros(n, int)
    eps_impl, eps_model, dep_z = [], [], []
    neg = np.full((n, ne - 1), np.nan)
    cost_neg = np.full((n, ne - 1), np.nan)
    cost_bas = np.full((n, ne - 1), np.nan)

    t0 = time.time()
    for i, h in enumerate(recs_s):
        B, em, dmh, good = build_basis(spec, theta, h)
        wmod = model_weights(spec, theta, h, good)
        got, extra = solve_galaxy(B, em, dmh, wmod, d_s[i], u_s[i])
        for k, v in got.items():
            preds[k][i] = v
        preds["model"][i] = (_design(B, em) @ wmod).reshape(ne, -1)
        n_act[i] = int((extra["w"] > 0).sum())
        n_act_fb[i] = int((extra["w_fb"] > 1e-9 * max(extra["w_fb"].max(), 1.0)).sum())
        eps_impl.append(extra["w"] / np.clip(dmh, 1.0, None))
        eps_model.append(wmod / np.clip(dmh, 1.0, None))
        dep_z.append(h.z[good])
        neg[i], cost_neg[i], cost_bas[i] = increment_test(B, em, d_s[i], u_s[i])
    if not quiet:
        print(f"    [{tag}] {n} galaxies in {time.time() - t0:.1f} s "
              f"({1000 * (time.time() - t0) / n:.1f} ms each)")

    # The permuted control: this galaxy's deposits, fitted to ANOTHER galaxy's
    # measured profile. The epoch mask must be the TARGET's, because the score
    # is taken against the target's data on the target's mask -- solving on the
    # host's epochs and scoring on the target's would compare two different
    # problems and was a defect in the first draft of this file.
    perm = (np.arange(n) + max(n // 3, 1)) % n
    pperm = np.empty_like(preds["ceiling"])
    for i, h in enumerate(recs_s):
        B, em, dmh, good = build_basis(spec, theta, h)
        A = _design(B, em)
        rws = _rows_of(u_s[perm[i]], len(R))
        y = d_s[perm[i]].ravel()
        pperm[i] = (A @ _solve(A[rws], y[rws])).reshape(ne, -1)

    preds["monotone_log"] = monotone_reference(d_s, u_s, kind="log")
    preds["monotone_obj"] = monotone_reference(d_s, u_s, kind="obj")
    scores = {k: _score_masked(v, d_s, m_s) for k, v in preds.items()}
    scores["permuted"] = _score_masked(pperm, d_s[perm], m_s[perm])
    scores["truth-vs-truth"] = _score_masked(d_s[perm], d_s, m_s)

    return dict(tag=tag, idx=idx, spec=spec, theta=theta, preds=preds,
                scores=scores, mask=m_s, use=u_s, data=d_s,
                rms=per_galaxy_shape_rms(preds["ceiling"], d_s),
                rms_model=per_galaxy_shape_rms(preds["model"], d_s),
                rms_alone=per_galaxy_shape_rms(preds["epoch-alone"], d_s),
                n_act=n_act, n_act_fb=n_act_fb, eps_impl=eps_impl,
                eps_model=eps_model, dep_z=dep_z, neg=neg,
                cost_neg=cost_neg, cost_bas=cost_bas, perm=perm)


def report_ladder(res):
    m_s, d_s = res["mask"], res["data"]
    print(f"\n  A[{res['tag']}]. THE LADDER — score_F "
          f"(production shape loss / L_F_ref; lower is better)")
    print(f"     {'rung':<20}{'free/galaxy':>13}"
          + "".join(f"{f'z={z}':>9}" for z in Z) + f"{'mean':>9}")
    for k in LADDER:
        if k not in res["scores"]:
            continue
        sf = res["scores"][k][1]
        print(f"     {k:<20}{FREEDOM[k]:>13}"
              + "".join(f"{v:>9.3f}" for v in sf) + f"{np.nanmean(sf):>9.3f}")
    print(f"     {'-' * 92}")
    for k, note in (("permuted", "ceiling fitted to ANOTHER galaxy"),
                    ("truth-vs-truth", "another galaxy's truth, unfitted")):
        sf = res["scores"][k][1]
        print(f"     {note:<33}" + "".join(f"{v:>9.3f}" for v in sf)
              + f"{np.nanmean(sf):>9.3f}")
    print(f"     * the monotone projections are basis-free: they constrain "
          f"EVERY deposition-only model.")

    print(f"\n     score_A (1.0 = the halo-only regression)")
    print(f"     {'rung':<20}{'':>13}" + "".join(f"{f'z={z}':>9}" for z in Z)
          + f"{'mean':>9}")
    for k in LADDER:
        if k not in res["scores"]:
            continue
        sa = res["scores"][k][0]
        print(f"     {k:<20}{'':>13}" + "".join(f"{v:>9.3f}" for v in sa)
              + f"{np.nanmean(sa):>9.3f}")
    sa = res["scores"]["permuted"][0]
    print(f"     {'permuted control':<33}" + "".join(f"{v:>9.3f}" for v in sa)
          + f"{np.nanmean(sa):>9.3f}")

    idx = [0, 3, 6, 11, 18, 23]
    print(f"\n  B[{res['tag']}]. WHERE THE RESIDUAL SITS — median "
          f"100*(prediction - truth)/truth")
    for k in (0, 4):
        print(f"\n     z = {Z[k]}   (n = {int(m_s[:, k].sum())})")
        print(f"       {'rung':<20}" + "".join(f"{R[c]:>9.1f}" for c in idx)
              + "   kpc")
        for nm in ("monotone_log", "monotone_obj", "epoch-alone", "ceiling",
                   "bins3", "bins8", "interval5", "model"):
            p, g = res["preds"][nm], m_s[:, k]
            if g.sum() == 0:
                continue
            print(f"       {nm:<20}" + "".join(
                f"{100 * np.median(p[g, k, c] / d_s[g, k, c] - 1):>9.1f}"
                for c in idx))

    print(f"\n     per-galaxy RMS fractional error of the ABSOLUTE curve of "
          f"growth [{'%'}], median (16th-84th)")
    print(f"       {'rung':<20}" + "".join(f"{f'z={z}':>18}" for z in Z))
    for nm, arr in (("epoch-alone", res["rms_alone"]), ("ceiling", res["rms"]),
                    ("model", res["rms_model"])):
        cells = []
        for k in range(5):
            g = m_s[:, k]
            v = 100 * arr[g, k] if g.sum() else np.array([np.nan])
            cells.append(f"{np.median(v):5.2f} ({np.percentile(v, 16):4.2f}-"
                         f"{np.percentile(v, 84):5.2f})" if g.sum() else "  --")
        print(f"       {nm:<20}" + "".join(f"{c:>18}" for c in cells))


def report_diagnostics(res, label):
    m_s, d_s = res["mask"], res["data"]
    print(f"\n  C[{res['tag']}]. THE INCREMENT TEST — solved in the units it "
          f"reports (v1 was not)")
    print(f"     Only the deposits of one epoch interval can build that "
          f"interval's rise in M*(<R).\n     Both costs are an RMS error in "
          f"the cumulative curve, in per cent of M*(<R) at the later epoch,\n"
          f"     with a 400-sample galaxy bootstrap on the median.")
    print(f"       {'interval':<16}{'n':>6}{'radii where':>14}"
          f"{'cost of clipping':>26}{'cost of the basis':>26}")
    print(f"       {'':<16}{'':>6}{'truth FALLS':>14}{'that to zero':>26}"
          f"{'on the RISE':>26}")
    for g in range(4):
        lab = f"z={Z[4 - g]} to {Z[3 - g]}"
        n = int(np.isfinite(res["cost_neg"][:, g]).sum())
        mn, lo_n, hi_n = _boot_ci(100 * res["cost_neg"][:, g])
        mb, lo_b, hi_b = _boot_ci(100 * res["cost_bas"][:, g])
        print(f"       {lab:<16}{n:>6}"
              f"{100 * np.nanmean(res['neg'][:, g]):>13.1f}%"
              f"{f'{mn:.2f}% [{lo_n:.2f},{hi_n:.2f}]':>26}"
              f"{f'{mb:.2f}% [{lo_b:.2f},{hi_b:.2f}]':>26}")

    dec = decline_flag(d_s)
    dec2 = decline_flag(d_s, DECLINE_R2)
    print(f"\n  D[{res['tag']}]. THE CENTRAL-DECLINE SPLIT — preregistered at "
          f"{R[int(np.argmin(np.abs(R - DECLINE_R)))]:.2f} kpc, z=2 to z=0.4")
    print(f"     A static non-negative deposition model CANNOT reproduce a "
          f"decline, so a compact channel\n     can only be the right "
          f"mechanism for galaxies whose centres did not decline.")
    print(f"     declining at {R[int(np.argmin(np.abs(R - DECLINE_R)))]:.2f} "
          f"kpc: {dec.sum()}/{len(dec)} ({100 * dec.mean():.1f} per cent);  at "
          f"{R[int(np.argmin(np.abs(R - DECLINE_R2)))]:.2f} kpc: "
          f"{dec2.sum()}/{len(dec2)} ({100 * dec2.mean():.1f} per cent)")
    i2 = int(np.argmin(np.abs(R - 2.0)))
    print(f"\n     median 100*(prediction - truth)/truth at "
          f"{R[i2]:.1f} kpc, z = 2")
    print(f"       {'group':<24}{'n':>6}" + "".join(
        f"{nm:>14}" for nm in ("model", "ceiling", "epoch-alone",
                               "monotone_obj")))
    for nm, sel in (("declining centre", dec), ("NOT declining", ~dec)):
        g = m_s[:, 4] & sel
        if g.sum() < 10:
            continue
        cells = [100 * np.median(res["preds"][k][g, 4, i2] / d_s[g, 4, i2] - 1)
                 for k in ("model", "ceiling", "epoch-alone", "monotone_obj")]
        print(f"       {nm:<24}{int(g.sum()):>6}"
              + "".join(f"{v:>14.1f}" for v in cells))
    print(f"\n     positive-increment basis cost by group, z=2 to z=1.5 "
          f"[per cent of M*(<R)]")
    for nm, sel in (("declining centre", dec), ("NOT declining", ~dec)):
        v = 100 * res["cost_bas"][sel, 0]
        med, lo, hi = _boot_ci(v)
        print(f"       {nm:<24}{int(np.isfinite(v).sum()):>6}"
              f"       {med:.2f}% [{lo:.2f},{hi:.2f}]")

    print(f"\n  E[{res['tag']}]. THE FREEDOM AUDIT")
    print(f"     active weights per galaxy: median {np.median(res['n_act']):.0f}"
          f", 10th-90th pct {np.percentile(res['n_act'], 10):.0f}-"
          f"{np.percentile(res['n_act'], 90):.0f}"
          f"   [capped solve: median {np.median(res['n_act_fb']):.0f}]")
    ei = np.concatenate(res["eps_impl"])
    em_ = np.concatenate(res["eps_model"])
    print(f"     implied eps_surv: ceiling max {ei.max():.2f}, 99th pct "
          f"{np.percentile(ei, 99):.4f};  model max {em_.max():.4f}   "
          f"(f_b = {F_B:.4f})")
    cost = float(np.nanmean(res["scores"]["ceiling_fb"][1]
                            - res["scores"]["ceiling"][1]))
    print(f"     {100 * float(np.mean(ei > F_B)):.2f} per cent of the ceiling's "
          f"deposits ask for eps_surv above the baryon\n     fraction; "
          f"forbidding it costs {cost:+.4f} in mean score_F, so the ceiling "
          f"PASSES the universal-baryon\n     mass-budget check. That is a "
          f"necessary condition, not a sufficient one: smoothness, merger "
          f"delivery\n     and stellar mass loss are still unconstrained, and "
          f"a median of {np.median(res['n_act']):.0f} active deposits is a "
          f"bursty history.")


def report_epoch_count(res):
    """The masked solve's trivialisation, shown rather than hidden."""
    if res["tag"] != "masked":
        return
    n_used = res["use"].sum(1)
    print(f"\n  F[masked]. HOW MANY EPOCHS EACH GALAXY ACTUALLY CONSTRAINED")
    print(f"     A galaxy admitted at ONE epoch has 24 radii against ~71 free "
          f"weights, so its ceiling is\n     near zero by construction and "
          f"bounds nothing. The z=0.4 column is dominated by such galaxies.")
    print(f"       {'epochs used':<14}{'n':>7}{'median per-galaxy RMS of the '
                                              'ceiling at z=0.4 [per cent]':>62}")
    for k in range(1, 6):
        g = n_used == k
        if g.sum() == 0:
            continue
        print(f"       {k:<14}{int(g.sum()):>7}"
              f"{100 * np.median(res['rms'][g, 0]):>62.3f}")


def _gate_basis(spec, theta, recs):
    """`B @ (eps dMh)` MUST equal `model.forward`, or the basis is not the
    model's. The experiment is 'replace the weights, keep everything else', so
    'everything else' has to be provably identical."""
    worst = 0.0
    for h in recs[: min(len(recs), 40)]:
        B, em, dmh, good = build_basis(spec, theta, h)
        mine = (_design(B, em) @ model_weights(spec, theta, h, good)
                ).reshape(5, len(R))
        theirs = M.forward(spec, theta, h, [0, 1, 2, 3, 4], R)
        if theirs is None:
            continue
        worst = max(worst, float(np.max(np.abs(mine - theirs)
                                        / np.clip(theirs, 1.0, None))))
    print(f"  BASIS GATE (`B @ eps dMh` vs `model.forward`)")
    print(f"    max relative difference over 40 galaxies = {worst:.2e}")
    if worst > 1e-12:
        raise SystemExit(
            "REFUSING TO RUN: the basis this module builds is not the basis "
            "the fitted model uses, so its ceiling bounds a different model.")
    print(f"    OK -- the ceiling is computed on the model's own deposits\n")


def _gate_all_true(spec, theta, recs, data):
    """An all-true `use` mask must reproduce the unmasked solve exactly."""
    B, em, dmh, good = build_basis(spec, theta, recs[0])
    wmod = model_weights(spec, theta, recs[0], good)
    a, _ = solve_galaxy(B, em, dmh, wmod, data[0], np.ones(5, bool))
    A = _design(B, em)
    ref = (A @ _solve(A, data[0].ravel())).reshape(5, -1)
    dev = float(np.max(np.abs(a["ceiling"] - ref) / np.clip(ref, 1.0, None)))
    print(f"  MASK GATE (all-true `use` must reproduce the unmasked solve)")
    print(f"    max relative difference = {dev:.2e}")
    if dev > 1e-12:
        raise SystemExit("REFUSING TO RUN: the epoch mask changes the solve "
                         "even when it admits everything.")
    print(f"    OK -- the mask only removes rows\n")


def main(smoke=False, figures=False):
    import selection as S
    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(POP)
    all_rows = np.array([g["row"] for g in s2._W["gals"]])
    if not SEL.exists():
        raise SystemExit("run selection.py first — the cuts come from there")
    cuts = np.load(SEL, allow_pickle=True)["cuts"]
    m200 = S.sample_masses(np.load(S.HS_NPZ, allow_pickle=True))
    complete_all = np.isfinite(m200) & (m200 >= cuts[None, :])

    rows = all_rows[::40] if smoke else all_rows
    recs = H.build_records(rows=rows, verbose=not smoke)
    sel = np.array([h.row for h in recs])
    data = pop["data"][sel]
    mask = complete_all[sel]

    thetas = {}
    pe = np.load(PEREPOCH, allow_pickle=True)
    s33 = np.load(STAGE33, allow_pickle=True)
    for label, src in BASES:
        thetas[label] = np.asarray(
            (pe if src == "perepoch" else s33)[f"{label}_theta"], float)

    print(f"exp54 STAGE 3.4 — THE PROFILE-SHAPE REPRESENTATIONAL CEILING (v2)")
    print(f"  galaxies: {len(recs)}   mh-complete per epoch: "
          + " ".join(str(int(v)) for v in mask.sum(0)))
    print(f"  'mh-complete' = above the halo mass at which our completeness "
          f"first reaches 60 per cent of\n  its z=0.4 ceiling of 0.781, i.e. "
          f"absolute completeness ~0.47. It is a threshold, not completeness 1.")
    print(f"  SIGMA_A : " + " ".join(f"{v:.4f}" for v in F.SIGMA_A))
    print(f"  bases   : " + ",  ".join(
        f"{lb} ({src}: " + " ".join(
            f"{n}={v:+.3f}" for n, v in
            zip(S33.spec_from_label(lb).theta_names[-3:], thetas[lb][-3:]))
        + ")" for lb, src in BASES) + "\n")

    label0, _ = BASES[0]
    spec0 = S33.spec_from_label(label0)
    _gate_scoring(spec0, thetas[label0], recs, data, mask)
    _gate_basis(spec0, thetas[label0], recs)
    _gate_all_true(spec0, thetas[label0], recs, data)

    n = len(recs)
    all_true = np.ones((n, 5), bool)
    cohort_rows = np.where(mask[:, 4])[0]
    definitions = {
        "all5": (all_true, np.arange(n)),
        "masked": (mask, np.arange(n)),
        "cohort": (all_true, cohort_rows),
    }
    print(f"  SAMPLE DEFINITIONS")
    print(f"    all5    solve on all 5 epochs, score on the mask  "
          f"(v1's object, kept to show the change)")
    print(f"    masked  solve only on each galaxy's admitted epochs "
          f"(matches MaskedProblem)")
    print(f"    cohort  the {len(cohort_rows)} galaxies above the z=2 "
          f"threshold, all 5 epochs — the EVOLUTION sample")
    print(f"    galaxies admitted at 1/2/3/4/5 epochs: "
          + "/".join(str(int(v)) for v in np.bincount(mask.sum(1),
                                                      minlength=6)[1:]))

    out = {}
    for label, _ in (BASES[:1] if smoke else BASES):
        spec = S33.spec_from_label(label)
        print(f"\n{RULE}\n{label}\n{RULE}")
        for tag in (SAMPLES[:2] if smoke else SAMPLES):
            use, keep = definitions[tag]
            res = run_sample(spec, thetas[label], recs, data, mask, use, keep,
                             tag)
            report_ladder(res)
            report_epoch_count(res)
            if tag == "cohort" or (smoke and tag == "masked"):
                report_diagnostics(res, label)
            out[(label, tag)] = res

    if figures:
        import stage34_figures as SF
        for (label, tag), res in out.items():
            if tag == "cohort":
                SF.ladder_figure(label, res, res["data"], res["mask"],
                                 name=f"{label}_smoke" if smoke else label)

    if smoke:
        print("\n  smoke run: writing nothing")
        return out
    np.savez_compressed(
        OUT, version=2, labels=np.array([b for b, _ in BASES]),
        samples=np.array(SAMPLES), rows=sel, mask=mask, cuts=cuts,
        sigma_a=F.SIGMA_A, ladder=np.array(LADDER), k_ladder=np.array(K_LADDER),
        f_b=F_B, decline_r=DECLINE_R,
        **{f"{lb}|{tg}|{k}_{s}": res["scores"][k][i]
           for (lb, tg), res in out.items() for k in res["scores"]
           for i, s in enumerate(("sA", "sF"))},
        **{f"{lb}|{tg}|pred_{k}": res["preds"][k]
           for (lb, tg), res in out.items() for k in res["preds"]},
        **{f"{lb}|{tg}|{k}": res[k] for (lb, tg), res in out.items()
           for k in ("idx", "n_act", "n_act_fb", "neg", "cost_neg", "cost_bas",
                     "rms", "rms_model", "rms_alone", "use")})
    print(f"\n  wrote {OUT}")
    return out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, figures="--figures" in sys.argv)
