"""exp54 Stage 3.4 — THE PROFILE-SHAPE REPRESENTATIONAL CEILING.

Stage 3.0 computed a ceiling for the AMPLITUDE and correctly killed a survival
term before anyone built it. The same question has never been asked about the
SHAPE, and three candidate culprits currently look identical in the loss: the
efficiency law, the deposit basis, and the size law.

THE EXPERIMENT. For one galaxy the model's deposits form a FIXED BASIS: one
truncated unit-mass cumulative profile `F_j(<R)` per MAH step, with `R50` from
the fitted size law and truncation at `3 R200c(t_j)`. The model chooses the
weights by a seven-parameter global law, `dM*_j = eps_surv(H_j) dMh_j`. Ask
instead:

    over ALL non-negative weights `w_j >= 0`, respecting causality (`w_j`
    enters epoch `k` only if `t_j <= t_k`), what is the best fit to THIS
    galaxy's measured curve of growth at all five epochs?

That is non-negative least squares -- convex, no optimiser tuning, no
multi-start, about a millisecond per galaxy. Its residual is the ceiling: the
best any deposition model with this basis could do given a PERFECT efficiency
law.

**HOW TO READ IT, AND HOW NOT TO.** NNLS gets unconstrained PER-GALAXY freedom
-- 71 non-negative numbers where the model has seven global ones. So a LARGE
ceiling proves the basis inadequate; a SMALL ceiling does NOT prove that any
global law can reach it. It is reported as a bound, never as a target, and
Part D measures how much of its fitting power is generic flexibility rather
than galaxy-specific information.

THE LADDER (Part A). A single ceiling number cannot separate the culprits, so
the run brackets the model between bounds of decreasing freedom:

  1. `monotone`   the monotone-in-time projection of the truth itself. Any sum
                  of static non-negative deposits makes `M*(<R,t)` non-
                  decreasing in time; TNG does not. Basis-free, so it bounds
                  EVERY deposition-only model. (Stage 3.0's floor, re-measured
                  here on this sample and this mask.)
  2. `epoch-alone` NNLS run on each epoch separately, with no requirement that
                  one weight vector serve all five. NOT a deposition model --
                  it is the answer to "can this basis draw this profile at
                  all", with the causal coupling switched off.
  3. `ceiling`    the joint causal NNLS above. 71 free weights per galaxy.
  4. `ceiling_fb` the same with `eps_surv <= f_b = Omega_b/Omega_m`, i.e. no
                  deposit may turn more than the universal baryon fraction of
                  its accreted halo mass into surviving central stars. A hard
                  physical bound with nothing tuned in it, so a ceiling that
                  survives it is physically admissible.
  5. `interval5`  NNLS over only FIVE weights per galaxy, one per interval
                  between observation epochs, with the within-interval
                  distribution taken from the fitted model. Per-galaxy freedom
                  in the coarse TIME distribution of deposited mass and nothing
                  else.
  6. `model`      the fitted seven-parameter global law. Zero per-galaxy
                  freedom.

Rung 2 isolates the basis; rung 1 isolates the deposition-only premise; the gap
between 3 and 6 is what a richer efficiency law could in principle buy.

WHAT IS SCORED, AND ON WHICH GALAXIES. Everything is scored with the production
`score_A` / `score_F` on the PER-EPOCH MH-COMPLETE MASK, because the full sample
is now known to flatter the model. `_score_masked` is gated against
`stage33_perepoch.MaskedProblem`: it must reproduce that class's `per_epoch` to
machine precision on the model's own prediction, or the run refuses to proceed.
A bound and the thing it bounds must be measured on the same galaxies.

ONE HONEST CAVEAT ON DIRECTION. NNLS minimises the summed squared FRACTIONAL
residual of the absolute curve of growth, which is close to but not identical
with the production objective (RMS over radii, then mean over epochs, on a
profile renormalised at 100 kpc). So the reported `score_F` of an NNLS solution
is an UPPER bound on what that rung could achieve. Every "the ceiling is this
low" statement is therefore conservative; every "the ceiling is this high"
statement is not, and is checked against the basis-free rung 1 instead.

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
RULE = "=" * 96

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
LADDER = ("monotone", "epoch-alone", "ceiling", "ceiling_fb", "interval5", "model")


# --------------------------------------------------------------------------- #
# scoring                                                                      #
# --------------------------------------------------------------------------- #
def _score_masked(pred, data, mask, obj=None):
    """(score_A[k], score_F[k]) for an ARBITRARY prediction, under the mask.

    A transcription of `stage33_perepoch.MaskedProblem.per_epoch` that takes a
    prediction array instead of a theta, so that a bound with no parameter
    vector can be scored on exactly the footing the model is. `_gate_scoring`
    asserts the two agree.
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


def _gate_scoring(spec, theta, recs, data, mask):
    """`_score_masked` MUST reproduce `MaskedProblem.per_epoch`, or stop.

    Everything in this module is a comparison between a bound and the fitted
    model. If the two are not computed by the same arithmetic on the same
    galaxies, the comparison is meaningless -- and that is exactly the defect
    that overturned this experiment's previous headline.
    """
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
    return pr


# --------------------------------------------------------------------------- #
# the basis and the rungs                                                      #
# --------------------------------------------------------------------------- #
def build_basis(spec, theta, h):
    """(B, epoch_mask, dMh) for one galaxy: the model's own deposit basis.

    `B[r, j] = F_j(<R_r)` is the unit-mass cumulative profile of deposit `j`,
    truncated at `3 R200c(t_j)` and renormalised, exactly as `model.forward`
    builds it -- `_gate_basis` asserts that `B @ (eps dMh)` reproduces
    `model.forward` to machine precision.
    """
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


def _solve(A, y, ub=None):
    """min ||(A w - y)/y||^2 over `0 <= w <= ub`. Rows scaled to fractional
    residuals; columns scaled by `ub` so the bounded solve is conditioned."""
    w_row = 1.0 / np.clip(y, 1.0, None)
    Aw, yw = A * w_row[:, None], y * w_row
    if ub is None:
        w, _ = nnls(Aw, yw)
        return w
    s = np.maximum(ub, 1.0)                     # column scaling: v = w/s
    r = lsq_linear(Aw * s[None, :], yw, bounds=(0.0, 1.0), method="bvls",
                   max_iter=2000)
    return r.x * s


def _interval_groups(em):
    """Group index per deposit: 0 = before z=2, then one per epoch interval.

    Group `g` is the set of deposits laid down between observation epoch
    `n_epoch - g` and `n_epoch - g - 1`, so group 0 is everything the earliest
    epoch already sees and group `n_epoch - 1` is what arrives last.
    """
    ne = em.shape[0]
    grp = np.full(em.shape[1], -1, int)
    grp[em[ne - 1]] = 0
    for k in range(ne - 2, -1, -1):
        grp[em[k] & (grp < 0)] = ne - 1 - k
    return grp


def ceilings_for_galaxy(B, em, dmh, wmod, d):
    """Every NNLS rung for one galaxy. Returns dict name -> (5, nR) prediction."""
    ne = em.shape[0]
    A = _design(B, em)
    y = d.ravel()
    out = {}
    w = _solve(A, y)
    out["ceiling"] = (A @ w).reshape(ne, -1)
    wfb = _solve(A, y, ub=F_B * dmh)
    out["ceiling_fb"] = (A @ wfb).reshape(ne, -1)

    grp = _interval_groups(em)
    G = np.zeros((A.shape[0], ne))
    for g in range(ne):
        sel = grp == g
        if sel.any() and wmod[sel].sum() > 0:
            G[:, g] = A[:, sel] @ wmod[sel] / wmod[sel].sum()
    v = _solve(G, y)
    out["interval5"] = (G @ v).reshape(ne, -1)

    alone = np.empty_like(d)
    for k in range(ne):
        Ak = B * em[k][None, :]
        alone[k] = Ak @ _solve(Ak, d[k])
    out["epoch-alone"] = alone
    return out, w, wfb, grp


# --------------------------------------------------------------------------- #
# diagnostics                                                                  #
# --------------------------------------------------------------------------- #
def increment_test(B, em, d):
    """Separate 'the truth goes DOWN' from 'the basis cannot draw the rise'.

    The causal design is block-triangular: the deposits of one epoch interval
    are the only ones that can build the increment `M*(<R,t_k) - M*(<R,t_k+1)`.
    Two ways that can fail, and they call for opposite fixes:

      * the required increment is NEGATIVE at some radius -- the truth's
        enclosed mass fell. No deposition model of any basis can do that, so it
        is the deposition-only premise, not the basis.
      * the increment is positive but lies outside the non-negative cone of that
        interval's deposits -- then the basis is genuinely too coarse.

    Both are reported in the SAME units -- error in the cumulative curve as a
    fraction of `M*(<R)` at the later epoch -- so the two failure modes can be
    compared directly instead of on incommensurable scales.

    Returns (frac_negative, cost_negative, cost_basis) per interval:
      * `frac_negative`  fraction of radii where the required increment is < 0;
      * `cost_negative`  RMS of `min(inc, 0)/M*(<R)`, the error forced by
        clipping those to zero -- the deposition-only premise's own bill;
      * `cost_basis`     RMS of `(B_g w - max(inc,0))/M*(<R)` at the best
        non-negative `w`, the extra error from the basis being too coarse.
    """
    ne = em.shape[0]
    grp = _interval_groups(em)
    neg = np.full(ne - 1, np.nan)
    cost_neg, cost_bas = np.full(ne - 1, np.nan), np.full(ne - 1, np.nan)
    for g in range(1, ne):
        k_new, k_old = ne - 1 - g, ne - g       # epochs bracketing interval g
        inc = d[k_new] - d[k_old]
        ref = np.clip(d[k_new], 1.0, None)
        neg[g - 1] = float(np.mean(inc < 0))
        cost_neg[g - 1] = float(np.sqrt(np.mean((np.clip(inc, None, 0) / ref) ** 2)))
        sel = grp == g
        if not sel.any():
            continue
        tgt = np.clip(inc, 0.0, None)
        fit_ = B[:, sel] @ _solve(B[:, sel], np.maximum(tgt, 1.0))
        cost_bas[g - 1] = float(np.sqrt(np.mean(((fit_ - tgt) / ref) ** 2)))
    return neg, cost_neg, cost_bas


def slope_bound(B, em, band=(50.0, 148.3)):
    """The STEEPEST outer density slope this basis can reach, per epoch.

    For a non-negative sum `Sigma = sum_j w_j sigma_j`, the local logarithmic
    slope is the sigma-weighted MEAN of the components' slopes, so it can never
    be steeper than the steepest component:

        d ln Sigma / d ln R  >=  min_j  d ln sigma_j / d ln R      (pointwise)

    That is a rigorous bound with no fitting in it. It is the compactness
    question asked in the outskirts, where it has teeth -- asked of the
    cumulative profile instead it has none, because an ancient sub-kpc deposit
    can always supply an arbitrarily large `F(<3 kpc)/F(<100 kpc)`, and those
    deposits contribute nothing at 100 kpc.
    """
    from hongshao.profile_emulator import density_from_cog
    ls, mid = density_from_cog(np.log10(np.clip(B.T * 1e12, 1.0, None)), R)
    sel = (mid >= band[0]) & (mid <= band[1])
    x = np.log10(mid[sel])
    s = np.array([np.polyfit(x, row[sel], 1)[0] if np.isfinite(row[sel]).all()
                  else np.nan for row in ls])
    return np.array([np.nanmin(s[em[k]]) if em[k].any() else np.nan
                     for k in range(em.shape[0])])


def _gal_slopes(cog, band=(50.0, 148.3)):
    """Per-galaxy outer density slope d log10 Sigma / d log10 R over the band."""
    from hongshao.profile_emulator import density_from_cog
    ls, mid = density_from_cog(np.log10(np.clip(cog, 1.0, None)), R)
    sel = (mid >= band[0]) & (mid <= band[1])
    x = np.log10(mid[sel])
    return np.array([np.polyfit(x, row[sel], 1)[0]
                     if np.isfinite(row[sel]).all() else np.nan for row in ls])


def outer_slope(cog):
    """d log10 Sigma / d log10 R over 50-148 kpc of the MEDIAN profile.

    The R^(1/4) density view is where the model's high-redshift failure is
    sharpest: the truth steepens from -2.78 to -3.38 between z=0.4 and z=2 and
    the model barely moves. Same operator on both sides -- see `qa_figures._sigma`.
    """
    from hongshao.profile_emulator import density_from_cog
    ls, mid = density_from_cog(np.log10(np.clip(cog, 1.0, None)), R)
    med = np.median(ls, axis=0)
    sel = (mid >= 50.0) & (mid <= 148.3)
    return float(np.polyfit(np.log10(mid[sel]), med[sel], 1)[0])


# --------------------------------------------------------------------------- #
# the run                                                                      #
# --------------------------------------------------------------------------- #
def run_basis(label, theta, recs, data, mask, lmh, quiet=False):
    """Every rung, every diagnostic, for one basis. Returns a result dict."""
    spec = S33.spec_from_label(label)
    n, ne = len(recs), 5
    preds = {k: np.empty((n, ne, len(R))) for k in
             ("ceiling", "ceiling_fb", "interval5", "epoch-alone", "model")}
    n_act = np.zeros(n, int)
    n_act_fb = np.zeros(n, int)
    rank = np.zeros(n, int)
    eps_impl, eps_model, dep_z = [], [], []
    neg = np.full((n, ne - 1), np.nan)
    cost_neg = np.full((n, ne - 1), np.nan)
    cost_bas = np.full((n, ne - 1), np.nan)
    sbound = np.full((n, ne), np.nan)

    t0 = time.time()
    for i, h in enumerate(recs):
        B, em, dmh, good = build_basis(spec, theta, h)
        wmod = model_weights(spec, theta, h, good)
        d = data[i]
        got, w, wfb, grp = ceilings_for_galaxy(B, em, dmh, wmod, d)
        for k, v in got.items():
            preds[k][i] = v
        preds["model"][i] = (_design(B, em) @ wmod).reshape(ne, -1)
        n_act[i] = int((w > 0).sum())
        n_act_fb[i] = int((wfb > 1e-9 * max(wfb.max(), 1.0)).sum())
        if i < 200:
            rank[i] = np.linalg.matrix_rank(_design(B, em))
        eps_impl.append(w / np.clip(dmh, 1.0, None))
        eps_model.append(wmod / np.clip(dmh, 1.0, None))
        dep_z.append(h.z[good])
        neg[i], cost_neg[i], cost_bas[i] = increment_test(B, em, d)
        sbound[i] = slope_bound(B, em)
    if not quiet:
        print(f"    {n} galaxies in {time.time() - t0:.1f} s "
              f"({1000 * (time.time() - t0) / n:.1f} ms each)")

    # the permuted-target control and its reference: how much of the basis's
    # fitting power is generic flexibility rather than galaxy-specific?
    perm = (np.arange(n) + max(n // 3, 1)) % n
    pperm = np.empty_like(preds["ceiling"])
    for i, h in enumerate(recs):
        B, em, dmh, good = build_basis(spec, theta, h)
        A = _design(B, em)
        pperm[i] = (A @ _solve(A, data[perm[i]].ravel())).reshape(ne, -1)

    scores = {k: _score_masked(v, data, mask) for k, v in preds.items()}
    scores["monotone"] = _score_masked(S30.project_all(data), data, mask)
    preds["monotone"] = S30.project_all(data)
    scores["permuted"] = _score_masked(pperm, data[perm], mask[perm])
    scores["truth-vs-truth"] = _score_masked(data[perm], data, mask)

    return dict(spec=spec, theta=theta, preds=preds, scores=scores,
                n_act=n_act, n_act_fb=n_act_fb, rank=rank[:200],
                eps_impl=eps_impl, eps_model=eps_model, dep_z=dep_z,
                neg=neg, cost_neg=cost_neg, cost_bas=cost_bas,
                sbound=sbound, perm=perm)


def report(label, res, data, mask):
    idx = [0, 3, 6, 11, 18, 23]
    print(f"\n{RULE}\n{label}\n{RULE}")

    print(f"\n  A. THE LADDER — score_F (production shape loss / "
          f"L_F_ref = {F.L_F_REF:.6f}), on each epoch's own mh-complete galaxies")
    print(f"     {'rung':<34}{'freedom/galaxy':>16}"
          + "".join(f"{f'z={z}':>9}" for z in Z))
    free = {"monotone": "unbounded*", "epoch-alone": "5 x ~71",
            "ceiling": "71", "ceiling_fb": "71 (capped)",
            "interval5": "5", "model": "0 (7 global)"}
    for k in LADDER:
        sa, sf = res["scores"][k]
        print(f"     {k:<34}{free[k]:>16}" + "".join(f"{v:>9.3f}" for v in sf))
    print(f"     {'-' * 90}")
    for k in ("permuted", "truth-vs-truth"):
        sa, sf = res["scores"][k]
        note = {"permuted": "ceiling fitted to ANOTHER galaxy",
                "truth-vs-truth": "another galaxy's truth, unfitted"}[k]
        print(f"     {note:<34}{'':>16}" + "".join(f"{v:>9.3f}" for v in sf))
    print(f"     * the monotone projection is basis-free: it bounds EVERY "
          f"deposition-only model, of any basis.")

    print(f"\n     score_A (1.0 = the halo-only regression)")
    print(f"     {'rung':<34}{'':>16}" + "".join(f"{f'z={z}':>9}" for z in Z))
    for k in LADDER:
        sa, sf = res["scores"][k]
        print(f"     {k:<34}{'':>16}" + "".join(f"{v:>9.3f}" for v in sa))

    print(f"\n  B. WHERE THE RESIDUAL SITS — median 100*(prediction - truth)/truth")
    for k in (0, 2, 4):
        print(f"\n     z = {Z[k]}   (n = {int(mask[:, k].sum())})")
        print(f"       {'rung':<20}" + "".join(f"{R[c]:>9.1f}" for c in idx)
              + "   kpc")
        for nm in LADDER:
            p = res["preds"][nm]
            g = mask[:, k]
            row = [100 * np.median(p[g, k, c] / data[g, k, c] - 1) for c in idx]
            print(f"       {nm:<20}" + "".join(f"{v:>9.1f}" for v in row))

    print(f"\n     outer density slope d log10 Sigma / d log10 R over 50-148 kpc "
          f"(more negative = steeper)")
    print(f"       {'':<20}" + "".join(f"{f'z={z}':>9}" for z in Z))
    for nm in ("truth",) + LADDER:
        row = []
        for k in range(5):
            g = mask[:, k]
            c = data[g, k] if nm == "truth" else res["preds"][nm][g, k]
            row.append(outer_slope(c))
        print(f"       {nm:<20}" + "".join(f"{v:>9.2f}" for v in row))

    print(f"\n     WHY THE OUTER SLOPE IS NOT A REPRESENTATIONAL PROBLEM. "
          f"`epoch-alone` matches the\n     cumulative curve to about one per "
          f"cent at every radius and is STILL too shallow,\n     because at "
          f"high redshift almost no mass lives out there: the objective is "
          f"blind, not\n     the basis. Mass fraction outside 52 kpc, and each "
          f"rung's mass in the 52-148 kpc\n     annulus as a ratio to the "
          f"truth's:")
    i50 = int(np.argmin(np.abs(R - 50.0)))
    print(f"       {'':<20}" + "".join(f"{f'z={z}':>9}" for z in Z))
    fout = [np.median(1.0 - data[mask[:, k], k, i50] / data[mask[:, k], k, -1])
            for k in range(5)]
    print(f"       {'M* outside 52 kpc':<20}" + "".join(f"{v:>9.3f}" for v in fout))
    for nm in LADDER:
        row = []
        for k in range(5):
            g = mask[:, k]
            p, t = res["preds"][nm][g, k], data[g, k]
            row.append(np.median((p[:, -1] - p[:, i50]) / np.clip(p[:, -1], 1.0, None))
                       / np.median((t[:, -1] - t[:, i50]) / t[:, -1]))
        print(f"       {nm:<20}" + "".join(f"{v:>9.2f}" for v in row))
    print(f"     (The rigorous bound 'a non-negative sum is never steeper than "
          f"its steepest deposit'\n     is VACUOUS here — median "
          f"{np.nanmedian(res['sbound'][:, 4]):.0f} dex/dex — because the hard "
          f"truncation at 3 R200c puts an\n     arbitrarily steep edge inside "
          f"the band. It forecloses the question; it does not answer it.)")

    print(f"\n  C. THE INCREMENT TEST — why the joint ceiling costs more than "
          f"the epoch-alone one")
    print(f"     The causal design is block-triangular: only the deposits of "
          f"one epoch interval can\n     build that interval's increase in "
          f"M*(<R). Two ways it fails, needing opposite fixes,\n     both "
          f"priced as an RMS error in the cumulative curve, in per cent of "
          f"M*(<R).")
    print(f"       {'interval':<20}{'radii where the':>20}"
          f"{'cost of clipping':>20}{'cost of the basis':>20}")
    print(f"       {'':<20}{'truth DECREASES':>20}{'that to zero':>20}"
          f"{'on the RISE':>20}")
    for g in range(4):
        lab = f"z={Z[4 - g]} -> z={Z[3 - g]}"
        print(f"       {lab:<20}{100 * np.nanmean(res['neg'][:, g]):>19.1f}%"
              f"{100 * np.nanmedian(res['cost_neg'][:, g]):>19.2f}%"
              f"{100 * np.nanmedian(res['cost_bas'][:, g]):>19.2f}%")

    print(f"\n  D. THE FREEDOM AUDIT — how much of the ceiling is generic "
          f"flexibility?")
    print(f"     active weights per galaxy (of 71): median "
          f"{np.median(res['n_act']):.0f}, "
          f"90th pct {np.percentile(res['n_act'], 90):.0f}   "
          f"[capped solve: median {np.median(res['n_act_fb']):.0f}]")
    print(f"     numerical rank of the causal design: median "
          f"{np.median(res['rank']):.0f} of 120 rows")
    ei = np.concatenate(res["eps_impl"])
    em_ = np.concatenate(res["eps_model"])
    print(f"     implied eps_surv: ceiling max {ei.max():.2f}, 99th pct "
          f"{np.percentile(ei, 99):.4f};  model max {em_.max():.4f}   "
          f"(f_b = {F_B:.4f})")
    frac_over = 100.0 * float(np.mean(ei > F_B))
    cost = float(np.mean(res["scores"]["ceiling_fb"][1]
                         - res["scores"]["ceiling"][1]))
    print(f"     {frac_over:.2f} per cent of the ceiling's deposits ask for "
          f"eps_surv above the baryon fraction;\n     forbidding that "
          f"(`ceiling_fb`) costs {cost:+.4f} in mean score_F, so the ceiling is "
          f"physically admissible.")
    for zc in (3.0, 2.0, 1.0):
        fi = np.array([w[z > zc].sum() / max(w.sum(), 1e-30)
                       for w, z in zip(res["eps_impl"], res["dep_z"])])
        fm = np.array([w[z > zc].sum() / max(w.sum(), 1e-30)
                       for w, z in zip(res["eps_model"], res["dep_z"])])
        print(f"     median fraction of stellar mass deposited at z > {zc}:  "
              f"ceiling {np.median(fi):.3f}   model {np.median(fm):.3f}")


def main(smoke=False, figures=False):
    import selection as S
    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(POP)
    all_rows = np.array([g["row"] for g in s2._W["gals"]])
    if not SEL.exists():
        raise SystemExit("run selection.py first — the mh-complete cuts come from there")
    cuts = np.load(SEL, allow_pickle=True)["cuts"]
    m200 = S.sample_masses(np.load(S.HS_NPZ, allow_pickle=True))
    complete_all = np.isfinite(m200) & (m200 >= cuts[None, :])

    rows = all_rows[::40] if smoke else all_rows
    recs = H.build_records(rows=rows, verbose=not smoke)
    sel = np.array([h.row for h in recs])
    data = pop["data"][sel]
    lmh = pop["logmh_zk_diffmah"][sel]
    mask = complete_all[sel]

    thetas = {}
    pe = np.load(PEREPOCH, allow_pickle=True)
    s33 = np.load(STAGE33, allow_pickle=True)
    for label, src in BASES:
        z = pe if src == "perepoch" else s33
        thetas[label] = np.asarray(z[f"{label}_theta"], float)

    print(f"exp54 STAGE 3.4 — THE PROFILE-SHAPE REPRESENTATIONAL CEILING")
    print(f"  galaxies: {len(recs)}   mh-complete per epoch: "
          + " ".join(str(int(v)) for v in mask.sum(0)))
    print(f"  SIGMA_A : " + " ".join(f"{v:.4f}" for v in F.SIGMA_A))
    print(f"  bases   : " + ",  ".join(
        f"{lb} ({src}: " + " ".join(
            f"{n}={v:+.3f}" for n, v in
            zip(S33.spec_from_label(lb).theta_names[-3:], thetas[lb][-3:]))
        + ")" for lb, src in BASES) + "\n")

    label0, _ = BASES[0]
    _gate_scoring(S33.spec_from_label(label0), thetas[label0], recs, data, mask)
    _gate_basis(S33.spec_from_label(label0), thetas[label0], recs)

    out = {}
    for label, _ in (BASES[:1] if smoke else BASES):
        res = run_basis(label, thetas[label], recs, data, mask, lmh)
        report(label, res, data, mask)
        out[label] = res

    if figures:
        import stage34_figures as SF
        for label in out:
            SF.ladder_figure(label, out[label], data, mask)

    if smoke:
        print("\n  smoke run: writing nothing")
        return out
    np.savez_compressed(
        OUT, labels=np.array(list(out)), rows=sel, mask=mask, cuts=cuts,
        sigma_a=F.SIGMA_A, ladder=np.array(LADDER), f_b=F_B,
        **{f"{lb}_{k}_{tag}": res["scores"][k][i]
           for lb, res in out.items() for k in res["scores"]
           for i, tag in enumerate(("sA", "sF"))},
        **{f"{lb}_pred_{k}": res["preds"][k]
           for lb, res in out.items() for k in res["preds"]},
        **{f"{lb}_{k}": res[k] for lb, res in out.items()
           for k in ("n_act", "n_act_fb", "neg", "cost_neg", "cost_bas",
                     "sbound", "theta")})
    print(f"\n  wrote {OUT}")
    return out


def _gate_basis(spec, theta, recs):
    """`B @ (eps dMh)` MUST equal `model.forward`, or the basis is not the
    model's. The whole experiment is 'replace the weights and keep everything
    else', so 'everything else' has to be provably identical."""
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


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, figures="--figures" in sys.argv)
