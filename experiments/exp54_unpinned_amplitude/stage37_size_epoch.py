"""exp54 Stage 3.7 — the size law's dependence on epoch.

Plan: `doc/plans/2026-08-25-exp54-stage37-size-epoch.md`.

THE DEFECT. The model's error in the stellar mass inside 2 kpc, as a median
over the completeness-controlled galaxies, in dex, negative meaning too light:

    z = 0.4   +0.020      z = 1.0   -0.021      z = 2.0   -0.126
    z = 0.7   -0.003      z = 1.5   -0.062      span      0.145

Right in the centre today, 25% too light in the centre at redshift 2. Two of
the three available levers are now known not to touch that span:

  * the efficiency law's dependence on TIME cannot (Stage 3.5): five richer
    forms and a free shared curve all sit within 0.2 percentage points of the
    incumbent, which is the whole budget for a shared time law;
  * the OBJECTIVE cannot (Stage 3.6): five objectives shift this column by a
    nearly constant offset -- the shift varies by 0.006 dex across five epochs
    -- and leave the span at 0.145 to 0.152 dex in every case.

The third lever is WHERE each deposit puts its mass as a function of when it
was laid down, and it has never been enriched. `model.r50_of` gives every
deposit `R50 = 10^(log_f0 + b log10(1+z)) * R200c`, a single power law in
redshift with ONE exponent shared by every galaxy and every halo mass. The
previously-tested size extension (`S3`) varies the size NORMALISATION with halo
properties; it never varies the exponent.

WHAT THIS SCRIPT DOES, in the order it must be done.

`--precheck`  **The measurement that can cancel the whole stage.** The
    completeness-controlled sample runs from 2397 galaxies at redshift 0.4 to
    839 at redshift 2, and the survivors are more massive. If the central error
    is a function of halo MASS rather than of EPOCH, the 0.145 dex span is the
    sample's composition moving and there is nothing here to fix. So the error
    is re-measured in fixed bins of epoch-matched halo mass, and on the fixed
    cohort followed at all five epochs. Lesson 22 — "the completeness trap
    recurs in every new view" — applied before a headline instead of after one.

`--bound`     Stage 3.5's most useful product was not its candidates but `E8`,
    the free shared curve that showed the budget was 0.2 points before any
    candidate was fitted. `S6f6` is the same thing for the size law: a free
    piecewise-linear curve in `ln(1+z)` on six knots, with the constant and
    linear directions projected out analytically so it cannot trade against
    `log_f0` and `b`. Not a candidate; a ceiling on every shared size law.

`--fit`       The candidates, only worth running if the bound leaves room:
    `S4` (the exponent depends on halo mass, 1 parameter) and `S5` (a low-order
    polynomial in redshift, 1-2 parameters), and both together.

HOW IT IS JUDGED. The profile-shape error is NOT the criterion: it is dominated
by radii where the model is already right, and this stage is aimed at a defect
the aggregate barely feels. The criterion is the central error's SPAN across
epochs, with the outskirts, the outer slope, the amplitude and the halo-mass
tilt all required not to get worse, plus the identifiability report.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp54_unpinned_amplitude/stage37_size_epoch.py --precheck
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import halo as H                                         # noqa: E402
import model as M                                        # noqa: E402
import stage32 as S32                                    # noqa: E402
import stage33 as S33                                    # noqa: E402
import stage35_time_law as S35                           # noqa: E402
import stage36_objective as S36                          # noqa: E402

POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
SEL = HERE / "outputs" / "selection.npz"
OUT = HERE / "outputs" / "stage37_size_epoch.npz"
PRECHECK = HERE / "outputs" / "stage37_precheck.npz"
FITDIR = HERE / "outputs" / "stage37_fits"

BASE = "gompertz_log-E2-S2"
#: the bound first, then the candidates
BOUNDS = (f"{BASE}+S6f6",)
CANDIDATES = (f"{BASE}+S4", f"{BASE}+S5p1", f"{BASE}+S5p2", f"{BASE}+S4+S5p2")
Z = (0.4, 0.7, 1.0, 1.5, 2.0)
#: fixed bins of epoch-matched log10 halo mass for the pre-check
MASS_BINS = (12.8, 13.1, 13.4, 13.7, 15.5)
RULE = "=" * 100


def build(smoke=False):
    """Records, data, the completeness-and-sanity mask, and halo masses."""
    import selection as S
    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(POP)
    all_rows = np.array([g["row"] for g in s2._W["gals"]])
    cuts = np.load(SEL, allow_pickle=True)["cuts"]
    m200 = S.sample_masses(np.load(S.HS_NPZ, allow_pickle=True))
    complete = (np.isfinite(m200) & (m200 >= cuts[None, :])
                & S.sane_history_mask(pop["data"]))
    rows = all_rows[::24] if smoke else all_rows
    recs = H.build_records(rows=rows, verbose=False)
    sel = np.array([h.row for h in recs])
    return (recs, sel, pop["data"][sel], complete[sel],
            pop["logmh_zk_diffmah"][sel], cuts)


def incumbent():
    """The incumbent's parameters, from the CLEAN-sample fit, or refuse."""
    p = HERE / "outputs" / "stage35_fits" / f"{BASE}.npz"
    if not p.exists():
        raise SystemExit(
            "REFUSING TO RUN: no clean-sample fit for the incumbent. Stage 3.5 "
            "must be re-run after `selection.sane_history_mask` was applied; "
            "the pre-fix fits in stage35_fits_PRE_ROW181_FIX are NOT a "
            "substitute, because this stage's whole subject is a 0.1 dex "
            "effect and that contamination moved the loss by 18%.")
    return np.asarray(np.load(p)["theta"], float)


def central_error(pred, data, mask, shell=0):
    """log10(model/truth) in one shell, per galaxy per epoch, NaN elsewhere."""
    def shells(c):
        return np.concatenate([c[:, :, :1], np.diff(c, axis=2)], axis=2)
    sm, sd = shells(pred)[:, :, shell], shells(data)[:, :, shell]
    ok = mask & np.isfinite(sm) & (sm > 0) & (sd > 0)
    out = np.full(sm.shape, np.nan)
    out[ok] = np.log10(sm[ok]) - np.log10(sd[ok])
    return out


# --------------------------------------------------------------------------- #
def precheck(smoke=False):
    """Is the epoch span real at fixed halo mass, or is it the sample moving?"""
    recs, sel, data, mask, lmh, cuts = build(smoke)
    spec = S33.spec_from_label(BASE)
    th = incumbent()
    pr = S35.StackedProblem(spec, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    pred = pr.predict(th)
    err = central_error(pred, data, mask)

    print(f"exp54 STAGE 3.7 PRE-CHECK — is the central error's epoch span "
          f"real at fixed halo mass?")
    print(f"  model {BASE}, clean-sample fit; {len(recs)} galaxies, "
          f"{int(mask.sum())} galaxy-epochs")
    print(f"  the quantity: median log10(model/truth) for M*(<2 kpc); "
          f"negative = the model is too light\n")

    print(f"  (1) AS REPORTED — every galaxy the mask admits at each epoch")
    print(f"      {'':<22}" + "".join(f"{f'z={z}':>10}" for z in Z)
          + f"{'span':>10}")
    med = np.array([np.nanmedian(err[:, k]) for k in range(5)])
    n = [int(np.isfinite(err[:, k]).sum()) for k in range(5)]
    print(f"      {'all admitted':<22}" + "".join(f"{v:>10.3f}" for v in med)
          + f"{med[0] - med[4]:>10.3f}")
    print(f"      {'n':<22}" + "".join(f"{v:>10d}" for v in n))

    print(f"\n  (2) AT FIXED HALO MASS — epoch-matched log10 M200c bins.")
    print(f"      If the span collapses here, it was the sample's mass "
          f"composition moving,\n      not the galaxies, and this stage is "
          f"aimed at nothing.")
    print(f"      {'mass bin':<22}" + "".join(f"{f'z={z}':>10}" for z in Z)
          + f"{'span':>10}")
    spans = []
    for lo, hi in zip(MASS_BINS[:-1], MASS_BINS[1:]):
        row, cnt = [], []
        for k in range(5):
            g = np.isfinite(err[:, k]) & (lmh[:, k] >= lo) & (lmh[:, k] < hi)
            row.append(np.median(err[g, k]) if g.sum() >= 25 else np.nan)
            cnt.append(int(g.sum()))
        row = np.array(row)
        sp_ = row[0] - row[4]
        spans.append(sp_)
        print(f"      {f'{lo:.1f}-{hi:.1f}':<22}"
              + "".join(f"{v:>10.3f}" if np.isfinite(v) else f"{'-':>10}"
                        for v in row) + f"{sp_:>10.3f}")
        print(f"      {'   n':<22}" + "".join(f"{v:>10d}" for v in cnt))

    print(f"\n  (2b) THE SAME TABLE READ DOWN THE COLUMNS — the central error's"
          f"\n       dependence on HALO MASS, at fixed epoch. Not asked for, "
          f"but it decides\n       which candidate is the right one.")
    print(f"      {'':<22}" + "".join(f"{f'z={z}':>10}" for z in Z))
    grad = []
    for k in range(5):
        col = []
        for lo, hi in zip(MASS_BINS[:-1], MASS_BINS[1:]):
            g = np.isfinite(err[:, k]) & (lmh[:, k] >= lo) & (lmh[:, k] < hi)
            col.append(np.median(err[g, k]) if g.sum() >= 25 else np.nan)
        col = np.array(col)
        grad.append(np.nanmax(col) - np.nanmin(col))
    print(f"      {'spread across mass':<22}"
          + "".join(f"{v:>10.3f}" for v in grad))
    print(f"      The model's central error depends on halo mass AND on epoch, "
          f"and the mass\n      gradient STEEPENS with redshift "
          f"({grad[0]:.3f} dex at z=0.4 to {np.nanmax(grad):.3f} at its worst)."
          f" That is an\n      interaction between mass and epoch, which is "
          f"exactly the functional form of\n      S4 (`b -> b + b_M m`) and "
          f"NOT of S5 (a curve in redshift alone).")

    print(f"\n  (3) THE FIXED COHORT — the same galaxies at all five epochs,")
    print(f"      which is the only set for which 'evolution' is even defined")
    coh = mask.all(axis=1)
    row = np.array([np.nanmedian(err[coh, k]) for k in range(5)])
    print(f"      {'cohort':<22}" + "".join(f"{v:>10.3f}" for v in row)
          + f"{row[0] - row[4]:>10.3f}")
    print(f"      {'n':<22}" + "".join(f"{int(coh.sum()):>10d}" for _ in Z))
    print(f"      their median log10 M200c: "
          + "  ".join(f"{np.median(lmh[coh, k]):.2f}" for k in range(5)))

    fixed = np.nanmedian(np.array(spans))
    print(f"\n{RULE}")
    print(f"  VERDICT")
    print(f"    span, all admitted galaxies      {med[0] - med[4]:+.3f} dex")
    print(f"    span, median over fixed-mass bins {fixed:+.3f} dex")
    print(f"    span, the fixed cohort           {row[0] - row[4]:+.3f} dex")
    if abs(fixed) < 0.4 * abs(med[0] - med[4]):
        print(f"    -> MOSTLY SAMPLE COMPOSITION. The span shrinks by more "
              f"than 60% at fixed\n       halo mass, so the defect is a "
              f"mass-dependent central error, not an\n       epoch-dependent "
              f"one. STOP: the size law's epoch axis is the wrong lever.")
    else:
        print(f"    -> THE SPAN IS REAL AT FIXED MASS. It survives at "
              f"{100 * abs(fixed / (med[0] - med[4])):.0f}% of its\n       "
              f"full-sample value, so it is the galaxies and not the sample. "
              f"Proceed.")
    print(RULE)
    if not smoke:
        np.savez(PRECHECK, err=err, lmh=lmh, mask=mask, rows=sel,
                 med_all=med, spans=np.array(spans), cohort_row=row,
                 mass_grad=np.array(grad), mass_bins=np.array(MASS_BINS))
        print(f"\n  wrote {PRECHECK}")
    return med, spans, row


# --------------------------------------------------------------------------- #
# the fits                                                                     #
# --------------------------------------------------------------------------- #
def nested_start(spec, th_base, spec_base):
    """The incumbent lifted into `spec`, BY PARAMETER NAME.

    Every extension in this experiment defaults its own new coefficients to the
    value that reproduces the parent — zero for the size-law and efficiency
    axes, `w0 = 0` for the compact channel — so the lift is just
    `default_theta(spec)` with the shared names overwritten from the parent's
    fit. That start guarantees the enriched fit cannot converge worse than the
    incumbent, which is what makes a negative result reportable rather than an
    optimiser failure.

    **This used to splice by POSITION**, inserting zeros after the size block.
    That is right for `S4`/`S5`/`S6`, whose parameters sit in the middle of the
    vector, and silently wrong for the compact channel, whose parameters are
    appended after the profile shape: the zeros landed on the shape parameter
    and the parent's shape landed on a compact coefficient. Matching by name
    cannot make that mistake, and `assert_lift_nests` checks the result.
    """
    th_base = np.asarray(th_base, float)
    base_names = list(spec_base.theta_names)
    out = np.asarray(M.default_theta(spec), float).copy()
    for i, n in enumerate(spec.theta_names):
        if n in base_names:
            out[i] = th_base[base_names.index(n)]
    return out


def assert_lift_nests(spec, th_base, spec_base, recs, data, mask, tol=1e-9):
    """The lifted start must score EXACTLY what the parent scores.

    Cheap, and it is the only thing standing between "the extension did not
    help" and "the extension was wired in wrong" -- which is a distinction this
    experiment has already got wrong once.
    """
    th = nested_start(spec, th_base, spec_base)
    a = S35.StackedProblem(spec_base, recs, data, mask,
                           epochs=(0, 1, 2, 3, 4)).loss(th_base)
    b = S35.StackedProblem(spec, recs, data, mask,
                           epochs=(0, 1, 2, 3, 4)).loss(th)
    if not abs(a - b) < tol:
        raise SystemExit(
            f"REFUSING TO RUN: lifting the incumbent into {spec.label} changes "
            f"its loss, {a:.12f} -> {b:.12f}. The extension does not nest as "
            f"wired, so nothing fitted from it can be compared with the "
            f"incumbent.")
    return th


def report_one(label, theta, recs, data, mask, lmh, prod_spec, prod):
    """Every diagnostic, the same for every candidate and for the bound."""
    spec = S33.spec_from_label(label)
    pr = S35.StackedProblem(spec, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    pred = pr.predict(theta)
    sa, sf = pr.per_epoch(theta)
    err = central_error(pred, data, mask)
    cen = np.array([np.nanmedian(err[:, k]) for k in range(5)])
    se = S36.shell_errors(pred, data, mask)
    os_ = S36.outer_slope(pred, data, mask)
    gr = S36.growth_report(pred, data, mask)
    # `masked_diagnostics` measures the tilt against the z=0.4 halo mass and
    # wants that column, not the whole epoch-matched array the pre-check uses
    over, tilt, _ = S35.masked_diagnostics(pr, theta, np.asarray(lmh)[:, 0])
    J, _ = F.jacobian(pr, theta)
    sv = np.linalg.svd(J, compute_uv=False)
    return dict(label=label, npar=spec.n_theta, theta=theta, sA=sa, sF=sf,
                shape_pct=S35.shape_pct(sf), central=cen,
                span=float(cen[0] - cen[4]), shells=se, slope=os_, growth=gr,
                over=over, tilt=tilt, ident=float(sv[-1] / sv[0]),
                sA0=sa[0], sA4=sa[4], sF0=sf[0], sF4=sf[4])


def print_one(r):
    print(f"\n    PARAMETERS  " + "  ".join(
        f"{n}={v:+.4f}" for n, v in
        zip(S33.spec_from_label(r["label"]).theta_names, r["theta"])))
    print(f"\n    CENTRAL ERROR, median log10(model/truth) inside 2 kpc")
    print(f"      {'':<14}" + "".join(f"{f'z={z}':>10}" for z in Z)
          + f"{'span':>10}")
    print(f"      {'this fit':<14}" + "".join(f"{v:>10.3f}" for v in r["central"])
          + f"{r['span']:>10.3f}")
    R = F.R_GRID
    print(f"    beyond 52 kpc   " + "".join(
        f"{np.nanmedian(r['shells'][k, R > 52]):>10.3f}" for k in range(5)))
    print(f"    outer slope err " + "".join(
        f"{r['slope'][k, 0] - r['slope'][k, 1]:>+10.2f}" for k in range(5)))
    print(f"    amplitude       " + "".join(f"{v:>10.3f}" for v in r["sA"]))
    g = r["growth"]
    print(f"\n    profile-shape error {r['shape_pct']:.3f}%   "
          f"outer residual {r['over']:+.4f} dex   tilt {r['tilt']:+.4f}   "
          f"ident {r['ident']:.1e}")
    print(f"    growth z=2->0.4: model {g['med_model']:+.3f}+-{g['sd_model']:.3f}"
          f"  truth {g['med_truth']:+.3f}+-{g['sd_truth']:.3f} dex")


def run(labels, n_starts=4, smoke=False, fit_only=False):
    recs, sel, data, mask, lmh, cuts = build(smoke)
    prod_spec = S33.spec_from_label(BASE)
    th_inc = incumbent()
    prod = S35.StackedProblem(prod_spec, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    print(f"exp54 STAGE 3.7 — the size law's dependence on epoch")
    print(f"  base {BASE}; {len(recs)} galaxies, {int(mask.sum())} "
          f"galaxy-epochs, sanity mask APPLIED")
    print(f"  incumbent loss {prod.loss(th_inc):.6f}, central span "
          f"{float(np.nanmedian(central_error(prod.predict(th_inc), data, mask)[:, 0])):+.3f}"
          f" -> ... (see the table per fit)\n")

    done = {}
    if not smoke:
        FITDIR.mkdir(exist_ok=True)
        for f in sorted(FITDIR.glob("*.npz")):
            z = np.load(f)
            done[str(z["label"])] = dict(theta=z["theta"], loss=float(z["loss"]))

    results = [report_one(BASE, th_inc, recs, data, mask, lmh, prod_spec, prod)]
    results[0]["loss"] = prod.loss(th_inc)
    print(f"{RULE}\n{BASE}   the incumbent, refitted in Stage 3.5\n{RULE}")
    print_one(results[0])

    for label in labels:
        spec = S33.spec_from_label(label)
        print(f"\n{RULE}\n{label}   {spec.n_theta} parameters: "
              f"{', '.join(spec.theta_names)}\n{RULE}")
        pr = S35.StackedProblem(spec, recs, data, mask, epochs=(0, 1, 2, 3, 4))
        if label in done:
            th, lo_ = np.asarray(done[label]["theta"]), done[label]["loss"]
            print(f"    loss {lo_:.6f}  (cached)")
        else:
            st = S35.starts_for(spec, n_starts)
            st.append(nested_start(spec, th_inc, prod_spec))
            t0 = time.time()
            th, lo_ = F.fit(pr, starts=st, maxiter=600 * spec.n_theta,
                            verbose=True)
            print(f"    fitted in {(time.time() - t0) / 60:.1f} min from "
                  f"{len(st)} starts, loss {lo_:.6f}", flush=True)
            if not smoke:
                FITDIR.mkdir(exist_ok=True)
                np.savez(FITDIR / f"{label}.npz", label=label, theta=th,
                         loss=lo_)
        r = report_one(label, th, recs, data, mask, lmh, prod_spec, prod)
        r["loss"] = lo_
        print_one(r)
        results.append(r)

    if smoke or fit_only:
        print(f"\n  {'smoke' if smoke else '--only'} run: writing nothing")
        return results

    print(f"\n{RULE}\nWHAT THIS STAGE IS ACTUALLY ABOUT: the central error's "
          f"SPAN across epochs\n{RULE}")
    print(f"  {'model':<26}{'p':>3}{'span':>9}{'vs base':>10}{'shape%':>9}"
          f"{'>52kpc':>9}{'tilt':>9}{'ident':>10}")
    base_span = results[0]["span"]
    for r in results:
        R = F.R_GRID
        out52 = float(np.nanmean([np.nanmedian(r["shells"][k, R > 52])
                                  for k in range(5)]))
        print(f"  {r['label'].replace(BASE, 'base'):<26}{r['npar']:>3}"
              f"{r['span']:>9.3f}{r['span'] - base_span:>+10.3f}"
              f"{r['shape_pct']:>9.3f}{out52:>9.3f}{r['tilt']:>+9.4f}"
              f"{r['ident']:>10.1e}")
    good, rk = S32.judge(results)
    order = np.argsort(rk.mean(1))
    print(f"\n  the seven-criterion judge, for continuity: "
          + " > ".join(good[i]["label"].replace(BASE, "base") for i in order))
    np.savez_compressed(
        OUT, labels=np.array([r["label"] for r in results]),
        **{f"theta::{r['label']}": r["theta"] for r in results},
        **{f"central::{r['label']}": r["central"] for r in results},
        **{f"shells::{r['label']}": r["shells"] for r in results},
        **{f"slope::{r['label']}": r["slope"] for r in results},
        **{f"sA::{r['label']}": r["sA"] for r in results},
        **{f"sF::{r['label']}": r["sF"] for r in results},
        span=np.array([r["span"] for r in results]),
        shape_pct=np.array([r["shape_pct"] for r in results]),
        tilt=np.array([r["tilt"] for r in results]),
        ident=np.array([r["ident"] for r in results]),
        npar=np.array([r["npar"] for r in results]),
        loss=np.array([r["loss"] for r in results]))
    print(f"\n  wrote {OUT}")
    return results


def span_bound(label, n_starts=6, tol=0.05, smoke=False):
    """The bound this stage actually needs: minimise the SPAN, not the loss.

    **WHY THE FIRST BOUND WAS THE WRONG ONE.** `--bound` fitted a free shared
    size law under the production loss and then reported its central span. That
    is precisely the error this project keeps re-learning: a bound must be
    optimised on the same quantity it is reported on. Fitted for the loss, the
    free curve found the BEST loss of any cell (1.8659) and a span of 0.139,
    WORSE than a one-parameter polynomial's 0.133 -- not because it could not
    reach 0.133, but because nothing asked it to.

    So here the free size law is fitted to minimise the central span directly,
    with the overall loss required to stay within `tol` of the incumbent's so
    the answer cannot be bought by making the model uniformly bad:

        minimise   span^2 + 100 * max(0, loss/loss_base - (1+tol))^2

    The result is the honest answer to "how much of the 0.145 dex span can ANY
    shared size law remove while remaining a comparably good model".
    """
    recs, sel, data, mask, lmh, cuts = build(smoke)
    spec = S33.spec_from_label(label)
    base_spec = S33.spec_from_label(BASE)
    th_inc = incumbent()
    pr = S35.StackedProblem(spec, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    pr_base = S35.StackedProblem(base_spec, recs, data, mask,
                                 epochs=(0, 1, 2, 3, 4))
    loss_base = pr_base.loss(th_inc)
    span_base = float(np.nanmedian(central_error(
        pr_base.predict(th_inc), data, mask)[:, 0])
        - np.nanmedian(central_error(
            pr_base.predict(th_inc), data, mask)[:, 4]))
    print(f"exp54 STAGE 3.7 — the SPAN bound for {label}")
    print(f"  incumbent: loss {loss_base:.6f}, central span {span_base:+.4f} dex")
    print(f"  minimising the span, with the loss held within "
          f"{100 * tol:.0f}% of the incumbent's\n")

    lo_b, hi_b = np.array(spec.bounds()).T

    def cost(th):
        if np.any(th < lo_b - 1e-9) or np.any(th > hi_b + 1e-9):
            return F.FAIL
        e = central_error(pr.predict(th), data, mask)
        if not np.isfinite(e).any():
            return F.FAIL
        sp_ = np.nanmedian(e[:, 0]) - np.nanmedian(e[:, 4])
        if not np.isfinite(sp_):
            return F.FAIL
        pen = max(0.0, pr.loss(th) / loss_base - (1.0 + tol))
        return float(sp_ ** 2 + 100.0 * pen ** 2)

    from scipy.optimize import minimize
    starts = S35.starts_for(spec, n_starts)
    starts.append(assert_lift_nests(spec, th_inc, base_spec, recs, data, mask))
    fit_p = HERE / "outputs" / "stage37_fits" / f"{label}.npz"
    if fit_p.exists():
        starts.append(np.asarray(np.load(fit_p)["theta"], float))
    best = None
    for k, s0 in enumerate(starts):
        r = minimize(cost, s0, method="Nelder-Mead",
                     options=dict(maxiter=800 * spec.n_theta, xatol=1e-5,
                                  fatol=1e-12))
        e = central_error(pr.predict(r.x), data, mask)
        sp_ = float(np.nanmedian(e[:, 0]) - np.nanmedian(e[:, 4]))
        print(f"    start {k}: span {sp_:+.4f}  loss {pr.loss(r.x):.6f}",
              flush=True)
        if best is None or r.fun < best[0]:
            best = (r.fun, r.x, sp_)
    _, th, sp_ = best
    e = central_error(pr.predict(th), data, mask)
    cen = np.array([np.nanmedian(e[:, k]) for k in range(5)])
    print(f"\n  BEST: span {sp_:+.4f} dex against the incumbent's "
          f"{span_base:+.4f}, loss {pr.loss(th):.6f}")
    print(f"  central error " + "  ".join(f"{v:+.3f}" for v in cen))
    print(f"  parameters    " + "  ".join(
        f"{n}={v:+.3f}" for n, v in zip(spec.theta_names, th)))
    at_bound = [n for n, v, a, b2 in zip(spec.theta_names, th, lo_b, hi_b)
                if abs(v - a) < 1e-6 or abs(v - b2) < 1e-6]
    if at_bound:
        print(f"  WARNING: at a parameter bound: {at_bound} — the reported "
              f"span is then a\n           property of the bound, not of the "
              f"model")
    if not smoke:
        # save the FULL diagnostic set, not just the span: what this solution
        # costs elsewhere is the whole point, and a figure that had to hardcode
        # those numbers would go stale the first time the fit was re-run
        base_prod = S35.StackedProblem(base_spec, recs, data, mask,
                                       epochs=(0, 1, 2, 3, 4))
        rep = report_one(label, th, recs, data, mask, lmh, base_spec, base_prod)
        np.savez(HERE / "outputs" / f"stage37_spanbound_{label}.npz",
                 label=label, theta=th, span=sp_, central=cen,
                 loss=pr.loss(th), span_base=span_base, loss_base=loss_base,
                 tol=tol, shells=rep["shells"], slope=rep["slope"],
                 sA=rep["sA"], sF=rep["sF"], shape_pct=rep["shape_pct"],
                 tilt=rep["tilt"], ident=rep["ident"])
    return sp_, th


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    ns = (int(sys.argv[sys.argv.index("--starts") + 1])
          if "--starts" in sys.argv else 4)
    only = (sys.argv[sys.argv.index("--only") + 1]
            if "--only" in sys.argv else None)
    if "--precheck" in sys.argv:
        precheck(smoke)
    elif "--spanbound" in sys.argv:
        lab = (sys.argv[sys.argv.index("--spanbound") + 1]
               if len(sys.argv) > sys.argv.index("--spanbound") + 1
               and not sys.argv[sys.argv.index("--spanbound") + 1].startswith("-")
               else BOUNDS[0])
        span_bound(lab, n_starts=ns, smoke=smoke)
    elif only:
        run([only], ns, smoke, fit_only=True)
    elif "--bound" in sys.argv:
        run(list(BOUNDS), ns, smoke)
    else:
        run(list(BOUNDS) + list(CANDIDATES), ns, smoke)
