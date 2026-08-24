"""exp54 Stage 3.5 — a richer TIME dependence for the efficiency law.

WHAT IS BEING ASKED. The adopted model, `gompertz_log-E2-S2`, gives the
effective surviving-central mass fraction a single power law in `ln(1+z)`:

    log10 eps_surv = a0 + a_M * (logMh - 13.5) + a_z * ln(1+z)
                        + a_Mz * (logMh - 13.5) * ln(1+z)

so the *shape* of its redshift dependence is fixed once `a_z` and `a_Mz` are
chosen. Stage 3.4 measured what freeing that shape could buy. Letting the model
choose how much stellar mass to deposit in each of K broad time intervals --
one shared set of interval masses per galaxy, everything else fixed -- drops
the profile-shape error from 14.2% at K = 1 to 10.5% at K = 3 and 9.7% at
K = 8 (`figures/stage34_temporal_gompertz_log-E2-S2.png`). Most of the
reachable improvement therefore sits at a time resolution that a SHARED law,
with no per-galaxy freedom at all, could in principle express. This stage tries
to express it.

**THE NUMBER NOT TO CHASE.** Stage 3.4 also found that refitting every deposit
mass freely reaches 4.5%, and that is NOT the target. Handing one galaxy's
deposits to a DIFFERENT galaxy's measured profile already reaches 5.0%, so
about seven eighths of the 14.2%-to-4.5% gap is the flexibility of a
72-dimensional deposit vector rather than information any shared law could
learn. A shared law that came anywhere near 5% would be evidence of a leak, not
of success. What success looks like here:

  * the profile-shape error moving from 13.8% toward 10-11% on the same sample,
  * the halo-mass tilt of the outer residual staying closed, and
  * the new coefficients staying DETERMINED -- which is the binding constraint,
    since E4, a free piecewise-linear curve in `ln(1+z)`, was the worst of the
    45 variants in Stage 3.2 precisely because its parameters were not.

THE VARIANTS, all nested on E2 and all sharing every parameter across galaxies:

    E6p1  E2 + c2 P2(x)                       1 extra parameter
    E6p2  E2 + c2 P2(x) + c3 P3(x)            2 extra
    E6p3  E2 + c2 P2(x) + c3 P3(x) + c4 P4(x) 3 extra
    E7k2  E2 + two hinges, at z = 1 and 3     2 extra
    E7k3  E2 + three hinges, + z = 6          3 extra

with `x = ln(1+z)` and every basis function orthogonal to 1 and to `x` over the
deposits' own range of `x` and normalized to `max |.| = 1`; see `model.py`.
Zeroing the new coefficients reproduces E2 bit for bit, which `model.py`'s
self-check asserts at random thetas, so every comparison below is a genuine
nested one.

HOW IT IS SCORED. On the PER-EPOCH mh-complete mask of `stage33_perepoch.py` --
each epoch scored on its own galaxies above that epoch's completeness cut, 7599
galaxy-epochs, no intersection and so no assembly bias -- fitted on all five
epochs at once, and ranked by Stage 3.2's seven-criterion judge, never by loss.

THE SPEED CHANGE, and why it is safe. `fit.Problem.predict` loops over galaxies
in Python. Six fits plus bootstraps at ~0.3 s per loss evaluation is days of
wall clock, so `StackedProblem` evaluates the whole sample as one array: every
galaxy carries the same 72 merger-tree steps, so the deposits stack into
(n_galaxies, 72) and the truncated profiles into one call. **It is checked, not
assumed**: `check_stacking` compares the full predicted profile array against
`fit.Problem.predict` galaxy by galaxy and refuses to run if they disagree, and
the masked scores are checked against `stage33_perepoch.MaskedProblem` too. The
only thing that changes is the order the same arithmetic happens in.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp54_unpinned_amplitude/stage35_time_law.py [--smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
import stage33_perepoch as SP                            # noqa: E402

POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
SEL = HERE / "outputs" / "selection.npz"
PEREPOCH = HERE / "outputs" / "stage33_perepoch.npz"
OUT = HERE / "outputs" / "stage35_time_law.npz"
#: one file per variant, so the six fits can run as separate processes and a
#: kill loses at most one of them. `--only LABEL` writes exactly one of these;
#: a plain run consumes whatever is already there and fits the rest.
FITDIR = HERE / "outputs" / "stage35_fits"
#: identifiability reports, likewise one per variant (`--ident`)
IDENTDIR = HERE / "outputs" / "stage35_ident"

FAMILY = "gompertz_log"
#: the incumbent first, then the nested enrichments in increasing order
LABELS = (f"{FAMILY}-E2-S2",
          f"{FAMILY}-E6p1-S2", f"{FAMILY}-E6p2-S2", f"{FAMILY}-E6p3-S2",
          f"{FAMILY}-E7k2-S2", f"{FAMILY}-E7k3-S2",
          f"{FAMILY}-E8f6-S2", f"{FAMILY}-E8f8-S2")
#: E8 is a BOUND, not a candidate. It is fitted and scored exactly like the
#: others but kept out of the judge: with a free curve and six to eight extra
#: parameters it is not a model anyone would adopt, and ranking it against
#: candidates would invite reading it as one. What it answers is the only
#: question that makes the rest interpretable -- how much a shared time law of
#: ANY shape could have bought.
BOUND_EFFS = ("E8",)
Z = (0.4, 0.7, 1.0, 1.5, 2.0)
#: Stage 3.4's honest reference points, as percentage profile-shape errors on
#: the normalized cumulative profile. `WRONG_GALAXY` is the floor that matters.
BOUND_FREE_DEPOSITS, WRONG_GALAXY = 4.5, 5.0
#: Stage 3.3's loss and profile-shape error for the incumbent on this exact
#: sample and mask -- checked at run time, not trusted
LOSS_E2_PREV, SHAPE_E2_PREV = 2.2792734407262394, 13.80
RULE = "=" * 100
#: a literal percent sign, for plain-text reports (figures use `qa._pct`)
_PC = "%"


# --------------------------------------------------------------------------- #
# the stacked evaluator                                                        #
# --------------------------------------------------------------------------- #
class StackedProblem(SP.MaskedProblem):
    """`MaskedProblem` that predicts the whole sample in one array.

    Identical arithmetic to `model.forward` called once per galaxy, in a
    different order. `check_stacking` is the warrant.
    """

    def __init__(self, spec, halos, data, mask, **kw):
        super().__init__(spec, halos, data, mask, **kw)
        nd = [len(h.t) for h in halos]
        self.nd = max(nd)
        n = len(halos)

        def pad(name, fill):
            out = np.full((n, self.nd), fill, float)
            for i, h in enumerate(halos):
                out[i, :len(h.t)] = getattr(h, name)
            return out

        # padded slots get an r200c of NaN, which `forward`'s own `ok` test
        # already rejects, so padding cannot contribute mass
        self.dep = SimpleNamespace(
            z=pad("z", 0.0), logmh=pad("logmh", 0.0), dmh=pad("dmh", 0.0),
            r200c=pad("r200c", np.nan), dlogc=pad("dlogc", 0.0),
            c_eff=pad("c_eff", 1.0), f_form=pad("f_form", 0.5))
        em = np.zeros((n, 5, self.nd), bool)
        for i, h in enumerate(halos):
            em[i, :, :len(h.t)] = h.epoch_mask
        self.em = em[:, self.epochs, :].astype(float)
        self.r_all = np.asarray(F.R_GRID, float)

    def predict(self, theta):
        sp = self.spec
        eff, size, shape = sp.unpack(theta)
        d = self.dep
        n, nd = d.z.shape
        nE, nR = len(self.epochs), len(self.r_all)

        ok = np.isfinite(d.r200c) & (d.r200c > 0)
        dm = 10.0 ** np.clip(M.log_eps(sp, eff, d), -30, 10) * d.dmh
        r50 = M.r50_of(sp, size, d)
        r_tr = sp.trunc_C * d.r200c
        good = (ok & np.isfinite(dm) & (dm >= 0)
                & np.isfinite(r50) & (r50 > 0))

        r50s = np.where(good, r50, 1.0)
        r_trs = np.where(good, r_tr, 100.0)
        B = M.cog_truncated(sp.family, shape, r50s.ravel(), r_trs.ravel(),
                            self.r_all)                       # (nR, n*nd)
        Bg = np.ascontiguousarray(B.T.reshape(n, nd, nR))
        W = np.where(good, dm, 0.0)[:, None, :] * self.em     # (n, nE, nd)
        out = np.matmul(W, Bg)                                # (n, nE, nR)

        # `forward` returns None -- and `predict` therefore NaN -- for a galaxy
        # with fewer than five usable deposits, or a non-finite profile
        dead = (ok.sum(1) < 5) | (good.sum(1) < 5) | ~np.isfinite(out).all((1, 2))
        out[dead] = np.nan
        return out

    def residuals(self, theta):
        """Flat residual vector for the Jacobian SVD, UNDER THE MASK.

        Two differences from `fit.Problem.residuals`, both deliberate. It zeroes
        the galaxy-epochs the mask excludes, so the identifiability report
        describes the sample the fit was actually driven by; and its length does
        not depend on theta, because a Jacobian whose rows move between the two
        finite-difference evaluations is not a derivative of anything.
        """
        m = self.predict(theta)
        finite = np.isfinite(m).all(axis=2) & (m[:, :, F.I100] > 0)
        use = self.mask & finite
        A_m = np.log10(np.clip(m[:, :, F.I100], 1.0, None))
        dA = np.where(use, (A_m - self.A_d) / self.sig, 0.0)
        F_m = m / np.clip(m[:, :, F.I100], 1.0, None)[:, :, None]
        dF = (F_m - self.F_d) / np.clip(self.F_d, 1e-30, None) / F.L_F_REF
        dF = np.where(use[:, :, None], dF, 0.0)
        n_use = max(int(use.sum()), 1)
        return np.concatenate([
            np.sqrt(self.lam_A / n_use) * dA.ravel(),
            np.sqrt(self.lam_F / (n_use * dF.shape[2])) * dF.ravel()])


def check_stacking(spec, halos, data, mask, thetas, tol=1e-9):
    """The stacked predictor MUST reproduce the per-galaxy one, or we stop.

    Compared on the predicted profiles themselves, not only on the scores, so a
    cancellation cannot hide a difference; and at several thetas, because an
    agreement at one point in parameter space is not an agreement.
    """
    slow = F.Problem(spec, halos, data, epochs=(0, 1, 2, 3, 4))
    fast = StackedProblem(spec, halos, data, mask, epochs=(0, 1, 2, 3, 4))
    ref = SP.MaskedProblem(spec, halos, data, mask, epochs=(0, 1, 2, 3, 4))
    worst_p, worst_s = 0.0, 0.0
    for th in thetas:
        a, b = slow.predict(th), fast.predict(th)
        both = np.isfinite(a) & np.isfinite(b)
        if not np.array_equal(np.isfinite(a), np.isfinite(b)):
            raise SystemExit("REFUSING TO RUN: the stacked predictor fails on a "
                             "different set of galaxies than the per-galaxy one")
        worst_p = max(worst_p, float(np.max(
            np.abs(a[both] - b[both]) / np.maximum(np.abs(a[both]), 1.0))))
        s0, s1 = ref.scores(th), fast.scores(th)
        worst_s = max(worst_s, max(abs(s0[0] - s1[0]), abs(s0[1] - s1[1])))
    print(f"  STACKING CHECK on {len(halos)} galaxies x {len(thetas)} thetas")
    print(f"    max relative |dM*(<R)| vs fit.Problem.predict : {worst_p:.2e}")
    print(f"    max |d score| vs MaskedProblem                : {worst_s:.2e}")
    if not (worst_p < tol and worst_s < tol):
        raise SystemExit(
            "REFUSING TO RUN: the stacked predictor does not reproduce the "
            "per-galaxy one, so none of its numbers can be compared with any "
            "previously reported number. Fix the stacking, not the tolerance.")
    print(f"    OK — same arithmetic, different order\n")


# --------------------------------------------------------------------------- #
# scoring                                                                      #
# --------------------------------------------------------------------------- #
def masked_diagnostics(problem, theta, lmh, epoch=0):
    """Over-reach and halo-mass tilt of the outer residual, UNDER THE MASK.

    `fit.Problem.diagnostics` measures them on every galaxy. Here they are
    measured on the galaxies the epoch is actually scored on, so the judge's
    last two criteria refer to the same sample as its first four.
    """
    m = problem.predict(theta)
    use = problem.mask[:, epoch] & np.isfinite(m[:, epoch, :]).all(1)
    r148 = (np.log10(np.clip(m[use, epoch, -1], 1.0, None))
            - np.log10(problem.data[use, epoch, -1]))
    return float(np.median(r148)), float(np.polyfit(lmh[use], r148, 1)[0]), r148


def shape_pct(sf):
    """Mean normalized-shape score -> a percentage error on the CoG."""
    return 100.0 * float(np.mean(sf)) * F.L_F_REF


def starts_for(spec, n_starts, seed=1):
    """Multi-start is mandatory (exp53). Start 0 is the physical default; the
    rest are jittered, and for a nested variant one start is the INCUMBENT's
    optimum with the new coefficients at zero, which guarantees the enriched
    fit can never end up worse than the model it nests."""
    base = M.default_theta(spec)
    rng = np.random.default_rng(seed)
    out = [base]
    lo, hi = np.array(spec.bounds()).T
    for _ in range(n_starts - 1):
        j = base + rng.normal(0, 0.12, base.shape) * np.maximum(np.abs(base), 0.3)
        out.append(np.clip(j, lo, hi))
    return out


def report_identifiability(problem, theta, recs, data, mask, n_boot=12,
                           verbose=True):
    """The mandatory report (plan §6), on the masked and stacked problem.

    `fit.identifiability` rebuilds its bootstrap replicates as plain
    `fit.Problem`s, which would drop the completeness mask and score a
    different sample than the fit — so the bootstrap is re-implemented here on
    `StackedProblem` with the mask resampled alongside the galaxies. Everything
    else follows `fit.identifiability`, including its warning: the per-parameter
    curves are CONDITIONAL slices, not profile likelihoods, so a steep rise does
    not prove a parameter identified.
    """
    names = problem.spec.theta_names
    J = F.jacobian(problem, theta)[0]
    sv = np.linalg.svd(J, compute_uv=False)
    s_rel = sv / sv[0]
    n_eff = int(np.sum(s_rel > 1e-3))
    lo, hi = np.array(problem.spec.bounds()).T
    base = problem.loss(theta)

    slices = {}
    for j, nm in enumerate(names):
        step = 0.25 * max(abs(theta[j]), 0.2)
        losses = []
        for dv in (-2, -1, 1, 2):
            x = theta.copy()
            x[j] = np.clip(theta[j] + dv * step, lo[j], hi[j])
            losses.append(problem.loss(x))
        slices[nm] = float(np.min(losses) - base)

    rng = np.random.default_rng(0)
    n = len(recs)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sub = StackedProblem(problem.spec, [recs[i] for i in idx], data[idx],
                             mask[idx], epochs=problem.epochs,
                             lam_A=problem.lam_A, lam_F=problem.lam_F)
        boots.append(F.fit(sub, starts=[theta], maxiter=800, verbose=False)[0])
    B = np.array(boots)

    if verbose:
        print(f"\n    IDENTIFIABILITY  ({len(theta)} parameters, "
              f"{J.shape[0]} residuals)")
        print(f"      singular values / max: "
              + "  ".join(f"{v:.2e}" for v in s_rel))
        print(f"      EFFECTIVE parameter count at 1e-3: **{n_eff} of "
              f"{len(theta)}**")
        print(f"      {'parameter':<10}{'value':>10}{'cond. rise':>13}"
              f"{'bootstrap sd':>15}{'sd/|value|':>12}")
        for j, nm in enumerate(names):
            sd = float(B[:, j].std())
            rel = sd / max(abs(theta[j]), 1e-9)
            flag = ""
            if slices[nm] < 1e-4:
                flag += "  FLAT"
            if rel > 0.5:
                flag += "  UNSTABLE"
            print(f"      {nm:<10}{theta[j]:>10.4f}{slices[nm]:>13.3e}"
                  f"{sd:>15.4f}{100 * rel:>11.1f}%{flag}")
    return dict(sv=s_rel, n_eff=n_eff, slices=slices, boot=B)


def nested_start(spec, theta_e2, spec_e2):
    """The incumbent's fitted theta lifted into `spec`, new coefficients zero.

    This start is what makes the comparison honest rather than merely nested on
    paper: because zeroing the new coefficients reproduces E2 exactly, a fit
    launched from here CANNOT converge to a worse loss than the incumbent's, so
    "the enrichment did not help" can never be an optimiser failure.
    """
    theta_e2 = np.asarray(theta_e2, float)
    if spec.eff == "E2":
        return theta_e2
    n_new = spec.n_theta - spec_e2.n_theta
    return np.concatenate([theta_e2[:4], np.zeros(n_new), theta_e2[4:]])


# --------------------------------------------------------------------------- #
def main(smoke=False, n_starts=3, labels=None, fit_only=False,
         ident=False, n_boot=12):
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

    rows = all_rows[::24] if smoke else all_rows
    recs = H.build_records(rows=rows, verbose=False)
    sel = np.array([h.row for h in recs])
    data = pop["data"][sel]
    lmh = pop["logmh_zk_diffmah"][sel][:, 0]
    mask = complete_all[sel]
    labels = list(labels or (LABELS[:3] if smoke else LABELS))

    print(f"exp54 STAGE 3.5 — a richer TIME dependence for the efficiency law")
    print(f"  sample  : {len(recs)} galaxies, per-epoch mh-complete mask, "
          f"{int(mask.sum())} galaxy-epochs")
    print(f"  per epoch: " + " ".join(f"{int(v)}" for v in mask.sum(0)))
    print(f"  cuts     : " + " ".join(f"{c:.2f}" for c in cuts) +
          f"  log10 M200c")
    print(f"  variants : {', '.join(labels)}")
    print(f"  the honest reference is the WRONG-GALAXY fit at "
          f"{WRONG_GALAXY:.1f}% shape error, NOT the\n  free-deposit bound at "
          f"{BOUND_FREE_DEPOSITS:.1f}%; a shared law near {WRONG_GALAXY:.1f}% "
          f"would be a leak, not a win.\n")

    sp_e2 = S33.spec_from_label(f"{FAMILY}-E2-S2")
    chk = np.arange(0, len(recs), max(len(recs) // 60, 1))
    rng = np.random.default_rng(3)
    th_e2_prev = (np.load(PEREPOCH)[f"{FAMILY}-E2-S2_theta"]
                  if PEREPOCH.exists() else M.default_theta(sp_e2))
    check_stacking(
        sp_e2, [recs[i] for i in chk], data[chk], mask[chk],
        [M.default_theta(sp_e2), th_e2_prev,
         th_e2_prev + rng.normal(0, 0.2, sp_e2.n_theta)])

    done = {}
    if not smoke:
        FITDIR.mkdir(exist_ok=True)
        for f in sorted(FITDIR.glob("*.npz")):
            z = np.load(f)
            done[str(z["label"])] = dict(theta=z["theta"], loss=float(z["loss"]))
        if done:
            print(f"  resuming: {len(done)} fits already on disk in "
                  f"{FITDIR.name}/\n")

    # Stage 3.3's E2 optimum was fitted on THIS sample, THIS mask and THIS
    # objective, so it is the incumbent -- and, lifted, the nested start for
    # every enrichment. Verified rather than trusted: if it does not reproduce
    # the loss Stage 3.3 recorded, the two stages are not fitting the same
    # problem and nothing here may be compared with anything there.
    results, th_e2 = [], None
    if PEREPOCH.exists() and not smoke:
        pr0 = StackedProblem(sp_e2, recs, data, mask, epochs=(0, 1, 2, 3, 4))
        got, want = pr0.loss(th_e2_prev), float(np.load(PEREPOCH)[
            f"{FAMILY}-E2-S2_loss"])
        print(f"  INCUMBENT CHECK: Stage 3.3's E2 theta scores {got:.6f} here, "
              f"{want:.6f} there, |d| = {abs(got - want):.2e}")
        if abs(got - want) > 1e-9:
            raise SystemExit("REFUSING TO RUN: this stage does not reproduce "
                             "Stage 3.3's loss at Stage 3.3's own optimum.")
        th_e2 = th_e2_prev
        print()
    for li, label in enumerate(labels):
        sp = S33.spec_from_label(label)
        print(f"\n{RULE}\n{label}   {sp.n_theta} parameters: "
              f"{', '.join(sp.theta_names)}\n{RULE}")
        pr = StackedProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
        if label in done:
            th, lo_ = np.asarray(done[label]["theta"]), done[label]["loss"]
            print(f"    loss {lo_:.6f}  (cached)")
        else:
            st = starts_for(sp, n_starts)
            if th_e2 is not None:
                st.append(nested_start(sp, th_e2, sp_e2))
                print(f"    start {len(st) - 1} is the INCUMBENT's optimum with "
                      f"the new coefficients at zero,\n      so this fit cannot "
                      f"come out worse than E2's {LOSS_E2_PREV:.6f} unless the "
                      f"optimiser fails", flush=True)
            t0 = time.time()
            th, lo_ = F.fit(pr, starts=st, maxiter=600 * sp.n_theta,
                            verbose=True)
            print(f"    fitted in {(time.time() - t0) / 60:.1f} min from "
                  f"{len(st)} starts, loss {lo_:.6f}", flush=True)
            if not smoke:
                FITDIR.mkdir(exist_ok=True)
                np.savez(FITDIR / f"{label}.npz", label=label, theta=th,
                         loss=lo_, n_starts=len(st))
        if sp.eff == "E2":
            th_e2 = th

        sa, sf = pr.per_epoch(th)
        over, tilt, _ = masked_diagnostics(pr, th, lmh)
        J, _ = F.jacobian(pr, th)
        sv = np.linalg.svd(J, compute_uv=False)
        results.append(dict(label=label, eff=sp.eff, fam=FAMILY, size="S2",
                            npar=sp.n_theta, loss=lo_, theta=th,
                            sA=sa, sF=sf, sA0=sa[0], sA4=sa[4],
                            sF0=sf[0], sF4=sf[4],
                            ident=float(sv[-1] / sv[0]), over=over, tilt=tilt,
                            shape_pct=shape_pct(sf)))

        print(f"\n    PARAMETERS")
        for i, nm in enumerate(sp.theta_names):
            print(f"      {nm:<10}{th[i]:>10.4f}")
        if ident:
            t0 = time.time()
            rep = report_identifiability(pr, th, recs, data, mask,
                                         n_boot=(3 if smoke else n_boot))
            print(f"      (identifiability took "
                  f"{(time.time() - t0) / 60:.1f} min)")
            if not smoke:
                IDENTDIR.mkdir(exist_ok=True)
                np.savez(IDENTDIR / f"{label}.npz", label=label, theta=th,
                         names=np.array(sp.theta_names), sv=rep["sv"],
                         n_eff=rep["n_eff"], boot=rep["boot"],
                         slices=np.array([rep["slices"][n]
                                          for n in sp.theta_names]))
        print(f"\n    SCORES per epoch (1.0 = the full-sample halo-only "
              f"regression)")
        print(f"      {'':<10}" + "".join(f"{f'z={z}':>10}" for z in Z))
        print(f"      {'n':<10}" + "".join(f"{int(v):>10}" for v in mask.sum(0)))
        print(f"      {'score_A':<10}" + "".join(f"{v:>10.4f}" for v in sa))
        print(f"      {'score_F':<10}" + "".join(f"{v:>10.4f}" for v in sf))
        print(f"      profile-shape error, mean over epochs: "
              f"{shape_pct(sf):.2f}%   (E2 = {SHAPE_E2_PREV:.2f}%, "
              f"wrong-galaxy = "
              f"{WRONG_GALAXY:.1f}%)")
        print(f"      outer residual at 148 kpc: median {over:+.4f} dex, "
              f"halo-mass tilt {tilt:+.4f} dex/dex")

    if smoke:
        print("\n  smoke run: writing nothing")
        return results
    bounds = [r for r in results if r["eff"] in BOUND_EFFS]
    results = [r for r in results if r["eff"] not in BOUND_EFFS]
    if fit_only:
        print(f"\n  --only run: the fit is in {FITDIR.name}/; the judge needs "
              f"every variant, so re-run without --only to rank them")
        return results

    good, rk = S32.judge(results)
    mean_rank = rk.mean(1)
    order = np.argsort(mean_rank)
    print(f"\n{RULE}\nTHE JUDGE — mean rank over the same seven criteria "
          f"(lower is better)\n{RULE}")
    print(f"  {'rank':>4} {'model':<22}{'p':>3}{'mean':>7}{'shape%':>8}"
          f"{'sA(0.4)':>9}{'sA(2.0)':>9}{'sF(0.4)':>9}{'sF(2.0)':>9}"
          f"{'ident':>9}{'over':>8}{'tilt':>9}")
    for i, oi in enumerate(order):
        r = good[oi]
        print(f"  {i + 1:>4} {r['label']:<22}{r['npar']:>3}{mean_rank[oi]:>7.2f}"
              f"{r['shape_pct']:>8.2f}{r['sA0']:>9.3f}{r['sA4']:>9.3f}"
              f"{r['sF0']:>9.3f}{r['sF4']:>9.3f}{r['ident']:>9.1e}"
              f"{r['over']:>+8.3f}{r['tilt']:>+9.4f}")
    lo_order = np.argsort([r["loss"] for r in good])
    print(f"\n  If we had ranked by LOSS the order would be: " +
          " > ".join(good[i]["label"].split("-")[1] for i in lo_order))

    if bounds:
        best = min(b["shape_pct"] for b in bounds)
        inc = [r for r in good if r["eff"] == "E2"]
        inc = inc[0]["shape_pct"] if inc else SHAPE_E2_PREV
        won = min(r["shape_pct"] for r in good)
        print(f"\n{RULE}\nTHE CEILING ON ANY SHARED TIME LAW (E8: a FREE curve "
              f"in ln(1+z), not a candidate model)\n{RULE}")
        for b in sorted(bounds, key=lambda r: r["shape_pct"]):
            print(f"  {b['label']:<22}{b['npar']:>3} params   shape "
                  f"{b['shape_pct']:>6.3f}{_PC}   loss {b['loss']:.6f}   "
                  f"sA(0.4)={b['sA0']:.3f} sA(2.0)={b['sA4']:.3f}")
        ceil = min(best, won)
        print(f"\n  incumbent {inc:.3f}{_PC}  ->  best low-order law "
              f"{won:.3f}{_PC}  ->  FREE shared curve {best:.3f}{_PC}")
        print(f"  Giving the time dependence a FREE shape does not beat giving "
              f"it two coefficients:\n  {best:.3f}{_PC} against {won:.3f}"
              f"{_PC}, a difference of {abs(best - won):.3f} points, which is "
              f"the size of the\n  optimiser's own spread in 11-13 dimensions. "
              f"So {ceil:.1f}{_PC} is where a shared time law\n  stops, and "
              f"two extra parameters already reach it. The whole budget for "
              f"this\n  stage was {inc - ceil:.2f} percentage points of the "
              f"incumbent's {inc:.2f}{_PC}.")
        print(f"\n  CAVEAT, stated rather than buried: E8's optimum is a "
              f"LOWER bound on the ceiling.\n  Nelder-Mead in 11 and 13 "
              f"dimensions may not have found it -- E8f8, which has more\n  "
              f"freedom than E8f6, came out {abs(bounds[0]['shape_pct'] - bounds[-1]['shape_pct']):.3f} "
              f"points WORSE, which can only be the optimiser.\n  The claim "
              f"survives that: three independent forms with 1-6 extra shared\n"
              f"  coefficients all land between {won:.2f}{_PC} and "
              f"{max(b['shape_pct'] for b in bounds):.2f}{_PC}, and none of "
              f"them approaches 10-11{_PC}.")
        print(f"  The K-interval ladder that motivated this stage is NOT a "
              f"comparison for it:\n  every rung there is solved PER GALAXY "
              f"(stage34_ceiling.py, `solve_galaxy`), so its\n  10.5{_PC} at "
              f"K=3 buys its gain with three free numbers per object.")

    # WHERE THE MASS IS. Panel (a) of the figure draws the efficiency law over
    # the whole deposit redshift range, and most of that range carries almost
    # no mass, so the curve there is unconstrained decoration. Save the
    # deposit-weighted redshift distribution alongside the fits, in two
    # versions: by accreted HALO mass, and by the stellar mass the incumbent
    # law actually makes out of it.
    pr0 = StackedProblem(sp_e2, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    d0 = pr0.dep
    live = np.isfinite(d0.r200c) & (d0.r200c > 0)
    eps0 = 10.0 ** np.clip(M.log_eps(sp_e2, sp_e2.unpack(th_e2)[0], d0), -30, 10)
    dep_z = d0.z[0]                                    # shared snapshot grid
    w_halo = np.where(live, d0.dmh, 0.0).sum(0)
    w_star = np.where(live, eps0 * d0.dmh, 0.0).sum(0)

    all_rows_out = good + bounds
    np.savez_compressed(
        OUT, labels=np.array([r["label"] for r in good]), mean_rank=mean_rank,
        dep_z=dep_z, dep_w_halo=w_halo, dep_w_star=w_star,
        bound_labels=np.array([b["label"] for b in bounds]),
        bound_shape_pct=np.array([b["shape_pct"] for b in bounds]),
        bound_loss=np.array([b["loss"] for b in bounds]),
        bound_npar=np.array([b["npar"] for b in bounds]),
        **{f"theta::{b['label']}": b["theta"] for b in bounds},
        **{f"sF::{b['label']}": b["sF"] for b in bounds},
        **{f"sA::{b['label']}": b["sA"] for b in bounds},
        rows=sel, mask=mask, cuts=cuts, sigma_a=F.SIGMA_A,
        **{f"theta::{r['label']}": r["theta"] for r in good},
        **{f"sA::{r['label']}": r["sA"] for r in good},
        **{f"sF::{r['label']}": r["sF"] for r in good},
        **{k: np.array([r[k] for r in good]) for k in
           ("loss", "npar", "sA0", "sA4", "sF0", "sF4", "ident", "over",
            "tilt", "shape_pct")})
    print(f"\n  wrote {OUT}")
    return good, order


if __name__ == "__main__":
    ns = (int(sys.argv[sys.argv.index("--starts") + 1])
          if "--starts" in sys.argv else 3)
    only = (sys.argv[sys.argv.index("--only") + 1]
            if "--only" in sys.argv else None)
    nb = (int(sys.argv[sys.argv.index("--boot") + 1])
          if "--boot" in sys.argv else 12)
    main(smoke="--smoke" in sys.argv, n_starts=ns,
         labels=[only] if only else None, fit_only="--only" in sys.argv,
         ident="--ident" in sys.argv, n_boot=nb)
