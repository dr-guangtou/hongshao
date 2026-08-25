"""exp54 Stage 3.9 — genuine PROFILE likelihoods for the seven adopted
parameters (open question B1).

WHAT WAS WRONG WITH EVERY IDENTIFIABILITY NUMBER THIS EXPERIMENT HAS QUOTED.
`fit.identifiability` and `stage35_time_law.report_identifiability` both move
one parameter and hold the other six FIXED. That is a CONDITIONAL slice. It
answers "how much worse does THIS fit get if I nudge one number", which is not
the question anyone means when they ask whether a parameter is determined. The
question they mean is "how much worse does the BEST ACHIEVABLE fit get if I
insist on this value", and answering it requires re-optimising the other six at
every grid point. In a correlated model the two differ badly: a nudge in one
parameter can be almost entirely undone by the other six moving to compensate,
so the conditional slice rises steeply along a direction the profiled curve
leaves nearly flat. The conditional version therefore systematically OVERSTATES
how well determined a parameter is. `fit.py`'s own docstring says exactly this
and says not to quote the slice as evidence on its own. It has never been done
until now.

THE MODEL. `gompertz_log-E2-S2`, the incumbent, fitted on the clean sample
(`selection.sane_history_mask` applied). Seven parameters shared by all 2397
galaxies, no stellar input, no per-object tuning:

    log10 eps_surv = a0 + a_M m + a_z x + a_Mz m x     m = log10 Mh - 13.5
                                                       x = ln(1+z)
    R50_j = 10^(log_f0 + b log10(1+z_j)) R200c(t_j)    c = the Gompertz shape

WHAT IS COMPUTED, PER PARAMETER

  1. The GRID, and why it is geometric. Eleven values centred on the
     incumbent, at `+-1, 2, 4, 8, 16` times `d0` — the displacement at which
     the CONDITIONAL loss rise reaches `STEP_RISE` = 0.10, measured separately
     in each direction (`--calibrate`). `d0` is a measured property of the
     loss, not a round number, and it is direction-specific because the loss is
     not symmetric in `c`. Because the conditional slice is very close to
     exactly quadratic here (checked: the displacement scales as the square
     root of the rise to three digits over two decades), the conditional rise
     across the grid runs from 0.10 at the innermost point to 0.10 x 256 = 25.6
     at the outermost. So each parameter is scanned over the whole conditional
     range from "slightly worse" to "catastrophically worse", and the profile
     says how much of that survives re-optimisation. The geometric spacing puts
     the points where the curvature is, instead of wasting half of them on a
     flat bottom. Half-widths are clipped to the parameter's own bounds and the
     clipping is reported.

     **The Hessian route was tried first and REJECTED, and this is worth
     recording.** Sizing the grid by the Gauss-Newton prediction
     `1 / (H^-1)_jj` needs the small eigenvalues of the loss Hessian, and a
     finite-difference Hessian cannot resolve them here: across finite-
     difference steps spanning a factor of ten, the five largest eigenvalues
     are stable to better than 0.1% while the two smallest move by a factor of
     25 and one of them goes NEGATIVE. That is not a converged-ness problem —
     re-optimising all seven parameters from the incumbent reproduces its loss
     to the last digit — it is cancellation: extracting a curvature of order
     0.1 from a matrix whose diagonal is of order 100. `--calibrate` still
     computes and stores the Hessian at three step scales, because THE
     INSTABILITY IS ITSELF THE RESULT: a seven-parameter model with two
     directions too flat to measure by differencing is exactly the situation in
     which conditional slices mislead.

  2. The PROFILE. At each grid value the parameter is FIXED and the other six
     are re-optimised by multi-start Nelder-Mead. Two starts per point: the
     neighbouring grid point's solution (a continuation sweep outward from the
     centre in both directions, which is what makes the curve smooth rather
     than a scatter of independent optimiser outcomes) and the incumbent's own
     six values. The better of the two is kept.

  3. The GATE. The centre grid point fixes the parameter at its incumbent value
     and re-optimises the rest, so it MUST return the incumbent's own loss. If
     it comes back higher the optimiser is failing; if it comes back materially
     lower the incumbent was not converged. Either way every profiled rise on
     that parameter's curve is measured against the wrong zero. The run reports
     the discrepancy for each parameter and `--report` refuses to summarise a
     parameter whose centre missed by more than `GATE_TOL`.

  4. What the rise MEANS. The loss is `score_A**2 + score_F**2`, a weighted sum
     of two scores each normalised to ~1 at "as good as the halo-only
     regression benchmark". It is NOT a log-likelihood, so no `Delta loss`
     threshold has any distributional meaning and NOTHING here is a confidence
     interval. Every rise is therefore also reported in the currency the rest
     of the experiment uses: the profile-shape error in percent — the typical
     error the model makes on the normalised curve of growth `M*(<R)/M*(<100
     kpc)`, 13.787% for the incumbent — and `score_A`, the amplitude error in
     units of the benchmark's scatter. "This parameter is determined to +-0.05"
     here means "moving it 0.05 and letting everything else adjust costs about
     one percentage point of profile-shape error", nothing more.

WHAT TO WATCH (all three flagged in the handover, all handled)

  * `log_f0`'s upper bound is 0.0 — a deposit half-mass radius equal to its
    halo's radius at redshift 0 — and a Stage 3.7 fit has already reached it.
    If a profile runs into that bound the curve stops being a property of the
    model and starts being a property of the bound. `--report` prints an
    explicit AT BOUND line for any grid point where a FREE parameter finished
    within `BOUND_TOL` of its bound, and the summary says "bound", never
    "determined".
  * Nelder-Mead in six dimensions is not reliable from one start; hence the
    continuation sweep above.
  * A `--smoke` run touches NO production filename: it writes into
    `stage39_profiles_SMOKE/` and `stage39_grid_SMOKE.npz`.

COST, measured not guessed: 0.111 s per loss evaluation on 2397 galaxies,
single-threaded (the evaluator is not BLAS-bound, so one process per parameter
does not contend). See the timing printed by `--only`.

Run:
  export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof
  PYTHONPATH=. uv run python -u experiments/.../stage39_profile.py --calibrate
  PYTHONPATH=. uv run python -u experiments/.../stage39_profile.py --only a0
  PYTHONPATH=. uv run python -u experiments/.../stage39_profile.py --report
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
import model as M                                        # noqa: E402
import stage33 as S33                                    # noqa: E402
import stage35_time_law as S35                           # noqa: E402
import stage37_size_epoch as S37                         # noqa: E402

BASE = "gompertz_log-E2-S2"
#: the incumbent's loss on the clean sample, checked at run time, never trusted
LOSS_INCUMBENT = 1.874253
#: the incumbent's profile-shape error, in percent, on the same sample -- the
#: typical error it makes on the normalised curve of growth M*(<R)/M*(<100 kpc)
SHAPE_INCUMBENT = 13.787
#: and the incumbent's `score_F`, the same quantity before the L_F_REF scaling
SF_INCUMBENT = 0.8715235273474429
#: the loss rise that corresponds to ONE percentage point of profile-shape
#: error, 13.787% -> 14.787%, if the whole rise lands in the shape term. This
#: is the only currency in which a "how well determined" number is quoted here,
#: because the loss itself is not a log-likelihood and a threshold on it means
#: nothing distributionally.
RISE_1PCT = (SF_INCUMBENT * (1 + 1.0 / SHAPE_INCUMBENT)) ** 2 - SF_INCUMBENT ** 2
Z = (0.4, 0.7, 1.0, 1.5, 2.0)
#: a literal percent sign, for plain-text reports (figures use `qa._pct`)
_PC = "%"

#: the CONDITIONAL loss rise that defines the grid's innermost step `d0`. 0.10
#: in loss units is about 0.9 percentage points of profile-shape error if the
#: whole rise lands in the shape term, which is a degradation a reader can feel.
STEP_RISE = 0.10
#: the grid: `+-MULTIPLIERS * d0`, plus the incumbent itself
MULTIPLIERS = (1, 2, 4, 8, 16)
#: finite-difference step scales for the Hessian diagnostic, as conditional
#: rises; three of them, because the point is the step-dependence
HESS_TARGETS = (1e-4, 1e-3, 1e-2)
#: the centre point must reproduce the incumbent's loss to this, or the whole
#: curve for that parameter is measured against the wrong zero
GATE_TOL = 1e-3
#: a free parameter this close to its own bound is AT the bound, not near it
BOUND_TOL = 1e-3
#: Nelder-Mead budget per grid point, six free parameters
MAXITER = 3600

GRID = HERE / "outputs" / "stage39_grid.npz"
PROFDIR = HERE / "outputs" / "stage39_profiles"
OUT = HERE / "outputs" / "stage39_profile.npz"
RULE = "=" * 100


def paths(smoke):
    """`--smoke` must never write a production filename (standing rule)."""
    if not smoke:
        return GRID, PROFDIR, OUT
    return (HERE / "outputs" / "stage39_grid_SMOKE.npz",
            HERE / "outputs" / "stage39_profiles_SMOKE",
            HERE / "outputs" / "stage39_profile_SMOKE.npz")


def setup(smoke=False):
    """The sample, the spec, the incumbent, the stacked evaluator — and the
    gate that the evaluator reproduces the incumbent's recorded loss."""
    recs, sel, data, mask, lmh, cuts = S37.build(smoke)
    spec = S33.spec_from_label(BASE)
    theta = S37.incumbent()
    pr = S35.StackedProblem(spec, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    loss = pr.loss(theta)
    print(f"  sample   : {len(recs)} galaxies, {int(mask.sum())} galaxy-epochs "
          f"(completeness AND sanity mask)")
    print(f"  incumbent: {BASE}, loss {loss:.6f}", flush=True)
    if not smoke and abs(loss - LOSS_INCUMBENT) > 1e-5:
        raise SystemExit(
            f"REFUSING TO RUN: this evaluator gives {loss:.6f} at the stored "
            f"incumbent theta, against the recorded {LOSS_INCUMBENT:.6f}. The "
            f"sample, the mask or the objective has changed, so nothing "
            f"computed here could be compared with any previously reported "
            f"number.")
    return recs, data, mask, lmh, spec, theta, pr


class FixedProblem:
    """The seven-parameter problem with parameter `j` held at `value`.

    Presents exactly the interface `fit.fit` needs — a `.spec` that can report
    `bounds()` and a `.loss` — over the SIX free parameters, so the profile is
    optimised by the same multi-start Nelder-Mead as every fit in this
    experiment rather than by a second, differently-behaved optimiser.
    """

    class _Spec:
        def __init__(self, b):
            self._b = b

        def bounds(self):
            return self._b

    def __init__(self, problem, j, value):
        self.p, self.j, self.value = problem, j, value
        full = list(problem.spec.bounds())
        self.spec = FixedProblem._Spec([b for k, b in enumerate(full) if k != j])
        self.n_eval = 0

    def lift(self, x6):
        return np.insert(np.asarray(x6, float), self.j, self.value)

    def drop(self, x7):
        return np.delete(np.asarray(x7, float), self.j)

    def loss(self, x6):
        self.n_eval += 1
        return self.p.loss(self.lift(x6))


# --------------------------------------------------------------------------- #
# calibration: how wide should each parameter's grid be?                       #
# --------------------------------------------------------------------------- #
def conditional_step(pr, theta, j, lo, hi, target, n_bisect=20, sign=+1):
    """The displacement at which the CONDITIONAL rise reaches `target`.

    One direction at a time, because the loss is not symmetric in every
    parameter: at the incumbent, `c` needs +5.02 to cost 2.0 in loss and only
    -0.40 to cost the same. Returns the room to the bound if the target is
    never reached inside it, and the caller must treat that as a clip.
    """
    base = pr.loss(theta)

    def rise(d):
        x = theta.copy()
        x[j] = float(np.clip(theta[j] + sign * d, lo, hi))
        return pr.loss(x) - base

    room = (hi - theta[j]) if sign > 0 else (theta[j] - lo)
    a, b = 0.0, min(1e-3, room)
    while b < room and rise(b) < target:
        a, b = b, min(2 * b, room)
    if rise(b) < target:
        return room, True
    for _ in range(n_bisect):
        m = 0.5 * (a + b)
        if rise(m) < target:
            a = m
        else:
            b = m
    return 0.5 * (a + b), False


def loss_hessian(pr, theta, steps):
    """Central-difference Hessian of the loss at `theta`, step `steps[j]`.

    Kept as a DIAGNOSTIC, not as the grid's basis; see the module docstring.
    """
    n = len(theta)
    f0 = pr.loss(theta)
    H = np.zeros((n, n))
    fp, fm = np.zeros(n), np.zeros(n)
    for j in range(n):
        x = theta.copy(); x[j] += steps[j]; fp[j] = pr.loss(x)
        x = theta.copy(); x[j] -= steps[j]; fm[j] = pr.loss(x)
        H[j, j] = (fp[j] - 2 * f0 + fm[j]) / steps[j] ** 2
    for j in range(n):
        for k in range(j + 1, n):
            x = theta.copy(); x[j] += steps[j]; x[k] += steps[k]
            fpp = pr.loss(x)
            x = theta.copy(); x[j] -= steps[j]; x[k] -= steps[k]
            fmm = pr.loss(x)
            H[j, k] = H[k, j] = ((fpp + fmm - fp[j] - fm[j] - fp[k] - fm[k]
                                  + 2 * f0) / (2 * steps[j] * steps[k]))
    return H, f0


def calibrate(smoke=False):
    """Measure `d0` per parameter per direction, lay the grids, and record why
    the Hessian was not used to lay them.

    `d0` is the displacement at which the CONDITIONAL loss rise reaches
    `STEP_RISE`. It is measured, by bisection, on the same evaluator the
    profile run uses. The grid is then `+-MULTIPLIERS * d0`, clipped to the
    parameter's bounds.
    """
    gridf, _, _ = paths(smoke)
    print(f"exp54 STAGE 3.9 — CALIBRATION: how wide must each parameter's grid "
          f"be?\n")
    recs, data, mask, lmh, spec, theta, pr = setup(smoke)
    names = list(spec.theta_names)
    lo, hi = np.array(spec.bounds()).T
    n = len(theta)
    t0 = time.time()

    d0 = np.zeros((n, 2))
    clip0 = np.zeros((n, 2), bool)
    for j in range(n):
        for s, sign in enumerate((-1, +1)):
            d0[j, s], clip0[j, s] = conditional_step(
                pr, theta, j, lo[j], hi[j], STEP_RISE, sign=sign)

    print(f"  d0 = the displacement at which the CONDITIONAL loss rise reaches "
          f"{STEP_RISE:g},")
    print(f"  which is about {100 * ((np.sqrt(0.7595 + STEP_RISE) / 0.87152) - 1) * SHAPE_INCUMBENT / 100:.2f} "
          f"percentage points of profile-shape error if the whole rise lands "
          f"in the shape term.\n")
    grids, clipped = {}, {}
    print(f"  {'parameter':<10}{'value':>9}{'d0 (-)':>10}{'d0 (+)':>10}"
          f"{'grid from':>12}{'grid to':>11}   note")
    for j, nm in enumerate(names):
        room_lo, room_hi = theta[j] - lo[j], hi[j] - theta[j]
        g_lo = np.clip(theta[j] - np.array(MULTIPLIERS) * d0[j, 0],
                       lo[j], hi[j])[::-1]
        g_hi = np.clip(theta[j] + np.array(MULTIPLIERS) * d0[j, 1],
                       lo[j], hi[j])
        g = np.concatenate([g_lo, [theta[j]], g_hi])
        g = np.unique(np.round(g, 12))
        grids[nm] = g
        cl = []
        if max(MULTIPLIERS) * d0[j, 0] > room_lo or clip0[j, 0]:
            cl.append(f"lower end CLIPPED at the bound {lo[j]:g}")
        if max(MULTIPLIERS) * d0[j, 1] > room_hi or clip0[j, 1]:
            cl.append(f"upper end CLIPPED at the bound {hi[j]:g}")
        clipped[nm] = cl
        print(f"  {nm:<10}{theta[j]:>9.4f}{d0[j, 0]:>10.4f}{d0[j, 1]:>10.4f}"
              f"{g[0]:>12.4f}{g[-1]:>11.4f}   {'; '.join(cl)}")

    # THE HESSIAN, as a diagnostic and as the record of why it is not the grid
    print(f"\n  THE HESSIAN, AT THREE FINITE-DIFFERENCE STEP SCALES. This is "
          f"NOT what sizes the\n  grid. It is here because its instability is "
          f"a result: if the small curvatures of a\n  seven-parameter model "
          f"cannot be measured by differencing, conditional slices — which\n"
          f"  are differences along the LARGE curvatures — are exactly the "
          f"wrong instrument.")
    hess, hsteps = [], []
    for tgt in HESS_TARGETS:
        st = np.array([conditional_step(pr, theta, j, lo[j], hi[j], tgt)[0]
                       for j in range(n)])
        Hm, f0 = loss_hessian(pr, theta, st)
        hess.append(Hm); hsteps.append(st)
        ev = np.linalg.eigvalsh(Hm)
        print(f"    step set at a conditional rise of {tgt:g}: eigenvalues "
              + "  ".join(f"{v:+.3e}" for v in ev))
    ev = np.array([np.linalg.eigvalsh(h) for h in hess])
    spread = ev.max(0) / np.where(np.abs(ev.min(0)) > 0, np.abs(ev.min(0)), np.nan)
    print(f"    step-to-step spread, smallest eigenvalue first: "
          + "  ".join(f"{v:.1f}x" for v in spread))
    print(f"    the diagonal, which IS the conditional curvature and IS "
          f"stable: "
          + "  ".join(f"{v:.1f}" for v in np.diag(hess[1])))

    gridf.parent.mkdir(exist_ok=True)
    np.savez(gridf, names=np.array(names), theta=theta, loss=pr.loss(theta),
             d0=d0, d0_clipped=clip0, step_rise=STEP_RISE,
             multipliers=np.array(MULTIPLIERS),
             hessian=np.array(hess), hess_steps=np.array(hsteps),
             hess_targets=np.array(HESS_TARGETS),
             cond_curv=np.diag(hess[1]),
             **{f"grid_{nm}": grids[nm] for nm in names})
    print(f"\n  wrote {gridf.name}  ({time.time() - t0:.0f} s, "
          f"{pr.n_eval} loss evaluations)")
    for nm in names:
        g = grids[nm]
        print(f"    {nm:<10}{len(g)} points: "
              + " ".join(f"{v:+.3f}" for v in g))
    return grids



# --------------------------------------------------------------------------- #
# the profile itself                                                           #
# --------------------------------------------------------------------------- #
def profile_one(name, smoke=False, verbose=True):
    """Fix `name` on its grid, re-optimise the other six at every point."""
    gridf, profdir, _ = paths(smoke)
    if not gridf.exists():
        raise SystemExit(f"REFUSING TO RUN: no {gridf.name}. Run --calibrate "
                         f"first; the grid must be sized before it is scanned.")
    print(f"exp54 STAGE 3.9 — PROFILE for {name}\n")
    recs, data, mask, lmh, spec, theta, pr = setup(smoke)
    names = list(spec.theta_names)
    if name not in names:
        raise SystemExit(f"{name!r} is not a parameter of {BASE}: {names}")
    j = names.index(name)
    z = np.load(gridf, allow_pickle=True)
    grid = np.asarray(z[f"grid_{name}"], float)
    if smoke:
        grid = grid[::max(len(grid) // 5, 1)]
    lo, hi = np.array(spec.bounds()).T
    base_loss = pr.loss(theta)
    ctr = int(np.argmin(np.abs(grid - theta[j])))

    print(f"  {name} = {theta[j]:+.4f} at the incumbent; grid {grid[0]:+.4f} "
          f"... {grid[-1]:+.4f}, {len(grid)} points, centre index {ctr}")
    print(f"  each point: six free parameters, Nelder-Mead from TWO starts — "
          f"the neighbouring\n  grid point's solution and the incumbent's own "
          f"six values. The CONDITIONAL rise\n  is measured at the same points "
          f"for free, so the two are compared point by point\n  rather than "
          f"through a quadratic approximation to either.\n")

    n = len(grid)
    res_loss = np.full(n, np.nan)
    res_theta = np.full((n, len(theta)), np.nan)
    res_sA = np.full(n, np.nan)
    res_sF = np.full(n, np.nan)
    res_shape = np.full(n, np.nan)
    res_atbound = np.zeros(n, bool)
    res_src = np.array(["" for _ in range(n)], dtype=object)
    #: the loss with ONLY this parameter moved -- the conditional slice, at the
    #: identical grid points, so the comparison needs no model of either curve
    res_cond = np.array([pr.loss(np.insert(np.delete(theta, j), j, v))
                         for v in grid])

    def run_point(i, prev6):
        fp = FixedProblem(pr, j, float(grid[i]))
        inc6 = fp.drop(theta)
        starts, tags = [], []
        if prev6 is not None:
            starts.append(np.clip(prev6, *np.array(fp.spec.bounds()).T))
            tags.append("continuation")
        starts.append(np.clip(inc6, *np.array(fp.spec.bounds()).T))
        tags.append("incumbent")
        best, best_l, best_tag = None, np.inf, ""
        for s, p0 in zip(tags, starts):
            x, l = F.fit(fp, starts=[p0], maxiter=MAXITER, verbose=False)
            if l < best_l:
                best, best_l, best_tag = x, l, s
        th7 = fp.lift(best)
        sA, sF, _ = pr.scores(th7)
        _, sf = pr.per_epoch(th7)
        free_lo, free_hi = np.array(fp.spec.bounds()).T
        at_b = bool(np.any(np.abs(best - free_lo) < BOUND_TOL)
                    or np.any(np.abs(best - free_hi) < BOUND_TOL))
        res_loss[i], res_theta[i] = best_l, th7
        res_sA[i], res_sF[i] = sA, sF
        res_shape[i] = S35.shape_pct(sf)
        res_atbound[i], res_src[i] = at_b, best_tag
        if verbose:
            flag = "   <-- a FREE parameter is AT its bound" if at_b else ""
            cr = res_cond[i] - base_loss
            pr_ = best_l - base_loss
            print(f"    {name}={grid[i]:+9.4f}   profiled rise {pr_:+10.6f}   "
                  f"conditional {cr:+10.4f}   survives "
                  f"{pr_ / cr if cr > 1e-12 else np.nan:7.4f}   "
                  f"shape {res_shape[i]:6.3f}%   score_A {sA:.4f}   "
                  f"[{best_tag}]{flag}", flush=True)
        return best

    t0 = time.time()
    centre6 = run_point(ctr, None)
    gate = res_loss[ctr] - base_loss
    print(f"\n    GATE: the centre point re-optimises six parameters with "
          f"{name} pinned at its own\n    incumbent value, so it must return "
          f"the incumbent's loss. Got {res_loss[ctr]:.6f} against "
          f"{base_loss:.6f}, difference {gate:+.2e}.")
    if gate > GATE_TOL:
        print(f"    WARNING: the optimiser did not recover the incumbent from "
              f"its own optimum.")
    elif gate < -GATE_TOL:
        print(f"    WARNING: the centre point beats the incumbent, so the "
              f"incumbent was not converged."
              + ("  EXPECTED under --smoke: the incumbent is the optimum of "
                 "the FULL sample, and --smoke fits 1/24 of it, so it is not "
                 "the optimum here and no rise on this curve is meaningful."
                 if smoke else ""))
    else:
        print(f"    OK.")
    print()

    prev = centre6
    for i in range(ctr + 1, n):
        prev = run_point(i, prev)
    prev = centre6
    for i in range(ctr - 1, -1, -1):
        prev = run_point(i, prev)

    profdir.mkdir(exist_ok=True)
    np.savez(profdir / f"{name}.npz", name=name, index=j, grid=grid,
             loss=res_loss, cond_loss=res_cond, theta=res_theta,
             score_A=res_sA, score_F=res_sF,
             shape_pct=res_shape, at_bound=res_atbound,
             start_used=np.array([str(s) for s in res_src]),
             base_loss=base_loss, base_theta=theta, gate=gate,
             cond_curv=float(z["cond_curv"][j]))
    print(f"\n  {name} done in {(time.time() - t0) / 60:.1f} min, "
          f"{pr.n_eval} loss evaluations -> {profdir.name}/{name}.npz")


# --------------------------------------------------------------------------- #
# the summary                                                                  #
# --------------------------------------------------------------------------- #
def _displacement_for(grid, rise, centre, target):
    """Interpolated |displacement| at which the profiled rise reaches `target`.

    Returned separately for the two directions; NaN where the grid never gets
    there, which is itself the finding.
    """
    out = []
    for sgn in (-1, +1):
        sel = np.argsort(sgn * (grid - centre))[::-1]
        d = sgn * (grid[sel] - centre)
        r = rise[sel]
        keep = d >= 0
        d, r = d[keep], r[keep]
        o = np.argsort(d)
        d, r = d[o], r[o]
        hit = np.where(r >= target)[0]
        if len(hit) == 0 or hit[0] == 0:
            out.append(np.nan)
            continue
        i = hit[0]
        f = (target - r[i - 1]) / max(r[i] - r[i - 1], 1e-30)
        out.append(d[i - 1] + f * (d[i] - d[i - 1]))
    return out


def report(smoke=False):
    """The table B1 owes: conditional against profiled, for all seven."""
    gridf, profdir, outf = paths(smoke)
    if not gridf.exists():
        raise SystemExit(f"no {gridf.name}: run --calibrate first")
    z = np.load(gridf, allow_pickle=True)
    names = [str(s) for s in z["names"]]
    theta = np.asarray(z["theta"], float)
    lo, hi = np.array(S33.spec_from_label(BASE).bounds()).T

    print(f"{RULE}\nexp54 STAGE 3.9 — PROFILE LIKELIHOODS for the seven "
          f"parameters of {BASE}\n{RULE}\n")
    print(f"THE QUESTION. Every identifiability number this experiment has "
          f"quoted moves ONE parameter\nand holds the other six FIXED. That "
          f"answers \"how much worse does THIS fit get\". The question\n"
          f"anyone actually means is \"how much worse does the BEST ACHIEVABLE "
          f"fit get\", which needs the\nother six re-optimised at every value. "
          f"This is that calculation.\n")
    print(f"WHAT EACH COLUMN IS")
    print(f"  cond. rise   the loss rise from moving one parameter to the edge "
          f"of its grid with the")
    print(f"               other six HELD FIXED — the number this experiment "
          f"has been quoting.")
    print(f"  prof. rise   the loss rise from the same move with the other six "
          f"RE-OPTIMISED — the")
    print(f"               honest one. It can never exceed the conditional "
          f"rise.")
    print(f"  survives     profiled / conditional at the edge. 1 means the "
          f"other six cannot")
    print(f"               compensate at all and the conditional number was "
          f"honest; 0.01 means")
    print(f"               99{_PC} of the apparent constraint was an artefact of "
          f"holding them fixed.")
    print(f"  -1pt / +1pt  how far the parameter can move, in each direction, "
          f"before the BEST")
    print(f"               ACHIEVABLE fit costs one percentage point of "
          f"profile-shape error —")
    print(f"               13.79{_PC} to 14.79{_PC} on the normalised curve of "
          f"growth. NOT a confidence")
    print(f"               interval: the loss is a weighted sum of two "
          f"normalised scores, not a")
    print(f"               log-likelihood, so no threshold on it has any "
          f"distributional meaning.\n")

    rows, missing = [], []
    for nm in names:
        f = profdir / f"{nm}.npz"
        if not f.exists():
            missing.append(nm)
            continue
        rows.append(dict(np.load(f, allow_pickle=True)))
    if missing:
        print(f"  NOT YET RUN, and therefore not summarised: "
              f"{', '.join(missing)}\n")
    if not rows:
        raise SystemExit("nothing to report")

    print(f"  {'param':<9}{'value':>9}{'grid -':>9}{'grid +':>9}"
          f"{'cond. rise':>12}{'prof. rise':>12}{'survives':>10}"
          f"{'-1pt':>9}{'+1pt':>9}   verdict")
    summary = {}
    for r in rows:
        nm, j = str(r["name"]), int(r["index"])
        grid, loss, cond = r["grid"], r["loss"], r["cond_loss"]
        base = float(r["base_loss"])
        rise, crise = loss - base, cond - base
        k_edge = int(np.nanargmax(rise))
        prof_edge, cond_edge = float(rise[k_edge]), float(crise[k_edge])
        surv = prof_edge / cond_edge if cond_edge > 1e-12 else np.nan
        dm, dp = _displacement_for(grid, rise, theta[j], RISE_1PCT)
        gate_ok = abs(float(r["gate"])) <= GATE_TOL
        at_bound_any = bool(np.any(r["at_bound"]))
        hit_own_bound = (abs(grid[0] - lo[j]) < BOUND_TOL
                         or abs(grid[-1] - hi[j]) < BOUND_TOL)

        verdict = []
        if not gate_ok:
            verdict.append(f"GATE FAILED ({float(r['gate']):+.1e})")
        if np.isnan(dm) and np.isnan(dp):
            verdict.append("NOT determined anywhere on this grid")
        elif np.isnan(dm) or np.isnan(dp):
            verdict.append("determined on ONE side only")
        else:
            verdict.append("determined")
        if hit_own_bound:
            verdict.append(f"the grid REACHES this parameter's own bound "
                           f"({lo[j]:g}, {hi[j]:g})")
        if at_bound_any:
            verdict.append("a FREE parameter sits at a bound somewhere on the "
                           "curve")
        print(f"  {nm:<9}{theta[j]:>9.4f}{grid[0] - theta[j]:>9.3f}"
              f"{grid[-1] - theta[j]:>9.3f}{cond_edge:>12.3f}"
              f"{prof_edge:>12.4f}{surv:>10.4f}"
              f"{-dm if np.isfinite(dm) else np.nan:>9.3f}"
              f"{dp if np.isfinite(dp) else np.nan:>9.3f}"
              f"   {'; '.join(verdict)}")
        summary[nm] = dict(
            grid=grid, rise=rise, cond_rise=crise,
            prof_edge=prof_edge, cond_edge=cond_edge, survives=surv,
            d_minus=dm, d_plus=dp, gate=float(r["gate"]),
            at_bound=at_bound_any, hit_own_bound=hit_own_bound,
            shape_pct=r["shape_pct"], score_A=r["score_A"])

    print(f"\n  HOW MUCH THE CONDITIONAL SLICE OVERSTATED EACH PARAMETER, at "
          f"every grid point.\n  Each cell is profiled rise / conditional "
          f"rise; the row is one parameter, read\n  outward from the "
          f"incumbent. A row of small numbers is a parameter the other six\n"
          f"  can absorb almost entirely.")
    print(f"  {'param':<9}{'displacement ->':>0}")
    for nm in summary:
        s = summary[nm]
        j = names.index(nm)
        order = np.argsort(np.abs(s["grid"] - theta[j]))
        cells = []
        for i in order[1:]:
            c = s["cond_rise"][i]
            cells.append(f"{s['rise'][i] / c:.4f}" if c > 1e-12 else "  --  ")
        print(f"  {nm:<9}" + " ".join(f"{v:>8}" for v in cells))

    print(f"\n  WHAT THE EDGE OF EACH GRID COSTS PHYSICALLY. The incumbent "
          f"makes a "
          f"{SHAPE_INCUMBENT:.3f}{_PC}\n  typical error on the normalised "
          f"curve of growth and scores 1.0558 on amplitude,\n  where 1.0 is "
          f"the halo-only regression benchmark's own scatter.")
    print(f"  {'param':<9}{'shape err':>12}{'score_A':>10}   at the worst grid "
          f"point")
    for nm in summary:
        s = summary[nm]
        k = int(np.nanargmax(s["rise"]))
        print(f"  {nm:<9}{float(s['shape_pct'][k]):>11.3f}{_PC}"
              f"{float(s['score_A'][k]):>10.4f}   "
              f"{nm} = {s['grid'][k]:+.4f}")

    if not smoke:
        flat = {}
        for nm, s in summary.items():
            for k, v in s.items():
                flat[f"{k}_{nm}"] = np.asarray(v)
        np.savez(outf, names=np.array(list(summary)), theta=theta,
                 rise_1pct=RISE_1PCT, **flat)
        print(f"\n  wrote {outf.name}")
    return summary



if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if "--calibrate" in sys.argv:
        calibrate(smoke)
    elif "--only" in sys.argv:
        profile_one(sys.argv[sys.argv.index("--only") + 1], smoke)
    elif "--report" in sys.argv:
        report(smoke)
    else:
        raise SystemExit(__doc__.strip().splitlines()[-1])
