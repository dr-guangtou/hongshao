"""exp49 stage 2 — the widened-domain optimum and the PROFILED `g` curve.

Stage 1 showed the six base parameters are near-collinear (pairwise |rho|
0.834-0.995) with an effective rank of 7 of 12, and exp49's rail test showed `g`
pinned at its bound of 4.0 by the DATA rather than the optimizer. What is still
missing is the number itself: where does `g` sit when the box is not in the way,
and how sharply is it determined?

Two products, per objective:

  fit      all 12 parameters, L-BFGS-B with TRUE bounds and no box penalty,
           from five starts, on the widened domain g<=6 and log_rc<=3.5.
           BOTH must be widened -- widening only `g` moves the fit to the
           other corner, which is what three of exp48's six families did.
  profile  fix `g` on a grid and RE-OPTIMIZE THE OTHER 11 at each point.
           A slice with the others frozen is not a profile; reading one as a
           profile is the error this stage exists to correct.

Profiling is invariant to how the nuisance parameters are coordinatized, so
this curve does not need the R50/time-pivot reparameterization first -- and it
must come first anyway, because the pivot is derived from the curvature at the
widened INTERIOR solution, which does not exist until this runs.

**Objective values from the two objectives are NOT comparable.** Compare the
optimum `g`, the curve width, physical radii, and prediction changes.

Resumable: every cell is saved the moment it finishes, and `--max-hours` stops
cleanly before a wall-clock budget rather than being killed mid-fit.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
experiments/exp49_g_rail/widened.py {fit|profile|report} \
[--objective production|density_log] [--max-hours H] [--dev]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))

import stage2_multiepoch as s2                      # noqa: E402
from hongshao.fitting import minimize_loss          # noqa: E402
from hongshao.objective import Objective, reduce_galaxies   # noqa: E402

OUTDIR = HERE / "outputs"
KS = [0, 1, 2, 3]
NAMES = ["log_rc", "g", "q", "mu", "sig", "gamma",
         "rc:logMh", "rc:c200c", "rc:fz2", "sig:logMh", "sig:c200c", "sig:fz2"]
GI = 1                                              # index of `g`
FAIL_LOSS = 4.0                    # the fit-steering sentinel, applied HERE

# WIDENED domain. g and log_rc both widened (see module docstring); the other
# four base parameters keep their existing physical bounds; the six
# conditioning slopes carry no penalty in 1ch-mof and stay unbounded.
BOX_LO = [1.0, 0.0, 0.0, 0.0, 0.05, 1.05] + [None] * 6
BOX_HI = [3.5, 6.0, 3.0, 3.0, 2.00, 6.00] + [None] * 6
BOUNDS = list(zip(BOX_LO, BOX_HI))

OBJECTIVES = dict(production=Objective(),
                  density_log=Objective(quantity="density", residual="log"))

RIDGE_SLOPE, RIDGE_INTERCEPT = 1.4822, -0.0843      # exp49's measured valley
PIVOT_LOG = -0.6747                                 # log10(t_p/t_obs) from it
GRID_COARSE = np.round(np.arange(3.00, 5.51, 0.25), 4)
SEED = 20260816


def _clip(p):
    lo = np.array([-np.inf if x is None else x for x in BOX_LO])
    hi = np.array([np.inf if x is None else x for x in BOX_HI])
    return np.clip(np.asarray(p, float), lo, hi)


class Problem:
    """Model + data + objective, with the sentinel applied where it belongs."""

    def __init__(self, obj, gals, R):
        self.obj, self.gals, self.R = obj, gals, R
        self.data = np.stack([np.array([g["data"][k] for k in KS], float)
                              for g in gals])
        self.n_eval = 0

    def cogs(self, theta):
        out = np.full((len(self.gals), len(KS), len(self.R)), np.nan)
        for i, g in enumerate(self.gals):
            c = s2.model_cogs(theta, g, KS, "1ch-mof")
            if c is not None:
                out[i] = np.asarray(c)
        return out

    def loss(self, theta):
        """Population loss. Failed galaxies get the SENTINEL, never dropped --
        reduce_galaxies drops non-finite entries, so relying on the drop makes
        a fit that starts failing look like it improved."""
        self.n_eval += 1
        v = self.obj.per_galaxy(self.cogs(theta), self.data, self.R)
        return reduce_galaxies(np.where(np.isfinite(v), v, FAIL_LOSS),
                               self.obj.galaxy)


def starts(theta_bounded, theta_e40):
    """Five starts, per the plan. The pivot-preserving and ridge-following
    ones matter: a naive displaced start just slides back down the valley."""
    rng = np.random.default_rng(SEED)
    s = {}
    s["bounded"] = _clip(theta_bounded)
    s["exp40_widened"] = _clip(theta_e40)
    # g = 2 with log_rc placed on the measured ridge
    lo = theta_bounded.copy()
    lo[GI] = 2.0
    lo[0] = (2.0 - RIDGE_INTERCEPT) / RIDGE_SLOPE
    s["g2_on_ridge"] = _clip(lo)
    # g = 5 holding the deposit radius at the pivot epoch fixed:
    # log rc(t_p) = log_rc + g*log10(t_p/t_obs) is preserved
    hi = theta_bounded.copy()
    hi[GI] = 5.0
    hi[0] = theta_bounded[0] - PIVOT_LOG * (5.0 - theta_bounded[GI])
    s["g5_pivot_fixed"] = _clip(hi)
    rp = theta_bounded.copy()
    rp[:6] *= 1.0 + rng.normal(0, 0.12, 6)
    rp[6:] = rng.normal(0, 0.03, 6)
    s[f"random_seed{SEED}"] = _clip(rp)
    return s


def _store(name, dev):
    return OUTDIR / f"{name}{'_dev' if dev else ''}.npz"


def _save(store, done, key, theta, loss, res, extra=None):
    done[f"theta::{key}"] = theta
    done[f"loss::{key}"] = np.array([float(loss)])
    done[f"meta::{key}"] = np.array([json.dumps(dict(
        nfev=int(res.nfev), success=bool(res.success),
        status=str(res.message)[:120], **(extra or {})))])
    OUTDIR.mkdir(exist_ok=True)
    np.savez(store, **done)                      # atomic-enough: after EVERY cell


def _bound_audit(theta):
    """Normalized distance to bound for every bounded parameter. Reports the
    whole vector, not just g -- exp41 shows sig at a box edge for 4% of
    galaxies and exp38 flags mu railing at 2/5 epochs."""
    hits = []
    for j, (lo, hi) in enumerate(BOUNDS):
        if lo is None or hi is None:
            continue
        d = min(theta[j] - lo, hi - theta[j]) / (hi - lo)
        if d < 0.02:
            hits.append(f"{NAMES[j]}={theta[j]:.4f} ({d*100:.2f}% from a bound)")
    return hits


def _setup(dev, objective):
    rows = (np.load(ROOT / "experiments/exp32_full_population/outputs/"
                    "population.npz")["dev100"] if dev else None)
    s2._w_init(rows)
    gals, R = s2._W["gals"], s2._W["e"].R
    prob = Problem(OBJECTIVES[objective], gals, R)
    return prob, gals, R


def cmd_fit(objective="production", dev=False, max_hours=8.0):
    prob, gals, R = _setup(dev, objective)
    th_b = np.asarray(np.load(OUTDIR / "railtest.npz",
                              allow_pickle=True)["theta::bounded::adopted"], float)
    th_40 = np.asarray(np.load(ROOT / "experiments/exp40_epoch_objective/"
                               "outputs/latestress.npz")["theta"], float)
    store = _store(f"widened_fit_{objective}", dev)
    done = dict(np.load(store, allow_pickle=True)) if store.exists() else {}
    print(f"exp49 stage 2 FIT — objective={objective} ({prob.obj.name})\n"
          f"  n={len(gals)}, widened domain g<=6, log_rc<=3.5, "
          f"budget {max_hours:.1f} h", flush=True)
    t0 = time.time()
    for name, p0 in starts(th_b, th_40).items():
        if f"theta::{name}" in done:
            print(f"  [{name}] cached", flush=True)
            continue
        if (time.time() - t0) / 3600 > max_hours:
            print(f"  BUDGET REACHED — stopping before [{name}]", flush=True)
            break
        t1 = time.time()
        prob.n_eval = 0
        r = minimize_loss(prob.loss, p0, method="lbfgsb", bounds=BOUNDS,
                          max_evals=4000)
        _save(store, done, name, r.x, r.fun, r,
              dict(minutes=round((time.time() - t1) / 60, 2), start=p0.tolist()))
        print(f"  [{name:<16}] loss {r.fun:.8f}  g {p0[GI]:.3f} -> "
              f"{r.x[GI]:.5f}  log_rc {p0[0]:.3f} -> {r.x[0]:.5f}  "
              f"({r.nfev} evals, {(time.time()-t1)/60:.1f} min, "
              f"conv={r.success})", flush=True)
    _fit_summary(done, objective)


def _fit_summary(done, objective):
    keys = sorted(k.split("::", 1)[1] for k in done if k.startswith("theta::"))
    if not keys:
        return
    losses = np.array([done[f"loss::{k}"][0] for k in keys])
    best = keys[int(np.argmin(losses))]
    spread = losses.max() - losses.min()
    print(f"\n  === {objective}: {len(keys)}/5 starts done ===")
    for k in keys:
        th = done[f"theta::{k}"]
        print(f"    {k:<18}{done[f'loss::{k}'][0]:14.8f}  g={th[GI]:.5f}  "
              f"log_rc={th[0]:.5f}")
    print(f"    start-to-start spread {spread:.2e} "
          f"({'OK, < 1e-6' if spread < 1e-6 else 'EXCEEDS 1e-6 — DIAGNOSE'})")
    th = done[f"theta::{best}"]
    hits = _bound_audit(th)
    print(f"    best = {best}; bounds audit: "
          f"{'; '.join(hits) if hits else 'no parameter within 2% of a bound'}")


def cmd_profile(objective="production", dev=False, max_hours=8.0,
                grid=None):
    """Fix `g`, re-optimize the other 11. Warm-started outward from the
    optimum so each point starts from its nearest solved neighbour."""
    prob, gals, R = _setup(dev, objective)
    fit_store = _store(f"widened_fit_{objective}", dev)
    if not fit_store.exists():
        raise SystemExit("run `fit` first — the profile warm-starts from it")
    fit = dict(np.load(fit_store, allow_pickle=True))
    keys = [k.split("::", 1)[1] for k in fit if k.startswith("theta::")]
    best = keys[int(np.argmin([fit[f"loss::{k}"][0] for k in keys]))]
    th_opt = np.asarray(fit[f"theta::{best}"], float)
    grid = GRID_COARSE if grid is None else np.asarray(grid, float)
    order = sorted(grid, key=lambda x: abs(x - th_opt[GI]))

    store = _store(f"profile_{objective}", dev)
    done = dict(np.load(store, allow_pickle=True)) if store.exists() else {}
    print(f"exp49 stage 2 PROFILE — objective={objective}\n"
          f"  optimum g={th_opt[GI]:.5f} (start '{best}'), {len(grid)} grid "
          f"points, budget {max_hours:.1f} h", flush=True)
    t0 = time.time()
    for gv in order:
        key = f"g={gv:.4f}"
        if f"theta::{key}" in done:
            print(f"  [{key}] cached", flush=True)
            continue
        if (time.time() - t0) / 3600 > max_hours:
            print(f"  BUDGET REACHED — stopping before [{key}]", flush=True)
            break
        warm = _nearest_warm(done, gv, th_opt)
        t1 = time.time()
        r = _fit_fixed_g(prob, warm, gv)
        theta = np.insert(r.x, GI, gv)
        _save(store, done, key, theta, r.fun, r,
              dict(minutes=round((time.time() - t1) / 60, 2), g_fixed=float(gv)))
        print(f"  [{key:<11}] loss {r.fun:.8f}  log_rc {theta[0]:.5f}  "
              f"({r.nfev} evals, {(time.time()-t1)/60:.1f} min, "
              f"conv={r.success})", flush=True)
    _profile_summary(done, objective)


def _nearest_warm(done, gv, th_opt):
    """Warm start from the nearest already-solved g, else the optimum."""
    solved = [(abs(float(k.split("=")[1]) - gv), k)
              for k in [x.split("::", 1)[1] for x in done
                        if x.startswith("theta::")]]
    if not solved:
        return th_opt
    return np.asarray(done[f"theta::{min(solved)[1]}"], float)


def _fit_fixed_g(prob, warm, gv):
    """Optimize the OTHER 11 parameters with `g` held at gv.

    `g` is removed from the vector rather than pinned with equal bounds, so
    there is no ambiguity about whether the optimizer treats it as free.
    """
    p0 = np.delete(_clip(np.insert(np.delete(warm, GI), GI, gv)), GI)
    bounds = [b for j, b in enumerate(BOUNDS) if j != GI]

    def loss11(p):
        return prob.loss(np.insert(p, GI, gv))

    return minimize_loss(loss11, p0, method="lbfgsb", bounds=bounds,
                         max_evals=4000)


def _profile_summary(done, objective):
    keys = [k.split("::", 1)[1] for k in done if k.startswith("theta::")]
    if not keys:
        return
    gs = np.array([float(k.split("=")[1]) for k in keys])
    ls = np.array([done[f"loss::{k}"][0] for k in keys])
    o = np.argsort(gs)
    gs, ls = gs[o], ls[o]
    print(f"\n  === {objective}: profiled curve, {len(gs)} points ===")
    print(f"    {'g':>8}{'loss':>16}{'delta vs min':>16}{'log_rc':>10}")
    for gv, lv in zip(gs, ls):
        th = done[f"theta::g={gv:.4f}"]
        print(f"    {gv:8.2f}{lv:16.8f}{lv - ls.min():16.3e}{th[0]:10.5f}")
    print(f"    minimum on the grid: g = {gs[int(np.argmin(ls))]:.2f}")
    if np.argmin(ls) in (0, len(ls) - 1):
        print("    WARNING: minimum at a grid endpoint — EXTEND the grid")


def cmd_report(objective="production", dev=False):
    """Derived diagnostics from the stored cells. No fitting, so it is cheap
    and can be re-run as cells accumulate."""
    prob, gals, R = _setup(dev, objective)
    store = _store(f"profile_{objective}", dev)
    if not store.exists():
        raise SystemExit("no profile store yet")
    done = dict(np.load(store, allow_pickle=True))
    keys = sorted((float(k.split("=")[1]), k.split("::", 1)[1])
                  for k in [x for x in done if x.startswith("theta::")]
                  for k in [k])
    from hongshao import qa
    truth = prob.data
    r50 = qa._safe_rhalf(truth[:, 0], R, 0.5)
    compact = r50 < 5.0
    i5 = int(np.searchsorted(R, 5.0))
    i10 = int(np.searchsorted(R, 10.0))
    print(f"exp49 stage 2 REPORT — {objective}  (n={len(gals)}, "
          f"{compact.sum()} compact at z=0.4)")
    print(f"  {'g':>7}{'loss':>15}{'M5 all':>10}{'M5 cmp':>10}{'M5 ext':>10}"
          f"{'M10 all':>10}{'bounds':>28}")
    for gv, key in keys:
        th = np.asarray(done[f"theta::{key}"], float)
        cogs = prob.cogs(th)
        m5 = np.nanmedian(cogs[:, 0, i5] / truth[:, 0, i5] - 1.0)
        m5c = np.nanmedian(cogs[compact, 0, i5] / truth[compact, 0, i5] - 1.0)
        m5e = np.nanmedian(cogs[~compact, 0, i5] / truth[~compact, 0, i5] - 1.0)
        m10 = np.nanmedian(cogs[:, 0, i10] / truth[:, 0, i10] - 1.0)
        hits = _bound_audit(th)
        print(f"  {gv:7.2f}{done[f'loss::{key}'][0]:15.8f}"
              f"{100*m5:9.1f}%{100*m5c:9.1f}%{100*m5e:9.1f}%{100*m10:9.1f}%"
              f"  {'; '.join(hits)[:26] if hits else '-':>26}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    dev = "--dev" in sys.argv
    obj = "production"
    hours = 8.0
    if "--objective" in sys.argv:
        obj = sys.argv[sys.argv.index("--objective") + 1]
    if "--max-hours" in sys.argv:
        hours = float(sys.argv[sys.argv.index("--max-hours") + 1])
    if obj not in OBJECTIVES:
        raise SystemExit(f"--objective must be one of {list(OBJECTIVES)}")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=ROOT).stdout.strip()
    print(f"# git {sha[:10]}  objective={obj}  dev={dev}", flush=True)
    if cmd == "fit":
        cmd_fit(obj, dev, hours)
    elif cmd == "profile":
        cmd_profile(obj, dev, hours)
    elif cmd == "report":
        cmd_report(obj, dev)
    else:
        raise SystemExit(__doc__)
