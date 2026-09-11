"""exp80 Stage 1 — the fit of the changed size law under the STANDARD objective

    L = sum over epochs of  A^2 + F^2 + S^2 + B^2        (exp63's, adopted references)

No size term (exp78: a size term cannot buy this; the radius term is scored
AFTER the fit, in the judge). Everything else is the adopted baseline's:
exp63's 12-parameter two-channel model, the measured history input, the
adopted references, the fitting sample, exp63's bounds, the same engine and
quadrature. The size law is `size_law.predict_law` with the knobs named on
the command line appended to the twelve parameters (`--knobs q_e` is the
default: the expansion exponent chosen by Stage 0 C; `--knobs q_e,g_e` adds
the halo-mass exponent). Every knob nests at zero, where the model IS the
baseline.

PARAMETER COUNT: 12 + the number of knobs (13 with q_e alone). An observer
could vary all of them.

Starts, each its own process (`--starts k:k`), then `--merge`:
  tuned     the baseline with the size-law constants (log_f_e, b_e, knobs) at
            Stage 0 C's radius-term tune (`outputs/stage0_cand_*.npz`)
  baseline  the baseline with the knobs at 0 (the nesting point: does the
            loss have a gradient in the new parameter at the old optimum?)
  far       the baseline with the knobs at the far end of their plausible
            range (q_e = 0.5, g_e = -1/3): a new parameter is started from
            both ends (lesson, exp76)
  basin     the 14.63 basin (gate-rejected; the loss's favourite) with the
            knobs at the tuned values
  nested    the nested incumbent, clipped into the box, knobs at the tuned values
  --continue NAME   a continuation of a start that stopped at the cap

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=. \\
        nohup uv run python -u experiments/exp80_deposit_size_law/stage1_fit.py --starts k:k [--knobs q_e] \\
        > experiments/exp80_deposit_size_law/outputs/stage1_fit_s{k}.log 2>&1 &
    (caffeinate -i -w <pid> on each), then `--merge`.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp63_analytic_growth",
          ROOT / "experiments/exp73_size_relative",
          ROOT / "experiments/exp74_c19_history_leak", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
import size_law as SL                                    # noqa: E402
from hongshao.fitting import minimize_loss               # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RB = _by_path("exp74_rebaseline", ROOT / "experiments/exp74_c19_history_leak/rebaseline.py")   # exp54 shadows it
S0C = _by_path("exp80_stage0_candidates", HERE / "stage0_candidates.py")

RULE = "=" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR = HERE / "outputs"
E74 = ROOT / "experiments/exp74_c19_history_leak/outputs"
DEFAULT_KNOBS = ("q_e",)
FAR_VALUES = {"q_e": 0.5, "g_e": -1.0 / 3.0, "b_e2": 1.0, "s_floor_kpc": 5.0}
#: which Stage 0 C candidate file holds the tuned constants for a knob set
TUNED_SOURCE = {("q_e",): "a_expand_free", ("q_e", "g_e"): "a_expand_free_g", ("g_e",): "b_fixed_kpc"}
START_ORDER = ["tuned", "baseline", "far", "basin", "nested"]
#: the merged file names, shared with the judge (lesson: name the file once)
STAGE_FILES = {"stage1": "stage1_fit"}


def knob_tag(knobs):
    return "" if tuple(knobs) == DEFAULT_KNOBS else ("_noknob" if not knobs else "_" + "-".join(knobs))


class LawProblem:
    """exp63's joint problem with the size-law knobs appended to theta: one
    `predict_law` call on the standard grid per evaluation, scored by each
    epoch's `Problem2.score_model` (the adopted references live there).
    `law_base`: fixed law settings merged under the knobs (e.g. the smooth
    exponential arrival for a delay spec)."""

    def __init__(self, pr, spec2, knobs, law_base=None):
        self.pr, self.spec2, self.knobs = pr, spec2, tuple(knobs)
        self.law_base = dict(law_base or {})
        self.names = tuple(spec2.theta_names) + self.knobs
        self.n_theta = len(self.names)
        self.epochs = list(pr.epochs)
        self.n_eval = 0

    def split(self, theta):
        theta = np.asarray(theta, float)
        assert theta.shape == (self.n_theta,), (theta.shape, self.n_theta)
        th12 = theta[:self.spec2.n_theta]
        law = SL.with_law(**self.law_base, **{k: float(v) for k, v in zip(self.knobs, theta[self.spec2.n_theta:])})
        return th12, law

    def predict(self, theta, R, nodes=M2.FIT_NODES, curves=None):
        th12, law = self.split(theta)
        return SL.predict_law(self.spec2, th12, law, self.pr.curves if curves is None else curves, R,
                              epochs=tuple(self.epochs), nodes=nodes)

    def per_epoch(self, theta, nodes=M2.FIT_NODES):
        m = self.predict(theta, self.pr.R, nodes=nodes)
        return {k: self.pr.problems[k].score_model(m[self.pr.index[k], j]) for j, k in enumerate(self.epochs)}

    def loss(self, theta):
        self.n_eval += 1
        total = 0.0
        for k, sc in self.per_epoch(theta).items():
            pr = self.pr.problems[k]
            if len(sc) < 5:
                return F.FAIL * len(self.epochs)
            a, f, s, n_bad, b = sc
            if not np.all(np.isfinite([a, f, s, b])):
                return F.FAIL * len(self.epochs)
            total += a * a + f * f + s * s + b * b + F.FAIL * n_bad / len(pr.curves)
        # a model worse than the unbuildable-model penalty IS a failed model:
        # without this cap L-BFGS-B's first line-search trial from a start with
        # a sizeable gradient lands at the box corner (a0 = +2, an amplitude
        # 10^10 too large, loss 10^24), its cubic interpolation collapses the
        # step to zero and it aborts after three gradients with the loss
        # unchanged (seen on the 'tuned' start, 2026-09-10). Nothing inside a
        # basin is affected: every basin sits far below the penalty.
        return min(total, F.FAIL * len(self.epochs))

    def table(self, theta, nodes=M2.FULL_NODES):
        per = self.per_epoch(theta, nodes=nodes)
        return np.array([[per[k][0], per[k][1], per[k][2], per[k][4]] if len(per[k]) >= 5 else [np.nan] * 4
                         for k in self.epochs])

    def report(self, theta, label):
        tab = self.table(theta)
        print(f"  {label}: per-epoch (A, F, S, B) and loss")
        tot = 0.0
        for j, k in enumerate(self.epochs):
            lk = float((tab[j] ** 2).sum()); tot += lk
            print(f"    z={ANCHOR_Z[k]}: " + " ".join(f"{v:.3f}" for v in tab[j]) + f" -> {lk:.4f}")
        print(f"    total {tot:.4f}")
        return tot, tab

    def describe(self, theta):
        th12, law = self.split(theta)
        p = self.spec2.unpack(th12)
        return (", ".join(f"{n} {p[n]:+.3f}" for n in ("log_f_c", "b_c", "log_f_e", "b_e", "m_half", "d_split", "n_c", "c_e"))
                + (f", tau_d {p['tau_d']:+.3f}" if "tau_d" in p else "") + "; law: " + SL.describe(law))


def railed(names, theta, bounds, tol=1e-3):
    lo, hi = np.array(bounds).T
    return [n for n, v, a, b in zip(names, theta, lo, hi)
            if b - a > tol and (v - a < tol * max(abs(a), 1) or b - v < tol * max(abs(b), 1))]


def build(smoke=False, knobs=DEFAULT_KNOBS, delay=False, delay_form="step"):
    """`delay=True`: exp63 Stage 2b's deposition delay tau_d (Hubble times at
    accretion; the extended channel's deposits arrive later, mass in transit
    is not yet deposited) as a thirteenth model parameter, appended after
    the twelve and before the size-law knobs (exp81, 2026-09-12)."""
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr = RB.build(smoke)
    if delay:
        spec2 = M2.Spec2(theta_names=M2.THETA_NAMES_DELAY, extended_family=spec2.extended_family,
                         compact_in_kpc=spec2.compact_in_kpc)
    lp = LawProblem(pr, spec2, knobs, law_base=(dict(smooth_delay=True) if (delay and delay_form == "exp") else None))
    th_nested = M2.with_levers_theta(M2.nested_theta(th_inc, delay=spec2.delay, growth=spec2.growth_split), spec2)
    bounds = [tuple(b) for b in np.asarray(fz["bounds"], float).tolist()] + ([M2.BOUNDS["tau_d"]] if delay else []) \
        + [S0C.BOUNDS[k] for k in knobs]
    return recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, th_nested, pr, lp, bounds


def tuned_constants(knobs, smoke):
    """(log_f_e, b_e, {knob: value}) from Stage 0 C's tune for this knob set."""
    src = TUNED_SOURCE.get(tuple(knobs), "a_expand_free")
    p = OUTDIR / f"stage0_cand_{src}{'_smoke' if smoke else ''}.npz"
    p = p if p.exists() else OUTDIR / f"stage0_cand_{src}.npz"
    d = np.load(p, allow_pickle=True)
    vals = dict(zip([str(n) for n in d["free"]], np.asarray(d["x_tuned"], float)))
    return vals, p.name


def main(smoke=False, starts_sel=None, merge=False, cont=None, knobs=DEFAULT_KNOBS, fix=None, delay=None, delay_form="step",
         start_from=None):
    knobs = tuple(knobs)
    fix = dict(fix or {})
    tag = knob_tag(knobs) + (f"_delay{delay:g}" if delay is not None else "") + ("_exp" if delay_form == "exp" else "") \
        + ("".join(f"_fix-{k}{v:g}" for k, v in fix.items())) + ("_smoke" if smoke else "")
    if merge:
        return merge_starts(tag)
    print(f"{RULE}\nexp80 STAGE 1 — the fit of the size law with knobs {list(knobs)} under A^2+F^2+S^2+B^2 (no size term)"
          f"{' (SMOKE)' if smoke else ''}\n{RULE}\n")
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, th_nested, pr, lp, bounds = build(smoke, knobs, delay is not None, delay_form)
    for k, v in fix.items():
        bounds[lp.names.index(k)] = (float(v), float(v))
    n_par = lp.n_theta - len(fix)
    print(f"  PARAMETERS: {n_par} fitted (12 + {list(knobs)}" + (f", with {fix} FIXED" if fix else "") + f"); "
          f"an observer could vary all {lp.n_theta}; knob bounds " + ", ".join(f"{k} {S0C.BOUNDS[k]}" for k in knobs))
    th_nested13 = np.r_[th_nested, np.zeros(len(knobs))]
    tab = lp.table(th_nested13)
    assert np.allclose(tab[:, 2:], 1.0, atol=2e-2), tab
    print(f"  GATE  the nested incumbent (knobs at 0) scores 1.000 in S, B (max dev {np.max(np.abs(tab[:, 2:] - 1)):.1e})  OK")
    l_null = lp.loss(th_nested13)
    lp.report(th_nested13, "the null (nested incumbent, measured curves)")
    smoke_tag = "_smoke" if smoke else ""

    def theta_of(path):
        p = Path(str(path).replace(".npz", f"{smoke_tag}.npz"))
        p = p if p.exists() else Path(path)
        return np.asarray(np.load(p, allow_pickle=True)["theta_best"], float)

    th_base = theta_of(E74 / "stage1_refit_measured.npz")
    th_basin = theta_of(E74 / "rebaseline_exp63.npz")
    if delay is not None:
        # the twelve plus tau_d at the requested start value; the nested incumbent carries tau_d = 0 already
        th_base, th_basin = np.r_[th_base, delay], np.r_[th_basin, delay]
        print(f"  DELAY: tau_d fitted (bounds {M2.BOUNDS['tau_d']}), started at {delay:g} Hubble times; arrival form '{delay_form}'")
        if start_from is not None:
            # start the twelve (+ tau_d) from a named fit file's theta (its knobs dropped)
            fs = np.load(OUTDIR / start_from, allow_pickle=True)
            th_src = np.asarray(fs["theta"], float)
            th_base = np.r_[th_src[:12], delay]
            print(f"  the 'baseline' start's twelve taken from {start_from} (loss there {float(fs['loss']):.4f})")
    vals, src = tuned_constants(knobs, smoke)
    print(f"  Stage 0 C tuned constants from {src}: " + ", ".join(f"{k} {v:+.3f}" for k, v in vals.items()))

    def with_law(th12, knob_values, constants=True):
        th = np.asarray(th12, float).copy()
        if constants:
            for n in ("log_f_e", "b_e"):
                if n in vals:
                    th[spec2.index(n)] = vals[n]
        return np.r_[th, [knob_values[k] for k in knobs]]

    tuned_knobs = {k: fix.get(k, vals[k]) for k in knobs}
    zero = {k: fix.get(k, 0.0) for k in knobs}
    far = {k: fix.get(k, FAR_VALUES[k]) for k in knobs}
    starts = [("tuned", with_law(th_base, tuned_knobs)),
              ("baseline", with_law(th_base, zero, constants=False)),
              ("far", with_law(th_base, far, constants=False)),
              ("basin", with_law(th_basin, tuned_knobs, constants=False)),
              ("nested", with_law(M2.clip_to_bounds(spec2, th_nested), tuned_knobs, constants=False))]
    lp.report(starts[1][1], "THE BASELINE MEAN (knobs at 0), scored here")
    lp.report(starts[0][1], "the baseline with Stage 0 C's tuned law (the 'tuned' start), scored here")
    if cont is not None:
        src_f = OUTDIR / f"{STAGE_FILES['stage1']}{tag}_start_{cont}.npz"
        fb = np.load(src_f, allow_pickle=True)
        starts = [(f"cont_{cont}", np.asarray(fb["theta"], float))]
        print(f"  continuation of '{cont}' ({int(fb['n_eval'])} evals, loss {float(fb['loss']):.4f})")
    else:
        starts = [s for n in START_ORDER for s in starts if s[0] == n]
        if starts_sel is not None:
            starts = starts[starts_sel[0]:starts_sel[1] + 1]
    print(f"\n  {len(starts)} start(s): " + ", ".join(n for n, _ in starts))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, p0 in starts:
        p0 = np.clip(p0, np.array(bounds)[:, 0], np.array(bounds)[:, 1])
        t0 = time.time(); lp.n_eval = 0
        l0 = lp.loss(p0)
        r = minimize_loss(lp.loss, p0, method="lbfgsb", bounds=bounds,
                          max_evals=S2F.MAX_EVALS["smoke" if smoke else "full"], fd_step=1e-5)
        dt = time.time() - t0
        rl = railed(lp.names, r.x, bounds)
        print(f"  start {name:<9} loss {l0:.6f} -> {r.fun:.6f}  ({lp.n_eval} evals, {dt / 60:.1f} min)  "
              + ", ".join(f"{k} = {r.x[lp.names.index(k)]:+.3f}" for k in knobs)
              + (f"  RAILED: {','.join(rl)}" if rl else ""), flush=True)
        print(f"    {lp.describe(r.x)}")
        per_tab = lp.table(r.x)
        np.savez(OUTDIR / f"{STAGE_FILES['stage1']}{tag}_start_{name}.npz", name=name, theta0=p0, theta=np.asarray(r.x),
                 loss=float(r.fun), loss0=float(l0), n_eval=lp.n_eval, railed=np.array(rl), per_epoch=per_tab,
                 loss_null=l_null, bounds=np.array(bounds), theta_names=np.array(lp.names), knobs=np.array(knobs),
                 theta_nested=th_nested13, theta_baseline=starts_base(th_base, knobs), fit_rows=pr.all_rows,
                 n_fit=np.array([len(pr.rows[k]) for k in pr.epochs]), l_s_ref=pr.l_s_ref, l_b_ref=pr.l_b_ref,
                 extended_family=spec2.extended_family, objective="binned", compact_in_kpc=spec2.compact_in_kpc,
                 fit_epoch=-1, fit_sample="sane", curves="measured (adopted 2026-09-08)", bins="measured mass",
                 n_parameters=n_par)
        print(f"  wrote {OUTDIR / f'{STAGE_FILES['stage1']}{tag}_start_{name}.npz'}", flush=True)


def starts_base(th_base, knobs):
    return np.r_[th_base, np.zeros(len(knobs))]


def merge_starts(tag):
    files = sorted(OUTDIR.glob(f"{STAGE_FILES['stage1']}{tag}_start_*.npz"))
    assert files, "no start files to merge"
    res = [dict(np.load(f, allow_pickle=True)) for f in files]
    ls = np.array([float(r["loss"]) for r in res])
    best = res[int(np.argmin(ls))]
    names = [str(r["name"]) for r in res]
    knobs = [str(k) for k in best["knobs"]]
    tn = [str(n) for n in best["theta_names"]]
    print(f"  stage1{tag}: merged {len(res)} starts: " + ", ".join(
        f"{n} {l:.4f} (" + ", ".join(f"{k} {float(r['theta'][tn.index(k)]):+.3f}" for k in knobs) + ")" for n, l, r in zip(names, ls, res)))
    print(f"  best: {best['name']} {float(best['loss']):.6f} (null {float(best['loss_null']):.4f}, "
          f"{100 * (1 - float(best['loss']) / float(best['loss_null'])):+.1f}%)"
          + (f"; RAILED {list(best['railed'])}" if len(best["railed"]) else ""))
    n_agree = int((ls < ls.min() + 0.01).sum())
    print(f"  {n_agree} of {len(res)} within 0.01 of the best" + ("  (a single basin)" if n_agree >= 3 else ""))
    out = OUTDIR / f"{STAGE_FILES['stage1']}{tag}.npz"
    keep = {k: best[k] for k in best if k not in ("name", "theta0", "theta", "loss", "loss0", "n_eval", "railed", "per_epoch")}
    np.savez(out, theta_best=best["theta"], loss_best=float(best["loss"]), best_name=str(best["name"]),
             per_epoch_best=best["per_epoch"], names=np.array(names), losses=ls,
             thetas=np.array([r["theta"] for r in res]), losses0=np.array([float(r["loss0"]) for r in res]),
             n_evals=np.array([int(r["n_eval"]) for r in res]), railed_best=best["railed"], **keep)
    print(f"  wrote {out}")


if __name__ == "__main__":
    a = sys.argv
    ss = None
    if "--starts" in a:
        lo, hi = a[a.index("--starts") + 1].split(":")
        ss = (int(lo), int(hi))
    kn = tuple(a[a.index("--knobs") + 1].split(",")) if "--knobs" in a else DEFAULT_KNOBS
    if "--knobs" in a and a[a.index("--knobs") + 1] == "none":
        kn = ()
    fx = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a[a.index("--fix") + 1].split(",")} if "--fix" in a else None
    dl = float(a[a.index("--delay") + 1]) if "--delay" in a else None
    main(smoke="--smoke" in a, starts_sel=ss, merge="--merge" in a,
         cont=(a[a.index("--continue") + 1] if "--continue" in a else None), knobs=kn, fix=fx, delay=dl,
         delay_form=("exp" if "--delay-exp" in a else "step"),
         start_from=(a[a.index("--start-from") + 1] if "--start-from" in a else None))
