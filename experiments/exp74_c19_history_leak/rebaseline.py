"""ADOPTION of the measured halo history, and the re-baseline (2026-09-08, the
user's decision after exp74).

From here the application input is `build_input(recs, "measured")`
(`measured.DEFAULT_INPUT_KIND`). Adopting it means the programme's reference
points move with it:

  1. THE REFERENCES. exp63's objective normalises its shape and binned terms
     at the nested incumbent. They are rebuilt here with the nested incumbent
     ON THE MEASURED CURVES, so "1.000 at the null" means the null on the
     adopted input. The halo-mass terciles of the binned term are by the
     MEASURED mass at each epoch (the DiffMAH mass fills the 30 galaxy-epochs
     the catalog lacks), not by the official curve's mass.
  2. THE BASELINE MEAN: exp63's 12-parameter two-channel model refitted on the
     measured curves under those references (`--model exp63`). exp74's
     measured refit (official references, DiffMAH bins) is a start.
  3. THE INCUMBENT, re-baselined: the same engine with the compact channel
     switched off (exp63's nesting: `m_half` = -1000, `d_split` = 1, the
     compact size and shape at their nesting values) and its seven remaining
     parameters refitted on the measured curves (`--model incumbent`). This is
     the honest-input incumbent every later comparison nests against.

What is NOT re-baselined here, and why: the v1 stochastic layer (exp60) is
built on exp57's X3 expansion problem, which runs on the old step engine and
reads snapshot masses generated FROM the official DiffMAH curve
(`engine.build_curves` asserts exactly that). Rebuilding it needs the layer's
stages 1-3 re-run on a predictor that takes the measured history; that is a
half-day and is recorded as owed in the README.

Run (each start as its own process, the MAIN shell, `caffeinate -i -w <pid>`):
    HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. nohup uv run python -u \\
        experiments/exp74_c19_history_leak/rebaseline.py --model exp63 --starts k:k > ... &
    ... --model incumbent --starts k:k
then `--model exp63 --merge` and `--model incumbent --merge`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp63_analytic_growth", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import selection as SEL                                  # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
import measured as MB                                    # noqa: E402
from hongshao.fitting import minimize_loss               # noqa: E402

RULE = "=" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR = HERE / "outputs"
FIT_NPZ = ROOT / "experiments/exp63_analytic_growth/outputs/stage2_fit_joint_kpc_free_sane.npz"
#: the incumbent's nesting inside exp63's spec: these are held at their nesting
#: values; the other seven (a0, a_M, a_z, a_Mz, log_f_e, b_e, c_e) are the
#: incumbent's own parameters and are fitted
INCUMBENT_FROZEN = ("m_half", "d_split", "log_f_c", "b_c", "n_c")
START_ORDER = {"exp63": ["exp74meas", "exp63", "near", "nested", "cont"],
               "incumbent": ["nested", "near_inc", "jitter_inc"]}


def measured_mass_bins(recs, lmh_dm):
    """The binning variable of the adopted objective: the measured M200c at
    each epoch, the DiffMAH value where the catalog has none."""
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    rows = np.array([h.row for h in recs])
    lmh_cat = SEL.sample_masses(hs)[rows]
    n_fill = int((~np.isfinite(lmh_cat)).sum())
    return np.where(np.isfinite(lmh_cat), lmh_cat, lmh_dm), n_fill


def build(smoke=False):
    recs, data, mask_legacy, lmh_dm, spec_inc, th_inc, _ = S0.build(smoke)
    mask, _ = S2F.fit_masks(data, mask_legacy, lmh_dm, "sane")
    fz = np.load(FIT_NPZ, allow_pickle=True)
    spec2 = S2F.spec_from_fit(fz)
    hist_path = OUTDIR / f"history_curves{'_smoke' if smoke else ''}.npz"
    meas = MB.build_input(recs, "measured", hist_path=hist_path, verbose=True)[0]
    lmh_bins, n_fill = measured_mass_bins(recs, lmh_dm)
    print(f"  ADOPTED INPUT: measured curves; binned term by the MEASURED mass ({n_fill} galaxy-epochs filled "
          f"from DiffMAH); references = the nested incumbent on the measured curves")
    pr = S2F.JointProblem2(spec2, meas, data, mask, lmh_bins, F.R_GRID, th_inc, binned=True)
    return recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr


def main(smoke=False, model="exp63", starts_sel=None, merge=False):
    tag = f"_{model}" + ("_smoke" if smoke else "")
    if merge:
        return merge_starts(model, tag)
    print(f"{RULE}\nADOPTION + RE-BASELINE on the measured history — model: {model}\n{RULE}\n")
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr = build(smoke)
    th_nested = M2.with_levers_theta(M2.nested_theta(th_inc, delay=spec2.delay, growth=spec2.growth_split), spec2)
    th_exp63 = np.asarray(fz["theta_best"], float)
    ref = pr.per_epoch(th_nested, nodes=M2.FULL_NODES)
    terms = np.array([[ref[k][0], ref[k][1], ref[k][2], ref[k][4]] for k in pr.epochs])
    assert np.allclose(terms[:, 2:], 1.0, atol=2e-2), terms
    print(f"  GATE  the nested incumbent on the MEASURED curves scores 1.000 in the referenced terms S, B "
          f"(max dev {np.max(np.abs(terms[:, 2:] - 1)):.1e}); A, F per epoch "
          + " ".join(f"{a:.2f}/{f:.2f}" for a, f in terms[:, :2]) + "  OK")
    l_null = pr.loss(th_nested)
    pr.report(th_nested, "the null (nested incumbent, measured curves)")
    pr.report(th_exp63, "exp63's theta (fitted on the official curves), scored here")

    bounds = [tuple(b) for b in np.asarray(fz["bounds"], float).tolist()]
    frozen = {}
    if model == "incumbent":
        for nm in INCUMBENT_FROZEN:
            v = float(th_nested[spec2.index(nm)])
            bounds[spec2.index(nm)] = (v, v)
            frozen[nm] = v
        print(f"  INCUMBENT: frozen at their nesting values " + ", ".join(f"{k}={v:+.3f}" for k, v in frozen.items())
              + f"; {spec2.n_theta - len(frozen)} parameters fitted")
    rng = np.random.default_rng(78)
    span = np.array([hi - lo for lo, hi in bounds])
    starts = []
    if model == "exp63":
        prev = OUTDIR / f"stage1_refit_measured{'_smoke' if smoke else ''}.npz"
        th_prev = np.asarray(np.load(prev, allow_pickle=True)["theta_best"], float)
        starts = [("exp74meas", th_prev), ("exp63", th_exp63.copy()),
                  ("near", M2.clip_to_bounds(spec2, th_prev + 0.05 * span * rng.standard_normal(len(span)))),
                  ("nested", M2.clip_to_bounds(spec2, th_nested))]   # the nesting values sit outside the box
        # a continuation of a start that stopped at the evaluation cap
        prev_start = OUTDIR / f"rebaseline{tag}_start_exp63.npz"
        if prev_start.exists():
            starts.append(("cont", np.asarray(np.load(prev_start, allow_pickle=True)["theta"], float)))
    else:
        def with_frozen(th):
            th = np.asarray(th, float).copy()
            for nm, v in frozen.items():
                th[spec2.index(nm)] = v
            return th
        starts = [("nested", th_nested.copy()),
                  ("near_inc", with_frozen(M2.clip_to_bounds(spec2, th_nested + 0.05 * span * rng.standard_normal(len(span))))),
                  ("jitter_inc", with_frozen(M2.clip_to_bounds(spec2, th_nested + 0.15 * span * rng.standard_normal(len(span)))))]
    starts = [s for n in START_ORDER[model] for s in starts if s[0] == n]
    if starts_sel is not None:
        starts = starts[starts_sel[0]:starts_sel[1] + 1]
    print(f"\n  {len(starts)} start(s): " + ", ".join(n for n, _ in starts))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, p0 in starts:
        t0 = time.time(); pr.n_eval = 0
        l0 = pr.loss(p0)
        r = minimize_loss(pr.loss, p0, method="lbfgsb", bounds=bounds,
                          max_evals=S2F.MAX_EVALS["smoke" if smoke else "full"], fd_step=1e-5)
        dt = time.time() - t0
        rl = S2F.railed(spec2, r.x, bounds=bounds)
        rl = [n for n in rl if n not in frozen]
        print(f"  start {name:<10} loss {l0:.6f} -> {r.fun:.6f}  ({pr.n_eval} evals, {dt / 60:.1f} min)"
              f"{'  RAILED: ' + ','.join(rl) if rl else ''}", flush=True)
        per = pr.per_epoch(r.x, nodes=M2.FULL_NODES)
        per_tab = np.array([[per[k][0], per[k][1], per[k][2], per[k][4]] if len(per[k]) >= 5
                            else [np.nan] * 4 for k in pr.epochs])
        np.savez(OUTDIR / f"rebaseline{tag}_start_{name}.npz", name=name, theta0=p0, theta=np.asarray(r.x),
                 loss=float(r.fun), loss0=float(l0), n_eval=pr.n_eval, railed=np.array(rl), per_epoch=per_tab,
                 loss_null=l_null, bounds=np.array(bounds), theta_names=np.array(spec2.theta_names),
                 theta_nested=th_nested, theta_exp63=th_exp63, fit_rows=pr.all_rows,
                 n_fit=np.array([len(pr.rows[k]) for k in pr.epochs]),
                 l_s_ref=pr.l_s_ref, l_b_ref=pr.l_b_ref,
                 frozen_names=np.array(list(frozen)), frozen_values=np.array(list(frozen.values())),
                 extended_family=spec2.extended_family, objective="binned",
                 compact_in_kpc=spec2.compact_in_kpc, fit_epoch=-1, fit_sample="sane",
                 curves="measured (adopted 2026-09-08)", bins="measured mass", model=model)
        print(f"  wrote {OUTDIR / f'rebaseline{tag}_start_{name}.npz'}", flush=True)


def merge_starts(model, tag):
    files = [OUTDIR / f"rebaseline{tag}_start_{n}.npz" for n in START_ORDER[model]]
    files = [f for f in files if f.exists()]
    assert files, "no start files to merge"
    res = [dict(np.load(f, allow_pickle=True)) for f in files]
    ls = np.array([float(r["loss"]) for r in res])
    best = res[int(np.argmin(ls))]
    names = [str(r["name"]) for r in res]
    print(f"  {model}: merged {len(res)} starts: " + ", ".join(f"{n} {l:.4f}" for n, l in zip(names, ls)))
    print(f"  best: {best['name']} {float(best['loss']):.6f} (null {float(best['loss_null']):.4f}, "
          f"{100 * (1 - float(best['loss']) / float(best['loss_null'])):+.1f}%)")
    n_agree = int((ls < ls.min() + 0.01).sum())
    print(f"  {n_agree} of {len(res)} within 0.01 of the best" + ("  (a single basin)" if n_agree >= 3 else ""))
    out = OUTDIR / f"rebaseline{tag}.npz"
    keep = {k: best[k] for k in best if k not in ("name", "theta0", "theta", "loss", "loss0", "n_eval", "railed", "per_epoch")}
    np.savez(out, theta_best=best["theta"], loss_best=float(best["loss"]), best_name=str(best["name"]),
             per_epoch_best=best["per_epoch"], names=np.array(names), losses=ls,
             thetas=np.array([r["theta"] for r in res]), losses0=np.array([float(r["loss0"]) for r in res]),
             railed_best=best["railed"], **keep)
    print(f"  wrote {out}")


if __name__ == "__main__":
    a = sys.argv
    ss = None
    if "--starts" in a:
        lo, hi = a[a.index("--starts") + 1].split(":")
        ss = (int(lo), int(hi))
    main(smoke="--smoke" in a, model=(a[a.index("--model") + 1] if "--model" in a else "exp63"),
         starts_sel=ss, merge="--merge" in a)
