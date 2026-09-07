"""exp76 Stage 1 — the fit: exp63's model with the growth-rate split `g_split`
free, on the MEASURED curves under the ADOPTED references.

Same objective, sample, bounds and engine as the adopted baseline; one extra
parameter. The starts span Stage 0's sweep: exp74's measured optimum with
g = 0 (the nested start: g = 0 IS the baseline model), g = -1 and g = -2, and
a near start around g = -1.5. Each start is its own process (`--starts k:k`),
merged with `--merge`.

Run:
    HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. nohup uv run python -u \\
        experiments/exp76_growth_rate_split/stage1_fit.py --starts k:k > outputs/stage1_fit_s{k}.log 2>&1 &
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
          ROOT / "experiments/exp63_analytic_growth",
          ROOT / "experiments/exp74_c19_history_leak", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
import measured as MB                                    # noqa: E402
from stage0_leverage import adopted_problem, E74, FIT_NPZ  # noqa: E402
from hongshao.fitting import minimize_loss               # noqa: E402

RULE = "=" * 100
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR = HERE / "outputs"
START_ORDER = ["g0", "g-1", "g-2", "near-1.5"]


def build(smoke=False):
    recs, data, mask_legacy, lmh_dm, spec_inc, th_inc, _ = S0.build(smoke)
    mask, _ = S2F.fit_masks(data, mask_legacy, lmh_dm, "sane")
    fz = np.load(FIT_NPZ, allow_pickle=True)
    spec2 = S2F.spec_from_fit(fz)
    spec_g = M2.Spec2(theta_names=M2.THETA_NAMES_GROWTH, extended_family=spec2.extended_family,
                      compact_in_kpc=spec2.compact_in_kpc)
    hist_path = E74 / f"history_curves{'_smoke' if smoke else ''}.npz"
    meas = MB.build_input(recs, "measured", hist_path=hist_path, verbose=True)[0]
    pr, lmh_cat = adopted_problem(spec_g, recs, data, mask, lmh_dm, meas, th_inc)
    f0 = E74 / f"stage1_refit_measured{'_smoke' if smoke else ''}.npz"
    if not f0.exists():
        f0 = E74 / "stage1_refit_measured.npz"
    th0 = np.asarray(np.load(f0, allow_pickle=True)["theta_best"], float)
    return recs, data, mask, lmh_dm, meas, fz, spec2, spec_g, th_inc, pr, th0


def main(smoke=False, starts_sel=None, merge=False):
    tag = "_smoke" if smoke else ""
    if merge:
        return merge_starts(tag)
    print(f"{RULE}\nexp76 Stage 1 — the growth-rate split fitted on the measured curves, adopted references\n{RULE}\n")
    recs, data, mask, lmh_dm, meas, fz, spec2, spec_g, th_inc, pr, th0 = build(smoke)
    th_nested = M2.nested_theta(th_inc, growth=True)
    ref = pr.per_epoch(th_nested, nodes=M2.FULL_NODES)
    terms = np.array([[ref[k][0], ref[k][1], ref[k][2], ref[k][4]] for k in pr.epochs])
    assert np.allclose(terms[:, 2:], 1.0, atol=2e-2), terms
    l_null = pr.loss(th_nested)
    print(f"  GATE  the nested incumbent (g = 0) on the measured curves scores 1.000 in S, B  OK; null loss {l_null:.4f}")
    jg = spec_g.index("g_split")
    thg0 = np.r_[th0, 0.0]
    pr.report(thg0, "exp74's measured optimum with g = 0 (the baseline model, scored here)")
    bounds = [tuple(b) for b in np.asarray(fz["bounds"], float).tolist()] + [spec_g.bounds()[jg]]
    rng = np.random.default_rng(76)
    span = np.array([hi - lo for lo, hi in bounds])
    def at(g):
        th = thg0.copy(); th[jg] = g; return th
    near = M2.clip_to_bounds(spec_g, at(-1.5) + 0.05 * span * rng.standard_normal(len(span)))
    starts = [("g0", at(0.0)), ("g-1", at(-1.0)), ("g-2", at(-2.0)), ("near-1.5", near)]
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
        rl = S2F.railed(spec_g, r.x, bounds=bounds)
        print(f"  start {name:<9} loss {l0:.6f} -> {r.fun:.6f}  ({pr.n_eval} evals, {dt / 60:.1f} min)  "
              f"g = {r.x[jg]:+.3f}{'  RAILED: ' + ','.join(rl) if rl else ''}", flush=True)
        per = pr.per_epoch(r.x, nodes=M2.FULL_NODES)
        per_tab = np.array([[per[k][0], per[k][1], per[k][2], per[k][4]] if len(per[k]) >= 5
                            else [np.nan] * 4 for k in pr.epochs])
        np.savez(OUTDIR / f"stage1_fit{tag}_start_{name}.npz", name=name, theta0=p0, theta=np.asarray(r.x),
                 loss=float(r.fun), loss0=float(l0), n_eval=pr.n_eval, railed=np.array(rl), per_epoch=per_tab,
                 loss_null=l_null, bounds=np.array(bounds), theta_names=np.array(spec_g.theta_names),
                 theta_nested=th_nested, theta_g0=thg0, fit_rows=pr.all_rows,
                 n_fit=np.array([len(pr.rows[k]) for k in pr.epochs]), l_s_ref=pr.l_s_ref, l_b_ref=pr.l_b_ref,
                 extended_family=spec_g.extended_family, objective="binned", compact_in_kpc=spec_g.compact_in_kpc,
                 fit_epoch=-1, fit_sample="sane", curves="measured (adopted)", bins="measured mass",
                 growth_split=True)
        print(f"  wrote {OUTDIR / f'stage1_fit{tag}_start_{name}.npz'}", flush=True)


def merge_starts(tag):
    files = [OUTDIR / f"stage1_fit{tag}_start_{n}.npz" for n in START_ORDER]
    files = [f for f in files if f.exists()]
    assert files, "no start files to merge"
    res = [dict(np.load(f, allow_pickle=True)) for f in files]
    ls = np.array([float(r["loss"]) for r in res])
    best = res[int(np.argmin(ls))]
    names = [str(r["name"]) for r in res]
    jg = list(best["theta_names"]).index("g_split")
    print(f"  merged {len(res)} starts: " + ", ".join(f"{n} {l:.4f} (g {float(r['theta'][jg]):+.2f})"
                                                       for n, l, r in zip(names, ls, res)))
    print(f"  best: {best['name']} {float(best['loss']):.6f}, g = {float(best['theta'][jg]):+.3f} "
          f"(null {float(best['loss_null']):.4f})")
    n_agree = int((ls < ls.min() + 0.01).sum())
    print(f"  {n_agree} of {len(res)} within 0.01 of the best" + ("  (a single basin)" if n_agree >= 3 else ""))
    out = OUTDIR / f"stage1_fit{tag}.npz"
    keep = {k: best[k] for k in best if k not in ("name", "theta0", "theta", "loss", "loss0", "n_eval", "railed", "per_epoch")}
    np.savez(out, theta_best=best["theta"], loss_best=float(best["loss"]), best_name=str(best["name"]),
             per_epoch_best=best["per_epoch"], names=np.array(names), losses=ls,
             thetas=np.array([r["theta"] for r in res]), losses0=np.array([float(r["loss0"]) for r in res]),
             g_by_start=np.array([float(r["theta"][jg]) for r in res]), railed_best=best["railed"], **keep)
    print(f"  wrote {out}")


if __name__ == "__main__":
    a = sys.argv
    ss = None
    if "--starts" in a:
        lo, hi = a[a.index("--starts") + 1].split(":")
        ss = (int(lo), int(hi))
    main(smoke="--smoke" in a, starts_sel=ss, merge="--merge" in a)
