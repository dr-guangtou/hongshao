"""exp74 Stage 1 — exp63's joint mean REFITTED on the pre-epoch halo curves.

THE CURVES ARE THE ONLY CHANGE. Same model (`model2` two-channel mean, exp63's
spec), same sample rule, same objective (exp63's four-term binned fixed-kpc
loss with its references frozen at the nested incumbent ON THE OFFICIAL
CURVES, so every term is normalised exactly as exp63's was), same bounds,
same starts (nested, exp63's own optimum, default, jitter0).

What differs from exp63's `JointProblem2`: each epoch's prediction uses THAT
epoch's curve list (`history.load_curves`), the curve fitted only to the
history before the epoch and anchored there. That costs one `predict2` call
per epoch per evaluation (five instead of one), so the four starts are run as
four processes (`--starts i:i`), ~0.4 GB each, and merged by `--merge`.

Run (each start in its own process, the MAIN shell):
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. nohup uv run python -u experiments/exp74_c19_history_leak/stage1_refit.py \\
        --starts k:k > experiments/exp74_c19_history_leak/outputs/stage1_refit_s{k}.log 2>&1 &
then
    ... stage1_refit.py --merge
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
import stage0_cost as S0                                 # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
import history as HI                                     # noqa: E402
from hongshao.fitting import minimize_loss               # noqa: E402

RULE = "=" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR = HERE / "outputs"
FIT_NPZ = ROOT / "experiments/exp63_analytic_growth/outputs/stage2_fit_joint_kpc_free_sane.npz"
START_ORDER = ["nested", "exp63", "default", "jitter0"]


class PreEpochJoint(S2F.JointProblem2):
    """exp63's joint problem with one curve list PER EPOCH. The references are
    the parent's: the nested incumbent on the OFFICIAL curves."""

    def __init__(self, spec2, curves_official, curves_by_epoch, data, mask, lmh, R, th_inc, binned=True):
        super().__init__(spec2, curves_official, data, mask, lmh, R, th_inc, binned=binned)
        self.curves_k = {k: [curves_by_epoch[k][i] for i in self.all_rows] for k in self.epochs}

    def per_epoch(self, theta, nodes=M2.FIT_NODES):
        out = {}
        for k in self.epochs:
            m = M2.predict2(self.spec2, theta, self.curves_k[k], self.R, epochs=(k,), nodes=nodes)[:, 0, :]
            out[k] = self.problems[k].score_model(m[self.index[k]])
        return out


def build(smoke=False):
    recs, data, mask_legacy, lmh, spec_inc, th_inc, _ = S0.build(smoke)
    mask, _ = S2F.fit_masks(data, mask_legacy, lmh, "sane")
    curves = E.build_curves(recs, verbose=False)
    fz = np.load(FIT_NPZ, allow_pickle=True)
    spec2 = S2F.spec_from_fit(fz)
    hist_path = OUTDIR / f"history_curves{'_smoke' if smoke else ''}.npz"
    pre = {k: HI.load_curves(curves, k, hist_path) for k in EPOCHS}
    pr = PreEpochJoint(spec2, curves, pre, data, mask, lmh, F.R_GRID, th_inc, binned=True)
    return recs, data, mask, lmh, curves, pre, fz, spec2, th_inc, pr


def main(smoke=False, starts_sel=None, merge=False):
    tag = "_smoke" if smoke else ""
    if merge:
        return merge_starts(tag)
    print(f"{RULE}\nexp74 Stage 1 — exp63's joint mean refitted on PRE-EPOCH curves; the curves are the only "
          f"change\n{RULE}\n")
    recs, data, mask, lmh, curves, pre, fz, spec2, th_inc, pr = build(smoke)
    print(f"  JOINT FIT at all five epochs: masks " + " / ".join(f"{len(pr.rows[k])}" for k in pr.epochs)
          + f" galaxies (union {len(pr.all_rows)}); references = the nested incumbent on the OFFICIAL curves")
    th_nested = M2.with_levers_theta(M2.nested_theta(th_inc, delay=spec2.delay, growth=spec2.growth_split), spec2)
    th_exp63 = np.asarray(fz["theta_best"], float)
    # the control that keeps this honest: the parent's per_epoch on the official
    # curves must return every term = 1 at the nested theta
    ref = S2F.JointProblem2.per_epoch(pr, th_nested, nodes=M2.FULL_NODES)
    terms = np.array([[ref[k][0], ref[k][1], ref[k][2], ref[k][4]] for k in pr.epochs])
    # S and B carry references (1 at the null by construction); A and F are
    # absolute scores in units of the incumbent's own scatter, near but not
    # exactly 1 -- exactly as in exp63's joint fit
    assert np.allclose(terms[:, 2:], 1.0, atol=2e-2), terms
    print(f"  GATE  on the official curves the nested incumbent scores 1.000 in the referenced terms S, B "
          f"(max dev {np.max(np.abs(terms[:, 2:] - 1)):.1e}); A, F per epoch "
          + " ".join(f"{a:.2f}/{f:.2f}" for a, f in terms[:, :2]) + "  OK")
    l_nested = pr.loss(th_nested)
    pr.report(th_nested, "the incumbent's theta ON PRE-EPOCH curves (not a null here)")
    l_63 = pr.loss(th_exp63)
    pr.report(th_exp63, "exp63's theta ON PRE-EPOCH curves (frozen-theta cost of the input change)")
    print(f"  (exp63's theta on the official curves scored {float(fz['loss_best']):.4f})")

    bounds = np.asarray(fz["bounds"], float).tolist()
    rng = np.random.default_rng(63)
    starts = S2F.starts_for(spec2, th_inc, rng)
    starts.append(("exp63", th_exp63.copy()))
    starts = [s for n in START_ORDER for s in starts if s[0] == n]
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
        print(f"  start {name:<8} loss {l0:.6f} -> {r.fun:.6f}  ({pr.n_eval} evals, {dt / 60:.1f} min)"
              f"{'  RAILED: ' + ','.join(rl) if rl else ''}", flush=True)
        per = pr.per_epoch(r.x, nodes=M2.FULL_NODES)
        # a start that ends on an unbuildable model returns short score tuples
        per_tab = np.array([[per[k][0], per[k][1], per[k][2], per[k][4]] if len(per[k]) >= 5
                            else [np.nan] * 4 for k in pr.epochs])
        np.savez(OUTDIR / f"stage1_refit{tag}_start_{name}.npz", name=name, theta0=p0, theta=np.asarray(r.x),
                 loss=float(r.fun), loss0=float(l0), n_eval=pr.n_eval, railed=np.array(rl),
                 per_epoch=per_tab,
                 loss_nested_pre=l_nested, loss_exp63_pre=l_63, loss_exp63_official=float(fz["loss_best"]),
                 bounds=np.array(bounds), theta_names=np.array(spec2.theta_names),
                 theta_nested=th_nested, theta_exp63=th_exp63, fit_rows=pr.all_rows,
                 n_fit=np.array([len(pr.rows[k]) for k in pr.epochs]),
                 l_s_ref=pr.l_s_ref, l_b_ref=pr.l_b_ref,
                 extended_family=spec2.extended_family, objective="binned",
                 compact_in_kpc=spec2.compact_in_kpc, fit_epoch=-1, fit_sample="sane",
                 curves="pre-epoch (exp74 history.py)")
        print(f"  wrote {OUTDIR / f'stage1_refit{tag}_start_{name}.npz'}", flush=True)


def merge_starts(tag):
    files = [OUTDIR / f"stage1_refit{tag}_start_{n}.npz" for n in START_ORDER]
    files = [f for f in files if f.exists()]
    assert files, "no start files to merge"
    res = [dict(np.load(f, allow_pickle=True)) for f in files]
    ls = np.array([float(r["loss"]) for r in res])
    best = res[int(np.argmin(ls))]
    names = [str(r["name"]) for r in res]
    print(f"  merged {len(res)} starts: " + ", ".join(f"{n} {l:.4f}" for n, l in zip(names, ls)))
    print(f"  best: {best['name']} {float(best['loss']):.6f}; exp63's theta on pre-epoch curves "
          f"{float(best['loss_exp63_pre']):.4f}; exp63 on official {float(best['loss_exp63_official']):.4f}")
    n_agree = int((ls < ls.min() + 0.01).sum())
    print(f"  {n_agree} of {len(res)} within 0.01 of the best" + ("  (a single basin)" if n_agree >= 3 else ""))
    out = OUTDIR / f"stage1_refit{tag}.npz"
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
    main(smoke="--smoke" in a, starts_sel=ss, merge="--merge" in a)
