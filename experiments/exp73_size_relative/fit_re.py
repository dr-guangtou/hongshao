"""exp73 Block C — refit the exp63 model with the galaxy scored where the galaxy is.

WHAT THE OBJECTIVE IS, and why each piece is there. Per epoch,

    L_k = mean_g[ per-galaxy rms over R50 shells of (dM_shell / M_total)^2 ] / ref_pg[k]
        + rms over shells and halo-mass terciles of the tercile-MEDIAN of the same residual / ref_bin[k]

  the coordinate   20 shells at fixed multiples of each galaxy's OWN half-mass
                   radius, 0.5 to 6 R50 (Block A1: below 0.5 R50 the measurement
                   runs out at z = 2; above 6 the grid does at z = 0.4). The
                   truth's R50, never the model's -- a model scored on its own
                   size scale satisfies the objective by shrinking, and the
                   coordinate module asserts the model cannot move it.
  the residual     MASS-WEIGHTED, (model - truth) / the galaxy's total, not the
                   shell's own mass. Block B: the relative form lets two galaxies
                   carry 78 per cent of the whole population's loss at z = 2
                   through near-empty outer shells; this form is closest to
                   proportional at every epoch and the least concentrated, and
                   the user's threshold gate was disqualified by the rule fixed
                   in advance.
  the binned term  exp63's own tercile-median gate translated to the new
                   coordinate. Block A2: a per-galaxy metric puts a shared theta
                   and five per-epoch thetas within 2 per cent of each other
                   while the binned gate separates them 2.3-2.7x, so an objective
                   without it cannot see the tension it is meant to resolve
                   (memory `loss-is-blind-to-the-binned-gate`).
  the references   each term divided by the NESTED INCUMBENT's value at that
                   epoch, so every epoch starts at 2.000 and the five-epoch null
                   is 10.000 -- exp63's own convention, so the two fits' losses
                   read the same way.

WHAT IS HELD FIXED so the objective is the only change: the same twelve
parameters, bounds, starts, sample rule, engine and five epochs jointly as
`exp63/outputs/stage2_fit_joint_kpc_free_sane.npz`, and the same as exp72's
`fit_gls.py`. The curve of growth inside 2 kpc comes from the density-rebuilt
curve spliced onto `cog_provided` (Block A1, `coordinate.extended_cog`); outside
2 kpc the target IS `cog_provided`, unchanged, per the user's D1.

WHAT THE FIT IS JUDGED ON afterwards (`eval_re.py`): the exp72 transfer matrix
in BOTH coordinates, the standard battery, and the new tier 2d SIZE GATE --
which is where the mass-weighting's one measured cost (about 3x less sensitive to
a pure size error relative to amplitude at z = 2) will show if it matters.

Run (long; the MAIN shell, never a subagent; each start checkpoints):
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. nohup uv run python -u \\
        experiments/exp73_size_relative/fit_re.py [--smoke] [--starts 0:3] [--epoch k] \\
        > experiments/exp73_size_relative/outputs/fit_re.log 2>&1 &
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
          ROOT / "experiments/exp72_error_objective", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
import fit_gls as G                                      # noqa: E402
import coordinate as C                                   # noqa: E402
from risk_test import binned_loss                        # noqa: E402
from hongshao.fitting import minimize_loss               # noqa: E402
from hongshao.profile_data import load_profiles          # noqa: E402

OUTDIR = HERE / "outputs"
RULE = "=" * 100
ANCHOR_Z = E.ANCHOR_Z
EPOCHS = (0, 1, 2, 3, 4)
POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
EXP63_NPZ = ROOT / "experiments/exp63_analytic_growth/outputs/stage2_fit_joint_kpc_free_sane.npz"
MAX_EVALS = {"smoke": 60, "full": 3000}
#: the coordinate, as Blocks A and B settled it
N_EDGE, RE_LO, RE_HI = 21, 0.5, 6.0


class ReProblem:
    """One theta, five epochs, on the size-relative coordinate.

    Everything the objective needs is measured once from the NESTED INCUMBENT
    and frozen: the coordinate (from the truth), and the two per-epoch
    normalisations. Nothing here moves with the current theta, because an
    objective whose own scales move with the parameters is not an objective.
    """

    def __init__(self, spec2, curves, truth_m, Rm, lmh0, th_nested,
                 edges=None, epochs=EPOCHS):
        self.spec2, self.Rm = spec2, np.asarray(Rm, float)
        self.curves, self.lmh0 = curves, np.asarray(lmh0, float)
        self.n = len(curves)
        # a single-epoch problem sees only that epoch's truth and predicts only
        # that epoch (the model is the same integral truncated at its anchor)
        self.epochs = tuple(int(k) for k in epochs)
        self.anchor_z = [ANCHOR_Z[k] for k in self.epochs]
        edges = tuple(np.geomspace(RE_LO, RE_HI, N_EDGE)) if edges is None else edges
        self.grid = C.SizeGrid(np.asarray(truth_m)[:, list(self.epochs), :],
                               self.Rm, edges=edges)              # truth-only
        self.w = C.weights_uniform(self.grid)
        self.n_eval = 0
        null = M2.predict2(spec2, th_nested, curves, self.Rm, epochs=self.epochs,
                           nodes=M2.FULL_NODES)
        pg, bn = self._parts(null)
        self.ref_pg, self.ref_bin = pg, bn

    def _parts(self, cogs):
        """(per-galaxy term, binned term), each (5,), unnormalised."""
        r = self.grid.residual(cogs, "mass")                    # (n, 5, shells)
        per = C.apply_weights(r, self.w)                        # (n, 5)
        pg = np.array([np.nanmean(per[:, j]) for j in range(len(self.epochs))])
        bn = binned_loss(self.grid, r, self.lmh0)
        return pg, bn

    def per_epoch(self, theta, nodes=M2.FIT_NODES):
        """(5,) each epoch's loss, exactly 2.000 at the nested incumbent."""
        cogs = M2.predict2(self.spec2, theta, self.curves, self.Rm,
                           epochs=self.epochs, nodes=nodes)
        bad = ~(np.isfinite(cogs).all(axis=(1, 2)) & (cogs > 0).all(axis=(1, 2)))
        if bad.any():
            cogs = np.where(bad[:, None, None], np.nan, cogs)
        pg, bn = self._parts(cogs)
        return pg / self.ref_pg + bn / self.ref_bin, int(bad.sum())

    def loss(self, theta):
        self.n_eval += 1
        per, n_bad = self.per_epoch(theta)
        if not np.isfinite(per).all():
            return F.FAIL * len(self.epochs)
        return float(per.sum()) + F.FAIL * n_bad / self.n

    def report(self, theta, label, nodes=M2.FULL_NODES):
        per, n_bad = self.per_epoch(theta, nodes=nodes)
        print(f"  {label}: " + "  ".join(f"z={z}:{v:.4f}" for z, v in zip(self.anchor_z, per))
              + f"  -> total {np.nansum(per):.4f}"
              + (f"  ({n_bad} unbuildable)" if n_bad else ""))
        return per


def build(smoke=False):
    """The sample on the merged grid, and everything a fit or its judge needs."""
    (recs, data, lmh, curves, sel, spec2, th_inc, th_nested, fz63,
     keep, sig_cog, sig_ann, ann_ok) = G.build(smoke)
    pop = np.load(POP, allow_pickle=True)
    idx = np.asarray(pop["index"], int)[sel]
    d = load_profiles()
    Rm, truth_all = C.extended_cog(data, d["cog_from_density_shared"][idx],
                                   F.R_GRID, d["shared_grid_kpc"])
    # `good` is TRUTH-ONLY: it must not depend on any model, because the model
    # changes at every evaluation. A theta that cannot build a galaxy pays the
    # FAIL penalty for it instead.
    good = keep.all(1) & np.isfinite(truth_all).all(axis=(1, 2)) & (truth_all > 0).all(axis=(1, 2))
    rows = np.where(good)[0]
    print(f"  THE FITTING SAMPLE on the merged grid: {len(rows)} galaxies, "
          f"{len(Rm)} radii from {Rm[0]:.3f} to {Rm[-1]:.2f} kpc")
    return (spec2, th_inc, th_nested, fz63, [curves[i] for i in rows],
            truth_all[good], Rm, lmh[good][:, 0], rows, data, lmh, good)


def main(smoke=False, starts_sel=None, epoch=None):
    """``epoch=None`` fits the five epochs jointly (Block C); ``epoch=k`` fits
    that epoch ALONE with the four time exponents frozen exactly as exp63's
    per-epoch fits froze them (a single epoch cannot constrain them), so the
    result is comparable with `stage2_fit_frozen_binned_kpc_sane_e{k}` — the
    same freedom, the R50 objective instead of fixed kpc. That is A2's true
    per-epoch comparator."""
    tag = ("" if epoch is None else f"_e{epoch}") + ("_smoke" if smoke else "")
    print(f"{RULE}\nexp73 Block C — the exp63 model refitted on R50 shells with a "
          f"mass-weighted residual"
          + ("" if epoch is None else f", at z={ANCHOR_Z[epoch]} ALONE")
          + f"\n{RULE}\n")
    (spec2, th_inc, th_nested, fz63, curves, truth_m, Rm, lmh0,
     rows, data, lmh, good) = build(smoke)
    frozen, th_e63 = {}, None
    if epoch is not None:
        fze = np.load(EXP63_NPZ.parent / f"stage2_fit_frozen_binned_kpc_sane_e{epoch}.npz",
                      allow_pickle=True)
        frozen = {str(a): float(b) for a, b in zip(fze["frozen_names"], fze["frozen_values"])}
        th_e63 = np.asarray(fze["theta_best"], float)
        print(f"  single epoch z={ANCHOR_Z[epoch]}: FROZEN "
              + ", ".join(f"{k}={v:+.4f}" for k, v in frozen.items())
              + " (as exp63's per-epoch fit froze them)")
    pr = ReProblem(spec2, curves, truth_m, Rm, lmh0, th_nested,
                   epochs=EPOCHS if epoch is None else (epoch,))
    print(f"  {pr.grid.n_shell} shells at {RE_LO}-{RE_HI} x each galaxy's own R50; "
          f"the coordinate is truth-only")
    print(f"  per-epoch normalisation (frozen): per-galaxy "
          + " ".join(f"{v:.3g}" for v in pr.ref_pg) + " | binned "
          + " ".join(f"{v:.3g}" for v in pr.ref_bin))

    def with_frozen(th):
        th = np.asarray(th, float).copy()
        for nm, val in frozen.items():
            th[spec2.index(nm)] = val
        return th

    l_null = pr.loss(th_nested)
    per_null = pr.report(th_nested, "the null (nested incumbent)")
    assert np.allclose(per_null, 2.0, atol=2e-2), per_null
    print(f"  GATE  the nested start reproduces the incumbent and each epoch "
          f"normalises to 2.000 (max deviation {np.max(np.abs(per_null - 2.0)):.1e})  OK")
    th_exp63 = np.asarray(fz63["theta_best"], float)
    pr.report(th_exp63, "exp63's theta, scored under this objective")

    bounds = [tuple(b) for b in np.asarray(fz63["bounds"], float).tolist()]
    for nm, val in frozen.items():
        bounds[spec2.index(nm)] = (val, val)          # an equal-bounds box freezes it
    rng = np.random.default_rng(73)
    starts = S2F.starts_for(spec2, th_inc, rng)
    starts.append(("exp63", th_exp63.copy()))
    # the four that matter, in the order that tests the basin soonest: the
    # nested incumbent, exp63's own optimum, a default, and one jitter
    order = ["nested", "exp63", "default", "jitter0"]
    starts = [s for n in order for s in starts if s[0] == n]
    if epoch is not None:
        # a single-epoch fit also starts from exp63's own per-epoch optimum,
        # the fixed-kpc comparator it is meant to replace
        starts.insert(2, (f"exp63_e{epoch}", th_e63.copy()))
        starts = [(n, with_frozen(th)) for n, th in starts]
        pr.report(th_e63, f"exp63's per-epoch theta (e{epoch}), scored under this objective")
    if starts_sel is not None:
        starts = starts[starts_sel[0]:starts_sel[1] + 1]
    print(f"\n  {len(starts)} starts: " + ", ".join(n for n, _ in starts))

    results = []
    OUTDIR.mkdir(parents=True, exist_ok=True)
    partial = OUTDIR / f"fit_re{tag}.PARTIAL.npz"
    for name, p0 in starts:
        t0 = time.time(); pr.n_eval = 0
        l0 = pr.loss(p0)
        r = minimize_loss(pr.loss, p0, method="lbfgsb", bounds=bounds,
                          max_evals=MAX_EVALS["smoke" if smoke else "full"], fd_step=1e-5)
        dt = time.time() - t0
        rl = S2F.railed(spec2, r.x, bounds=bounds)
        print(f"  start {name:<8} loss {l0:.6f} -> {r.fun:.6f}  ({pr.n_eval} evals, "
              f"{dt / 60:.1f} min){'  RAILED: ' + ','.join(rl) if rl else ''}", flush=True)
        results.append(dict(name=name, theta=np.asarray(r.x), loss=float(r.fun),
                            loss0=float(l0), railed=rl))
        np.savez(partial, names=np.array([q["name"] for q in results]),
                 thetas=np.array([q["theta"] for q in results]),
                 losses=np.array([q["loss"] for q in results]),
                 loss_null=l_null, theta_nested=th_nested,
                 ref_pg=pr.ref_pg, ref_bin=pr.ref_bin,
                 theta_names=np.array(spec2.theta_names))

    best = min(results, key=lambda q: q["loss"])
    ls = np.array([q["loss"] for q in results])
    print(f"\n  best start: {best['name']} loss {best['loss']:.6f} "
          f"(null {l_null:.6f}, {100 * (1 - best['loss'] / l_null):+.1f}%)")
    print("  " + "  ".join(f"{n}={v:+.4f}" for n, v in zip(spec2.theta_names, best["theta"])))
    if best["railed"]:
        print(f"  RAILED at the bound: {', '.join(best['railed'])}")
    pr.report(best["theta"], "best (full nodes)")
    n_agree = int((ls < best["loss"] + 0.01).sum())
    print(f"  {n_agree} of {len(results)} starts within 0.01 of the best"
          + ("  (a single basin)" if n_agree >= 3 else
             "  — NOT enough agreement to call the basin settled"))

    out = OUTDIR / f"fit_re{tag}.npz"
    np.savez(out, theta_best=best["theta"], loss_best=best["loss"], best_name=best["name"],
             loss_null=l_null, ref_pg=pr.ref_pg, ref_bin=pr.ref_bin,
             edges=np.array(pr.grid.edges), bounds=np.array(bounds),
             theta_nested=th_nested, theta_exp63=th_exp63,
             theta_names=np.array(spec2.theta_names),
             extended_family=spec2.extended_family, objective="re_mass_binned",
             compact_in_kpc=spec2.compact_in_kpc,
             fit_epoch=-1 if epoch is None else int(epoch), fit_sample="sane",
             frozen_names=np.array(list(frozen)), frozen_values=np.array(list(frozen.values())),
             n_fit=len(rows), fit_rows=rows,
             names=np.array([q["name"] for q in results]),
             thetas=np.array([q["theta"] for q in results]), losses=ls,
             losses0=np.array([q["loss0"] for q in results]))
    print(f"\nwrote {out}")
    return best


if __name__ == "__main__":
    a = sys.argv
    ss = None
    if "--starts" in a:
        lo, hi = a[a.index("--starts") + 1].split(":")
        ss = (int(lo), int(hi))
    ep = int(a[a.index("--epoch") + 1]) if "--epoch" in a else None
    main(smoke="--smoke" in a, starts_sel=ss, epoch=ep)
