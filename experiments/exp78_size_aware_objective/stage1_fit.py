"""exp78 Stage 1 — the fit under the SIZE-AWARE objective

    L = sum over epochs of  A^2 + F^2 + S^2 + B^2 + Z^2

with Z the radius term of `size_terms.radius_term` (the rms over R20/R50/R80
x three halo-mass terciles of the tercile-median log10 R_f(model)/R_f(truth),
on the merged 0.673-148 kpc grid), normalised to the nested incumbent on the
measured curves like S and B. Everything else is the adopted baseline's:
exp63's 12-parameter two-channel model, the measured history input
(`measured.build_input(recs, "measured")`), the adopted references, the
fitting sample, exp63's bounds, the same engine and quadrature.

PARAMETER COUNT: 12 fitted, plus 0 for the term (it has no free constant; its
weight is 1 like every other term's). An observer could vary all twelve.
Stage 2 (`--growth`): 13, the growth-rate split `g_split` added (exp76's
lever), starting from the Stage 1 optimum with g = 0, -1, -2.

The truth appears only inside the loss; no per-galaxy or per-epoch quantity
from the simulation enters the model.

Starts, each its own process (`--starts k:k`), then `--merge`:
  baseline  exp74's measured optimum (THE BASELINE MEAN)
  basin     the 14.63 basin (gate-rejected; the loss's own favourite so far)
  exp63     exp63's joint fit on the official curves
  nested    the nested incumbent, clipped into the box
  --continue NAME   a continuation of a start that stopped at the cap, from
            its own point (near starts rail into the failure penalty under
            this objective; a basin is settled from its own point)

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=. \\
        nohup uv run python -u experiments/exp78_size_aware_objective/stage1_fit.py --starts k:k \\
        > experiments/exp78_size_aware_objective/outputs/stage1_fit_s{k}.log 2>&1 &
    (caffeinate -i -w <pid> on each), then `--merge`; `--growth` for Stage 2.
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
import size_terms as ST                                  # noqa: E402
from hongshao.fitting import minimize_loss               # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S0T = _by_path("exp78_stage0_terms", HERE / "stage0_terms.py")
RB = S0T.RB

RULE = "=" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR = HERE / "outputs"
E74 = ROOT / "experiments/exp74_c19_history_leak/outputs"
START_ORDER = {"stage1": ["baseline", "basin", "exp63", "nested"],
               "stage2": ["g0", "g-1", "g-2"]}


class SizeAwareProblem:
    """exp63's joint problem plus the radius term, one `predict2` call on the
    merged grid per evaluation (its last 24 radii ARE the standard grid)."""

    def __init__(self, pr, spec, truth_m, Rm, n_inner, lmh_bins, rows_m, th_null):
        self.pr, self.spec, self.Rm, self.n_inner = pr, spec, np.asarray(Rm, float), int(n_inner)
        self.epochs = list(pr.epochs)
        index_of = np.full(int(max(pr.all_rows)) + 1, -1)
        index_of[pr.all_rows] = np.arange(len(pr.all_rows))
        self.index_m = {k: index_of[rows_m[k]] for k in self.epochs}
        self.truth_k = {k: truth_m[rows_m[k], k] for k in self.epochs}
        self.lmh_k = {k: lmh_bins[rows_m[k], k] for k in self.epochs}
        # the truth's fractional radii, computed once
        self.r_truth = {k: {f: ST.fractional_radius(self.truth_k[k], self.Rm, f) for f in ST.FRACTIONS}
                        for k in self.epochs}
        self.n_eval = 0
        self.z_ref = None
        raw = self.per_epoch(th_null)[1]
        self.z_ref = raw.copy()

    def _median_table(self, mk, k):
        """(n_frac, 3) tercile medians of log10 R_f(model)/R_f(truth) at epoch k."""
        terc = ST.tercile_masks(self.lmh_k[k])
        good = np.isfinite(mk).all(1) & (mk[:, -1] > 0)
        med = np.full((len(ST.FRACTIONS), 3), np.nan)
        for i, f in enumerate(ST.FRACTIONS):
            d = np.log10(ST.fractional_radius(mk, self.Rm, f) / self.r_truth[k][f])
            for b, t in enumerate(terc):
                sel = t & good & np.isfinite(d)
                if sel.sum() >= 5:
                    med[i, b] = np.median(d[sel])
        return med

    def _radius_term(self, mk, k):
        return float(np.sqrt(np.nanmean(self._median_table(mk, k) ** 2)))

    def median_tables(self, spec, theta, nodes=M2.FULL_NODES):
        """(5, n_frac, 3) the term's tercile-median tables per epoch, for any spec."""
        m = M2.predict2(spec, theta, self.pr.curves, self.Rm, epochs=tuple(self.epochs), nodes=nodes)
        return np.array([self._median_table(m[self.index_m[k], j], k) for j, k in enumerate(self.epochs)])

    def per_epoch(self, theta, nodes=M2.FIT_NODES):
        """({k: (A, F, S, n_bad, B)}, (5,) raw radius term, (5,) normalised Z)."""
        m = M2.predict2(self.spec, theta, self.pr.curves, self.Rm, epochs=tuple(self.epochs), nodes=nodes)
        scores, raw = {}, np.full(len(self.epochs), np.nan)
        for j, k in enumerate(self.epochs):
            scores[k] = self.pr.problems[k].score_model(m[self.pr.index[k], j, self.n_inner:])
            raw[j] = self._radius_term(m[self.index_m[k], j], k)
        z = raw / self.z_ref if self.z_ref is not None else raw
        return scores, raw, z

    def table(self, theta, nodes=M2.FULL_NODES):
        """(5, 5) of A, F, S, B, Z per epoch."""
        scores, _, z = self.per_epoch(theta, nodes=nodes)
        return np.array([[scores[k][0], scores[k][1], scores[k][2], scores[k][4], z[j]]
                         if len(scores[k]) >= 5 else [np.nan] * 5 for j, k in enumerate(self.epochs)])

    def loss(self, theta):
        self.n_eval += 1
        scores, _, z = self.per_epoch(theta)
        total = 0.0
        for j, k in enumerate(self.epochs):
            sc = scores[k]
            if len(sc) < 5:
                return F.FAIL * len(self.epochs)
            a, f, s, n_bad, b = sc
            if not np.all(np.isfinite([a, f, s, b, z[j]])):
                return F.FAIL * len(self.epochs)
            total += a * a + f * f + s * s + b * b + z[j] * z[j] + F.FAIL * n_bad / len(self.pr.problems[k].curves)
        return total

    def report(self, theta, label):
        tab = self.table(theta)
        print(f"  {label}: per-epoch (A, F, S, B, Z) and loss")
        tot = 0.0
        for j, k in enumerate(self.epochs):
            lk = float((tab[j] ** 2).sum())
            tot += lk
            print(f"    z={ANCHOR_Z[k]}: " + " ".join(f"{v:.3f}" for v in tab[j]) + f" -> {lk:.4f}")
        print(f"    total {tot:.4f}")
        return tot, tab


def build(smoke=False, growth=False):
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr, Rm, truth_m, n_inner, rows_m = S0T.build(smoke)
    spec = spec2
    if growth:
        spec = M2.Spec2(theta_names=M2.THETA_NAMES_GROWTH, extended_family=spec2.extended_family,
                        compact_in_kpc=spec2.compact_in_kpc)
        pr = S2F.JointProblem2(spec, meas, data, mask, lmh_bins, F.R_GRID, th_inc, binned=True)
    th_nested = M2.with_levers_theta(M2.nested_theta(th_inc, delay=spec.delay, growth=spec.growth_split), spec)
    sap = SizeAwareProblem(pr, spec, truth_m, Rm, n_inner, lmh_bins, rows_m, th_nested)
    return recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, spec, th_inc, th_nested, pr, sap, Rm, truth_m, n_inner, rows_m


def main(smoke=False, starts_sel=None, merge=False, growth=False, cont=None):
    stage = "stage2" if growth else "stage1"
    tag = ("_growth" if growth else "") + ("_smoke" if smoke else "")
    if merge:
        return merge_starts(stage, tag)
    print(f"{RULE}\nexp78 {stage.upper()} — the fit under A^2+F^2+S^2+B^2+Z^2 (Z = the radius term)"
          + (" with the growth-rate split free" if growth else "") + f"\n{RULE}\n")
    (recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, spec, th_inc, th_nested,
     pr, sap, Rm, truth_m, n_inner, rows_m) = build(smoke, growth)
    n_par = spec.n_theta
    print(f"  PARAMETERS: {n_par} fitted ({'12 + g_split' if growth else '12'}), plus 0 for the size term; "
          f"an observer could vary all {n_par}")
    tab = sap.table(th_nested)
    assert np.allclose(tab[:, 2:], 1.0, atol=2e-2), tab
    print(f"  GATE  the nested incumbent on the measured curves scores 1.000 in S, B, Z "
          f"(max dev {np.max(np.abs(tab[:, 2:] - 1)):.1e})  OK")
    l_null = sap.loss(th_nested)
    sap.report(th_nested, "the null (nested incumbent, measured curves)")
    smoke_tag = "_smoke" if smoke else ""

    def theta_of(path):
        p = Path(str(path).replace(".npz", f"{smoke_tag}.npz"))
        p = p if p.exists() else Path(path)
        return np.asarray(np.load(p, allow_pickle=True)["theta_best"], float)

    bounds = [tuple(b) for b in np.asarray(fz["bounds"], float).tolist()]
    th_base = theta_of(E74 / "stage1_refit_measured.npz")
    if growth:
        jg = spec.index("g_split")
        bounds = bounds + [spec.bounds()[jg]]
        th_s1 = theta_of(OUTDIR / "stage1_fit.npz")
        def at(g):
            return np.r_[th_s1, g]
        starts = [("g0", at(0.0)), ("g-1", at(-1.0)), ("g-2", at(-2.0))]
        sap.report(at(0.0), "the Stage 1 optimum with g = 0 (scored here)")
    else:
        starts = [("baseline", th_base), ("basin", theta_of(E74 / "rebaseline_exp63.npz")),
                  ("exp63", np.asarray(fz["theta_best"], float)),
                  ("nested", M2.clip_to_bounds(spec, th_nested))]
        sap.report(th_base, "THE BASELINE MEAN (exp74's measured optimum), scored here")
    if cont is not None:
        # a continuation of a start that stopped at the evaluation cap (near
        # starts rail into the failure penalty; a basin is settled from its own point)
        src = OUTDIR / f"{stage}_fit{tag}_start_{cont}.npz"
        fb = np.load(src, allow_pickle=True)
        starts = [(f"cont_{cont}", np.asarray(fb["theta"], float))]
        print(f"  continuation of '{cont}' ({int(fb['n_eval'])} evals, loss {float(fb['loss']):.4f})")
    else:
        starts = [s for n in START_ORDER[stage] for s in starts if s[0] == n]
        if starts_sel is not None:
            starts = starts[starts_sel[0]:starts_sel[1] + 1]
    print(f"\n  {len(starts)} start(s): " + ", ".join(n for n, _ in starts))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, p0 in starts:
        t0 = time.time(); sap.n_eval = 0
        l0 = sap.loss(p0)
        r = minimize_loss(sap.loss, p0, method="lbfgsb", bounds=bounds,
                          max_evals=S2F.MAX_EVALS["smoke" if smoke else "full"], fd_step=1e-5)
        dt = time.time() - t0
        rl = S2F.railed(spec, r.x, bounds=bounds)
        extra = f"  g = {r.x[spec.index('g_split')]:+.3f}" if growth else ""
        print(f"  start {name:<9} loss {l0:.6f} -> {r.fun:.6f}  ({sap.n_eval} evals, {dt / 60:.1f} min){extra}"
              f"{'  RAILED: ' + ','.join(rl) if rl else ''}", flush=True)
        per_tab = sap.table(r.x)
        np.savez(OUTDIR / f"{stage}_fit{tag}_start_{name}.npz", name=name, theta0=p0, theta=np.asarray(r.x),
                 loss=float(r.fun), loss0=float(l0), n_eval=sap.n_eval, railed=np.array(rl), per_epoch=per_tab,
                 loss_null=l_null, bounds=np.array(bounds), theta_names=np.array(spec.theta_names),
                 theta_nested=th_nested, theta_baseline=th_base, fit_rows=pr.all_rows,
                 n_fit=np.array([len(pr.rows[k]) for k in pr.epochs]),
                 n_term=np.array([len(rows_m[k]) for k in pr.epochs]),
                 l_s_ref=pr.l_s_ref, l_b_ref=pr.l_b_ref, z_ref=sap.z_ref,
                 extended_family=spec.extended_family, objective="binned+radius",
                 compact_in_kpc=spec.compact_in_kpc, fit_epoch=-1, fit_sample="sane",
                 curves="measured (adopted 2026-09-08)", bins="measured mass", growth_split=growth,
                 n_parameters=n_par)
        print(f"  wrote {OUTDIR / f'{stage}_fit{tag}_start_{name}.npz'}", flush=True)


def merge_starts(stage, tag):
    files = sorted(OUTDIR.glob(f"{stage}_fit{tag}_start_*.npz"))
    assert files, "no start files to merge"
    res = [dict(np.load(f, allow_pickle=True)) for f in files]
    ls = np.array([float(r["loss"]) for r in res])
    best = res[int(np.argmin(ls))]
    names = [str(r["name"]) for r in res]
    growth = bool(best["growth_split"])
    jg = list(best["theta_names"]).index("g_split") if growth else None
    print(f"  {stage}: merged {len(res)} starts: " + ", ".join(
        f"{n} {l:.4f}" + (f" (g {float(r['theta'][jg]):+.2f})" if growth else "") for n, l, r in zip(names, ls, res)))
    print(f"  best: {best['name']} {float(best['loss']):.6f} (null {float(best['loss_null']):.4f}, "
          f"{100 * (1 - float(best['loss']) / float(best['loss_null'])):+.1f}%)"
          + (f", g = {float(best['theta'][jg]):+.3f}" if growth else "")
          + (f"; RAILED {list(best['railed'])}" if len(best["railed"]) else ""))
    n_agree = int((ls < ls.min() + 0.01).sum())
    print(f"  {n_agree} of {len(res)} within 0.01 of the best" + ("  (a single basin)" if n_agree >= 3 else ""))
    out = OUTDIR / f"{stage}_fit{tag}.npz"
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
    main(smoke="--smoke" in a, starts_sel=ss, merge="--merge" in a, growth="--growth" in a,
         cont=(a[a.index("--continue") + 1] if "--continue" in a else None))
