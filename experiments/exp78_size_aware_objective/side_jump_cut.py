"""exp78 side quest (the user, 2026-09-09) — do the few galaxies with a SUDDEN
stellar-mass increase between adjacent epochs move the fit?

Different models fail at the same few galaxies (`qa_cases_*`: the worst two
have M*(<100 kpc) at z = 2 three to ten times below their z = 1.5 value).
THE FITTING SAMPLE rule keeps them: its jump criterion removes only a rise
of M*(<100) above 1.0 dex between adjacent epochs (two galaxies). exp77
decided to keep them until a snapshot check; this measures what they cost.

THE STRICTER CUT (`--level L`, default 0.6 dex): a galaxy is removed at every
epoch if its M*(<100 kpc) OR M*(<30 kpc) rises by more than L dex between
any two adjacent epochs (a factor 4 in 1.1-2.1 Gyr). At 0.6 dex that is 45
galaxies (1.9 per cent of the fitting sample); at 0.5 dex, 109 (4.6 per
cent). The cut is a SAMPLE question, not a model one; the model never sees it.

Stage A, frozen (`--frozen`): the baseline mean scored on the full fitting
sample and on the stricter samples UNDER THE SAME (full-sample) REFERENCES,
so the terms are comparable galaxy for galaxy: the per-epoch A, F, S, B, the
share of the per-galaxy terms the removed galaxies carry, the profile medians
and the size gate on each sample.

Stage B, the refit (`--fit --starts k:k`, then `--merge`): exp63's objective
(A^2+F^2+S^2+B^2, no size term) on the stricter sample with its own
references, from the baseline and from the 14.63 basin; then `--judge`: the
refit against the baseline on BOTH samples (loss under the adopted
references, the profile, the future-dependence gate, the size gate).

Run:
    HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. nohup uv run python -u \\
        experiments/exp78_size_aware_objective/side_jump_cut.py --frozen > outputs/side_jump_frozen.log 2>&1 &
    ... --fit --starts 0:0 (and 1:1), --merge, --judge
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
import selection as SEL                                  # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
from hongshao import qa                                  # noqa: E402
from hongshao.fitting import minimize_loss               # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RB = _by_path("exp74_rebaseline", ROOT / "experiments/exp74_c19_history_leak/rebaseline.py")

RULE = "=" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR = HERE / "outputs"
E74 = ROOT / "experiments/exp74_c19_history_leak/outputs"
SEL_NPZ = ROOT / "experiments/exp54_unpinned_amplitude/outputs/selection.npz"
R_SHOW = (2.0, 10.25, 52.30, 103.45)
R_LEAK = 103.45
STARTS = ["baseline", "basin"]


def jump_flag(data, level):
    """(n,) True where M*(<100) or M*(<30) rises by more than `level` dex
    between adjacent epochs (later minus earlier, epochs ordered z = 0.4 ... 2)."""
    L = np.log10(np.clip(np.asarray(data, float), 1.0, None))
    i30 = int(np.argmin(np.abs(F.R_GRID - 30.0)))
    j100 = np.max(L[:, :4, F.I100] - L[:, 1:, F.I100], axis=1)
    j30 = np.max(L[:, :4, i30] - L[:, 1:, i30], axis=1)
    return (j100 > level) | (j30 > level), j100, j30


def build(level, smoke=False):
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr = RB.build(smoke)
    flag, j100, j30 = jump_flag(data, level)
    in_fit = mask.all(1)
    strict = mask & ~flag[:, None]
    print(f"  STRICTER CUT at {level} dex: {int((in_fit & flag).sum())} of {int(in_fit.sum())} fitting-sample galaxies "
          f"removed ({100 * (in_fit & flag).mean() / in_fit.mean():.1f}%); by interval (z 0.7<-0.4 ... 2.0<-1.5): "
          + "/".join(str(int((in_fit & flag & (np.argmax(np.log10(np.clip(data[:, :4, F.I100], 1, None))
                                                           - np.log10(np.clip(data[:, 1:, F.I100], 1, None)), axis=1) == k)).sum()))
                     for k in range(4)))
    th_base = np.asarray(np.load(E74 / "stage1_refit_measured.npz", allow_pickle=True)["theta_best"], float)
    th_basin = np.asarray(np.load(E74 / "rebaseline_exp63.npz", allow_pickle=True)["theta_best"], float)
    return recs, data, mask, strict, flag, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr, th_base, th_basin


def subset_problem(pr, spec2, meas, data, rows, lmh_bins, k):
    """exp63's per-epoch problem on `rows` with the FULL-sample references."""
    p = pr.problems[k]
    return S2F.Problem2(spec2, meas, data, rows, F.R_GRID, l_s_ref=p.l_s_ref, binned=True,
                        lmh=lmh_bins[:, k], l_b_ref=p.l_b_ref, epoch=k)


def score_on(pr, spec2, meas, data, lmh_bins, theta, mask):
    """(5, 4) A, F, S, B per epoch of `theta` on the galaxies of `mask`, under the adopted references."""
    rows = {k: np.where(mask[:, k] & np.isfinite(data[:, k]).all(1) & (data[:, k] > 0).all(1))[0] for k in EPOCHS}
    allr = np.unique(np.concatenate(list(rows.values())))
    m = M2.predict2(spec2, theta, [meas[i] for i in allr], F.R_GRID, epochs=EPOCHS, nodes=M2.FULL_NODES)
    pos = {r: i for i, r in enumerate(allr)}
    out = np.full((5, 4), np.nan)
    for j, k in enumerate(EPOCHS):
        sub = subset_problem(pr, spec2, meas, data, rows[k], lmh_bins, k)
        sc = sub.score_model(m[[pos[r] for r in rows[k]], j])
        out[j] = [sc[0], sc[1], sc[2], sc[4]]
    return out


def qa_tables(pred, data, mask_all, lmh_cat, lmh_bins, complete, label):
    """The profile medians, the leak and the size gate on the galaxies of `mask_all` (n,) bool."""
    ir = [int(np.argmin(np.abs(F.R_GRID - r))) for r in R_SHOW]
    c = int(np.argmin(np.abs(F.R_GRID - R_LEAK)))
    print(f"\n  --- {label}: {int(mask_all.sum())} galaxies ---")
    print(f"  profile, median (model-data)/data [%], fitting | mh-complete")
    print(f"  {'epoch':>6}" + "".join(f"{f'M(<{F.R_GRID[i]:.0f})':>22}" for i in ir))
    for k in EPOCHS:
        m_c = mask_all & complete[:, k]
        print(f"  {ANCHOR_Z[k]:>6}" + "".join(
            f"{100 * np.nanmedian((pred[mask_all, k, i] - data[mask_all, k, i]) / data[mask_all, k, i]):>+10.1f} |"
            f"{100 * np.nanmedian((pred[m_c, k, i] - data[m_c, k, i]) / data[m_c, k, i]):>+9.1f}" for i in ir))
    y_tru = np.log10(np.clip(data[:, :, c], 1.0, None))
    growth = lmh_cat[:, 0][:, None] - lmh_cat
    cells = []
    for k in range(1, 5):
        y = np.log10(np.clip(pred[:, k, c], 1.0, None)) - y_tru[:, k]
        rho, sl, n = SEL.partial_growth(y, lmh_cat[:, k], growth[:, k], mask_all & np.isfinite(lmh_cat[:, k]))
        cells.append(f"{sl:>+8.3f}")
    print(f"  future-dependence dy/dG at fixed mass, M(<{R_LEAK:g}), z=0.7..2: " + "".join(cells))
    out = qa.evaluate(pred[mask_all], data[mask_all], F.R_GRID, ANCHOR_Z, name=label, figdir=None, figures=False,
                      verbose=False, bin_by=lmh_bins[mask_all][:, 0], halo_mass_epochs=lmh_cat[mask_all])
    g, n_ok, n, n_off, n_wid = out["size_gate_ms"]
    print(f"  size gate: offset {n_off} of {n}, width {n_wid} of {n}; R50 offset z=1.5/2 "
          f"{g[('R50', 3)]['offset']:+.3f}/{g[('R50', 4)]['offset']:+.3f}; R80 z=2 {g[('R80', 4)]['offset']:+.3f}")
    for key in ("kpc:M(<10)", "kpc:M(<100)"):
        t, mm = out["truth"][key], out["model"][key]
        print(f"  {key:<12} median bias fitting: " + "".join(f"{100 * np.nanmedian(qa.relerr(mm[:, j], t[:, j])):>8.1f}%" for j in range(5)))
    return out


def main(level=0.6, frozen=False, fit=False, starts_sel=None, merge=False, judge=False, smoke=False, use=None):
    tag = f"_l{level:g}" + ("_smoke" if smoke else "")
    print(f"{RULE}\nexp78 side quest — sudden stellar-mass increases: the stricter cut at {level} dex\n{RULE}\n")
    (recs, data, mask, strict, flag, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr, th_base, th_basin) = build(level, smoke)
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    lmh_cat = SEL.sample_masses(hs)[np.array([h.row for h in recs])]
    sn = np.load(SEL_NPZ, allow_pickle=True)
    complete = np.isfinite(lmh_cat) & (lmh_cat >= sn["cuts"][None, :])
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    full_all, strict_all = mask.all(1) & good, strict.all(1) & good
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if frozen:
        print(f"\n{RULE}\nA. FROZEN: the baseline mean on the full sample, the stricter sample and the removed galaxies, "
              f"same references\n{RULE}")
        tabs = {}
        for lab, mk in (("full", mask), ("stricter", strict), ("removed only", mask & flag[:, None])):
            tabs[lab] = score_on(pr, spec2, meas, data, lmh_bins, th_base, mk)
        print(f"  {'sample':<14}{'epoch':>6}{'A':>8}{'F':>8}{'S':>8}{'B':>8}{'A2+F2+S2+B2':>13}")
        for lab, t in tabs.items():
            for j, k in enumerate(EPOCHS):
                print(f"  {lab if j == 0 else '':<14}{ANCHOR_Z[k]:>6}" + "".join(f"{v:>8.3f}" for v in t[j]) + f"{(t[j] ** 2).sum():>13.3f}")
            print(f"  {'':<14}{'total':>6}{'':>32}{(t ** 2).sum():>13.3f}")
        n_f = int((mask.all(1) & flag).sum()); n_all = int(mask.all(1).sum())
        print(f"\n  the removed galaxies' share of the per-galaxy terms (A^2 and S are means over galaxies: "
              f"share = n_removed x value_removed / (n_all x value_all)); {n_f} of {n_all} galaxies = {100 * n_f / n_all:.1f}%")
        for j, k in enumerate(EPOCHS):
            a_share = n_f * tabs["removed only"][j, 0] ** 2 / (n_all * tabs["full"][j, 0] ** 2)
            s_share = n_f * tabs["removed only"][j, 2] / (n_all * tabs["full"][j, 2])
            print(f"    z={ANCHOR_Z[k]}: A^2 share {100 * a_share:5.1f}%   S share {100 * s_share:5.1f}%")
        pred = M2.predict2(spec2, th_base, meas, F.R_GRID, epochs=EPOCHS, nodes=M2.FULL_NODES)
        for lab, mk in (("full", full_all), ("stricter", strict_all)):
            qa_tables(pred, data, mk, lmh_cat, lmh_bins, complete, f"baseline on the {lab} sample")
        np.savez(OUTDIR / f"side_jump_frozen{tag}.npz", level=level, flag=flag, tabs_keys=np.array(list(tabs)),
                 tabs=np.array(list(tabs.values())))
        return

    if fit:
        pr_s = S2F.JointProblem2(spec2, meas, data, strict, lmh_bins, F.R_GRID, th_inc, binned=True)
        th_nested = M2.with_levers_theta(M2.nested_theta(th_inc, delay=spec2.delay, growth=spec2.growth_split), spec2)
        l_null = pr_s.loss(th_nested)
        print(f"  the stricter sample's own references; null loss {l_null:.4f}; rows per epoch "
              + "/".join(str(len(pr_s.rows[k])) for k in EPOCHS))
        bounds = [tuple(b) for b in np.asarray(fz["bounds"], float).tolist()]
        starts = [("baseline", th_base), ("basin", th_basin)]
        if starts_sel is not None:
            starts = starts[starts_sel[0]:starts_sel[1] + 1]
        for name, p0 in starts:
            t0 = time.time(); pr_s.n_eval = 0
            l0 = pr_s.loss(p0)
            r = minimize_loss(pr_s.loss, p0, method="lbfgsb", bounds=bounds,
                              max_evals=S2F.MAX_EVALS["smoke" if smoke else "full"], fd_step=1e-5)
            rl = S2F.railed(spec2, r.x, bounds=bounds)
            print(f"  start {name:<9} loss {l0:.6f} -> {r.fun:.6f}  ({pr_s.n_eval} evals, {(time.time() - t0) / 60:.1f} min)"
                  f"{'  RAILED: ' + ','.join(rl) if rl else ''}", flush=True)
            per = pr_s.per_epoch(r.x, nodes=M2.FULL_NODES)
            np.savez(OUTDIR / f"side_jump_fit{tag}_start_{name}.npz", name=name, theta0=p0, theta=np.asarray(r.x),
                     loss=float(r.fun), loss0=float(l0), n_eval=pr_s.n_eval, railed=np.array(rl), loss_null=l_null,
                     per_epoch=np.array([[per[k][0], per[k][1], per[k][2], per[k][4]] for k in EPOCHS]),
                     theta_names=np.array(spec2.theta_names), level=level, fit_rows=pr_s.all_rows, bounds=np.array(bounds))
        return

    if merge:
        files = [OUTDIR / f"side_jump_fit{tag}_start_{n}.npz" for n in STARTS]
        res = [dict(np.load(f, allow_pickle=True)) for f in files if f.exists()]
        ls = np.array([float(r["loss"]) for r in res]); best = res[int(np.argmin(ls))]
        print("  merged: " + ", ".join(f"{str(r['name'])} {l:.4f} ({int(r['n_eval'])} evals)" for r, l in zip(res, ls))
              + f"; best {str(best['name'])} {float(best['loss']):.4f} (null {float(best['loss_null']):.4f})")
        np.savez(OUTDIR / f"side_jump_fit{tag}.npz", theta_best=best["theta"], loss_best=float(best["loss"]),
                 best_name=str(best["name"]), names=np.array([str(r["name"]) for r in res]), losses=ls,
                 thetas=np.array([r["theta"] for r in res]), theta_names=best["theta_names"], level=level,
                 loss_null=float(best["loss_null"]), railed_best=best["railed"])
        return

    if judge:
        fs = np.load(OUTDIR / f"side_jump_fit{tag}.npz", allow_pickle=True)
        th_s = np.asarray(fs["theta_best"], float)
        if use is not None:
            # judge one named start rather than the merged best
            fu = np.load(OUTDIR / f"side_jump_fit{tag}_start_{use}.npz", allow_pickle=True)
            th_s = np.asarray(fu["theta"], float)
            print(f"  judging the '{use}' start: loss {float(fu['loss']):.4f} ({int(fu['n_eval'])} evals)")
        names = list(spec2.theta_names)
        print(f"\n{RULE}\nB. THE REFIT on the stricter sample against the baseline\n{RULE}")
        print("  starts: " + ", ".join(f"{n} {l:.4f}" for n, l in zip(fs["names"], fs["losses"]))
              + f"; best {str(fs['best_name'])} {float(fs['loss_best']):.4f} (null {float(fs['loss_null']):.4f})"
              + (f"; RAILED {list(fs['railed_best'])}" if len(fs["railed_best"]) else ""))
        moved = [(n, a, b) for n, a, b in zip(names, th_base, th_s) if abs(a - b) > 0.02]
        print("  moved by > 0.02 from the baseline: " + (", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in moved) or "none"))
        print("  all parameters, baseline -> refit: " + ", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in zip(names, th_base, th_s)))
        print(f"\n  the loss under the ADOPTED (full-sample) references, per epoch, on each sample:")
        print(f"  {'model':<12}{'sample':<10}{'epoch':>6}{'A':>8}{'F':>8}{'S':>8}{'B':>8}{'A2+F2+S2+B2':>13}")
        for lab, th in (("baseline", th_base), ("refit", th_s)):
            for slab, mk in (("full", mask), ("stricter", strict)):
                t = score_on(pr, spec2, meas, data, lmh_bins, th, mk)
                for j, k in enumerate(EPOCHS):
                    print(f"  {lab if j == 0 else '':<12}{slab if j == 0 else '':<10}{ANCHOR_Z[k]:>6}"
                          + "".join(f"{v:>8.3f}" for v in t[j]) + f"{(t[j] ** 2).sum():>13.3f}")
                print(f"  {'':<12}{'':<10}{'total':>6}{'':>32}{(t ** 2).sum():>13.3f}")
        for lab, th in (("baseline", th_base), ("refit", th_s)):
            pred = M2.predict2(spec2, th, meas, F.R_GRID, epochs=EPOCHS, nodes=M2.FULL_NODES)
            for slab, mk in (("full", full_all), ("stricter", strict_all)):
                qa_tables(pred, data, mk, lmh_cat, lmh_bins, complete, f"{lab} on the {slab} sample")
        np.savez(OUTDIR / f"side_jump_judge{tag}{'_' + use if use else ''}.npz", theta_baseline=th_base, theta_refit=th_s,
                 level=level, flag=flag)


if __name__ == "__main__":
    a = sys.argv
    ss = None
    if "--starts" in a:
        lo, hi = a[a.index("--starts") + 1].split(":")
        ss = (int(lo), int(hi))
    main(level=float(a[a.index("--level") + 1]) if "--level" in a else 0.6, frozen="--frozen" in a, fit="--fit" in a,
         starts_sel=ss, merge="--merge" in a, judge="--judge" in a, smoke="--smoke" in a,
         use=(a[a.index("--use") + 1] if "--use" in a else None))
