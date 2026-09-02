"""exp73 Block A2, closed — THE TRUE PER-EPOCH CEILING under the R50 objective.

A2 asked whether the size-relative coordinate narrows the gap between one
shared law and five epoch-specialised fits, and answered 1.28 -> 1.07 with a
caveat it called "not small": the five per-epoch comparators had been optimised
under exp63's FIXED-KPC objective, so under the R50 objective they were
mis-specialised and the shared theta beat them at z >= 1.0 — a gap below 1,
which is a comparator artefact, not a ceiling.

`fit_re.py --epoch k` (the user's decision 2, 2026-09-02) refits each epoch
ALONE under the R50 objective with the same freedom exp63's per-epoch fits had
(the four time exponents frozen at the same values). This script is the judge:

1. the transfer matrix — every theta at every epoch under the R50 objective;
2. the gap, joint / own, with the R50 per-epoch fits as the comparator (the
   true ceiling) next to A2's fixed-kpc comparator (the number A2 reported);
3. what a per-epoch fit does to the population at its own epoch — R50 and the
   profile at 1, 2, 10, 100 kpc — because exp61 found per-epoch fits reach their
   loss by overshooting the population, and a ceiling that overshoots is not a
   target ([[per-epoch-ceiling-does-not-transfer]]);
4. the free parameters against log10(1+z), R50 per-epoch next to fixed-kpc
   per-epoch — whether the non-parametric time dependence changes with the
   objective;
5. exp63's four-term loss at each fit's own epoch — what the R50 per-epoch fit
   gave up under the objective the programme has always used.

Run after the five `fit_re.py --epoch k` runs:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u experiments/exp73_size_relative/ceiling_re.py [--smoke]
"""
from __future__ import annotations

import sys
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
import fit_re as FR                                      # noqa: E402
from risk_test import trend, TREND_EPOCHS, E63           # noqa: E402

RULE, THIN = "=" * 100, "-" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR = HERE / "outputs"
R_SHOW = (1.0, 2.0, 10.25, 103.45)


def load_models(spec2, th_nested, fz63, tag):
    """name -> (spec, theta); per-epoch files that do not exist are skipped
    with a note (the smoke run has only what was smoke-fitted)."""
    models = {"R50 joint": (spec2, None), "exp63 joint": (spec2, np.asarray(fz63["theta_best"], float))}
    fits = {}
    f = OUTDIR / f"fit_re{tag}.npz"
    if f.exists():
        fzr = np.load(f, allow_pickle=True)
        models["R50 joint"] = (spec2, np.asarray(fzr["theta_best"], float))
        fits["R50 joint"] = fzr
    else:
        print(f"  (no {f.name}; the R50 joint row falls back to exp63's theta)")
        models["R50 joint"] = models["exp63 joint"]
    for k in EPOCHS:
        f = OUTDIR / f"fit_re_e{k}{tag}.npz"
        if f.exists():
            fz = np.load(f, allow_pickle=True)
            models[f"R50 e{k}"] = (spec2, np.asarray(fz["theta_best"], float))
            fits[f"R50 e{k}"] = fz
        else:
            print(f"  (no {f.name}; R50 e{k} skipped)")
        fe = np.load(E63 / f"stage2_fit_frozen_binned_kpc_sane_e{k}.npz", allow_pickle=True)
        models[f"exp63 e{k}"] = (S2F.spec_from_fit(fe), np.asarray(fe["theta_best"], float))
        fits[f"exp63 e{k}"] = fe
    models["nested incumbent"] = (spec2, th_nested)
    return models, fits


def main(smoke=False):
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp73 A2 closed — the TRUE per-epoch ceiling under the R50 "
          f"objective\n{RULE}\n")
    (spec2, th_inc, th_nested, fz63, curves, truth_m, Rm, lmh0,
     rows, data, lmh, good) = FR.build(smoke)
    models, fits = load_models(spec2, th_nested, fz63, tag)
    names = list(models)
    pnames = list(spec2.theta_names)

    print(f"\n  the five single-epoch R50 fits (each: best start, loss vs null 2.000, "
          f"vs exp63's per-epoch theta scored under this objective)")
    for k in EPOCHS:
        n = f"R50 e{k}"
        if n not in fits:
            continue
        fz = fits[n]
        l0 = {str(a): float(b) for a, b in zip(fz["names"], fz["losses0"])}
        ls = {str(a): float(b) for a, b in zip(fz["names"], fz["losses"])}
        agree = int((np.asarray(fz["losses"], float) < float(fz["loss_best"]) + 0.01).sum())
        print(f"  z={ANCHOR_Z[k]}: best '{str(fz['best_name'])}' {float(fz['loss_best']):.4f}; "
              f"exp63's e{k} theta scored {l0.get(f'exp63_e{k}', np.nan):.4f} -> its start "
              f"reached {ls.get(f'exp63_e{k}', np.nan):.4f}; {agree} of {len(ls)} starts "
              f"within 0.01 of the best; frozen "
              + ", ".join(f"{a}={float(b):+.3f}" for a, b in zip(fz["frozen_names"], fz["frozen_values"])))

    # predictions on the merged grid; one `ok` for every model
    cog = {n: M2.predict2(sp, th, curves, Rm, nodes=M2.FULL_NODES) for n, (sp, th) in models.items()}
    ok = np.isfinite(truth_m).all(axis=(1, 2))
    for c in cog.values():
        ok &= np.isfinite(c).all(axis=(1, 2)) & (c > 0).all(axis=(1, 2))
    print(f"\n  {int(ok.sum())} galaxies scored")
    sel = np.where(ok)[0]
    pr = FR.ReProblem(spec2, [curves[i] for i in sel], truth_m[ok], Rm, lmh0[ok], th_nested)

    # ---- 1. transfer matrix --------------------------------------------- #
    print(f"\n{RULE}\n1. THE TRANSFER MATRIX under the R50 objective — every theta "
          f"at every epoch; the nested\n   incumbent is 2.000 in each column. A "
          f"per-epoch theta's own column is its ceiling.\n{RULE}")
    print(f"  {'model':<18}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z) + f"{'total':>10}")
    L = {}
    for n in names:
        per, _ = pr.per_epoch(models[n][1], nodes=M2.FULL_NODES)
        L[n] = per
        mark = "  <- own epoch " + n[-2:] if n[-3:-1] == " e" else ""
        print(f"  {n:<18}" + "".join(f"{v:>9.4f}" for v in per) + f"{np.nansum(per):>10.4f}{mark}")

    # ---- 2. the gap ------------------------------------------------------ #
    print(f"\n{RULE}\n2. THE GAP = loss(one shared theta) / loss(that epoch's own "
          f"theta), under the R50 objective.\n   A2 reported the first row (its "
          f"comparator was optimised on fixed kpc); the second row is\n   the true "
          f"ceiling. 1.00 = sharing is free; below 1 = the comparator is "
          f"mis-specialised.\n{RULE}")

    def gap(joint, own):
        return np.array([L[joint][k] / L[own(k)][k] if own(k) in L else np.nan for k in EPOCHS])

    rows_ = [("A2: exp63 joint / exp63 e_k (fixed-kpc comparator)", gap("exp63 joint", lambda k: f"exp63 e{k}")),
             ("TRUE: R50 joint / R50 e_k", gap("R50 joint", lambda k: f"R50 e{k}")),
             ("exp63 joint / R50 e_k", gap("exp63 joint", lambda k: f"R50 e{k}")),
             ("R50 e_k / exp63 e_k (what the objective bought per epoch)",
              np.array([L[f"R50 e{k}"][k] / L[f"exp63 e{k}"][k] if f"R50 e{k}" in L else np.nan for k in EPOCHS]))]
    print(f"  {'':<58}" + "".join(f"{f'z={z}':>8}" for z in ANCHOR_Z) + f"{'z<=1.5':>9}")
    for lab, g in rows_:
        print(f"  {lab:<58}" + "".join(f"{v:>8.3f}" for v in g) + f"{np.nanmean(g[:4]):>9.3f}")
    g_true = rows_[1][1]
    print(f"\n  READ: the cost of sharing under the R50 objective is "
          f"{100 * (np.nanmean(g_true[:4]) - 1):+.1f}% at z<=1.5 (true) against "
          f"{100 * (np.nanmean(rows_[0][1][:4]) - 1):+.1f}% (A2's comparator); "
          f"under exp63's fixed-kpc objective A2 measured +28%.")

    # ---- 3. the population at the own epoch ------------------------------ #
    print(f"\n{RULE}\n3. WHAT A PER-EPOCH FIT DOES TO THE POPULATION AT ITS OWN EPOCH — "
          f"median model/truth - 1 for R50,\n   and median (model - data)/data at "
          f"1, 2, 10, 103 kpc on the merged grid. exp61: per-epoch fits reach\n   "
          f"their loss by overshooting the population; a ceiling that overshoots is "
          f"not a target.\n{RULE}")
    r50_t = C.size_radius(truth_m[ok], Rm, 0.5)
    ir = [int(np.argmin(np.abs(Rm - r))) for r in R_SHOW]
    print(f"  {'model':<18}{'epoch':>6}{'R50':>8}" + "".join(f"{f'M(<{Rm[i]:.0f})':>10}" for i in ir))
    pop = {}
    for k in EPOCHS:
        for n in (f"R50 e{k}", f"exp63 e{k}", "R50 joint", "exp63 joint"):
            if n not in cog:
                continue
            m = cog[n][ok]
            r50_m = C.size_radius(m, Rm, 0.5)
            e_r50 = float(np.nanmedian(r50_m[:, k] / r50_t[:, k]) - 1)
            prof = [float(np.nanmedian((m[:, k, i] - truth_m[ok][:, k, i]) / truth_m[ok][:, k, i])) for i in ir]
            pop[(n, k)] = (e_r50, prof)
            print(f"  {n:<18}{ANCHOR_Z[k]:>6}{100 * e_r50:>+7.1f}%" + "".join(f"{100 * v:>+9.1f}%" for v in prof))
        print()

    # ---- 4. the free parameters against log10(1+z) ----------------------- #
    fe0 = fits["exp63 e0"]
    frozen = {str(a): float(b) for a, b in zip(fe0["frozen_names"], fe0["frozen_values"])}
    free = [p for p in pnames if p not in frozen]
    print(f"{RULE}\n4. THE FREE PARAMETERS AGAINST log10(1+z) — R50 per-epoch next to "
          f"fixed-kpc per-epoch.\n   Frozen in both: {', '.join(f'{a}={b:+.3f}' for a, b in frozen.items())}. "
          f"`b_c` = 0 so `log_f_c` IS the compact size in log10 kpc.\n   Trend fitted "
          f"on z<=1.5; `resid/range` = scatter about a line / span.\n{RULE}")
    print(f"  {'param':<9}{'fit':<7}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z)
          + f"{'slope':>9}{'resid/rng':>11}")
    trends = {}
    for p in free:
        j = pnames.index(p)
        for lab, pre in (("kpc", "exp63 e"), ("R50", "R50 e")):
            vals = np.array([models[f"{pre}{k}"][1][j] if f"{pre}{k}" in models else np.nan for k in EPOCHS])
            sl, r1, rng, _ = trend(vals, ANCHOR_Z)
            trends[(p, lab)] = (vals, sl, r1, rng)
            print(f"  {p:<9}{lab:<7}" + "".join(f"{v:>9.3f}" for v in vals)
                  + f"{sl:>9.3f}{(r1 / rng if rng else np.nan):>11.3f}")

    # ---- 5. exp63's four-term loss at the own epoch ---------------------- #
    print(f"\n{RULE}\n5. exp63's FOUR-TERM LOSS (fixed kpc, incumbent terms = 1 each) at "
          f"each fit's OWN epoch —\n   what the R50 per-epoch fit gives up under the "
          f"programme's objective.\n{RULE}")
    (_, data_all, lmh_all, curves_all, _, _, _, _, _, keep_all, _, _, _) = G.build(smoke)
    keep_k = np.zeros((len(data_all), 5), bool)
    keep_k[np.where(good)[0][ok]] = True
    print(f"  {'epoch':>6}{'R50 e_k':>10}{'exp63 e_k':>11}{'R50 joint':>11}{'exp63 joint':>13}")
    e63 = {}
    for k in EPOCHS:
        rws = np.where(keep_k[:, k])[0]
        cu = [curves_all[i] for i in rws]
        ref = S2F.Problem2(M2.Spec2(), curves_all, data_all, rws, F.R_GRID, l_s_ref=None,
                           binned=True, lmh=lmh_all[:, k], l_b_ref=None, epoch=k)
        sc0 = ref.score_model(M2.predict2(spec2, th_nested, cu, F.R_GRID, epochs=(k,),
                                          nodes=M2.FULL_NODES)[:, 0, :])
        prb = S2F.Problem2(M2.Spec2(), curves_all, data_all, rws, F.R_GRID, l_s_ref=sc0[2],
                           binned=True, lmh=lmh_all[:, k], l_b_ref=sc0[4], epoch=k)
        out = []
        for n in (f"R50 e{k}", f"exp63 e{k}", "R50 joint", "exp63 joint"):
            if n not in models:
                out.append(np.nan)
                continue
            sp, th = models[n]
            a, f_, s_, _, b_ = prb.score_model(M2.predict2(sp, th, cu, F.R_GRID, epochs=(k,),
                                                           nodes=M2.FULL_NODES)[:, 0, :])
            out.append(a * a + f_ * f_ + s_ * s_ + b_ * b_)
        e63[k] = out
        print(f"  {ANCHOR_Z[k]:>6}" + "".join(f"{v:>11.3f}" for v in out[:2]) + f"{out[2]:>11.3f}{out[3]:>13.3f}")

    np.savez(OUTDIR / f"ceiling_re{tag}.npz", models=np.array(names),
             re_loss=np.array([L[n] for n in names]),
             gap_a2=rows_[0][1], gap_true=rows_[1][1], gap_cross=rows_[2][1], gap_bought=rows_[3][1],
             pop_keys=np.array([f"{n}|{k}" for (n, k) in pop]),
             pop_r50=np.array([v[0] for v in pop.values()]),
             pop_prof=np.array([v[1] for v in pop.values()]), r_show=np.array(R_SHOW),
             trend_keys=np.array([f"{p}|{lab}" for (p, lab) in trends]),
             trend_vals=np.array([v[0] for v in trends.values()]),
             trend_slope=np.array([v[1] for v in trends.values()]),
             e63_loss=np.array([e63[k] for k in EPOCHS]))
    print(f"\nwrote {OUTDIR / f'ceiling_re{tag}.npz'}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
