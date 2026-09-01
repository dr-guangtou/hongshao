"""exp73 Block C, the judge — the size-relative refit against the fits it competes with.

Four models on ground none of them chose. The refit (Block C) is compared with
exp63's joint fit (the incumbent objective, fixed kpc), exp72's chi-square refit
(the error-weighted objective) and the nested incumbent (the null), so a reader
sees where each objective put the galaxy's mass and what it gave up.

IN ORDER OF HOW MUCH EACH CAN CHANGE THE VERDICT:

1. **The transfer matrix, in BOTH coordinates.** Each model's loss under this
   experiment's objective (R50 shells, mass-weighted, binned) and under exp63's
   four-term loss (fixed kpc), nested incumbent at the null in each. The diagonal
   is what each fit optimised and must win; the off-diagonal is what it gave up.
2. **The mass-size relation**, directly: median R50 model against truth per
   epoch. This one number decided exp72 (the chi-square made z = 2 galaxies 27
   per cent too big) and it is the user's D1 target.
3. **The tier 2d SIZE GATE** (Block D): the R20/R50/R80 distributions at fixed
   stellar mass and at fixed halo mass, pass/fail with thresholds fixed before
   the fit ran. This is where the mass-weighting's one measured cost -- lower
   sensitivity to a pure size error -- shows if it matters.
4. **The profiles, in plain per cent**, on the merged grid so the inside of
   2 kpc is measured rather than extrapolated: median (model - data)/data at
   fixed radii, every epoch. The cumulative profile, never a shell first.
5. **The standard battery**, `hongshao.qa.evaluate`, unchanged.
6. **exp71's diagnostics** -- the z = 2 halo-mass tilt and the future-growth
   leakage -- which are properties of the model CLASS and should not move.

Run after `fit_re.py`:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u \\
        experiments/exp73_size_relative/eval_re.py [--smoke] [--tables-only]
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
          ROOT / "experiments/exp71_c18_selection_leakage",
          ROOT / "experiments/exp72_error_objective", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import selection as SEL                                  # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
import fit_re as FR                                      # noqa: E402
from hongshao import qa                                  # noqa: E402

OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
RULE, THIN = "=" * 100, "-" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
EXP72_FIT = ROOT / "experiments/exp72_error_objective/outputs/fit_gls.npz"
SEL_NPZ = ROOT / "experiments/exp54_unpinned_amplitude/outputs/selection.npz"
TABLE_KEYS = ("kpc:M(<10)", "kpc:M(<100)", "kpc:M(50-100)")
R_SHOW = (1.0, 2.0, 4.92, 10.25, 52.30, 103.45, 148.22)


def main(smoke=False, tables_only=False):
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp73 Block C, the judge — the size-relative refit against "
          f"the fits it competes with\n{RULE}\n")
    fzr = np.load(OUTDIR / f"fit_re{tag}.npz", allow_pickle=True)
    (spec2, th_inc, th_nested, fz63, curves, truth_m, Rm, lmh0,
     rows, data, lmh, good) = FR.build(smoke)
    thetas = {
        "R50 refit": np.asarray(fzr["theta_best"], float),
        "exp63 joint": np.asarray(fz63["theta_best"], float),
        "chi2 refit": np.asarray(np.load(EXP72_FIT, allow_pickle=True)["theta_best"], float),
        "nested incumbent": th_nested,
    }
    names = list(thetas)
    print(f"  R50 refit: loss {float(fzr['loss_best']):.4f} vs null "
          f"{float(fzr['loss_null']):.4f}, best start '{str(fzr['best_name'])}'")
    print("  " + "  ".join(f"{n}={v:+.4f}" for n, v in zip(spec2.theta_names, thetas["R50 refit"])))
    moved = [(n, a, b) for n, a, b in zip(spec2.theta_names, thetas["exp63 joint"],
                                          thetas["R50 refit"]) if abs(a - b) > 0.05]
    print(f"  moved by > 0.05 from exp63: "
          + (", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in moved) or "none"))

    # predictions on both grids
    cog_m = {n: M2.predict2(spec2, t, curves, Rm, nodes=M2.FULL_NODES) for n, t in thetas.items()}
    cog_k = {n: M2.predict2(spec2, t, curves, F.R_GRID, nodes=M2.FULL_NODES) for n, t in thetas.items()}
    ok = np.ones(len(curves), bool)
    for c in list(cog_m.values()) + list(cog_k.values()):
        ok &= np.isfinite(c).all(axis=(1, 2)) & (c > 0).all(axis=(1, 2))
    d_k = data[good]
    print(f"  {int(ok.sum())} galaxies scored")

    # ---- 1. the transfer matrix in both coordinates ---------------------- #
    print(f"\n{RULE}\n1. THE TRANSFER MATRIX, BOTH COORDINATES — nested incumbent "
          f"at the null in each.\n   The diagonal is what each fit optimised and "
          f"must win; the off-diagonal is what it\n   gave up.\n{RULE}")
    pr = FR.ReProblem(spec2, [curves[i] for i in np.where(ok)[0]], truth_m[ok], Rm,
                      lmh0[ok], th_nested, edges=tuple(fzr["edges"]))
    re_tot = {}
    print(f"\n  under THIS objective (R50 shells, mass-weighted + binned; null = 10.000)")
    print(f"  {'model':<18}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z) + f"{'total':>10}")
    for n in names:
        per, _ = pr.per_epoch(thetas[n], nodes=M2.FULL_NODES)
        re_tot[n] = per
        print(f"  {n:<18}" + "".join(f"{v:>9.4f}" for v in per) + f"{np.nansum(per):>10.4f}")
    e63_tot = {}
    print(f"\n  under exp63's four-term loss (fixed kpc; incumbent terms = 1 each)")
    print(f"  {'model':<18}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z) + f"{'total':>10}")
    keep_k = np.zeros((len(data), 5), bool)
    keep_k[np.where(good)[0][ok]] = True
    # exp63's loss needs the FULL curve list indexed by population row; rebuild it
    # once through fit_gls' builder, which is what exp72 did
    import fit_gls as G
    (_, data_all, lmh_all, curves_all, _, _, _, _, _, keep_all, _, _, _) = G.build(smoke)
    for n in names:
        per = np.full(5, np.nan)
        for k in EPOCHS:
            rws = np.where(keep_k[:, k])[0]
            ref = S2F.Problem2(M2.Spec2(), curves_all, data_all, rws, F.R_GRID, l_s_ref=None,
                               binned=True, lmh=lmh_all[:, k], l_b_ref=None, epoch=k)
            full_k = M2.predict2(spec2, th_nested, [curves_all[i] for i in rws], F.R_GRID,
                                 epochs=(k,), nodes=M2.FULL_NODES)[:, 0, :]
            sc0 = ref.score_model(full_k)
            prb = S2F.Problem2(M2.Spec2(), curves_all, data_all, rws, F.R_GRID, l_s_ref=sc0[2],
                               binned=True, lmh=lmh_all[:, k], l_b_ref=sc0[4], epoch=k)
            mk = M2.predict2(spec2, thetas[n], [curves_all[i] for i in rws], F.R_GRID,
                             epochs=(k,), nodes=M2.FULL_NODES)[:, 0, :]
            a, f_, s_, _, b_ = prb.score_model(mk)
            per[k] = a * a + f_ * f_ + s_ * s_ + b_ * b_
        e63_tot[n] = per
        print(f"  {n:<18}" + "".join(f"{v:>9.4f}" for v in per) + f"{np.nansum(per):>10.4f}")
    print(f"\n{THIN}\n  as a percentage change from exp63 (negative = better)\n{THIN}")
    print(f"  {'':<18}{'this objective':>16}{'exp63 four-term':>18}")
    for n in ("R50 refit", "chi2 refit", "nested incumbent"):
        dg = 100 * (np.nansum(re_tot[n]) / np.nansum(re_tot["exp63 joint"]) - 1)
        de = 100 * (np.nansum(e63_tot[n]) / np.nansum(e63_tot["exp63 joint"]) - 1)
        print(f"  {n:<18}{dg:>+15.1f}%{de:>+17.1f}%")

    # ---- 2. the mass-size relation --------------------------------------- #
    print(f"\n{RULE}\n2. THE MASS-SIZE RELATION — median R50 [kpc], model against "
          f"TNG300, on the merged\n   grid so R50 is measured inside 2 kpc. This "
          f"one number decided exp72.\n{RULE}")
    def r50(c):
        out = np.full(c.shape[:2], np.nan)
        for i in range(c.shape[0]):
            for j in range(5):
                out[i, j] = qa.half_mass_radius(c[i, j], Rm)
        return out
    rt = r50(truth_m[ok])
    print(f"\n  {'':<18}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z))
    print(f"  {'TNG300':<18}" + "".join(f"{v:>9.2f}" for v in np.nanmedian(rt, axis=0)))
    r50_tab = {}
    for n in names:
        rm_ = r50(cog_m[n][ok])
        dex = np.nanmedian(np.log10(rm_ / rt), axis=0)
        r50_tab[n] = dex
        print(f"  {n:<18}" + "".join(f"{v:>9.2f}" for v in np.nanmedian(rm_, axis=0))
              + "   error " + " ".join(f"{100 * (10 ** v - 1):>+4.0f}%" for v in dex))

    # ---- 3 & 5. the size gate and the standard battery -------------------- #
    print(f"\n{RULE}\n3. THE SIZE GATE (Block D) and 5. THE STANDARD BATTERY — "
          f"on the standard grid,\n   so every number is comparable with every "
          f"product ever scored. Median relative\n   bias [%], fitting sample / "
          f"mh-complete.\n{RULE}")
    sn = np.load(SEL_NPZ, allow_pickle=True)
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    lmh_cat = SEL.sample_masses(hs)[np.where(good)[0]][ok]
    complete = np.isfinite(lmh_cat) & (lmh_cat >= sn["cuts"][None, :])
    logms = np.log10(np.clip(d_k[ok][:, 0, -1], 1.0, None))
    (FIGDIR / "qa").mkdir(parents=True, exist_ok=True)
    out = {}
    for n in names:
        print(f"\n  --- {n} ---")
        out[n] = qa.evaluate(cog_k[n][ok], d_k[ok], F.R_GRID, ANCHOR_Z,
                             name=f"exp73_{n.replace(' ', '_')}",
                             figdir=(None if tables_only else FIGDIR / "qa"),
                             figures=not tables_only, verbose=False,
                             bin_by=lmh0[ok], bin_label=r"logM$_h$(z=0.4)",
                             bin_by_ms=logms, ms_label=r"logM$_*$ (total)",
                             halo_mass_epochs=lmh[good][ok])
        qa.print_size_gate(out[n]["size_gate_ms"], ANCHOR_Z, "STELLAR mass")
        qa.print_size_gate(out[n]["size_gate_mh"], ANCHOR_Z, "HALO mass")
    for key in TABLE_KEYS:
        print(f"\n  {key}")
        print(f"  {'model':<18}{'sample':<16}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for n in names:
            t, mm = out[n]["truth"][key], out[n]["model"][key]
            print(f"  {n:<18}{'fitting sample':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[:, j], t[:, j])):>9.1f}%" for j in range(5)))
            print(f"  {'':<18}{'mh-complete':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[complete[:, j], j], t[complete[:, j], j])):>9.1f}%"
                for j in range(5)))
    print(f"\n  tier 3 — profile max|relative| beyond 5 kpc (median)")
    print(f"  {'model':<18}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    for n in names:
        print(f"  {n:<18}" + "".join(f"{np.nanmedian(out[n]['mr_out'][:, j]):>10.4f}" for j in range(5)))

    # ---- 4. the profiles in plain per cent, merged grid ------------------ #
    print(f"\n{RULE}\n4. THE PROFILES — median (model - data)/data [%] on the merged "
          f"grid. The cumulative\n   profile first; inside 2 kpc it is now "
          f"MEASURED (density-rebuilt), not extrapolated.\n{RULE}")
    cols = [int(np.argmin(np.abs(Rm - r))) for r in R_SHOW]
    for n in names:
        print(f"\n  {n}")
        print(f"  {'epoch':>6}" + "".join(f"{Rm[j]:>8.1f}" for j in cols) + "   kpc")
        for k in EPOCHS:
            rel = 100 * np.median((cog_m[n][ok][:, k, :] - truth_m[ok][:, k, :])
                                  / truth_m[ok][:, k, :], axis=0)
            print(f"  {ANCHOR_Z[k]:>6.1f}" + "".join(f"{rel[j]:>+8.1f}" for j in cols))

    # ---- 6. exp71's diagnostics ------------------------------------------ #
    print(f"\n{RULE}\n6. exp71's DIAGNOSTICS — the z=2 halo-mass tilt of M*(<103 "
          f"kpc) (fitting | mh-complete)\n   and the future-growth leakage "
          f"(partial rho | dy/dG). Class properties; should not move.\n{RULE}")
    growth = lmh_cat[:, 0][:, None] - lmh_cat
    c103 = int(np.argmin(np.abs(F.R_GRID - 103.45)))
    print(f"\n  {'model':<18}" + "".join(f"{f'z={z}':>18}" for z in ANCHOR_Z))
    for n in names:
        y = (np.log10(np.clip(cog_k[n][ok][:, :, c103], 1, None))
             - np.log10(np.clip(d_k[ok][:, :, c103], 1, None)))
        allm = np.ones(len(y), bool)
        row = [f"{SEL.tilt(y[:, k], lmh_cat[:, k], allm, seed=k)[0]:+.3f}|"
               f"{SEL.tilt(y[:, k], lmh_cat[:, k], complete[:, k], seed=100 + k)[0]:+.3f}"
               for k in EPOCHS]
        print(f"  {n:<18}" + "".join(f"{v:>18}" for v in row))
    print(f"\n  {'model':<18}" + "".join(f"{f'z={z}':>18}" for z in ANCHOR_Z[1:]) + "   leakage")
    for n in names:
        y = (np.log10(np.clip(cog_k[n][ok][:, :, c103], 1, None))
             - np.log10(np.clip(d_k[ok][:, :, c103], 1, None)))
        row = []
        for k in range(1, 5):
            r_, s_, _ = SEL.partial_growth(y[:, k], lmh_cat[:, k], growth[:, k], np.ones(len(y), bool))
            row.append(f"{r_:+.3f}|{s_:+.3f}")
        print(f"  {n:<18}" + "".join(f"{v:>18}" for v in row))

    np.savez(OUTDIR / f"eval_re{tag}.npz", models=np.array(names),
             **{f"re_{n.replace(' ', '_')}": re_tot[n] for n in names},
             **{f"e63_{n.replace(' ', '_')}": e63_tot[n] for n in names},
             **{f"r50dex_{n.replace(' ', '_')}": r50_tab[n] for n in names})
    print(f"\nwrote {OUTDIR / f'eval_re{tag}.npz'}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, tables_only="--tables-only" in sys.argv)
