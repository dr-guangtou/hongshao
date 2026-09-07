"""exp74 Stage 1, the judge — exp63 on the official curves against the refit on
the pre-epoch curves. THE CURVES ARE THE ONLY CHANGE, so every difference
below is what the model's knowledge of the halo's future was worth.

  1. the fit: starts, losses, which parameters moved;
  2. the 2 x 2 of theta against curves under exp63's four-term loss (the
     off-diagonals are the frozen-theta costs in both directions);
  3. the profile: median residual at 2 / 10 / 52 / 103 kpc, fitting sample |
     mh-complete;
  4. the leak: dy/dG at fixed catalog mass at 103 kpc (truth +0.003 at z = 2);
  5. the z = 2 halo-mass tilt, fitting sample | mh-complete (C18's numbers);
  6. the size gate and the standard battery, with figures.

Run after `stage1_refit.py --merge`:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u experiments/exp74_c19_history_leak/stage1_eval.py [--smoke] [--tables-only]
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
          ROOT / "experiments/exp73_size_relative", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import selection as SEL                                  # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
import coordinate as C                                   # noqa: E402
import stage1_refit as S1                                # noqa: E402
from stage0_frozen import tilt, R_SHOW, R_LEAK           # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE, THIN = "=" * 100, "-" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
SEL_NPZ = ROOT / "experiments/exp54_unpinned_amplitude/outputs/selection.npz"
TABLE_KEYS = ("kpc:M(<10)", "kpc:M(<100)")


def main(smoke=False, tables_only=False):
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp74 Stage 1, the judge — exp63 (official curves) against the refit (pre-epoch curves)"
          f"\n{RULE}\n")
    recs, data, mask, lmh, curves, pre, fz63, spec2, th_inc, pr = S1.build(smoke)
    fzr = np.load(OUTDIR / f"stage1_refit{tag}.npz", allow_pickle=True)
    th63 = np.asarray(fz63["theta_best"], float)
    thr = np.asarray(fzr["theta_best"], float)
    names = list(spec2.theta_names)
    # variant B, when its refit exists: the measured curves and their optimum
    fm = OUTDIR / f"stage1_refit_measured{tag}.npz"
    thm, pr_m, meas = None, None, None
    if fm.exists():
        import measured as MB
        hist = dict(np.load(OUTDIR / f"history_curves{tag}.npz", allow_pickle=True))
        meas, _ = MB.build_measured(recs, curves, np.load(SEL.HS_NPZ, allow_pickle=True), hist, verbose=False)
        pr_m = S1.PreEpochJoint(spec2, curves, {k: meas for k in EPOCHS}, data, mask, lmh, F.R_GRID, th_inc)
        fzm = np.load(fm, allow_pickle=True)
        thm = np.asarray(fzm["theta_best"], float)
        print(f"  MEASURED refit: starts " + ", ".join(f"{n} {l:.4f}" for n, l in zip(fzm["names"], fzm["losses"]))
              + f"; best '{str(fzm['best_name'])}' {float(fzm['loss_best']):.4f}")
        movedm = [(n, a, b) for n, a, b in zip(names, th63, thm) if abs(a - b) > 0.05]
        print(f"  moved by > 0.05 from exp63: " + (", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in movedm) or "none"))

    # ---- 1. the fit -------------------------------------------------------- #
    print(f"  starts: " + ", ".join(f"{n} {l:.4f}" for n, l in zip(fzr["names"], fzr["losses"]))
          + f"; best '{str(fzr['best_name'])}' {float(fzr['loss_best']):.4f}"
          + (f"; RAILED {list(fzr['railed_best'])}" if len(fzr["railed_best"]) else ""))
    moved = [(n, a, b) for n, a, b in zip(names, th63, thr) if abs(a - b) > 0.05]
    print(f"  moved by > 0.05 from exp63: " + (", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in moved) or "none"))
    print("  refit: " + "  ".join(f"{n}={v:+.4f}" for n, v in zip(names, thr)))

    # ---- 2. the 2 x 2 ------------------------------------------------------ #
    print(f"\n{RULE}\n2. THETA x CURVES under exp63's four-term loss (A^2+F^2+S^2+B^2 per epoch, references = "
          f"the nested incumbent\n   on the official curves). Diagonal: each fit on its own input. "
          f"Off-diagonal: the frozen-theta cost of the input change.\n{RULE}")
    print(f"  {'theta \\ curves':<28}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z) + f"{'total':>10}")
    grid = {}
    combos = [("exp63", th63, "official"), ("exp63", th63, "pre-epoch"),
              ("exp74 refit", thr, "official"), ("exp74 refit", thr, "pre-epoch")]
    if thm is not None:
        combos += [("exp63", th63, "measured"), ("refit measured", thm, "official"),
                   ("refit measured", thm, "measured")]
    for tl, th, cl in combos:
        if cl == "official":
            per = S2F.JointProblem2.per_epoch(pr, th, nodes=M2.FULL_NODES)
        elif cl == "pre-epoch":
            per = pr.per_epoch(th, nodes=M2.FULL_NODES)
        else:
            per = pr_m.per_epoch(th, nodes=M2.FULL_NODES)
        row = np.array([sum(x * x for x in (per[k][0], per[k][1], per[k][2], per[k][4])) for k in pr.epochs])
        grid[(tl, cl)] = row
        print(f"  {tl + ' on ' + cl:<28}" + "".join(f"{v:>9.3f}" for v in row) + f"{row.sum():>10.3f}")

    # ---- predictions ------------------------------------------------------- #
    pred = {"exp63 official": M2.predict2(spec2, th63, curves, F.R_GRID),
            "refit pre-epoch": np.stack([M2.predict2(spec2, thr, pre[k], F.R_GRID, epochs=(k,))[:, 0, :]
                                         for k in EPOCHS], axis=1)}
    if thm is not None:
        pred["refit measured"] = M2.predict2(spec2, thm, meas, F.R_GRID)
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for v in pred.values():
        good &= np.isfinite(v).all(axis=(1, 2)) & (v > 0).all(axis=(1, 2))
    fit_all = good & mask.all(1)
    _, complete = S2F.fit_masks(data, np.load(SEL_NPZ, allow_pickle=True)["mask"] if False else mask, lmh, "sane") \
        if False else (None, None)
    # the mh-complete after-fit mask, as every judge in the programme builds it
    sn = np.load(SEL_NPZ, allow_pickle=True)
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    rows = np.array([h.row for h in recs])
    lmh_cat = SEL.sample_masses(hs)[rows]
    complete = np.isfinite(lmh_cat) & (lmh_cat >= sn["cuts"][None, :])
    growth = lmh_cat[:, 0][:, None] - lmh_cat
    print(f"\n  fitting sample {int(fit_all.sum())}; mh-complete per epoch "
          + "/".join(str(int((fit_all & complete[:, k]).sum())) for k in EPOCHS))

    # ---- 3. the profile ---------------------------------------------------- #
    ir = [int(np.argmin(np.abs(F.R_GRID - r))) for r in R_SHOW]
    print(f"\n{RULE}\n3. THE PROFILE — median (model - data)/data [%], fitting sample | mh-complete\n{RULE}")
    print(f"  {'epoch':>6}{'':<18}" + "".join(f"{f'M(<{F.R_GRID[i]:.0f})':>22}" for i in ir))
    prof = {}
    for k in EPOCHS:
        m_c = fit_all & complete[:, k]
        for lab, prd in pred.items():
            res = [(100 * np.nanmedian((prd[fit_all, k, i] - data[fit_all, k, i]) / data[fit_all, k, i]),
                    100 * np.nanmedian((prd[m_c, k, i] - data[m_c, k, i]) / data[m_c, k, i])) for i in ir]
            prof[(lab, k)] = res
            print(f"  {ANCHOR_Z[k] if lab.startswith('exp63') else '':>6}{lab:<18}"
                  + "".join(f"{a:>+10.1f} |{b:>+9.1f}" for a, b in res))

    # ---- 4. the leak ------------------------------------------------------- #
    c = int(np.argmin(np.abs(F.R_GRID - R_LEAK)))
    y_tru = np.log10(np.clip(data[:, :, c], 1.0, None))
    print(f"\n{RULE}\n4. THE LEAK — the residual's dy/dG at fixed CATALOG mass, M(<{R_LEAK:g}) [partial rho | dex per "
          f"dex]; the truth's own dependence in the first row\n{RULE}")
    print(f"  {'':<22}" + "".join(f"{f'z={z}':>18}" for z in ANCHOR_Z[1:]))
    leak = {}
    for lab in ("truth", *pred):
        cells = []
        for k in range(1, 5):
            y = y_tru[:, k] if lab == "truth" else np.log10(np.clip(pred[lab][:, k, c], 1.0, None)) - y_tru[:, k]
            rho, sl, n = SEL.partial_growth(y, lmh_cat[:, k], growth[:, k], fit_all & np.isfinite(lmh_cat[:, k]))
            leak[(lab, k)] = (rho, sl)
            cells.append(f"{rho:>+7.3f} |{sl:>+8.3f}")
        print(f"  {lab:<22}" + "".join(f"{s:>18}" for s in cells))

    # ---- 5. the tilt ------------------------------------------------------- #
    print(f"\n{RULE}\n5. THE HALO-MASS TILT of the {R_LEAK:g} kpc residual vs the CATALOG mass [dex per dex], "
          f"fitting sample | mh-complete\n   (exp71 / C18 at z=2: -0.122 | +0.061 for exp63)\n{RULE}")
    print(f"  {'':<18}" + "".join(f"{f'z={z}':>20}" for z in ANCHOR_Z))
    tl = {}
    for lab, prd in pred.items():
        cells = []
        for k in EPOCHS:
            y = np.log10(np.clip(prd[:, k, c], 1.0, None)) - y_tru[:, k]
            a, b = tilt(y, lmh_cat[:, k], fit_all), tilt(y, lmh_cat[:, k], fit_all & complete[:, k])
            tl[(lab, k)] = (a, b)
            cells.append(f"{a:>+8.3f} |{b:>+8.3f}")
        print(f"  {lab:<18}" + "".join(f"{s:>20}" for s in cells))

    # ---- 6. sizes, the gate, the battery ----------------------------------- #
    print(f"\n{RULE}\n6. SIZES, THE SIZE GATE AND THE STANDARD BATTERY (figures in figures/qa)\n{RULE}")
    r50_t = C.size_radius(data[fit_all], F.R_GRID, 0.5)
    print(f"  median R50 model/truth - 1:")
    for lab, prd in pred.items():
        r50 = C.size_radius(prd[fit_all], F.R_GRID, 0.5)
        print(f"    {lab:<18}" + "".join(f"{100 * (np.nanmedian(r50[:, k] / r50_t[:, k]) - 1):>+8.1f}%" for k in EPOCHS))
    (FIGDIR / "qa").mkdir(parents=True, exist_ok=True)
    logms = np.log10(np.clip(data[fit_all][:, 0, -1], 1.0, None))
    out = {}
    for lab, prd in pred.items():
        print(f"\n  --- {lab} ---")
        out[lab] = qa.evaluate(prd[fit_all], data[fit_all], F.R_GRID, ANCHOR_Z,
                               name=f"exp74_{lab.replace(' ', '_')}{tag}",
                               figdir=(None if tables_only else FIGDIR / "qa"), figures=not tables_only,
                               verbose=False, bin_by=lmh[fit_all][:, 0], bin_label=r"logM$_h$(z=0.4)",
                               bin_by_ms=logms, ms_label=r"logM$_*$ (total)", halo_mass_epochs=lmh_cat[fit_all])
        qa.print_size_gate(out[lab]["size_gate_ms"], ANCHOR_Z, "STELLAR mass")
    for key in TABLE_KEYS:
        print(f"\n  {key} — median relative bias, fitting sample / mh-complete")
        print(f"  {'model':<18}{'sample':<16}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        comp = complete[fit_all]
        for lab in pred:
            t, mm = out[lab]["truth"][key], out[lab]["model"][key]
            print(f"  {lab:<18}{'fitting sample':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[:, j], t[:, j])):>9.1f}%" for j in range(5)))
            print(f"  {'':<18}{'mh-complete':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[comp[:, j], j], t[comp[:, j], j])):>9.1f}%" for j in range(5)))

    np.savez(OUTDIR / f"stage1_eval{tag}.npz",
             grid_keys=np.array([f"{a}|{b}" for (a, b) in grid]), grid=np.array(list(grid.values())),
             prof_keys=np.array([f"{a}|{k}" for (a, k) in prof]), prof=np.array(list(prof.values())),
             leak_keys=np.array([f"{a}|{k}" for (a, k) in leak]), leak=np.array(list(leak.values())),
             tilt_keys=np.array([f"{a}|{k}" for (a, k) in tl]), tilt=np.array(list(tl.values())),
             fit_all=fit_all, complete=complete)
    print(f"\nwrote {OUTDIR / f'stage1_eval{tag}.npz'}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, tables_only="--tables-only" in sys.argv)
