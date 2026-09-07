"""exp76 Stage 1, the judge — the growth-rate split against the adopted
baseline (exp63's model on the measured curves, g = 0), same input, same
references, same sample. Everything below is what one parameter bought.

  1. the fit: starts, the fitted g, what else moved;
  2. the loss per epoch, both models, under the adopted references;
  3. P9's table at the fitted g (Stage 0's measurement, at the fit's theta);
  4. the profile at 2 / 10 / 52 / 103 kpc, fitting sample | mh-complete;
  5. the future-dependence gate and the z = 2 tilt;
  6. sizes, the size gate (offset and WIDTH), the standard battery with
     figures (`figures/qa/*exp76_*`).

Run after `stage1_fit.py --merge` (and the adoption's `rebaseline.py --model exp63 --merge`):
    HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
        experiments/exp76_growth_rate_split/stage1_eval.py [--smoke] [--tables-only]
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
          ROOT / "experiments/exp73_size_relative",
          ROOT / "experiments/exp74_c19_history_leak", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import selection as SEL                                  # noqa: E402
import coordinate as C                                   # noqa: E402
import stage1_fit as S1                                  # noqa: E402
from stage0_leverage import (measured_assembly, partial_spearman, compact_share,  # noqa: E402
                             MEASURED_VARS, DIFFMAH_VARS, E74, E63, POP)
from hongshao import qa                                  # noqa: E402

RULE = "=" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
SEL_NPZ = ROOT / "experiments/exp54_unpinned_amplitude/outputs/selection.npz"
R_SHOW = (2.0, 10.25, 52.30, 103.45)
R_LEAK = 103.45
TABLE_KEYS = ("kpc:M(<10)", "kpc:M(<100)")


def tilt(y, x, mask):
    ok = mask & np.isfinite(y) & np.isfinite(x)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 30 else np.nan


def main(smoke=False, tables_only=False):
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp76 Stage 1, the judge — the growth-rate split against the adopted baseline\n{RULE}\n")
    recs, data, mask, lmh_dm, meas, fz, spec2, spec_g, th_inc, pr, th0 = S1.build(smoke)
    fg = np.load(OUTDIR / f"stage1_fit{tag}.npz", allow_pickle=True)
    thg = np.asarray(fg["theta_best"], float)
    jg = spec_g.index("g_split")
    # the adopted baseline (g = 0): the adoption's refit if it exists, else exp74's measured optimum
    fb = E74 / f"rebaseline_exp63{tag}.npz"
    thb = np.asarray(np.load(fb, allow_pickle=True)["theta_best"], float) if fb.exists() else th0
    print(f"  baseline theta from {'the adoption refit (rebaseline_exp63)' if fb.exists() else 'exp74 measured refit'}")
    thb_g = np.r_[thb, 0.0]
    names = list(spec_g.theta_names)
    print(f"  starts: " + ", ".join(f"{n} {l:.4f} (g {g:+.2f})" for n, l, g in zip(fg["names"], fg["losses"], fg["g_by_start"]))
          + f"; best '{str(fg['best_name'])}' {float(fg['loss_best']):.4f}, g = {thg[jg]:+.3f}"
          + (f"; RAILED {list(fg['railed_best'])}" if len(fg["railed_best"]) else ""))
    moved = [(n, a, b) for n, a, b in zip(names, thb_g, thg) if abs(a - b) > 0.05]
    print(f"  moved by > 0.05 from the baseline: " + (", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in moved) or "none"))

    # ---- 2. loss ----------------------------------------------------------- #
    print(f"\n{RULE}\n2. THE LOSS per epoch under the adopted references (null = nested incumbent on measured curves)\n{RULE}")
    print(f"  {'model':<24}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z) + f"{'total':>10}")
    grid = {}
    cands0 = [(str(n), float(l), th) for n, l, th in zip(fg["names"], fg["losses"], fg["thetas"]) if th[jg] < -0.05]
    rows_loss = [("baseline (g = 0)", thb_g), ("growth-rate split", thg)]
    if cands0:
        thn0 = min(cands0, key=lambda c: c[1])[2]
        rows_loss.append((f"split g={thn0[jg]:+.2f}", thn0))
    for lab, th in rows_loss:
        per = pr.per_epoch(th, nodes=M2.FULL_NODES)
        row = np.array([sum(x * x for x in (per[k][0], per[k][1], per[k][2], per[k][4])) for k in pr.epochs])
        grid[lab] = row
        print(f"  {lab:<24}" + "".join(f"{v:>9.3f}" for v in row) + f"{row.sum():>10.3f}")

    # the best solution with g < 0, when the best overall is g = 0 (the fit rejecting the split)
    cands = [(str(n), float(l), th) for n, l, th in zip(fg["names"], fg["losses"], fg["thetas"]) if th[jg] < -0.05]
    thn = None
    if cands:
        n_neg, l_neg, thn = min(cands, key=lambda c: c[1])
        print(f"  best start with g < 0: '{n_neg}' {l_neg:.4f}, g = {thn[jg]:+.3f} (shown as the third model)")
    pred = {"baseline (g = 0)": M2.predict2(spec_g, thb_g, meas, F.R_GRID),
            "growth-rate split": M2.predict2(spec_g, thg, meas, F.R_GRID)}
    if thn is not None:
        pred[f"split g={thn[jg]:+.2f}"] = M2.predict2(spec_g, thn, meas, F.R_GRID)
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for v in pred.values():
        good &= np.isfinite(v).all(axis=(1, 2)) & (v > 0).all(axis=(1, 2))
    fit_all = good & mask.all(1)
    sn = np.load(SEL_NPZ, allow_pickle=True)
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    rows = np.array([h.row for h in recs])
    lmh_cat = SEL.sample_masses(hs)[rows]
    complete = np.isfinite(lmh_cat) & (lmh_cat >= sn["cuts"][None, :])
    growth = lmh_cat[:, 0][:, None] - lmh_cat
    ctrl = np.where(np.isfinite(lmh_cat[:, 0]), lmh_cat[:, 0], lmh_dm[:, 0])

    # ---- 3. P9 at the fitted g --------------------------------------------- #
    print(f"\n{RULE}\n3. P9 AT THE FIT — partial Spearman rho at fixed z=0.4 halo mass of the compact share with "
          f"assembly variables\n   (measured, then DiffMAH's); the data's inner share first; rms distance to the data "
          f"over all eight\n{RULE}")
    va = measured_assembly(recs, hs)
    curves = E.build_curves(recs, verbose=False)
    pop = np.load(POP, allow_pickle=True)
    va["late"] = np.array([hc.late for hc in curves]); va["logtc"] = np.array([hc.logtc for hc in curves])
    va["t50_dmah"] = np.asarray(pop["t50"], float)[rows]
    va["f_form"] = np.array([h.f_form[h.epoch_mask[0]][-1] for h in recs])
    s1 = np.load(E63 / "stage1_deconvolve.npz", allow_pickle=True)
    pos = {int(i): j for j, i in enumerate(s1["index"])}
    w = np.asarray(s1["w_gompertz_c0.80"], float); s_grid = np.asarray(s1["s_grid"], float)
    share_data = np.full(len(recs), np.nan)
    for g, h in enumerate(recs):
        j = pos.get(int(h.index))
        if j is not None:
            share_data[g] = w[j, s_grid < 5].sum() / max(w[j].sum(), 1e-30)
    all_vars = MEASURED_VARS + DIFFMAH_VARS
    r_data = np.array([partial_spearman(share_data, va[v], ctrl, fit_all) for v in all_vars])
    print(f"  {'share':<24}" + "".join(f"{v:>13}" for v in all_vars) + f"{'rms dist':>10}")
    print(f"  {'DATA inner share':<24}" + "".join(f"{v:>13.3f}" for v in r_data))
    p9 = {}
    for lab, th in rows_loss:
        sh = compact_share(spec_g, th, meas, None)
        r = np.array([partial_spearman(sh[:, 0], va[v], ctrl, fit_all) for v in all_vars])
        p9[lab] = (r, float(np.nanmedian(sh[fit_all, 0])))
        print(f"  {lab + f' (share {p9[lab][1]:.2f})':<24}" + "".join(f"{v:>13.3f}" for v in r)
              + f"{np.sqrt(np.nanmean((r - r_data) ** 2)):>10.3f}")

    # ---- 4. profile ---------------------------------------------------------- #
    ir = [int(np.argmin(np.abs(F.R_GRID - r))) for r in R_SHOW]
    print(f"\n{RULE}\n4. THE PROFILE — median (model - data)/data [%], fitting sample | mh-complete\n{RULE}")
    print(f"  {'epoch':>6}{'':<20}" + "".join(f"{f'M(<{F.R_GRID[i]:.0f})':>22}" for i in ir))
    prof = {}
    for k in EPOCHS:
        m_c = fit_all & complete[:, k]
        for j, (lab, prd) in enumerate(pred.items()):
            res = [(100 * np.nanmedian((prd[fit_all, k, i] - data[fit_all, k, i]) / data[fit_all, k, i]),
                    100 * np.nanmedian((prd[m_c, k, i] - data[m_c, k, i]) / data[m_c, k, i])) for i in ir]
            prof[(lab, k)] = res
            print(f"  {ANCHOR_Z[k] if j == 0 else '':>6}{lab:<20}" + "".join(f"{a:>+10.1f} |{b:>+9.1f}" for a, b in res))

    # ---- 5. leak + tilt ------------------------------------------------------ #
    c = int(np.argmin(np.abs(F.R_GRID - R_LEAK)))
    y_tru = np.log10(np.clip(data[:, :, c], 1.0, None))
    print(f"\n{RULE}\n5. THE FUTURE-DEPENDENCE GATE (residual dy/dG at fixed catalog mass, dex/dex; truth first) and "
          f"the z-tilt\n   of the {R_LEAK:g} kpc residual vs catalog mass, fitting | mh-complete\n{RULE}")
    print(f"  {'':<22}" + "".join(f"{f'z={z}':>12}" for z in ANCHOR_Z[1:]))
    leak, tl = {}, {}
    for lab in ("truth", *pred):
        cells = []
        for k in range(1, 5):
            y = y_tru[:, k] if lab == "truth" else np.log10(np.clip(pred[lab][:, k, c], 1.0, None)) - y_tru[:, k]
            rho, sl, n = SEL.partial_growth(y, lmh_cat[:, k], growth[:, k], fit_all & np.isfinite(lmh_cat[:, k]))
            leak[(lab, k)] = (rho, sl); cells.append(f"{sl:>+12.3f}")
        print(f"  {lab:<22}" + "".join(cells))
    print(f"  {'tilt':<22}" + "".join(f"{f'z={z}':>20}" for z in ANCHOR_Z))
    for lab, prd in pred.items():
        cells = []
        for k in EPOCHS:
            y = np.log10(np.clip(prd[:, k, c], 1.0, None)) - y_tru[:, k]
            a, b = tilt(y, lmh_cat[:, k], fit_all), tilt(y, lmh_cat[:, k], fit_all & complete[:, k])
            tl[(lab, k)] = (a, b); cells.append(f"{a:>+8.3f} |{b:>+8.3f}")
        print(f"  {lab:<22}" + "".join(f"{s:>20}" for s in cells))

    # ---- 6. sizes, gate, battery -------------------------------------------- #
    print(f"\n{RULE}\n6. SIZES, THE SIZE GATE (offset AND width) AND THE STANDARD BATTERY (figures/qa/*exp76_*)\n{RULE}")
    r50_t = C.size_radius(data[fit_all], F.R_GRID, 0.5)
    print(f"  median R50 model/truth - 1:")
    for lab, prd in pred.items():
        r50 = C.size_radius(prd[fit_all], F.R_GRID, 0.5)
        print(f"    {lab:<22}" + "".join(f"{100 * (np.nanmedian(r50[:, k] / r50_t[:, k]) - 1):>+8.1f}%" for k in EPOCHS))
    (FIGDIR / "qa").mkdir(parents=True, exist_ok=True)
    logms = np.log10(np.clip(data[fit_all][:, 0, -1], 1.0, None))
    lmh_bins = np.where(np.isfinite(lmh_cat), lmh_cat, lmh_dm)
    out = {}
    for lab, prd in pred.items():
        print(f"\n  --- {lab} ---")
        out[lab] = qa.evaluate(prd[fit_all], data[fit_all], F.R_GRID, ANCHOR_Z,
                               name=f"exp76_{lab.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '')}{tag}",
                               figdir=(None if tables_only else FIGDIR / "qa"), figures=not tables_only,
                               verbose=False, bin_by=lmh_bins[fit_all][:, 0], bin_label=r"logM$_h$(z=0.4)",
                               bin_by_ms=logms, ms_label=r"logM$_*$ (total)", halo_mass_epochs=lmh_cat[fit_all])
        qa.print_size_gate(out[lab]["size_gate_ms"], ANCHOR_Z, "STELLAR mass")
    for key in TABLE_KEYS:
        print(f"\n  {key} — median relative bias, fitting sample / mh-complete")
        print(f"  {'model':<22}{'sample':<16}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        comp = complete[fit_all]
        for lab in pred:
            t, mm = out[lab]["truth"][key], out[lab]["model"][key]
            print(f"  {lab:<22}{'fitting sample':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[:, j], t[:, j])):>9.1f}%" for j in range(5)))
            print(f"  {'':<22}{'mh-complete':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[comp[:, j], j], t[comp[:, j], j])):>9.1f}%" for j in range(5)))

    np.savez(OUTDIR / f"stage1_eval{tag}.npz", grid_keys=np.array(list(grid)), grid=np.array(list(grid.values())),
             p9_vars=np.array(all_vars), p9_data=r_data, p9_keys=np.array(list(p9)),
             p9_model=np.array([v[0] for v in p9.values()]),
             prof_keys=np.array([f"{a}|{k}" for (a, k) in prof]), prof=np.array(list(prof.values())),
             leak_keys=np.array([f"{a}|{k}" for (a, k) in leak]), leak=np.array(list(leak.values())),
             tilt_keys=np.array([f"{a}|{k}" for (a, k) in tl]), tilt=np.array(list(tl.values())),
             fit_all=fit_all, complete=complete, theta_baseline=thb_g, theta_fit=thg)
    print(f"\nwrote {OUTDIR / f'stage1_eval{tag}.npz'}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, tables_only="--tables-only" in sys.argv)
