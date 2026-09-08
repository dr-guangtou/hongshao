"""exp78 — the judge for the size-aware fit(s), against the adopted baseline.

Models (all on the measured curves, adopted references, fitting sample):
  baseline        THE BASELINE MEAN (exp74's measured optimum), fitted WITHOUT Z
  14.63 basin     the loss-preferred, gate-rejected basin (C23), for reference
  size-aware      Stage 1's optimum under A^2+F^2+S^2+B^2+Z^2 (12 parameters)
  size-aware + g  Stage 2's optimum with the growth-rate split free (13), if it exists

Sections: 1 the fits; 2 the loss grid (A, F, S, B, Z per epoch, with and
without Z); 3 the profile at 2 / 10 / 52 / 103 kpc, fitting sample |
mh-complete; 4 the future-dependence gate and the halo-mass tilt; 5 sizes,
the size gate with OFFSET and WIDTH counted separately (at fixed stellar mass
and at fixed halo mass), the standard battery with figures
(`figures/qa/*exp78_*`), the bias tables on both samples.

Run after `stage1_fit.py --merge` (and `--growth --merge`):
    HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. nohup uv run python -u \\
        experiments/exp78_size_aware_objective/stage1_eval.py [--smoke] [--tables-only] > outputs/stage1_eval.log 2>&1 &
"""
from __future__ import annotations

import importlib.util
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
from hongshao import qa                                  # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S1 = _by_path("exp78_stage1_fit", HERE / "stage1_fit.py")     # exp76 has a `stage1_fit` too

RULE = "=" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
E74 = ROOT / "experiments/exp74_c19_history_leak/outputs"
SEL_NPZ = ROOT / "experiments/exp54_unpinned_amplitude/outputs/selection.npz"
R_SHOW = (2.0, 10.25, 52.30, 103.45)
R_LEAK = 103.45
TABLE_KEYS = ("kpc:M(<10)", "kpc:M(<100)", "kpc:M(50-100)")


def tilt(y, x, mask):
    ok = mask & np.isfinite(y) & np.isfinite(x)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 30 else np.nan


def main(smoke=False, tables_only=False):
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp78 — the judge: the size-aware fit against the adopted baseline\n{RULE}\n")
    (recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, spec, th_inc, th_nested,
     pr, sap, Rm, truth_m, n_inner, rows_m) = S1.build(smoke)
    spec_g = M2.Spec2(theta_names=M2.THETA_NAMES_GROWTH, extended_family=spec2.extended_family,
                      compact_in_kpc=spec2.compact_in_kpc)
    names = list(spec2.theta_names)

    def theta_of(path):
        p = Path(str(path).replace(".npz", f"{tag}.npz"))
        p = p if p.exists() else Path(path)
        return np.asarray(np.load(p, allow_pickle=True)["theta_best"], float)

    th_base = theta_of(E74 / "stage1_refit_measured.npz")
    th_basin = theta_of(E74 / "rebaseline_exp63.npz")
    f1 = np.load(OUTDIR / f"stage1_fit{tag}.npz", allow_pickle=True)
    th1 = np.asarray(f1["theta_best"], float)
    models = {"baseline": (spec2, th_base), "14.63 basin": (spec2, th_basin), "size-aware": (spec2, th1)}
    print(f"  Stage 1: starts " + ", ".join(f"{n} {l:.4f} ({e} evals)" for n, l, e in zip(f1["names"], f1["losses"], f1["n_evals"]))
          + f"; best '{str(f1['best_name'])}' {float(f1['loss_best']):.4f} (null {float(f1['loss_null']):.4f})"
          + (f"; RAILED {list(f1['railed_best'])}" if len(f1["railed_best"]) else ""))
    moved = [(n, a, b) for n, a, b in zip(names, th_base, th1) if abs(a - b) > 0.05]
    print(f"  size-aware moved by > 0.05 from the baseline: "
          + (", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in moved) or "none"))
    moved = [(n, a, b) for n, a, b in zip(names, th_basin, th1) if abs(a - b) > 0.05]
    print(f"  size-aware moved by > 0.05 from the 14.63 basin: "
          + (", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in moved) or "none"))
    f2p = OUTDIR / f"stage1_fit_growth{tag}.npz"
    if f2p.exists():
        f2 = np.load(f2p, allow_pickle=True)
        th2 = np.asarray(f2["theta_best"], float)
        jg = spec_g.index("g_split")
        models["size-aware + g"] = (spec_g, th2)
        print(f"  Stage 2: starts " + ", ".join(f"{n} {l:.4f}" for n, l in zip(f2["names"], f2["losses"]))
              + f"; best '{str(f2['best_name'])}' {float(f2['loss_best']):.4f}, g = {th2[jg]:+.3f}"
              + (f"; RAILED {list(f2['railed_best'])}" if len(f2["railed_best"]) else ""))
        moved = [(n, a, b) for n, a, b in zip(names, th1, th2[:-1]) if abs(a - b) > 0.05]
        print(f"  Stage 2 moved by > 0.05 from Stage 1: "
              + (", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in moved) or "none"))
    print(f"  PARAMETERS: baseline 12; size-aware 12 (+0 for the term); size-aware + g 13")

    # ---- 2. the loss grid ------------------------------------------------------ #
    print(f"\n{RULE}\n2. THE LOSS GRID under the adopted references — A, F, S, B, Z per epoch; the loss with and without Z\n{RULE}")
    print(f"  {'model':<16}{'epoch':>6}{'A':>8}{'F':>8}{'S':>8}{'B':>8}{'Z':>8}{'A2+F2+S2+B2':>13}{'+Z2':>8}")
    grid, pred = {}, {}
    for lab, (sp, th) in models.items():
        scores, raw, z = sap.per_epoch(th, nodes=M2.FULL_NODES) if sp is spec2 else _scores_growth(sap, sp, th)
        tab = np.array([[scores[k][0], scores[k][1], scores[k][2], scores[k][4], z[j]] for j, k in enumerate(EPOCHS)])
        grid[lab] = tab
        for j, k in enumerate(EPOCHS):
            print(f"  {lab if j == 0 else '':<16}{ANCHOR_Z[k]:>6}" + "".join(f"{v:>8.3f}" for v in tab[j])
                  + f"{(tab[j, :4] ** 2).sum():>13.3f}{(tab[j] ** 2).sum():>8.3f}")
        print(f"  {'':<16}{'total':>6}{'':>40}{(tab[:, :4] ** 2).sum():>13.3f}{(tab ** 2).sum():>8.3f}")
        full = M2.predict2(sp, th, [meas[i] for i in range(len(recs))], F.R_GRID, epochs=EPOCHS, nodes=M2.FULL_NODES)
        pred[lab] = full

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
    print(f"\n  fitting sample {int(fit_all.sum())}; mh-complete per epoch "
          + "/".join(str(int((fit_all & complete[:, k]).sum())) for k in EPOCHS))

    # ---- 3. the profile ---------------------------------------------------- #
    ir = [int(np.argmin(np.abs(F.R_GRID - r))) for r in R_SHOW]
    print(f"\n{RULE}\n3. THE PROFILE — median (model - data)/data [%], fitting sample | mh-complete\n{RULE}")
    print(f"  {'epoch':>6}{'':<18}" + "".join(f"{f'M(<{F.R_GRID[i]:.0f})':>22}" for i in ir))
    prof = {}
    for k in EPOCHS:
        m_c = fit_all & complete[:, k]
        for j, (lab, prd) in enumerate(pred.items()):
            res = [(100 * np.nanmedian((prd[fit_all, k, i] - data[fit_all, k, i]) / data[fit_all, k, i]),
                    100 * np.nanmedian((prd[m_c, k, i] - data[m_c, k, i]) / data[m_c, k, i])) for i in ir]
            prof[(lab, k)] = res
            print(f"  {ANCHOR_Z[k] if j == 0 else '':>6}{lab:<18}" + "".join(f"{a:>+10.1f} |{b:>+9.1f}" for a, b in res))

    # ---- 4. the leak + tilt ------------------------------------------------- #
    c = int(np.argmin(np.abs(F.R_GRID - R_LEAK)))
    y_tru = np.log10(np.clip(data[:, :, c], 1.0, None))
    print(f"\n{RULE}\n4. THE FUTURE-DEPENDENCE GATE — the residual's dy/dG at fixed catalog mass, M(<{R_LEAK:g}) "
          f"[partial rho | dex per dex]; the truth's own in the first row\n{RULE}")
    print(f"  {'':<18}" + "".join(f"{f'z={z}':>18}" for z in ANCHOR_Z[1:]))
    leak = {}
    for lab in ("truth", *pred):
        cells = []
        for k in range(1, 5):
            y = y_tru[:, k] if lab == "truth" else np.log10(np.clip(pred[lab][:, k, c], 1.0, None)) - y_tru[:, k]
            rho, sl, n = SEL.partial_growth(y, lmh_cat[:, k], growth[:, k], fit_all & np.isfinite(lmh_cat[:, k]))
            leak[(lab, k)] = (rho, sl)
            cells.append(f"{rho:>+7.3f} |{sl:>+8.3f}")
        print(f"  {lab:<18}" + "".join(f"{s:>18}" for s in cells))
    print(f"\n  the halo-mass TILT of the {R_LEAK:g} kpc residual vs the catalog mass [dex per dex], fitting | mh-complete")
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

    # ---- 5. sizes, gate, battery ------------------------------------------- #
    print(f"\n{RULE}\n5. SIZES, THE SIZE GATE (OFFSET and WIDTH separately) AND THE STANDARD BATTERY "
          f"(figures in figures/qa, *exp78_*)\n{RULE}")
    print(f"  median R_f model/truth - 1 on the standard grid, fitting sample:")
    for frac in (0.2, 0.5, 0.8):
        r_t = C.size_radius(data[fit_all], F.R_GRID, frac)
        for lab, prd in pred.items():
            r_m = C.size_radius(prd[fit_all], F.R_GRID, frac)
            print(f"    R{int(100 * frac):<3}{lab:<18}" + "".join(
                f"{100 * (np.nanmedian(r_m[:, k] / r_t[:, k]) - 1):>+8.1f}%" for k in EPOCHS))
    (FIGDIR / "qa").mkdir(parents=True, exist_ok=True)
    logms = np.log10(np.clip(data[fit_all][:, 0, -1], 1.0, None))
    out, gates = {}, {}
    for lab, prd in pred.items():
        print(f"\n  --- {lab} ---")
        out[lab] = qa.evaluate(prd[fit_all], data[fit_all], F.R_GRID, ANCHOR_Z,
                               name=f"exp78_{lab.replace(' ', '_').replace('+', 'plus').replace('.', 'p')}{tag}",
                               figdir=(None if tables_only else FIGDIR / "qa"), figures=not tables_only,
                               verbose=False, bin_by=lmh_bins[fit_all][:, 0], bin_label=r"logM$_h$(z=0.4)",
                               bin_by_ms=logms, ms_label=r"logM$_*$ (total)", halo_mass_epochs=lmh_cat[fit_all])
        qa.print_size_gate(out[lab]["size_gate_ms"], ANCHOR_Z, "STELLAR mass")
        qa.print_size_gate(out[lab]["size_gate_mh"], ANCHOR_Z, "HALO mass")
        gates[lab] = (out[lab]["size_gate_ms"][3], out[lab]["size_gate_ms"][4],
                      out[lab]["size_gate_mh"][3], out[lab]["size_gate_mh"][4])
    print(f"\n  SIZE GATE SUMMARY (of 15): offset | width, at fixed stellar mass; offset | width, at fixed halo mass")
    for lab, g in gates.items():
        print(f"    {lab:<18}{g[0]:>4} | {g[1]:<4}   {g[2]:>4} | {g[3]:<4}")
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
             grid_keys=np.array(list(grid)), grid=np.array(list(grid.values())),
             prof_keys=np.array([f"{a}|{k}" for (a, k) in prof]), prof=np.array(list(prof.values())),
             leak_keys=np.array([f"{a}|{k}" for (a, k) in leak]), leak=np.array(list(leak.values())),
             tilt_keys=np.array([f"{a}|{k}" for (a, k) in tl]), tilt=np.array(list(tl.values())),
             gate_keys=np.array(list(gates)), gates=np.array(list(gates.values())),
             fit_all=fit_all, complete=complete)
    print(f"\nwrote {OUTDIR / f'stage1_eval{tag}.npz'}")


def _scores_growth(sap, spec_g, th):
    """Score a 13-parameter theta on the 12-parameter problem's references
    (the split nests at g = 0; the problems' references do not depend on the spec)."""
    m = M2.predict2(spec_g, th, sap.pr.curves, sap.Rm, epochs=tuple(sap.epochs), nodes=M2.FULL_NODES)
    scores, raw = {}, np.full(len(sap.epochs), np.nan)
    for j, k in enumerate(sap.epochs):
        scores[k] = sap.pr.problems[k].score_model(m[sap.pr.index[k], j, sap.n_inner:])
        raw[j] = sap._radius_term(m[sap.index_m[k], j], k)
    return scores, raw, raw / sap.z_ref


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, tables_only="--tables-only" in sys.argv)
