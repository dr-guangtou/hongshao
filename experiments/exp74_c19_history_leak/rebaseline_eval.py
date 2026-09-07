"""The judge for the ADOPTION + RE-BASELINE (2026-09-08).

Four models on the standard battery, under the ADOPTED references (the nested
incumbent on the measured curves, bins by the measured mass):

  exp63 official        the model as it was fitted, on the official curves —
                        the last product of the old input, for continuity
  baseline measured     exp63's 12-parameter model refitted on the measured
                        curves under the adopted references — THE NEW BASELINE
  incumbent measured    the incumbent (compact channel off) refitted on the
                        measured curves — the new null every later fit nests in
  exp74 measured        exp74's refit (official references, DiffMAH bins) — to
                        show what the reference change alone did

Sections as `stage1_eval.py`: the loss grid, the profile, the leak, the tilt,
sizes + size gate + battery with figures (`figures/qa/*exp74_adopt_*`).

Run after both `rebaseline.py --merge` calls:
    HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
        experiments/exp74_c19_history_leak/rebaseline_eval.py [--smoke] [--tables-only]
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
import coordinate as C                                   # noqa: E402
import importlib.util                                    # noqa: E402
_spec = importlib.util.spec_from_file_location("exp74_rebaseline", HERE / "rebaseline.py")
RB = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(RB)   # another `rebaseline` shadows it
from stage0_frozen import tilt, R_SHOW, R_LEAK           # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE = "=" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
SEL_NPZ = ROOT / "experiments/exp54_unpinned_amplitude/outputs/selection.npz"
TABLE_KEYS = ("kpc:M(<10)", "kpc:M(<100)")


def main(smoke=False, tables_only=False):
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nADOPTION + RE-BASELINE — the judge\n{RULE}\n")
    recs, data, mask, lmh_dm, lmh_bins, meas, fz63, spec2, th_inc, pr = RB.build(smoke)
    curves = E.build_curves(recs, verbose=False)
    th63 = np.asarray(fz63["theta_best"], float)
    fb = np.load(OUTDIR / f"rebaseline_exp63{tag}.npz", allow_pickle=True)
    fi = np.load(OUTDIR / f"rebaseline_incumbent{tag}.npz", allow_pickle=True)
    f74 = np.load(OUTDIR / f"stage1_refit_measured{tag}.npz", allow_pickle=True)
    thb, thi, th74 = (np.asarray(f["theta_best"], float) for f in (fb, fi, f74))
    names = list(spec2.theta_names)
    for lab, f in (("baseline measured", fb), ("incumbent measured", fi)):
        print(f"  {lab}: starts " + ", ".join(f"{n} {l:.4f}" for n, l in zip(f["names"], f["losses"]))
              + f"; best '{str(f['best_name'])}' {float(f['loss_best']):.4f} (null {float(f['loss_null']):.4f})"
              + (f"; RAILED {list(f['railed_best'])}" if len(f["railed_best"]) else ""))
    moved = [(n, a, b) for n, a, b in zip(names, th63, thb) if abs(a - b) > 0.05]
    print(f"  baseline moved by > 0.05 from exp63: " + (", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in moved) or "none"))
    moved = [(n, a, b) for n, a, b in zip(names, th74, thb) if abs(a - b) > 0.05]
    print(f"  baseline moved by > 0.05 from exp74's measured refit: "
          + (", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in moved) or "none"))
    th_nested = np.asarray(fb["theta_nested"], float)
    moved = [(n, a, b) for n, a, b in zip(names, th_nested, thi) if abs(a - b) > 0.05]
    print(f"  incumbent moved by > 0.05 from its official-curve values: "
          + (", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in moved) or "none"))

    # ---- 1. the loss grid under the ADOPTED references -------------------- #
    print(f"\n{RULE}\n1. THE LOSS under the adopted references (nested incumbent on measured curves = 1 per "
          f"referenced term;\n   bins by the measured mass), every model on its own input\n{RULE}")
    print(f"  {'model':<26}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z) + f"{'total':>10}")
    models = {"exp63 official": (curves, th63), "baseline measured": (meas, thb),
              "incumbent measured": (meas, thi), "exp74 measured": (meas, th74),
              "nested incumbent, measured": (meas, th_nested)}
    grid = {}
    for lab, (cv, th) in models.items():
        m = M2.predict2(spec2, th, [cv[i] for i in pr.all_rows], F.R_GRID, epochs=tuple(pr.epochs),
                        nodes=M2.FULL_NODES)
        per = {k: pr.problems[k].score_model(m[pr.index[k], j]) for j, k in enumerate(pr.epochs)}
        row = np.array([sum(x * x for x in (per[k][0], per[k][1], per[k][2], per[k][4])) for k in pr.epochs])
        grid[lab] = row
        print(f"  {lab:<26}" + "".join(f"{v:>9.3f}" for v in row) + f"{row.sum():>10.3f}")

    pred = {lab: M2.predict2(spec2, th, cv, F.R_GRID) for lab, (cv, th) in models.items()
            if lab != "nested incumbent, measured"}
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

    # ---- 2. the profile ---------------------------------------------------- #
    ir = [int(np.argmin(np.abs(F.R_GRID - r))) for r in R_SHOW]
    print(f"\n{RULE}\n2. THE PROFILE — median (model - data)/data [%], fitting sample | mh-complete\n{RULE}")
    print(f"  {'epoch':>6}{'':<20}" + "".join(f"{f'M(<{F.R_GRID[i]:.0f})':>22}" for i in ir))
    prof = {}
    for k in EPOCHS:
        m_c = fit_all & complete[:, k]
        for j, (lab, prd) in enumerate(pred.items()):
            res = [(100 * np.nanmedian((prd[fit_all, k, i] - data[fit_all, k, i]) / data[fit_all, k, i]),
                    100 * np.nanmedian((prd[m_c, k, i] - data[m_c, k, i]) / data[m_c, k, i])) for i in ir]
            prof[(lab, k)] = res
            print(f"  {ANCHOR_Z[k] if j == 0 else '':>6}{lab:<20}" + "".join(f"{a:>+10.1f} |{b:>+9.1f}" for a, b in res))

    # ---- 3. the leak ------------------------------------------------------- #
    c = int(np.argmin(np.abs(F.R_GRID - R_LEAK)))
    y_tru = np.log10(np.clip(data[:, :, c], 1.0, None))
    print(f"\n{RULE}\n3. THE FUTURE-DEPENDENCE GATE — the residual's dy/dG at fixed catalog mass, M(<{R_LEAK:g}) "
          f"[partial rho | dex per dex]; the truth's own in the first row\n{RULE}")
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

    # ---- 4. the tilt ------------------------------------------------------- #
    print(f"\n{RULE}\n4. THE HALO-MASS TILT of the {R_LEAK:g} kpc residual vs the catalog mass [dex per dex], "
          f"fitting sample | mh-complete\n{RULE}")
    print(f"  {'':<20}" + "".join(f"{f'z={z}':>20}" for z in ANCHOR_Z))
    tl = {}
    for lab, prd in pred.items():
        cells = []
        for k in EPOCHS:
            y = np.log10(np.clip(prd[:, k, c], 1.0, None)) - y_tru[:, k]
            a, b = tilt(y, lmh_cat[:, k], fit_all), tilt(y, lmh_cat[:, k], fit_all & complete[:, k])
            tl[(lab, k)] = (a, b)
            cells.append(f"{a:>+8.3f} |{b:>+8.3f}")
        print(f"  {lab:<20}" + "".join(f"{s:>20}" for s in cells))

    # ---- 5. sizes, gate, battery ------------------------------------------- #
    print(f"\n{RULE}\n5. SIZES, THE SIZE GATE AND THE STANDARD BATTERY (figures in figures/qa, *exp74_adopt_*)\n{RULE}")
    r50_t = C.size_radius(data[fit_all], F.R_GRID, 0.5)
    print(f"  median R50 model/truth - 1:")
    for lab, prd in pred.items():
        r50 = C.size_radius(prd[fit_all], F.R_GRID, 0.5)
        print(f"    {lab:<20}" + "".join(f"{100 * (np.nanmedian(r50[:, k] / r50_t[:, k]) - 1):>+8.1f}%" for k in EPOCHS))
    (FIGDIR / "qa").mkdir(parents=True, exist_ok=True)
    logms = np.log10(np.clip(data[fit_all][:, 0, -1], 1.0, None))
    out = {}
    for lab, prd in pred.items():
        print(f"\n  --- {lab} ---")
        out[lab] = qa.evaluate(prd[fit_all], data[fit_all], F.R_GRID, ANCHOR_Z,
                               name=f"exp74_adopt_{lab.replace(' ', '_')}{tag}",
                               figdir=(None if tables_only else FIGDIR / "qa"), figures=not tables_only,
                               verbose=False, bin_by=lmh_bins[fit_all][:, 0], bin_label=r"logM$_h$(z=0.4)",
                               bin_by_ms=logms, ms_label=r"logM$_*$ (total)", halo_mass_epochs=lmh_cat[fit_all])
        qa.print_size_gate(out[lab]["size_gate_ms"], ANCHOR_Z, "STELLAR mass")
    for key in TABLE_KEYS:
        print(f"\n  {key} — median relative bias, fitting sample / mh-complete")
        print(f"  {'model':<20}{'sample':<16}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        comp = complete[fit_all]
        for lab in pred:
            t, mm = out[lab]["truth"][key], out[lab]["model"][key]
            print(f"  {lab:<20}{'fitting sample':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[:, j], t[:, j])):>9.1f}%" for j in range(5)))
            print(f"  {'':<20}{'mh-complete':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[comp[:, j], j], t[comp[:, j], j])):>9.1f}%" for j in range(5)))

    np.savez(OUTDIR / f"rebaseline_eval{tag}.npz",
             grid_keys=np.array(list(grid)), grid=np.array(list(grid.values())),
             prof_keys=np.array([f"{a}|{k}" for (a, k) in prof]), prof=np.array(list(prof.values())),
             leak_keys=np.array([f"{a}|{k}" for (a, k) in leak]), leak=np.array(list(leak.values())),
             tilt_keys=np.array([f"{a}|{k}" for (a, k) in tl]), tilt=np.array(list(tl.values())),
             fit_all=fit_all, complete=complete)
    print(f"\nwrote {OUTDIR / f'rebaseline_eval{tag}.npz'}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, tables_only="--tables-only" in sys.argv)
