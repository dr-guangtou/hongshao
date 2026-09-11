"""exp80 — the judge for the Stage 1 fit (the size law with the expansion
exponent q_e), against the adopted baseline.

Models (all on the measured curves, adopted references, fitting sample):
  baseline      THE BASELINE MEAN (exp74's measured optimum, 12 parameters)
  14.63 basin   the loss-preferred, gate-rejected basin (C23), for reference
  exp80         Stage 1's optimum (13 parameters: 12 + q_e)
  exp80 (NAME)  any other Stage 1 start that settled in a different basin
                (`--also NAME`), for the record

Sections: 1 the fits; 2 the loss grid (A, F, S, B per epoch) and the radius
term scored AFTER the fit as a report (exp78's Z, normalised to the null);
3 the profile at 2 / 10 / 52 / 103 kpc, fitting sample | mh-complete; 4 the
future-dependence gate and the halo-mass tilt; 5 sizes, the size gate with
OFFSET and WIDTH counted separately (at fixed stellar mass and at fixed halo
mass), the standard battery with figures (`figures/qa/*exp80_*`, including
the mass planes `qa_planes_*`), the bias tables on both samples.

Run after `stage1_fit.py --merge`:
    HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. nohup uv run python -u \\
        experiments/exp80_deposit_size_law/stage1_eval.py [--smoke] [--tables-only] [--also NAME] > outputs/stage1_eval.log 2>&1 &
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
          ROOT / "experiments/exp74_c19_history_leak",
          ROOT / "experiments/exp78_size_aware_objective", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import selection as SEL                                  # noqa: E402
import coordinate as C                                   # noqa: E402
import size_terms as ST                                  # noqa: E402
import size_law as SL                                    # noqa: E402
from hongshao import qa                                  # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S1 = _by_path("exp80_stage1_fit", HERE / "stage1_fit.py")            # exp76/exp78 have a `stage1_fit` too
S0T = _by_path("exp78_stage0_terms", ROOT / "experiments/exp78_size_aware_objective/stage0_terms.py")

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


def main(smoke=False, tables_only=False, also=(), knobs=S1.DEFAULT_KNOBS, best=None, fit_tag=None, delay=False):
    tag = "_smoke" if smoke else ""
    ktag = fit_tag if fit_tag is not None else S1.knob_tag(knobs)
    print(f"{RULE}\nexp80 — the judge: the size law with {list(knobs)} against the adopted baseline\n{RULE}\n")
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, th_nested, pr, lp, bounds = S1.build(smoke, knobs, delay)
    Rm, truth_m, n_inner = S0T.merged_truth(recs, data)
    good_m = np.isfinite(truth_m).all(2) & (truth_m > 0).all(2)
    rows_m = {k: pr.rows[k][good_m[pr.rows[k], k]] for k in pr.epochs}
    index_of = np.full(len(recs), -1); index_of[pr.all_rows] = np.arange(len(pr.all_rows))
    nk = len(knobs) + (1 if delay else 0)      # tau_d rides with the twelve; baseline/basin get tau_d = 0

    def theta_of(path):
        p = Path(str(path).replace(".npz", f"{tag}.npz"))
        p = p if p.exists() else Path(path)
        return np.asarray(np.load(p, allow_pickle=True)["theta_best"], float)

    th_base = np.r_[theta_of(E74 / "stage1_refit_measured.npz"), np.zeros(nk)]
    th_basin = np.r_[theta_of(E74 / "rebaseline_exp63.npz"), np.zeros(nk)]
    f1 = np.load(OUTDIR / f"{S1.STAGE_FILES['stage1']}{ktag}{tag}.npz", allow_pickle=True)
    th1 = np.asarray(f1["theta_best"], float)
    if best is not None:
        # the model under judgement is a named start, not the loss's favourite
        # (the loss ranks the 14.63 basin first with q_e -> 0: C23 again; the gates decide)
        fb = np.load(OUTDIR / f"{S1.STAGE_FILES['stage1']}{ktag}{tag}_start_{best}.npz", allow_pickle=True)
        th1 = np.asarray(fb["theta"], float)
        print(f"  JUDGED AS exp80: the start '{best}' (loss {float(fb['loss']):.4f}), not the loss's best '{str(f1['best_name'])}'")
    names = list(lp.names)
    models = {"baseline": th_base, "14.63 basin": th_basin, "exp80": th1}
    tn = [str(n) for n in f1["theta_names"]]
    print(f"  Stage 1: starts " + ", ".join(
        f"{n} {l:.4f} ({e} evals, " + ", ".join(f"{k} {float(t[tn.index(k)]):+.3f}" for k in knobs) + ")"
        for n, l, e, t in zip(f1["names"], f1["losses"], f1["n_evals"], f1["thetas"]))
          + f"; best '{str(f1['best_name'])}' {float(f1['loss_best']):.4f} (null {float(f1['loss_null']):.4f})"
          + (f"; RAILED {list(f1['railed_best'])}" if len(f1["railed_best"]) else ""))
    moved = [(n, a, b) for n, a, b in zip(names, th_base, th1) if abs(a - b) > 0.05]
    print(f"  exp80 moved by > 0.05 from the baseline: " + (", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in moved) or "none"))
    # Stage 0 C's frozen-theta point (the baseline with log_f_e, b_e, q_e at the
    # radius-term tune): the size-clean point the fit started from
    src = S1.TUNED_SOURCE.get(tuple(knobs))
    if src is not None and (OUTDIR / f"stage0_cand_{src}.npz").exists():
        d0 = np.load(OUTDIR / f"stage0_cand_{src}.npz", allow_pickle=True)
        th0 = th_base.copy()
        for n_, v_ in zip([str(n) for n in d0["free"]], np.asarray(d0["x_tuned"], float)):
            th0[names.index(n_)] = v_
        models["exp80 (S0C point)"] = th0
        print(f"  exp80 (S0C point): the baseline with Stage 0 C's frozen-theta tune of {list(d0['free'])} (no fit)")
    for nm in also:
        fa = OUTDIR / f"{S1.STAGE_FILES['stage1']}{ktag}{tag}_start_{nm}.npz"
        if fa.exists():
            tha = np.asarray(np.load(fa, allow_pickle=True)["theta"], float)
            models[f"exp80 ({nm})"] = tha
            print(f"  exp80 ({nm}): loss {float(np.load(fa, allow_pickle=True)['loss']):.4f}; moved by > 0.05 from exp80: "
                  + (", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in zip(names, th1, tha) if abs(a - b) > 0.05) or "none"))
    print(f"  PARAMETERS: baseline 12; 14.63 basin 12; exp80 {lp.n_theta} (12 + {list(knobs)}); the size law in words: "
          f"an extended deposit made at t' has size 10^log_f_e (1+z')^b_e R200c(t') (R200c(t_obs)/R200c(t'))^q_e — "
          f"it expands with the halo after deposition by the fraction q_e of the halo's growth in radius")
    for lab, th in models.items():
        print(f"    {lab:<14}{lp.describe(th)}")

    # ---- 2. the loss grid + the radius term as a report ------------------------ #
    print(f"\n{RULE}\n2. THE LOSS GRID under the adopted references — A, F, S, B per epoch; the radius term Z (exp78) scored after the fit\n{RULE}")
    z_ref = None
    grid, pred, pred_m = {}, {}, {}
    print(f"  {'model':<16}{'epoch':>6}{'A':>8}{'F':>8}{'S':>8}{'B':>8}{'Z raw':>8}{'Z norm':>8}{'A2+F2+S2+B2':>13}")
    all_curves = [meas[i] for i in range(len(recs))]
    for lab, th in list({"null": np.r_[th_nested, np.zeros(nk)]}.items()) + list(models.items()):
        m = lp.predict(th, Rm, nodes=M2.FULL_NODES)
        raw = np.array([ST.radius_term(m[index_of[rows_m[k]], j], truth_m[rows_m[k], k], Rm, lmh_bins[rows_m[k], k])[0]
                        for j, k in enumerate(EPOCHS)])
        if lab == "null":
            z_ref = raw
            continue
        tab = lp.table(th)
        grid[lab] = np.column_stack([tab, raw, raw / z_ref])
        for j, k in enumerate(EPOCHS):
            print(f"  {lab if j == 0 else '':<16}{ANCHOR_Z[k]:>6}" + "".join(f"{v:>8.3f}" for v in tab[j])
                  + f"{raw[j]:>8.4f}{raw[j] / z_ref[j]:>8.3f}{(tab[j] ** 2).sum():>13.3f}")
        print(f"  {'':<16}{'total':>6}{'':>48}{(tab ** 2).sum():>13.3f}   Z rms {np.sqrt(np.mean((raw / z_ref) ** 2)):.3f}")
        pred_m[lab] = m
        pred[lab] = lp.predict(th, F.R_GRID, nodes=M2.FULL_NODES, curves=all_curves)
    print(f"\n  the radius term's tercile medians of log R_f(model)/R_f(truth) [dex], low / mid / high halo-mass tercile")
    for lab, m in pred_m.items():
        print(f"  {lab}:")
        med = np.array([ST.radius_term(m[index_of[rows_m[k]], j], truth_m[rows_m[k], k], Rm, lmh_bins[rows_m[k], k])[1]
                        for j, k in enumerate(EPOCHS)])
        for i, f in enumerate(ST.FRACTIONS):
            print(f"    R{int(100 * f):<3}" + "".join("  z=" + f"{ANCHOR_Z[j]}:" + "/".join(f"{v:+.3f}" for v in med[j, i]) for j in range(5)))

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
          f"(figures in figures/qa, *exp80_*; the mass planes are qa_planes_*)\n{RULE}")
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
                               name=f"exp80_{lab.replace(' ', '_').replace('(', '').replace(')', '').replace('.', 'p')}{tag}",
                               figdir=(None if tables_only else FIGDIR / "qa"), figures=not tables_only,
                               verbose=False, bin_by=lmh_bins[fit_all][:, 0], bin_label=r"logM$_h$(z=0.4)",
                               bin_by_ms=logms, ms_label=r"logM$_*$ (total)", halo_mass_epochs=lmh_cat[fit_all])
        qa.print_size_gate(out[lab]["size_gate_ms"], ANCHOR_Z, "STELLAR mass")
        qa.print_size_gate(out[lab]["size_gate_mh"], ANCHOR_Z, "HALO mass")
        gates[lab] = (out[lab]["size_gate_ms"][3], out[lab]["size_gate_ms"][4],
                      out[lab]["size_gate_mh"][3], out[lab]["size_gate_mh"][4])
        s = out[lab]["sizes"]
        print(f"    mass-size slope R50 vs M*(<148): model " + " / ".join(f"{s[('R50', k)][1]['slope']:.2f}" for k in EPOCHS)
              + "; truth " + " / ".join(f"{s[('R50', k)][0]['slope']:.2f}" for k in EPOCHS))
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

    np.savez(OUTDIR / f"stage1_eval{ktag}{tag}.npz",
             grid_keys=np.array(list(grid)), grid=np.array(list(grid.values())),
             prof_keys=np.array([f"{a}|{k}" for (a, k) in prof]), prof=np.array(list(prof.values())),
             leak_keys=np.array([f"{a}|{k}" for (a, k) in leak]), leak=np.array(list(leak.values())),
             tilt_keys=np.array([f"{a}|{k}" for (a, k) in tl]), tilt=np.array(list(tl.values())),
             gate_keys=np.array(list(gates)), gates=np.array(list(gates.values())),
             thetas=np.array(list(models.values())), model_keys=np.array(list(models)),
             fit_all=fit_all, complete=complete)
    print(f"\nwrote {OUTDIR / f'stage1_eval{ktag}{tag}.npz'}")


if __name__ == "__main__":
    a = sys.argv
    also = tuple(a[a.index("--also") + 1].split(",")) if "--also" in a else ()
    kn = tuple(a[a.index("--knobs") + 1].split(",")) if "--knobs" in a else S1.DEFAULT_KNOBS
    if "--knobs" in a and a[a.index("--knobs") + 1] == "none":
        kn = ()
    main(smoke="--smoke" in a, tables_only="--tables-only" in a, also=also, knobs=kn,
         best=(a[a.index("--best") + 1] if "--best" in a else None),
         fit_tag=(a[a.index("--fit-tag") + 1] if "--fit-tag" in a else None), delay="--delay" in a)
