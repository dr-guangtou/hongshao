"""exp78 Stage 0 — no fit. The two candidate size terms on the adopted
baseline: their blind spots asserted, then the models we already have
scored under each, next to the loss they were fitted on and the size gate.

The decision this stage makes: a term may enter the loss only if it ranks
the known models the way the QA gates do -- exp74's measured optimum (the
adopted baseline, size-gate offset 12 of 15) above the 14.63 basin (C23, 10
of 15), and exp76's growth-rate split (11 of 15, R80 width 0.22 -> 0.48 at
z = 2) above its own g = 0 baseline (the 14.63 basin it was started from).

Models scored (all on the fitting sample, every epoch, full quadrature):
  null               the nested incumbent on the measured curves (= 1.000)
  exp63 official     exp63's joint fit on the OFFICIAL DiffMAH curves
  exp74 optimum      THE BASELINE MEAN (`rebaseline.adopted_baseline()`)
  14.63 basin        `exp74/outputs/rebaseline_exp63.npz` (gate-rejected)
  incumbent          the incumbent re-baselined on the measured curves
  exp76 split        g = -0.27, `exp76/outputs/stage1_fit_start_b-1.npz`

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=. \\
        nohup uv run python -u experiments/exp78_size_aware_objective/stage0_terms.py [--smoke] \\
        > experiments/exp78_size_aware_objective/outputs/stage0_terms.log 2>&1 &
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
import size_terms as ST                                  # noqa: E402
from hongshao import qa                                  # noqa: E402
from hongshao.profile_data import load_profiles          # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RB = _by_path("exp74_rebaseline", ROOT / "experiments/exp74_c19_history_leak/rebaseline.py")  # exp54 shadows it

RULE = "=" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR = HERE / "outputs"
E74 = ROOT / "experiments/exp74_c19_history_leak/outputs"
E76 = ROOT / "experiments/exp76_growth_rate_split/outputs"
POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
TERM_NAMES = tuple(ST.TERMS)
#: the gate reading each term must reproduce (offset sub-gate, 2026-09-08 evals)
GATE_ORDER = (("exp74 optimum", "14.63 basin"), ("exp76 split", "14.63 basin"))


def merged_truth(recs, data):
    """exp73's merged 0.673-148 kpc grid: the density-rebuilt curve inside
    2 kpc spliced onto the stored CoG outside it."""
    pop = np.load(POP, allow_pickle=True)
    sel = np.array([h.row for h in recs])
    idx = np.asarray(pop["index"], int)[sel]
    d = load_profiles()
    Rm, truth_m = C.extended_cog(data, d["cog_from_density_shared"][idx], F.R_GRID, d["shared_grid_kpc"])
    n_inner = len(Rm) - len(F.R_GRID)
    assert np.allclose(Rm[n_inner:], F.R_GRID)
    return Rm, truth_m, n_inner


def build(smoke=False):
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr = RB.build(smoke)
    Rm, truth_m, n_inner = merged_truth(recs, data)
    good_m = np.isfinite(truth_m).all(2) & (truth_m > 0).all(2)          # (n, 5)
    rows_m = {k: pr.rows[k][good_m[pr.rows[k], k]] for k in pr.epochs}
    print(f"  merged grid: {len(Rm)} radii, {Rm[0]:.3f}-{Rm[-1]:.2f} kpc ({n_inner} inside 2 kpc)")
    print(f"  the term's sample per epoch (fitting rows with a usable merged truth): "
          + "/".join(f"{len(rows_m[k])}" for k in pr.epochs) + "  of the loss's "
          + "/".join(f"{len(pr.rows[k])}" for k in pr.epochs))
    return recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr, Rm, truth_m, n_inner, rows_m


def models_to_score(fz, spec2, th_inc, meas, recs, smoke):
    tag = "_smoke" if smoke else ""
    official = E.build_curves(recs, verbose=False)
    th63 = np.asarray(fz["theta_best"], float)
    th_nested = M2.with_levers_theta(M2.nested_theta(th_inc, delay=spec2.delay, growth=spec2.growth_split), spec2)

    def theta(path):
        p = Path(str(path).replace(".npz", f"{tag}.npz"))
        p = p if p.exists() else Path(path)
        return np.asarray(np.load(p, allow_pickle=True)["theta_best"], float)

    spec_g = M2.Spec2(theta_names=M2.THETA_NAMES_GROWTH, extended_family=spec2.extended_family,
                      compact_in_kpc=spec2.compact_in_kpc)
    f76 = E76 / "stage1_fit_start_b-1.npz"
    th76 = np.asarray(np.load(f76, allow_pickle=True)["theta"], float)
    out = {"null": (spec2, th_nested, meas),
           "exp63 official": (spec2, th63, official),
           "exp74 optimum": (spec2, theta(E74 / "stage1_refit_measured.npz"), meas),
           "14.63 basin": (spec2, theta(E74 / "rebaseline_exp63.npz"), meas),
           "incumbent": (spec2, theta(E74 / "rebaseline_incumbent.npz"), meas),
           "exp76 split": (spec_g, th76, meas)}
    jg = spec_g.index("g_split")
    print(f"  exp76 split: g = {th76[jg]:+.3f} from {f76.name}")
    return out


def score_terms(m, truth_m, Rm, lmh_bins, rows_m, index_of):
    """Raw per-epoch values of every term for one model's (n_all, 5, nRm)
    prediction; also the radius term's median table (5, 3, 3)."""
    raw = {t: np.full(5, np.nan) for t in TERM_NAMES}
    med = np.full((5, len(ST.FRACTIONS), 3), np.nan)
    ann = np.full((5, 2), np.nan)
    n_bad = np.zeros(5, int)
    for j, k in enumerate(EPOCHS):
        rows = rows_m[k]
        mk = m[index_of[rows], j]
        tk = truth_m[rows, k]
        lk = lmh_bins[rows, k]
        raw["radius"][j], med[j], n_bad[j] = ST.radius_term(mk, tk, Rm, lk)
        raw["annular (truth total, exp77)"][j], ann[j], _ = ST.annular_term(mk, tk, Rm, "truth")
        raw["annular (own total)"][j], _, _ = ST.annular_term(mk, tk, Rm, "own")
    return raw, med, ann, n_bad


def main(smoke=False):
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp78 Stage 0 — two candidate size terms on the adopted baseline; NO FIT\n{RULE}\n")
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr, Rm, truth_m, n_inner, rows_m = build(smoke)
    index_of = np.full(len(recs), -1)
    index_of[pr.all_rows] = np.arange(len(pr.all_rows))

    # ---- 1. the probes ------------------------------------------------------- #
    print(f"\n{RULE}\n1. THE PROBES — each term on synthetic models built from the truth (z=0.4 and z=2 rows)\n{RULE}")
    ST.selftest(truth_m[rows_m[0]][:, [0, 4]], Rm)
    probes = {}
    for k in (0, 4):
        print(f"\n  z = {ANCHOR_Z[k]} ({len(rows_m[k])} galaxies):")
        probes[k] = ST.run_probes(truth_m[rows_m[k], k], Rm, lmh_bins[rows_m[k], k])

    # ---- 2. the models ------------------------------------------------------- #
    print(f"\n{RULE}\n2. THE MODELS — every term per epoch, raw [dex] and normalised to the null\n{RULE}")
    models = models_to_score(fz, spec2, th_inc, meas, recs, smoke)
    raw, med, ann, loss_terms, sizes_gate, preds = {}, {}, {}, {}, {}, {}
    for lab, (spec, th, curves) in models.items():
        cv = [curves[i] for i in pr.all_rows]
        m = M2.predict2(spec, th, cv, Rm, epochs=EPOCHS, nodes=M2.FULL_NODES)
        preds[lab] = m
        raw[lab], med[lab], ann[lab], nb = score_terms(m, truth_m, Rm, lmh_bins, rows_m, index_of)
        per = {k: pr.problems[k].score_model(m[pr.index[k], j, n_inner:]) for j, k in enumerate(EPOCHS)}
        loss_terms[lab] = np.array([[per[k][0], per[k][1], per[k][2], per[k][4]] for k in EPOCHS])
        if nb.any():
            print(f"  {lab}: {nb.tolist()} unbuildable galaxies per epoch")
    ref = {t: raw["null"][t] for t in TERM_NAMES}
    norm = {lab: {t: raw[lab][t] / ref[t] for t in TERM_NAMES} for lab in models}
    for t in TERM_NAMES:
        print(f"\n  {t}: raw [dex] per epoch | normalised to the null | rms of the five normalised values")
        print(f"  {'model':<18}" + "".join(f"{f'z={z}':>8}" for z in ANCHOR_Z) + "   |"
              + "".join(f"{f'z={z}':>8}" for z in ANCHOR_Z) + f"{'Z':>8}")
        for lab in models:
            r, nz = raw[lab][t], norm[lab][t]
            print(f"  {lab:<18}" + "".join(f"{v:>8.4f}" for v in r) + "   |" + "".join(f"{v:>8.3f}" for v in nz)
                  + f"{np.sqrt(np.mean(nz ** 2)):>8.3f}")
    print(f"\n  radius term, the tercile medians of log10 R_f(model)/R_f(truth) [dex] (low / mid / high halo-mass tercile):")
    for lab in models:
        if lab == "null":
            continue
        print(f"  {lab}:")
        for i, f in enumerate(ST.FRACTIONS):
            print(f"    R{int(100 * f):<3}" + "".join(
                "  z=" + f"{ANCHOR_Z[j]}:" + "/".join(f"{v:+.3f}" for v in med[lab][j, i]) for j in range(5)))

    # ---- 3. the loss they were fitted on, and the size gate ------------------ #
    print(f"\n{RULE}\n3. THE LOSS (A^2+F^2+S^2+B^2, adopted references) and the SIZE GATE (offset sub-gate, standard "
          f"grid, fixed stellar mass)\n{RULE}")
    fit_all = np.zeros(len(recs), bool); fit_all[pr.all_rows] = True
    fit_all &= np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for lab in models:
        if lab == "null":
            continue
        fit_all &= np.isfinite(preds[lab][index_of.clip(0)]).all(axis=(1, 2)) & (preds[lab][index_of.clip(0)] > 0).all(axis=(1, 2))
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    lmh_cat = SEL.sample_masses(hs)[np.array([h.row for h in recs])]
    logms = np.log10(np.clip(data[fit_all][:, 0, -1], 1.0, None))
    print(f"  {'model':<18}{'loss':>8}" + "".join(f"{f'z={z}':>8}" for z in ANCHOR_Z)
          + f"{'offset pass':>13}{'R50 z=2':>9}{'R80 z=2':>9}{'R80w z=2':>10}")
    gate_off, loss_tot = {}, {}
    for lab in models:
        lt = loss_terms[lab]
        per = (lt ** 2).sum(1)
        loss_tot[lab] = float(per.sum())
        if lab == "null":
            print(f"  {lab:<18}{per.sum():>8.3f}" + "".join(f"{v:>8.3f}" for v in per))
            continue
        prd = preds[lab][index_of[fit_all], :, n_inner:]
        out = qa.evaluate(prd, data[fit_all], F.R_GRID, ANCHOR_Z, name=f"exp78_stage0_{lab}", figdir=None,
                          figures=False, verbose=False, bin_by=lmh_bins[fit_all][:, 0], bin_by_ms=logms,
                          halo_mass_epochs=lmh_cat[fit_all])
        g, n_ok, n, n_off, n_wid = out["size_gate_ms"]
        sizes_gate[lab] = g
        gate_off[lab] = n_off
        print(f"  {lab:<18}{per.sum():>8.3f}" + "".join(f"{v:>8.3f}" for v in per)
              + f"{f'{n_off} of {n}':>13}{g[('R50', 4)]['offset']:>+9.3f}{g[('R80', 4)]['offset']:>+9.3f}"
              + f"{g[('R80', 4)]['width_ratio']:>10.2f}")

    # ---- 4. the ranking check ------------------------------------------------ #
    print(f"\n{RULE}\n4. THE RANKING — does each term order the models as the gates do?\n{RULE}")
    verdict = {}
    for t in TERM_NAMES:
        z = {lab: float(np.sqrt(np.mean(norm[lab][t] ** 2))) for lab in models}
        ok = all(z[a] < z[b] for a, b in GATE_ORDER)
        verdict[t] = ok
        order = sorted((lab for lab in models if lab != "null"), key=lambda l: z[l])
        print(f"\n  {t}: {'PASS' if ok else 'FAIL'}")
        print(f"    best to worst: " + " < ".join(f"{l} ({z[l]:.3f})" for l in order))
        for a, b in GATE_ORDER:
            print(f"    {a} ({z[a]:.3f}) {'<' if z[a] < z[b] else '>='} {b} ({z[b]:.3f})  "
                  f"gate: offset {gate_off[a]} vs {gate_off[b]} of 15  {'ok' if z[a] < z[b] else 'WRONG WAY'}")
        # the loss with the term added, weight 1, per epoch, so the reader sees whether the ordering flips
        print(f"    loss + Z^2 per epoch: " + "; ".join(
            f"{lab} {loss_tot[lab]:.2f} -> {loss_tot[lab] + float((norm[lab][t] ** 2).sum()):.2f}"
            for lab in ("exp74 optimum", "14.63 basin", "exp76 split")))
    print(f"\n  the loss alone orders: " + " < ".join(
        f"{l} ({loss_tot[l]:.2f})" for l in sorted((l for l in models if l != "null"), key=lambda l: loss_tot[l])))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTDIR / f"stage0_terms{tag}.npz", models=np.array(list(models)), term_names=np.array(TERM_NAMES),
             raw=np.array([[raw[l][t] for t in TERM_NAMES] for l in models]),
             ref=np.array([ref[t] for t in TERM_NAMES]), med=np.array([med[l] for l in models]),
             ann=np.array([ann[l] for l in models]), loss_terms=np.array([loss_terms[l] for l in models]),
             gate_offset=np.array([gate_off.get(l, -1) for l in models]),
             verdict=np.array([verdict[t] for t in TERM_NAMES]), Rm=Rm, n_inner=n_inner,
             n_rows=np.array([len(rows_m[k]) for k in EPOCHS]))
    print(f"\nwrote {OUTDIR / f'stage0_terms{tag}.npz'}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
