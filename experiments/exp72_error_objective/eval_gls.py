"""exp72 Step 2b — judge the error-weighted refit against the fit it replaces.

Two models, fitted on the same galaxies, with the same twelve parameters, the
same bounds, the same starts and the same engine. The ONLY difference is the
objective. This scores them on ground neither of them chose.

FIVE THINGS, in the order they can change the verdict:

1. **The transfer matrix.** Each fit's loss under each objective, normalised so
   the nested incumbent is 1. The diagonal is what each fit optimised and must
   win; the off-diagonal is what it gave up. A fit that wins its own objective
   and does no worse on the other one is a free improvement; a fit that wins its
   own and loses badly on the other has bought a preference, not a better model.
2. **The standard battery**, `hongshao.qa.evaluate`, unchanged — the same tiers
   every product in this programme is quoted on, on the fitting sample and on
   the mh-complete subset as the after-fit check.
3. **Did the shape actually improve?** The GLS objective was adopted BECAUSE it
   charges for shape at z = 2 where the incumbent objective charges 0.2 per cent.
   If the refit does not move the z = 2 shape, the objective was not the binding
   constraint and that is the result.
4. **What did the amplitude cost?** The other half of the same question.
5. **exp71's diagnostics, re-run.** Does the halo-mass tilt at z = 2, and the
   future-growth leakage that explains it, change under a different objective?
   Both are properties of the model class rather than of one fit, so they should
   NOT move much; if they do, something has been misunderstood.

Run after `fit_gls.py`:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u \\
        experiments/exp72_error_objective/eval_gls.py [--smoke] [--tables-only]
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
          ROOT / "experiments/exp71_c18_selection_leakage", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import selection as SEL                                  # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
from hongshao import qa                                  # noqa: E402
import fit_gls as G                                      # noqa: E402
from errors import RULE, THIN                            # noqa: E402

OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
ANCHOR_Z = E.ANCHOR_Z
EPOCHS = (0, 1, 2, 3, 4)
TABLE_KEYS = ("kpc:M(<10)", "kpc:M(<100)", "kpc:M(50-100)")
SEL_NPZ = ROOT / "experiments/exp54_unpinned_amplitude/outputs/selection.npz"
R_TILT = 103.45


def exp63_loss(spec2, curves, data, lmh, keep, cogs, th_nested):
    """Yield (epoch, Problem2, rows) with every term normalised at the nested
    incumbent exactly as `stage2_fit` does, so the caller can score any model
    under exp63's real loss rather than a reimplementation of it."""
    for k in EPOCHS:
        rows = np.where(keep[:, k])[0]
        ref = S2F.Problem2(M2.Spec2(), curves, data, rows, F.R_GRID, l_s_ref=None,
                           binned=True, lmh=lmh[:, k], l_b_ref=None, epoch=k)
        sc0 = ref.score_model(cogs["nested incumbent"][rows, k])
        pr = S2F.Problem2(M2.Spec2(), curves, data, rows, F.R_GRID, l_s_ref=sc0[2],
                          binned=True, lmh=lmh[:, k], l_b_ref=sc0[4], epoch=k)
        yield k, pr, rows


def main(smoke=False, tables_only=False):
    print(f"{RULE}\nexp72 Step 2b — the error-weighted refit against the fit it "
          f"replaces\n{RULE}\n")
    tag = "_smoke" if smoke else ""
    fzg = np.load(OUTDIR / f"fit_gls{tag}.npz", allow_pickle=True)
    (recs, data, lmh, curves, sel, spec2, th_inc, th_nested, fz63,
     keep, sig_cog, sig_ann, ann_ok) = G.build(smoke, float(fzg["floor"]))

    thetas = {
        "gls refit": np.asarray(fzg["theta_best"], float),
        "exp63 joint": np.asarray(fz63["theta_best"], float),
        "nested incumbent": th_nested,
    }
    print(f"\n  gls refit  : loss {float(fzg['loss_best']):.4f} vs null "
          f"{float(fzg['loss_null']):.4f}, best start '{str(fzg['best_name'])}', "
          f"floor f = {float(fzg['floor'])}")
    print("  " + "  ".join(f"{n}={v:+.4f}" for n, v in
                           zip(spec2.theta_names, thetas["gls refit"])))
    print("  exp63 joint: " + "  ".join(f"{n}={v:+.4f}" for n, v in
                                        zip(spec2.theta_names, thetas["exp63 joint"])))
    moved = [(n, a, b) for n, a, b in zip(spec2.theta_names, thetas["exp63 joint"],
                                          thetas["gls refit"]) if abs(a - b) > 0.05]
    print(f"\n  parameters that moved by more than 0.05: "
          + (", ".join(f"{n} {a:+.3f}->{b:+.3f}" for n, a, b in moved) or "none"))

    cogs = {n: M2.predict2(spec2, th, curves, F.R_GRID, nodes=M2.FULL_NODES)
            for n, th in thetas.items()}
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for m in cogs.values():
        good &= np.isfinite(m).all(axis=(1, 2)) & (m > 0).all(axis=(1, 2))
    good &= keep.all(1)
    keep = keep & good[:, None]
    print(f"  scored on {int(good.sum())} galaxies")

    # ---- 1. the transfer matrix ----------------------------------------- #
    print(f"\n{RULE}\n1. THE TRANSFER MATRIX — each fit's loss under each "
          f"objective, with the nested\n   incumbent at 1.000 by construction. "
          f"The DIAGONAL is what each fit optimised and\n   must win; the "
          f"OFF-DIAGONAL is what it gave up.\n{RULE}")
    pr_gls = G.GlsProblem(spec2, curves, data, keep, sig_cog, sig_ann, ann_ok,
                          th_nested, floor=float(fzg["floor"]))
    gls_tot, e63_tot = {}, {}
    print(f"\n  under chi2_gls_amp (the refit's own objective), per epoch")
    print(f"  {'model':<20}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z) + f"{'total':>10}")
    for n in cogs:
        per, _ = pr_gls.per_epoch(thetas[n], nodes=M2.FULL_NODES)
        gls_tot[n] = per
        print(f"  {n:<20}" + "".join(f"{v:>9.4f}" for v in per)
              + f"{np.nansum(per):>10.4f}")

    print(f"\n  under exp63's four-term loss (the incumbent objective), per epoch")
    print(f"  {'model':<20}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z) + f"{'total':>10}")
    for n in cogs:
        per = np.full(5, np.nan)
        for k, pr, rows in exp63_loss(spec2, curves, data, lmh, keep, cogs, th_nested):
            a, f_, s_, _, b_ = pr.score_model(cogs[n][rows, k])
            per[k] = a * a + f_ * f_ + s_ * s_ + b_ * b_
        e63_tot[n] = per
        print(f"  {n:<20}" + "".join(f"{v:>9.4f}" for v in per)
              + f"{np.nansum(per):>10.4f}")

    print(f"\n{THIN}\n  the matrix, as a percentage change from the exp63 fit "
          f"(negative = better)\n{THIN}")
    print(f"  {'':<20}{'chi2_gls_amp':>16}{'exp63 four-term':>18}")
    for n in ("gls refit", "nested incumbent"):
        dg = 100 * (np.nansum(gls_tot[n]) / np.nansum(gls_tot["exp63 joint"]) - 1)
        de = 100 * (np.nansum(e63_tot[n]) / np.nansum(e63_tot["exp63 joint"]) - 1)
        print(f"  {n:<20}{dg:>+15.1f}%{de:>+17.1f}%")

    # ---- 2. the standard battery ---------------------------------------- #
    print(f"\n{RULE}\n2. THE STANDARD BATTERY — `hongshao.qa.evaluate`, "
          f"unchanged. Median relative bias\n   [%], fitting sample / "
          f"mh-complete after-fit check.\n{RULE}")
    sn = np.load(SEL_NPZ, allow_pickle=True)
    cuts = sn["cuts"]
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    lmh_cat = SEL.sample_masses(hs)[sel]
    complete = keep & np.isfinite(lmh_cat) & (lmh_cat >= cuts[None, :])
    msk = complete[good]
    logms = np.log10(np.clip(data[good][:, 0, -1], 1.0, None))
    QADIR = FIGDIR / "qa"
    QADIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for n in cogs:
        out[n] = qa.evaluate(
            cogs[n][good], data[good], F.R_GRID, list(ANCHOR_Z),
            name=f"exp72_{n.replace(' ', '_')}",
            figdir=(None if tables_only else QADIR), figures=not tables_only,
            verbose=False, bin_by=lmh[good][:, 0], bin_label=r"logM$_h$(z=0.4)",
            bin_by_ms=logms, ms_label=r"logM$_*$ (total)")
    for key in TABLE_KEYS:
        print(f"\n  {key}")
        print(f"  {'model':<20}{'sample':<16}"
              + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for n in cogs:
            t, mm = out[n]["truth"][key], out[n]["model"][key]
            print(f"  {n:<20}{'fitting sample':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[:, j], t[:, j])):>9.1f}%"
                for j in range(5)))
            print(f"  {'':<20}{'mh-complete':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[msk[:, j], j], t[msk[:, j], j])):>9.1f}%"
                for j in range(5)))
    print(f"\n  tier 3 — profile max|relative| beyond 5 kpc (median); lower is "
          f"a better SHAPE")
    print(f"  {'model':<20}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    for n in cogs:
        print(f"  {n:<20}"
              + "".join(f"{np.nanmedian(out[n]['mr_out'][:, j]):>10.4f}" for j in range(5)))

    # ---- 3 & 4. shape and amplitude, separated -------------------------- #
    print(f"\n{RULE}\n3. DID THE SHAPE MOVE, AND WHAT DID THE AMPLITUDE COST? — "
          f"the two halves of the\n   objective, measured directly. SHAPE is "
          f"the median absolute residual of the\n   profile PINNED at "
          f"M*(<103 kpc); AMPLITUDE is the median log10(model/truth) of\n   "
          f"M*(<103 kpc) itself.\n{RULE}")
    i100 = F.I100
    print(f"\n  shape: median over galaxies of the mean |log10(model/truth)| of "
          f"the pinned profile, dex")
    print(f"  {'model':<20}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    shape_tab = {}
    for n in cogs:
        m = cogs[n][good]
        d = data[good]
        pm = m / m[:, :, i100][:, :, None]
        pd = d / d[:, :, i100][:, :, None]
        r = np.abs(np.log10(np.clip(pm, 1e-30, None)) - np.log10(np.clip(pd, 1e-30, None)))
        shape_tab[n] = np.nanmedian(np.nanmean(r, axis=2), axis=0)
        print(f"  {n:<20}" + "".join(f"{v:>10.4f}" for v in shape_tab[n]))
    print(f"\n  amplitude: median log10(model/truth) of M*(<103 kpc), dex")
    print(f"  {'model':<20}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    amp_tab = {}
    for n in cogs:
        y = (np.log10(np.clip(cogs[n][good][:, :, i100], 1.0, None))
             - np.log10(np.clip(data[good][:, :, i100], 1.0, None)))
        amp_tab[n] = np.nanmedian(y, axis=0)
        print(f"  {n:<20}" + "".join(f"{v:>+10.4f}" for v in amp_tab[n]))

    # ---- 5. exp71's diagnostics ----------------------------------------- #
    print(f"\n{RULE}\n5. exp71's DIAGNOSTICS, RE-RUN — the halo-mass tilt at "
          f"z = 2 and the future-growth\n   leakage that explains it. Both are "
          f"properties of the model CLASS, so a change of\n   objective should "
          f"not move them much. If it does, something has been "
          f"misunderstood.\n{RULE}")
    growth = lmh_cat[:, 0][:, None] - lmh_cat
    print(f"\n  halo-mass tilt of M*(<{R_TILT:g} kpc), dex per dex "
          f"(fitting sample | mh-complete)")
    print(f"  {'model':<20}" + "".join(f"{f'z={z}':>18}" for z in ANCHOR_Z))
    c_idx = int(np.argmin(np.abs(F.R_GRID - R_TILT)))
    tilt_tab, leak_tab = {}, {}
    for n in cogs:
        y = (np.log10(np.clip(cogs[n][:, :, c_idx], 1.0, None))
             - np.log10(np.clip(data[:, list(EPOCHS), c_idx], 1.0, None)))
        row, tl = [], np.full((5, 2), np.nan)
        for k in EPOCHS:
            a = SEL.tilt(y[:, k], lmh_cat[:, k], keep[:, k], seed=k)[0]
            b = SEL.tilt(y[:, k], lmh_cat[:, k], complete[:, k], seed=100 + k)[0]
            tl[k] = [a, b]
            row.append(f"{a:+.3f}|{b:+.3f}")
        tilt_tab[n] = tl
        print(f"  {n:<20}" + "".join(f"{v:>18}" for v in row))
        lk = np.full((5, 2), np.nan)
        for k in range(1, 5):
            r_, s_, _ = SEL.partial_growth(y[:, k], lmh_cat[:, k], growth[:, k], keep[:, k])
            lk[k] = [r_, s_]
        leak_tab[n] = lk
    print(f"\n  future-growth leakage: partial rho | dy/dG at fixed halo mass")
    print(f"  {'model':<20}" + "".join(f"{f'z={z}':>18}" for z in ANCHOR_Z[1:]))
    for n in cogs:
        print(f"  {n:<20}" + "".join(
            f"{leak_tab[n][k, 0]:+.3f}|{leak_tab[n][k, 1]:+.3f}".rjust(18)
            for k in range(1, 5)))

    np.savez(OUTDIR / f"eval_gls{tag}.npz",
             models=np.array(list(cogs)), anchor_z=np.array(ANCHOR_Z),
             **{f"gls_{n.replace(' ', '_')}": gls_tot[n] for n in cogs},
             **{f"e63_{n.replace(' ', '_')}": e63_tot[n] for n in cogs},
             **{f"shape_{n.replace(' ', '_')}": shape_tab[n] for n in cogs},
             **{f"amp_{n.replace(' ', '_')}": amp_tab[n] for n in cogs},
             **{f"tilt_{n.replace(' ', '_')}": tilt_tab[n] for n in cogs},
             **{f"leak_{n.replace(' ', '_')}": leak_tab[n] for n in cogs})
    print(f"\nwrote {OUTDIR / f'eval_gls{tag}.npz'}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, tables_only="--tables-only" in sys.argv)
